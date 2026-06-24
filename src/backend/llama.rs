//! llama-cpp-2 backend (the `llama` feature). ALL the version-fragile llama-cpp-2 calls live here —
//! the rest of the crate talks only to the `Backend`/`Session` traits. See docs/BACKEND.md.
//!
//! FRAGILE-API NOTE: llama-cpp-2 tracks upstream llama.cpp with no stable API. The calls tagged
//! `// FRAGILE:` are the ones most likely to drift between crate versions; verified against 0.1.150.

#![allow(deprecated)] // token_to_str is deprecated → token_to_piece; we want the convenience.

use anyhow::{Context, Result};
use std::num::NonZeroU32;

use llama_cpp_2::context::params::{LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend as LlamaRuntime;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::token::LlamaToken;

use super::{Backend, Decode, Detok, Session, SessionConfig, Token};

/// Largest single decode we submit (chunked prefill caps at this; fused/observe are ≤ 2 tokens).
const BATCH_CAP: usize = 512;

/// Owns the llama runtime + model. Must outlive any session (which borrows the model).
pub struct LlamaBackend {
    rt: LlamaRuntime,
    model: LlamaModel,
}

impl Backend for LlamaBackend {
    type Session<'a> = LlamaSession<'a>;

    fn load(model_path: &str, gpu_layers: u32, verbose: bool) -> Result<Self> {
        let mut rt = LlamaRuntime::init().context("LlamaBackend::init failed")?;
        // llama.cpp/ggml are chatty at load (every Metal kernel compile, etc.). Void unless asked,
        // so stdout JSONL stays clean and the TUI's alternate screen isn't corrupted. (0.1.150.)
        if !verbose {
            rt.void_logs();
        }
        let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers); // FRAGILE
        let model = LlamaModel::load_from_file(&rt, model_path, &model_params) // FRAGILE
            .with_context(|| format!("failed to load model from {model_path}"))?;
        Ok(Self { rt, model })
    }

    fn n_vocab(&self) -> usize {
        self.model.n_vocab() as usize // FRAGILE: n_vocab()
    }

    fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<Token>> {
        let bos = if add_bos { AddBos::Always } else { AddBos::Never };
        Ok(self
            .model
            .str_to_token(text, bos)? // FRAGILE: str_to_token (parses special tokens)
            .into_iter()
            .map(|t| Token(t.0))
            .collect())
    }

    fn detok(&self, token: Token, mode: Detok) -> String {
        let special = match mode {
            Detok::Text => Special::Tokenize,   // keep <...> markers (scoring / RAG parse)
            Detok::Plain => Special::Plaintext, // suppress specials (user-facing reply)
        };
        // FRAGILE: token_to_str(LlamaToken, Special) -> Result<String>.
        match self.model.token_to_str(LlamaToken(token.0), special) {
            Ok(s) => s,
            // Text mode wants a visible replacement char on failure; Plain wants nothing to leak.
            Err(_) if mode == Detok::Text => String::from("\u{fffd}"),
            Err(_) => String::new(),
        }
    }

    fn is_eog(&self, token: Token) -> bool {
        self.model.is_eog_token(LlamaToken(token.0)) // FRAGILE: is_eog_token
    }

    fn special_token(&self, text: &str) -> Option<Token> {
        // parse_special tokenization; a true special token maps to exactly one id.
        match self.model.str_to_token(text, AddBos::Never) {
            Ok(v) if v.len() == 1 => Some(Token(v[0].0)),
            _ => None,
        }
    }

    fn session(&self, cfg: SessionConfig) -> Result<LlamaSession<'_>> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(cfg.n_ctx)) // FRAGILE: Option<NonZeroU32>
            .with_n_batch(cfg.n_batch) // FRAGILE
            .with_n_seq_max(cfg.n_seq_max)
            .with_kv_unified(cfg.kv_unified); // #6: required, else seq 0 dies at n_ctx/2
        let ctx = self
            .model
            .new_context(&self.rt, ctx_params) // FRAGILE: new_context
            .context("new_context failed")?;
        let n_vocab = self.n_vocab();
        Ok(LlamaSession {
            ctx,
            batch: LlamaBatch::new(BATCH_CAP, 1),
            n_vocab,
        })
    }
}

/// An inference session: the KV cache + a reusable batch. Borrows the model via `ctx`.
pub struct LlamaSession<'a> {
    ctx: LlamaContext<'a>,
    batch: LlamaBatch<'a>, // gained a phantom lifetime in 0.1.150
    n_vocab: usize,
}

impl Session for LlamaSession<'_> {
    fn decode(&mut self, batch: &[Decode]) -> Result<()> {
        debug_assert!(!batch.is_empty(), "decode requires at least one token");
        self.batch.clear();
        for d in batch {
            // FRAGILE: add(token, pos, seq_ids, compute_logits)
            self.batch
                .add(LlamaToken(d.token.0), d.pos, &[d.seq as i32], d.logits)?;
        }
        self.ctx.decode(&mut self.batch)?; // FRAGILE: decode
        Ok(())
    }

    fn logits(&self, i: usize) -> &[f32] {
        // FRAGILE: get_logits_ith(i) -> &[f32] of length n_vocab (i = batch index with logits on).
        &self.ctx.get_logits_ith(i as i32)[..self.n_vocab]
    }

    fn clear_seq(&mut self, seq: u32) -> Result<()> {
        self.ctx.clear_kv_cache_seq(Some(seq), None, None)?; // FRAGILE: = seq_rm
        Ok(())
    }

    fn copy_seq(&mut self, src: u32, dst: u32) -> Result<()> {
        self.ctx
            .copy_kv_cache_seq(src as i32, dst as i32, None, None)?; // FRAGILE: = seq_cp
        Ok(())
    }

    fn seq_pos_max(&self, seq: u32) -> i32 {
        self.ctx.kv_cache_seq_pos_max(seq as i32) // FRAGILE: = llama_kv_cache_seq_pos_max
    }
}

// --- Embedding model (#8 semantic retrieval) -------------------------------------------------------
// A SECOND model (harrier-oss-v1-270m, a gemma3-arch embedding model) loaded into the SAME runtime as
// the main model — `LlamaBackend::init()` guards double-init, so the embedder must share it. Lives
// behind the llama feature; `main` uses it only when `--rag-embed-model` is given.

impl LlamaBackend {
    /// Load an embedding model into this backend's runtime and open a LAST-token-pooling embeddings
    /// context. The model handle is leaked to `'static`: it's a process-lifetime resource (loaded once,
    /// used till exit — like the main model), and leaking lets the returned `Embedder` own its context
    /// without a self-referential struct or threaded lifetimes. Sound because the runtime (this
    /// backend) outlives the run; `'static` here means "as long as the process", which it is.
    pub fn load_embedder(&self, model_path: &str, gpu_layers: u32, n_ctx: u32) -> Result<Embedder> {
        let model_params = LlamaModelParams::default().with_n_gpu_layers(gpu_layers); // FRAGILE
        let model = LlamaModel::load_from_file(&self.rt, model_path, &model_params) // FRAGILE
            .with_context(|| format!("failed to load embedder model from {model_path}"))?;
        let model: &'static LlamaModel = Box::leak(Box::new(model)); // process-lifetime; see above
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx)) // FRAGILE: Option<NonZeroU32>
            .with_n_batch(n_ctx) // a whole text embeds in one decode
            .with_embeddings(true) // FRAGILE: turns on embedding extraction
            // LAST-token pooling — verified empirically as harrier's correct pooling (matches the
            // model card; Mean/Cls invert or degenerate). FRAGILE.
            .with_pooling_type(LlamaPoolingType::Last);
        let ctx = model
            .new_context(&self.rt, ctx_params) // FRAGILE: new_context (rt not tied to ctx lifetime)
            .context("embedder new_context failed")?;
        let n_embd = model.n_embd() as usize;
        Ok(Embedder {
            model,
            ctx,
            batch: LlamaBatch::new(n_ctx as usize, 1),
            cap: n_ctx as usize,
            n_embd,
        })
    }
}

/// An embeddings session: feeds a text through the model and reads its pooled, L2-normalized vector.
/// Owns everything (the model handle is leaked `'static`), so it moves freely into a closure.
pub struct Embedder {
    #[allow(dead_code)] // held so the leaked model's lifetime intent is explicit; ctx uses it
    model: &'static LlamaModel,
    ctx: LlamaContext<'static>,
    batch: LlamaBatch<'static>,
    cap: usize,
    #[allow(dead_code)] // read by dim(), which only the smoke test calls
    n_embd: usize,
}

impl Embedder {
    /// Embedding dimension (harrier: 640).
    #[allow(dead_code)] // smoke-test only
    pub fn dim(&self) -> usize {
        self.n_embd
    }

    /// Embed `text` → an L2-normalized vector (so cosine similarity == dot product). Each call is an
    /// independent sequence: clear seq 0, decode the whole text in one batch, read the pooled (last-
    /// token) embedding. The caller applies harrier's query instruction prefix when embedding a query.
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut toks = self.model.str_to_token(text, AddBos::Always)?; // FRAGILE: BOS like gemma chat
        toks.truncate(self.cap); // never exceed the context/batch (chunks are short; this is a guard)
        self.ctx.clear_kv_cache_seq(Some(0), None, None)?; // each embed is a fresh sequence
        self.batch.clear();
        let last = toks.len().saturating_sub(1);
        for (i, t) in toks.iter().enumerate() {
            // logits on every token: pooling reads them; for LAST pooling the final one is decisive.
            self.batch.add(*t, i as i32, &[0], i == last)?; // FRAGILE: add(token,pos,seqs,logits)
        }
        self.ctx.decode(&mut self.batch)?; // FRAGILE: decode
        let mut v = self.ctx.embeddings_seq_ith(0)?.to_vec(); // FRAGILE: pooled seq embedding
        l2_normalize(&mut v);
        Ok(v)
    }
}

/// L2-normalize in place (harrier embeddings are L2-normalized; cosine then reduces to a dot product).
fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke-test the real harrier GGUF: 640-dim, and a related passage out-scores an unrelated one.
    /// `#[ignore]` — loads models from disk (Qwen as the throwaway main model, harrier as the
    /// embedder); run manually: `cargo test --features metal -- --ignored embedder_smoke --nocapture`.
    #[test]
    #[ignore = "loads real GGUFs from models/; run manually"]
    fn embedder_smoke() {
        let backend =
            LlamaBackend::load("models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", 999, false).unwrap();
        let mut e = backend
            .load_embedder("models/harrier-oss-v1-270m-BF16.gguf", 999, 2048)
            .unwrap();
        assert_eq!(e.dim(), 640, "harrier embedding dimension");

        let instruct = "Instruct: Given a search query, retrieve relevant passages.\nQuery: ";
        let query = e.embed(&format!("{instruct}a dog barking in the yard")).unwrap();
        let related = e.embed("The puppy wagged its tail and barked loudly.").unwrap();
        let unrelated = e.embed("The Boeing 747 taxied slowly down the runway.").unwrap();
        let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let (near, far) = (cos(&query, &related), cos(&query, &unrelated));
        eprintln!("cos(q,related)={near:.4}  cos(q,unrelated)={far:.4}  gap={:.4}", near - far);
        assert!(near - far > 0.1, "related must clearly out-score unrelated (LAST pooling)");
    }
}
