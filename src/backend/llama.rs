//! llama-cpp-2 backend (the `llama` feature). ALL the version-fragile llama-cpp-2 calls live here —
//! the rest of the crate talks only to the `Backend`/`Session` traits. See docs/BACKEND.md.
//!
//! FRAGILE-API NOTE: llama-cpp-2 tracks upstream llama.cpp with no stable API. The calls tagged
//! `// FRAGILE:` are the ones most likely to drift between crate versions; verified against 0.1.150.

#![allow(deprecated)] // token_to_str is deprecated → token_to_piece; we want the convenience.

use anyhow::{Context, Result};
use std::num::NonZeroU32;

use llama_cpp_2::context::params::LlamaContextParams;
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
