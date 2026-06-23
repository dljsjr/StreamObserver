# BACKEND.md — inference-backend abstraction (trait + cargo-feature toggle)

Goal: put a trait between the observer (`lobe.rs`) and the inference engine so the backend is
swappable — **llama-cpp-2** today, **Candle** later (the only stack that gets per-layer KV +
PLE-offload + concurrent fork-and-generate; see `CANDLE_DESIGN.md`). One backend compiles at a time,
chosen by a cargo feature.

## The surface to abstract (surveyed from `lobe.rs`)

The backend-specific calls are few and confined. Everything else (surprisal, top-k, Welford,
trigger gating, `Step`/`Trigger`, the scratch-seq pattern, sampling, dedup, CLI, JSONL, TUI,
`--debug-log`) is already backend-agnostic — it operates on logits/text.

- **Model + tokenizer:** `str_to_token` (tokenize), `token_to_str` (detok; Text vs Plain renders
  specials or not), `is_eog_token`, `n_vocab`, special-token lookup (`<|turn>`/`<turn|>`), load.
- **Session (KV state):** create-context (n_ctx/n_batch/n_seq_max/kv_unified), `batch.add` +
  `decode` (a batch of (token,pos,seq,want_logits)), `get_logits_ith`, `clear_kv_cache_seq` (evict /
  scratch cleanup), `copy_kv_cache_seq` (the interjection fork).

## The traits (`src/backend/mod.rs`)

```rust
pub struct Token(pub i32);                                  // backend-agnostic token id
pub struct Decode { token: Token, pos: i32, seq: u32, logits: bool }  // one batched token
pub enum Detok { Text, Plain }                              // render specials, or suppress them
pub struct SessionConfig { n_ctx, n_batch, n_seq_max: u32, kv_unified: bool }

pub trait Backend: Sized {
    type Session<'a>: Session where Self: 'a;              // GAT: session borrows the model
    fn load(model_path: &str, gpu_layers: u32, verbose: bool) -> Result<Self>;
    fn n_vocab(&self) -> usize;
    fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<Token>>;
    fn detok(&self, token: Token, mode: Detok) -> String;
    fn is_eog(&self, token: Token) -> bool;
    fn special_token(&self, text: &str) -> Option<Token>;  // single special token, or None
    fn session(&self, cfg: SessionConfig) -> Result<Self::Session<'_>>;
}

pub trait Session {
    fn decode(&mut self, batch: &[Decode]) -> Result<()>;  // one fused forward pass
    fn logits(&self, ith: usize) -> &[f32];                // logits for batch index `ith` (len n_vocab)
    fn clear_seq(&mut self, seq: u32) -> Result<()>;       // = clear_kv_cache_seq(Some(seq), ..)
    fn copy_seq(&mut self, src: u32, dst: u32) -> Result<()>; // = copy_kv_cache_seq(src, dst, ..)
}
```

Notes:
- **`Decode { logits: bool }`** generalizes both single-token decode and the fused two-sequence batch
  (`get_logits_ith(0)`/`(1)`), and the chunked prefill — all become "push N `Decode`s, mark which
  want logits."
- **GAT `Session<'a>`** models "the session borrows the model" (forced by llama-cpp-2's
  `LlamaContext<'a>`; Candle borrows the weights too). Lobe stays `Lobe<'a>`.
- The traits are the *contract*; only one impl is compiled (below), so there are no `dyn` objects and
  no runtime dispatch.

## Module layout & feature toggle

```
src/backend/
  mod.rs     — traits, Token/Decode/Detok/SessionConfig, and the cfg-selected aliases:
               #[cfg(feature="llama")]  pub use llama::LlamaBackend  as ActiveBackend;
               #[cfg(feature="candle")] pub use candle::CandleBackend as ActiveBackend;
  llama.rs   — LlamaBackend + LlamaSession<'a>: the current FRAGILE calls live ONLY here.
  candle.rs  — CandleBackend + CandleSession: the from-scratch gemma-4 forward pass (bring-up).
```

```toml
[features]
default = ["llama"]
llama  = ["dep:llama-cpp-2"]
candle = ["dep:candle-core", "dep:candle-nn", "dep:tokenizers"]  # added at bring-up
metal  = ["llama-cpp-2?/metal"]   # accel passthrough; only bites when llama is on
cuda   = ["llama-cpp-2?/cuda"]
vulkan = ["llama-cpp-2?/vulkan"]
```
`cargo build --release --features metal` still works (default `llama` + metal). Candle later:
`--no-default-features --features candle`.

## `lobe.rs` migration (the cutover — mechanical, the one invasive step)

`Lobe<'a>` swaps its concrete llama types for the cfg-aliased backend:
- `LlamaToken` → `backend::Token` everywhere (struct fields, `argmax`/`sample_topp` returns, literals).
- `ctx`/`batch` fields → one `session: <ActiveBackend as Backend>::Session<'a>`; `engine: &'a ActiveBackend`.
- `self.ctx.decode(&mut self.batch)` (after `batch.clear()` + `batch.add`s) → build a `Vec<Decode>`
  and `self.session.decode(&batch)`; `get_logits_ith(i)` → `self.session.logits(i)`.
- `clear_kv_cache_seq` → `session.clear_seq`; `copy_kv_cache_seq` → `session.copy_seq`.
- `model.str_to_token`/`token_to_str`/`is_eog_token` → `engine.tokenize`/`detok`/`is_eog`;
  `special_id(model, s)` → `engine.special_token(s)`.

This is the only step that touches the observer logic; it's a rename + small restructuring of the
~5 decode sites. **Status: traits + both backend impls land first (green on `llama`); the cutover is
the next step** (until then `lobe.rs` still uses llama directly, so `--features candle` won't link —
expected).

## Candle backend — what "started" means here

`candle.rs` lands as a compiling skeleton implementing both traits, with the forward pass
`unimplemented!()`. The real bring-up (per `CANDLE_DESIGN.md`) is the gemma-4 E-model forward pass:
dual head_dim (256 local / 512 global), Proportional RoPE on global layers, iSWA layer schedule,
Per-Layer Embeddings (the offload that unlocks Reason 2), GeGLU/RMSNorm, 262k byte-BPE vocab. That's
a model bring-up, not an afternoon — the skeleton makes the seam real so the bring-up has a home.
