//! Inference-backend abstraction (see docs/BACKEND.md).
//!
//! Two traits sit between the observer (`lobe.rs`) and the inference engine: `Backend` (model +
//! tokenizer) and `Session` (KV state + decode). The observer is written against these, so the
//! engine is swappable — **llama-cpp-2** today, **Candle** later (`CANDLE_DESIGN.md`). Exactly one
//! backend compiles at a time, chosen by the `llama` / `candle` cargo feature; `ActiveBackend`
//! aliases the chosen impl. No `dyn`, no runtime dispatch — the trait is the contract.

use anyhow::Result;

#[cfg(feature = "llama")]
pub mod llama;
#[cfg(feature = "candle")]
pub mod candle;

#[cfg(feature = "llama")]
pub use llama::LlamaBackend as ActiveBackend;
#[cfg(feature = "candle")]
pub use candle::CandleBackend as ActiveBackend;

/// Backend-agnostic token id (a vocabulary index).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Token(pub i32);

/// One token to feed in a batched forward pass: its position in its sequence, which sequence it
/// belongs to, and whether to compute (and expose) logits for it.
#[derive(Copy, Clone, Debug)]
pub struct Decode {
    pub token: Token,
    pub pos: i32,
    pub seq: u32,
    pub logits: bool,
}

/// Detokenization mode: render special/control tokens as their literal text (`Text`, for scoring/
/// debug where the markers matter), or suppress them (`Plain`, for user-facing generated output).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Detok {
    Text,
    Plain,
}

/// KV/session configuration (mirrors the llama context params we rely on; #6 requires `kv_unified`).
#[derive(Copy, Clone, Debug)]
pub struct SessionConfig {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_seq_max: u32,
    pub kv_unified: bool,
}

/// A loaded model + tokenizer. Owns the weights; produces `Session`s that borrow it.
pub trait Backend: Sized {
    /// A session borrowing this backend's model (GAT: `LlamaContext<'a>` / Candle weights borrow).
    type Session<'a>: Session
    where
        Self: 'a;

    /// Load the model. `gpu_layers` = layers to offload (999 = all on Apple Silicon); `verbose`
    /// surfaces backend load logs (otherwise voided so stdout JSONL / the TUI stay clean).
    fn load(model_path: &str, gpu_layers: u32, verbose: bool) -> Result<Self>;

    /// Vocabulary size (logits length).
    fn n_vocab(&self) -> usize;

    /// Tokenize text. `add_bos` prepends the BOS token (used only for the pinned preamble / a
    /// standalone snippet prompt; stream tokens are a continuation → false).
    fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<Token>>;

    /// Render a single token to text. `Detok::Text` keeps special-token markers (the surprisal path
    /// and RAG tool-call parsing need them); `Detok::Plain` suppresses them (generated reply text).
    fn detok(&self, token: Token, mode: Detok) -> String;

    /// Is this an end-of-generation token (EOS/EOT family)? NB does not reliably flag gemma-4's
    /// `<turn|>`; the observer also stops on `special_token("<turn|>")` / `("<|turn>")`.
    fn is_eog(&self, token: Token) -> bool;

    /// Resolve a single special token by its literal text (e.g. `<|turn>`). `None` if it isn't a
    /// single vocab token in this model — caller should treat that as "feature unavailable".
    fn special_token(&self, text: &str) -> Option<Token>;

    /// Open an inference session (allocates the KV cache).
    fn session(&self, cfg: SessionConfig) -> Result<Self::Session<'_>>;
}

/// An inference session: holds KV state for one or more sequences and runs forward passes.
pub trait Session {
    /// Decode a batch of tokens in ONE forward pass (the fused-batch primitive). Logits are computed
    /// for entries with `logits = true`; read them afterward via `logits(i)`, where `i` is that
    /// entry's index in `batch`. `batch` must be non-empty and fit the configured `n_batch`.
    fn decode(&mut self, batch: &[Decode]) -> Result<()>;

    /// Logits (length `n_vocab`) for batch index `i` of the most recent `decode`. `i` must have had
    /// `logits = true`. The slice is valid until the next `decode`.
    fn logits(&self, i: usize) -> &[f32];

    /// Remove sequence `seq`'s cells from the KV cache (cap+reset eviction of seq 0; scratch cleanup
    /// of the interjection sequence). Shared/forked cells lose only this seq's tag.
    fn clear_seq(&mut self, seq: u32) -> Result<()>;

    /// Fork: make sequence `dst` share/copy sequence `src`'s cells (the interjection context fork).
    fn copy_seq(&mut self, src: u32, dst: u32) -> Result<()>;

    /// Highest position held for sequence `seq` (or -1 if it has no cells). Instrumentation for KV
    /// occupancy: lets the observer roll on TOTAL extent (not just seq 0's position) and detect a
    /// GEN_SEQ cell leak (after a clear it must drop to -1). Best-effort; -1 if the backend can't.
    fn seq_pos_max(&self, seq: u32) -> i32;
}
