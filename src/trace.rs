//! Structured observability for the observer (`--debug-log <path>`).
//!
//! Deep, granular JSONL via `tracing`: one JSON object per event, written non-blocking to a file so
//! logging never stalls the decode loop. Everything is gated — with no subscriber installed (the
//! default), the `tracing` macros and `enabled!` guards compile to near-nothing, so normal runs pay
//! nothing. Turn it on and you get the whole picture, no guessing: every surprisal score with its
//! logit distribution and observer context, every interjection's exact prompt / raw output / timing,
//! every window-slide reset, every RAG call.
//!
//! Event taxonomy (filter via `LOBE_LOG`, e.g. `LOBE_LOG=lobe::interject=trace,lobe::observe=info`):
//!   - `lobe::run`       one config dump at startup
//!   - `lobe::observe`   per scored token: surprisal/entropy/z/fired (+ TRACE: logit dump, context)
//!   - `lobe::trigger`   on each fire: the trigger + expected top-k + captured delta span
//!   - `lobe::interject` begin (mode, span, novelty block, EXACT prompt) / step (per gen token) / done
//!   - `lobe::roll`      cap+reset window slides: pos before/after, replayed window, latency
//!   - `lobe::rag`       native tool-call hook: prompt, raw output, parsed thought + directive
//!
//! llama.cpp's own C-level logs do NOT flow through `tracing`, so the file stays clean JSONL even
//! without `--verbose`.

use anyhow::Result;
use std::path::Path;
use tracing_appender::non_blocking::WorkerGuard;

/// Install the JSONL subscriber writing to `path`. Returns a guard that MUST be kept alive for the
/// process lifetime — dropping it flushes and stops the background writer. Captures TRACE for every
/// target by default; override the filter via the `LOBE_LOG` env var.
pub fn init(path: &Path) -> Result<WorkerGuard> {
    let file = std::fs::File::create(path)?;
    let (writer, guard) = tracing_appender::non_blocking(file);
    // Default: all our `lobe::*` events at TRACE + the llama-cpp-2 crate's own load/decode events
    // (it emits tracing too). Override entirely via `LOBE_LOG` (e.g. `lobe::interject=trace`).
    let filter = tracing_subscriber::EnvFilter::try_from_env("LOBE_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("lobe=trace,llama_cpp_2=debug"));
    tracing_subscriber::fmt()
        .json()
        .flatten_event(true) // event fields at the top level of each JSON object (analysis-friendly)
        .with_current_span(false)
        .with_span_list(false)
        .with_writer(writer)
        .with_env_filter(filter)
        .try_init()
        .map_err(|e| anyhow::anyhow!("tracing subscriber init failed: {e}"))?;
    Ok(guard)
}

/// How much per-inference detail to dump. The `tracing` macros are gated by `enabled!`, so this only
/// bounds the size of the blobs that *do* get dumped when a target is live.
#[derive(Copy, Clone, Debug)]
pub struct DebugCfg {
    /// Top-K logits to attach to each inference event (the predicting distribution).
    pub topk: usize,
    /// Also dump the FULL n_vocab logit vector — only on fires and interjection generation, never on
    /// every observed token. Huge (≈262k floats/event); off by default. `--debug-full-logits`.
    pub full_logits: bool,
}

impl Default for DebugCfg {
    fn default() -> Self {
        Self {
            topk: 64,
            full_logits: false,
        }
    }
}
