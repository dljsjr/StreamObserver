//! The observer's public vocabulary — the CLI-facing enums (`Signal`, `EvictMode`, `InterjectMode`,
//! `AskMode`, `NoveltyMode`) and the per-token result types (`Trigger`, `Step`, `StepOutcome`). Plain
//! data, no behavior; re-exported from the `lobe` root so existing `crate::lobe::Signal` paths hold.

use super::InterjectStatus;
use clap::ValueEnum;

/// Which scalar the observer thresholds on to decide a token "fires" (punch-list #4, pluggable).
/// Both are z-scored against the running Welford baseline, which tracks whichever signal is
/// active — so `--z` means the same thing (sigmas above baseline) regardless of choice.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum Signal {
    /// -ln P(actual token): "that specific token was unexpected." The default.
    Surprisal,
    /// Entropy (nats) of the next-token distribution: "the model was uncertain *here*,"
    /// independent of which token actually arrived. Catches confusion/forking points.
    Entropy,
}

/// How the observer keeps streaming once sequence 0's KV cache fills (punch-list #6).
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum EvictMode {
    /// No eviction: decode until the context is full, then error. For bounded streams, or as the
    /// large-`n_ctx` control arm in validation (isolates corpus drift from reset drift).
    Off,
    /// Cap + reset — the supported path on Gemma's iSWA cache. When the cache is near full, clear
    /// sequence 0 and re-prime the pinned preamble + a rolling window of recent stream tokens, then
    /// continue. No position-shift, so it never hits Gemma's iSWA context-shift limitation.
    Reset,
}

/// What context an interjection reflects on.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum InterjectMode {
    /// Re-encode only the last `RECENT_TOKENS` stream tokens as a fresh prompt and react to the
    /// surprising token. Narrow but self-contained; doesn't see the observer's real context.
    Snippet,
    /// Fork the observer's FULL live context (the seq-0 KV) onto the scratch sequence via
    /// `copy_kv_cache_seq`, then ask it to reflect on that whole context. Cheaper (the context is
    /// copied, not re-prefilled) and far richer. Cleanest with `--frame` (seq 0 is a real turn).
    Context,
}

/// EXPERIMENT (templating study): how the context-mode ask FRAMES the request. `Passage` (control)
/// hands the model a discrete quoted span to "comment on" — a self-contained micro-essay each fire,
/// which may force a per-aside reset (H2). `Continuous` frames it as picking up an ongoing thread.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum AskMode {
    Passage,
    Continuous,
}

/// EXPERIMENT (templating study): the novelty-memory framing (H4). `Fresh` (control) shows the last
/// asides and asks for "a fresh angle" (content novelty). `Form` asks to vary rhythm/openings (form
/// novelty). `Off` omits the novelty memory entirely (isolates whether showing prior asides matters).
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum NoveltyMode {
    Fresh,
    Form,
    Off,
}

/// A trigger emitted when the observer is surprised enough to "speak".
#[derive(Debug, Clone)]
pub struct Trigger {
    /// Index of the token in the stream (0-based, counting only stream tokens).
    pub stream_index: usize,
    /// The surprising token, rendered to text.
    pub token_text: String,
    /// Raw surprisal in nats: -ln P(token).
    pub surprisal: f32,
    /// Entropy in nats of the distribution that predicted this token (model uncertainty here).
    pub entropy: f32,
    /// z-score of the ACTIVE signal against the running baseline (the value that actually fired).
    pub z: f32,
    /// What the model expected instead: top-k (text, probability), highest first.
    pub expected: Vec<(String, f32)>,
}

/// One scored step of observation.
#[derive(Debug, Clone)]
pub struct Step {
    pub stream_index: usize,
    pub token_text: String,
    pub surprisal: f32,
    pub entropy: f32,
    /// z-score of the ACTIVE signal (surprisal or entropy) against the running baseline.
    pub z: f32,
    pub fired: bool,
    /// Only populated when `fired` (top-k is comparatively expensive, so we gate it).
    pub trigger: Option<Trigger>,
}

/// Result of one fused `step()`: the observation plus whatever the concurrent interjection did.
pub struct StepOutcome {
    pub step: Step,
    pub interjection: InterjectStatus,
}

/// Everything the caller configures at construction, in one value (`Lobe::new(engine, n_ctx, cfg)`).
/// Built once from the CLI; `new` distributes it into the sub-structs. Replaces the old swarm of
/// `set_*` setters and the "must call before `prime()`" ordering hazard — there is no post-`new`
/// configuration step, so the cache-sizing knobs (`evict`/`keep_recent`/`interject_max`) are
/// guaranteed in place before `prime`. `Default` is the converged config (= every CLI default).
pub struct LobeConfig {
    /// Trigger signal (#4) + the identifier/entity firing gate.
    pub signal: Signal,
    pub identifiers_only: bool,
    /// Cap+reset (#6): eviction policy + rolling-window size.
    pub evict: EvictMode,
    pub keep_recent: usize,
    /// Interjection generation knobs.
    pub interject_mode: InterjectMode,
    pub ask_mode: AskMode,
    pub novelty_mode: NoveltyMode,
    pub interject_temp: f32,
    pub interject_top_p: f32,
    pub interject_max: usize,
    /// Ask tokens prefilled per tick when a fused interjection starts (CHUNKED PREFILL; ≥1). Smaller =
    /// smoother prose scroll during the prefill (each tick's decode stays lighter) but the aside takes
    /// more ticks to start; larger = the aside starts sooner but the per-tick decode is heavier. The
    /// old behavior was "the whole ask in one decode" (a ~250–500ms stream freeze at each fire).
    pub prefill_chunk: usize,
    /// Post-fire refractory period (tokens; 0 = off) and the opt-in dedup backstop (0 = off).
    pub refractory: usize,
    pub dedup: f32,
    /// Structured-observability detail (`--debug-log`).
    pub debug: crate::trace::DebugCfg,
    /// `None` = the fixed reproducible seed (byte-identical runs, the default); `Some(s)` seeds both
    /// RNG streams from entropy (`--non-deterministic`), so asides — and, with `fire_softness > 0`,
    /// which tokens fire — vary run-to-run.
    pub seed: Option<u64>,
    /// Stochastic-firing softness (z-units); `0` = the deterministic hard threshold (default).
    pub fire_softness: f32,
}

impl Default for LobeConfig {
    fn default() -> Self {
        Self {
            signal: Signal::Surprisal,
            identifiers_only: false,
            evict: EvictMode::Reset,
            keep_recent: 4096,
            interject_mode: InterjectMode::Context,
            ask_mode: AskMode::Passage,
            novelty_mode: NoveltyMode::Fresh,
            interject_temp: 0.7,
            interject_top_p: 0.95,
            interject_max: 96,
            prefill_chunk: 8,
            refractory: 0,
            dedup: 0.0,
            debug: crate::trace::DebugCfg::default(),
            seed: None,
            fire_softness: 0.0,
        }
    }
}
