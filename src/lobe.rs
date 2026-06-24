//! The observer "lobe": a thin wrapper over `llama-cpp-2` that does exactly the
//! mechanic discussed in design — teacher-force an incoming token stream into the
//! KV cache, and at every step read the next-token distribution so we can score how
//! *surprised* the model is by whatever actually arrives next.
//!
//! Observe and generate use separate paths so that generation never pollutes observation:
//!   - OBSERVE:  next token comes from the stream -> `observe()` -> `decode_one()` on KV
//!               sequence 0. This advances the observation position and captures the next-token
//!               distribution used to score the following token.
//!   - GENERATE: on a trigger, `interject()` runs a chat-framed reply on a *scratch* KV
//!               sequence (`GEN_SEQ`) via `decode_seq()` + greedy `argmax()`, then `seq_rm`s it.
//!               Sequence 0, `pos`, and `last_logits` are left untouched (details in `interject`).
//!
//! Surprisal of token t = -ln P(t | context up to t-1), read from the logits produced
//! when we decoded t-1. So we are always one step behind: predict, the real token
//! arrives, score it, then feed it and predict the next.
//!
//! ---------------------------------------------------------------------------------
//! FRAGILE-API NOTE: `llama-cpp-2` tracks upstream llama.cpp and does NOT keep a stable
//! API (no meaningful semver). Every line tagged `// FRAGILE:` is a call whose exact
//! name/signature is the most likely thing to have drifted in your installed version.
//! If it doesn't compile, the fix is almost always a renamed method on the same object
//! — check `docs.rs/llama-cpp-2/<your-version>`. The control flow is correct regardless.
//! ---------------------------------------------------------------------------------

use anyhow::Result;
use clap::ValueEnum;
use std::collections::VecDeque;

/// How many of the most-recently-observed token texts to keep as rolling context for the
/// chat-framed interjection prompt. Small on purpose: enough to situate the observer, but it
/// keeps the (one-shot) interjection prompt well under the decode batch size.
const RECENT_TOKENS: usize = 48;
/// Scratch KV-cache sequence id used for interjection generation. The observation stream lives
/// on sequence 0; generation runs entirely on this one and is discarded, so it never pollutes
/// observation. Requires the context to be built with `n_seq_max >= 2`.
const GEN_SEQ: i32 = 1;

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

/// Headroom (cells) kept below `n_ctx` so a reset fires *before* the cache is physically full. Also
/// reserves slots for an in-flight interjection/RAG prefill, which shares the unified cache via
/// GEN_SEQ. Clamped to n_ctx/4 so it stays sane at tiny test contexts.
const RESET_MARGIN: i32 = 256;
/// Tokens after a reset during which triggers are suppressed: the context is momentarily shorter,
/// so the first few surprisals shift until the window refills. Keeps a reset from self-firing.
const RESET_SETTLE: usize = 16;
/// Cap on the "delta" span (tokens since the last fire) handed to a context-mode interjection, so a
/// long quiet stretch can't blow up the prompt. The span ends with the surprising token.
const MAX_SPAN_TOKENS: usize = 128;
/// `interject_max` is a SOFT length target: once an interjection reaches it, generation continues
/// only until the next sentence boundary (so it never ends mid-clause), bounded by this much extra —
/// a HARD ceiling of `interject_max + this`, after which it stops regardless (rare runaway guard).
const INTERJECT_SENTENCE_SLACK: usize = 64;

/// Result of one fused `step()`: the observation plus whatever the concurrent interjection did.
pub struct StepOutcome {
    pub step: Step,
    pub interjection: InterjectStatus,
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

// #8 RAG lives in its own module (the result types, the rag() pass, the tool-call parser). `pub mod`
// so `rag()`'s public return type (`RagOutcome`) is reachable; `Source` is re-exported for `main`.
pub mod rag;
pub use rag::Source;

// The interjection machinery (the generation side of observe → generate → resume) lives in its own
// module. `InterjectState` is `pub(crate)` (the `interjection` field below holds it); `InterjectStatus`
// (the `StepOutcome` field + the frontends) is re-exported. `InterjectStep` stays inside the module —
// it only appears in `interject_step`'s signature there, with no external consumer.
mod interject;
use interject::InterjectState;
pub use interject::InterjectStatus;

// Interjection anti-fixation memory (recent asides + delta span + dedup) lives in its own module —
// pure state, no session. `Lobe` holds one and delegates the dedup/novelty queries to it.
mod novelty;
use novelty::InterjectionMemory;

// Backend abstraction (docs/BACKEND.md): the observer talks only to these traits + types, never to
// a concrete inference engine. `ActiveBackend` is the cfg-selected impl (llama today, candle later).
use crate::backend::{ActiveBackend, Backend, Decode, Detok, Session, SessionConfig, Token};

// The loaded model + tokenizer now lives behind the backend abstraction: `ActiveBackend`
// (= `backend::llama::LlamaBackend` or `backend::candle::CandleBackend`, per cargo feature).
// Load it with `ActiveBackend::load(path, gpu_layers, verbose)`; it plays the old `Engine` role.

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

pub struct Lobe<'a> {
    /// The backend (model + tokenizer), borrowed for tokenize/detok/is_eog/special-token lookups.
    engine: &'a ActiveBackend,
    /// The inference session (KV cache + decode). All forward passes go through this.
    session: <ActiveBackend as Backend>::Session<'a>,
    /// The next-token distribution produced by the most recent decode. We copy it out
    /// of the session immediately so we can score the *next* arriving token against it.
    last_logits: Vec<f32>,
    /// Current absolute position in the KV cache (number of tokens decoded so far).
    pos: i32,
    /// How many stream tokens we've scored (used for Trigger/Step indexing).
    stream_index: usize,
    /// Rolling window of recently-observed token texts, used to situate the interjection
    /// prompt ("here is the recent text it produced"). Capped at `RECENT_TOKENS`.
    recent: VecDeque<String>,
    /// Active trigger signal (#4) and whether firing is gated to identifier/entity-like tokens.
    signal: Signal,
    identifiers_only: bool,
    // --- #6 cap + reset state ---
    /// Context size the cache was built with (positions 0..n_ctx-1 are valid).
    n_ctx: i32,
    /// Eviction policy.
    evict: EvictMode,
    /// Pinned-prefix length: the preamble tokens replayed verbatim on every reset (the
    /// StreamingLLM "sink", except here it carries real content). Set in `prime`.
    n_keep: i32,
    /// The preamble tokens, kept for replay on reset.
    preamble: Vec<Token>,
    /// Rolling ring of recent STREAM token *ids* — distinct from `recent`'s strings, because
    /// detok→retok is not round-trip safe, so we replay the actual ids. Capped at `keep_recent`.
    recent_ids: VecDeque<Token>,
    /// The COMPLETE stream-token content currently in seq 0 (everything decoded since the last reset
    /// / prime — NOT capped, so it can grow to ~n_ctx between resets). The full live context is
    /// `preamble + context_ids`; this is what the context-dumping diagnostics replay so they show
    /// *all* the tokens the model is attending to, not just the recent window. Reset to `recent_ids`
    /// on a roll (the post-reset seq-0 stream content).
    context_ids: Vec<Token>,
    /// How many recent stream tokens to replay after a reset (the rolling window).
    keep_recent: usize,
    /// Post-reset trigger-suppression countdown.
    settle: usize,
    /// Post-fire refractory countdown: after the observer remarks, it stays quiet for
    /// `refractory_period` tokens so it doesn't obsess over the same salient thing while it lingers
    /// in the window. Counts down each observed token; reset to `refractory_period` on each fire.
    refractory: usize,
    refractory_period: usize,
    /// Count of resets so far (for the TUI status line / validation).
    resets: u64,
    /// Turn-boundary token ids (gemma chat). Used to stop interjection generation cleanly —
    /// `is_eog_token` does NOT reliably flag `<end_of_turn>` in this build, and `<start_of_turn>`
    /// is never EOG but signals the model has begun a new turn (its reply is over).
    eot: Option<Token>,
    sot: Option<Token>,
    /// In-flight streaming interjection (pumped by `interject_step`); None when idle.
    interjection: Option<InterjectState>,
    /// Whether interjections reflect on the full forked context or a re-encoded snippet.
    interject_mode: InterjectMode,
    /// EXPERIMENT (templating study): context-mode ask framing (H2) + novelty framing (H4). Both
    /// default to the control variant, so behavior is unchanged unless a flag is set.
    ask_mode: AskMode,
    novelty_mode: NoveltyMode,
    /// Interjection anti-fixation memory: recent asides (novelty memory) + the delta-since-last-fire
    /// span + the opt-in dedup backstop. Pure state (no session); see `novelty::InterjectionMemory`.
    memory: InterjectionMemory,
    /// Structured-observability detail level (`--debug-log`). Inert unless a tracing subscriber is
    /// installed (the `enabled!` guards skip all dump work otherwise).
    debug: crate::trace::DebugCfg,
    /// Interjection sampling (experiment): temperature + top-p applied ONLY to interjection
    /// generation. `temp <= 0` = greedy argmax (default). Observation scoring is always exact.
    interject_temp: f32,
    interject_top_p: f32,
    /// xorshift64 state for interjection sampling; constant-seeded so a run is reproducible.
    rng_state: u64,
    /// Independent xorshift64 state for the PROBABILISTIC firing decision (decorrelated from
    /// `rng_state` so trigger draws and interjection draws don't interfere). Seeded alongside it.
    fire_rng: u64,
    /// Softness of the stochastic firing sigmoid, in z-units (`--fire-softness`). `<= 0` = the
    /// deterministic hard threshold (`z >= z_threshold`, the default). `> 0` = fire with probability
    /// `sigmoid((z - z_threshold)/softness)` — so triggers vary run-to-run under `--random-seed`.
    fire_softness: f32,

    // --- Fused concurrent forward pass (CONCURRENT_FORWARD_PASS.md, TUI path only) ---
    // When an interjection is in flight, `step()` co-batches the next GEN_SEQ token with the stream
    // token into ONE decode, so observation (seq 0) never stalls while the lobe generates. These
    // cursors advance independently of seq-0's `pos`. (Headless still uses the blocking `interject`.)
    /// Is an interjection generating on GEN_SEQ right now (fused mode)?
    gen_in_flight: bool,
    /// GEN_SEQ's position cursor, independent of seq-0's `pos`.
    gen_pos: i32,
    /// The GEN_SEQ token to decode next tick (sampled from last tick's gen logits).
    pending_gen_tok: Option<Token>,
    /// Accumulated interjection text so far (fused mode).
    gen_out: String,
    /// Interjection tokens produced so far, and the cap.
    gen_produced: usize,
    gen_max: usize,
    /// Max interjection length (tokens), used to size the cap+reset roll margin so an interjection's
    /// full concurrent KV footprint (ask + generated tokens + seq-0 growth during gen) always fits.
    interject_max_hint: usize,
}

/// The outcome of `Lobe::fire_decision` — whether the token fired, plus the gate states for the
/// observe trace (`suppressed_settle` / `in_refractory` / `gate_pass`).
struct FireOutcome {
    fired: bool,
    suppressed: bool,
    in_refractory: bool,
    gate: bool,
}

impl<'a> Lobe<'a> {
    pub fn new(engine: &'a ActiveBackend, n_ctx: u32) -> Result<Self> {
        // Two sequences (0 = observation stream, 1 = interjection scratch) over a UNIFIED KV cache:
        // one shared pool of n_ctx cells so seq 0 gets the full n_ctx (without unified the cache
        // partitions per sequence and seq 0 dies at n_ctx/2 — #6). The interjection on seq 1 borrows
        // transiently, covered by RESET_MARGIN.
        let session = engine.session(SessionConfig {
            n_ctx,
            n_batch: 2048,
            n_seq_max: 2,
            kv_unified: true,
        })?;

        // Resolve gemma-4 chat turn-boundary tokens once. gemma-4 uses <|turn> to OPEN a turn and
        // <turn|> to CLOSE it — NOT gemma-2/3's <start_of_turn>/<end_of_turn> (absent from the vocab).
        let eot = engine.special_token("<turn|>"); // turn close = the model's reply is done
        let sot = engine.special_token("<|turn>"); // turn open  = model started a new turn

        Ok(Self {
            engine,
            session,
            last_logits: Vec::new(),
            pos: 0,
            stream_index: 0,
            recent: VecDeque::with_capacity(RECENT_TOKENS),
            signal: Signal::Surprisal,
            identifiers_only: false,
            n_ctx: n_ctx as i32,
            evict: EvictMode::Reset,
            n_keep: 0,
            preamble: Vec::new(),
            recent_ids: VecDeque::new(),
            context_ids: Vec::new(),
            keep_recent: 4096,
            settle: 0,
            refractory: 0,
            refractory_period: 0,
            resets: 0,
            eot,
            sot,
            interjection: None,
            interject_mode: InterjectMode::Context,
            ask_mode: AskMode::Passage, // control
            novelty_mode: NoveltyMode::Fresh, // control
            memory: InterjectionMemory::default(),
            debug: crate::trace::DebugCfg::default(),
            interject_temp: 0.7, // default: the fixation fix (see set_interject_sampling / main)
            interject_top_p: 0.95,
            rng_state: 0x9E3779B97F4A7C15, // fixed seed → reproducible runs (overridden via set_seed)
            fire_rng: 0x2545F4914F6CDD1D, // independent trigger-RNG stream (re-seeded in set_seed)
            fire_softness: 0.0, // default: deterministic hard threshold (no stochastic firing)
            gen_in_flight: false,
            gen_pos: 0,
            pending_gen_tok: None,
            gen_out: String::new(),
            gen_produced: 0,
            gen_max: 0,
            interject_max_hint: 96,
        })
    }

    /// Set the structured-observability detail level (`--debug-log`). Has no effect unless a tracing
    /// subscriber is installed; it only bounds how big the per-event dumps get.
    pub fn set_debug(&mut self, cfg: crate::trace::DebugCfg) {
        self.debug = cfg;
    }

    /// Configure interjection sampling (experiment). `temp <= 0` = greedy (default). Applies ONLY to
    /// interjection generation; observation scoring is always exact greedy/argmax off real logits.
    pub fn set_interject_sampling(&mut self, temp: f32, top_p: f32) {
        self.interject_temp = temp;
        self.interject_top_p = top_p;
    }

    /// Re-seed BOTH RNG streams from one seed: the interjection sampler (`rng_state`) and the
    /// stochastic-firing RNG (`fire_rng`). Constant by default → reproducible runs; seed from entropy
    /// (`--random-seed`) for non-determinism. The interjection sampler always affects aside *content*;
    /// the firing RNG only matters when `--fire-softness > 0` (else the trigger is the deterministic
    /// hard threshold). `fire_rng` is derived via a splitmix64 step so the two streams are independent
    /// and decorrelated; both guard the xorshift64* `0` fixed point.
    pub fn set_seed(&mut self, seed: u64) {
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        self.rng_state = s;
        let mut z = s.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        self.fire_rng = if z == 0 { 0x2545F4914F6CDD1D } else { z };
    }

    /// Set the stochastic-firing softness (z-units). `0` (default) = deterministic hard threshold;
    /// `> 0` makes firing probabilistic (`sigmoid((z - threshold)/softness)`) off the seeded
    /// `fire_rng`, so which tokens fire varies under `--random-seed` and reproduces under `--seed`.
    pub fn set_fire_softness(&mut self, softness: f32) {
        self.fire_softness = softness;
    }

    /// Decide whether surprisal `z` crosses the firing threshold. `fire_softness <= 0` → the exact
    /// deterministic hard threshold (`z >= z_threshold`), drawing no randomness. `> 0` → a
    /// PROBABILISTIC fire, `P = sigmoid((z - z_threshold)/softness)`, sampled from the seeded
    /// `fire_rng`. The surprisal value itself is always the exact read off the forward pass; only the
    /// fire/no-fire *decision* near the threshold is softened.
    fn crosses(&mut self, z: f32, z_threshold: f32) -> bool {
        if self.fire_softness <= 0.0 {
            return z >= z_threshold;
        }
        let p = 1.0 / (1.0 + (-(z - z_threshold) / self.fire_softness).exp());
        next_unit_f32(&mut self.fire_rng) < p
    }

    /// The firing decision, shared by `observe` and `step`. Advances the settle (#6, post-reset
    /// suppression) and refractory counters, applies the identifier gate (#4) and the — possibly
    /// stochastic — threshold crossing, and on a fire arms the refractory cooldown and snapshots the
    /// delta span (tokens since the last fire). Returns whether the token fired. `text` is the
    /// just-decoded token's detok; `stats` is the running baseline (read-only here).
    ///
    /// Ordering matters: `crosses` is evaluated LAST (it may draw the firing RNG) so randomness is
    /// consumed only once the cheap deterministic gates pass — keeping the draw sequence stable and
    /// reproducible per seed.
    fn fire_decision(
        &mut self,
        text: &str,
        z: f32,
        z_threshold: f32,
        stats: &crate::stats::Welford,
    ) -> FireOutcome {
        let suppressed = self.settle > 0;
        self.settle = self.settle.saturating_sub(1);
        let in_refractory = self.refractory > 0;
        self.refractory = self.refractory.saturating_sub(1);
        let gate = !self.identifiers_only || looks_like_identifier(text);
        let fired = !suppressed
            && !in_refractory
            && gate
            && stats.count() > stats.warmup()
            && self.crosses(z, z_threshold);
        if fired {
            self.refractory = self.refractory_period;
            self.memory.snapshot_span();
        }
        FireOutcome { fired, suppressed, in_refractory, gate }
    }

    /// Hint for the max interjection length (tokens) — sizes the cap+reset roll margin (#6 / fused).
    pub fn set_interject_max_hint(&mut self, m: usize) {
        self.interject_max_hint = m;
    }

    /// Cap+reset roll margin (FUSED_CACHE_GO_NOGO §4a): how far below `n_ctx` seq 0 must roll so that
    /// an interjection's full CONCURRENT KV footprint fits in the unified pool — the context-mode ask
    /// (delta span ≤ MAX_SPAN_TOKENS + up to 2 prior interjections of novelty memory + framing) PLUS
    /// the generated tokens PLUS seq-0's growth during the (deferred-roll) generation. Sized to that
    /// peak so total occupancy never overruns; clamped so seq 0 still gets a usable window.
    fn roll_margin(&self) -> i32 {
        let m = self.interject_max_hint as i32;
        // ask = span(≤128) + novelty(2·m) + framing(~128); + gen(m) + seq0-growth-during-gen(m).
        // Total above the fork ≈ 256 + 4·m; keep generous so the exact pre-fork check rarely fires.
        (MAX_SPAN_TOKENS as i32 + 128 + 5 * m)
            .min(self.n_ctx * 3 / 4)
            .max(RESET_MARGIN)
    }

    /// Build a debug payload for a logit vector: summary stats (max, argmax, entropy) + top-K tokens
    /// with logit and probability. Only called behind an `enabled!` guard, so it's free when off.
    fn logits_debug(&self, logits: &[f32], k: usize) -> serde_json::Value {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|&v| (v - max).exp()).sum();
        let entropy: f32 = logits
            .iter()
            .map(|&v| {
                let p = (v - max).exp() / sum;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        let argmax = idx.first().copied().unwrap_or(0);
        let top: Vec<_> = idx
            .iter()
            .take(k)
            .map(|&i| {
                serde_json::json!({
                    "id": i,
                    "tok": self.detok(Token(i as i32)),
                    "logit": logits[i],
                    "prob": (logits[i] - max).exp() / sum,
                })
            })
            .collect();
        serde_json::json!({
            "n_vocab": logits.len(),
            "max_logit": max,
            "argmax_id": argmax,
            "argmax_tok": self.detok(Token(argmax as i32)),
            "entropy_nats": entropy,
            "topk": top,
        })
    }

    /// The full n_vocab logit vector as a JSON array string (huge; only when `--debug-full-logits`).
    fn logits_full_json(&self, logits: &[f32]) -> String {
        serde_json::to_string(logits).unwrap_or_default()
    }

    /// The COMPLETE live context the observer is attending to, as text — the verbatim framing
    /// (preamble: BOS + the pinned `<|turn>system…<turn|>` sink + open model turn) followed by ALL
    /// stream tokens currently in seq 0 (`context_ids`, uncapped). This is what every context-dumping
    /// diagnostic emits, so a trace shows *all* the tokens in context, not a truncated recent window.
    /// Specials render as text (`detok`) so the turn markers are visible.
    fn full_context_text(&self) -> String {
        self.preamble
            .iter()
            .chain(self.context_ids.iter())
            .map(|&t| self.detok(t))
            .collect()
    }

    /// Choose how interjections see context: forked-full-context (`Context`) or snippet (#7 nuance).
    pub fn set_interject_mode(&mut self, mode: InterjectMode) {
        self.interject_mode = mode;
    }

    /// EXPERIMENT: set the context-mode ask framing (H2) and novelty framing (H4). Defaults are the
    /// control variants (`Passage` / `Fresh`); these only change behavior when set off-control.
    pub fn set_ask_mode(&mut self, ask: AskMode, novelty: NoveltyMode) {
        self.ask_mode = ask;
        self.novelty_mode = novelty;
    }

    /// Set the near-duplicate interjection suppression threshold (Jaccard word overlap; 0 = off).
    pub fn set_dedup(&mut self, threshold: f32) {
        self.memory.set_dedup(threshold);
    }

    /// Record an emitted interjection as novelty memory for the next ask (see
    /// `InterjectionMemory::record` — the primary anti-fixation mechanism).
    pub fn record_interjection(&mut self, text: &str) {
        self.memory.record(text);
    }

    /// Is `text` novel vs the recently-emitted interjections? (Opt-in dedup backstop; always true
    /// when dedup is off. See `InterjectionMemory::is_novel`.)
    pub fn interjection_is_novel(&self, text: &str) -> bool {
        self.memory.is_novel(text)
    }

    /// EARLY streaming dedup: will an in-flight interjection be a duplicate once its opening stem is
    /// known? Lets a frontend abort before rendering. See `InterjectionMemory::doomed`.
    pub fn interjection_doomed(&self, partial: &str) -> bool {
        self.memory.doomed(partial)
    }

    /// Is there enough of an in-flight interjection to decide novelty yet? See
    /// `InterjectionMemory::decidable`.
    pub fn interjection_decidable(&self, partial: &str) -> bool {
        self.memory.decidable(partial)
    }

    /// Abort an in-flight streaming interjection (a frontend decided not to show it): drop the
    /// scratch sequence and clear the generation state. Observation (seq 0) is untouched.
    pub fn abort_interjection(&mut self) -> Result<()> {
        self.session.clear_seq(GEN_SEQ as u32)?;
        self.interjection = None; // timesliced state machine (headless)
        // fused state (TUI)
        self.gen_in_flight = false;
        self.pending_gen_tok = None;
        self.gen_out.clear();
        self.gen_produced = 0;
        Ok(())
    }

    /// Configure the pluggable trigger signal (#4) and the identifier/entity firing gate.
    pub fn set_signal(&mut self, signal: Signal, identifiers_only: bool) {
        self.signal = signal;
        self.identifiers_only = identifiers_only;
    }

    /// Post-fire refractory period (tokens): how long the observer stays quiet after remarking, so
    /// it doesn't obsess over the same salient thing while it lingers in the window. 0 disables.
    pub fn set_refractory(&mut self, period: usize) {
        self.refractory_period = period;
    }

    /// Configure cap + reset (#6): eviction mode and how many recent stream tokens to replay on a
    /// reset. The pinned prefix `n_keep` is captured separately in `prime`. Call before `prime`.
    pub fn set_eviction(&mut self, evict: EvictMode, keep_recent: usize) {
        self.evict = evict;
        self.keep_recent = keep_recent.max(1);
    }

    /// Resets performed so far (validation / TUI status).
    pub fn resets(&self) -> u64 {
        self.resets
    }

    /// Append a committed stream token id to the rolling reset window AND to the full live-context
    /// record (`context_ids`, uncapped — the complete seq-0 stream content for diagnostics).
    fn push_recent_id(&mut self, tok: Token) {
        self.recent_ids.push_back(tok);
        while self.recent_ids.len() > self.keep_recent {
            self.recent_ids.pop_front();
        }
        self.context_ids.push(tok);
    }

    /// Append an observed token's text to the rolling recent-context window.
    fn remember(&mut self, text: &str) {
        self.recent.push_back(text.to_string());
        while self.recent.len() > RECENT_TOKENS {
            self.recent.pop_front();
        }
    }

    /// Tokenize text. `add_bos` should be true only for the very first call (the system
    /// prompt / preamble), false for stream continuations.
    pub fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<Token>> {
        self.engine.tokenize(text, add_bos)
    }

    /// Render a single token to a string, KEEPING special-token markers (`<...>`). Used for the
    /// surprisal/observation path and RAG tool-call parsing, where the markers matter.
    pub fn detok(&self, tok: Token) -> String {
        self.engine.detok(tok, Detok::Text)
    }

    /// Like `detok`, but SUPPRESSES special/control tokens (renders them empty). Used for generated
    /// interjection output so a stray control token can never leak into the user-facing text.
    fn detok_gen(&self, tok: Token) -> String {
        self.engine.detok(tok, Detok::Plain)
    }

    /// Decode a single token into the KV cache and capture the resulting next-token
    /// distribution. This is the one primitive both observe and generate are built on.
    fn decode_one(&mut self, tok: Token) -> Result<()> {
        // #6: cap + reset. Roll over BEFORE the cache physically fills; the margin also reserves
        // slots for an in-flight interjection/RAG prefill that shares the unified cache (GEN_SEQ).
        if self.evict == EvictMode::Reset {
            let margin = self.roll_margin();
            if self.pos >= self.n_ctx - margin {
                self.roll()?;
            }
        }
        self.session.decode(&[Decode {
            token: tok,
            pos: self.pos,
            seq: 0,
            logits: true,
        }])?;
        // Copy out the next-token distribution (the only logits-enabled entry, index 0).
        self.last_logits.clear();
        self.last_logits.extend_from_slice(self.session.logits(0));

        self.pos += 1;
        Ok(())
    }

    /// Batched prefill of `toks` onto sequence 0, computing logits only for the final token (which
    /// become `last_logits`); advances `pos` by `toks.len()`. Chunked to the 512-token batch
    /// capacity. Used by `prime` and `roll` — NOT `decode_one`, so the reset guard can't re-enter.
    fn prefill_seq0(&mut self, toks: &[Token]) -> Result<()> {
        if toks.is_empty() {
            return Ok(());
        }
        let cap = 512usize; // session batch capacity (BATCH_CAP)
        let n = toks.len();
        for start in (0..n).step_by(cap) {
            let end = (start + cap).min(n);
            let is_final_chunk = end == n;
            let batch: Vec<Decode> = toks[start..end]
                .iter()
                .enumerate()
                .map(|(i, &t)| Decode {
                    token: t,
                    pos: self.pos + i as i32,
                    seq: 0,
                    // logits only on the very last token of the whole replay
                    logits: is_final_chunk && (start + i == n - 1),
                })
                .collect();
            self.session.decode(&batch)?;
            if is_final_chunk {
                // the final token's logits sit at its offset within this last batch
                let last_off = batch.len() - 1;
                self.last_logits.clear();
                self.last_logits
                    .extend_from_slice(self.session.logits(last_off));
            }
            self.pos += (end - start) as i32;
        }
        Ok(())
    }

    /// Prime the context with preamble tokens (system prompt etc.). Records them for replay on a
    /// reset and captures the pinned-prefix length. No scoring.
    pub fn prime(&mut self, tokens: &[Token]) -> Result<()> {
        self.preamble = tokens.to_vec();
        self.n_keep = tokens.len() as i32;
        // Keep pinned prefix + rolling window comfortably inside the context (leaving room for an
        // interjection's concurrent footprint); clamp + warn if the configured window is too large.
        let margin = self.roll_margin();
        let room = (self.n_ctx - self.n_keep - margin).max(1) as usize;
        if self.keep_recent > room {
            eprintln!(
                "[lobe] keep_recent {} too large for n_ctx {} (n_keep {}); clamping to {}",
                self.keep_recent, self.n_ctx, self.n_keep, room
            );
            self.keep_recent = room;
        }
        self.recent_ids = VecDeque::with_capacity(self.keep_recent);
        self.context_ids.clear(); // the stream part of seq 0 starts empty (preamble is separate)
        self.prefill_seq0(tokens)
    }

    /// Cap + reset (#6): clear sequence 0 and rebuild it from the pinned preamble plus the rolling
    /// recent-token window, then continue. Sequence 1 (interjection scratch) is untouched; the
    /// Welford baseline and `stream_index` are global and survive the reset.
    fn roll(&mut self) -> Result<()> {
        let t0 = std::time::Instant::now();
        let pos_before = self.pos;
        let window = self.recent_ids.len();
        self.session.clear_seq(0)?;
        let mut replay = self.preamble.clone();
        replay.extend(self.recent_ids.iter().copied());
        let replay_len = replay.len();
        self.pos = 0;
        self.last_logits.clear();
        self.prefill_seq0(&replay)?;
        // The rebuilt seq-0 stream content is exactly the replayed window, so the full-context record
        // tracks it (then grows again as new tokens stream in).
        self.context_ids = self.recent_ids.iter().copied().collect();
        self.settle = RESET_SETTLE;
        self.resets += 1;
        // Window-slide observability: a reset cleared seq 0 and rebuilt sink + recent window. At INFO
        // we also dump the reconstructed context split into the two parts, so a trace can SEE that the
        // opening framing is preserved intact: `framing` = the verbatim-replayed preamble (the BOS +
        // `<|turn>system…<turn|>` sink — CONSTANT across every reset), `window` = the rolling stream
        // tokens (the ONLY part that slides). Gated by `enabled!` so it's free when no subscriber is on.
        let dump = tracing::enabled!(target: "lobe::roll", tracing::Level::INFO);
        let framing = if dump {
            self.preamble.iter().map(|&t| self.detok(t)).collect::<String>()
        } else {
            String::new()
        };
        let window_text = if dump {
            self.recent_ids.iter().map(|&t| self.detok(t)).collect::<String>()
        } else {
            String::new()
        };
        tracing::info!(
            target: "lobe::roll", kind = "window_slide",
            reset_index = self.resets, stream_index = self.stream_index as u64,
            pos_before = pos_before as i64, pos_after = self.pos as i64, n_keep = self.n_keep as i64,
            recent_window = window as u64, replay_len = replay_len as u64, n_ctx = self.n_ctx as i64,
            latency_us = t0.elapsed().as_micros() as u64,
            framing = %framing, window = %window_text,
            "window_slide"
        );
        Ok(())
    }

    /// Observe one stream token: score it against the *previous* step's distribution,
    /// then feed it so the next step is conditioned on it.
    ///
    /// `stats` is the running baseline; it is updated with this token's surprisal.
    pub fn observe(
        &mut self,
        tok: Token,
        stats: &mut crate::stats::Welford,
        z_threshold: f32,
        topk: usize,
    ) -> Result<Step> {
        // If we have no prior distribution (very first stream token with an empty
        // preamble), feed it and report a neutral step.
        if self.last_logits.is_empty() {
            let text = self.detok(tok);
            self.remember(&text);
            self.memory.push_span(&text);
            self.decode_one(tok)?;
            self.push_recent_id(tok);
            let idx = self.stream_index;
            self.stream_index += 1;
            return Ok(Step {
                stream_index: idx,
                token_text: text,
                surprisal: 0.0,
                entropy: 0.0,
                z: 0.0,
                fired: false,
                trigger: None,
            });
        }

        // Both metrics are read off the same distribution (the one that predicted `tok`):
        // surprisal = -ln P(tok); entropy = H of the whole distribution. We z-score whichever
        // signal is active and feed THAT to the baseline, but report both for inspection.
        let surprisal = self.surprisal_of(tok);
        let entropy = self.entropy_of();
        let fire_value = match self.signal {
            Signal::Surprisal => surprisal,
            Signal::Entropy => entropy,
        };
        let z = stats.z(fire_value);
        stats.update(fire_value);

        let text = self.detok(tok);
        self.remember(&text);
        self.memory.push_span(&text); // grow the delta-since-last-fire buffer; current token ends the span

        // Settle (#6 post-reset) + refractory + identifier gate (#4) + threshold crossing, and on a
        // fire arm the cooldown and snapshot the delta span — see fire_decision.
        let FireOutcome { fired, suppressed, in_refractory, gate } =
            self.fire_decision(&text, z, z_threshold, stats);

        let trigger = if fired {
            let expected = self.top_k(topk);
            Some(Trigger {
                stream_index: self.stream_index,
                token_text: text.clone(),
                surprisal,
                entropy,
                z,
                expected,
            })
        } else {
            None
        };

        // --- observability: capture the PREDICTING distribution (`last_logits`, which scored this
        // token) before `decode_one` overwrites it; time the decode; emit a per-token observe event
        // and a richer trigger event on a fire. All gated by `enabled!` → free when --debug-log off.
        let obs_debug = tracing::enabled!(target: "lobe::observe", tracing::Level::DEBUG);
        let obs_trace = tracing::enabled!(target: "lobe::observe", tracing::Level::TRACE);
        let trig_on = fired && tracing::enabled!(target: "lobe::trigger", tracing::Level::INFO);
        let logit_dump = (obs_trace || trig_on)
            .then(|| self.logits_debug(&self.last_logits, self.debug.topk).to_string());
        let full_logits = (trig_on && self.debug.full_logits)
            .then(|| self.logits_full_json(&self.last_logits));
        let pos_before = self.pos;

        // Feed it so subsequent observation is conditioned on what actually arrived.
        let dec_t = std::time::Instant::now();
        self.decode_one(tok)?;
        let decode_us = dec_t.elapsed().as_micros() as u64;
        self.push_recent_id(tok);

        if obs_debug {
            tracing::debug!(
                target: "lobe::observe", kind = "observe",
                stream_index = self.stream_index as u64, token = %text, token_id = tok.0 as i64,
                pos_before = pos_before as i64, pos_after = self.pos as i64,
                surprisal = surprisal as f64, entropy = entropy as f64, z = z as f64,
                signal = ?self.signal, baseline_mean = stats.mean() as f64,
                baseline_std = stats.std() as f64, fired, suppressed_settle = suppressed,
                in_refractory, gate_pass = gate, decode_us,
                logits = logit_dump.as_deref().unwrap_or(""),
                context = if obs_trace { self.full_context_text() } else { String::new() },
                "observe"
            );
        }
        if trig_on {
            tracing::info!(
                target: "lobe::trigger", kind = "trigger",
                stream_index = self.stream_index as u64, token = %text, token_id = tok.0 as i64,
                pos = pos_before as i64, surprisal = surprisal as f64, entropy = entropy as f64,
                z = z as f64, baseline_mean = stats.mean() as f64, baseline_std = stats.std() as f64,
                delta_span = %self.memory.last_span, logits = logit_dump.as_deref().unwrap_or(""),
                full_logits = full_logits.as_deref().unwrap_or(""),
                context = %self.full_context_text(),
                "trigger"
            );
        }

        let idx = self.stream_index;
        self.stream_index += 1;
        Ok(Step {
            stream_index: idx,
            token_text: text,
            surprisal,
            entropy,
            z,
            fired,
            trigger,
        })
    }

    /// FUSED concurrent forward pass (CONCURRENT_FORWARD_PASS.md — the TUI path). One stream token in,
    /// one `decode` out: the stream token (seq 0) and, if an interjection is generating, its next
    /// token (GEN_SEQ) are co-batched into a SINGLE kernel launch — so observation never stalls while
    /// the lobe "thinks" (weights are read once; the second sequence is ~free, decode is
    /// bandwidth-bound). Returns the observation plus the interjection's progress this tick.
    ///
    /// Does NOT touch the headless path (`observe()` + blocking `interject()`), which keeps the
    /// simpler "interjection attached to its trigger" semantics. Mirror of `observe()`'s scoring/
    /// gating, with the decode fused and a concurrent generation arm.
    pub fn step(
        &mut self,
        stream_tok: Token,
        stats: &mut crate::stats::Welford,
        z_threshold: f32,
        topk: usize,
        interject_max: usize,
    ) -> Result<StepOutcome> {
        // Neutral first token (no prior distribution): feed it, no scoring, no interjection.
        if self.last_logits.is_empty() {
            let text = self.detok(stream_tok);
            self.remember(&text);
            self.memory.push_span(&text);
            self.decode_one(stream_tok)?;
            self.push_recent_id(stream_tok);
            let idx = self.stream_index;
            self.stream_index += 1;
            return Ok(StepOutcome {
                step: Step {
                    stream_index: idx,
                    token_text: text,
                    surprisal: 0.0,
                    entropy: 0.0,
                    z: 0.0,
                    fired: false,
                    trigger: None,
                },
                interjection: InterjectStatus::Idle,
            });
        }

        // 1. Cap+reset roll guard, DEFERRED while generating (the forked GEN_SEQ cells must not be
        //    disturbed mid-stream). Margin is widened to cover the concurrent growth during an
        //    interjection: both seq 0 and GEN_SEQ add cells each tick (~2×interject_max + the ask),
        //    and a fire can land just under the threshold — so reserve for the whole interjection.
        self.interject_max_hint = interject_max; // keep the roll margin sized to the actual cap
        if self.evict == EvictMode::Reset && !self.gen_in_flight {
            let margin = self.roll_margin();
            if self.pos >= self.n_ctx - margin {
                self.roll()?;
            }
        }

        // 2. Score the stream token against the PRIOR distribution (last_logits) — exactly observe().
        let surprisal = self.surprisal_of(stream_tok);
        let entropy = self.entropy_of();
        let fire_value = match self.signal {
            Signal::Surprisal => surprisal,
            Signal::Entropy => entropy,
        };
        let z = stats.z(fire_value);
        stats.update(fire_value);

        let text = self.detok(stream_tok);
        self.remember(&text);
        self.memory.push_span(&text);

        // Shared firing decision (settle/refractory/gate/crossing + span capture) — see fire_decision.
        let fired = self.fire_decision(&text, z, z_threshold, stats).fired;
        let trigger = if fired {
            let expected = self.top_k(topk);
            Some(Trigger {
                stream_index: self.stream_index,
                token_text: text.clone(),
                surprisal,
                entropy,
                z,
                expected,
            })
        } else {
            None
        };

        // observability: capture the predicting distribution before the decode overwrites it.
        let trig_on = fired && tracing::enabled!(target: "lobe::trigger", tracing::Level::INFO);
        let logit_dump = trig_on
            .then(|| self.logits_debug(&self.last_logits, self.debug.topk).to_string());

        // 3. FUSED decode: stream token @ seq 0 (idx 0) + pending gen token @ GEN_SEQ (idx 1), both
        //    with logits, in ONE pass. (One decode → both `logits(0)` and `logits(1)` available.)
        let mut batch = vec![Decode {
            token: stream_tok,
            pos: self.pos,
            seq: 0,
            logits: true,
        }];
        let gen_idx = if self.gen_in_flight {
            let t = self
                .pending_gen_tok
                .expect("gen_in_flight implies a pending token");
            batch.push(Decode {
                token: t,
                pos: self.gen_pos,
                seq: GEN_SEQ as u32,
                logits: true,
            });
            Some(1usize)
        } else {
            None
        };
        self.session.decode(&batch)?;

        // 3a. Observation: row 0 is the distribution after the stream token → next tick's last_logits.
        self.last_logits.clear();
        self.last_logits.extend_from_slice(self.session.logits(0));

        // 3b. Generation: the token we just decoded (pending) is committed → emit it; sample the next.
        let mut interjection = InterjectStatus::Idle;
        if let Some(gi) = gen_idx {
            let just = self.pending_gen_tok.take().expect("gen_idx implies pending");
            self.gen_out.push_str(&self.detok_gen(just));
            self.gen_produced += 1;
            self.gen_pos += 1;
            let gen_logits: Vec<f32> = self.session.logits(gi).to_vec();
            let next = sample_topp(
                &gen_logits,
                self.interject_temp,
                self.interject_top_p,
                &mut self.rng_state,
            );
            // Soft length cap: past `gen_max` (interject_max), stop at the next sentence boundary so
            // the aside never ends mid-clause; a hard ceiling (+SLACK) guards against runaway.
            let stop = self.engine.is_eog(next)
                || Some(next) == self.eot
                || Some(next) == self.sot
                || self.gen_produced >= self.gen_max + INTERJECT_SENTENCE_SLACK
                || (self.gen_produced >= self.gen_max && ends_sentence(&self.gen_out));
            if stop {
                self.session.clear_seq(GEN_SEQ as u32)?;
                self.gen_in_flight = false;
                self.gen_produced = 0;
                interjection = InterjectStatus::Done(std::mem::take(&mut self.gen_out));
            } else {
                self.pending_gen_tok = Some(next);
                interjection = InterjectStatus::Working(self.gen_out.clone());
            }
        }

        // 4. Advance seq 0 (mirrors decode_one's pos++ and observe's recent_id push).
        self.push_recent_id(stream_tok);
        self.pos += 1;

        // 5. Start a new interjection on a fresh fire (only if not already generating). The fork +
        //    ask prefill is one separate decode this tick — the single per-interjection stall;
        //    NB the ask can't co-batch with this tick's stream token because `fired` is only known
        //    after the decode above. The per-token gen decodes (3b) are what get fused.
        // `Idle` ⟺ no gen activity this tick (not Working, not just-finished Done) ⟺ safe to start.
        // Deferring a start on a Done tick avoids clobbering the finished text in the status enum.
        if fired && matches!(interjection, InterjectStatus::Idle) {
            self.start_fused_interjection(interject_max)?;
            if self.gen_in_flight {
                interjection = InterjectStatus::Started;
            }
        }

        if trig_on {
            tracing::info!(
                target: "lobe::trigger", kind = "trigger", fused = true,
                stream_index = self.stream_index as u64, token = %text, token_id = stream_tok.0 as i64,
                pos = (self.pos - 1) as i64, surprisal = surprisal as f64, entropy = entropy as f64,
                z = z as f64, delta_span = %self.memory.last_span,
                logits = logit_dump.as_deref().unwrap_or(""),
                "trigger"
            );
        }

        let idx = self.stream_index;
        self.stream_index += 1;
        Ok(StepOutcome {
            step: Step {
                stream_index: idx,
                token_text: text,
                surprisal,
                entropy,
                z,
                fired,
                trigger,
            },
            interjection,
        })
    }

    /// Decode `toks` onto sequence `seq` starting at `start_pos`, computing logits only for the
    /// final token, and return a copy of those logits (length `n_vocab`). Used for both the
    /// scratch-sequence prefill and the single-token generation steps. `toks` must be non-empty
    /// and fit the batch (interjection prompts are kept well under the 512-token batch).
    fn decode_seq(&mut self, toks: &[Token], start_pos: i32, seq: i32) -> Result<Vec<f32>> {
        debug_assert!(!toks.is_empty(), "decode_seq requires at least one token");
        let last = toks.len() - 1;
        let batch: Vec<Decode> = toks
            .iter()
            .enumerate()
            .map(|(i, &t)| Decode {
                token: t,
                pos: start_pos + i as i32,
                seq: seq as u32,
                logits: i == last,
            })
            .collect();
        self.session.decode(&batch)?;
        Ok(self.session.logits(last).to_vec())
    }

    /// -ln P(tok) under last_logits, via a stable log-sum-exp.
    fn surprisal_of(&self, tok: Token) -> f32 {
        let i = tok.0 as usize;
        let logits = &self.last_logits;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for &v in logits {
            sum += (v - max).exp();
        }
        let logsumexp = max + sum.ln();
        // -ln softmax[i] = logsumexp - logits[i]
        logsumexp - logits[i]
    }

    /// Shannon entropy (nats) of the current next-token distribution: H = -Σ p ln p. High H
    /// means the model is spread thin / uncertain at this position, regardless of which token
    /// actually arrives. Computed via the same stable log-sum-exp as `surprisal_of`.
    fn entropy_of(&self) -> f32 {
        let logits = &self.last_logits;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for &v in logits {
            sum += (v - max).exp();
        }
        let logsumexp = max + sum.ln();
        // p_i = exp(logit_i - logsumexp); -ln p_i = logsumexp - logit_i;  H = Σ p_i (-ln p_i).
        let mut h = 0.0f32;
        for &v in logits {
            let p = (v - logsumexp).exp();
            h += p * (logsumexp - v);
        }
        h
    }

    /// Top-k (text, probability) from last_logits, highest first.
    fn top_k(&self, k: usize) -> Vec<(String, f32)> {
        let logits = &self.last_logits;
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for &v in logits {
            sum += (v - max).exp();
        }
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        idx.into_iter()
            .take(k)
            .map(|i| {
                let p = ((logits[i] - max).exp()) / sum;
                (self.detok(Token(i as i32)), p)
            })
            .collect()
    }

    pub fn position(&self) -> i32 {
        self.pos
    }

    /// KV-occupancy instrumentation (FUSED_CACHE_GO_NOGO §3): seq-0 extent, GEN_SEQ extent (-1 when
    /// reclaimed), and whether a fused interjection is in flight. `gen_pos_max` must sawtooth to -1
    /// after each interjection — a ratchet means a GEN_SEQ leak (§4b).
    pub fn kv_debug(&self) -> (i32, i32, bool) {
        (
            self.session.seq_pos_max(0),
            self.session.seq_pos_max(GEN_SEQ as u32),
            self.gen_in_flight,
        )
    }
}

/// Heuristic: does this token's text look like an identifier or named entity — the kind of
/// "objective" surprise worth flagging (a code identifier, a proper noun, a number-bearing
/// token) rather than a rare-but-irrelevant function word? Used by the `--identifiers-only`
/// firing gate (#4). Deliberately cheap and approximate.
fn looks_like_identifier(text: &str) -> bool {
    let t = text.trim();
    if t.len() < 2 {
        return false;
    }
    // Must be "wordy": only alphanumerics or underscores (no punctuation/whitespace inside).
    if !t.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return false;
    }
    let has_digit = t.chars().any(|c| c.is_ascii_digit());
    let has_underscore = t.contains('_');
    let starts_upper = t.chars().next().is_some_and(char::is_uppercase);
    // code-identifier-ish (snake_case / has a digit) OR proper-noun-ish (Capitalized).
    has_underscore || has_digit || starts_upper
}

/// Drop a leading partial-word fragment from a span of concatenated token pieces. Surprisal fires on
/// subword tokens, so a span cut at a fire boundary can start mid-word: e.g. the trigger `ETY` (start
/// of "ETYMOLOGY") ends one span, leaving the next span to begin with the orphan tail "MOLOGY". gemma
/// renders word-start tokens with a leading space, so if the span's first char is NOT whitespace it
/// began mid-word — skip to the first whitespace boundary so the model sees whole words, not stems.
/// If there's no whitespace at all, return as-is (better a fragment than nothing).
fn word_aligned(s: &str) -> &str {
    match s.chars().next() {
        Some(c) if !c.is_whitespace() => match s.find(char::is_whitespace) {
            Some(i) => &s[i..],
            None => s,
        },
        _ => s,
    }
}

/// True if a partial generation ends at a natural sentence boundary, so a soft length cap can stop
/// here instead of mid-clause. Tolerant of trailing markdown emphasis and closing quotes/brackets
/// (e.g. `…the void.”` or `…*insists.*`).
fn ends_sentence(s: &str) -> bool {
    let t = s
        .trim_end()
        .trim_end_matches(|c: char| matches!(c, '"' | '\'' | '*' | '_' | '`' | ')' | ']' | '’' | '”' | '»'));
    t.ends_with(|c: char| matches!(c, '.' | '!' | '?' | '…'))
}


/// Index of the maximum logit, as a token (greedy argmax).
fn argmax(logits: &[f32]) -> Token {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    Token(best as i32)
}

/// Temperature + top-p (nucleus) sampling over a logit vector. `temp <= 0` falls back to greedy
/// argmax. Used ONLY on the interjection generation path (the experiment: break greedy's verbatim
/// collapse and surface latent varied observations); observation scoring stays exact. `rng` is an
/// xorshift64 state advanced in place — seeded to a constant so a run is reproducible.
fn sample_topp(logits: &[f32], temp: f32, top_p: f32, rng: &mut u64) -> Token {
    if temp <= 0.0 {
        return argmax(logits);
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    let inv_t = 1.0 / temp;
    // Temperature-scaled softmax weights in descending-logit order.
    let weights: Vec<f32> = idx.iter().map(|&i| ((logits[i] - max) * inv_t).exp()).collect();
    let total: f32 = weights.iter().sum();
    // Nucleus: smallest prefix whose cumulative prob ≥ top_p.
    let mut cum = 0.0f32;
    let mut cutoff = weights.len();
    for (j, &w) in weights.iter().enumerate() {
        cum += w / total;
        if cum >= top_p {
            cutoff = j + 1;
            break;
        }
    }
    let nucleus_sum: f32 = weights[..cutoff].iter().sum();
    let r = next_unit_f32(rng) * nucleus_sum;
    let mut acc = 0.0f32;
    for j in 0..cutoff {
        acc += weights[j];
        if r <= acc {
            return Token(idx[j] as i32);
        }
    }
    Token(idx[cutoff - 1] as i32)
}

/// xorshift64* → uniform f32 in [0, 1). Advances `state` in place.
fn next_unit_f32(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    ((x >> 40) as f32) / ((1u32 << 24) as f32)
}
