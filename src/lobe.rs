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
use std::collections::VecDeque;

/// How many of the most-recently-observed token texts to keep as rolling context for the
/// chat-framed interjection prompt. Small on purpose: enough to situate the observer, but it
/// keeps the (one-shot) interjection prompt well under the decode batch size.
const RECENT_TOKENS: usize = 48;
/// Scratch KV-cache sequence id used for interjection generation. The observation stream lives
/// on sequence 0; generation runs entirely on this one and is discarded, so it never pollutes
/// observation. Requires the context to be built with `n_seq_max >= 2`.
const GEN_SEQ: i32 = 1;

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

// The public vocabulary (CLI enums + per-token result types) lives in its own module; re-exported
// here so existing `crate::lobe::{Signal, Step, ...}` paths hold for `main` and the frontends.
mod types;
pub use types::{AskMode, EvictMode, InterjectMode, NoveltyMode, Signal, Step, StepOutcome, Trigger};

// #8 RAG lives in its own module (the result types, the rag() pass, the tool-call parser). `pub mod`
// so `rag()`'s public return type (`RagOutcome`) is reachable; `Source` is re-exported for `main`.
pub mod rag;
pub use rag::Source;

// The interjection machinery (the generation side of observe → generate → resume) lives in its own
// module. `InterjectState` is `pub(crate)` (the `interjection` field below holds it); `InterjectStatus`
// (the `StepOutcome` field + the frontends) is re-exported. `InterjectStep` stays inside the module —
// it only appears in `interject_step`'s signature there, with no external consumer.
mod interject;
use interject::{GenState, InterjectConfig};
pub use interject::InterjectStatus;

// Interjection anti-fixation memory (recent asides + delta span + dedup) lives in its own module —
// pure state, no session. `Lobe` holds one and delegates the dedup/novelty queries to it.
mod novelty;
use novelty::InterjectionMemory;

// The firing decision (#4): trigger signal + identifier gate + refractory + stochastic-fire RNG.
// Pure of inference state; `Lobe::fire_decision` orchestrates it with the window's `settle` + memory.
mod firing;
use firing::Firing;

// StreamingLLM cap+reset window (#6): pinned-prefix sink + rolling recent-id window + reset counters.
// Owns the data + pure bookkeeping; `Lobe::roll`/`prime` do the session-side KV rebuild around it.
mod window;
use window::StreamWindow;

// Pure token-selection + text-shaping helpers (argmax, top-p sampling, RNG, span/sentence guards).
// `pub(crate) use` re-export so the sibling modules keep reaching them via `super::` unchanged.
mod sampling;
pub(crate) use sampling::{argmax, ends_sentence, next_unit_f32, sample_topp, word_aligned};

// Backend abstraction (docs/BACKEND.md): the observer talks only to these traits + types, never to
// a concrete inference engine. `ActiveBackend` is the cfg-selected impl (llama today, candle later).
use crate::backend::{ActiveBackend, Backend, Decode, Detok, Session, SessionConfig, Token};

// The loaded model + tokenizer now lives behind the backend abstraction: `ActiveBackend`
// (= `backend::llama::LlamaBackend` or `backend::candle::CandleBackend`, per cargo feature).
// Load it with `ActiveBackend::load(path, gpu_layers, verbose)`; it plays the old `Engine` role.

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
    /// Firing decision (#4): the trigger signal, identifier gate, refractory cooldown, stochastic-fire
    /// RNG + softness. Pure of inference state; see `firing::Firing`.
    firing: Firing,
    /// Context size the cache was built with (positions 0..n_ctx-1 are valid).
    n_ctx: i32,
    /// StreamingLLM cap+reset window (#6): the pinned-prefix sink (preamble), the rolling recent-id
    /// window, the full-context record, and the reset/settle counters. Pure state (no session); see
    /// `window::StreamWindow`. `roll`/`prime` do the KV rebuild around it.
    window: StreamWindow,
    /// Turn-boundary token ids (gemma chat). Used to stop interjection generation cleanly —
    /// `is_eog_token` does NOT reliably flag `<end_of_turn>` in this build, and `<start_of_turn>`
    /// is never EOG but signals the model has begun a new turn (its reply is over).
    eot: Option<Token>,
    sot: Option<Token>,
    /// How interjections are produced (mode, experiment arms, sampling, length hint). Pure config;
    /// see `interject::InterjectConfig`.
    icfg: InterjectConfig,
    /// Interjection anti-fixation memory: recent asides (novelty memory) + the delta-since-last-fire
    /// span + the opt-in dedup backstop. Pure state (no session); see `novelty::InterjectionMemory`.
    memory: InterjectionMemory,
    /// In-flight interjection generation runtime: the timesliced FSM + the fused concurrent cursors
    /// (CONCURRENT_FORWARD_PASS.md, TUI path) + the sampler RNG. See `interject::GenState`.
    gen: GenState,
    /// Structured-observability detail level (`--debug-log`). Inert unless a tracing subscriber is
    /// installed (the `enabled!` guards skip all dump work otherwise).
    debug: crate::trace::DebugCfg,
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
            firing: Firing::default(),
            n_ctx: n_ctx as i32,
            window: StreamWindow::default(),
            eot,
            sot,
            icfg: InterjectConfig::default(),
            memory: InterjectionMemory::default(),
            gen: GenState::default(),
            debug: crate::trace::DebugCfg::default(),
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
        self.icfg.temp = temp;
        self.icfg.top_p = top_p;
    }

    /// Re-seed BOTH RNG streams from one seed: the interjection sampler (`rng_state`) and the
    /// stochastic-firing RNG (`fire_rng`). Constant by default → reproducible runs; seed from entropy
    /// (`--random-seed`) for non-determinism. The interjection sampler always affects aside *content*;
    /// the firing RNG only matters when `--fire-softness > 0` (else the trigger is the deterministic
    /// hard threshold). `fire_rng` is derived via a splitmix64 step so the two streams are independent
    /// and decorrelated; both guard the xorshift64* `0` fixed point.
    pub fn set_seed(&mut self, seed: u64) {
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        self.gen.rng = s;
        let mut z = s.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        self.firing.seed_rng(if z == 0 { 0x2545F4914F6CDD1D } else { z });
    }

    /// Set the stochastic-firing softness (z-units). `0` (default) = deterministic hard threshold;
    /// `> 0` makes firing probabilistic (`sigmoid((z - threshold)/softness)`) off the seeded firing
    /// RNG, so which tokens fire varies under `--random-seed` and reproduces under `--seed`.
    pub fn set_fire_softness(&mut self, softness: f32) {
        self.firing.set_softness(softness);
    }

    /// The firing decision, shared by `observe` and `step`. Advances the post-reset `settle` (#6)
    /// suppression counter (owned here), then delegates the refractory + identifier gate + threshold
    /// crossing to `Firing::decide`; on a fire it snapshots the delta span (memory). `text` is the
    /// just-decoded token's detok; `stats` is the running baseline (read-only here).
    fn fire_decision(
        &mut self,
        text: &str,
        z: f32,
        z_threshold: f32,
        stats: &crate::stats::Welford,
    ) -> FireOutcome {
        let suppressed = self.window.tick_settle();
        let warm = stats.count() > stats.warmup();
        let (fired, in_refractory, gate) = self.firing.decide(text, z, z_threshold, warm, suppressed);
        if fired {
            self.memory.snapshot_span();
        }
        FireOutcome { fired, suppressed, in_refractory, gate }
    }

    /// Hint for the max interjection length (tokens) — sizes the cap+reset roll margin (#6 / fused).
    pub fn set_interject_max_hint(&mut self, m: usize) {
        self.icfg.max_hint = m;
    }

    /// Cap+reset roll margin (FUSED_CACHE_GO_NOGO §4a): how far below `n_ctx` seq 0 must roll so that
    /// an interjection's full CONCURRENT KV footprint fits in the unified pool — the context-mode ask
    /// (delta span ≤ MAX_SPAN_TOKENS + up to 2 prior interjections of novelty memory + framing) PLUS
    /// the generated tokens PLUS seq-0's growth during the (deferred-roll) generation. Sized to that
    /// peak so total occupancy never overruns; clamped so seq 0 still gets a usable window.
    fn roll_margin(&self) -> i32 {
        let m = self.icfg.max_hint as i32;
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
        self.window.full_ids().map(|&t| self.detok(t)).collect()
    }

    /// Choose how interjections see context: forked-full-context (`Context`) or snippet (#7 nuance).
    pub fn set_interject_mode(&mut self, mode: InterjectMode) {
        self.icfg.mode = mode;
    }

    /// EXPERIMENT: set the context-mode ask framing (H2) and novelty framing (H4). Defaults are the
    /// control variants (`Passage` / `Fresh`); these only change behavior when set off-control.
    pub fn set_ask_mode(&mut self, ask: AskMode, novelty: NoveltyMode) {
        self.icfg.ask_mode = ask;
        self.icfg.novelty_mode = novelty;
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
        self.gen.fsm = None; // timesliced state machine (headless)
        // fused state (TUI)
        self.gen.in_flight = false;
        self.gen.pending = None;
        self.gen.out.clear();
        self.gen.produced = 0;
        Ok(())
    }

    /// Configure the pluggable trigger signal (#4) and the identifier/entity firing gate.
    pub fn set_signal(&mut self, signal: Signal, identifiers_only: bool) {
        self.firing.set_signal(signal, identifiers_only);
    }

    /// Post-fire refractory period (tokens): how long the observer stays quiet after remarking, so
    /// it doesn't obsess over the same salient thing while it lingers in the window. 0 disables.
    pub fn set_refractory(&mut self, period: usize) {
        self.firing.set_refractory(period);
    }

    /// Configure cap + reset (#6): eviction mode and how many recent stream tokens to replay on a
    /// reset. The pinned prefix `n_keep` is captured separately in `prime`. Call before `prime`.
    pub fn set_eviction(&mut self, evict: EvictMode, keep_recent: usize) {
        self.window.set_eviction(evict, keep_recent);
    }

    /// Resets performed so far (validation / TUI status).
    pub fn resets(&self) -> u64 {
        self.window.resets()
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
        if self.window.evict == EvictMode::Reset {
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
        // Keep pinned prefix + rolling window comfortably inside the context (leaving room for an
        // interjection's concurrent footprint); the window clamps + warns if it's too large.
        let n_keep = tokens.len() as i32;
        let margin = self.roll_margin();
        let room = (self.n_ctx - n_keep - margin).max(1) as usize;
        self.window.begin_prime(tokens.to_vec(), room, self.n_ctx);
        self.prefill_seq0(tokens)
    }

    /// Cap + reset (#6): clear sequence 0 and rebuild it from the pinned preamble plus the rolling
    /// recent-token window, then continue. Sequence 1 (interjection scratch) is untouched; the
    /// Welford baseline and `stream_index` are global and survive the reset.
    fn roll(&mut self) -> Result<()> {
        let t0 = std::time::Instant::now();
        let pos_before = self.pos;
        let window = self.window.window_len();
        self.session.clear_seq(0)?;
        let replay = self.window.replay_tokens();
        let replay_len = replay.len();
        self.pos = 0;
        self.last_logits.clear();
        self.prefill_seq0(&replay)?;
        // The rebuilt seq-0 stream content is exactly the replayed window, so the window syncs its
        // full-context record to it, arms the post-reset settle, and bumps the reset counter.
        self.window.mark_rolled();
        // Window-slide observability: a reset cleared seq 0 and rebuilt sink + recent window. At INFO
        // we also dump the reconstructed context split into the two parts, so a trace can SEE that the
        // opening framing is preserved intact: `framing` = the verbatim-replayed preamble (the BOS +
        // `<|turn>system…<turn|>` sink — CONSTANT across every reset), `window` = the rolling stream
        // tokens (the ONLY part that slides). Gated by `enabled!` so it's free when no subscriber is on.
        let dump = tracing::enabled!(target: "lobe::roll", tracing::Level::INFO);
        let framing = if dump {
            self.window.preamble.iter().map(|&t| self.detok(t)).collect::<String>()
        } else {
            String::new()
        };
        let window_text = if dump {
            self.window.recent_ids().map(|&t| self.detok(t)).collect::<String>()
        } else {
            String::new()
        };
        tracing::info!(
            target: "lobe::roll", kind = "window_slide",
            reset_index = self.window.resets(), stream_index = self.stream_index as u64,
            pos_before = pos_before as i64, pos_after = self.pos as i64, n_keep = self.window.n_keep as i64,
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
            self.window.push_id(tok);
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
        let fire_value = match self.firing.signal {
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
        self.window.push_id(tok);

        if obs_debug {
            tracing::debug!(
                target: "lobe::observe", kind = "observe",
                stream_index = self.stream_index as u64, token = %text, token_id = tok.0 as i64,
                pos_before = pos_before as i64, pos_after = self.pos as i64,
                surprisal = surprisal as f64, entropy = entropy as f64, z = z as f64,
                signal = ?self.firing.signal, baseline_mean = stats.mean() as f64,
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
            self.window.push_id(stream_tok);
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
        self.icfg.max_hint = interject_max; // keep the roll margin sized to the actual cap
        if self.window.evict == EvictMode::Reset && !self.gen.in_flight {
            let margin = self.roll_margin();
            if self.pos >= self.n_ctx - margin {
                self.roll()?;
            }
        }

        // 2. Score the stream token against the PRIOR distribution (last_logits) — exactly observe().
        let surprisal = self.surprisal_of(stream_tok);
        let entropy = self.entropy_of();
        let fire_value = match self.firing.signal {
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
        let gen_idx = if self.gen.in_flight {
            let t = self
                .gen
                .pending
                .expect("gen_in_flight implies a pending token");
            batch.push(Decode {
                token: t,
                pos: self.gen.pos,
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
            let just = self.gen.pending.take().expect("gen_idx implies pending");
            self.gen.out.push_str(&self.detok_gen(just));
            self.gen.produced += 1;
            self.gen.pos += 1;
            let gen_logits: Vec<f32> = self.session.logits(gi).to_vec();
            let next = sample_topp(
                &gen_logits,
                self.icfg.temp,
                self.icfg.top_p,
                &mut self.gen.rng,
            );
            // Soft length cap: past `gen_max` (interject_max), stop at the next sentence boundary so
            // the aside never ends mid-clause; a hard ceiling (+SLACK) guards against runaway.
            let stop = self.engine.is_eog(next)
                || Some(next) == self.eot
                || Some(next) == self.sot
                || self.gen.produced >= self.gen.max + INTERJECT_SENTENCE_SLACK
                || (self.gen.produced >= self.gen.max && ends_sentence(&self.gen.out));
            if stop {
                self.session.clear_seq(GEN_SEQ as u32)?;
                self.gen.in_flight = false;
                self.gen.produced = 0;
                interjection = InterjectStatus::Done(std::mem::take(&mut self.gen.out));
            } else {
                self.gen.pending = Some(next);
                interjection = InterjectStatus::Working(self.gen.out.clone());
            }
        }

        // 4. Advance seq 0 (mirrors decode_one's pos++ and observe's recent_id push).
        self.window.push_id(stream_tok);
        self.pos += 1;

        // 5. Start a new interjection on a fresh fire (only if not already generating). The fork +
        //    ask prefill is one separate decode this tick — the single per-interjection stall;
        //    NB the ask can't co-batch with this tick's stream token because `fired` is only known
        //    after the decode above. The per-token gen decodes (3b) are what get fused.
        // `Idle` ⟺ no gen activity this tick (not Working, not just-finished Done) ⟺ safe to start.
        // Deferring a start on a Done tick avoids clobbering the finished text in the status enum.
        if fired && matches!(interjection, InterjectStatus::Idle) {
            self.start_fused_interjection(interject_max)?;
            if self.gen.in_flight {
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
            self.gen.in_flight,
        )
    }
}
