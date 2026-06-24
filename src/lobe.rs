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
pub use types::{
    AskMode, EvictMode, InterjectMode, LobeConfig, NoveltyMode, Signal, Step, StepOutcome, Trigger,
};

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

// Pure token-selection + text-shaping helpers; `pub(crate) use` so siblings reach them via `super::`.
mod sampling;
pub(crate) use sampling::{argmax, ends_sentence, next_unit_f32, sample_topp, trim_snippet, word_aligned};

// The scoring functional-core: pure functions of a logit slice (surprisal/entropy/top-k).
mod scoring;
pub(crate) use scoring::{entropy_of, surprisal_of, top_k};

// The fused forward pass: the imperative-shell boundary around the shared decode (Lane → logits).
mod fused;

// Shared test doubles (the mock Backend/Session) — compiled only under test.
#[cfg(test)]
mod testutil;

// The observation engine (`observe`/`step` + firing + logit dumps) — a big `impl Lobe`.
mod observe;

// Backend abstraction (docs/BACKEND.md): the observer talks only to these traits + types, never to
// a concrete inference engine. `ActiveBackend` is the cfg-selected impl (llama today, candle later).
use crate::backend::{ActiveBackend, Backend, Decode, Detok, Session, SessionConfig, Token};

// The loaded model + tokenizer now lives behind the backend abstraction: `ActiveBackend`
// (= `backend::llama::LlamaBackend` or `backend::candle::CandleBackend`, per cargo feature).
// Load it with `ActiveBackend::load(path, gpu_layers, verbose)`; it plays the old `Engine` role.

// Generic over the backend `B` (the `Backend`/`Session` trait pair), defaulting to the cfg-selected
// `ActiveBackend` so every caller (`main`, the frontends) writes plain `Lobe` and is unaffected. The
// default + the trait seam are what let a test inject a mock backend with scripted logits.
pub struct Lobe<'a, B: Backend = ActiveBackend> {
    /// The backend (model + tokenizer), borrowed for tokenize/detok/is_eog/special-token lookups.
    engine: &'a B,
    /// The inference session (KV cache + decode). All forward passes go through this.
    session: B::Session<'a>,
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

impl<'a, B: Backend> Lobe<'a, B> {
    pub fn new(engine: &'a B, n_ctx: u32, cfg: LobeConfig) -> Result<Self> {
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

        // Distribute the config into the per-concern sub-structs (the construction is the only place
        // they're configured — no post-`new` setters, so the ordering hazard with `prime` is gone).
        let mut firing = Firing::default();
        firing.set_signal(cfg.signal, cfg.identifiers_only);
        firing.set_refractory(cfg.refractory);
        firing.set_softness(cfg.fire_softness);
        let icfg = InterjectConfig {
            mode: cfg.interject_mode,
            ask_mode: cfg.ask_mode,
            novelty_mode: cfg.novelty_mode,
            temp: cfg.interject_temp,
            top_p: cfg.interject_top_p,
            max_hint: cfg.interject_max,
        };
        let mut window = StreamWindow::default();
        window.set_eviction(cfg.evict, cfg.keep_recent);
        let mut memory = InterjectionMemory::default();
        memory.set_dedup(cfg.dedup);

        let mut lobe = Self {
            engine,
            session,
            last_logits: Vec::new(),
            pos: 0,
            stream_index: 0,
            recent: VecDeque::with_capacity(RECENT_TOKENS),
            firing,
            n_ctx: n_ctx as i32,
            window,
            eot,
            sot,
            icfg,
            memory,
            gen: GenState::default(),
            debug: cfg.debug,
        };
        // Seed: leave the fixed reproducible defaults unless an entropy seed was supplied.
        if let Some(seed) = cfg.seed {
            lobe.seed_rngs(seed);
        }
        Ok(lobe)
    }

    /// Re-seed BOTH RNG streams from one seed: the interjection sampler (`gen.rng`) and the
    /// stochastic-firing RNG (`firing.fire_rng`). Constant by default → reproducible runs; seeded
    /// from entropy (`--non-deterministic`) otherwise. The interjection sampler always affects aside
    /// *content*; the firing RNG only matters when `fire_softness > 0` (else the trigger is the
    /// deterministic hard threshold). The firing seed is derived via a splitmix64 step so the two
    /// streams are independent and decorrelated; both guard the xorshift64* `0` fixed point.
    fn seed_rngs(&mut self, seed: u64) {
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        self.gen.rng = s;
        let mut z = s.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        self.firing.seed_rng(if z == 0 { 0x2545F4914F6CDD1D } else { z });
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

    /// The COMPLETE live context the observer is attending to, as text — the verbatim framing
    /// (preamble: BOS + the pinned `<|turn>system…<turn|>` sink + open model turn) followed by ALL
    /// stream tokens currently in seq 0 (`context_ids`, uncapped). This is what every context-dumping
    /// diagnostic emits, so a trace shows *all* the tokens in context, not a truncated recent window.
    /// Specials render as text (`detok`) so the turn markers are visible.
    fn full_context_text(&self) -> String {
        self.window.full_ids().map(|&t| self.detok(t)).collect()
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
    /// `InterjectionMemory::decidable`. (Private: only the shared `advance_reveal` machine needs it.)
    fn interjection_decidable(&self, partial: &str) -> bool {
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

    /// Decode `toks` onto sequence `seq` starting at `start_pos`, computing logits only for the
    /// final token, and return a copy of those logits (length `n_vocab`). Used for both the
    /// scratch-sequence prefill and the single-token generation steps. `toks` must be non-empty
    /// and fit the batch (interjection prompts are kept well under the 512-token batch).
    fn decode_seq(&mut self, toks: &[Token], start_pos: i32, seq: i32) -> Result<Vec<f32>> {
        debug_assert!(!toks.is_empty(), "decode_seq requires at least one token");
        // Chunk at the session batch capacity, exactly like prefill_seq0 — an ask can exceed 512
        // tokens (a long delta span + novelty memory + a #8 recall block), and a single oversized
        // decode would overrun the batch ("Insufficient Space of 512"). Logits only on the very last.
        const CAP: usize = 512; // = backend BATCH_CAP
        let n = toks.len();
        let mut last_logits = Vec::new();
        for start in (0..n).step_by(CAP) {
            let end = (start + CAP).min(n);
            let is_final = end == n;
            let batch: Vec<Decode> = toks[start..end]
                .iter()
                .enumerate()
                .map(|(i, &t)| Decode {
                    token: t,
                    pos: start_pos + (start + i) as i32,
                    seq: seq as u32,
                    logits: is_final && (start + i == n - 1),
                })
                .collect();
            self.session.decode(&batch)?;
            if is_final {
                last_logits = self.session.logits(batch.len() - 1).to_vec();
            }
        }
        Ok(last_logits)
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
