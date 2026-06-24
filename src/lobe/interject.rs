//! The interjection machinery: the observe → generate → resume loop's generation side. Holds the
//! in-flight generation state types and the methods that fork seq 0 onto the scratch sequence
//! (`GEN_SEQ`), build the gemma-4 chat-framed ask, and pump reply tokens — the fused concurrent path
//! (`start_fused_interjection`, driven by `Lobe::step`) and the timesliced one-shot path
//! (`interject`/`interject_begin`/`interject_step`, used by headless). The prompts themselves live in
//! `crate::prompt` (the gemma-4 chat-format owner); observation/firing stays in the `lobe` core.

use super::*;
use crate::backend::{Backend, Session};
use anyhow::Result;

/// In-flight interjection generation, pumped one unit per call (`interject_step`) so a frontend
/// event loop never freezes. The observation stream still pauses while it runs (the GPU is serial),
/// but generation streams visibly instead of blocking in one shot. Lives on the scratch sequence
/// `GEN_SEQ`; never touches observation state.
pub(crate) enum InterjectState {
    /// Prompt/ask tokens staged; the next step prefills them on GEN_SEQ starting at `start_pos`
    /// (0 for a fresh snippet prompt, or the forked context length for `Context` mode).
    Prefill {
        toks: Vec<Token>,
        start_pos: i32,
        max: usize,
        /// When the interjection began (for total-latency observability).
        t_start: std::time::Instant,
        /// Prompt token count (carried into Gen for the done event).
        prompt_len: usize,
    },
    /// Greedy-decoding the reply, one token per step.
    Gen {
        pos: i32,
        produced: usize,
        max: usize,
        out: String,
        logits: Vec<f32>,
        t_start: std::time::Instant,
        prompt_len: usize,
    },
}

/// Result of one `interject_step` (the timesliced path; headless `interject()` drives it one-shot).
/// `Working`'s partial is the streaming contract for a live frontend; the one-shot headless caller
/// ignores it (the TUI now uses the fused `step()` instead), hence `allow(dead_code)`.
#[allow(dead_code)]
pub enum InterjectStep {
    /// No interjection in progress.
    Idle,
    /// Still generating; carries the text produced so far (for streaming display).
    Working(String),
    /// Finished; carries the final text.
    Done(String),
}

/// What the interjection generator did during one fused `step()` (CONCURRENT_FORWARD_PASS).
pub enum InterjectStatus {
    /// No interjection generating this tick.
    Idle,
    /// A new interjection began this tick (forked + ask prefilled); no reply token yet.
    Started,
    /// Generating; carries the reply text so far (for live display).
    Working(String),
    /// Finished this tick; carries the final reply text.
    Done(String),
}

/// How interjections are produced — the generation knobs, set once from the CLI. Pure config (no
/// runtime state). Defaults are the converged config (Context mode, control experiment arms, the
/// fixation-fixing sampling temp/top-p).
pub(crate) struct InterjectConfig {
    /// Whether interjections reflect on the full forked context or a re-encoded snippet.
    pub mode: InterjectMode,
    /// EXPERIMENT (templating study): context-mode ask framing (H2) + novelty framing (H4). Both
    /// default to the control variant, so behavior is unchanged unless a flag is set.
    pub ask_mode: AskMode,
    pub novelty_mode: NoveltyMode,
    /// Interjection sampling: temperature + top-p applied ONLY to interjection generation.
    /// `temp <= 0` = greedy argmax. Observation scoring is always exact regardless.
    pub temp: f32,
    pub top_p: f32,
    /// Max interjection length (tokens), used to size the cap+reset roll margin so an interjection's
    /// full concurrent KV footprint (ask + generated tokens + seq-0 growth during gen) always fits.
    pub max_hint: usize,
}

impl Default for InterjectConfig {
    fn default() -> Self {
        Self {
            mode: InterjectMode::Context,
            ask_mode: AskMode::Passage,   // control
            novelty_mode: NoveltyMode::Fresh, // control
            temp: 0.7,                    // the fixation fix (see Lobe::set_interject_sampling)
            top_p: 0.95,
            max_hint: 96,
        }
    }
}

impl Lobe<'_> {
    /// Fork seq 0 onto GEN_SEQ and prefill the interjection ask in ONE decode (the single
    /// per-interjection stall), then seed the fused gen cursors. Subsequent reply tokens co-batch in
    /// `step()`. Mirrors `interject_begin` but drives the fused `gen_*` state instead of the
    /// timesliced `InterjectState`.
    pub(crate) fn start_fused_interjection(&mut self, max: usize) -> Result<()> {
        self.session.clear_seq(GEN_SEQ as u32)?;
        let (toks, start_pos) = match self.icfg.mode {
            InterjectMode::Snippet => {
                let p = self.interject_prompt_snippet();
                (self.tokenize(&p, true)?, 0)
            }
            InterjectMode::Context => {
                // EXACT cap+reset fit check (FUSED_CACHE_GO_NOGO §4a): tokenize the ask FIRST, then —
                // if forking here + the ask + the generated tokens + a little headroom would overrun
                // the unified pool — roll seq 0 NOW (no gen is in flight at start). After the roll
                // `self.pos` is the small pinned-prefix+window, so the fork+ask+gen fits exactly,
                // never an estimate. Re-tokenize after the roll (the span/novelty text is unchanged,
                // but `self.recent`/`last_span` survive the roll so the ask is identical anyway).
                let ask = self.interject_ask_context();
                let mut toks = self.tokenize(&ask, false)?;
                // Footprint above the fork = ask (GEN_SEQ) + gen tokens (GEN_SEQ) + seq-0's growth
                // during the deferred-roll generation (BOTH sequences grow concurrently) ≈
                // ask + 2·(max+SLACK), using the HARD ceiling since the soft cap may overrun by SLACK.
                // Roll now if that won't fit (no gen in flight while starting).
                let gen_ceiling = (max + INTERJECT_SENTENCE_SLACK) as i32;
                if self.evict == EvictMode::Reset
                    && self.pos + toks.len() as i32 + 2 * gen_ceiling + 32 > self.n_ctx
                {
                    self.roll()?;
                    let ask = self.interject_ask_context();
                    toks = self.tokenize(&ask, false)?;
                }
                self.session.copy_seq(0, GEN_SEQ as u32)?;
                (toks, self.pos)
            }
        };
        let logits = self.decode_seq(&toks, start_pos, GEN_SEQ)?;
        let first = sample_topp(
            &logits,
            self.icfg.temp,
            self.icfg.top_p,
            &mut self.rng_state,
        );
        self.gen_pos = start_pos + toks.len() as i32;
        self.gen_out.clear();
        self.gen_produced = 0;
        self.gen_max = max;
        // Empty interjection (first token is a stop): don't enter the fused loop.
        if self.engine.is_eog(first) || Some(first) == self.eot || Some(first) == self.sot {
            self.session.clear_seq(GEN_SEQ as u32)?;
            self.gen_in_flight = false;
            self.pending_gen_tok = None;
        } else {
            self.pending_gen_tok = Some(first);
            self.gen_in_flight = true;
        }
        Ok(())
    }

    /// SNIPPET mode prompt: a self-contained gemma-4 chat turn carrying only the last
    /// `RECENT_TOKENS` of stream text. The model sees just this snippet, not its real context.
    ///
    /// The surprisal spike is purely the HARNESS trigger (it decides WHEN to speak) — the aside is
    /// NOT conditioned on it. The model simply discusses the passage the text has reached; the
    /// surprising token is never named or referenced.
    ///
    /// NB: phrasing chosen empirically — gemma-4-E2B is sensitive to it. Variants that put the
    /// recent text last, or say "reply with only the sentence", made the model *continue* the recent
    /// text instead of reacting. Keep the recent block first and the instruction after it.
    fn interject_prompt_snippet(&self) -> String {
        // Snippet mode can't fork the live KV, so it carries the recent window as text. (Modality
        // lives in the system prompt; length in interject_max.) Template: prompt::interject_prompt_snippet.
        let recent: String = self.recent.iter().map(String::as_str).collect();
        crate::prompt::interject_prompt_snippet(word_aligned(&recent).trim())
    }

    /// CONTEXT mode ask: appended AFTER the forked full context (no BOS — it continues that
    /// context). Closes the in-progress turn, then a user turn that spotlights the *delta* — the
    /// span of text since the last fire — plus NOVELTY MEMORY (what it just said), then opens the
    /// model turn.
    fn interject_ask_context(&self) -> String {
        // The surprisal spike is purely the HARNESS trigger — it decides WHEN to interject, nothing
        // more. The aside is NOT conditioned on "what was surprising"; the model simply discusses the
        // current chunk (the delta span — the text since the last fire). The surprising token is never
        // named or referenced. NOVELTY MEMORY (the model's own recent asides) is the anti-fixation
        // mechanism: on a sustained region the span alone would draw the same standing attractor fire
        // after fire, so we show the model what it just said and ask for a fresh angle — without ever
        // mentioning surprise. The full forked context stays available, so a genuine "X because of Y
        // from earlier" connection can still form.
        let span = word_aligned(&self.memory.last_span).trim();

        // Novelty memory (H4). Off = omit it entirely; otherwise show the last 1–2 asides, with the
        // framing chosen by `novelty_mode` (Fresh = content novelty [control]; Form = form novelty).
        let novelty_block = if self.icfg.novelty_mode == NoveltyMode::Off {
            String::new()
        } else {
            let noted: String = self
                .memory
                .recent(2)
                .map(|s| format!("- {}", s.trim()))
                .collect::<Vec<_>>()
                .join("\n");
            if noted.is_empty() {
                String::new()
            } else {
                format!("\n\nYou recently said:\n{noted}")
            }
        };
        // The closing instruction's novelty clause depends on `novelty_mode`.
        let novelty_clause = match self.icfg.novelty_mode {
            NoveltyMode::Fresh => " — but find a fresh angle, not one you've already made", // control
            NoveltyMode::Form => " — but vary your rhythm and how you begin, not just the subject",
            NoveltyMode::Off => "",
        };
        // Ask framing (H2): Passage = "comment on this quoted span" (control); Continuous = pick up an
        // ongoing thread. Template: prompt::interject_ask_context.
        crate::prompt::interject_ask_context(
            span,
            &novelty_block,
            novelty_clause,
            self.icfg.ask_mode == AskMode::Continuous,
        )
    }

    /// One-shot, blocking chat-framed *observation* about the surprising token (used by headless,
    /// where there's no event loop to keep responsive). Built on the streaming `interject_begin`/
    /// `interject_step` machinery so there's a single generation code path.
    ///
    /// Generation runs entirely on the scratch sequence `GEN_SEQ`; sequence 0, `self.pos`, and
    /// `self.last_logits` are never touched, so observation is byte-identical with or without it.
    /// It's a *reframe*, not a continuation — the (instruct-tuned) observer reacts to the stream
    /// rather than extends it. Greedy, capped at `max`, stopping at any turn boundary.
    pub fn interject(&mut self, surprising: &str, max: usize) -> Result<String> {
        self.interject_begin(surprising, max)?;
        loop {
            match self.interject_step()? {
                InterjectStep::Done(text) => return Ok(text),
                InterjectStep::Idle => return Ok(String::new()),
                InterjectStep::Working(_) => {}
            }
        }
    }

    /// Begin a streaming interjection: clear the scratch sequence and stage the prompt/ask.
    /// Cheap (the heavy prefill happens on the first `interject_step`), so the trigger tick stays
    /// light. Pump `interject_step` once per frontend tick until it returns `Done`.
    ///
    /// In `Context` mode this forks the observer's full live context (seq 0) onto `GEN_SEQ` with a
    /// cheap cell copy (no recompute) and the ask continues from `self.pos`; in `Snippet` mode it
    /// stages a fresh re-encoded prompt from position 0.
    pub fn interject_begin(&mut self, surprising: &str, max: usize) -> Result<()> {
        let t_start = std::time::Instant::now();
        self.session.clear_seq(GEN_SEQ as u32)?;
        let forked = matches!(self.icfg.mode, InterjectMode::Context);
        // Build the EXACT prompt the model will see (raw model input), keeping the text for the
        // observability event. Context mode forks the live seq-0 KV first (cheap cell copy).
        let prompt_text = match self.icfg.mode {
            InterjectMode::Snippet => self.interject_prompt_snippet(),
            InterjectMode::Context => {
                self.session.copy_seq(0, GEN_SEQ as u32)?;
                self.interject_ask_context()
            }
        };
        let toks = self.tokenize(&prompt_text, !forked)?; // BOS only for the standalone snippet prompt
        let start_pos = if forked { self.pos } else { 0 };

        tracing::info!(
            target: "lobe::interject", kind = "interject_begin",
            stream_index = self.stream_index as u64, mode = ?self.icfg.mode, forked,
            trigger_token = %surprising, start_pos = start_pos as i64,
            prompt_tokens = toks.len() as u64, max = max as u64,
            delta_span = %word_aligned(&self.memory.last_span).trim(),
            // The delta span (above) and the ENTIRE live context (below), dumped separately: the span
            // is what the ask spotlights; `full_context` is the whole forked seq-0 the model reflects
            // over (framing + all stream tokens). Lazily evaluated — only when the trace is enabled.
            full_context = %self.full_context_text(),
            novelty_memory = %self.memory.recent(2)
                .cloned().collect::<Vec<_>>().join(" ||| "),
            prompt = %prompt_text, // the raw model input, verbatim
            "interject_begin"
        );

        self.interjection = Some(InterjectState::Prefill {
            prompt_len: toks.len(),
            toks,
            start_pos,
            max,
            t_start,
        });
        Ok(())
    }

    /// Advance an in-flight interjection by one unit of work: the first call prefills the prompt on
    /// `GEN_SEQ`; each subsequent call greedily decodes one reply token. Returns `Working(partial)`
    /// with the text so far, or `Done(text)` when it stops (turn boundary / EOG / `max`).
    pub fn interject_step(&mut self) -> Result<InterjectStep> {
        match self.interjection.take() {
            None => Ok(InterjectStep::Idle),
            Some(InterjectState::Prefill {
                toks,
                start_pos,
                max,
                t_start,
                prompt_len,
            }) => {
                // Prefill the staged tokens on the scratch sequence (after any forked context),
                // then transition to per-token generation.
                let pre_t = std::time::Instant::now();
                let logits = self.decode_seq(&toks, start_pos, GEN_SEQ)?;
                tracing::debug!(
                    target: "lobe::interject", kind = "interject_prefill",
                    prompt_tokens = prompt_len as u64, start_pos = start_pos as i64,
                    latency_us = pre_t.elapsed().as_micros() as u64,
                    "interject_prefill"
                );
                self.interjection = Some(InterjectState::Gen {
                    pos: start_pos + toks.len() as i32,
                    produced: 0,
                    max,
                    out: String::new(),
                    logits,
                    t_start,
                    prompt_len,
                });
                Ok(InterjectStep::Working(String::new()))
            }
            Some(InterjectState::Gen {
                pos,
                produced,
                max,
                mut out,
                logits,
                t_start,
                prompt_len,
            }) => {
                // Interjection generation: temperature/top-p if configured (experiment), else greedy.
                let tok = sample_topp(
                    &logits,
                    self.icfg.temp,
                    self.icfg.top_p,
                    &mut self.rng_state,
                );
                // Stop at any turn boundary (is_eog alone misses gemma-4's <turn|>; <|turn> means
                // the model has begun a new turn) or the cap. Drop the scratch sequence on finish.
                // Soft length cap (mirrors step()): past `max` stop at the next sentence boundary so
                // the aside never ends mid-clause; a hard ceiling (+SLACK) guards against runaway.
                let stop_reason = if self.engine.is_eog(tok) {
                    Some("eog")
                } else if Some(tok) == self.eot {
                    Some("turn_close")
                } else if Some(tok) == self.sot {
                    Some("turn_open")
                } else if produced >= max + INTERJECT_SENTENCE_SLACK {
                    Some("max")
                } else if produced >= max && ends_sentence(&out) {
                    Some("max_sentence")
                } else {
                    None
                };
                if let Some(reason) = stop_reason {
                    self.session.clear_seq(GEN_SEQ as u32)?;
                    tracing::info!(
                        target: "lobe::interject", kind = "interject_done",
                        stream_index = self.stream_index as u64, stop_reason = reason,
                        produced = produced as u64, prompt_tokens = prompt_len as u64,
                        latency_us = t_start.elapsed().as_micros() as u64,
                        output = %out, "interject_done"
                    );
                    return Ok(InterjectStep::Done(out));
                }
                // Per-gen-token detail (TRACE): the chosen token + its distribution.
                if tracing::enabled!(target: "lobe::interject", tracing::Level::TRACE) {
                    tracing::trace!(
                        target: "lobe::interject", kind = "interject_token",
                        produced = produced as u64, pos = pos as i64, token_id = tok.0 as i64,
                        token = %self.detok_gen(tok),
                        logits = %self.logits_debug(&logits, self.debug.topk),
                        "interject_token"
                    );
                }
                out.push_str(&self.detok_gen(tok));
                let next_logits = self.decode_seq(&[tok], pos, GEN_SEQ)?;
                let partial = out.clone();
                self.interjection = Some(InterjectState::Gen {
                    pos: pos + 1,
                    produced: produced + 1,
                    max,
                    out,
                    logits: next_logits,
                    t_start,
                    prompt_len,
                });
                Ok(InterjectStep::Working(partial))
            }
        }
    }
}
