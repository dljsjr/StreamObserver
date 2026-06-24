//! The observation engine: the heart of the lobe. `observe` (score the next stream token against the
//! previous step's distribution, then teacher-force it) and the fused concurrent `step` (co-batch the
//! stream token with an in-flight interjection token), plus the firing decision they share, the
//! surprisal/entropy/top-k scoring read off `last_logits`, and the logit-vector debug dumps. The KV
//! primitives (`decode_one`/`roll`/...) and the per-concern sub-structs stay in the `lobe` root; this
//! module is `impl Lobe` over them.

use super::*;
use crate::backend::Backend;
use anyhow::Result;

/// The outcome of `Lobe::fire_decision` — whether the token fired, plus the gate states for the
/// observe trace (`suppressed_settle` / `in_refractory` / `gate_pass`).
struct FireOutcome {
    fired: bool,
    suppressed: bool,
    in_refractory: bool,
    gate: bool,
}

impl<B: Backend> Lobe<'_, B> {
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

    /// Build a debug payload for a logit vector: summary stats (max, argmax, entropy) + top-K tokens
    /// with logit and probability. Only called behind an `enabled!` guard, so it's free when off.
    pub(crate) fn logits_debug(&self, logits: &[f32], k: usize) -> serde_json::Value {
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
        let surprisal = surprisal_of(&self.last_logits, tok);
        let entropy = entropy_of(&self.last_logits);
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
            let expected = top_k(&self.last_logits, topk)
                .into_iter()
                .map(|(id, p)| (self.detok(Token(id as i32)), p))
                .collect();
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
        retrieve: &mut crate::retrieval::RetrieveFn,
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
        let surprisal = surprisal_of(&self.last_logits, stream_tok);
        let entropy = entropy_of(&self.last_logits);
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
            let expected = top_k(&self.last_logits, topk)
                .into_iter()
                .map(|(id, p)| (self.detok(Token(id as i32)), p))
                .collect();
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

        // 3. FUSED decode (the imperative shell, `fused::decode_lanes`): the stream lane (seq 0) and,
        //    if an interjection is in flight, the generation lane (GEN_SEQ) ride in ONE decode. The
        //    lanes are the injected "token sources"; the per-lane logits come back for interpretation.
        let mut lanes = vec![fused::Lane { token: stream_tok, pos: self.pos, seq: 0 }];
        if self.gen.in_flight {
            let pending = self.gen.pending.expect("gen_in_flight implies a pending token");
            lanes.push(fused::Lane { token: pending, pos: self.gen.pos, seq: GEN_SEQ as u32 });
        }
        let logits = fused::decode_lanes(&mut self.session, &lanes)?;

        // 3a. Observation: lane 0's distribution becomes next tick's `last_logits`.
        self.last_logits.clear();
        self.last_logits.extend_from_slice(&logits[0]);

        // 3b. Generation: if a gen lane rode along, advance it from its lane's logits (interject.rs).
        let mut interjection = match logits.get(1) {
            Some(gen_logits) => self.advance_fused_gen(gen_logits)?,
            None => InterjectStatus::Idle,
        };

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
            // #8 RAG (option E): retrieve on the surprising entity and weave the hit into the aside IN
            // VOICE. `retrieve` is a no-op (→ None) unless `--rag` is on, so a plain run is unchanged.
            // Only this query embed + the ask-prefill stall; the aside itself streams concurrently (3b).
            let recalled = retrieve(&text);
            self.start_fused_interjection(interject_max, recalled.as_deref())?;
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
}

#[cfg(test)]
mod tests {
    use crate::backend::Token;
    use crate::lobe::testutil::MockBackend;
    use crate::lobe::{Lobe, LobeConfig};
    use crate::stats::Welford;
    use anyhow::Result;

    // The very first stream token has no prior distribution to score against → a neutral step.
    #[test]
    fn first_token_is_neutral() -> Result<()> {
        let backend = MockBackend { n_vocab: 8 };
        let mut lobe = Lobe::new(&backend, 2048, LobeConfig::default())?;
        let mut stats = Welford::new(2, 0);
        let step = lobe.observe(Token(3), &mut stats, 3.0, 5)?;
        assert_eq!(step.surprisal, 0.0);
        assert!(!step.fired);
        assert_eq!(step.stream_index, 0);
        assert_eq!(lobe.position(), 1); // the token was still fed into the cache
        Ok(())
    }

    // Each observed token advances the KV position and the stream index.
    #[test]
    fn observe_advances_position_and_index() -> Result<()> {
        let backend = MockBackend { n_vocab: 8 };
        let mut lobe = Lobe::new(&backend, 2048, LobeConfig::default())?;
        let mut stats = Welford::new(2, 0);
        for i in 0..5 {
            let step = lobe.observe(Token(i % 4), &mut stats, 3.0, 5)?;
            assert_eq!(step.stream_index, i as usize);
        }
        assert_eq!(lobe.position(), 5);
        Ok(())
    }

    // A token far above the warmed baseline fires a trigger carrying the model's expectations.
    #[test]
    fn high_surprisal_fires_after_warmup() -> Result<()> {
        let backend = MockBackend { n_vocab: 8 };
        let mut lobe = Lobe::new(&backend, 2048, LobeConfig::default())?;
        let mut stats = Welford::new(2, 0);
        // Warm the baseline with low-id (low-surprisal) tokens; none should fire.
        for &id in &[0, 1, 0, 1, 0] {
            assert!(!lobe.observe(Token(id), &mut stats, 3.0, 5)?.fired);
        }
        // A high-id token sits far up the ramp → a big surprisal spike → fires.
        let spike = lobe.observe(Token(7), &mut stats, 3.0, 5)?;
        assert!(spike.fired);
        let trigger = spike.trigger.expect("a fired step carries a trigger");
        assert!(trigger.surprisal > 5.0); // ~7.5 on the ramp for token id 7
        assert!(!trigger.expected.is_empty()); // top-k expectations populated
        Ok(())
    }

    // Entropy is the same for every ramp distribution, so the entropy signal never spikes — a guard
    // that the pluggable signal is actually wired to the firing decision.
    #[test]
    fn entropy_signal_does_not_fire_on_constant_distribution() -> Result<()> {
        let backend = MockBackend { n_vocab: 8 };
        let cfg = LobeConfig {
            signal: crate::lobe::Signal::Entropy,
            ..LobeConfig::default()
        };
        let mut lobe = Lobe::new(&backend, 2048, cfg)?;
        let mut stats = Welford::new(2, 0);
        for &id in &[0, 1, 7, 2, 7, 0, 7] {
            assert!(!lobe.observe(Token(id), &mut stats, 3.0, 5)?.fired);
        }
        Ok(())
    }

    // Cap+reset (#6): a small context rolls before it fills, and observation continues across resets.
    #[test]
    fn cap_reset_rolls_and_continues() -> Result<()> {
        let backend = MockBackend { n_vocab: 8 };
        let cfg = LobeConfig {
            keep_recent: 64,
            ..LobeConfig::default()
        };
        let mut lobe = Lobe::new(&backend, 512, cfg)?;
        let preamble = lobe.tokenize("sink", true)?;
        lobe.prime(&preamble)?;
        let mut stats = Welford::new(2, 0);
        for i in 0..400 {
            lobe.observe(Token(i % 5), &mut stats, 3.0, 5)?;
        }
        assert!(lobe.resets() >= 1); // it rolled at least once
        assert!(lobe.position() < 512); // pos sawtooths below n_ctx, never overruns
        Ok(())
    }
}
