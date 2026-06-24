//! The firing decision (#4) — when a scored token "fires" a trigger. Owns the firing config and
//! state (the active signal, the identifier gate, the post-fire refractory cooldown, and the
//! stochastic-firing RNG), pure of any inference/KV state. The `Lobe` core keeps the per-reset
//! `settle` suppression and the delta-span snapshot (window / memory concerns); `Firing::decide`
//! handles the refractory, the identifier gate, and the (possibly stochastic) threshold crossing.

use super::{next_unit_f32, Signal};

/// Firing config + state: the pluggable trigger signal, the identifier gate, the refractory
/// cooldown, and the stochastic-firing RNG. Pure (no session/KV).
pub(crate) struct Firing {
    /// Active trigger signal (#4): which scalar is z-scored against the running baseline.
    pub signal: Signal,
    /// Gate firing to identifier/entity-like tokens (`--identifiers-only`).
    identifiers_only: bool,
    /// Post-fire refractory countdown: after the observer remarks, it stays quiet for
    /// `period` tokens so it doesn't obsess over the same salient thing while it lingers in the
    /// window. Counts down each observed token; reset to `period` on each fire.
    refractory: usize,
    period: usize,
    /// Independent xorshift64 RNG for the PROBABILISTIC firing decision (decorrelated from the
    /// interjection sampler so trigger draws and aside draws don't interfere).
    fire_rng: u64,
    /// Softness of the stochastic firing sigmoid, in z-units (`--fire-softness`). `<= 0` = the
    /// deterministic hard threshold (`z >= z_threshold`, the default). `> 0` = fire with probability
    /// `sigmoid((z - z_threshold)/softness)` — so triggers vary run-to-run under `--random-seed`.
    softness: f32,
}

impl Default for Firing {
    fn default() -> Self {
        Self {
            signal: Signal::Surprisal,
            identifiers_only: false,
            refractory: 0,
            period: 0,
            fire_rng: 0x2545F4914F6CDD1D,
            softness: 0.0,
        }
    }
}

impl Firing {
    /// Configure the pluggable trigger signal (#4) and the identifier/entity firing gate.
    pub fn set_signal(&mut self, signal: Signal, identifiers_only: bool) {
        self.signal = signal;
        self.identifiers_only = identifiers_only;
    }

    /// Post-fire refractory period (tokens): how long the observer stays quiet after remarking. 0 off.
    pub fn set_refractory(&mut self, period: usize) {
        self.period = period;
    }

    /// Set the stochastic-firing softness (z-units; `<= 0` = deterministic hard threshold).
    pub fn set_softness(&mut self, softness: f32) {
        self.softness = softness;
    }

    /// Re-seed the firing RNG (called from `Lobe::set_seed` alongside the interjection sampler).
    pub fn seed_rng(&mut self, seed: u64) {
        self.fire_rng = seed;
    }

    /// The firing computation, shared by `observe` and `step`. Ticks the refractory counter, applies
    /// the identifier gate (#4) and the — possibly stochastic — threshold crossing, and on a fire
    /// re-arms the refractory cooldown. `suppressed` (post-reset settle, owned by the window) and
    /// `warm` (baseline past warmup) are passed in. Returns `(fired, in_refractory, gate)`; the
    /// caller snapshots the delta span on a fire.
    ///
    /// Ordering matters: `crosses` is evaluated LAST (it may draw the firing RNG) so randomness is
    /// consumed only once the cheap deterministic gates pass — keeping the draw sequence stable and
    /// reproducible per seed.
    pub fn decide(
        &mut self,
        text: &str,
        z: f32,
        z_threshold: f32,
        warm: bool,
        suppressed: bool,
    ) -> (bool, bool, bool) {
        let in_refractory = self.refractory > 0;
        self.refractory = self.refractory.saturating_sub(1);
        let gate = !self.identifiers_only || looks_like_identifier(text);
        let fired = !suppressed && !in_refractory && gate && warm && self.crosses(z, z_threshold);
        if fired {
            self.refractory = self.period;
        }
        (fired, in_refractory, gate)
    }

    /// Does `z` cross the threshold? Deterministic `z >= z_threshold` when `softness <= 0`; otherwise
    /// a stochastic draw with probability `sigmoid((z - z_threshold)/softness)`.
    fn crosses(&mut self, z: f32, z_threshold: f32) -> bool {
        if self.softness <= 0.0 {
            return z >= z_threshold;
        }
        let p = 1.0 / (1.0 + (-(z - z_threshold) / self.softness).exp());
        next_unit_f32(&mut self.fire_rng) < p
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
