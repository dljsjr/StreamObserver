//! Running baseline for surprisal, with two modes.
//!
//! The whole point (from the design discussion): an off-the-shelf small model has an
//! *uncalibrated* sense of what's surprising, and absolute surprisal varies wildly by
//! stream. So we don't threshold raw surprisal — we threshold a z-score against a
//! running mean/variance, so spikes are relative to the stream's own texture.
//!
//! Two baselines:
//! - **Global** (Welford, `adapt_window == 0`): mean/variance over the *whole* stream so far.
//!   Stable, but it never habituates — a sustained surprising region (e.g. a long catalog of
//!   chapter titles) keeps spiking forever because the all-time baseline barely moves.
//! - **Adaptive** (EWMA, `adapt_window > 0`): exponentially-weighted mean/variance with an
//!   effective window of ~`adapt_window` tokens. This *habituates*: a sustained surprising
//!   texture rapidly becomes the new normal (z collapses toward 0), so the observer stops
//!   obsessing over it; only genuine novelty — a spike *above* the elevated recent texture —
//!   breaks through. This is sensory adaptation, and it's the right model for a reflex observer.

pub struct Welford {
    count: u64,
    mean: f32,
    /// Welford sum of squared deviations (global mode only).
    m2: f32,
    /// EWMA variance (adaptive mode only).
    var_ewma: f32,
    warmup: u64,
    /// EWMA smoothing factor; `Some` → adaptive, `None` → global Welford.
    alpha: Option<f32>,
}

impl Welford {
    /// `adapt_window == 0` → global Welford; `> 0` → adaptive EWMA with that effective window
    /// (standard `alpha = 2 / (window + 1)`).
    pub fn new(warmup: u64, adapt_window: usize) -> Self {
        let alpha = (adapt_window > 0).then(|| 2.0 / (adapt_window as f32 + 1.0));
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            var_ewma: 0.0,
            warmup,
            alpha,
        }
    }

    pub fn update(&mut self, x: f32) {
        self.count += 1;
        match self.alpha {
            Some(a) => {
                // EWMA mean + variance (West's incremental form). `mean` is the pre-update value,
                // matching `z(x)` being read before `update(x)`.
                let delta = x - self.mean;
                self.mean += a * delta;
                self.var_ewma = (1.0 - a) * (self.var_ewma + a * delta * delta);
            }
            None => {
                let delta = x - self.mean;
                self.mean += delta / self.count as f32;
                let delta2 = x - self.mean;
                self.m2 += delta * delta2;
            }
        }
    }

    pub fn variance(&self) -> f32 {
        match self.alpha {
            Some(_) => self.var_ewma,
            None => {
                if self.count < 2 {
                    0.0
                } else {
                    self.m2 / (self.count - 1) as f32
                }
            }
        }
    }

    pub fn std(&self) -> f32 {
        self.variance().sqrt()
    }

    /// z-score of x against the current baseline. 0 until we have a usable std.
    pub fn z(&self, x: f32) -> f32 {
        let s = self.std();
        if s <= 1e-6 {
            0.0
        } else {
            (x - self.mean) / s
        }
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn warmup(&self) -> u64 {
        self.warmup
    }

    pub fn mean(&self) -> f32 {
        self.mean
    }
}
