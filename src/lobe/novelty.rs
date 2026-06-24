//! Interjection novelty/dedup memory — the anti-fixation state, pure (no session/engine). Owns the
//! rolling record of recently-emitted asides (the "novelty memory" the next ask shows the model), the
//! "delta since last fire" span buffer (what each interjection focuses on), and the opt-in
//! near-duplicate dedup (`--dedup`, a backstop). It decides whether an aside is fresh and snapshots
//! the delta span; the `Lobe` core and `interject` module read through it.

use super::MAX_SPAN_TOKENS;
use std::collections::HashSet;
use std::collections::VecDeque;

/// How many recent interjections to compare against for near-duplicate suppression.
const DEDUP_HISTORY: usize = 6;
/// Leading words that define an interjection's "opening stem" for theme dedup. 2 measured best on
/// the catalog (collapses "The repetition. It's stark" with "The repetition. It's a stutter") while
/// `DEDUP_HISTORY`'s short window keeps distant same-opening reactions; higher under-collapses theme
/// recurrence, and the model varies openings enough that 2 didn't merge distinct "The word X" lines.
const DEDUP_OPENING_WORDS: usize = 2;

/// Interjection anti-fixation memory: recent asides + the delta span + dedup config. Pure data and
/// text logic — no inference state, so it never touches the KV cache.
#[derive(Default)]
pub(crate) struct InterjectionMemory {
    /// Recent emitted interjection texts, for near-duplicate suppression and the novelty memory the
    /// next ask shows the model (the observer shouldn't repeat itself while a salient thing lingers).
    recent_interjections: VecDeque<String>,
    /// Token texts observed since the last fire — the "delta" the next interjection focuses on, so
    /// each one reacts to *new* content instead of re-reflecting on the dominant feature of the
    /// whole window. A text buffer (not KV positions), so it's immune to cap+reset / window slide.
    since_last_fire: VecDeque<String>,
    /// The delta span captured at the last fire (what `interject` in Context mode focuses on).
    pub last_span: String,
    /// Jaccard word-overlap threshold above which a new interjection is treated as a repeat and
    /// suppressed. 0 disables dedup.
    dedup_threshold: f32,
}

impl InterjectionMemory {
    /// Set the near-duplicate interjection suppression threshold (Jaccard word overlap; 0 = off).
    pub fn set_dedup(&mut self, threshold: f32) {
        self.dedup_threshold = threshold;
    }

    /// Append an observed token's text to the rolling "delta since last fire" buffer (capped).
    pub fn push_span(&mut self, text: &str) {
        self.since_last_fire.push_back(text.to_string());
        while self.since_last_fire.len() > MAX_SPAN_TOKENS {
            self.since_last_fire.pop_front();
        }
    }

    /// Snapshot the delta span at a fire: `last_span` becomes the text since the last fire, then the
    /// buffer clears so the next interjection focuses only on what arrives next.
    pub fn snapshot_span(&mut self) {
        self.last_span = self.since_last_fire.iter().map(String::as_str).collect();
        self.since_last_fire.clear();
    }

    /// Record an emitted interjection so the NEXT ask can show it as novelty memory
    /// (`interject_ask_context`, the 1b fix). This is the PRIMARY anti-fixation mechanism — it feeds
    /// the model what it just said so it can move on, rather than filtering a duplicate after the
    /// fact. Call on every emitted interjection. (Distinct from `is_novel`, the post-hoc dedup
    /// filter, which only records when `dedup_threshold > 0` and is now a backstop.)
    pub fn record(&mut self, text: &str) {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        self.recent_interjections.push_back(text.to_string());
        while self.recent_interjections.len() > DEDUP_HISTORY {
            self.recent_interjections.pop_front();
        }
    }

    /// The most recent `n` asides, newest first — the novelty memory shown in the next ask.
    pub fn recent(&self, n: usize) -> impl Iterator<Item = &String> {
        self.recent_interjections.iter().rev().take(n)
    }

    /// PURE check (no recording — that's `record`'s job now): is `text` novel vs the recently-emitted
    /// interjections? Returns false (a repeat) if its opening stem matches a recent one, or
    /// char-shingle Jaccard exceeds `dedup_threshold`. This is the OPT-IN backstop filter (default
    /// off, `--dedup 0`); the primary anti-fixation mechanism is the novelty memory in the ask (1b).
    /// Returns true (always novel) when dedup is disabled.
    pub fn is_novel(&self, text: &str) -> bool {
        if self.dedup_threshold <= 0.0 {
            return true;
        }
        let open = opening(text, DEDUP_OPENING_WORDS);
        let sh = shingles(text);
        !self.recent_interjections.iter().any(|prev| {
            opening(prev, DEDUP_OPENING_WORDS) == open
                || jaccard(&sh, &shingles(prev)) > self.dedup_threshold
        })
    }

    /// EARLY dedup for streaming frontends: true once an in-flight interjection's *opening stem* is
    /// known AND already matches a recently-emitted one — i.e. it WILL be a duplicate — so the
    /// frontend can abort and never render it (instead of streaming it live then dropping it at
    /// `Done`). Only with dedup enabled; needs the opening stem complete to judge. (The opening stem
    /// is the dominant repeat signal; the full shingle backstop isn't checkable mid-stream and is
    /// dropped on this path — a fair trade to avoid render-then-delete.)
    pub fn doomed(&self, partial: &str) -> bool {
        if self.dedup_threshold <= 0.0 || !self.opening_complete(partial) {
            return false;
        }
        let open = opening(partial, DEDUP_OPENING_WORDS);
        self.recent_interjections
            .iter()
            .any(|prev| opening(prev, DEDUP_OPENING_WORDS) == open)
    }

    /// True when there's enough of an in-flight interjection to decide novelty: dedup off → always
    /// (stream immediately, no change), else once the opening stem is complete. A streaming frontend
    /// buffers (shows a neutral "thinking…") until this, then reveals if not `doomed`.
    pub fn decidable(&self, partial: &str) -> bool {
        self.dedup_threshold <= 0.0 || self.opening_complete(partial)
    }

    /// Does `text` already contain a full `DEDUP_OPENING_WORDS`-word opening stem?
    fn opening_complete(&self, text: &str) -> bool {
        opening(text, DEDUP_OPENING_WORDS)
            .split(' ')
            .filter(|w| !w.is_empty())
            .count()
            >= DEDUP_OPENING_WORDS
    }
}

/// Character n-gram (shingle) set of a normalized string, for interjection dedup. More robust than
/// word sets when two interjections share a long opening but diverge in the tail (the common case:
/// "The structure… a catalog of mundane things" vs "…a catalog of things") — word-set Jaccard gets
/// diluted by the differing tails, char-shingle Jaccard stays high on the shared span.
fn shingles(s: &str) -> HashSet<String> {
    const N: usize = 4;
    let norm = s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    let chars: Vec<char> = norm.chars().collect();
    if chars.len() < N {
        return std::iter::once(norm).collect();
    }
    chars.windows(N).map(|w| w.iter().collect()).collect()
}

/// First `k` lowercased alphanumeric words of a string (its "opening stem"), for theme dedup.
fn opening(s: &str, k: usize) -> String {
    s.split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .take(k)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Jaccard similarity (|∩| / |∪|) of two shingle sets.
fn jaccard(a: &HashSet<String>, b: &HashSet<String>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f32;
    let union = a.union(b).count() as f32;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}
