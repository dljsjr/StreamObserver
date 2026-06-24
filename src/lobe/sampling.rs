//! Token-selection and text-shaping helpers, pure and free-standing (no `Lobe`/session): greedy
//! argmax, temperature+top-p nucleus sampling (interjection path only — observation scoring stays
//! exact), the xorshift64* uniform RNG, and the span/sentence text guards. Re-exported from the
//! `lobe` root so the sibling modules reach them via `super::`.

use super::Token;

/// Index of the maximum logit, as a token (greedy argmax).
pub(crate) fn argmax(logits: &[f32]) -> Token {
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
pub(crate) fn sample_topp(logits: &[f32], temp: f32, top_p: f32, rng: &mut u64) -> Token {
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
pub(crate) fn next_unit_f32(state: &mut u64) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    ((x >> 40) as f32) / ((1u32 << 24) as f32)
}

/// Drop a leading partial-word fragment from a span of concatenated token pieces. Surprisal fires on
/// subword tokens, so a span cut at a fire boundary can start mid-word: e.g. the trigger `ETY` (start
/// of "ETYMOLOGY") ends one span, leaving the next span to begin with the orphan tail "MOLOGY". gemma
/// renders word-start tokens with a leading space, so if the span's first char is NOT whitespace it
/// began mid-word — skip to the first whitespace boundary so the model sees whole words, not stems.
/// If there's no whitespace at all, return as-is (better a fragment than nothing).
pub(crate) fn word_aligned(s: &str) -> &str {
    match s.chars().next() {
        Some(c) if !c.is_whitespace() => match s.find(char::is_whitespace) {
            Some(i) => &s[i..],
            None => s,
        },
        _ => s,
    }
}

/// A word-aligned head of `s`, at most `max` bytes (back up to the last whitespace within the limit
/// so a word isn't split). Keeps a retrieved chunk to a punchy quote in the #8 recall block.
pub(crate) fn trim_snippet(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    match s[..end].rfind(char::is_whitespace) {
        Some(i) if i > 0 => &s[..i],
        _ => &s[..end],
    }
}

/// True if a partial generation ends at a natural sentence boundary, so a soft length cap can stop
/// here instead of mid-clause. Tolerant of trailing markdown emphasis and closing quotes/brackets
/// (e.g. `…the void.”` or `…*insists.*`).
pub(crate) fn ends_sentence(s: &str) -> bool {
    let t = s
        .trim_end()
        .trim_end_matches(|c: char| matches!(c, '"' | '\'' | '*' | '_' | '`' | ')' | ']' | '’' | '”' | '»'));
    t.ends_with(|c: char| matches!(c, '.' | '!' | '?' | '…'))
}
