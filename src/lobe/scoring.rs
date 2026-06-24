//! The scoring functional-core: pure functions of a logit slice, no `Lobe`/session/tokenizer. These
//! are the arithmetic the observer reads off each forward pass — surprisal (`-ln P(tok)`), entropy of
//! the distribution, and the top-k by probability. Kept free-standing (FUNCTIONAL CORE / imperative
//! shell): the impure decode lives in `fused`/`Session`; this is just math over its output, so it
//! tests with literal arrays and never needs a mock backend. `top_k` returns token *ids* — turning
//! ids into text needs the tokenizer, which is the shell's job (the caller detoks).

use super::Token;

/// `-ln P(tok)` under `logits`, via a stable log-sum-exp. The surprise of the token that arrived.
pub(crate) fn surprisal_of(logits: &[f32], tok: Token) -> f32 {
    let i = tok.0 as usize;
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &v in logits {
        sum += (v - max).exp();
    }
    let logsumexp = max + sum.ln();
    // -ln softmax[i] = logsumexp - logits[i]
    logsumexp - logits[i]
}

/// Shannon entropy (nats) of the distribution: `H = -Σ p ln p`. High H means the model is spread
/// thin / uncertain at this position, regardless of which token actually arrives. Same stable
/// log-sum-exp as `surprisal_of`.
pub(crate) fn entropy_of(logits: &[f32]) -> f32 {
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

/// Top-k `(token_id, probability)` by descending probability. Pure — the caller renders ids to text.
pub(crate) fn top_k(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for &v in logits {
        sum += (v - max).exp();
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.into_iter()
        .take(k)
        .map(|i| (i, (logits[i] - max).exp() / sum))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pure math → tested with literal logit arrays, no Lobe and no mock backend.
    #[test]
    fn surprisal_is_lower_for_likelier_tokens() {
        let logits = [3.0, 0.0, -1.0]; // token 0 is the most likely
        let s0 = surprisal_of(&logits, Token(0));
        let s2 = surprisal_of(&logits, Token(2));
        assert!(s0 < s2);
        assert!(s0 > 0.0); // still costs something
    }

    #[test]
    fn surprisal_of_near_certain_token_is_tiny() {
        let logits = [100.0, 0.0, 0.0]; // token 0 ~ probability 1
        assert!(surprisal_of(&logits, Token(0)) < 1e-3);
    }

    #[test]
    fn entropy_is_max_for_uniform_and_zero_for_peaked() {
        let uniform = [0.0, 0.0, 0.0, 0.0];
        let peaked = [100.0, 0.0, 0.0, 0.0];
        assert!((entropy_of(&uniform) - (4.0f32).ln()).abs() < 1e-4); // ln(n) for n uniform
        assert!(entropy_of(&peaked) < 1e-3);
    }

    #[test]
    fn top_k_is_descending_by_probability_and_normalized() {
        let logits = [1.0, 3.0, 2.0, 0.0];
        let top = top_k(&logits, 2);
        assert_eq!(top[0].0, 1); // highest logit first
        assert_eq!(top[1].0, 2);
        assert!(top[0].1 > top[1].1);
        assert!(top[0].1 < 1.0 && top[0].1 > 0.0); // a probability
    }
}
