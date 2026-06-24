//! Shared test doubles for the lobe modules (compiled only under `cfg(test)`). The `Backend`/`Session`
//! seam realized as a fake: no model, no GPU. Its decode always yields the same RAMP distribution
//! `logit[i] = -i`, so surprisal grows smoothly and predictably with the scored token's id
//! (`surprisal(id) ≈ logsumexp + id`) — letting a test warm the baseline with low-id tokens and then
//! spike with a high-id one. Lets the observation engine and the fused pass be tested without weights.

use crate::backend::{Backend, Decode, Detok, Session, SessionConfig, Token};
use anyhow::Result;
use std::collections::HashMap;

pub(crate) struct MockBackend {
    pub n_vocab: usize,
}

/// A fake session: tracks per-seq max position (for `seq_pos_max`/leak checks) and holds the last
/// decode's logits. The ramp is recomputed each decode; the input tokens only move the cursors.
pub(crate) struct MockSession<'a> {
    backend: &'a MockBackend,
    last: Vec<f32>,
    seq_pos: HashMap<u32, i32>,
}

impl Backend for MockBackend {
    type Session<'a>
        = MockSession<'a>
    where
        Self: 'a;
    fn load(_path: &str, _gpu_layers: u32, _verbose: bool) -> Result<Self> {
        Ok(Self { n_vocab: 8 })
    }
    fn n_vocab(&self) -> usize {
        self.n_vocab
    }
    fn tokenize(&self, text: &str, _add_bos: bool) -> Result<Vec<Token>> {
        // Deterministic byte→id mapping; enough to prime a preamble in the cap+reset test.
        Ok(text
            .bytes()
            .map(|b| Token((b as usize % self.n_vocab) as i32))
            .collect())
    }
    fn detok(&self, token: Token, _mode: Detok) -> String {
        format!("t{}", token.0)
    }
    fn is_eog(&self, _token: Token) -> bool {
        false
    }
    fn special_token(&self, _text: &str) -> Option<Token> {
        None
    }
    fn session(&self, _cfg: SessionConfig) -> Result<MockSession<'_>> {
        Ok(MockSession {
            backend: self,
            last: Vec::new(),
            seq_pos: HashMap::new(),
        })
    }
}

impl Session for MockSession<'_> {
    fn decode(&mut self, batch: &[Decode]) -> Result<()> {
        self.last = (0..self.backend.n_vocab).map(|i| -(i as f32)).collect();
        for d in batch {
            let e = self.seq_pos.entry(d.seq).or_insert(-1);
            *e = (*e).max(d.pos);
        }
        Ok(())
    }
    fn logits(&self, _i: usize) -> &[f32] {
        &self.last
    }
    fn clear_seq(&mut self, seq: u32) -> Result<()> {
        self.seq_pos.insert(seq, -1);
        Ok(())
    }
    fn copy_seq(&mut self, _src: u32, _dst: u32) -> Result<()> {
        Ok(())
    }
    fn seq_pos_max(&self, seq: u32) -> i32 {
        self.seq_pos.get(&seq).copied().unwrap_or(-1)
    }
}
