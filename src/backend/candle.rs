//! Candle backend (the `candle` feature) — SKELETON. The seam is real (it implements `Backend` /
//! `Session`); the gemma-4 E-model forward pass is the bring-up and is `unimplemented!()` for now.
//!
//! Why Candle (see CANDLE_DESIGN.md): owning the forward pass + KV cache is the only way to get all
//! three of — seamless per-layer infinite context, PLE-offload efficiency *with* logit access, and a
//! truly concurrent fused observe+generate (the last three reasons llama.cpp can't satisfy together).
//!
//! Bring-up checklist (what `decode` needs): dual head_dim (256 local / 512 global), Proportional
//! RoPE on global layers, the iSWA layer schedule (4 sliding : 1 global, 512 window), Per-Layer
//! Embeddings (the offloadable tables — Reason 2), GeGLU + RMSNorm, 262k byte-fallback BPE tokenizer.
//! First action: check `candle-transformers` Gemma coverage (gemma-4's features are almost certainly
//! net-new). Add deps then: candle-core, candle-nn, tokenizers (gated by the `candle` feature).

// Skeleton: fields/args are unused until the gemma-4 forward pass is implemented (see the module
// doc / CANDLE_DESIGN.md). Silence the placeholder dead-code; the default `llama` build is unaffected.
#![allow(dead_code, unused_variables)]

use anyhow::{bail, Result};
use std::marker::PhantomData;

use super::{Backend, Decode, Detok, Session, SessionConfig, Token};

const TODO: &str = "Candle backend: gemma-4 forward pass not implemented yet (see CANDLE_DESIGN.md)";

/// Owns the (eventual) Candle model weights + tokenizer.
pub struct CandleBackend {
    // weights: candle gemma-4 model, tokenizer: tokenizers::Tokenizer, device: candle Device, …
}

impl Backend for CandleBackend {
    type Session<'a> = CandleSession<'a>;

    fn load(_model_path: &str, _gpu_layers: u32, _verbose: bool) -> Result<Self> {
        bail!("{TODO}");
    }
    fn n_vocab(&self) -> usize {
        unimplemented!("{TODO}")
    }
    fn tokenize(&self, _text: &str, _add_bos: bool) -> Result<Vec<Token>> {
        unimplemented!("{TODO}")
    }
    fn detok(&self, _token: Token, _mode: Detok) -> String {
        unimplemented!("{TODO}")
    }
    fn is_eog(&self, _token: Token) -> bool {
        unimplemented!("{TODO}")
    }
    fn special_token(&self, _text: &str) -> Option<Token> {
        unimplemented!("{TODO}")
    }
    fn session(&self, _cfg: SessionConfig) -> Result<CandleSession<'_>> {
        unimplemented!("{TODO}")
    }
}

/// Per-layer KV cache + forward-pass state (the heart of the port — see CANDLE_DESIGN.md "KV cache
/// design": local layers windowed, global layers sink+window+shift, all owned here).
pub struct CandleSession<'a> {
    _borrow: PhantomData<&'a CandleBackend>,
}

impl Session for CandleSession<'_> {
    fn decode(&mut self, _batch: &[Decode]) -> Result<()> {
        unimplemented!("{TODO}")
    }
    fn logits(&self, _i: usize) -> &[f32] {
        unimplemented!("{TODO}")
    }
    fn clear_seq(&mut self, _seq: u32) -> Result<()> {
        unimplemented!("{TODO}")
    }
    fn copy_seq(&mut self, _src: u32, _dst: u32) -> Result<()> {
        unimplemented!("{TODO}")
    }
    fn seq_pos_max(&self, _seq: u32) -> i32 {
        -1
    }
}
