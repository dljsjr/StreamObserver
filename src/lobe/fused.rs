//! The fused forward pass — the imperative-shell boundary around the one impure resource (the KV
//! cache). This is the ONLY place that knows "many lanes share a single decode": the concurrent-
//! forward-pass primitive (CONCURRENT_FORWARD_PASS.md). It is stateless — a module of free functions,
//! not a struct — because the state lives in the `Session` (the KV cache) and in the observer/
//! generator records; the pass itself just multiplexes them onto one decode per tick. The "token
//! sources" are injected as arguments (`&[Lane]`) and the per-lane logits returned, so the caller
//! owns all interpretation (scoring vs sampling) — keeping that interpretation a pure functional core.

use crate::backend::{Decode, Session, Token};
use anyhow::Result;

/// One lane's contribution to a fused decode: a token to feed at `pos` on sequence `seq`. (The stream
/// observation lane is seq 0; an in-flight interjection is the generation lane on `GEN_SEQ`.) `logits`
/// requests this lane's next-token distribution be computed — gemma-4's vocab is 262k, so projecting
/// logits is NOT free; a chunked-prefill lane that's only building KV sets it false to skip the
/// projection (only the FINAL ask token of a prefill needs logits, to seed the first reply token).
pub(crate) struct Lane {
    pub token: Token,
    pub pos: i32,
    pub seq: u32,
    pub logits: bool,
}

/// Co-batch every lane into a SINGLE decode (weights read once; extra lanes are ~free on a
/// bandwidth-bound kernel) and return each lane's next-token logits, in lane order. Pure orchestration
/// over the impure `session`; holds no observation/generation state. `lanes` must be non-empty and fit
/// the configured batch. Returns owned logit vectors so the caller can read them after the borrow of
/// `session` ends (and route lane 0 → scoring, lane 1 → sampling, etc.); a lane with `logits: false`
/// gets an empty vector (no projection computed — the caller must not read it).
pub(crate) fn decode_lanes(session: &mut impl Session, lanes: &[Lane]) -> Result<Vec<Vec<f32>>> {
    let batch: Vec<Decode> = lanes
        .iter()
        .map(|l| Decode {
            token: l.token,
            pos: l.pos,
            seq: l.seq,
            logits: l.logits,
        })
        .collect();
    session.decode(&batch)?;
    Ok((0..lanes.len())
        .map(|i| {
            if lanes[i].logits {
                session.logits(i).to_vec()
            } else {
                Vec::new()
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Backend;
    use crate::lobe::testutil::MockBackend;

    // The multiplex: N lanes → one decode → N logit vectors, one per lane, each n_vocab long.
    #[test]
    fn decode_lanes_returns_one_logit_row_per_lane() {
        let backend = MockBackend { n_vocab: 8 };
        let mut session = backend.session(crate::backend::SessionConfig {
            n_ctx: 64,
            n_batch: 64,
            n_seq_max: 2,
            kv_unified: true,
        })
        .unwrap();
        let lanes = vec![
            Lane { token: Token(3), pos: 0, seq: 0, logits: true },
            Lane { token: Token(5), pos: 0, seq: 1, logits: true },
        ];
        let logits = decode_lanes(&mut session, &lanes).unwrap();
        assert_eq!(logits.len(), 2); // one row per lane
        assert!(logits.iter().all(|row| row.len() == 8)); // each is a full distribution
        // Single-lane is the observation-only (non-fused) shape.
        let one = decode_lanes(&mut session, &lanes[..1]).unwrap();
        assert_eq!(one.len(), 1);
        // A logits:false lane builds KV but skips the projection → an empty row the caller skips.
        let mixed = vec![
            Lane { token: Token(2), pos: 1, seq: 0, logits: true },
            Lane { token: Token(4), pos: 1, seq: 1, logits: false },
        ];
        let rows = decode_lanes(&mut session, &mixed).unwrap();
        assert_eq!(rows[0].len(), 8); // requested → full distribution
        assert!(rows[1].is_empty()); // not requested → empty
    }
}
