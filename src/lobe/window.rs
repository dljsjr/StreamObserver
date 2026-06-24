//! StreamingLLM cap + reset window (#6) — the bounded-memory-over-infinite-input state. Holds the
//! pinned-prefix sink (the verbatim-replayed preamble), a rolling ring of recent stream-token ids,
//! the full live-context record, and the reset/settle counters. This module owns the DATA and the
//! pure bookkeeping (capping, replay-token assembly, reset accounting); the actual KV rebuild —
//! clearing sequence 0 and re-prefilling — stays in `Lobe::roll`/`prime` because it needs the
//! session. `Lobe` reads `evict`/`n_keep`/`preamble` directly and calls the methods for the rest.

use super::{EvictMode, Token, RESET_SETTLE};
use std::collections::VecDeque;

/// The cap+reset window state (#6). Pure of any session/KV handle.
pub(crate) struct StreamWindow {
    /// Eviction policy.
    pub evict: EvictMode,
    /// Pinned-prefix length: the preamble tokens replayed verbatim on every reset (the StreamingLLM
    /// "sink", except here it carries real content). Set in `prime`.
    pub n_keep: i32,
    /// The preamble tokens, kept for replay on reset.
    pub preamble: Vec<Token>,
    /// Rolling ring of recent STREAM token *ids* — distinct from the `recent` text window, because
    /// detok→retok is not round-trip safe, so we replay the actual ids. Capped at `keep_recent`.
    recent_ids: VecDeque<Token>,
    /// The COMPLETE stream-token content currently in seq 0 (everything decoded since the last reset /
    /// prime — NOT capped, so it can grow to ~n_ctx between resets). The full live context is
    /// `preamble + context_ids`; the context-dumping diagnostics replay this so they show *all* the
    /// tokens the model attends to, not just the recent window. Reset to `recent_ids` on a roll.
    context_ids: Vec<Token>,
    /// How many recent stream tokens to replay after a reset (the rolling window).
    keep_recent: usize,
    /// Post-reset trigger-suppression countdown.
    settle: usize,
    /// Count of resets so far (for the TUI status line / validation).
    resets: u64,
}

impl Default for StreamWindow {
    fn default() -> Self {
        Self {
            evict: EvictMode::Reset,
            n_keep: 0,
            preamble: Vec::new(),
            recent_ids: VecDeque::new(),
            context_ids: Vec::new(),
            keep_recent: 4096,
            settle: 0,
            resets: 0,
        }
    }
}

impl StreamWindow {
    /// Configure cap + reset (#6): eviction mode + how many recent stream tokens to replay on a reset.
    /// The pinned prefix `n_keep` is captured separately in `prime`. Call before `prime`.
    pub fn set_eviction(&mut self, evict: EvictMode, keep_recent: usize) {
        self.evict = evict;
        self.keep_recent = keep_recent.max(1);
    }

    /// Resets performed so far (validation / TUI status).
    pub fn resets(&self) -> u64 {
        self.resets
    }

    /// Tick the post-reset settle counter; returns whether we WERE suppressed before this tick.
    pub fn tick_settle(&mut self) -> bool {
        let suppressed = self.settle > 0;
        self.settle = self.settle.saturating_sub(1);
        suppressed
    }

    /// Append a committed stream-token id to the rolling reset window AND to the full live-context
    /// record (`context_ids`, uncapped — the complete seq-0 stream content for diagnostics).
    pub fn push_id(&mut self, tok: Token) {
        self.recent_ids.push_back(tok);
        while self.recent_ids.len() > self.keep_recent {
            self.recent_ids.pop_front();
        }
        self.context_ids.push(tok);
    }

    /// `prime` bookkeeping (no KV work): pin the preamble, capture `n_keep`, clamp `keep_recent` into
    /// `room` (warning if it was too large for the context), and start the stream content empty. The
    /// caller does the actual KV prefill; `n_ctx` is only for the clamp warning.
    pub fn begin_prime(&mut self, preamble: Vec<Token>, room: usize, n_ctx: i32) {
        self.n_keep = preamble.len() as i32;
        self.preamble = preamble;
        if self.keep_recent > room {
            eprintln!(
                "[lobe] keep_recent {} too large for n_ctx {} (n_keep {}); clamping to {}",
                self.keep_recent, n_ctx, self.n_keep, room
            );
            self.keep_recent = room;
        }
        self.recent_ids = VecDeque::with_capacity(self.keep_recent);
        self.context_ids.clear(); // the stream part of seq 0 starts empty (preamble is separate)
    }

    /// The reset replay sequence: the pinned preamble followed by the rolling recent-id window. This
    /// is exactly what `roll` re-prefills onto a cleared sequence 0.
    pub fn replay_tokens(&self) -> Vec<Token> {
        let mut replay = self.preamble.clone();
        replay.extend(self.recent_ids.iter().copied());
        replay
    }

    /// `roll` bookkeeping (no KV work): the rebuilt seq-0 stream content is exactly the replayed
    /// window, so sync the full-context record to it, arm the post-reset settle suppression, and bump
    /// the reset counter.
    pub fn mark_rolled(&mut self) {
        self.context_ids = self.recent_ids.iter().copied().collect();
        self.settle = RESET_SETTLE;
        self.resets += 1;
    }

    /// Current rolling-window length (number of recent ids), for the window-slide trace.
    pub fn window_len(&self) -> usize {
        self.recent_ids.len()
    }

    /// Iterate the rolling recent-id window (for the window-slide trace dump).
    pub fn recent_ids(&self) -> impl Iterator<Item = &Token> {
        self.recent_ids.iter()
    }

    /// Iterate the full live seq-0 content: pinned preamble followed by all stream ids since the last
    /// reset. What the context-dumping diagnostics replay.
    pub fn full_ids(&self) -> impl Iterator<Item = &Token> {
        self.preamble.iter().chain(self.context_ids.iter())
    }
}
