# INFINITE_CONTEXT.md — handoff for punch-list item 6

Implement unbounded streaming for the observer via **cap + reset**. This is grounded in the
current `src/lobe.rs` (llama-cpp-2 0.1.150). Read this whole file before touching code.

## Decision: cap + reset (not context-shift)

The textbook StreamingLLM move — `seq_rm` the middle + `seq_add` to shift the tail + let the
K-shift re-rotate RoPE — **does not work on Gemma in llama.cpp**. Gemma 4 E2B uses interleaved
sliding-window attention (iSWA); llama.cpp's compact iSWA cache discards the local-layers' KV
once it leaves the 512 window, irrecoverably, so context-shift / prefix-cache / context-reuse
are all unsupported on it (confirmed by the maintainer on the SWA cache PR). You can only do the
real shift with `--swa-full` (full KV on every layer), which throws away Gemma's memory design
and forbids q4_0 K-cache. See "Upgrade path" at the bottom.

**cap + reset** sidesteps all of that: when the observation sequence fills, clear it, re-prime
the pinned preamble plus a rolling window of recent stream tokens, and continue. It never slides,
so it never needs sinks and never hits the iSWA limitation. It uses only `clear_kv_cache_seq`
(already validated in `interject`) plus the existing prefill path. Works with the compact iSWA
cache and with KV-cache quantization.

### What this does NOT do (set expectations)

It does not extend recall. Tokens evicted at a reset are gone — exactly as they'd be gone under
real StreamingLLM, whose attention sinks carry no semantic content either. The only thing we give
up vs real StreamingLLM is the *no-recompute* property: each reset re-prefills the recent window
(a brief stall), instead of sliding for free. With a large `n_ctx` and a modest recent window,
that stall is sub-second on E2B and paid only every (n_ctx − n_keep − keep_recent) tokens. For a
reflex observer that's a good trade.

## State to add to `Lobe`

```rust
n_ctx: i32,                       // store the value passed to `new` (currently only forwarded)
n_keep: i32,                      // length of the pinned preamble; set in `prime`
preamble: Vec<LlamaToken>,        // the preamble tokens, kept for replay on reset
recent_ids: VecDeque<LlamaToken>, // rolling ring of recent STREAM token IDs, capped at keep_recent
keep_recent: usize,               // window size to replay on reset (e.g. 4096–8192). Make it a CLI arg.
settle: usize,                    // post-reset trigger-suppression counter (see Edge cases)
```

Note `recent_ids` is **token IDs**, distinct from the existing `recent: VecDeque<String>` (which
stays as-is, sized for interjection framing). Do not re-tokenize the texts — detok→retokenize is
not round-trip safe. Keep a parallel ID ring.

## The reset primitive

```rust
/// Re-prime sequence 0 from scratch: pinned preamble + the rolling recent window.
/// Sequence 0's KV is cleared and rebuilt; `pos` and `last_logits` are reset to the new state.
/// `stream_index` and the Welford baseline are NOT touched (they're global / external).
fn roll(&mut self) -> Result<()> {
    self.ctx.clear_kv_cache_seq(Some(0), None, None)?;     // already-validated call shape
    let mut replay = self.preamble.clone();
    replay.extend(self.recent_ids.iter().copied());
    self.pos = 0;
    self.last_logits.clear();
    self.prefill_seq0(&replay)?;                            // batched; sets last_logits + pos
    self.settle = self.warmup_after_reset;                  // suppress triggers briefly (optional)
    Ok(())
}

/// Batched prefill of `toks` onto seq 0, logits computed only on the final token, captured into
/// `last_logits`; advances `pos` by toks.len(). Chunked to the batch capacity (LlamaBatch is 512
/// here) — a 8k replay is ~16 decode calls, fast. Use this in BOTH `prime` and `roll`, NOT
/// `decode_one` (so the reset guard can't re-enter).
fn prefill_seq0(&mut self, toks: &[LlamaToken]) -> Result<()> {
    let cap = 512; // batch capacity from LlamaBatch::new(512, 1)
    for chunk_start in (0..toks.len()).step_by(cap) {
        let chunk = &toks[chunk_start..(chunk_start + cap).min(toks.len())];
        let is_final_chunk = chunk_start + cap >= toks.len();
        self.batch.clear();
        let last = chunk.len() - 1;
        for (i, &t) in chunk.iter().enumerate() {
            // logits only on the very last token of the whole replay
            self.batch.add(t, self.pos + i as i32, &[0], is_final_chunk && i == last)?;
        }
        self.ctx.decode(&mut self.batch)?;
        if is_final_chunk {
            let logits = self.ctx.get_logits_ith(last as i32);
            self.last_logits = logits[..self.n_vocab].to_vec();
        }
        self.pos += chunk.len() as i32;
    }
    Ok(())
}
```

## Hooks into existing methods

**`new`**: store `self.n_ctx = n_ctx as i32;` and init the new fields (`keep_recent` from a CLI
arg, `recent_ids` with that capacity, `settle = 0`).

**`prime`**: record the preamble and use the batched prefill:
```rust
pub fn prime(&mut self, tokens: &[LlamaToken]) -> Result<()> {
    self.preamble = tokens.to_vec();
    self.n_keep = tokens.len() as i32;
    self.prefill_seq0(tokens)
}
```

**`decode_one`**: add the reset guard at the very top (this is the single-token observation path;
`prefill_seq0` is the only other writer of seq 0 and is exempt, so no recursion):
```rust
fn decode_one(&mut self, tok: LlamaToken) -> Result<()> {
    if self.pos >= self.n_ctx - 1 { self.roll()?; }
    // ... unchanged body ...
}
```

**`observe`**: append the committed token to `recent_ids` **after** `decode_one` returns (so the
ring reflects what's actually in the cache, and a `roll` triggered inside `decode_one` replays the
prior window — not the not-yet-decoded current token). Do this in both the neutral-first-token
branch and the main branch:
```rust
self.decode_one(tok)?;
self.recent_ids.push_back(tok);
while self.recent_ids.len() > self.keep_recent { self.recent_ids.pop_front(); }
```
Also, near the top of `observe`, honor the settle counter so the post-reset blip doesn't fire
spuriously:
```rust
let suppressed = self.settle > 0;
if suppressed { self.settle -= 1; }
// ... compute surprisal/z as normal ...
let fired = !suppressed && z >= z_threshold && stats.count() > stats.warmup();
```

## Invariants / edge cases

- `n_keep + keep_recent` must stay comfortably under `n_ctx`. Assert it in `new`; if a user sets
  a huge preamble or recent window, clamp `keep_recent` and warn.
- `stream_index` must **not** reset on roll — it's the global event counter the JSONL/TUI use.
- The Welford baseline lives outside the cache and continues across resets — don't reset it. (It's
  why `settle` exists: the model's context is momentarily shorter right after a reset, so the first
  few surprisals can shift slightly until the window refills; suppress triggers for ~warmup tokens.)
- The interjection scratch sequence (`GEN_SEQ = 1`) is untouched by all of this — `roll` only
  clears seq 0. Good. Just make sure an interjection is not mid-flight when `roll` fires (it isn't,
  given the current single-threaded tick loop — but if generation ever moves off-thread, gate it).
- Headless and TUI both get this for free since both go through `observe`.

## Optional: compaction-aligned reset

The observer rolls many times per session on its own. As a *coarse, opportunistic* extra trigger,
the harness can also force a roll when the upstream model auto-compacts (a real semantic boundary:
the context the prior thinking reasoned about was just summarized away). Expose a public
`pub fn force_roll(&mut self) -> Result<()> { self.roll() }` and call it from the harness on a
compaction event. Caveat: the observer accumulates *thinking* tokens the cortex discards between
turns, so the two contexts aren't identical — this is a good reset trigger, not a sync mechanism.

## Upgrade path (only if cap+reset's hiccup proves unacceptable)

Real StreamingLLM (smooth, no-recompute sliding) on Gemma requires:
1. Build the context with full attention on all layers (`--swa-full` equivalent context param —
   **verify it's exposed in llama-cpp-2 0.1.150**; may need a newer crate or a patch).
2. `kv_cache_seq_rm(0, n_keep, n_keep + n_discard)` then `kv_cache_seq_add(0, n_keep + n_discard,
   pos, -n_discard)` then `pos -= n_discard`. The K-shift (RoPE re-rotation) is applied
   automatically on the next decode. **Verify these methods exist in the crate** (C symbols:
   `llama_kv_cache_seq_rm` / `llama_kv_cache_seq_add`, possibly under the newer `llama_memory_*`).
3. Do **not** quantize the K cache (`--cache-type-k q4_0` breaks the shift — documented bug).
4. Before relying on it, confirm context-*shift* (distinct from context-*reuse*, which is reported
   still-broken for Gemma) actually works on your pinned llama.cpp commit. If it logs "not
   supported," you're stuck on cap+reset regardless.

The cost of this path is memory: `--swa-full` abandons Gemma's hybrid efficiency (all layers store
full KV). On E2B that's only a couple GB at 128k — affordable on the M4 Pro — but it's the price.