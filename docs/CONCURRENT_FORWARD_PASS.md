# CONCURRENT_FORWARD_PASS.md — fused observation+generation (status: PARTIAL, see §0)

Goal (from the spec-author handoff): make observation and interjection-generation run in **one fused
`llama_decode`** per tick instead of timesliced, so the observer never stalls while the lobe
generates — the "parallel subconscious" made real (strengthens the PRIMARY claim: streaming in +
*concurrent* out). Implemented against `src/lobe.rs` (llama-cpp-2 0.1.150). This doc is the design
**plus** the implementation findings.

---

## §0. STATUS — what landed, what works, what doesn't (READ FIRST)

Implemented: `Lobe::step()` (the fused tick), `StepOutcome`/`InterjectStatus`, the `gen_*` fused
state, and `start_fused_interjection`. Exposed behind an **experimental, opt-in `--fused` flag on
`headless`** (not the default). The **TUI stays on the proven timesliced path** — see why below.

**Verified working:**
- **The fused two-sequence decode is sound.** Confirmed against the crate source: `batch.add(tok,
  pos, &[seq], logits=true)` pushes each token's batch index into `initialized_logits`, so adding
  stream@0 and gen@1 (both logits on) makes `get_logits_ith(0)` and `get_logits_ith(1)` each return
  their own row. One decode, two logit rows. ✔
- **Concurrent interjections are coherent** (same analytical quality as timesliced).
- **Observation is mathematically preserved** but **NOT bit-identical** to the unfused path. Co-
  batching a second sequence doesn't *pollute* seq 0 (sequences are independent), but it changes the
  GPU **batch shape**, so Metal/GGML reduction order differs → seq-0 logits differ at the **~1e-3**
  level (verified: many tokens identical, a few differ by 0.001). In a *thresholded* trigger those
  tiny diffs **compound through the Welford baseline into different borderline fires.** So fused
  observation is valid and gap-free, but reproducibility vs. unfused is float-level, not exact.
  (The timesliced path *was* byte-identical because it used identical single-token batches.)
- **Chunked ask-prefill (the hiccup fix) co-batches during prefill too.** The interjection ask (often
  200–350 tokens) used to prefill in ONE `decode_seq` at the fire — a ~250–500ms freeze of the stream
  (measured: median ~256ms on E2B, ~73ms/chunk × the same total on E4B). Now `start_fused_interjection`
  only STAGES the ask; `step()` co-batches `prefill_chunk` ask tokens per tick alongside the stream
  token (`GenKind::Prefill`, `advance_fused_prefill`), so the prose never stalls — the aside just starts
  a few ticks later (the bubble shows "musing…"). Only the FINAL ask token requests logits (gemma's 262k
  vocab makes the projection non-free), so `fused::Lane` grew a per-lane `logits: bool`. This is the same
  FP story as above (now the prefill ticks are co-batched too → the ~1e-3 drift extends to them); the
  interjection OUTPUT is unchanged (same final-token logits + RNG) — `chunked_prefill_changes_neither_
  observation_nor_interjection` proves both exactly on the deterministic mock. `--prefill-chunk` (default
  8) trades scroll-smoothness-during-prefill against aside-start latency; the cap+reset fit-check reserves
  the extra seq-0 growth (`ask/prefill_chunk` ticks).

**The blocker — context-mode fork overruns the unified KV cache under load:**
- `--interject-mode context` forks seq 0 onto GEN_SEQ via `copy_kv_cache_seq`. In **timesliced**
  mode seq 0 is *frozen* during generation, so the fork + frozen seq 0 fit. In **fused** mode seq 0
  **keeps growing** alongside the forked, also-growing GEN_SEQ, and the unified cache hits
  **`Decode Error 1: NoKvCacheSlot`** under sustained operation. Crash point scales with `n_ctx`
  (ctx 1024 → ~2nd interjection; ctx 4096 → ~20th).
- **Isolated by experiment:** fused + `--interject-mode snippet` (no fork) runs clean (exit 0, many
  interjections); fused + `context` (fork) crashes at the same config. → the **fork** is the cause.
- Because the crash scales with ctx, fused-context at the demo's ctx 32768 *would* crash partway
  through a long read — so making it the TUI default would regress the demo. Hence the TUI revert.

**Latent finding (pre-existing, not caused by this work):** even the **timesliced** context-mode
fork has cache pressure that can exhaust a **small** cache (ctx 4096) under *sustained* interjections
(exposed by a 13k-token stress run). The demo's **ctx 32768 has ample headroom — confirmed clean**
on a 13k-token run and the earlier **311k-token / 10-reset** full-book run. So the demo is safe; the
constraint is "ctx must be comfortably larger than the interjection load when context-mode forking."

**Corrections to the original design doc:**
- The ask prefill **cannot** co-batch with the stream token on the fire tick (the doc's "fuse it"
  optimization) — `fired` is only known *after* that tick's decode. Interjection start-up is one
  separate decode (one stall per interjection); the per-gen-token decodes are what fuse. The doc's
  "simpler-but-fine alternative" is the only option.
- The `roll()`-during-generation deferral (gate on `!gen_in_flight`) is implemented and was **not**
  the crash cause; the fork cache-pressure is. The widened reset margin (`2·interject_max + 96`) is
  in `step()` for the deferral, but it can't save a small ctx from the fork itself.

**Bottom line:** the fused mechanic is real and verified; the obstacle is precise multi-sequence KV
management (forked + concurrently-growing sequences) that llama.cpp's unified cache doesn't sustain.
This is exactly the cache-control limitation `CANDLE_DESIGN.md` is about — **a third reason for the
Candle port** (own the cache → fork-and-generate concurrently without overrun, *and* keep logits).
Until then: TUI = timesliced (works at ctx 32768); `--fused` headless = experimental, snippet-safe.

---

## Design (as implemented)

State on `Lobe`: `gen_in_flight`, `gen_pos` (GEN_SEQ cursor, independent of seq-0 `pos`),
`pending_gen_tok`, `gen_out`, `gen_produced`, `gen_max`.

`step(stream_tok, stats, z, topk, interject_max) -> StepOutcome`:
1. Roll guard, deferred while `gen_in_flight` (widened margin for the concurrent growth).
2. Score `stream_tok` vs the prior `last_logits` (identical to `observe()`); compute `fired`/trigger.
3. **Fused decode:** `stream_tok @ seq 0` (logits) + `pending_gen_tok @ GEN_SEQ` (logits) → one
   `decode`. Row 0 → next `last_logits`; row 1 → sample next gen token (temp/top-p), emit the just-
   decoded token, stop on EOG/turn-marker/`gen_max`.
4. Advance seq 0 (`pos++`, `push_recent_id`).
5. On a fresh fire with no gen active (`InterjectStatus::Idle`), `start_fused_interjection` (fork +
   ask prefill, one decode) and seed the gen cursors. Deferred if a gen finished this same tick (so
   a `Done` isn't clobbered).

`StepOutcome { step: Step, interjection: InterjectStatus { Idle | Started | Working(s) | Done(s) } }`.
Frontends apply the same early-abort dedup as before (buffer → reveal-if-novel / abort-if-doomed).

## Reproduce / verify
```bash
M=models/gemma-4-E2B_q4_0-it.gguf
# works (snippet, no fork):
cat in.txt | streaming-lobe --model $M --ctx 1024 --interject-mode snippet --mode headless --fused
# crashes under load (context fork):
cat in.txt | streaming-lobe --model $M --ctx 1024 --interject-mode context --mode headless --fused
# observation float-parity check: diff trigger streams of `--no-interject` vs `--fused`.
```

## If picking this up
- **Fix path A (llama.cpp):** pin the exact cell behavior of `copy_kv_cache_seq` + concurrent growth
  (share vs. copy-on-write; whether overlapping cross-seq positions leak); maybe re-fork periodically
  or bound the fork. Uncertain; deep in the cache internals.
- **Fix path B (the real one):** Candle — own the KV cache, fork-and-generate with a second sequence
  whose cells you control, no unified-cache overrun. See `CANDLE_DESIGN.md` (now a 3-reason port).
- **Cheap interim:** fused + snippet mode works today (loses the connective full-context reflection).
