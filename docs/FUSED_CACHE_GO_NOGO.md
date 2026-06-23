# FUSED_CACHE_GO_NOGO.md — overrun triage + Candle go/no-go → **RESOLVED: NO-GO (fixed on llama.cpp)**

The fused concurrent forward pass (observation + interjection in ONE decode, so observation never
stalls) was crashing `NoKvCacheSlot` in context-fork mode, and that crash was being used to justify a
Candle port. This doc is the falsifiable go/no-go that settled it. **Outcome: it was a cache-*management*
bug (pool sizing), not a missing primitive. Fixed in hours. Stay on llama.cpp.** Reasoning below.

## The two original Candle justifications, both retired

1. **PLE-offload efficiency (was CANDLE_DESIGN "Reason 2") — NON-ISSUE on this hardware.** PLE offloads
   embedding tables off the *accelerator* to spare scarce GPU/TPU/NPU memory (phones, discrete GPUs).
   The M4 Pro is **unified memory** — CPU and GPU share one physical pool, so "offload to CPU" saves
   nothing, and the absolute delta is ~1–1.5 GB at 4-bit, which this machine doesn't notice. llama
   loading the full set costs the same here. Removed from the Candle case.
2. **Concurrent fork-and-generate overrun (was "Reason 3") — a MANAGEMENT bug, now fixed.** See below.

Custom cache **management** (orchestrating `seq_cp`/`seq_rm`/`clear` for the lifecycle) ≠ custom cache
**implementation** (writing the KV data structure). An overrun needs only the former, which llama.cpp
already gives. The latter is what Candle costs (weeks).

## What we measured (§3 instrumentation)

Added `Session::seq_pos_max(seq)` (llama `kv_cache_seq_pos_max`; no used-cells accessor exists in
0.1.150) + a `[kv]` stats row (`Lobe::kv_debug`): per-tick `seq0_max`, `gen_max`, `gen_inflight`.
Ran the crashing config (ctx 1024 / 2048, fused, context-mode, interjections firing).

Findings:
- **No leak.** `gen_max` sawtoothed to **-1** after every interjection — GEN_SEQ reclaims cleanly
  (§4b was already correct). Ruled out the leak hypothesis by measurement, not assumption.
- **The ask is large.** At a fire, `gen_max` jumped ~300 above `seq0_max`: the context-mode ask =
  delta span (≤`MAX_SPAN_TOKENS`=128) + novelty memory (≤2·`interject_max`) + framing (~64) ≈ 300
  tokens, decoded onto GEN_SEQ starting at the *current* seq-0 position.
- **The bug = sizing (§4a).** The roll guard fired on **seq-0 position** with a margin that never
  accounted for the interjection's footprint. So a fire near the top decoded its ~300-token ask
  *past* `n_ctx` → `NoKvCacheSlot`. Peak concurrent footprint above the fork = `ask + 2·interject_max`
  (gen tokens on GEN_SEQ **plus** seq-0's own growth during the deferred-roll generation — both
  sequences grow at once).

## The fix (§4a, exact not estimated)

`src/lobe.rs`:
- `roll_margin()` sizes the cap+reset margin to the full concurrent footprint
  (`MAX_SPAN_TOKENS + 128 + 5·interject_max`, clamped), used by `decode_one`/`step`/`prime`.
- `start_fused_interjection` does an **exact** pre-fork fit check: it tokenizes the ask first, and if
  `pos + ask_len + 2·interject_max + 32 > n_ctx` it **rolls seq 0 first** (no gen is in flight at
  start), so the fork + ask + gen always fits — a precise check, never an estimate.

## §5 go/no-go result — **PASS → NO-GO on Candle**

Sustained run: ~23k tokens (>10×n_ctx at ctx 2048), fused, context-mode, interjections throughout
(GEN_SEQ fork/generate/clear exercised repeatedly, 24 cap+resets).
- **exit 0, zero `NoKvCacheSlot`.**
- `seq0_max` bounded and sawtoothing (282–1502, well under n_ctx 2048); `gen_max` sawtooths to -1.
- No monotonic growth. 234 triggers+interjections, coherent.
- Demo config (ctx 32768, interject_max 96) also exits 0.

Per the decision rule: **PASS reached via §4 ⇒ the crash was a management bug; it's fixed; stay on
llama.cpp.** The GO bar (a *named per-layer primitive* that llama.cpp rejects or computes incorrectly
— never an overrun) was never reached: this was a capacity failure, which is NO-GO by definition.

## Consequence

The fused concurrent forward pass now works on llama.cpp. The TUI uses it by default (observation
never stalls; the reply streams in alongside the text). The PRIMARY claim — streaming input + output
generated **concurrent to** the stream (not as an interrupting turn), infinite context, no compaction —
is delivered on llama-cpp-2, no Candle bring-up required.

## The one Candle reason that survives (still conditional, still not now)

CANDLE_DESIGN "Reason 1" — *seamless* per-layer infinite context (local layers windowed + global
layers sink+shift simultaneously, no re-prefill hiccup). cap+reset already gives bounded memory +
compute over unbounded input; Reason 1 only removes the sub-second re-prefill stall per reset. That's
a long-term polish decision, gated on the hiccup actually being felt — not a PoC blocker. The GO
trigger for it remains a *correctness/unsupported* failure of per-layer cache shiftability (iSWA
non-shiftability), never a capacity crash.
