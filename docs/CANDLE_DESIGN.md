# CANDLE_DESIGN.md — minimal design for a Candle port

**Status: future / conditional — and the case got WEAKER on 2026-06-21.** This is not the next step.
Build the PoC on llama-cpp-2, ship cap+reset. **Two of the three reasons were retired** after review +
measurement (see `FUSED_CACHE_GO_NOGO.md`):

- ~~Reason 2 (PLE-offload efficiency)~~ **RETIRED — non-issue on unified memory.** PLE spares *accelerator*
  memory; the M4 Pro shares one pool, so offload saves nothing here (~1–1.5 GB at 4-bit, unnoticed).
- ~~Reason 3 (concurrent fork-and-generate)~~ **RETIRED — it was a cache-*management* bug, now fixed on
  llama.cpp.** The fused pass overran because the roll guard didn't reserve the interjection's
  concurrent footprint; fixed (roll on the full `ask + 2·interject_max` footprint), validated crash-free
  over >10×n_ctx. The fused concurrent path is now the TUI default. Overrun = sizing/leak = management,
  never a Candle GO signal.

So **only Reason 1 survives, and it's conditional**: read on only if (a) cap+reset's sub-second
re-prefill hiccup is actually *felt* at your window sizes AND (b) you want *seamless* per-layer
infinite context AND (c) you'll own a from-scratch Gemma-4 forward pass. Otherwise llama.cpp remains
the right home: the live choice is just E2B (footprint) vs E4B-on-llama.cpp (demo voice).

## Reason 1 — the per-layer KV cache

Everything else in this project is equal or worse in Candle. The single thing that flips is the KV
cache. In llama.cpp the two memory wins are mutually exclusive — Gemma's iSWA windowing is exactly
what makes the cache non-shiftable, so you must pick between Gemma's native efficiency and
StreamingLLM's free sliding. **In Candle you write the attention, so you own each layer's cache
independently:** local layers keep a 512-token window (Gemma-native), global layers run
sink+window+shift (real StreamingLLM) — *at the same time*. Result: both memory wins, seamless
unbounded context, no `--swa-full` memory tax, no re-prefill stall. The thing that's architecturally
impossible in current llama.cpp is just a per-layer policy you implement here.

## Reason 2 — PLE-offload efficiency *without* losing logits ~~(surfaced 2026-06-21)~~ — **RETIRED 2026-06-21**

> **RETIRED — non-issue on this hardware.** PLE's benefit is sparing *accelerator* memory by keeping
> embedding tables off the GPU/TPU/NPU (the win on phones / discrete GPUs). The M4 Pro is **unified
> memory**: CPU and GPU share one physical pool, so "offload to CPU" frees nothing, and the absolute
> embedding footprint is ~1–1.5 GB at 4-bit — negligible here. llama.cpp loading PLE densely costs the
> same as offloading it on this machine. This is NOT a reason to leave llama.cpp on unified memory (it
> would be on a discrete GPU). The original analysis is kept below for the discrete-GPU case only.

The second payoff, same enabler (you own the forward pass). The E-models' headline efficiency —
E4B is "4.5B **effective** / 8B **with embeddings**" — comes from **Per-Layer Embeddings (PLE)**
being **offloadable** (kept off the accelerator, streamed per layer). There's a real
three-way bind across stacks:

- **llama.cpp (our stack):** gives raw per-token logits (the surprisal trigger needs them) but —
  verify, but near-certainly — loads PLE **densely**, so you pay the full 8B-with-embeddings
  footprint and forfeit the "effective param" win. (E4B on llama-cpp-2: 5.15 GB / ~30 tok/s.)
- **The "mobile" build + LiteRT-LM** (`-mobile-ct` / `-mobile-transformers`): realizes the PLE
  offload + 2-bit decode + optimized KV — i.e. E4B quality at mobile footprint — but LiteRT-LM is a
  high-level *generation* runtime that does **not** expose raw next-token logits or arbitrary
  token-forcing. That **breaks the surprisal mechanic at its foundation** (the whole project is
  `-ln P(token)` read off a teacher-forced pass). So the efficiency is real but unreachable here.
- **Candle:** because you write the forward pass, you can implement PLE offload **yourself** (keep
  the per-layer embedding tables on CPU / stream them in) to capture the effective-param memory win,
  **and** keep full logit access (you wrote the head), **and** get Reason 1's per-layer KV. It is
  the only stack that gets *all three* — the mobile build's efficiency without LiteRT's logit
  blocker, plus the streaming cache.

Note the cost is already on the books: the Gemma-4 forward-pass bring-up (net-new item 1) **already
includes implementing PLE** — so Reason 2's payoff rides the same surface as Reason 1 (and the
bring-up you're paying for regardless). Caveat (fundamental, not fixable by stack): E4B's ~4.5B
*effective* **active** params are ~2× E2B's compute, so even with PLE offloaded, E4B-class throughput
won't match E2B — Candle narrows *memory*, not the active-compute gap. PLE offload also trades some
accelerator-memory for per-layer CPU↔GPU traffic; net throughput effect is worth measuring, not
assuming.

The strategic upshot: this took the port from a **one-reason** move (seamless infinite context) to a
**two-reason** one (… + reach E4B-class quality at near-E-model footprint while keeping the logit
access the architecture is built on). Neither reason alone may clear the bar; together they're a
stronger case for eventually owning the inference stack.

## Reason 3 — concurrent fork-and-generate without cache overrun ~~(surfaced 2026-06-21)~~ — **RETIRED 2026-06-21**

> **RETIRED — it was a cache-*management* bug, fixed on llama.cpp.** The fused pass's context-fork
> `NoKvCacheSlot` was NOT a missing primitive: the cap+reset roll guard fired on seq-0 position without
> reserving the interjection's concurrent footprint (the ~300-token context ask + gen tokens + seq-0's
> own growth during generation = `ask + 2·interject_max` cells above the fork). Fixed by rolling on
> that full footprint (`roll_margin` + an exact pre-fork fit check in `start_fused_interjection`).
> **§5 result: crash-free over >10×n_ctx with interjections firing throughout; the fused concurrent
> path is now the TUI default.** Per the go/no-go's hard rule, an *overrun* (out of cells) is always
> sizing or a leak — both management — and can NEVER be a Candle GO signal; GO requires a named
> per-layer primitive llama.cpp rejects or computes incorrectly. See `FUSED_CACHE_GO_NOGO.md`.

## The pivotal insight (why the cost and the payoff are the same surface)

Gemma 4 **forces** per-layer attention dispatch regardless of cache strategy: sliding layers and
global layers have different head dimensions (256 vs 512), different RoPE handling (p-RoPE on
global), different windowing, and the final layer is always global. You cannot write Gemma 4 as a
uniform attention loop — you're writing per-layer dispatch no matter what. And per-layer dispatch is
precisely what unlocks per-layer cache control. So the expensive part of the port (bringing up
Gemma 4's unusual attention) is the same surface that delivers the prize (per-layer streaming
cache). You're not paying twice.

## What ports nearly unchanged

The observer *logic* is backend-agnostic. From the current repo, these move with little or no
change because they operate on logits/text, not on llama.cpp:
- surprisal (`surprisal_of`, the log-sum-exp), `top_k`, the Welford baseline (`stats.rs`)
- the trigger gating, `Step`/`Trigger`, stream-index bookkeeping
- the scratch-sequence generation *pattern* (`interject`, `decide_retrieval`) — the structure is
  identical; only the decode/sample calls underneath change
- the cap-vs-stream cache *policy decisions*, the CLI, headless JSONL, the TUI
- the retrieval grammar's parse/feedback logic

Only `Lobe`'s decode→logits primitive and its KV-cache internals are replaced. Most of the codebase
survives the port; treat `lobe.rs` as the seam.

## What's net-new (the actual work, roughly in order of risk)

1. **The Gemma 4 E2B forward pass in Candle.** This is the big lift, and it's bigger than "tweak the
   existing Gemma model" because of the novel pieces: dual head_dim (256 local / 512 global),
   Proportional RoPE on global layers, unified/shared KV in global layers (verify exact semantics —
   possibly cross-layer KV sharing à la Gemma 3n), Per-Layer Embeddings in the E-models, RMSNorm,
   GeGLU, the iSWA layer schedule, 262k vocab with byte-fallback BPE. **First action: check current
   `candle-transformers` Gemma coverage** — it has historically carried `gemma`/`gemma2` as a
   skeleton, but Gemma 4's features are net-new and almost certainly not yet present. Budget this as
   a model bring-up, not an afternoon.
2. **The per-layer KV cache** (the payoff — see sketch below).
3. **Constrained / grammar sampling.** Candle has **no GBNF engine** (llama.cpp's big freebie you'd
   be giving up). So item 8's retrieval grammar goes the *decomposed* route from
   `RETRIEVAL_GRAMMAR.md` — logprob classify (`NONE`/`SEARCH`, then `mem`/`rag`) + free query
   generation — which needs no grammar engine and is the reason that fallback was specified. If you
   later want true grammar constraints, you implement a token-mask sampler yourself.
4. **Metal device setup.** Candle's Metal backend is solid; this is the easy part. (Still Metal, not
   MLX — same as llama.cpp; MLX-in-Rust remains immature and on an M4 Pro buys little.)

## KV cache design (the heart)

Own the cache per layer. Sketch:

```rust
enum LayerKvCache {
    /// Local sliding-window layers: keep only the last `window` tokens. Native Gemma behavior;
    /// bounded by construction, nothing to shift.
    WindowedLocal { window: usize, k: RingKv, v: RingKv },
    /// Global layers: StreamingLLM. Keep the first `n_keep` (pinned prefix / sinks) + a rolling
    /// window; on overflow evict the middle and shift remaining positions down, re-applying p-RoPE
    /// for the new positions (correct *because you control RoPE application here*).
    StreamingGlobal { n_keep: usize, window: usize, k: ShiftKv, v: ShiftKv },
}

// The model's attention loop dispatches per layer from the iSWA schedule:
for (il, layer) in layers.iter().enumerate() {
    let cache = &mut caches[il];                  // WindowedLocal or StreamingGlobal per schedule
    let (k, v) = layer.project_kv(x);
    let k = apply_rope(k, positions, layer.rope); // p-RoPE on global, standard on local
    cache.append(k, v);                            // ring-evicts (local) or evict+shift (global)
    let out = attend(layer.project_q(x), cache.k(), cache.v(), layer.mask);
    // ...
}
```

Key correctness points, all of which llama.cpp's abstraction *prevented* you from getting right and
Candle now *requires* you to get right:
- **Per-layer RoPE on shift.** The global cache shifts positions on eviction; re-apply p-RoPE for the
  shifted cells (the same principle as standard-RoPE re-rotation, with Gemma 4's p-RoPE formula).
  Local caches never shift, so this only touches global layers.
- **The pinned prefix lives in the global caches.** That's where the sinks + preamble persist across
  unbounded streaming. Local layers legitimately forget the prefix (it's outside their 512 window) —
  which is fine and is exactly Gemma's native behavior; the long-range signal routes through the
  global layers regardless.
- **Both cache types are bounded**, so memory is flat over an unbounded stream with no `--swa-full`
  and no reset hiccup. That is the entire point of the port.

## Simplifications Candle buys you

- **You can probably skip quantization.** E2B in bf16 is a couple GB — trivial in the M4 Pro's
  unified memory. Running unquantized removes the whole q4_0-K-cache/shift hazard and the KV-quant
  pain that haunt the llama.cpp path. (Candle does support GGUF/quantized loading if you want it
  later, but for E2B you likely won't need to.)
- **The FRAGILE-API churn disappears.** No more chasing renamed methods across llama-cpp-2 releases —
  you own the model code. The trade is that correctness is now entirely yours; there's no
  battle-tested kernel to lean on.

## Decision gate

Do the port only when all three hold:
1. cap+reset's hiccup is something you actually feel at your real window sizes (measure first — it
   may be imperceptible);
2. you specifically want seamless, both-memory-wins, per-layer infinite context;
3. you're willing to own a from-scratch Gemma 4 E2B forward pass, including its genuinely unusual
   attention (the head_dim=512 global layers that FlashAttention itself can't yet serve, p-RoPE, PLE).

If any fails, stay on llama-cpp-2 + cap+reset. The Candle move is the deliberate "I've outgrown
llama.cpp's cache abstraction" decision — and Gemma 4's architecture makes it a real project, so go
in with the model bring-up scoped, not as a weekend refactor.
```