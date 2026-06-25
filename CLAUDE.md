# CLAUDE.md — stream-observer

Read this first. This is a handoff brief for continuing the prototype in Claude Code.

## Status — VERIFIED RUNNABLE (2026-06-19)

It compiles and runs. The "⚠️ unverified" caveat below is now resolved: every `// FRAGILE:`
call matched the installed **`llama-cpp-2 0.1.150`** API as written (the upstream scaffold
got them right); only a handful of *layout/lifetime* fixes were needed, not method renames.

- **Toolchain:** Rust 1.96, Apple clang 21, cmake 4.3, on Apple M4 Pro / Metal.
- **Model:** Google's official QAT 4-bit GGUF, `models/gemma-4-E2B_q4_0-it.gguf`
  (from `google/gemma-4-E2B-it-qat-q4_0-gguf`, Apache, ungated). `gemma4` arch is present
  in the llama.cpp vendored by 0.1.150, so it loads natively. A tiny known-good fallback,
  `models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`, is also present.
- **PoC confirmed:** on `sample_thinking.txt` (a reasoning stream with planted anomalies)
  the top z-scores are the anomalies — ` fl`(ibbertigibbet) z=4.57, ` moon` (green cheese)
  z=3.85, ` elevator`/` Tuesday` — while function words sit at ~0 nats. At `--z 3` it fires
  exactly on those, reporting what it expected instead.

Fixes applied (all small): files moved to `src/`; `Cargo.toml` → `default-features = false`
(drops `openmp`, whose cmake `find_package(OpenMP)` is flaky with Apple clang); `LlamaBatch`
→ `LlamaBatch<'a>` (it gained a phantom lifetime); `main.rs` match borrows `&cli.mode`
instead of moving; `token_to_str`/`Special` deprecation allow-listed. Punch-list 1–3 done
(see below): live `--z` adjust (`+`/`-`) and lobe position in the TUI status bar are wired,
and a `--verbose` flag now gates llama.cpp/Metal log spam (off by default via `void_logs`).

**Interjections reworked (punch-list #7 done).** The first cut decoded the interjection into
the *observation sequence* (seq 0) at stream positions, so the observer attended to its own
output — slowing observation and shifting later surprisals, and the "interjection" was really
a *completion* of the stream. Both are fixed:
- **No pollution:** generation runs on a scratch KV sequence (seq 1; context now built with
  `with_n_seq_max(2)`), then `seq_rm`'d. The observation stream, `pos`, and `last_logits` are
  untouched — *proven* byte-identical: the per-step surprisal sequence is the same with and
  without `--interject`.
- **Observation, not completion:** generation is a **chat-framed reframe** — the (instruct)
  observer is prompted to *react* to the surprising token given recent context, not extend it.
  Output is genuine commentary (e.g. ` spiral` → "an odd identifier given the preceding list of
  physical weapons"), even on out-of-domain literary prose.
- ⚠️ **CRITICAL gemma-4 chat-format gotcha.** gemma-4 does **not** use gemma-2/3's
  `<start_of_turn>`/`<end_of_turn>` — those are **absent from the vocab**. It uses `<|turn>`
  (open, id 105) + role + `\n` + content + `<turn|>` (close, id 106); generation prompt is
  `<|turn>model\n`; EOS is `<eos>` (id 1) and `is_eog_token` does NOT flag `<turn|>`. Using the
  wrong markers tokenizes them as *literal text* → the model never enters a chat turn → it
  *completes* (continues prose, leaks literal `<...>` markers) instead of replying. This was a
  real bug (interjection AND `--frame`). Fixed: `Lobe::interject` and `main.rs` framing use the
  gemma-4 tokens; generation stops on `<turn|>`/`<|turn>`/EOG; output detok suppresses specials.
  **If you swap models, re-derive the chat format from the GGUF `tokenizer.chat_template`.**
- `--interject` is now also available in **headless** mode: each `trigger` is followed by an
  `interjection` JSONL event (the observe → generate → resume shape, scriptable).

## North star — two goals, two modes (read before tuning interjections)

The thing being validated, stated plainly so it stops getting re-litigated:

**The PRIMARY claim (the whole point):** an agent that operates over **infinite streaming input** and
generates its output **concurrently with the input stream** — not as an interrupting, turn-based chat
turn. Streaming in + concurrent out, unbounded, bounded-memory. (This is *already validated*: #6 +
the 311k-token/10-reset run = unbounded streaming, flat memory/throughput, while interjecting
concurrently. The rest is making it *good*, not proving it *possible*.)

**Short-term goal (what we're building NOW): the party trick.** An agent "reads" a long novel and
talks about it — no compaction, just its bounded window + cap+reset. The interjection IS the show, so
it must read like a *thoughtful reader* (connective musing — the gears→inventory→stalled thread is the
demo working), NOT a highlighter (terse pointers). → **protect length; kill repetition; don't cap
hard.** The surprisal trigger is exactly right for this.

**Long-term goal (the dream): a limbic lobe embedded in an agent harness**, watching the turn-based
chat stream on a separate thread. Here the interjection is **not** output — it's a *typed promotion*
into the main loop. Promotions are **polymorphic** (E2B's native tool-calling is why): the lobe can
*act* (run RAG / memory / web / `duckdb` and promote the **result**) or *observe* (promote a reminder
/ warning / "you're violating the rule at line Y" / "remember earlier the user said Z"). It's
"supercharged reasoning": detection is **free** (off the forward pass, parallel, zero main-loop cost),
promotion is **selective-cost** (inject only validated hits → save a tool call / a turn). The
influence mechanism is the **injection**, not shared attention (the lobe reasons in its *own* context;
promoting is what turns subconscious → conscious / attended).

**Trigger ↔ promotion are coupled by type** — what we've built covers some cells, not all:

| Promotion (long-term) | Detect via | Free off surprisal/entropy? |
|---|---|---|
| RAG/memory result (unfamiliar entity / gap) | high surprisal = "didn't expect / don't know" | yes |
| anomaly / consistency flag | surprisal spike vs. established context | yes |
| constraint enforcement ("violates rule at line Y") | **pin the rules in the sink** → a violation is a low-prob token given rule-primed context = a surprisal spike | yes, *if rules are in the preamble/sink* |
| lost-in-the-middle resurfacing ("remember earlier, Z") | *relevance* to distant de-weighted context | **no** — relevance-match, not surprise; needs a different detector |

Timing wrinkle for the harness (not the demo): the subconscious is **slower** than the conscious — by
the time a RAG round-trip promotes, the conversation may have moved on. Promotions must be fast enough
or staleness-robust. Single-threaded novel-reading demo doesn't have this; the harness does.

## What this is

A small, fast local model (GGUF via `llama-cpp-2`) that **observes** a token stream and
flags the tokens it finds surprising — a "limbic" / subconscious observer lobe meant to
run alongside a frontier reasoning model, watching its (streamed) thinking and chiming in.

> **Two distinct ideas, don't conflate them (the name does):**
> 1. **StreamingLLM** (the paper, arXiv 2309.17453) is *only* a KV-cache trick — attention sinks +
>    rolling window → bounded-memory infinite *input*. That is **exactly and only punch-list #6**
>    (cap+reset). It has nothing to do with triggers, output tokens, or interjections.
> 2. **The surprisal observer** (the trigger + interjection) is this project's *own* idea, not from
>    the paper. The "trigger" is **not** the model deciding/emitting anything — it's `-ln P(token |
>    full context)` read straight off the forward pass (objective, free, context-conditional; the
>    token is just where a context-level surprise localizes). That free salience signal is what lets
>    a small model watch continuously and only *generate* on a spike. (A "generative observer" that
>    emits its own interject token — like reasoning-mode `<think>` delimiters, which gemma-4 has as
>    `<|think|>`/`<|channel>` — is a valid *alternative* architecture, but it trades away the
>    free-detection property; noted as a future direction, not the current design.)

The core mechanic, which is the whole reason this is feasible on an off-the-shelf model
with no fine-tuning:

- **Observing == teacher forcing.** We feed the stream's next token into the KV cache via
  `decode`, exactly as generation does, except the next token comes from the stream rather
  than from sampling. The model doesn't care where a token came from.
- **Surprisal as the trigger.** After decoding token `t-1` we have the next-token
  distribution; when token `t` arrives we score `-ln P(t)`. A spike means "something
  unexpected just happened." This needs zero generation — it's read off the forward pass.
- **Baselined, not absolute.** Raw surprisal varies wildly by stream, and a 2B model's
  sense of "surprising" is uncalibrated, so we z-score against a running Welford
  mean/variance (`src/stats.rs`) and fire on the z-score. This is the calibration knob.
- **Observe → generate → resume.** When a token fires, we can optionally flip the *same*
  context into greedy generation for a short interjection, then resume observing. One
  context, timesliced. (See the note in `tui.rs` about the interjection sharing the cache.)

## Structure

- `src/backend/` — **inference-backend abstraction** (docs/BACKEND.md). `mod.rs` = the `Backend`
  (model+tokenizer) and `Session` (KV+decode) traits + `Token`/`Decode`/`Detok`/`SessionConfig` +
  the cfg-selected `ActiveBackend` alias. `llama.rs` = the llama-cpp-2 impl — **ALL version-fragile
  `// FRAGILE:` calls now live ONLY here.** `candle.rs` = the Candle backend (skeleton; gemma-4
  forward pass is the bring-up). Exactly one backend compiles, chosen by the `llama`/`candle` feature.
- `src/lobe.rs`  — the observer core, now **backend-agnostic** (talks only to the traits; no
  `llama_cpp_2` references). Teacher-forced decode, surprisal, top-k, sampling, interjection, #6/#7/#8.
- `src/stats.rs` — Welford running mean/variance + z-score. Stable, no external API.
- `src/trace.rs` — **structured observability** (`--debug-log`). `tracing` → JSONL.
- `src/main.rs`  — CLI + **headless** mode (stdin → JSONL on stdout). The Sandpiper-facing
  shape: tee a frontier model's thinking-delta stream into stdin. `with_lobe(cli, body)` is the shared
  construction seam — loads the engine, builds + primes the lobe, builds the retriever (all share the
  self-referential `&engine` borrow), then hands `&mut Lobe` + the retrieval fn to `body`; every mode
  funnels through it (the present worker calls it on its own thread).
- `src/tui.rs`   — **`--mode debug-tui`** (ratatui). Paces a transcript through the same lobe and
  renders surprisal live. The calibration instrument (sparkline, z-heatmap, triggers, knobs). Still
  SYNCHRONOUS (lobe on the render thread) — it's the instrument, not the show.
- `src/present.rs` / `src/present_scene.rs` — **`--mode demo-tui`** / **`--mode demo-tui --scene`**
  (ratatui), the DEFAULT mode. The
  SHOWCASE, not the instrument: clean stage (prose pane + spotlit interjection / pixel-art study with
  speech bubbles). **`src/present_worker.rs` runs ALL llama work (observe + retrieve + fused
  interjection) on a WORKER THREAD** that owns the engine/lobe/retriever (built inside the thread via
  `with_lobe`, so the `!Send` llama handles never cross a boundary); the render thread is render-only,
  consuming `UiEvent`s over an mpsc channel and sending `Control` back. This is the HICCUP FIX: the
  fire/glow animation never freezes on a retrieval or interjection. `space` pauses, `q` quits.
- `src/sprite.rs` — half-block sprite widget for the scene; art in `assets/scene/study.py`→`study.json`.
  **See `docs/SCENE_HANDOFF.md`** (state + the rasterize design loop) before touching the art.

> **Key docs added (read these — they hold the hard-won decisions):** `docs/BACKEND.md` (trait +
> cargo-feature backend), `docs/FUSED_CACHE_GO_NOGO.md` (the cache fix → fused works on llama, Candle
> NO-GO), `docs/TEMPLATING_STUDY.md` (the converged voice: `herzog_varyform.txt` @ temp 0.6 — temp
> owns depth, prompt owns structure), `docs/SCENE_HANDOFF.md` (the `--scene` diorama + art loop).

Two frontends, one core. Headless is what you wire into the harness; the TUI is what you
stare at to tune `--z`.

## Observability — `--debug-log <file>` (don't guess; measure)

Deep structured tracing to JSONL, gated behind a flag, **zero-overhead when off** (the `tracing`
`enabled!` guards skip all dump work when no subscriber is installed). Separate from the stdout
JSONL and `--stats`; safe in the TUI (writes to a file, not the terminal). This exists because we
kept reasoning from screenshots — never again. One JSON object per event, `kind` field per type:

- `run_start` — full config dump (every knob), so each trace file is self-describing.
- `observe` (per scored token) — `surprisal`/`entropy`/`z`, `baseline_mean`/`std`, `fired`,
  `suppressed_settle`/`in_refractory`/`gate_pass` (exactly *why* it did/didn't fire), `decode_us`.
  At TRACE: the `logits` top-K dump (id/tok/logit/prob + argmax/entropy) of the **predicting**
  distribution, and the recent-context window text.
- `trigger` (per fire) — the above + `delta_span` (what the interjection will focus on) + full
  context; `--debug-full-logits` also dumps the entire 262k-vocab vector here.
- `interject_begin` — `mode`/`forked`/`start_pos`, the captured `delta_span`, the `novelty_memory`
  (what 1b shows the model), and the **exact raw `prompt`** the model sees, verbatim.
- `interject_prefill` / `interject_token` (per gen token, TRACE) / `interject_done` (`output`,
  `stop_reason`, `produced`, `latency_us` — e.g. ~700ms is the "visible hiccup").
- `window_slide` — cap+reset: `pos_before`/`pos_after`, `recent_window`, `replay_len`, `latency_us`.
- `rag` — `prompt`, `raw_output`, parsed `thought` + `directive_source`/`directive_query`, timing.

Flags: `--debug-log <file>` (enable), `--debug-topk N` (logit-dump width, default 64),
`--debug-full-logits` (full vector on fires/gen only — huge). Filter targets with the `LOBE_LOG`
env var (default `lobe=trace,llama_cpp_2=debug`; the crate emits its own load/decode events too).
Analyze post-hoc with `jq`/python over the JSONL — nested blobs (`logits`, `full_logits`) are
JSON-encoded strings (double-decode).

## Build

Requires the Rust toolchain and `clang` (the llama backend compiles llama.cpp via bindgen).

**Backend is a cargo feature — exactly one** (`llama` default, `candle` WIP; see docs/BACKEND.md).
`metal`/`cuda`/`vulkan` are accelerator passthroughs to the llama backend.

```bash
# Apple Silicon (M-series) — Metal (= default llama + metal):
cargo build --release --features metal
# NVIDIA:
cargo build --release --features cuda
# CPU only (llama):
cargo build --release
# Candle backend (skeleton — compiles; gemma-4 forward pass is unimplemented):
cargo check --no-default-features --features candle
```

First llama build is slow (it compiles llama.cpp from source).

## Run

**The DEFAULTS ARE THE SHOWCASE.** The entire converged demo config is baked in as defaults, so the
only things you pass are `--input` and `--skip-to`. Out of the box: `--mode demo-tui`, gemma-4-**E4B**,
the **herzog_varyform** persona, **RAG ON** (hybrid BM25+harrier-270m over the corpus), **`--scene`**,
**non-deterministic**, z 4.1, refractory 320, interject-max 80, interject-temp 0.6, prefill-chunk 8,
32k ctx, cap+reset. **The CLI is FLAT — no subcommands;** `--mode {demo-tui|debug-tui|headless}` picks
the frontend (default `demo-tui`), everything else is a top-level flag, order-independent. Build once
(`cargo build --release --features metal`), then:

```bash
# THE SHOWCASE — Werner Herzog reads Moby Dick's narrative, muses, and RECALLS over the text. Just
# input + skip-to; everything else (E4B, persona, RAG, scene, non-det, the tuned cadence) is default.
# (--skip-to jumps past the front-matter to the narrative. Interactive; +/- tunes z; space pauses; q quits.)
./target/release/stream-observer --input corpus/pg2701.txt --skip-to "Call me Ishmael"

# Bare (no args) = the same showcase on the bundled sample_thinking.txt.
./target/release/stream-observer

# Opt OUT of showcase pieces (each default-on thing has a negation):
#   --no-scene (plain clean stage) · --no-rag (free-association, skips the embed-model load) ·
#   --deterministic (byte-identical runs) · --no-frame · --preamble-file "" (drop the persona) ·
#   --model models/gemma-4-E2B_q4_0-it.gguf (the leaner model) · --mode debug-tui (calibration instrument).
./target/release/stream-observer --mode debug-tui --input corpus/pg2701.txt --skip-to "Call me Ishmael"

# Headless (stdin → JSONL). For the LEAN pipe, turn RAG off (else it loads the embedder + indexes the
# corpus at startup) and/or interjections off:
cat sample_thinking.txt | ./target/release/stream-observer --mode headless --no-rag                 # interjections, no retrieval
cat sample_stream.txt   | ./target/release/stream-observer --mode headless --no-rag --no-interject  # pure surprisal
cat sample_stream.txt   | ./target/release/stream-observer --mode headless --no-rag --no-interject --all-steps > steps.jsonl
```

VOICE finding (docs/TEMPLATING_STUDY.md, baked into the defaults): the converged voice is
`herzog_varyform.txt` at LOW temp (0.6). TEMPERATURE owns DEPTH (low temp = commitment = rich musing);
the PROMPT owns STRUCTURE (the "vary your openings" nudge breaks lockstep at ~zero depth cost). Cranking
temp for variety was the wrong lever — it diluted the voice.

**Flag layout:** the CLI is FLAT (no subcommands) — `--mode {demo-tui|debug-tui|headless}` picks the
frontend (default `demo-tui`) and EVERY other option is a top-level flag, so flag order never matters.
The defaults bake the showcase (see above), so the notable flags are the **negations / overrides**:
`--no-scene`, `--no-rag`, `--no-frame`, `--no-interject`, `--deterministic`, `--preamble-file ""` (drop
the persona → `--preamble`/built-in), `--rag-embed-model ""` (BM25-only), `--model …` (default E4B),
`--rag-corpus …` (defaults to `--input`). Tuning knobs: `--z` (4.1), `--refractory` (320),
`--interject-max` (80), `--interject-temp` (0.6), `--prefill-chunk` (8), `--ctx` (32768),
`--interject-mode` (context), `--evict` (reset), `--debug-log`. **Determinism:**
the showcase default is NON-deterministic — a random OS-entropy seed each run AND a softened firing
decision (`P=sigmoid((z-z_threshold)/softness)`, softness 0.5, off an independent seeded `fire_rng`),
so BOTH the asides AND which tokens fire vary run-to-run, like a chat API. `--deterministic` pins it:
fixed sampler seed + hard firing threshold (the surprisal trigger is greedy/teacher-forced, so it's
deterministic by construction anyway; the surprisal VALUE is always exact regardless). Mode-specific
flags (top-level like all the rest): `--mode headless` → `--granularity`/`--all-steps`/`--fused`;
`--mode demo-tui`/`debug-tui` → `--input` (default `sample_thinking.txt`)/`--tick-ms` (30)/`--skip-to`,
and `--scene`/`--no-scene` (demo-tui only). Default granularity is `line`, which
preserves Gemma's subword tokenization. stdout is clean JSONL; a tiny Metal banner goes to stderr
(`--verbose` for full logs). (`--interject` is a deprecated no-op kept so older commands don't break.)

## ✅ FRAGILE calls — resolved against `llama-cpp-2 0.1.150`

The code was authored without compiling, but against accurate docs: against 0.1.150 the
`// FRAGILE:` calls were all correct as written. For the record, verified against the
installed crate source (`~/.cargo/registry/.../llama-cpp-2-0.1.150/src/`):

1. `get_logits_ith(i32) -> &[f32]` of length `n_vocab` — correct (batch offset `0`, which is
   the single logits-enabled token after `clear`+`add`; the slice is `n_vocab` long).
2. `token_to_str(tok, Special::Tokenize)` — exists (now deprecated → `token_to_piece`; we
   `#[allow(deprecated)]` it, it's the convenience we want).
3. `batch.add(tok, pos, &[0], logits)` — exact signature match (`pos: llama_pos = i32`).
4. `with_n_ctx(Option<NonZeroU32>)` / `with_n_batch(u32)` — correct.
5. `token_eos() -> LlamaToken` — exists (`is_eog_token(tok)` also available if preferred).
6. Features `metal`/`cuda`/`vulkan` — all present in the crate `[features]`.

The only real edits were structural, not name drift: see the Status block at the top.
`// FRAGILE:` tags are kept in `src/lobe.rs` as a reminder that this surface drifts between
crate versions — re-verify against `docs.rs/llama-cpp-2/<version>` if you bump the dep.

## Punch-list (suggested order)

1. ✅ **DONE — compiles + runs** against `llama-cpp-2 0.1.150` on Metal. Surprisal verified
   sane (function words ≈0 nats, rare/anomalous tokens high; see Status block).
2. ✅ **DONE — live `--z` adjust in the TUI.** `+`/`-` retune the threshold while watching
   (`src/tui.rs`: a mutable `z` seeded from `--z`, threaded into `observe` and the status).
3. ✅ **DONE — lobe position in the status bar** (`Lobe::position()` → `draw_status`).
4. ✅ **DONE — pluggable trigger signal.** `--signal {surprisal|entropy}` selects what the
   Welford z-score thresholds on: surprisal (-ln P, unexpected token) or entropy (model
   uncertainty at the position). `--identifiers-only` gates firing to identifier/entity-like
   tokens (`looks_like_identifier` in `src/lobe.rs`). Both metrics are always computed and
   reported (`surprisal`/`entropy`/`signal` in the JSONL). Verified the signals catch
   different things (surprisal → anomaly onset; entropy → uncertain interior of garbage).
5. ✅ **DONE — stream framing.** `--frame` pins a fixed Gemma chat wrapper (`THINKING_FRAME`
   in `src/main.rs`) so the stream is scored in the assistant-reasoning register rather than
   as raw text; appended to `--preamble`. Demonstrably shifts the surprisal distribution.
6. ✅ **DONE (core) — infinite-context streaming via cap + reset (StreamingLLM on iSWA).**
   The original "evict middle + `seq_add` shift" StreamingLLM move is NOT viable on Gemma here:
   gemma-4 uses an interleaved-SWA (iSWA) KV cache whose two sub-caches differ in size, so
   `can_shift()` is false and the first shifted `decode()` hard-`GGML_ABORT`s (would need
   `with_swa_full(true)`, which forfeits Gemma's memory design). So we do **cap + reset**: a
   stable pinned **system-prompt sink** (the attention sink + framing, replayed verbatim on every
   reset) + a **rolling window** of recent stream-token *ids* (`recent_ids`/`--keep-recent`);
   when seq 0 nears `n_ctx`, `roll()` clears it and re-primes sink + window, then continues. Same
   StreamingLLM result (bounded context = sink + recent), rebuilt instead of shifted (a
   sub-second recompute stall per reset; irrelevant for a reflex observer). Key impl facts:
   `with_kv_unified(true)` is REQUIRED (else the cache partitions `n_ctx` across the 2 sequences
   and seq 0 dies at `n_ctx/2`); eviction is seq-0-only so the interjection scratch seq (`GEN_SEQ`)
   is untouched; a `settle` counter suppresses firing right after a reset. Flags: `--evict
   {reset|off}`, `--keep-recent`, `--stats`. Core-validated: survives 35+ resets, byte-identical
   to `--evict off` pre-reset, surprisal stable on varied corpus text, and resets provably clear
   context (pos sawtooths; surprisal jumps at each reset boundary, softened by the rolling window).
   **Validated at scale (213k tokens, 64 resets, ~38 min, ctx 4096):** RSS flat at ~3.35 GB
   (3315–3378 MB), throughput flat at ~94 tok/s, and windowed surprisal mean stable (drift +0.067
   over the run — ~1.5%); post-reset surprisal ≈ pre-reset on varied text (no spike with
   keep_recent=512). The "infinite context" claim holds: unbounded streaming with bounded memory +
   compute + stable signal.
   **Re-validated at session-realistic ctx (full Moby Dick, 311,660 tokens, 10 resets, ~81 min, ctx
   32768, z 3.5, --interject ON):** RSS flat at ~3.6 GB (3511–3672 MB, 161 MB / 4.4% spread, no
   trend) — only ~250 MB over the 4k run, confirming iSWA's sublinear KV growth (only ~1/6 global
   layers hold the full window). Throughput flat at ~64 tok/s across all 10 resets (62–67 range;
   ~64→4k's 94 is the cost of attending over the bigger window, but it does NOT degrade as resets
   accumulate). `pos` sawtooths every ~28k tokens. Behavior held over the whole book (pathological
   front-matter + narrative + trailing license boilerplate): 285 interjections, 4% structure-lead,
   1 chapter-list continuation total. Takeaway: a multi-million-token coding session is just more of
   the same sawtooth — nothing here grows with token count.
7. ✅ **DONE — interjection as a branch, not inline.** Generation runs on a scratch KV
   sequence (`GEN_SEQ = 1`; `Lobe::interject` / `decode_seq` in `src/lobe.rs`), discarded
   with `seq_rm` afterward, so observation (seq 0) is never polluted (proven byte-identical).
   It's also chat-framed so the output is an observation, not a completion. Available in both
   the TUI (`--interject`) and headless (`--interject`, emits an `interjection` event).
   **FUSED concurrent (TUI default + headless `--fused`):** `Lobe::step()` co-batches the stream
   token (seq 0) and the in-flight interjection token (`GEN_SEQ`) into ONE decode, so observation
   **never stalls** while the lobe generates — the reply streams in *alongside* the text
   (`StepOutcome{step, interjection: InterjectStatus}`). This is the PRIMARY claim made visible:
   output generated concurrent to the input stream, not as an interrupting turn. The earlier
   context-fork `NoKvCacheSlot` was a cap+reset **sizing** bug (the roll didn't reserve the
   interjection's concurrent footprint = `ask + 2·interject_max`); fixed via `roll_margin` + an exact
   pre-fork fit check, validated crash-free over >10×n_ctx (see `docs/FUSED_CACHE_GO_NOGO.md` — this is
   why the Candle port was downgraded). The **timesliced** state machine (`interject_begin` /
   `interject_step` → `InterjectStep`, observation pauses during gen) survives as the headless one-shot
   `interject()` path used when `--fused` is off.
   **CHUNKED ASK-PREFILL (the prose-stall fix).** The interjection ask is often 200–350 tokens (delta
   span + novelty memory + recall); prefilling it in ONE decode at the fire froze the *prose stream*
   ~250–500ms (measured median ~256ms E2B, ~73ms × the same total E4B) — the fused gen-token co-batch
   kept the animation alive but the firing `step()` itself blocked. Now `start_fused_interjection` only
   STAGES the ask; `step()` co-batches `--prefill-chunk` (default 8) ask tokens per tick with the stream
   token (`GenKind::Prefill` / `advance_fused_prefill`), so the prose keeps scrolling and the aside
   starts a few ticks later ("musing…" until then). Only the final ask token requests logits (262k vocab
   → `fused::Lane` gained `logits: bool`); OUTPUT is byte-identical to one-shot (same final logits + RNG
   — `chunked_prefill_changes_neither_observation_nor_interjection` proves it on the mock). Smaller chunk
   = smoother scroll, later aside; the cap+reset fit-check reserves the prefill ticks' seq-0 growth.
   **Two context modes (`--interject-mode`, `Lobe::set_interject_mode`):** `snippet` re-encodes
   only the last `RECENT_TOKENS` as a fresh prompt and reacts to the trigger token (myopic);
   `context` (DEFAULT) forks the observer's FULL live seq-0 KV onto `GEN_SEQ` via
   `copy_kv_cache_seq` (cheap — shares cells, no recompute), appends a reflection turn, and
   generates, so it reflects on the *entire context window* rather than one token. `context` is
   cheaper on prefill (~40-tok ask vs ~180-tok snippet) and cleanest with `--frame`. Both run on
   the streaming `interject_begin`/`interject_step` machinery.
   ⚠️ **DESIGN PHILOSOPHY — surprisal is the HARNESS TRIGGER, not the interjection's subject; the
   aside is a free internal monologue, NOT structured output.** The surprisal spike decides *when* to
   speak — nothing more. The ask is NOT conditioned on "what was surprising": it simply hands the
   model the current chunk (the delta span) and asks it to discuss it — "Here is the passage the text
   has just reached… Give your aside on it — what's happening here, what it's doing, what it means."
   The surprising token is never named or referenced in the ask (gemma fixates on it → "the token X is
   notable because it appears…"); the builders (`interject_ask_context`/`interject_prompt_snippet`)
   don't even receive it — only the debug trace logs `trigger_token` for observability. The ask
   imposes no structure ("one sentence") or verdict ("on/off track") either; all of that suffocates
   the "thinking out loud" the lobe is for. Output length is bounded by `interject_max` (tokens),
   never the prompt — but it's a SOFT cap: past `interject_max` generation runs only to the next
   sentence boundary (`ends_sentence`), with a hard ceiling of `+INTERJECT_SENTENCE_SLACK` (64), so an
   aside never ends mid-clause. (The cap+reset pre-fork fit check uses the hard ceiling, `max+SLACK`.) Anti-fixation is NOVELTY MEMORY (the model's own recent asides, "find a fresh
   angle"), not surprise framing. The observer's *voice/modality* lives in the pinned system prompt
   (`DEFAULT_SYSTEM_PROMPT`, or `--preamble-file` for a persona like `personas/herzog.txt`); if you
   want to bias behavior (e.g. toward flagging problems vs free association), nudge the SYSTEM PROMPT,
   never the ask. (#8/RAG is the exception — there, structured/constrained output is wanted.)

   **Content-vs-structure bias (system-prompt nudge, the right lever).** On structural/boilerplate
   text (Moby Dick's front-matter: table of contents, etymology, copyright) the observer would muse
   on *form* — "it's a list", "the repetition", "linear structure" — because at that point the
   content genuinely *is* structure. The fix is a SYSTEM-PROMPT nudge (modality belongs there, not
   the ask): `DEFAULT_SYSTEM_PROMPT` now adds "what pulls your attention is the substance — a
   specific word, image, name, claim, or turn — not how the text is laid out; you barely notice
   structure, lists, or repetition." Measured: catalog interjections flip from mostly structure-talk
   to mostly specifics ("by a Sub-Sub-Librarian", "mockingly", "Nescio quid sit", "a grove of pikes
   appears"); a few still fight the bias where the region has no content to grab, which is honest. On
   a real reasoning stream it still nails anomalies ("the moon is made of green cheese … so random").
   **The deeper version (not built, RAG-coupled): structure should produce NO interjection at all,
   and the only reason to wake on something in boilerplate is a retrievable entity.** That composes
   *today* as content-bias prompt + `--identifiers-only` (the gate silences structure, fires only on
   entity-shaped tokens — Knights/Affidavit/Usher/Commonwealth, not "the"/"is"/list-ness) + `--rag`
   (each surviving entity → a native `search{query, source: mem|rag}` decision; `source: mem` IS
   "do I remember something about <foo>"). The missing piece is the feedback loop (feed the
   `<|tool_response>` back), which needs a real backend behind `run_retrieval` — see #8 increment (a).

   **Obsession control (the "it fixates on the same thing until it scrolls out" problem).** When a
   salient thing lingers in the window (e.g. Moby Dick's chapter catalog), naive context-mode keeps
   surfacing that one dominant feature, so the observer repeats near-identical observations. The fix
   is in `lobe.rs`, shared by TUI + headless, and has three layers (root → polish → backstop):
   - **Delta-focus (the ROOT fix).** The context-mode interjection ask spotlights the *delta* — the
     span of stream text *since the last fire* (`since_last_fire` ring → `last_span`, capped at
     `MAX_SPAN_TOKENS`) — rather than asking generically "what caught your attention." The FULL
     forked context is still present (the model's whole "world"; it can and does reach back), but
     pointing it at *new* content makes each interjection react to something different, so it varies
     on its own — no count-limiting needed. It's a **text buffer**, not KV positions, so it's immune
     to cap+reset / window slide. (Idea credit: this was the user's, and it's the real fix — it
     attacks the cause, not the symptom. Took the catalog from a wall of identical "the structure is
     linear" to ~26 distinct reactions: "unsplinterable glasses", "*Pehee-Nuee-Nuee*", etc.)
   - **Word-aligned spans (subword-boundary fix).** Surprisal fires on *subword* tokens, so a span
     cut at a fire boundary can start mid-word: the trigger `ETY` (start of "ETYMOLOGY") ends one
     span, leaving the next to begin with the orphan tail "MOLOGY" — and the model then fixates on it
     as a word ("a clipped syllable, hanging there"). `word_aligned()` trims a leading partial-word
     fragment (gemma renders word-starts with a leading space, so a span starting with a non-space
     char began mid-word → skip to the first whitespace). Applied to the delta span and the snippet/
     RAG `recent` buffers. NB: the *concatenation* was always correct (`last_span` is a separator-
     less `collect::<String>()`); the bug was the boundary, not the join.
   - **React-frame (anti-continuation polish).** Pasting the span invites the model to *continue*
     the prose (the system prompt's "experience the thought as your own" actively encourages this),
     so the span is curly-quoted and framed "quoted for you to react to, NOT to continue." That took
     continuations from ~7/33 to ~0 on raw/quote-list spans.
   - **Novelty memory in the ask (1b) + SAMPLING (the two that actually killed fixation; proven
     with `--debug-log`).** Two findings, only decisive together:
     - *Novelty memory (1b):* the context ask now shows the model its last 1–2 interjections ("You
       recently said: …") and asks for a fresh angle. `record_interjection` records every emitted
       interjection unconditionally (the `recent_interjections` ring); `interject_ask_context`
       injects them. ALONE this barely helped — the structured log proved why: on repetitive content
       (the chapter catalog) a 2B under **greedy** decoding *ignores* it — 13/34 outputs reproduced
       verbatim the exact text they were shown they'd just said, 7/34 byte-identical to the prior.
     - *Sampling (`--interject-temp`, DEFAULT 0.7; `--interject-top-p` 0.95):* the actual fix. Greedy
       is a deterministic collapse onto the single dominant phrasing per region; the varied
       observations are *latent below the argmax*. Temperature+top-p (`sample_topp`, applied ONLY to
       interjection generation — observation scoring is provably byte-identical regardless, md5-
       verified) surfaces them AND lets 1b finally function (greedy couldn't act on "fresh angle").
       Measured on the catalog: reproduces-shown 13/34→2/34, verbatim-prev 7/34→2/34, distinct
       openings 19/34→31/34, and it *develops a thread* across fires (gears → inventory → the tension
       between them) instead of repeating. xorshift64 RNG, constant-seeded → runs reproducible.
   - **Refractory** (`--refractory`, default 64): coarse post-fire cooldown so it doesn't re-fire on
     every spike in a dense region. NB it's *input-shaping*, not a filter — at `--refractory 0` the
     observer fires nearly every token, so consecutive deltas are ~1 token and even a perfect ask
     gets "nothing new" → ~20× verbatim run-ons. Don't test fixation at refractory 0.
   - **Dedup** (`--dedup`, default 0 = OFF, opt-in backstop): suppresses an emitted repeat (opening
     stem match or char-shingle Jaccard). Demoted from the primary mechanism to a backstop now that
     1b+sampling fix the cause; `interjection_is_novel` is a pure check (recording is separate), so
     dedup gates emission only, never the novelty memory. Per the design principle (don't filter
     fixations, *stop* them), it's off by default.
   - **NOT the fix:** adaptive/EWMA baseline (`--adapt`, 0/off) and the old delta-focus-only /
     dedup-only attempts. The catalog is a train of *spikes*, repetition was a *content+greedy*
     problem; only making generation non-deterministic (sampling) addressed the cause.
   - **The structural-region GATE (mode #3) — investigated, MEASURED, DEFERRED (don't naively
     re-attempt).** The idea: in repetitive/boilerplate regions (chapter list, license, etymology
     list) the model has nothing new, so *don't fire* there. But a content-diversity gate does NOT
     cleanly detect those regions: measured on Moby Dick, the chapter-list and narrative have the
     **same median unique-word ratio (0.88)** — the list looks diverse because the titles are all
     different words; only "CHAPTER"/digits repeat. The signals that *would* separate them (the
     literal token "CHAPTER", digit density) are corpus-specific and won't generalize. Likewise
     `--identifiers-only` is the wrong gate (it keys on the *trigger token* shape and would suppress
     the *good* terse non-identifier observations like "the date"/"the shift in tone" — fixation is a
     property of the interjection *content*, only loosely coupled to the trigger token). So the gate
     is a **real research problem (the long-term mode #3), not a quick wire-up.** For the demo we
     sidestep it: read the *narrative* (front-matter is the least-watchable part anyway — see the
     `--skip-to` demo command) + `--dedup 0.5` as a cheap repeat backstop.
   **Methodology note:** all the above was settled with `--debug-log` (see the Observability section)
   — per-fire prompt/span/novelty-memory/output dumps, not screenshots. Don't regress to guessing.
   Reading the output is the eval; metrics only flag failure modes (see [[interjection-eval-is-judgment-based]]).
8. ✅ **DONE (first cut) — RAG hook via NATIVE TOOL CALLING (not GBNF).** Key discovery: gemma-4
   has a full native tool protocol (`<|tool>` define / `<|tool_call>call:name{…}<tool_call|>` /
   `<|tool_response>`), all single special tokens, and E2B emits clean, well-formed calls with
   sensible queries (it even expands the trigger token — `ceti`→`Sperma-ceti`, `Macy`→`Obed Macy`).
   So we skip grammar entirely (no per-token vocab-mask latency): `--rag` (headless) defines a
   `search(query, source)` tool in gemma-4's native format, the observer thinks + (if warranted)
   emits a native call, and `Lobe::rag` → `parse_rag_output` parses it into a free `thought`
   (emitted as an `interjection`) + a `RetrievalDirective {source: Mem|Rag, query}` (emitted as a
   `retrieval` event). Think→act emerges on its own (no forced `<|channel>` or budget-forcing
   needed for the basic case). Abstain = no call → just the thought. `run_retrieval` (in `main.rs`)
   is the stub seam (returns None → `found:false`); a real backend (file-memory / external KB)
   replaces only that body. Combine with `--identifiers-only` to gate retrieval to entity-shaped
   triggers. **Remaining increments (not yet built):** (a) the feedback loop — feed the hit back as
   a native `<|tool_response>` and continue generating (the agentic part; needs a backend);
   (b) `--rag` in the TUI (streaming, like interjection); (c) the explicit `<|channel>thought` +
   budget-forcing modal version if we want to *guarantee* the think-then-act split rather than rely
   on it emerging.

**🎉 PUNCH-LIST COMPLETE: #1–8 all done.** The prototype compiles + runs on Metal with gemma-4-E2B,
streams unboundedly (cap+reset), fires on a pluggable surprisal/entropy signal, interjects as a
free first-person monologue over its full forked context (streaming, refractory-gated), and now
emits native tool calls for retrieval.

## Design context (why these choices)

- The observer must stay *reflexive and shallow*. A 2B model is a good salience/pattern
  detector and a bad second brain; the architecture only works if you ask it to be the
  former. Delegate reflexes (anomaly flags, "this looks wrong", loop detection), not
  reasoning.
- The hard part is **calibration**, not plumbing. Whether the trigger fires usefully is the
  thing most likely to disappoint, which is exactly why the TUI exists and why headless can
  emit every step for offline threshold sweeps.
- This prototype is the Rust path. For fastest iteration on an M-series Mac, prototyping the
  cache policy + calibration in Python `mlx-lm` is also reasonable; this crate is the lean,
  shippable, in-process-with-a-TS-harness target on Metal.
