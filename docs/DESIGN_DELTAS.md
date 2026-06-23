# DESIGN_DELTAS.md — what we built that the original design didn't anticipate

This is the changelog of *thinking*, not of code. The original design = the scaffold + the punch-list
in `CLAUDE.md` + the three spec docs (`INFINITE_CONTEXT.md`, `GRAMMAR.md`, `CANDLE_DESIGN.md`). This
doc records the **deltas**: things we discovered the design got wrong, places we deliberately chose
differently than spec'd, and net-new mechanisms invented while making the PoC actually work.

If you only read the punch-list (which is now all ✅), you'd think we executed a plan. We mostly
*re-derived* the plan against reality — gemma-4 is not gemma-2/3, a 2B is not a reasoner, and the
"obsession" failure mode wasn't on anyone's radar. Those are the interesting parts.

---

## 1. Discoveries — where reality contradicted the design

These weren't choices; the design was simply wrong or silent about them, and the model/crate told us
so. Each was a real bug or a hard constraint, found by building and looking.

### 1a. gemma-4's chat format is NOT gemma-2/3's (the most expensive bug)
The original `GRAMMAR.md` prompt used `<start_of_turn>` / `<end_of_turn>`. **Those tokens are not in
gemma-4's vocab.** gemma-4 uses `<|turn>` (id 105, opens) + role + `\n` + content + `<turn|>` (id 106,
closes); the generation prompt is `<|turn>model\n`; EOS is `<eos>` (id 1) and `is_eog_token` does
*not* flag `<turn|>`. Using the wrong markers tokenizes them as **literal text**, so the model never
enters a chat turn — it *completes* the prose (and leaks literal `<...>` markers) instead of replying.
This was the root cause of the "interjections are nonsensical / barfing the prompt" symptom. The user
correctly insisted it was a parsing bug, not "2B metacognition," before we found it.
**Rule going forward:** if you swap models, re-derive the chat format from the GGUF
`tokenizer.chat_template` — do not assume the prior Gemma generation's tokens.

### 1b. `with_kv_unified(true)` is REQUIRED, or seq 0 dies at n_ctx/2
With two sequences (observation + interjection scratch) and a non-unified cache, llama.cpp
**partitions `n_ctx` across the sequences**, so the observation sequence hits `NoKvCacheSlot` at
≈`n_ctx/2` instead of `n_ctx`. Not documented in the design; found by crashing. `with_n_seq_max(2)` +
`with_kv_unified(true)` is the working combination.

### 1c. gemma-4's iSWA cache genuinely cannot context-shift
The textbook StreamingLLM move (evict middle + `seq_add` to shift the tail + RoPE re-rotation) is
**not viable** on gemma-4 in llama.cpp: the interleaved sliding-window-attention cache discards local
layers' KV once outside the 512 window, so `can_shift()` is false and the first shifted `decode()`
hard-`GGML_ABORT`s. (`--swa-full` would allow it but forfeits Gemma's whole memory design.) The
spec author reached this conclusion independently in `INFINITE_CONTEXT.md`; we confirmed it the hard
way. This is *why* #6 is cap+reset and not a shift (see §2a).

### 1d. gemma-4 has a full native tool-calling protocol (a happy discovery)
The design assumed a 2B couldn't be trusted to emit structured output without a GBNF grammar. But
gemma-4 ships a **complete native tool protocol** in its template — define (`<|tool>declaration:…
<tool|>`), call (`<|tool_call>call:name{…}<tool_call|>`), and response (`<|tool_response>…
<tool_response|>`) — all as single special tokens. And E2B emits clean, well-formed calls with
sensible queries, even *expanding* the trigger token (`ceti`→`Sperma-ceti`, `Macy`→`Obed Macy`). This
discovery rewrote #8 (see §2b).

---

## 2. Departures — where we deliberately chose differently than the spec

### 2a. #6 infinite context: cap+reset, not context-shift
Forced by §1c. Instead of sliding, we **cap + reset**: a pinned system-prompt sink (replayed
verbatim) + a rolling window of recent stream-token *ids*; when seq 0 nears `n_ctx`, `roll()` clears
it and re-primes sink + window. Same StreamingLLM *result* (bounded context = sink + recent),
rebuilt instead of shifted — a sub-second recompute per reset, irrelevant for a reflex observer.
Validated at scale: **213k tokens, 64 resets, ~38 min, RSS flat at ~3.35 GB, ~94 tok/s flat,
windowed surprisal drift ~1.5%.** (The seamless per-layer version lives in `CANDLE_DESIGN.md` as a
conditional future, gated on the re-prefill hiccup actually being felt — it isn't, yet.)

### 2b. #8 retrieval: native tool calls, NOT GBNF
`GRAMMAR.md` spec'd a delimited GBNF grammar (`NONE | SEARCH mem|rag <query>`) with a decomposed
logprob-classification fallback. Given §1d we **skipped grammar entirely**: define a `search(query,
source)` tool in gemma-4's native format, let the observer think and (if warranted) emit a native
call, and parse it (`parse_rag_output`) into a free `thought` + a `RetrievalDirective {source, query}`.
Why this is *better*, not just easier:
- **No per-token vocab-mask latency** (the grammar tax the design feared).
- **Think→act emerges on its own** — no forced `<|channel>` or budget-forcing needed for the basic
  case; the model reasons, then calls.
- **Abstain is free** — no `NONE` token; the model just doesn't emit a call.
- **The feedback loop is native too** — `<|tool_response>` is gemma-4's own mechanism for handing a
  result back, so the agentic loop is the model's native protocol, not a bolt-on.

GBNF stays in the back pocket (the `common` feature + a token-mask sampler) only if native-format
reliability ever disappoints; it hasn't.

### 2c. Interjection: a branch on a scratch sequence, not inline
The first cut (implied by "observe → generate → resume" sharing one context) decoded the interjection
*into the observation sequence*, so the observer attended to its own output — slowing observation,
shifting later surprisals, and producing a *completion* rather than an observation. We moved
generation to a **scratch KV sequence** (`GEN_SEQ = 1`), discarded with `seq_rm` after — proven
byte-identical observation with/without `--interject` — and made it a **chat-framed reframe** so the
output is genuine commentary, not a continuation.

---

## 3. Net-new — mechanisms invented during the work, in no spec

### 3a. The "obsession" failure mode and its three-layer fix (the biggest net-new)
Nobody anticipated this: when a salient thing **lingers in the context window** (e.g. Moby Dick's
chapter catalog), naive context-mode keeps reflecting on that one dominant feature, so the observer
emits a *wall of near-identical interjections* until it scrolls out. The fix, root → polish →
backstop, all in `lobe.rs`:
- **Delta-focus (the root fix; the user's idea).** The interjection ask spotlights the *delta* — the
  span of stream text *since the last fire* — instead of asking generically "what caught your
  attention." The full forked context is still present (it can and does reach back), but pointing it
  at *new* content makes each interjection react to something different, so it varies on its own. It's
  a **text buffer** (`since_last_fire` → `last_span`), not KV positions, so it's immune to
  cap+reset / window slide. Took the catalog from a wall of identical "the structure is linear" to
  ~26 *distinct* reactions.
- **React-frame (anti-continuation polish).** Pasting a span invites the model to *continue* the
  prose (the system prompt's "experience the thought as your own" encourages exactly that), so the
  span is curly-quoted and framed "quoted for you to react to, NOT to continue." Cut continuations
  from ~7/33 to ~0.
- **Word-aligned spans (subword-boundary fix).** Surprisal fires on *subword* tokens, so a delta
  span cut at a fire boundary can start mid-word — the trigger `ETY` (start of "ETYMOLOGY") ends one
  span, orphaning the tail "MOLOGY" into the start of the next, and the model fixates on it as a word
  ("a clipped syllable, hanging there"). `word_aligned()` trims a leading partial-word fragment.
  Subtlety worth remembering: the *concatenation* was never the bug (spans are a separator-less
  `collect::<String>()` of detok'd pieces, which faithfully reconstructs text — exactly how streaming
  generation prints); the bug was the span *boundary*. Easy to misdiagnose as a join/spacing issue.
- **Refractory + dedup (backstops).** `--refractory` (post-fire cooldown) and `--dedup` (opening-stem
  + char-shingle near-duplicate suppression) mop up residual repeats/echoes. The opening-stem window
  (`DEDUP_OPENING_WORDS`) was tuned to 2 by measurement: it collapses theme recurrence ("The
  repetition…", "The list…") on a catalog without merging genuinely-distinct reactions, because the
  model varies its openings enough and `DEDUP_HISTORY` bounds the comparison to recent interjections.
- **What did NOT work:** an adaptive EWMA surprisal baseline (`--adapt`). The catalog is a *train of
  spikes*, not a plateau, so habituating the baseline barely moved firing — the repetition was a
  *content* problem (dominant-feature reflection), not a count problem. Kept as an experimental flag.

### 3b. Interjection as a free first-person monologue (a design philosophy, not a feature)
We converged — via the user's feedback — on the interjection being a **secondary stream of
consciousness**, NOT structured/prescriptive output. Concretely: the per-trigger ask is minimal and
hands the floor back ("something caught your attention; think out loud"); it does **not** name the
token (gemma fixates on it), impose a structure ("one sentence"), or a verdict ("on/off track"). The
observer's *voice/modality* lives in the pinned system prompt; length is bounded by `interject_max`
(tokens), never the prompt. (#8/RAG is the explicit *exception* — there, structured output is wanted.)

### 3c. Two interjection context modes
`--interject-mode`: **`snippet`** re-encodes only the last N tokens as a fresh prompt (myopic);
**`context`** (default) forks the observer's FULL live seq-0 KV onto `GEN_SEQ` via `copy_kv_cache_seq`
(cheap — shares cells, no recompute) and reflects on the whole context window. Neither was in the
original design; `context` is what makes delta-focus (§3a) meaningful.

### 3d. Streaming interjection via an explicit state machine — no async runtime
The interjection is pumped one token per UI tick (`interject_begin` / `interject_step` →
`InterjectStep`), so the reply streams in live and the event loop never freezes. Deliberately a
single-threaded tick loop with an explicit state machine **instead of Tokio**: the work is GPU-serial
anyway, so async buys nothing but a runtime, `Send`/`Sync` fights with `LlamaContext`, and harder
interleaving. (The user specifically called this out as the right pragmatic call.)

### 3e. Pluggable trigger signal + identifier gate
`--signal {surprisal|entropy}` selects what the Welford z-score thresholds on (unexpected token vs
model uncertainty at the position); `--identifiers-only` gates firing to identifier/entity-shaped
tokens. Both metrics are always computed and reported. (This was punch-list #4, but the *entropy*
option and its framing as "where the model would want to think" came out of the design conversation,
not the original spec.)

---

## 4. Conceptual clarifications that changed how we reason about the project

Not code, but load-bearing understanding we didn't start with:

- **"StreamingLLM" is ONLY the KV trick (= #6).** The paper (2309.17453) is attention sinks + rolling
  window → bounded-memory infinite *input*. It has nothing to do with triggers, output tokens, or
  interjections. The **surprisal observer** (trigger + interjection) is this project's *own* idea. The
  project name conflates them; don't.
- **Detection must stay passive — there's a circularity.** "Could thinking *effort* be the trigger?"
  No: thinking means the observer *generating*, which requires it to *stop consuming the stream* — a
  decision that must be made *before* any thinking, i.e. from the forward pass (surprisal/entropy).
  Thinking is the **response**, not the detector. The free, passive, per-token logit read is the only
  thing that can answer *when?* — and that free detection is the whole reason a small model can ride
  alongside a frontier model continuously. The entropy signal (§3e) is the passive proxy for "where
  it would want to think."
- **Thinking budgets are inference-level, not model-level.** The modal "think up to N tokens then
  answer" structure is *trained* (the model knows `<|channel>thought`…`<channel|>`), but the hard cap
  is **budget forcing** — the harness injects the close token at the cap, or suppresses it + injects
  "Wait," to *extend*. Budget forcing is the same family of mechanism as GBNF (shape output by
  forcing/banning tokens). This is why the channel structure could, in principle, host a free thought
  phase + a structured tool-call phase in one generation — noted as a future direction.

---

## 5. Status of the deltas

| Delta | State |
|---|---|
| 1a gemma-4 chat format | Fixed, baked into `Lobe` + `main.rs` framing |
| 1b `kv_unified` | Fixed in `Lobe::new` |
| 1c iSWA can't shift | Confirmed; drove §2a |
| 1d native tool protocol | Discovered; drove §2b |
| 2a cap+reset | ✅ done, validated at 213k tokens |
| 2b native tool-call RAG | ✅ first cut; feedback loop / TUI wiring / modal-forced split are the remaining increments |
| 2c scratch-sequence interjection | ✅ done, byte-identical observation proven |
| 3a obsession fix | ✅ done (delta-focus + react-frame + refractory + dedup) |
| 3b free-monologue philosophy | ✅ done; the standing design principle for interjections |
| 3c context/snippet modes | ✅ done |
| 3d streaming state machine | ✅ done (TUI streams live, no freeze) |
| 3e pluggable signal + gate | ✅ done |
| 4 conceptual clarifications | Recorded here + in `CLAUDE.md` |

The punch-list says "done." This doc says *how the plan changed on contact with gemma-4 and a 2B's
actual behavior* — which is the part worth remembering.
