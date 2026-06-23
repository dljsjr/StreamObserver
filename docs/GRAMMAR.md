# RETRIEVAL_GRAMMAR.md — handoff for punch-list item 8

Implement trigger-gated retrieval: on a fired token, the observer emits a constrained directive
(abstain, or search file-memory / RAG), the harness runs the lookup, and the result is surfaced.
Grounded in the current `src/lobe.rs` (llama-cpp-2 0.1.150). This reuses the scratch-sequence
machinery you already built for `interject`.

## Where it fits

- **Gated by the trigger.** Retrieval only runs when a token fires (the surprisal/z spike). It is
  not a per-token operation. The objective trigger is what keeps retrieval off noise — see Caveats.
- **Runs on the scratch sequence** (`GEN_SEQ = 1`), exactly like `interject`: clear it, prefill a
  framed prompt, decode a short constrained reply, clear it again. Sequence 0 (observation),
  `pos`, and `last_logits` are never touched.
- **Division of labor:** the grammar enforces the output *format* so a 2B model can't malform it;
  the prompt supplies the *judgment*. The grammar cannot make the decision good — only parseable.
- **Abstain is first-class.** `NONE` is a one-token bail and should be the common answer. Most
  surprising tokens are not lookups (a number, a typo, a non-sequitur). Forcing a query is worse
  than none.

## The grammar

Minimal delimited format — every extra delimiter is a token the model has to place exactly, so we
avoid JSON. `RETRIEVAL_GBNF`:

```gbnf
# Either abstain, or one retrieval directive.
root    ::= none | search
none    ::= "NONE"
search  ::= "SEARCH " source " " query
source  ::= "mem" | "rag"
query   ::= [^\n]+
```

Prefer the **bounded** query rule if your llama.cpp supports `{m,n}` repetition (current builds do)
— it stops a runaway query and gives the grammar a clean terminal:

```gbnf
query   ::= [^\n]{1,80}
```

Design notes:
- `none` / `search` are plain ASCII keywords — robust under tokenization, trivial to parse.
- `source` is a 2-way enum so routing (file-memory vs RAG) is a forced choice, not parsing.
- `query` is free-form on purpose. You cannot grammar-constrain a *good* query, only the envelope;
  the query's content is steered from the prompt (echo the surprising identifier — see below).

## The prompt

Gemma chat format, same framing style as `interject` (which you found E2B is sensitive to — keep
the recent block before the instruction, keep the surprising token and the ask last):

```rust
const RETRIEVAL_PROMPT: &str = "\
<start_of_turn>user
You are a fast reflex observer watching another AI model think out loud. You decide whether a \
surprising token is worth looking up. Here is the most recent text it produced:

{recent}

The token it just emitted, \"{surprising}\", was statistically surprising in that context.
If it is an unfamiliar identifier, name, term, or reference worth retrieving, respond with:
  SEARCH mem <query>   to look in this session's own memory / earlier context
  SEARCH rag <query>   to look in the external knowledge base
Echo the surprising identifier verbatim in <query>.
If it is NOT something to look up (a number, a typo, an odd phrasing, a non-sequitur that isn't a \
reference), respond with exactly:
  NONE
Respond with one line and nothing else.<end_of_turn>
<start_of_turn>model
";
```

Fill `{recent}` from the existing `self.recent` window and `{surprising}` from the fired token's
text. Telling it to *echo* the identifier matters: a 2B is more reliable copying the surprising
token into the query than composing a query, and "echo from context" can't be expressed in static
GBNF — it has to come from the prompt.

## Generation: grammar-constrained sampling

This is the one real new mechanic vs `interject` (which uses raw `argmax`). Grammar sampling
**masks** illegal tokens, so you can't argmax raw logits — you go through the sampler chain, and
you **must accept each chosen token so the grammar state advances**. Forgetting `accept` is the
single most common bug here (the grammar never progresses).

```rust
pub fn decide_retrieval(&mut self, surprising: &str, max: usize) -> Result<RetrievalDirective> {
    self.ctx.clear_kv_cache_seq(Some(GEN_SEQ as u32), None, None)?;

    let recent: String = self.recent.iter().map(String::as_str).collect();
    let prompt = RETRIEVAL_PROMPT
        .replace("{recent}", recent.trim())
        .replace("{surprising}", surprising);

    // FRAGILE: grammar sampler constructor. C symbol llama_sampler_init_grammar(vocab, gbnf, root).
    // In llama-cpp-2 0.1.150 check `sampling::LlamaSampler` for `grammar(...)`; arg order is likely
    // (&LlamaModel, &str gbnf, &str root). Build a FRESH sampler each call (grammar is stateful).
    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::grammar(self.model, RETRIEVAL_GBNF, "root"), // masks illegal tokens
        LlamaSampler::greedy(),                                    // argmax over the legal set
    ]);

    let prompt_tokens = self.tokenize(&prompt, true)?;
    self.decode_seq(&prompt_tokens, 0, GEN_SEQ)?; // prefill on scratch seq (logits land at last idx)

    let mut out = String::new();
    let mut pos = prompt_tokens.len() as i32;
    let mut last_idx = (prompt_tokens.len() - 1) as i32;
    for _ in 0..max {
        // FRAGILE: sampler.sample(&ctx, idx) reads logits from the context, applies the chain,
        //          returns a token. C symbols llama_sampler_sample / llama_sampler_accept.
        let tok = sampler.sample(&self.ctx, last_idx);
        sampler.accept(tok); // <-- REQUIRED: advances grammar state. Do not omit.
        if self.model.is_eog_token(tok) { break; }
        let piece = self.detok(tok);
        if piece.contains('\n') { out.push_str(piece.trim_end_matches('\n')); break; }
        out.push_str(&piece);
        self.decode_seq(&[tok], pos, GEN_SEQ)?; // decode_seq computes logits at its last index (0)
        pos += 1;
        last_idx = 0;
    }

    self.ctx.clear_kv_cache_seq(Some(GEN_SEQ as u32), None, None)?;
    Ok(parse_directive(&out))
}
```

Stop conditions: EOG, a newline, or `max` tokens. With the bounded `{1,80}` query the grammar also
naturally runs out of legal non-EOG tokens, so greedy will pick EOG on its own.

### Alternative if the grammar sampler fights you (no grammar engine)

The grammar is tiny enough to decompose, using only machinery you already have (`decode_seq` +
`argmax`), zero grammar-API dependency — and this is arguably *more* robust on a 2B:

1. **Classify** by scoring fixed continuations: prefill the prompt, then compare the model's
   logprob of `"NONE"` vs `"SEARCH"` (forced-decode each candidate string, sum token logprobs,
   pick higher). If `NONE`, return abstain.
2. **Source**: same logprob comparison of `"mem"` vs `"rag"`.
3. **Query**: plain free generation (no grammar) until newline / EOG / cap.

This trades a few extra decode passes for eliminating the FRAGILE grammar-sampler surface and
making the gate deterministic. Pick whichever you can get working first; the parsed output is the
same `RetrievalDirective` either way.

## Parsing

```rust
pub enum RetrievalDirective {
    None,
    Search { source: Source, query: String },
}
pub enum Source { Mem, Rag }

fn parse_directive(s: &str) -> RetrievalDirective {
    let s = s.trim();
    if let Some(rest) = s.strip_prefix("SEARCH ") {
        let (src, query) = rest.split_once(' ').unwrap_or((rest, ""));
        let source = if src == "rag" { Source::Rag } else { Source::Mem };
        return RetrievalDirective::Search { source, query: query.trim().to_string() };
    }
    RetrievalDirective::None
}
```

The grammar guarantees one of the two shapes, so parsing is total; default to `None` on anything
unexpected.

## Retriever seam + feedback loop

The actual lookup backends are out of scope for this item — define the seam and stub them:

```rust
pub trait Retriever {
    fn retrieve(&self, source: Source, query: &str) -> Result<Option<String>>;
}
```

The harness owns the impls (file-memory = this session's stored context; rag = external KB). On a
`Search` directive, call `retrieve`; on `Some(snippet)`, surface it.

**Feedback — recommended for the PoC:** surface the snippet via a *follow-up interjection on the
scratch sequence* — i.e., prompt `interject`-style with "you looked up X and found: {snippet};
note in one line whether it resolves the surprise." This keeps everything on `GEN_SEQ`, never
mutates observation, and composes with what exists.

**Do NOT (yet)** inject the retrieved snippet into sequence 0. It's the richer "limbic feeds the
cortex" move, but it reintroduces the observation-pollution problem `interject` was rewritten to
avoid, and it collides with the cap+reset bookkeeping (you'd have to push the injected tokens into
`recent_ids` so they survive a roll). Defer it until the basic loop is proven.

## Events

- **Headless:** emit a `retrieval` JSONL event on each non-abstain directive — `{event, source,
  query, found: bool, snippet}` — alongside the existing `trigger` events.
- **TUI:** a panel like the triggers/interjections panels, showing the query, source, and a
  one-line result.

## Caveats

- **Metacognition is the weak link, not the format.** The grammar makes the output parseable; it
  does not make "is this worth looking up" correct. A 2B's judgment here is unreliable. Mitigate by
  (a) only invoking `decide_retrieval` when the fired token is *identifier-shaped* — this is where
  item 4's better trigger pays off; don't run retrieval on every spike — (b) keeping `NONE` cheap
  and the default, and (c) if it over-fires, bias toward `NONE` (e.g., require the model's `SEARCH`
  margin over `NONE` to clear a threshold in the decomposed path).
- **Cost.** Each retrieving trigger is a scratch prefill + short gen + a lookup + maybe a follow-up
  interjection — heavier than a bare interjection. Gating to identifier-shaped triggers keeps it
  affordable.
- **Suggested refactor:** `interject` and `decide_retrieval` share the scratch-sequence dance
  (clear → frame → `decode_seq` prefill → gen loop → clear). Extract
  `fn generate_on_scratch(&mut self, prompt: &str, max: usize, sampler: Option<LlamaSampler>) -> Result<String>`
  and have both call it (interjection passes `None` → greedy; retrieval passes the grammar sampler).
  Reduces the surface where the FRAGILE scratch-seq code can drift.
```