# StreamObserver

Proof-of-concept for an LLM inferencing implementation that consumes streaming input with the ability
to concurrently generate output based on the stream.

Importantly, it does so without compacting, and without yielding to a traditional user/assistant agent turn
structure; everything happens concurrently.

The demo in this repo uses Gemma E4B to read novels off
[Project Gutenberg](https://www.gutenberg.org/) and do a Werner Herzog impression about them, in a way
the real Werner Herzog would almost certainly find revolting.

I'm okay with that.

![StreamObserver reading Moby Dick by the fire, as Werner Herzog](demo.gif)

## So what is it, really

The input text is streamed into the observer token-by-token, [teacher-forced](https://en.wikipedia.org/wiki/Teacher_forcing) through a small local model.
Because we're pushing tokens through a model one at a time, we can take advantage of knowing the probability of the next token to compute
the [surprisal](https://en.wikipedia.org/wiki/Information_content) for that token when it actually arrives (`-ln P(token | everything so far)`),
and we get this for free by fully controlling the forward pass and the token stream. No second model, no classifier, no extra work.
We use this as a proxy for "something interesting just happened"; we force the text through the model as if the model was writing it itself,
and then when the model is surprised by "its own" output, we use that as our activation signal for "talk about this content out loud".

When we hit this signal, that's our cue to generate the musing/output/interjection/aside/whatever.
We are able to do this concurrent with the observer without instantiating a second model by fusing/batching the forward passes of both sequences;
we use a scratch KV sequence on the *same* model decode pathway to drive this.

No "turns" involved, no buffering to spoof concurrency; we're driving both sequences through the model at the same time, allowing the model to
continue to actively observe while also interjecting.

If you want the gory details, Claude maintained a pile of design notes in [`docs/`](docs/), plus the running
development log in [`CLAUDE.md`](CLAUDE.md) — every hard-won decision, dead end, and "well, *that*
didn't work" from building this.

### What About Context Windows?

The architeture here is built on top of a foundation that is heavily inspired by the [StreamingLLM](https://arxiv.org/abs/2309.17453) findings,
wherein maintaining a very small number of "attention sinks" (stable prefix tokens) allows for you to treat the rest of your context as a sliding
window without coherence/perplexity suffering.

Here, our "attention sinks" are the tokens in our persona prompt; Werner Herzog is keeping the whole thing from blowing up. As long as the model
knows that it's a budget-rate Werner Herzog, it won't freak out when its context window slips around underneath it.

What this means is that we don't ever have to worry about compacting; we just slide the old tokens out to make room for the new tokens as they come in,
while maintaining our persona prompt as the stable prefix.

This does *not* equate to infinite recall/memory; context is still context. It simply means that the behavior of the model is stable under this
sliding window.

### Augmenting Memory

To give some amount of recall over stuff that may slide out of context without compacting, we chunk-and-embed the input corpus using an extremely
small embedding model.

Whenever a surprisal threshold is crossed, we perform a small hybrid vector search operation over the chunked corpus based on the token that
triggered the interjection, and supplement the context handed off to the interjection pass with the results of the hybrid search.

## The party trick

That gif up top is gemma-4-E4B reading *Moby Dick* by a (pixel-art) fire, impersonating Werner Herzog, musing and
recalling as it goes. The whole showcase config is baked in as the defaults, so you only pass the text
and where to start reading:

```console
$ ./target/release/stream-observer --input corpus/pg2701.txt --skip-to "Call me Ishmael"
```

`space` pauses · `q` quits · `+`/`-` nudge the surprise threshold while it runs. You'll want a real,
wide, truecolor terminal for the fireside scene.

## Dependencies

You'll need to download models and text to run the demo yourself. The defaults use the files and their expected download locations are below, but you can use any GGUF to experiment with it yourself;
these can all be set with CLI flags.

| Path | What | Notes |
|---|---|---|
| `models/gemma-4-E4B_q4_0-it.gguf` | the reader (default model) | Google's QAT 4-bit instruct GGUF. The persona needs the bigger E4B; E2B collapses into mush trying to be Herzog. Override with `--model`. |
| `models/gemma-4-E2B_q4_0-it.gguf` | leaner reader | Faster, fine without a persona. Use it with `--model …E2B…`. |
| `models/harrier-oss-v1-270m-BF16.gguf` | retrieval embedder | A tiny embedding GGUF for the hybrid (lexical + semantic) recall. Swap with `--rag-embed-model`, or `--rag-embed-model ""` for lexical-only, or `--no-rag` to turn recall off entirely. |
| `corpus/pg2701.txt` | the book | *Moby Dick*, plain text, from [Project Gutenberg #2701](https://www.gutenberg.org/ebooks/2701). Any UTF-8 text works — point `--input` at whatever you like. |

There's a bundled `sample_thinking.txt` so a bare `stream-observer` does *something* without a corpus.

## Build

You'll need Rust and `clang` (the default backend compiles llama.cpp from source, so the first build
is slow — go get a coffee):

```console
$ cargo build --release --features metal   # Apple Silicon (Metal)
$ cargo build --release --features cuda     # NVIDIA
$ cargo build --release                     # CPU only
```

## Modes

The CLI is flat: `--mode` picks the frontend, everything else is a top-level flag.

| `--mode` | What it is |
|---|---|
| `demo-tui` *(default)* | The show: a clean stage (or the `--scene` pixel-art study) with the prose streaming past and the asides forming alongside it. |
| `debug-tui` | The instrument I actually used to tune it — live surprisal sparkline, z-score heatmap, the trigger list, knobs. |
| `headless` | `stdin` → JSONL on `stdout`. The shape you'd wire into something real. For a lean pipe, add `--no-rag` (and `--no-interject` if you just want the raw surprisal numbers). |

The showcase is the default, and every default-on piece has an off switch: `--no-scene`, `--no-rag`,
`--no-frame`, `--no-interject`, `--deterministic`, `--preamble-file ""` (drop the persona). Run
`--help` for the whole surface.

## Acknowledgements

The overall design and implementation here owe a real debt to **StreamingLLM**
("Efficient Streaming Language Models with Attention Sinks," Xiao et al., 2023 —
[arXiv:2309.17453](https://arxiv.org/abs/2309.17453)). Their result is that a handful of "attention
sink" tokens pinned at the front of the context, plus a rolling window of the recent past, is enough to
keep a model coherent over a token stream of effectively unbounded length, in bounded memory, with no
fine-tuning. That is the foundation everything else stands on — it's what makes it possible for a small
model to read an entire novel front to back without compaction, without summarization, and without the
context (or the memory footprint) blowing up partway through.

The surprisal trigger and the concurrent interjection are this project's own contributions on top. But
the streaming substrate they ride on is StreamingLLM's, and a serious read of the paper is the best way
to understand why any of this works at all. Credit belongs there.

## License

MIT. Knock yourself out. See [LICENSE](LICENSE).
