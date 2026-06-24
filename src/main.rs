//! streaming-lobe — a tiny local model that *observes* a token stream and flags the
//! tokens it finds surprising, optionally interjecting. Two frontends, one core:
//!
//!   headless : reads the stream from stdin, emits one JSON object per scored token on
//!              stdout (JSONL). This is the Sandpiper-facing shape — tee a frontier
//!              model's thinking-delta stream into stdin and consume the events.
//!
//!   tui      : paces a transcript through the same observer and shows surprisal live,
//!              with per-token shading and a spike log. This is the calibration
//!              instrument — the thing you actually stare at to tune the z-threshold.
//!
//! Both share `Lobe` (src/lobe.rs): teacher-force the stream into the KV cache, read the
//! next-token distribution each step, score -ln P(actual), fire on a Welford z-spike.

mod backend;
mod lobe;
mod present;
mod present_scene;
mod prompt;
mod retrieval;
mod sprite;
mod stats;
mod trace;
mod tui;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::io::{BufRead, Write};
use std::time::Instant;

use backend::{ActiveBackend, Backend};
use lobe::{
    AskMode, EvictMode, InterjectMode, Lobe, LobeConfig, NoveltyMode, Signal, Source, Step, Trigger,
};
use stats::Welford;

/// A non-deterministic u64 seed from OS entropy, for `--random-seed`. Uses `RandomState` (seeded from
/// the OS RNG for HashMap DoS-resistance) mixed with the wall-clock nanos — no `rand` dependency. The
/// `0` xorshift fixed point is handled downstream by `Lobe::set_seed`.
fn entropy_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    if let Ok(d) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        h.write_u128(d.as_nanos());
    }
    h.finish()
}

#[derive(Parser)]
#[command(name = "streaming-lobe", version, about)]
struct Cli {
    /// Path to a GGUF model (e.g. a small Gemma/Qwen at Q4). The observer should be
    /// small and fast; this is the "limbic" lobe, not the cortex. Defaults to the gemma-4-E2B QAT
    /// 4-bit GGUF shipped in `models/` — so a bare `streaming-lobe tui` just works.
    #[arg(long, default_value = "models/gemma-4-E2B_q4_0-it.gguf")]
    model: String,

    /// Context window size for the observer. Default 32k (gemma-4 iSWA keeps the KV cost sublinear;
    /// validated flat-memory/throughput over a full book at this size).
    #[arg(long, default_value_t = 32768)]
    ctx: u32,

    /// GPU layers to offload. Honored on metal/cuda/vulkan builds, ignored on CPU.
    /// 999 = offload everything (right default for Apple Silicon).
    #[arg(long, default_value_t = 999)]
    gpu_layers: u32,

    /// z-score threshold above which a token "fires" (the observer speaks up).
    /// Lower = chattier. This is the main calibration knob.
    #[arg(long, default_value_t = 3.0)]
    z: f32,

    /// Number of warmup tokens before triggers are allowed (lets the baseline settle).
    #[arg(long, default_value_t = 32)]
    warmup: u64,

    /// Refractory period (tokens): after the observer fires, it stays quiet this long so it doesn't
    /// obsess over the same salient thing while it lingers in the context window. 0 disables.
    #[arg(long, default_value_t = 64)]
    refractory: usize,

    /// Surprisal-baseline adaptation window (tokens). 0 = global all-stream baseline (default).
    /// >0 = adaptive EWMA over ~N recent tokens (z is relative to recent texture). Experimental:
    /// it does NOT fix obsessing over a sustained-salient region — the catalog is spiky, not a
    /// plateau, and the repetition is a content issue (see --dedup). Left as an option.
    #[arg(long, default_value_t = 0)]
    adapt: usize,

    /// OPT-IN backstop filter (default OFF). Suppresses an interjection that repeats a recent one
    /// (opening-stem match or char-shingle Jaccard over this threshold). The PRIMARY anti-fixation
    /// mechanism is now the novelty memory in the ask (the model is shown what it just said and asked
    /// for a fresh angle) — this only filters residual repeats it can't escape. >0 to enable.
    #[arg(long, default_value_t = 0.0)]
    dedup: f32,

    /// Top-k expected alternatives to attach to each trigger.
    #[arg(long, default_value_t = 5)]
    topk: usize,

    /// Trigger signal (#4): `surprisal` (the actual token was unexpected — default) or
    /// `entropy` (the model was uncertain at this position, regardless of the token).
    #[arg(long, value_enum, default_value = "surprisal")]
    signal: Signal,

    /// Only fire on identifier/entity-looking tokens — code identifiers, proper nouns,
    /// number-bearing tokens (#4). An objective gate that cuts rare-but-irrelevant triggers.
    #[arg(long, default_value_t = false)]
    identifiers_only: bool,

    /// Override the built-in system prompt that's pinned at the front of the context — the
    /// StreamingLLM attention sink, replayed verbatim on every cap+reset (#6). Empty = use the
    /// built-in default. With --frame it becomes the body of the model's reasoning turn.
    #[arg(long, default_value = "")]
    preamble: String,

    /// Read the system-prompt sink from a file instead of `--preamble` (avoids shell-quoting a
    /// paragraph; lets you keep a persona library, e.g. `personas/herzog.txt`). Trimmed of
    /// surrounding whitespace. Takes precedence over `--preamble` when both are given.
    #[arg(long)]
    preamble_file: Option<String>,

    /// Frame the incoming stream as the frontier model's *thinking* (#5): pin a fixed chat
    /// wrapper so the observer scores the stream in the "assistant reasoning" register instead
    /// of as arbitrary raw text. Prepended to --preamble. Recommended for the real use case.
    #[arg(long, default_value_t = false)]
    frame: bool,

    /// What an interjection reflects on: `context` (default — fork the observer's FULL live context
    /// so it reacts to the whole reasoning) or `snippet` (re-encode only the recent window and react
    /// to the surprising token). `context` is cleanest with --frame.
    #[arg(long, value_enum, default_value = "context")]
    interject_mode: InterjectMode,

    /// EXPERIMENT (templating study): context-mode ask framing. `passage` (control) hands the model a
    /// discrete quoted span to comment on; `continuous` frames it as picking up an ongoing thread (H2).
    #[arg(long, value_enum, default_value = "passage")]
    ask_mode: AskMode,

    /// EXPERIMENT (templating study): novelty-memory framing. `fresh` (control) asks for a fresh
    /// angle (content); `form` asks to vary rhythm/openings; `off` omits the novelty memory (H4).
    #[arg(long, value_enum, default_value = "fresh")]
    novelty: NoveltyMode,

    /// Eviction policy when the context fills (#6): `reset` (cap + reset — re-prime preamble +
    /// recent window; the supported path on Gemma's iSWA cache) or `off` (no eviction; errors at
    /// the cap — use with a large --ctx as the validation control arm).
    #[arg(long, value_enum, default_value = "reset")]
    evict: EvictMode,

    /// Cap + reset: how many recent stream tokens to replay after a reset (the rolling window).
    /// Clamped to fit the context. Larger = more continuity across resets, but costlier resets.
    #[arg(long, default_value_t = 4096)]
    keep_recent: usize,

    /// Emit periodic throughput/eviction stats to stderr during headless streaming (every ~25k
    /// tokens) plus a final summary — for validating bounded throughput on long streams.
    #[arg(long, default_value_t = false)]
    stats: bool,

    /// Print llama.cpp / Metal load logs to stderr. Off by default (logs are voided) so the
    /// JSONL on stdout stays clean and the TUI isn't corrupted by model-load spam.
    #[arg(long, default_value_t = false)]
    verbose: bool,

    /// Deep structured observability: write granular JSONL traces of every observe/trigger/
    /// interject/roll/rag event to this file (via `tracing`). Off unless a path is given. Separate
    /// from stdout JSONL and `--stats`; safe to use in the TUI (file, not terminal). Filter targets
    /// with the `LOBE_LOG` env var (default: everything at TRACE).
    #[arg(long, value_name = "FILE")]
    debug_log: Option<std::path::PathBuf>,

    /// Top-K logits to attach to each inference event in the debug log (the predicting distribution).
    #[arg(long, default_value_t = 64)]
    debug_topk: usize,

    /// Also dump the FULL n_vocab logit vector on fires + interjection generation (huge — ≈262k
    /// floats per event; bounded to fires/gen, not every token). Requires --debug-log.
    #[arg(long, default_value_t = false)]
    debug_full_logits: bool,

    /// Sampling temperature for INTERJECTION generation only (observation scoring always stays
    /// exact/greedy — provably byte-identical surprisals regardless of this). Default 0.7: the
    /// fixation fix. Greedy (0) deterministically collapses to one dominant phrasing per region and
    /// repeats it verbatim even when shown its own prior output; sampling surfaces the varied
    /// observations latent below the argmax and lets the novelty memory actually take effect
    /// (measured: verbatim-repeats 7/34→2/34, distinct openings 19/34→31/34). 0 = greedy.
    #[arg(long, default_value_t = 0.7)]
    interject_temp: f32,

    /// Top-p (nucleus) cutoff for interjection sampling when --interject-temp > 0.
    #[arg(long, default_value_t = 0.95)]
    interject_top_p: f32,

    /// Disable interjections. They are ON by default in BOTH modes (the observe → react loop is the
    /// point); pass this for pure surprisal output (e.g. headless JSONL with no generation latency).
    #[arg(long, default_value_t = false)]
    no_interject: bool,

    /// Deprecated/no-op: interjections are on by default now. Accepted so older commands don't break.
    #[arg(long, default_value_t = false, hide = true)]
    interject: bool,

    /// Max tokens per interjection (the length control — the prompt no longer caps it). Global:
    /// applies to both `headless` and `tui`.
    #[arg(long, default_value_t = 96)]
    interject_max: usize,

    /// Make the observer NON-deterministic. By default every run is byte-identical: the sampler uses
    /// a fixed seed, and the surprisal trigger is teacher-forced/greedy (deterministic by construction
    /// — there's no sampler on the observation path to seed). With this flag the code picks a random
    /// seed from OS entropy each run AND softens the firing decision (probabilistic sigmoid), so BOTH
    /// the asides AND which tokens fire vary run-to-run — like a chat API. No argument; no tuning.
    #[arg(long, default_value_t = false)]
    non_deterministic: bool,

    #[command(subcommand)]
    mode: Mode,
}

impl Cli {
    /// Whether interjections are active (on by default; `--no-interject` disables; the legacy
    /// `--interject` flag is a redundant no-op kept for back-compat).
    fn interject_on(&self) -> bool {
        !self.no_interject
    }
}

#[derive(Subcommand)]
enum Mode {
    /// Read the stream from stdin, emit JSONL events on stdout.
    Headless {
        /// Granularity of a stream "token" as read from stdin.
        #[arg(long, value_enum, default_value_t = Granularity::Line)]
        granularity: Granularity,

        /// Emit a uniform `step` event for every scored token (fired or not), alongside any
        /// `trigger` events — so an offline threshold sweep gets a complete per-token stream.
        #[arg(long, default_value_t = false)]
        all_steps: bool,

        /// On each trigger, probe the native tool-calling RAG hook (#8): define a `search` tool and
        /// let the observer think + (maybe) call it. Emits a `rag_probe` event with the RAW output.
        #[arg(long, default_value_t = false)]
        rag: bool,

        /// Corpus text file to retrieve over (#8): the "external knowledge base" a `search` tool call
        /// is answered from (e.g. the novel being read). When set, retrieval is REAL (BM25 over the
        /// file) and the model's hits feed back into a grounded aside; unset → retrieval finds nothing.
        #[arg(long)]
        rag_corpus: Option<String>,

        /// Use the FUSED concurrent forward pass (CONCURRENT_FORWARD_PASS): observation and
        /// interjection generation co-batch into one decode per stream token, so observation never
        /// stalls (the interjection forms in the background and emits ~N tokens after its trigger).
        /// Default off → the blocking path (interjection attached to its trigger). Mutually exclusive
        /// with --rag.
        #[arg(long, default_value_t = false)]
        fused: bool,
    },
    /// Pace a transcript file through the observer in a live TUI.
    Tui {
        /// Transcript file to stream through the observer. Defaults to the bundled
        /// `sample_thinking.txt` demo (a reasoning stream with planted anomalies).
        #[arg(long, default_value = "sample_thinking.txt")]
        input: String,

        /// Milliseconds between tokens (paces the stream so you can watch it).
        #[arg(long, default_value_t = 40)]
        tick_ms: u64,

        /// Start streaming at the first occurrence of this substring, skipping everything before it.
        /// For the novel-reading demo: skip a book's front-matter (license / table of contents /
        /// etymology) straight to the narrative, e.g. `--skip-to "Call me Ishmael"`. Empty = start
        /// at the beginning. (The structural front-matter is the least-watchable part and the hardest
        /// to gate — see CLAUDE.md; reading the narrative is what "watch it read the novel" means.)
        #[arg(long, default_value = "")]
        skip_to: String,
    },
    /// Presentation view: a clean stage that just shows the lobe reading and musing — the prose
    /// streaming past while asides form alongside it. No debug chrome (vs `tui`, the instrument).
    Present {
        /// Transcript file to stream. The novel-reading showcase: `--input corpus/pg2701.txt`.
        #[arg(long, default_value = "sample_thinking.txt")]
        input: String,

        /// Milliseconds between tokens (paces the stream; a calmer default than `tui` for watching).
        #[arg(long, default_value_t = 30)]
        tick_ms: u64,

        /// Start at the first occurrence of this substring (skip front-matter), e.g.
        /// `--skip-to "Call me Ishmael"`. Empty = start at the beginning.
        #[arg(long, default_value = "")]
        skip_to: String,

        /// Cosmetic SCENE skin: render the observer as a bearded gent in a Victorian study (a
        /// half-block pixel-art scene) with each aside as a speech bubble, and the prose reduced to a
        /// quiet ticker. Same observe→react loop as plain `present`; just a different stage. Off by
        /// default (plain `present` is the default view).
        #[arg(long, default_value_t = false)]
        scene: bool,
    },
}

#[derive(Copy, Clone, ValueEnum)]
enum Granularity {
    /// Each newline-delimited line of stdin is tokenized and fed as a chunk.
    Line,
    /// Each whitespace-delimited word is fed as a chunk.
    Word,
}

/// Built-in default system prompt (#6 sink + #5 framing). This is the stable, pinned prefix that
/// serves three jobs at once: the StreamingLLM attention *sink* (it sits at position 0 and is
/// replayed verbatim on every cap+reset, so the sink never changes), the observer's framing, and
/// — with --frame — the body of the Gemma user turn that opens the model's reasoning. It's a few
/// dozen tokens, far more than the ~4 a sink actually needs. Overridden entirely by --preamble.
// Register = the COMMENTATOR (the Princess-Bride-grandpa aside): asides ABOUT the content — what a
// passage is doing, what it means, how it connects — not introspection about how the observer feels.
// The earlier "stream of consciousness… what struck you" framing produced emotive "it feels like / it
// struck me" output; this directs attention to the text itself. NB: embodied framing ("reading
// alongside you / leaning in") makes it emit stage directions ("(I nod slowly)") — keep it
// disembodied. (Modality/voice belongs HERE, the sink — never the per-fire ask. Override w/ --preamble
// for a different register; this default is tuned for the novel-reading demo.)
const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a perceptive literary commentator following a text as it streams past. Now and then you \
     offer a brief aside about what's happening on the page — what a passage is doing, what a word or \
     image is up to, what it means, how it ties to something earlier. Keep your eye on the text \
     itself: its moves, its meaning, its craft.";

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Structured observability (--debug-log): install the JSONL subscriber FIRST so every subsequent
    // event is captured. `_trace_guard` must live for the whole run (drop flushes the writer).
    let _trace_guard = match &cli.debug_log {
        Some(path) => Some(trace::init(path)?),
        None => None,
    };

    let engine = ActiveBackend::load(&cli.model, cli.gpu_layers, cli.verbose)?;
    // All construction-time config in one value (no post-`new` setters, no before-`prime` ordering
    // hazard). Determinism: byte-identical by default (seed None = fixed reproducible RNG + hard
    // firing threshold). --non-deterministic seeds both RNG streams from OS entropy AND softens the
    // firing decision (sigmoid, z-units), so both the asides AND which tokens fire vary run-to-run;
    // 0.5 is a ~±1z transition band around z_threshold. Only interjection CONTENT and (softened)
    // firing are affected — the surprisal value itself is always the exact greedy read.
    let config = LobeConfig {
        signal: cli.signal,
        identifiers_only: cli.identifiers_only,
        evict: cli.evict,
        keep_recent: cli.keep_recent,
        interject_mode: cli.interject_mode,
        ask_mode: cli.ask_mode, // EXPERIMENT toggles (default = control)
        novelty_mode: cli.novelty,
        interject_temp: cli.interject_temp,
        interject_top_p: cli.interject_top_p,
        interject_max: cli.interject_max,
        refractory: cli.refractory,
        dedup: cli.dedup,
        debug: trace::DebugCfg {
            topk: cli.debug_topk,
            full_logits: cli.debug_full_logits,
        },
        seed: cli.non_deterministic.then(entropy_seed),
        fire_softness: if cli.non_deterministic { 0.5 } else { 0.0 },
    };
    let mut lobe = Lobe::new(&engine, cli.ctx, config)?;

    // One config dump so every trace file is self-describing (what produced this run).
    tracing::info!(
        target: "lobe::run", kind = "run_start",
        model = %cli.model, ctx = cli.ctx as u64, z = cli.z as f64, warmup = cli.warmup,
        signal = ?cli.signal, identifiers_only = cli.identifiers_only, frame = cli.frame,
        interject_mode = ?cli.interject_mode, refractory = cli.refractory as u64,
        dedup = cli.dedup as f64, adapt = cli.adapt as u64, evict = ?cli.evict,
        keep_recent = cli.keep_recent as u64,
        interject_temp = cli.interject_temp as f64, interject_top_p = cli.interject_top_p as f64,
        non_deterministic = cli.non_deterministic,
        "run_start"
    );

    // The pinned system-prompt "sink" (StreamingLLM): a stable prefix at position 0 — the
    // attention sink, the observer framing, AND exactly what roll() replays verbatim on every
    // reset. (c): the built-in default is used unless --preamble overrides it.
    // (file takes precedence over --preamble; trimmed so a trailing newline doesn't enter the sink).
    let system_prompt: String = if let Some(path) = &cli.preamble_file {
        std::fs::read_to_string(path)
            .with_context(|| format!("failed to read --preamble-file {path}"))?
            .trim()
            .to_string()
    } else if cli.preamble.is_empty() {
        DEFAULT_SYSTEM_PROMPT.to_string()
    } else {
        cli.preamble.clone()
    };
    // 1d (latent-bug hygiene): context interject-mode's ask opens with `<turn|>` (turn-close), which
    // is only well-formed if the preamble opened a `<|turn>model` turn — i.e. with --frame. Since
    // --frame defaults off but --interject-mode defaults to `context`, a default invocation would
    // build malformed prompts. Derive an effective frame flag so context mode always frames.
    let frame = cli.frame || cli.interject_mode == InterjectMode::Context;
    if cli.interject_mode == InterjectMode::Context && !cli.frame {
        eprintln!("[lobe] context interject-mode requires framing; enabling --frame implicitly");
    }
    // #5: with frame the persona goes in gemma-4's dedicated SYSTEM turn (see prompt::system_preamble
    // for the chat-format rationale); add_bos=true — BOS is the canonical first attention sink.
    let preamble_text = prompt::system_preamble(&system_prompt, frame);
    let preamble_tokens = lobe.tokenize(&preamble_text, true)?;
    lobe.prime(&preamble_tokens)?;

    // Borrow `cli.mode` rather than moving out of it — the handlers also take `&cli`, so a
    // partial move of `cli.mode` (the non-Copy `input: String`) would invalidate that borrow.
    match &cli.mode {
        Mode::Headless {
            granularity,
            all_steps,
            rag,
            rag_corpus,
            fused,
        } => {
            // Build the retrieval index once (BM25 over the corpus file) if --rag-corpus was given.
            let corpus = match rag_corpus {
                Some(path) => Some(retrieval::index(
                    &std::fs::read_to_string(path)
                        .with_context(|| format!("failed to read --rag-corpus {path}"))?,
                    80, // ~80-word chunks: a readable passage, small enough to localize a hit
                )),
                None => None,
            };
            run_headless(
                &mut lobe,
                &cli,
                *granularity,
                *all_steps,
                *rag,
                *fused,
                corpus.as_ref(),
            )
        }
        Mode::Tui {
            input,
            tick_ms,
            skip_to,
        } => tui::run(&mut lobe, &cli, input, *tick_ms, skip_to),
        Mode::Present {
            input,
            tick_ms,
            skip_to,
            scene,
        } => {
            if *scene {
                present_scene::run(&mut lobe, &cli, input, *tick_ms, skip_to)
            } else {
                present::run(&mut lobe, &cli, input, *tick_ms, skip_to)
            }
        }
    }
}

/// Headless: stdin -> scored tokens -> JSONL on stdout.
///
/// Write a `trigger` JSONL event. `with_expected` attaches the model's top-k expectations (the
/// non-fused path includes them; the fused path keeps the line lean).
fn emit_trigger(out: &mut impl Write, t: &Trigger, signal: &str, pos: i32, with_expected: bool) -> Result<()> {
    let mut ev = serde_json::json!({
        "event": "trigger", "stream_index": t.stream_index, "token": t.token_text,
        "surprisal": t.surprisal, "entropy": t.entropy, "z": t.z, "signal": signal, "pos": pos,
    });
    if with_expected {
        let expected: Vec<_> = t
            .expected
            .iter()
            .map(|(s, p)| serde_json::json!({ "tok": s, "p": p }))
            .collect();
        ev["expected"] = serde_json::json!(expected);
    }
    writeln!(out, "{ev}")?;
    Ok(())
}

/// Write an `interjection` JSONL event. `trigger_token` is attached on the non-fused path (the fused
/// one tags by stream_index only) and `fused` flags which path produced it.
fn emit_interjection(
    out: &mut impl Write,
    stream_index: usize,
    trigger_token: Option<&str>,
    text: &str,
    fused: bool,
) -> Result<()> {
    let mut ev = serde_json::json!({ "event": "interjection", "stream_index": stream_index, "text": text });
    if let Some(tok) = trigger_token {
        ev["trigger_token"] = serde_json::json!(tok);
    }
    if fused {
        ev["fused"] = serde_json::json!(true);
    }
    writeln!(out, "{ev}")?;
    Ok(())
}

/// Write a uniform per-token `step` JSONL event (`--all-steps`).
fn emit_step(out: &mut impl Write, step: &Step, signal: &str, pos: i32) -> Result<()> {
    let ev = serde_json::json!({
        "event": "step", "stream_index": step.stream_index, "token": step.token_text,
        "surprisal": step.surprisal, "entropy": step.entropy, "z": step.z, "signal": signal, "pos": pos,
    });
    writeln!(out, "{ev}")?;
    Ok(())
}

/// One stream token on the FUSED path: co-batch observation + any in-flight interjection in a single
/// `step()`, emitting the trigger and (when the concurrent aside completes) the interjection.
fn process_fused_token(
    lobe: &mut Lobe,
    tok: backend::Token,
    stats: &mut Welford,
    cli: &Cli,
    signal: &str,
    out: &mut impl Write,
) -> Result<()> {
    let outcome = lobe.step(tok, stats, cli.z, cli.topk, cli.interject_max)?;
    let pos = lobe.position();
    if let Some(t) = &outcome.step.trigger {
        emit_trigger(out, t, signal, pos, false)?;
    }
    match outcome.interjection {
        lobe::InterjectStatus::Working(partial) => {
            // early-abort a known duplicate (exercises the abort path)
            if lobe.interjection_doomed(&partial) {
                lobe.abort_interjection()?;
            }
        }
        lobe::InterjectStatus::Done(text) => {
            let text = text.trim();
            if !text.is_empty() {
                emit_interjection(out, outcome.step.stream_index, None, text, true)?;
                lobe.record_interjection(text);
            }
        }
        _ => {}
    }
    Ok(())
}

/// One stream token on the NON-fused path: observe, emit the trigger, then (blocking) the optional
/// interjection and #8 RAG, plus the `--all-steps` event. Returns whether the token fired.
#[allow(clippy::too_many_arguments)]
fn process_observe_token(
    lobe: &mut Lobe,
    tok: backend::Token,
    stats: &mut Welford,
    cli: &Cli,
    signal: &str,
    interject: bool,
    rag: bool,
    all_steps: bool,
    corpus: Option<&retrieval::Corpus>,
    out: &mut impl Write,
) -> Result<bool> {
    let step = lobe.observe(tok, stats, cli.z, cli.topk)?;
    let pos = lobe.position(); // post-decode KV position; sawtooths under cap+reset

    if let Some(t) = &step.trigger {
        emit_trigger(out, t, signal, pos, true)?;
        // Clone the trigger token out before mutably borrowing lobe for generation (t borrows step).
        let stream_index = t.stream_index;
        let surprising = t.token_text.clone();
        if interject {
            handle_interjection(lobe, stream_index, &surprising, cli.interject_max, out)?;
        }
        if rag {
            handle_rag(lobe, stream_index, &surprising, corpus, out)?;
        }
    }

    // --all-steps: a uniform per-token `step` for EVERY observed token, alongside any trigger above.
    if all_steps {
        emit_step(out, &step, signal, pos)?;
    }
    Ok(step.fired)
}

/// Record a (trimmed) aside and emit it. The aside is ALWAYS recorded — anti-fixation lives in the
/// ask's novelty memory; `--dedup` (via `interjection_is_novel`) gates only whether it's emitted,
/// never the recording. Empty → no-op. Shared by the plain interjection and the RAG thought.
fn record_and_emit_interjection(
    lobe: &mut Lobe,
    stream_index: usize,
    trigger_token: &str,
    text: &str,
    out: &mut impl Write,
) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    let emit = lobe.interjection_is_novel(text);
    lobe.record_interjection(text);
    if emit {
        emit_interjection(out, stream_index, Some(trigger_token), text, false)?;
    }
    Ok(())
}

/// Generate + emit the blocking interjection for a fire (non-fused path).
fn handle_interjection(
    lobe: &mut Lobe,
    stream_index: usize,
    surprising: &str,
    max: usize,
    out: &mut impl Write,
) -> Result<()> {
    let note = lobe.interject(surprising, max)?;
    record_and_emit_interjection(lobe, stream_index, surprising, note.trim(), out)
}

/// #8 native tool-calling RAG pass for a fire: the free thought is emitted as an `interjection`, and
/// a parsed tool call drives a `retrieval` event answered by BM25 over the `--rag-corpus` (or nothing
/// when no corpus was given). Abstain → no retrieval, just the thought.
fn handle_rag(
    lobe: &mut Lobe,
    stream_index: usize,
    surprising: &str,
    corpus: Option<&retrieval::Corpus>,
    out: &mut impl Write,
) -> Result<()> {
    // Retrieval is injected as a function argument (the `--rag-corpus` BM25 index, or nothing).
    let rag_out = lobe.rag(surprising, 160, |_source, query| {
        corpus.and_then(|c| retrieval::search(c, query))
    })?;
    record_and_emit_interjection(lobe, stream_index, surprising, rag_out.thought.trim(), out)?;
    if let Some(d) = &rag_out.directive {
        let src = match d.source {
            Source::Mem => "mem",
            Source::Rag => "rag",
        };
        let rev = serde_json::json!({
            "event": "retrieval", "stream_index": stream_index, "source": src,
            "query": d.query, "found": rag_out.retrieved.is_some(), "snippet": rag_out.retrieved,
        });
        writeln!(out, "{rev}")?;
        // The grounded reply (after the snippet was fed back) is the actual promotion — emit it too.
        if let Some(resp) = rag_out.response.as_deref() {
            if !resp.is_empty() {
                record_and_emit_interjection(lobe, stream_index, surprising, resp, out)?;
            }
        }
    }
    Ok(())
}

/// Each input chunk (line or word) is tokenized and each of its tokens is observed and
/// scored individually, so a single noisy line can produce several events. Designed to
/// be piped: `cat thinking_stream.txt | streaming-lobe --model m.gguf headless`.
#[allow(clippy::too_many_arguments)]
fn run_headless(
    lobe: &mut Lobe,
    cli: &Cli,
    granularity: Granularity,
    all_steps: bool,
    rag: bool,
    fused: bool,
    corpus: Option<&retrieval::Corpus>,
) -> Result<()> {
    let interject = cli.interject_on(); // global flag (on by default; --no-interject disables)
    let mut stats = Welford::new(cli.warmup, cli.adapt);
    let signal_name = match cli.signal {
        Signal::Surprisal => "surprisal",
        Signal::Entropy => "entropy",
    };
    // #6 validation: throughput / eviction stats (stderr only — never pollutes stdout JSONL).
    let t_start = Instant::now();
    let mut t_window = t_start;
    let mut tok_count: u64 = 0;
    let mut window_count: u64 = 0;
    const STATS_EVERY: u64 = 25_000;
    const FUSED_STATS_EVERY: u64 = 200; // tighter KV-occupancy diagnostic for the fused path (§3)
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line?;
        let chunks: Vec<String> = match granularity {
            Granularity::Line => vec![format!("{line}\n")],
            Granularity::Word => line
                .split_whitespace()
                .map(|w| format!("{w} "))
                .collect(),
        };

        for chunk in chunks {
            // add_bos=false: stream tokens are a continuation of the pinned preamble.
            let tokens = lobe.tokenize(&chunk, false)?;
            for tok in tokens {
                // Two per-token paths (see the helpers): FUSED co-batches observation + an in-flight
                // interjection in one step(); the default path observes then blocks on the optional
                // interjection / #8 RAG. Both share the counters + periodic stats below.
                if fused {
                    process_fused_token(lobe, tok, &mut stats, cli, signal_name, &mut out)?;
                } else {
                    let fired = process_observe_token(
                        lobe, tok, &mut stats, cli, signal_name, interject, rag, all_steps, corpus,
                        &mut out,
                    )?;
                    // Flush promptly on triggers so live consumers see them; plain steps stay buffered
                    // for throughput when piping --all-steps to a file.
                    if fired {
                        out.flush()?;
                    }
                }

                tok_count += 1;
                window_count += 1;
                // Periodic throughput / eviction stats (stderr only — never pollutes stdout JSONL).
                // The fused path dumps KV occupancy at a tighter cadence (FUSED_CACHE_GO_NOGO §3).
                let stats_every = if fused { FUSED_STATS_EVERY } else { STATS_EVERY };
                if cli.stats && window_count >= stats_every {
                    let dt = t_window.elapsed().as_secs_f64().max(1e-9);
                    if fused {
                        let (s0, gen, inflight) = lobe.kv_debug();
                        eprintln!(
                            "[kv] tok={tok_count} resets={} pos={} seq0_max={s0} gen_max={gen} \
                             gen_inflight={inflight} tok/s={:.0}",
                            lobe.resets(), lobe.position(), window_count as f64 / dt,
                        );
                    } else {
                        eprintln!(
                            "[stats] tok={tok_count} resets={} pos={} window_tok/s={:.0}",
                            lobe.resets(), lobe.position(), window_count as f64 / dt,
                        );
                    }
                    t_window = Instant::now();
                    window_count = 0;
                }
            }
        }
    }

    if cli.stats {
        let dt = t_start.elapsed().as_secs_f64().max(1e-9);
        eprintln!(
            "[stats] DONE tok={tok_count} resets={} elapsed={:.1}s avg_tok/s={:.0}",
            lobe.resets(),
            dt,
            tok_count as f64 / dt,
        );
    }
    Ok(())
}
