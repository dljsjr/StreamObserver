//! stream-observer — a tiny local model that *observes* a token stream and flags the
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
mod present_worker;
mod prompt;
mod retrieval;
mod sprite;
mod stats;
mod trace;
mod tui;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
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

#[derive(Parser, Clone)]
#[command(name = "stream-observer", version, about)]
struct Cli {
    /// Path to a GGUF model (e.g. a small Gemma/Qwen at Q4). The observer should be
    /// small and fast; this is the "limbic" lobe, not the cortex. Defaults to the gemma-4-E4B QAT
    /// 4-bit GGUF shipped in `models/` — the showcase model (the persona needs E4B; E2B collapses on
    /// it) — so a bare `stream-observer --input … --skip-to …` runs the full demo. (E2B is also in
    /// `models/` for the leaner/headless use: `--model models/gemma-4-E2B_q4_0-it.gguf`.)
    #[arg(long, default_value = "models/gemma-4-E4B_q4_0-it.gguf")]
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
    /// Lower = chattier. This is the main calibration knob. Default 4.1 = the showcase cadence (sparse
    /// close-reading; E4B fires hotter than E2B, so a higher z than E2B's old 3.0).
    #[arg(long, default_value_t = 4.1)]
    z: f32,

    /// Number of warmup tokens before triggers are allowed (lets the baseline settle).
    #[arg(long, default_value_t = 32)]
    warmup: u64,

    /// Refractory period (tokens): after the observer fires, it stays quiet this long so it doesn't
    /// obsess over the same salient thing while it lingers in the context window. 0 disables. Default
    /// 320 = the showcase cadence (sparse close-reading).
    #[arg(long, default_value_t = 320)]
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
    /// surrounding whitespace. Takes precedence over `--preamble`. Defaults to the showcase persona
    /// `personas/herzog_varyform.txt`; point it at another persona file for a different voice, or pass
    /// `--preamble-file ""` (empty) to drop the persona and fall back to `--preamble` / the built-in
    /// default prompt.
    #[arg(long, default_value = "personas/herzog_varyform.txt")]
    preamble_file: Option<String>,

    /// Frame the incoming stream as the frontier model's *thinking* (#5): pin a fixed chat
    /// wrapper so the observer scores the stream in the "assistant reasoning" register instead
    /// of as arbitrary raw text. ON by default (the showcase config; also implied by context
    /// interject-mode); `--no-frame` disables it.
    #[arg(long = "no-frame", default_value_t = false)]
    no_frame: bool,

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
    /// (measured: verbatim-repeats 7/34→2/34, distinct openings 19/34→31/34). 0 = greedy. Default 0.6
    /// = the showcase voice (the templating study's converged setting: low temp owns DEPTH).
    #[arg(long, default_value_t = 0.6)]
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
    /// applies to every mode. Default 80 = the showcase length.
    #[arg(long, default_value_t = 80)]
    interject_max: usize,

    /// Ask tokens prefilled per stream tick when a (fused) interjection starts. The interjection's ask
    /// is often 200–350 tokens; prefilling it in one decode froze the prose ~250–500ms at every fire.
    /// It's now spread this many tokens per tick (co-batched with the stream), so the prose keeps
    /// scrolling. SMALLER = smoother scroll during prefill but the aside starts a few more ticks later;
    /// LARGER = the aside starts sooner but each tick's decode is heavier (a bigger scroll slowdown).
    /// Live frontends only (`present`/`tui`); ≥1.
    #[arg(long, default_value_t = 8)]
    prefill_chunk: usize,

    /// Make the observer DETERMINISTIC (byte-identical runs). By DEFAULT the showcase is
    /// NON-deterministic: a random OS-entropy seed each run + a softened (probabilistic-sigmoid) firing
    /// decision, so BOTH the asides AND which tokens fire vary run-to-run — like a chat API. Pass this
    /// to pin it: a fixed sampler seed + a hard firing threshold (the surprisal trigger is greedy/
    /// teacher-forced, so it's deterministic by construction anyway). No argument; no tuning.
    #[arg(long, default_value_t = false)]
    deterministic: bool,

    /// Disable #8 RAG. RAG is ON by default (the showcase config): on each fire the observer retrieves
    /// over the corpus and weaves the hit into its aside in voice (live modes), or emits the native
    /// tool-call JSONL (headless). Pass `--no-rag` for the lean free-association / pure-surprisal path
    /// (and to skip the embed-model load + corpus index at startup).
    #[arg(long = "no-rag", default_value_t = false)]
    no_rag: bool,

    /// Corpus text file the `search` tool retrieves over (#8) — the "external knowledge base", e.g. the
    /// novel being read. DEFAULTS TO `--input` when omitted (so the gent retrieves over the very text
    /// he's reading — the showcase shape). Hits feed back into a grounded/woven aside.
    #[arg(long)]
    rag_corpus: Option<String>,

    /// Embedding-model GGUF. Retrieval becomes HYBRID: BM25 and semantic-cosine rankings fused by
    /// Reciprocal Rank Fusion (k=60). Requires the llama backend; embeds the corpus once at startup.
    /// Defaults to the showcase embedder `models/harrier-oss-v1-270m-BF16.gguf`; set to "" to fall back
    /// to lexical-only (BM25) retrieval.
    #[arg(long, default_value = "models/harrier-oss-v1-270m-BF16.gguf")]
    rag_embed_model: Option<String>,

    /// Which frontend to run. `demo-tui` (default) = the SHOWCASE: clean stage, prose + the lobe's
    /// asides (add `--scene` for the pixel-art study skin). `debug-tui` = the calibration INSTRUMENT
    /// (sparkline, z-heatmap, trigger list, live knobs). `headless` = stdin → JSONL on stdout (the
    /// harness-pipe shape).
    #[arg(long, value_enum, default_value_t = Mode::DemoTui)]
    mode: Mode,

    // ── Mode-specific options (flattened from the old subcommands; each applies in the modes noted) ──
    /// [demo-tui/debug-tui] Transcript file to stream through the observer. The novel-reading showcase:
    /// `--input corpus/pg2701.txt`. (headless reads the stream from stdin instead.)
    #[arg(long, default_value = "sample_thinking.txt")]
    input: String,

    /// [demo-tui/debug-tui] Milliseconds between tokens (paces the stream so you can watch it).
    #[arg(long, default_value_t = 30)]
    tick_ms: u64,

    /// [demo-tui/debug-tui] Start streaming at the first occurrence of this substring, skipping
    /// everything before it — e.g. `--skip-to "Call me Ishmael"` to skip a book's front-matter (license
    /// / table of contents / etymology) to the narrative. Empty = start at the beginning.
    #[arg(long, default_value = "")]
    skip_to: String,

    /// [demo-tui] The pixel-art SCENE skin: the observer as a bearded gent in a Victorian study (a
    /// half-block pixel-art scene) with each aside as a speech bubble, and the prose reduced to a quiet
    /// ticker. ON by default (the showcase); `--no-scene` falls back to the plain clean stage. Ignored
    /// by the other modes.
    #[arg(long = "no-scene", default_value_t = false)]
    no_scene: bool,

    /// [headless] Granularity of a stream "token" as read from stdin.
    #[arg(long, value_enum, default_value_t = Granularity::Line)]
    granularity: Granularity,

    /// [headless] Emit a uniform `step` event for every scored token (fired or not), alongside any
    /// `trigger` events — so an offline threshold sweep gets a complete per-token stream.
    #[arg(long, default_value_t = false)]
    all_steps: bool,

    /// [headless] Use the FUSED concurrent forward pass (CONCURRENT_FORWARD_PASS): observation and
    /// interjection generation co-batch into one decode per stream token, so observation never stalls
    /// (the interjection forms in the background, emitting ~N tokens after its trigger). Default off →
    /// the blocking path (interjection attached to its trigger). Mutually exclusive with --rag.
    #[arg(long, default_value_t = false)]
    fused: bool,
}

impl Cli {
    /// Whether interjections are active (on by default; `--no-interject` disables; the legacy
    /// `--interject` flag is a redundant no-op kept for back-compat).
    fn interject_on(&self) -> bool {
        !self.no_interject
    }

    /// Stream framing (#5), ON by default; `--no-frame` disables. (Context interject-mode also forces
    /// it on — see `with_lobe`.)
    fn frame(&self) -> bool {
        !self.no_frame
    }

    /// #8 RAG retrieval, ON by default; `--no-rag` disables.
    fn rag(&self) -> bool {
        !self.no_rag
    }

    /// The pixel-art scene skin (demo-tui), ON by default; `--no-scene` falls back to the plain stage.
    fn scene(&self) -> bool {
        !self.no_scene
    }

    /// The corpus the `search` tool retrieves over — `--rag-corpus`, defaulting to `--input` (the text
    /// being read) when omitted, so by default the gent retrieves over his own reading.
    fn rag_corpus_path(&self) -> &str {
        self.rag_corpus.as_deref().unwrap_or(&self.input)
    }
}

/// Which frontend to run (selected by `--mode`). Was a clap subcommand; flattened to a value-enum flag
/// so every option lives at one level (mode-specific options are now top-level `--` flags). Variant
/// names map to kebab-case on the CLI (`DebugTui` → `debug-tui`).
#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum Mode {
    /// stdin → JSONL on stdout (the harness-pipe shape).
    Headless,
    /// The calibration instrument: sparkline, z-heatmap, trigger list, live knobs.
    DebugTui,
    /// The showcase: clean stage, prose + the lobe's asides (+ `--scene` for the pixel-art study).
    DemoTui,
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
    // `tracing` subscribers are global across threads, so the presentation WORKER thread's events are
    // captured too; `present_worker::run` joins the worker before returning, keeping this guard alive.
    let _trace_guard = match &cli.debug_log {
        Some(path) => Some(trace::init(path)?),
        None => None,
    };

    // `Mode` is Copy, so matching it doesn't borrow `cli` — the mode-specific options are now plain
    // top-level fields read directly off `cli`.
    match cli.mode {
        Mode::Headless => with_lobe(&cli, |lobe, retrieve| {
            run_headless(lobe, &cli, cli.granularity, cli.all_steps, cli.fused, retrieve)
        }),
        Mode::DebugTui => with_lobe(&cli, |lobe, retrieve| {
            tui::run(lobe, &cli, &cli.input, cli.tick_ms, &cli.skip_to, retrieve)
        }),
        // The showcase runs ALL llama work on a worker thread (so the scene animation never stalls on
        // retrieval / ask-prefill); the worker builds its own engine + lobe via `with_lobe`.
        Mode::DemoTui => present_worker::run(cli.clone()),
    }
}

/// Build the engine, lobe, and retriever — all share self-referential `&engine` borrows, so they
/// must live in one stack frame — prime the sink, then hand `&mut Lobe` + the retrieval fn to `body`.
/// Every run mode funnels its construction through here; the presentation worker calls it on its own
/// thread so the `!Send` llama handles are created and destroyed entirely on that thread.
pub(crate) fn with_lobe<R>(
    cli: &Cli,
    body: impl FnOnce(&mut Lobe, &mut crate::retrieval::RetrieveFn) -> Result<R>,
) -> Result<R> {
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
        prefill_chunk: cli.prefill_chunk,
        refractory: cli.refractory,
        dedup: cli.dedup,
        debug: trace::DebugCfg {
            topk: cli.debug_topk,
            full_logits: cli.debug_full_logits,
        },
        seed: (!cli.deterministic).then(entropy_seed),
        fire_softness: if cli.deterministic { 0.0 } else { 0.5 },
    };
    let mut lobe = Lobe::new(&engine, cli.ctx, config)?;

    // One config dump so every trace file is self-describing (what produced this run).
    tracing::info!(
        target: "lobe::run", kind = "run_start",
        model = %cli.model, ctx = cli.ctx as u64, z = cli.z as f64, warmup = cli.warmup,
        signal = ?cli.signal, identifiers_only = cli.identifiers_only, frame = cli.frame(),
        interject_mode = ?cli.interject_mode, refractory = cli.refractory as u64,
        dedup = cli.dedup as f64, adapt = cli.adapt as u64, evict = ?cli.evict,
        keep_recent = cli.keep_recent as u64,
        interject_temp = cli.interject_temp as f64, interject_top_p = cli.interject_top_p as f64,
        non_deterministic = !cli.deterministic,
        "run_start"
    );

    // The pinned system-prompt "sink" (StreamingLLM): a stable prefix at position 0 — the
    // attention sink, the observer framing, AND exactly what roll() replays verbatim on every
    // reset. (c): the built-in default is used unless --preamble overrides it.
    // (file takes precedence over --preamble; trimmed so a trailing newline doesn't enter the sink).
    // Precedence: --preamble-file (a non-empty path) > --preamble > built-in default. The persona file
    // defaults to the showcase persona, so an EMPTY `--preamble-file ""` is the escape hatch that drops
    // back to --preamble / the built-in (otherwise the default file would always shadow --preamble).
    let system_prompt: String = match cli.preamble_file.as_deref() {
        Some(path) if !path.is_empty() => std::fs::read_to_string(path)
            .with_context(|| format!("failed to read --preamble-file {path}"))?
            .trim()
            .to_string(),
        _ if cli.preamble.is_empty() => DEFAULT_SYSTEM_PROMPT.to_string(),
        _ => cli.preamble.clone(),
    };
    // 1d (latent-bug hygiene): context interject-mode's ask opens with `<turn|>` (turn-close), which
    // is only well-formed if the preamble opened a `<|turn>model` turn — i.e. with framing. Since
    // --interject-mode defaults to `context`, derive an effective frame flag so context mode always
    // frames even if `--no-frame` was passed.
    let frame = cli.frame() || cli.interject_mode == InterjectMode::Context;
    if cli.interject_mode == InterjectMode::Context && !cli.frame() {
        eprintln!("[lobe] context interject-mode requires framing; enabling it implicitly");
    }
    // #5: with frame the persona goes in gemma-4's dedicated SYSTEM turn (see prompt::system_preamble
    // for the chat-format rationale); add_bos=true — BOS is the canonical first attention sink.
    let preamble_text = prompt::system_preamble(&system_prompt, frame);
    let preamble_tokens = lobe.tokenize(&preamble_text, true)?;
    lobe.prime(&preamble_tokens)?;

    // Build the retrieval function ONCE (it's the mode-agnostic `--rag` seam): HYBRID RRF (embed model
    // + corpus) > LEXICAL BM25 (corpus only) > none; a no-op without `--rag` (so `step()`/`handle_rag`
    // see `None` and behave as plain interjection).
    let mut retrieve = build_retriever(&engine, cli)?;

    body(&mut lobe, retrieve.as_mut())
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
    retrieve: &mut crate::retrieval::RetrieveFn<'_>,
    out: &mut impl Write,
) -> Result<()> {
    let outcome = lobe.step(tok, stats, cli.z, cli.topk, cli.interject_max, retrieve)?;
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
    retrieve: &mut crate::retrieval::RetrieveFn<'_>,
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
            handle_rag(lobe, stream_index, &surprising, retrieve, out)?;
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
    let note = lobe.interject(surprising, max, None)?;
    record_and_emit_interjection(lobe, stream_index, surprising, note.trim(), out)
}

/// The harrier query instruction (#8 semantic retrieval). harrier needs a one-sentence task
/// instruction prepended to QUERIES (not documents); format is its native `Instruct: …\nQuery: …`.
const RAG_INSTRUCT: &str = "Instruct: Given a search query, retrieve relevant passages.\nQuery: ";

/// Reciprocal Rank Fusion constant (the canonical default).
const RRF_K: f32 = 60.0;

/// Build the `--rag` retrieval function (the seam `step`/`handle_rag` call per query): HYBRID RRF
/// (embed model + corpus: fuse BM25 and semantic rankings) > LEXICAL BM25 (corpus only) > none. The
/// closure OWNS its indexes (and, in the hybrid case, the `EmbedderModel` — owned, not leaked, so it
/// drops before the device at exit). Without `--rag` it's a no-op. The `'a` lifetime ties the closure
/// to `engine` (the embedder borrows its runtime).
fn build_retriever<'a>(
    engine: &'a ActiveBackend,
    cli: &Cli,
) -> Result<Box<crate::retrieval::RetrieveFn<'a>>> {
    const CHUNK_WORDS: usize = 80;
    if !cli.rag() {
        return Ok(Box::new(|_q: &str| None));
    }
    // The corpus defaults to --input (the text being read) when --rag-corpus is omitted.
    let corpus_path = cli.rag_corpus_path();
    let text = std::fs::read_to_string(corpus_path)
        .with_context(|| format!("failed to read rag corpus {corpus_path}"))?;
    // The embed model defaults to the showcase embedder; an empty `--rag-embed-model ""` opts down to
    // lexical-only (BM25) retrieval.
    let embed_path = cli.rag_embed_model.as_deref().filter(|p| !p.is_empty());
    match embed_path {
        Some(embed_path) => {
            #[cfg(feature = "llama")]
            {
                // One chunking shared by both indexes (RRF fuses by chunk index, so they must align).
                let chunks = retrieval::chunk(&text, CHUNK_WORDS);
                let embedder = engine.load_embedder(embed_path, 999, 2048)?; // owned (no leak)
                eprintln!("[lobe] embedding {} corpus chunks for hybrid retrieval…", chunks.len());
                let embeddings = embedder.embed_all(&chunks)?; // one shared context for the corpus
                let corpus = retrieval::index_chunks(chunks);
                let semantic = retrieval::SemanticIndex::new(embeddings);
                Ok(Box::new(move |q: &str| {
                    // Fuse BM25 (lexical) + semantic rankings with RRF — keeps both signals (exact-term
                    // hits AND paraphrase/concept matches) without normalizing their disparate scores.
                    // embed_one spins up a transient context per query (sparse → cheap).
                    let bm = retrieval::rank_bm25(&corpus, q);
                    let qe = embedder.embed_one(&format!("{RAG_INSTRUCT}{q}")).ok()?;
                    let sem = retrieval::rank_semantic(&semantic, &qe);
                    retrieval::rrf_best(&[&bm, &sem], RRF_K).map(|i| corpus.chunk_text(i).to_string())
                }))
            }
            #[cfg(not(feature = "llama"))]
            {
                let _ = (text, embed_path, engine);
                anyhow::bail!("--rag-embed-model requires the llama backend")
            }
        }
        None => {
            let corpus = retrieval::index(&text, CHUNK_WORDS);
            Ok(Box::new(move |q: &str| retrieval::search(&corpus, q)))
        }
    }
}

/// #8 native tool-calling RAG pass for a fire: the free thought is emitted as an `interjection`, the
/// parsed tool call is answered by `retrieve` (semantic or BM25 over `--rag-corpus`, or nothing), and
/// the grounded reply (after the snippet is fed back) is emitted as a second interjection.
fn handle_rag(
    lobe: &mut Lobe,
    stream_index: usize,
    surprising: &str,
    retrieve: &mut crate::retrieval::RetrieveFn<'_>,
    out: &mut impl Write,
) -> Result<()> {
    // Retrieval is injected as a function argument (the pre-built `--rag` retriever).
    let rag_out = lobe.rag(surprising, 160, |_source, query| retrieve(query))?;
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
/// be piped: `cat thinking_stream.txt | stream-observer --model m.gguf --mode headless`.
#[allow(clippy::too_many_arguments)]
fn run_headless(
    lobe: &mut Lobe,
    cli: &Cli,
    granularity: Granularity,
    all_steps: bool,
    fused: bool,
    retrieve: &mut crate::retrieval::RetrieveFn<'_>,
) -> Result<()> {
    let interject = cli.interject_on(); // global flag (on by default; --no-interject disables)
    let rag = cli.rag();
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
                    process_fused_token(lobe, tok, &mut stats, cli, signal_name, retrieve, &mut out)?;
                } else {
                    let fired = process_observe_token(
                        lobe, tok, &mut stats, cli, signal_name, interject, rag, all_steps, retrieve,
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
