//! Worker-thread split for the presentation frontends (`present` / `present --scene`).
//!
//! THE HICCUP FIX. The presentation render loop animates the fire + glow every frame and must never
//! block. Previously it ALSO drove the lobe (`step()`), so one fire's worth of retrieval + ask-prefill
//! froze the whole scene (fire stops, glow stops). Here ALL llama work — observe, retrieve, the fused
//! interjection — runs on a dedicated worker thread that OWNS the engine/lobe/retriever (built INSIDE
//! the thread via `with_lobe`, so the `!Send` llama handles never cross a thread boundary). The worker
//! streams display events; the UI thread only ever renders the latest snapshot.
//!
//! Both properties the demo wants are preserved: the prose keeps SCROLLING (the worker still runs the
//! fused `step()`, so observation and generation co-batch) and the aside still types itself out
//! INCREMENTALLY (`Pending` carries the growing text). The retrieval pause simply happens off the
//! render thread now, so the animation stays buttery while the gent "reaches for a memory".

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};

use crate::stats::Welford;
use crate::{with_lobe, Cli};

const PROSE_TAIL_CHARS: usize = 12_000; // bound display memory (the lobe keeps its own context)
const MAX_RECENT: usize = 24; // recent asides kept for the feed under the live one

/// Events the worker streams to the UI thread, applied to `Display` in arrival order.
pub enum UiEvent {
    /// Construction (model load + prime) finished — safe for the UI to take over the terminal.
    Ready,
    /// A stream token's text → append to the prose ticker.
    Token(String),
    /// The in-flight aside's current text (incremental reveal); `None` between asides.
    Pending(Option<String>),
    /// A completed aside, to push onto the feed.
    Settled(String),
    /// The stream is exhausted.
    Done,
}

/// Control messages the UI thread sends back to the worker.
pub enum Control {
    /// Pause/resume feeding the stream (the fire keeps animating regardless; this only gates tokens).
    Pause(bool),
    /// Nudge the firing threshold live (presenter knob; `+`/`-`).
    AdjustZ(f32),
    /// Stop and exit (the worker finishes its current step, then returns).
    Quit,
}

/// The display snapshot the UI renders. The worker owns all lobe state; this is the projection of it.
#[derive(Default)]
pub struct Display {
    pub prose: String,
    pub pending: Option<String>,
    pub asides: Vec<String>,
    pub done: bool,
}

impl Display {
    fn apply(&mut self, ev: UiEvent) {
        match ev {
            UiEvent::Ready => {}
            UiEvent::Token(t) => {
                self.prose.push_str(&t);
                if self.prose.len() > PROSE_TAIL_CHARS {
                    let want = self.prose.len() - PROSE_TAIL_CHARS;
                    let cut = (want..self.prose.len())
                        .find(|&i| self.prose.is_char_boundary(i))
                        .unwrap_or(self.prose.len());
                    self.prose.drain(0..cut);
                }
            }
            UiEvent::Pending(p) => self.pending = p,
            UiEvent::Settled(s) => {
                self.asides.push(s);
                if self.asides.len() > MAX_RECENT {
                    self.asides.remove(0);
                }
            }
            UiEvent::Done => self.done = true,
        }
    }

    /// Apply everything the worker has queued (non-blocking). Returns `false` once the worker is gone
    /// (channel disconnected) — the caller exits if that happened before the stream finished.
    pub fn drain(&mut self, rx: &Receiver<UiEvent>) -> bool {
        loop {
            match rx.try_recv() {
                Ok(ev) => self.apply(ev),
                Err(TryRecvError::Empty) => return true,
                Err(TryRecvError::Disconnected) => return false,
            }
        }
    }
}

/// Entry point for `--mode demo-tui`: spawn the llama worker, wait for it to finish loading, then run
/// the chosen render loop on THIS (the main) thread. On exit, signal the worker and reap it.
pub fn run(cli: Cli) -> Result<()> {
    let (input, tick_ms, skip_to, scene) =
        (cli.input.clone(), cli.tick_ms, cli.skip_to.clone(), cli.scene());
    let title = std::path::Path::new(&input)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("stream")
        .to_string();

    let (ui_tx, ui_rx) = mpsc::channel::<UiEvent>();
    let (ctrl_tx, ctrl_rx) = mpsc::channel::<Control>();

    let handle: JoinHandle<Result<()>> = thread::spawn(move || {
        with_lobe(&cli, |lobe, retrieve| {
            worker_loop(lobe, retrieve, &cli, &input, tick_ms, &skip_to, &ui_tx, &ctrl_rx)
        })
    });

    // Block until the worker signals it has loaded + primed the model (the slow part) BEFORE taking
    // the terminal — so the Metal load banner prints to a normal stderr and we don't render an empty
    // scene for several seconds. A disconnect here means the worker errored during construction (e.g.
    // a bad --model); join to surface that error rather than hang or draw a broken TUI.
    match ui_rx.recv() {
        Ok(UiEvent::Ready) => {}
        _ => return handle.join().unwrap_or_else(|_| Err(anyhow!("worker thread panicked"))),
    }

    let render_res = if scene {
        crate::present_scene::render(&title, &ui_rx, &ctrl_tx)
    } else {
        crate::present::render(&title, &ui_rx, &ctrl_tx)
    };

    // Stop the worker and reap it (it may be mid-interjection; the wait is bounded by one aside).
    let _ = ctrl_tx.send(Control::Quit);
    let join_res = handle.join().unwrap_or_else(|_| Err(anyhow!("worker thread panicked")));
    render_res.and(join_res)
}

/// The worker body: own the lobe + retriever, pace the stream, and stream display events. Runs the
/// SAME fused `step()` the old single-threaded loop ran — only now off the render thread.
#[allow(clippy::too_many_arguments)]
fn worker_loop(
    lobe: &mut crate::lobe::Lobe,
    retrieve: &mut crate::retrieval::RetrieveFn,
    cli: &Cli,
    input_path: &str,
    tick_ms: u64,
    skip_to: &str,
    tx: &Sender<UiEvent>,
    ctrl: &Receiver<Control>,
) -> Result<()> {
    let interject = cli.interject_on();
    let interject_max = cli.interject_max;
    let mut raw = std::fs::read_to_string(input_path)?;
    if !skip_to.is_empty() {
        if let Some(idx) = raw.find(skip_to) {
            raw = raw[idx..].to_string();
        }
    }
    let tokens = lobe.tokenize(&raw, false)?;
    let mut feed = tokens.into_iter();

    let mut stats = Welford::new(cli.warmup, cli.adapt);
    let mut z = cli.z; // pre-tuned; +/- still nudges silently
    let mut pending: Option<String> = None; // the in-flight aside (mirrors present.rs)
    let mut revealed = false;
    let mut paused = false;

    // Construction done — release the UI to take the terminal.
    if tx.send(UiEvent::Ready).is_err() {
        return Ok(()); // UI already gone
    }

    let tick = Duration::from_millis(tick_ms);
    loop {
        // Drain control first (non-blocking); Quit wins immediately.
        loop {
            match ctrl.try_recv() {
                Ok(Control::Pause(b)) => paused = b,
                Ok(Control::AdjustZ(d)) => z = (z + d).clamp(0.0, 12.0),
                Ok(Control::Quit) => return Ok(()),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return Ok(()), // UI gone
            }
        }
        if paused {
            thread::sleep(Duration::from_millis(16)); // stay responsive to unpause/quit
            continue;
        }

        let step_start = Instant::now();
        match feed.next() {
            Some(tok) => {
                // One fused step: observe + advance any in-flight aside in one decode (prose keeps
                // scrolling). With --rag, step() retrieves on a fire and weaves the recall into the
                // aside IN VOICE — the per-fire retrieval + ask-prefill stall now blocks only THIS
                // thread, never the render loop.
                let status = if interject {
                    let out = lobe.step(tok, &mut stats, z, cli.topk, interject_max, retrieve)?;
                    if tx.send(UiEvent::Token(out.step.token_text)).is_err() {
                        return Ok(());
                    }
                    Some(out.interjection)
                } else {
                    let s = lobe.observe(tok, &mut stats, z, cli.topk)?;
                    if tx.send(UiEvent::Token(s.token_text)).is_err() {
                        return Ok(());
                    }
                    None
                };

                // The dedup/reveal policy lives in the lobe; emit whatever survives + the live aside.
                if let Some(text) = lobe.advance_reveal(status, &mut pending, &mut revealed)? {
                    let _ = tx.send(UiEvent::Settled(text));
                }
                let _ = tx.send(UiEvent::Pending(pending.clone()));
            }
            None => {
                let _ = tx.send(UiEvent::Done);
                return Ok(());
            }
        }

        // Pace: one token per `tick`, minus the time the step itself took (a long interjection step
        // already "spent" the tick). The render thread runs at its own cadence regardless.
        if let Some(rem) = tick.checked_sub(step_start.elapsed()) {
            thread::sleep(rem);
        }
    }
}
