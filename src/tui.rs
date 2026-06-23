//! The TUI calibration instrument. Paces a transcript through the same `Lobe` the
//! headless mode uses, and renders:
//!   - the transcript, each token shaded by its z-score (calm -> hot)
//!   - a live surprisal sparkline
//!   - a baseline/threshold readout
//!   - a scrolling log of triggers, each with the top-k the model expected instead
//!   - optional chat-framed interjections, to show the observe -> generate -> resume switch
//!
//! Single-threaded by design: each UI tick does exactly one unit of work — observe one stream
//! token, or, while an interjection is in flight, generate one of its tokens (pumped via
//! `lobe.interject_step`). So the interjection streams in token-by-token and the event loop never
//! freezes — input and redraw stay live throughout. Observation pauses while it generates (the GPU
//! is serial), but generation is on a scratch sequence, so observer output is byte-identical to
//! running without `--interject`. The in-flight reply renders live at the top of the interjections
//! panel with a caret.
//!
//! FRAGILE note applies to nothing here — this is all ratatui/crossterm, which are
//! stable. The version-sensitive surface is entirely in lobe.rs.

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline, Wrap};
use ratatui::Frame;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::lobe::{InterjectStatus, Lobe, Trigger};
use crate::stats::Welford;
use crate::Cli;

struct Shaded {
    text: String,
    z: f32,
}

pub fn run(lobe: &mut Lobe, cli: &Cli, input_path: &str, tick_ms: u64, skip_to: &str) -> Result<()> {
    let interject = cli.interject_on(); // global flag (on by default; --no-interject disables)
    let interject_max = cli.interject_max;
    // Pre-tokenize the whole transcript as the stream. add_bos=false (continuation).
    let mut raw = std::fs::read_to_string(input_path)?;
    // Optional: skip to a marker substring (the novel-reading demo skips front-matter to the
    // narrative). If the marker isn't found, fall through and stream from the top.
    if !skip_to.is_empty() {
        if let Some(idx) = raw.find(skip_to) {
            raw = raw[idx..].to_string();
        }
    }
    let tokens = lobe.tokenize(&raw, false)?;
    let mut feed = tokens.into_iter();

    let mut stats = Welford::new(cli.warmup, cli.adapt);
    // Live, mutable firing threshold so it can be tuned with +/- while watching (this is the
    // whole point of the calibration TUI). Seeded from --z.
    let mut z = cli.z;

    // Display state.
    let mut shaded: Vec<Shaded> = Vec::new(); // transcript, shaded per token
    let mut spark: VecDeque<u64> = VecDeque::with_capacity(120);
    let mut triggers: VecDeque<Trigger> = VecDeque::with_capacity(64);
    let mut interjections: Vec<String> = Vec::new();
    let mut pending: Option<String> = None; // in-flight interjection text, streamed in live
    // Whether the in-flight interjection has been "revealed" (shown as live content). We buffer
    // silently (showing "💭 thinking…") until novelty is decidable, then reveal if it's not a known
    // duplicate — so a filtered interjection is never rendered then deleted. With --dedup 0 this is
    // decidable immediately, so streaming is unchanged.
    let mut revealed = false;
    let mut paused = false;
    let mut done = false;
    let mut last_tick = Instant::now();
    let tick = Duration::from_millis(tick_ms);

    let mut terminal = ratatui::init();
    let res = (|| -> Result<()> {
        loop {
            let pos = lobe.position();
            let resets = lobe.resets();
            terminal.draw(|f| {
                draw(
                    f,
                    cli,
                    z,
                    pos,
                    resets,
                    &shaded,
                    &spark,
                    &triggers,
                    &interjections,
                    pending.as_deref(),
                    &stats,
                    paused,
                    done,
                )
            })?;

            // Input (non-blocking-ish): poll for the remainder of the tick.
            let timeout = tick
                .checked_sub(last_tick.elapsed())
                .unwrap_or_else(|| Duration::from_millis(0));
            if event::poll(timeout)? {
                if let Event::Key(k) = event::read()? {
                    match k.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char(' ') => paused = !paused,
                        KeyCode::Char('+') | KeyCode::Char('=') => z = (z + 0.25).min(12.0),
                        KeyCode::Char('-') | KeyCode::Char('_') => z = (z - 0.25).max(0.0),
                        _ => {}
                    }
                }
            }

            if last_tick.elapsed() < tick {
                continue;
            }
            last_tick = Instant::now();
            if paused || done {
                continue;
            }

            // One FUSED step per tick (CONCURRENT_FORWARD_PASS.md / `step()`): observation (seq 0)
            // and any in-flight interjection token (GEN_SEQ) are co-batched into ONE decode, so
            // observation NEVER stalls while the lobe generates — the reply streams in alongside the
            // text. (The cap+reset overrun that previously forced the timesliced fallback is fixed —
            // FUSED_CACHE_GO_NOGO: the roll now reserves the interjection's full concurrent footprint;
            // validated crash-free over >10×n_ctx with interjections firing throughout.) `--no-interject`
            // drops to plain observation.
            if let Some(tok) = feed.next() {
                let (s, status) = if interject {
                    let out = lobe.step(tok, &mut stats, z, cli.topk, interject_max)?;
                    (out.step, Some(out.interjection))
                } else {
                    (lobe.observe(tok, &mut stats, z, cli.topk)?, None)
                };
                push_spark(&mut spark, s.surprisal);
                shaded.push(Shaded {
                    text: s.token_text.clone(),
                    z: s.z,
                });
                if shaded.len() > 4000 {
                    shaded.drain(0..1000); // cap transcript memory for the display
                }
                if let Some(t) = s.trigger {
                    if triggers.len() == 64 {
                        triggers.pop_front();
                    }
                    triggers.push_back(t);
                }
                // Render the fused interjection state. The dedup/reveal logic is identical to before;
                // only its driver changed (step()'s InterjectStatus instead of interject_step()).
                match status {
                    Some(InterjectStatus::Started) => {
                        pending = Some(String::new()); // buffering the opening → "💭 thinking…"
                        revealed = false;
                    }
                    Some(InterjectStatus::Working(partial)) => {
                        if revealed {
                            pending = Some(partial);
                        } else if lobe.interjection_doomed(&partial) {
                            lobe.abort_interjection()?; // dup — never render it
                            pending = None;
                        } else if lobe.interjection_decidable(&partial) {
                            revealed = true; // novel — reveal + stream from here
                            pending = Some(partial);
                        } else {
                            pending = Some(String::new());
                        }
                    }
                    Some(InterjectStatus::Done(text)) => {
                        let text = text.trim();
                        if !text.is_empty() && (revealed || !lobe.interjection_doomed(text)) {
                            lobe.record_interjection(text); // novelty memory (1b)
                            interjections.push(format!(">> {text}"));
                            if interjections.len() > 50 {
                                interjections.remove(0);
                            }
                        }
                        pending = None;
                        revealed = false;
                    }
                    Some(InterjectStatus::Idle) | None => {}
                }
            } else {
                done = true;
            }
        }
        Ok(())
    })();

    ratatui::restore();
    res
}

fn push_spark(spark: &mut VecDeque<u64>, surprisal: f32) {
    // Sparkline wants u64; scale nats into a small integer band.
    let v = (surprisal * 10.0).clamp(0.0, 1000.0) as u64;
    if spark.len() == 120 {
        spark.pop_front();
    }
    spark.push_back(v);
}

#[allow(clippy::too_many_arguments)]
fn draw(
    f: &mut Frame,
    cli: &Cli,
    z: f32,
    pos: i32,
    resets: u64,
    shaded: &[Shaded],
    spark: &VecDeque<u64>,
    triggers: &VecDeque<Trigger>,
    interjections: &[String],
    pending: Option<&str>,
    stats: &Welford,
    paused: bool,
    done: bool,
) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),    // transcript
            Constraint::Length(3), // sparkline
            Constraint::Length(3), // baseline gauge
            Constraint::Length(1), // status
        ])
        .split(f.area());

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(outer[0]);

    draw_transcript(f, body[0], shaded);
    draw_side(f, body[1], triggers, interjections, pending);
    draw_sparkline(f, outer[1], spark);
    draw_gauge(f, outer[2], stats, z);
    draw_status(f, outer[3], stats, cli, z, pos, resets, paused, done);
}

fn draw_transcript(f: &mut Frame, area: Rect, shaded: &[Shaded]) {
    // Show only the tail that plausibly fits.
    let tail = shaded.len().saturating_sub(1200);
    let spans: Vec<Span> = shaded[tail..]
        .iter()
        .map(|s| Span::styled(s.text.clone(), shade_style(s.z)))
        .collect();
    let para = Paragraph::new(Line::from(spans))
        .block(Block::default().borders(Borders::ALL).title(" stream "))
        .wrap(Wrap { trim: false });
    f.render_widget(para, area);
}

/// Calm -> hot shading by z-score.
fn shade_style(z: f32) -> Style {
    let base = if z >= 4.0 {
        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
    } else if z >= 3.0 {
        Style::default().fg(Color::LightRed)
    } else if z >= 2.0 {
        Style::default().fg(Color::Yellow)
    } else if z >= 1.0 {
        Style::default().fg(Color::Green)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    base
}

fn draw_side(
    f: &mut Frame,
    area: Rect,
    triggers: &VecDeque<Trigger>,
    interjections: &[String],
    pending: Option<&str>,
) {
    let split = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    let items: Vec<ListItem> = triggers
        .iter()
        .rev()
        .map(|t| {
            let expected: String = t
                .expected
                .iter()
                .map(|(s, p)| format!("{}·{:.2}", clean(s), p))
                .collect::<Vec<_>>()
                .join(" ");
            let header = Line::from(vec![
                Span::styled(
                    format!("[{}] ", t.stream_index),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("{:?}", clean(&t.token_text)),
                    Style::default().fg(Color::LightRed).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("  z={:.1}  s={:.1}", t.z, t.surprisal),
                    Style::default().fg(Color::Gray),
                ),
            ]);
            let exp = Line::from(Span::styled(
                format!("    expected: {expected}"),
                Style::default().fg(Color::Blue),
            ));
            ListItem::new(vec![header, exp])
        })
        .collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" triggers (newest first) "));
    f.render_widget(list, split[0]);

    // Newest first, one Line per interjection. A wrapping Paragraph (not a List) so long
    // observations wrap to the panel width instead of being truncated at the right edge. The
    // in-flight interjection (if any) streams at the top with a live caret.
    let mut inter: Vec<Line> = Vec::new();
    if let Some(p) = pending {
        let shown = if p.is_empty() {
            "💭 thinking…".to_string()
        } else {
            format!("💭 {p}▋")
        };
        inter.push(Line::from(Span::styled(
            shown,
            Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
        )));
    }
    inter.extend(
        interjections
            .iter()
            .rev()
            .map(|s| Line::from(Span::styled(s.clone(), Style::default().fg(Color::Cyan)))),
    );
    let inter_para = Paragraph::new(inter)
        .block(Block::default().borders(Borders::ALL).title(" interjections "))
        .wrap(Wrap { trim: false });
    f.render_widget(inter_para, split[1]);
}

fn draw_sparkline(f: &mut Frame, area: Rect, spark: &VecDeque<u64>) {
    let data: Vec<u64> = spark.iter().copied().collect();
    let s = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title(" surprisal (nats x10) "))
        .data(&data)
        .style(Style::default().fg(Color::Magenta));
    f.render_widget(s, area);
}

fn draw_gauge(f: &mut Frame, area: Rect, stats: &Welford, z_threshold: f32) {
    // Show the firing line (mean + z*std) as a fraction of a ceiling that sits a few sigma
    // *above* it. The ceiling tracks the live threshold (tunable up to z=12) rather than a
    // fixed 6-sigma mark, so the bar keeps moving across the whole range instead of pegging
    // at 100% the moment z passes 6.
    let fire_at = stats.mean() + z_threshold * stats.std();
    let headroom = (z_threshold + 3.0).max(6.0);
    let ceiling = (stats.mean() + headroom * stats.std()).max(1.0);
    let ratio = (fire_at / ceiling).clamp(0.0, 1.0) as f64;
    let g = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(" fire threshold "))
        .gauge_style(Style::default().fg(Color::LightRed))
        .ratio(ratio)
        .label(format!(
            "fire@ {:.2} nats (mean {:.2}, sd {:.2}, z {:.1})",
            fire_at,
            stats.mean(),
            stats.std(),
            z_threshold
        ));
    f.render_widget(g, area);
}

fn draw_status(
    f: &mut Frame,
    area: Rect,
    stats: &Welford,
    cli: &Cli,
    z: f32,
    pos: i32,
    resets: u64,
    paused: bool,
    done: bool,
) {
    let state = if done {
        "DONE"
    } else if paused {
        "PAUSED"
    } else {
        "OBSERVING"
    };
    let line = Line::from(vec![
        Span::styled(format!(" {state} "), Style::default().fg(Color::Black).bg(Color::White)),
        Span::raw(format!(
            "  tokens={}  ctx_pos={}  resets={}  z={:.2}  warmup={}   ",
            stats.count(),
            pos,    // live KV-cache position, via Lobe::position() — sawtooths with cap+reset
            resets, // #6 eviction count
            z,      // live threshold (tunable with +/-), not the static cli.z
            cli.warmup
        )),
        Span::styled("[space]pause  [+/-]z  [q]quit", Style::default().fg(Color::DarkGray)),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

/// Make whitespace tokens visible in compact UI contexts.
fn clean(s: &str) -> String {
    s.replace('\n', "⏎").replace('\t', "→")
}
