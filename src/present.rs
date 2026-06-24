//! Presentation TUI — the SHOWCASE, as opposed to `tui` (the calibration instrument). Same `Lobe`,
//! same fused loop; a clean stage. No sparkline, no z-heatmap, no per-token shading, no trigger
//! list, no knobs readout — just the prose streaming past like a teleprompter and the lobe's asides
//! forming *alongside* it (the concurrent fused forward pass made visible: the text keeps scrolling
//! while the reply types itself out below). For the "watch it read the novel" demo.
//!
//! Controls are deliberately minimal: `space` pauses, `q` quits. `+`/`-` still nudge the threshold
//! live (handy for a presenter) but aren't advertised — the config is meant to be pre-tuned.

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph, Wrap};
use ratatui::Frame;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use crate::present_worker::{Control, Display, UiEvent};

/// ~30fps render cadence — independent of the token pace (which the worker owns). Keeps the frame
/// loop responsive so input and (in the scene skin) the fire/glow stay smooth no matter the tick.
const FRAME_MS: u64 = 33;

/// The animated "thinking" placeholder shown while an aside is prefilling (pending is `Some("")` — a
/// fire has happened but no reply token exists yet; the prefill is spreading across ticks). The
/// trailing ellipsis cycles 1→3 dots ~every 350ms so it reads as ALIVE and distinct from the typed-out
/// aside. `anim_ms` is wall-clock-since-render-start, so the animation is smooth regardless of tick.
/// `pub(crate)` so the scene skin shares the identical label.
pub(crate) fn musing_label(anim_ms: u64) -> String {
    let dots = (anim_ms / 350 % 3) as usize + 1; // 1..=3
    format!("musing{}", ".".repeat(dots))
}

/// Render-only loop: consume the worker's display events, draw, and forward key controls. ALL llama
/// work lives on the worker thread (`present_worker`), so this loop never blocks — that's the hiccup
/// fix. `space` pauses the stream, `q` quits, `+`/`-` nudge the threshold (silently).
pub fn render(
    title: &str,
    ui_rx: &Receiver<UiEvent>,
    ctrl_tx: &Sender<Control>,
) -> Result<()> {
    let mut display = Display::default();
    let mut paused = false; // local mirror, for the footer; the worker is the source of truth
    let frame = Duration::from_millis(FRAME_MS);
    let mut last = Instant::now();
    let anim_start = Instant::now(); // drives the "musing…" ellipsis, smooth regardless of tick

    let mut terminal = ratatui::init();
    let res = (|| -> Result<()> {
        loop {
            // Pull whatever the worker has produced; exit if it vanished before the stream finished.
            if !display.drain(ui_rx) && !display.done {
                break;
            }
            let anim_ms = anim_start.elapsed().as_millis() as u64;
            terminal.draw(|f| {
                draw(f, title, &display.prose, &display.asides,
                     display.pending.as_deref(), paused, display.done, anim_ms)
            })?;

            let timeout = frame.checked_sub(last.elapsed()).unwrap_or_default();
            if event::poll(timeout)? {
                if let Event::Key(k) = event::read()? {
                    match k.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char(' ') => {
                            paused = !paused;
                            let _ = ctrl_tx.send(Control::Pause(paused));
                        }
                        KeyCode::Char('+') | KeyCode::Char('=') => {
                            let _ = ctrl_tx.send(Control::AdjustZ(0.25));
                        }
                        KeyCode::Char('-') | KeyCode::Char('_') => {
                            let _ = ctrl_tx.send(Control::AdjustZ(-0.25));
                        }
                        _ => {}
                    }
                }
            }
            last = Instant::now();
        }
        Ok(())
    })();

    ratatui::restore();
    res
}

#[allow(clippy::too_many_arguments)]
fn draw(
    f: &mut Frame,
    title: &str,
    prose: &str,
    asides: &[String],
    pending: Option<&str>,
    paused: bool,
    done: bool,
    anim_ms: u64,
) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(30), // the prose — readable, but still the quiet source
            Constraint::Min(8),         // the asides feed — the focus
            Constraint::Length(1),      // a quiet footer
        ])
        .split(f.area());

    draw_prose(f, outer[0], title, prose);
    draw_feed(f, outer[1], asides, pending, anim_ms);
    draw_footer(f, outer[2], paused, done);
}

/// The streamed text as a teleprompter — scrolled so the newest line sits at the bottom.
/// `pub(crate)` so the scene skin (`present_scene`) reuses the identical prose pane.
pub(crate) fn draw_prose(f: &mut Frame, area: Rect, title: &str, prose: &str) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            format!(" reading · {title} "),
            Style::default().fg(Color::DarkGray),
        ))
        .padding(Padding::horizontal(2));
    let inner = block.inner(area);
    let w = inner.width.max(1) as usize;
    let h = inner.height.max(1) as usize;
    // Scroll so the tail is visible (newest line at the bottom).
    let total = wrap_count(prose, w);
    let scroll = total.saturating_sub(h) as u16;
    // Readable but quiet — the source the lobe is reading; the asides feed below is the focus.
    let para = Paragraph::new(prose)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0))
        .style(Style::default().fg(Color::Gray));
    f.render_widget(para, area);
}

/// The asides feed — the focus. A centered, narrow column reading as a stack of spotlit musings,
/// OLDEST→NEWEST top→bottom: the in-flight one forms at the very bottom (warm, with a caret) and as
/// each settles the older ones slide up and off the top. Bottom-anchored (scrolled so the newest is
/// always at the bottom edge), with ample spacing between items.
fn draw_feed(f: &mut Frame, area: Rect, asides: &[String], pending: Option<&str>, anim_ms: u64) {
    // Blank display lines between items — "ample spacing" (tune here).
    const ASIDE_GAP: usize = 2;

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(16),
            Constraint::Percentage(68),
            Constraint::Percentage(16),
        ])
        .split(area);
    let col = cols[1];
    let block = Block::default().padding(Padding::horizontal(1));
    let inner = block.inner(col);
    let w = inner.width.max(1) as usize;
    let h = inner.height.max(1) as usize;

    let mut lines: Vec<Line> = Vec::new();
    let mut total = 0usize; // running wrapped-line count, for bottom-anchoring the scroll
    let n = asides.len();
    for (i, s) in asides.iter().enumerate() {
        if i > 0 {
            for _ in 0..ASIDE_GAP {
                lines.push(Line::from(""));
                total += 1;
            }
        }
        // The most recent settled aside (only when nothing is forming) gets a touch more weight.
        let style = if i + 1 == n && pending.is_none() {
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Cyan)
        };
        lines.push(Line::from(Span::styled(s.clone(), style)));
        total += wrap_count(s, w);
    }
    if let Some(p) = pending {
        if n > 0 {
            for _ in 0..ASIDE_GAP {
                lines.push(Line::from(""));
                total += 1;
            }
        }
        let (body, style) = if p.is_empty() {
            // Prefilling: the gent is thinking, not speaking yet — a dim, italic, animated placeholder
            // (low contrast + a breathing ellipsis), deliberately distinct from the typed-out aside.
            (
                musing_label(anim_ms),
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC | Modifier::DIM),
            )
        } else {
            // The live aside — warm and bold, forming at the bottom of the feed.
            (format!("{p}▍"), Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        };
        // +1 safety: never under-count the focal (live) item, so its last line is never clipped.
        total += wrap_count(&body, w) + 1;
        lines.push(Line::from(Span::styled(body, style)));
    } else if n == 0 {
        lines.push(Line::from(Span::styled("…", Style::default().fg(Color::DarkGray))).centered());
        total += 1;
    }

    // Bottom-anchor: when the feed is shorter than the pane, pad the TOP so it sits at the bottom
    // (chat-style) from the very first aside; once it overflows, scroll so the newest/in-flight aside
    // stays pinned to the bottom edge and older ones slide up out of view.
    let scroll = if total < h {
        let pad = h - total;
        let mut padded = Vec::with_capacity(pad + lines.len());
        padded.resize(pad, Line::from(""));
        padded.extend(lines);
        lines = padded;
        0
    } else {
        (total - h) as u16
    };
    let para = Paragraph::new(lines)
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0))
        .block(block);
    f.render_widget(para, col);
}

pub(crate) fn draw_footer(f: &mut Frame, area: Rect, paused: bool, done: bool) {
    let (label, color) = if done {
        ("finished", Color::Green)
    } else if paused {
        ("paused", Color::Yellow)
    } else {
        ("reading", Color::Cyan)
    };
    let line = Line::from(vec![
        Span::styled(
            format!(" {label} "),
            Style::default().fg(Color::Black).bg(color),
        ),
        Span::styled(
            "  space pause · q quit",
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    f.render_widget(Paragraph::new(line).alignment(Alignment::Left), area);
}

/// Count the display lines `s` occupies after greedy word-wrap at inner width `w` — matching
/// ratatui's `Wrap` closely enough to bottom-anchor a feed without clipping. (The earlier ceil(chars/w)
/// estimate under-counted word-wrapped paragraphs, which would clip the live aside off the bottom.)
fn wrap_count(s: &str, w: usize) -> usize {
    if w == 0 {
        return 1;
    }
    let mut lines = 0usize;
    for hard in s.split('\n') {
        let mut cur = 0usize; // chars used on the current visual line
        let mut started = false; // any word placed on this hard line yet
        for word in hard.split(' ') {
            let wl = word.chars().count();
            if wl == 0 {
                cur += 1; // a space (collapsed run / leading space)
                if cur > w {
                    lines += 1;
                    cur = 0;
                }
                continue;
            }
            let need = if started { cur + 1 + wl } else { wl };
            if !started || need <= w {
                cur = need;
                started = true;
            } else {
                lines += 1; // wrap to a new visual line
                cur = wl;
            }
            if cur > w {
                // a word longer than the column spills across multiple lines
                lines += cur / w;
                cur %= w;
            }
        }
        lines += 1; // the final visual line of this hard line
    }
    lines.max(1)
}
