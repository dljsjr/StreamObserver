//! Presentation TUI — SCENE skin (`present --scene`). A cosmetic alternate to `present`: the observer
//! rendered as a bearded gent in a wingback chair by a fire in a Victorian study (a chunky half-block
//! sprite, `assets/scene/study.json`), with each interjection appearing as a rounded speech bubble.
//!
//! It reuses `present`'s EXACT base presentation — the same fused loop, the same top prose pane
//! (`present::draw_prose`) and footer (`present::draw_footer`) — and only replaces the interjection
//! display: where plain `present` shows the asides feed, this stages the sprite (anchored
//! bottom-center) with the current aside as a speech bubble in the open band ABOVE it. Controls match
//! `present`: `space` pauses, `q` quits, `+`/`-` nudge the threshold silently.

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode};
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph, Wrap};
use ratatui::Frame;
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};
use tachyonfx::ToRgbComponents;

use crate::present_worker::{Control, Display, UiEvent};
use crate::sprite::{Anim, Sprite};

/// ~30fps render cadence — independent of the token pace (the worker owns that). Keeps the fire/glow
/// animation and input smooth regardless of `--tick-ms`.
const FRAME_MS: u64 = 33;

/// The study scene art (sprite-grid JSON), embedded at compile time. 100 px wide × 48 px tall =
/// 100 cells wide × 24 cell-rows tall. FIXED size — never scaled; anchored bottom-center.
const STUDY_JSON: &str = include_str!("../assets/scene/study.json");

/// The animated fireplace flame (4-frame loop, `fire.py` → `fire.anim.json`), blitted OVER the dark
/// firebox each tick. 9 px × 12 px. Its top-left anchors at scene pixel (FIRE_PX_X, FIRE_PX_Y).
const FIRE_JSON: &str = include_str!("../assets/scene/fire.anim.json");

/// Firebox interior top-left in SCENE PIXEL coords — where the flame overlay anchors. FIRE_PX_Y is
/// EVEN so the flame's pixel rows pair into cells in lockstep with the scene (no half-cell shear).
const FIRE_PX_X: u16 = 46;
const FIRE_PX_Y: u16 = 24;

/// The fire's glow center in SCENE PIXEL coords (the flame core, ≈ firebox center). The tachyonfx
/// glow shader pulses warm light that falls off with distance from here.
const GLOW_PX_X: f32 = 50.0;
const GLOW_PX_Y: f32 = 33.0;

/// Glow-shader tunables (all distances in CELLS from the fire center; see `make_glow`).
const GLOW_R: f32 = 26.0; // bright-zone reach: cells within ~this far pulse/warm with the fire
const GLOW_AMP: f32 = 0.30; // flicker amplitude at the hearth — how dramatic the brightness pulse is
const VIG_R0: f32 = 20.0; // vignette: cells start dimming beyond this distance
const VIG_R1: f32 = 58.0; // vignette: …reaching full dimming by here (the far corners)
const VIG_MAX_DIM: f32 = 0.55; // vignette: max brightness cut at the darkest edges (0=off, 1=black)

/// Where the man's head sits in the scene, in SCENE COLUMNS (0-based); the bubble's tail points at it.
/// (Werner stands reading in front of his chair; his head is centered at col 64 of the 100-wide room.)
const HEAD_COL: u16 = 64;

/// Render-only loop for the scene skin: consume the worker's display events and animate the study.
/// ALL llama work lives on the worker thread (`present_worker`), so the fire/glow NEVER freeze on a
/// retrieval or interjection — that's the hiccup fix. Controls match plain `present`: `space` pauses
/// the stream, `q` quits, `+`/`-` nudge the threshold silently.
pub fn render(
    title: &str,
    ui_rx: &Receiver<UiEvent>,
    ctrl_tx: &Sender<Control>,
) -> Result<()> {
    // Parse the scene + flame animation once, up front, so a malformed asset fails loudly before
    // entering raw mode.
    let scene = Sprite::from_json(STUDY_JSON)?;
    let fire = Anim::from_json(FIRE_JSON)?;

    let mut display = Display::default();
    let mut paused = false; // local mirror for the footer; the worker is the source of truth

    // Fireplace animation state: the flame loops at its own fps independent of the stream, and the
    // glow is a per-frame pass driven by wall-clock time (`anim_start`). Both keep crackling even
    // while paused (or while the worker is mid-interjection) — a fire doesn't stop.
    let mut fire_idx = 0usize;
    let mut last_fire = Instant::now();
    let fire_period = Duration::from_secs_f32(1.0 / fire.fps.max(1.0));
    let frame = Duration::from_millis(FRAME_MS);
    let mut last = Instant::now();
    let anim_start = Instant::now(); // drives the glow flicker AND the "musing…" ellipsis

    let mut terminal = ratatui::init();
    let res = (|| -> Result<()> {
        loop {
            // Pull whatever the worker produced; exit if it vanished before the stream finished.
            if !display.drain(ui_rx) && !display.done {
                break;
            }

            if last_fire.elapsed() >= fire_period {
                fire_idx = (fire_idx + 1) % fire.frames.len();
                last_fire = Instant::now();
            }
            let flame = &fire.frames[fire_idx];
            let last_aside = display.asides.last().map(|s| s.as_str());
            let anim_ms = anim_start.elapsed().as_millis() as u64;
            terminal.draw(|f| {
                draw(f, &scene, flame, title, &display.prose,
                     last_aside, display.pending.as_deref(), paused, display.done, anim_ms)
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

/// Same base layout as `present` (prose pane on top, footer at the bottom); the middle region —
/// where plain `present` shows the asides feed — instead stages the scene + speech bubble.
#[allow(clippy::too_many_arguments)]
fn draw(
    f: &mut Frame,
    scene: &Sprite,
    flame: &Sprite,
    title: &str,
    prose: &str,
    last_aside: Option<&str>,
    pending: Option<&str>,
    paused: bool,
    done: bool,
    anim_ms: u64,
) {
    // Paint the whole frame the room's deep-shadow tone so transparent sprite cells + margins read as
    // one continuous dark study, not the terminal default. The glow pass (in draw_stage) then lights
    // the WHOLE stage region from this base, so the room dissolves into the surrounding shadow.
    let backdrop = Block::default().style(Style::default().bg(Color::Rgb(34, 26, 22)));
    f.render_widget(backdrop, f.area());

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(30), // prose pane — IDENTICAL to present
            Constraint::Min(8),         // the stage: scene + speech bubble (replaces the asides feed)
            Constraint::Length(1),      // footer — IDENTICAL to present
        ])
        .split(f.area());

    crate::present::draw_prose(f, outer[0], title, prose);
    draw_stage(f, outer[1], scene, flame, last_aside, pending, anim_ms);
    crate::present::draw_footer(f, outer[2], paused, done);
}

/// The stage: sprite anchored bottom-center; the current aside as a speech bubble in the open band
/// ABOVE the sprite (never over it), with a short tail pointing down at the gent's head.
fn draw_stage(
    f: &mut Frame,
    area: Rect,
    scene: &Sprite,
    flame: &Sprite,
    last_aside: Option<&str>,
    pending: Option<&str>,
    anim_ms: u64,
) {
    let (sw, sh) = scene.cell_size();
    let sprite_w = sw.min(area.width);
    let sprite_h = sh.min(area.height);
    let sprite_x = area.x + (area.width.saturating_sub(sprite_w)) / 2;
    let sprite_y = area.bottom().saturating_sub(sprite_h); // anchor to the BOTTOM of the region
    let sprite_rect = Rect { x: sprite_x, y: sprite_y, width: sprite_w, height: sprite_h };
    f.render_widget(scene, sprite_rect);

    // Animated flame, blitted OVER the firebox. The firebox top-left is at scene pixel
    // (FIRE_PX_X, FIRE_PX_Y); convert to a cell offset within the scene's render rect (pixel row →
    // cell row = /2, valid because FIRE_PX_Y is even). The flame's transparent cells let the dark
    // firebox show through; its partial cells composite over it (see sprite.rs render).
    let (fw, fh) = flame.cell_size();
    let flame_rect = Rect {
        x: sprite_rect.x + FIRE_PX_X,
        y: sprite_rect.y + FIRE_PX_Y / 2,
        width: fw,
        height: fh,
    };
    f.render_widget(flame, flame_rect);

    // Fireplace glow: warm + brighten near the hearth, vignette to shadow at the edges, with an
    // organic flicker. Applied AFTER the scene + flame are drawn (it modulates the already-painted
    // cells) and over the WHOLE stage `area` — NOT just the sprite rect — so the room's walls dissolve
    // into the surrounding shadow as one continuous firelit field, instead of reading as a lit
    // rectangle floating on a flat backdrop. The fire center is the flame in absolute buffer cells.
    let t = anim_ms as f32 / 1000.0;
    let fcx = sprite_rect.x as f32 + GLOW_PX_X;
    let fcy = sprite_rect.y as f32 + GLOW_PX_Y / 2.0;
    apply_glow(f.buffer_mut(), area, fcx, fcy, t);

    // The bubble lives in the band ABOVE the sprite (never over it).
    let band = Rect {
        x: area.x,
        y: area.y,
        width: area.width,
        height: sprite_y.saturating_sub(area.y),
    };
    draw_bubble(f, band, sprite_rect, last_aside, pending, anim_ms);
}

/// The bubble's three visual states: thinking (prefill, before any reply token), actively typing the
/// reply, and settled. They get deliberately distinct chrome so "musing" never reads as a real aside.
enum BubbleState {
    Musing,  // prefilling — dim, italic, animated ellipsis (low contrast: the gent is thinking)
    Forming, // typing the reply — warm/bold with a caret
    Settled, // the last completed aside — calm cream
}

/// A rounded speech bubble in `band` (the open space above the sprite), holding the CURRENT aside —
/// the in-flight one (warm/bold, with a caret) while generating, else the most recent settled one.
/// Bottom-anchored just above the sprite, with a short downward tail stub in the gap pointing at the
/// gent (so the tail never has to draw over the art).
fn draw_bubble(
    f: &mut Frame,
    band: Rect,
    sprite_rect: Rect,
    last_aside: Option<&str>,
    pending: Option<&str>,
    anim_ms: u64,
) {
    let (text, state): (String, BubbleState) = match pending {
        Some(p) if !p.is_empty() => (format!("{p}▍"), BubbleState::Forming),
        // Prefilling: thinking, not speaking yet — the dim animated placeholder (see musing_label).
        Some(_) => (crate::present::musing_label(anim_ms), BubbleState::Musing),
        None => match last_aside {
            Some(a) if !a.is_empty() => (a.to_string(), BubbleState::Settled),
            _ => return, // nothing to say yet → no bubble (just the quiet scene)
        },
    };
    if band.height < 4 || band.width < 14 {
        return; // not enough open space above the sprite to stage a readable bubble
    }

    // Centered horizontally; bottom-anchored in the band, leaving one gap row for the tail.
    let bw = band.width.saturating_sub(8).clamp(12, 60);
    let bh = (band.height.saturating_sub(1)).clamp(3, 12);
    let bx = band.x + (band.width.saturating_sub(bw)) / 2;
    let by = band.bottom().saturating_sub(bh + 1);
    let bubble = Rect { x: bx, y: by, width: bw, height: bh };

    let (border_color, text_style) = match state {
        // Low contrast: a muted border that recedes into the warm dark + dim italic text — clearly
        // "thinking", not "speaking". The animated ellipsis (in `text`) carries the liveness.
        BubbleState::Musing => (
            Color::Rgb(92, 80, 74),
            Style::default()
                .fg(Color::Rgb(150, 138, 130))
                .add_modifier(Modifier::ITALIC | Modifier::DIM),
        ),
        BubbleState::Forming => (
            Color::Yellow,
            Style::default().fg(Color::Rgb(255, 244, 214)).add_modifier(Modifier::BOLD),
        ),
        BubbleState::Settled => (
            Color::Rgb(214, 214, 208),
            Style::default().fg(Color::Rgb(232, 230, 224)),
        ),
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color))
        .style(Style::default().bg(Color::Rgb(40, 32, 31)))
        .padding(Padding::horizontal(1));
    f.render_widget(
        Paragraph::new(text).block(block).wrap(Wrap { trim: false }).style(text_style),
        bubble,
    );

    // Tail: a short run in the gap rows between the bubble and the sprite, at the gent's head column,
    // ending in a down-pointer. Lives entirely in the gap → never overlaps the art.
    let head_x = (sprite_rect.x + HEAD_COL).min(sprite_rect.right().saturating_sub(1));
    let tail_style = Style::default().fg(border_color).bg(Color::Rgb(40, 32, 31));
    for y in bubble.bottom()..sprite_rect.y {
        let glyph = if y + 1 == sprite_rect.y { '▾' } else { '│' };
        if let Some(cell) = f.buffer_mut().cell_mut((head_x, y)) {
            cell.set_symbol(&glyph.to_string());
            cell.set_style(tail_style);
        }
    }
}

/// The fireplace glow: every frame, warm + pulse each cell in `area` by an amount that falls off with
/// distance from the fire, vignetting to shadow at the edges. A direct per-frame pass (not a tachyonfx
/// effect) so it can cover the WHOLE stage with the fire center at an ABSOLUTE buffer position (`fcx`,
/// `fcy`) — the effect-relative center couldn't, which is why the glow used to be confined to the
/// sprite rect (→ the room read as a lit rectangle on a flat backdrop). `t` is wall-clock seconds,
/// driving the flicker; the falloff/warmth math is unchanged.
fn apply_glow(buf: &mut Buffer, area: Rect, fcx: f32, fcy: f32, t: f32) {
    // Layered sines → an organic flicker in roughly [-1, 1]: a fast crackle over a slower swell.
    let flick = 0.50 * (t * 11.0).sin() + 0.30 * (t * 19.0 + 1.3).sin() + 0.20 * (t * 6.1 + 2.1).sin();
    for y in area.top()..area.bottom() {
        for x in area.left()..area.right() {
            let Some(cell) = buf.cell_mut((x, y)) else { continue };
            let dx = x as f32 - fcx;
            let dy = (y as f32 - fcy) * 2.0; // cells are ~2× tall → keep the falloff circular
            let dist = (dx * dx + dy * dy).sqrt();
            // Near-fire GLOW: brighten + warm + flicker, hugging the hearth (eased reach GLOW_R).
            let near = (1.0 - dist / GLOW_R).clamp(0.0, 1.0);
            let near = near * near;
            // VIGNETTE: dim toward the edges/corners (firelight falloff → the room recedes to shadow).
            let vig = ((dist - VIG_R0) / (VIG_R1 - VIG_R0)).clamp(0.0, 1.0);
            // Combined brightness: a dramatic pulse at the hearth, multiplied by the edge dimming.
            let bright = (1.0 + GLOW_AMP * near * flick) * (1.0 - VIG_MAX_DIM * vig);
            let warmth = 0.07 * near; // warm bias only near the fire
            cell.set_fg(warm(cell.fg.to_rgb(), bright, warmth));
            cell.set_bg(warm(cell.bg.to_rgb(), bright, warmth));
        }
    }
}

/// Re-light one color: scale brightness by `bright` (the glow pulse × vignette dimming), then nudge
/// toward firelight orange by `warmth` (proximity to the fire). Clamped to valid RGB.
fn warm((r, g, b): (u8, u8, u8), bright: f32, warmth: f32) -> Color {
    let chan = |c: u8, target: f32| {
        let v = c as f32 * bright;
        (v + (target - v) * warmth).clamp(0.0, 255.0) as u8
    };
    Color::Rgb(chan(r, 255.0), chan(g, 150.0), chan(b, 60.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::buffer::Cell;

    /// The glow pass (the part no TTY can show offline): run one frame over a flat buffer with the fire
    /// center at the hearth, and confirm it (a) modulates cells AT the hearth and (b) DARKENS the far
    /// corners (the vignette). Guards against the pass silently no-op'ing — and, since it now covers the
    /// whole stage, that the absolute-center math lights the right spot regardless of the area origin.
    #[test]
    fn glow_lights_the_hearth_and_vignettes_the_corners() {
        let area = Rect::new(0, 0, 100, 24);
        let base = Color::Rgb(90, 90, 90);
        let mut seed = Cell::default();
        seed.set_fg(base).set_bg(base);
        let mut buf = Buffer::filled(area, seed);

        // Fire center at the hearth: scene-pixel (50,33) → cell (50,16) in absolute buffer coords.
        let (fcx, fcy) = (GLOW_PX_X, GLOW_PX_Y / 2.0);
        apply_glow(&mut buf, area, fcx, fcy, 0.0);

        let near = buf[(fcx as u16, fcy as u16)].clone();
        assert_ne!(near.bg, base, "glow must modulate cells at the hearth");
        // A far corner is well beyond VIG_R0 → the vignette must dim it (darker than the base gray).
        let far = buf[(99, 0)].clone();
        let far_lum = far.bg.to_rgb().0;
        assert!(far_lum < 90, "vignette must darken the far corner (got {far_lum}, base 90)");
    }
}
