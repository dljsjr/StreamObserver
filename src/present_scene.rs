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
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::{Block, BorderType, Borders, Padding, Paragraph, Wrap};
use ratatui::Frame;
use std::time::{Duration, Instant};
use tachyonfx::{fx, Duration as FxDuration, Effect, EffectRenderer, ToRgbComponents};

use crate::lobe::Lobe;
use crate::sprite::{Anim, Sprite};
use crate::stats::Welford;
use crate::Cli;

const PROSE_TAIL_CHARS: usize = 12_000; // bound display memory (the lobe keeps its own context)

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

pub fn run(
    lobe: &mut Lobe,
    cli: &Cli,
    input_path: &str,
    tick_ms: u64,
    skip_to: &str,
    retrieve: &mut crate::retrieval::RetrieveFn,
) -> Result<()> {
    // Parse the scene + flame animation once, up front, so a malformed asset fails loudly before
    // entering raw mode.
    let scene = Sprite::from_json(STUDY_JSON)?;
    let fire = Anim::from_json(FIRE_JSON)?;

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
    let mut z = cli.z; // pre-tuned; +/- still adjusts silently
    let title = std::path::Path::new(input_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("stream")
        .to_string();

    // Display state — same shape as present.rs.
    let mut prose = String::new();
    let mut last_aside: Option<String> = None;
    let mut pending: Option<String> = None;
    let mut revealed = false;
    let mut paused = false;
    let mut done = false;
    let mut last_tick = Instant::now();
    let tick = Duration::from_millis(tick_ms);

    // Fireplace animation state: the flame loops at its own fps independent of the stream tick, and
    // the glow shader is a continuous tachyonfx effect advanced by per-frame wall-clock delta. Both
    // keep crackling even while paused — a fire doesn't stop when you stop reading.
    let mut fire_idx = 0usize;
    let mut last_fire = Instant::now();
    let fire_period = Duration::from_secs_f32(1.0 / fire.fps.max(1.0));
    let mut glow = make_glow();
    let mut last_draw = Instant::now();

    let mut terminal = ratatui::init();
    let res = (|| -> Result<()> {
        loop {
            if last_fire.elapsed() >= fire_period {
                fire_idx = (fire_idx + 1) % fire.frames.len();
                last_fire = Instant::now();
            }
            let glow_dt: FxDuration = last_draw.elapsed().into();
            last_draw = Instant::now();
            let flame = &fire.frames[fire_idx];
            terminal.draw(|f| {
                draw(f, &scene, flame, &mut glow, glow_dt, &title, &prose,
                     last_aside.as_deref(), pending.as_deref(), paused, done)
            })?;

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

            // One fused step per tick — COPIED from present.rs (identical pending/revealed/dedup logic);
            // only the display differs.
            if let Some(tok) = feed.next() {
                // One fused step per tick: observe + advance any in-flight aside in one decode. With
                // --rag, step() retrieves on a fire and weaves the recall into the aside IN VOICE
                // (the aside still streams concurrently; only the query embed + ask-prefill stall).
                let (s, status) = if interject {
                    let out = lobe.step(tok, &mut stats, z, cli.topk, interject_max, retrieve)?;
                    (out.step, Some(out.interjection))
                } else {
                    (lobe.observe(tok, &mut stats, z, cli.topk)?, None)
                };

                prose.push_str(&s.token_text);
                if prose.len() > PROSE_TAIL_CHARS {
                    let want = prose.len() - PROSE_TAIL_CHARS;
                    let cut = (want..prose.len())
                        .find(|&i| prose.is_char_boundary(i))
                        .unwrap_or(prose.len());
                    prose.drain(0..cut);
                }

                // The dedup/reveal policy lives in the lobe; we just store whatever survives.
                if let Some(text) = lobe.advance_reveal(status, &mut pending, &mut revealed)? {
                    last_aside = Some(text);
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

/// Same base layout as `present` (prose pane on top, footer at the bottom); the middle region —
/// where plain `present` shows the asides feed — instead stages the scene + speech bubble.
#[allow(clippy::too_many_arguments)]
fn draw(
    f: &mut Frame,
    scene: &Sprite,
    flame: &Sprite,
    glow: &mut Effect,
    glow_dt: FxDuration,
    title: &str,
    prose: &str,
    last_aside: Option<&str>,
    pending: Option<&str>,
    paused: bool,
    done: bool,
) {
    // Paint the whole frame the room's deep-shadow tone so transparent sprite cells + margins read as
    // one continuous dark study, not the terminal default.
    let backdrop = Block::default().style(Style::default().bg(Color::Rgb(28, 22, 21)));
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
    draw_stage(f, outer[1], scene, flame, glow, glow_dt, last_aside, pending);
    crate::present::draw_footer(f, outer[2], paused, done);
}

/// The stage: sprite anchored bottom-center; the current aside as a speech bubble in the open band
/// ABOVE the sprite (never over it), with a short tail pointing down at the gent's head.
#[allow(clippy::too_many_arguments)]
fn draw_stage(
    f: &mut Frame,
    area: Rect,
    scene: &Sprite,
    flame: &Sprite,
    glow: &mut Effect,
    glow_dt: FxDuration,
    last_aside: Option<&str>,
    pending: Option<&str>,
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

    // Fireplace glow: a continuous tachyonfx shader pulsing warm light over the scene, falling off
    // with distance from the fire. Applied AFTER the scene + flame are drawn (it modulates the
    // already-painted cells), and scoped to the scene rect so it never touches the bubble/prose.
    f.render_effect(glow, sprite_rect, glow_dt);

    // The bubble lives in the band ABOVE the sprite (never over it).
    let band = Rect {
        x: area.x,
        y: area.y,
        width: area.width,
        height: sprite_y.saturating_sub(area.y),
    };
    draw_bubble(f, band, sprite_rect, last_aside, pending);
}

/// A rounded speech bubble in `band` (the open space above the sprite), holding the CURRENT aside —
/// the in-flight one (warm/bold, with a caret) while generating, else the most recent settled one.
/// Bottom-anchored just above the sprite, with a short downward tail stub in the gap pointing at the
/// gent (so the tail never has to draw over the art).
fn draw_bubble(f: &mut Frame, band: Rect, sprite_rect: Rect, last_aside: Option<&str>, pending: Option<&str>) {
    let (text, forming): (String, bool) = match pending {
        Some(p) if !p.is_empty() => (format!("{p}▍"), true),
        Some(_) => ("musing…".to_string(), true), // buffering the opening (caret-only phase)
        None => match last_aside {
            Some(a) if !a.is_empty() => (a.to_string(), false),
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

    let (border_color, text_style) = if forming {
        (Color::Yellow, Style::default().fg(Color::Rgb(255, 244, 214)).add_modifier(Modifier::BOLD))
    } else {
        (Color::Rgb(214, 214, 208), Style::default().fg(Color::Rgb(232, 230, 224)))
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

/// Build the continuous fireplace-glow shader. A tachyonfx `effect_fn` that, every frame, warms and
/// pulses each cell by an amount that falls off with distance from the fire. The flicker is driven by
/// an `Instant` state (wall-clock continuous, independent of the effect timer), over a timer so long
/// it never "completes" within a session. The effect's `area` is the scene's render rect, so the fire
/// center is derived from it each frame → correct after a terminal resize.
fn make_glow() -> Effect {
    let start = Instant::now();
    fx::effect_fn(start, FxDuration::from_millis(u32::MAX), |start, ctx, cells| {
        let t = start.elapsed().as_secs_f32();
        let area = ctx.area;
        let fcx = area.x as f32 + GLOW_PX_X;
        let fcy = area.y as f32 + GLOW_PX_Y / 2.0; // scene pixel row → cell row
        // Layered sines → an organic flicker in roughly [-1, 1]: a fast crackle over a slower swell.
        let flick = 0.50 * (t * 11.0).sin() + 0.30 * (t * 19.0 + 1.3).sin() + 0.20 * (t * 6.1 + 2.1).sin();
        for (pos, cell) in cells {
            let dx = pos.x as f32 - fcx;
            let dy = (pos.y as f32 - fcy) * 2.0; // cells are ~2× tall → keep the falloff circular
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
    })
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
    use ratatui::buffer::{Buffer, Cell};

    /// End-to-end check of the tachyonfx glow wiring (the part no TTY can show offline): build the
    /// effect, process one frame over a flat buffer, and confirm it (a) modulates cells AT the hearth
    /// and (b) DARKENS the far corners (the vignette). Guards against the effect silently no-op'ing.
    #[test]
    fn glow_lights_the_hearth_and_vignettes_the_corners() {
        let area = Rect::new(0, 0, 100, 24);
        let base = Color::Rgb(90, 90, 90);
        let mut seed = Cell::default();
        seed.set_fg(base).set_bg(base);
        let mut buf = Buffer::filled(area, seed);

        let mut glow = make_glow();
        buf.render_effect(&mut glow, area, FxDuration::from_millis(16));

        // The fire center is scene-pixel (50,33) → cell (50,16); that cell must be modulated.
        let near = buf[(GLOW_PX_X as u16, (GLOW_PX_Y / 2.0) as u16)].clone();
        assert_ne!(near.bg, base, "glow must modulate cells at the hearth");
        // A far corner is well beyond VIG_R0 → the vignette must dim it (darker than the base gray).
        let far = buf[(99, 0)].clone();
        let far_lum = far.bg.to_rgb().0;
        assert!(far_lum < 90, "vignette must darken the far corner (got {far_lum}, base 90)");
    }
}
