//! Reusable half-block sprite widget — blit a sprite-grid JSON into a ratatui `Buffer`.
//!
//! THE CORE TECHNIQUE (mirrors `tui-sprite-art/scripts/halfblock_render.py`): a terminal cell shows
//! ONE glyph with ONE fg + ONE bg. Print U+2580 ▀ (UPPER HALF BLOCK) and the foreground paints the
//! TOP half of the cell, the background the BOTTOM half — two independently colored, ~square pixels
//! stacked in one character cell. To draw a bitmap we walk the pixel grid two rows at a time:
//!   pixel row 2N   -> fg (top)
//!   pixel row 2N+1 -> bg (bottom)
//! A `null` palette entry is transparent: that half keeps the terminal background. The four
//! (top, bottom) transparency cases match the Python encoder exactly:
//!   (None, None)     -> leave the cell untouched (transparent)
//!   (Some, None)     -> ▀ with fg = top                 (bottom transparent)
//!   (None, Some)     -> ▄ (LOWER HALF BLOCK) with fg = bottom (top transparent)
//!   (Some, Some)     -> ▀ with fg = top, bg = bottom
//!
//! The grid is the SPRITE authoring format: `{"palette": {"<char>": [r,g,b] | null}, "rows": [...]}`.
//! Each character in a row indexes the palette; all rows must be the same length. (The RGB
//! `{width,height,pixels}` format the image importer emits is not needed here; we only ship sprite
//! assets.) Loaded with `serde_json::Value` — no `serde` derive dependency required.

use anyhow::{anyhow, Context, Result};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::widgets::Widget;

/// One pixel: `None` = transparent, `Some((r,g,b))` = an opaque color.
pub type Px = Option<(u8, u8, u8)>;

/// A loaded sprite grid: `w` x `h` pixels, row-major in `px`.
pub struct Sprite {
    pub w: u16,
    pub h: u16,
    px: Vec<Px>,
}

impl Sprite {
    /// Parse a sprite-grid JSON string (the `{palette, rows}` authoring format) into a `Sprite`.
    /// Mirrors `halfblock_render.py::parse_grid`: each row char indexes the palette; a `null`
    /// palette value is transparent; all rows must share the first row's width.
    pub fn from_json(text: &str) -> Result<Sprite> {
        let v: serde_json::Value =
            serde_json::from_str(text).context("sprite JSON is not valid JSON")?;
        Sprite::from_value(&v)
    }

    /// Parse one already-decoded `{palette, rows}` grid object into a `Sprite`. Factored out of
    /// `from_json` so `Anim::from_json` can reuse it per frame (frames share this exact grid shape).
    pub fn from_value(v: &serde_json::Value) -> Result<Sprite> {
        let palette_obj = v
            .get("palette")
            .and_then(|p| p.as_object())
            .ok_or_else(|| anyhow!("sprite JSON missing object field `palette`"))?;
        // Resolve each palette entry to a Px once, up front.
        let mut palette: std::collections::HashMap<char, Px> = std::collections::HashMap::new();
        for (k, val) in palette_obj {
            let mut chars = k.chars();
            let ch = chars
                .next()
                .ok_or_else(|| anyhow!("empty palette key in sprite JSON"))?;
            if chars.next().is_some() {
                return Err(anyhow!("palette key {k:?} must be a single character"));
            }
            palette.insert(ch, coerce_pixel(val)?);
        }

        let rows = v
            .get("rows")
            .and_then(|r| r.as_array())
            .ok_or_else(|| anyhow!("sprite JSON missing array field `rows`"))?;
        if rows.is_empty() {
            return Err(anyhow!("sprite JSON has empty `rows`"));
        }

        let mut grid: Vec<Px> = Vec::new();
        let mut width: Option<usize> = None;
        for (y, row) in rows.iter().enumerate() {
            let s = row
                .as_str()
                .ok_or_else(|| anyhow!("row {y} is not a string"))?;
            let row_chars: Vec<char> = s.chars().collect();
            match width {
                None => width = Some(row_chars.len()),
                Some(w) if w != row_chars.len() => {
                    return Err(anyhow!(
                        "row {y} has width {} (expected {w}; all rows must match)",
                        row_chars.len()
                    ));
                }
                _ => {}
            }
            for ch in row_chars {
                let px = *palette
                    .get(&ch)
                    .ok_or_else(|| anyhow!("row {y} uses undefined palette char {ch:?}"))?;
                grid.push(px);
            }
        }

        let w = width.unwrap_or(0) as u16;
        let h = rows.len() as u16;
        Ok(Sprite { w, h, px: grid })
    }

    #[inline]
    fn get(&self, x: u16, y: u16) -> Px {
        if x < self.w && y < self.h {
            self.px[y as usize * self.w as usize + x as usize]
        } else {
            None
        }
    }

    /// Cell footprint: `w` cells wide, `ceil(h/2)` cell-rows tall (two pixel rows per cell).
    pub fn cell_size(&self) -> (u16, u16) {
        (self.w, self.h.div_ceil(2))
    }
}

/// A loaded frame animation — the sprite skill's `{fps, loop, frames: [<grid>, ...]}` format, where
/// each frame is a full `{palette, rows}` grid. Used for the fireplace flame overlay (`fire.anim.json`):
/// `present_scene` advances `frames` at `fps` and blits the current one over the (dark) firebox.
pub struct Anim {
    pub fps: f32,
    pub frames: Vec<Sprite>,
}

impl Anim {
    /// Parse a `{fps, loop, frames: [...]}` animation JSON. `fps` defaults to 8 if absent; `loop` is
    /// ignored here (the caller cycles). Each frame is parsed via `Sprite::from_value`.
    pub fn from_json(text: &str) -> Result<Anim> {
        let v: serde_json::Value =
            serde_json::from_str(text).context("animation JSON is not valid JSON")?;
        let fps = v.get("fps").and_then(|f| f.as_f64()).unwrap_or(8.0) as f32;
        let frames_val = v
            .get("frames")
            .and_then(|f| f.as_array())
            .ok_or_else(|| anyhow!("animation JSON missing array field `frames`"))?;
        if frames_val.is_empty() {
            return Err(anyhow!("animation JSON has empty `frames`"));
        }
        let frames = frames_val
            .iter()
            .map(Sprite::from_value)
            .collect::<Result<Vec<_>>>()?;
        Ok(Anim { fps, frames })
    }
}

/// Coerce one palette value (`null` or `[r,g,b]`) into a `Px`. Mirrors `_coerce_pixel`.
fn coerce_pixel(val: &serde_json::Value) -> Result<Px> {
    if val.is_null() {
        return Ok(None);
    }
    let arr = val
        .as_array()
        .ok_or_else(|| anyhow!("palette color must be null or [r,g,b], got {val}"))?;
    if arr.len() != 3 {
        return Err(anyhow!("palette color must have 3 components, got {val}"));
    }
    let comp = |i: usize| -> Result<u8> {
        let n = arr[i]
            .as_i64()
            .ok_or_else(|| anyhow!("palette color component must be an integer, got {val}"))?;
        Ok(n.clamp(0, 255) as u8)
    };
    Ok(Some((comp(0)?, comp(1)?, comp(2)?)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped study asset parses and has the documented footprint (100 px × 48 px = 100 cells
    /// wide × 24 cell-rows tall). This pins the encoding the scene presenter depends on.
    #[test]
    fn study_asset_parses_to_expected_footprint() {
        let json = include_str!("../assets/scene/study.json");
        let s = Sprite::from_json(json).expect("study.json must parse");
        assert_eq!((s.w, s.h), (100, 48));
        assert_eq!(s.cell_size(), (100, 24)); // ceil(48/2)
    }

    /// The shipped flame animation parses into its frames at the firebox footprint (9 px × 12 px =
    /// 9 cells × 6 cell-rows). This pins the overlay `present_scene` blits over the dark firebox.
    #[test]
    fn fire_anim_parses_to_expected_footprint() {
        let json = include_str!("../assets/scene/fire.anim.json");
        let a = Anim::from_json(json).expect("fire.anim.json must parse");
        assert!(a.frames.len() >= 2, "flame must have multiple frames to flicker");
        for f in &a.frames {
            assert_eq!((f.w, f.h), (9, 12));
            assert_eq!(f.cell_size(), (9, 6));
        }
    }

    /// Transparency + half-block pairing: a `null` palette entry yields a transparent (None) pixel;
    /// an `[r,g,b]` entry yields the exact opaque color; row chars index the palette in order.
    #[test]
    fn parses_palette_transparency_and_pairing() {
        let json = r#"{
            "palette": {".": null, "R": [255, 0, 0], "G": [0, 128, 0]},
            "rows": [".R", "G."]
        }"#;
        let s = Sprite::from_json(json).unwrap();
        assert_eq!((s.w, s.h), (2, 2));
        assert_eq!(s.get(0, 0), None); // '.' transparent
        assert_eq!(s.get(1, 0), Some((255, 0, 0))); // 'R'
        assert_eq!(s.get(0, 1), Some((0, 128, 0))); // 'G'
        assert_eq!(s.get(1, 1), None); // '.'
    }

    #[test]
    fn rejects_ragged_rows() {
        let json = r#"{"palette": {".": null}, "rows": [".", ".."]}"#;
        assert!(Sprite::from_json(json).is_err());
    }

    #[test]
    fn rejects_undefined_palette_char() {
        let json = r#"{"palette": {".": null}, "rows": ["X"]}"#;
        assert!(Sprite::from_json(json).is_err());
    }
}

impl Widget for &Sprite {
    /// Blit the sprite into `area`, top-left aligned, clipped to `area`. Transparent cells are left
    /// untouched so whatever was drawn underneath (the cleared background) shows through.
    fn render(self, area: Rect, buf: &mut Buffer) {
        const UPPER: &str = "\u{2580}"; // ▀ fg=top, bg=bottom
        const LOWER: &str = "\u{2584}"; // ▄ fg=bottom (top transparent)
        let cell_rows = self.h.div_ceil(2);
        for cy in 0..cell_rows {
            let ty = cy * 2;
            let by = cy * 2 + 1;
            for x in 0..self.w {
                let sx = area.x + x;
                let sy = area.y + cy;
                if sx >= area.right() || sy >= area.bottom() {
                    continue;
                }
                let top = self.get(x, ty);
                let bot = self.get(x, by);
                let Some(cell) = buf.cell_mut((sx, sy)) else {
                    continue;
                };
                match (top, bot) {
                    (None, None) => { /* transparent: leave the underlying cell as-is */ }
                    (Some(c), None) => {
                        cell.set_symbol(UPPER);
                        cell.set_fg(Color::Rgb(c.0, c.1, c.2));
                    }
                    (None, Some(c)) => {
                        cell.set_symbol(LOWER);
                        cell.set_fg(Color::Rgb(c.0, c.1, c.2));
                    }
                    (Some(t), Some(b)) => {
                        cell.set_symbol(UPPER);
                        cell.set_fg(Color::Rgb(t.0, t.1, t.2));
                        cell.set_bg(Color::Rgb(b.0, b.1, b.2));
                    }
                }
            }
        }
    }
}
