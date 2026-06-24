# SCENE_HANDOFF.md — `present --scene` pixel-art diorama (state + how to iterate)

The `present --scene` skin renders the observer as a tiny chibi gent reading by the fire in a cozy
pixel-art study, with each interjection as a speech bubble. **Done and working**; actively being
polished. The fireplace is now **alive**: an animated 4-frame flame + a continuous **firelight glow
pass** pulse warm light over the room (now across the whole stage, so the room blends into the
surrounding shadow rather than floating as a lit box). This doc is so a fresh session can keep iterating.

## What exists (file map)

- **`src/present_scene.rs`** — the scene TUI (a skin over `present`). Reuses `present`'s base: the same
  fused `lobe.step()` loop + `present::draw_prose` (top prose pane) + `present::draw_footer`. The
  MIDDLE region (where plain `present` shows the asides feed) instead stages the scene sprite
  (bottom-centered) + a rounded **speech bubble** ABOVE it (tail points down at `HEAD_COL`). The scene
  art is `include_str!("../assets/scene/study.json")`. **NEW:** it also blits the animated flame
  (`fire.anim.json`) over the firebox each tick and runs the glow pass (`apply_glow`/`warm`) over the
  full stage.
- **`src/sprite.rs`** — reusable half-block sprite widget: parses `{palette, rows}` JSON, blits into a
  ratatui `Buffer` as `▀` cells (top px → fg, bottom px → bg; `null` = transparent). **NEW:** `Anim`
  loads the skill's `{fps, loop, frames:[grid…]}` format (used for the flame). Has unit tests.
- **`assets/scene/study.py`** — the GENERATOR. Programmatically draws the whole scene (walls, floor,
  bookshelves, terracotta fireplace, wingback chair, lamp+table, rug, pictures, the warm radial
  glow vignette, and Werner) and writes `study.json`. **Edit the art here, not the JSON.** NB the
  firebox interior is left **dark** — the fire is the animated overlay, not baked here.
- **`assets/scene/fire.py`** — the FLAME GENERATOR. Parametric 4-frame flame (varying tip height,
  lean, white-hot core) sized to the firebox opening (9×12 px); writes `fire.anim.json`. **Edit the
  flame here.**
- **`assets/scene/study.json`** — the generated scene grid (100×48 px = 100 cells × 24 cell-rows).
- **`assets/scene/fire.anim.json`** — the generated 4-frame flame (`{fps:8, loop, frames}`).
- **`assets/scene/study.png`** — rasterized reference of the current scene.
- **`assets/scene/werner.py`** — standalone bobble-Werner sprite (older design reference; the live copy
  is inlined in `study.py`'s "WERNER" block at the end, now a fuller chibi — see below).
- **`personas/*.txt`** — voice presets (`herzog_varyform.txt` is the shipped voice; see below).

## The art design loop (DO THIS to iterate the art)

The rasterizer needs Pillow, which lived in a `/tmp` venv (gone after a reboot). Recreate it:
```bash
python3 -m venv /tmp/artvenv && /tmp/artvenv/bin/pip install Pillow
```
Then the loop: edit `assets/scene/study.py` → regenerate + rasterize → LOOK at the PNG → repeat:
```bash
python3 assets/scene/study.py
/tmp/artvenv/bin/python ~/.claude/skills/tui-sprite-art/scripts/rasterize.py assets/scene/study.json -o /tmp/study.png
# then Read /tmp/study.png to view it
```
For a single sprite at high zoom (e.g. tweaking Werner): rasterize a standalone JSON with `--scale 36`
(or crop a sub-rect of `study.json` first — see the `werner_crop` snippet pattern in session history).
For the FLAME: `python3 assets/scene/fire.py` then rasterize `fire.anim.json` with `--strip` to see all
4 frames side by side. **The static `study.json` firebox is dark** (the flame is a runtime overlay), so
to preview the fire IN CONTEXT, composite a flame frame onto the scene at scene-pixel (46,24) and
rasterize that — that's the only way to eyeball flame placement without a live TTY.
The `tui-sprite-art` skill (`~/.claude/skills/tui-sprite-art/`) is the authority on the format/aesthetic.

**CRITICAL:** `study.json` is `include_str!`'d at COMPILE time. The rasterize loop above does NOT need
a rebuild (it reads the JSON directly). But to see art changes in the LIVE TUI you must
`cargo build --release --features metal`. Also: if Werner moves, update `HEAD_COL` in `present_scene.rs`
(the speech-bubble tail column) to his new head column.

## Current design (as of this handoff)

- **Werner:** a fuller **chibi** (~11 px tall, 9 wide; head:body ≈ 6:5) — big round bald head, grey-blond
  hair at the temples/ears (color `A` = muted `[158,151,138]`, NOT near-white — that read as earmuffs),
  clean-shaven, **big eyes** (white upper = catch-light, dark pupil below, looking down at the book),
  small mouth, green smoking jacket + light collar, hands holding an **open book** (cream pages / red
  cover), little shoes. Standing in front of his (empty, kept) wingback. Centered at scene col 64;
  `HEAD_COL = 64`. Redraw lives in `study.py`'s "WERNER" block.
- **Fireplace:** clean stylized terracotta (#13-style): cream mantel + base shelves (dark-outlined,
  overhanging), terracotta body, cream-framed **arched** opening, dark interior. The fire is an
  **animated 4-frame flame** (`fire.py`) blitted over the (dark) firebox at scene-pixel (46,24), looping
  at 8 fps — chunky triangular flame, red edges → orange → yellow → white-hot core, ember/log base, with
  per-frame flicker (tip height + lean + core). The flame is kept **short** (raised `tip_row` in
  `FRAMES`) so there's clear dark headroom at the top of the firebox.
- **Glow pass:** `apply_glow()` in `present_scene.rs` — a direct per-frame pass (was a tachyonfx
  `fx::effect_fn`; switched to a manual pass so it can take the fire center in ABSOLUTE buffer coords)
  that re-lights every cell by distance from the fire center (scene-pixel (50,33)). Two zones: a NEAR
  **glow** (brighten + warm + a dramatic layered-sine flicker, reach `GLOW_R`) and a far **vignette**
  (dim toward the corners so the room recedes into shadow, `VIG_R0`→`VIG_R1`, up to `VIG_MAX_DIM`).
  Driven by wall-clock `t` (the same `anim_start` that drives the "musing…" ellipsis); runs even while
  paused. **Applied over the WHOLE stage region, not just the sprite rect** — so the room's walls
  dissolve into the surrounding shadow as one continuous firelit field, instead of reading as a lit
  rectangle floating on a flat backdrop (the old confined-to-sprite glow was the cause of that seam).
  This is ON TOP of the baked static vignette in `study.py`. **Tunables are named consts** at the top
  of `present_scene.rs`:
  `GLOW_R 26`, `GLOW_AMP 0.30` (flicker drama), `VIG_R0 20`, `VIG_R1 58`, `VIG_MAX_DIM 0.55` (edge
  darkness); the three sine freqs/weights are in `apply_glow`. To preview the lit look offline (no TTY),
  replicate the `apply_glow`/`warm` math per-pixel over the composited scene (the cell-space distance ==
  pixel distance from fire-pixel (50,33)) and render with Pillow — see session history.
- **Room:** wide cozy study — bookshelves both walls, lamp+side table (left), rug, framed pictures,
  warm radial fire-glow vignette (`FX,FY,R1,R2` in study.py; brightens wall/floor near the fire).

## Open polish (not yet addressed)
- The **rug** is mostly hidden by the hearth lip; **floor** is a thin strip — could add depth.
- The **left side** is emptier than the right.
- Speech-bubble placement/tail aim is tuned to constants — needs a TTY eyeball.
- The **glow shader amplitude/warmth** were set by reasoning, not a live eyeball — may want tuning on a
  real terminal (could expose as flags). Consider syncing the glow pulse tighter to the flame frame.
- Possible: round the fireplace arch more; light spill from the lamp as a second (cooler) glow.

## Build / run
```bash
cargo build --release --features metal
# the showcase (voice + scene):
./target/release/streaming-lobe --model models/gemma-4-E4B_q4_0-it.gguf \
  --z 4.1 --refractory 320 --frame --interject-max 120 --dedup 0.5 --interject-temp 0.6 \
  --preamble-file personas/herzog_varyform.txt \
  present --scene --input corpus/pg2701.txt --skip-to "Call me Ishmael" --tick-ms 30
```
Needs a real, wide, truecolor terminal. Drop `--scene` for the plain clean `present` view.
