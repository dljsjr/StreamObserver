import json, math

# Animated fireplace flame — a 4-frame loop blitted OVER the (dark) firebox of study.json each tick.
# Sized to the firebox opening: 9 px wide x 12 px tall (cols 0-8 -> scene x46..54; rows 0-11 ->
# scene y24..35). Top-left anchors at scene pixel (46,24) — an EVEN row, so it cell-aligns cleanly.
# Transparent (".") cells let the dark firebox show through. Parametric so the flicker is natural:
# each frame varies tip height, a small lean, and the white-hot core. See SCENE_HANDOFF.md.
W, H = 9, 12
PAL = {
 ".": None,
 "e": [150, 40, 18],    # deep-red outer edge
 "R": [212, 72, 30],    # red flame body
 "O": [246, 152, 52],   # orange
 "Y": [255, 226, 142],  # yellow
 "C": [255, 250, 214],  # white-hot core
 "m": [196, 74, 30],    # glowing ember
 "M": [248, 170, 70],   # bright ember
 "g": [78, 46, 36],     # charred log
}
CX = 4.0          # flame centerline column
BASE = H - 1      # bottom pixel row (embers sit here)

# Per-frame flicker params: (tip_row, lean, base_half_width, core_boost)
#  tip_row     = how high the flame reaches (smaller = taller)
#  lean        = horizontal drift of the tip (sway)
#  base_half   = half-width of the flame at its base
#  core_boost  = how much white-hot core shows this frame
# tip_row raised vs. the first cut → a SHORTER flame that leaves headroom at the top of the firebox.
FRAMES = [
    (5, 0.3, 3.7, 1.0),
    (4, 1.1, 3.9, 1.3),
    (6, -0.5, 3.4, 0.7),
    (5, -1.1, 3.8, 1.1),
]


def build(tip_row, lean, base_half, core_boost):
    g = [["." for _ in range(W)] for _ in range(H)]
    span = max(1.0, BASE - tip_row)
    for y in range(H):
        # embers / logs along the very bottom two rows
        if y >= BASE - 1:
            for x in range(W):
                d = abs(x - CX)
                if y == BASE:
                    g[y][x] = "g" if d > 2.6 else ("m" if d > 1.2 else "M")
                else:  # BASE-1: low glow over the logs
                    if d <= base_half - 0.4:
                        g[y][x] = "M" if d < 1.0 else "m"
            continue
        if y < tip_row:
            continue
        t = (y - tip_row) / span          # 0 at tip .. 1 at base
        # centerline sways toward the tip; flame is wide at base, narrow at tip.
        # sqrt profile keeps the body chunky and rounds the tip (vs. a thin spike).
        center = CX + lean * (1.0 - t)
        half = max(0.7, base_half * math.sqrt(0.12 + 0.88 * t))
        for x in range(W):
            r = abs(x - center) / half     # 0 center .. 1 edge
            if r > 1.0:
                continue
            if r > 0.92:
                c = "e"
            elif r > 0.62:
                c = "R"
            elif r > 0.32:
                c = "O"
            else:
                c = "Y"
            # white-hot core: lower-middle of the flame, tight to the centerline
            if r < 0.24 and 0.5 < t < 0.85 and core_boost > 0.8:
                c = "C"
            g[y][x] = c
    return ["".join(r) for r in g]


frames = [{"palette": PAL, "rows": build(*p)} for p in FRAMES]
out = {"fps": 8, "loop": True, "frames": frames}
json.dump(out, open(__file__.replace("fire.py", "fire.anim.json"), "w"))
print("wrote fire.anim.json", W, "x", H, "x", len(frames), "frames")
