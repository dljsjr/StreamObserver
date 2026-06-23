import json, math
# Wide, zoomed-OUT cozy study: central brick fireplace, wingback chair, flanking bookshelves, a
# lamp+side table, a rug, framed pictures — and the gent as a SMALL NES-style sprite in the chair.
# A warm radial fire-glow vignette (bright near the hearth, dark at the edges) sets the mood.
W, H = 100, 48
PAL = {
 # ambient surfaces have dark/mid/bright variants; the glow pass picks one by distance from the fire.
 "w": [34,26,22], "W": [52,38,30], "V": [82,58,40],     # wall  dark / mid / bright(fire-lit)
 "f": [40,28,22], "F": [60,42,30], "G": [98,66,42],     # floor dark / mid / bright
 "b": [74,44,30], "B": [116,72,46], "N": [160,104,64],  # brick dark / mid / bright
 "r": [48,32,24],                                       # mortar/recess
 "M": [96,62,36], "m": [124,84,50],                     # mantel wood / highlight
 "k": [40,28,24],                                       # firebox interior (dark brown)
 "Q": [216,186,146], "T": [86,52,34], "z": [192,112,84], "Z": [150,80,58], # fireplace: cream shelf / dark outline / terracotta / shade
 "e": [122,36,18], "R": [212,72,30], "O": [246,152,52], "Y": [255,226,142], "C": [255,250,214], # fire
 "L": [255,222,150], "l": [150,108,58], "p": [70,48,30],# lamp shade / glow halo / base
 "t": [82,54,32], "s": [70,46,28], "S": [46,30,20],     # table / shelf wood / shelf dark
 "1": [150,54,48], "2": [58,98,96], "3": [184,148,62], "4": [66,74,118], "5": [150,140,120], # books
 "h": [108,54,48], "H": [144,80,62], "d": [72,38,38],   # chair mid / highlight / shadow
 "g": [122,56,46], "q": [156,84,52], "a": [88,42,34],   # rug / pattern / dark
 "i": [120,82,48], "I": [58,46,40],                     # picture frame / inner
 "K": [232,184,152], "A": [158,151,138], "J": [46,122,86], "j": [30,90,62], "y": [88,158,118], "n": [46,44,52], "E": [248,248,246], "P": [232,226,206], # WERNER skin/hair/jacket/.../eye-white/book-pages
 "x": [16,12,10],
}
g = [["W" for _ in range(W)] for _ in range(H)]
def rect(x0,y0,x1,y1,c):
    for y in range(max(0,y0),min(H,y1+1)):
        for x in range(max(0,x0),min(W,x1+1)):
            g[y][x]=c
def px(x,y,c):
    if 0<=x<W and 0<=y<H: g[y][x]=c

FLOOR=40
rect(0,FLOOR,99,47,"F")

# --- bookshelves (both walls) ---
def shelf(x0,x1):
    rect(x0,3,x1,39,"S")                      # case
    rect(x0+1,4,x1-1,38,"s")                  # back
    cols="1234521534215342"
    for sy in range(5,38,6):                  # shelves with books
        rect(x0+1,sy+5,x1-1,sy+5,"S")
        for x in range(x0+2,x1-1):
            c=cols[(x+sy)%len(cols)]
            px(x,sy+ (1 if x%3 else 2),c)
            rect(x,sy+ (1 if x%3 else 2),x,sy+4,c)
shelf(2,20)
shelf(80,98)

# --- fireplace (#13-style: cream mantel/base shelves, terracotta body, cream-framed arched opening) ---
rect(42,17,58,37,"z")                          # terracotta body
rect(42,17,43,37,"Z"); rect(57,17,58,37,"Z")   # darker side shading
# mantel shelf (cream, dark-outlined, overhangs)
rect(39,14,61,14,"T"); rect(39,15,61,16,"Q"); rect(40,17,60,17,"T")
# arched opening: dark interior with a cream frame, rounded top
rect(45,22,55,37,"Q")                          # cream frame block
rect(46,24,54,36,"k")                          # dark interior
rect(47,23,53,23,"k")                          # arch shoulder
px(45,22,"z"); px(55,22,"z"); px(45,23,"z"); px(55,23,"z")   # round the top corners
# base shelf (cream, dark-outlined, overhangs) + little feet
rect(40,38,60,38,"T"); rect(39,39,61,40,"Q")
rect(40,41,42,41,"T"); rect(58,41,60,41,"T")

# --- framed pictures above the mantel ---
for fx in (40,53):
    rect(fx,4,fx+7,10,"i"); rect(fx+1,5,fx+6,9,"I")

# --- side table + lamp (left of the fire) ---
rect(23,35,32,36,"t"); rect(24,37,25,40,"S"); rect(30,37,31,40,"S")  # table + legs
rect(27,31,28,34,"p")                         # lamp stem
rect(25,27,30,30,"L"); rect(26,26,29,26,"L")  # shade (glows), smaller + softer
for yy in range(23,40):                        # soft lamp glow halo
    for xx in range(20,36):
        if g[yy][xx]=="W" and (xx-27.5)**2+((yy-29)*1.5)**2 < 50: g[yy][xx]="l"

# --- wingback chair (right of the fire, FRONT-FACING so the silhouette reads; gent sits in it) ---
rect(63,25,75,37,"d")                          # back panel — DARK so it recedes behind the gent
rect(60,23,64,31,"h")                          # left wing ("ear")
rect(74,23,78,31,"h")                          # right wing
rect(60,31,64,40,"h")                          # left arm
rect(74,31,78,40,"h")                          # right arm
rect(60,31,64,31,"H"); rect(74,31,78,31,"H")   # rolled arm-tops (highlight)
rect(63,37,75,40,"H")                          # seat cushion — LIGHT, distinct, in front
rect(60,23,60,40,"H")                          # fire-lit (left) outer edge
rect(77,23,78,40,"d")                          # shadow (right) outer edge
rect(60,40,78,41,"d")                          # base shadow on the floor

# (chair left empty for now — standing Werner is drawn last, on top of everything)

# --- rug on the floor in front of the hearth ---
rect(28,42,72,46,"g"); rect(28,42,72,42,"a"); rect(28,46,72,46,"a")
for x in range(31,72,5): rect(x,43,x,45,"q")

# NB: the fire itself is NOT baked here — it's an ANIMATED overlay (assets/scene/fire.anim.json,
# generated by fire.py) blitted over this dark firebox each tick by present_scene.rs. The firebox
# interior stays dark "k" so the flame's transparent cells read against it. See SCENE_HANDOFF.md.

# --- GLOW: warm radial vignette over the ambient surfaces (wall/floor/brick) ---
FX, FY, R1, R2 = 50, 30, 22, 46
VAR = {"W":("V","w"), "F":("G","f"), "B":("N","b")}
for y in range(H):
    for x in range(W):
        c=g[y][x]
        if c in VAR:
            d=math.hypot(x-FX,(y-FY)*1.7)      # *1.7: cells are ~2x tall, keep the glow circular
            bright,dark=VAR[c]
            g[y][x]= bright if d<R1 else (dark if d>R2 else c)

# (the #13 fireplace is a clean stylized terracotta — no brown-brick mortar pass)

# --- WERNER: a CHIBI gent — oversized round bald head (head:body ~6:5), big catch-lit eyes, green
# smoking jacket, holding an open book; standing reading in front of his wingback. (drawn LAST, on top.)
# Footprint cols 61-67 (head) / 61-67 (body), rows 36-46 (~9px wide x 11px tall). HEAD_COL=64.
# HEAD (rows 36-41): big and round — bald skin crown, grey hair at the temples, clean-shaven.
rect(62,36,66,36,"K")                          # crown top (rounded: 5 wide)
px(61,37,"A"); rect(62,37,66,37,"K"); px(67,37,"A")          # temples (grey hair) frame the bald crown
rect(61,38,67,38,"K")                          # brow row
rect(61,39,67,39,"K")                          # cheek row
px(61,38,"A"); px(67,38,"A"); px(61,39,"A"); px(67,39,"A")   # grey side hair (over the ears)
rect(62,38,63,38,"E"); rect(65,38,66,38,"E")   # big eyes — white upper (the catch-light)
px(62,39,"E"); px(63,39,"x"); px(65,39,"x"); px(66,39,"E")   # pupils (looking down at the book)
rect(62,40,66,40,"K"); px(64,40,"i")           # jaw narrows + a small mouth
rect(63,41,65,41,"K")                          # chin
# BODY (rows 42-46): small and stubby under the big head.
rect(61,42,67,42,"J"); rect(63,42,65,42,"y")   # shoulders + light collar
rect(61,43,67,43,"J"); px(62,43,"K"); px(66,43,"K"); rect(63,43,65,43,"P")  # arms + hands + open pages
px(61,44,"J"); rect(62,44,66,44,"1"); px(67,44,"J")          # held book: red cover
rect(62,45,66,45,"J")                          # lower jacket hem
px(62,46,"n"); px(63,46,"n"); px(65,46,"n"); px(66,46,"n")   # little shoes

rows=["".join(r) for r in g]
json.dump({"palette":PAL,"rows":rows}, open(__file__.replace("study.py","study.json"),"w"))
print("wrote study.json", W,"x",H)
