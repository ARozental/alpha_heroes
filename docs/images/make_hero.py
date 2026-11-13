"""The README's first image (docs/images/hero.jpg): an adventure-map screen with what the agent sees and decides drawn
over it. An illustration: the position and every number are examples, not a recorded decision.

	python docs/images/make_hero.py [screen.jpg] [out.jpg]

Needs Pillow (not in the project's requirements) and macOS's Avenir Next. The screen is docs/images/adventure_map.jpg
(1920 x 1080; Heroes of Might & Magic III HD Edition, Ubisoft). Everything drawn is set below: the tiles the hero can
reach with their probabilities, the labelled entities, the panel's three parts (action type, where = prior + learned
match, the value heads' four horizons). Colours: the dataviz palette's dark steps, checked on the panel's surface."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
SRC = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "adventure_map.jpg")
OUT = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "hero.jpg")
S = 3  # drawn at 3x, saved at 2x: every edge antialiased
W, H = 1200, 675  # the design canvas
CROP = (420, 100, 1860, 910)  # of the 1920 x 1080 screen; 1440 x 810
AVENIR = "/System/Library/Fonts/Avenir Next.ttc"
WEIGHT = {"regular": 7, "medium": 5, "demi": 2, "bold": 0}

INK, INK2, MUTED = (240, 239, 236), (195, 194, 183), (150, 149, 141)
SURFACE = (26, 26, 25)
BLUE, ORANGE, AQUA, GOLD, VIOLET = (57, 135, 229), (217, 89, 38), (25, 158, 112), (201, 133, 0), (144, 133, 233)
HEAT = (255, 196, 64)  # the gold of the policy, lit for the photograph


def px(v):
	return int(round(v * S))


def box(x0, y0, x1, y1):
	return (px(x0), px(y0), px(x1), px(y1))


def font(size, weight="medium"):
	return ImageFont.truetype(AVENIR, px(size), index=WEIGHT[weight])


def tile(i, j):
	"""A map tile's rectangle on the design canvas (the screen's tiles are 72 px, columns start at -6)."""
	x0, y0 = 60 * i - 355, 60 * j - 83.33
	return x0, y0, x0 + 60, y0 + 60


def centre(i, j):
	x0, y0, x1, y1 = tile(i, j)
	return (x0 + x1) / 2, (y0 + y1) / 2


def mix(colour, alpha, under=SURFACE):
	"""colour at alpha over the panel's surface, as a solid colour."""
	return tuple(int(round(under[c] + (colour[c] - under[c]) * alpha)) for c in range(3)) + (255,)


def layer():
	return Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))


def tracked(draw, xy, text, f, fill, spacing):
	"""Text with letter spacing (design px); returns its width."""
	x, y = xy
	for ch in text:
		draw.text((px(x), px(y)), ch, font=f, fill=fill)
		x += draw.textlength(ch, font=f) / S + spacing
	return x - xy[0]


def width(draw, text, f):
	return draw.textlength(text, font=f) / S


# ---- what is on the map (an example, not a recorded position) ---------------------------------------------------
HERO = (14, 7)
TARGETS = {(13, 10): 0.41, (12, 10): 0.18, (14, 10): 0.12, (16, 10): 0.08, (11, 2): 0.04, (17, 12): 0.04, (13, 2): 0.02}
REST = [(11, j) for j in range(3, 10)] + [(i, 9) for i in range(12, 19)] + [(14, 8), (13, 8), (15, 8), (15, 10), (11, 10), (10, 10), (9, 10), (8, 10), (7, 10), (17, 10), (18, 10), (18, 11), (18, 12), (12, 2)]
REACH = dict(TARGETS)
REACH.update({t: (1 - sum(TARGETS.values())) / len(REST) for t in REST})
PATH = [HERO, (14, 8), (14, 9), (13, 10)]

# ---- the photograph: dimmed, lit again where the hero can go ----------------------------------------------------
src = Image.open(SRC).convert("RGB").crop(CROP).resize((W * S, H * S), Image.LANCZOS)
gray = ImageOps.grayscale(src).convert("RGB")
dark = ImageEnhance.Brightness(Image.blend(src, gray, 0.30)).enhance(0.50)
lit = ImageEnhance.Brightness(Image.blend(src, gray, 0.05)).enhance(0.92)
mask = Image.new("L", (W * S, H * S), 0)
m = ImageDraw.Draw(mask)
for t in list(REACH) + [HERO]:
	x0, y0, x1, y1 = tile(*t)
	m.rectangle(box(x0 - 16, y0 - 16, x1 + 16, y1 + 16), fill=235)
m.rectangle(box(372, 118, 638, 412), fill=235)  # the town
mask = mask.filter(ImageFilter.GaussianBlur(px(26)))
img = Image.composite(lit, dark, mask).convert("RGBA")

# ---- the move policy: one cell a reachable tile ------------------------------------------------------------------
heat = layer()
d = ImageDraw.Draw(heat)
top = max(REACH.values())
for t, p in REACH.items():
	x0, y0, x1, y1 = tile(*t)
	alpha = int(255 * (0.13 + 0.60 * (p / top) ** 0.5))
	d.rounded_rectangle(box(x0 + 3, y0 + 3, x1 - 3, y1 - 3), radius=px(8), fill=HEAT + (alpha,))
	if t in TARGETS:
		d.rounded_rectangle(box(x0 + 3, y0 + 3, x1 - 3, y1 - 3), radius=px(8), outline=HEAT + (235,), width=px(1.6))
img = Image.alpha_composite(img, heat)

# ---- attention: the hero's token reads the others -----------------------------------------------------------------
lines = layer()
d = ImageDraw.Draw(lines)
hx, hy = centre(*HERO)
for end in ((700, 132), centre(17, 12)):
	d.line([px(hx), px(hy), px(end[0]), px(end[1])], fill=(255, 255, 255, 70), width=px(1.2))
	d.ellipse(box(end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3), fill=(255, 255, 255, 150))
img = Image.alpha_composite(img, lines)

# ---- tokens: the entities the encoders name ----------------------------------------------------------------------
over = layer()
d = ImageDraw.Draw(over)


def outline(rect, colour):
	x0, y0, x1, y1 = rect
	d.rounded_rectangle(box(x0 - 1.5, y0 - 1.5, x1 + 1.5, y1 + 1.5), radius=px(11), outline=(0, 0, 0, 120), width=px(5))
	d.rounded_rectangle(box(x0, y0, x1, y1), radius=px(10), outline=colour + (255,), width=px(2.2))


def chip(x, y, colour, name, detail, align="left"):
	"""A label: a coloured dot (the kind), the name, a detail. Text in ink, never in the kind's colour."""
	fn, fd = font(15.5, "bold"), font(15.5, "medium")
	wn, wd = width(d, name, fn), width(d, detail, fd)
	w = 12 + 9 + 7 + wn + (9 + wd if detail else 0) + 12
	if align == "right":
		x -= w
	d.rounded_rectangle(box(x, y, x + w, y + 29), radius=px(14.5), fill=SURFACE + (238,), outline=(255, 255, 255, 46), width=px(1))
	d.ellipse(box(x + 12, y + 10, x + 21, y + 19), fill=colour + (255,))
	d.text((px(x + 28), px(y + 14.5)), name, font=fn, fill=INK, anchor="lm")
	if detail:
		d.text((px(x + 28 + wn + 9), px(y + 14.5)), detail, font=fd, fill=INK2, anchor="lm")
	return w


town = (372, 118, 638, 412)
gate = (22, 22, 190, 185)  # a Subterranean Gate: the way to the underground level
mill = (682, 8, 812, 132)
outline(town, BLUE)
outline(gate, AQUA)
outline(mill, AQUA)
x0, y0, x1, y1 = tile(*HERO)
outline((x0 + 2, y0 - 6, x1 - 2, y1 + 6), BLUE)
x0, y0, x1, y1 = tile(16, 10)
outline((x0 + 3, y0 + 3, x1 - 3, y1 + 3), ORANGE)

chip(380, 126, BLUE, "TOWN", "9 buildings · +1,000 a day")
chip(552, 352, BLUE, "HERO", "level 5 · army 8,400")
chip(22, 192, AQUA, "SUBTERRANEAN GATE", "")

# the gate's distance: the pathfinder's route along the roads, in days, not the straight line in tiles
route = [centre(14, 9), centre(11, 9), centre(11, 3), (gate[2] + 5, centre(11, 3)[1])]
for a, b in zip(route, route[1:]):
	length = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
	n = max(int(length / 16), 1)
	for k in range(n):
		q0 = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
		q1 = (a[0] + (b[0] - a[0]) * (k + 0.56) / n, a[1] + (b[1] - a[1]) * (k + 0.56) / n)
		d.line([px(q0[0]), px(q0[1]), px(q1[0]), px(q1[1])], fill=(0, 0, 0, 150), width=px(4.6))
		d.line([px(q0[0]), px(q0[1]), px(q1[0]), px(q1[1])], fill=(255, 255, 255, 235), width=px(2.2))
gx, gy = route[-1]
d.polygon([px(gx - 9), px(gy), px(gx + 5), px(gy - 8), px(gx + 5), px(gy + 8)], fill=(0, 0, 0, 150))
d.polygon([px(gx - 7), px(gy), px(gx + 4), px(gy - 6), px(gx + 4), px(gy + 6)], fill=(255, 255, 255, 255))
fr = font(14.5, "bold")
wr = width(d, "2 days", fr) + 22
d.rounded_rectangle(box(250 - wr / 2, 88, 250 + wr / 2, 114), radius=px(13), fill=SURFACE + (238,), outline=(255, 255, 255, 46), width=px(1))
d.text((px(250), px(101)), "2 days", font=fr, fill=INK, anchor="mm")

# what the gate means: a small graph of the two levels, its edges in days
ix, iy, iw, ih = 22, 232, 252, 136
d.rounded_rectangle(box(ix, iy, ix + iw, iy + ih), radius=px(14), fill=SURFACE + (238,), outline=(255, 255, 255, 46), width=px(1))
tracked(d, (ix + 14, iy + 11), "DISTANCE IS DAYS, NOT TILES", font(11.2, "demi"), MUTED, 1.1)
fs, fsb = font(12.5, "medium"), font(12.5, "demi")
ys, yu = iy + 58, iy + 102
xh, xg, xm = ix + 112, ix + 216, ix + 132
d.text((px(ix + 14), px(ys)), "surface", font=fs, fill=MUTED, anchor="lm")
d.text((px(ix + 14), px(yu)), "underground", font=fs, fill=MUTED, anchor="lm")
for xa, xb, yy in ((xh, xg, ys), (xm, xg, yu)):  # a level's road: dashed
	for k in range(int((xb - xa - 16) / 9)):
		d.line([px(xa + 8 + k * 9), px(yy), px(xa + 8 + k * 9 + 5), px(yy)], fill=(255, 255, 255, 215), width=px(1.8))
d.line([px(xg), px(ys + 8), px(xg), px(yu - 8)], fill=AQUA + (255,), width=px(2.4))  # the gate: one step between the levels
d.ellipse(box(xh - 5, ys - 5, xh + 5, ys + 5), fill=BLUE + (255,))
for yy in (ys, yu):
	d.ellipse(box(xg - 6.5, yy - 6.5, xg + 6.5, yy + 6.5), fill=SURFACE + (255,), outline=AQUA + (255,), width=px(2.4))
d.ellipse(box(xm - 5, yu - 5, xm + 5, yu + 5), fill=AQUA + (255,))
d.text((px(xh), px(ys - 14)), "hero", font=fs, fill=INK2, anchor="mm")
d.text((px(xg), px(ys - 14)), "gate", font=fs, fill=INK2, anchor="mm")
d.text((px((xh + xg) / 2), px(ys - 14)), "2 days", font=fsb, fill=INK, anchor="mm")
d.text((px(xg - 11), px((ys + yu) / 2)), "1 step", font=fsb, fill=INK, anchor="rm")
d.text((px(xm), px(yu + 15)), "mine", font=fs, fill=INK2, anchor="mm")
d.text((px((xm + xg) / 2 + 6), px(yu + 15)), "1 day", font=fsb, fill=INK, anchor="mm")
chip(814, 140, AQUA, "MILL", "weekly", align="right")
chip(598, 482, ORANGE, "GUARD", "P(win) 0.97")
chip(330, 584, AQUA, "ARTIFACT", "worth 2,000")
chip(612, 602, AQUA, "CHEST", "1,500")

# the chosen move: a dotted path and an arrow head
pts = [centre(*t) for t in PATH]
pts[0] = (pts[0][0], pts[0][1] + 34)
pts[-1] = (pts[-1][0] + 27, pts[-1][1] - 27)
for a, b in zip(pts, pts[1:]):
	length = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
	n = max(int(length / 11), 1)
	for k in range(n + 1):
		x, y = a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n
		d.ellipse(box(x - 4.4, y - 4.4, x + 4.4, y + 4.4), fill=(0, 0, 0, 150))
		d.ellipse(box(x - 3, y - 3, x + 3, y + 3), fill=(255, 255, 255, 255))
ex, ey = pts[-1]
ux, uy = -0.7071, 0.7071  # the last step's direction


def head(size, colour):
	tip = (ex + ux * size * 1.25, ey + uy * size * 1.25)
	left = (ex - ux * size * 0.15 + uy * size * 0.8, ey - uy * size * 0.15 - ux * size * 0.8)
	right = (ex - ux * size * 0.15 - uy * size * 0.8, ey - uy * size * 0.15 + ux * size * 0.8)
	d.polygon([c for q in (tip, left, right) for c in (px(q[0]), px(q[1]))], fill=colour)


head(13.5, (0, 0, 0, 150))
head(10.5, (255, 255, 255, 255))

# the probabilities of the likeliest targets, on their cells
fp = font(16, "bold")
for t, p in TARGETS.items():
	if p < 0.08:
		continue
	x0, y0, x1, y1 = tile(*t)
	d.text((px((x0 + x1) / 2), px(y1 - 15)), f"{round(p * 100)}%", font=fp, fill=(255, 255, 255, 255), anchor="mm", stroke_width=px(1.6), stroke_fill=(0, 0, 0, 200))

# the legend, bottom left
lx, ly = 16, 632
fl = font(13.5, "medium")
items = (("ours", BLUE), ("object", AQUA), ("guarded", ORANGE))
wl = 14 + sum(9 + 6 + width(d, name, fl) + 16 for name, _ in items) + 70 + 8 + width(d, "P(the hero moves there)", fl) + 14
d.rounded_rectangle(box(lx, ly, lx + wl, ly + 27), radius=px(13.5), fill=SURFACE + (235,), outline=(255, 255, 255, 46), width=px(1))
x = lx + 14
for name, colour in items:
	d.ellipse(box(x, ly + 9, x + 9, ly + 18), fill=colour + (255,))
	d.text((px(x + 15), px(ly + 13.5)), name, font=fl, fill=INK2, anchor="lm")
	x += 9 + 6 + width(d, name, fl) + 16
for k in range(5):
	a = int(255 * (0.13 + 0.60 * (k / 4) ** 0.5))
	d.rounded_rectangle(box(x + k * 14, ly + 7.5, x + k * 14 + 12, ly + 19.5), radius=px(3), fill=tuple(int(SURFACE[c] + (HEAT[c] - SURFACE[c]) * a / 255) for c in range(3)) + (255,))
d.text((px(x + 78), px(ly + 13.5)), "P(the hero moves there)", font=fl, fill=INK2, anchor="lm")
img = Image.alpha_composite(img, over)

# ---- the panel: one decision ----------------------------------------------------------------------------------------
panel = layer()
d = ImageDraw.Draw(panel)
PX0, PY0, PX1, PY1 = 836, 14, 1182, 661
d.rounded_rectangle(box(PX0, PY0, PX1, PY1), radius=px(18), fill=SURFACE + (240,), outline=(255, 255, 255, 40), width=px(1))
L, R = PX0 + 22, PX1 - 22
eyebrow, note = font(12, "demi"), font(12.5, "medium")
fb, fv = font(15, "medium"), font(15, "demi")
y = PY0 + 18
tracked(d, (L, y), "ONE DECISION", eyebrow, MUTED, 1.6)
d.text((px(R), px(y + 8)), "day 12 · 1 of 3 heroes", font=note, fill=MUTED, anchor="rm")
y += 25
d.text((px(L), px(y)), "Move the hero", font=font(25, "demi"), fill=INK)
d.text((px(L), px(y + 30)), "to the artifact", font=font(25, "demi"), fill=INK)

# action types: one series, the chosen one solid
y += 78
tracked(d, (L, y), "ACTION TYPE", eyebrow, MUTED, 1.6)
y += 24
rows = (("MOVE", 0.62), ("RECRUIT", 0.12), ("BUILD", 0.09), ("TRAVEL", 0.07), ("END TURN", 0.03), ("9 others", 0.07))
bx0, bx1 = L + 92, R - 48
for k, (name, p) in enumerate(rows):
	cy = y + k * 23 + 8
	d.text((px(L), px(cy)), name, font=fb, fill=INK if k == 0 else INK2, anchor="lm")
	d.rounded_rectangle(box(bx0, cy - 4.5, bx1, cy + 4.5), radius=px(4), fill=mix((255, 255, 255), 0.07))
	w = max((bx1 - bx0) * p / 0.62, 5)
	d.rounded_rectangle(box(bx0, cy - 4.5, bx0 + w, cy + 4.5), radius=px(4), fill=mix(GOLD, 1.0 if k == 0 else 0.62), corners=(False, True, True, False))
	d.text((px(R), px(cy)), f"{round(p * 100)}%", font=fv, fill=INK if k == 0 else INK2, anchor="rm")

# where: a prior from the game's own numbers plus a learned match
y += len(rows) * 23 + 22
tracked(d, (L, y), "WHERE  =  PRIOR  +  LEARNED MATCH", eyebrow, MUTED, 1.4)
y += 23
cols = (L, L + 172, L + 236, R)
for x, name, anchor in ((cols[0], "hero and tile", "lm"), (cols[1], "prior", "rm"), (cols[2], "match", "rm"), (cols[3], "p", "rm")):
	d.text((px(x), px(y + 6)), name, font=note, fill=MUTED, anchor=anchor)
y += 20
table = (("artifact", "+3.1", "+0.4", "41%"), ("gold", "+2.3", "+0.1", "18%"), ("gold", "+1.9", "+0.1", "12%"), ("guard, then loot", "+1.5", "+0.2", "8%"))
for k, (name, prior, match, p) in enumerate(table):
	cy = y + k * 23 + 9
	ink = INK if k == 0 else INK2
	if k == 0:
		d.rounded_rectangle(box(L - 9, cy - 11.5, R + 9, cy + 11.5), radius=px(6), fill=mix(GOLD, 0.30))
	d.text((px(cols[0]), px(cy)), name, font=fb, fill=ink, anchor="lm")
	d.text((px(cols[1]), px(cy)), prior, font=fb, fill=ink, anchor="rm")
	d.text((px(cols[2]), px(cy)), match, font=fb, fill=ink, anchor="rm")
	d.text((px(cols[3]), px(cy)), p, font=fv, fill=ink, anchor="rm")
y += len(table) * 23 + 4
d.text((px(L), px(y + 8)), "prior: the tile's value × P(we win its guard)", font=note, fill=MUTED, anchor="lm")

# value heads: one forecast at four horizons
y += 34
tracked(d, (L, y), "VALUE: OUR SHARE OF ALL EMPIRES", eyebrow, MUTED, 1.4)
y += 30
cx0, cx1, cy0, cy1 = L + 8, R - 26, y, y + 70
values = (("now", 0.0, 0.52), ("+2 d", 0.2, 0.535), ("+5 d", 0.43, 0.56), ("+7 d", 0.6, 0.575), ("end", 1.0, 0.66))
lo, hi = 0.485, 0.675


def at(u, v):
	return cx0 + (cx1 - cx0) * u, cy1 - (cy1 - cy0) * (v - lo) / (hi - lo)


py = at(0, 0.5)[1]
for k in range(0, int(R - cx0), 9):  # parity, a recessive dashed line
	d.line([px(cx0 + k), px(py), px(min(cx0 + k + 4, R)), px(py)], fill=mix((255, 255, 255), 0.26), width=px(1))
d.text((px(cx0 + (cx1 - cx0) * 0.80), px(py - 9)), "parity 0.50", font=font(11.5, "medium"), fill=MUTED, anchor="mm")
pts = [at(u, v) for _, u, v in values]
d.line([c for q in pts[:4] for c in (px(q[0]), px(q[1]))], fill=VIOLET + (255,), width=px(2), joint="curve")
a, b = pts[3], pts[4]
n = 9
for k in range(n):  # to the game's end: far, so dashed
	if k % 2 == 0:
		d.line([px(a[0] + (b[0] - a[0]) * k / n), px(a[1] + (b[1] - a[1]) * k / n), px(a[0] + (b[0] - a[0]) * (k + 1) / n), px(a[1] + (b[1] - a[1]) * (k + 1) / n)], fill=VIOLET + (255,), width=px(2))
for (name, u, v), q in zip(values, pts):
	d.ellipse(box(q[0] - 6.5, q[1] - 6.5, q[0] + 6.5, q[1] + 6.5), fill=SURFACE + (255,))
	d.ellipse(box(q[0] - 4.5, q[1] - 4.5, q[0] + 4.5, q[1] + 4.5), fill=VIOLET + (255,))
	d.text((px(q[0]), px(cy1 + 16)), name, font=note, fill=MUTED, anchor="mm")
d.text((px(pts[0][0] + 4), px(pts[0][1] - 15)), "0.52", font=fv, fill=INK2, anchor="mm")
d.text((px(pts[4][0]), px(pts[4][1] - 15)), "0.66", font=fv, fill=INK, anchor="mm")
y = cy1 + 38
d.text((px(L), px(y)), "each head is graded by the next one's later forecast", font=font(11.8, "medium"), fill=MUTED, anchor="lm")
d.text((px(R), px(PY1 - 13)), "illustration · example numbers", font=font(10.5, "medium"), fill=(120, 119, 112), anchor="rm")
img = Image.alpha_composite(img, panel)

out = img.convert("RGB").resize((W * 2, H * 2), Image.LANCZOS)
out.save(OUT, quality=90, optimize=True)
print(OUT, out.size, "panel text ends at y", round(y + 7), "of", PY1)
