"""The README's battle image (docs/images/battle.jpg): a siege screen with what the battle model sees and decides drawn
over it, the companion of make_hero.py. An illustration: the reading of the position and every number are examples,
not a recorded decision.

	python docs/images/make_battle.py [screen.jpg] [out.jpg]

Needs Pillow (not in the project's requirements) and macOS's Avenir Next. The screen is docs/images/battle_field.jpg
(1470 x 1000; Heroes of Might & Magic III HD Edition, Ubisoft).

The position, as read from the screen: units that face right are the attacker's, so the dragon and the red flyer
inside the walls are the enemy's, and we are the garrison. The acting stack is our shooters in the castle's corner. A
shot moves nobody, so the picture makes no claim about where a stack should stand; what decides between the targets is
the engine's damage rule, of which one part shows on the field: a shot over more than 10 hexes does half damage
(battleHasDistancePenalty; the wall penalty is for shooters outside the walls only). The hexes are VCMI's (x = 14 + 22
* ((row + 1) % 2) + 44 * col, y = 86 + 42 * row on the 800 x 556 field, drawn here at x 1.5), and so is the distance
between two of them (BattleHex::getDistance). The model's side of it: battle_model/model.py adds the engine's value of
each shot to that shot's logit (value_prior) beside a learned match, and its four value heads forecast the reward at
the end of this round, the next two, and the battle (layers.ValueHeads). Colours: the dataviz palette's dark steps."""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

HERE = Path(__file__).resolve().parent
SRC = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "battle_field.jpg")
OUT = sys.argv[2] if len(sys.argv) > 2 else str(HERE / "battle.jpg")
S = 3  # drawn at 3x, saved at 2x: every edge antialiased
W, H = 1200, 816  # the design canvas: the whole screen at 1200 px
K = 1200 / 1470  # the screen's pixels to the canvas's
AVENIR = "/System/Library/Fonts/Avenir Next.ttc"
WEIGHT = {"regular": 7, "medium": 5, "demi": 2, "bold": 0}

INK, INK2, MUTED = (240, 239, 236), (195, 194, 183), (150, 149, 141)
SURFACE = (26, 26, 25)
BLUE, ORANGE, GOLD, VIOLET = (57, 135, 229), (217, 89, 38), (201, 133, 0), (144, 133, 233)
HEAT = (255, 196, 64)  # the gold of the policy, lit for the photograph
WHITE = (255, 255, 255, 255)


def px(v):
	return int(round(v * S))


def box(x0, y0, x1, y1):
	return (px(x0), px(y0), px(x1), px(y1))


def font(size, weight="medium"):
	return ImageFont.truetype(AVENIR, px(size), index=WEIGHT[weight])


def mix(colour, alpha, under=SURFACE):
	"""colour at alpha over the panel's surface, as a solid colour (PIL's draw replaces alpha, it does not blend)."""
	return tuple(int(round(under[c] + (colour[c] - under[c]) * alpha)) for c in range(3)) + (255,)


def layer():
	return Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))


def tracked(draw, xy, text, f, fill, spacing):
	x, y = xy
	for ch in text:
		draw.text((px(x), px(y)), ch, font=f, fill=fill)
		x += draw.textlength(ch, font=f) / S + spacing


def width(draw, text, f):
	return draw.textlength(text, font=f) / S


def flat(points):
	return [c for q in points for c in (px(q[0]), px(q[1]))]


def hexagon(col, row, inset=0.0):
	"""A battle hex's six corners on the canvas, from the top one clockwise. 66 wide and rows 63 apart as the engine
	draws them; 84 tall (the engine's cell is 78) so that neighbours share their edges."""
	x = 1.5 * (14 + 22 * ((row + 1) % 2) + 44 * col) + inset
	y = 1.5 * (86 + 42 * row) - 3 + inset
	w, h = 66 - 2 * inset, 84 - 2 * inset
	return [(x + w / 2, y), (x + w, y + h * 0.25), (x + w, y + h * 0.75), (x + w / 2, y + h), (x, y + h * 0.75), (x, y + h * 0.25)]


def neighbour(col, row, side):
	"""The hex across the edge from corner `side` to the next one (even rows sit half a hex to the right)."""
	shifted = (row + 1) % 2
	return ((col + shifted, row - 1), (col + 1, row), (col + shifted, row + 1), (col + shifted - 1, row + 1), (col - 1, row), (col + shifted - 1, row - 1))[side]


def distance(a, b):
	"""VCMI's BattleHex::getDistance between two (col, row) hexes."""
	x1, y1, x2, y2 = a[0] + a[1] // 2, a[1], b[0] + b[1] // 2, b[1]
	dx, dy = x2 - x1, y2 - y1
	return max(abs(dx), abs(dy)) if (dx >= 0 and dy >= 0) or (dx < 0 and dy < 0) else abs(dx) + abs(dy)


def screen(x0, y0, x1, y1):
	"""A rectangle given in the screen's pixels, on the canvas."""
	return (x0 * K, y0 * K, x1 * K, y1 * K)


def mid(rect):
	return ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)


# ---- the position (an example reading of the screen, not a recorded battle) -------------------------------------------
SHOOTER, RANGE = (15, 0), 10
ACTING = screen(1270, 70, 1350, 226)  # our shooters in the castle's corner, 14 of them
OURS = (screen(1196, 452, 1322, 528), screen(1232, 775, 1312, 905))
# the stacks it can shoot, by the policy's rank: the hexes each stands on, its box, the chance, the value the shot destroys
TARGETS = (
	{"hexes": ((7, 0),), "rect": screen(638, 88, 716, 228), "p": 0.46, "value": 760, "tag": "below"},
	{"hexes": ((13, 9), (12, 9)), "rect": screen(1080, 750, 1224, 905), "p": 0.22, "value": 640, "tag": "above"},
	{"hexes": ((9, 6),), "rect": screen(798, 528, 882, 672), "p": 0.14, "value": 550, "tag": "below"},
	{"hexes": ((13, 4), (12, 4)), "rect": screen(1035, 345, 1190, 532), "p": 0.12, "value": 450, "tag": "below"},
	{"hexes": ((5, 1),), "rect": screen(398, 186, 528, 334), "p": 0.04, "value": 300, "tag": "below"},
)
REST = 0.02  # two more stacks, far left, behind the panel
for t in TARGETS:
	t["far_by"] = min(distance(SHOOTER, h) for h in t["hexes"])  # any of its hexes in range is in range
	t["half"] = t["far_by"] > RANGE
FIELD = [(c, r) for r in range(11) for c in range(17)]
NEAR = {cell for cell in FIELD if distance(SHOOTER, cell) <= RANGE}
TOP = max(t["p"] for t in TARGETS)
BOXES = (ACTING,) + OURS + tuple(t["rect"] for t in TARGETS)


def heat(p):
	"""The policy's gold at a chance p, over the photograph."""
	return mix(HEAT, 0.30 + 0.62 * (p / TOP) ** 0.5, under=(40, 34, 24))


# ---- the photograph: dimmed; lit where a shot does full damage, and on every stack ---------------------------------------
src = Image.open(SRC).convert("RGB").resize((W * S, H * S), Image.LANCZOS)
gray = ImageOps.grayscale(src).convert("RGB")
dark = ImageEnhance.Brightness(Image.blend(src, gray, 0.30)).enhance(0.40)
lit = ImageEnhance.Brightness(Image.blend(src, gray, 0.05)).enhance(0.95)
mask = Image.new("L", (W * S, H * S), 0)
m = ImageDraw.Draw(mask)
for cell in NEAR:
	m.polygon(flat(hexagon(*cell, inset=-2)), fill=112)
for x0, y0, x1, y1 in BOXES:
	m.rectangle(box(x0 - 12, y0 - 12, x1 + 12, y1 + 12), fill=240)
mask = mask.filter(ImageFilter.GaussianBlur(px(15)))
img = Image.alpha_composite(Image.new("RGBA", (W * S, H * S), SURFACE + (255,)), Image.composite(lit, dark, mask).convert("RGBA"))

# ---- the hexes: every one a token; the range's edge, where the rule changes ---------------------------------------------
cells = layer()
d = ImageDraw.Draw(cells)
for col, row in FIELD:
	if col >= 4:
		d.polygon(flat(hexagon(col, row, inset=1.2)), outline=(255, 255, 255, 30))
for col, row in sorted(NEAR):
	corners = hexagon(col, row)
	for side in range(6):
		other = neighbour(col, row, side)
		if other in NEAR or other not in FIELD:
			continue
		a, b = corners[side], corners[(side + 1) % 6]
		for shade, weight in (((0, 0, 0, 120), 4.0), ((255, 255, 255, 225), 1.9)):
			for k in range(3):  # three dashes an edge, none inside a stack's box
				q0 = (a[0] + (b[0] - a[0]) * (k + 0.20) / 3, a[1] + (b[1] - a[1]) * (k + 0.20) / 3)
				q1 = (a[0] + (b[0] - a[0]) * (k + 0.80) / 3, a[1] + (b[1] - a[1]) * (k + 0.80) / 3)
				if (q0[1] + q1[1]) / 2 < 676 and not any(x0 - 6 <= (q0[0] + q1[0]) / 2 <= x1 + 6 and y0 - 6 <= (q0[1] + q1[1]) / 2 <= y1 + 6 for x0, y0, x1, y1 in BOXES):  # 676: the formulas' box
					d.line(flat((q0, q1)), fill=shade, width=px(weight))
img = Image.alpha_composite(img, cells)

# ---- tokens, labels, the shot ------------------------------------------------------------------------------------------------------
over = layer()
d = ImageDraw.Draw(over)


def outline(rect, colour, weight=2.2):
	x0, y0, x1, y1 = rect
	d.rounded_rectangle(box(x0 - 1.5, y0 - 1.5, x1 + 1.5, y1 + 1.5), radius=px(11), outline=(0, 0, 0, 120), width=px(weight + 2.8))
	d.rounded_rectangle(box(x0, y0, x1, y1), radius=px(10), outline=colour + (255,), width=px(weight))


def chip(x, y, colour, name, detail, align="left"):
	"""A label: a coloured dot (the side), the name, a detail. Text in ink, never in the side's colour."""
	fn, fd = font(15.5, "bold"), font(15.5, "medium")
	wn, wd = width(d, name, fn), width(d, detail, fd)
	w = 12 + (16 if colour else 0) + wn + (9 + wd if detail else 0) + 12
	if align == "right":
		x -= w
	elif align == "centre":
		x -= w / 2
	d.rounded_rectangle(box(x, y, x + w, y + 29), radius=px(14.5), fill=SURFACE + (238,), outline=(255, 255, 255, 46), width=px(1))
	tx = x + 12
	if colour:
		d.ellipse(box(x + 12, y + 10, x + 21, y + 19), fill=colour + (255,))
		tx += 16
	d.text((px(tx), px(y + 14.5)), name, font=fn, fill=INK, anchor="lm")
	if detail:
		d.text((px(tx + wn + 9), px(y + 14.5)), detail, font=fd, fill=INK2, anchor="lm")
	return x, w


def number(draw, cx, cy, n, r=10.5):
	"""A target's number: the enemy's orange, a white numeral."""
	draw.ellipse(box(cx - r - 1.6, cy - r - 1.6, cx + r + 1.6, cy + r + 1.6), fill=SURFACE + (255,))
	draw.ellipse(box(cx - r, cy - r, cx + r, cy + r), fill=ORANGE + (255,))
	draw.text((px(cx), px(cy - 0.5)), str(n), font=font(13.5, "bold"), fill=WHITE, anchor="mm")


def tag(x, y, n, p):
	"""A target's number and the chance the policy gives a shot at it."""
	w = 84
	d.rounded_rectangle(box(x, y, x + w, y + 29), radius=px(14.5), fill=heat(p), outline=HEAT + (245,), width=px(1.6))
	number(d, x + 15, y + 14.5, n)
	d.text((px(x + 54), px(y + 14.5)), f"{round(p * 100)}%", font=font(16.5, "bold"), fill=WHITE, anchor="mm", stroke_width=px(1.4), stroke_fill=(0, 0, 0, 190))
	return w


for rect in OURS:
	outline(rect, BLUE, weight=1.5)
outline(ACTING, BLUE)
for n, t in enumerate(TARGETS, start=1):
	outline(t["rect"], ORANGE)
for n, t in enumerate(TARGETS, start=1):
	x0, y0, x1, y1 = t["rect"]
	ty = y1 + 7 if t["tag"] == "below" else y0 - 36
	w = tag(x0, ty, n, t["p"])
	if t["half"]:
		chip(x0 + w + 6, ty, None, f"{t['far_by']} hexes", "half damage")

# the chosen shot: dotted, with a head, and what the rule says of it
first = TARGETS[0]
a = (ACTING[0] - 6, mid(ACTING)[1] - 10)
b = (first["rect"][2] + 12, mid(first["rect"])[1] - 14)
c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 30)
curve = [((1 - u) ** 2 * a[0] + 2 * (1 - u) * u * c[0] + u * u * b[0], (1 - u) ** 2 * a[1] + 2 * (1 - u) * u * c[1] + u * u * b[1]) for u in (k / 44 for k in range(45))]
for x, y in curve[1:-1]:
	d.ellipse(box(x - 4.4, y - 4.4, x + 4.4, y + 4.4), fill=(0, 0, 0, 150))
for x, y in curve[1:-1]:
	d.ellipse(box(x - 3, y - 3, x + 3, y + 3), fill=WHITE)
(ex, ey), (qx, qy) = curve[-1], curve[-3]
norm = ((ex - qx) ** 2 + (ey - qy) ** 2) ** 0.5
ux, uy = (ex - qx) / norm, (ey - qy) / norm
for size, colour in ((13.5, (0, 0, 0, 150)), (10.5, WHITE)):
	d.polygon(flat(((ex + ux * size * 1.25, ey + uy * size * 1.25), (ex - ux * size * 0.15 + uy * size * 0.8, ey - uy * size * 0.15 - ux * size * 0.8), (ex - ux * size * 0.15 - uy * size * 0.8, ey - uy * size * 0.15 + ux * size * 0.8))), fill=colour)
hx, hy = curve[22]
chip(hx, hy - 14.5, None, f"{first['far_by']} hexes", "full damage", align="centre")
chip(W - 16, 16, BLUE, "ACTING", "our shooters × 14", align="right")

# the rule's edge, named once, on the far side of it
ex_, ey_ = hexagon(9, 7)[5]
chip(ex_ - 16, ey_ - 6, None, f"beyond {RANGE} hexes", "half damage", align="right")

# the legend
lx, ly = 332, 16
fl = font(13.5, "medium")
items = (("ours", BLUE), ("enemy", ORANGE))
last = "P(shoot it)"
wl = 14 + sum(9 + 6 + width(d, name, fl) + 16 for name, _ in items) + 70 + 8 + width(d, last, fl) + 14
d.rounded_rectangle(box(lx, ly, lx + wl, ly + 27), radius=px(13.5), fill=SURFACE + (238,), outline=(255, 255, 255, 46), width=px(1))
x = lx + 14
for name, colour in items:
	d.ellipse(box(x, ly + 9, x + 9, ly + 18), fill=colour + (255,))
	d.text((px(x + 15), px(ly + 13.5)), name, font=fl, fill=INK2, anchor="lm")
	x += 9 + 6 + width(d, name, fl) + 16
for k in range(5):
	d.rounded_rectangle(box(x + k * 14, ly + 7.5, x + k * 14 + 12, ly + 19.5), radius=px(3), fill=heat(TOP * (k / 4) ** 2))
d.text((px(x + 78), px(ly + 13.5)), last, font=fl, fill=INK2, anchor="lm")

# formulas become ops: the damage rule as a chain of nodes, no arithmetic
ix, iy, iw, ih = 332, 684, 532, 118
d.rounded_rectangle(box(ix, iy, ix + iw, iy + ih), radius=px(14), fill=SURFACE + (240,), outline=(255, 255, 255, 46), width=px(1))
tracked(d, (ix + 18, iy + 13), "THE GAME'S FORMULAS RUN AS OPS INSIDE THE NETWORK", font(11.2, "demi"), MUTED, 1.1)
fo, fob = font(13.5, "medium"), font(13.5, "demi")
line = mix((255, 255, 255), 0.42)


def node(x, y, text, kind="input"):
	"""An input (quiet), a result (gold) or the logit (the policy's gold): a pill; returns its right edge."""
	f = fo if kind == "input" else fob
	w = width(d, text, f) + 20
	fill, edge, ink = {"input": (mix((255, 255, 255), 0.09), mix((255, 255, 255), 0.24), INK2), "result": (mix(GOLD, 0.34), mix(GOLD, 0.9), INK), "logit": (heat(TOP), HEAT + (255,), (30, 24, 8))}[kind]
	d.rounded_rectangle(box(x, y, x + w, y + 26), radius=px(13), fill=fill, outline=edge, width=px(1))
	d.text((px(x + w / 2), px(y + 13)), text, font=f, fill=ink, anchor="mm")
	return x + w


def op(x, y, sign):
	"""An op between two nodes: a small circle; returns its right edge."""
	d.ellipse(box(x, y + 3, x + 20, y + 23), fill=mix(GOLD, 0.34), outline=mix(GOLD, 0.9), width=px(1))
	d.text((px(x + 10), px(y + 12.2)), sign, font=font(14.5, "demi"), fill=INK, anchor="mm")
	return x + 20


def arrow(x, y, length=24):
	d.line(flat(((x, y), (x + length - 7, y))), fill=line, width=px(1.4))
	d.polygon(flat(((x + length - 8, y - 4.5), (x + length, y), (x + length - 8, y + 4.5))), fill=line)
	return x + length


r1, r2 = iy + 38, iy + 78
x = ix + 18
for k, (name, sign) in enumerate((("count", None), ("damage", "×"), ("attack − defence", "×"), ("range", "×"), ("health", "÷"))):
	if sign:
		x = op(x + 5, r1, sign) + 5
	x = node(x, r1, name)
row_end = x
x = ix + 18
d.line(flat(((x + 9, r1 + 30), (x + 9, r2 + 13), (x + 20, r2 + 13))), fill=line, width=px(1.4), joint="curve")  # the chain turns down to its result
x = arrow(x + 18, r2 + 13, 16)
x = node(x + 5, r2, "kills", "result")
x = arrow(x + 6, r2 + 13)
x = node(x + 5, r2, "value destroyed", "result")
x = arrow(x + 6, r2 + 13)
x = node(x + 5, r2, "this shot's logit", "logit")
img = Image.alpha_composite(img, over)

# ---- the panel: one decision ----------------------------------------------------------------------------------------------------
panel = layer()
d = ImageDraw.Draw(panel)
PX0, PY0, PX1, PY1 = 14, 14, 314, 802
d.rounded_rectangle(box(PX0, PY0, PX1, PY1), radius=px(18), fill=SURFACE + (244,), outline=(255, 255, 255, 40), width=px(1))
L, R = PX0 + 22, PX1 - 22
eyebrow, note = font(12, "demi"), font(12.5, "medium")
fb, fv = font(15, "medium"), font(15, "demi")
y = PY0 + 20
tracked(d, (L, y), "ONE DECISION", eyebrow, MUTED, 1.6)
d.text((px(R), px(y + 8)), "defending a siege", font=note, fill=MUTED, anchor="rm")
y += 30
d.text((px(L), px(y)), "Shoot where the", font=font(25, "demi"), fill=INK)
d.text((px(L), px(y + 30)), "most value falls", font=font(25, "demi"), fill=INK)

# action types: one series, the chosen one solid
y += 86
tracked(d, (L, y), "ACTION TYPE", eyebrow, MUTED, 1.6)
y += 26
rows = (("SHOOT", 0.88), ("WAIT", 0.05), ("DEFEND", 0.03), ("MELEE", 0.02), ("MOVE", 0.01), ("8 others", 0.01))
bx0, bx1 = L + 80, R - 46
for k, (name, p) in enumerate(rows):
	cy = y + k * 25 + 8
	d.text((px(L), px(cy)), name, font=fb, fill=INK if k == 0 else INK2, anchor="lm")
	d.rounded_rectangle(box(bx0, cy - 4.5, bx1, cy + 4.5), radius=px(4), fill=mix((255, 255, 255), 0.07))
	w = max((bx1 - bx0) * p / rows[0][1], 5)
	d.rounded_rectangle(box(bx0, cy - 4.5, bx0 + w, cy + 4.5), radius=px(4), fill=mix(GOLD, 1.0 if k == 0 else 0.62), corners=(False, True, True, False))
	d.text((px(R), px(cy)), f"{round(p * 100)}%", font=fv, fill=INK if k == 0 else INK2, anchor="rm")

# whom: what the rules say each shot does, plus a learned match
y += len(rows) * 25 + 26
tracked(d, (L, y), "WHOM = RULES + LEARNED MATCH", eyebrow, MUTED, 1.3)
y += 26
cols = (L + 84, L + 146, L + 204, R)
for x, name in zip(cols, ("hexes", "damage", "value", "p")):
	d.text((px(x), px(y + 6)), name, font=note, fill=MUTED, anchor="rm")
y += 24
for k, t in enumerate(TARGETS):
	cy = y + k * 26 + 9
	ink = INK if k == 0 else INK2
	if k == 0:
		d.rounded_rectangle(box(L - 9, cy - 13, R + 9, cy + 13), radius=px(6), fill=mix(GOLD, 0.30))
	number(d, L + 11, cy, k + 1, r=10)
	for x, text, f in ((cols[0], str(t["far_by"]), fb), (cols[1], "half" if t["half"] else "full", fb), (cols[2], str(t["value"]), fb), (cols[3], f"{round(t['p'] * 100)}%", fv)):
		d.text((px(x), px(cy)), text, font=f, fill=ink, anchor="rm")
cy = y + len(TARGETS) * 26 + 9
d.text((px(L), px(cy)), "2 more, far left", font=fb, fill=INK2, anchor="lm")
d.text((px(cols[1]), px(cy)), "half", font=fb, fill=INK2, anchor="rm")
d.text((px(cols[3]), px(cy)), f"{round(REST * 100)}%", font=fv, fill=INK2, anchor="rm")
y = cy + 25
d.text((px(L), px(y)), "value: the kills the engine expects, in gold", font=note, fill=MUTED, anchor="lm")

# value heads: the battle's reward at four horizons
y += 34
tracked(d, (L, y), "VALUE: THE REWARD, 4 HORIZONS", eyebrow, MUTED, 1.4)
y += 34
cx0, cx1, cy0, cy1 = L + 28, R - 22, y, y + 62
values = (("this round", 0.0, -0.01), ("next", 0.34, -0.02), ("+2", 0.62, -0.04), ("the end", 1.0, -0.05))
lo, hi = -0.07, 0.012


def at(u, v):
	return cx0 + (cx1 - cx0) * u, cy1 - (cy1 - cy0) * (v - lo) / (hi - lo)


zy = at(0, 0.0)[1]
for k in range(0, int(R - L), 9):  # even, a recessive dashed line
	d.line([px(L + k), px(zy), px(min(L + k + 4, R)), px(zy)], fill=mix((255, 255, 255), 0.26), width=px(1))
d.text((px(R), px(zy - 9)), "even 0", font=font(11.5, "medium"), fill=MUTED, anchor="rm")
pts = [at(u, v) for _, u, v in values]
d.line(flat(pts[:3]), fill=VIOLET + (255,), width=px(2), joint="curve")
a, b = pts[2], pts[3]
for k in range(9):  # to the battle's end: far, so dashed
	if k % 2 == 0:
		d.line([px(a[0] + (b[0] - a[0]) * k / 9), px(a[1] + (b[1] - a[1]) * k / 9), px(a[0] + (b[0] - a[0]) * (k + 1) / 9), px(a[1] + (b[1] - a[1]) * (k + 1) / 9)], fill=VIOLET + (255,), width=px(2))
for (name, u, v), q in zip(values, pts):
	d.ellipse(box(q[0] - 6.5, q[1] - 6.5, q[0] + 6.5, q[1] + 6.5), fill=SURFACE + (255,))
	d.ellipse(box(q[0] - 4.5, q[1] - 4.5, q[0] + 4.5, q[1] + 4.5), fill=VIOLET + (255,))
	d.text((px(q[0]), px(cy1 + 14)), name, font=note, fill=MUTED, anchor="mm")
d.text((px(pts[0][0] + 6), px(pts[0][1] + 17)), "−0.01", font=fv, fill=INK2, anchor="mm")
d.text((px(pts[3][0] - 8), px(pts[3][1] - 16)), "−0.05", font=fv, fill=INK, anchor="mm")
y = cy1 + 40
for k, text in enumerate(("Enemy value destroyed − ours lost, of both", "armies. Outnumbered, every kill still counts.")):
	d.text((px(L), px(y + k * 17)), text, font=note, fill=MUTED, anchor="lm")
y += 17
d.text((px(R), px(PY1 - 14)), "illustration · example numbers", font=font(10.5, "medium"), fill=(120, 119, 112), anchor="rm")
img = Image.alpha_composite(img, panel)

out = img.convert("RGB").resize((W * 2, H * 2), Image.LANCZOS)
out.save(OUT, quality=90, optimize=True)
print(OUT, out.size, "| panel text ends at y", round(y + 8), "of", PY1, "| ops row ends at x", round(row_end), "of", ix + iw - 18, "| chain ends at", round(x), "| hexes", [t["far_by"] for t in TARGETS], "| p sums to", round(sum(t["p"] for t in TARGETS) + REST, 2))
