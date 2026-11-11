"""Bounded views of the numbers the learned layers read, and Nullkiller2's candidates as tables shaped like inputs.

The user's rules (2026-11-11), for learning that is smooth and fast:
- A number is a channel on a pair the model already has ((hero, tile) of a window, (hero, area) of TRAVEL), never a
  row of its own: the layer that scores the pair then reads it beside the hero's and the tile's vectors.
- Value is counted the same several bounded ways everywhere (value_views): absolute at two scales, as a share of our
  empire at two scales, in days and in weeks of our own growth. Each is x / (|x| + half): signed, smooth, within
  -1..1, never flat at a scale that matters (one of the two scales is in its sensitive range).
- Time is a discount, never a day count or a day's embedding (discounts), for this hero's own days and for the
  path's movement points in a baseline hero's days (a faster hero shows in the first, not in the second), and
  value comes already divided by it (rate_views).
- Preference is a gap to the best alternative (gap, preference), within -1..0.
- Odds are one-sided (risk): nothing grows because a fight is easy. A loss on the way is the share and its log
  survival, finite at certain death and never without a gradient (log_survival).

candidate_tables turns Nullkiller2's candidate rows (schema 19's k_*: Advice.cpp) into dense tables at our MOVE and
TRAVEL entries, with only pairwise comparisons over the rows and scatter_add: fixed shapes, no nonzero, nothing the
CUDA step graph or Apple's GPU cannot record.
"""
import torch

from . import constants as c

BASE_DAY = 1500.0  # movement points a baseline hero walks a day (encoders.UNITS["movement"])
VALUE_HALVES = (2_000.0, 20_000.0)  # gold: where the two absolute views stand half-way
SHARE_HALVES = (0.02, 0.2)  # of our empire's value
PACE_HALVES = (1.0, 7.0)  # days of our own growth
PACE_FLOOR = 0.005  # our daily gain is counted as at least this share of our empire (a week that lost value, or no history)
PHI_FLOOR = 1000.0  # gold: an empire below it is all but gone (labels.PHI_FLOOR)
RATE_FLOOR = 0.25  # days: a pickup beside the hero counts at most 4 times its value a day
RATE_HALVES = (20_000.0, 0.02, 1.0)  # value a day: gold, share of our empire, days of our own growth
FAR_DAYS = 30.0  # the days of a tile no path reaches: its discounts ~0 (the reach's own flag says "never")
GAP_SCALE = 4.0  # prior logits: a gap of 4 reads -0.76
PREFERENCE_SCALE = 3.0  # e-folds below the best priority: 3 read -0.76
SURVIVAL_FLOOR = 0.01  # log_survival at certain death: ln 0.01 = -4.6
PLAN_SHARPNESS = 4.0  # a hero's plan vector weighs a candidate e^(4 x its hero-relative preference)

# Schema 19's candidate rows are integers on the wire (records' missing tail reads 0, so "none" is 0 and indices are
# shifted by one): the scales to divide by, and candidate_rows to read them
SCORE_SCALE, DAYS_SCALE, LOSS_SCALE, DANGER_SCALE = 100.0, 1000.0, 10000.0, 1000.0
KINDS = 7  # Advice.cpp's k_kind: visit or capture, attack a hero, attack a town, defend, meet our hero, explore, other
FLAG_ARRIVES, FLAG_UNREACHABLE, FLAG_TELEPORT, FLAG_CHAIN = 1, 2, 4, 8  # k_flags' bits
CHANNELS = (
	"has", "preference_all", "preference_hero", "gather_all", "gather_hero", "fallback", "arrives", "soon", "later", "loss", "log_survival", "risk",
	"unreachable", "teleport", "chain", "settled", *(f"kind_{i}" for i in range(KINDS)), "others", "mass",
)
F = len(CHANNELS)
# Each channel's bound (the gate "no explosion": scratchpad ov/gate_channels.py reads them)
BOUNDS = {name: (0.0, 1.0) for name in CHANNELS}
BOUNDS.update({name: (-1.0, 0.0) for name in ("preference_all", "preference_hero", "gather_all", "gather_hero")}, log_survival=(-4.61, 0.01))


def soft(x, half):
	"""x / (|x| + half): signed, smooth, within -1..1, half-way at `half`."""
	return x / (x.abs() + half)


def _like(value, target):
	"""A per-decision value [B] shaped to broadcast against target [B, ...]."""
	return value.reshape(value.shape + (1,) * (target.dim() - value.dim())) if value.dim() < target.dim() else value


def daily_gain(phi, phi_past, days_past):
	"""[B] our empire value's gain a day over the last days_past days (phi_past: the model's own output then, passed
	back: schema 19's g_phi_past, g_phi_past_days), at least PACE_FLOOR of our empire: a week that lost value, or the
	first day (days_past 0), counts as that floor."""
	phi = phi.float().clamp(min=PHI_FLOOR)
	days = days_past.float()
	gain = torch.where((days > 0) & (phi_past > 0), (phi - phi_past.float()) / days.clamp(min=1.0), torch.zeros_like(phi))
	return torch.maximum(gain, PACE_FLOOR * phi)


def value_views(gold, phi, pace):
	"""A gold amount [...] in six bounded views [..., 6], each signed and within -1..1: absolute (half-way at 2,000 and
	at 20,000), as a share of our empire (2%, 20%), in our own growth (a day's, a week's). phi, pace: [B] (our empire's
	value, daily_gain's) or anything that broadcasts against gold."""
	gold = gold.float()
	phi, pace = _like(phi.float().clamp(min=PHI_FLOOR), gold), _like(pace.float().clamp(min=1.0), gold)
	share, days = gold / phi, gold / pace
	return torch.stack(
		[soft(gold, VALUE_HALVES[0]), soft(gold, VALUE_HALVES[1]), soft(share, SHARE_HALVES[0]), soft(share, SHARE_HALVES[1]), soft(days, PACE_HALVES[0]), soft(days, PACE_HALVES[1])],
		dim=-1,
	)


def rate_views(gold, days, phi, pace):
	"""Value a day of movement [..., 3], each within -1..1: gold / (days + RATE_FLOOR) as an absolute amount (half-way at
	20,000 a day), as a share of our empire (2% a day) and in days of our own growth (1)."""
	rate = gold.float() / (days.float().clamp(min=0.0) + RATE_FLOOR)
	phi, pace = _like(phi.float().clamp(min=PHI_FLOOR), rate), _like(pace.float().clamp(min=1.0), rate)
	return torch.stack([soft(rate, RATE_HALVES[0]), soft(rate / phi, RATE_HALVES[1]), soft(rate / pace, RATE_HALVES[2])], dim=-1)


def own_days(turns, left, move_now, move_max):
	"""The days this hero needs to a tile [...], fractional: reached today, the share of a full day's movement spent on
	arriving; on a later day, the whole days and the share of the last day's movement spent (the encoder's travel_days).
	turns: days (0 today, < 0 never); left: movement points left on arriving (schema 19's w_arrive_left); move_now,
	move_max: the hero's movement left and a full day's, broadcasting. FAR_DAYS where no path reaches."""
	turns, left = turns.float(), left.float().clamp(min=0.0)
	move_max = move_max.float().clamp(min=1.0)
	today = (move_now.float() - left).clamp(min=0.0) / move_max
	later = turns + (1.0 - left / move_max).clamp(0.0, 1.0)
	return torch.where(turns < 0, torch.full_like(today, FAR_DAYS), torch.where(turns < 1, today, later))


def base_days(turns, left, move_now, move_max):
	"""The path's movement points in a baseline hero's days [...] (BASE_DAY points a day): what is left of today, the
	whole days between and the last day's part. A fast hero's own_days are fewer than these."""
	turns, left = turns.float(), left.float().clamp(min=0.0)
	move_now, move_max = move_now.float(), move_max.float().clamp(min=1.0)
	today = (move_now - left).clamp(min=0.0)
	later = move_now + (turns - 1.0).clamp(min=0.0) * move_max + (move_max - left).clamp(min=0.0)
	return torch.where(turns < 0, torch.full_like(today, FAR_DAYS), torch.where(turns < 1, today, later) / BASE_DAY)


def discounts(days):
	"""Days [...] as two discounts [..., 2], each within 0..1: 1 / (1 + t) and 1 / (1 + t / 3). Any discount between a
	one-day and a three-day half-life is a weighted sum of the two."""
	days = days.float().clamp(min=0.0)
	return torch.stack([1.0 / (1.0 + days), 1.0 / (1.0 + days / 3.0)], dim=-1)


def gap(value, best):
	"""How far a value (prior logits) stands below the best alternative, within -1..0: 0 the best itself."""
	return -torch.tanh((_like(best, value) - value).clamp(min=0.0) / GAP_SCALE)


def preference(score, best):
	"""Nullkiller2's priority against the best one, within -1..0: -tanh(ln(best / score) / 3), 0 the best itself, -1
	where it scored nothing. Its priorities are products (reward x (1 - loss) / time) spanning six orders of magnitude
	between decisions: only the ratio within one means anything, and its log is a sum, as our logits are."""
	score, best = score.float(), _like(best.float(), score)
	folds = (torch.log(best.clamp(min=1e-12)) - torch.log(score.clamp(min=1e-12))).clamp(min=0.0)
	return torch.where(score > 0, -torch.tanh(folds / PREFERENCE_SCALE), torch.full_like(score, -1.0))


def risk(ratio):
	"""Their strength over ours [...] as a danger within 0..1: r^2 / (1 + r^2), 1/2 when even, flat at 0 when they are
	nothing. One-sided (the user: pixies beside a strong hero must not become a reason by themselves)."""
	squared = ratio.float().clamp(min=0.0) ** 2
	return squared / (1.0 + squared)


def log_survival(loss):
	"""The share of the army lost on a path [...] as ln(1 - loss + 0.01), within -4.6..0.01: finite at certain death,
	its gradient never 0. It reaches only the entries of that one path, so a large value there floods only them."""
	return torch.log(1.0 - loss.float().clamp(0.0, 1.0) + SURVIVAL_FLOOR)


def settled(count):
	"""How settled a candidate is, within 0..1: 1 - 1 / (1 + the decisions in a row it has been among its hero's top 3)."""
	return 1.0 - 1.0 / (1.0 + count.float().clamp(min=0.0))


def hero_bests(move_value, move_legal, travel_value, travel_legal, slot_heroes):
	"""(each window slot's hero's best legal value [B, S], any hero's [B]) over MOVE's tiles and TRAVEL's areas, for
	gap. move_value, move_legal: [B, S, ...] by window slot; travel_value, travel_legal: [B, HO, ...] by hero;
	slot_heroes: [B, S] the hero of each window slot (features.slot_heroes: its own, or a further window's). A hero
	with nothing legal: -1e4 (its gaps are then not read: nothing of its is legal)."""
	b, s = move_value.shape[:2]
	ho = travel_value.shape[1]
	low = -1e4
	by_slot = move_value.reshape(b, s, -1).masked_fill(~move_legal.reshape(b, s, -1), low).amax(dim=-1)  # [B, S]
	by_hero = travel_value.reshape(b, ho, -1).masked_fill(~travel_legal.reshape(b, ho, -1), low).amax(dim=-1)  # [B, HO]
	owner = slot_heroes.long().clamp(0, ho - 1)
	same = owner.unsqueeze(-1) == torch.arange(ho, device=owner.device)  # [B, S, HO]
	best = torch.maximum(by_hero, torch.where(same, by_slot.unsqueeze(-1), torch.full_like(by_slot, low).unsqueeze(-1)).amax(dim=1))  # [B, HO]
	return best.gather(1, owner), best.amax(dim=1)


def candidate_rows(x):
	"""Schema 19's candidate rows in their own units: hero (-1: an empty row), dest [.., 3], dest_cell, move, travel (the
	entries' indices, -1 none), score [.., 2] (the priority at the winning tier and at the gather tier, 0 not scored),
	days, loss (0..1), danger (their strength over ours), kind, settled, flags; present [B] (the advisor ran)."""
	return {
		"hero": x["k_hero"].long() - 1, "dest": x["k_dest"].long(), "dest_cell": x["k_dest_cell"].long() - 1, "move": x["k_move"].long() - 1, "travel": x["k_travel"].long() - 1,
		"score": x["k_score"].float() / SCORE_SCALE, "days": x["k_days"].float() / DAYS_SCALE, "loss": x["k_loss"].float() / LOSS_SCALE, "danger": x["k_danger"].float() / DANGER_SCALE,
		"kind": x["k_kind"].long(), "settled": x["k_settled"].long(), "flags": x["k_flags"].long(), "present": x["g_nk_present"].bool(),
	}


def arrive_left(x):
	"""Schema 19's w_arrive_left as movement points left on arriving [B, HO, W, W], -1 where no path reaches (the wire
	holds it + 1: an old record's 0 reads as none)."""
	return x["w_arrive_left"].long() - 1


def _scatter(index, values, size):
	"""values [B, NK, ...] added into a table [B, size, ...] at index [B, NK] (rows to skip: index == size)."""
	shape = values.shape[2:]
	table = torch.zeros((values.shape[0], size + 1) + shape, device=values.device, dtype=values.dtype)
	where = index.reshape(index.shape + (1,) * len(shape)).expand_as(values)
	return table.scatter_add(1, where, values)[:, :size]


def candidate_tables(x):
	"""Nullkiller2's candidate rows (x's k_*, g_nk_present; schema 19: candidate_rows) as dense tables:

	  channels            [B, NK, F]            each candidate's channels (CHANNELS)
	  move, dest          [B, HO, W * W, F]     the channels of the best candidate whose first step (k_move) is the
	                                            entry; of the best whose destination is the cell (k_dest_cell)
	  travel              [B, HO, AREAS, F]     the same at TRAVEL entries (k_travel)
	  move_dest_cell      [B, HO, W * W]        that candidate's destination cell (k_dest_cell), -1 none: where the
	  travel_dest_cell    [B, HO, AREAS]        model gathers the destination's pair vector and own value views
	  prior_move, prior_travel                  1 + its preference against the decision's best (0..1), where the
	                                            entry's best candidate scored at the winning tier: x beta in the model
	  prior_move_fallback, prior_travel_fallback  the same from the gather tier, for a hero the winning tier says
	                                            nothing about (an INSTADEFEND turn's other heroes): x beta_fallback
	  plan_weight         [B, NK]               each candidate's weight in its hero's plan vector (they sum to 1 a
	                                            hero with a scored candidate): e^(4 x its hero-relative preference)

	The best candidate through an entry: one scored at the winning tier before a fallback before any other
	gather-tier one, then by preference, then the earlier row. One candidate an entry, so every table is a plain sum.
	An absent advisor (g_nk_present false: the dropout, deployed play without it) leaves every table 0."""
	r = candidate_rows(x)
	hero = r["hero"]
	b, nk = hero.shape
	ho, cells, areas = c.OWN_HEROES, c.WINDOW * c.WINDOW, c.MAP_LEVELS * c.TRAVEL_GRID * c.TRAVEL_GRID
	valid = (hero >= 0) & (hero < ho) & r["present"].reshape(b, 1)
	hero_index = torch.where(valid, hero, torch.full_like(hero, ho))
	score, gather_score = r["score"][..., 0] * valid, r["score"][..., 1] * valid
	flags = r["flags"]

	def best_of(values):  # (the decision's best [B], each row's hero's best [B, NK])
		return values.amax(dim=1), _hero_max(values, hero_index, ho).gather(1, hero_index)

	best_all, best_hero = best_of(score)
	gather_all, gather_hero = best_of(gather_score)
	winning = valid & (score > 0)
	fallback = valid & ~winning & (gather_score > 0) & (best_hero <= 0)  # (its hero has nothing at the winning tier)
	shadow = valid & ~winning & ~fallback & (gather_score > 0)
	p_all, p_hero = preference(score, best_all), preference(score, best_hero)
	g_all, g_hero = preference(gather_score, gather_all), preference(gather_score, gather_hero)
	# One ranking for "the best candidate through an entry": winning in 2..3, fallback in 1..2, any other gather-tier one (shadow) in 0..1, unscored 0
	key = torch.where(winning, 3.0 + p_all, torch.where(fallback, 2.0 + g_all, torch.where(shadow, 1.0 + g_all, torch.zeros_like(p_all))))
	live = torch.where(winning, 1.0 + p_all, torch.where(fallback, 1.0 + g_all, torch.zeros_like(p_all)))  # (0..1: what a prior counts)
	# Another hero's candidate for the same destination: the best live preference among them
	dest = r["dest"]
	same_dest = (dest.unsqueeze(2) == dest.unsqueeze(1)).all(dim=-1) & (hero.unsqueeze(2) != hero.unsqueeze(1)) & valid.unsqueeze(1) & valid.unsqueeze(2)
	others = (same_dest.float() * live.unsqueeze(1)).amax(dim=2)
	kind = (r["kind"].clamp(0, KINDS - 1).unsqueeze(-1) == torch.arange(KINDS, device=hero.device)).float()
	soon_later = discounts(r["days"])
	bit = lambda mask: (torch.div(flags, mask, rounding_mode="floor") % 2).float()  # noqa: E731
	scored = (key > 0).float()
	channels = torch.cat(
		[
			torch.stack(
				[
					scored, p_all, p_hero, g_all, g_hero, fallback.float(), bit(FLAG_ARRIVES), soon_later[..., 0], soon_later[..., 1], r["loss"].clamp(0.0, 1.0),
					log_survival(r["loss"]), risk(r["danger"]), bit(FLAG_UNREACHABLE), bit(FLAG_TELEPORT), bit(FLAG_CHAIN), settled(r["settled"]),
				],
				dim=-1,
			),
			kind, others.unsqueeze(-1), torch.zeros(b, nk, 1, device=hero.device),  # (mass: the entry's, set below)
		],
		dim=-1,
	) * scored.unsqueeze(-1)
	out = {"channels": channels}
	# (a hero the winning tier scored: its plan is that tier's, its other gather-tier candidates at e^-4; a fallback hero: the gather tier's)
	weight = torch.exp(PLAN_SHARPNESS * torch.where(fallback, g_hero, p_hero)) * scored
	total = torch.zeros(b, ho + 1, device=hero.device).scatter_add(1, hero_index, weight)
	out["plan_weight"] = weight / total.gather(1, hero_index).clamp(min=1e-12)
	earlier = torch.arange(nk, device=hero.device).unsqueeze(0) < torch.arange(nk, device=hero.device).unsqueeze(1)  # [k, j]: j before k
	dest_cell = r["dest_cell"]
	for name, column, size in (("move", r["move"], ho * cells), ("travel", r["travel"], ho * areas), ("dest", dest_cell, ho * cells)):
		through = (column >= 0) & (column < size) & (key > 0)
		same = (column.unsqueeze(2) == column.unsqueeze(1)) & through.unsqueeze(1) & through.unsqueeze(2)  # [B, k, j]: j goes through k's entry too
		beaten = same & ((key.unsqueeze(1) > key.unsqueeze(2)) | ((key.unsqueeze(1) == key.unsqueeze(2)) & earlier.unsqueeze(0)))
		best = through & ~beaten.any(dim=2)
		index = torch.where(best, column, torch.full_like(column, size))
		table = _scatter(index, channels * best.unsqueeze(-1), size)
		mass = _scatter(torch.where(through, column, torch.full_like(column, size)), live * through, size)
		table = torch.cat([table[..., :-1], (mass / (mass + 1.0)).unsqueeze(-1) * table[..., :1]], dim=-1)
		out[name] = table.reshape(b, ho, size // ho, F)
		if name == "dest":
			continue
		out[f"{name}_dest_cell"] = (_scatter(index, (dest_cell + 1).clamp(min=0) * best, size) - 1).reshape(b, ho, size // ho)
		out[f"prior_{name}"] = _scatter(index, (1.0 + p_all) * (best & winning), size).reshape(b, ho, size // ho)
		out[f"prior_{name}_fallback"] = _scatter(index, (1.0 + g_all) * (best & fallback), size).reshape(b, ho, size // ho)
	return out


def _hero_max(values, hero_index, ho):
	"""[B, HO + 1] the largest of values [B, NK] among each hero's rows (0 with none; hero_index == ho: no hero)."""
	same = hero_index.unsqueeze(-1) == torch.arange(ho + 1, device=hero_index.device)  # [B, NK, HO + 1]
	return (values.unsqueeze(-1) * same).amax(dim=1)
