"""Dumb decisions in campaign games, as the battle model's failure_modes: each a share 0..1 of the cases where it could
happen (nan when none came up), logged by the learner (its training games, campaign_failures/ in TensorBoard) and by
the evaluations against Nullkiller2. They should be near 0, though not always 0.

	pickup_passed_end     of the turns ended, those where one of our heroes could still take a pickup today: a legal
	                      move onto an unvisited, unguarded resource pile, chest, campfire, artifact, ... (PICKUP,
	                      TREASURE) with no enemy hero in sight (near one, walking off for a pile can cost the hero)
	pickup_passed_travel  of the TRAVEL decisions, those where the travelling hero could take such a pickup today
	troops_left           of the troop moves, those that left our troops in a garrison not a town's (a mine's): Φ prices
	                      them as in the hero, but there they guard next to nothing (rarer still than passing a pile)
"""
import numpy as np

from . import constants as c
from .records import decode_fields

HO, W, L, TG, HD, A = c.OWN_HEROES, c.WINDOW, c.MAP_LEVELS, c.TRAVEL_GRID, c.N_HOLDERS, c.ARMY_SLOTS
SAFE_CLASSES = (c.ObjectClass.PICKUP, c.ObjectClass.TREASURE)
CLASS_OF = np.array([int(c.object_class(i)) for i in range(c.OBJECT_VOCAB)] + [int(c.ObjectClass.OTHER)])
NAMES = ("pickup_passed_end", "pickup_passed_travel", "troops_left")
# The inputs safe_pickups reads, decoded CHUNK decisions at a time: the update's END and TRAVEL decisions (~a third of
# them) decoded whole at once were ~10 GB (1.7 MB a decision), the spike that stopped the PC's run (2026-10-30)
FIELDS = ("h_present", "h_owner", "w_object", "w_flags", "move_mask", "w_creature", "w_owner")
CHUNK = 256


def safe_pickups(x, b):
	"""[HO]: each of our heroes has a pickup it may take today at decision b, with no enemy hero in sight."""
	if (x["h_present"][b, HO:] & (x["h_owner"][b, HO:] != c.OWNER_SELF)).any():
		return np.zeros(HO, dtype=bool)
	obj, flags = x["w_object"][b], x["w_flags"][b]
	kind = CLASS_OF[np.clip(obj, 0, c.OBJECT_VOCAB)]
	bit = lambda value: (flags // value) % 2 == 1  # noqa: E731
	ok = (
		x["move_mask"][b] & np.isin(kind, SAFE_CLASSES) & (obj >= 0) & bit(c.TILE_VISITABLE) & ~bit(c.TILE_VISITED_BY_PLAYER) & ~bit(c.TILE_VISITED_BY_HERO)
		& ~bit(c.TILE_GUARDED) & (x["w_creature"][b] < 0) & (x["w_owner"][b] != c.OWNER_SELF)
	)
	return ok.reshape(HO, -1).any(axis=1)


def failure_counts(decisions):
	"""{name: (dumb decisions, decisions where it could happen)} over one player's (or several players') decisions."""
	counts = {name: [0, 0] for name in NAMES}
	ends = [d for d in decisions if d.kind == c.DecisionKind.TURN and d.action_type == c.ActionType.END]
	travels = [d for d in decisions if d.kind == c.DecisionKind.TURN and d.action_type == c.ActionType.TRAVEL]
	for name, chosen in (("pickup_passed_end", ends), ("pickup_passed_travel", travels)):
		if not chosen:
			continue
		for start in range(0, len(chosen), CHUNK):
			part = chosen[start : start + CHUNK]
			x = decode_fields(part, FIELDS)
			for b, d in enumerate(part):
				pickups = safe_pickups(x, b)
				counts[name][0] += bool(pickups.any() if name == "pickup_passed_end" else pickups[min(d.index // (L * TG * TG), HO - 1)])
				counts[name][1] += 1
	for d in decisions:
		if d.action_type == c.ActionType.ARMY and d.index >= 0:
			source, _, target, _ = np.unravel_index(d.index, (HD, A, HD, c.N_ARMY_AMOUNTS))
			counts["troops_left"][0] += bool(target == c.OTHER_ARMY and source != c.OTHER_ARMY)
			counts["troops_left"][1] += 1
	return {name: tuple(v) for name, v in counts.items()}


def failure_modes(decisions=None, counts=None):
	"""{name: share} of failure_counts (given, or of the decisions)."""
	counts = counts if counts is not None else failure_counts(decisions)
	return {name: (hit / total if total else float("nan")) for name, (hit, total) in counts.items()}
