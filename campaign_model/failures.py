"""Dumb decisions in campaign games, as the battle model's failure_modes: each a share 0..1 of the cases where it could
happen (nan when none came up), logged by the learner (its training games, campaign_failures/ in TensorBoard) and by
the evaluations against Nullkiller2. They should be near 0, though not always 0.

	pickup_passed_end     of the turns ended, those where one of our heroes could still take a pickup today: a legal
	                      move onto an unvisited, unguarded resource pile, chest, campfire, artifact, ... (PICKUP,
	                      TREASURE) with no enemy hero in sight (near one, walking off for a pile can cost the hero)
	pickup_passed_travel  of the TRAVEL decisions, those where the travelling hero could take such a pickup today
	troops_left           of the dialogs with a garrison not a town's (a mine's) where we moved troops into it, those that
	                      closed with more of our creatures there than when they opened: Φ prices them as in the hero,
	                      but there they guard next to nothing. Until 2026-11-06 each move into it counted, of all troop
	                      moves: in 39 games all 17 such moves were taken back within their dialog, none left
	exchange_shuffle      of those dialogs, the ones where a move into it was taken back (a creature in and out again:
	                      decisions wasted, nothing left; 11 of 11 dialogs in those 39 games)
	own_hero_moves        of our MOVEs, those onto a hero of ours (a meeting: each opens an exchange). Nullkiller2's own
	                      play 1.7% (2,678 moves); joint7 at v2481 54.7% (2026-11-09 audit: its advice loops on a
	                      phantom delivery and on meetings already made)
	repeat_meetings       of those meetings, the ones repeating a meeting of the same two heroes that day (70.5%)
	meeting_movement      of the movement points our heroes spend on MOVE and TRAVEL, the share spent stepping onto a
	                      hero of ours (25.7% at v2481)
	town_movement         the share spent stepping into a town of ours (18.7%: a visit takes nothing for us, GATHER
	                      being untaught)
	idle_movement         of the movement points our heroes have in a day, the share still unused when we end the turn
	                      (heroes not in a garrison): 0.095 in 8 joint7 games at v2479-2481, 0.269 in 6 at ~v2625 after
	                      the meeting fixes (2026-11-10: the meetings had burned it; END took their place, not the
	                      targets in reach, which the model sees: the 7-day credit pays END as well as a small visit)
	idle_movement_town    the same for our heroes standing in a town of ours (0.41 before, 0.65 after)
	unmoved_heroes        of our heroes at the turn's end, those that did not move that day (0.048 before, 0.195 after)
	unmoved_heroes_town   the same for those in a town of ours (0.20 before, 0.44 after)
"""
from collections import defaultdict

import numpy as np

from . import constants as c
from .records import decode_fields

HO, W, L, TG, HD, A = c.OWN_HEROES, c.WINDOW, c.MAP_LEVELS, c.TRAVEL_GRID, c.N_HOLDERS, c.ARMY_SLOTS
SAFE_CLASSES = (c.ObjectClass.PICKUP, c.ObjectClass.TREASURE)
CLASS_OF = np.array([int(c.object_class(i)) for i in range(c.OBJECT_VOCAB)] + [int(c.ObjectClass.OTHER)])
NAMES = (
	"pickup_passed_end", "pickup_passed_travel", "troops_left", "exchange_shuffle", "own_hero_moves", "repeat_meetings", "meeting_movement", "town_movement",
	"idle_movement", "idle_movement_town", "unmoved_heroes", "unmoved_heroes_town",
)
IDLE = ("idle_movement", "idle_movement_town", "unmoved_heroes", "unmoved_heroes_town")
# The inputs safe_pickups reads, decoded CHUNK decisions at a time: the update's END and TRAVEL decisions (~a third of
# them) decoded whole at once were ~10 GB (1.7 MB a decision), the spike that stopped the PC's run (2026-10-30)
FIELDS = ("h_present", "h_owner", "w_object", "w_flags", "move_mask", "w_creature", "w_owner")
CHUNK = 256
# What the turn's end reads: our heroes' movement left and their day's, where they stand
END_FIELDS = ("h_present", "h_owner", "h_garrisoned", "h_town", "h_move", "h_move_max")
# What the meetings read: our heroes' places, identities (h_type, unique a hero) and movement left, our towns' places
MOVE_FIELDS = ("h_present", "h_owner", "h_pos", "h_type", "h_move", "h_garrisoned", "t_present", "t_owner", "t_pos")
R = c.WINDOW_RADIUS


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


def failure_counts(decisions, moves_limit=None):
	"""{name: (dumb decisions, decisions where it could happen)} over one player's (or several players') decisions;
	moves_limit: the meetings' counts over the first players' moves to about that many (movement_counts)."""
	counts = {name: [0, 0] for name in NAMES}
	parts = (
		(("pickup_passed_end", "pickup_passed_travel"), lambda: pickup_counts(decisions, counts)),
		(("troops_left", "exchange_shuffle"), lambda: garrison_counts(decisions, counts)),
		(("own_hero_moves", "repeat_meetings", "meeting_movement", "town_movement"), lambda: movement_counts(decisions, counts, moves_limit)),
		(IDLE, lambda: idle_counts(decisions, counts, moves_limit)),
	)
	for names, part in parts:
		# (each part a diagnostic of its own, 2026-11-03: never the update's end, nor an evaluation game's, which an
		# error here would leave out of the score; its shares nan)
		try:
			part()
		except Exception as error:  # noqa: BLE001
			print(f"failures: the counts of {', '.join(names)} failed: {error!r}", flush=True)
			for name in names:
				counts[name] = [0, 0]
	return {name: tuple(v) for name, v in counts.items()}


def pickup_counts(decisions, counts):
	"""Adds pickup_passed_end and pickup_passed_travel: the turns ended, and the travels, with a safe pickup in reach
	(safe_pickups)."""
	ends = [d for d in decisions if d.kind == c.DecisionKind.TURN and d.action_type == c.ActionType.END]
	travels = [d for d in decisions if d.kind == c.DecisionKind.TURN and d.action_type == c.ActionType.TRAVEL]
	for name, chosen in (("pickup_passed_end", ends), ("pickup_passed_travel", travels)):
		for start in range(0, len(chosen), CHUNK):
			part = chosen[start : start + CHUNK]
			x = decode_fields(part, FIELDS)
			for b, d in enumerate(part):
				pickups = safe_pickups(x, b)
				counts[name][0] += bool(pickups.any() if name == "pickup_passed_end" else pickups[min(d.index // (L * TG * TG), HO - 1)])
				counts[name][1] += 1


def garrison_counts(decisions, counts):
	"""Adds troops_left and exchange_shuffle over the dialogs with a garrison not a town's (garrison_dialogs)."""
	for first, last, gives, takes in garrison_dialogs(decisions):
		x = decode_fields([first, last], ("x_count",))
		counts["troops_left"][0] += bool(x["x_count"][1].sum() > x["x_count"][0].sum())
		counts["troops_left"][1] += 1
		counts["exchange_shuffle"][0] += bool(takes > 0)
		counts["exchange_shuffle"][1] += 1


def idle_counts(decisions, counts, limit=None):
	"""Adds IDLE's counts: at each turn decision we ended (END), our heroes (not a town's garrison hero) and the
	movement each still has of its day's (h_move of h_move_max), in a town of ours (h_town) or not. limit: about that
	many ends (an update's first; a share needs few)."""
	ends = [d for d in decisions if d.kind == c.DecisionKind.TURN and d.action_type == c.ActionType.END]
	if limit is not None:
		ends = ends[:limit]
	for start in range(0, len(ends), CHUNK):
		x = decode_fields(ends[start : start + CHUNK], END_FIELDS)
		ours = x["h_present"][:, :HO] & (x["h_owner"][:, :HO] == c.OWNER_SELF) & ~x["h_garrisoned"][:, :HO].astype(bool)
		town = ours & (x["h_town"][:, :HO] >= 0)
		left, day = x["h_move"][:, :HO].astype(np.int64), np.maximum(x["h_move_max"][:, :HO].astype(np.int64), 1)
		unmoved = left >= day
		for name, rows in (("idle_movement", ours), ("idle_movement_town", town)):
			counts[name][0] += int(np.minimum(left, day)[rows].sum())
			counts[name][1] += int(day[rows].sum())
		for name, rows in (("unmoved_heroes", ours), ("unmoved_heroes_town", town)):
			counts[name][0] += int((unmoved & rows).sum())
			counts[name][1] += int(rows.sum())


def movement_counts(decisions, counts, limit=None):
	"""Adds own_hero_moves, repeat_meetings, meeting_movement and town_movement to counts: each player's MOVE and TRAVEL
	turn decisions in order (a day going back: the next game's), what each spent of its hero's movement (its points
	then less at that player's next decision of the day, the hero found by h_type), where each MOVE stepped. limit:
	whole players' games only until about that many moves (an update's ~25k took ~9 s on the Mac; a share needs few)."""
	streams, game, last = defaultdict(list), defaultdict(int), {}
	for d in decisions:
		if d.color in last and d.day < last[d.color]:
			game[d.color] += 1
		last[d.color] = d.day
		streams[(d.color, game[d.color])].append(d)
	steps = []  # (stream, the move, the next decision of its day or None)
	for key, stream in streams.items():
		if limit is not None and len(steps) >= limit:
			break
		for p, d in enumerate(stream):
			if d.kind == c.DecisionKind.TURN and d.action_type in (c.ActionType.MOVE, c.ActionType.TRAVEL) and d.index >= 0:
				nxt = stream[p + 1] if p + 1 < len(stream) and stream[p + 1].day == d.day else None
				steps.append((key, d, nxt))
	met = defaultdict(set)  # (stream, day) -> the pairs of heroes met
	for start in range(0, len(steps), CHUNK):
		part = steps[start : start + CHUNK]
		rows = [d for _, d, _ in part] + [nxt for _, _, nxt in part if nxt is not None]
		x = decode_fields(rows, MOVE_FIELDS)
		row_of = {}
		for i, d in enumerate(rows):
			row_of.setdefault(id(d), i)
		for key, d, nxt in part:
			b = row_of[id(d)]
			move = d.action_type == c.ActionType.MOVE
			h = min(d.index // (W * W) if move else d.index // (L * TG * TG), HO - 1)
			spent = 0
			if nxt is not None:
				later = row_of[id(nxt)]
				same = np.flatnonzero(x["h_present"][later, :HO] & (x["h_type"][later, :HO] == x["h_type"][b, h]))
				if len(same):
					spent = max(0, int(x["h_move"][b, h]) - int(x["h_move"][later, same[0]]))
			meeting = town = False
			if move:
				rest = d.index % (W * W)
				target = np.array([x["h_pos"][b, h, 0] + rest % W - R, x["h_pos"][b, h, 1] + rest // W - R, x["h_pos"][b, h, 2]])
				# (a hero in a town's garrison shares its tile and meets no one: a step there is a visit of the town, 2026-11-09)
				ours = x["h_present"][b, :HO] & (x["h_owner"][b, :HO] == c.OWNER_SELF) & (np.arange(HO) != h) & ~x["h_garrisoned"][b, :HO].astype(bool)
				there = np.flatnonzero(ours & (x["h_pos"][b, :HO] == target).all(axis=-1))
				meeting = len(there) > 0
				towns = x["t_present"][b, : c.OWN_TOWNS] & (x["t_owner"][b, : c.OWN_TOWNS] == c.OWNER_SELF)
				town = not meeting and bool((towns & (x["t_pos"][b, : c.OWN_TOWNS] == target).all(axis=-1)).any())
				counts["own_hero_moves"][0] += meeting
				counts["own_hero_moves"][1] += 1
				if meeting:
					pair = frozenset((int(x["h_type"][b, h]), int(x["h_type"][b, there[0]])))
					counts["repeat_meetings"][0] += pair in met[(key, d.day)]
					counts["repeat_meetings"][1] += 1
					met[(key, d.day)].add(pair)
			for name, hit in (("meeting_movement", meeting), ("town_movement", town)):
				counts[name][0] += spent if hit else 0
				counts[name][1] += spent


def garrison_dialogs(decisions):
	"""[(the dialog's first decision, its last, our moves into the other army, moves back out of it)] of each exchange
	dialog (a player's consecutive EXCHANGE decisions) in which we moved troops into a garrison's other army (holder
	OTHER_ARMY: a mine's, not a town's); decisions in their order, one game's or several games' one after the other."""
	out, open_ = [], {}  # color -> [first, last, gives, takes, day]
	def close(color):
		dialog = open_.pop(color, None)
		if dialog is not None and dialog[2] > 0:
			out.append(tuple(dialog[:4]))
	for d in decisions:
		dialog = open_.get(d.color)
		if dialog is not None and (d.kind != c.DecisionKind.EXCHANGE or d.day < dialog[4]):  # (closed, or a next game's)
			close(d.color)
			dialog = None
		if d.kind != c.DecisionKind.EXCHANGE:
			continue
		if dialog is None:
			dialog = open_[d.color] = [d, d, 0, 0, d.day]
		dialog[1] = d
		if d.action_type == c.ActionType.ARMY and d.index >= 0:
			source, _, target, _ = np.unravel_index(d.index, (HD, A, HD, c.N_ARMY_AMOUNTS))
			dialog[2] += bool(target == c.OTHER_ARMY and source != c.OTHER_ARMY)
			dialog[3] += bool(source == c.OTHER_ARMY and target != c.OTHER_ARMY)
	for color in list(open_):
		close(color)
	return out


def failure_modes(decisions=None, counts=None, moves_limit=None):
	"""{name: share} of failure_counts (given, or of the decisions)."""
	counts = counts if counts is not None else failure_counts(decisions, moves_limit)
	return {name: (hit / total if total else float("nan")) for name, (hit, total) in counts.items()}
