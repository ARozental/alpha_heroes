"""Nullkiller2's advice as the teacher's targets over the turn, not action by action (config.nk2_advice; the user,
2026-11-02: the order of a turn's actions should not matter, and a target must have been visible when it is used): at
each decision the teacher's entries are its advice's own (Decision.advice) and every other entry of ours that follows
it there (teacher_sets); train.micro_loss takes the cross-entropy of their summed probability.

	at the decision  recorded by Advice.cpp (Decision.accept, from the plan Nullkiller2 made there): each hero's MOVE to
	                 any tile of its path, a short step along it or a long one alike, its TRAVEL to an area the path
	                 reaches in a day or more; at an exchange every troop move toward the army it wants
	pending          a town action it advised earlier that turn (build, recruit, hire, upgrade, trade, buy) that we have
	                 not done since stays the teacher's at the turn's later decisions, at any amount; once it is not
	                 legal its probability is 0 and it adds nothing. Never at a decision before the advice: what
	                 Nullkiller2 named later may rest on what we had not seen yet
"""
import numpy as np

from . import constants as c
from .schema import SPECS

A = c.ActionType
SET_SIZE = 16  # a decision's entries at most beyond the advice's own (the plan's steps first, then the pending town actions; 11 at most in 2,473 decisions of a smoke run, 2026-11-02)
_RECRUIT = SPECS["recruit_mask"].shape  # (town, level, amount, into)
_TRADE = SPECS["trade_mask"].shape  # (sell, buy, amount)


def town_item(action, index):
	"""(identity, entries) of a town action: the same item whatever its amount, each of its table's entries; None for any
	other action."""
	if action == A.RECRUIT:
		town, level, _, into = np.unravel_index(index, _RECRUIT)
		return (int(action), int(town), int(level), int(into)), [int(np.ravel_multi_index((town, level, a, into), _RECRUIT)) for a in range(_RECRUIT[2])]
	if action == A.TRADE:
		sell, buy, _ = np.unravel_index(index, _TRADE)
		return (int(action), int(sell), int(buy)), [int(np.ravel_multi_index((sell, buy, a), _TRADE)) for a in range(_TRADE[2])]
	if action in (A.BUILD, A.HIRE, A.UPGRADE, A.BUY):
		return (int(action), int(index)), [int(index)]
	return None


def teacher_sets(decisions, size=SET_SIZE):
	"""Each decision's teacher entries beyond its advice's own: (types [N, size], indices [N, size], valid [N, size]).
	decisions: a game's, in each player's order, all of them (what we did counts, usable or not)."""
	n = len(decisions)
	types, indices, valid = np.zeros((n, size), np.int64), np.zeros((n, size), np.int64), np.zeros((n, size), bool)
	pending = {}  # (color, day) -> {identity: (action, entries)}: the town actions advised earlier that turn, not done
	for i, d in enumerate(decisions):
		accept = getattr(d, "accept", None) or ()
		extra = [(int(accept[k]), int(accept[k + 1])) for k in range(0, len(accept) - 1, 2)]
		if d.kind == c.DecisionKind.TURN:
			todo = pending.setdefault((d.color, d.day), {})
			extra += [(action, entry) for action, entries in todo.values() for entry in entries]
			ours = town_item(d.action_type, d.index) if d.action_type >= 0 else None
			if ours is not None:
				todo.pop(ours[0], None)
			named = town_item(d.advice[0], d.advice[1]) if d.advice is not None else None
			if named is not None and (ours is None or ours[0] != named[0]):
				todo[named[0]] = (int(d.advice[0]), named[1])  # (the teacher's from the next decision on)
		own = (int(d.advice[0]), int(d.advice[1])) if d.advice is not None else None
		seen, k = set(), 0
		for entry in extra:
			if entry == own or entry in seen:
				continue
			if k == size:
				break
			seen.add(entry)
			types[i, k], indices[i, k], valid[i, k] = entry[0], entry[1], True
			k += 1
	return types, indices, valid
