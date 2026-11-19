"""The critic's NK2 rows as game-day groups (e5 for the user, 2026-11-18): a whole game's samples gave the critic its few
outcomes a pass, memorised inside it; a group is one game-day instead. Since 2026-11-19 the games come from the NK2 stream
(nk2_stream.py, supply.Games' kind "nk2": made on the fly, each used once); the static pool, the disk store, its intake
and the own replay ring are gone (the user: "there shouldn't be a pool at all!").

A group is one game-day: one row of every player who has a decision that day (one of the day's kept decisions, drawn),
trained, with its labels from the real outcomes alone (trust 0); and each of those rows' rivals' rows (privileged_rivals':
a rival's last decision on a day BEFORE ours, so nothing after our decision is in it), carried as rival-only rows (their
states feed the privileged head; no loss on them). Rival indices are local to the group. assemble: groups into
train_critic's pool dict, "train" naming the trained rows."""
from collections import defaultdict

import numpy as np

from .empire import EmpireConfig


def game_groups(game, game_id, config, empire=EmpireConfig(), labels=None, rng=None, keep=1):
	"""A game's groups, one a game-day (see the module). config: share_target and outcome_share_min; labels:
	labels.LabelConfig for share_target "share". keep 2 (our own games' replay, e5 2026-11-18): each player-day's TWO
	drawn decisions kept as its choices, one drawn at each use (assemble); keep 1: the one drawn here."""
	from .labels import LabelConfig, end_labels, end_targets, growth_labels, label, ladder_targets, share_change_labels
	from .learner import privileged_rivals

	rng = rng if rng is not None else np.random.default_rng()
	n = len(game.decisions)
	if not n or not game.ledgers:
		return []
	t = share_change_labels(game, empire) if config.share_target == "change" else label(game, labels or LabelConfig(), empire)[0]
	z, real = growth_labels(game, empire), end_labels(game, empire)
	graded, grown = ladder_targets(game, t, z, {}, {}, config.outcome_share_min)
	ends = end_targets(game, real, {}, 0.0, config.outcome_share_min)
	rivals = privileged_rivals(game, np.ones(n, dtype=bool), 0, empire)
	by_day = defaultdict(lambda: defaultdict(list))
	for i, d in enumerate(game.decisions):
		by_day[d.day][d.color].append(i)
	groups = []
	for day in sorted(by_day):
		picks = [sorted(int(i) for i in rng.choice(rows, size=min(keep, len(rows)), replace=False)) for _, rows in sorted(by_day[day].items())]
		trained = [i for pick in picks for i in pick]
		rival_rows = sorted({int(r) for i in trained for r in rivals["rivals"][i] if r >= 0} - set(trained))
		order = trained + rival_rows
		local = {row: k for k, row in enumerate(order)}
		r = rivals["rivals"][trained]
		if keep > 1:  # (each player's choices: indices among the trained rows, one drawn a use)
			choices, k = [], 0
			for pick in picks:
				choices.append(list(range(k, k + len(pick))))
				k += len(pick)
		groups.append({
			"game": game_id, "day": int(day), "rows": [game.decisions[i] for i in order], "trained": len(trained),
			"targets": np.asarray(graded[trained], np.float32), "growth": np.asarray(grown[trained], np.float32), "end": np.asarray(ends[trained], np.float32),
			"share_logit": np.asarray(rivals["share_logit"][trained], np.float32), "present": np.asarray(rivals["present"][trained], np.float32),
			"rivals": np.where(r >= 0, np.vectorize(lambda x: local.get(int(x), -1))(np.maximum(r, 0)), -1).astype(np.int64),
		})
		if keep > 1:
			groups[-1]["choices"] = choices
	return groups


def choose(group, rng):
	"""A group with choices (keep 2) as one with one trained row a player, drawn now; others as they are."""
	if "choices" not in group:
		return group
	pick = [int(rng.choice(c)) for c in group["choices"]]
	rival_rows = sorted({int(r) for i in pick for r in group["rivals"][i] if r >= 0})
	order = pick + rival_rows
	local = {row: j for j, row in enumerate(order)}
	r = group["rivals"][pick]
	return {
		"game": group["game"], "day": group["day"], "rows": [group["rows"][i] for i in order], "trained": len(pick),
		**{key: group[key][pick] for key in ("targets", "growth", "end", "share_logit", "present")},
		"rivals": np.where(r >= 0, np.vectorize(lambda x: local.get(int(x), -1))(np.maximum(r, 0)), -1).astype(np.int64),
	}


def assemble(groups, rivals_width=None, rng=None):
	"""Groups into train_critic's pool dict: every row (trained, then rival-only, group by group), each row's rivals as
	indices into it (-1 none), targets NaN on rival-only rows, and "train": the trained rows' indices. A group with
	choices (keep 2) gives one of each player's, drawn with rng."""
	from .model_v2 import PRIVILEGED_RIVALS

	width = rivals_width or PRIVILEGED_RIVALS
	rng = rng if rng is not None else np.random.default_rng()
	decisions, train = [], []
	parts = {"rivals": [], "present": [], "share_logit": [], "targets": [], "growth": [], "end": []}
	for g in (choose(g, rng) for g in groups):
		base, k, m = len(decisions), g["trained"], len(g["rows"])
		decisions += g["rows"]
		train += range(base, base + k)
		rivals = np.full((m, width), -1, np.int64)
		rivals[:k] = np.where(g["rivals"] >= 0, g["rivals"] + base, -1)
		present = np.zeros((m, width), np.float32)
		present[:k] = g["present"]
		share = np.zeros(m, np.float32)
		share[:k] = g["share_logit"]
		targets = np.full((m,) + g["targets"].shape[1:], np.nan, np.float32)
		targets[:k] = g["targets"]
		growth = np.full((m,) + g["growth"].shape[1:], np.nan, np.float32)
		growth[:k] = g["growth"]
		end = np.full(m, np.nan, np.float32)
		end[:k] = g["end"]
		for name, value in (("rivals", rivals), ("present", present), ("share_logit", share), ("targets", targets), ("growth", growth), ("end", end)):
			parts[name].append(value)
	if not decisions:
		return None
	return {"decisions": decisions, "train": np.asarray(train, np.int64), **{name: np.concatenate(values) for name, values in parts.items()}}
