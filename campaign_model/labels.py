"""Labels: what each decision of a game was worth, from the empire values (empire.py) of the days after it.

The user's rules (2026-10-27, 2026-10-28):
- The game is played to its end (or a day cap). A decision on day d is labelled with the player's share of all
  empires' value at the end of days d + 2, d + 5 and d + 7 (VALUE_OFFSETS): never the same day's or the next day's,
  so what a day's moves leave open to the enemy's next turn is learned by the value heads, not priced.
- A share after the player's elimination, or after the game ended, is its share at that end (0 for the eliminated,
  1 for the winner of a 2-player game). Past a day cap there is no label (nan): the cap is no outcome, and the share
  there taught greed before it (the model sees the day).
- Own growth (the growth heads, helpers the policy's credit never reads): the log of our empire's value at the end of
  day d + k over its value at the decision, within x10 and /10 (GROWTH_CAP). Only our side: none of an opponent's luck
  that never touches us, and in a 2-player game two signals where the shares, mirror images, give one.
- Each head past the first is graded (ladder_targets) against the next shorter one's forecast of the same day, made
  later: V7 against V5 two days on, V5 against V2 three days on, the growth heads alike, mixed with the real outcome
  (all of it while that head is not trusted, at least outcome_share_min of it once it is). A day's decisions share
  their label's luck and labels days apart share most of it (the 7-day label, one day apart: 6 days of 7); graded
  against a later forecast, a label carries 2 or 3 days of luck, not 7 (a linear toy: ~1.1-3.4x the data's worth,
  by how long an edge lasts). The forecast made later is a pure forecast of the same day: on average it is what can
  be forecast now, so a head graded against it learns the same forecast, with less luck in it.
- The end head V_end (the user, 2026-10-28: a 7-day horizon never learns what pays later: an Estates' gold beats an
  Offense or a Wisdom whose worth comes weeks on): the player's share at the game's end, where it ended (one player
  left) or reached its day cap, drawn per game so that past days_min every day is as likely to be the last
  (config.days_min: the cap is then an end the model can't see coming, not a cutoff to plan around). Graded against
  itself at the player's first turn decision the next day, mixed with the real end by the trust in it, as the ladder.
  Its credit (end_advantages) takes the policy over from the 7-day one as V_end proves itself (learner.py).
- Every turn decision has a small price (DECISION_PRICE for each turn decision still to come that day, itself
  included), so a turn does not drag on with moves that do nothing; the AI still stops at MAX_DECISIONS_PER_DAY.
Questions the engine asks are answered without a price: they are not optional.

The end of day d is the ledger taken at the start of day d + 1 (after the night's income and growth).
"""
import bisect
import math
from dataclasses import dataclass

import numpy as np

from . import constants as c
from .empire import STATUS_INGAME, STATUS_LOST, EmpireConfig, empire_value, shares

COLORS = ("RED", "BLUE", "TAN", "GREEN", "ORANGE", "PURPLE", "TEAL", "PINK")


@dataclass(frozen=True)
class LabelConfig:
	weights: tuple = (1 / 3, 1 / 3, 1 / 3)  # of the shares VALUE_OFFSETS days on
	decision_price: float = 1e-4  # share points per turn decision


GROWTH_CAP = math.log(10.0)  # the growth heads' range: our empire's value x10 to /10 of its value at the decision
PHI_FLOOR = 1000.0  # gold: growth counts from at least this much (an empire below it is all but gone)


def tables(game, empire=EmpireConfig()):
	"""({color name: {day: share at the end of that day}}, {color name: {day: its empire value then}}, the last day with
	a snapshot after it, and whether the game ended there: its last snapshot has at most one player still in the game.
	Otherwise it stopped at the day cap, or was stopped (a player knocked out, then the time limit: 2026-10-28 review)."""
	share, value = {}, {}

	def note(snapshot, day):
		for color, s in shares(snapshot, empire).items():
			share.setdefault(COLORS[color], {})[day] = s
		for p in snapshot["players"]:
			value.setdefault(COLORS[p["color"]], {})[day] = max(empire_value(p, empire)["total"], 0.0)

	for day, snapshot in sorted(game.ledgers.items()):
		note(snapshot, day - 1)
	last = max(game.ledgers) - 1 if game.ledgers else 0
	latest = game.ledgers[max(game.ledgers)] if game.ledgers else None
	# Snapshots taken when a player won or lost, after the last daily one: the state the game stopped in (a player
	# knocked out of a game that goes on shows as out of it in the next daily snapshot anyway)
	for snapshot in sorted(game.final, key=lambda s: s["day"]):
		if snapshot["day"] > last:
			last, latest = snapshot["day"], snapshot
			note(snapshot, last)
	ended = latest is not None and sum(p.get("status", STATUS_INGAME) == STATUS_INGAME for p in latest["players"]) <= 1
	return share, value, last, ended


def share_table(game, empire=EmpireConfig()):
	"""{color name: {day: share at the end of that day}}, and the last day with a snapshot after it."""
	share, _, last, _ = tables(game, empire)
	return share, last


def share_at(table, color, day, last):
	"""The share (or value: tables') at the end of `day`, or at the last day before it that has one (the end of the game
	or the cap)."""
	days = table.get(color, {})
	known = [d for d in days if d <= min(day, last)]
	return days[max(known)] if known else 0.0


def label(game, config=LabelConfig(), empire=EmpireConfig()):
	"""Per decision (game.decisions' order): the real shares [N, N_VALUE_OFFSETS] (nan past the last day played unless
	the game ended there), the label (their weighted mix) [N] and the decision price [N]."""
	table, _, last, ended = tables(game, empire)
	targets = np.array(
		[[share_at(table, d.color, d.day + k, last) if ended or d.day + k <= last else np.nan for k in c.VALUE_OFFSETS] for d in game.decisions], dtype=np.float32
	).reshape(-1, c.N_VALUE_OFFSETS)
	mixed = targets @ np.asarray(config.weights, dtype=np.float32)
	# How many turn decisions each player made on each day: the price counts those still to come
	per_day = {}
	for d in game.decisions:
		if d.kind == c.DecisionKind.TURN:
			per_day[(d.color, d.day)] = max(per_day.get((d.color, d.day), 0), d.today + 1)
	price = np.array(
		[config.decision_price * (per_day[(d.color, d.day)] - d.today) if d.kind == c.DecisionKind.TURN else 0.0 for d in game.decisions], dtype=np.float32
	)
	return targets, mixed, price


def growth_labels(game, empire=EmpireConfig()):
	"""Per decision: the log of our empire's value at the end of day d + k [N, N_VALUE_OFFSETS] (of at least 1 gold:
	once knocked out, far below the cap), nan where label has none. The growth heads learn it over our value at the
	decision, which the model computes from the decision's inputs (train.ppo_update)."""
	_, values, last, ended = tables(game, empire)
	return np.array(
		[[math.log(max(share_at(values, d.color, d.day + k, last), 1.0)) if ended or d.day + k <= last else np.nan for k in c.VALUE_OFFSETS] for d in game.decisions],
		dtype=np.float32,
	).reshape(-1, c.N_VALUE_OFFSETS)


def end_labels(game, empire=EmpireConfig()):
	"""Per decision: the player's share at the game's end [N]. The game's end: where one player was left, or where it
	reached its day cap (game.days); a player knocked out ended with 0 whenever the game stopped; nan for the others of
	a game stopped before either (the time limit: its last day is no end)."""
	table, _, last, ended = tables(game, empire)
	complete = ended or (game.days > 0 and last >= game.days)
	out = {COLORS[p["color"]] for s in [*game.ledgers.values(), *game.final] for p in s["players"] if p.get("status") == STATUS_LOST}
	return np.array([0.0 if d.color in out else share_at(table, d.color, last, last) if complete else np.nan for d in game.decisions], dtype=np.float32)


def end_targets(game, real, starts, trust, outcome_share_min=0.25):
	"""The end head's grade [N]: its own forecast at the player's first turn decision the next day (starts: day_starts'),
	mixed with the real end (real: end_labels'), whose weight is 1 - (1 - outcome_share_min) x the trust in V_end, as
	ladder_targets'; the forecast alone where the end is unknown (the time limit), the real end alone on the player's
	last day."""
	graded = real.copy()
	w = 1 - (1 - outcome_share_min) * trust
	for i, d in enumerate(game.decisions):
		start = starts.get((d.color, d.day + 1))
		if start is not None:
			graded[i] = start[4] if np.isnan(real[i]) else (1 - w) * start[4] + w * real[i]
	return graded


def end_steps(game, ahead=c.VALUE_OFFSETS[-1]):
	"""Per decision (game.decisions' order): the step's index, the player's next turn decision (that day or later), and
	the week's, its first turn decision `ahead` days on or later; -1 where there is none (the game's end stands)."""
	decisions = game.decisions
	n = len(decisions)
	step, week, following, firsts = np.full(n, -1), np.full(n, -1), {}, {}
	for i in reversed(range(n)):
		d = decisions[i]
		step[i] = following.get(d.color, -1)
		if d.kind == c.DecisionKind.TURN:
			following[d.color] = i
			firsts[(d.color, d.day)] = i
	days = {}
	for color, day in sorted(firsts):
		days.setdefault(color, []).append(day)
	for i, d in enumerate(decisions):
		later = days.get(d.color, [])
		k = bisect.bisect_left(later, d.day + ahead)
		if k < len(later):
			week[i] = firsts[(d.color, later[k])]
	return step, week


def end_advantages(values, step, week, real, price, turn, decision_price, outcome):
	"""The end head's credit [N]: how much better than expected each decision did for the player's share at the game's
	end. values: V_end at each decision (the model as the update starts: train.ppo_update); step, week: end_steps'
	indices into the same rows; real: end_labels'; price: label's (the turn decisions still to come that day).
	  td    V_end at the player's next turn decision (the real end where there is none) - V_end here, one turn
	        decision's price: little noise, as good as V_end
	  week  V_end at the player's first turn decision 7 days on (the real end where there is none) - V_end here, the
	        day's price: a week of what then happened, so td's blind spots within a week are corrected
	advantage = (1 - outcome) td + outcome week; td alone where the week's is unknown, nan where neither is (a game
	stopped by the time limit: the end of it). No real end beyond a week: the whole game's luck would drown the step
	(what V_end learns from the real end, the credit gets through it)."""
	there = np.where(step >= 0, values[np.maximum(step, 0)], real)
	td = there - values - np.where(turn, decision_price, 0.0)
	later = np.where(week >= 0, values[np.maximum(week, 0)], real)
	weekly = later - values - price
	return np.where(np.isfinite(weekly), (1 - outcome) * td + outcome * weekly, td).astype(np.float32)


def blend_credit(seven, end, weight):
	"""The policy's advantage [N]: (1 - weight) x the 7-day credit (horizon_advantages') + weight x the end credit
	(end_advantages'); where one is unknown (nan) it adds nothing (the 7-day one within a week of a day cap, which has
	no share 7 days on), and nan where both are."""
	known = np.isfinite(seven) | np.isfinite(end)
	mixed = np.where(np.isfinite(seven), (1 - weight) * seven, 0.0) + np.where(np.isfinite(end), weight * end, 0.0)
	return np.where(known, mixed, np.nan).astype(np.float32)


def growth_scale(log_growth):
	"""Log growth to the growth heads' 0..1 (their cross-entropy's target): /10 or less 0, no growth 1/2, x10 or more 1."""
	return (np.clip(log_growth, -GROWTH_CAP, GROWTH_CAP) + GROWTH_CAP) / (2 * GROWTH_CAP)


def ladder_targets(game, real, growth, starts, trust, outcome_share_min=0.25):
	"""The heads' grades for game.decisions: shares [N, N_VALUE_OFFSETS] and growth (as growth_labels'). Each head past
	the first is graded against the next shorter one at the player's first turn decision of the day where they meet,
	the same day's forecast made later (V5 against V2 three days on, V7 against V5 two days on), mixed with the real
	outcome (real: label's shares; growth: growth_labels'), whose weight is 1 - (1 - outcome_share_min) x the trust in
	the shorter head (trust: {"V2", "V5", "G2", "G5"} 0..1: all real while it is not trusted). starts: {(color, day):
	(decision index, shares, log growth, log of our value there, ...)}, the model's now (learner.day_starts). With no such
	decision (the game ended first) the real outcome stands; with none (past the cap), the grade is nan."""
	real, growth = real.copy(), growth.copy()
	for h in range(1, c.N_VALUE_OFFSETS):
		g = h - 1
		ahead = c.VALUE_OFFSETS[h] - c.VALUE_OFFSETS[g]
		w = 1 - (1 - outcome_share_min) * trust.get(f"V{c.VALUE_OFFSETS[g]}", 0.0)
		u = 1 - (1 - outcome_share_min) * trust.get(f"G{c.VALUE_OFFSETS[g]}", 0.0)
		for i, d in enumerate(game.decisions):
			start = starts.get((d.color, d.day + ahead))
			if start is None:
				continue
			shares_there, growth_there, log_value_there = start[1], start[2], start[3]
			real[i, h] = (1 - w) * shares_there[g] + w * real[i, h]
			growth[i, h] = (1 - u) * (log_value_there + growth_there[g]) + u * growth[i, h]
	return real, growth


def explained_variance(predicted, actual):
	"""1 - var(actual - predicted) / var(actual) over the rows with an actual value (nan: none, or fewer than 2)."""
	ok = np.isfinite(actual)
	if ok.sum() < 2:
		return float("nan")
	return float(1 - np.var(actual[ok] - predicted[ok]) / max(np.var(actual[ok]), 1e-12))


def residual_variances(predicted, actual):
	"""(var(actual - predicted), var(actual)) over the rows with an actual value (nan, nan: fewer than 2): explained
	variance's parts, to pool over updates (pooled_ev)."""
	ok = np.isfinite(actual)
	if ok.sum() < 2:
		return float("nan"), float("nan")
	return float(np.var(actual[ok] - predicted[ok])), float(np.var(actual[ok]))


def pooled_ev(pool, now, memory=0.9):
	"""(the new pool [residual variance, variance], its explained variance) after one update's residual_variances: each
	part smoothed apart (~10 updates), then the ratio. Smoothing the ratios let one update of a few games with outcomes
	close together (a small variance: the explained variance -1 or worse, 2026-10-29) pull the trust down for ten."""
	if not all(np.isfinite(now)):
		return pool, (1 - pool[0] / max(pool[1], 1e-12) if pool else float("nan"))
	pool = list(now) if pool is None else [memory * a + (1 - memory) * b for a, b in zip(pool, now)]
	return pool, 1 - pool[0] / max(pool[1], 1e-12)


def horizon_advantages(game, targets, price, trust, config=LabelConfig(), outcome_share_min=0.25, myopic_weight=0.5, check=1.0, parts=False):
	"""The policy's advantage [N] (game.decisions' order): aimed at the share VALUE_OFFSETS[-1] = 7 days on, the
	longest horizon (the user, 2026-10-28: the shorter heads are there to help learn V7, not to be maximized).

	Three estimates of how much better than expected a decision on day d did, for the share at the end of day d + 7:
	  td      the step's: a turn decision followed by another the same day, V7 there - V7 here (both predict the same
	          day's share); the day's last one (and a question), V5 at the player's first turn decision of day d + 2
	          (5 days on from d + 2 is d + 7: the horizons compose) - V7 here, or the share itself when the game had
	          ended. Little noise, as good as the value heads
	  mc      the share at the end of day d + 7 itself - V7 here: unbiased, all of the week's luck
	  myopic  the share 2 days on - V2 here: a shortcut aimed at the wrong horizon, used only while the heads can't
	          be trusted
	advantage = (1 - s) td + s mc + myopic_weight (1 - trust) check myopic, s = 1 - (1 - outcome_share_min) min(trust,
	check): with untrusted heads (trust 0) the real outcome and the 2-day shortcut; with trusted ones (trust 1, learner.py:
	the heads' explained variance) mostly the step's estimate, a quarter the outcome's (its blind spots still get
	corrected). check (credit_trust: does the step's estimate price each action type as what then happened?) only moves
	the credit to the real outcome: where the heads overprice a quick gain, the 2-day shortcut, which prices quick gains
	by design, is no fallback (the check once fed it: a failed check raised its weight).
	Prices: td charges one turn decision, mc every turn decision left that day (price, labels.label's). V2, V5, V7 are
	what the heads said when the decision was made (records). parts: also return the step's error, its estimate of the
	share at the end of day d + 7 minus that share (nan where it took the share itself), for the credit check."""
	decisions = game.decisions
	n = len(decisions)
	values = np.array([d.value for d in decisions], dtype=np.float32).reshape(n, c.N_VALUE_OFFSETS)
	short, middle, long = values[:, 0], values[:, 1], values[:, -1]
	turn = np.array([d.kind == c.DecisionKind.TURN for d in decisions])
	first_turn, next_turn, latest = {}, np.full(n, -1), {}
	for i, d in enumerate(decisions):
		if turn[i]:
			first_turn.setdefault((d.color, d.day), i)
	for i in reversed(range(n)):
		d = decisions[i]
		next_turn[i] = latest.get((d.color, d.day), -1)
		if turn[i]:
			latest[(d.color, d.day)] = i
	ahead = c.VALUE_OFFSETS[-1] - c.VALUE_OFFSETS[1]  # V5 this many days on predicts the share VALUE_OFFSETS[-1] days on
	target, estimated = np.empty(n, dtype=np.float32), np.zeros(n, dtype=bool)
	for i, d in enumerate(decisions):
		j = next_turn[i]
		k = first_turn.get((d.color, d.day + ahead))
		target[i] = long[j] if j >= 0 else middle[k] if k is not None else targets[i, -1]
		estimated[i] = j >= 0 or k is not None
	td = target - long - np.where(turn, config.decision_price, 0.0)
	mc = targets[:, -1] - long - price
	s = 1.0 - (1.0 - outcome_share_min) * min(trust, check)
	advantages = ((1 - s) * td + s * mc + myopic_weight * (1 - trust) * check * (targets[:, 0] - short)).astype(np.float32)
	error = np.where(estimated, target - targets[:, -1], np.nan).astype(np.float32)
	return (advantages, error) if parts else advantages


CREDIT_T = 3.0  # an action type's step estimate is off when its mean error is this many standard errors from 0
CREDIT_MEMORY = 0.9  # the check's evidence fades this much an update: it weighs the last ~10
CREDIT_MIN_PLAYERS = 8.0  # game-players (faded) with decisions of a type before the check judges the type


def credit_evidence(decisions, errors, players, evidence=None, memory=CREDIT_MEMORY):
	"""The credit check's evidence, {action type: [n, sum, sum of squares]} over game-players of the step's mean error
	after that type minus after the player's other turn decisions (errors: horizon_advantages' parts, nan where none;
	players: an id per decision), added to the earlier updates' (evidence) faded by memory.
	- A game-player is one sample, not a decision: its decisions share their week's luck (one share at the end of day
	  d + 7 for all of a day's), so as samples one lucky game made the check flag noise.
	- Against the player's other decisions: the luck they share cancels, and so does an error common to all of them,
	  which costs the policy nothing (ppo_update centres the advantages); what is left is the heads pricing what follows
	  one type differently from the rest, a quick gain's mark.
	- One update has ~16 game-players; the evidence is kept (~10 updates) to see a small gap."""
	evidence = {kind: [memory * v for v in sums] for kind, sums in (evidence or {}).items()}
	kinds = np.array([d.action_type if d.kind == c.DecisionKind.TURN else -1 for d in decisions])
	keep = (kinds >= 0) & np.isfinite(errors)
	players, kinds, errors = np.asarray(players)[keep], kinds[keep], np.asarray(errors, dtype=np.float64)[keep]
	for kind in np.unique(kinds):
		gaps = []
		for player in np.unique(players[kinds == kind]):
			mine, others = players == player, (players == player) & (kinds != kind)
			if others.any():
				gaps.append(errors[mine & (kinds == kind)].mean() - errors[others].mean())
		if gaps:
			gaps = np.array(gaps)
			sums = evidence.setdefault(c.ActionType(int(kind)).name, [0.0, 0.0, 0.0])
			sums[0], sums[1], sums[2] = sums[0] + len(gaps), sums[1] + float(gaps.sum()), sums[2] + float((gaps**2).sum())
	return evidence


def credit_gaps(evidence, min_players=CREDIT_MIN_PLAYERS):
	"""{action type: the t-statistic of its gap} from credit_evidence, for each type with min_players. If V7 (and V5 two
	days on) are right, the step's error has mean 0 after every type, so every gap is 0; a gap is the heads pricing what
	follows that type differently from what then happened by day 7: a quick gain that costs later is above 0, a slow
	gain that pays later below. (The faded count stands in for the number of samples: it is below the effective one,
	so the t-statistic errs small.)"""
	gaps = {}
	for kind, (n, total, squares) in evidence.items():
		if n < min_players:
			continue
		mean = total / n
		variance = max(squares / n - mean * mean, 0.0) * n / (n - 1)
		gaps[kind] = float(mean / (np.sqrt(variance / n) + 1e-12))
	return gaps


def credit_trust(gaps, limit=CREDIT_T):
	"""How far the step estimate is to be trusted given the gaps: 1 when every type agrees (|t| <= limit), falling
	linearly to 0 at twice the limit."""
	worst = max((abs(t) for t in gaps.values()), default=0.0)
	return float(min(1.0, max(0.0, 2.0 - worst / limit)))


# ---- what we take from the other players: the "attributed" credit (train.ppo_update) ----

def _battle_loss(battle, side):
	"""What a side of one of our battles lost (gold, its assets as the battle reward prices them: its army's casualties,
	its hero and mana if it lost them). foresight.battle_score's terms, without its torch imports."""
	import game_values
	from battle_model import reward

	army = sum(count * game_values.creature_value(creature) for creature, count in battle["armies"][side])
	hero, mana = 0.0, 0.0
	if battle["heroes"][side]:
		h = battle["heroes"][side]
		artifacts = sum(game_values.artifact_price(a) for a in h.get("artifacts", []))
		hero, mana = reward.hero_asset(h.get("level", 1), artifacts), reward.mana_value(h.get("mana", 0), h.get("power", 0))
	lost = sum(count * game_values.creature_value(creature) for creature, count in battle["casualties"][side])
	outcome = reward.Outcome.WIN if battle["winner"] == side else {1: reward.Outcome.RETREAT, 2: reward.Outcome.SURRENDER}.get(battle["result"], reward.Outcome.LOSS)
	return max(army + hero + mana - reward.assets_after(outcome, max(army - lost, 0.0), hero, mana), 0.0)


def taken_events(game, empire=EmpireConfig()):
	"""{color: [(day, serial or None, what it took from the other players then, as a share of all players' value)]}:
	- a battle against another player's hero: the value that side lost (_battle_loss), serial: the battle's
	- a town of another player's that became ours that day (our towns +1, theirs -1): its buildings' and income's value
	  as they lost it (split between the day's gainers)
	- a player's income that became ours that day (ours up, theirs down, no town changing hands): the smaller change
	Value over all players' as the day starts (tables'). Wandering monsters and a garrison without a hero are no player's
	loss here; the snapshots carry no ids, so a capture is read from the counts and the parts."""
	_, value, last, _ = tables(game, empire)
	colors = sorted(value)
	total = lambda day: max(sum(share_at(value, color, day, last) for color in colors), 1.0)  # noqa: E731 (the end of `day`)
	events = {}
	for color, battles in game.battles.items():
		for battle in battles:
			enemy = 1 - battle["side"]
			if battle.get("heroes") and battle["heroes"][enemy] and battle.get("casualties"):
				loss = _battle_loss(battle, enemy)
				if loss > 0:
					events.setdefault(color, []).append((battle["day"], battle["serial"], loss / total(battle["day"] - 1)))
	days = sorted(game.ledgers)
	for first, then in zip(days, days[1:]):
		before = {COLORS[p["color"]]: p for p in game.ledgers[first]["players"]}
		after = {COLORS[p["color"]]: p for p in game.ledgers[then]["players"]}
		parts = {color: (empire_value(before[color], empire), empire_value(after[color], empire)) for color in before if color in after}
		towns = {color: len(after[color]["towns"]) - len(before[color]["towns"]) for color in parts}
		income = {color: parts[color][1].get("income", 0.0) - parts[color][0].get("income", 0.0) for color in parts}  # (a player out of the game: {"total": 0})
		day, all_value = first, total(first - 1)
		gainers = [color for color in parts if towns[color] > 0]
		for loser in (color for color in parts if towns[color] < 0):
			lost = max(parts[loser][0].get("buildings", 0.0) - parts[loser][1].get("buildings", 0.0), 0.0) + max(-income[loser], 0.0)
			for color in gainers:
				events.setdefault(color, []).append((day, None, lost / len(gainers) / all_value))
		for color in (color for color in parts if towns[color] == 0 and income[color] > 0):
			for loser in (other for other in parts if other != color and towns[other] == 0 and income[other] < 0):
				events.setdefault(color, []).append((day, None, min(income[color], -income[loser]) / all_value))
	return events


def attributed_labels(game, empire=EmpireConfig(), horizon=c.VALUE_OFFSETS[-1]):
	"""Per decision (game.decisions' order; the battles' serials count game.all_decisions when it is set): (our share as
	its day starts [N], what we took from the other players in the `horizon` days from it [N]: taken_events after the
	decision, as shares of all players' value). With our log growth over the same days these are the parts of our log
	share's change our decisions make: d log share = (1 - share) d log V_ours + taken / V_all - the others' own growth
	(the last no decision of ours makes, and it carries the luck we cannot see: the user and e5's idea, 2026-10-31)."""
	share, _, last, _ = tables(game, empire)
	events = taken_events(game, empire)
	full = getattr(game, "all_decisions", game.decisions)
	serial, position = {}, {}
	for d in full:
		serial[id(d)] = position[d.color] = position.get(d.color, 0) + 1
	rows = [d for d in game.decisions]
	now = np.array([share_at(share, d.color, d.day - 1, last) for d in rows], dtype=np.float32)
	taken = np.zeros(len(rows), dtype=np.float32)
	for i, d in enumerate(rows):
		s = serial.get(id(d), 0)
		taken[i] = sum(f for day, at, f in events.get(d.color, ()) if d.day <= day <= d.day + horizon and (at is None or at >= s))
	return now, taken
