import numpy as np
import pytest

from campaign_model import constants as c
from campaign_model.empire import empire_value
from campaign_model.labels import LabelConfig, blend_credit, end_advantages, end_labels, end_steps, end_targets, growth_labels, label, ladder_targets
from campaign_model.records import Decision, Game

TURN = int(c.DecisionKind.TURN)


def snapshot(day, red, blue, blue_status=0):
	"""Two players whose only asset is gold (resources), so the shares are red / (red + blue)."""
	def p(color, gold, status=0):
		return {"color": color, "status": status, "resources": [0] * 6 + [gold], "income": [0] * 7, "explored": 0, "towns": [], "heroes": [], "dwellings": [], "armies": []}
	return {"day": day, "players": [p(0, red), p(1, blue, blue_status)]}


def decision(color, day, today=0, kind=TURN):
	return Decision(color, day, kind, today, 1, 0, -1, -1, 0.0, (0.5,) * c.N_VALUE_OFFSETS, 0, b"")


def test_labels_are_the_shares_two_five_and_seven_days_on():
	game = Game(ledgers={d: snapshot(d, 1000 * d, 1000) for d in range(1, 15)})
	game.decisions = [decision("RED", 3)]
	targets, mixed, _ = label(game, LabelConfig(decision_price=0))
	# the end of day d is the snapshot of day d + 1: red has 1000 (d + 1) against 1000
	want = [(d + 1) / (d + 2) for d in (5, 8, 10)]
	assert targets[0] == pytest.approx(want)
	assert mixed[0] == pytest.approx(np.mean(want))


def test_past_the_last_day_the_last_share_stands_and_eliminated_players_get_zero():
	game = Game(ledgers={1: snapshot(1, 1000, 1000), 2: snapshot(2, 2000, 1000), 3: snapshot(3, 3000, 500, blue_status=1)})
	game.decisions = [decision("RED", 1), decision("BLUE", 1)]
	targets, _, _ = label(game)
	assert (targets[0] == 1).all() and (targets[1] == 0).all()


def test_the_end_of_the_game_stands_for_every_day_after():
	game = Game(ledgers={1: snapshot(1, 1000, 1000), 2: snapshot(2, 1000, 1000)}, final=[{**snapshot(2, 4000, 1000, blue_status=1), "final": True}])
	game.decisions = [decision("RED", 1)]
	targets, _, _ = label(game)
	assert (targets[0] == 1).all()


def test_the_price_counts_the_turn_decisions_still_to_come_that_day():
	game = Game(ledgers={d: snapshot(d, 1000, 1000) for d in range(1, 4)})
	game.decisions = [decision("RED", 1, today=i) for i in range(3)] + [decision("RED", 1, kind=int(c.DecisionKind.YES_NO))]
	_, _, price = label(game, LabelConfig(decision_price=0.01))
	assert price == pytest.approx([0.03, 0.02, 0.01, 0.0])


def test_the_winner_ends_with_every_share():
	"""The end-of-game snapshot marks the winner WINNER (2), the others LOSER (1): only losers are worth 0."""
	end = {**snapshot(2, 4000, 1000, blue_status=1), "final": True}
	end["players"][0]["status"] = 2
	game = Game(ledgers={1: snapshot(1, 1000, 1000), 2: snapshot(2, 1000, 1000)}, final=[end])
	game.decisions = [decision("RED", 1), decision("BLUE", 1)]
	targets, _, _ = label(game)
	assert (targets[0] == 1).all() and (targets[1] == 0).all()


def test_players_knocked_out_together_get_nothing():
	end = {**snapshot(2, 0, 0, blue_status=1), "final": True}
	end["players"][0]["status"] = 1
	game = Game(ledgers={1: snapshot(1, 1000, 1000), 2: snapshot(2, 1000, 1000)}, final=[end])
	game.decisions = [decision("RED", 1), decision("BLUE", 1)]
	targets, _, _ = label(game)
	assert (targets == 0).all()


def test_past_a_day_cap_there_is_no_label():
	"""A game stopped at the cap (both players still in it) has no outcome past it: those labels are nan, the label mix
	and the growth labels too; before it they stand."""
	game = Game(ledgers={d: snapshot(d, 1000 * d, 1000) for d in range(1, 6)})  # the end of day 4 is the last
	game.decisions = [decision("RED", 1), decision("RED", 2)]
	targets, mixed, _ = label(game)
	assert targets[0, 0] == pytest.approx(4 / 5) and targets[1, 0] == pytest.approx(5 / 6)  # the ends of days 3, 4
	assert np.isnan(targets[:, 1:]).all() and np.isnan(mixed).all()
	growth = growth_labels(game)
	assert growth[0, 0] == pytest.approx(np.log(empire_value(snapshot(4, 4000, 1000)["players"][0])["total"]), abs=1e-5)
	assert np.isnan(growth[:, 1:]).all()


def test_growth_labels_are_our_own_value_and_nothing_once_knocked_out():
	game = Game(ledgers={1: snapshot(1, 1000, 1000), 2: snapshot(2, 2000, 1000), 3: snapshot(3, 3000, 500, blue_status=1)})
	game.decisions = [decision("RED", 1), decision("BLUE", 1)]
	growth = growth_labels(game)
	assert growth[0] == pytest.approx([np.log(empire_value(snapshot(3, 3000, 500)["players"][0])["total"])] * 3, abs=1e-5)
	assert (growth[1] == 0).all()  # (log of 1 gold: far below /10 of anything at the decision)


def test_the_longer_heads_are_graded_against_the_shorter_ones_forecasts_later():
	"""V5's grade: V2's forecast at the player's first turn decision three days on (a forecast of the same day, d + 5),
	mixed with the real share by the trust in V2; V7's: V5's two days on. The growth heads alike, the forecast added to
	the log of our value there. Untrusted: the real outcome; no decision there (the game ended): the real outcome; no
	real outcome (past the cap): no grade."""
	game = Game(decisions=[decision("RED", 1), decision("RED", 3), decision("RED", 4)])
	real = np.array([[0.4, 0.5, 0.6], [0.45, 0.55, np.nan], [0.5, np.nan, np.nan]], dtype=np.float32)
	growth = np.log(np.array([[2000, 3000, 4000], [2500, 3500, np.nan], [3000, np.nan, np.nan]], dtype=np.float32))
	starts = {
		("RED", 3): (1, np.array([0.52, 0.58, 0.61]), np.array([0.1, 0.3, 0.4]), np.log(2200.0)),
		("RED", 4): (2, np.array([0.53, 0.59, 0.62]), np.array([0.05, 0.2, 0.3]), np.log(2400.0)),
		("RED", 5): (0, np.array([0.54, 0.60, 0.63]), np.array([0.0, 0.1, 0.2]), np.log(2600.0)),
	}
	trust = {"V2": 1.0, "V5": 0.5, "G2": 1.0, "G5": 0.0}
	shares, grown = ladder_targets(game, real, growth, starts, trust, outcome_share_min=0.25)
	assert shares[0] == pytest.approx([0.4, 0.75 * 0.53 + 0.25 * 0.5, 0.375 * 0.58 + 0.625 * 0.6])
	assert grown[0] == pytest.approx([np.log(2000), 0.75 * (np.log(2400) + 0.05) + 0.25 * np.log(3000), np.log(4000)], abs=1e-5)
	assert shares[1, :2] == pytest.approx([0.45, 0.55]) and np.isnan(shares[1, 2])  # (no decision on day 6; past the cap)
	assert np.isnan(shares[2, 1:]).all() and np.isnan(grown[2, 1:]).all()
	untrusted, _ = ladder_targets(game, real, growth, starts, {}, outcome_share_min=0.25)
	assert np.array_equal(untrusted, real, equal_nan=True)


def test_a_player_knocked_out_before_the_time_limit_does_not_end_the_game():
	"""Three players; one is knocked out on day 3 (its end-of-game snapshot) and the game is stopped before day 4's
	ledger: two players still in it, so it did not end: the labels past day 3 are none (2026-10-28 review)."""
	def three(day, statuses):
		players = [{"color": i, "status": st, "resources": [0] * 6 + [1000 * (i + 1)], "income": [0] * 7, "explored": 0, "towns": [], "heroes": [], "dwellings": [], "armies": []} for i, st in enumerate(statuses)]
		return {"day": day, "players": players}

	game = Game(ledgers={1: three(1, [0, 0, 0]), 2: three(2, [0, 0, 0]), 3: three(3, [0, 0, 0])}, final=[{**three(3, [0, 0, 1]), "final": True}])
	game.decisions = [decision("RED", 1)]
	targets, _, _ = label(game)
	assert np.isfinite(targets[0, 0]) and np.isnan(targets[0, 1:]).all()  # day 3 was seen; past it, the game went on
	game.final = [{**three(3, [2, 1, 1]), "final": True}]  # this time the others are out: it ended
	targets, _, _ = label(game)
	assert np.isfinite(targets[0]).all()


def test_the_end_is_the_share_where_the_game_ended_or_reached_its_cap_and_unknown_at_a_time_limit():
	game = Game(ledgers={d: snapshot(d, 1000 * d, 1000) for d in range(1, 12)}, days=10)
	game.decisions = [decision("RED", 1), decision("BLUE", 3)]
	assert end_labels(game) == pytest.approx([11 / 12, 1 / 12])  # the cap's day 10 ends with the snapshot of day 11
	game.days = 30  # stopped before its cap (the time limit): no end
	assert np.isnan(end_labels(game)).all()
	game.ledgers[11] = snapshot(11, 5000, 1000, blue_status=1)  # blue knocked out: one player left, the game ended
	game.days = 0  # (whatever the cap)
	assert end_labels(game) == pytest.approx([1.0, 0.0])
	third = {"color": 2, "status": 0, "resources": [0] * 6 + [1000], "income": [0] * 7, "explored": 0, "towns": [], "heroes": [], "dwellings": [], "armies": []}
	for day, ledger in game.ledgers.items():  # three players: blue's knock-out ends the game for blue alone
		ledger["players"] = ledger["players"][:2] + [{**third}]
	game.days = 30
	labels = end_labels(game)
	assert np.isnan(labels[0]) and labels[1] == 0.0


def test_the_end_head_is_graded_against_itself_the_next_day_mixed_with_the_real_end():
	game = Game(ledgers={d: snapshot(d, 1000 * d, 1000) for d in range(1, 12)}, days=10)
	game.decisions = [decision("RED", 1), decision("RED", 2), decision("RED", 10)]
	real = end_labels(game)
	starts = {("RED", 2): (1, np.zeros(3), np.zeros(3), 0.0, 0.6, 0.5)}
	graded = end_targets(game, real, starts, 1.0, outcome_share_min=0.25)
	assert graded[0] == pytest.approx(0.75 * 0.6 + 0.25 * real[0])
	assert graded[1] == pytest.approx(real[1]) and graded[2] == pytest.approx(real[2])  # no start the next day: the end
	assert end_targets(game, real, starts, 0.0)[0] == pytest.approx(real[0])  # untrusted: the real end alone
	unknown = np.full(3, np.nan, dtype=np.float32)  # a game stopped by the time limit: the forecast alone
	graded = end_targets(game, unknown, starts, 1.0)
	assert graded[0] == pytest.approx(0.6) and np.isnan(graded[1:]).all()


def test_the_end_credit_steps_to_the_next_turn_decision_and_a_week_on():
	game = Game(ledgers={d: snapshot(d, 1000, 1000) for d in range(1, 20)})
	question = int(c.DecisionKind.YES_NO)
	game.decisions = [decision("RED", 1), decision("RED", 1, kind=question), decision("RED", 1, today=1), decision("RED", 3), decision("RED", 9), decision("BLUE", 1), decision("BLUE", 8)]
	step, week = end_steps(game)
	assert list(step) == [2, 2, 3, 4, -1, 6, -1]  # (a question steps to the next turn decision; never across players)
	assert list(week) == [4, 4, 4, -1, -1, 6, -1]  # (the first turn decision 7 days on or later)


def test_the_end_credit_mixes_the_step_and_the_week():
	values = np.array([0.4, 0.5, 0.45, 0.6], dtype=np.float32)
	step, week = np.array([1, 2, 3, -1]), np.array([3, -1, -1, -1])
	real = np.full(4, 0.7, dtype=np.float32)
	price, turn = np.zeros(4, dtype=np.float32), np.ones(4, dtype=bool)
	a = end_advantages(values, step, week, real, price, turn, 0.0, 0.25)
	assert a[0] == pytest.approx(0.75 * (0.5 - 0.4) + 0.25 * (0.6 - 0.4))
	assert a[1] == pytest.approx(0.75 * (0.45 - 0.5) + 0.25 * (0.7 - 0.5))  # no decision a week on: the real end
	assert a[3] == pytest.approx(0.7 - 0.6)
	real[:] = np.nan  # a game stopped by the time limit: the step alone, and none at the last decision
	a = end_advantages(values, step, week, real, price, turn, 0.0, 0.25)
	assert a[0] == pytest.approx(0.75 * 0.1 + 0.25 * 0.2) and a[1] == pytest.approx(-0.05) and np.isnan(a[3])
	assert end_advantages(values, step, week, real, price, turn, 0.01, 0.0)[0] == pytest.approx(0.09)  # one turn decision's price


def test_the_policy_credit_blends_the_seven_day_and_the_end_credit():
	seven = np.array([1.0, np.nan, 2.0, np.nan], dtype=np.float32)
	end = np.array([3.0, 4.0, np.nan, np.nan], dtype=np.float32)
	mixed = blend_credit(seven, end, 0.25)
	assert mixed[:3] == pytest.approx([0.75 + 0.75, 1.0, 1.5]) and np.isnan(mixed[3])

def test_explained_variance_pools_its_parts_not_its_ratios():
	"""One update whose outcomes lie close together (a small variance) gives an explained variance far below 0; pooled
	by parts it weighs as little as its variance does."""
	from campaign_model.labels import pooled_ev, residual_variances

	pool, ev = pooled_ev(None, (0.5, 1.0))
	assert ev == pytest.approx(0.5)
	pool, ev = pooled_ev(pool, (0.03, 0.01))  # its own explained variance: -2
	assert ev == pytest.approx(1 - (0.9 * 0.5 + 0.1 * 0.03) / (0.9 * 1.0 + 0.1 * 0.01))  # ~0.5 still
	assert pooled_ev(pool, (float("nan"), float("nan")))[0] == pool  # nothing measured: unchanged
	res, var = residual_variances(np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.4, np.nan]))  # (no outcome: left out)
	assert var == pytest.approx(np.var([0.2, 0.4])) and res == pytest.approx(np.var([0.1, 0.2]))


def test_the_attributed_credit_counts_what_we_take_after_each_decision():
	"""(2026-10-31) attributed_labels: our share as the day starts, and what we took from the others in the 7 days from
	each decision: a battle won against their hero (after the decision: its serial), a town of theirs that became ours."""
	from campaign_model.labels import attributed_labels, taken_events

	def town():
		return {"faction": 0, "built": [0, 1, 10], "garrison": [], "available": []}

	def snap(day, red_towns, blue_towns):
		s = snapshot(day, 10000, 10000)
		s["players"][0]["towns"], s["players"][1]["towns"] = [town() for _ in range(red_towns)], [town() for _ in range(blue_towns)]
		return s

	ledgers = {d: snap(d, 1 if d <= 5 else 2, 2 if d <= 5 else 1) for d in range(1, 12)}  # red takes a town of blue's on day 5
	battle = {"serial": 2, "day": 3, "side": 0, "armies": [[[1, 10]], [[1, 20]]], "heroes": [{"level": 2, "mana": 0, "power": 1, "artifacts": []}, {"level": 3, "mana": 10, "power": 2, "artifacts": []}],
		"result": 0, "winner": 0, "casualties": [[[1, 2]], [[1, 20]]], "exp": [0, 0]}
	game = Game(ledgers=ledgers, battles={"RED": [battle]})
	game.decisions = [decision("RED", 1), decision("RED", 3), decision("RED", 4), decision("RED", 9), decision("BLUE", 3)]
	events = taken_events(game)
	assert len(events["RED"]) == 2 and "BLUE" not in events  # the battle, the town
	now, taken = attributed_labels(game)
	battle_part, town_part = (f for _, _, f in sorted(events["RED"], key=lambda e: e[0]))
	assert 0.0 < now[0] < 0.5  # (red's one town against blue's two)
	assert taken[0] == pytest.approx(battle_part + town_part)  # day 1: both within 7 days, the battle after it
	assert taken[1] == pytest.approx(battle_part + town_part)  # serial 2 led to the battle
	assert taken[2] == pytest.approx(town_part)  # after the battle
	assert taken[3] == 0.0 and taken[4] == 0.0  # past it all; blue took nothing


def test_the_attributed_credit_takes_a_player_out_of_the_game():
	"""A player knocked out (its empire value {"total": 0}, no parts) loses its towns to us: the capture counts, no
	KeyError (it stopped joint5, 2026-10-31 14:00)."""
	from campaign_model.labels import attributed_labels, taken_events

	def snap(day, blue_status):
		s = snapshot(day, 10000, 10000, blue_status=blue_status)
		s["players"][0]["towns"] = [{"faction": 0, "built": [0, 1], "garrison": [], "available": []} for _ in range(1 if day <= 3 else 2)]
		s["players"][1]["towns"] = [{"faction": 0, "built": [0, 1], "garrison": [], "available": []}] if day <= 3 else []
		return s

	game = Game(ledgers={d: snap(d, 0 if d <= 3 else 1) for d in range(1, 8)})
	game.decisions = [decision("RED", 2)]
	assert taken_events(game)["RED"][0][0] == 3
	now, taken = attributed_labels(game)
	assert taken[0] > 0
