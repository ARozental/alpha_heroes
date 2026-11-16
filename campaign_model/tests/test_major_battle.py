"""evaluate.major_battle (the user via e5, 2026-11-16: "save the major ... highest value battle going by ... harmonic mean
of value from both sides"): among our battles against a rival hero or a town, the highest harmonic mean of the two
armies' worth at the start; a neutral fight never counts; its fields in score()'s result (eval_games.jsonl's line)."""
from campaign_model.empire import army_worth
from campaign_model.evaluate import EVAL_EMPIRE, major_battle, score
from campaign_model.records import Game

RED, BLUE = 0, 1


def snap(day, red_towns=1):
	town = {"faction": 0, "built": [10], "garrison": [], "available": []}
	return {"day": day, "final": False, "players": [
		{"color": RED, "status": 0, "towns": [dict(town) for _ in range(red_towns)], "heroes": [{"type": 5, "level": 3, "exp": 2500, "army": [[0, 40]], "primary": [1, 1, 1, 1], "skills": [], "spells": []}], "resources": [0, 0, 0, 0, 0, 0, 5000], "income": [0, 0, 0, 0, 0, 0, 0], "dwellings": [], "armies": [], "explored": 0},
		{"color": BLUE, "status": 0, "towns": [dict(town)], "heroes": [{"type": 7, "level": 5, "exp": 6000, "army": [[2, 30]], "primary": [2, 2, 1, 1], "skills": [], "spells": []}], "resources": [0, 0, 0, 0, 0, 0, 5000], "income": [0, 0, 0, 0, 0, 0, 0], "dwellings": [], "armies": [], "explored": 0},
	]}


def battle(day, side, ours, theirs, their_hero=True, winner=None, serial=0):
	hero = {"level": 3, "exp": 2500}
	return {"day": day, "side": side, "armies": [ours, theirs] if side == 0 else [theirs, ours], "heroes": [hero, {"level": 6} if their_hero else None] if side == 0 else [{"level": 6} if their_hero else None, hero],
		"winner": side if winner else 1 - side, "result": 0, "casualties": [[], []], "exp": [0, 0], "serial": serial}


def test_the_largest_harmonic_mean_against_a_hero_or_a_town_and_never_a_neutral():
	game = Game(ledgers={d: snap(d) for d in range(1, 30)}, battles={"RED": [
		battle(5, 0, [[0, 400]], [[3, 300]], their_hero=False, winner=True),  # (a neutral fight: the biggest, never the major one)
		battle(12, 0, [[0, 40]], [[2, 30]], winner=True),  # (a rival hero, small)
		battle(20, 1, [[0, 60]], [[2, 90]], winner=False),  # (a rival hero, the largest harmonic mean of the two)
	]})
	m = major_battle(game, "RED")
	a, b = army_worth([(0, 60)], EVAL_EMPIRE), army_worth([(2, 90)], EVAL_EMPIRE)
	assert m["major_day"] == 20 and m["major_kind"] == "rival hero" and m["major_attacked"] is False and m["major_won"] is False
	assert abs(m["major_ours"] - a) < 0.1 and abs(m["major_theirs"] - b) < 0.1
	assert m["major_level_ours"] == 3 and m["major_level_theirs"] == 6
	assert m["major_at_town"] is None  # (no decisions recorded: unknown)


def test_a_town_taken_counts_and_no_battle_gives_nones():
	ledgers = {d: snap(d, red_towns=1 if d < 15 else 2) for d in range(1, 30)}
	game = Game(ledgers=ledgers, battles={"RED": [battle(15, 0, [[0, 50]], [[4, 20]], their_hero=False, winner=True)]})
	m = major_battle(game, "RED")
	assert m["major_day"] == 15 and m["major_kind"] == "town" and m["major_won"] is True
	empty = major_battle(Game(ledgers={d: snap(d) for d in range(1, 5)}), "RED")
	assert set(empty.values()) == {None}


def test_score_carries_the_fields():
	from campaign_model.tests.test_learner import snapshot

	game = Game(ledgers={d: snapshot(d, 3000, 1000) for d in range(1, 8)}, battles={"RED": [battle(3, 0, [[0, 40]], [[2, 30]], winner=True)]})
	result = score(game, "RED", players=2, days=5)
	assert result["major_day"] == 3 and result["major_won"] is True
