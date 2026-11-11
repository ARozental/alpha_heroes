"""The split evaluation (config.eval_own_games, eval_follow_games; the user, 2026-11-11): its arms' setups and the
own-choice switch each game gets, the row's columns (the old score columns empty, OWN's and FOLLOW's apart, their
pairing), what reads the score (evaluate.eval_score: OWN's), and an old eval.csv taking the new columns."""
import csv
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

import campaign_model.evaluate as evaluate
from campaign_model.config import CampaignConfig
from campaign_model.evaluate import COLUMNS, SHOWN_ONLY, eval_score, plan, setups

SPLIT = dict(eval_games=48, eval_own_games=32, eval_follow_games=16)


def test_the_arms_setups_are_fixed_spread_and_follow_is_a_subset_of_own():
	assert plan(CampaignConfig(eval_games=48)) == [(i, "shown") for i in range(48)]  # (unsplit: as before)
	arms = plan(CampaignConfig(**SPLIT))
	assert arms == plan(CampaignConfig(**SPLIT)) and len(arms) == 48  # (the same every evaluation; 48 games as before)
	own, follow = [i for i, a in arms if a == "own"], [i for i, a in arms if a == "shown"]
	assert own == list(range(32)) and len(follow) == 16 and set(follow) <= set(own) and len(set(follow)) == 16
	games = setups(48)
	assert sorted(Counter(games[i][1] for i in own).values()) == [10, 11, 11]
	assert sorted(Counter(games[i][1] for i in follow).values()) == [5, 5, 6]
	assert max(Counter(games[i][2] for i in follow).values()) <= 8  # (our seat spread, not one colour's)


def fake_evaluation(monkeypatch, config):
	"""evaluate.evaluate with its games faked (each game's settings kept, every score 0)."""
	seen = {}

	def play(socket, name, days, seed, settings, game_key, **kwargs):
		seen[game_key] = dict(settings), list(settings)
		return SimpleNamespace(decisions=[], done={}), SimpleNamespace(status="stopped at day 57", seconds=1.0, run_dir="/nonexistent")

	monkeypatch.setattr(evaluate, "play", play)
	monkeypatch.setattr(evaluate, "eval_map", lambda seed, players, config: f"Maps/x{seed}")
	monkeypatch.setattr(evaluate, "memory_allows", lambda **kwargs: True)
	arms = plan(config)

	def score(game, color, players, days, empire=None):
		return {"score": 0.0, "score_d28": 0.0, "players": players, "share": 0.0, "phi": 1.0, "phi_nullkiller": 1.0, "heroes": 1, "heroes_nullkiller": 1.0, "eliminated": False, "won": False, "short": False, "ai_errors": 0}

	monkeypatch.setattr(evaluate, "score", score)
	server = SimpleNamespace(socket_for=lambda j: f"s{j}")
	battle = SimpleNamespace(eval_settings=lambda j: (("ALPHA_BATTLE_SLOT", str(j)),))
	row = evaluate.evaluate(config, server, battle, version=7, epoch=1)
	return row, seen, arms


def test_each_game_gets_its_arms_switch_once(monkeypatch):
	config = CampaignConfig(nk2_inputs=True, eval_parallel=8, **SPLIT)
	row, seen, arms = fake_evaluation(monkeypatch, config)
	assert len(seen) == 48 and row["games"] == 48 and row["failed"] == 0
	for j, (i, arm) in enumerate(arms):
		settings, pairs = seen[f"e{j}"]
		assert [name for name, _ in pairs].count("ALPHA_MAP_NK2_CHOICE") == 1
		assert settings["ALPHA_MAP_NK2_CHOICE"] == ("0" if arm == "own" else "1") and settings["ALPHA_MAP_NK2_INPUTS"] == "1"
	assert {g["arm"] for g in row["per_game"]} == {"own", "shown"}
	assert all(row[key] == "" for key in SHOWN_ONLY) and row["games_own"] == 32 and row["games_follow"] == 16
	unsplit, seen, _ = fake_evaluation(monkeypatch, CampaignConfig(nk2_inputs=True, eval_games=6, eval_parallel=6))
	assert all(settings["ALPHA_MAP_NK2_CHOICE"] == "1" for settings, _ in seen.values()) and "score_own" not in unsplit and np.isfinite(unsplit["score"])


def game(setup, arm, score, seed=None):
	return {"setup": setup, "arm": arm, "seed": seed if seed is not None else 1000 + setup, "map": f"m{setup}", "color": "RED", "score": score, "score_d28": score / 2, "phi": 10.0 * score, "eliminated": score == 0, "won": score > 2}


def split_row(version, own_scores, follow_scores):
	played = [game(i, "own", s) for i, s in enumerate(own_scores)] + [game(i, "shown", s) for i, s in follow_scores.items()]
	row = {"version": version, "epoch": 1, "games": len(played), "short": 0, "ai_errors": 0, "failed": 0, "seconds": 9.0, **{f"failure_{k}": 0.0 for k in evaluate.FAILURES}, "per_game": played}
	row.update(evaluate.split_row(played))
	return row


def test_the_rows_arms_their_pairing_and_the_score_read(tmp_path):
	path = tmp_path / "eval.csv"
	first = split_row(1, [0.5, 1.0, 0.0, 1.5], {0: 0.9, 2: 0.4})
	assert first["score_own"] == pytest.approx(0.75) and first["score_follow"] == pytest.approx(0.65)
	assert first["follow_minus_own"] == pytest.approx(((0.9 - 0.5) + (0.4 - 0.0)) / 2) and first["eliminated_own"] == 1
	evaluate.log(first, path)
	second = split_row(2, [0.7, 1.2, 0.1, 1.4], {0: 1.0, 2: 0.5})
	evaluate.log(second, path)
	rows = list(csv.DictReader(path.open()))
	assert all(rows[-1][key] == "" for key in SHOWN_ONLY)  # (the old score columns: empty, never a different measure under their name)
	assert float(rows[-1]["own_vs_previous"]) == pytest.approx(np.mean([0.2, 0.2, 0.1, -0.1]))  # (OWN against the previous OWN only)
	assert eval_score(rows[-1]) == pytest.approx(float(rows[-1]["score_own"]))  # (what reads the score: OWN's)
	assert eval_score({"score": "0.6", "score_own": ""}) == pytest.approx(0.6) and np.isnan(eval_score({"score": ""}))


def test_an_old_eval_csv_takes_the_new_columns(tmp_path):
	from battle_model.train import upgrade_csv

	path = tmp_path / "eval.csv"
	old = COLUMNS[: COLUMNS.index("games_own")]
	with path.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(old)
		writer.writerow([3120 if name == "version" else 0.75 if name == "score" else 0 for name in old])
	upgrade_csv(path, COLUMNS)
	evaluate.log(split_row(3200, [0.5, 1.0], {0: 0.8}), path)
	rows = list(csv.DictReader(path.open()))
	assert list(rows[0]) == COLUMNS and rows[0]["score_own"] == "" and float(rows[0]["score"]) == 0.75
	assert eval_score(rows[0]) == pytest.approx(0.75) and eval_score(rows[1]) == pytest.approx(0.75)
