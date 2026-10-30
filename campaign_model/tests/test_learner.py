import json
import threading
from types import SimpleNamespace

import pytest

from battle_model.train import TrainConfig, with_overrides
from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.evaluate import score, setups
from campaign_model.learner import CampaignLearner
from campaign_model.play import finished
from campaign_model.records import Decision, Game
from campaign_model.tests.test_labels import snapshot


def test_campaign_settings_from_the_command_line():
	config = with_overrides(TrainConfig(), ["campaign.games=4", "campaign.players=2,3", "campaign.label_weights=0.2,0.3,0.5", "campaign.device=cpu"])
	assert config.campaign.games == 4 and config.campaign.players == (2, 3) and config.campaign.device == "cpu"
	assert config.campaign.label_weights == (0.2, 0.3, 0.5)
	assert TrainConfig().campaign.games == 0  # a battle-only run unless asked


def ledger(day, players, final=False):
	return json.dumps({"day": day, "final": final, "players": [{"color": color, "status": status} for color, status in players]}).encode()


def test_a_game_stops_when_our_players_have_sent_their_last_records():
	every = {"g:ledger": [ledger(1, [(0, 0), (1, 0)]), ledger(6, [(0, 0), (1, 0)])], "g:RED:done": [b""]}
	assert not finished(every, days=5)  # BLUE (ours too) has not finished
	assert finished({**every, "g:BLUE:done": [b""]}, days=5)
	one_out = {"g:ledger": [ledger(1, [(0, 0), (1, 0)]), ledger(6, [(0, 0), (1, 1)])], "g:RED:done": [b""]}
	assert finished(one_out, days=5) and not finished(one_out, days=6)  # BLUE is out of the game; day 6 is no day past a cap of 6
	# An evaluation: only RED is ours, Nullkiller2 never sends a "done"
	assert finished(every, days=5, ours=("RED",))
	assert not finished({"g:ledger": [ledger(1, [(0, 0), (1, 0)])]}, days=5, ours=("RED",))


def test_the_eval_score_is_our_share_times_the_players():
	game = Game(ledgers={d: snapshot(d, 3000, 1000) for d in range(1, 8)})
	result = score(game, "RED", players=2, days=5)
	assert result["score"] == pytest.approx(1.5) and result["share"] == pytest.approx(0.75)
	assert result["phi"] == pytest.approx(3000) and result["phi_nullkiller"] == pytest.approx(1000) and not result["short"]
	knocked_out = Game(ledgers={1: snapshot(1, 1000, 1000), 2: snapshot(2, 1000, 1000), 3: snapshot(3, 2000, 0, blue_status=1)})
	result = score(knocked_out, "BLUE", players=2, days=5)
	assert result["score"] == 0 and result["eliminated"] and not result["short"]
	# Knocked out during a day: only the end-of-game snapshot says so
	that_day = Game(ledgers={1: snapshot(1, 1000, 1000), 2: snapshot(2, 1000, 1000)}, final=[{**snapshot(2, 3000, 0, blue_status=1), "final": True}])
	result = score(that_day, "BLUE", players=2, days=5)
	assert result["score"] == 0 and result["eliminated"] and result["heroes"] == 0 and not result["short"]
	won = Game(ledgers={1: snapshot(1, 1000, 1000)}, final=[{**snapshot(1, 3000, 0, blue_status=1), "final": True}])
	won.final[0]["players"][0]["status"] = 2
	result = score(won, "RED", players=2, days=5)
	assert result["won"] and not result["eliminated"] and not result["short"] and result["score"] == pytest.approx(2)
	cut = Game(ledgers={1: snapshot(1, 1000, 1000), 2: snapshot(2, 1000, 1000)})  # a time limit at day 2
	assert score(cut, "BLUE", players=2, days=5)["short"] and score(cut, "BLUE", players=2, days=5)["score"] == pytest.approx(1)


def test_eval_games_are_fixed_and_move_our_seat():
	games = setups(6)
	assert games == setups(6)
	assert [g[1] for g in games] == [2, 3, 4, 2, 3, 4]
	assert [g[2] for g in games] == ["RED", "RED", "RED", "BLUE", "BLUE", "BLUE"]


def test_evals_compare_only_with_evals_played_the_same_way(tmp_path):
	"""(2026-10-29) The eval went greedy: a greedy eval's paired differences are against earlier greedy ones only (the
	sampled evals before it measured another thing), and it still gets the next eval number."""
	import json
	import math

	from campaign_model.evaluate import COLUMNS, log

	def row(version, scores, temperature=None):
		games = [{"seed": i, "map": f"m{i}", "color": "RED", "score": x, **({} if temperature is None else {"temperature": temperature})} for i, x in enumerate(scores)]
		return {**{k: 0.0 for k in COLUMNS}, "version": version, "per_game": games}

	path = tmp_path / "eval.csv"
	log(row(1, [0.1, 0.2, 0.3]), path)  # sampled, from before the field
	log(row(2, [0.2, 0.3, 0.4]), path)
	greedy = row(3, [0.5, 0.5, 0.5], temperature=0.0)
	log(greedy, path)
	assert math.isnan(greedy["vs_previous"]) and math.isnan(greedy["vs_first"])
	later = row(4, [0.6, 0.7, 0.6], temperature=0.0)
	log(later, path)
	assert later["vs_previous"] == pytest.approx(0.1333, abs=1e-3) and later["vs_first"] == later["vs_previous"]
	numbers = [json.loads(line)["eval"] for line in (tmp_path / "eval_games.jsonl").read_text().splitlines()]
	assert numbers == [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3


def game_of(decisions):
	return Game(decisions=[Decision("RED", 1, int(c.DecisionKind.TURN), 0, 1, 0, -1, -1, 0.0, (0.5,) * 3, 0, b"")] * decisions)


def test_an_update_waits_for_enough_decisions_of_enough_games():
	learner = SimpleNamespace(config=CampaignConfig(decisions_per_update=100, min_games_per_update=2), lock=threading.Condition(), finished=[], stats=__import__("collections").Counter())
	take = lambda: CampaignLearner._take(learner)  # noqa: E731
	learner.finished = [(game_of(300), None)]
	assert take() is None  # one game, however long
	learner.finished.append((game_of(10), None))
	assert len(take()) == 2 and learner.finished == []
	learner.finished = [(game_of(60), "a"), (game_of(60), "b"), (game_of(60), "c")]
	assert [info for _, info in take()] == ["a", "b"] and len(learner.finished) == 1  # the oldest first; the rest wait


def test_maps_with_a_player_left_without_a_town_are_found(tmp_path):
	import zipfile

	from campaign_model.play import players_without_a_town

	path = tmp_path / "m.vmap"
	header = '{\n\t// generated\n\t"players": {"red": {"canPlay": "PlayerOrAI"}, "blue": {"mainTown": {"x": 1, "y": 2, "l": 0}, "canPlay": "PlayerOrAI"},},\n}'
	with zipfile.ZipFile(path, "w") as z:
		z.writestr("header.json", header)
	assert players_without_a_town(path) == ["red"]


def test_a_backlog_beyond_two_updates_drops_its_oldest_games():
	learner = SimpleNamespace(config=CampaignConfig(decisions_per_update=100, min_games_per_update=2), lock=threading.Condition(), finished=[], stats=__import__("collections").Counter())
	learner.finished = [(game_of(60), i) for i in range(8)]  # 480 decisions: room for 200
	taken = CampaignLearner._take(learner)
	assert [info for _, info in taken] == [4, 5] and [info for _, info in learner.finished] == [6, 7] and learner.stats["dropped_games"] == 4


def test_an_update_runs_with_the_horizon_credit_and_learns_how_far_to_trust_the_heads(tmp_path):
	"""CampaignLearner.step end to end on a stub (no games, no servers): the labels, the model's forecasts at the day
	starts that grade the longer heads, the horizon advantages, one PPO update, the heads' smoothed explained variance
	(trust), the checkpoint and the log row."""
	import collections
	import dataclasses

	import numpy as np
	import torch

	from campaign_model.empire import EmpireConfig
	from campaign_model.labels import LabelConfig
	from campaign_model.learner import COLUMNS
	from campaign_model.model import CampaignModel
	from campaign_model.schema import random_inputs
	from campaign_model.tests.test_model import encoded
	from campaign_model.tests.test_train import decisions_for
	from campaign_model.train import make_optimizer

	torch.manual_seed(5)
	kwargs = {"dim": 32, "cell_dim": 16, "layers": 1}
	model = CampaignModel(**kwargs).eval()
	config = CampaignConfig(decisions_per_update=8, min_games_per_update=2, batch_size=4, passes=1, learner="thread", device="cpu")
	games = []
	for g in range(2):
		x = random_inputs(8, seed=20 + g)
		decisions = [dataclasses.replace(d, inputs=encoded(x, b), day=1 + b // 3) for b, d in enumerate(decisions_for(model, x, seed=g))]
		games.append((Game(decisions=decisions, ledgers={d: snapshot(d, 1000 + 300 * d, 1000) for d in range(1, 14)}, days=12 if g == 0 else 30), SimpleNamespace(seconds=1.0, map_name="Maps/alpha_random/x")))

	class Writer:
		def add_scalar(self, *args):
			pass

		def flush(self):
			pass

	learner = SimpleNamespace(
		config=config, labels=LabelConfig(), empire=EmpireConfig(), version=1, finished=list(games), lock=threading.Condition(), stats=collections.Counter(),
		model=model, optimizer=make_optimizer(model, config), device=torch.device("cpu"), worker=None, model_lock=threading.Lock(), lr_scale=1.0,
		server=SimpleNamespace(set_model=lambda *args: None), games_used=0, out=tmp_path, kwargs=kwargs, value_ev=None, credit_evidence={}, credit_check=1.0, head_ev={"Vend": 0.2}, value_pool=None, head_pool={}, progress=tmp_path / "progress.csv", writer=Writer(),
	)
	for name in ("_take", "trust", "_save", "_log", "_note_value_ev", "_note_head_ev", "grade_trust", "end_credit"):
		setattr(learner, name, getattr(CampaignLearner, name).__get__(learner))
	learner.progress.write_text(",".join(COLUMNS) + "\n")
	assert learner.trust() == 0.0  # nothing known yet: the credit leans on the real outcome
	assert learner.end_credit() == pytest.approx((0.5, 1 - 0.75 * 0.5))  # V_end half trusted: half the credit is the end's
	assert CampaignLearner.step(learner, epoch=1)
	row = dict(zip(COLUMNS, learner.progress.read_text().splitlines()[1].split(",")))
	assert float(row["end_credit"]) == pytest.approx(0.5) and row["head_ev_vend"] != "nan"  # (game 0 reached its cap: an end)
	assert learner.version == 2 and learner.value_ev is not None and len(learner.value_ev) == 2 and 0.0 <= learner.trust() <= 1.0
	assert (tmp_path / "checkpoint_latest.pt").exists() and len(learner.progress.read_text().splitlines()) == 2
	saved = torch.load(tmp_path / "checkpoint_latest.pt")
	assert np.isfinite(saved["value_ev"]).all() and set(saved["credit_evidence"]) <= {k.name for k in c.ActionType}
	assert learner.credit_check == 1.0  # (2 game-players: too few to judge a type)
	heads = {f"{kind}{k}" for kind in "VG" for k in c.VALUE_OFFSETS}
	assert heads <= set(learner.head_ev) <= heads | {"Vend"} and set(saved["head_ev"]) == set(learner.head_ev)  # (measured at the day starts)
	assert all(0.0 <= t <= 1.0 for t in learner.grade_trust().values())


def test_a_games_cap_is_memoryless_past_its_minimum():
	"""draw_days: past days_min, the chance that a day is the last is the same whatever day it is (the model can't
	see the end coming); mean days_min + days_tail (a little under: days_max cuts the tail)."""
	import random

	import numpy as np

	from campaign_model.learner import draw_days

	config = CampaignConfig(days_min=28, days_tail=28.0, days_max=112)
	rng = random.Random(0)
	days = np.array([draw_days(rng, config) for _ in range(40000)])
	assert days.min() == 28 and days.max() == 112 and 53 < days.mean() < 56
	hazard = [np.mean(days[days >= d] == d) for d in (30, 50, 80)]
	assert np.allclose(hazard, 1 / 29, atol=0.006)


def test_skill_picks_count_each_skill_taken_and_offered():
	import dataclasses

	from campaign_model.learner import skill_picks
	from campaign_model.schema import random_inputs
	from campaign_model.tests.test_model import encoded

	x = random_inputs(2, seed=3)
	x["o_kind"][:] = c.OptionKind.NONE
	x["o_kind"][:, :2] = c.OptionKind.SKILL
	x["o_type"][0, :2], x["o_type"][1, :2] = (13, 7), (7, 22)  # estates or wisdom; wisdom or offence
	ups = [dataclasses.replace(Decision("RED", 1, int(c.DecisionKind.LEVEL_UP), 0, 1, -1, i, -1, 0.0, (0.5,) * 3, 0, b""), inputs=encoded(x, b)) for b, i in ((0, 1), (1, 1))]
	assert skill_picks(ups) == {"estates": (0, 1), "wisdom": (1, 2), "offence": (1, 1)}
	assert skill_picks([]) == {}
