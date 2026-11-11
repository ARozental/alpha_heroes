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
	new_phi = row(5, [0.1, 0.1, 0.1], temperature=0.0)
	for game in new_phi["per_game"]:
		game["empire"] = 3  # Φ's rules changed again: another measure
	log(new_phi, path)
	assert math.isnan(new_phi["vs_previous"])


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


def test_a_backlog_beyond_four_updates_drops_its_oldest_games():
	"""(2026-10-30) The supply starts games only while under `ahead` updates' worth wait (supply.Supply.allow), so the
	learner drops nothing short of a safety cap of DROP_BEYOND (4) updates' worth: games that were in flight as it stalled."""
	from campaign_model.learner import DROP_BEYOND

	assert DROP_BEYOND == 4
	# (an update's worth: what it takes, 2 games of 60 here, 120 decisions, over decisions_per_update's 100: 2026-11-01)
	learner = SimpleNamespace(config=CampaignConfig(decisions_per_update=100, min_games_per_update=2), lock=threading.Condition(), finished=[], stats=__import__("collections").Counter())
	learner.finished = [(game_of(60), i) for i in range(9)]  # 540 decisions: 480 past the oldest, four updates' worth
	taken = CampaignLearner._take(learner)
	assert [info for _, info in taken] == [1, 2] and [info for _, info in learner.finished] == list(range(3, 9)) and learner.stats["dropped_games"] == 1
	learner.finished = [(game_of(60), i) for i in range(12)]  # 720: four more past it
	taken = CampaignLearner._take(learner)
	assert [info for _, info in taken] == [4, 5] and learner.stats["dropped_games"] == 5


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
	config = CampaignConfig(decisions_per_update=8, min_games_per_update=2, batch_size=4, passes=1, learner="thread", device="cpu", credit="horizon", share_target="share")  # (the horizon credit on share heads: no longer the defaults)
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
		server=SimpleNamespace(set_model=lambda *args: None), teacher=None, teacher_weight=0.0, games_used=0, out=tmp_path, kwargs=kwargs, value_ev=None, credit_evidence={}, credit_check=1.0, head_ev={"Vend": 0.2}, value_pool=None, head_pool={}, progress=tmp_path / "progress.csv", writer=Writer(),
		features_cpu=model.features, trust_ema=None, grad_clips=None, clip_samples=[], head_residual=None, head_var_slow=None, head_residual_updates=0, held=None, held_updates=0, gather_ramp_start=None, valve=None, value_readings=[], value_own=0,
	)
	for name in ("_take", "trust", "trust_today", "_save", "_log", "_note_value_ev", "_note_head_ev", "grade_trust", "end_credit", "_refresh_features", "_relabel_waiting", "loss_shares", "loss_weights", "_note_gradients", "gather_weight_now", "valve_now", "_note_valve", "_valve_reading", "_head_scale", "_calibrate_clips", "_judge_value"):
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
	assert "gather_ramp_start" in saved and float(row["gather_weight"]) == 0.0 and row["eliminated"] != ""  # (GATHER's ramp: a restart goes on)
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


def test_the_labeler_computes_what_step_did(tmp_path):
	"""(2026-10-30) The labels a game needs that no model forward gives are computed as it arrives, in a labeler process
	(labeler.py), and are what learner.step computed in the trainer's thread: the same arrays."""
	import dataclasses

	import numpy as np
	import torch

	from campaign_model.foresight import labels_of
	from campaign_model.labeler import Labeler, precompute
	from campaign_model.labels import EmpireConfig, LabelConfig, end_labels, end_steps, growth_labels, label
	from campaign_model.model import CampaignModel
	from campaign_model.schema import random_inputs
	from campaign_model.tests.test_model import encoded
	from campaign_model.tests.test_train import decisions_for
	from campaign_model.train import usable

	torch.manual_seed(5)
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	x = random_inputs(8, seed=20)
	decisions = [dataclasses.replace(d, inputs=encoded(x, b), day=1 + b // 3) for b, d in enumerate(decisions_for(model, x, seed=0))]
	game = Game(decisions=decisions, ledgers={d: snapshot(d, 1000 + 300 * d, 1000) for d in range(1, 14)}, days=12)
	pre = precompute(game, model.features.state_dict(), LabelConfig(), EmpireConfig(), foresight=True)
	kept = usable(game.decisions)
	assert pre["n"] == len(kept) > 0
	g = dataclasses.replace(game, decisions=kept)
	g.all_decisions = game.decisions
	t, mixed, price = label(g, LabelConfig(), EmpireConfig())
	np.testing.assert_array_equal(pre["t"], t)
	np.testing.assert_array_equal(pre["mixed"], mixed)
	np.testing.assert_array_equal(pre["price"], price)
	np.testing.assert_array_equal(pre["z"], growth_labels(g, EmpireConfig()))
	np.testing.assert_array_equal(pre["real"], end_labels(g, EmpireConfig()))
	for got, want in zip((pre["step"], pre["week"]), end_steps(g)):
		np.testing.assert_array_equal(got, want)
	rows = labels_of(g, model.features)
	assert set(pre["foresight"]) == set(rows)
	for name in rows:
		np.testing.assert_array_equal(pre["foresight"][name], rows[name])
	labeler = Labeler(foresight=True, workers=1)  # the same through a labeler process (the game pickled there and back)
	try:
		again = labeler.submit(game, model.features).result(timeout=120)
	finally:
		labeler.close()
	assert again["n"] == pre["n"] and np.array_equal(again["t"], pre["t"]) and np.array_equal(again["foresight"]["turn"], pre["foresight"]["turn"])


def test_an_update_runs_with_the_attributed_credit(tmp_path):
	"""CampaignLearner.step end to end with credit "attributed" (2026-10-31): our share and what we took per decision, the
	growth head's forecast at the update's start, the end credit standardized and mixed in, one PPO update."""
	import collections
	import dataclasses

	import torch

	from campaign_model.empire import EmpireConfig
	from campaign_model.labels import LabelConfig
	from campaign_model.learner import COLUMNS
	from campaign_model.model import CampaignModel
	from campaign_model.schema import random_inputs
	from campaign_model.tests.test_model import encoded
	from campaign_model.tests.test_train import decisions_for
	from campaign_model.train import make_optimizer

	torch.manual_seed(6)
	kwargs = {"dim": 32, "cell_dim": 16, "layers": 1}
	model = CampaignModel(**kwargs).eval()
	config = CampaignConfig(decisions_per_update=8, min_games_per_update=2, batch_size=4, passes=1, learner="thread", device="cpu", credit="attributed", noise_every=0)
	games = []
	for g in range(2):
		x = random_inputs(8, seed=30 + g)
		decisions = [dataclasses.replace(d, inputs=encoded(x, b), day=1 + b // 3) for b, d in enumerate(decisions_for(model, x, seed=g))]
		games.append((Game(decisions=decisions, ledgers={d: snapshot(d, 1000 + 300 * d, 1000) for d in range(1, 14)}, days=12), SimpleNamespace(seconds=1.0, map_name="Maps/alpha_random/x")))

	class Writer:
		def add_scalar(self, *args):
			pass

		def flush(self):
			pass

	learner = SimpleNamespace(
		config=config, labels=LabelConfig(), empire=EmpireConfig(), version=1, finished=list(games), lock=threading.Condition(), stats=collections.Counter(),
		model=model, optimizer=make_optimizer(model, config), device=torch.device("cpu"), worker=None, model_lock=threading.Lock(), lr_scale=1.0,
		server=SimpleNamespace(set_model=lambda *args: None), teacher=None, teacher_weight=0.0, games_used=0, out=tmp_path, kwargs=kwargs, value_ev=None, credit_evidence={}, credit_check=1.0, head_ev={"Vend": 0.2}, value_pool=None, head_pool={}, progress=tmp_path / "progress.csv", writer=Writer(),
		features_cpu=model.features, trust_ema=None, grad_clips=None, clip_samples=[], head_residual=None, head_var_slow=None, head_residual_updates=0, held=None, held_updates=0, gather_ramp_start=None, valve=None, value_readings=[], value_own=0,
	)
	for name in ("_take", "trust", "trust_today", "_save", "_log", "_note_value_ev", "_note_head_ev", "grade_trust", "end_credit", "_refresh_features", "_relabel_waiting", "loss_shares", "loss_weights", "_note_gradients", "gather_weight_now", "valve_now", "_note_valve", "_valve_reading", "_head_scale", "_calibrate_clips", "_judge_value"):
		setattr(learner, name, getattr(CampaignLearner, name).__get__(learner))
	learner.progress.write_text(",".join(COLUMNS) + "\n")
	assert CampaignLearner.step(learner, epoch=1)
	row = dict(zip(COLUMNS, learner.progress.read_text().splitlines()[1].split(",")))
	assert learner.version == 2 and float(row["end_credit"]) == pytest.approx(config.attributed_end_weight)


def test_the_teacher_weight_follows_its_kl_target():
	from campaign_model.learner import adapted_teacher_weight

	assert adapted_teacher_weight(0.2, 0.5, 0.1) == pytest.approx(0.3)  # far over: stronger
	assert adapted_teacher_weight(0.2, 0.01, 0.1) == pytest.approx(0.2 / 1.5)  # far under: weaker
	assert adapted_teacher_weight(0.2, 0.1, 0.1) == 0.2  # near: kept
	assert adapted_teacher_weight(0.2, 0.5, 0.0) == 0.2  # no target: the fixed weight
	assert adapted_teacher_weight(9.0, 5.0, 0.1) == 10.0 and adapted_teacher_weight(0.01, 0.0, 0.1) == 0.01  # bounded


def test_the_teachers_weight_follows_our_score_against_it():
	from campaign_model.learner import teacher_weight_for

	share = lambda score: teacher_weight_for(score) / (1 + teacher_weight_for(score))  # noqa: E731
	assert share(0.1) == pytest.approx(0.95) and share(0.5) == pytest.approx(0.7)  # much better than us: it dominates
	assert share(1.0) == pytest.approx(0.2) and share(2.0) == pytest.approx(0.05)  # as good or worse: small, never 0


def test_an_end_agrees_with_the_advice_of_an_end_whatever_its_index():
	from campaign_model.learner import agrees

	ours = lambda action, index, advice: SimpleNamespace(action_type=action, index=index, advice=advice)  # noqa: E731
	assert agrees(ours(c.ActionType.END, -1, (c.ActionType.END, 0, -1)))  # (ours has no index, the advice's is 0)
	assert agrees(ours(c.ActionType.ARMY, 7, (c.ActionType.ARMY, 7, -1)))
	assert not agrees(ours(c.ActionType.ARMY, 7, (c.ActionType.ARMY, 8, -1)))
	assert not agrees(ours(c.ActionType.ARMY, 7, (c.ActionType.END, 0, -1)))


def test_an_update_runs_with_the_schedule_credits_action_price(tmp_path):
	"""CampaignLearner.step end to end with credit "schedule" and config.schedule_action_price (the user, 2026-11-09):
	each decision's actions through its week (labels.actions_ahead) reach the update with the end's arrays, and the
	update's stats carry the charge; with the price at 0 (the default) none are counted."""
	import collections
	import dataclasses

	import torch

	from campaign_model.empire import EmpireConfig
	from campaign_model.labels import LabelConfig
	from campaign_model.learner import COLUMNS
	from campaign_model.model import build_model
	from campaign_model.schema import random_inputs
	from campaign_model.tests.test_model import encoded
	from campaign_model.tests.test_train import decisions_for
	from campaign_model.train import make_optimizer
	import campaign_model.learner as learner_module

	seen = []
	real_update = learner_module.ppo_update

	def spy(*args, **kwargs):
		seen.append(kwargs.get("end"))
		return real_update(*args, **kwargs)

	for price in (2e-6, 0.0):
		torch.manual_seed(6)
		kwargs = {"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"}
		model = build_model(kwargs).eval()
		config = CampaignConfig(decisions_per_update=8, min_games_per_update=2, batch_size=4, passes=1, learner="thread", device="cpu", credit="schedule", share_target="change", model_architecture="v2", noise_every=0, schedule_action_price=price)
		games = []
		for g in range(2):
			x = random_inputs(8, seed=30 + g)
			decisions = [dataclasses.replace(d, inputs=encoded(x, b), day=1 + b // 3) for b, d in enumerate(decisions_for(model, x, seed=g))]
			games.append((Game(decisions=decisions, ledgers={d: snapshot(d, 1000 + 300 * d, 1000) for d in range(1, 14)}, days=12), SimpleNamespace(seconds=1.0, map_name="Maps/alpha_random/x")))

		class Writer:
			def add_scalar(self, *args):
				pass

			def flush(self):
				pass

		out = tmp_path / str(price)
		out.mkdir()
		learner = SimpleNamespace(
			config=config, labels=LabelConfig(), empire=EmpireConfig(), version=1, finished=list(games), lock=threading.Condition(), stats=collections.Counter(),
			model=model, optimizer=make_optimizer(model, config), device=torch.device("cpu"), worker=None, model_lock=threading.Lock(), lr_scale=1.0,
			server=SimpleNamespace(set_model=lambda *args: None), teacher=None, teacher_weight=0.0, games_used=0, out=out, kwargs=kwargs, value_ev=None, credit_evidence={}, credit_check=1.0, head_ev={"Vend": 0.2}, value_pool=None, head_pool={}, progress=out / "progress.csv", writer=Writer(),
			features_cpu=model.features, trust_ema=None, grad_clips=None, clip_samples=[], head_residual=None, head_var_slow=None, head_residual_updates=0, held=None, held_updates=0, gather_ramp_start=None, valve=None, value_readings=[], value_own=0,
		)
		for name in ("_take", "trust", "trust_today", "_save", "_log", "_note_value_ev", "_note_head_ev", "grade_trust", "end_credit", "_refresh_features", "_relabel_waiting", "loss_shares", "loss_weights", "_note_gradients", "gather_weight_now", "valve_now", "_note_valve", "_valve_reading", "_head_scale", "_calibrate_clips", "_judge_value"):
			setattr(learner, name, getattr(CampaignLearner, name).__get__(learner))
		learner.progress.write_text(",".join(COLUMNS) + "\n")
		learner_module.ppo_update = spy
		try:
			assert CampaignLearner.step(learner, epoch=1)
		finally:
			learner_module.ppo_update = real_update
		assert learner.version == 2
		end = seen[-1]
		if price:
			counted = sum(len(game.decisions) for game, _ in games)
			assert len(end["actions"]) == len(end["price"]) == counted and float(end["actions"].max()) >= 1 and float(end["actions"].min()) >= 0
		else:
			assert "actions" not in end


def test_gathers_label_ramps_in_and_the_collapse_readings():
	"""config.nk2_gather_ramp: the weight at the ramp's first update is 1 / ramp of nk2_gather_weight, the start that
	version (kept for the checkpoint), then a straight line to the weight, never past it; no ramp (or no weight): the
	weight as configured. army_on_heroes: the heroes' army worth over all of it (garrisons, armies no hero leads), players
	still in; eliminated_share: players LOSER at a game's last snapshot."""
	import numpy as np

	from campaign_model.empire import EmpireConfig, army_worth
	from campaign_model.learner import army_on_heroes, eliminated_share

	learner = SimpleNamespace(config=CampaignConfig(nk2_gather_weight=0.1, nk2_gather_ramp=200), version=50, gather_ramp_start=None)
	now = lambda: CampaignLearner.gather_weight_now(learner)  # noqa: E731
	assert now() == pytest.approx(0.1 / 200) and learner.gather_ramp_start == 50
	learner.version = 149
	assert now() == pytest.approx(0.1 * 100 / 200)
	learner.version = 400
	assert now() == pytest.approx(0.1)
	assert CampaignLearner.gather_weight_now(SimpleNamespace(config=CampaignConfig(nk2_gather_weight=0.1), version=7, gather_ramp_start=None)) == 0.1
	assert CampaignLearner.gather_weight_now(SimpleNamespace(config=CampaignConfig(nk2_gather_ramp=200), version=7, gather_ramp_start=None)) == 0.0

	stacks = [[0, 10]]  # (ten pikemen)
	def player(status, on_hero, in_town):
		return {"color": 0, "status": status, "heroes": [{"army": stacks}] if on_hero else [], "towns": [{"garrison": stacks}] if in_town else [], "armies": []}
	games = [(Game(ledgers={1: {"day": 1, "players": [player(0, True, True), player(1, True, False)]}, 2: {"day": 2, "players": [player(0, True, False), player(0, False, True)]}}), None)]
	assert army_on_heroes(games, EmpireConfig()) == pytest.approx(2 / 4) and army_worth(stacks, EmpireConfig()) > 0  # (day 1's second player is out)
	assert eliminated_share(games) == 0.0
	games[0][0].ledgers[3] = {"day": 3, "players": [player(0, True, False), player(1, False, False)]}
	assert eliminated_share(games) == 0.5 and np.isnan(eliminated_share([]))


def test_a_reading_that_fails_is_nan():
	"""learner.reading: the update's diagnostics (army_on_heroes, eliminated_share) give nan on any error, never the
	update's end (games whose snapshots are not what they expect)."""
	import math

	from campaign_model.learner import army_on_heroes, eliminated_share, reading

	broken = [(SimpleNamespace(ledgers={1: {"day": 1}}, final=[]), None)]  # (a snapshot without its players)
	assert math.isnan(reading(army_on_heroes, broken, None)) and math.isnan(reading(eliminated_share, broken))
	assert reading(lambda a: a + 1, 2) == 3


def test_the_value_valves_close_on_a_jump_and_reopen_as_its_level_is_absorbed(tmp_path):
	"""learner.valve_step: steady readings leave both valves open; a lasting jump to 3x closes s and k (to ~0.41: the
	reference absorbs ~2% of it an update meanwhile), at most /1.25 an update; as the reference absorbs the new level both reopen, at most x1.25 an update; a missing reading
	changes nothing. valve_reference starts from progress.csv's readings (their median), not from the first one; valve_now
	gives the state's own s and k back (a restart does not snap them to 1) and 1, 1 with the switches off."""
	import numpy as np

	from campaign_model.learner import VALVE_K_MIN, VALVE_RATE, VALVE_S_MIN, valve_reference, valve_step

	state = valve_reference([0.001] * 60)
	for _ in range(20):
		state = valve_step(state, 0.001)
	assert state["s"] == 1.0 and state["k"] == 1.0 and abs(state["r"] - 1.0) < 1e-9
	s_seen, k_seen = [], []
	for _ in range(200):
		previous = dict(state)
		state = valve_step(state, 0.003)
		assert state["s"] >= previous["s"] / VALVE_RATE - 1e-12 and state["s"] <= previous["s"] * VALVE_RATE + 1e-12
		s_seen.append(state["s"])
		k_seen.append(state["k"])
	assert min(s_seen) < 0.45 and min(k_seen) < 0.45 and min(s_seen) >= VALVE_S_MIN and min(k_seen) >= VALVE_K_MIN  # (the 7-median waits 4 readings while the reference absorbs: ~0.41)
	assert s_seen[-1] > 0.95 and k_seen[-1] > 0.95  # (the new level absorbed: open again)
	assert valve_step(state, float("nan")) == state
	progress = tmp_path / "progress.csv"
	progress.write_text("version,value_loss_fresh\n" + "".join(f"{v},{0.002 if v < 40 else 0.009}\n" for v in range(60)))
	learner = SimpleNamespace(config=CampaignConfig(value_valve=True, kl_valve=True), progress=progress, version=60, valve=None)
	assert CampaignLearner.valve_now(learner) == (1.0, 1.0) and learner.valve["reference"] == pytest.approx(0.002)  # (the median of 50: the last 20 at 0.009 do not set it)
	learner.valve.update(s=0.3, k=0.5)
	assert CampaignLearner.valve_now(learner) == (0.3, 0.5)
	off = SimpleNamespace(config=CampaignConfig(), progress=progress, version=60, valve=None)
	assert CampaignLearner.valve_now(off) == (1.0, 1.0) and off.valve is None and np.isnan(np.nan)
