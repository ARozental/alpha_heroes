"""Own games (config.nk2_choice_games; the user, 2026-11-11): a share of the training games hides Nullkiller2's own
choice from the inputs (ALPHA_MAP_NK2_CHOICE=0), one setting a game drawn from its seed apart from the inputs' dropout;
evaluations show it; a hidden-choice decision is our own policy's, the follow mixture off; the update's readings
apart for the two kinds of games."""
from types import SimpleNamespace

import numpy as np
import torch

from campaign_model.config import CampaignConfig
from campaign_model.learner import choice_readings
from campaign_model.model_v2 import CampaignModelV2
from campaign_model.play import choice_hidden, nk2_settings
from campaign_model.records import Decision, Game
from campaign_model.tests.test_model_v2 import rich_inputs


def test_the_default_shows_it_as_before():
	config = CampaignConfig(nk2_inputs=True)
	for seed in range(200):
		assert dict(nk2_settings(config, seed)) == {"ALPHA_MAP_NK2_INPUTS": "1", "ALPHA_MAP_GATHER_ADVICE": "1", "ALPHA_MAP_NK2_CHOICE": "1"}
	assert dict(nk2_settings(CampaignConfig(), 5)) == {"ALPHA_MAP_NK2_INPUTS": "0"}  # (without the inputs: no choice to hide)


def test_a_share_of_training_games_hides_it_drawn_apart_from_the_dropout_and_evaluations_show_it():
	config = CampaignConfig(nk2_inputs=True, nk2_choice_games=0.5)
	hidden = [dict(nk2_settings(config, seed))["ALPHA_MAP_NK2_CHOICE"] == "0" for seed in range(4000)]
	assert 0.46 < np.mean(hidden) < 0.54
	assert [dict(nk2_settings(config, seed))["ALPHA_MAP_NK2_CHOICE"] for seed in range(50)] == [dict(nk2_settings(config, seed))["ALPHA_MAP_NK2_CHOICE"] for seed in range(50)]  # (the seed's, every machine alike)
	assert dict(nk2_settings(config))["ALPHA_MAP_NK2_CHOICE"] == "1"  # (an evaluation, a paired test: seed None)
	both = CampaignConfig(nk2_inputs=True, nk2_choice_games=0.5, nk2_input_dropout=0.5)
	kinds = [tuple(sorted(dict(nk2_settings(both, seed)).items())) for seed in range(4000)]
	off = sum(dict(k)["ALPHA_MAP_NK2_INPUTS"] == "0" for k in kinds) / len(kinds)
	hid = sum(dict(k).get("ALPHA_MAP_NK2_CHOICE") == "0" for k in kinds) / len(kinds)
	assert 0.46 < off < 0.54 and 0.21 < hid < 0.29  # (independent draws: hidden in half of the games with the inputs)


def test_a_games_players_share_the_setting():
	"""The setting is the game's environment (supply.Runner.campaign_settings: one tuple for the whole game, every
	player's AI reads it), never a player's: no colour-keyed choice switch."""
	from battle_model.supply import Runner

	config = SimpleNamespace(campaign=CampaignConfig(nk2_inputs=True, nk2_choice_games=0.5))
	runner = SimpleNamespace(battle_socket="b", link="l", records=SimpleNamespace(path="r"), config=config)
	for seed in range(40):
		settings = Runner.campaign_settings(runner, f"k{seed}", seed)
		names = [name for name, _ in settings if "NK2_CHOICE" in name]
		assert names == ["ALPHA_MAP_NK2_CHOICE"], names
		assert dict(settings)["ALPHA_MAP_NK2_CHOICE"] == dict(nk2_settings(config.campaign, seed))["ALPHA_MAP_NK2_CHOICE"]


def test_a_hidden_choice_decision_is_our_own_policys():
	"""Its choice left 0 (the hidden game's inputs): the outputs with the follow mixture at 0.9 are the outputs with it off."""
	torch.manual_seed(3)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = rich_inputs(12, 4)
	assert (x["g_nk_action"] > 0).any() and (x["k_hero"] > 0).any()
	for name in ("g_nk_action", "g_nk_index", "g_nk_cell"):
		x[name][:] = 0
	t = {k: torch.from_numpy(v) for k, v in x.items()}
	with torch.no_grad():
		model.features.nk_follow.fill_(0.9)
		followed = model(t)
		model.features.nk_follow.fill_(0.0)
		own = model(t)
	for name, value in own.items():
		assert torch.equal(value, followed[name]), name


def game_with(shown, colors=("RED", "BLUE"), out=()):
	"""A game whose players' summaries say whether its choice was shown (None: no word, a game from before the switch)."""
	done = {color: ({"advice": {"asked": 3, "choice_shown": shown}} if shown is not None else {}) for color in colors}
	decisions = [Decision("RED", 1, 0, 0, 1, 1, 5, -1, -0.1, (0.0,) * 3, 0, b"", advice=(1, 5, -1)), Decision("RED", 1, 0, 1, 1, 0, -1, -1, -0.1, (0.0,) * 3, 0, b"", advice=(1, 7, -1))]
	players = [{"color": color, "status": 1 if color in out else 0} for color in colors]  # (empire.STATUS_LOST 1)
	return Game(decisions=decisions, done=done, ledgers={1: {"day": 1, "players": players}})


def test_a_game_says_whether_it_hid_the_choice_and_the_readings_go_apart():
	assert choice_hidden(game_with(0)) and not choice_hidden(game_with(1)) and not choice_hidden(game_with(None))
	games = [(game_with(0, out=("BLUE",)), None), (game_with(0), None), (game_with(1), None), (game_with(None), None)]
	read = choice_readings(games, {"adv_ev_shown": 0.1, "adv_ev_hidden": 0.2})
	assert read["choice_hidden"] == 0.5
	assert read["teacher_agree_shown"] == 0.5 and read["teacher_agree_hidden"] == 0.5  # (each game: one of its two choices its advice)
	assert read["adv_ev_shown"] == 0.1 and read["adv_ev_hidden"] == 0.2
	assert read["eliminated_hidden"] == 0.25 and read["eliminated_shown"] == 0.0
	assert np.isnan(choice_readings([(game_with(1), None)], {})["teacher_agree_hidden"])


def test_the_heads_unnamed_types_count_only_where_no_choice_is_shown():
	"""AdvantageHead.unnamed: 0 to start (the head as before), and once set it moves adv_type at the decisions without
	Nullkiller2's choice alone."""
	torch.manual_seed(3)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	model.adv_outputs = True
	x = rich_inputs(12, 4)
	t = {k: torch.from_numpy(v) for k, v in x.items()}
	with torch.no_grad():
		assert float(model.advantage.unnamed.abs().max()) == 0.0
		before = model(t)["adv_type"]
		model.advantage.unnamed.fill_(0.5)
		after = model(t)["adv_type"]
	none = torch.from_numpy(x["g_nk_action"] == 0)
	assert none.any() and (~none).any()
	assert torch.allclose(after[none], before[none] + 0.5) and torch.equal(after[~none], before[~none])


def test_an_update_reads_the_head_apart_for_the_two_kinds_of_games():
	from campaign_model.tests.test_advantage_head import setup, update

	model, x, decisions = setup()
	advantages = np.random.default_rng(3).standard_normal(len(decisions)).astype(np.float32)
	hidden = np.arange(len(decisions)) % 2 == 0
	optimizer_config = CampaignConfig(batch_size=8, passes=1, model_architecture="v2", credit="schedule", share_target="change", adv_weight=1.0)
	from campaign_model.tests.test_advantage_head import CLIPS
	from campaign_model.train import make_optimizer, ppo_update

	torch.manual_seed(7)
	from campaign_model import constants as c

	targets = (0.5 + 0.05 * np.random.default_rng(0).standard_normal((len(decisions), c.N_VALUE_OFFSETS))).astype(np.float32)
	metrics, _ = ppo_update(model, make_optimizer(model, optimizer_config), decisions, targets, None, optimizer_config, decode=lambda rows: {k: v[rows] for k, v in x.items()}, advantages=advantages, clips=CLIPS, hidden=hidden)
	for name in ("adv_ev_shown", "adv_ev_hidden", "adv_raw_sd_shown", "adv_raw_sd_hidden"):
		assert np.isfinite(metrics[name]), name


def test_the_learner_logs_every_reading_of_deploy_59(tmp_path, capsys):
	"""CampaignLearner.step end to end as deploy 59's line runs it (the head at weight 1 and read at gain 1, half the games
	hiding the choice, the action price at 2e-6): every new column in progress.csv, every part of the update's line."""
	import collections
	import csv
	import dataclasses
	import threading

	from campaign_model.empire import EmpireConfig
	from campaign_model.labels import LabelConfig
	from campaign_model.learner import COLUMNS, CampaignLearner, set_advice_prior
	from campaign_model.model import build_model
	from campaign_model.schema import random_inputs
	from campaign_model.tests.test_labels import snapshot
	from campaign_model.tests.test_model import encoded
	from campaign_model.tests.test_train import decisions_for
	from campaign_model.train import make_optimizer

	torch.manual_seed(6)
	kwargs = {"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"}
	model = build_model(kwargs).eval()
	config = CampaignConfig(decisions_per_update=8, min_games_per_update=2, batch_size=4, passes=1, learner="thread", device="cpu", credit="schedule", share_target="change", model_architecture="v2", noise_every=0, adv_weight=1.0, adv_gain=1.0, nk2_choice_games=0.5, schedule_action_price=2e-6)
	set_advice_prior(model, config)
	games = []
	for g in range(2):
		x = random_inputs(8, seed=30 + g)
		decisions = [dataclasses.replace(d, inputs=encoded(x, b), day=1 + b // 3, advice=(d.action_type, d.index, d.cell) if b % 2 else None) for b, d in enumerate(decisions_for(model, x, seed=g))]
		done = {"RED": {"advice": {"asked": 4, "choice_shown": g}}, "BLUE": {"advice": {"asked": 4, "choice_shown": g}}}
		games.append((Game(decisions=decisions, ledgers={d: snapshot(d, 1000 + 300 * d, 1000) for d in range(1, 14)}, days=12, done=done), SimpleNamespace(seconds=1.0, map_name="Maps/alpha_random/x")))

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
	capsys.readouterr()
	assert CampaignLearner.step(learner, epoch=1)
	printed = capsys.readouterr().out
	row = list(csv.DictReader(open(learner.progress)))[-1]
	for part in ("advantage head: loss", "read by our policy at", "action price", "its choice hidden in 50% of the games", "idle movement", "a day begun with END"):
		assert part in printed, part
	assert float(row["choice_hidden"]) == 0.5 and float(row["action_price_mean"]) > 0 and float(row["adv_kl"]) >= 0 and row["adv_read"] not in ("", "nan")
	assert "FAILURE" not in printed
