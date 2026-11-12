"""The credit by segments (config.credit_segments; e5's design, 2026-11-11): the three segments sum to the 7-day credit;
with both long segments' k at 1 the update is the 7-day credit's, at 0 its credit is the 2-day one; the shrinkage reads
k = max(0, 1 - 1/F) from the action types' gaps; the learner moves the k's 0.05 a step toward it, from 0."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.learner import CampaignLearner
from campaign_model.model_v2 import CampaignModelV2
from campaign_model.tests.test_model_v2 import decisions_for, rich_inputs
from campaign_model.train import make_optimizer, ppo_update, segment_shrinkage, segment_surprises

CONFIG = CampaignConfig(batch_size=4, passes=1, model_architecture="v2", credit="schedule", share_target="change")


def labels(n, seed=0):
	rng = np.random.default_rng(seed)
	targets = (0.5 + 0.05 * rng.standard_normal((n, c.N_VALUE_OFFSETS))).astype(np.float32)
	growth = (rng.standard_normal((n, c.N_VALUE_OFFSETS)) * 0.3).astype(np.float32)
	return targets, growth


def test_the_segments_sum_to_the_7_day_credit():
	n = 12
	targets, growth = labels(n)
	targets[3, 1:] = np.nan  # (a decision whose 5- and 7-day labels are unknown)
	value_all, growth_all = torch.randn(n, c.N_VALUE_OFFSETS), torch.randn(n, c.N_VALUE_OFFSETS)
	log_phi, w = torch.randn(n) * 0.1, torch.rand(n)
	pieces = segment_surprises(targets, growth, value_all, growth_all, log_phi, 0.7, w, "cpu")
	alone = segment_surprises(targets[:, -1:], growth[:, -1:], value_all[:, -1:], growth_all[:, -1:], log_phi, 0.7, w, "cpu")[:, 0]
	assert pieces.shape == (n, 3) and torch.isfinite(pieces[3, 0]) and not torch.isfinite(pieces[3, 1:]).any()
	known = torch.isfinite(alone)
	assert torch.allclose(pieces.sum(1)[known], alone[known], atol=1e-5)


def update(model, decisions, x, config, segment_k):
	torch.manual_seed(7)
	m = copy.deepcopy(model)
	optimizer = make_optimizer(m, config)
	n = len(decisions)
	targets, growth = labels(n)
	metrics, _ = ppo_update(m, optimizer, decisions, targets, None, config, decode=lambda rows: {k: v[rows] for k, v in x.items()}, advantages=np.zeros(n, np.float32),
		growth=growth, growth_real=growth, labels=targets, segment_k=segment_k)
	return m, metrics


@pytest.fixture(scope="module")
def setup():
	torch.manual_seed(5)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	x = rich_inputs(8, seed=6)
	return model, x, decisions_for(model.eval(), x)


def test_k_at_1_is_the_7_day_credit_and_off_reports_nothing(setup):
	model, x, decisions = setup
	plain, off = update(model, decisions, x, CONFIG, None)
	assert "segment_surprise" not in off and "segment_credit_std" not in off
	full, on = update(model, decisions, x, replace(CONFIG, credit_segments=True), (1.0, 1.0))
	assert on["segment_surprise"].shape == (len(decisions), 3)
	assert on["segment_credit_std"] == pytest.approx(on["segment_full_std"], rel=1e-5)
	for (name, a), (_, b) in zip(plain.named_parameters(), full.named_parameters()):
		assert torch.allclose(a, b, atol=1e-6), name
	short, at0 = update(model, decisions, x, replace(CONFIG, credit_segments=True), (0.0, 0.0))
	first = at0["segment_surprise"][:, 0]
	assert at0["segment_credit_std"] == pytest.approx(float(np.std(first[np.isfinite(first)], ddof=1)), rel=1e-4)  # (the 2-day credit)
	assert any(not torch.equal(a, b) for (_, a), (_, b) in zip(plain.named_parameters(), short.named_parameters()))


def evidence(means, players=40, spread=1.0, seed=0):
	rng = np.random.default_rng(seed)
	out = {}
	for t, mean in enumerate(means):
		gaps = mean + spread * rng.standard_normal(players)
		out[f"T{t}"] = [float(players), float(gaps.sum()), float((gaps**2).sum())]
	return out


def test_the_shrinkage_reads_the_types_spread_against_sampling():
	k, f, types = segment_shrinkage(evidence([0.0] * 6), 8)
	assert types == 6 and f < 3 and k < 0.7  # (no spread between the types: F near 1)
	k, f, types = segment_shrinkage(evidence([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]), 8)
	assert f > 50 and k > 0.98
	k, f, types = segment_shrinkage(evidence([0.0, 1.0], players=5), 8)
	assert types == 0 and np.isnan(k) and np.isnan(f)


def test_the_learner_moves_k_slowly_toward_its_reading_and_keeps_it():
	"""A type whose 2-5 day surprise stands far above the rest's: k5 climbs 0.05 of the way to ~1 a step, from 0; the
	5-7 segment, no type apart, stays low."""
	learner = CampaignLearner.__new__(CampaignLearner)
	learner.segment_evidence, learner.segment_k = [{}, {}, {}], (0.0, 0.0)
	rng = np.random.default_rng(1)
	n, ks = 600, []
	for _ in range(20):  # (each update its own games)
		decisions = [SimpleNamespace(kind=c.DecisionKind.TURN, action_type=int(t)) for t in rng.integers(0, 8, n)]
		surprises = rng.standard_normal((n, 3)).astype(np.float32)
		surprises[:, 1] += 3.0 * (np.array([d.action_type for d in decisions]) == 0)
		row = learner._note_segments(decisions, surprises, np.arange(n) % 16)
		ks.append(learner.segment_k)
	assert row["segment_types5"] == 8 and row["segment_f5"] > 50 and row["segment_k5"] == pytest.approx(ks[-2][0])  # (the row: the k this update used)
	assert ks[0][0] == pytest.approx(0.05, abs=0.002) and ks[-1][0] == pytest.approx(1 - 0.95**20, abs=0.01)
	assert ks[-1][1] < 0.1


def test_the_learner_step_end_to_end(tmp_path, capsys):
	"""CampaignLearner.step at deploy 60's line with the segments on: the columns in progress.csv, the update's line, the
	k's and the evidence in the checkpoint."""
	import collections
	import csv
	import dataclasses
	import threading

	from campaign_model.empire import EmpireConfig
	from campaign_model.labels import LabelConfig
	from campaign_model.learner import COLUMNS, set_advice_prior
	from campaign_model.model import build_model
	from campaign_model.records import Game
	from campaign_model.schema import random_inputs
	from campaign_model.tests.test_labels import snapshot
	from campaign_model.tests.test_model import encoded
	from campaign_model.tests.test_train import decisions_for as train_decisions

	torch.manual_seed(6)
	kwargs = {"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"}
	model = build_model(kwargs).eval()
	config = CampaignConfig(decisions_per_update=8, min_games_per_update=2, batch_size=4, passes=1, learner="thread", device="cpu", credit="schedule", share_target="change", model_architecture="v2", noise_every=0, adv_weight=1.0, adv_gain=1.0, nk2_follow=0.0, nk2_choice_games=0.0, schedule_action_price=2e-6, credit_segments=True)
	set_advice_prior(model, config)
	games = []
	for g in range(2):
		x = random_inputs(8, seed=30 + g)
		decisions = [dataclasses.replace(d, inputs=encoded(x, b), day=1 + b // 3) for b, d in enumerate(train_decisions(model, x, seed=g))]
		done = {"RED": {"advice": {"asked": 4, "choice_shown": 0}}, "BLUE": {"advice": {"asked": 4, "choice_shown": 0}}}
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
		segment_evidence=[{}, {}, {}], segment_k=(0.0, 0.0),
	)
	for name in ("_take", "trust", "trust_today", "_save", "_log", "_note_value_ev", "_note_head_ev", "grade_trust", "end_credit", "_refresh_features", "_relabel_waiting", "loss_shares", "loss_weights", "_note_gradients", "gather_weight_now", "valve_now", "_note_valve", "_valve_reading", "_head_scale", "_calibrate_clips", "_judge_value", "_note_segments"):
		setattr(learner, name, getattr(CampaignLearner, name).__get__(learner))
	learner.progress.write_text(",".join(COLUMNS) + "\n")
	capsys.readouterr()
	assert CampaignLearner.step(learner, epoch=1)
	printed = capsys.readouterr().out
	row = list(csv.DictReader(open(learner.progress)))[-1]
	assert "credit by segments: k5 0.000 k7 0.000" in printed and "FAILURE" not in printed
	assert float(row["segment_k5"]) == 0.0 and row["segment_types5"] != "" and float(row["segment_credit_std"]) > 0 and float(row["segment_full_std"]) > 0
	assert all(isinstance(e, dict) for e in learner.segment_evidence) and any(learner.segment_evidence)
	assert float(row["loops_per_player_day"]) == 0.0  # (no loops in these games; the column there)
	for name in ("signal_all", "signal_trunk", "signal_body", "signal_heads", "ratio_body", "ratio_heads", "lr_scale", "kl", "learning_rate"):  # (e5's learning-rate readings)
		assert row[name] not in ("", "nan"), name
	assert 0 < float(row["ratio_body"]) <= 1 and 0 < float(row["ratio_heads"]) <= 1
	saved = torch.load(tmp_path / "checkpoint_latest.pt", weights_only=False)
	assert saved["segment_k"] == learner.segment_k and len(saved["segment_evidence"]) == 3


def test_the_trunk_moments_match_the_groups_signal_fraction_when_pooled():
	"""train.trunk_moments' two parts pooled by their weights' counts are signal_fraction's trunk reading."""
	from campaign_model.train import SHARED_PREFIXES, signal_fraction, trunk_moments

	torch.manual_seed(2)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	optimizer = make_optimizer(model, CONFIG)
	for _ in range(3):
		for p in model.parameters():
			p.grad = torch.randn_like(p) + 0.5
		optimizer.step()
	out = trunk_moments(model, optimizer)
	trunk = next(g for g in optimizer.param_groups if g["name"] == "trunk")
	names = {id(p): n for n, p in model.named_parameters()}
	n_body = sum(p.numel() for p in trunk["params"] if names[id(p)].startswith(SHARED_PREFIXES))
	n_heads = sum(p.numel() for p in trunk["params"]) - n_body
	assert n_body > 0 and n_heads > 0
	pooled = (out["signal_body"] * n_body + out["signal_heads"] * n_heads) / (n_body + n_heads)
	assert pooled == pytest.approx(signal_fraction(optimizer)["trunk"], rel=1e-4)
