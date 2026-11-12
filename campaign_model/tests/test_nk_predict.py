"""nk_predict's stage 1 (2026-11-11): the head never sees the planner (its blind pass reads none of the planner's fields
nor the memory), its targets from the live rows, its loss reaches its own weights alone, it learns, weight 0 computes
nothing, and the learner logs its fit."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model import nk_predict as N
from campaign_model.config import CampaignConfig
from campaign_model.model_v2 import NKP_ROWS, CampaignModelV2
from campaign_model.tests.test_model_v2 import decisions_for, rich_inputs
from campaign_model.train import GRAD_GROUPS, make_optimizer, ppo_update

HO, W = c.OWN_HEROES, c.WINDOW


def model_and_inputs(rows=16, seed=4):
	torch.manual_seed(3)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = {k: torch.from_numpy(v) for k, v in rich_inputs(rows, seed).items()}
	return model, x


def head_of(model, x):
	model.nkp_outputs = True
	with torch.no_grad():
		out = model(x)
	model.nkp_outputs = False
	return {k: v for k, v in out.items() if k.startswith("nkp_")}, {k[4:]: v for k, v in out.items() if k.startswith("nkt:")}


def test_the_blind_pass_reads_none_of_the_planner_nor_the_memory():
	model, x = model_and_inputs()
	pred, _ = head_of(model, x)
	assert pred["nkp_move"].shape[0] == max(1, 16 // NKP_ROWS)
	changed = dict(x)
	g = torch.Generator().manual_seed(9)
	for name in N.BLIND:
		changed[name] = torch.randint(0, 50, x[name].shape, generator=g).to(x[name].dtype)
	changed["mem"] = torch.randint(-1000, 1000, x["mem"].shape, generator=g).to(x["mem"].dtype)
	other, targets = head_of(model, changed)
	for name in pred:
		assert torch.equal(pred[name], other[name]), name
	assert any(not torch.equal(targets[k], head_of(model, x)[1][k]) for k in targets)  # (the targets are the live rows')
	ours = dict(x)
	ours["h_army_count"] = x["h_army_count"] * 3 + 1  # (a field of ours: the blind pass sees it)
	assert any(not torch.equal(pred[k], head_of(model, ours)[0][k]) for k in pred)


def one_candidate(x, row=0, hero=0, cell=None, area=None, score=500):
	"""x with every candidate row cleared but one of hero `hero` at MOVE cell `cell` of its own window (or TRAVEL area)."""
	x = {k: v.clone() for k, v in x.items()}
	for name in N.BLIND:
		if name.startswith("k_"):
			x[name][row] = N.SPECS[name].fill
	x["g_nk_present"][row] = True
	x["k_hero"][row, 0] = hero + 1
	if cell is not None:
		x["move_mask"][row, hero].view(-1)[cell] = True
		x["k_move"][row, 0] = hero * W * W + cell + 1
	if area is not None:
		x["travel_mask"][row, hero].view(-1)[area] = True
		x["k_travel"][row, 0] = hero * N.AREAS + area + 1
	x["k_score"][row, 0, 0] = score
	x["h_present"][row, hero] = True
	x["h_owner"][row, hero] = c.OWNER_SELF
	return x


def test_the_targets_of_one_candidate():
	_, x = model_and_inputs(rows=2)
	x = one_candidate(x, hero=1, cell=600)
	t = {k: v[:1] for k, v in N.targets(x).items()}
	assert bool(t["move_has"][0, 1, 600]) and int(t["move_has"].sum()) == 1
	forms = t["move_forms"][0, 1, 600]
	assert forms[0] == pytest.approx(1.0) and forms[1] == pytest.approx(1.0) and forms[4] == pytest.approx(1.0)  # (the best: preference 0 -> 1; its rank 1st)
	assert t["plan"][0, 1, 1 * W * W + 600] == pytest.approx(1.0) and t["plan_best"][0, 1] == 1 * W * W + 600
	others = t["plan_heroes"][0].clone()
	others[1] = False
	assert torch.all(t["plan"][0][others][:, -1] == 1.0)  # (our other heroes: no task)
	x = one_candidate(x, hero=1, area=40)
	t = N.targets(x)
	assert bool(t["travel_has"][0, 1, 40]) and t["plan"][0, 1, HO * W * W + 40] == pytest.approx(1.0)
	x["g_nk_present"][0] = False
	t = N.targets(x)
	assert not t["move_has"][0].any() and not t["travel_has"][0].any() and not t["plan_heroes"][0].any()  # (no advisor: nothing counts)


def test_it_learns_and_its_loss_reaches_the_head_alone():
	model, x = model_and_inputs(rows=32, seed=6)
	model.nkp_outputs = True
	for p in model.parameters():
		p.grad = None
	out = model(x)
	loss, _ = N.loss({k: v for k, v in out.items() if k.startswith("nkp_")}, {k[4:]: v for k, v in out.items() if k.startswith("nkt:")})
	loss.backward()
	touched = {name.split(".")[0] for name, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0}
	assert touched == {"nkp"}, touched
	optimizer = torch.optim.Adam(model.nkp.parameters(), lr=1e-2)
	_, first = N.loss({k: v.detach() for k, v in out.items() if k.startswith("nkp_")}, {k[4:]: v for k, v in out.items() if k.startswith("nkt:")})
	for _ in range(60):
		optimizer.zero_grad()
		out = model(x)
		loss, parts = N.loss({k: v for k, v in out.items() if k.startswith("nkp_")}, {k[4:]: v for k, v in out.items() if k.startswith("nkt:")})
		loss.backward()
		optimizer.step()
	assert float(parts["a"]) < 0.5 * float(first["a"]) and float(parts["b"]) < float(first["b"]) and float(parts["c"]) < 0.95 * float(first["c"]), (first, parts)
	model.nkp_outputs = False


def update(model, decisions, x, config):
	torch.manual_seed(7)
	np.random.seed(7)
	m = copy.deepcopy(model)
	optimizer = make_optimizer(m, config)
	n = len(decisions)
	rng = np.random.default_rng(0)
	targets = (0.5 + 0.05 * rng.standard_normal((n, c.N_VALUE_OFFSETS))).astype(np.float32)
	growth = (rng.standard_normal((n, c.N_VALUE_OFFSETS)) * 0.3).astype(np.float32)
	clips = {"typical": 10.0, **{name: 1e9 for name in GRAD_GROUPS}}
	metrics, _ = ppo_update(m, optimizer, decisions, targets, None, config, decode=lambda rows: {k: v[rows] for k, v in x.items()}, advantages=np.zeros(n, np.float32), growth=growth, growth_real=growth, labels=targets, clips=clips)
	return m, metrics


def test_weight_0_computes_nothing_and_on_it_moves_the_head_alone():
	torch.manual_seed(5)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	x = rich_inputs(16, seed=6)
	decisions = decisions_for(model.eval(), x)
	config = CampaignConfig(batch_size=8, passes=1, model_architecture="v2", credit="schedule", share_target="change")
	off, metrics_off = update(model, decisions, x, config)
	assert "nkp_auc" not in metrics_off and not off.nkp_outputs
	for (name, a), (_, b) in zip(model.named_parameters(), off.named_parameters()):
		if name.startswith("nkp."):
			assert torch.equal(a, b), name  # (no gradient: never stepped)
	on, metrics_on = update(model, decisions, x, replace(config, nk_predict_weight=1.0))
	assert not on.nkp_outputs  # (off again after the update)
	for (name, a), (_, b) in zip(off.named_parameters(), on.named_parameters()):
		if not name.startswith("nkp."):
			assert torch.equal(a, b), name
	assert any(not torch.equal(a, b) for (name, a), (_, b) in zip(model.named_parameters(), on.named_parameters()) if name.startswith("nkp."))
	assert metrics_on["nkp_loss"] > 0 and all(name in metrics_on for name in ("nkp_auc", "nkp_ce", "nkp_ce_uniform", "nkp_top1_best", "nkp_ev_rank"))


def test_the_auc():
	assert N.auc(torch.tensor([0.1, 0.4, 0.35, 0.8], dtype=torch.float64), torch.tensor([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.75)
	assert np.isnan(N.auc(torch.tensor([0.1, 0.2], dtype=torch.float64), torch.tensor([1.0, 1.0])))


def test_the_learner_logs_its_fit(tmp_path, capsys):
	"""CampaignLearner.step at deploy 62's line with the head on: its columns in progress.csv and its part of the update's line."""
	import csv

	from campaign_model.tests import test_credit_segments as T

	saved = T.CampaignConfig
	try:
		T.CampaignConfig = lambda **kwargs: saved(**{**kwargs, "nk_predict_weight": 1.0, "credit_segments": True})
		T.test_the_learner_step_end_to_end(tmp_path, capsys)
	finally:
		T.CampaignConfig = saved
	row = list(csv.DictReader(open(tmp_path / "progress.csv")))[-1]
	assert float(row["nkp_loss"]) > 0 and row["nkp_ce"] not in ("", "nan") and row["nkp_ce_uniform"] not in ("", "nan")
