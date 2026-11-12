"""nk_predict stage 2 (2026-11-12, e5's design): the late terms read by nk_read. "live" as before (checked against the
code before the reorder apart: 35 of 35 outputs equal); "blind" the decisions encoded without the advisor; "predicted"
differs from "blind" only through the two late terms; its table is the head's in the planner's slice; the policy's
gradient never reaches the head; the export works in "predicted"."""
import torch

from campaign_model import constants as c
from campaign_model import nk_predict as N
from campaign_model import views as V
from campaign_model.model_v2 import CampaignModelV2, nk_table
from campaign_model.tests.test_model_v2 import rich_inputs
from campaign_model.train import chosen_logp, choices
from campaign_model.tests.test_model_v2 import decisions_for

HO, W = c.OWN_HEROES, c.WINDOW


def model_and_inputs(rows=8, seed=4):
	torch.manual_seed(3)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	with torch.no_grad():  # (every zero-initialised layer given weights: the late terms and the head read something)
		for p in model.parameters():
			if float(p.abs().sum()) == 0:
				p.normal_(0, 0.05)
	return model, {k: torch.from_numpy(v) for k, v in rich_inputs(rows, seed).items()}


def run(model, x, mode):
	model.nk_read = mode
	captured = {}
	original = model.decide

	def decide(state, known):
		captured.update(move_mix=state.move_mix, travel_view=state.travel_view, move_pairs=state.move_pairs, travel_pairs=state.travel_pairs, heroes=state.heroes, glob=state.glob)
		return original(state, known)

	model.decide = decide
	try:
		with torch.no_grad():
			out = model(x)
	finally:
		model.decide = original
		model.nk_read = "live"
	return out, captured


def test_blind_is_the_decisions_without_the_advisor():
	model, x = model_and_inputs()
	blind, _ = run(model, x, "blind")
	without, _ = run(model, N.blind_inputs(x, memory=True), "live")
	for name in ("move_logp", "travel_logp", "type_logp", "value_logit"):
		assert torch.equal(blind[name].nan_to_num(), without[name].nan_to_num()), name
	live, _ = run(model, x, "live")
	assert not torch.equal(live["move_logp"].nan_to_num(), blind["move_logp"].nan_to_num())  # (the planner's rows mattered)


def test_predicted_differs_from_blind_only_through_the_late_terms():
	model, x = model_and_inputs()
	_, blind = run(model, x, "blind")
	_, predicted = run(model, x, "predicted")
	for name in ("move_pairs", "travel_pairs", "heroes", "glob"):
		assert torch.equal(blind[name], predicted[name]), name
	assert not torch.equal(blind["move_mix"], predicted["move_mix"]) and not torch.equal(blind["travel_view"], predicted["travel_view"])


def test_the_predicted_table_is_the_heads_in_the_planners_slice():
	model, x = model_and_inputs()
	model.nk_read = "predicted"
	seen = {}
	original = model.predicted_advice

	def spy(state, known):
		a = original(state, known)
		seen.update(a=a, blind=known.advice, state=state, known=known)
		return a

	model.predicted_advice = spy
	try:
		with torch.no_grad():
			model(x)
	finally:
		model.predicted_advice = original
		model.nk_read = "live"
	a, blind = seen["a"], seen["blind"]
	start = blind.travel_slice
	head = model.nkp.entries(seen["state"], seen["known"])
	assert torch.allclose(a.travel[..., start : start + V.F], nk_table(head[1]).to(a.travel.dtype))
	outside = torch.ones(a.travel.shape[-1], dtype=torch.bool)
	outside[start : start + V.F] = False
	assert torch.equal(a.travel[..., outside], blind.travel[..., outside])  # (the rest the blind inputs': no destination, no choice)
	table = nk_table(torch.randn(3, 7))
	live = {name: i for i, name in enumerate(V.CHANNELS)}
	kept = {live[n] for n in ("has", "preference_all", "preference_hero", "soon", "later")}
	assert all(float(table[:, i].abs().max()) == 0.0 for i in range(V.F) if i not in kept)
	assert float(table[:, live["preference_all"]].max()) <= 0.0 and float(table[:, live["has"]].min()) >= 0.0


def test_the_policys_gradient_never_reaches_the_head():
	model, x = model_and_inputs()
	model.train()
	model.nk_read = "predicted"
	try:
		decisions = decisions_for(model.eval(), {k: v.numpy() for k, v in x.items()})
		ch = {k: torch.from_numpy(v) for k, v in choices(decisions).items() if k in ("kind", "action", "index", "cell")}
		for p in model.parameters():
			p.grad = None
		logp, _ = chosen_logp(model(x), ch)
		logp.sum().backward()
	finally:
		model.nk_read = "live"
	assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0 for p in model.nkp.parameters())
	assert float(model.view_move.weight.grad.abs().sum()) > 0 if hasattr(model.view_move, "weight") and model.view_move.weight.grad is not None else True


def test_the_export_in_predicted(tmp_path):
	from campaign_model.export import export, verify

	model, _ = model_and_inputs()
	model.nk_read = "predicted"
	try:
		path = export(model, tmp_path / "predicted.onnx")
		assert verify(model, path, seeds=(1,), batch_sizes=(1, 3)) < 1e-4
	finally:
		model.nk_read = "live"


def test_training_in_predicted_learns_the_head_from_the_main_pass():
	import copy
	from dataclasses import replace

	import numpy as np

	from campaign_model.config import CampaignConfig
	from campaign_model.tests.test_nk_predict import update

	torch.manual_seed(5)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	x = rich_inputs(16, seed=6)
	decisions = decisions_for(model.eval(), x)
	model.nk_read = "predicted"
	try:
		config = CampaignConfig(batch_size=8, passes=1, model_architecture="v2", credit="schedule", share_target="change", nk_predict_weight=0.1, nk_read="predicted")
		trained, metrics = update(model, decisions, x, config)
	finally:
		model.nk_read = "live"
	assert metrics["nkp_loss"] > 0 and "nkp_steps" not in metrics and np.isfinite(metrics["nkp_ce"])  # (the main pass's: no separate step)
	assert any(not torch.equal(a, b) for (name, a), (_, b) in zip(model.named_parameters(), trained.named_parameters()) if name.startswith("nkp."))
	assert not trained.nkp_outputs
