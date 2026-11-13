"""config.value_shared (e5 for the user, 2026-11-13): the value terms' gradient into the shared layers at a fixed scale
through the model's value_valve, ramped by the version alone: off (1.0) is 1 at every version, the model and the update
as before; the schedule 0.8, 0.64, ... and value_shared from the 7th update; the shared layers' gradient from the value
term scales by exactly s while the value group's own is unchanged; a restart mid-ramp gives the same s; the learner's
valve_now hands it on; value_shared with the value valve's controller is a config error. config.head_scale_freeze holds
the heads' weights."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import campaign_model.train as train_module
from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.learner import CampaignLearner, value_shared_now
from campaign_model.model_v2 import CampaignModelV2
from campaign_model.tests.test_model_v2 import decisions_for, rich_inputs
from campaign_model.train import SHARED_PREFIXES, VALUE_MODULES, make_optimizer, ppo_update


def setup(n=8, seed=6):
	torch.manual_seed(5)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1, share_target="change")
	with torch.no_grad():  # (a trained model's read-outs: a new model's start at 0 and pass no gradient back)
		for head in (model.value_heads, model.growth_heads, model.end_head):
			for p in head.parameters():
				p.add_(0.2 * torch.randn_like(p))
	x = rich_inputs(n, seed=seed)
	return model, x, decisions_for(model.eval(), x)


def config_of(**kw):
	return CampaignConfig(batch_size=8, passes=1, model_architecture="v2", share_target="change", credit="schedule", **kw)


def value_term(model, x, decisions, config):
	"""micro_loss's value term on the update's first micro-batch (ppo_update builds the batch), then the update stops."""
	caught, real = {}, train_module.micro_loss

	class Done(Exception):
		pass

	def micro(out, batch, config_, *args, **kwargs):
		if kwargs.get("split"):
			return real(out, batch, config_, *args, **kwargs)
		caught["value"] = real(out, batch, config_, *args, **{**kwargs, "split": True})[4]["value"]
		raise Done

	n = len(decisions)
	targets = (0.5 + 0.05 * np.random.default_rng(0).standard_normal((n, c.N_VALUE_OFFSETS))).astype(np.float32)
	train_module.micro_loss = micro
	try:
		torch.manual_seed(7)
		ppo_update(model, make_optimizer(model, config), decisions, targets, None, config, decode=lambda rows: {k: v[rows] for k, v in x.items()}, advantages=np.random.default_rng(1).standard_normal(n).astype(np.float32))
	except Done:
		pass
	finally:
		train_module.micro_loss = real
	return caught["value"]


def grads(model, term):
	params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
	got = torch.autograd.grad(term, [p for _, p in params], retain_graph=True, allow_unused=True)
	return {name: g for (name, _), g in zip(params, got) if g is not None}


def test_off_is_1_at_every_version():
	config = config_of()
	assert all(value_shared_now(config, v) == 1.0 for v in (0, 1, 3300, 10**6))
	assert value_shared_now(config_of(value_shared=0.25, value_shared_from=3400), 3399) == 1.0  # (before the start)


def test_the_schedule_reaches_value_shared_at_the_7th_update():
	config = config_of(value_shared=0.25, value_shared_from=3340)
	got = [value_shared_now(config, 3340 + k) for k in range(9)]
	assert got[:6] == pytest.approx([1.25**-1, 1.25**-2, 1.25**-3, 1.25**-4, 1.25**-5, 1.25**-6])
	assert got[0] == pytest.approx(0.8) and got[5] > 0.25 and got[6] == 0.25 and got[8] == 0.25
	# (a restart mid-ramp: the version's alone, whatever came before)
	assert value_shared_now(config, 3343) == value_shared_now(config_of(value_shared=0.25, value_shared_from=3340), 3343)


def test_the_shared_layers_gradient_scales_by_s_and_the_value_groups_does_not():
	model, x, decisions = setup()
	config = config_of()
	full = grads(model, value_term(model, x, decisions, config))
	model.value_valve.fill_(0.25)
	scaled = grads(model, value_term(model, x, decisions, config))
	shared = [name for name in full if name.startswith(SHARED_PREFIXES) and float(full[name].abs().max()) > 0]
	own = [name for name in full if name.startswith(VALUE_MODULES) and float(full[name].abs().max()) > 0]
	assert shared and own
	for name in shared:
		assert torch.allclose(scaled[name], 0.25 * full[name], rtol=1e-4, atol=1e-9), name
	for name in own:
		assert torch.allclose(scaled[name], full[name], rtol=1e-5, atol=1e-9), name


def test_valve_now_hands_it_on():
	config = config_of(value_shared=0.25, value_shared_from=3340)
	learner = SimpleNamespace(config=config, version=3342, valve=None)
	assert CampaignLearner.valve_now(learner) == (pytest.approx(1.25**-3), 1.0)
	assert CampaignLearner.valve_now(SimpleNamespace(config=config_of(), version=3342, valve=None)) == (1.0, 1.0)


def test_value_shared_with_the_valve_controller_is_a_config_error(tmp_path):
	config = CampaignConfig(value_shared=0.25, value_valve=True, learner="thread", device="cpu", model_architecture="v2")
	with pytest.raises(ValueError, match="value_shared and config.value_valve"):
		CampaignLearner(config, tmp_path, torch.device("cpu"), battle=SimpleNamespace(config=SimpleNamespace(remote_port=0)), server=SimpleNamespace(set_model=lambda *a, **k: None, socket_path="/nonexistent", close=lambda: None, stop=lambda: None))


def test_value_shared_with_the_loss_shares_not_frozen_is_a_config_error(tmp_path):
	# (e5, 2026-11-13: their balancing would raise the value's weight against the scale; a new run is at version 1)
	server = SimpleNamespace(set_model=lambda *a, **k: None, socket_path="/nonexistent", close=lambda: None, stop=lambda: None)
	battle = SimpleNamespace(config=SimpleNamespace(remote_port=0))
	for freeze in (0, 1420):
		config = CampaignConfig(value_shared=0.25, loss_shares=True, loss_shares_freeze=freeze, learner="thread", device="cpu", model_architecture="v2")
		with pytest.raises(ValueError, match="loss shares not frozen"):
			CampaignLearner(config, tmp_path / str(freeze), torch.device("cpu"), battle=battle, server=server)


def test_head_scale_freeze_holds_the_weights():
	model, x, decisions = setup()
	n = len(decisions)
	targets = (0.5 + 0.05 * np.random.default_rng(0).standard_normal((n, c.N_VALUE_OFFSETS))).astype(np.float32)
	frozen = {"share": [3.0, 2.0, 1.0]}
	config = config_of(head_scale=True)
	metrics, _ = ppo_update(model, make_optimizer(model, config), decisions, targets, None, config, decode=lambda rows: {k: v[rows] for k, v in x.items()}, advantages=np.zeros(n, np.float32), head_scale={"var": None, "ramp": 1.0, "frozen": frozen})
	assert metrics["head_weights"]["share"] == pytest.approx(frozen["share"])
