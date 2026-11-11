"""The advantage head (model_v2.AdvantageHead, config.adv_weight; stage 1, 2026-11-11: it only learns, the policy
reads none of it): inert at weight 0, an older checkpoint loads it at 0, its loss reaches its own weights alone, and
it learns a planted advantage (the choice being Nullkiller2's own) with its held-out reading (adv_ev) rising."""
import copy

import numpy as np
import torch
from torch import nn

from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.model_v2 import CampaignModelV2
from campaign_model.records import Decision
from campaign_model.tests.test_model_v2 import decisions_for, rich_inputs
from campaign_model.train import GRAD_GROUPS, STEP_PARTS, advantage_of, chosen_logp, make_optimizer, parameter_groups, ppo_update

CLIPS = {"typical": 10.0, **{name: 1e9 for name in GRAD_GROUPS}}  # (no group clip binds: the groups' steps apart)


def tensors(x):
	return {k: torch.from_numpy(v) for k, v in x.items()}


def update(model, decisions, x, config, advantages, passes=1):
	torch.manual_seed(7)
	optimizer = make_optimizer(model, config)
	n = len(decisions)
	targets = (0.5 + 0.05 * np.random.default_rng(0).standard_normal((n, c.N_VALUE_OFFSETS))).astype(np.float32)
	metrics, _ = ppo_update(model, optimizer, decisions, targets, None, config, decode=lambda rows: {k: v[rows] for k, v in x.items()}, advantages=advantages, clips=CLIPS)
	return metrics


def setup(n=16, seed=6):
	torch.manual_seed(5)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	x = rich_inputs(n, seed=seed)
	return model, x, decisions_for(model.eval(), x)


def test_it_is_off_the_contract_and_in_the_value_group():
	model, x, _ = setup(4)
	with torch.no_grad():
		out = model.eval()(tensors(x))
	assert not any(name.startswith("adv_") for name in out)  # (a server's model, an export: never set)
	groups = {name: group for group, members in parameter_groups(model).items() for name, _ in members}
	assert {groups[name] for name, _ in model.advantage.named_parameters(prefix="advantage")} == {"value"}
	model.adv_outputs = True
	with torch.no_grad():
		out = model(tensors(x))
	b = x["g_day"].shape[0]
	assert out["adv_type"].shape == (b, len(c.ActionType)) and out["adv_named"].shape == (b, len(c.ActionType)) and out["adv_label"].shape == (b, 2)
	assert out["adv_move"].shape == out["move_logp"].shape and out["adv_travel"].shape == out["travel_logp"].shape
	assert all(float(out[name].abs().max()) == 0.0 for name in ("adv_type", "adv_move", "adv_travel", "adv_named"))  # (0 to start)


def test_an_older_checkpoint_loads_it_at_zero():
	model, x, _ = setup(4)
	state = {k: v for k, v in model.state_dict().items() if not k.startswith("advantage.")}
	old = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	old.load_state_dict(state)
	assert all(float(p.detach().abs().max()) == 0.0 for p in old.advantage.parameters())
	assert all(torch.equal(a, b) for (_, a), (_, b) in zip(model.state_dict().items(), old.state_dict().items()))


def test_at_weight_0_the_update_is_the_one_without_the_head():
	"""Every weight bit-identical to an update of the same model with no head at all (its parameters out of the value
	group, nothing computed), the head unmoved, its loss 0 and no reading of it."""
	model, x, decisions = setup()
	config = CampaignConfig(batch_size=8, passes=1, model_architecture="v2", credit="schedule", share_target="change")
	advantages = np.random.default_rng(3).standard_normal(len(decisions)).astype(np.float32)
	with_head, without = copy.deepcopy(model), copy.deepcopy(model)
	without.advantage = nn.Module()
	metrics = update(with_head, decisions, x, config, advantages)
	update(without, decisions, x, config, advantages)
	plain = dict(without.named_parameters())
	for name, p in with_head.named_parameters():
		if name.startswith("advantage."):
			assert float(p.abs().max()) == 0.0, name
		else:
			assert torch.equal(p, plain[name]), name
	assert not with_head.adv_outputs and metrics["adv_loss"] == 0.0 and "adv_ev" not in metrics


def test_its_loss_reaches_its_own_weights_alone():
	"""The head's term alone backpropagates into the head's weights and nothing else; and an update with it moves every
	other weight exactly as one without it (no group clip binding)."""
	model, x, decisions = setup()
	model.adv_outputs = True
	out = model.train()(tensors(x))
	action = torch.tensor([d.action_type for d in decisions])
	index = torch.tensor([d.index for d in decisions])
	predicted, _ = advantage_of(out, action, index)
	# (a nonzero gradient needs nonzero weights downstream of the inputs: the head's weights away from 0 first)
	with torch.no_grad():
		for p in model.advantage.parameters():
			p.normal_(0.0, 0.1)
	out = model(tensors(x))
	predicted, _ = advantage_of(out, action, index)
	(predicted - 1.0).pow(2).sum().backward()
	reached = {name for name, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().max()) > 0}
	assert reached and all(name.startswith("advantage.") for name in reached), sorted(reached)[:5]
	assert {"advantage.types.weight", "advantage.move.weight", "advantage.travel.weight"} <= reached
	model.adv_outputs = False
	model.zero_grad(set_to_none=True)
	model, x, decisions = setup()
	advantages = np.random.default_rng(3).standard_normal(len(decisions)).astype(np.float32)
	off, on = copy.deepcopy(model), copy.deepcopy(model)
	update(off, decisions, x, CampaignConfig(batch_size=8, passes=1, model_architecture="v2", credit="schedule", share_target="change"), advantages)
	metrics = update(on, decisions, x, CampaignConfig(batch_size=8, passes=1, model_architecture="v2", credit="schedule", share_target="change", adv_weight=1.0), advantages)
	plain = dict(off.named_parameters())
	moved = [name for name, p in on.named_parameters() if name.startswith("advantage.") and float(p.abs().max()) > 0]
	assert moved
	for name, p in on.named_parameters():
		if not name.startswith("advantage."):
			assert torch.equal(p, plain[name]), name
	assert metrics["adv_loss"] > 0 and np.isfinite(metrics["adv_ev"]) and np.isfinite(metrics["adv_raw_sd"])
	assert {"adv_ev_followed", "adv_ev_deviating", "adv_followed"} <= set(metrics)
	assert "adv_loss" in STEP_PARTS


def planted(n=24, seed=11):
	"""Decisions whose choice is Nullkiller2's own at about half the rows it has one, our sampled one elsewhere, each
	with its log-probability now; the advantage +1 where the choice is its own, -1 elsewhere."""
	torch.manual_seed(5)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = rich_inputs(n, seed=seed)
	decisions = decisions_for(model, x)
	rng = np.random.default_rng(1)
	for b, d in enumerate(decisions):
		if x["g_nk_action"][b] > 0 and d.kind == c.DecisionKind.TURN and rng.random() < 0.5:
			action, index, cell = int(x["g_nk_action"][b]) - 1, int(x["g_nk_index"][b]) - 1, int(x["g_nk_cell"][b]) - 1  # (schema 19: +1, 0 none)
			decisions[b] = Decision(**{**d.__dict__, "action_type": action, "index": -1 if action == c.ActionType.END else index, "cell": cell})
	with torch.no_grad():
		out = model(tensors(x))
		choice = {"kind": torch.tensor([d.kind for d in decisions]), "action": torch.tensor([d.action_type for d in decisions]), "index": torch.tensor([d.index for d in decisions]), "cell": torch.tensor([d.cell for d in decisions])}
		logp, _ = chosen_logp(out, choice)
		model.adv_outputs = True
		_, named = advantage_of(model(tensors(x)), choice["action"], choice["index"])
		model.adv_outputs = False
	decisions = [Decision(**{**d.__dict__, "logp": float(lp)}) for d, lp in zip(decisions, logp.tolist())]
	return model, x, decisions, np.where(named.numpy(), 1.0, -1.0).astype(np.float32), named.numpy()


def test_it_learns_a_planted_advantage():
	"""The advantage +1 where the choice was Nullkiller2's own, -1 elsewhere: about a hundred steps of the head (the
	policy at a negligible rate) and its held-out reading, the explained variance before each update's steps, rises
	from 0 to most of it."""
	model, x, decisions, advantages, named = planted()
	assert 4 <= named.sum() <= len(named) - 4, named.sum()
	config = CampaignConfig(batch_size=8, passes=12, model_architecture="v2", credit="schedule", share_target="change", adv_weight=1.0, learning_rate=1e-7, value_lr_factor=3e5, target_kl=0.0)
	readings = []
	for _ in range(4):  # (3 minibatches x 12 passes: 144 steps, the readings before steps 0, 36, 72 and 108)
		metrics = update(model, decisions, x, config, advantages)
		readings.append(metrics["adv_ev"])
	assert abs(readings[0]) < 1e-6 and readings[-1] > 0.8, readings
	# (each group's own target is constant here, +1 or -1: its explained variance within the group is not a number)
	assert 0 < metrics["adv_followed"] < 1
	assert float(model.advantage.named.detach().abs().max()) > 0  # (the named weights carry it)
