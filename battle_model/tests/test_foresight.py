"""The foresight heads (model.Foresight) and their migration: an older checkpoint plays exactly as before, the labels
say what happened before our next decision, and the foresight's loss reaches the trunk without moving the policy."""
import copy

import numpy as np
import torch

from battle_model.constants import N_HEXES
from battle_model.data import foresight_labels
from battle_model.losses import FORESIGHT_PARTS, PPOConfig, foresight_loss
from battle_model.model import CONTEXT_HIDDEN, BattleModel
from battle_model.schema import empty_inputs, random_inputs
from battle_model.train import trunk_gradients


def tensors(x):
	return {k: torch.from_numpy(v) for k, v in x.items()}


def old_checkpoint(model):
	"""model's weights as a checkpoint from before the foresight, the actor queries and the wider context."""
	state = {k: v.clone() for k, v in model.state_dict().items() if not k.startswith(BattleModel.ADDED)}
	dim = model.context.norm.weight.shape[0]
	new, old = CONTEXT_HIDDEN * dim, dim // 2
	up, bias = state["context.mlp.up.weight"], state["context.mlp.up.bias"]
	state["context.mlp.up.weight"] = torch.cat([up[:old], up[new : new + old]])
	state["context.mlp.up.bias"] = torch.cat([bias[:old], bias[new : new + old]])
	state["context.mlp.down.weight"] = state["context.mlp.down.weight"][:, :old]
	return state, old


def test_an_older_checkpoint_plays_exactly_as_before():
	torch.manual_seed(0)
	model = BattleModel().eval()
	state, old = old_checkpoint(model)
	reference = copy.deepcopy(model)  # what the old checkpoint computed: the context's extra units out, the rest at 0
	with torch.no_grad():
		reference.context.mlp.down.weight[:, old:] = 0
	loaded = BattleModel().eval()
	loaded.load_state_dict(state)
	x = tensors(random_inputs(batch=3, seed=4))
	with torch.no_grad():
		a, b = reference(x), loaded(x)
	for name in a:
		if not name.startswith("foresight"):
			assert torch.allclose(a[name], b[name], atol=1e-5, equal_nan=True), name


def two_decisions():
	"""Our unit (slot 0, 50 creatures of 10 HP) and an enemy's (slot 1, 20 of 30 HP): the enemy then dealt 120 HP (a
	quarter of ours: 12 of ours died and its stack lost a fifth) and moved to hex 40; slot 2, an enemy that dies."""
	x = empty_inputs(2)
	for d in (0, 1):
		x["g_our_side"][d] = 0
		x["u_present"][d, :3] = 1
		x["u_creature"][d, :3] = [5, 7, 9]
		x["u_owner_side"][d, :3] = [0, 1, 1]
		x["u_max_hp"][d, :3] = [10, 30, 5]
	x["u_alive"][0, :3] = [1, 1, 1]
	x["u_count"][0, :3], x["u_first_hp"][0, :3], x["u_hex"][0, :3] = [50, 20, 4], [10, 30, 5], [10, 20, 30]
	x["u_alive"][1, :3] = [1, 1, 0]
	x["u_count"][1, :3], x["u_first_hp"][1, :3], x["u_hex"][1, :3] = [38, 16, 0], [10, 30, 0], [10, 40, 30]
	x["u_dealt"][1, 1] = 120
	x["u_rear_hex"][:, :3] = -1
	return [({k: v[d] for k, v in x.items()}, {}, {}) for d in (0, 1)]


def test_the_labels_say_what_happened_before_our_next_decision():
	labels = foresight_labels(two_decisions())
	first = labels[0]
	assert labels[1] is None  # the battle's last decision: nothing after it
	assert first["unit_mask"][:3].tolist() == [1, 1, 1] and first["unit_mask"][3:].sum() == 0
	assert first["dies"][:3].tolist() == [0, 0, 1]
	assert np.allclose(first["lost"][:3], [12 * 10 / 500, 4 * 30 / 600, 1.0])
	assert np.isclose(first["dealt"][1], 120 / 500)  # of our 500 HP
	assert first["hex_enemy"][40] == 1 and first["hex_enemy"].sum() == 1  # the dead one no longer stands anywhere
	assert first["hex_enemy"].shape == (N_HEXES,)


def foresight_batch(batch, seed=0):
	rng = np.random.default_rng(seed)
	units = 28
	return {
		"dies": torch.from_numpy((rng.random((batch, units)) < 0.1).astype(np.float32)),
		"lost": torch.from_numpy(rng.random((batch, units)).astype(np.float32)),
		"dealt": torch.from_numpy((rng.random((batch, units)) * 0.2).astype(np.float32)),
		"unit_mask": torch.from_numpy((rng.random((batch, units)) < 0.5).astype(np.float32)),
		"hex_enemy": torch.from_numpy((rng.random((batch, N_HEXES)) < 0.05).astype(np.float32)),
		"valid": torch.ones(batch),
	}


def test_the_foresight_loss_reaches_the_trunk_and_starts_without_moving_the_policy():
	torch.manual_seed(1)
	model = BattleModel()
	x = tensors(random_inputs(batch=4, seed=2))
	out = model(x)
	loss, parts = foresight_loss(out, foresight_batch(4), PPOConfig(foresight_weight=1.0))
	assert set(parts) == set(FORESIGHT_PARTS) and all(float(v) > 0 for v in parts.values())
	model.zero_grad()
	loss.backward()
	trunk = [p.grad for p in model.trunk.parameters() if p.requires_grad]
	assert all(g is not None and torch.isfinite(g).all() for g in trunk)
	assert sum(float(g.abs().sum()) for g in trunk) > 0  # it shapes the trunk
	join = model.foresight.join_unit.weight.grad
	assert join is None or float(join.abs().sum()) == 0  # (the foresight's own loss does not go through the joins)
	# The joins start at 0: the policy is the same with the foresight's hidden states scrambled
	with torch.no_grad():
		before = model(x)["logp_type"]
		for layer in (model.foresight.unit.up, model.foresight.hex.up):
			layer.weight.normal_()
		after = model(x)["logp_type"]
	assert torch.allclose(before, after, equal_nan=True)


def test_the_gradient_check_measures_both_losses_on_the_trunk():
	torch.manual_seed(2)
	model = BattleModel()
	with torch.no_grad():  # (untrained, the heads' readouts are 0: no gradient reaches the trunk from the PPO losses)
		for module in (model.values, model.actions.queries, model.actions.type_head):
			for parameter in module.parameters():
				parameter.normal_(0, 0.1)
	out = model(tensors(random_inputs(batch=4, seed=3)))
	main = out["value"].sum() + out["logp_type"].clamp(min=-30).sum()
	foresight, _ = foresight_loss(out, foresight_batch(4, seed=1), PPOConfig(foresight_weight=1.0))
	check = trunk_gradients(model, main, foresight)
	assert check["grad_trunk_main"] > 0 and check["grad_trunk_foresight"] > 0
	assert -1.0 <= check["grad_trunk_cosine"] <= 1.0
	assert check["grad_dead_share"] < 0.5
