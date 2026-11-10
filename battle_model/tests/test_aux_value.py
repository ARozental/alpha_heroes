"""aux_value.py: the value's own phase (Phasic Policy Gradient's auxiliary phase, 2026-11-10)."""
import copy
from types import SimpleNamespace

import numpy as np
import torch

from battle_model.aux_value import AuxValuePhase, ValueReplay, aux_allowed, take_rows
from battle_model.data import SPARSE_INPUTS, Sparse
from battle_model.distributions import enumerate_actions, kl
from battle_model.losses import value_loss
from battle_model.model import BattleModel
from battle_model.schema import random_inputs
from battle_model.tests.test_model import _batched, small_masks


def run(model, x):
	with torch.no_grad():
		return model({k: torch.from_numpy(v) for k, v in x.items()})


def perturbed(model, scale=0.05, seed=1):
	other = copy.deepcopy(model)
	g = torch.Generator().manual_seed(seed)
	with torch.no_grad():
		for p in other.parameters():
			p.add_(scale * torch.randn(p.shape, generator=g))
	return other


def test_the_kl_is_the_full_distributions_exactly():
	"""distributions.kl against KL(p || q) summed over every legal full action (enumerate_actions), on inputs thin enough
	to enumerate; 0 between a model and itself."""
	torch.manual_seed(0)
	a = BattleModel().eval()
	b = perturbed(a)
	for seed in range(4):
		x = small_masks(seed)
		out_a, out_b = run(a, x), run(b, x)
		actions = enumerate_actions(x)
		lp_a, lp_b = _batched(out_a, actions), _batched(out_b, actions)
		exact = float((lp_a.exp() * (lp_a - lp_b)).sum())
		assert np.isclose(float(kl(out_a, out_b)[0]), exact, atol=1e-4), (seed, float(kl(out_a, out_b)[0]), exact)
		assert abs(float(kl(out_a, out_a)[0])) < 1e-6


def test_take_rows_keeps_dense_and_sparse_rows_alike():
	rng = np.random.default_rng(2)
	big = rng.normal(size=(6, 9, 4)).astype(np.float32) * (rng.random((6, 9, 4)) < 0.2)
	mask = rng.random((6, 5, 7)) < 0.3
	data = {"inputs": {"big": Sparse(big), "mask": Sparse(mask), "small": rng.normal(size=(6, 3)).astype(np.float32)}}
	rows = np.array([1, 3, 4])
	got = take_rows(data, rows)["inputs"]
	assert np.array_equal(got["big"].dense(), big[rows]) and np.array_equal(got["mask"].dense(), mask[rows])
	assert np.array_equal(got["small"], data["inputs"]["small"][rows])


def battle_item(seed, n, version, heads):
	"""A replay item as ActorPool.take hands it (supply.prepare's data), from random inputs: its big tables Sparse."""
	x = random_inputs(batch=n, seed=seed)
	inputs = {name: Sparse(v) if name in SPARSE_INPUTS else v for name, v in x.items()}
	labels = np.random.default_rng(seed).uniform(-1, 1, size=(n, heads)).astype(np.float32)
	return {"data": {"inputs": inputs, "value_labels": labels}, "decisions": n, "version": version}


def test_the_replay_keeps_a_spread_of_each_battle_within_its_limits():
	heads = BattleModel().eval()({k: torch.from_numpy(v) for k, v in random_inputs(batch=1).items()})["value"].shape[1]
	replay = ValueReplay(max_decisions=12, max_age=2, per_battle=3, seed=0)
	replay.add([battle_item(1, 7, 10, heads), battle_item(2, 2, 10, heads)], version=10)
	assert replay.decisions() == 5 and replay.bytes > 0  # (3 of 7, 2 of 2)
	replay.add([battle_item(3, 9, 11, heads), battle_item(4, 9, 11, heads)], version=11)
	assert replay.decisions() <= 12  # (the oldest out first)
	replay.add([battle_item(5, 9, 14, heads)], version=14)
	assert all(e["version"] >= 12 for e in replay.entries)  # (more than 2 versions old: out)
	batch = replay.batch(2, torch.device("cpu"))
	assert batch["value_labels"].shape == (2, heads) and set(batch["inputs"]) == set(random_inputs(batch=1))
	assert replay.batch(5, torch.device("cpu"), exclude_version=14) is None


def test_the_phase_lowers_the_value_loss_and_its_kl_holds_the_policy():
	"""On a replay of random battles: the value loss on them falls; with the KL's weight the policy moves less from where
	the phase found it than without (both phases from the same weights, the same draws)."""
	torch.manual_seed(0)
	# (a trained model's weights: a new one's value and policy heads start at 0, so the trunk moves neither the policy
	# nor gets a gradient from V until they have moved)
	base = perturbed(BattleModel().eval(), scale=0.05, seed=3)
	heads = run(base, random_inputs(batch=1))["value"].shape[1]
	x_check = random_inputs(batch=16, seed=99)
	check = {k: torch.from_numpy(v) for k, v in x_check.items()}
	moved = {}
	for weight in (0.0, 10.0):
		model = copy.deepcopy(base)
		replay = ValueReplay(max_decisions=10_000, max_age=8, per_battle=8, seed=0)
		replay.add([battle_item(s, 8, 5, heads) for s in range(12)], version=5)
		from battle_model.train import battle_parameter_groups

		groups = battle_parameter_groups(model)
		phase = AuxValuePhase(groups, {name: (3e-3 if name == "values" else 3e-4) for name in groups}, kl_max=1.0)  # (no net: the KL's own effect)
		every = replay.batch(10_000, torch.device("cpu"))
		with torch.no_grad():
			before = float(value_loss(model({k: v.long() if v.dtype == torch.int32 else v for k, v in every["inputs"].items()}), every["value_labels"])[0])
			snapshot = model(check)
		readings = phase.run(model, replay, torch.device("cpu"), steps=40, batch_size=32, kl_weight=weight, seconds=120)
		with torch.no_grad():
			after = float(value_loss(model({k: v.long() if v.dtype == torch.int32 else v for k, v in every["inputs"].items()}), every["value_labels"])[0])
			moved[weight] = float(kl(snapshot, model(check)).mean())
		assert readings["aux_steps"] == 40 and after < before, (weight, before, after)
		assert np.isfinite(readings["aux_grad_cosine"]) and readings["aux_trunk_move"] > 0
	# (measured alone 3.6e-3 against 2.4e-2, in the whole suite 5.1e-3 against 4.2e-2: torch's state differs; the loss
	# 0.796 -> 0.694 against -> 0.684)
	assert moved[10.0] < 0.3 * moved[0.0] and moved[10.0] < 1e-2, moved


def test_the_phase_never_runs_beside_a_campaign_update():
	"""aux_allowed: no campaign learner, or one on the battle trainer's own thread ("turns"); not a thread or process of
	its own (its update could be running on the device meanwhile)."""
	assert aux_allowed(None, SimpleNamespace(campaign=SimpleNamespace(learner="thread")))
	assert aux_allowed(object(), SimpleNamespace(campaign=SimpleNamespace(learner="turns")))
	assert not aux_allowed(object(), SimpleNamespace(campaign=SimpleNamespace(learner="thread")))
	assert not aux_allowed(object(), SimpleNamespace(campaign=SimpleNamespace(learner="process")))


def setup_phase(rates, kl_max=3e-4, seed=3):
	from battle_model.train import battle_parameter_groups

	torch.manual_seed(0)
	model = perturbed(BattleModel().eval(), scale=0.05, seed=seed)
	heads = run(model, random_inputs(batch=1))["value"].shape[1]
	replay = ValueReplay(max_decisions=10_000, max_age=8, per_battle=8, seed=0)
	replay.add([battle_item(s, 8, 5, heads) for s in range(10)] + [battle_item(50 + s, 8, 6, heads) for s in range(3)], version=6)
	groups = battle_parameter_groups(model)
	return model, replay, AuxValuePhase(groups, {name: rates for name in groups}, kl_max=kl_max)


def test_a_phase_that_moves_the_policy_past_kl_max_is_undone_bit_for_bit():
	"""e5's net: kl_weight 0 at a large rate moves the policy far; the phase is undone, every weight the start's exactly,
	its Adam cold again (aux_rejected)."""
	model, replay, phase = setup_phase(rates=1e-2)
	start = {name: p.detach().clone() for name, p in model.state_dict().items()}
	readings = phase.run(model, replay, torch.device("cpu"), steps=10, batch_size=32, kl_weight=0.0, seconds=120, held_out_version=6)
	assert readings["aux_rejected"] == 1 and readings["aux_kl_check"] > 3e-4, readings
	assert all(torch.equal(start[name], value) for name, value in model.state_dict().items())
	assert phase.steps_done == 0 and not phase.optimizer.state  # (cold again)


def test_an_error_in_the_phase_puts_the_weights_back_and_guarded_run_turns_it_off(capsys):
	"""e5's net: the replay failing on its second batch; run() puts the weights back and re-raises; guarded_run says
	FAILURE once and gives None (the phase off for the run)."""
	from battle_model.aux_value import guarded_run

	model, replay, phase = setup_phase(rates=1e-3)
	start = {name: p.detach().clone() for name, p in model.state_dict().items()}
	draws = {"n": 0}
	real_batch = replay.batch

	def failing(*args, **kwargs):
		draws["n"] += 1
		if draws["n"] >= 3:  # (the held-out check, the first step's batch, then this)
			raise RuntimeError("a replay batch that fails")
		return real_batch(*args, **kwargs)

	replay.batch = failing
	readings, still = guarded_run(phase, model, replay, torch.device("cpu"), steps=5, batch_size=32, kl_weight=10.0, seconds=120, held_out_version=6)
	assert still is None and readings == {"aux_failed": 1}
	assert "aux: FAILURE" in capsys.readouterr().out
	assert all(torch.equal(start[name], value) for name, value in model.state_dict().items()) and not model.training


def test_the_phase_warms_its_rates_and_its_state_goes_through_a_checkpoint():
	"""A cold Adam's rate is rate x 1/WARMUP on its first step; the state (moments, steps done) saved and loaded again."""
	from battle_model.aux_value import WARMUP

	model, replay, phase = setup_phase(rates=1e-4, kl_max=1.0)
	phase.run(model, replay, torch.device("cpu"), steps=1, batch_size=32, kl_weight=10.0, seconds=120, held_out_version=6)
	assert phase.steps_done == 1 and all(abs(g["lr"] - 1e-4 / WARMUP) < 1e-12 for g in phase.optimizer.param_groups)
	saved = phase.state_dict()
	_, _, other = setup_phase(rates=1e-4, kl_max=1.0)
	other.load_state_dict(saved)
	assert other.steps_done == 1 and len(other.optimizer.state_dict()["state"]) == len(saved["optimizer"]["state"])
