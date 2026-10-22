import numpy as np
import pytest
import torch

from battle_model.constants import RESERVED_ACTION_TYPES, ActionType
from battle_model.distributions import entropy, enumerate_actions, is_legal, log_prob, sample, stack_actions
from battle_model.losses import ppo_loss
from battle_model.model import BattleModel
from battle_model.schema import INPUTS, OUTPUTS, finalize_masks, random_inputs


@pytest.fixture(scope="module")
def model():
	torch.manual_seed(0)
	return BattleModel().eval()


def run(model, x):
	with torch.no_grad():
		return model({k: torch.from_numpy(v) for k, v in x.items()})


def small_masks(seed):
	"""Random inputs whose masks are thinned so all legal actions can be enumerated."""
	x = random_inputs(batch=1, seed=seed)
	rng = np.random.default_rng(seed)
	for spec in INPUTS:
		if spec.name.startswith("m_") and spec.name != "m_type":
			x[spec.name] &= rng.random(x[spec.name].shape) < 0.35
	finalize_masks(x, 0)
	return x


def test_output_shapes(model):
	for batch in (1, 3):
		out = run(model, random_inputs(batch=batch, seed=batch))
		assert set(out) == {s.name for s in OUTPUTS}
		for spec in OUTPUTS:
			assert out[spec.name].shape == (batch,) + spec.shape, spec.name


def test_illegal_entries_are_neg_inf_and_rows_normalised(model):
	x = random_inputs(batch=2, seed=7)
	out = run(model, x)
	pairs = [("logp_type", "m_type"), ("logp_move_hex", "m_move_hex"), ("logp_melee_target", "m_melee_target"), ("logp_melee_from", "m_melee_from"), ("logp_shoot_target", "m_shoot_target"), ("logp_spell", "m_spell"), ("logp_spell_unit", "m_spell_unit"), ("logp_spell_hex", "m_spell_hex"), ("logp_spell2_hex", "m_spell2_hex"), ("logp_spell2_unit", "m_spell2_unit")]
	for logp_name, mask_name in pairs:
		logp, mask = out[logp_name].numpy(), x[mask_name]
		assert np.isneginf(logp[~mask]).all(), logp_name
		assert np.isfinite(logp[mask]).all(), logp_name
		rows = mask.any(axis=-1)
		sums = np.exp(logp).sum(axis=-1)
		assert np.allclose(sums[rows], 1.0, atol=1e-5), logp_name


def test_reserved_types_never_legal_in_random_inputs(model):
	x = random_inputs(batch=4, seed=3)
	assert not x["m_type"][:, [int(a) for a in RESERVED_ACTION_TYPES]].any()


def test_full_action_probabilities_sum_to_one_and_entropy_is_exact(model):
	for seed in range(5):
		x = small_masks(seed)
		out = run(model, x)
		actions = enumerate_actions(x)
		lp = _batched(out, actions)
		p = lp.exp().numpy()
		assert np.isclose(p.sum(), 1.0, atol=1e-4), (seed, p.sum())
		exact = float(-(p * lp.numpy()).sum())
		assert np.isclose(float(entropy(out)[0]), exact, atol=1e-4), seed


def _batched(out, actions):
	"""log_prob of many actions against one state: repeat the outputs along the batch axis."""
	n = len(actions)
	rep = {k: v.expand((n,) + v.shape[1:]) for k, v in out.items()}
	return log_prob(rep, stack_actions(actions))


def test_samples_are_legal_and_match_log_prob(model):
	rng = np.random.default_rng(0)
	for seed in range(10):
		x = random_inputs(batch=1, seed=seed)
		out = run(model, x)
		for _ in range(50):
			a = sample(out, rng=rng)
			assert is_legal(a, x), (seed, a)
			assert a["type"] not in [int(r) for r in RESERVED_ACTION_TYPES]
			assert np.isfinite(float(log_prob(out, stack_actions([a]))[0]))
		greedy = sample(out, greedy=True)
		assert is_legal(greedy, x)


def test_sampling_frequencies_match_probabilities(model):
	x = small_masks(11)
	out = run(model, x)
	actions = enumerate_actions(x)
	p = _batched(out, actions).exp().numpy()
	rng = np.random.default_rng(1)
	index = {tuple(sorted(a.items())): i for i, a in enumerate(actions)}
	counts = np.zeros(len(actions))
	n = 20000
	for _ in range(n):
		counts[index[tuple(sorted(sample(out, rng=rng).items()))]] += 1
	assert np.abs(counts / n - p).max() < 0.02


def test_ppo_loss_gradients_are_finite():
	torch.manual_seed(0)
	model = BattleModel().train()
	x = random_inputs(batch=4, seed=5)
	tx = {k: torch.from_numpy(v) for k, v in x.items()}
	with torch.no_grad():
		out = model(tx)
	actions = stack_actions([sample(out, b=b, rng=np.random.default_rng(b)) for b in range(4)])
	old = log_prob(out, actions)
	loss, parts = ppo_loss(model(tx), actions, old, advantage=torch.randn(4), returns=torch.randn(4))
	loss.backward()
	assert torch.isfinite(loss)
	for name, param in model.named_parameters():
		if param.grad is not None:
			assert torch.isfinite(param.grad).all(), name
	assert abs(float(parts["ratio"]) - 1.0) < 1e-4  # same parameters: ratio is 1
