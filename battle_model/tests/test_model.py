import numpy as np
import pytest
import torch

from battle_model.constants import RESERVED_ACTION_TYPES, ActionType
from battle_model.distributions import entropy, enumerate_actions, is_legal, log_prob, sample, stack_actions
from battle_model.losses import ppo_loss
from battle_model.model import BattleModel
from battle_model.model_v1 import BattleModelV1
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
		assert set(out) == {s.name for s in OUTPUTS} | {"value_logit"}  # value_logit: training only
		for spec in OUTPUTS:
			assert out[spec.name].shape == (batch,) + spec.shape, spec.name


def test_illegal_entries_are_neg_inf_and_rows_normalised(model):
	x = random_inputs(batch=2, seed=7)
	out = run(model, x)
	pairs = [("logp_type", "m_type"), ("logp_move_hex", "m_move_hex"), ("logp_melee_target", "m_melee_target"), ("logp_melee_from", "m_melee_from"), ("logp_shoot_target", "m_shoot_target"), ("logp_spell", "m_spell"), ("logp_spell_unit", "m_spell_unit"), ("logp_spell_hex", "m_spell_hex"), ("logp_spell2_hex", "m_spell2_hex"), ("logp_spell2_unit", "m_spell2_unit"), ("logp_creature_spell_unit", "m_creature_spell_unit"), ("logp_creature_spell_hex", "m_creature_spell_hex")]
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


def test_entropy_bonus_does_not_favour_types_with_many_options(model):
	"""The bonus's gradient on the type logits is the type entropy's alone: without that, each type was paid the entropy of
	what follows it (MOVE's ~100 hexes against WAIT's nothing), a constant push toward moving."""
	from battle_model.distributions import _entropy
	for seed in range(3):
		out = run(model, small_masks(seed))
		logp_type = out["logp_type"].clone().requires_grad_()
		(full,) = torch.autograd.grad(entropy({**out, "logp_type": logp_type}).sum(), logp_type, allow_unused=True)
		(alone,) = torch.autograd.grad(_entropy(logp_type).sum(), logp_type)
		legal = torch.isfinite(out["logp_type"])
		assert torch.allclose(full[legal], alone[legal], atol=1e-6), seed


def _batched(out, actions):
	"""log_prob of many actions against one state: repeat the outputs along the batch axis."""
	n = len(actions)
	rep = {k: v.expand((n,) + v.shape[1:]) for k, v in out.items()}
	return log_prob(rep, stack_actions(actions))


SUMMON_DEMONS = 102  # past the hero spells' ids (N_SPELLS)


@pytest.mark.parametrize("on_hexes", (False, True))
def test_creature_spells_are_full_actions(model, on_hexes):
	"""CREATURE_SPELL and its targets (units or hexes) sum to one with every other action, the entropy stays exact,
	and samples are legal; the spell is the acting unit's, so it only informs."""
	x = small_masks(2)
	x["g_creature_spell"][0] = SUMMON_DEMONS
	x["m_creature_spell_unit"][0] = False
	x["m_creature_spell_hex"][0] = False
	if on_hexes:
		x["m_creature_spell_hex"][0, [20, 40, 60]] = True
	else:
		x["m_creature_spell_unit"][0, [1, 3]] = True
	finalize_masks(x, 0)
	assert x["m_type"][0, ActionType.CREATURE_SPELL]
	out = run(model, x)
	actions = enumerate_actions(x)
	casts = [a for a in actions if a["type"] == ActionType.CREATURE_SPELL]
	assert len(casts) == (3 if on_hexes else 2) and all(a["spell"] == SUMMON_DEMONS for a in casts)
	lp = _batched(out, actions)
	assert np.isclose(float(lp.exp().sum()), 1.0, atol=1e-4)
	assert np.isclose(float(entropy(out)[0]), float(-(lp.exp() * lp).sum()), atol=1e-4)
	rng = np.random.default_rng(0)
	samples = [sample(out, rng=rng) for _ in range(300)]
	assert all(is_legal(a, x) for a in samples)
	assert any(a["type"] == ActionType.CREATURE_SPELL for a in samples)


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
	labels = torch.rand(4, out["value"].shape[1]) * 2 - 1  # value labels are on the -1..1 scale
	loss, parts = ppo_loss(model(tx), actions, old, advantage=torch.randn(4), value_labels=labels)
	loss.backward()
	assert torch.isfinite(loss)
	for name, param in model.named_parameters():
		if param.grad is not None:
			assert torch.isfinite(param.grad).all(), name
	assert abs(float(parts["ratio"]) - 1.0) < 1e-4  # same parameters: ratio is 1


def test_untrained_model_picks_types_with_the_prior_odds(monkeypatch):
	"""With its rule weights at 0 and the value priors off, the untrained model's types follow ACTION_TYPE_PRIOR exactly
	(zero pointer scores: every follow-up is worth 0, so the best one adds nothing)."""
	import battle_model.model as model_module
	from battle_model.layers import ACTION_TYPE_PRIOR

	monkeypatch.setattr(model_module, "VALUE_PRIOR_LOGITS", 0.0)
	torch.manual_seed(3)
	m = BattleModel().eval()
	with torch.no_grad():  # the spells' biases lift HERO_SPELL by the best castable spell's: zero them here
		m.actions.spell_head.bias.zero_()
		m.actions.spell_mastery.zero_()
	x = random_inputs(batch=1, seed=4)
	out = run(m, x)
	legal = np.flatnonzero(x["m_type"][0])
	prior = np.array([ACTION_TYPE_PRIOR.get(ActionType(t), 0.0) for t in legal])
	expected = np.exp(prior) / np.exp(prior).sum()
	assert np.allclose(out["logp_type"][0, legal].exp().numpy(), expected, atol=1e-5)
	# The round heads start mid-range; the final one at the strength prior, V = 2 share - 1 = tanh(z / 2), z the
	# "all units" Lanchester share's log-odds (exponents 1: log health ratio + log damage ratio)
	assert np.allclose(out["value"][:, :-1].numpy(), 0.0)
	ratios = m.facts({k: torch.from_numpy(v) for k, v in x.items()})["strength"][:, 0]
	assert np.allclose(out["value"][:, -1].numpy(), torch.tanh(ratios.sum(dim=-1) / 2).numpy(), atol=1e-6)


def test_values_stay_in_the_unit_range():
	torch.manual_seed(0)
	m = BattleModel().eval()
	for head in m.values.heads:  # large random heads to push the outputs to their limits
		torch.nn.init.normal_(head.weight, std=50.0)
	out = run(m, random_inputs(batch=8, seed=2))
	assert (out["value"] >= -1).all() and (out["value"] <= 1).all()
	assert torch.allclose(out["value"], torch.tanh(out["value_logit"]))


def test_boost_stop_gradient_keeps_long_horizon_labels_off_short_heads():
	torch.manual_seed(0)
	x = {k: torch.from_numpy(v) for k, v in random_inputs(batch=2, seed=3).items()}
	for stop, expect_grad in ((True, False), (False, True)):
		m = BattleModel(boost_stop_gradient=stop)
		m(x)["value_logit"][:, -2].sum().backward()  # a loss on V2, the longest round horizon, only
		v0_grad = m.values.heads[0].weight.grad
		assert (v0_grad is not None and v0_grad.abs().sum() > 0) == expect_grad


def test_final_value_head_stands_alone():
	"""V, used for the advantages, predicts the final reward itself: it must not inherit V0..V2's swings."""
	torch.manual_seed(0)
	x = {k: torch.from_numpy(v) for k, v in random_inputs(batch=2, seed=3).items()}
	m = BattleModelV1(boost_stop_gradient=False)
	for head in m.value_heads:  # non-zero heads, so a chained logit would differ from the head's own output
		torch.nn.init.normal_(head.weight)
	out = m(x)
	out["value_logit"][:, -1].sum().backward()
	assert all(h.weight.grad is None or h.weight.grad.abs().sum() == 0 for h in m.value_heads[:-1])
	chained = BattleModelV1(boost_stop_gradient=False, chain_final_value=True)
	chained.load_state_dict(m.state_dict())
	assert not torch.allclose(chained(x)["value_logit"][:, -1], out["value_logit"][:, -1])


def test_training_with_precomputed_facts_matches_the_exported_forward():
	"""train.py computes model.facts once per epoch and trains on model(x, facts): it must be the forward that the
	ONNX plays (else PPO's ratio to the recorded policy is off from the first step)."""
	torch.manual_seed(1)
	m = BattleModel()
	x = {k: torch.from_numpy(v) for k, v in random_inputs(batch=6, seed=3).items()}
	with torch.no_grad():
		played = m.eval()(x)
		trained = m.train()(x, m.facts(x))
	for spec in OUTPUTS:
		a, b = played[spec.name], trained[spec.name]
		assert torch.equal(torch.isneginf(a), torch.isneginf(b)), spec.name
		finite = torch.isfinite(a)
		assert torch.allclose(a[finite], b[finite], atol=1e-6), spec.name


def test_history_reaches_the_outputs():
	"""The event log and the enemy hero's seen spells change the model's outputs; an empty log is fine. With random
	weights: an untrained model's heads start from their rule priors, the trunk's part at 0, so nothing reaches them."""
	torch.manual_seed(1)
	model = BattleModel().eval()
	with torch.no_grad():
		for p in model.parameters():
			p.add_(0.05 * torch.randn_like(p))
	x = random_inputs(batch=1, seed=3)
	base = run(model, x)
	no_events = dict(x, e_present=np.zeros_like(x["e_present"]))
	assert all(np.isfinite(run(model, no_events)["value"]).all() for _ in (0,))
	enemy = 1 - int(x["g_our_side"][0])
	seen = {k: v.copy() for k, v in x.items()}
	seen["h_seen_spells"][0, enemy] = 0
	seen["h_seen_spells"][0, enemy, 15] = 2
	finite = lambda t: torch.nan_to_num(t, neginf=0.0)
	for changed in (no_events, seen):
		out = run(model, changed)
		assert not torch.allclose(finite(out["logp_move_hex"]), finite(base["logp_move_hex"])), "history did not reach the policy"
		assert not torch.allclose(out["value"], base["value"]), "history did not reach the value"



def test_the_value_priors_lift_what_gains_and_load_into_old_checkpoints(monkeypatch):
	"""Each option's engine value adds VALUE_PRIOR_LOGITS x gain x tanh(value / VALUE_PRIOR_SCALE) to its logit, and its
	type gets its best option's through the follow-up: a valuable shot puts SHOOT above defending and waiting by about
	that (the rest of the untrained model is 0). A state dict from before the priors loads (strictly), at gain 1."""
	import battle_model.model as model_module
	from battle_model import features as F
	from battle_model.model import VALUE_PRIOR_LOGITS, VALUE_PRIOR_SCALE

	torch.manual_seed(5)
	m = BattleModel().eval()
	x = random_inputs(batch=1, seed=7)
	x["m_type"][0] = False
	x["m_type"][0, [ActionType.WAIT, ActionType.DEFEND, ActionType.SHOOT]] = True
	target = int(np.flatnonzero(x["m_shoot_target"][0])[0])
	x["a_shoot_value"][0, target] = 10**6 * (1 - 2 * int(x["g_our_side"][0]))  # the engine: a shot that gains a lot
	t = {k: torch.from_numpy(v) for k, v in x.items()}
	with torch.no_grad():
		facts = m.facts(t)
		shot = facts["shot"][0, :, F.ATTACK_FEATURES.index("net_value")]
		legal = t["m_shoot_target"][0]
		assert float(shot[legal].max()) > 0.04  # (2x VALUE_PRIOR_SCALE: the prior near its full VALUE_PRIOR_LOGITS)
		out = m(t, facts)
		with_prior = out["logp_type"][0, ActionType.SHOOT] - out["logp_type"][0, ActionType.DEFEND]
		monkeypatch.setattr(model_module, "VALUE_PRIOR_LOGITS", 0.0)
		without = m(t, facts)["logp_type"][0]
	lift = float(with_prior - (without[ActionType.SHOOT] - without[ActionType.DEFEND]))
	assert abs(lift - VALUE_PRIOR_LOGITS * float(torch.tanh(shot[legal].max() / VALUE_PRIOR_SCALE))) < 1e-4
	old = {k: v for k, v in m.state_dict().items() if not k.endswith("value_gains")}
	fresh = BattleModel()
	fresh.load_state_dict(old)  # strict: the hook adds the priors' gains
	assert torch.equal(fresh.actions.value_gains.detach(), torch.zeros(4))


def test_the_trunks_folded_relation_bias_is_the_bias():
	"""Block folds its relation bias r[h, relation(i, j)] into q and k (scaled_dot_product_attention): the same attention
	as adding the bias to the logits (relation 0 opposite sides, 1 either has none, 2 same side)."""
	import math

	from battle_model.constants import MASKED_LOGIT
	from battle_model.model import Block

	torch.manual_seed(3)
	block = Block(64, 4)
	with torch.no_grad():
		block.relation.copy_(torch.randn(4, 3))
	b, n = 3, 40
	x = torch.randn(b, n, 64)
	team = torch.randint(-1, 2, (b, n))
	valid = torch.rand(b, n) > 0.3
	valid[:, 0] = True
	q, k, v = block.qkv(block.norm1(x)).chunk(3, dim=-1)
	q, k, v = (t.reshape(b, n, 4, 16).transpose(1, 2) for t in (q, k, v))
	relation = (team[:, :, None] * team[:, None, :]).sign() + 1
	bias = block.relation[torch.arange(4)[None, :, None, None], relation[:, None]]
	att = ((q @ k.transpose(-1, -2)) / math.sqrt(16) + bias).masked_fill(~valid[:, None, None, :], MASKED_LOGIT).softmax(dim=-1)
	h = x + block.proj((att @ v).transpose(1, 2).reshape(b, n, 64))
	expected = h + block.mlp(block.norm2(h))
	assert torch.allclose(block(x, valid, team), expected, atol=1e-5)
