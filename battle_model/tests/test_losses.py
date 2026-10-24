import numpy as np
import torch

from battle_model.data import collate
from battle_model.distributions import ACTION_FIELDS, enumerate_actions, log_prob, log_prob_at, stack_actions
from battle_model.losses import AdvantageConfig, battle_returns_and_advantages, search_loss
from battle_model.model import BattleModel
from battle_model.schema import INPUTS, random_inputs

R = 2.5  # terminal reward of a won battle
VALUES = [1.0, 1.5, 0.5, 2.0]
REWARDS = [0.0, 0.0, 0.0, R]


def test_lambda_one_judges_every_decision_by_the_final_outcome():
	returns, adv = battle_returns_and_advantages(VALUES, REWARDS, AdvantageConfig(gamma=1.0, lam=1.0))
	assert torch.allclose(returns, torch.full((4,), R))
	assert torch.allclose(adv, R - torch.tensor(VALUES))


def test_lambda_zero_is_the_one_step_change_of_the_prediction():
	_, adv = battle_returns_and_advantages(VALUES, REWARDS, AdvantageConfig(gamma=1.0, lam=0.0))
	expected = [1.5 - 1.0, 0.5 - 1.5, 2.0 - 0.5, R - 2.0]  # V(next) - V(now), last step: R - V
	assert torch.allclose(adv, torch.tensor(expected))


def test_decision_that_raised_the_prediction_gets_credit():
	# With a decent value head, the decision after which V jumped (t=2: 0.5 -> 2.0) is the most credited
	_, adv = battle_returns_and_advantages(VALUES, REWARDS, AdvantageConfig(gamma=1.0, lam=0.5))
	assert int(adv.argmax()) == 2
	assert float(adv[1]) < 0  # after t=1 the prediction dropped


def test_discount():
	returns, _ = battle_returns_and_advantages([0.0, 0.0], [0.0, 1.0], AdvantageConfig(gamma=0.9, lam=1.0))
	assert torch.allclose(returns, torch.tensor([0.9, 1.0]))


def test_value_targets_per_round_and_final():
	from battle_model.losses import value_targets
	# Decisions in rounds 0, 0, 1, 2, 3; rounds 0..2 ended with the battle going on, it ended in round 3
	scores = {0: 0.2, 1: 0.5, 2: 0.9}
	labels = value_targets([0, 0, 1, 2, 3], scores, final_reward=2.4, horizons=3)
	expected = torch.tensor([
		[0.2, 0.5, 0.9, 2.4],  # round 0: end of rounds 0, 1, 2, then the battle
		[0.2, 0.5, 0.9, 2.4],  # same round: same labels
		[0.5, 0.9, 2.4, 2.4],  # round 1: round 3 is where it ended, so V2 = V
		[0.9, 2.4, 2.4, 2.4],
		[2.4, 2.4, 2.4, 2.4],  # last round: V0 == V1 == V2 == V
	])
	from battle_model.reward import to_scale
	assert torch.allclose(labels, to_scale(expected))


def test_value_loss_weights_and_minimum():
	from battle_model.losses import PPOConfig, value_loss
	labels = torch.tensor([[-0.4, 0.0, 0.4, 0.8]])
	perfect = {"value_logit": torch.atanh(labels)}  # heads predict exactly the labels
	off = {"value_logit": perfect["value_logit"] + torch.tensor([[2.0, 0.0, 0.0, 0.0]])}  # only V0 wrong
	config = PPOConfig()
	assert value_loss(off, labels, config)[0] > value_loss(perfect, labels, config)[0]
	only_v = PPOConfig(value_head_weights=(0.0, 0.0, 0.0, 1.0))
	assert torch.isclose(value_loss(off, labels, only_v)[0], value_loss(perfect, labels, only_v)[0])


def test_advantages_normalised_per_side_only_with_enough_decisions():
	from battle_model.losses import normalize_advantages
	config = AdvantageConfig(normalize_min_decisions=8)
	few = torch.tensor([0.3, -0.1, 0.2])
	assert torch.equal(normalize_advantages(few, torch.zeros(3), torch.zeros(3), config), few)

	# Two sides with very different scales, two battles each: each side ends up with mean 0, std 1
	attack = torch.arange(10, dtype=torch.float32) * 10
	defence = torch.arange(10, dtype=torch.float32) * 0.01 + 5
	sides = torch.tensor([0] * 10 + [1] * 10)
	battles = torch.tensor([0] * 5 + [1] * 5 + [2] * 5 + [3] * 5)
	scaled = normalize_advantages(torch.cat([attack, defence]), sides, battles, config)
	for side in (0, 1):
		part = scaled[sides == side]
		assert abs(float(part.mean())) < 1e-4 and abs(float(part.std()) - 1) < 1e-4


def test_a_single_battle_is_scaled_not_centred():
	from battle_model.losses import normalize_advantages
	won = torch.linspace(0.1, 0.3, 10)  # every decision of a won battle did better than expected
	scaled = normalize_advantages(won, torch.zeros(10), torch.zeros(10), AdvantageConfig(normalize_min_decisions=8))
	assert (scaled > 0).all() and abs(float(scaled.pow(2).mean()) - 1) < 1e-5



def test_policy_follows_only_the_heads_switched_on():
	from battle_model.losses import policy_advantages
	final = torch.tensor([0.5, -0.2])
	labels = torch.tensor([[0.3, 0.1, 0.4, 0.6], [0.2, 0.0, -0.1, 0.6]])
	values = torch.tensor([[0.1, 0.1, 0.1, 0.2], [0.4, 0.0, 0.3, 0.1]])
	assert torch.equal(policy_advantages(final, labels, values, AdvantageConfig(policy_heads=(0, 0, 0, 1))), final)
	both = policy_advantages(final, labels, values)  # default: V and V2
	assert torch.allclose(both, final + (labels[:, 2] - values[:, 2]))
	only_v2 = policy_advantages(final, labels, values, AdvantageConfig(policy_heads=(0, 0, 1, 0)))
	assert torch.allclose(only_v2, labels[:, 2] - values[:, 2])


def search_examples(seed=0, batch=3):
	"""Examples of random decisions, each with a search policy over a few of its legal actions (the last one none)."""
	x = random_inputs(batch=batch, seed=seed)
	rng = np.random.default_rng(seed)
	examples = []
	for b in range(batch):
		actions = enumerate_actions(x, b)
		picked = [actions[i] for i in rng.choice(len(actions), size=min(len(actions), 2 + b), replace=False)]
		probs = rng.random(len(picked))
		search = (torch.tensor([[a[f] for f in ACTION_FIELDS] for a in picked]), torch.tensor(probs / probs.sum(), dtype=torch.float32))
		examples.append({"inputs": {s.name: x[s.name][b] for s in INPUTS}, "action": picked[0], "old_log_prob": torch.tensor(0.0), "advantage": torch.tensor(0.0), "value_labels": torch.zeros(4), "search": search if b < batch - 1 else None})
	return examples


def test_log_prob_at_scores_actions_under_any_row():
	torch.manual_seed(0)
	examples = search_examples()
	batch = collate(examples)
	with torch.no_grad():
		out = BattleModel()(batch["inputs"])
	for b, e in enumerate(examples[:-1]):
		actions = {f: e["search"][0][:, j] for j, f in enumerate(ACTION_FIELDS)}
		alone = log_prob({k: v[b : b + 1].expand((len(e["search"][1]),) + v.shape[1:]) for k, v in out.items()}, actions)
		assert torch.allclose(log_prob_at(out, torch.full_like(actions["type"], b), actions), alone)


def test_search_loss_is_the_cross_entropy_to_the_search_policy():
	"""Padding (probability 0, fields -1) and a decision without a search add nothing; the loss is finite with finite
	gradients, and descending it moves the network toward the search's policy."""
	torch.manual_seed(0)
	examples = search_examples()
	batch = collate(examples)
	assert batch["search"]["probs"].shape == (3, 3) and batch["search"]["probs"][-1].sum() == 0
	model = BattleModel()
	out = model(batch["inputs"])
	loss, parts = search_loss(out, batch["search"], batch["value_labels"])
	expected = []
	for b, e in enumerate(examples[:-1]):
		actions, probs = e["search"]
		lp = log_prob({k: v[b : b + 1].expand((len(probs),) + v.shape[1:]) for k, v in out.items()}, {f: actions[:, j] for j, f in enumerate(ACTION_FIELDS)})
		expected.append(-(probs * lp).sum())
	assert torch.allclose(parts["policy"], torch.stack(expected).mean())
	kl_before = float(parts["approx_kl"])
	loss.backward()
	assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
	for _ in range(30):
		optimizer.zero_grad()
		loss, parts = search_loss(model(batch["inputs"]), batch["search"], batch["value_labels"])
		loss.backward()
		optimizer.step()
	assert float(parts["approx_kl"]) < 0.5 * kl_before
