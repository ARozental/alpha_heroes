import numpy as np
import torch

from battle_model.losses import AdvantageConfig, battle_returns_and_advantages

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


def test_self_imitation_pulls_only_toward_lines_better_than_expected():
	from battle_model.losses import PPOConfig, self_imitation_loss
	from battle_model.model import BattleModel
	from battle_model.schema import random_inputs
	from battle_model.distributions import log_prob, sample, stack_actions
	torch.manual_seed(0)
	model = BattleModel(dim=32, layers=1)
	inputs = {k: torch.as_tensor(v) for k, v in random_inputs(4).items()}
	out = model(inputs)
	action = stack_actions([sample(out, b, np.random.default_rng(b)) for b in range(4)])
	config = PPOConfig(self_imitation_weight=1.0)
	# V starts at 0: a line worth -0.5 is worse than expected, so it is not imitated
	assert float(self_imitation_loss(out, action, -0.5, config)[0]) == 0.0
	loss, _ = self_imitation_loss(out, action, 0.8, config)
	loss.backward()
	with torch.no_grad():
		for p in model.parameters():
			if p.grad is not None:
				p -= 0.05 * p.grad
	assert float(log_prob(model(inputs), action).sum()) > float(log_prob(out, action).sum())
