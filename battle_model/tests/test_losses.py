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
	from battle_model.reward import to_unit
	assert torch.allclose(labels, to_unit(expected))


def test_value_loss_weights_and_minimum():
	from battle_model.losses import PPOConfig, value_loss
	labels = torch.tensor([[0.3, 0.5, 0.7, 0.9]])
	perfect = {"value_logit": torch.log(labels / (1 - labels))}  # heads predict exactly the labels
	off = {"value_logit": perfect["value_logit"] + torch.tensor([[2.0, 0.0, 0.0, 0.0]])}  # only V0 wrong
	config = PPOConfig()
	assert value_loss(off, labels, config)[0] > value_loss(perfect, labels, config)[0]
	only_v = PPOConfig(value_head_weights=(0.0, 0.0, 0.0, 1.0))
	assert torch.isclose(value_loss(off, labels, only_v)[0], value_loss(perfect, labels, only_v)[0])
