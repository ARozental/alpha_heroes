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
