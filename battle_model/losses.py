"""How the model learns from experience: the PPO loss. (What counts as good is reward.py.)

PPO (proximal policy optimisation) trains on batches of decisions collected by playing with a
recent copy of the model. Each decision in a batch carries:
	inputs          the model inputs at that moment
	action          the full action that was taken (distributions.ACTION_FIELDS)
	old_log_prob    log p(action) under the model that played it
	advantage       how much better than expected this action turned out: discounted future
	                reward - value estimate (computed by the training loop, e.g. with GAE)
	returns         the reward that followed from this state, the target of the value head

The loss has three terms, weighted by PPOConfig:
	policy   make actions with positive advantage more likely and negative ones less likely.
	         The ratio p_new(action) / p_old(action) is clipped to [1 - clip_range, 1 + clip_range]
	         so one batch cannot move the policy far from the one that collected the data;
	         that is what keeps PPO stable.
	value    squared error between the value head and the observed returns. The value head
	         predicts the terminal reward (reward.py) from a state; the training loop uses it to
	         compute advantages, and later it serves as the battle outcome predictor.
	entropy  a small bonus for keeping the action distribution spread out, so the model keeps
	         exploring instead of locking onto its first habits.
"""
from dataclasses import dataclass

import torch

from .distributions import entropy, log_prob


@dataclass(frozen=True)
class PPOConfig:
	clip_range: float = 0.2  # max change of p(action) per update, as a ratio
	value_weight: float = 0.5  # weight of the value loss against the policy loss
	entropy_weight: float = 0.01  # weight of the exploration bonus


def ppo_loss(out, action, old_log_prob, advantage, returns, config=PPOConfig()):
	"""Total loss to minimise, and its parts for logging."""
	ratio = torch.exp(log_prob(out, action) - old_log_prob)
	clipped = ratio.clamp(1 - config.clip_range, 1 + config.clip_range)
	policy = -torch.min(ratio * advantage, clipped * advantage).mean()
	value = (out["value"] - returns).pow(2).mean()
	bonus = entropy(out).mean()
	total = policy + config.value_weight * value - config.entropy_weight * bonus
	return total, {"policy": policy.detach(), "value": value.detach(), "entropy": bonus.detach(), "ratio": ratio.mean().detach()}
