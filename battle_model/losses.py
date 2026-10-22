"""Training losses: clipped PPO policy loss, value loss and entropy bonus.

No training loop yet; these are the pieces it will call. A batch holds the model inputs,
the actions that were taken, their log-probabilities when they were taken (old_log_prob),
advantages and returns.
"""
from dataclasses import dataclass

import torch

from .distributions import entropy, log_prob


@dataclass
class PPOConfig:
	clip: float = 0.2
	value_coef: float = 0.5
	entropy_coef: float = 0.01


def ppo_loss(out, action, old_log_prob, advantage, returns, config=PPOConfig()):
	"""Returns (total loss, dict of components for logging)."""
	new_log_prob = log_prob(out, action)
	ratio = torch.exp(new_log_prob - old_log_prob)
	clipped = ratio.clamp(1 - config.clip, 1 + config.clip)
	policy = -torch.min(ratio * advantage, clipped * advantage).mean()
	value = (out["value"] - returns).pow(2).mean()
	ent = entropy(out).mean()
	total = policy + config.value_coef * value - config.entropy_coef * ent
	return total, {"policy": policy.detach(), "value": value.detach(), "entropy": ent.detach(), "ratio": ratio.mean().detach()}
