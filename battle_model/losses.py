"""How the reward becomes a training signal: returns, advantages and the PPO loss.
(What counts as good is reward.py; this file is how the model learns from it.)

From reward to loss, for one battle seen by one side:

1. Trajectory. The side's model makes decisions d_0 .. d_{T-1}: one per unit turn it plays, plus one
   per hero cast. Between them the opponent acts and dice are rolled; to the model that is simply the
   environment. Turns played by the scripted fallback (war machines, tactics, round 30+) are not
   decisions of the model, but they do affect the outcome.

2. Reward. reward.py gives one number R when the battle ends: +-1 for the outcome, the value fractions
   and the round and mana penalties. Every earlier decision gets reward 0. Per-decision rewards, if we
   ever add some, would simply go into the same `rewards` array.

3. Return. The return G_t of decision t is the reward that followed it: G_t = gamma^(T-1-t) * R.
   With gamma = 1 (our choice: battles are short, and the round penalty already prices slowness)
   every decision of a battle has the same return R.

4. Value. The value head predicts V(s_t) ~ E[G_t]: the final reward expected from this state when
   playing with the current model. The value loss (V(s_t) - G_t)^2 trains that prediction.

5. Advantage. A_t is how much better than expected the outcome of decision t was. GAE(lambda) mixes
	   one step:    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)   how much the prediction rose after the action
	   full battle: G_t - V(s_t)                                   final outcome against the prediction
   lambda = 1 uses only the full-battle estimate: unbiased but noisy, since every decision of a won
   battle looks good, even the bad ones. lambda = 0 uses only the one-step estimate: little noise,
   but it trusts V completely. 0.95 is the usual compromise.
   This is how a single number at the end becomes credit for individual decisions: once the value
   head is any good, a decision after which the predicted outcome jumps gets a positive advantage
   and one after which it drops gets a negative one.

6. Policy loss. Make decisions with A_t > 0 more likely and A_t < 0 less likely:
	   loss_t = -A_t * ratio_t,   ratio_t = p_new(a_t | s_t) / p_old(a_t | s_t)
   At ratio 1 its gradient is the classic policy gradient -A_t * grad log p(a_t | s_t). The data was
   collected by the old model, so it stops being representative once the model moves far from it;
   PPO therefore clips the ratio to [1 - clip_range, 1 + clip_range] and takes the more pessimistic
   of the clipped and unclipped terms.

7. Total loss = policy + value_weight * value - entropy_weight * entropy.
   The entropy bonus keeps the action distribution from collapsing onto early habits. The ACTION_TYPE_PRIOR
   in model.py already starts the model at sensible odds, so this bonus can stay small.
"""
from dataclasses import dataclass

import torch

from .distributions import entropy, log_prob


@dataclass(frozen=True)
class AdvantageConfig:
	gamma: float = 1.0  # discount per decision (step 3); 1 = the battle's end counts fully for every decision
	lam: float = 0.95  # GAE lambda (step 5); 1 = judge each decision by the final outcome only


@dataclass(frozen=True)
class PPOConfig:
	clip_range: float = 0.2  # step 6: one update may change p(action) by at most +-20%; PPO's standard value
	value_weight: float = 0.5  # step 7: the value loss shares the network with the policy, standard PPO weight
	entropy_weight: float = 0.01  # step 7: small exploration bonus
	normalize_advantages: bool = True  # scale advantages to mean 0, std 1 per batch: the step size then does not depend on the reward scale


def battle_returns_and_advantages(values, rewards, config=AdvantageConfig()):
	"""Steps 3-5 for one battle seen by one side.

	values:  [T] V(s_t) from the model that played the battle (recorded while playing)
	rewards: [T] reward right after each decision: 0, except the last, which is reward.py's terminal reward
	Returns (returns [T], advantages [T]). returns is the value head's target: the GAE(lambda) return,
	equal to G_t for lambda = 1. The battle is over after the last decision, so its next value is 0.
	"""
	values = torch.as_tensor(values, dtype=torch.float32)
	rewards = torch.as_tensor(rewards, dtype=torch.float32)
	advantages = torch.zeros_like(values)
	running, next_value = 0.0, 0.0
	for t in reversed(range(len(values))):
		delta = rewards[t] + config.gamma * next_value - values[t]
		running = delta + config.gamma * config.lam * running
		advantages[t] = running
		next_value = values[t]
	return advantages + values, advantages


def ppo_loss(out, action, old_log_prob, advantage, returns, config=PPOConfig()):
	"""Steps 6-7 for a batch of decisions. out: model outputs for the batch inputs. Returns (loss, parts for logging)."""
	if config.normalize_advantages and advantage.numel() > 1:
		advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
	ratio = torch.exp(log_prob(out, action) - old_log_prob)
	clipped = ratio.clamp(1 - config.clip_range, 1 + config.clip_range)
	policy = -torch.min(ratio * advantage, clipped * advantage).mean()
	value = (out["value"] - returns).pow(2).mean()
	bonus = entropy(out).mean()
	total = policy + config.value_weight * value - config.entropy_weight * bonus
	return total, {"policy": policy.detach(), "value": value.detach(), "entropy": bonus.detach(), "ratio": ratio.mean().detach()}
