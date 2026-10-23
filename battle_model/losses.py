"""How the reward becomes a training signal: returns, advantages and the PPO loss.
(What counts as good is reward.py; this file is how the model learns from it.)

PPO = Proximal Policy Optimization: improve the policy from games it played itself, in small steps
("proximal": each update stays close to the policy that collected the data).
Plan: PPO now, AlphaZero-style search (MCTS) later as an upgrade reusing this network; the vcmi-gym
author dropped MCTS because VCMI cannot copy and restore a battle state, which search needs.

Every value, label and reward below is on the -1..1 scale (reward.to_scale: -1 = worst possible
reward, +1 = best possible). Value heads are tanh outputs trained with cross-entropy; tanh(z) is
2 * sigmoid(2z) - 1, so this is the same model and loss as a sigmoid on 0..1, just easier to read.

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

4. Value. The value heads predict, from the state of decision t (played in battle round r_t):
	   V0  the reward formula evaluated at the end of round r_t      (reward.round_score)
	   V1  ... at the end of round r_t + 1
	   V2  ... at the end of round r_t + 2
	   V   the final reward R                                         (reward.terminal_reward)
   So every decision has N_VALUE_HORIZONS + 1 labels (value_targets). V's label is R for every
   decision of the battle; V0's label is shared by the decisions of one round and differs between
   rounds. If the battle ends before round r_t + k ends, Vk's label is R, so in the last round
   V0 = V1 = V2 = V. The shorter horizons are easier, less noisy targets that track the final
   outcome closely; the model's heads are boosted (model.py), each learning the difference to the
   previous one. The value loss is a weighted sum over the heads (value_head_weights, equal by default).
   Each head outputs tanh (-1..1) and its loss is the cross-entropy against its -1..1 label (both
   mapped to 0..1): same minimum as the squared error, but the gradient does not vanish when the
   output saturates on a wrong answer.
   Only V, the final-reward head, is used for the advantages below.

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

7. Total loss = policy + value_weight * (weighted value losses) - entropy_weight * entropy.
   The entropy bonus keeps the action distribution from collapsing onto early habits. The ACTION_TYPE_PRIOR
   in model.py already starts the model at sensible odds, so this bonus can stay small.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .constants import N_VALUE_HORIZONS
from .distributions import entropy, log_prob
from .reward import RewardConfig, to_scale


@dataclass(frozen=True)
class AdvantageConfig:
	gamma: float = 1.0  # discount per decision (step 3); 1 = the battle's end counts fully for every decision
	lam: float = 0.95  # GAE lambda (step 5); 1 = judge each decision by the final outcome only
	normalize_min_decisions: int = 8  # step 5: rescale a group's advantages only if it has at least this many decisions


@dataclass(frozen=True)
class PPOConfig:
	clip_range: float = 0.2  # step 6: one update may change p(action) by at most +-20%; PPO's standard value
	value_weight: float = 0.5  # step 7: the value loss shares the network with the policy, standard PPO weight
	entropy_weight: float = 0.01  # step 7: small exploration bonus
	value_head_weights: tuple = (1.0,) * (N_VALUE_HORIZONS + 1)  # step 4: share of each value head (V0, V1, V2, V) in the value loss


def value_targets(decision_rounds, round_scores, final_reward, horizons=N_VALUE_HORIZONS, reward_config=RewardConfig()):
	"""Step 4: the labels of the value heads for every decision of one battle seen by one side.

	decision_rounds: [T] battle round of each decision
	round_scores:    {round: reward.round_score at the end of that round} for every round that ended
	                 with the battle still going on (reward units)
	final_reward:    reward.terminal_reward of the battle (reward units)
	Returns [T, horizons + 1] on the -1..1 scale: column k < horizons is the score at the end of round
	r_t + k, or the final reward if the battle ended before that; the last column is the final reward.
	"""
	labels = torch.full((len(decision_rounds), horizons + 1), float(final_reward))
	for t, r in enumerate(decision_rounds):
		for k in range(horizons):
			if r + k in round_scores:
				labels[t, k] = round_scores[r + k]
	return to_scale(labels, reward_config)


def battle_returns_and_advantages(values, rewards, config=AdvantageConfig()):
	"""Steps 3-5 for one battle seen by one side.

	values:  [T] V(s_t), the final-reward head (-1..1), from the model that played the battle (recorded while playing)
	rewards: [T] reward right after each decision: 0, except the last, which is reward.to_scale(terminal reward)
	Returns (returns [T], advantages [T]). returns is the GAE(lambda) return (G_t for lambda = 1); the
	value heads are trained on value_targets instead, where V's label is the final reward itself.
	The battle is over after the last decision, so its next value is 0.
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


def value_loss(out, targets, config=PPOConfig()):
	"""Step 4 for a batch: weighted cross-entropy of each value head against its -1..1 label.

	A head's output is tanh(z) = 2 * sigmoid(2z) - 1, so its cross-entropy is the one of sigmoid(2z)
	against the label mapped to 0..1. targets: [B, N_VALUE_HORIZONS + 1] from value_targets.
	Returns (loss, per-head losses [heads]).
	"""
	probabilities = ((targets + 1) / 2).clamp(0.0, 1.0)
	per_head = F.binary_cross_entropy_with_logits(2 * out["value_logit"], probabilities, reduction="none").mean(dim=0)
	weights = torch.tensor(config.value_head_weights, dtype=per_head.dtype, device=per_head.device)
	return (per_head * weights).sum() / weights.sum(), per_head.detach()


def normalize_advantages(advantages, groups, battles, config=AdvantageConfig()):
	"""Step 5: rescale one epoch's advantages to mean 0, std 1 per group, so the step size does not depend on the reward scale.

	advantages, groups, battles: [N], for all decisions of the epoch. groups: the problem each decision
	belongs to (the side we played: attacking and defending are different problems); battles: the
	battle it was made in.

	Per group, never across groups, and never per batch. Normalising both sides together (as before)
	was a mistake: the centring shifted one side's advantages by the other side's luck (an unexpected
	attack win pushed every defence decision down), and the joint std is dominated by the side whose
	outcomes vary most, so the other side learned as if with a ~10x smaller learning rate (the lr 3e-4
	run's defence stopped improving once its attack started winning).

	A group is centred only if it has at least 2 battles: centring a single battle's advantages would
	make half the decisions of a won battle look bad. A single battle is only scaled, by its root mean
	square. With fewer than normalize_min_decisions decisions the statistics are meaningless (with 2
	decisions it only ranks them, whatever the outcome), so those advantages are left as they are.
	"""
	out = advantages.clone()
	for group in groups.unique():
		member = groups == group
		a = advantages[member]
		if len(a) < config.normalize_min_decisions:
			continue
		if len(battles[member].unique()) >= 2:
			out[member] = (a - a.mean()) / (a.std() + 1e-8)
		else:
			out[member] = a / (a.pow(2).mean().sqrt() + 1e-8)
	return out


def ppo_loss(out, action, old_log_prob, advantage, value_labels, config=PPOConfig()):
	"""Steps 4-7 for a batch of decisions. out: model outputs for the batch inputs.

	advantage: [B] from battle_returns_and_advantages (using V, the final-reward head, as values), already
	           normalised per side over the epoch by normalize_advantages
	value_labels: [B, N_VALUE_HORIZONS + 1] from value_targets
	Returns (loss, parts for logging). approx_kl (mean old - new log-prob) and clip_fraction (share of
	decisions whose ratio left the clip range) show how far this update moved the policy from the one
	that played: PPO's usual warning signs of too large steps.
	"""
	new_log_prob = log_prob(out, action)
	ratio = torch.exp(new_log_prob - old_log_prob)
	clipped = ratio.clamp(1 - config.clip_range, 1 + config.clip_range)
	policy = -torch.min(ratio * advantage, clipped * advantage).mean()
	value, per_head = value_loss(out, value_labels, config)
	bonus = entropy(out).mean()
	total = policy + config.value_weight * value - config.entropy_weight * bonus
	return total, {
		"policy": policy.detach(),
		"value": value.detach(),
		"value_per_head": per_head,
		"entropy": bonus.detach(),
		"ratio": ratio.mean().detach(),
		"approx_kl": (old_log_prob - new_log_prob).mean().detach(),
		"clip_fraction": ((ratio - 1).abs() > config.clip_range).float().mean().detach(),
	}
