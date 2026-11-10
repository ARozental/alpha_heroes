"""How the reward becomes a training signal: returns, advantages and the PPO loss.
(What counts as good is reward.py; this file is how the model learns from it.)

PPO = Proximal Policy Optimization: improve the policy from games it played itself, in small steps
("proximal": each update stays close to the policy that collected the data).
Plan: PPO now, AlphaZero-style search (MCTS) later as an upgrade reusing this network; the vcmi-gym
author dropped MCTS because VCMI cannot copy and restore a battle state, which search needs.

Every value, label and reward below is on the -1..1 scale of the reward (reward.py: zero-sum, -1 = we lost
everything and the enemy nothing, +1 the reverse). Value heads are tanh outputs trained with cross-entropy; tanh(z) is
2 * sigmoid(2z) - 1, so this is the same model and loss as a sigmoid on 0..1, just easier to read.

From reward to loss, for one battle seen by one side:

1. Trajectory. The side's model makes decisions d_0 .. d_{T-1}: one per unit turn it plays, plus one
   per hero cast. Between them the opponent acts and dice are rolled; to the model that is simply the
   environment. War machines under our control (a catapult's wall part, a First Aid Tent's patient, a ballista's
   shot) and tactics placements are the model's decisions too; the engine plays a war machine itself when the hero
   lacks its skill, and a stalemate (STALEMATE_ROUNDS without damage) ends in a retreat, as chess's 50-move rule.

2. Reward. reward.py gives one number R when the battle ends: the assets (army, hero, mana) the enemy lost
   minus those we lost, of both sides' assets at the start. Every earlier decision gets reward 0. Per-decision rewards, if we
   ever add some, would simply go into the same `rewards` array.

3. Return. The return G_t of decision t is the reward that followed it: G_t = gamma^(T-1-t) * R.
   With gamma = 1 (our choice: battles are short)
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
   V, the final-reward head, makes the advantages below, and V stands alone (not chained to V0..V2,
   model.py: chained, it inherited their material swings and taught the defence passivity, 2026-10-23). The
   policy maximizes V alone (AdvantageConfig.policy_heads, step 5b); V0..V2 are auxiliary predictions that shape
   the network.

5. Advantage. A_t is how much better than expected the outcome of decision t was (AdvantageConfig.credit):
	   "window" (the default): V at our first decision window_rounds (2) rounds on (the result once the battle has
	                ended) - V(s_t), mixed with the real outcome R - V(s_t): its share outcome_share (15%) once V is
	                good, more (up to outcome_share_max, 50%) while V's explained variance is low (train.py)
	   "gae":       GAE(lambda), between one step, delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) (how much the
	                prediction rose after the action), and the full battle, G_t - V(s_t) (lambda = 1: unbiased but
	                noisy, every decision of a won battle looks good, even the bad ones)
   This is how a single number at the end becomes credit for individual decisions: once the value
   head is any good, a decision after which the predicted outcome jumps gets a positive advantage
   and one after which it drops gets a negative one.

6. Policy loss. Make decisions with A_t > 0 more likely and A_t < 0 less likely:
	   loss_t = -A_t * ratio_t,   ratio_t = p_new(a_t | s_t) / p_old(a_t | s_t)
   At ratio 1 its gradient is the classic policy gradient -A_t * grad log p(a_t | s_t). The data was
   collected by the old model, so it stops being representative once the model moves far from it;
   PPO therefore clips the ratio to [1 - clip_range, 1 + clip_range] and takes the more pessimistic
   of the clipped and unclipped terms.

7. Total loss = policy + value_weight * (weighted value losses) + entropy_floor_weight * mean(max(0, floor - H)).
   The entropy floor keeps a decision from collapsing onto one choice: only decisions sharper than entropy_floor nats
   are pushed back up. It replaced a constant bonus (- entropy_weight * mean H, now 0 by default) on 2026-10-28, after
   joint1's median p(SHOOT) where shooting was legal fell from 0.99 to 0.32 in ~300 updates while entropy rose.
   What battle_model/gradient_study.py showed (runtime/studies/drift2): the bonus pushed every minibatch the same way,
   lowering p(SHOOT) by 4-15 a minibatch (Adam's units, t = -9 to -28); the policy gradient's mean along the same
   direction was not resolved (-59 to +48, t within +-0.7: ~+-650 a minibatch). So which push was the bigger is not
   shown, and v15-window trained 2,290 updates with the same bonus without that drift (the 2026-10-28 review). The
   floor stays for what it is: bounded (it lets a decision go at entropy_floor, a bonus pushes forever) and silent
   above it; whether it beats the bonus is an ablation from checkpoint 2840 not yet run. The ACTION_TYPE_PRIOR in
   model.py starts the model at sensible odds.
"""
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .constants import N_VALUE_HORIZONS
from .device import constant
from .distributions import entropy, log_prob, log_prob_at


@dataclass(frozen=True)
class AdvantageConfig:
	gamma: float = 1.0  # discount per decision (step 3); 1 = the battle's end counts fully for every decision
	# How the final-reward credit of a decision is made (step 5):
	#   "window"  the user's linear mix (2026-10-26):
	#             (1 - outcome_share) * [V(our first decision window_rounds rounds later) - V(now)]
	#             + outcome_share * [final reward - V(now)]
	#             the value head's view of the position a while later (the battle's result if it ends before), plus a
	#             share of the real outcome so the value head's blind spots still get corrected; the window is game
	#             time (rounds), the same for any army size, where lam counts decisions
	#   "gae"     GAE(lam): exponentially decaying shares of every later step (lam below)
	credit: str = "window"
	window_rounds: int = 2
	# The user, 2026-11-05: half the credit the outcome's, fixed (it was 0.15 at a value EV >= 0.4, all of joint7's 1,090
	# updates: 85% of a decision's credit was the value head's view two rounds on, its blind spots the policy's)
	outcome_share: float = 0.5
	# "window": the outcome's share follows how good the value head is (train.py, by its explained variance EV):
	# outcome_share_max at EV <= 0, falling linearly to outcome_share at EV >= outcome_share_full_ev. A weak value head is
	# not trusted with most of the credit: v15-window's first two hours (EV < 0) went flat and half unlearned casting,
	# and it was well again once EV was ~0.2-0.4. outcome_share_max = outcome_share (the default since 2026-11-05): off.
	outcome_share_max: float = 0.5
	outcome_share_full_ev: float = 0.4
	# GAE lambda (step 5), read by credit "gae" only: the "window" default has no decay of any kind. 1 = judge each
	# decision by the final outcome only. 0.8 was the default for 40 minutes on 2026-10-26 (da86ba9 to ca3ee7b: the
	# v15-history run; earlier runs 0.95); v15-window, joint1 and every run since used the window, so no lambda
	# comparison is pending. At 0.8 a decision 12 steps from the end gets 7% of its credit from the outcome (0.95: 54%)
	lam: float = 0.8
	normalize_min_decisions: int = 8  # step 5: rescale a group's advantages only if it has at least this many decisions
	# Step 5a: each battle's advantages in the unit of its weaker side (data.lopsided_scale), before the normalisation. The
	# reward is a share of both sides' assets, so what a lopsided battle's decisions could change was ~1/100 of an even
	# fight's, and 60% of joint7's battle updates were lopsided campaign battles (attack won 84%): near-silent (2026-11-05).
	# The reward, the value heads and the evals keep their unit
	lopsided_scale: bool = True
	# How much of each value head (V0, V1, V2, V) the policy maximizes (policy_advantages): weights. The policy
	# maximizes the final outcome V alone (the user, 2026-10-28: the longest horizon, as the campaign's day 7); the
	# window credit reaches it through V two rounds on, and the outcome's share covers V's blind spots while V is weak.
	# The round heads are chained to each other (each learns the difference to the previous, detached) and only shape
	# the network: a round head's term pays for material kept k rounds on, so it punishes a sacrifice that wins later (a
	# weak stack dying to shield a shooter). (V2's term once helped early runs: 2026-10-23, one seed each, V + V2 beat V
	# alone against BattleAI, attack -0.09 vs -0.24, defence +0.10 vs +0.01; V's explained variance was ~0 then and is
	# ~0.4 now.) fade_round_heads: a round head given a weight here counts less as V's trust rises (train.py)
	policy_heads: tuple = (0, 0, 0, 1)
	fade_round_heads: bool = True


@dataclass(frozen=True)
class PPOConfig:
	clip_range: float = 0.2  # step 6: one update may change p(action) by at most +-20%; PPO's standard value
	value_weight: float = 0.5  # step 7: the value loss shares the network with the policy, standard PPO weight
	entropy_weight: float = 0.0  # step 7: the constant bonus on every decision's entropy (0.01 until 2026-10-28; see step 7)
	entropy_floor: float = 0.3  # step 7: nats; ~the 10th percentile of decisions' entropy at v15/joint1 checkpoints
	# step 7: how hard a decision below the floor is pushed back up. 0 since 2026-10-29: in joint2 the floor was the one
	# steady push against shooting (gradient_study on 3078: t -5.8 generated, -8.8 campaign battles), as the sharpest
	# decisions it lifts are confident shots (23-29% of shooter states sat below it, 4-7% of the rest); PPO's clip and the
	# KL control already limit how fast a choice sharpens
	entropy_floor_weight: float = 0.0
	value_head_weights: tuple = (1.0,) * (N_VALUE_HORIZONS + 1)  # step 4: share of each value head (V0, V1, V2, V) in the value loss
	# The foresight heads (model.Foresight: what happens before our next decision), each target's loss weight: auxiliary,
	# so they shape the trunk the policy reads without outweighing the policy's own gradient (train.py logs the ratio of
	# their gradients on the trunk: at 0.05 the foresight's was 2.6% of the PPO losses' on joint2's 5520, too little to
	# teach the trunk anything; 0.4 for ~20%)
	foresight_weight: float = 0.4


def value_targets(decision_rounds, round_scores, final_reward, horizons=N_VALUE_HORIZONS):
	"""Step 4: the labels of the value heads for every decision of one battle seen by one side.

	decision_rounds: [T] battle round of each decision
	round_scores:    {round: reward.round_score at the end of that round} for every round that ended
	                 with the battle still going on (-1..1)
	final_reward:    reward.terminal_reward of the battle (-1..1)
	Returns [T, horizons + 1] on the -1..1 scale: column k < horizons is the score at the end of round
	r_t + k, or the final reward if the battle ended before that; the last column is the final reward.
	"""
	labels = torch.full((len(decision_rounds), horizons + 1), float(final_reward))
	for t, r in enumerate(decision_rounds):
		for k in range(horizons):
			if r + k in round_scores:
				labels[t, k] = round_scores[r + k]
	return labels


def battle_returns_and_advantages(values, rewards, config=AdvantageConfig(), rounds=None):
	"""Steps 3-5 for one battle seen by one side.

	values:  [T] V(s_t), the final-reward head (-1..1), from the model that played the battle (recorded while playing)
	rewards: [T] reward right after each decision: 0, except the last, which is the terminal reward
	rounds:  [T] the battle round of each decision (config.credit "window" needs them)
	Returns (returns [T], advantages [T]). returns is the advantage plus V (G_t for GAE with lambda = 1); the
	value heads are trained on value_targets instead, where V's label is the final reward itself.
	The battle is over after the last decision, so its next value is 0.
	"""
	values = torch.as_tensor(values, dtype=torch.float32)
	rewards = torch.as_tensor(rewards, dtype=torch.float32)
	if config.credit == "window":
		if rounds is None:
			raise ValueError("credit 'window' needs each decision's round")
		rounds = torch.as_tensor(rounds, dtype=torch.long)
		final = rewards.sum()  # the terminal reward (rewards are 0 before it)
		# Our first decision window_rounds rounds on (rounds never decrease): its V, or the result if there is none
		later_index = torch.searchsorted(rounds, rounds + config.window_rounds)
		later = torch.cat([values, final[None]])[later_index]
		advantages = (1 - config.outcome_share) * (later - values) + config.outcome_share * (final - values)
		return advantages + values, advantages
	advantages = torch.zeros_like(values)
	running, next_value = 0.0, 0.0
	for t in reversed(range(len(values))):
		delta = rewards[t] + config.gamma * next_value - values[t]
		running = delta + config.gamma * config.lam * running
		advantages[t] = running
		next_value = values[t]
	return advantages + values, advantages


def weighted_mean(values, weight):
	"""Mean of values [B] (or [B, ...] over the first axis) with weights [B] (None: plain mean)."""
	if weight is None:
		return values.mean(dim=0)
	w = weight.reshape(-1, *([1] * (values.dim() - 1)))
	return (values * w).sum(dim=0) / weight.sum().clamp(min=1e-8)


def value_loss(out, targets, config=PPOConfig(), weight=None):
	"""Step 4 for a batch: weighted cross-entropy of each value head against its -1..1 label.

	A head's output is tanh(z) = 2 * sigmoid(2z) - 1, so its cross-entropy is the one of sigmoid(2z)
	against the label mapped to 0..1. targets: [B, N_VALUE_HORIZONS + 1] from value_targets.
	Returns (loss, per-head losses [heads]).
	"""
	probabilities = ((targets + 1) / 2).clamp(0.0, 1.0)
	per_head = weighted_mean(F.binary_cross_entropy_with_logits(2 * out["value_logit"], probabilities, reduction="none"), weight)
	weights = constant(config.value_head_weights, per_head.device, per_head.dtype)
	return (per_head * weights).sum() / weights.sum(), per_head.detach()


def value_excess(out, targets, config=PPOConfig()):
	"""[B] each decision's value loss (value_loss's cross-entropy, its heads weighted as there) above its labels' own
	entropy, the cross-entropy's floor: 0 when every head predicts its label exactly. The plain cross-entropy moves
	with how decisive the battles were (a label near 0 has an entropy near ln 2), this only with how wrong V is."""
	p = ((targets + 1) / 2).clamp(0.0, 1.0)
	cross = F.binary_cross_entropy_with_logits(2 * out["value_logit"], p, reduction="none")
	floor = -(torch.special.xlogy(p, p) + torch.special.xlogy(1 - p, 1 - p))
	weights = constant(config.value_head_weights, cross.device, cross.dtype)
	return ((cross - floor) * weights).sum(dim=-1) / weights.sum()


def policy_advantages(final_advantages, value_labels, values, config=AdvantageConfig()):
	"""Step 5b: the advantage the policy follows, from the value heads that config.policy_heads switches on.

	final_advantages: [T] each decision's final-reward credit (battle_returns_and_advantages: the window by default)
	value_labels:     [T, heads] value_targets; values: [T, heads] the heads' outputs, both -1..1
	V counts with that credit; a round head k with its label (the reward formula at the end of round
	r_t + k) minus its own prediction: how much better than it expected that round went.
	"""
	heads = config.policy_heads
	if len(heads) != values.shape[1]:
		raise ValueError(f"policy_heads needs one weight per value head ({values.shape[1]}), got {heads}")
	total = final_advantages * float(heads[-1])
	for k, weight in enumerate(heads[:-1]):
		if weight:
			total = total + float(weight) * (value_labels[:, k] - values[:, k])
	return total


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


def ppo_loss(out, action, old_log_prob, advantage, value_labels, config=PPOConfig(), weight=None, weight_base=None):
	"""Steps 4-7 for a batch of decisions. out: model outputs for the batch inputs. weight: [B] how much each decision
	counts (data.example_weight: a battle that hardly matters counts little), or None for all alike.

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
	policy = -weighted_mean(torch.min(ratio * advantage, clipped * advantage), weight)
	value, per_head = value_loss(out, value_labels, config, weight)
	h = entropy(out)
	entropy_loss, bonus, below = entropy_term(out, config, weight, h=h)
	total = policy + config.value_weight * value + entropy_loss
	return total, {
		"policy": policy.detach(),
		"value": value.detach(),
		"value_per_head": per_head,
		"entropy": bonus.detach(),
		# (with the weights before TrainConfig.battle_length_scale: the entropy series keeps its meaning; weight_base None:
		# the weights are those)
		"entropy_base": weighted_mean(h.detach(), weight_base) if weight_base is not None else bonus.detach(),
		"below_floor": below,
		"ratio": ratio.mean().detach(),
		"approx_kl": (old_log_prob - new_log_prob).mean().detach(),
		"clip_fraction": ((ratio - 1).abs() > config.clip_range).float().mean().detach(),
	}


FORESIGHT_PARTS = ("dies", "lost", "dealt", "hex_enemy")


def foresight_loss(out, labels, config=PPOConfig(), weight=None):
	"""model.Foresight's loss: binary cross-entropy (soft for the shares lost and dealt, 0..1) of each target against what
	happened before our next decision (data.foresight_labels), over the unit slots alive then and every hex, each
	decision weighted as its battle (weight); a battle's last decision has none (labels["valid"] 0). Returns (loss,
	{part: its mean loss}): FORESIGHT_PARTS, each part's weight config.foresight_weight."""
	valid = labels["valid"] * (weight if weight is not None else 1.0)
	denominator = valid.sum().clamp(min=1e-6)
	mask = labels["unit_mask"]
	targets = torch.stack([labels["dies"], labels["lost"], labels["dealt"]], dim=-1)
	unit = nn.functional.binary_cross_entropy_with_logits(out["foresight_unit"], targets, reduction="none")  # [B, U, 3]
	unit = (unit * mask[..., None]).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 3]
	hexes = nn.functional.binary_cross_entropy_with_logits(out["foresight_hex"], labels["hex_enemy"], reduction="none").mean(dim=-1)  # [B]
	per_part = torch.cat([unit, hexes[:, None]], dim=-1)  # [B, 4]
	parts = (per_part * valid[:, None]).sum(dim=0) / denominator
	return config.foresight_weight * parts.sum(), {name: parts[i].detach() for i, name in enumerate(FORESIGHT_PARTS)}


def entropy_term(out, config=PPOConfig(), weight=None, h=None):
	"""Step 7: (the loss term, the mean entropy, the share of decisions below the floor). The floor pushes up only
	decisions sharper than config.entropy_floor; the constant bonus (entropy_weight) pushes every decision. h: the
	decisions' entropies when the caller has them (entropy(out))."""
	h = entropy(out) if h is None else h
	bonus = weighted_mean(h, weight)
	term = -config.entropy_weight * bonus
	if config.entropy_floor_weight:
		term = term + config.entropy_floor_weight * weighted_mean(torch.relu(config.entropy_floor - h), weight)
	return term, bonus, (h < config.entropy_floor).float().mean().detach()


def search_loss(out, search, value_labels, config=PPOConfig()):
	"""AlphaZero's loss, for decisions a search played (Gumbel AlphaZero, agents/AlphaBattleAI/GumbelSearch.h): the
	policy's cross-entropy against the search's improved policy over the root actions (softmax of the network's
	logits plus the completed Q-values), instead of PPO's policy loss; the value loss and the entropy bonus as in
	ppo_loss (the value heads learn the battle's outcome either way).

	search: {"action": {field: [B, K]}, "probs": [B, K]} (data.collate), padding at probability 0. Rows without a
	search (it failed) train only the value. approx_kl here is KL(search policy || network): how far the network is
	from what the search found; clip_fraction is 0 (nothing is clipped).
	"""
	probs = search["probs"]
	b, k = probs.shape
	rows = torch.arange(b, device=probs.device)[:, None].expand(b, k).reshape(-1)
	logp = log_prob_at(out, rows, {f: v.reshape(-1) for f, v in search["action"].items()}).view(b, k)
	used = probs > 0
	cross = -(probs * torch.where(used, logp, torch.zeros_like(logp))).sum(dim=-1)  # 0 * -inf would be NaN
	own = -(probs * torch.where(used, probs.clamp(min=1e-12).log(), torch.zeros_like(probs))).sum(dim=-1)
	searched = used.any(dim=-1).float()
	count = searched.sum().clamp(min=1.0)
	policy = (cross * searched).sum() / count
	value, per_head = value_loss(out, value_labels, config)
	entropy_loss, bonus, below = entropy_term(out, config)
	total = policy + config.value_weight * value + entropy_loss
	return total, {
		"policy": policy.detach(),
		"value": value.detach(),
		"value_per_head": per_head,
		"entropy": bonus.detach(),
		"entropy_base": bonus.detach(),  # (no weights here: the same)
		"below_floor": below,
		"ratio": torch.ones((), device=probs.device),
		"approx_kl": (((cross - own) * searched).sum() / count).detach(),
		"clip_fraction": torch.zeros((), device=probs.device),
	}
