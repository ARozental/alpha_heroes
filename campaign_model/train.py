"""PPO for the campaign model: the update a joint run takes (learner.py) and a one-game smoke test.

	.venv/bin/python -m campaign_model.train --days 5 --seed 1     # one self-play game, then one update

PPO as the battle model's (battle_model/train.py): the clipped surrogate on advantages normalized over the update,
passes over minibatches of fixed size, the KL control (an update stops once its KL from the policy at its start passes
1.5 x target_kl, and the rate's scale is cut by 1.5, or raised back by 1.5 after an update that ended under half of it,
never above lr_scale_max: the user's rule, never above 3e-5; with a teacher and config.kl_teacher_inflation, the
target is (1 + its share against PPO's) x target_kl, as until 2026-11-06). Entropy: a floor, not a bonus (battle_model/losses.py's
lesson of 2026-10-28: a constant bonus is a consistent push that beats a noise-level policy gradient), with the
parents' probabilities detached (a type's table entropy weighs by p(type), which gets no gradient from it:
test_entropy_bonus_does_not_favour_types_with_many_options). The value heads are trained with cross-entropy (the
user's rule for 0..1 targets: it learns far better than squared error, whose gradient dies near 0 and 1) against
their grades (labels.ladder_targets: the shares VALUE_OFFSETS days on, the longer heads' graded against the shorter
ones' later forecasts), and the growth heads, at growth_weight each, against our own value's log growth on their 0..1
scale (labels.growth_scale), and V_end, at a share head's weight, against its grades (labels.end_targets); a missing
grade (nan: past the day cap) is left out, and so is a decision without credit from the policy's loss. The optimizer (make_optimizer) is Adam, with weight decay only on the
corrections of the model's fixed priors (model.prior_corrections): back to the prior unless the games keep pushing. Its
groups (GRAD_GROUPS, docs/optimizer_plan.md): the value heads' at value_lr_factor x the rate (the critic on the fast
timescale), the trunk's at trunk_beta1, each group's gradient clipped apart (clip_gradients).
"""
import argparse
import collections
import contextlib
import itertools
import json
import queue
import random
import threading
import time
import traceback
from collections import Counter, deque

import numpy as np
import torch
import torch.nn.functional as F

from battle_model import play as battle_play
from battle_model.inference import InferenceServer
from battle_model.optim import clip_gradients, moments_by_name, signal_fraction

from . import constants as c
from .config import CampaignConfig
from .empire import EmpireConfig, empire_value
from .labels import COLORS, GROWTH_CAP, PHI_FLOOR, SHARE_CHANGE_CAP, LabelConfig, blend_credit, end_advantages, end_labels, end_steps, explained_variance, growth_labels, label, residual_variances
from .model import build_model
from .play import make_map, play
from .graphs import Graphs, StepGraph
from .records import IN_BYTES, NO_EFFECT, decode_inputs, decode_wire, heroes_in, input_on_device, inputs_on_device
from .schema import HEADS

RUNS = battle_play.ROOT / "runtime" / "campaign"
TURN_KINDS = (c.DecisionKind.TURN, c.DecisionKind.EXCHANGE)


def model_kwargs(config):
	kwargs = {"dim": config.model_dim, "cell_dim": config.model_cell_dim, "heads": config.model_heads, "layers": config.model_layers}
	if config.model_architecture == "v1":
		return kwargs
	return {**kwargs, "architecture": config.model_architecture, **({"share_target": config.share_target} if config.share_target != "share" else {})}


@contextlib.contextmanager
def serve_model(checkpoint="", model_kwargs=None, processes=1, threads=2):
	"""A model (new, or a checkpoint's) served under "current" by an inference server for the games."""
	model, version = build_model(model_kwargs), 0
	if checkpoint:
		state = torch.load(checkpoint, map_location="cpu")
		model_kwargs = state.get("model_kwargs", {})
		model = build_model(model_kwargs)
		migrate_subtypes(state, model)  # (a checkpoint from before the dense subtype tables, 2026-10-31)
		model.load_state_dict(state["model"])
		version = state.get("version", 0)
	battle_play.SOCKET_DIR.mkdir(parents=True, exist_ok=True)
	server = InferenceServer(battle_play.SOCKET_DIR / f"campaign-{random.getrandbits(32):08x}.sock", family="campaign", model_kwargs=model_kwargs, threads=threads, processes=processes)
	try:
		server.set_model("current", version + 1, model)
		yield model, server
	finally:
		server.close()


def usable(decisions):
	"""The decisions to train on: the ones the AI made (a turn decision with no legal entry, or no model, has none)."""
	return [d for d in decisions if (d.action_type == c.ActionType.END or d.index >= 0) and (d.kind not in TURN_KINDS or d.action_type >= 0)]


def head_entropy(logp):
	"""Entropy of each row of log-probability tables [N, ...] (-inf entries are impossible)."""
	flat = logp.reshape(logp.shape[0], -1)
	p = flat.exp()
	return -(p * torch.where(torch.isfinite(flat), flat, torch.zeros_like(flat))).sum(dim=-1)


def choices(decisions):
	"""The decisions' choices and what the model said when they were made, as arrays."""
	return {
		"kind": np.array([d.kind for d in decisions], dtype=np.int64),
		"action": np.array([d.action_type for d in decisions], dtype=np.int64),
		"index": np.array([d.index for d in decisions], dtype=np.int64),
		"cell": np.array([d.cell for d in decisions], dtype=np.int64),
		"old_logp": np.array([d.logp for d in decisions], dtype=np.float32),
		"old_value": np.array([d.value for d in decisions], dtype=np.float32).reshape(-1, c.N_VALUE_OFFSETS),
	}


def chosen_logp(out, choice):
	"""The log-probability (now) of each decision's choice [N] and its decision's entropy [N]: a turn's is the type's
	plus each type's table entropy weighted by p(type) without its gradient; a question's is its options'.

	choice: tensors [N] kind, action (ActionType, -1 for a question), index (in the type's table, or the option), cell
	(a targeted spell's tile, -1 none)."""
	kind, action, index, cell = choice["kind"], choice["action"], choice["index"], choice["cell"]
	n = kind.shape[0]
	rows = torch.arange(n, device=kind.device)
	turn = (kind == c.DecisionKind.TURN) | (kind == c.DecisionKind.EXCHANGE)
	type_logp = out["type_logp"]
	type_p = type_logp.exp().detach()
	zero = torch.zeros((), device=kind.device)
	logp = torch.where(turn, type_logp[rows, action.clamp(min=0)], out["option_logp"][rows, index.clamp(0, c.N_OPTIONS - 1)])
	entropy = head_entropy(type_logp)
	for a, (output, _) in HEADS.items():
		table = out[output].reshape(n, -1)
		chosen = turn & (action == a)
		logp = logp + torch.where(chosen, table[rows, index.clamp(0, table.shape[1] - 1)], zero)
		entropy = entropy + type_p[:, a] * head_entropy(table)
	# A targeted spell's tile: its hero's own table
	hero = torch.div(index.clamp(min=0), c.N_ADVENTURE_SPELLS, rounding_mode="floor").clamp(max=c.OWN_HEROES - 1)
	cells = out["cast_cell_logp"].reshape(n, c.OWN_HEROES, -1)[rows, hero]
	targeted = turn & (action == c.ActionType.CAST) & (cell >= 0)
	logp = logp + torch.where(targeted, cells[rows, cell.clamp(min=0)], zero)
	# ARMY's target slot (schema 16; a record from before has none, cell -1): its row of army_slot_logp (normalized over
	# every slot legal for the stack or a part of it), renormalized over the whole stack's legal slots or a part's fit
	# ones (an empty slot, its own creature's: AlphaMapAI's draw), every value finite (an illegal slot's -inf in a branch
	# not taken would make the gradient NaN)
	if "army_slot_logp" in out:
		flat = index.clamp(min=0)
		amount = flat % c.N_ARMY_AMOUNTS
		to = torch.div(flat, c.N_ARMY_AMOUNTS, rounding_mode="floor") % c.N_HOLDERS
		slot = torch.div(flat, c.N_ARMY_AMOUNTS * c.N_HOLDERS, rounding_mode="floor") % c.ARMY_SLOTS
		holder = torch.div(flat, c.N_ARMY_AMOUNTS * c.N_HOLDERS * c.ARMY_SLOTS, rounding_mode="floor").clamp(max=c.N_HOLDERS - 1)
		row = out["army_slot_logp"][rows, holder, slot, to].clamp(min=-1e4)  # [N, ARMY_SLOT_CHOICES]
		fit, none = out["army_split_mask"][rows, holder, slot, to], torch.zeros_like(out["army_split_mask"][rows, holder, slot, to])
		whole = torch.cat([out["army_whole_mask"][rows, holder, slot, to], none], dim=-1)
		allowed = torch.where((amount == c.ArmyAmount.ALL)[:, None], whole, torch.cat([fit, none], dim=-1))
		allowed = torch.where((amount == c.ArmyAmount.ONE)[:, None], torch.cat([fit, fit], dim=-1), allowed)
		taken = row[rows, cell.clamp(0, c.ARMY_SLOT_CHOICES - 1)] - torch.logsumexp(row.masked_fill(~allowed, -1e4), dim=-1)
		moved = turn & (action == c.ActionType.ARMY) & (cell >= 0)
		logp = logp + torch.where(moved, taken, zero)
	entropy = torch.where(turn, entropy, head_entropy(out["option_logp"]))
	return logp, entropy


def swap_only_moves(out, types, indices):
	"""Whether each entry (types, indices: [N] or [N, K]) is an ARMY whole stack legal only as a swap (schema 16: a
	hero's last stack, a full army without its creature), from the outputs' slot masks."""
	flat = indices.clamp(min=0)
	amount = flat % c.N_ARMY_AMOUNTS
	to = torch.div(flat, c.N_ARMY_AMOUNTS, rounding_mode="floor") % c.N_HOLDERS
	slot = torch.div(flat, c.N_ARMY_AMOUNTS * c.N_HOLDERS, rounding_mode="floor") % c.ARMY_SLOTS
	holder = torch.div(flat, c.N_ARMY_AMOUNTS * c.N_HOLDERS * c.ARMY_SLOTS, rounding_mode="floor").clamp(max=c.N_HOLDERS - 1)
	rows = torch.arange(flat.shape[0], device=flat.device).reshape((-1,) + (1,) * (flat.dim() - 1))
	whole, fit = out["army_whole_mask"][rows, holder, slot, to], out["army_split_mask"][rows, holder, slot, to]
	return (types == c.ActionType.ARMY) & (amount == c.ArmyAmount.ALL) & whole.any(dim=-1) & ~(whole & fit).any(dim=-1)


def entries_logp(out, types, indices, valid):
	"""[N, K] log-probabilities of K turn entries a row (types: ActionType [N, K]; indices: in their tables [N, K]), -1e4
	where not valid or not legal: their probability 0, every gradient finite (an all-illegal row's log-sum-exp is not
	-inf, whose gradient would be NaN)."""
	logp = out["type_logp"].gather(1, types.clamp(0, c.N_ACTION_TYPES - 1))
	zero = torch.zeros((), device=types.device, dtype=logp.dtype)
	for a, (output, _) in HEADS.items():
		table = out[output].reshape(types.shape[0], -1)
		logp = logp + torch.where(types == a, table.gather(1, indices.clamp(0, table.shape[1] - 1)), zero)
	return logp.clamp(min=-1e4).masked_fill(~valid, -1e4)


def _teacher_span(exchange_multi=False):
	"""TEACHER_SPAN [2 (turn, exchange), label type, type]: the types Nullkiller2's label is weighed against.
	exchange_multi: MULTI in an exchange's too (config.exchange_span_multi)."""
	A = c.ActionType
	hero, town = {A.MOVE, A.TRAVEL, A.ARMY}, {A.BUILD, A.RECRUIT, A.HIRE, A.UPGRADE, A.TRADE, A.BUY}
	span = torch.zeros(2, c.N_ACTION_TYPES, c.N_ACTION_TYPES, dtype=torch.bool)
	for label in A:
		# (a MULTI label is a town's doing before its visitor leaves: Nullkiller2's GATHER, config.nk2_gather_weight, the
		# only one taught; alone with END, TRADE and MULTI the hero's leaving without the troops was not pushed down)
		family = hero | town if label == A.END or label in town or label == A.MULTI else hero if label in hero or label == A.CAST else set()
		span[0, label, sorted(family | {label, A.END, A.TRADE, A.MULTI})] = True  # (TRADE and MULTI pushed down at every turn label, as before)
		span[1, label, sorted({A.ARMY, A.END, label} | ({A.MULTI} if exchange_multi else set()))] = True
	return span


# The teacher's cross-entropy is of its entries among what it could have asked for there, not among everything we can do
# (2026-11-05). Match.cpp maps none of Nullkiller2's requests to ARTIFACT, DISMISS or a MULTI (only the accept set's), and
# it casts a few spells only, so every advised decision pushed those toward 0: ARTIFACT 0.020 of the decisions at schema 13
# (v1025) to 0.00002 by v1200, CAST ~0 since v475, though nothing of the teacher's said they were wrong. And its label is
# the first request of its turn: a town action it would make later was pushed down at a hero move's decision (the
# turn's order should not matter: the user, 2026-11-02). So a hero's label (MOVE, TRAVEL, ARMY; a CAST) is weighed
# against the hero's actions and END only, a town action's label (BUILD, RECRUIT, HIRE, UPGRADE, TRADE, BUY) and an END
# against both: a town's actions before its visitor leaves are worth teaching (the troops go with the hero, and
# Nullkiller2 buys first for that), and with the moves out of a town label's span our heroes left towns before buying
# (3 updates of 0d54e0f: RECRUIT 0.119 -> 0.078 and BUILD 0.060 -> 0.042 at decisions in a town of ours, MOVE and TRAVEL
# up). An exchange's against ARMY and END; every accepted entry against itself too. What lies outside gets no gradient
# from the teacher (teacher_outside: its share). MULTI stays in every turn decision's (as TRADE below): out of it, the
# turn's DROP and GATHER, which the teacher had held down, doubled within 15 updates (v1219-v1236: DROP 6 -> 14 a game,
# its disbands destroying ~12k gold of army a game; GATHER 4.8 -> 8.6, the morning's emptied towns); at an exchange
# TAKE_BEST and GIVE_ALL stay outside (52% of exchanges took one, as long as before, no loops).
# TRADE stays in every turn decision's: its labels are not taught (taught: Nullkiller2's trades), and the push at every
# other label is part of what ended 135 trades a game (2026-11-04); left to the family it would be pushed at a town's only.
# A turn's MULTI label (2026-11-09: Nullkiller2's GATHER, Advice.cpp visitAgain; none was taught before) against both
# families as a town action's: our model left the town without the garrison's troops at 92% of those labels.
# TEACHER_SPAN_MULTI (config.exchange_span_multi, 2026-11-09): MULTI in an exchange's as well, the TAKE_BEST its advice
# accepts (into the hero that gets its army) the only one not pushed down
TEACHER_SPAN = _teacher_span()
TEACHER_SPAN_MULTI = _teacher_span(exchange_multi=True)


def teacher_span_logp(out, kind, label, accepted=None, accept_types=None, exchange_multi=False):
	"""[N] the log-probability (now) of what each decision's teacher label is weighed against (TEACHER_SPAN; with
	exchange_multi TEACHER_SPAN_MULTI: True, or [N] the rows it holds for), its accepted entries of other types
	included (accepted [N, K] entries_logp, accept_types [N, K]); 0 (all of it) at a question."""
	from battle_model.device import moved

	exchange, label_type = (kind == c.DecisionKind.EXCHANGE).long(), label.clamp(0, c.N_ACTION_TYPES - 1)
	if torch.is_tensor(exchange_multi):  # (a row's own: config.exchange_span_advised, advised_exchanges)
		span = torch.where(exchange_multi.bool().unsqueeze(1), moved(TEACHER_SPAN_MULTI, kind.device)[exchange, label_type], moved(TEACHER_SPAN, kind.device)[exchange, label_type])
	else:
		span = moved(TEACHER_SPAN_MULTI if exchange_multi else TEACHER_SPAN, kind.device)[exchange, label_type]  # [N, types]
	parts = [torch.logsumexp(out["type_logp"].clamp(min=-1e4).masked_fill(~span, -1e4), dim=1, keepdim=True)]
	if accepted is not None:
		inside = span.gather(1, accept_types.clamp(0, c.N_ACTION_TYPES - 1))  # (counted with their type already)
		parts.append(accepted.masked_fill(inside, -1e4))
	logp = torch.logsumexp(torch.cat(parts, dim=1), dim=1)
	turn = (kind == c.DecisionKind.TURN) | (kind == c.DecisionKind.EXCHANGE)
	return torch.where(turn, logp, torch.zeros_like(logp))


DECODE_THREADS = 3  # batches decoded at once: one thread's ~0.3 s a 128-decision micro-batch (110 MB of zlib and dtype copies, which release the interpreter lock) held the GPU's 146 ms step to it (2026-10-30, the PC)


def _prefetched(batches, decode, device, convert=None, threads=DECODE_THREADS, measure=None):
	"""(rows, inputs on device) of each batch of rows, decoded ahead in `threads` threads (zlib and numpy let the
	training step go on meanwhile) and yielded in order. convert: applied on the device to what decode gave (the wire
	bytes cut and converted there: records.inputs_on_device), else decode gives the inputs themselves. measure: a
	function of what decode gave, on the host in the batch's thread, its result yielded third ((rows, inputs, it)). A
	consumer that stops early (the KL control) ends the threads: they were left blocked on the full queue with their
	batches, every such update (2026-10-30)."""
	import concurrent.futures

	stop = threading.Event()
	pinned = device.type == "cuda"  # (a copy from pageable memory waits for the GPU: ~100 of them a minibatch)

	def work(rows):
		if stop.is_set():
			return None
		decoded = decode(rows)
		measured = () if measure is None else (measure(decoded),)
		if torch.is_tensor(decoded):  # (decode filled a tensor of its own: wire_for's, pinned already, pin_memory() then itself)
			inputs = {"wire": decoded}
		else:
			inputs = {name: torch.from_numpy(value) for name, value in decoded.items()} if isinstance(decoded, dict) else {"wire": torch.from_numpy(decoded)}
		return rows, ({name: value.pin_memory() for name, value in inputs.items()} if pinned else inputs), measured

	pool = concurrent.futures.ThreadPoolExecutor(max_workers=threads)
	pending = collections.deque()
	batches = iter(batches)
	try:
		for rows in itertools.islice(batches, threads + 1):
			pending.append(pool.submit(work, rows))
		while pending:
			item = pending.popleft().result()
			for rows in itertools.islice(batches, 1):
				pending.append(pool.submit(work, rows))
			if item is None:
				break
			rows, inputs, measured = item
			inputs = {name: value.to(device, non_blocking=pinned) for name, value in inputs.items()}
			yield (rows, convert(inputs["wire"]) if convert is not None and "wire" in inputs else inputs, *measured)
	finally:
		stop.set()
		pool.shutdown(wait=False, cancel_futures=True)


def converts_on_device(device):
	"""Whether the update sends the decisions' bytes as recorded to the device and converts them there (CUDA)."""
	return device.type == "cuda"


def wire_for(decisions, device):
	"""decode_wire into memory the copy to `device` reads directly: pinned on CUDA (no staging copy; the caching host
	allocator's buffers, resident and reused, which let the decoding threads scale: records.decode_wire)."""
	out = torch.empty((len(decisions), IN_BYTES), dtype=torch.uint8, pin_memory=device.type == "cuda")
	decode_wire(decisions, out=out.numpy())
	return out


def share_weight(day, side, config):
	"""The schedule credit's share weight [N] at each decision's day: share_weight_start + share_weight_per_day x day,
	the rate scaled by the map's side (share_weight_side / side: a map twice as wide ramps half as fast), within 0..1."""
	rate = config.share_weight_per_day * (config.share_weight_side / side.clamp(min=1.0) if config.share_weight_side > 0 else 1.0)
	return (config.share_weight_start + rate * day).clamp(0.0, 1.0)


def batch_heroes(decoded):
	"""Our heroes in a decoded batch (decode_inputs' or decode_wire's), on the host."""
	return int(heroes_in(decoded).sum())


def sequence_heroes(steps):
	"""batch_heroes for a micro-batch of sequence_order's: steps x the most any one step holds, so that the whole batch's
	window budget holds for each step's share of it (model_v2.forward_steps)."""
	return lambda decoded: steps * int(heroes_in(decoded).reshape(steps, -1).sum(axis=1).max())


def sequence_order(stream, position, steps, size, micro_batches):
	"""A pass's rows as sequences of `steps` of a player's consecutive decisions (config.memory_steps): each run of one
	player's consecutive records (stream [N]: the game and player; position [N]: the record's place among the player's)
	cut into sequences from a random offset, the sequences shuffled and laid out a micro-batch at a time, step-major
	(step 0 of its sequences, then step 1, ...: model_v2.forward_steps' order). Whole minibatches of `size` rows only:
	the rows left over (a run's ends, a last partial minibatch) sit this pass out, another offset the next."""
	n = len(stream)
	by = np.lexsort((position, stream))  # (each player's records in their order)
	s, p = stream[by], position[by]
	new = np.ones(n, dtype=bool)
	new[1:] = (s[1:] != s[:-1]) | (p[1:] != p[:-1] + 1)
	bounds = np.append(np.flatnonzero(new), n)
	starts = np.concatenate([np.zeros(0, dtype=np.int64)] + [np.arange(a + np.random.randint(steps), b - steps + 1, steps) for a, b in zip(bounds[:-1], bounds[1:])])
	per_micro = size // micro_batches // steps
	usable = len(starts) - len(starts) % (per_micro * micro_batches)
	starts = np.random.permutation(starts)[:usable].reshape(-1, 1, per_micro)
	return by[(starts + np.arange(steps).reshape(1, steps, 1)).reshape(-1)]


def windows_for(rows, heroes, config):
	"""The hero windows a step's graph computes for a batch of `rows` decisions holding `heroes` of ours
	(model_v2.windows_of's budget): config.window_budget a decision, or None (every slot) when the batch has more."""
	budget = int(round(rows * config.window_budget))
	return budget if 0 < budget < rows * c.OWN_HEROES and heroes <= budget else None


# The parameter groups (docs/optimizer_plan.md, 2026-11-06), each an Adam group of its own: their gradient norms logged
# (campaign/grad_norm_<group>), each clipped apart (clip_gradients), the value heads' at their own rate
# (config.value_lr_factor), the trunk's, the embeddings' and the action types' bias at the trunk's momentum
# (config.trunk_beta1)
VALUE_MODULES = ("value_hidden", "value_heads", "growth_heads", "end_head")
# (the privileged critic its own group, last: the clips calibrated before it hold no clip for it, and its gradient
# never moves another group's clip; at the value heads' rate)
GRAD_GROUPS = ("trunk", "embeddings", "type", "value", "corrections", "privileged")
TRUNK_GROUPS = ("trunk", "embeddings", "type")


def parameter_groups(model):
	"""{group: [(name, parameter)]} over GRAD_GROUPS, each parameter in one, in the model's order: the prior corrections
	(model.prior_corrections), the value heads' (VALUE_MODULES), every embedding table's (nn.Embedding), the action
	types' bias (type_bias), the trunk the rest. A group may be empty (v1 has no value_hidden)."""
	corrections = {id(p) for p in model.prior_corrections()}
	value = {id(p) for name in VALUE_MODULES if isinstance(getattr(model, name, None), torch.nn.Module) for p in getattr(model, name).parameters()}
	privileged = {id(p) for p in model.privileged.parameters()} if isinstance(getattr(model, "privileged", None), torch.nn.Module) else set()
	embeddings = {id(m.weight) for m in model.modules() if isinstance(m, torch.nn.Embedding)}
	groups = {name: [] for name in GRAD_GROUPS}
	for name, p in model.named_parameters():
		group = "corrections" if id(p) in corrections else "privileged" if id(p) in privileged else "value" if id(p) in value else "embeddings" if id(p) in embeddings else "type" if name == "type_bias" else "trunk"
		groups[group].append((name, p))
	return groups


def group_lr(group, config, lr_scale=1.0):
	"""An Adam group's rate: learning_rate x the KL control's lr_scale, the value heads' x value_lr_factor (and without
	lr_scale unless config.value_lr_kl)."""
	if group.get("name") in ("value", "privileged"):
		return config.learning_rate * config.value_lr_factor * (lr_scale if config.value_lr_kl else 1.0)
	return config.learning_rate * lr_scale


def group_settings(name, config):
	"""A group's hyperparameters from the config (load_optimizer sets them again after a checkpoint's: its saved ones
	would win): weight decay on the corrections alone, the trunk's beta1 for TRUNK_GROUPS."""
	return {"name": name, "weight_decay": config.prior_decay if name == "corrections" else 0.0, "betas": (config.trunk_beta1 if name in TRUNK_GROUPS else 0.9, 0.999)}


def make_optimizer(model, config=CampaignConfig()):
	"""AdamW over parameter_groups' groups, each named (its "name") with its parameters' names (torch's "param_names", which
	the checkpoint keeps: load_optimizer matches moments by them). No weight decay but on the corrections of the model's
	fixed priors (model.prior_corrections: the object values' gains and deltas), decayed by config.prior_decay toward 0,
	the prior (the user, 2026-10-28: the regularization penalizes the change from Nullkiller2's values). Adam moves a
	weight ~lr a step when its gradient is consistent and the decay pulls lr x prior_decay x weight back, so a
	correction's weight settles near 1 / prior_decay under a steady push (its effect: CORRECTION_STEP times that) and
	near 0 under noise."""
	groups = [{"params": params, **group_settings(name, config)} for name, params in parameter_groups(model).items() if params]
	optimizer = torch.optim.AdamW(groups, lr=config.learning_rate)
	for group in optimizer.param_groups:
		group["lr"] = group_lr(group, config)
	return optimizer


def saved_parameter_names(saved, model, saved_keys):
	"""The names of a saved optimizer's parameters, in its numbering (its groups' order). Its groups' "param_names" since
	2026-11-06; before, two groups: the rest as the checkpoint's state_dict lists them but its buffers, the corrections
	as ours (prior_corrections' list: a new one takes its place in it)."""
	if all("param_names" in group for group in saved["param_groups"]):
		if any(len(group["param_names"]) != len(group["params"]) for group in saved["param_groups"]):
			raise ValueError("the checkpoint's optimizer groups name a different number of parameters than they hold")
		return [name for group in saved["param_groups"] for name in group["param_names"]]
	names = {id(p): name for name, p in model.named_parameters()}
	corrections = [names[id(p)] for p in model.prior_corrections()]
	present, buffers, skip = set(saved_keys), {name for name, _ in model.named_buffers()}, set(corrections)
	orders = [[key for key in saved_keys if key not in buffers and key not in skip], [name for name in corrections if name in present]]
	if len(saved["param_groups"]) != len(orders):
		raise ValueError(f"the checkpoint's optimizer has {len(saved['param_groups'])} groups and no parameter names (2 expected)")
	for order, group in zip(orders, saved["param_groups"]):
		if len(order) != len(group["params"]):
			raise ValueError(f"the checkpoint's optimizer group has {len(group['params'])} parameters, its model {len(order)}")
	return [name for order in orders for name in order]


def migrate_subtypes(state, model):
	"""A checkpoint (state: its "model" and "optimizer") from before the subtype tables changed (2026-10-31: SUBTYPE_HASH
	rows each), for model, in place; returns the tables migrated.
	- The (object, subtype) pairs' tables (PAIR_ROWS: constants.OBJECT_PAIRS): each pair's row is its old hashed one,
	  exactly, Adam's moments too; PAIR_OTHER and PAIR_BLOCKED are the fresh model's
	- The options' (SUBTYPE_ROWS, still hashed): a pair's row is its old one's modulo SUBTYPE_BUCKETS (features.bucket),
	  each new row the mean of the old rows folded into it that a gradient ever reached (Adam's second moment above 0;
	  every row without the optimizer's state), its moments alike; a row none reached keeps the fresh model's"""
	own = model.state_dict()
	names = [n for n, v in state["model"].items() if n in own and v.dim() == 2 and v.shape[0] == c.SUBTYPE_HASH and own[n].shape[0] in (c.SUBTYPE_ROWS, c.PAIR_ROWS)]
	if not names:
		return []
	saved = state.get("optimizer")
	index = {}  # a parameter's key in the optimizer's state: its place in the saved optimizer's groups
	if saved:
		flat = [i for group in saved["param_groups"] for i in group["params"]]
		try:
			order = saved_parameter_names(saved, model, state["model"].keys())
		except ValueError:
			order = []
		if len(flat) == len(order):
			index = dict(zip(order, flat))
	pairs = list(c.OBJECT_PAIRS.items())
	for name in names:
		old = state["model"][name]
		device = old.device
		moments = saved["state"].get(index[name]) if name in index else None
		if own[name].shape[0] == c.PAIR_ROWS:  # (bucket(obj, obj, subtype) before: the pair's hashed row)
			rows = torch.tensor([row for _, row in pairs], device=device)
			hashed = torch.tensor([((o + 1) * 7919 + (o + 1) * 131 + t * 7) % c.SUBTYPE_HASH for (o, t), _ in pairs], device=device)

			def moved(table, fresh):
				out = fresh.to(device=device, dtype=table.dtype).clone()
				out[rows] = table[hashed]
				return out
		else:
			live = moments["exp_avg_sq"].sum(dim=1) > 0 if moments is not None else torch.ones(c.SUBTYPE_HASH, dtype=torch.bool, device=device)
			fold = (torch.arange(c.SUBTYPE_HASH, device=device) % c.SUBTYPE_BUCKETS)[live]
			count = torch.zeros(c.SUBTYPE_ROWS, device=device).index_add_(0, fold, torch.ones(len(fold), device=device)).unsqueeze(1)

			def moved(table, fresh):
				total = torch.zeros(c.SUBTYPE_ROWS, table.shape[1], device=device).index_add_(0, fold, table[live].float())
				return torch.where(count > 0, total / count.clamp(min=1), fresh.to(device).float()).to(table.dtype)

		state["model"][name] = moved(old, own[name])
		if moments is not None:
			for key in ("exp_avg", "exp_avg_sq"):
				moments[key] = moved(moments[key], torch.zeros_like(own[name]))
	return names


def load_optimizer(optimizer, model, saved, saved_keys):
	"""The optimizer's state from a checkpoint (saved: its state_dict; saved_keys: its model's state_dict keys, in their
	order). Adam's moments follow their parameters by name (saved_parameter_names) into whatever group make_optimizer
	puts them now (2026-11-06: two groups became GRAD_GROUPS' five): a model that has gained parameters since (schema 7:
	hire_worth) starts the new ones fresh, one that has lost some (2026-11-04: the question head's option_q,
	answer_fight, answer_join) drops theirs. The groups' hyperparameters stay the optimizer's, the config's (the
	checkpoint's would win in load_state_dict: a changed prior_decay or beta1 never applied to a resumed run)."""
	saved_names = saved_parameter_names(saved, model, list(saved_keys))
	fresh, gone = moments_by_name(optimizer, model, saved, saved_names)
	fit_moments(optimizer, model)
	moved = sorted({group.get("name", "?") for group in optimizer.param_groups} - {group.get("name", "?") for group in saved["param_groups"]})
	if fresh or gone or moved:
		print(f"campaign optimizer: Adam's moments kept by name{' (new groups: ' + ', '.join(moved) + ')' if moved else ''}, fresh for the new parameters: {', '.join(fresh) or 'none'}; dropped for the gone: {', '.join(gone) or 'none'}", flush=True)


def fit_moments(optimizer, model):
	"""Adam's moments loaded for a parameter whose shape has changed since (optimizer.load_state_dict does not check, and
	the first step would fail): move_pair's first layer from before 2026-11-04's MOVE features moved with its weights
	(model_v2.move_pair_columns), any other started fresh (logged)."""
	from .model_v2 import move_pair_columns

	names = {id(p): name for name, p in model.named_parameters()}
	for group in optimizer.param_groups:
		for p in group["params"]:
			state = optimizer.state.get(p)
			if not state or "exp_avg" not in state or state["exp_avg"].shape == p.shape:
				continue
			name = names.get(id(p), "?")
			if name == "move_pair.up.weight" and move_pair_columns(state["exp_avg"], fold=False).shape == p.shape:
				for key in ("exp_avg", "exp_avg_sq"):
					state[key] = move_pair_columns(state[key], fold=False)
				print(f"campaign optimizer: {name}'s moments moved to its inputs now (model_v2.move_pair_columns)", flush=True)
			else:
				del optimizer.state[p]
				print(f"campaign optimizer: {name}'s moments did not fit its shape {tuple(p.shape)}: fresh", flush=True)


def value_fit(predicted, actual):
	"""The 7-day head's fit, both off the share prior: the correlation of its forecasts with what then happened, and each
	one's spread (std): a negative explained variance is noise (correlation ~0) or overconfidence (correlation > 0,
	the forecasts' spread above the correlation x the outcomes')."""
	ok = np.isfinite(actual) & np.isfinite(predicted)
	if ok.sum() < 3:
		return {"value_corr": float("nan"), "value_spread": float("nan"), "label_spread": float("nan")}
	p, a = predicted[ok], actual[ok]
	return {"value_corr": float(np.corrcoef(p, a)[0, 1]) if p.std() > 0 and a.std() > 0 else 0.0, "value_spread": float(p.std()), "label_spread": float(a.std())}


def gpu_held(device, release=False):
	"""On Apple's GPU: the memory the process holds there (MB); release: its cached blocks freed after. The batches'
	shapes vary (windows only for the heroes present) and MPS keeps a block of every size it saw: the update held ~8 GB
	more than it used (2026-10-28), next to the battle learner's, and the machine swapped. Freeing them every step
	doubled the update's time; the update process sets MPS's low watermark instead (learner.MPS_LOW_WATERMARK), so the
	allocator frees them itself when it holds too much."""
	if device.type == "cuda":  # the update's peak of live tensors (the card shows the allocator's reserve, which keeps the peak)
		held = torch.cuda.max_memory_allocated(device) / 2**20
		if release:
			torch.cuda.reset_peak_memory_stats(device)
			torch.cuda.empty_cache()
		return held
	if device.type != "mps":
		return 0.0
	held = torch.mps.driver_allocated_memory() / 2**20
	if release:
		torch.mps.empty_cache()
	return held


# The KL control judges the update's KL by the mean of its last minibatches' (k3 estimates): one minibatch's is noise
# (2026-10-28 review: 15% of v15-window's updates stopped early, 99% of them with a mean KL under half the stop line)
KL_WINDOW = 4


def head_loss(logits, targets, valid, count=None):
	"""[heads] each head's cross-entropy against its soft targets, over the decisions that have one (valid): its sum over
	them / their count (count: the whole minibatch's, for one micro-batch of it)."""
	loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * valid
	return loss.sum(dim=0) / (valid.sum(dim=0) if count is None else count).clamp(min=1)


HEAD_SCALE_EPS = 1e-4  # (config.head_scale) a head's target variance floor, on its 0..1 scale (a spread of 0.01)
HEAD_SCALE_RAMP = 20  # updates over which the heads' weights go from 1 (and growth_weight) to head_weights' (Adam's second moments adapt over ~1,000 steps)
HEAD_SCALE_MEMORY = 0.8  # the running variance's share of the old estimate at each update
# (config.head_scale_by "residual") each head's fresh excess loss, averaged at this memory an update (the plan's 0.9),
# floored at HEAD_RESIDUAL_EPS (the share-change heads' ~5e-4); the variances that scale the weights to the old total
# at a slower memory: at 0.8 the total, V_end's spread most of it, swung every head's weight with the batch
HEAD_RESIDUAL_MEMORY = 0.9
HEAD_RESIDUAL_EPS = 1e-5
HEAD_SCALE_SLOW_MEMORY = 0.98


def head_weights(variances, ramp, growth_weight, eps=HEAD_SCALE_EPS, residuals=None):
	"""(config.head_scale) Each value head's weight in the value loss, {"share": [V2, V5, V7], "growth": [G2, G5, G7],
	"end": [Vend]}: inversely to its targets' variance (each head's loss, a constant prediction's, equally large), scaled
	so the heads' losses at a constant prediction add up to what they did at the old weights (1, growth_weight, 1): the
	value loss keeps its size against the policy's and the teacher's, only its split between the heads changes; and ramp
	(0..1) of the way from the old weights. variances: the same layout (a missing head keeps its old weight).
	2026-11-02: V_end's targets spread 0.097 on the 0..1 scale and the share-change heads' 0.010-0.017, V_end's loss at
	a constant prediction 35-100x theirs: the share-change heads were near-constant after a day.
	residuals (config.head_scale_by "residual", docs/optimizer_plan.md step 7): inversely to each head's running fresh
	excess loss instead, its residual (a head learning well gets more of the shared layers, one whose residual stays
	large, the long horizons and V_end, less), scaled the same way."""
	old = {"share": [1.0] * c.N_VALUE_OFFSETS, "growth": [growth_weight] * c.N_VALUE_OFFSETS, "end": [1.0]}
	known = lambda table, name, k: table is not None and table.get(name) is not None and k < len(table[name]) and table[name][k] == table[name][k]  # noqa: E731 (nan: unknown)
	pairs = [(name, k) for name in old for k in range(len(old[name])) if known(variances, name, k) and (residuals is None or known(residuals, name, k))]
	if not pairs:
		return old
	baseline = sum(old[name][k] * variances[name][k] for name, k in pairs)
	raw = {(name, k): old[name][k] / (variances[name][k] + eps if residuals is None else max(residuals[name][k], HEAD_RESIDUAL_EPS)) for name, k in pairs}
	scale = baseline / max(sum(raw[p] * variances[p[0]][p[1]] for p in pairs), 1e-12)
	out = {name: list(values) for name, values in old.items()}
	for name, k in pairs:
		out[name][k] = (1 - ramp) * old[name][k] + ramp * scale * raw[(name, k)]
	return out


def running(old, measured, memory):
	"""{name: [values]}: measured blended into old at memory, list by list (a name old lacks, or has at another length,
	starts at measured; a value not measured, nan, keeps the old one)."""
	old = old or {}
	blend = lambda o, m: o if m != m else memory * o + (1 - memory) * m  # noqa: E731
	return {name: [blend(o, m) for o, m in zip(old[name], values)] if old.get(name) is not None and len(old[name]) == len(values) else list(values) for name, values in measured.items()}


def value_readout_rms(model):
	"""The value heads' readouts' root mean square (value_heads', growth_heads', end_head's weights): they start at 0, so
	how far the critic has come (docs/optimizer_plan.md step 1: 0.006-0.012 after 1,300 updates at the policy's rate)."""
	weights = [getattr(model, name).weight for name in ("value_heads", "growth_heads", "end_head") if isinstance(getattr(model, name, None), torch.nn.Linear)]
	if not weights:
		return float("nan")
	with torch.no_grad():
		return float(torch.cat([w.detach().flatten().float() for w in weights]).pow(2).mean().sqrt())


def masked_variance(t, valid):
	"""[heads] each column's variance over its valid rows (0 with fewer than 2)."""
	n = valid.sum(dim=0)
	mean = (t * valid).sum(dim=0) / n.clamp(min=1)
	return torch.where(n > 1, (((t - mean) ** 2) * valid).sum(dim=0) / n.clamp(min=1), torch.zeros_like(n))


def excess_loss(logits, targets):
	"""Each head's cross-entropy against its soft targets above the targets' own entropy (its floor): 0 when the head
	predicts its target exactly."""
	floor = -(torch.special.xlogy(targets, targets) + torch.special.xlogy(1 - targets, 1 - targets))
	return F.binary_cross_entropy_with_logits(logits, targets, reduction="none") - floor


# A step's metrics, in ppo_update's order: value the heads' loss at their own weights (comparable whatever head_scale
# does), value_weighted at head_scale's (what the update takes)
STEP_PARTS = ("policy", "value", "entropy", "below_floor", "clip_fraction", "kl", "teacher_kl", "teacher_ce", "value_weighted", "teacher_outside", "value_privileged")
# The teacher's outputs its KL reads (teacher_kl), put in each micro-batch as "t:<name>"
TEACHER_OUTPUTS = ("type_logp", "option_logp") + tuple(output for output, _ in HEADS.values())


def teacher_kl(out, batch):
	"""KL(teacher || model) of each decision [N] (config.teacher_kl): a turn decision's, over the action type plus each
	type's table weighted by the teacher's p(type); a question's whose answers the teacher learned (records.
	TEACHER_QUESTIONS: yes/no, selection, level up), over its options; 0 for the others (exchanges, the other
	questions: there the teacher's outputs are its untrained start, which the KL would pull the answers to). An entry
	the teacher gives no probability adds nothing (both models see the same legal masks); the model's
	log-probabilities are floored at -50 there."""
	def part(t, s):
		p = t.exp()
		return torch.where(p > 0, p * (t - s.clamp(min=-50.0)), torch.zeros_like(p)).sum(dim=-1)

	kind = batch["kind"]
	n = kind.shape[0]
	t_type = batch["t:type_logp"]
	kl = part(t_type, out["type_logp"])
	p_type = t_type.exp()
	for a, (output, _) in HEADS.items():
		kl = kl + p_type[:, a] * part(batch["t:" + output].reshape(n, -1), out[output].reshape(n, -1))
	answered = (kind == c.DecisionKind.YES_NO) | (kind == c.DecisionKind.SELECT) | (kind == c.DecisionKind.LEVEL_UP)  # (records.TEACHER_QUESTIONS)
	return torch.where(kind == c.DecisionKind.TURN, kl, torch.where(answered, part(batch["t:option_logp"], out["option_logp"]), torch.zeros_like(kl)))


def signal_ratio(optimizer):
	"""The median over the weights that have had a gradient of |m| / sqrt(v) in Adam's state, bias-corrected (1: the
	steps' gradients agree, 0: they cancel out; battle_model.train.signal_ratio's, without the ~half of this model's
	weights no gradient reaches, embedding rows of ids never seen, which held its median at 0)."""
	ratios = []
	for group in optimizer.param_groups:
		if "betas" not in group:  # (not Adam: no such state)
			continue
		beta1, beta2 = group["betas"]
		for p in group["params"]:
			state = optimizer.state.get(p, {})
			if "exp_avg" in state:
				step = float(state["step"])
				v = state["exp_avg_sq"].flatten() / (1 - beta2**step)
				live = v > 0
				ratios.append((state["exp_avg"].flatten()[live] / (1 - beta1**step)).abs() / v[live].sqrt())
	return float(torch.cat(ratios).median()) if ratios else float("nan")


NOISE_PHASES = (("days 1-14", 1, 14), ("days 15-35", 15, 35), ("days 36+", 36, 10**6))  # loss_noise's, by the decisions' day


def standardized(values):
	"""values [N] less their mean over the finite ones, over their standard deviation; nan stays nan."""
	known = np.isfinite(values)
	if known.sum() < 2:
		return values
	return ((values - values[known].mean()) / (values[known].std() + 1e-8)).astype(np.float32)


def loss_noise(model, decisions, decode, convert, device, t, config, rows=128, samples=12):
	"""The gradient noise scale of each loss term, by the decisions' day: tr(cov) / |mean|^2 of the term's gradient over
	`samples` micro-batches of `rows` decisions (McCandlish et al. 2018's B_simple; |mean|^2 less its bias, tr(cov) / samples),
	times rows: the batch, in decisions, at which the term's gradient noise equals its signal; inf where no signal shows.
	The user's question (2026-10-31): our empire value's growth against our share of all, early and late. Terms:
	  policy         the clipped surrogate with the update's advantage (the credit the policy follows)
	  policy_share   the same with the 7-day share's Monte Carlo advantage: our share at the end of day d + 7 - V7
	  policy_growth  the same with our value's 7-day growth's: its log growth (the growth heads' 0..1 scale) - G7
	  value_share, value_growth, value_end   the value heads' losses
	t: the update's tensors (choice, advantage, credited, unsound, targets, share_valid; growth, growth_valid; end, end_valid;
	labels: the real shares; growth_real: the real log values). The weights are left as they were (torch.autograd.grad)."""
	params = [p for p in model.parameters() if p.requires_grad]
	day = np.array([d.day for d in decisions])
	result = {}
	for phase, first, last in NOISE_PHASES:
		pool = np.nonzero((day >= first) & (day <= last))[0]
		if len(pool) < 2 * rows:
			continue
		rng = np.random.default_rng(len(pool))
		sums, squares, n = {}, {}, 0
		for _ in range(samples):
			chunk = np.sort(rng.choice(pool, rows, replace=False))
			raw = decode(chunk)
			inputs = {name: torch.from_numpy(value).to(device) for name, value in raw.items()} if isinstance(raw, dict) else convert(torch.as_tensor(raw).to(device))  # (as_tensor: decode's wire is wire_for's tensor, or an array)
			sub = torch.from_numpy(chunk).to(device)
			keep = (~t["unsound"][sub]).float()
			out = model(inputs)
			old = t["choice"]["old_logp"][sub]
			logp, _ = chosen_logp(out, {k: t["choice"][k][sub] for k in ("kind", "action", "index", "cell")})
			ratio = (torch.where(keep > 0, logp, old) - old).exp()

			def surrogate(a, valid):
				w = keep * valid
				return -(torch.minimum(ratio * a, ratio.clamp(1 - config.clip, 1 + config.clip) * a) * w).sum() / w.sum().clamp(min=1)

			now = out["phi"].detach().float().clamp(min=PHI_FLOOR).log()
			share7, real7 = t["labels"][sub, -1], t["growth_real"][sub, -1]
			grown7 = ((real7 - now).clamp(-GROWTH_CAP, GROWTH_CAP) + GROWTH_CAP) / (2 * GROWTH_CAP)
			terms = {
				"policy": surrogate(t["advantage"][sub], t["credited"][sub]),
				"policy_share": surrogate((share7 - torch.sigmoid(out["value_logit"][:, -1].float()).detach()).nan_to_num(0.0), torch.isfinite(share7).float()),
				"policy_growth": surrogate((grown7 - torch.sigmoid(out["growth_logit"][:, -1].float()).detach()).nan_to_num(0.0), torch.isfinite(real7).float()),
				"value_share": head_loss(out["value_logit"], t["targets"][sub], t["share_valid"][sub]).sum(),
			}
			if "growth" in t:
				scaled = ((t["growth"][sub] - now.unsqueeze(-1)).clamp(-GROWTH_CAP, GROWTH_CAP) + GROWTH_CAP) / (2 * GROWTH_CAP)
				terms["value_growth"] = head_loss(out["growth_logit"], scaled, t["growth_valid"][sub]).sum()
			if "end" in t:
				terms["value_end"] = head_loss(out["end_logit"].unsqueeze(-1), t["end"][sub].unsqueeze(-1), t["end_valid"][sub].unsqueeze(-1)).sum()
			for name, term in terms.items():
				grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
				flat = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten().float() for g, p in zip(grads, params)])
				sums[name] = flat if name not in sums else sums[name] + flat
				squares[name] = flat * flat if name not in squares else squares[name] + flat * flat
			n += 1
			del out, terms, logp, ratio
		for name in sums:
			mean = sums[name] / n
			trace = float(((squares[name] / n - mean * mean).sum()) * n / (n - 1))
			signal = float((mean * mean).sum()) - trace / n
			result[f"{name} {phase}"] = trace / signal * rows if signal > 0 else float("inf")
	return result


def micro_loss(out, batch, config, foresight_config=None, split=False, privileged=None):
	"""A micro-batch's loss and metrics from the model's outputs and its rows, for the eager step and the recorded one
	alike (graphs.StepGraph): (loss, STEP_PARTS [len], the foresight's parts [len(PART_NAMES)] or None, the foresight's
	share of the loss or None); split: and the loss's terms ({name: tensor}, their sum the loss: gradient_check's).
	batch: tensors on the device, each a row of the micro-batch (kind, action,
	index, cell, old_logp, keep, advantage, credited, targets, share_valid, anchor; growth, growth_valid; end, end_valid;
	"fs:<label>": the foresight's), and its minibatch's counts (count_credit, count_keep, count_share, count_growth,
	count_end) and its share of the minibatch ("share"): the micro-batches' gradients add up to the minibatch's.
	privileged: the model's PrivilegedCritic, trained here on the batch's rivals ("priv_*", train.ppo_update's) at the
	share heads' and V_end's own losses, every input detached (its loss reaches only its own weights); logged as
	value_privileged (those heads' mean)."""
	from .foresight import PART_NAMES, ForesightConfig, foresight_loss

	keep = batch["keep"]
	logp, entropy = chosen_logp(out, {k: batch[k] for k in ("kind", "action", "index", "cell")})
	logp = torch.where(keep > 0, logp, batch["old_logp"])
	ratio = (logp - batch["old_logp"]).exp()
	a, credit = batch["advantage"], keep * batch["credited"]
	policy = -(torch.minimum(ratio * a, ratio.clamp(1 - config.clip, 1 + config.clip) * a) * credit).sum() / batch["count_credit"]
	# (config.head_scale: each head at its weight, "w:share" "w:growth" "w:end", train.head_weights; the denominator stays the
	# old weights' sum, the weights keeping the heads' total)
	value, heads = head_loss(out["value_logit"], batch["targets"], batch["share_valid"], batch["count_share"]), float(c.N_VALUE_OFFSETS)
	own = value.sum()  # (each head at its own weight: the value_loss logged, comparable whatever head_scale does)
	value = (value * batch["w:share"]).sum() if "w:share" in batch else own
	if "growth" in batch:  # our value's log growth from the decision, on the heads' 0..1 scale
		now = out["phi"].detach().clamp(min=PHI_FLOOR).log().unsqueeze(-1)
		scaled = ((batch["growth"] - now).clamp(-GROWTH_CAP, GROWTH_CAP) + GROWTH_CAP) / (2 * GROWTH_CAP)
		grown = head_loss(out["growth_logit"], scaled, batch["growth_valid"], batch["count_growth"])
		own = own + config.growth_weight * grown.sum()
		value = value + ((grown * batch["w:growth"]).sum() if "w:growth" in batch else config.growth_weight * grown.sum())
		heads += config.growth_weight * c.N_VALUE_OFFSETS
	if "end" in batch:  # V_end, at a share head's weight
		ended = head_loss(out["end_logit"].unsqueeze(-1), batch["end"].unsqueeze(-1), batch["end_valid"].unsqueeze(-1), batch["count_end"].unsqueeze(-1))
		own = own + ended.sum()
		value = value + ((ended * batch["w:end"]).sum() if "w:end" in batch else ended.sum())
		heads += 1.0
	value, own = value / heads, own / heads
	privileged_part = torch.zeros((), device=keep.device)
	if privileged is not None and "priv_rivals" in batch:  # (the privileged critic: on top of the critic, detached)
		corr = privileged(out["value_hidden"].detach(), batch["priv_rivals"], batch["priv_mask"], batch["priv_share"])
		k = c.N_VALUE_OFFSETS
		privileged_part = head_loss(out["value_logit"].detach() + corr[:, :k], batch["targets"], batch["share_valid"], batch["count_share"]).sum()
		if "end" in batch:
			privileged_part = privileged_part + head_loss((out["end_logit"].detach() + corr[:, k]).unsqueeze(-1), batch["end"].unsqueeze(-1), batch["end_valid"].unsqueeze(-1), batch["count_end"].unsqueeze(-1)).sum()
		privileged_part = privileged_part / (k + (1 if "end" in batch else 0))  # (its heads' mean: the share heads and V_end)
	bonus = (entropy * keep).sum() / batch["count_keep"]
	floor = (torch.relu(config.entropy_floor - entropy) * keep).sum() / batch["count_keep"]  # only decisions below it
	# (the value's and the foresight's weights from the learner's loss shares, "lw:value" "lw:foresight"; 1 without)
	loss = policy + config.value_weight * value * batch.get("lw:value", 1.0) - config.entropy_weight * bonus + config.entropy_floor_weight * floor + privileged_part
	teacher_part = torch.zeros((), device=keep.device)
	if "t:type_logp" in batch:  # the teacher's KL (ppo_update's teacher), at its weight (a tensor: a recorded step reads it)
		teacher_part = (teacher_kl(out, batch) * keep).sum() / batch["count_keep"]
		loss = loss + batch["teacher_weight"] * teacher_part
	advice_part = torch.zeros((), device=keep.device)
	outside_part = torch.zeros((), device=keep.device)
	if "advice_action" in batch:  # Nullkiller2's advice (config.nk2_advice): its choice's cross-entropy, at the teacher's weight
		advised = batch["advised"] * keep
		# (schema 16's: a whole stack legal only as a swap, new since; Advice.cpp's nearest move took it for a hero's last
		# stack, which Nullkiller2 never moves whole, and taught two heroes to swap their stacks back and forth)
		swapping = None if config.nk2_new_advice else swap_only_moves(out, batch["advice_action"], batch["advice_index"])
		if swapping is not None:
			advised = advised * (~swapping).float()
		advice_logp, _ = chosen_logp(out, {"kind": batch["kind"], "action": batch["advice_action"], "index": batch["advice_index"], "cell": batch["advice_cell"]})
		accepted = None
		if "accept_types" in batch:  # (and every other entry that follows it there, advice.teacher_sets: their summed probability)
			advised_own = torch.where(batch["advice_has"], advice_logp.clamp(min=-1e4), torch.full_like(advice_logp, -1e4)).unsqueeze(1)
			valid = batch["accept_valid"] if config.nk2_new_advice else batch["accept_valid"] & ~swap_only_moves(out, batch["accept_types"], batch["accept_index"])
			accepted = entries_logp(out, batch["accept_types"], batch["accept_index"], valid)
			advice_logp = torch.logsumexp(torch.cat([advised_own, accepted], dim=1), dim=1)
			advised = advised * (advice_logp > -1e3).float()  # (a row none of whose entries is legal any more: none)
		# among what it could have asked for there (TEACHER_SPAN), not among everything we can do
		span_logp = teacher_span_logp(out, batch["kind"], batch["advice_action"], accepted, batch.get("accept_types"), batch["exchange_multi"] if "exchange_multi" in batch else config.exchange_span_multi)
		advice_logp = advice_logp - span_logp
		advice_part = -(torch.where(advised > 0, advice_logp, torch.zeros_like(advice_logp)) * advised).sum() / batch["count_advised"]
		outside_part = ((1.0 - span_logp.detach().exp()) * advised).sum() / batch["count_advised"]
		loss = loss + batch["teacher_weight"] * advice_part
	fs_parts = fs_share = None
	labels = {name[3:]: value for name, value in batch.items() if name.startswith("fs:")}
	if labels and "fs_turn" in out:
		fs_config = foresight_config or ForesightConfig()
		fs_loss, parts = foresight_loss(out, labels, fs_config)
		fs_share = batch["share"] * fs_loss * batch.get("lw:foresight", 1.0)
		loss = loss + fs_share
		fs_parts = torch.stack([parts[name].detach().float() for name in PART_NAMES])
	drift = logp.detach() - batch["anchor"]
	metrics = torch.stack([
		policy.detach(),
		own.detach(),
		bonus.detach(),
		((entropy < config.entropy_floor).float() * keep).sum() / batch["count_keep"],  # below_floor
		(((ratio - 1).abs() > config.clip).float() * keep).sum() / batch["count_keep"],  # clip_fraction
		(((drift.exp() - 1) - drift) * keep).sum() / batch["count_keep"],  # kl: k3, before this step
		teacher_part.detach(),  # teacher_kl
		advice_part.detach(),  # teacher_ce
		value.detach(),  # value_weighted
		outside_part,  # teacher_outside: our probability outside what the teacher's label is weighed against
		privileged_part.detach(),  # value_privileged: the privileged critic's loss, its heads' mean at their own weights
	]).float()
	if split:  # (each term at weight 1 against the learner's loss shares, the loss_weights it sets: the plan's apart too;
		# the privileged critic's not among them: it reaches its own weights only)
		terms = {
			"policy": policy, "value": config.value_weight * value,
			"entropy": config.entropy_floor_weight * floor - config.entropy_weight * bonus, "teacher": teacher_part + advice_part,
		}
		if fs_share is not None:
			terms["foresight"], terms["plan"] = batch["share"] * fs_loss, batch["share"] * fs_config.weight * parts["plan"]
		return loss, metrics, fs_parts, fs_share, terms
	return loss, metrics, fs_parts, fs_share


def share_baseline(real_change, predicted, trust):
	"""The schedule credit's baseline for the share's real change: the V7 change head's prediction as far as it is
	trusted (0..1: its explained variance's, CampaignLearner.trust), the batch's mean change for the rest. joint7's head
	was worse than a constant through version 55 (explained variance -0.02 to -0.12), and from day 45 the credit is all
	this term (share_weight_per_day): the head's error was most of the advantage there. A baseline does not change what
	the policy is credited for, only the advantage's variance."""
	known = torch.isfinite(real_change)
	mean = real_change[known].mean() if int(known.sum()) else torch.zeros((), device=real_change.device)
	return trust * predicted + (1.0 - trust) * mean


STEP_CLASS_PRIOR = 50.0  # rows at 0 added to each class's mean rest (step_credit): a class of few rows is barely corrected


def step_classes(kind, action):
	"""[N] each decision's class for the step credit's correction (step_credit): its action type, an exchange's apart
	from a turn's (a MULTI there is another thing); every question the class 0."""
	return (action + 1).clamp(min=0) + (kind == c.DecisionKind.EXCHANGE).long() * (c.N_ACTION_TYPES + 1)


def forecast_trust(real, forecast, full_ev):
	"""0..1: how far a forecast [N] of the real outcome [N] is trusted for its step (step_credit's trust): its explained
	variance on the rows where both are known, over full_ev (config.trust_full_ev). A fit, not a slope: the baseline's
	coefficient (share_baseline's trust) says how to scale a forecast, not how good its steps are."""
	known = torch.isfinite(real) & torch.isfinite(forecast)
	if int(known.sum()) < 2:
		return 0.0
	spread = float(real[known].var())
	if spread < 1e-12:
		return 0.0
	explained = 1.0 - float((real[known] - forecast[known]).pow(2).mean()) / spread
	return float(min(1.0, max(0.0, explained / full_ev)))


def forecast_steps(forecast, step, day):
	"""[N] the forecast at the player's next turn decision that day (step [N]: its row, -1 none) less the forecast here;
	nan where there is none that day."""
	j = step.clamp(min=0)
	same = (step >= 0) & (day[j] == day)
	return torch.where(same, forecast[j] - forecast, torch.full_like(forecast, float("nan")))


# The classes whose mean own step an update reports (schedule_stats' step_<name>: TensorBoard campaign_step/<name>): what
# the step credit says of each action type before the week's outcome corrects it (step_credit)
STEP_REPORT = (("move", 0, c.ActionType.MOVE), ("travel", 0, c.ActionType.TRAVEL), ("build", 0, c.ActionType.BUILD), ("recruit", 0, c.ActionType.RECRUIT), ("upgrade", 0, c.ActionType.UPGRADE),
	("hire", 0, c.ActionType.HIRE), ("trade", 0, c.ActionType.TRADE), ("multi", 0, c.ActionType.MULTI), ("exchange_multi", 1, c.ActionType.MULTI), ("exchange_army", 1, c.ActionType.ARMY))


def step_report(moved, kind, action):
	"""{step_<name>: the mean own step of STEP_REPORT's classes with a row (floats)}: moved [N] forecast_steps'."""
	out = {}
	for name, exchange, kind_of in STEP_REPORT:
		rows = torch.isfinite(moved) & (action == int(kind_of)) & ((kind == c.DecisionKind.EXCHANGE) == bool(exchange))
		if int(rows.sum()):
			out[f"step_{name}"] = float(moved[rows].mean())
	return out


def step_credit(term, forecast, step, day, trust, classes=None, price=None, outcome_share_min=0.25):
	"""SET ASIDE (the user, 2026-11-09: "I do not want any value on actual steps, only on day+2/5/7", until its effect
	can be measured apart): nothing calls it, ppo_update's call is commented out. What gave pause: it credits what a
	decision does at once, so a purchase paid with traded resources starts from a loss in our value, and the price was
	felt too high (2e-4 is ~21 gold a decision at the 8 joint7 games' median value, 0.27% of our value a player-day).
	Measured there for when it is tried (scratchpad audit/f1.py, f2.py): a MOVE collecting 250+ gold +0.0209 (97%
	positive), a MOVE changing nothing -0.0007; a BUILD paid by trades +0.0007 +- 0.0043 (our value -0.082 at once,
	G7's forecast +0.083; its 7-day credit -0.006), a RECRUIT paid by trades -0.0079 +- 0.0020 (7-day credit -0.035).

	A schedule credit term [N] with its forecast's step (config.step_credit; the 2026-11-09 audit). The term is the
	outcome less the forecast here: the step (the forecast [N] at the player's next turn decision that day, step [N] its
	row, less the forecast here) + the rest (the outcome less the forecast there: a week of everyone's luck, shared by
	the day's decisions: 96.5% of the advantage's variance was). The step is the decision's own; the rest counts at
	  s = 1 - (1 - outcome_share_min) x trust       (the old horizon credit's rule, the battle credit's)
	credit = step - price + s x rest + (1 - s) x the class's mean rest
	classes [N] (step_classes; None: no correction): the mean rest of a class's rows this update (shrunk toward 0 by
	STEP_CLASS_PRIOR rows) is what the forecast gets wrong right after such a decision, on average. Without it a class
	is taught the forecast's habit as if it were the action's effect (measured on 8 joint7 games: after an UPGRADE G7's
	forecast falls 0.0070 while Φ loses 0.0002 and the week's outcome says +0.007 +- 0.008; after an exchange's MULTI
	-0.0028 against +0.0008 +- 0.0014). With it each class's mean credit is the outcome's, as before the step; the step
	sorts the decisions within the class.
	price [N] (None: none): what a decision costs (config.step_price, 0 for END and a question). A useless action's step
	is ~0, which is right (it loses nothing) and no reason to stop: the price is the reason, a little under the
	smallest action worth making. It is charged here, on the decision that made it, not in the 7-day labels: there
	the whole day's count would be every decision's of that day (one number a day again), the heads would have to learn
	it, and most of it would be a bonus for ending the day.
	The step alone (with its class's whole mean rest) where the term is unknown (its 7-day outcome), the term alone
	where the step is (the day's last turn decision; a step to a later day carries the night's luck and another day's
	horizon)."""
	s = 1.0 - (1.0 - outcome_share_min) * float(min(1.0, max(0.0, trust)))
	moved = forecast_steps(forecast, step, day)
	stepped = torch.isfinite(moved)
	both = torch.isfinite(term) & stepped
	rest = torch.where(both, term - moved, torch.zeros_like(term))
	correction = torch.zeros_like(term)
	if classes is not None and int(both.sum()):
		count = int(classes.max()) + 1
		sums = torch.zeros(count, device=term.device, dtype=term.dtype).index_add_(0, classes[both], rest[both])
		rows = torch.zeros(count, device=term.device, dtype=term.dtype).index_add_(0, classes[both], torch.ones_like(rest[both]))
		correction = (sums / (rows + STEP_CLASS_PRIOR))[classes]
	own = moved - price if price is not None else moved
	return torch.where(both, own + s * rest + (1 - s) * correction, torch.where(stepped, own + correction, term))


def advised_exchanges(decisions, stream=None):
	"""[N] bool: the exchange decisions opened by a move Nullkiller2 advised: the player's last turn decision before them
	was its advice's own MOVE (config.exchange_span_advised). There its exchange label is the delivery its plan asked
	for (Advice.cpp adviseExchange: the hero met gets the army), and the TAKE_BEST it accepts is that delivery in one
	decision. decisions: in each player's order; stream [N] (None: the color): each one's game and player."""
	out, last = np.zeros(len(decisions), dtype=bool), {}
	for i, d in enumerate(decisions):
		key = int(stream[i]) if stream is not None else d.color
		if d.kind == c.DecisionKind.TURN:
			last[key] = d.advice is not None and d.action_type == c.ActionType.MOVE and d.advice[0] == c.ActionType.MOVE and d.index == d.advice[1]
		elif d.kind == c.DecisionKind.EXCHANGE:
			out[i] = last.get(key, False)
	return out


def taught(advice, advised, config):
	"""Which advised decisions the teacher's cross-entropy counts [N] (advice: [N, 3] type, index, cell): Nullkiller2's
	TRADE advice only with config.nk2_trade_advice (False since 2026-11-04); schema 16's newly matched advice only with
	config.nk2_new_advice (new_advice); its GATHER advice (gather_advice) only with config.nk2_gather_weight."""
	advice = np.asarray(advice).reshape(-1, 3)
	if not config.nk2_trade_advice:
		advised = advised & (advice[:, 0] != c.ActionType.TRADE)
	if config.nk2_gather_weight <= 0:
		advised = advised & ~gather_advice(advice)
	return advised if config.nk2_new_advice else advised & ~new_advice(advice)


def gather_advice(advice):
	"""[N] whether each advice (type, index, cell) is a MULTI GATHER: Nullkiller2's visit again of the town its hero
	stands in (Advice.cpp visitAgain, ALPHA_MAP_GATHER_ADVICE), the garrison's troops into the hero."""
	advice = np.asarray(advice).reshape(-1, 3)
	return (advice[:, 0] == c.ActionType.MULTI) & (advice[:, 1] % c.N_MULTI == c.MultiKind.GATHER)


def advice_weights(advice, advised, config):
	"""[N] float32: each advised decision's weight in the teacher's cross-entropy: 1 where it counts (taught), 0 where
	not, config.nk2_gather_weight at a GATHER advice (a label our model gives ~0: a small weight of its own, the
	2026-11-05 collapse's rule)."""
	counted = taught(advice, advised, config)
	weights = counted.astype(np.float32)
	weights[counted & gather_advice(advice)] = config.nk2_gather_weight
	return weights


def new_advice(advice):
	"""[N] whether each advice (type, index, cell) is of a kind Match.cpp maps only since schema 16 (2026-11-07):
	ARTIFACT, MULTI's SPLIT_SINGLES, MERGE_ALL and GARRISON_SWAP, ARMY within one army or ONE's all-but-one half."""
	action, index, cell = advice[:, 0], advice[:, 1], advice[:, 2]
	multi = (action == c.ActionType.MULTI) & (index % c.N_MULTI >= c.MultiKind.SPLIT_SINGLES)
	f, _, to, _ = np.unravel_index(np.clip(index, 0, None) % (c.N_HOLDERS * c.ARMY_SLOTS * c.N_HOLDERS * c.N_ARMY_AMOUNTS), (c.N_HOLDERS, c.ARMY_SLOTS, c.N_HOLDERS, c.N_ARMY_AMOUNTS))
	army = (action == c.ActionType.ARMY) & ((f == to) | (cell >= c.ARMY_SLOTS))
	return (action == c.ActionType.ARTIFACT) | multi | army


def night_dropped(model, decisions, advice, has, inputs_of, device, config):
	"""[N] bool (config.nk2_night_drop): the advised turn decisions whose label, a MOVE or a TRAVEL, the night terms
	price more than config.nk2_night_drop_logit below staying with the hero's night there reading
	config.nk2_night_drop_reading or more: MOVE's tile_shift and tile_night at its tile, TRAVEL's logits less its prior
	and features.night_danger at its stop (a record without stops: none). The model's hand-made features, no gradient;
	inputs_of(rows) the rows' inputs."""
	from . import features as ft
	from .model_v2 import windows_of

	out = np.zeros(len(decisions), dtype=bool)
	kinds = np.array([d.kind for d in decisions])
	rows = np.flatnonzero(has & (kinds == c.DecisionKind.TURN) & np.isin(advice[:, 0], (c.ActionType.MOVE, c.ActionType.TRAVEL)))
	features, areas = model.features, c.MAP_LEVELS * c.TRAVEL_GRID * c.TRAVEL_GRID
	for start in range(0, len(rows), 64):
		part = rows[start : start + 64]
		x = {k: torch.as_tensor(v).to(device) for k, v in inputs_of(part).items()}
		with torch.no_grad():
			windows = model.windows(x)[0] if hasattr(model, "windows") else windows_of(x)[0]
			f = features(x, windows)
			shift_travel = (f.travel_logits - f.travel_prior).cpu().numpy()
			tonight = known = None
			if "travel_stop" in x:
				stops, known = ft.travel_stops(x)
				tonight = ft.night_danger(x, f.hero_strength, features.town_strength(x, f.town_power, manned=True), stops).cpu().numpy()
				known = known.cpu().numpy()
			windows = windows.cpu().numpy()
			shift_move, night_move = (f.tile_shift.cpu().numpy(), f.tile_night.cpu().numpy()) if len(windows) else (None, None)
		for j, r in enumerate(part):
			kind, index = int(advice[r, 0]), int(advice[r, 1])
			if kind == c.ActionType.MOVE:
				slot, cell = divmod(index, c.WINDOW * c.WINDOW)
				k = np.flatnonzero(windows == j * c.OWN_HEROES + slot)
				if shift_move is None or not len(k):
					continue
				row, column = divmod(cell, c.WINDOW)
				shift, reading = float(shift_move[k[0], row, column]), float(night_move[k[0], row, column])
			else:
				hero, area = divmod(index, areas)
				if tonight is None or not known[j, hero, area]:
					continue
				shift, reading = float(shift_travel[j, hero, area]), float(tonight[j, hero, area])
			out[r] = shift < -config.nk2_night_drop_logit and reading >= config.nk2_night_drop_reading
	return out


def ppo_update(
	model, optimizer, decisions, targets, returns, config=CampaignConfig(), decode=None, device=torch.device("cpu"), lr_scale=1.0, weights=LabelConfig().weights, advantages=None,
	growth=None, labels=None, end=None, foresight=None, growth_real=None, measure_noise=False, attributed=None, teacher=None, teacher_weight=None,
	share_trust=1.0, head_scale=None, teacher_sets=None, loss_weights=None, streams=None, clips=None, privileged=None,
	growth_trust=1.0,
):
	"""One update: config.passes over the decisions in minibatches of config.batch_size (each pass leaves out its
	last partial one: fixed shapes for Apple's GPU), stopped by the KL control. targets [N, N_VALUE_OFFSETS]: the
	share heads' grades (labels.ladder_targets; nan: none); growth [N, N_VALUE_OFFSETS] or None: the growth heads',
	the log of our value at the end of each day (labels.growth_labels), learned over our value at the decision (the
	model's Φ of its inputs, detached); labels: the real shares, for the heads' explained variance (default: targets).
	advantages [N]: the policy's (labels.horizon_advantages; nan: no credit, left out of the policy's loss), or None
	for the mix of returns [N] (the label mix minus the decisions' price) minus the heads' mix (weights).
	end or None: the end head's, {"targets" [N]: its grades (labels.end_targets), "real" [N] (labels.end_labels),
	"step", "week" [N]: indices into these decisions (labels.end_steps), "price" [N] (labels.label's), "actions" [N]
	(labels.actions_ahead; only with config.schedule_action_price), "weight": the
	end credit's share of the policy's advantage, "outcome": the week's share in it (labels.end_advantages)}; the end
	credit is computed here, from V_end of the model as the update starts (the KL anchor's pass), and blended with
	advantages (labels.blend_credit) before they are normalized. decode(rows) gives the inputs of those decisions
	(default: decode_inputs). share_trust 0..1: how far the schedule credit's share term takes the V7 change head as its
	baseline (share_baseline; the learner's trust); with config.step_credit, it and growth_trust (G7's, the learner's
	grade_trust) set how far each term is its step (step_credit). streams: (stream [N], position [N]) of each decision, its game and
	player and its place among the player's records, for config.memory_steps (sequence_order; None: each decision
	alone). clips: the groups' gradient clips (clip_gradients; None: the one clip at grad_clip). privileged: each
	decision's rivals for the privileged critic (config.privileged_critic; learner.privileged_rivals: "rivals" [N, R]
	their decisions' rows here, -1 none, "present" [N, R] 1 a rival alive in the game, "share_logit" [N] our true share's
	logit as the day began), or None. Returns (metrics, the KL control's new lr_scale)."""
	n, started = len(decisions), time.time()
	given_decode = decode  # (night_dropped's inputs: a caller's own, else the records')
	night_drop_share = float("nan")  # (config.nk2_night_drop: the share of the advised labels it left out)
	clock = {"prepass": 0.0, "wait": 0.0, "steps": 0.0, "forward": 0.0, "losses": 0.0, "backward": 0.0, "optimizer": 0.0, "cpu": time.thread_time()}  # where the update's time goes (logged: the PC's step ran at half the GPU's pace, 2026-10-30)
	# The foresight's labels (foresight.game_labels, v2), each decision's row: its loss joins the minibatch's, a share
	# per micro-batch as the others'; the shared layers' gradient from it against the PPO losses', on the first one
	from .foresight import PART_NAMES, WATCHED, ForesightConfig, gradient_check

	foresight_t = {name: torch.from_numpy(value).to(device) for name, value in foresight.items()} if foresight is not None else None
	if foresight_t is not None and hasattr(model, "plan_outputs"):  # (v2: its fs_plan for the plan's loss, in every graph recorded below)
		model.plan_outputs = True
	foresight_config = ForesightConfig(weight=getattr(config, "foresight_weight", ForesightConfig.weight))
	foresight_sums, foresight_sum, checked = Counter(), None, {}  # (foresight_sum: its parts, PART_NAMES' order, on the device)
	# On CUDA the bytes as sent go to the device and are cut and converted there (half the copy, none of the CPU's dtype copies)
	convert = inputs_on_device if decode is None and converts_on_device(device) else None
	decode = decode or ((lambda rows: wire_for([decisions[r] for r in rows], device)) if convert else (lambda rows: decode_inputs([decisions[r] for r in rows])))
	choice = {name: torch.from_numpy(value).to(device) for name, value in choices(decisions).items()}
	# Nullkiller2's advice (config.nk2_advice), where a decision has it: its type, entry and tile; advised 1 there
	advice_t = None
	# (teacher_sets: advice.teacher_sets' (types, indices, valid) of the decisions, every other entry that follows it)
	has = np.array([d.advice is not None for d in decisions], dtype=bool)
	if has.any() or (teacher_sets is not None and teacher_sets[2].any()):
		advice = np.array([d.advice if d.advice is not None else (0, 0, -1) for d in decisions], dtype=np.int64).reshape(-1, 3)
		# Only where its request was matched (has): a decision where Nullkiller2 asked for what we cannot do (an artifact
		# swap, a hero dismissed) has no label to weigh its plan's steps against (TEACHER_SPAN), 2026-11-05
		advised = advice_weights(advice, has, config)
		if getattr(config, "nk2_night_drop", False):  # (Nullkiller2's labels into the night left out: night_dropped)
			counted = int((advised > 0).sum())
			dropped = night_dropped(model, decisions, advice, has, given_decode or (lambda rows: decode_inputs([decisions[r] for r in rows])), device, config) & (advised > 0)
			advised[dropped] = 0.0
			night_drop_share = float(dropped.sum()) / max(counted, 1)
		advice_t = {
			"advice_action": torch.from_numpy(advice[:, 0]).to(device), "advice_index": torch.from_numpy(advice[:, 1]).to(device), "advice_cell": torch.from_numpy(advice[:, 2]).to(device),
			"advised": torch.from_numpy(advised).to(device),
		}
		if config.exchange_span_advised and not config.exchange_span_multi:  # (MULTI in the span at the exchanges its advice opened)
			advice_t["exchange_multi"] = torch.from_numpy(advised_exchanges(decisions, streams[0] if streams is not None else None)).to(device)
		if teacher_sets is not None:
			types, indices, valid = teacher_sets
			advice_t.update(advice_has=torch.from_numpy(has).to(device), accept_types=torch.from_numpy(types).to(device), accept_index=torch.from_numpy(indices).to(device), accept_valid=torch.from_numpy(valid).to(device))
	if advantages is None:
		baseline = choice["old_value"] @ torch.tensor(weights, dtype=torch.float32, device=device)
		advantage = torch.from_numpy(np.asarray(returns, dtype=np.float32)).to(device) - baseline
	else:
		advantage = torch.from_numpy(np.asarray(advantages, dtype=np.float32)).to(device)
	targets_t = torch.from_numpy(np.asarray(targets, dtype=np.float32)).to(device)
	share_valid = torch.isfinite(targets_t).float()
	targets_t = targets_t.nan_to_num(0.0)
	if growth is not None:
		growth_t = torch.from_numpy(np.asarray(growth, dtype=np.float32)).to(device)
		growth_valid = torch.isfinite(growth_t).float()
		growth_t = growth_t.nan_to_num(0.0)
	if end is not None:
		end_t = torch.from_numpy(np.asarray(end["targets"], dtype=np.float32)).to(device)
		end_valid = torch.isfinite(end_t).float()
		end_t = end_t.nan_to_num(0.0)
	for group in optimizer.param_groups:
		group["lr"] = group_lr(group, config, lr_scale)  # (the value heads' at value_lr_factor, apart from lr_scale)
	size = config.batch_size
	# config.memory_steps: the passes' micro-batches as sequences of a player's decisions, the memory carried through
	# them (model_v2.forward_steps) and its gradient back into the writes; the first pass's (the anchor's, the old
	# values') each decision on its recorded memory, as played
	unroll = max(1, int(getattr(config, "memory_steps", 1))) if streams is not None else 1
	if unroll > 1 and (size % config.micro_batches or (size // config.micro_batches) % unroll):
		print(f"campaign: FAILURE: memory_steps {unroll} does not divide a micro-batch ({size} / {config.micro_batches} rows): each decision alone", flush=True)
		unroll = 1
	forward = (lambda inputs: model.forward_steps(inputs, unroll)) if unroll > 1 else model
	# The KL control's anchor: each decision's choice under the model as the update starts; and V_end there (the end
	# credit's)
	graphs = None
	if config.cuda_graphs and Graphs.usable(model, device):  # (recorded once a process: kept with the model)
		graphs = model.__dict__.setdefault("_graphs", Graphs(model))
	# A micro-batch's forward, loss and backward recorded as one graph (graphs.StepGraph), checked against the eager step
	# the first time: the update launched ~1,000 small kernels a micro-batch outside the model's graph (its loss and the
	# loss's backward: ~45 of a 76 s update, 2026-10-31)
	teacher_graphs = None  # (teacher: a frozen model whose KL the loss adds at teacher_weight, else config.teacher_kl; its forward graphed apart)
	if teacher is not None and graphs is not None:
		teacher_graphs = teacher.__dict__.setdefault("_graphs", Graphs(teacher))
	teacher_weight_t = torch.full((), float(config.teacher_kl if teacher_weight is None else teacher_weight), device=device)
	# The KL control's target: target_kl bounds how far PPO's part of the loss, noise where the credit is, moves the policy
	# an update; a teacher's pull (its advice, its KL) is supervised, not noise. At teacher_weight against PPO's 1 the
	# update may move (1 + teacher_weight) x as far: at the teacher's 95% of the loss, PPO's target alone stopped every
	# joint7 update after 2-6 of its ~56 steps and cut the rate 1.5x each time (1e-5 to 3e-6 in 3 updates, 2026-11-01)
	# (loss_weights, the learner's loss shares: the value's and the foresight's weights, and kl_teacher, the teacher's share
	# against the policy's: the KL its pull makes, as teacher_weight was against PPO's 1 before them)
	kl_teacher = float(loss_weights["kl_teacher"]) if loss_weights is not None else float(teacher_weight_t)
	# (config.kl_teacher_inflation False since 2026-11-06, docs/optimizer_plan.md step 2: the target is target_kl, the
	# actor on the slow timescale: inflated, the policy's budget was 0.015 x (1 + 0.6 / 0.2) = 0.06 an update)
	kl_target = config.target_kl * (1.0 + kl_teacher) if config.kl_teacher_inflation and (advice_t is not None or teacher is not None) else config.target_kl
	weights_t = {f"lw:{name}": torch.full((), float(loss_weights[name]), device=device) for name in ("value", "foresight")} if loss_weights is not None else {}
	steps_graph = None
	privileged_module = getattr(model, "privileged", None)  # (its loss only where a batch has rivals: config.privileged_critic)
	if graphs is not None and config.step_graphs:
		steps_graph = model.__dict__.setdefault("_step_graph" if unroll == 1 else f"_step_graph_{unroll}", StepGraph(model, lambda out, batch: micro_loss(out, batch, config, foresight_config, privileged=privileged_module)[:3], forward))
	# The wire's conversion recorded in the graphs (config.convert_in_graphs): they take the decoded batches as sent
	# ({"wire": [N, IN_BYTES]}) and prepare them inside; everything else here converts them itself (model_inputs)
	prepare = (lambda raw: convert(raw["wire"])) if convert is not None and graphs is not None and config.convert_in_graphs else None
	model_inputs = prepare or (lambda raw: raw)
	fed = None if prepare is not None else convert  # (_prefetched's conversion: on the main thread, outside the graphs)
	model.eval()
	anchor, end_now, prior = torch.empty(n, device=device), torch.empty(n, device=device), torch.empty(n, device=device)
	growth_now, log_phi = torch.zeros(n, device=device), torch.zeros(n, device=device)  # (G7's logit, our log value: the attributed credit's)
	side = torch.full((n,), 36.0, device=device)  # (each decision's map side: the schedule credit's share weight)
	value_now = torch.zeros(n, device=device)  # (V7's logit: the schedule credit's share forecast)
	# The same pass measures the value loss on these decisions before the update learns from them (the user, 2026-10-29:
	# it should go down over time): every value head weighted as the loss below weighs it, above its targets' entropy
	fresh = {"share": torch.zeros(n, c.N_VALUE_OFFSETS, device=device), "growth": torch.zeros(n, c.N_VALUE_OFFSETS, device=device), "end": torch.zeros(n, 1, device=device)}
	# The privileged critic's inputs (config.privileged_critic): every decision's critic state and the value heads'
	# logits from this pass, the rivals' states gathered from them below
	priv_t = None
	if privileged is not None and config.privileged_critic and privileged_module is not None:
		priv_t = {name: torch.from_numpy(np.asarray(privileged[name])).to(device) for name in ("rivals", "present", "share_logit")}
		priv_t["present"] = priv_t["present"].float()
		priv_t["mask"] = priv_t["present"] * (priv_t["rivals"] >= 0).float()  # (pooled: the rivals alive with a state)
		priv_t["rivals"] = priv_t["rivals"].clamp(min=0)
		priv_t["share_logit"] = priv_t["share_logit"].float()
		state_all = torch.zeros(n, model.value_heads.in_features, device=device)
		logits_all = {"share": torch.zeros(n, c.N_VALUE_OFFSETS, device=device), "end": torch.zeros(n, device=device)}
	gpu_anchor_mb = 0.0
	with torch.no_grad():
		chunk = config.prepass_batch or max(1, size // max(1, config.micro_batches))  # (no gradients, but at a whole minibatch it was the update's peak: 5.2 GB)
		chunks = [np.arange(i, i + chunk).clip(max=n - 1) for i in range(0, n, chunk)]  # the last one padded: fixed shapes
		prepass_started = time.perf_counter()
		for rows, raw, heroes in _prefetched(chunks, decode, device, fed, config.decode_threads, measure=batch_heroes):
			index = torch.arange(int(rows[0]), int(rows[0]) + len(rows), device=device).clamp(max=n - 1)  # (made there: a copy from pageable memory waited for the GPU)
			# (fewer windows once a step's graph with them was checked: graphs.StepGraph.budget_ok, none here)
			budget = windows_for(len(rows), heroes, config) if steps_graph is not None and steps_graph.budget_ok else None
			out = graphs.eval(raw, budget, prepare) if graphs else model(model_inputs(raw))
			logp, _ = chosen_logp(out, {k: choice[k][index] for k in ("kind", "action", "index", "cell")})
			anchor[index] = logp
			end_now[index] = torch.sigmoid(out["end_logit"].float())
			if "growth_logit" in out:
				growth_now[index] = out["growth_logit"][:, -1].float()
				value_now[index] = out["value_logit"][:, -1].float()
				log_phi[index] = out["phi"].float().clamp(min=PHI_FLOOR).log()
			players = input_on_device(raw["wire"], "g_players") if prepare is not None else raw["g_players"]
			side[index] = (input_on_device(raw["wire"], "g_map") if prepare is not None else raw["g_map"])[:, :2].amax(dim=1).float()
			prior[index] = 1.0 / players.float().clamp(min=2) if config.share_target == "share" else 0.5  # the share heads' start: 1 / players (no change: 1/2)
			fresh["share"][index] = excess_loss(out["value_logit"].float(), targets_t[index])
			if growth is not None:
				now = out["phi"].float().clamp(min=PHI_FLOOR).log().unsqueeze(-1)
				fresh["growth"][index] = excess_loss(out["growth_logit"].float(), ((growth_t[index] - now).clamp(-GROWTH_CAP, GROWTH_CAP) + GROWTH_CAP) / (2 * GROWTH_CAP))
			if end is not None:
				fresh["end"][index] = excess_loss(out["end_logit"].float().unsqueeze(-1), end_t[index].unsqueeze(-1))
			if priv_t is not None:
				state_all[index] = out["value_hidden"].float()
				logits_all["share"][index], logits_all["end"][index] = out["value_logit"].float(), out["end_logit"].float()
			gpu_anchor_mb = max(gpu_anchor_mb, gpu_held(device))
		if device.type == "cuda":
			torch.cuda.synchronize(device)
		clock["prepass"] = time.perf_counter() - prepass_started
	masked = lambda x, valid: ((x * valid).sum(dim=0) / valid.sum(dim=0).clamp(min=1)).sum()
	# The share heads' fresh loss over a constant guess's, each head's mean target in this batch (a head that knew nothing
	# but the batch): 1 is no fit at all. value_fresh's jumps that come with the targets' spread (v1265-1266, 2026-11-05:
	# a batch of lopsided games) leave it flat, a worse fit does not (TensorBoard eval/campaign_value_unexplained)
	share_mean = (targets_t * share_valid).sum(dim=0) / share_valid.sum(dim=0).clamp(min=1)
	guess = torch.logit(share_mean.clamp(1e-4, 1 - 1e-4)).expand_as(targets_t)
	baseline = float(masked(excess_loss(guess, targets_t), share_valid))
	value_unexplained = float(masked(fresh["share"], share_valid)) / baseline if baseline > 1e-6 else float("nan")  # (nan: targets all alike)
	value_fresh, heads = masked(fresh["share"], share_valid), float(c.N_VALUE_OFFSETS)
	if growth is not None:
		value_fresh, heads = value_fresh + config.growth_weight * masked(fresh["growth"], growth_valid), heads + config.growth_weight * c.N_VALUE_OFFSETS
	if end is not None:
		value_fresh, heads = value_fresh + masked(fresh["end"], end_valid.unsqueeze(-1)), heads + 1.0
	value_fresh = float(value_fresh) / heads
	# Each head's fresh excess loss (TensorBoard campaign_heads/fresh_<head>; config.head_scale_by "residual" weighs by it)
	per_head = lambda x, valid: torch.where(valid.sum(dim=0) > 0, (x * valid).sum(dim=0) / valid.sum(dim=0).clamp(min=1), torch.full_like(valid[0], float("nan"))).tolist()  # noqa: E731 (nan: no grade in this batch)
	head_fresh = {"share": per_head(fresh["share"], share_valid)}
	if growth is not None:
		head_fresh["growth"] = per_head(fresh["growth"], growth_valid)
	if end is not None:
		head_fresh["end"] = per_head(fresh["end"], end_valid.unsqueeze(-1))
	del fresh
	# The privileged critic's corrections as the update starts (no gradient), its fit (the share heads' unexplained, as
	# value_unexplained's), and with config.privileged_credit the schedule credit's baselines from it (V7, V_end; G7 the
	# public one: the growth heads have no correction)
	privileged_stats = {}
	if priv_t is not None:
		k = c.N_VALUE_OFFSETS
		corr_all = torch.zeros(n, k + 1, device=device)
		with torch.no_grad():
			for first in range(0, n, 4096):
				part = slice(first, first + 4096)
				corr_all[part] = privileged_module(state_all[part], state_all[priv_t["rivals"][part]], priv_t["mask"][part], priv_t["share_logit"][part]).float()
		fresh_priv = excess_loss(logits_all["share"] + corr_all[:, :k], targets_t)
		privileged_stats = {
			"value_unexplained_privileged": float(masked(fresh_priv, share_valid)) / baseline if baseline > 1e-6 else float("nan"),
			"privileged_correction_rms": float(corr_all.pow(2).mean().sqrt()),
			"privileged_rivals": float(priv_t["mask"].sum(dim=1).mean()),  # (the rivals pooled: alive with a state, of privileged_present's alive)
			"privileged_present": float(priv_t["present"].sum(dim=1).mean()),
			"privileged_head_fresh": {"share": per_head(fresh_priv, share_valid)},
		}
		if end is not None:
			privileged_stats["privileged_head_fresh"]["end"] = per_head(excess_loss((logits_all["end"] + corr_all[:, k]).unsqueeze(-1), end_t.unsqueeze(-1)), end_valid.unsqueeze(-1))
		if config.privileged_credit:  # (the credit's baselines from the privileged critic: still functions of the state alone)
			value_now = value_now + corr_all[:, k - 1]
			end_now = torch.sigmoid(logits_all["end"] + corr_all[:, k])
		del logits_all, corr_all
	# config.head_scale: the heads' target variances this update (the growth heads' on their 0..1 scale, over the value
	# now), into the running ones (head_scale["var"]), and the heads' weights from them (head_weights)
	head_w, head_var, head_residual, head_var_slow = None, None, None, None
	if head_scale is not None:
		measured = {"share": masked_variance(targets_t, share_valid).tolist()}
		if growth is not None:
			scaled_growth = ((growth_t - log_phi.unsqueeze(-1)).clamp(-GROWTH_CAP, GROWTH_CAP) + GROWTH_CAP) / (2 * GROWTH_CAP)
			measured["growth"] = masked_variance(scaled_growth, growth_valid).tolist()
		if end is not None:
			measured["end"] = masked_variance(end_t.unsqueeze(-1), end_valid.unsqueeze(-1)).tolist()
		old = head_scale.get("var") or {}
		head_var = running(old, measured, HEAD_SCALE_MEMORY)
		weights = head_weights(head_var, float(head_scale.get("ramp", 1.0)), config.growth_weight)
		if config.head_scale_by == "residual":  # (ramped in from the variance's weights over HEAD_SCALE_RAMP updates)
			head_residual = running(head_scale.get("residual"), {name: values for name, values in head_fresh.items() if name in measured}, HEAD_RESIDUAL_MEMORY)
			head_var_slow = running(head_scale.get("var_slow") or old, measured, HEAD_SCALE_SLOW_MEMORY)
			by_residual = head_weights(head_var_slow, float(head_scale.get("ramp", 1.0)), config.growth_weight, residuals=head_residual)
			share = float(head_scale.get("residual_ramp", 1.0))
			weights = {name: [(1 - share) * a + share * b for a, b in zip(values, by_residual[name])] for name, values in weights.items()}
		head_w = {f"w:{name}": torch.tensor(values, dtype=torch.float32, device=device) for name, values in weights.items() if name in measured}
	gpu_mb = 0.0
	end_step_error = None
	attributed_stats = {}
	if attributed is not None and growth_real is not None:  # (config.credit "attributed": the parts of our log share's change we make)
		real7 = torch.from_numpy(np.asarray(growth_real, dtype=np.float32)[:, -1]).to(device)
		own = (1 - torch.from_numpy(attributed["share_now"]).to(device)) * ((real7 - log_phi).clamp(-GROWTH_CAP, GROWTH_CAP) - (torch.sigmoid(growth_now) * 2 * GROWTH_CAP - GROWTH_CAP))
		taken = torch.from_numpy(attributed["taken"]).to(device)
		advantage = own + taken  # (nan where our value 7 days on is unknown: no credit)
		known = torch.isfinite(advantage)
		if int(known.sum()) > 1:
			attributed_stats = {"attributed_taken_share": float((taken[known] > 0).float().mean()), "attributed_taken_mean": float(taken[known].mean()), "attributed_own_std": float(own[known].std()), "attributed_taken_std": float(taken[known].std())}
	schedule_stats = {}
	if config.credit == "schedule" and growth_real is not None:  # (config.credit: the share's weight grows by the day)
		real7 = torch.from_numpy(np.asarray(growth_real, dtype=np.float32)[:, -1]).to(device)
		grow = (real7 - log_phi).clamp(-GROWTH_CAP, GROWTH_CAP) - (torch.sigmoid(growth_now) * 2 * GROWTH_CAP - GROWTH_CAP)
		real_change = torch.from_numpy(np.asarray(targets, dtype=np.float32)[:, -1]).to(device) * 2 * SHARE_CHANGE_CAP - SHARE_CHANGE_CAP
		predicted_change = torch.sigmoid(value_now) * 2 * SHARE_CHANGE_CAP - SHARE_CHANGE_CAP
		change = real_change - share_baseline(real_change, predicted_change, share_trust)
		day = torch.tensor([d.day for d in decisions], dtype=torch.float32, device=device)
		step_stats = {}
		# SET ASIDE (the user, 2026-11-09): no credit on a decision's own step for now, the schedule credit is the 7-day
		# outcome alone (as it runs); config.step_credit changes nothing while these lines stay commented out. The step
		# credit (step_credit below, with its price a decision, its classes, its readings) is kept to measure later: to
		# try it, uncomment this block and set config.step_credit.
		# if config.step_credit and end is not None:  # (each term partly its forecast's step to the next turn decision that day)
		# 	step = torch.from_numpy(np.asarray(end["step"], dtype=np.int64)).to(device)
		# 	classes = step_classes(choice["kind"], choice["action"]) if config.step_class_correction else None
		# 	price = config.step_price * ((choice["action"] >= 0) & (choice["action"] != c.ActionType.END)).to(grow.dtype)
		# 	growth_forecast = log_phi + torch.sigmoid(growth_now) * 2 * GROWTH_CAP - GROWTH_CAP
		# 	# The share's forecast as its baseline scales it (share_baseline: the batch's mean is the same at both ends of a
		# 	# step), trusted for its step as far as that baseline explains the real change (forecast_trust: a fit;
		# 	# share_trust is the baseline's coefficient, and a slope of 1 would say nothing of its steps)
		# 	share_forecast = share_trust * predicted_change
		# 	share_step_trust = forecast_trust(real_change, real_change - change, config.trust_full_ev)
		# 	moved = forecast_steps(growth_forecast, step, day)
		# 	step_stats = {"schedule_step_trust": share_step_trust, **step_report(moved, choice["kind"], choice["action"])}
		# 	if int(torch.isfinite(moved).sum()) > 1:
		# 		step_stats["schedule_step_growth_std"] = float(moved[torch.isfinite(moved)].std())
		# 	grow = step_credit(grow, growth_forecast, step, day, growth_trust, classes, price, config.outcome_share_min)
		# 	change = step_credit(change, share_forecast, step, day, share_step_trust, classes, price, config.outcome_share_min)
		w = share_weight(day, side, config)
		advantage = w * change + (1 - w) * grow  # (nan where a label is unknown: no credit)
		known = torch.isfinite(advantage)
		if int(known.sum()) > 1:
			schedule_stats = {"schedule_share_weight": float(w[known].mean()), "schedule_share_std": float(change[known].std()), "schedule_growth_std": float(grow[known].std()), "schedule_share_trust": float(share_trust), "schedule_growth_trust": float(growth_trust), **step_stats}
		# The actions' price (config.schedule_action_price; the user, 2026-11-09): the 7-day outcome less a tiny amount for
		# each action the player makes from this decision on through that week (labels.actions_ahead, the learner's
		# end["actions"]), so that of two ways to the same outcome the one with fewer actions is the better
		if config.schedule_action_price > 0 and end is not None and "actions" in end:
			charged = config.schedule_action_price * torch.from_numpy(np.asarray(end["actions"], dtype=np.float32)).to(device)
			if int(known.sum()) > 1:
				schedule_stats.update(schedule_action_price_mean=float(charged[known].mean()), schedule_action_price_std=float(charged[known].std()))
			advantage = advantage - charged
	if end is not None and end["weight"] > 0:  # the policy's credit moves toward the end's as V_end proves itself
		turn = choices(decisions)["kind"] == c.DecisionKind.TURN
		later = end_advantages(end_now.cpu().numpy(), end["step"], end["week"], end["real"], end["price"], turn, config.decision_price, end["outcome"])
		# V_end's step estimate of the end against the end itself, for the end credit's check by action type (a diagnostic
		# in the learner, as the 7-day credit's check: the end credit is two thirds of the advantage, and no check saw it)
		values = end_now.cpu().numpy()
		end_step_error = np.where(end["step"] >= 0, values[np.maximum(end["step"], 0)] - end["real"], np.nan).astype(np.float32)
		if attributed is not None or config.credit == "schedule":  # (two units, log growth and share: each standardized before they mix)
			advantage = torch.from_numpy(blend_credit(standardized(advantage.cpu().numpy()), standardized(later), end["weight"])).to(device)
		else:
			advantage = torch.from_numpy(blend_credit(advantage.cpu().numpy(), later, end["weight"])).to(device)
	credited = torch.isfinite(advantage)
	if int(credited.sum()) > 1:
		advantage = (advantage - advantage[credited].mean()) / (advantage[credited].std() + 1e-8)
	advantage = torch.where(credited, advantage, torch.zeros_like(advantage))
	unsound = ~torch.isfinite(anchor)  # a choice the model calls impossible now (its legality changed): left out
	credited_f, anchor_safe, shares = credited.float(), anchor.nan_to_num(neginf=0.0), {}
	noise = {}
	if measure_noise and growth_real is not None and labels is not None:  # (the loss terms' gradient noise, at the update's start)
		tensors = {"choice": choice, "advantage": advantage, "credited": credited_f, "unsound": unsound, "targets": targets_t, "share_valid": share_valid,
			"labels": torch.from_numpy(np.asarray(labels, dtype=np.float32)).to(device), "growth_real": torch.from_numpy(np.asarray(growth_real, dtype=np.float32)).to(device)}
		if growth is not None:
			tensors.update(growth=growth_t, growth_valid=growth_valid)
		if end is not None:
			tensors.update(end=end_t, end_valid=end_valid)
		model.train()
		# At the update's micro-batch (not loss_noise's 128) and never the update's end: a diagnostic. At 128 rows, eager,
		# beside the recorded step's pool, it ran out of the 27 GB (joint7's v11, 2026-11-01) and ended the run
		try:
			noise = loss_noise(model, decisions, decode, convert, device, tensors, config, rows=max(1, size // max(1, config.micro_batches)))
		except torch.OutOfMemoryError as error:
			noise = {}
			print(f"campaign: the loss noise not measured (out of GPU memory: {str(error)[:120]})", flush=True)
		except Exception:  # (a diagnostic never ends the run: 2026-11-03 07:25, a TypeError here stopped joint7, its games in flight lost; the traceback logged, which the watcher sees)
			noise = {}
			traceback.print_exc()
			print("campaign: the loss noise not measured (the error above)", flush=True)
		if device.type == "cuda":
			torch.cuda.empty_cache()
	totals, steps, stopped, update_kl, recent = Counter(), 0, False, 0.0, deque(maxlen=KL_WINDOW)
	step_norms, step_bound = [], []  # each step's gradient norms before the clip ([total, GRAD_GROUPS']) and which clips bound
	kl_trace = []  # each step's KL before it, in order (the KL control's reading: kl_first, kl_growth)
	full_windows = 0  # micro-batches stepped over every hero slot (more of our heroes than config.window_budget allows)
	stopped_pass = 0  # the pass the KL control ended (0: none; b1, 2026-10-31: with passes=2, a second pass mostly cut short adds nothing)
	model.train()
	for pass_number in range(1, config.passes + 1):
		if stopped:
			break
		if unroll > 1:
			order = sequence_order(*streams, unroll, size, config.micro_batches)
		else:
			order = torch.randperm(n).numpy()
			if n >= size:
				order = order[: n - n % size]
		# Decoded a micro-batch at a time (a minibatch at once held ~4 of them decoded in pinned memory, a campaign
		# decision several MB: the PC's learner at 9.5 GB, 2026-10-30); the minibatch's gradient as before
		minibatches = [order[i : i + size] for i in range(0, len(order), size)]
		splits = [np.array_split(np.arange(len(rows)), max(1, min(config.micro_batches, len(rows)))) for rows in minibatches]
		decoded = _prefetched([rows[part] for rows, parts in zip(minibatches, splits) for part in parts], decode, device, fed, config.decode_threads, measure=sequence_heroes(unroll) if unroll > 1 else batch_heroes)
		# The pass's order on the device once: a minibatch's and each micro-batch's rows as slices of it (each copy of a
		# small index array from pageable memory waited for the GPU: five a minibatch, 2026-10-31)
		order_t = torch.from_numpy(order).to(device)
		for first, (rows, parts_of_rows) in zip(range(0, len(order), size), zip(minibatches, splits)):
			index = order_t[first : first + len(rows)]
			whole = (~unsound[index]).float()
			# The minibatch's counts: each micro-batch's losses are its sums over these, so the micro-batches' gradients add
			# up to the whole minibatch's, at 1 / micro_batches of its peak memory (2026-10-28: the update's ~8 GB on
			# Apple's GPU next to the battle learner's sent the machine into swap)
			counts = {"credit": (whole * credited[index].float()).sum().clamp(min=1), "keep": whole.sum().clamp(min=1), "share": share_valid[index].sum(dim=0)}
			if advice_t is not None:
				counts["advised"] = (whole * advice_t["advised"][index]).sum().clamp(min=1)
			if growth is not None:
				counts["growth"] = growth_valid[index].sum(dim=0)
			if end is not None:
				counts["end"] = end_valid[index].sum(dim=0)
			optimizer.zero_grad(set_to_none=False)  # (in place: a recorded step adds into these .grad tensors)
			parts = []
			for part in parts_of_rows:
				sub, keep = index[part[0] : part[-1] + 1], whole[part[0] : part[-1] + 1]  # (array_split's parts: consecutive)
				waited = time.perf_counter()
				_, raw, heroes = next(decoded)
				clock["wait"] += time.perf_counter() - waited
				budget = windows_for(len(part), heroes, config) if steps_graph is not None else None
				over = steps_graph is not None and config.window_budget > 0 and budget is None  # (more of our heroes than the budget)
				full_windows += over
				# A step over budget while budgets work is stepped eagerly, never recorded: a graph over every slot holds its
				# pool for the rest of the run (~6 GB of the GPU, 2026-11-03: recorded for one such step in hours, the card
				# then 7.6 GB from full and the loss noise out of memory)
				eager = over and steps_graph.budget_ok is not False
				share = len(part) / len(rows)
				batch = {
					**{k: choice[k][sub] for k in ("kind", "action", "index", "cell", "old_logp")},
					"keep": keep, "advantage": advantage[sub], "credited": credited_f[sub], "targets": targets_t[sub], "share_valid": share_valid[sub], "anchor": anchor_safe[sub],
					"count_credit": counts["credit"], "count_keep": counts["keep"], "count_share": counts["share"], "share": shares.setdefault(share, torch.full((), share, device=device)),
				}
				if growth is not None:
					batch.update(growth=growth_t[sub], growth_valid=growth_valid[sub], count_growth=counts["growth"])
				if end is not None:
					batch.update(end=end_t[sub], end_valid=end_valid[sub], count_end=counts["end"])
				if head_w is not None:  # (config.head_scale: each value head's weight, the update's)
					batch.update(head_w)
				if priv_t is not None:  # (the privileged critic's rivals: their states as the update started)
					batch.update(priv_rivals=state_all[priv_t["rivals"][sub]], priv_mask=priv_t["mask"][sub], priv_share=priv_t["share_logit"][sub])
				batch.update(weights_t)
				if advice_t is not None:  # Nullkiller2's advice, at the teacher's weight
					batch.update({name: value[sub] for name, value in advice_t.items()}, count_advised=counts["advised"], teacher_weight=teacher_weight_t)
				if foresight_t is not None:
					batch.update({"fs:" + name: value[sub] for name, value in foresight_t.items()})
				if teacher is not None:  # the teacher's distributions on the same inputs
					with torch.no_grad():
						t_out = teacher_graphs.eval(raw, prepare=prepare) if teacher_graphs is not None else teacher(model_inputs(raw))
					batch.update({"t:" + name: t_out[name] for name in TEACHER_OUTPUTS}, teacher_weight=teacher_weight_t)
				if foresight_t is not None and not checked:
					# The gradient check, once an update, on an eager forward of its own, through torch.autograd.grad; every tensor
					# of that forward is dropped before the graphed step: its autograd graph alive (one tensor of it kept) made
					# the parameters' AccumulateGrad nodes reference the default stream, and the capture refused them ("a stale
					# reference to the default stream (stream 0) from warmup", 2026-10-30)
					check = micro_loss(forward(model_inputs(raw)), batch, config, foresight_config, split=True, privileged=privileged_module)
					if check[3] is not None:
						checked = gradient_check(model, check[0] - check[3], check[3], check[4], prefix="grad_raw_", watch=WATCHED)
					del check
				mark = time.perf_counter()
				recorded = steps_graph.run(raw, batch, budget, prepare) if steps_graph is not None and not eager else None
				if recorded is not None:  # forward, loss and backward in one replay, its gradient added to .grad
					metrics, fs_parts = recorded
					clock["forward"] += time.perf_counter() - mark
				else:
					inputs = model_inputs(raw)
					out = graphs.train(inputs) if graphs and not eager and unroll == 1 else forward(inputs)
					clock["forward"] += time.perf_counter() - mark
					mark = time.perf_counter()
					loss, metrics, fs_parts, _ = micro_loss(out, batch, config, foresight_config, privileged=privileged_module)
					clock["losses"] += time.perf_counter() - mark
					mark = time.perf_counter()
					loss.backward()
					clock["backward"] += time.perf_counter() - mark
				if fs_parts is not None:  # (summed on the device: a float() here waited for the GPU every micro-batch)
					foresight_sum = fs_parts * share if foresight_sum is None else foresight_sum + fs_parts * share
					foresight_sums["steps_seen"] += share
				parts.append(metrics)  # (summed on the device: one sync a step, not six a micro-batch)
			mark = time.perf_counter()
			grad_norm, group_norms, bound = clip_gradients(optimizer, config.grad_clip, clips, GRAD_GROUPS)
			optimizer.step()
			stepped = time.perf_counter()
			clock["optimizer"] += stepped - mark
			read = torch.cat([torch.stack(parts).sum(dim=0), grad_norm.detach().float().reshape(1), group_norms, bound]).tolist()  # (a sync: the minibatch's GPU work is done here)
			sums, grad_norm, norms_now, bound_now = read[: len(STEP_PARTS)], read[len(STEP_PARTS)], read[len(STEP_PARTS) + 1 : len(STEP_PARTS) + 1 + len(GRAD_GROUPS)], read[len(STEP_PARTS) + 1 + len(GRAD_GROUPS) :]
			step_norms.append([grad_norm, *norms_now])
			step_bound.append(bound_now)
			clock["steps"] += time.perf_counter() - stepped
			step = Counter(dict(zip(STEP_PARTS, sums)))
			gpu_mb = max(gpu_mb, gpu_held(device))
			steps += 1
			recent.append(step["kl"])
			kl_trace.append(step["kl"])
			update_kl = float(np.mean(recent))  # the update's KL: the last KL_WINDOW minibatches', not one's
			totals.update(step, grad_norm=grad_norm)
			if config.target_kl and update_kl > 1.5 * kl_target:
				stopped, stopped_pass = True, pass_number  # this update has moved the policy far enough
				break
		decoded.close()  # (its thread ends, its decoded batches freed)
	model.eval()
	gpu_mb = max(gpu_mb, gpu_held(device, release=True))  # (CUDA: the peak logged, the cache released: the other GPU process keeps its headroom)
	if config.target_kl and steps:
		if stopped:
			lr_scale = max(0.1, lr_scale / 1.5)
		elif update_kl < kl_target / 2:
			lr_scale = min(config.lr_scale_max, lr_scale * 1.5)
	# The share heads' explained variance over the share prior's: the player count alone explains much of a share (1/2 in
	# a 2-player game, 1/4 in a 4-player one), and the heads start there, so plain EV trusted them before they learned
	predicted, actual = choice["old_value"].cpu().numpy(), np.asarray(targets if labels is None else labels, dtype=np.float32)
	predicted, actual = predicted - prior.cpu().numpy()[:, None], actual - prior.cpu().numpy()[:, None]
	metrics = {key: value / max(steps, 1) for key, value in totals.items()}
	metrics.update(
		steps=steps,
		stopped=stopped,
		stopped_pass=stopped_pass,
		noise=noise,  # (loss_noise's, when measured)
		**attributed_stats,
		**schedule_stats,
		signal_ratio=signal_ratio(optimizer),  # (the steps' noise: the user's 2026-10-31 aim of 0.25+, with 4,096-decision steps)
		signal_fraction=signal_fraction(optimizer),  # (the share of each group's gradient that is signal, at any beta1)
		grad_norms=step_norms,  # (each step's: the learner's clip calibration, config.grad_clip_by_group)
		**{f"grad_norm_{name}": float(np.median([norms[1 + i] for norms in step_norms])) for i, name in enumerate(GRAD_GROUPS) if step_norms},
		**{f"clip_bound_{name}": float(np.mean([bound[i] for bound in step_bound])) for i, name in enumerate(GRAD_GROUPS) if step_bound},
		clipped_by_group=clips is not None,
		value_learning_rate=group_lr({"name": "value"}, config, lr_scale),
		value_readout_rms=value_readout_rms(model),
		head_fresh=head_fresh,
		end_step_error=end_step_error,
		last_kl=update_kl,
		kl_target=kl_target,
		# (2026-11-06: the KL barely followed the rate, 2.5x less rate for 1.6x less KL) the first step's KL, before any
		# step of this update (what the training forward differs from the anchor's by at the same weights:
		# config.memory_steps' memory, its window budget), and its growth a step (a line through the steps' KLs)
		kl_first=kl_trace[0] if kl_trace else float("nan"),
		kl_growth=float(np.polyfit(np.arange(len(kl_trace)), kl_trace, 1)[0]) if len(kl_trace) > 2 else float("nan"),
		unsound=int(unsound.sum()),
		decisions=n,
		value_fresh=value_fresh,
		value_unexplained=value_unexplained,
		**privileged_stats,  # (config.privileged_critic: its fit, its corrections' size, the rivals a decision had)
		learning_rate=config.learning_rate * lr_scale,
		credited=int(credited.sum()),
		gpu_mb=gpu_mb,
		gpu_anchor_mb=gpu_anchor_mb,
		update_seconds=time.time() - started,
		prepass_seconds=clock["prepass"],
		wait_seconds=clock["wait"],  # blocked on the decoded batches
		sync_seconds=clock["steps"],  # blocked on the GPU at each minibatch's end
		# The thread's time launching each part of the micro-batches (2026-10-31: the update was CPU-bound, the thread busy ~78%)
		forward_seconds=clock["forward"],
		losses_seconds=clock["losses"],
		backward_seconds=clock["backward"],
		optimizer_seconds=clock["optimizer"],
		cpu_share=(time.thread_time() - clock["cpu"]) / max(time.time() - started, 1e-9),  # of the update's time this thread ran on the CPU
		full_windows=full_windows,
		memory_steps=unroll,
		window_budget_ok=steps_graph.budget_ok if steps_graph is not None else None,
		**({f"foresight_{name}": value / max(foresight_sums["steps_seen"], 1e-9) for name, value in zip(PART_NAMES, foresight_sum.tolist())} if foresight_sum is not None else {}),
		**checked,
		**{f"value_ev_d{k}": explained_variance(predicted[:, i], actual[:, i]) for i, k in enumerate(c.VALUE_OFFSETS)},
		**{f"value_parts_d{k}": residual_variances(predicted[:, i], actual[:, i]) for i, k in enumerate(c.VALUE_OFFSETS)},
		**value_fit(predicted[:, -1], actual[:, -1]),
		head_var=head_var,  # (config.head_scale: the running target variances, for the next update)
		head_weights={name[2:]: tensor.tolist() for name, tensor in head_w.items()} if head_w is not None else None,
		head_residual=head_residual,  # (config.head_scale_by "residual": the heads' running fresh excess, for the next update)
		head_var_slow=head_var_slow,
		night_dropped=night_drop_share,
	)
	return metrics, lr_scale


def summarize(game, empire=EmpireConfig()):
	"""Per day and player: Φ and its parts, the share, heroes; and the decisions by kind and action type."""
	lines = []
	heroes = []
	for day, snapshot in sorted(game.ledgers.items()):
		values = {COLORS[p["color"]]: empire_value(p, empire) for p in snapshot["players"]}
		total = sum(max(v["total"], 0) for v in values.values()) or 1
		for p in snapshot["players"]:
			if p["status"] == 0:
				heroes.append(len(p["heroes"]))
		cells = "  ".join(f"{color} {v['total']:8.0f} ({max(v['total'], 0) / total:.3f})" for color, v in values.items())
		lines.append(f"start of day {day}: {cells}")
	last = game.ledgers[max(game.ledgers)] if game.ledgers else None
	if last:
		for p in last["players"]:
			parts = empire_value(p, empire)
			lines.append(f"  {COLORS[p['color']]:6s} " + " ".join(f"{k} {v:.0f}" for k, v in parts.items() if k != "total"))
	kinds = Counter(c.DecisionKind(d.kind).name for d in game.decisions)
	types = Counter(c.ActionType(d.action_type).name for d in game.decisions if d.action_type >= 0)
	no_effect = Counter(c.ActionType(d.action_type).name for d in game.decisions if d.flags & NO_EFFECT)
	lines.append(f"decisions by kind {dict(kinds)}; turn actions {dict(types)}; changed nothing {dict(no_effect)}")
	lines.append(f"heroes (mean over players and days): {np.mean(heroes) if heroes else float('nan'):.2f}")
	return "\n".join(lines), (float(np.mean(heroes)) if heroes else float("nan"))


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--days", type=int, default=5)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--players", type=int, default=0, help="0: 2 to 4 at random")
	parser.add_argument("--size", type=int, default=36)
	parser.add_argument("--water", default="random")
	parser.add_argument("--checkpoint", default="", help="start from this checkpoint (default: a new model)")
	parser.add_argument("--max-seconds", type=int, default=170, help="the game's time limit (headless.sh kills it)")
	parser.add_argument("--battle-ai", default="BattleAI")
	parser.add_argument("--name", default="")
	args = parser.parse_args()
	torch.manual_seed(args.seed)
	config, labels = CampaignConfig(passes=1), LabelConfig()
	name = args.name or time.strftime("%Y%m%d-%H%M%S")
	out_dir = RUNS / name
	out_dir.mkdir(parents=True, exist_ok=True)

	players = args.players or random.Random(args.seed).randint(2, 4)
	map_name, template = make_map(args.seed, players, args.size, water=args.water)
	print(f"map {map_name} ({template}), {players} players, {args.days} days")
	with serve_model(args.checkpoint) as (model, server):
		game, info = play(server.socket_path, map_name, args.days, args.seed, max_seconds=args.max_seconds, battle_ai=args.battle_ai)
	(battle_play.USER_DIR / (map_name + ".vmap")).unlink(missing_ok=True)
	info.template = template
	print(f"game: {info.status}, {info.seconds:.1f}s, logs {info.run_mb:.1f} MB; records {game.record_bytes / 2**20:.2f} MB "
		  f"({game.record_bytes / max(len(game.decisions), 1) / 1024:.1f} KB a decision, {game.record_bytes / 2**20 / max(info.seconds, 1e-9) * 60:.2f} MB/min)")
	print(f"AIs: {game.done}")
	summary, heroes_mean = summarize(game)
	print(summary)
	(out_dir / "ledgers.json").write_text(json.dumps({"days": game.ledgers, "final": game.final}))
	(out_dir / "decisions.json").write_text(json.dumps([{k: v for k, v in vars(d).items() if k != "inputs"} for d in game.decisions]))
	game.decisions, game.days = usable(game.decisions), args.days
	if not game.decisions or not game.ledgers:
		raise SystemExit("no decisions or no ledger: nothing to train on")
	targets, mixed, price = label(game, labels)
	real = end_labels(game)
	step, week = end_steps(game)
	end = {"targets": real, "real": real, "step": step, "week": week, "price": price, "weight": 0.0, "outcome": 1.0}  # (V_end trained, not yet credited)
	optimizer = make_optimizer(model, config)
	started = time.time()
	metrics, _ = ppo_update(model, optimizer, game.decisions, targets, mixed - price, config, growth=growth_labels(game), end=end)
	metrics["seconds"] = time.time() - started
	metrics.update(heroes_mean=heroes_mean, label_mean=float(np.nanmean(mixed)), price_mean=float(price.mean()), game_seconds=info.seconds, game_status=info.status)
	print("update: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in metrics.items()))
	checkpoint = out_dir / "model_00001.pt"
	torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "version": 1, "model_kwargs": {}}, checkpoint)
	(out_dir / "metrics.json").write_text(json.dumps({"map": map_name, "template": template, "players": players, **metrics}, indent=1, default=str))
	print(f"saved {checkpoint}")


if __name__ == "__main__":
	main()
