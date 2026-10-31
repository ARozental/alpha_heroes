"""PPO for the campaign model: the update a joint run takes (learner.py) and a one-game smoke test.

	.venv/bin/python -m campaign_model.train --days 5 --seed 1     # one self-play game, then one update

PPO as the battle model's (battle_model/train.py): the clipped surrogate on advantages normalized over the update,
passes over minibatches of fixed size, the KL control (an update stops once its KL from the policy at its start passes
1.5 x target_kl, and the rate's scale is cut by 1.5, or raised back by 1.5 after an update that ended under half of it,
never above lr_scale_max: the user's rule, never above 1e-4). Entropy: a floor, not a bonus (battle_model/losses.py's
lesson of 2026-10-28: a constant bonus is a consistent push that beats a noise-level policy gradient), with the
parents' probabilities detached (a type's table entropy weighs by p(type), which gets no gradient from it:
test_entropy_bonus_does_not_favour_types_with_many_options). The value heads are trained with cross-entropy (the
user's rule for 0..1 targets: it learns far better than squared error, whose gradient dies near 0 and 1) against
their grades (labels.ladder_targets: the shares VALUE_OFFSETS days on, the longer heads' graded against the shorter
ones' later forecasts), and the growth heads, at growth_weight each, against our own value's log growth on their 0..1
scale (labels.growth_scale), and V_end, at a share head's weight, against its grades (labels.end_targets); a missing
grade (nan: past the day cap) is left out, and so is a decision without credit from the policy's loss. The optimizer (make_optimizer) is Adam, with weight decay only on the
corrections of the model's fixed priors (model.prior_corrections): back to the prior unless the games keep pushing.
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
from collections import Counter, deque

import numpy as np
import torch
import torch.nn.functional as F

from battle_model import play as battle_play
from battle_model.inference import InferenceServer

from . import constants as c
from .config import CampaignConfig
from .empire import EmpireConfig, empire_value
from .labels import COLORS, GROWTH_CAP, PHI_FLOOR, LabelConfig, blend_credit, end_advantages, end_labels, end_steps, explained_variance, growth_labels, label, residual_variances
from .model import CampaignModel, build_model
from .play import make_map, play
from .graphs import Graphs, StepGraph
from .records import NO_EFFECT, decode_inputs, decode_wire, inputs_on_device
from .schema import HEADS

RUNS = battle_play.ROOT / "runtime" / "campaign"
TURN_KINDS = (c.DecisionKind.TURN, c.DecisionKind.EXCHANGE)


def model_kwargs(config):
	kwargs = {"dim": config.model_dim, "cell_dim": config.model_cell_dim, "heads": config.model_heads, "layers": config.model_layers}
	return {**kwargs, "architecture": config.model_architecture} if config.model_architecture != "v1" else kwargs


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
	entropy = torch.where(turn, entropy, head_entropy(out["option_logp"]))
	return logp, entropy


DECODE_THREADS = 3  # batches decoded at once: one thread's ~0.3 s a 128-decision micro-batch (110 MB of zlib and dtype copies, which release the interpreter lock) held the GPU's 146 ms step to it (2026-10-30, the PC)


def _prefetched(batches, decode, device, convert=None, threads=DECODE_THREADS):
	"""(rows, inputs on device) of each batch of rows, decoded ahead in `threads` threads (zlib and numpy let the
	training step go on meanwhile) and yielded in order. convert: applied on the device to what decode gave (the wire
	bytes cut and converted there: records.inputs_on_device), else decode gives the inputs themselves. A consumer that
	stops early (the KL control) ends the threads: they were left blocked on the full queue with their batches, every
	such update (2026-10-30)."""
	import concurrent.futures

	stop = threading.Event()
	pinned = device.type == "cuda"  # (a copy from pageable memory waits for the GPU: ~100 of them a minibatch)

	def work(rows):
		if stop.is_set():
			return None
		decoded = decode(rows)
		inputs = {name: torch.from_numpy(value) for name, value in decoded.items()} if isinstance(decoded, dict) else {"wire": torch.from_numpy(decoded)}
		return rows, ({name: value.pin_memory() for name, value in inputs.items()} if pinned else inputs)

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
			rows, inputs = item
			inputs = {name: value.to(device, non_blocking=pinned) for name, value in inputs.items()}
			yield rows, (convert(inputs["wire"]) if convert is not None and "wire" in inputs else inputs)
	finally:
		stop.set()
		pool.shutdown(wait=False, cancel_futures=True)


def make_optimizer(model, config=CampaignConfig()):
	"""AdamW: no weight decay but on the corrections of the model's fixed priors (model.prior_corrections: the object
	values' gains and deltas), decayed by config.prior_decay toward 0, the prior (the user, 2026-10-28: the
	regularization penalizes the change from Nullkiller2's values). Adam moves a weight ~lr a step when its gradient is
	consistent and the decay pulls lr x prior_decay x weight back, so a correction's weight settles near 1 / prior_decay
	under a steady push (its effect: CORRECTION_STEP times that) and near 0 under noise."""
	corrections = model.prior_corrections()
	ids = {id(p) for p in corrections}
	rest = [p for p in model.parameters() if id(p) not in ids]
	return torch.optim.AdamW([{"params": rest, "weight_decay": 0.0}, {"params": corrections, "weight_decay": config.prior_decay}], lr=config.learning_rate)


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
	index = {}  # a parameter's key in the optimizer's state: its place in make_optimizer's groups
	if saved:
		corrections = model.prior_corrections()
		ids, by_id = {id(p) for p in corrections}, {id(p): n for n, p in model.named_parameters()}
		order = [by_id[id(p)] for p in model.parameters() if id(p) not in ids] + [by_id[id(p)] for p in corrections]
		flat = [i for group in saved["param_groups"] for i in group["params"]]
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
	"""The optimizer's state from a checkpoint (saved: its state_dict; saved_keys: its model's state_dict keys). A model
	that has gained parameters since (schema 7: hire_worth) keeps Adam's moments for the ones it had, matched by name
	within each group, and starts the new ones fresh."""
	names = {id(p): name for name, p in model.named_parameters()}
	groups = [[names[id(p)] for p in group["params"]] for group in optimizer.param_groups]
	if all(len(g) == len(s["params"]) for g, s in zip(groups, saved["param_groups"])):
		optimizer.load_state_dict(saved)
		return
	saved_keys = set(saved_keys)
	state, param_groups, index = {}, [], 0
	for group, saved_group in zip(groups, saved["param_groups"]):
		had = [name for name in group if name in saved_keys]  # the checkpoint's parameters, in its order
		if len(had) != len(saved_group["params"]):
			raise ValueError(f"the checkpoint's optimizer group has {len(saved_group['params'])} parameters, its model {len(had)}")
		old = dict(zip(had, saved_group["params"]))
		ids = []
		for name in group:
			if name in old and old[name] in saved["state"]:
				state[index] = saved["state"][old[name]]
			ids.append(index)
			index += 1
		param_groups.append({**saved_group, "params": ids})
	optimizer.load_state_dict({"state": state, "param_groups": param_groups})
	fresh = [name for group in groups for name in group if name not in saved_keys]
	print(f"campaign optimizer: Adam's moments kept, fresh for the new parameters: {', '.join(fresh)}", flush=True)


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


def excess_loss(logits, targets):
	"""Each head's cross-entropy against its soft targets above the targets' own entropy (its floor): 0 when the head
	predicts its target exactly."""
	floor = -(torch.special.xlogy(targets, targets) + torch.special.xlogy(1 - targets, 1 - targets))
	return F.binary_cross_entropy_with_logits(logits, targets, reduction="none") - floor


STEP_PARTS = ("policy", "value", "entropy", "below_floor", "clip_fraction", "kl", "teacher_kl")  # a step's metrics, in ppo_update's order
# The teacher's outputs its KL reads (teacher_kl), put in each micro-batch as "t:<name>"
TEACHER_OUTPUTS = ("type_logp", "option_logp") + tuple(output for output, _ in HEADS.values())


def teacher_kl(out, batch):
	"""KL(teacher || model) of each decision [N] (config.teacher_kl): a turn's over the action type plus each type's
	table weighted by the teacher's p(type); a question's over its options. An entry the teacher gives no probability
	adds nothing (both models see the same legal masks); the model's log-probabilities are floored at -50 there."""
	def part(t, s):
		p = t.exp()
		return torch.where(p > 0, p * (t - s.clamp(min=-50.0)), torch.zeros_like(p)).sum(dim=-1)

	kind = batch["kind"]
	n = kind.shape[0]
	turn = (kind == c.DecisionKind.TURN) | (kind == c.DecisionKind.EXCHANGE)
	t_type = batch["t:type_logp"]
	kl = part(t_type, out["type_logp"])
	p_type = t_type.exp()
	for a, (output, _) in HEADS.items():
		kl = kl + p_type[:, a] * part(batch["t:" + output].reshape(n, -1), out[output].reshape(n, -1))
	return torch.where(turn, kl, part(batch["t:option_logp"], out["option_logp"]))


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
			inputs = {name: torch.from_numpy(value).to(device) for name, value in raw.items()} if isinstance(raw, dict) else convert(torch.from_numpy(raw).to(device))
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


def micro_loss(out, batch, config, foresight_config=None):
	"""A micro-batch's loss and metrics from the model's outputs and its rows, for the eager step and the recorded one
	alike (graphs.StepGraph): (loss, the rest of STEP_PARTS [6], the foresight's parts [len(PART_NAMES)] or None, the
	foresight's share of the loss or None). batch: tensors on the device, each a row of the micro-batch (kind, action,
	index, cell, old_logp, keep, advantage, credited, targets, share_valid, anchor; growth, growth_valid; end, end_valid;
	"fs:<label>": the foresight's), and its minibatch's counts (count_credit, count_keep, count_share, count_growth,
	count_end) and its share of the minibatch ("share"): the micro-batches' gradients add up to the minibatch's."""
	from .foresight import PART_NAMES, ForesightConfig, foresight_loss

	keep = batch["keep"]
	logp, entropy = chosen_logp(out, {k: batch[k] for k in ("kind", "action", "index", "cell")})
	logp = torch.where(keep > 0, logp, batch["old_logp"])
	ratio = (logp - batch["old_logp"]).exp()
	a, credit = batch["advantage"], keep * batch["credited"]
	policy = -(torch.minimum(ratio * a, ratio.clamp(1 - config.clip, 1 + config.clip) * a) * credit).sum() / batch["count_credit"]
	value, heads = head_loss(out["value_logit"], batch["targets"], batch["share_valid"], batch["count_share"]).sum(), float(c.N_VALUE_OFFSETS)
	if "growth" in batch:  # our value's log growth from the decision, on the heads' 0..1 scale
		now = out["phi"].detach().clamp(min=PHI_FLOOR).log().unsqueeze(-1)
		scaled = ((batch["growth"] - now).clamp(-GROWTH_CAP, GROWTH_CAP) + GROWTH_CAP) / (2 * GROWTH_CAP)
		value = value + config.growth_weight * head_loss(out["growth_logit"], scaled, batch["growth_valid"], batch["count_growth"]).sum()
		heads += config.growth_weight * c.N_VALUE_OFFSETS
	if "end" in batch:  # V_end, at a share head's weight
		value = value + head_loss(out["end_logit"].unsqueeze(-1), batch["end"].unsqueeze(-1), batch["end_valid"].unsqueeze(-1), batch["count_end"].unsqueeze(-1)).sum()
		heads += 1.0
	value = value / heads
	bonus = (entropy * keep).sum() / batch["count_keep"]
	floor = (torch.relu(config.entropy_floor - entropy) * keep).sum() / batch["count_keep"]  # only decisions below it
	loss = policy + config.value_weight * value - config.entropy_weight * bonus + config.entropy_floor_weight * floor
	teacher_part = torch.zeros((), device=keep.device)
	if "t:type_logp" in batch:  # the teacher's KL (ppo_update's teacher), at its weight (a tensor: a recorded step reads it)
		teacher_part = (teacher_kl(out, batch) * keep).sum() / batch["count_keep"]
		loss = loss + batch["teacher_weight"] * teacher_part
	fs_parts = fs_share = None
	labels = {name[3:]: value for name, value in batch.items() if name.startswith("fs:")}
	if labels and "fs_turn" in out:
		fs_loss, parts = foresight_loss(out, labels, foresight_config or ForesightConfig())
		fs_share = batch["share"] * fs_loss
		loss = loss + fs_share
		fs_parts = torch.stack([parts[name].float() for name in PART_NAMES])
	drift = logp.detach() - batch["anchor"]
	metrics = torch.stack([
		policy.detach(),
		value.detach(),
		bonus.detach(),
		((entropy < config.entropy_floor).float() * keep).sum() / batch["count_keep"],  # below_floor
		(((ratio - 1).abs() > config.clip).float() * keep).sum() / batch["count_keep"],  # clip_fraction
		(((drift.exp() - 1) - drift) * keep).sum() / batch["count_keep"],  # kl: k3, before this step
		teacher_part.detach(),  # teacher_kl
	]).float()
	return loss, metrics, fs_parts, fs_share


def ppo_update(
	model, optimizer, decisions, targets, returns, config=CampaignConfig(), decode=None, device=torch.device("cpu"), lr_scale=1.0, weights=LabelConfig().weights, advantages=None,
	growth=None, labels=None, end=None, foresight=None, growth_real=None, measure_noise=False, attributed=None, teacher=None, teacher_weight=None,
):
	"""One update: config.passes over the decisions in minibatches of config.batch_size (each pass leaves out its
	last partial one: fixed shapes for Apple's GPU), stopped by the KL control. targets [N, N_VALUE_OFFSETS]: the
	share heads' grades (labels.ladder_targets; nan: none); growth [N, N_VALUE_OFFSETS] or None: the growth heads',
	the log of our value at the end of each day (labels.growth_labels), learned over our value at the decision (the
	model's Φ of its inputs, detached); labels: the real shares, for the heads' explained variance (default: targets).
	advantages [N]: the policy's (labels.horizon_advantages; nan: no credit, left out of the policy's loss), or None
	for the mix of returns [N] (the label mix minus the decisions' price) minus the heads' mix (weights).
	end or None: the end head's, {"targets" [N]: its grades (labels.end_targets), "real" [N] (labels.end_labels),
	"step", "week" [N]: indices into these decisions (labels.end_steps), "price" [N] (labels.label's), "weight": the
	end credit's share of the policy's advantage, "outcome": the week's share in it (labels.end_advantages)}; the end
	credit is computed here, from V_end of the model as the update starts (the KL anchor's pass), and blended with
	advantages (labels.blend_credit) before they are normalized. decode(rows) gives the inputs of those decisions
	(default: decode_inputs). Returns (metrics, the KL control's new lr_scale)."""
	n, started = len(decisions), time.time()
	clock = {"prepass": 0.0, "wait": 0.0, "steps": 0.0, "forward": 0.0, "losses": 0.0, "backward": 0.0, "optimizer": 0.0, "cpu": time.thread_time()}  # where the update's time goes (logged: the PC's step ran at half the GPU's pace, 2026-10-30)
	# The foresight's labels (foresight.game_labels, v2), each decision's row: its loss joins the minibatch's, a share
	# per micro-batch as the others'; the shared layers' gradient from it against the PPO losses', on the first one
	from .foresight import PART_NAMES, ForesightConfig, gradient_check

	foresight_t = {name: torch.from_numpy(value).to(device) for name, value in foresight.items()} if foresight is not None else None
	foresight_config = ForesightConfig(weight=getattr(config, "foresight_weight", ForesightConfig.weight))
	foresight_sums, foresight_sum, checked = Counter(), None, {}  # (foresight_sum: its parts, PART_NAMES' order, on the device)
	# On CUDA the bytes as sent go to the device and are cut and converted there (half the copy, none of the CPU's dtype copies)
	convert = inputs_on_device if decode is None and device.type == "cuda" else None
	decode = decode or ((lambda rows: decode_wire([decisions[r] for r in rows])) if convert else (lambda rows: decode_inputs([decisions[r] for r in rows])))
	choice = {name: torch.from_numpy(value).to(device) for name, value in choices(decisions).items()}
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
		group["lr"] = config.learning_rate * lr_scale
	size = config.batch_size
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
	steps_graph = None
	if graphs is not None and config.step_graphs:
		steps_graph = model.__dict__.setdefault("_step_graph", StepGraph(model, lambda out, batch: micro_loss(out, batch, config, foresight_config)[:3]))
	model.eval()
	anchor, end_now, prior = torch.empty(n, device=device), torch.empty(n, device=device), torch.empty(n, device=device)
	growth_now, log_phi = torch.zeros(n, device=device), torch.zeros(n, device=device)  # (G7's logit, our log value: the attributed credit's)
	# The same pass measures the value loss on these decisions before the update learns from them (the user, 2026-10-29:
	# it should go down over time): every value head weighted as the loss below weighs it, above its targets' entropy
	fresh = {"share": torch.zeros(n, c.N_VALUE_OFFSETS, device=device), "growth": torch.zeros(n, c.N_VALUE_OFFSETS, device=device), "end": torch.zeros(n, 1, device=device)}
	gpu_anchor_mb = 0.0
	with torch.no_grad():
		chunk = config.prepass_batch or max(1, size // max(1, config.micro_batches))  # (no gradients, but at a whole minibatch it was the update's peak: 5.2 GB)
		chunks = [np.arange(i, i + chunk).clip(max=n - 1) for i in range(0, n, chunk)]  # the last one padded: fixed shapes
		prepass_started = time.perf_counter()
		for rows, inputs in _prefetched(chunks, decode, device, convert, config.decode_threads):
			index = torch.arange(int(rows[0]), int(rows[0]) + len(rows), device=device).clamp(max=n - 1)  # (made there: a copy from pageable memory waited for the GPU)
			out = graphs.eval(inputs) if graphs else model(inputs)
			logp, _ = chosen_logp(out, {k: choice[k][index] for k in ("kind", "action", "index", "cell")})
			anchor[index] = logp
			end_now[index] = torch.sigmoid(out["end_logit"].float())
			if "growth_logit" in out:
				growth_now[index] = out["growth_logit"][:, -1].float()
				log_phi[index] = out["phi"].float().clamp(min=PHI_FLOOR).log()
			prior[index] = 1.0 / inputs["g_players"].float().clamp(min=2)  # the share heads' start: 1 / players
			fresh["share"][index] = excess_loss(out["value_logit"].float(), targets_t[index])
			if growth is not None:
				now = out["phi"].float().clamp(min=PHI_FLOOR).log().unsqueeze(-1)
				fresh["growth"][index] = excess_loss(out["growth_logit"].float(), ((growth_t[index] - now).clamp(-GROWTH_CAP, GROWTH_CAP) + GROWTH_CAP) / (2 * GROWTH_CAP))
			if end is not None:
				fresh["end"][index] = excess_loss(out["end_logit"].float().unsqueeze(-1), end_t[index].unsqueeze(-1))
			gpu_anchor_mb = max(gpu_anchor_mb, gpu_held(device))
		if device.type == "cuda":
			torch.cuda.synchronize(device)
		clock["prepass"] = time.perf_counter() - prepass_started
	masked = lambda x, valid: ((x * valid).sum(dim=0) / valid.sum(dim=0).clamp(min=1)).sum()
	value_fresh, heads = masked(fresh["share"], share_valid), float(c.N_VALUE_OFFSETS)
	if growth is not None:
		value_fresh, heads = value_fresh + config.growth_weight * masked(fresh["growth"], growth_valid), heads + config.growth_weight * c.N_VALUE_OFFSETS
	if end is not None:
		value_fresh, heads = value_fresh + masked(fresh["end"], end_valid.unsqueeze(-1)), heads + 1.0
	value_fresh = float(value_fresh) / heads
	del fresh
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
	if end is not None and end["weight"] > 0:  # the policy's credit moves toward the end's as V_end proves itself
		turn = choices(decisions)["kind"] == c.DecisionKind.TURN
		later = end_advantages(end_now.cpu().numpy(), end["step"], end["week"], end["real"], end["price"], turn, config.decision_price, end["outcome"])
		# V_end's step estimate of the end against the end itself, for the end credit's check by action type (a diagnostic
		# in the learner, as the 7-day credit's check: the end credit is two thirds of the advantage, and no check saw it)
		values = end_now.cpu().numpy()
		end_step_error = np.where(end["step"] >= 0, values[np.maximum(end["step"], 0)] - end["real"], np.nan).astype(np.float32)
		if attributed is not None:  # (two units, log growth and share: each standardized before they mix)
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
		noise = loss_noise(model, decisions, decode, convert, device, tensors, config)
	totals, steps, stopped, update_kl, recent = Counter(), 0, False, 0.0, deque(maxlen=KL_WINDOW)
	stopped_pass = 0  # the pass the KL control ended (0: none; b1, 2026-10-31: with passes=2, a second pass mostly cut short adds nothing)
	model.train()
	for pass_number in range(1, config.passes + 1):
		if stopped:
			break
		order = torch.randperm(n).numpy()
		if n >= size:
			order = order[: n - n % size]
		# Decoded a micro-batch at a time (a minibatch at once held ~4 of them decoded in pinned memory, a campaign
		# decision several MB: the PC's learner at 9.5 GB, 2026-10-30); the minibatch's gradient as before
		minibatches = [order[i : i + size] for i in range(0, len(order), size)]
		splits = [np.array_split(np.arange(len(rows)), max(1, min(config.micro_batches, len(rows)))) for rows in minibatches]
		decoded = _prefetched([rows[part] for rows, parts in zip(minibatches, splits) for part in parts], decode, device, convert, config.decode_threads)
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
			if growth is not None:
				counts["growth"] = growth_valid[index].sum(dim=0)
			if end is not None:
				counts["end"] = end_valid[index].sum(dim=0)
			optimizer.zero_grad(set_to_none=False)  # (in place: a recorded step adds into these .grad tensors)
			parts = []
			for part in parts_of_rows:
				sub, keep = index[part[0] : part[-1] + 1], whole[part[0] : part[-1] + 1]  # (array_split's parts: consecutive)
				waited = time.perf_counter()
				_, inputs = next(decoded)
				clock["wait"] += time.perf_counter() - waited
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
				if foresight_t is not None:
					batch.update({"fs:" + name: value[sub] for name, value in foresight_t.items()})
				if teacher is not None:  # the teacher's distributions on the same inputs
					with torch.no_grad():
						t_out = teacher_graphs.eval(inputs) if teacher_graphs is not None else teacher(inputs)
					batch.update({"t:" + name: t_out[name] for name in TEACHER_OUTPUTS}, teacher_weight=teacher_weight_t)
				if foresight_t is not None and not checked:
					# The gradient check, once an update, on an eager forward of its own, through torch.autograd.grad; every tensor
					# of that forward is dropped before the graphed step: its autograd graph alive (one tensor of it kept) made
					# the parameters' AccumulateGrad nodes reference the default stream, and the capture refused them ("a stale
					# reference to the default stream (stream 0) from warmup", 2026-10-30)
					check = micro_loss(model(inputs), batch, config, foresight_config)
					if check[3] is not None:
						checked = gradient_check(model, check[0] - check[3], check[3])
					del check
				mark = time.perf_counter()
				recorded = steps_graph.run(inputs, batch) if steps_graph is not None else None
				if recorded is not None:  # forward, loss and backward in one replay, its gradient added to .grad
					metrics, fs_parts = recorded
					clock["forward"] += time.perf_counter() - mark
				else:
					out = graphs.train(inputs) if graphs else model(inputs)
					clock["forward"] += time.perf_counter() - mark
					mark = time.perf_counter()
					loss, metrics, fs_parts, _ = micro_loss(out, batch, config, foresight_config)
					clock["losses"] += time.perf_counter() - mark
					mark = time.perf_counter()
					loss.backward()
					clock["backward"] += time.perf_counter() - mark
				if fs_parts is not None:  # (summed on the device: a float() here waited for the GPU every micro-batch)
					foresight_sum = fs_parts * share if foresight_sum is None else foresight_sum + fs_parts * share
					foresight_sums["steps_seen"] += share
				parts.append(metrics)  # (summed on the device: one sync a step, not six a micro-batch)
			mark = time.perf_counter()
			grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
			optimizer.step()
			stepped = time.perf_counter()
			clock["optimizer"] += stepped - mark
			*sums, grad_norm = torch.cat([torch.stack(parts).sum(dim=0), grad_norm.detach().float().reshape(1)]).tolist()  # (a sync: the minibatch's GPU work is done here)
			clock["steps"] += time.perf_counter() - stepped
			step = Counter(dict(zip(STEP_PARTS, sums)))
			gpu_mb = max(gpu_mb, gpu_held(device))
			steps += 1
			recent.append(step["kl"])
			update_kl = float(np.mean(recent))  # the update's KL: the last KL_WINDOW minibatches', not one's
			totals.update(step, grad_norm=grad_norm)
			if config.target_kl and update_kl > 1.5 * config.target_kl:
				stopped, stopped_pass = True, pass_number  # this update has moved the policy far enough
				break
		decoded.close()  # (its thread ends, its decoded batches freed)
	model.eval()
	gpu_mb = max(gpu_mb, gpu_held(device, release=True))  # (CUDA: the peak logged, the cache released: the other GPU process keeps its headroom)
	if config.target_kl and steps:
		if stopped:
			lr_scale = max(0.1, lr_scale / 1.5)
		elif update_kl < config.target_kl / 2:
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
		signal_ratio=signal_ratio(optimizer),  # (the steps' noise: the user's 2026-10-31 aim of 0.25+, with 4,096-decision steps)
		end_step_error=end_step_error,
		last_kl=update_kl,
		unsound=int(unsound.sum()),
		decisions=n,
		value_fresh=value_fresh,
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
		**({f"foresight_{name}": value / max(foresight_sums["steps_seen"], 1e-9) for name, value in zip(PART_NAMES, foresight_sum.tolist())} if foresight_sum is not None else {}),
		**checked,
		**{f"value_ev_d{k}": explained_variance(predicted[:, i], actual[:, i]) for i, k in enumerate(c.VALUE_OFFSETS)},
		**{f"value_parts_d{k}": residual_variances(predicted[:, i], actual[:, i]) for i, k in enumerate(c.VALUE_OFFSETS)},
		**value_fit(predicted[:, -1], actual[:, -1]),
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
