"""The battle value's own phase (2026-11-10; Phasic Policy Gradient's auxiliary phase, Cobbe et al. 2020).

Measured on joint7 (e5, a stop's buffer of 168 battles): the value loss's gradient on the shared trunk was 0.4% of a
PPO step's, its direction steady (cosine +0.73 between steps) but its share of the trunk's Adam second moment ~0.1%:
the trunk's features were never trained by V. And the 276-weight value readout, fitted to each update's ~117 battles
at 10x the rate, jittered between checkpoints (+10% value loss on fixed battles on average, +25% at worst).

So V gets a phase of its own between the policy's updates, in the time the trainer waits for battles (97% of it):
	loss = value_loss(V(s), the battle's labels) + kl_weight x KL(policy as the phase starts || policy now)
over a replay of the last updates' battles (ValueReplay: a few decisions of each, spread over its rounds; the labels
are the battle's outcome and round scores, losses.value_targets, so they never go stale as the policy moves: only the
states' mix ages). Its own Adam (never the PPO optimizer's, whose second moment is the policy's noise), every
parameter at its group's rate (battle_group_lr: the user's cap on the trunk). The KL holds the policy where the phase
found it: what the trunk learns for V, the policy heads follow without changing what they choose.

Only between a battle update and the next: with the campaign learner "turns" (its updates on the same thread) or no
campaign, never beside a campaign update (aux_allowed)."""
import collections
import copy
import random
import time

import numpy as np
import torch

from .data import Sparse, widened
from .distributions import kl
from .losses import PPOConfig, value_excess, value_loss
from .supply import dense_on


def take_rows(value, rows):
	"""Rows `rows` (sorted indices into the first axis) of an item's array: a numpy array's, a Sparse's (its entries of
	those rows, renumbered), each of a dict's."""
	if isinstance(value, dict):
		return {key: take_rows(v, rows) for key, v in value.items()}
	if isinstance(value, Sparse):
		per_row = int(np.prod(value.shape[1:])) if len(value.shape) > 1 else 1
		row_of = value.index // per_row
		keep = np.isin(row_of, rows)
		out = Sparse.__new__(Sparse)
		out.shape, out.dtype = (len(rows),) + tuple(value.shape[1:]), value.dtype
		out.index = (np.searchsorted(rows, row_of[keep]) * per_row + value.index[keep] % per_row).astype(value.index.dtype)
		out.values = None if value.values is None else value.values[keep]
		return out
	return value[rows]


def spread(n, k, rng):
	"""k distinct sorted rows of n, one drawn in each of k equal stretches (a battle's early, middle and late rounds)."""
	if k >= n:
		return np.arange(n)
	edges = np.linspace(0, n, k + 1)
	rows = sorted({rng.randrange(int(edges[i]), max(int(edges[i]) + 1, int(edges[i + 1]))) for i in range(k)})
	return np.array(rows, dtype=np.int64)


def nbytes(value):
	if isinstance(value, dict):
		return sum(nbytes(v) for v in value.values())
	if isinstance(value, (Sparse, np.ndarray)):
		return int(value.nbytes)
	return 0


def join(values, device):
	"""Entries' arrays joined along their rows, as tensors on device (a Sparse one made dense there: supply.dense_on)."""
	if isinstance(values[0], dict):
		return {key: join([v[key] for v in values], device) for key in values[0]}
	if any(isinstance(v, Sparse) for v in values):
		return dense_on(values, device)
	return torch.from_numpy(np.ascontiguousarray(np.concatenate(values))).to(device)


class ValueReplay:
	"""The battles of the last updates for the value's phase: of each battle an update took (ActorPool.take), up to
	per_battle decisions spread over it, their inputs and value labels; the oldest out past max_decisions, and every
	entry more than max_age versions older than the newest added."""

	def __init__(self, max_decisions=20_000, max_age=8, per_battle=5, seed=0):
		self.entries = collections.deque()
		self.max_decisions, self.max_age, self.per_battle = max_decisions, max_age, per_battle
		self.rng = random.Random(seed)
		self.held = 0  # decisions held
		self.bytes = 0

	def add(self, items, version):
		for item in items:
			data = item.get("data")
			if not data or not item.get("decisions"):
				continue
			rows = spread(item["decisions"], self.per_battle, self.rng)
			entry = {"inputs": take_rows(data["inputs"], rows), "value_labels": np.asarray(data["value_labels"])[rows], "version": int(item.get("version", version)), "decisions": len(rows)}
			if data.get("weight") is not None:
				entry["weight"] = np.asarray(data["weight"])[rows]
			entry["bytes"] = nbytes(entry["inputs"]) + entry["value_labels"].nbytes
			self.entries.append(entry)
			self.held += entry["decisions"]
			self.bytes += entry["bytes"]
		self._trim(version)

	def _trim(self, version):
		while self.entries and (self.held > self.max_decisions or version - self.entries[0]["version"] > self.max_age):
			old = self.entries.popleft()
			self.held -= old["decisions"]
			self.bytes -= old["bytes"]

	def decisions(self, exclude_version=None, only_version=None):
		return sum(e["decisions"] for e in self.entries if (exclude_version is None or e["version"] != exclude_version) and (only_version is None or e["version"] == only_version))

	def batch(self, n, device, exclude_version=None, only_version=None):
		"""n decisions drawn at random (without replacement) from the entries (those of exclude_version left out, or only
		those of only_version), as a batch on device: "inputs" (int64 ones as the entries keep them: widened() for the
		model), "value_labels", "weight" when the entries have it; None when there are none."""
		pool = [(i, r) for i, e in enumerate(self.entries) if (exclude_version is None or e["version"] != exclude_version) and (only_version is None or e["version"] == only_version) for r in range(e["decisions"])]
		if not pool:
			return None
		chosen = sorted(self.rng.sample(pool, min(n, len(pool))))
		by_entry = collections.defaultdict(list)
		for i, r in chosen:
			by_entry[i].append(r)
		parts = [{key: take_rows(self.entries[i][key], np.array(rows)) for key in ("inputs", "value_labels", "weight") if key in self.entries[i]} for i, rows in sorted(by_entry.items())]
		keys = set.intersection(*(set(p) for p in parts))
		return join([{key: p[key] for key in sorted(keys)} for p in parts], device)


def aux_allowed(campaign, config):
	"""Whether the value's phase may run now: never beside a campaign update. With no campaign learner, or one on this
	thread ("turns": its updates run between the battle updates, not during them), the phase has the device alone."""
	return campaign is None or getattr(getattr(config, "campaign", None), "learner", "turns") == "turns"


class AuxValuePhase:
	"""The value's phase (the module's docstring), its own Adam over the model's groups (groups: {name: [(name,
	parameter)]}, train.battle_parameter_groups; rates: {group: its rate})."""

	def __init__(self, groups, rates, betas=(0.9, 0.999)):
		self.trunk = [p for _, p in groups.get("trunk", [])]
		self.params = [p for params in groups.values() for _, p in params]
		self.optimizer = torch.optim.Adam([{"params": [p for _, p in params], "name": name, "lr": rates[name]} for name, params in groups.items() if params], betas=betas)

	def run(self, model, replay, device, steps=8, batch_size=512, kl_weight=1.0, seconds=30.0, clip=1.0, ppo_config=None, held_out_version=None, held_out=512):
		"""Up to `steps` steps (and `seconds`) on replay batches of batch_size (the battles of held_out_version left out
		of them: the readings' held-out set); the readings (aux_*): steps, seconds, mean value loss and KL, the cosine
		between successive steps' trunk gradients (signal when steady; ~0 is noise), the trunk's move over the phase (of
		its norm), value_excess on the held-out battles before and after, and the policy's KL from the snapshot there."""
		readings = {"aux_steps": 0}
		ppo_config = ppo_config if ppo_config is not None else PPOConfig()
		if replay.decisions(exclude_version=held_out_version) < batch_size:
			return readings
		started = time.monotonic()
		snapshot = copy.deepcopy(model).eval()  # (the policy as the phase starts: what the KL holds it to)
		for p in snapshot.parameters():
			p.requires_grad_(False)
		check = replay.batch(held_out, device, only_version=held_out_version) if held_out_version is not None else None

		def held_out_readings():
			if check is None:
				return float("nan"), float("nan")
			with torch.no_grad():
				inputs = widened(check["inputs"])
				out, snap = model(inputs), snapshot(inputs)
				excess = value_excess(out, check["value_labels"], ppo_config)
				return float(excess.mean()), float(kl(snap, out).mean())

		model.eval()
		readings["aux_excess_before"], _ = held_out_readings()
		trunk_before = torch.cat([p.detach().flatten() for p in self.trunk]) if self.trunk else None
		values, kls, cosines, previous = [], [], [], None
		model.train()
		for _ in range(steps):
			if time.monotonic() - started > seconds:
				break
			batch = replay.batch(batch_size, device, exclude_version=held_out_version)
			inputs = widened(batch["inputs"])
			out = model(inputs)
			with torch.no_grad():
				snap = snapshot(inputs)
			value, _ = value_loss(out, batch["value_labels"], ppo_config, batch.get("weight"))
			divergence = kl(snap, out).mean()
			self.optimizer.zero_grad(set_to_none=True)
			(value + kl_weight * divergence).backward()
			torch.nn.utils.clip_grad_norm_(self.params, clip)
			if self.trunk:
				grad = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in self.trunk])
				if previous is not None:
					cosines.append(torch.nn.functional.cosine_similarity(grad, previous, dim=0))
				previous = grad.detach().clone()
			self.optimizer.step()
			values.append(value.detach())
			kls.append(divergence.detach())
			readings["aux_steps"] += 1
		model.eval()
		readings["aux_excess_after"], readings["aux_kl_held_out"] = held_out_readings()
		readings["aux_value_loss"] = float(torch.stack(values).mean()) if values else float("nan")
		readings["aux_kl"] = float(torch.stack(kls).mean()) if kls else float("nan")
		readings["aux_grad_cosine"] = float(torch.stack(cosines).mean()) if cosines else float("nan")
		if trunk_before is not None:
			after = torch.cat([p.detach().flatten() for p in self.trunk])
			readings["aux_trunk_move"] = float((after - trunk_before).norm() / trunk_before.norm().clamp(min=1e-12))
		readings["aux_seconds"] = round(time.monotonic() - started, 2)
		return readings
