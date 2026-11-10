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
campaign, never beside a campaign update (aux_allowed).

Nets (e5's review): a step whose loss is not finite is skipped (aux_skipped); a phase that ends with a weight or a
reading not finite, or with the policy moved more than kl_max from where it started (on the held-out battles), is
undone, its weights the snapshot's again and its Adam reset (aux_rejected); an error inside it puts the weights back
and re-raises, and guarded_run turns the phase off for the run with one "aux: FAILURE" line. A cold Adam's first steps
are full-rate sign steps on one batch's noise: its rates rise linearly over its first WARMUP steps, and its state is
kept in the checkpoints (state_dict), so a restart does not start cold.

The KL's net never sees the value: with the replay's labels shuffled across battles (2026-11-10, 5 splits by game) the
phase doubled the held-out value error and no phase was rejected. So each phase is read on the battles that come after
it (judge: the models before and after it, on the next update's battles, which neither trained on: aux_fresh_change),
and a window of WINDOW readings whose mean is above 0 by more than HARM_SE standard errors stops the phase for the run.

The readings' held-out battles are the newest version's, which PPO trained on just before: they hold the KL's check,
but their value error before and after a phase is no measure of it (its steps toward the other battles undo part of
PPO's fit to these: a bias against the phase, of the order of its gain). Neither level is comparable with
value_loss_fresh (which the trainer reads before PPO trains on its battles), the judged series."""

WARMUP = 16  # steps of a cold Adam over which its rates rise from 1/WARMUP to their full value
WINDOW, HARM_SE = 16, 3.0  # AuxValuePhase.judge: fresh readings to a verdict; standard errors of harm that stop the phase
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
		self.adds = 0  # add() calls so far; an entry's "add" is the call that brought it (judge: the battles after a phase)

	def add(self, items, version):
		self.adds += 1
		for item in items:
			data = item.get("data")
			if not data or not item.get("decisions"):
				continue
			rows = spread(item["decisions"], self.per_battle, self.rng)
			entry = {"inputs": take_rows(data["inputs"], rows), "value_labels": np.asarray(data["value_labels"])[rows], "version": int(item.get("version", version)), "decisions": len(rows), "add": self.adds}
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

	def chosen(self, exclude_version=None, only_version=None, only_add=None):
		return [(i, e) for i, e in enumerate(self.entries) if (exclude_version is None or e["version"] != exclude_version) and (only_version is None or e["version"] == only_version) and (only_add is None or e["add"] == only_add)]

	def decisions(self, exclude_version=None, only_version=None, only_add=None):
		return sum(e["decisions"] for _, e in self.chosen(exclude_version, only_version, only_add))

	def batch(self, n, device, exclude_version=None, only_version=None, only_add=None):
		"""n decisions drawn at random (without replacement) from the entries (those of exclude_version left out, or only
		those of only_version, or of the add() call only_add), as a batch on device: "inputs" (int64 ones as the entries
		keep them: widened() for the model), "value_labels", "weight" when the entries have it; None when there are none."""
		pool = [(i, r) for i, e in self.chosen(exclude_version, only_version, only_add) for r in range(e["decisions"])]
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

	def __init__(self, groups, rates, betas=(0.9, 0.999), kl_max=3e-4):
		self.trunk = [p for _, p in groups.get("trunk", [])]
		self.params = [p for params in groups.values() for _, p in params]
		self.groups, self.rates, self.betas, self.kl_max = groups, dict(rates), betas, kl_max
		self.on, self.pending, self.fresh_readings = True, None, 0  # (judge's)
		self.reset()

	def reset(self):
		"""A cold Adam (its rates warmed up again over WARMUP steps); judge's window starts again, its first one not judged
		(e5: the warm-up's steps, cosine -0.28 between them in the real battles' run, are no verdict on the phase)."""
		self.optimizer = torch.optim.Adam([{"params": [p for _, p in params], "name": name, "lr": self.rates[name]} for name, params in self.groups.items() if params], betas=self.betas)
		self.steps_done = 0
		self.changes, self.warm_window = [], True

	def state_dict(self):
		return {"optimizer": self.optimizer.state_dict(), "steps_done": self.steps_done}

	def load_state_dict(self, state):
		"""A checkpoint's phase state; a cold one kept when it does not fit (other groups or shapes), said once."""
		try:
			self.optimizer.load_state_dict(state["optimizer"])
			self.steps_done = int(state.get("steps_done", 0))
		except (KeyError, ValueError, RuntimeError) as error:
			print(f"aux: the checkpoint's phase state does not fit ({str(error)[:200]}): a cold one", flush=True)
			self.reset()

	def judge(self, replay, device, ppo_config, n=512):
		"""The last phase read on the battles that came after it (the replay's newest add(): neither the model before the
		phase nor after it trained on them): value_excess's relative change there, aux_fresh_change. Over WINDOW readings
		(the first window after a cold Adam not judged), a mean above 0 by more than HARM_SE standard errors turns the
		phase off for the run, said in one "aux: FAILURE" line with the readings and the mean it saw."""
		pending, self.pending = self.pending, None
		if pending is None or replay.adds <= pending[2]:
			return {}
		check = replay.batch(n, device, only_add=replay.adds)
		if check is None:
			return {}
		with torch.no_grad():
			inputs = widened(check["inputs"])
			old, new = (float(value_excess(m(inputs), check["value_labels"], ppo_config).mean()) for m in pending[:2])
		change = (new - old) / old if old > 0 else float("nan")
		if not np.isfinite(change):
			return {"aux_fresh_change": change}
		self.fresh_readings += 1
		self.changes.append(change)
		if len(self.changes) >= WINDOW:
			window, self.changes = np.array(self.changes), []
			if self.warm_window:
				self.warm_window = False
			else:
				mean, se = float(window.mean()), float(window.std(ddof=1) / np.sqrt(len(window)))
				if mean > HARM_SE * se:
					self.on = False
					print(f"aux: FAILURE: the value's phase raised the value error on the battles that came after it, {mean:+.2%} +- {se:.2%} (se) over its fresh readings {self.fresh_readings - len(window) + 1}-{self.fresh_readings} of this start (aux_fresh_change; more than {HARM_SE:g} se); the phase off for the run", flush=True)
		return {"aux_fresh_change": change}

	def run(self, model, replay, device, steps=8, batch_size=512, kl_weight=1.0, seconds=30.0, clip=1.0, ppo_config=None, held_out_version=None, held_out=512):
		"""The last phase judged (judge), then up to `steps` steps (and `seconds`) on replay batches of batch_size (the
		battles of held_out_version left out of them: the readings' held-out set); the readings (aux_*): steps, seconds,
		mean value loss and KL, the cosine between successive steps' trunk gradients (signal when steady; ~0 is noise),
		the trunk's move over the phase (of its norm), value_excess on the held-out battles before and after, and the
		policy's KL from the snapshot there."""
		readings = {"aux_steps": 0, "aux_skipped": 0, "aux_rejected": 0}
		ppo_config = ppo_config if ppo_config is not None else PPOConfig()
		readings.update(self.judge(replay, device, ppo_config, held_out))
		if not self.on or replay.decisions(exclude_version=held_out_version) < batch_size:
			return readings
		started = time.monotonic()
		snapshot = copy.deepcopy(model).eval()  # (the policy as the phase starts: what the KL holds it to)
		for p in snapshot.parameters():
			p.requires_grad_(False)

		def undo():
			model.load_state_dict(snapshot.state_dict())  # (in place: the PPO optimizer's parameters are the same tensors)
			model.eval()

		try:
			readings.update(self._run(model, snapshot, replay, device, steps, batch_size, kl_weight, seconds, clip, ppo_config, held_out_version, held_out, started))
		except BaseException:
			undo()
			raise
		rejected = not all(np.isfinite(readings[k]) for k in ("aux_value_loss", "aux_kl", "aux_kl_check")) or not all(bool(torch.isfinite(p).all()) for p in model.parameters())
		rejected = readings["aux_steps"] > 0 and (rejected or readings["aux_kl_check"] > self.kl_max)  # (no step taken: nothing to undo)
		if rejected:  # (the phase undone, its Adam cold again: what led here must not carry on)
			undo()
			self.reset()
			readings["aux_rejected"] = 1
		elif readings["aux_steps"] > 0:  # (the models before and after it, read on the battles that come next: judge)
			after = copy.deepcopy(model).eval()
			for p in after.parameters():
				p.requires_grad_(False)
			self.pending = (snapshot, after, replay.adds)
		readings["aux_seconds"] = round(time.monotonic() - started, 2)
		return readings

	def _run(self, model, snapshot, replay, device, steps, batch_size, kl_weight, seconds, clip, ppo_config, held_out_version, held_out, started):
		readings = {"aux_steps": 0, "aux_skipped": 0}
		check = replay.batch(held_out, device, only_version=held_out_version) if held_out_version is not None else None
		if check is None:  # (no held-out battles: the KL's check on a draw of the replay as the phase starts)
			check = replay.batch(held_out, device, exclude_version=held_out_version)

		def held_out_readings():
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
			loss = value + kl_weight * divergence
			self.optimizer.zero_grad(set_to_none=True)
			if not bool(torch.isfinite(loss)):
				readings["aux_skipped"] += 1
				continue
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.params, clip)
			warm = min(1.0, (self.steps_done + 1) / WARMUP)
			for group in self.optimizer.param_groups:
				group["lr"] = self.rates[group["name"]] * warm
			if self.trunk:
				grad = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in self.trunk])
				if previous is not None:
					cosines.append(torch.nn.functional.cosine_similarity(grad, previous, dim=0))
				previous = grad.detach().clone()
			self.optimizer.step()
			self.steps_done += 1
			values.append(value.detach())
			kls.append(divergence.detach())
			readings["aux_steps"] += 1
		model.eval()
		readings["aux_excess_after"], readings["aux_kl_check"] = held_out_readings()
		readings["aux_kl_held_out"] = readings["aux_kl_check"] if held_out_version is not None else float("nan")
		readings["aux_value_loss"] = float(torch.stack(values).mean()) if values else float("nan")
		readings["aux_kl"] = float(torch.stack(kls).mean()) if kls else float("nan")
		readings["aux_grad_cosine"] = float(torch.stack(cosines).mean()) if cosines else float("nan")
		if trunk_before is not None:
			after = torch.cat([p.detach().flatten() for p in self.trunk])
			readings["aux_trunk_move"] = float((after - trunk_before).norm() / trunk_before.norm().clamp(min=1e-12))
		return readings


def guarded_run(phase, *args, **kwargs):
	"""(readings, phase or None): phase.run's, or, when it raised (its weights already put back), one "aux: FAILURE" line
	and None: the phase off for the run (the value's phase must never end the training); None too when the phase stopped
	itself (judge)."""
	try:
		readings = phase.run(*args, **kwargs)
	except Exception as error:  # noqa: BLE001
		print(f"aux: FAILURE: the value's phase failed ({type(error).__name__}: {str(error)[:300]}); its weights put back, the phase off for the run", flush=True)
		return {"aux_failed": 1}, None
	return readings, (phase if phase.on else None)
