"""The optimizers' shared parts (docs/optimizer_plan.md, 2026-11-06): clipping by parameter group and its calibration,
and the share of a group's gradient that is signal. The campaign's Adam groups (campaign_model/train.py GRAD_GROUPS)
and the battle model's (battle_model/train.py BATTLE_GROUPS) are named ("name"), their parameters too ("param_names",
which a checkpoint keeps: the moments follow their parameters by name).

Why clip by group: one clip over every parameter at grad_clip always bound (the campaign's norm ~10 against 0.5), and
with Adam an always-binding clip is a per-step normalisation, harmless while one group makes the norm; but the norm was
made by the prior corrections (their x10 step) and the value heads' own parameters (the value loss's ~350 weight), so
their spikes shrank the trunk's gradient on that step. Each group is clipped at its own median norm, then all times
grad_clip / the total's median: Adam is blind to a constant scale, so that is the same training, at the scale its
moments were built at (clipped at their medians alone the trunk's gradients would be ~25x larger at once, and with a
second moment ~1,000 steps slow its steps would run up to ~6x their size for hours)."""
import numpy as np
import torch


def signal_fraction(optimizer):
	"""{group's name: f, "all": f over every group's weights}: the share of a group's gradient that is signal, from Adam's
	moments, f = (the mean of min(1, m^2 / v) - k) / (1 - k), k = (1 - beta1) / (1 + beta1) what pure noise gives
	(bias-corrected, over the weights that have had a gradient). For a gradient of steady mean mu and noise sigma,
	E[m^2] = mu^2 + k sigma^2 and v = mu^2 + sigma^2, so f = mu^2 / (mu^2 + sigma^2): 0 for noise, 1 for a constant
	gradient, at any beta1. The median |m| / sqrt(v) (signal_ratio) falls with a longer beta1 whatever the signal (pure
	noise: 0.155 at 0.9, 0.108 at 0.95): it cannot judge a beta1 change or compare groups of different betas."""
	names, sums, counts, floors = [], [], [], []
	for group in optimizer.param_groups:
		if "betas" not in group:  # (not Adam: no such state)
			continue
		beta1, beta2 = group["betas"]
		parts = []
		for p in group["params"]:
			state = optimizer.state.get(p, {})
			if "exp_avg" in state:
				step = float(state["step"])
				v = state["exp_avg_sq"].flatten() / (1 - beta2**step)
				live = v > 0
				parts.append(((state["exp_avg"].flatten()[live] / (1 - beta1**step)) ** 2 / v[live]).clamp(max=1.0))
		if parts:
			r2 = torch.cat(parts)
			names.append(group.get("name", str(len(names))))
			sums.append(r2.sum())
			counts.append(r2.numel())
			floors.append((1 - beta1) / (1 + beta1))
	if not names:
		return {}
	values = torch.stack(sums).tolist()  # (one read of the device)
	out = {name: (s / max(n, 1) - k) / (1 - k) for name, s, n, k in zip(names, values, counts, floors)}
	out["all"] = sum((s - k * n) / (1 - k) for s, n, k in zip(values, counts, floors)) / max(sum(counts), 1)
	return out


def clip_settings(items, setting="grad_clip_groups"):
	"""Clips by hand, items "trunk:0.4", "value:9", ..., "typical:12" (each group's clip and the total norm's median, a
	config tuple): {name: clip}, {} for none."""
	clips = {}
	for item in items:
		name, _, value = str(item).partition(":")
		clips[name.strip()] = float(value)
	if clips and "typical" not in clips:
		raise ValueError(f"{setting} needs typical:<the total norm's median> beside the groups' clips")
	return clips


def calibrated_clips(samples, names):
	"""The groups' clips from the norms of a calibration's steps (samples: [total, each of names' norm] a step): each
	group's median ("typical" the total's); a group that never had a gradient is not clipped."""
	columns = np.asarray(samples, dtype=np.float64)
	return {"typical": float(np.median(columns[:, 0])), **{name: float(np.median(columns[:, 1 + i])) or float("inf") for i, name in enumerate(names)}}


def stale_clips(clips, since):
	"""Whether calibrated clips (their "version": the version or epoch they were measured at; none before 2026-11-06's
	recalibration switch) are older than since (a config's grad_clip_since; 0: never stale)."""
	return clips is not None and since > 0 and clips.get("version", 0) < since


def clip_gradients(optimizer, grad_clip, clips=None, names=None):
	"""The gradient's clip before an optimizer step, on the device (no sync). Returns the total norm before it, and each
	group of names' (default: the optimizer's) norm before it and whether its clip bound, [len(names)] each, 0 for a
	group the optimizer lacks. clips ({group: its clip, "typical": the total norm's median}): each group at its own
	clip, then all times grad_clip / typical; None: the one clip at grad_clip over all of them (as torch's
	clip_grad_norm_)."""
	groups = optimizer.param_groups
	device = groups[0]["params"][0].device
	grads = [[p.grad for p in group["params"] if p.grad is not None] for group in groups]
	norms = torch.stack([torch.nn.utils.get_total_norm(g).float() if g else torch.zeros((), device=device) for g in grads])
	total = torch.linalg.vector_norm(norms)
	if clips:
		limit = torch.tensor([clips.get(group.get("name"), float("inf")) for group in groups], device=device)
		scale = (limit / (norms + 1e-6)).clamp(max=1.0) * (grad_clip / clips["typical"])
		bound = norms > limit
	else:
		scale = (grad_clip / (total + 1e-6)).clamp(max=1.0).expand(len(groups))
		bound = (total > grad_clip).expand(len(groups))
	for i, g in enumerate(grads):
		if g:
			torch._foreach_mul_(g, scale[i].to(g[0].dtype))
	names = names or [group.get("name", str(i)) for i, group in enumerate(groups)]
	place = {group.get("name"): i for i, group in enumerate(groups)}
	index = torch.tensor([place.get(name, len(groups)) for name in names], device=device)
	padded = lambda t: torch.cat([t.float(), torch.zeros(1, device=device)])[index]  # noqa: E731 (names' order, 0 where absent)
	return total, padded(norms), padded(bound)


def moments_by_name(optimizer, model, saved, saved_names):
	"""The saved optimizer state (saved: its state_dict; saved_names: its parameters' names in its numbering) loaded into
	optimizer by name, whatever its groups now; the groups' hyperparameters stay optimizer's (load_state_dict would
	take the checkpoint's). Returns (the names that started fresh, the saved names now gone)."""
	names = {id(p): name for name, p in model.named_parameters()}
	flat = [i for group in saved["param_groups"] for i in group["params"]]
	by_name = {name: saved["state"][i] for name, i in zip(saved_names, flat) if i in saved["state"]}
	state, param_groups, index, fresh, had = {}, [], 0, [], set(saved_names)
	for group in optimizer.param_groups:
		ids = []
		for p in group["params"]:
			name = names[id(p)]
			if name in by_name and by_name[name].get("exp_avg", p).shape == p.shape:
				state[index] = by_name[name]
			elif name not in had or name in by_name:  # (new, or of another shape now: fresh moments)
				fresh.append(name)
			ids.append(index)
			index += 1
		param_groups.append({**{key: value for key, value in group.items() if key != "params"}, "params": ids})
	optimizer.load_state_dict({"state": state, "param_groups": param_groups})
	return fresh, [name for name in saved_names if name not in set(names.values())]


class SignalShare:
	"""The share of an update's gradient that other battles repeat (e5, 2026-11-10: signal_ratio and signal_fraction read
	pure noise as signal at accumulate 24, an update's 4 steps being 4 passes over the same decisions: Adam's m held 4
	copies of one gradient, |m|/sqrt(v) 0.31 for noise). At each update's first optimizer step, before any step on its
	battles, each named group's clipped gradient g against S, a bias-corrected running mean (decay) of the earlier
	updates' first-step gradients, updated after the reading: (g . S) / |g|^2, raw and with both divided by the
	optimizer's sqrt(v-hat) (as Adam applies them). Its mean over ~25-50 updates is that share (expected ~0.005 as
	applied, 2026-11-10). A diagnostic: an error turns it off with one line, never an update's end; S starts again at
	a restart (not kept in the checkpoints)."""

	def __init__(self, decay=0.98):
		self.decay, self.mean, self.count, self.on = decay, {}, {}, True

	def read(self, optimizer):
		if not self.on:
			return {}
		try:
			readings, gradients = {}, {}
			for group in optimizer.param_groups:
				params = [p for p in group["params"] if p.grad is not None]
				if not params:
					continue
				name = group.get("name", "all")
				g = torch.cat([p.grad.detach().float().flatten() for p in params])
				scale = []
				for p in params:
					state = optimizer.state.get(p, {})
					if "exp_avg_sq" not in state or "step" not in state:
						scale = None
						break
					step = float(state["step"])
					scale.append((state["exp_avg_sq"].float().flatten() / (1 - group["betas"][1] ** step)).sqrt() + group.get("eps", 1e-8))
				gradients[name] = (g, torch.cat(scale) if scale else None)
			for name, (g, scale) in gradients.items():
				mean, count = self.mean.get(name), self.count.get(name, 0)
				if mean is not None and count and mean.shape == g.shape:
					s = mean / (1 - self.decay**count)
					readings[f"signal_share_{name}"] = float((g @ s) / (g @ g).clamp(min=1e-30))
					if scale is not None:
						ga, sa = g / scale, s / scale
						readings[f"signal_share_applied_{name}"] = float((ga @ sa) / (ga @ ga).clamp(min=1e-30))
				if mean is None or mean.shape != g.shape:
					mean, count = torch.zeros_like(g), 0
				self.mean[name] = self.decay * mean + (1 - self.decay) * g
				self.count[name] = count + 1
			return readings
		except Exception as error:  # noqa: BLE001
			print(f"signal_share: FAILURE ({type(error).__name__}: {str(error)[:200]}): the reading turned off for the run", flush=True)
			self.on = False
			return {}
