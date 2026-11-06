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
			if name in by_name:
				state[index] = by_name[name]
			elif name not in had:
				fresh.append(name)
			ids.append(index)
			index += 1
		param_groups.append({**{key: value for key, value in group.items() if key != "params"}, "params": ids})
	optimizer.load_state_dict({"state": state, "param_groups": param_groups})
	return fresh, [name for name in saved_names if name not in set(names.values())]
