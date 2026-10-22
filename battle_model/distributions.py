"""Probability of full actions under the model's conditional outputs.

An action is a dict of int64 tensors [B], -1 where a field does not apply:
	type, hex (MOVE destination / MELEE from-hex), unit (MELEE / SHOOT target),
	spell, target_unit / target_hex (spell first target), target2_hex / target2_unit (second target).

log p(action) = log p(type) + log p(arguments | type), each argument conditional on the previous ones.
sample() is the reference for the C++ sampler in agents/AlphaBattleAI.
"""
import numpy as np
import torch

from .constants import N_SPELLS, SECOND_TARGET_KIND, ActionType, TargetKind

ACTION_FIELDS = ("type", "hex", "unit", "spell", "target_unit", "target_hex", "target2_hex", "target2_unit")

# Static second-target kind per spell, as tensors for vectorised lookups
_SECOND_KIND = torch.zeros(N_SPELLS, dtype=torch.long)
for _spell, _kind in SECOND_TARGET_KIND.items():
	_SECOND_KIND[_spell] = int(_kind)


def _take(table, index):
	"""table[b, index[b]] for a [B, N] table; index -1 gives 0 (field unused)."""
	picked = table.gather(1, index.clamp(min=0).unsqueeze(1)).squeeze(1)
	return torch.where(index >= 0, picked, torch.zeros_like(picked))


def _row(table, index):
	"""table[b, index[b], :] for a [B, N, M] table."""
	return table[torch.arange(table.shape[0], device=table.device), index.clamp(min=0)]


def log_prob(out, action):
	"""log p(action) for a batch, shape [B]."""
	t = action["type"]
	lp = _take(out["logp_type"], t)
	is_move = t == ActionType.MOVE
	is_melee = t == ActionType.MELEE
	is_shoot = t == ActionType.SHOOT
	is_spell = t == ActionType.HERO_SPELL

	lp = lp + torch.where(is_move, _take(out["logp_move_hex"], action["hex"]), 0.0)
	melee = _take(out["logp_melee_target"], action["unit"]) + _take(_row(out["logp_melee_from"], action["unit"]), action["hex"])
	lp = lp + torch.where(is_melee, melee, 0.0)
	lp = lp + torch.where(is_shoot, _take(out["logp_shoot_target"], action["unit"]), 0.0)

	spell = action["spell"]
	spell_lp = _take(out["logp_spell"], spell)
	spell_lp = spell_lp + _take(_row(out["logp_spell_unit"], spell), action["target_unit"])
	spell_lp = spell_lp + _take(_row(out["logp_spell_hex"], spell), action["target_hex"])
	spell_lp = spell_lp + _take(_row(out["logp_spell2_hex"], action["target_unit"]), action["target2_hex"])
	spell_lp = spell_lp + _take(_row(out["logp_spell2_unit"], action["target_unit"]), action["target2_unit"])
	return lp + torch.where(is_spell, spell_lp, 0.0)


def _entropy(logp):
	"""Entropy over the last axis; rows with no legal entry have entropy 0.

	Illegal entries are -inf: zero them before multiplying, 0 * -inf would make the gradient NaN.
	"""
	finite_logp = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
	return -(logp.exp() * finite_logp).sum(dim=-1)


def _expect(logp, values):
	"""E[values] under the distribution exp(logp) over the last axis."""
	return (logp.exp() * values).sum(dim=-1)


def entropy(out):
	"""Exact entropy of the full action distribution, shape [B]."""
	p_type = out["logp_type"].exp()
	h = _entropy(out["logp_type"])
	h = h + p_type[:, ActionType.MOVE] * _entropy(out["logp_move_hex"])
	h = h + p_type[:, ActionType.MELEE] * (_entropy(out["logp_melee_target"]) + _expect(out["logp_melee_target"], _entropy(out["logp_melee_from"])))
	h = h + p_type[:, ActionType.SHOOT] * _entropy(out["logp_shoot_target"])

	# Per spell: entropy of its first target (one of the two rows is empty) plus, for
	# two-target spells, the expected entropy of the second target given the first.
	second_kind = _SECOND_KIND.to(out["logp_spell"].device)
	h2_hex = _entropy(out["logp_spell2_hex"])  # [B, U]
	h2_unit = _entropy(out["logp_spell2_unit"])
	h2 = torch.where(second_kind[None, :, None] == TargetKind.HEX, h2_hex[:, None, :], torch.where(second_kind[None, :, None] == TargetKind.UNIT, h2_unit[:, None, :], 0.0))  # [B, S, U]
	per_spell = _entropy(out["logp_spell_unit"]) + _entropy(out["logp_spell_hex"]) + _expect(out["logp_spell_unit"], h2)
	h_spell = _entropy(out["logp_spell"]) + _expect(out["logp_spell"], per_spell)
	return h + p_type[:, ActionType.HERO_SPELL] * h_spell


def _choose(logp, rng, greedy):
	logp = np.asarray(logp, dtype=np.float64)
	legal = np.isfinite(logp)
	if not legal.any():
		raise ValueError("no legal entry to choose from")
	if greedy:
		return int(np.argmax(np.where(legal, logp, -np.inf)))
	p = np.where(legal, np.exp(logp - logp[legal].max()), 0.0)
	return int(rng.choice(len(p), p=p / p.sum()))


def sample(out, b=0, rng=None, greedy=False):
	"""Sample one full action for batch row b from numpy/torch outputs. Returns a dict of ints."""
	rng = rng or np.random.default_rng()
	o = {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))[b] for k, v in out.items()}
	a = dict.fromkeys(ACTION_FIELDS, -1)
	a["type"] = t = _choose(o["logp_type"], rng, greedy)
	if t == ActionType.MOVE:
		a["hex"] = _choose(o["logp_move_hex"], rng, greedy)
	elif t == ActionType.MELEE:
		a["unit"] = u = _choose(o["logp_melee_target"], rng, greedy)
		a["hex"] = _choose(o["logp_melee_from"][u], rng, greedy)
	elif t == ActionType.SHOOT:
		a["unit"] = _choose(o["logp_shoot_target"], rng, greedy)
	elif t == ActionType.HERO_SPELL:
		a["spell"] = s = _choose(o["logp_spell"], rng, greedy)
		if np.isfinite(o["logp_spell_unit"][s]).any():
			a["target_unit"] = u = _choose(o["logp_spell_unit"][s], rng, greedy)
			second = SECOND_TARGET_KIND.get(s)
			if second == TargetKind.HEX:
				a["target2_hex"] = _choose(o["logp_spell2_hex"][u], rng, greedy)
			elif second == TargetKind.UNIT:
				a["target2_unit"] = _choose(o["logp_spell2_unit"][u], rng, greedy)
		elif np.isfinite(o["logp_spell_hex"][s]).any():
			a["target_hex"] = _choose(o["logp_spell_hex"][s], rng, greedy)
	return a


def is_legal(action, masks, b=0):
	"""Check a decoded action against the engine masks (numpy, batch row b)."""
	m = {k: np.asarray(v)[b] for k, v in masks.items()}
	t = action["type"]
	if t < 0 or not m["m_type"][t]:
		return False
	if t == ActionType.MOVE:
		return bool(m["m_move_hex"][action["hex"]])
	if t == ActionType.MELEE:
		return bool(m["m_melee_target"][action["unit"]] and m["m_melee_from"][action["unit"], action["hex"]])
	if t == ActionType.SHOOT:
		return bool(m["m_shoot_target"][action["unit"]])
	if t == ActionType.HERO_SPELL:
		s = action["spell"]
		if not m["m_spell"][s]:
			return False
		if action["target_unit"] >= 0 and not m["m_spell_unit"][s, action["target_unit"]]:
			return False
		if action["target_hex"] >= 0 and not m["m_spell_hex"][s, action["target_hex"]]:
			return False
		if action["target2_hex"] >= 0 and not m["m_spell2_hex"][action["target_unit"], action["target2_hex"]]:
			return False
		if action["target2_unit"] >= 0 and not m["m_spell2_unit"][action["target_unit"], action["target2_unit"]]:
			return False
	return True


def stack_actions(actions):
	"""List of action dicts -> dict of int64 tensors [B]."""
	return {f: torch.tensor([a[f] for a in actions], dtype=torch.long) for f in ACTION_FIELDS}


def enumerate_actions(masks, b=0):
	"""Every legal full action of batch row b, from the masks alone (for tests and dump checks)."""
	m = {k: np.asarray(v)[b] for k, v in masks.items()}
	blank = dict.fromkeys(ACTION_FIELDS, -1)
	actions = []
	for t in np.flatnonzero(m["m_type"]):
		t = int(t)
		if t == ActionType.MOVE:
			actions += [{**blank, "type": t, "hex": int(h)} for h in np.flatnonzero(m["m_move_hex"])]
		elif t == ActionType.MELEE:
			for u in np.flatnonzero(m["m_melee_target"]):
				actions += [{**blank, "type": t, "unit": int(u), "hex": int(h)} for h in np.flatnonzero(m["m_melee_from"][u])]
		elif t == ActionType.SHOOT:
			actions += [{**blank, "type": t, "unit": int(u)} for u in np.flatnonzero(m["m_shoot_target"])]
		elif t == ActionType.HERO_SPELL:
			for s in np.flatnonzero(m["m_spell"]):
				s = int(s)
				base = {**blank, "type": t, "spell": s}
				if m["m_spell_unit"][s].any():
					second = SECOND_TARGET_KIND.get(s)
					for u in np.flatnonzero(m["m_spell_unit"][s]):
						first = {**base, "target_unit": int(u)}
						if second == TargetKind.HEX:
							actions += [{**first, "target2_hex": int(h)} for h in np.flatnonzero(m["m_spell2_hex"][u])]
						elif second == TargetKind.UNIT:
							actions += [{**first, "target2_unit": int(v)} for v in np.flatnonzero(m["m_spell2_unit"][u])]
						else:
							actions.append(first)
				elif m["m_spell_hex"][s].any():
					actions += [{**base, "target_hex": int(h)} for h in np.flatnonzero(m["m_spell_hex"][s])]
				else:
					actions.append(base)
		else:
			actions.append({**blank, "type": t})
	return actions
