"""Starting biases, in human-readable files next to this one (battle_model/priors/*.csv, columns name,id,bias), to
review and edit: the models' bias parameters start at these values and are learned from there.

	action_types.csv   per action type, the type head's bias: e.g. -6 for retreat and surrender (giving up is offered
	                   at every decision: at -6 it ends ~4% of untrained battles, at -2 two in three)
	spells.csv         per spell, the spell head's bias: some spells are usually better than others
	MASTERY_BIAS       per spell, by our hero's mastery of the spell (none, basic, advanced, expert, as the engine casts
	                   it): added to that spell's logit, learned per spell and mastery, starting the same for all

tests/test_priors.py checks the files: every id once, names as the engine's.
"""
import csv
from pathlib import Path

import torch

from .constants import N_ACTION_TYPES, N_SPELLS, ActionType

PRIORS_DIR = Path(__file__).resolve().parent / "priors"
MASTERY_BIAS = (0.0, 0.25, 0.5, 0.75)


def read(name):
	"""{id: (name, bias)} from a priors file."""
	with (PRIORS_DIR / name).open(newline="") as f:
		return {int(row["id"]): (row["name"], float(row["bias"])) for row in csv.DictReader(f)}


def biases(name, count):
	"""The biases of a priors file as a tensor [count]; ids the file does not list get 0."""
	rows = read(name)
	return torch.tensor([rows.get(i, ("", 0.0))[1] for i in range(count)])


def action_types():
	return biases("action_types.csv", N_ACTION_TYPES)


# RETREAT and SURRENDER start at about never (action_types.csv: -6), and that bias lifts as our side gets much weaker:
# by escape_relief(z) of it, z the log of our health x damage per round over the enemy's (features.side_strengths,
# all units; Lanchester's square law: an army twice the size has 4x). Even or stronger it stays ~-6, at 3x the
# enemy's army (9x) ~-5.3, at 5x (25x) half (-3), at 10x (100x) ~-0.3, about as likely as waiting. Considered then,
# never forced; the learned type head adds to it. Low on purpose: a losing side gets weaker every turn and decides many
# times, so a lift at 3x (tried first: half of it there) ended a third of the first epochs' lost battles in an escape.
ESCAPE_AT = -3.2  # log(1 / 25)
ESCAPE_WIDTH = 0.5


def escape_relief(z):
	"""The share (0..1) of the escapes' bias lifted at strength log-ratio z (above)."""
	return torch.sigmoid((ESCAPE_AT - z) / ESCAPE_WIDTH)


# Escaping while ahead (the user, 2026-11-06: "another -3 for the logit if we are ahead and maybe something that scales
# nicely with how much we are ahead"): both escapes' logits get w_step x tanh(lead / ESCAPE_AHEAD_WIDTH) + w_scale x
# lead, lead = max(0, V), V the value head's own estimate of the battle's end (-1..1; the failure count's "ahead" is
# V > 0): 0 even or behind, -1.5 at V 0.05, -2.6 at 0.1, -3.5 at 0.2, -4.5 at 0.5, -6 at a sure win. The relief above
# judges by the armies' strength alone, and the learned type head's RETREAT row (its norm 7x MOVE's by epoch 8476) could
# lift an escape past the -6 while V said we were winning (~2% of escapes, 2026-11-06). Learned from (-3, -3), as the
# idle shooter's -5: a prior that presses, no mask
ESCAPE_AHEAD = (-3.0, -3.0)  # (w_step, w_scale)
ESCAPE_AHEAD_WIDTH = 0.1


def escape_ahead(value, weights):
	"""[B] the logit both escapes get at the value head's estimate value [B] (above); weights: (w_step, w_scale)."""
	lead = value.clamp(min=0.0)
	return weights[0] * torch.tanh(lead / ESCAPE_AHEAD_WIDTH) + weights[1] * lead


def spells():
	return biases("spells.csv", N_SPELLS)


def mastery():
	"""[N_SPELLS, 4]: MASTERY_BIAS for every spell."""
	return torch.tensor(MASTERY_BIAS).expand(N_SPELLS, -1).clone()


# A unit that can shoot at full damage shoots: BattleAI's rule, as the type rules' prior (the user's learnable penalty for
# idle shooters, 2026-11-05). DEFEND starts this much lower there (features.SHOT_TYPE_FEATURES' full_shot); every other
# type rule starts at 0. In joint2 the 5% of shooter states where an idle type beat SHOOT were behind by up to ~1 logit.
# -2 first, -5 since the evening (the user: "maybe have -5 there just in case")
IDLE_SHOT_BIAS = -5.0


def type_rules(names):
	"""[N_ACTION_TYPES, len(names)]: the type rules' start (model.ActionHeads.type_rules); names: features.TYPE_FEATURES."""
	weight = torch.zeros(N_ACTION_TYPES, len(names))
	weight[ActionType.DEFEND, names.index("full_shot")] = IDLE_SHOT_BIAS
	return weight


def widen_type_rules(state_dict, key, names):
	"""A checkpoint (or a league snapshot) from before features.SHOT_TYPE_FEATURES: its type rules' columns kept, the
	new ones at their start (type_rules)."""
	saved = state_dict.get(key)
	if saved is not None and saved.shape[-1] < len(names):
		widened = type_rules(names).to(device=saved.device, dtype=saved.dtype)
		widened[:, : saved.shape[-1]] = saved
		state_dict[key] = widened
