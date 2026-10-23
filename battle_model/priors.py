"""Starting biases, in human-readable files next to this one (battle_model/priors/*.csv, columns name,id,bias), to
review and edit: the models' bias parameters start at these values and are learned from there.

	action_types.csv   per action type, the type head's bias: e.g. -6 for retreat and surrender (giving up is offered
	                   at every decision: at -6 it ends ~4% of untrained battles, at -2 two in three)
	spells.csv         per spell, the spell head's bias: some spells are usually better than others
	MASTERY_BIAS       per spell, by our hero's mastery of the spell's school (none, basic, advanced, expert; the best of
	                   its schools): added to that spell's logit, learned per spell and mastery, starting the same for all

tests/test_priors.py checks the files: every id once, names as the engine's.
"""
import csv
from pathlib import Path

import torch

from .constants import N_ACTION_TYPES, N_SPELLS

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


def spells():
	return biases("spells.csv", N_SPELLS)


def mastery():
	"""[N_SPELLS, 4]: MASTERY_BIAS for every spell."""
	return torch.tensor(MASTERY_BIAS).expand(N_SPELLS, -1).clone()
