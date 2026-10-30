"""Starting biases, in human-readable files next to this one (campaign_model/priors/*.csv, columns name,id,bias), as the
battle model's (battle_model/priors.py): the model's bias parameters start at these values and are learned from there.

	action_types.csv   per action type, the type head's bias: explore by moving and building rather than by ending the
	                   turn at once, leave the market (whose spread loses value) mostly alone, never dismiss a hero
	bank_guards.csv    per creature bank (object, subtype), its guards' expected strength (count x fight value) and hit
	                   points over its levels, from VCMI's own config (write_bank_guards): what a player knows of a bank
	                   it sees, so its fight has a return (features.fight_return) like a wandering monster's
	object_values.csv  per object type (and a few subtypes), what visiting it is worth to Nullkiller2: its own reward
	                   numbers (PriorityEvaluator.cpp: gold, army, skill and strategic rewards) on one gold scale, the
	                   "how" column says which (gold; model.CustomFeatures adds the fights and exploration in gold and
	                   maps the sum to the MOVE logit's prior with features.value_logit)
"""
import csv
from pathlib import Path

import torch

from . import constants as c

PRIORS_DIR = Path(__file__).resolve().parent / "priors"


def read(name):
	"""{id: (name, bias)} from a priors file."""
	with (PRIORS_DIR / name).open(newline="") as f:
		return {int(row["id"]): (row["name"], float(row["bias"])) for row in csv.DictReader(f)}


def action_types():
	"""The type head's starting biases [N_ACTION_TYPES]; types the file does not list get 0."""
	rows = read("action_types.csv")
	return torch.tensor([rows.get(int(a), ("", 0.0))[1] for a in c.ActionType], dtype=torch.float32)


def object_values():
	"""(each object type's worth in gold [OBJECT_VOCAB + 1] (the last row: none), [(object, subtype, worth - its
	type's)] for the rows of one subtype). The worth is Φ's immediate gain (the "gold" column: what taking it adds to
	empire.py's value) where the table has one, else Nullkiller2's score (what Φ adds by hero: skills, spells, luck and
	morale; or nothing Φ holds: information, access; docs/empire_value.md); a creature bank's is its expected reward
	(bank_guards.csv). The model turns gold into prior logits
	with features.value_logit; campaign_model/audit.py checks the immediate ones against what taking them did."""
	types, subtypes = torch.zeros(c.OBJECT_VOCAB + 1), []
	with (PRIORS_DIR / "object_values.csv").open(newline="") as f:
		rows = list(csv.DictReader(f))
	worth = lambda row: float(row["gold"]) if row.get("gold", "") != "" else float(row["score"])  # noqa: E731
	for row in rows:
		if row["subtype"] == "*":
			types[int(row["id"])] = worth(row)
	for row in rows:
		if row["subtype"] != "*":
			subtypes.append((int(row["id"]), int(row["subtype"]), worth(row) - float(types[int(row["id"])])))
	for (obj, subtype), (_, _, reward) in bank_table().items():
		if 0 <= obj < c.OBJECT_VOCAB:
			subtypes.append((obj, subtype, reward - float(types[obj])))
	return types, subtypes


def immediate_objects():
	"""The object types whose worth is Φ's immediate gain (a "gold" value, or a bank's reward): what the value audit
	checks against what taking them did. The rest add to Φ by hero (skills, spells, luck and morale) or not at all
	(information, access)."""
	with (PRIORS_DIR / "object_values.csv").open(newline="") as f:
		ids = {int(row["id"]) for row in csv.DictReader(f) if row.get("gold", "") != ""}
	return ids | {obj for obj, _ in bank_table()}


def class_values():
	"""Each ObjectClass's worth in gold [N_OBJECT_CLASSES], for TRAVEL's areas: the mean of its objects' worth in
	object_values.csv (what we may take: pickups, treasure, banks, skills, boosts, mines, dwellings, towns, info); 0 for
	what is ours or visited, heroes, monsters, transport and the rest (their worth is the learned layers' to find)."""
	types, _ = object_values()
	values = torch.zeros(c.N_OBJECT_CLASSES)
	for kind, ids in c.OBJECT_CLASSES.items():
		if kind in (c.ObjectClass.HERO, c.ObjectClass.MONSTER, c.ObjectClass.TRANSPORT):
			continue
		scored = [float(types[i]) for i in ids if float(types[i]) > 0]
		values[kind] = sum(scored) / len(scored) if scored else 0.0
	return values


BANK_CONFIG = PRIORS_DIR.parent.parent / "vcmi" / "config" / "objects" / "creatureBanks.json"


def write_bank_guards(path=PRIORS_DIR / "bank_guards.csv"):
	"""bank_guards.csv from VCMI's creatureBanks.json: per bank, over its levels (each with its chance of appearing),
	the expected strength and hit points of its guards (a stack's upgradeChance counts the first upgrade that often)
	and the expected reward in Φ's gold (resources at their prices, creatures at cost + u, artifacts at their class's
	mean price).

		.venv/bin/python -c "from campaign_model.priors import write_bank_guards; write_bank_guards()"
	"""
	import json
	import re

	from .vocab import creatures

	from game_values import resources_value

	from .empire import EmpireConfig, _unit
	from .vocab import creature_cost, raw

	by_key = {row["key"].split(":")[-1]: row for row in creatures().values()}
	keys = {row["key"]: row for row in creatures().values()}
	prices = {}
	for artifact in raw()["artifacts"]:
		prices.setdefault(artifact["class"].upper(), []).append(artifact["price"])
	mean_price = {k: sum(v) / len(v) for k, v in prices.items()}
	config = json.loads(re.sub(r"//[^\n]*", "", BANK_CONFIG.read_text()))
	rows = []
	for name, bank in config.items():
		for kind, entry in bank.get("types", {}).items():
			levels = entry.get("levels") or entry.get("rewards") or []
			power = health = reward = 0.0
			for level in levels:
				chance = (level.get("appearChance", {}).get("max", 100) - level.get("appearChance", {}).get("min", 0)) / 100
				reward += chance * resources_value(level.get("resources", {}))
				reward += chance * sum(mean_price.get(a.get("class", "TREASURE").upper(), 0) for a in level.get("artifacts", []))
				for creature in level.get("creatures", []):
					row = by_key[creature["type"]]
					reward += chance * creature["amount"] * (creature_cost(row["id"]) + _unit(row["id"], EmpireConfig()))
				for guard in level.get("guards", []):
					base = by_key[guard["type"]]
					upgrade = keys.get(base.get("upgrades", [None])[0] or "", base) if base.get("upgrades") else base
					p = guard.get("upgradeChance", 0) / 100
					fight = (1 - p) * base["fight_value"] + p * upgrade["fight_value"]
					points = (1 - p) * base["hit_points"] + p * upgrade["hit_points"]
					power += chance * guard["amount"] * fight
					health += chance * guard["amount"] * points
			rows.append((bank["index"], entry.get("index", 0), f"{name}/{kind}", round(power), round(health), round(reward)))
	with path.open("w", newline="") as f:
		out = csv.writer(f)
		out.writerow(["id", "subtype", "name", "guard_power", "guard_health", "reward_gold"])
		out.writerows(sorted(rows))
	return rows


def bank_table():
	"""{(object, subtype): (the guards' expected strength, their hit points, the expected reward in gold)} of
	bank_guards.csv."""
	with (PRIORS_DIR / "bank_guards.csv").open(newline="") as f:
		return {(int(row["id"]), int(row["subtype"])): (float(row["guard_power"]), float(row["guard_health"]), float(row["reward_gold"])) for row in csv.DictReader(f)}


def bank_guards():
	"""{(object, subtype): (the guards' expected strength, their expected hit points)}."""
	return {key: value[:2] for key, value in bank_table().items()}
