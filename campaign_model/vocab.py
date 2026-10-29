"""The engine's tables the campaign model needs (battle_model/artifacts/vocab.json, from scripts/export_vocab.sh):
creatures (cost, growth, fight value, upgrades), town buildings (cost, daily produce, what each upgrades), each
faction's creatures by level, the experience table, artifacts' prices and spells' mana costs."""
import json
from functools import lru_cache

from battle_model.vocab import VOCAB_PATH
from game_values import resources_value

DWELLING_FIRST, DWELLING_LAST = 30, 36  # BuildingID DWELL_LVL_1..7: their first batch is free (empire.py)


@lru_cache(maxsize=None)
def raw(path=VOCAB_PATH):
	return json.loads(path.read_text())


@lru_cache(maxsize=None)
def creatures():
	return {row["id"]: row for row in raw()["creatures"]}


@lru_cache(maxsize=None)
def base_forms():
	"""creature id -> the creature it is an upgrade of, followed to the bottom (itself if it is no upgrade)."""
	by_key = {row["key"]: row["id"] for row in raw()["creatures"]}
	parent = {}
	for row in raw()["creatures"]:
		for upgrade in row.get("upgrades", []):
			if upgrade in by_key and by_key[upgrade] != row["id"]:
				parent.setdefault(by_key[upgrade], row["id"])
	base = {}
	for creature in creatures():
		seen, current = set(), creature
		while current in parent and current not in seen:
			seen.add(current)
			current = parent[current]
		base[creature] = current
	return base


def creature_cost(creature_id):
	"""Gold value of one creature (its recruit cost at the shared resource prices); 0 for an unknown id."""
	row = creatures().get(creature_id)
	return resources_value(row.get("cost", {})) if row else 0


@lru_cache(maxsize=None)
def buildings():
	"""(faction, building id) -> row with cost, produce, upgrade."""
	return {(row["faction"], row["id"]): row for row in raw()["buildings"]}


@lru_cache(maxsize=None)
def town_creatures():
	"""faction -> creatures[level] = [base, upgrade, ...]."""
	return {row["faction"]: row["creatures"] for row in raw()["town_creatures"]}


def experience_table():
	return raw()["experience"]


@lru_cache(maxsize=None)
def artifact_prices():
	"""artifact id -> its worth as an asset (game_values.artifact_price: spellbook and Grail 0, war machines half)."""
	from game_values import UNVALUED_ARTIFACTS, WAR_MACHINE_ARTIFACTS

	return {row["id"]: WAR_MACHINE_ARTIFACTS.get(row["id"], 0 if row["id"] in UNVALUED_ARTIFACTS else row["price"]) for row in raw()["artifacts"]}
