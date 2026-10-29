"""What an empire is worth (Φ), in gold, from a ledger snapshot (agents/AlphaMapAI/Ledger.cpp): every player's assets
as raw engine values. The labels (labels.py) are each player's share of all players' Φ some days later.

The user's rules (2026-10-27), and why each component is priced as it is:
- Buying anything at its list price leaves Φ unchanged; value appears when what was bought produces, through pickups,
  fights and captures. Prices are the shared list's (game_values): the battle model prices with the same.
- Resources: at the resource prices.
- Buildings: their cost. A building with an income (the halls, the resource silo, the grail) is worth
  max(cost, INCOME_DAYS x the daily income it adds over the building it upgrades): the Town, City and Capitol halls cost
  exactly 5 days of the income they add, so both rules agree on them, and the village hall (pre-built, no real cost)
  gets 5 days of its income.
- Income without a price (mines and other owned objects): INCOME_DAYS x its daily value.
- Creatures: a hired creature is worth its cost + u, a creature for hire (in our towns and dwellings) u, where
  u = ALPHA x the cost of its unupgraded form; recruiting is neutral (cost + u for cost and u), upgrading too (u does
  not change with the upgrade). A dwelling is worth its cost minus u x its creature's growth: the first batch, which
  comes with it, would otherwise make building it a gain. Growth raises Φ the day it arrives (Monday), never before:
  the AI is to learn the calendar (build dwellings at the end of the week), not to see it smoothed out.
- Heroes: the battle model's value of their levels (game_values.hero_level_value: levels 1-3 free, then
  2500 x (L - 3) ^ 1.3, the level counting its way to the next), their artifacts at the game's prices, and their mana
  at the battle price but only what they can spend in a 3-round battle (game_values.capped_mana_value). Hiring is not
  neutral: 2500 buys a hero worth its starting army (and its level, if any), so a hire must pay for itself through
  what the hero does. And HERO_PENALTY x heroes^2 off: a small, growing cost of every extra hero.
- Explored tiles: EXPLORED_TILE gold for each tile the player ever revealed (ever: Cover of Darkness hides tiles again,
  which must not pay twice).
- Nothing for threats or positions: those are left for the value head to learn (labels start 2 days after a decision).
A player out of the game is worth 0.
"""
from dataclasses import dataclass

from game_values import capped_mana_value, fractional_level, hero_level_value, resources_value

from . import vocab
from .vocab import DWELLING_FIRST, DWELLING_LAST

STATUS_INGAME, STATUS_LOST = 0, 1  # EPlayerStatus::INGAME, LOSER (WINNER is 2)


@dataclass(frozen=True)
class EmpireConfig:
	alpha: float = 0.25  # u = ALPHA x a creature's (unupgraded) cost: what a creature for hire is worth
	income_days: float = 5.0  # unpriced income is worth this many days of it (the user: "sane for this hyperparameter")
	hero_penalty: float = 100.0  # gold x heroes^2
	explored_tile: float = 1.0  # gold per tile ever revealed (the user, 2026-10-27: 10 was too much; a 72x72 two-level map is 10k)


def _unit(creature_id, config):
	"""u: what one creature for hire is worth."""
	return config.alpha * vocab.creature_cost(vocab.base_forms().get(creature_id, creature_id))


def building_value(faction, building, config=EmpireConfig()):
	row = vocab.buildings().get((faction, building))
	if row is None:
		return 0.0
	cost = resources_value(row.get("cost") or {})
	produce = resources_value(row.get("produce") or {})
	if produce:
		below = vocab.buildings().get((faction, row.get("upgrade", -1)))
		added = produce - (resources_value(below.get("produce") or {}) if below else 0)
		return max(cost, config.income_days * added)
	if DWELLING_FIRST <= building <= DWELLING_LAST:
		levels = vocab.town_creatures().get(faction, [])
		level = building - DWELLING_FIRST
		if level < len(levels) and levels[level]:
			base = levels[level][0]
			cost -= _unit(base, config) * vocab.creatures()[base].get("growth", 0)
	return cost


def empire_value(player, config=EmpireConfig()):
	"""{component: gold} of one player of a ledger snapshot, and "total"."""
	if player.get("status", STATUS_INGAME) == STATUS_LOST:  # (a winner keeps its value: it ends with every share)
		return {"total": 0.0}
	parts = dict.fromkeys(("resources", "buildings", "income", "army", "for_hire", "heroes", "artifacts", "mana", "hero_penalty", "explored"), 0.0)
	parts["resources"] = resources_value(player["resources"])
	stacks = [pair for hero in player["heroes"] for pair in hero["army"]] + list(player.get("armies", []))
	for town in player["towns"]:
		parts["buildings"] += sum(building_value(town["faction"], b, config) for b in town["built"])
		stacks += town["garrison"]
		parts["for_hire"] += sum(count * _unit(creature, config) for creature, count in town["available"] if creature >= 0)
	for dwelling in player.get("dwellings", []):
		parts["for_hire"] += sum(count * _unit(creature, config) for creature, count in dwelling["available"] if creature >= 0)
	parts["army"] = sum(count * (vocab.creature_cost(creature) + _unit(creature, config)) for creature, count in stacks)
	parts["income"] = config.income_days * resources_value(player.get("income", [0] * 7))
	experience = vocab.experience_table()
	prices = vocab.artifact_prices()
	for hero in player["heroes"]:
		parts["heroes"] += hero_level_value(fractional_level(hero["exp"], experience))
		parts["artifacts"] += sum(prices.get(a, 0) for a in hero["artifacts"])
		parts["mana"] += capped_mana_value(hero["mana"], hero["power"], hero["spell_cost"])
	parts["hero_penalty"] = -config.hero_penalty * len(player["heroes"]) ** 2
	parts["explored"] = config.explored_tile * player.get("explored", 0)
	parts["total"] = sum(parts.values())
	return parts


def shares(snapshot, config=EmpireConfig()):
	"""{color: Φ share} of one ledger snapshot: each player's Φ over all players' (0 for a player out of the game)."""
	values = {p["color"]: max(empire_value(p, config)["total"], 0.0) for p in snapshot["players"]}
	total = sum(values.values())
	alive = [p["color"] for p in snapshot["players"] if p.get("status", STATUS_INGAME) != STATUS_LOST]
	if total > 0:
		return {color: value / total for color, value in values.items()}
	# Nothing of value anywhere: those still in split it; losers get 0 (everyone knocked out together: all 0)
	return {color: (1.0 / len(alive) if color in alive else 0.0) for color in values}

