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
- Heroes (the user, 2026-10-29): not their level alone, which would never pay for a primary point or a secondary
  skill (an Arena, a Witch Hut, a University), but what a level brings, so a level is worth what it gives: HERO_BASE
  for a fresh hero (its two skill levels and five primary points), and on top of that (skill_value) SKILL_LEVEL for
  each secondary skill level beyond those two (a University's price), the k-th primary point beyond the five
  PRIMARY_POINT x k (the more a hero has, the more the next one is worth), LEVEL_BONUS a level (its specialty), and
  the way to the next level at what that level-up brings (a primary point, a skill level, the bonus). Only the
  strongest hero's skills (the most skill_value) count in full, the others' at SECONDARY_SHARE: a scout's skills matter
  less than the main hero's. A level-up is then worth ~2.6k at level 1, ~4.2k at 5, ~6.2k at 10; over the experience a
  level takes, more than a treasure chest's 2 gold an experience point up to level ~11, less after. The battle reward
  values a hero by its level alone (game_values.hero_level_value), good enough for a battle.
  Their artifacts at the game's prices (a spellbook at its 500: SPELLBOOK_GOLD_COST), and their mana at the battle
  price but only what they can spend in a 3-round battle (game_values.capped_mana_value). A hire with the starting
  army a tavern's heroes of the week come with is a gain (2500 for 1500 and an army of ~1500-2500 at cost + u); one
  with a lone creature, a loss. And HERO_PENALTY x heroes^2 off: a small, growing cost of every extra hero.
- Explored tiles: EXPLORED_TILE gold for each tile the player ever revealed (ever: Cover of Darkness hides tiles again,
  which must not pay twice).
- Nothing for threats or positions: those are left for the value head to learn (labels start 2 days after a decision).
A player out of the game is worth 0.
"""
from dataclasses import dataclass

import math

from game_values import capped_mana_value, fractional_level, resources_value

from . import vocab
from .vocab import DWELLING_FIRST, DWELLING_LAST

STATUS_INGAME, STATUS_LOST = 0, 1  # EPlayerStatus::INGAME, LOSER (WINNER is 2)
# Bumped when Φ's rules change (the evals' scores are shares of it: evaluate.py compares only evals of one version).
# 2 (2026-10-29): heroes by their skills (skill_value), the strongest in full; a fresh hero HERO_BASE; the spellbook
EMPIRE_VERSION = 2


@dataclass(frozen=True)
class EmpireConfig:
	alpha: float = 0.25  # u = ALPHA x a creature's (unupgraded) cost: what a creature for hire is worth
	income_days: float = 5.0  # unpriced income is worth this many days of it (the user: "sane for this hyperparameter")
	hero_base: float = 1500.0  # gold: a fresh hero (its first two skill levels and five primary points), besides its army and artifacts
	skill_level: float = 2000.0  # each secondary skill level beyond a fresh hero's two
	fresh_skill_levels: float = 2.0
	primary_point: float = 400.0  # the k-th primary point beyond a fresh hero's five: k x this
	primary_base: float = 5.0
	level_bonus: float = 200.0  # a level's own worth (its specialty)
	secondary_share: float = 0.5  # how much a hero's skills count unless it is the strongest
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


def skill_value(hero, config=EmpireConfig()):
	"""What a hero's own skills are worth (gold), 0 for a fresh hero's (see the top): its secondary skill levels, its
	primary points (without items), its levels and its way to the next."""
	points = sum(max(p, 0) for p in hero["primary"])
	extra = max(points - config.primary_base, 0.0)
	level = fractional_level(max(hero["exp"], 0), vocab.experience_table())
	whole = math.floor(level)
	levels = sum(max(level_, 0) for _, level_ in hero["skills"])
	next_gain = config.primary_point * max(points - config.primary_base + 1, 0.0) + config.skill_level + config.level_bonus
	return config.skill_level * (levels - config.fresh_skill_levels) + config.primary_point * extra * (extra + 1) / 2 + config.level_bonus * (whole - 1) + (level - whole) * next_gain


def heroes_value(heroes, config=EmpireConfig()):
	"""Φ's heroes part (gold): HERO_BASE each, the strongest hero's skill_value in full, the others' at SECONDARY_SHARE."""
	if not heroes:
		return 0.0
	values = [skill_value(h, config) for h in heroes]
	main = max(values)
	return config.hero_base * len(heroes) + main + config.secondary_share * (sum(values) - main)


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
	prices = vocab.artifact_prices()
	parts["heroes"] = heroes_value(player["heroes"], config)
	for hero in player["heroes"]:
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

