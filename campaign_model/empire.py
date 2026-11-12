"""What an empire is worth (Φ), in gold, from a ledger snapshot (agents/AlphaMapAI/Ledger.cpp): every player's assets
as raw engine values. The labels (labels.py) are each player's share of all players' Φ some days later. Every map
object and town building, what it gives and what Φ adds for it (or why nothing): docs/empire_value.md.

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
  Its spells are among its skills (the user, 2026-10-30: a Shrine must not be skipped): SPELL_LEVEL x each spell's
  level for every spell in its spellbook (a Shrine of Magic Incantation's 500, a Pyramid's 2,500, a first mage guild's
  five ~2,500 to the main hero).
  Its luck and morale until the next battle (a Temple, a Buoy, a Rally Flag, a Warrior's Tomb's -3 morale): each 1/24
  of chance a battle stack gains (a lucky double-damage hit, an extra turn) or loses (a turn lost to bad morale, 2/24
  a point) is worth BOOST of the army it lifts at cost + u (luck: every creature; morale: the living), within the
  battle's +-3 caps, and goes when the battle uses it: a Temple's +1 morale is ~400 to a 20,000 army.
  Their artifacts at the game's prices (a spellbook at its 500: SPELLBOOK_GOLD_COST), and their mana at the battle
  price but only what they can spend in a 3-round battle (game_values.capped_mana_value). A hire with the starting
  army a tavern's heroes of the week come with is a gain (2500 for 1500 and an army of ~1500-2500 at cost + u); one
  with a lone creature, a loss. And HERO_PENALTY x heroes^2 off: a small, growing cost of every extra hero.
- Explored tiles: EXPLORED_TILE gold for each tile the player ever revealed (ever: Cover of Darkness hides tiles again,
  which must not pay twice).
- Nothing for threats or positions: those are left for the value head to learn (labels start 2 days after a decision).
A player out of the game is worth 0.
"""
from dataclasses import dataclass, field

import math
import os

from game_values import ARMY_MORALE, SPEED_POWER, SPELL_STEP, army_multiplier, capped_mana_value, fractional_level, make_up_morale, movement_factor, resources_value, unled_share

from . import vocab
from .vocab import DWELLING_FIRST, DWELLING_LAST

STATUS_INGAME, STATUS_LOST = 0, 1  # EPlayerStatus::INGAME, LOSER (WINNER is 2)
# Bumped when Φ's rules change (the evals' scores are shares of it: evaluate.py compares only evals of one version).
# 2 (2026-10-29): heroes by their skills (skill_value), the strongest in full; a fresh hero HERO_BASE; the spellbook.
# 3 (2026-10-30): heroes' spells (SPELL_LEVEL a level), their luck and morale until the next battle (boost_value)
# 4 (2026-11-05): armies no hero leads at EmpireConfig.unled_share, a hero's spells in what it makes of its army.
# 5 (2026-11-05): a hero's army at its movement (speed_factor) and its make-up's morale (make_up_value). Training's; the
# evaluation keeps 3's rules (EVAL_EMPIRE, below EmpireConfig) and its version
EMPIRE_VERSION = 5
EVAL_EMPIRE_VERSION = 3
# A battle's luck and morale (VCMI config/gameConfig.json "combat": goodLuckChance [1, 2, 3], badLuckChance [],
# goodMoraleChance [1, 2, 3], badMoraleChance [2, 4, 6], all out of 24): at each value -3..+3 the chance in 24 that a
# stack's attack is lucky (double damage), or that its turn brings an extra one (+) or is lost (-)
LUCK_CHANCE = (0, 0, 0, 0, 1, 2, 3)
MORALE_CHANCE = (-6, -4, -2, 0, 1, 2, 3)


@dataclass(frozen=True)
class EmpireConfig:
	# u = ALPHA x a creature's (unupgraded) cost: what a creature for hire is worth. ALPHA_EMPIRE_ALPHA sets it for a run
	# (the user, 2026-11-12: "going from 0.25 to 0.5 should be fine"; purchases stay neutral: hired cost + u, for hire u):
	# read when this module is imported, since every default EmpireConfig() is made then, so it is the run's environment
	# (runtime/advice_run.sh), the same in every process it starts. The evaluation's EVAL_EMPIRE keeps 0.25
	alpha: float = field(default_factory=lambda: float(os.environ.get("ALPHA_EMPIRE_ALPHA", "") or 0.25))
	income_days: float = 5.0  # unpriced income is worth this many days of it (the user: "sane for this hyperparameter")
	hero_base: float = 1500.0  # gold: a fresh hero (its first two skill levels and five primary points), besides its army and artifacts
	skill_level: float = 2000.0  # each secondary skill level beyond a fresh hero's two
	fresh_skill_levels: float = 2.0
	primary_point: float = 400.0  # the k-th primary point beyond a fresh hero's five: k x this
	primary_base: float = 5.0
	level_bonus: float = 200.0  # a level's own worth (its specialty)
	secondary_share: float = 0.5  # how much a hero's skills count unless it is the strongest
	spell_level: float = 500.0  # each spell in a hero's spellbook: this x its level (1-5)
	boost: float = 0.02  # each 1/24 of battle chance luck or morale gives until the next battle: this x the army it lifts
	hero_penalty: float = 100.0  # gold x heroes^2
	explored_tile: float = 5.0  # gold per tile ever revealed (the user, 2026-11-03: 5, our 36x36 two-level maps 13k explored in full, ~3.4k at a game's usual 26%; 2026-10-27: 10 was too much)
	# A hero's HERO_BASE by its army (the user, 2026-11-03: "a hero with nothing slightly negative, maybe -1k on average
	# ... heroes with creatures should stay positive"): armyless_share of it with no army, all of it once the army is
	# worth armed_army (gold at cost + u, as Φ's army part counts it). The hires of joint7's games (471): +1,943 in Φ at the
	# median, -366 without their army; at 0.55 an empty hire comes to ~-1,040, one with its usual army (1,938) as before
	armyless_share: float = 0.55
	armed_army: float = 1000.0
	# A hero's spellbook in what it makes of its army (game_values.spell_factor), and so Φ's unit: an army no hero leads
	# at game_values.unled_share (0.953 at 0.02), a hero's at that x its multiplier (the user, 2026-11-05: buying an army
	# slightly negative, neutral once even a starting hero with spell power 3 and knowledge 2 leads it)
	spell_step: float = SPELL_STEP
	# A hero's army at its movement (game_values.movement_factor at this power) and the battle chance its make-up gives or
	# takes (game_values.ARMY_MORALE a 24th of the living army; the user, 2026-11-05)
	speed_power: float = SPEED_POWER
	army_morale: float = ARMY_MORALE


# The evaluation's Φ (its scores, evaluate.score): as of 2026-11-04's army multiplier, before the spellbook's factor and
# the unled share, the army's movement and make-up, so its scores stay comparable across the changes
EVAL_EMPIRE = EmpireConfig(alpha=0.25, spell_step=0.0, speed_power=0.0, army_morale=0.0)


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
	primary points (without items), its levels and its way to the next, its spells."""
	points = sum(max(p, 0) for p in hero["primary"])
	extra = max(points - config.primary_base, 0.0)
	level = fractional_level(max(hero["exp"], 0), vocab.experience_table())
	whole = math.floor(level)
	levels = sum(max(level_, 0) for _, level_ in hero["skills"])
	next_gain = config.primary_point * max(points - config.primary_base + 1, 0.0) + config.skill_level + config.level_bonus
	spells = config.spell_level * sum(vocab.spell_levels().get(spell, 0) for spell in hero.get("spells", []))
	return config.skill_level * (levels - config.fresh_skill_levels) + config.primary_point * extra * (extra + 1) / 2 + config.level_bonus * (whole - 1) + (level - whole) * next_gain + spells


def _chance(table, value):
	"""The battle chance (in 24ths) at a luck or morale value, capped at -3..+3 (LUCK_CHANCE, MORALE_CHANCE)."""
	return table[min(max(value, -3), 3) + 3]


def boost_value(hero, config=EmpireConfig()):
	"""What a hero's luck and morale until the next battle are worth (gold; see the top): the battle chance they add
	over the rest of its luck and morale (within the caps: a hero at +3 morale gains nothing from a Temple; a Warrior's
	Tomb's -3 costs twice a point as a lost turn), BOOST a 24th, of the army each lifts at cost + u."""
	luck, morale = hero.get("luck", 0), hero.get("morale", 0)
	gained_luck = _chance(LUCK_CHANCE, luck) - _chance(LUCK_CHANCE, luck - hero.get("battle_luck", 0))
	gained_morale = _chance(MORALE_CHANCE, morale) - _chance(MORALE_CHANCE, morale - hero.get("battle_morale", 0))
	if not gained_luck and not gained_morale:
		return 0.0
	lucky = living = 0.0
	for creature, count in hero["army"]:
		worth = count * (vocab.creature_cost(creature) + _unit(creature, config))
		lucky += worth * vocab.feels(creature, vocab.NO_LUCK)
		living += worth * vocab.feels(creature, vocab.NO_MORALE)
	return config.boost * (gained_luck * lucky + gained_morale * living)


def army_worth(stacks, config=EmpireConfig()):
	"""An army's worth in Φ (gold): each creature at cost + u, as Φ's army part counts it."""
	return sum(count * (vocab.creature_cost(creature) + _unit(creature, config)) for creature, count in stacks)


def _present(stacks):
	return [(creature, count) for creature, count in stacks if creature in vocab.creatures() and count > 0]


def speed_factor(stacks, config=EmpireConfig()):
	"""What a hero's movement makes of the army it leads (game_values.movement_factor at its slowest creature's speed,
	at speed_power); 1 for none."""
	speeds = [vocab.creatures()[creature].get("speed", 0) for creature, _ in _present(stacks)]
	return movement_factor(min(speeds), config.speed_power) if speeds else 1.0


def army_make_up(stacks):
	"""An army's morale from its make-up (game_values.make_up_morale: its factions, any undead)."""
	present = _present(stacks)
	factions = {vocab.creatures()[creature].get("faction") for creature, _ in present}
	undead = any("UNDEAD" in vocab.creatures()[creature].get("abilities", []) for creature, _ in present)
	return make_up_morale(len(factions), undead)


def make_up_value(hero, config=EmpireConfig()):
	"""What a hero's army's make-up gives or takes in battle (gold): the chance its morale (army_make_up) adds over the
	rest of the hero's morale (its "morale" holds the make-up's: VCMI's bonus of source ARMY), within the caps, ARMY_MORALE
	a 24th of the army it lifts at cost + u (its living creatures)."""
	make_up, morale = army_make_up(hero["army"]), hero.get("morale", 0)
	gained = _chance(MORALE_CHANCE, morale) - _chance(MORALE_CHANCE, morale - make_up)
	if not gained:
		return 0.0
	living = sum(count * (vocab.creature_cost(creature) + _unit(creature, config)) * vocab.feels(creature, vocab.NO_MORALE) for creature, count in hero["army"])
	return config.army_morale * gained * living


def hero_multiplier(hero, config=EmpireConfig()):
	"""What a hero's army counts at in Φ: game_values.unled_share x army_multiplier at its battle attack, defense, spell
	power and knowledge, items included (the ledger's "attack", "defense", "power", "knowledge"; its base primary skills
	in a ledger from before them), and its spellbook (the ledger's "spellbook", else artifact 0 among its "artifacts")."""
	primary = hero["primary"]
	spellbook = hero.get("spellbook", 0 in hero.get("artifacts", []))
	multiplier = army_multiplier(
		hero.get("attack", primary[0]), hero.get("defense", primary[1]), hero.get("power", primary[2]), hero.get("knowledge", primary[3]), spellbook, config.spell_step
	)
	return unled_share(config.spell_step) * multiplier


def hero_base_value(army, config=EmpireConfig()):
	"""A hero's HERO_BASE at its army's worth (gold): armyless_share of it with none, all of it from armed_army on."""
	return config.hero_base * (config.armyless_share + (1 - config.armyless_share) * min(1.0, max(army, 0.0) / config.armed_army))


def heroes_value(heroes, config=EmpireConfig()):
	"""Φ's heroes part (gold): each hero's HERO_BASE by its army (hero_base_value), the strongest hero's skill_value in
	full, the others' at SECONDARY_SHARE, every hero's luck and morale until the next battle (boost_value) and its army's
	make-up (make_up_value)."""
	if not heroes:
		return 0.0
	values = [skill_value(h, config) for h in heroes]
	main = max(values)
	base = sum(hero_base_value(army_worth(h["army"], config), config) for h in heroes)
	return base + main + config.secondary_share * (sum(values) - main) + sum(boost_value(h, config) + make_up_value(h, config) for h in heroes)


def empire_value(player, config=EmpireConfig()):
	"""{component: gold} of one player of a ledger snapshot, and "total"."""
	if player.get("status", STATUS_INGAME) == STATUS_LOST:  # (a winner keeps its value: it ends with every share)
		return {"total": 0.0}
	parts = dict.fromkeys(("resources", "buildings", "income", "army", "for_hire", "heroes", "artifacts", "mana", "hero_penalty", "explored"), 0.0)
	parts["resources"] = resources_value(player["resources"])
	stacks = list(player.get("armies", []))  # (armies no hero leads, at their worth; the heroes' by army_multiplier below)
	for town in player["towns"]:
		parts["buildings"] += sum(building_value(town["faction"], b, config) for b in town["built"])
		stacks += town["garrison"]
		parts["for_hire"] += sum(count * _unit(creature, config) for creature, count in town["available"] if creature >= 0)
	for dwelling in player.get("dwellings", []):
		parts["for_hire"] += sum(count * _unit(creature, config) for creature, count in dwelling["available"] if creature >= 0)
	parts["army"] = unled_share(config.spell_step) * army_worth(stacks, config) + sum(army_worth(hero["army"], config) * hero_multiplier(hero, config) * speed_factor(hero["army"], config) for hero in player["heroes"])
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

