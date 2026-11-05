"""What things are worth, in gold: the one list the battle model (battle_model/reward.py) and the campaign model
(campaign_model/empire.py) both price with. The campaign sees more of a hero's and an artifact's worth than a battle
does, so it is expected to make most of the later changes here; every change moves both models.

	resources      config/resources.json prices: wood and ore 250, mercury, sulfur, crystal and gems 500
	creatures      their recruit cost at those prices (vocab.json)
	artifacts      their price in the game's artifact table (ARTRAITS, vocab.json)
	war machines   half their price (ballista, ammo cart, first aid tent), the catapult and arrow towers 0
	hero levels    a level-up from level L is worth LEVEL_VALUE x min(L, LEVEL_STEPS): 2000, 4000, then 6000 each;
	               level 2 2000, 3 6000, 4 12 000, 10 48 000, 20 108 000 (hero_level_value)
	mana           battle: MANA_VALUE x spell power x mana;
	               campaign: the same, but only of the mana a hero can spend in a short battle (capped_mana_value)
	an army        campaign: its worth x what the hero leading it makes of it (army_multiplier: VCMI's damage
	               modifiers at the hero's battle attack and defense), 1 for no hero (a garrison's)

The tables are the engine's (battle_model/artifacts/vocab.json, scripts/export_vocab.sh); nothing here is trained.
"""
import math

# config/resources.json "price", in VCMI's resource order
RESOURCE_PRICES = {"wood": 250, "mercury": 500, "ore": 250, "sulfur": 500, "crystal": 500, "gems": 500, "gold": 1}
RESOURCE_ORDER = tuple(RESOURCE_PRICES)
# GameConstants::HERO_GOLD_COST: hiring a hero in a tavern
HERO_HIRE_COST = 2500
# GameConstants::SPELLBOOK_GOLD_COST: a spellbook in a town with a mage guild
SPELLBOOK_GOLD_COST = 500

# Hero levels (the user, 2026-10-29: every level counts, so the main hero grows; levels 1-3 had been free, and no hero of
# the campaign gained one): the first level-up's worth, and how many level-ups each add that much again
LEVEL_VALUE = 2000.0
LEVEL_STEPS = 3
# Gold per mana point per point of spell power. At 2, a power-5 hero's 50 mana is worth 500 gold
MANA_VALUE = 2.0
# The battle the campaign prices mana for: a hero casts at most once a round
MANA_ROUNDS = 3


# VCMI's damage modifiers (an attack against the defender's defense): +5% damage a point over it, at most +300%; -2.5%
# a point under it, at most -70%
ATTACK_STEP, ATTACK_POINTS = 0.05, 60
DEFENSE_STEP, DEFENSE_POINTS = 0.025, 28


def army_multiplier(attack, defense):
	"""What a hero's battle attack and defense (its items' included) make of the army it leads (the user, 2026-11-04:
	keeping the army with the strongest hero must be worth more), against an enemy of none: the root of the damage it
	deals over the damage it takes, by VCMI's modifiers; 1 for attack and defense 0 (a garrison's)."""
	dealt = 1.0 + ATTACK_STEP * min(max(attack, 0), ATTACK_POINTS)
	taken = 1.0 - DEFENSE_STEP * min(max(defense, 0), DEFENSE_POINTS)
	return math.sqrt(dealt / taken)


def resources_value(resources):
	"""Gold value of a cost dict, e.g. {"gold": 3000, "gems": 1}, or of 7 amounts in VCMI's order."""
	if isinstance(resources, dict):
		return sum(RESOURCE_PRICES[name] * amount for name, amount in resources.items())
	return sum(RESOURCE_PRICES[name] * amount for name, amount in zip(RESOURCE_ORDER, resources))


def hero_level_value(level, level_value=LEVEL_VALUE, level_steps=LEVEL_STEPS):
	"""What a hero's levels are worth: the level-up from level L to L + 1 is worth level_value x min(L, level_steps),
	2000, 4000, then 6000 each (level 2 2000, 3 6000, 4 12000, 10 48000, 20 108000); level counts its way to the next
	(12.5: halfway to 13), each experience point of a level worth the same. Over the experience each level takes, that
	is 2 gold a point at level 1, 4-5 at levels 2-4, falling under 2 from level 11 (a treasure chest's 1000 gold against
	500 experience: experience for a young hero, gold for a seasoned one). A lone level-2 hero (2000) is worth less than
	a hire (2500), so it is not worth a retreat; a level-3 one is."""
	if level <= 1:
		return 0.0
	whole = math.floor(level)
	done = whole - 1  # the level-ups made
	steps = min(done, level_steps)
	made = steps * (steps + 1) / 2 + level_steps * (done - steps)
	return level_value * (made + (level - whole) * min(whole, level_steps))


def fractional_level(experience, levels_experience):
	"""The level with the way to the next one: levels_experience[i] is the experience level i + 1 needs (vocab.json
	"experience": [0, 1000, 2000, ...]). 1500 experience -> 2.5."""
	level = 1
	while level < len(levels_experience) and experience >= levels_experience[level]:
		level += 1
	if level >= len(levels_experience):
		return float(level)
	low, high = levels_experience[level - 1], levels_experience[level]
	return level + (experience - low) / (high - low)


def capped_mana_value(mana, spell_power, max_spell_cost, rounds=MANA_ROUNDS, mana_value=MANA_VALUE):
	"""Campaign mana: only what the hero can spend in a battle of `rounds` rounds (one cast a round at the cost of
	its dearest combat spell) is worth anything, at the battle's price. 1000 -> 900 mana costs nothing, 20 -> 5 does;
	no combat spell, no value; a weak caster's mana is worth little."""
	usable = min(max(mana, 0), rounds * max(max_spell_cost, 0))
	return mana_value * max(spell_power, 0) * usable


# War machines (decided 2026-10-28, memory war-machine-value): a hero's ballista, ammo cart and first aid tent are worth
# half their price, the catapult and the arrow towers nothing, in the campaign's Φ and the battle reward alike. Not
# full price: a fresh hero often comes with one and every hero with a catapult, so at the game's prices a hire of 2500
# brought ~3600 of value (joint1, 2026-10-27: heroes per player 1.5 -> 3.2 in an hour). Not 0: then a destroyed ballista
# cost nothing and a bought one was a pure loss. They cannot be sold; a hero that is defeated loses them, one that
# retreats or surrenders keeps those still standing; the winner takes none (VCMI: BattleResultProcessor).
WAR_MACHINE_ARTIFACTS = {3: 0.0, 4: 1250.0, 5: 500.0, 6: 375.0}  # catapult, ballista, ammo cart, first aid tent (a hero's slots)
WAR_MACHINE_CREATURES = {145: 0.0, 146: 1250.0, 147: 375.0, 148: 500.0, 149: 0.0}  # the same in battle, and the arrow tower


def creature_value(creature_id):
	"""Gold value of one creature: its recruit cost at the resource prices (a war machine: WAR_MACHINE_CREATURES)."""
	from battle_model.vocab import creature

	if creature_id in WAR_MACHINE_CREATURES:
		return WAR_MACHINE_CREATURES[creature_id]
	return resources_value(creature(creature_id).get("cost", {}))


# Artifacts that are no asset of a hero: the spellbook and the Grail (valued once built); war machines at
# WAR_MACHINE_ARTIFACTS. The battle's winner takes neither (agents/AlphaBattleAI Encoder.cpp heroArtifacts)
UNVALUED_ARTIFACTS = frozenset({0, 2})


def artifact_price(artifact_id):
	"""What an artifact is worth as an asset: the game's price; 0 for UNVALUED_ARTIFACTS, WAR_MACHINE_ARTIFACTS'."""
	from battle_model.vocab import tables

	if artifact_id in WAR_MACHINE_ARTIFACTS:
		return WAR_MACHINE_ARTIFACTS[artifact_id]
	return 0 if artifact_id in UNVALUED_ARTIFACTS else tables()["artifacts"][artifact_id]["price"]
