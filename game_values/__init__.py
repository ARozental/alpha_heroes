"""What things are worth, in gold: the one list the battle model (battle_model/reward.py) and the campaign model
(campaign_model/empire.py) both price with. The campaign sees more of a hero's and an artifact's worth than a battle
does, so it is expected to make most of the later changes here; every change moves both models.

	resources      config/resources.json prices: wood and ore 250, mercury, sulfur, crystal and gems 500
	creatures      their recruit cost at those prices (vocab.json)
	artifacts      their price in the game's artifact table (ARTRAITS, vocab.json)
	war machines   half their price (ballista, ammo cart, first aid tent), the catapult and arrow towers 0
	hero levels    LEVEL_VALUE x (level - FREE_LEVELS) ^ LEVEL_EXPONENT: 2500 x (L - 3) ^ 1.3; levels 1-3 are free
	               (a fresh hero's hire price buys it with a few creatures), level 4 2500, 10 ~31 600, 20 ~100 000
	mana           battle: MANA_VALUE x spell power x mana;
	               campaign: the same, but only of the mana a hero can spend in a short battle (capped_mana_value)

The tables are the engine's (battle_model/artifacts/vocab.json, scripts/export_vocab.sh); nothing here is trained.
"""

# config/resources.json "price", in VCMI's resource order
RESOURCE_PRICES = {"wood": 250, "mercury": 500, "ore": 250, "sulfur": 500, "crystal": 500, "gems": 500, "gold": 1}
RESOURCE_ORDER = tuple(RESOURCE_PRICES)
# GameConstants::HERO_GOLD_COST: hiring a hero in a tavern
HERO_HIRE_COST = 2500
# GameConstants::SPELLBOOK_GOLD_COST: a spellbook in a town with a mage guild
SPELLBOOK_GOLD_COST = 500

LEVEL_VALUE = 2500.0
FREE_LEVELS = 3.0
LEVEL_EXPONENT = 1.3
# Gold per mana point per point of spell power. At 2, a power-5 hero's 50 mana is worth 500 gold
MANA_VALUE = 2.0
# The battle the campaign prices mana for: a hero casts at most once a round
MANA_ROUNDS = 3


def resources_value(resources):
	"""Gold value of a cost dict, e.g. {"gold": 3000, "gems": 1}, or of 7 amounts in VCMI's order."""
	if isinstance(resources, dict):
		return sum(RESOURCE_PRICES[name] * amount for name, amount in resources.items())
	return sum(RESOURCE_PRICES[name] * amount for name, amount in zip(RESOURCE_ORDER, resources))


def hero_level_value(level, level_value=LEVEL_VALUE, free_levels=FREE_LEVELS, level_exponent=LEVEL_EXPONENT):
	"""What a hero's levels are worth; level counts its way to the next level (12.5: halfway to 13)."""
	return level_value * max(0.0, level - free_levels) ** level_exponent


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
