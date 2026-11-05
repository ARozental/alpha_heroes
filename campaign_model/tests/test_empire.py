import copy
import math

import pytest

from campaign_model import vocab
from campaign_model.empire import EmpireConfig, army_worth, boost_value, building_value, empire_value, hero_base_value, hero_multiplier, heroes_value, shares, skill_value
from game_values import REFERENCE_KNOWLEDGE, REFERENCE_POWER, unled_share
from game_values import HERO_HIRE_COST, capped_mana_value, resources_value

CASTLE = 0
VILLAGE_HALL, TOWN_HALL, CITY_HALL, CAPITOL, DWELLING_1, DWELLING_1_UP = 10, 11, 12, 13, 30, 37
PIKEMAN, HALBERDIER, SKELETON = 0, 1, 56
MAGIC_ARROW, FIREBALL, HASTE = 15, 21, 53


def cost(building):
	return resources_value(vocab.buildings()[(CASTLE, building)]["cost"])


def fresh_hero(kind, army=((PIKEMAN, 10),), **changes):
	"""A level-1 hero with a fresh hero's two skill levels and five primary points (empire.py: its skills worth 0)."""
	return {"type": kind, "level": 1, "exp": 0, "mana": 10, "power": 1, "spell_cost": 0, "primary": [1, 1, 1, 2], "skills": [[0, 1], [1, 1]], "army": [list(a) for a in army], "artifacts": [], **changes}


def player(**changes):
	p = {
		"color": 0, "status": 0, "resources": [10, 4, 10, 4, 4, 4, 10000], "income": [0] * 7, "explored": 100,
		"towns": [{"faction": CASTLE, "built": [VILLAGE_HALL, DWELLING_1], "garrison": [], "available": [[PIKEMAN, 14]] + [[-1, 0]] * 6}],
		"heroes": [fresh_hero(0)],
		"dwellings": [], "armies": [],
	}
	p.update(changes)
	return p


# Φ without the army's movement and make-up (2026-11-05, v5): the tests of the other rules, each its own
ISOLATED = EmpireConfig(speed_power=0.0, army_morale=0.0)


def total(p, config=ISOLATED):
	return empire_value(p, config)["total"]


def test_the_halls_are_worth_their_cost_and_the_village_hall_five_days_of_income():
	for hall in (TOWN_HALL, CITY_HALL, CAPITOL):
		assert building_value(CASTLE, hall) == cost(hall)
	assert building_value(CASTLE, VILLAGE_HALL) == 5 * 500


def test_building_at_list_price_is_neutral():
	before = player()
	after = copy.deepcopy(before)
	after["resources"][6] -= cost(TOWN_HALL)
	after["towns"][0]["built"].append(TOWN_HALL)
	assert total(after) == pytest.approx(total(before))


def test_a_new_dwelling_with_its_first_batch_is_neutral_and_growth_is_a_gain():
	before = player()
	before["towns"][0]["built"] = [VILLAGE_HALL]
	before["towns"][0]["available"] = [[-1, 0]] * 7
	after = copy.deepcopy(before)
	price = vocab.buildings()[(CASTLE, DWELLING_1)]["cost"]
	for i, name in enumerate(("wood", "mercury", "ore", "sulfur", "crystal", "gems", "gold")):
		after["resources"][i] -= price.get(name, 0)
	after["towns"][0]["built"].append(DWELLING_1)
	after["towns"][0]["available"][0] = [PIKEMAN, vocab.creatures()[PIKEMAN]["growth"]]
	assert total(after) == pytest.approx(total(before))
	monday = copy.deepcopy(after)
	monday["towns"][0]["available"][0][1] *= 2
	assert total(monday) > total(after)


def test_recruiting_into_a_town_is_slightly_negative_into_a_starting_hero_neutral_and_an_upgrade_gains_what_its_hero_makes_of_it():
	"""For a hero already armed (an army worth armed_army or more): below that, creatures also lift its HERO_BASE
	(test_a_hero_with_nothing_is_worth_less_than_one_with_creatures). A garrison's creatures count at unled_share of their
	worth (the user, 2026-11-05: buying an army slightly negative, ~4.7% at 0.953), a starting hero with only spell power 3
	and knowledge 2 and its spellbook makes it neutral; a hero's at its multiplier, so upgrading them adds that much more
	of the upgrade's worth (2026-11-04)."""
	before = player()
	before["heroes"][0]["army"] = [[PIKEMAN, 30]]  # (30 x (60 + u): armed)
	before["heroes"][0].update(attack=2, defense=2)  # (a starting might hero's: above the unit hero, x1.026)
	assert army_worth(before["heroes"][0]["army"]) >= EmpireConfig().armed_army
	recruited = copy.deepcopy(before)
	recruited["resources"][6] -= 5 * vocab.creature_cost(PIKEMAN)
	recruited["towns"][0]["available"][0][1] -= 5
	recruited["towns"][0]["garrison"] = [[PIKEMAN, 5]]
	assert 0.95 < unled_share() < 1 and total(recruited) == pytest.approx(total(before) - (1 - unled_share()) * army_worth([[PIKEMAN, 5]]))
	starting = copy.deepcopy(before)  # (a starting hero: attack and defense 0, spell power 3, knowledge 2, its spellbook)
	starting["heroes"][0].update(attack=0, defense=0, power=REFERENCE_POWER, knowledge=REFERENCE_KNOWLEDGE, spellbook=True)
	into_hero = copy.deepcopy(starting)
	into_hero["resources"][6] -= 5 * vocab.creature_cost(PIKEMAN)
	into_hero["towns"][0]["available"][0][1] -= 5
	into_hero["heroes"][0]["army"] = [[PIKEMAN, 35]]
	assert hero_multiplier(starting["heroes"][0]) == pytest.approx(1.0) and total(into_hero) == pytest.approx(total(starting))
	upgraded = copy.deepcopy(before)
	upgraded["resources"][6] -= 30 * (vocab.creature_cost(HALBERDIER) - vocab.creature_cost(PIKEMAN))
	upgraded["heroes"][0]["army"] = [[HALBERDIER, 30]]
	gain = (hero_multiplier(before["heroes"][0]) - 1) * 30 * (vocab.creature_cost(HALBERDIER) - vocab.creature_cost(PIKEMAN))
	assert gain > 0 and total(upgraded) - total(before) == pytest.approx(gain)


def test_an_army_is_worth_more_with_the_stronger_hero():
	"""(the user, 2026-11-04: keeping the army with the strongest hero must be good) Moving an army from a hero of attack
	and defense 1 to one of 10 and 10 adds its worth x the difference of their army_multiplier; VCMI's modifiers: 10
	attack deals x1.5, 10 defense takes x0.75."""
	before = player()
	weak = fresh_hero(1, army=((PIKEMAN, 30),), attack=1, defense=1)
	strong = fresh_hero(2, army=((PIKEMAN, 1),), attack=10, defense=10)
	before["heroes"] = [weak, strong]
	after = copy.deepcopy(before)
	after["heroes"][0]["army"], after["heroes"][1]["army"] = [[PIKEMAN, 1]], [[PIKEMAN, 30]]
	moved = army_worth(((PIKEMAN, 29),))
	assert hero_multiplier(strong) == pytest.approx(unled_share() * math.sqrt(1.5 / 0.75))  # (no spellbook: its attack and defense alone)
	assert total(after) - total(before) == pytest.approx(moved * (hero_multiplier(strong) - hero_multiplier(weak)))


def test_a_hire_brings_a_hero_and_its_army_for_its_price_and_the_crowding():
	"""(the user, 2026-10-29) A hero with the army a week's tavern heroes come with is a gain; one with a lone creature
	a loss."""
	config, before = EmpireConfig(), player()

	def hire(army):
		after = copy.deepcopy(before)
		after["resources"][6] -= HERO_HIRE_COST
		after["heroes"].append(fresh_hero(1, army=army, mana=0, power=0))
		return total(after) - total(before)

	crowding = config.hero_penalty * (2**2 - 1**2)
	ten = army_worth(((PIKEMAN, 10),))
	assert ten == pytest.approx(10 * vocab.creature_cost(PIKEMAN) * (1 + config.alpha))
	assert hire(((PIKEMAN, 10),)) == pytest.approx(hero_base_value(ten) + ten * hero_multiplier(fresh_hero(1)) - HERO_HIRE_COST - crowding)
	assert hire(((PIKEMAN, 30), (HALBERDIER, 10))) > 0 and hire(((PIKEMAN, 1),)) < 0


def test_a_hero_with_nothing_is_worth_less_than_one_with_creatures():
	"""(the user, 2026-11-03) "a hero with nothing slightly negative, maybe -1k on average ... heroes with creatures
	should stay positive": HERO_BASE at armyless_share with no army, rising with the army to all of it at armed_army."""
	config, before = EmpireConfig(), player()

	def hire(army):
		after = copy.deepcopy(before)
		after["resources"][6] -= HERO_HIRE_COST
		after["heroes"].append(fresh_hero(1, army=army, mana=0, power=0))
		return total(after) - total(before)

	assert hero_base_value(0.0) == pytest.approx(config.armyless_share * config.hero_base)
	assert hero_base_value(config.armed_army) == pytest.approx(config.hero_base) == hero_base_value(10 * config.armed_army)
	assert hero_base_value(config.armed_army / 2) == pytest.approx(config.hero_base * (1 + config.armyless_share) / 2)
	crowding = config.hero_penalty * (2**2 - 1**2)
	# (a bare hero: no skills beyond a fresh one's, no artifacts; joint7's hires without their army came to ~-1,040)
	assert hire(()) == pytest.approx(config.armyless_share * config.hero_base - HERO_HIRE_COST - crowding)
	assert hire(()) < -1000
	assert hire(((PIKEMAN, 30),)) > 0


def test_a_level_is_worth_what_it_brings():
	"""A level's worth is its primary point, its skill level and its bonus (empire.skill_value); the way to it counts
	linearly; the k-th primary point beyond a fresh hero's five is worth k primary points; a level-up is ~2.6k at level
	1 and grows with the hero's points."""
	config = EmpireConfig()
	fresh = fresh_hero(0)
	assert skill_value(fresh) == 0
	leveled = fresh_hero(0, exp=1000, primary=[2, 1, 1, 2], skills=[[0, 2], [1, 1]])  # level 2: +1 attack, Advanced skill 0
	gain = config.primary_point * 1 + config.skill_level + config.level_bonus
	assert skill_value(leveled) == pytest.approx(gain) and 2000 < gain < 3000
	assert skill_value(fresh_hero(0, exp=500)) == pytest.approx(gain / 2)  # halfway to level 2: half its level-up
	strong = fresh_hero(0, primary=[6, 5, 2, 1])  # 14 points: 9 beyond five
	assert skill_value(strong) == pytest.approx(config.primary_point * 9 * 10 / 2)
	stronger = fresh_hero(0, primary=[7, 5, 2, 1])
	assert skill_value(stronger) - skill_value(strong) == pytest.approx(config.primary_point * 10)  # the 10th point beyond: worth 10


def test_only_the_strongest_heros_skills_count_in_full():
	config = EmpireConfig()
	main = fresh_hero(0, exp=1000, primary=[2, 1, 1, 2], skills=[[0, 2], [1, 1]])
	scout = fresh_hero(1, skills=[[0, 1], [1, 1], [2, 1]])  # one skill level beyond a fresh hero's
	value = heroes_value([main, scout], ISOLATED)
	bases = hero_base_value(army_worth(main["army"])) + hero_base_value(army_worth(scout["army"]))
	assert value == pytest.approx(bases + skill_value(main) + config.secondary_share * skill_value(scout))
	assert heroes_value([scout, main], ISOLATED) == pytest.approx(value)  # whatever the order
	assert heroes_value([]) == 0


def test_spells_count_by_their_level_among_a_heros_skills():
	"""Each spell in a hero's spellbook is worth SPELL_LEVEL x its level, among its skills: a scout's at SECONDARY_SHARE."""
	config = EmpireConfig()
	caster = fresh_hero(0, spells=[MAGIC_ARROW, HASTE, FIREBALL])  # levels 1, 1 and 3
	assert skill_value(caster) == pytest.approx(config.spell_level * 5)
	assert skill_value(fresh_hero(0, spells=[])) == 0 and skill_value(fresh_hero(0)) == 0  # (a ledger before Φ v3: no spells)
	main = fresh_hero(1, exp=1000, primary=[2, 1, 1, 2], skills=[[0, 2], [1, 1]])
	assert heroes_value([main, caster]) - heroes_value([main, fresh_hero(0)]) == pytest.approx(config.secondary_share * config.spell_level * 5)


def test_luck_and_morale_until_the_next_battle_are_worth_the_battle_chance_they_add_to_the_army_they_lift():
	"""BOOST of the army's worth (cost + u) for each 24th of battle chance: luck lifts every creature, morale only the
	living; within the +-3 caps; bad morale costs 2/24 a point, bad luck nothing (VCMI's gameConfig.json)."""
	config = EmpireConfig()
	army = 10 * (vocab.creature_cost(PIKEMAN) + config.alpha * vocab.creature_cost(PIKEMAN))
	temple = fresh_hero(0, morale=1, battle_morale=1)
	assert boost_value(temple) == pytest.approx(config.boost * army)
	rally_flag = fresh_hero(0, luck=1, battle_luck=1, morale=2, battle_morale=1)  # +1 morale over a permanent +1
	assert boost_value(rally_flag) == pytest.approx(config.boost * 2 * army)
	assert boost_value(fresh_hero(0, morale=4, battle_morale=1)) == 0  # +3 already: a Temple adds nothing
	assert boost_value(fresh_hero(0, morale=3, battle_morale=2)) == pytest.approx(config.boost * 2 * army)  # +1 -> +3
	tomb = fresh_hero(0, morale=-3, battle_morale=-3)  # a Warrior's Tomb: 6/24 of the turns lost
	assert boost_value(tomb) == pytest.approx(-6 * config.boost * army)
	assert boost_value(fresh_hero(0, luck=-1, battle_luck=-1)) == 0  # bad luck does nothing in this configuration
	undead = fresh_hero(0, army=((SKELETON, 10),), luck=1, battle_luck=1, morale=1, battle_morale=1)
	skeletons = 10 * (vocab.creature_cost(SKELETON) + config.alpha * vocab.creature_cost(vocab.base_forms()[SKELETON]))
	assert boost_value(undead) == pytest.approx(config.boost * skeletons)  # the luck only
	assert boost_value(fresh_hero(0, morale=2)) == 0  # nothing until the next battle: nothing to price
	assert heroes_value([temple], ISOLATED) - heroes_value([fresh_hero(0, morale=0)], ISOLATED) == pytest.approx(boost_value(temple))  # in Φ, every hero's own


def test_mana_counts_only_what_a_short_battle_can_spend():
	assert capped_mana_value(1000, 3, 24) == capped_mana_value(900, 3, 24)
	assert capped_mana_value(20, 3, 24) > capped_mana_value(5, 3, 24)
	assert capped_mana_value(100, 3, 0) == 0  # no combat spell


def test_explored_tiles_and_elimination():
	assert total(player(explored=200)) - total(player(explored=100)) == pytest.approx(100 * EmpireConfig().explored_tile)
	assert total(player(status=1)) == 0


def test_shares_add_up_to_one():
	snapshot = {"players": [player(), player(color=1, explored=300), player(color=2, status=1)]}
	s = shares(snapshot)
	assert sum(s.values()) == pytest.approx(1) and s[2] == 0 and s[1] > s[0]


def test_a_heros_war_machines_are_half_their_price_and_the_catapult_nothing():
	"""(2026-10-28) The catapult every hero has is no asset (the spellbook is, at its price: 2026-10-29); a ballista, ammo cart or first aid tent
	half its price (game_values.WAR_MACHINE_ARTIFACTS), as in the battle reward: at full price a fresh hero's kit made a
	hire pay for itself in Φ (joint1's hero spam), at 0 a bought machine was a pure loss."""
	from campaign_model.empire import empire_value

	base = player()
	kit = player()
	kit["heroes"] = [dict(h, artifacts=[3]) for h in base["heroes"]]
	assert total(kit) == total(base)
	kit["heroes"] = [dict(h, artifacts=[3, 4, 5, 6]) for h in base["heroes"]]
	assert total(kit) == pytest.approx(total(base) + len(base["heroes"]) * (1250 + 500 + 375))
	kit["heroes"] = [dict(h, artifacts=[0]) for h in base["heroes"]]  # (2026-10-29) a spellbook at its price, and (2026-11-05) its spell factor on the army
	factor = sum(army_worth(h["army"]) * (hero_multiplier(dict(h, artifacts=[0])) - hero_multiplier(h)) for h in base["heroes"])
	assert factor > 0 and total(kit) == pytest.approx(total(base) + len(base["heroes"]) * 500 + factor)


def _creature(key):
	return next(i for i, row in vocab.creatures().items() if row["key"] == key)


def test_a_slow_or_foreign_creature_costs_the_army_its_movement_and_morale():
	"""Φ v5 (the user, 2026-11-05): a hero's army counts at its movement, game_values.movement_factor ((the movement of its
	slowest creature's speed / 2000) ^ 0.25), and its make-up's morale (one faction +1, two 0, any undead -1 more; 2% of
	the living army a 24th of battle chance, -4% a point below 0). Five Angels are worth more alone than with two Dwarves
	(speed 3: 1500 a day, x0.931; two factions: +1 lost) or a Zombie (speed 4, undead: morale -1), whichever garrison
	keeps them; ten Crusaders (speed 6, Castle's too) still add more than they slow. The evaluation's Φ (EVAL_EMPIRE)
	sees none of it."""
	from campaign_model.empire import EVAL_EMPIRE, army_make_up, make_up_value, speed_factor

	angel, dwarf, zombie, crusader = _creature("core:angel"), _creature("core:dwarf"), _creature("core:zombieLord"), _creature("core:crusader")
	assert speed_factor([[angel, 5]]) == 1.0 and speed_factor([[angel, 5], [dwarf, 2]]) == pytest.approx((1500 / 2000) ** 0.25)
	assert (army_make_up([[angel, 5]]), army_make_up([[angel, 5], [dwarf, 2]]), army_make_up([[angel, 5], [zombie, 1]])) == (1, 0, -1)

	def with_hero(army, garrison):
		p = player()
		p["heroes"] = [fresh_hero(0, army=army, morale=army_make_up(army))]  # (a hero's morale holds its army's make-up's)
		p["towns"][0]["garrison"] = [list(g) for g in garrison]
		return p

	angels = with_hero(((angel, 5),), ((dwarf, 2), (zombie, 1)))
	for kept, left in ((((angel, 5), (dwarf, 2)), ((zombie, 1),)), (((angel, 5), (zombie, 1)), ((dwarf, 2),))):
		slowed = with_hero(kept, left)
		assert total(angels, EmpireConfig()) > total(slowed, EmpireConfig())
		moved = army_worth(kept) - army_worth([[angel, 5]])  # (under the evaluation's rules a move changes only what the hero makes of it)
		assert total(slowed, EVAL_EMPIRE) - total(angels, EVAL_EMPIRE) == pytest.approx(moved * (hero_multiplier(slowed["heroes"][0], EVAL_EMPIRE) - 1.0), rel=1e-6)
	assert make_up_value(angels["heroes"][0]) == pytest.approx(0.02 * army_worth([[angel, 5]]))  # (+1: 1/24 more turns)
	crusaders = with_hero(((angel, 5), (crusader, 10)), ())
	assert total(crusaders, EmpireConfig()) > total(with_hero(((angel, 5),), ()), EmpireConfig()) + 0.9 * army_worth([[crusader, 10]]) - 0.05 * army_worth([[angel, 5], [crusader, 10]])
