import copy

import pytest

from campaign_model import vocab
from campaign_model.empire import EmpireConfig, building_value, empire_value, shares
from game_values import HERO_HIRE_COST, capped_mana_value, resources_value

CASTLE = 0
VILLAGE_HALL, TOWN_HALL, CITY_HALL, CAPITOL, DWELLING_1, DWELLING_1_UP = 10, 11, 12, 13, 30, 37
PIKEMAN, HALBERDIER = 0, 1


def cost(building):
	return resources_value(vocab.buildings()[(CASTLE, building)]["cost"])


def player(**changes):
	p = {
		"color": 0, "status": 0, "resources": [10, 4, 10, 4, 4, 4, 10000], "income": [0] * 7, "explored": 100,
		"towns": [{"faction": CASTLE, "built": [VILLAGE_HALL, DWELLING_1], "garrison": [], "available": [[PIKEMAN, 14]] + [[-1, 0]] * 6}],
		"heroes": [{"type": 0, "level": 1, "exp": 0, "mana": 10, "power": 1, "spell_cost": 0, "army": [[PIKEMAN, 10]], "artifacts": []}],
		"dwellings": [], "armies": [],
	}
	p.update(changes)
	return p


def total(p):
	return empire_value(p)["total"]


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


def test_recruiting_and_upgrading_are_neutral():
	before = player()
	recruited = copy.deepcopy(before)
	recruited["resources"][6] -= 5 * vocab.creature_cost(PIKEMAN)
	recruited["towns"][0]["available"][0][1] -= 5
	recruited["towns"][0]["garrison"] = [[PIKEMAN, 5]]
	assert total(recruited) == pytest.approx(total(before))
	upgraded = copy.deepcopy(before)
	upgraded["resources"][6] -= 10 * (vocab.creature_cost(HALBERDIER) - vocab.creature_cost(PIKEMAN))
	upgraded["heroes"][0]["army"] = [[HALBERDIER, 10]]
	assert total(upgraded) == pytest.approx(total(before))


def test_hiring_a_fresh_hero_costs_its_price_less_its_army_and_the_crowding():
	before = player()
	after = copy.deepcopy(before)
	after["resources"][6] -= HERO_HIRE_COST
	after["heroes"].append({"type": 1, "level": 1, "exp": 0, "mana": 0, "power": 0, "spell_cost": 0, "army": [[PIKEMAN, 10]], "artifacts": []})
	army = 10 * vocab.creature_cost(PIKEMAN) * (1 + EmpireConfig().alpha)
	crowding = EmpireConfig().hero_penalty * (2**2 - 1**2)
	assert total(after) - total(before) == pytest.approx(army - HERO_HIRE_COST - crowding)


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
	"""(2026-10-28) The catapult every hero has and the spellbook are no asset; a ballista, ammo cart or first aid tent
	half its price (game_values.WAR_MACHINE_ARTIFACTS), as in the battle reward: at full price a fresh hero's kit made a
	hire pay for itself in Φ (joint1's hero spam), at 0 a bought machine was a pure loss."""
	from campaign_model.empire import empire_value

	base = player()
	kit = player()
	kit["heroes"] = [dict(h, artifacts=[3, 0]) for h in base["heroes"]]
	assert empire_value(kit)["total"] == empire_value(base)["total"]
	kit["heroes"] = [dict(h, artifacts=[3, 4, 5, 6, 0]) for h in base["heroes"]]
	assert empire_value(kit)["total"] == pytest.approx(empire_value(base)["total"] + len(base["heroes"]) * (1250 + 500 + 375))