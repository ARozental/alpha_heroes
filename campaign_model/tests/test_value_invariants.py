"""Invariants of what the campaign model's hand-made values claim: nothing can be scored that cannot be had. Each
test is a rule a killer bug would break (2026-10-28: a guarded treasure counted when its fight is lost made the prior
send heroes to their death); campaign_model/audit.py checks the same claims against real games."""
import pytest
import torch

from campaign_model import constants as c
from campaign_model import features as ft
from campaign_model.model import CampaignModel
from campaign_model.schema import empty_inputs
from campaign_model.tests.test_model import PIKEMAN, window_with

CASTLE, VILLAGE_HALL, TOWN_HALL, ANGEL = 0, 10, 11, 13


@pytest.fixture(scope="module")
def model():
	torch.manual_seed(0)
	return CampaignModel(dim=32, cell_dim=16, layers=1).eval()


def features(model, x):
	t = {k: torch.from_numpy(v) for k, v in x.items()}
	with torch.no_grad():
		windows, _ = model.windows(t)
		return model.features(t, windows)


@pytest.mark.parametrize("obj", [79, 101, 53, 5, 6, 98, 16, 25, 112])
def test_nothing_behind_a_hopeless_fight_is_worth_anything(model, obj):
	"""Whatever a tile holds, guarded by an army far beyond ours, it is worth less than nothing."""
	x, tile = window_with(obj)
	row, column = tile
	x["w_creature"][0, 0, row, column], x["w_count"][0, 0, row, column] = ANGEL, 9  # a legion of angels on it
	f = features(model, x)
	assert float(f.tile_value[0][tile]) < 0 and float(f.tile_prior[0][tile]) < 0


def test_empty_revealed_land_is_worth_nothing(model):
	x, _ = window_with(-1)
	f = features(model, x)
	assert float(f.tile_value[0].abs().max()) == 0 and float(f.tile_prior[0].abs().max()) == 0


def test_an_empty_revealed_area_offers_nothing_to_travel_to(model):
	x = empty_inputs(1)
	x["g_map"][0], x["h_present"][0, 0], x["h_owner"][0, 0], x["h_move_max"][0, 0] = [36, 36, 1], True, c.OWNER_SELF, 1500
	x["g_tiles"][0, 0, :, :, c.GridCount.ON_MAP] = x["g_tiles"][0, 0, :, :, c.GridCount.REVEALED] = 1
	f = features(model, x)
	assert float(f.travel_prior[0, 0, : c.TRAVEL_GRID**2].abs().max()) == 0


def empire(model, x):
	parts, total = ft.empire_value({k: torch.from_numpy(v) for k, v in x.items()}, model.features)
	return float(total[0])


def own_town(x):
	x["t_present"][0, 0], x["t_owner"][0, 0], x["t_faction"][0, 0], x["t_detailed"][0, 0] = True, c.OWNER_SELF, CASTLE, True
	x["t_built"][0, 0, VILLAGE_HALL] = True
	x["g_resources"][0] = [20, 10, 20, 10, 10, 10, 20000]
	return x


def test_recruiting_at_list_price_into_a_garrison_costs_its_unled_share(model):
	"""Gold into creatures (each for hire at u, hired at cost + u): into a garrison they count at unled_share of their
	worth (the user, 2026-11-05: buying an army slightly negative unless a hero leads it), empire.py's."""
	from campaign_model.empire import army_worth
	from campaign_model.vocab import creature_cost
	from game_values import unled_share

	x = own_town(empty_inputs(1))
	x["t_available"][0, 0, 0], x["t_available_creature"][0, 0, 0] = 14, PIKEMAN
	before = empire(model, x)
	x["t_available"][0, 0, 0] -= 10
	x["t_garrison_creature"][0, 0, 0], x["t_garrison_count"][0, 0, 0] = PIKEMAN, 10
	x["g_resources"][0, 6] -= 10 * creature_cost(PIKEMAN)
	assert empire(model, x) == pytest.approx(before - (1 - unled_share()) * army_worth([[PIKEMAN, 10]]), abs=1.0)


def test_building_at_list_price_is_worth_nothing(model):
	from campaign_model.vocab import buildings
	from game_values import resources_value

	x = own_town(empty_inputs(1))
	before = empire(model, x)
	cost = buildings()[(CASTLE, 5)]["cost"]  # a building without income (the tavern)
	x["t_built"][0, 0, 5] = True
	order = ("wood", "mercury", "ore", "sulfur", "crystal", "gems", "gold")
	for r, name in enumerate(order):
		x["g_resources"][0, r] -= cost.get(name, 0)
	assert resources_value(cost) > 0 and empire(model, x) == pytest.approx(before, abs=1.0)
