"""The stockpile discount (the user, 2026-11-15; empire.py stock_floor f): above its threshold T a resource counts at a
marginal value falling from 1 toward f; T = the max of a floor, the most any startable building costs, and a week's
growth of our dwellings; the income at its flat price. The model's replica follows the ledger; a purchase's shown gain
is its change of Φ beyond today's; exactly as before at f = 1."""
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model import empire as E
from campaign_model import features as ft
from campaign_model.config import CampaignConfig
from campaign_model.learner import phi_now
from campaign_model.model import CampaignModel
from campaign_model.schema import empty_inputs
from campaign_model.tests.test_empire import CASTLE, PIKEMAN

TAVERN, VILLAGE_HALL, TOWN_HALL, CITY_HALL, CAPITOL = 5, 10, 11, 12, 13
MAGE_1, MARKET, BLACKSMITH, FORT, CITADEL, CASTLE_B = 0, 14, 16, 7, 8, 9
THIRD = 1.0 / 3.0


@pytest.mark.parametrize("threshold,amount,value", [
	(20000, 30000, 27954), (20000, 40000, 33991), (20000, 60000, 44063), (20000, 100000, 61315), (20000, 300000, 135782),
	(10000, 15000, 13977), (10000, 20000, 16995), (10000, 30000, 22031), (10000, 100000, 49815),
	(30, 45, 41.9), (30, 60, 51.0), (30, 100, 70.7),
])
def test_value_by_hand(threshold, amount, value):
	"""e5's table (f = 1/3)."""
	assert E.stock_value(amount, threshold, THIRD) == pytest.approx(value, abs=0.6 if threshold > 100 else 0.06)
	t = torch.tensor([float(amount)])
	assert float(ft.stock_value(t, torch.tensor([float(threshold)]), torch.tensor(THIRD))) == pytest.approx(E.stock_value(amount, threshold, THIRD), rel=1e-6)


def test_the_marginal_and_the_flat_value_at_floor_1():
	assert E.stock_marginal(20000, 20000.0, THIRD) == 1.0 and E.stock_marginal(30000, 20000.0, THIRD) == pytest.approx(2 / 3)
	assert E.stock_marginal(10**7, 20000.0, THIRD) == pytest.approx(THIRD, abs=1e-3)
	assert E.stock_value(123456, 20000.0, 1.0) == 123456 and E.stock_value(19999, 20000.0, THIRD) == 19999


def player(towns, resources, available=(), income=(0, 0, 0, 0, 0, 0, 0)):
	return {"color": 0, "status": 0, "resources": list(resources), "income": list(income), "explored": 0, "towns": [{"faction": CASTLE, "built": list(b), "garrison": [], "available": [list(a) for a in available] + [[-1, 0]] * (7 - len(available))} for b in towns], "heroes": [], "dwellings": [], "armies": []}


def inputs(p):
	x = empty_inputs(1)
	x["g_resources"][0] = p["resources"]
	x["g_object_income"][0] = p["income"]
	for t, town in enumerate(p["towns"]):
		x["t_present"][0, t], x["t_owner"][0, t], x["t_faction"][0, t] = True, c.OWNER_SELF, CASTLE
		x["t_built"][0, t, town["built"]] = True
		for level, (creature, count) in enumerate(town["available"][:7]):
			x["t_available_creature"][0, t, level], x["t_available"][0, t, level] = creature, count
	return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in x.items()}


EARLY = [TAVERN, VILLAGE_HALL, 30]
CAPITOL_OPEN = [TAVERN, VILLAGE_HALL, TOWN_HALL, CITY_HALL, MAGE_1, MARKET, BLACKSMITH, FORT, CITADEL, CASTLE_B, 30]
ALL = sorted({b for (f, b) in E.vocab.buildings() if f == CASTLE and b < 44})
CASES = {"early": [EARLY], "capitol startable": [CAPITOL_OPEN], "everything built": [ALL], "no town": []}


def test_the_thresholds_by_case():
	"""The max of three: the floors; one max over the towns of a startable building's cost; a week's growth."""
	early = E.stock_thresholds(player(CASES["early"], [0] * 7))
	capitol = E.stock_thresholds(player(CASES["capitol startable"], [0] * 7))
	nothing = E.stock_thresholds(player(CASES["everything built"], [0] * 7))
	assert E.stock_thresholds(player([], [0] * 7)) == list(E.STOCK_FLOORS)  # (no town: the floors, T_gold 10,000)
	assert capitol[6] == max(10000.0, 10000.0)  # (Capitol's 10,000 gold, the floor's 10,000: a week of two dwellings is less)
	assert early[6] == 10000.0  # (City Hall not startable early: the floor)
	growth = E.week_growth(CASTLE, set(ALL))
	week_gold = sum(count * E._resource_vector(E.vocab.creatures()[E.vocab.town_creatures()[CASTLE][lv][1]].get("cost"))[6] for lv, count in growth.items())
	assert nothing[6] == pytest.approx(max(10000.0, week_gold))  # (nothing startable: the week's growth, upgraded, castle x2, hordes)
	assert week_gold > 10000


@pytest.mark.parametrize("case", list(CASES))
def test_the_replica_follows_the_ledger(case):
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	t = model.features
	try:
		for resources in ([5, 2, 5, 2, 2, 2, 1500], [200, 60, 200, 50, 40, 70, 180000]):
			p = player(CASES[case], resources, available=[(PIKEMAN, 14)] if CASES[case] else (), income=(2, 1, 2, 0, 0, 1, 500))
			x = inputs(p)
			with torch.no_grad():
				assert ft.stock_thresholds(x, t)[0].tolist() == pytest.approx(E.stock_thresholds(p), rel=1e-6)
			for floor in (1.0, THIRD):
				t.phi_stock_floor.fill_(floor)
				expected = E.empire_value(p, replace(E.EmpireConfig(), stock_floor=floor))
				with torch.no_grad():
					assert float(ft.stock_resources(x, t)[0]) == pytest.approx(expected["resources"], rel=1e-5), (case, floor)
				assert expected["income"] == E.empire_value(p)["income"]  # (the income at its flat price)
	finally:
		t.phi_stock_floor.fill_(1.0)


def test_a_purchases_shown_gain_is_its_change_of_phi_beyond_todays_and_0_under_the_threshold():
	"""With the thresholds where they were (gold spent on nothing that moves them), stock_gain is the resources part's
	change less the price; nothing while the stock stays at or under the threshold; 0 at f = 1."""
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	t = model.features
	try:
		for floor in (1.0, THIRD):
			t.phi_stock_floor.fill_(floor)
			for gold, spend in ((180000, 30000), (180000, 2000), (5000, 2000)):
				before = inputs(player([ALL], [10, 4, 10, 4, 4, 4, gold]))
				after = inputs(player([ALL], [10, 4, 10, 4, 4, 4, gold - spend]))
				with torch.no_grad():
					threshold = ft.stock_thresholds(before, t)
					cost = torch.zeros(1, 7)
					cost[0, 6] = spend
					gain = float(ft.stock_gain(cost, before["g_resources"].clamp(min=0).float(), threshold, t)[0])
					change = float(ft.stock_resources(after, t)[0] - ft.stock_resources(before, t)[0])
				assert change == pytest.approx(-spend + gain, abs=0.05), (floor, gold, spend)
				if floor == 1.0 or gold <= float(threshold[0, 6]):
					assert gain == pytest.approx(0.0, abs=1e-6)
				else:
					assert gain > 0
	finally:
		t.phi_stock_floor.fill_(1.0)


def test_the_floor_rides_the_phi_ramp_and_the_evaluation_keeps_1(monkeypatch):
	config = CampaignConfig(phi_stock_floor=THIRD, phi_from=100, phi_ramp=20)
	assert phi_now(config, 99)["stock_floor"] == 1.0
	assert phi_now(config, 109) == pytest.approx({**phi_now(config, 109), "stock_floor": 1 - 10 * (2 / 3) / 20})
	assert phi_now(config, 200)["stock_floor"] == pytest.approx(THIRD)
	monkeypatch.setenv("ALPHA_EMPIRE_STOCK_FLOOR", "0.3333")
	assert E.EmpireConfig().stock_floor == pytest.approx(0.3333) and E.EVAL_EMPIRE.stock_floor == 1.0
