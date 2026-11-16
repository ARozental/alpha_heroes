"""The stockpile thresholds' gold floor and B's form (the user via e5, 2026-11-16: "We do want it to go down faster";
empire.gold_floor, stock_b_dwelling; the model's phi_gold_floor, phi_stock_b): today's values bit for bit, the replica
equal to empire at the new ones, B over the dwellings alone, the ramp by version, the evaluation's EVAL_EMPIRE untouched."""
from dataclasses import replace

import pytest
import torch

from campaign_model import empire as E
from campaign_model import features as ft
from campaign_model.config import CampaignConfig
from campaign_model.learner import gold_now, set_advice_prior
from campaign_model.model import CampaignModel
from campaign_model.tests.test_stockpile import CASES, PIKEMAN, inputs, player


def old_thresholds(p):
	"""empire.stock_thresholds before 2026-11-16 (STOCK_FLOORS whole, B over every building), for the bit-for-bit check."""
	return E.stock_thresholds(p, replace(E.EmpireConfig(), gold_floor=E.STOCK_FLOORS[6], stock_b_dwelling=False))


@pytest.mark.parametrize("case", list(CASES))
def test_todays_values_are_as_before_and_the_default(case, monkeypatch):
	monkeypatch.delenv("ALPHA_EMPIRE_GOLD_FLOOR", raising=False)
	monkeypatch.delenv("ALPHA_EMPIRE_STOCK_B", raising=False)
	p = player(CASES[case], [0] * 7)
	assert E.stock_thresholds(p) == old_thresholds(p)
	assert E.EmpireConfig().gold_floor == 10000.0 and E.EmpireConfig().stock_b_dwelling is False
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	assert float(model.features.phi_gold_floor) == 10000.0 and float(model.features.phi_stock_b) == 0.0


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("floor,b_on", [(5000.0, False), (5000.0, True), (2500.0, True), (10000.0, True)])
def test_the_replica_follows_empire_at_the_new_values(case, floor, b_on):
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	t = model.features
	config = replace(E.EmpireConfig(), gold_floor=floor, stock_b_dwelling=b_on)
	t.phi_gold_floor.fill_(floor)
	t.phi_stock_b.fill_(1.0 if b_on else 0.0)
	for resources in ([5, 2, 5, 2, 2, 2, 1500], [200, 60, 200, 50, 40, 70, 180000]):
		p = player(CASES[case], resources, available=[(PIKEMAN, 14)] if CASES[case] else (), income=(2, 1, 2, 0, 0, 1, 500))
		with torch.no_grad():
			assert ft.stock_thresholds(inputs(p), t)[0].tolist() == pytest.approx(E.stock_thresholds(p, config), rel=1e-6), (case, floor, b_on)
		for stock_floor in (1.0, 1 / 3):
			t.phi_stock_floor.fill_(stock_floor)
			expected = E.empire_value(p, replace(config, stock_floor=stock_floor))
			with torch.no_grad():
				assert float(ft.stock_resources(inputs(p), t)[0]) == pytest.approx(expected["resources"], rel=1e-5), (case, floor, b_on, stock_floor)


def test_b_over_the_dwellings_drops_a_capitol_from_the_threshold():
	p = player(CASES["capitol startable"], [0] * 7)
	today = E.stock_thresholds(p, replace(E.EmpireConfig(), gold_floor=5000.0))
	dwellings = E.stock_thresholds(p, replace(E.EmpireConfig(), gold_floor=5000.0, stock_b_dwelling=True))
	assert today[6] >= 10000.0  # (the Capitol's 10,000 sets B)
	assert dwellings[6] < today[6]  # (B over the dwellings alone: the Capitol no longer counts)


def test_the_ramp_by_version_and_the_one_source():
	off = CampaignConfig()
	assert gold_now(off, 5000) == (10000.0, False)
	on = CampaignConfig(phi_gold_floor=5000.0, phi_gold_from=100, phi_gold_ramp=10, phi_stock_b=True)
	assert gold_now(on, 99) == (10000.0, False)
	assert gold_now(on, 100) == (pytest.approx(9500.0), True)
	assert gold_now(on, 104)[0] == pytest.approx(7500.0) and gold_now(on, 109)[0] == pytest.approx(5000.0) and gold_now(on, 500) == (pytest.approx(5000.0), True)
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	set_advice_prior(model, on, 104)
	assert float(model.features.phi_gold_floor) == pytest.approx(7500.0) and float(model.features.phi_stock_b) == 1.0
	set_advice_prior(model, off, 104)
	assert float(model.features.phi_gold_floor) == 10000.0 and float(model.features.phi_stock_b) == 0.0


def test_the_evaluation_keeps_todays(monkeypatch):
	monkeypatch.setenv("ALPHA_EMPIRE_GOLD_FLOOR", "5000")
	monkeypatch.setenv("ALPHA_EMPIRE_STOCK_B", "1")
	assert E.EmpireConfig().gold_floor == 5000.0 and E.EmpireConfig().stock_b_dwelling is True
	assert E.EVAL_EMPIRE.gold_floor == 10000.0 and E.EVAL_EMPIRE.stock_b_dwelling is False
