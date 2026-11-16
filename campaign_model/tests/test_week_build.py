"""BUILD's week term (the user, 2026-11-15: "an input that is how many days we have until the next Sunday/7 ... I want to
be able to learn that new creature dwellings are getting the extra value on Sunday easily ... and have a reverse-ish,
learnable effect on the other dwelling types"; e5's spec): build_prior += week_weights[class] x near, near = (4 - 7 d) / 3
from the days until the next growth d (1 on Monday .. 1/7 on Sunday), the classes a new dwelling, a dwelling's upgrade, a
growth building (citadel, castle, a horde) and every other; four bounded corrections at 0 to begin with."""
import math
from types import SimpleNamespace

import pytest
import torch

from campaign_model import constants as c
from campaign_model import features as ft
from campaign_model.model import CORRECTION_STEP, CampaignModel
from campaign_model.schema import empty_inputs
from campaign_model.train import make_optimizer
from campaign_model.config import CampaignConfig
from campaign_model.tests.test_stockpile import CAPITOL_OPEN, inputs, player

DWELLING, UPGRADE, GROWTH, OTHER = range(4)


def raw(v):
	"""The raw parameter that bounded(-2, 2, 0) reads as v."""
	p = (v + 2.0) / 4.0
	return math.log(p / (1.0 - p)) / CORRECTION_STEP


def test_the_classes_of_each_factions_buildings():
	t = CampaignModel(dim=32, cell_dim=16, layers=1).features
	k = t.build_week_class
	for faction in range(c.FACTION_VOCAB):
		assert k[faction, 30:37].tolist() == [DWELLING] * 7 and k[faction, 37:44].tolist() == [UPGRADE] * 7
		assert k[faction, 8] == GROWTH and k[faction, 9] == GROWTH  # (citadel, castle)
		for b in (5, 10, 11, 12, 13, 0, 14, 7):  # (tavern, the halls, mage guild 1, marketplace, fort)
			assert k[faction, b] == OTHER, (faction, b)
	assert k[0, 18] == GROWTH and k[0, 19] == GROWTH and k[0, 24] == OTHER  # (Castle: the Griffin Bastion, no second horde)
	assert k[1, 24] == GROWTH and k[1, 25] == GROWTH  # (Rampart's second horde)
	assert ((k == GROWTH) == ((t.horde_level >= 0) | (torch.arange(c.N_BUILDINGS) == 8) | (torch.arange(c.N_BUILDINGS) == 9))[None]).all()


@pytest.mark.parametrize("day,near", [(1, -1.0), (4, 0.0), (7, 1.0), (8, -1.0), (14, 1.0), (2, -2 / 3), (29, -1.0)])
def test_near(day, near):
	x = empty_inputs(1)
	x["g_day"][0] = day
	assert float(ft.week_near({k: torch.from_numpy(v) for k, v in x.items()})[0]) == pytest.approx(near, abs=1e-6)


def build_prior(t, x):
	t.build_economy.fill_(1.0)
	with torch.no_grad():
		_, value = ft.build_features(x, t, ft.economy_features(x, t))
		return t.build_prior(x, SimpleNamespace(build_value=value, stock_threshold=ft.stock_thresholds(x, t)))


def test_zero_to_begin_with_and_a_weight_moves_only_its_class_by_near():
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	t = model.features
	assert any(p is t.week_weights for p in t.corrections())
	assert [t.correction_values()[f"w{k + 1}_week_{n}"] for k, n in enumerate(ft.WEEK_CLASSES)] == [0.0] * 4
	try:
		for day, near in ((7, 1.0), (1, -1.0), (4, 0.0)):
			x = inputs(player([CAPITOL_OPEN], [40, 20, 40, 20, 20, 20, 9000]))
			x["g_day"][0] = day
			with torch.no_grad():
				t.week_weights.zero_()
			plain = build_prior(t, x)
			assert float(ft.week_correction(x, t, torch.zeros(1, c.OWN_TOWNS, dtype=torch.long)).abs().max()) == 0.0
			with torch.no_grad():
				t.week_weights[DWELLING] = raw(1.0)
			moved = build_prior(t, x) - plain
			assert t.correction_values()["w1_week_dwelling"] == pytest.approx(1.0, abs=1e-5)
			assert torch.allclose(moved[0, 0, 30:37], torch.full((7,), near), atol=1e-5), (day, moved[0, 0, 30:37])
			others = torch.cat([moved[0, 0, :30], moved[0, 0, 37:]])
			assert float(others.abs().max()) <= 1e-5, (day, float(others.abs().max()))
	finally:
		with torch.no_grad():
			t.week_weights.zero_()
		t.build_economy.fill_(0.0)


def test_a_step_moves_the_four_weights():
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	t = model.features
	t.build_economy.fill_(1.0)
	x = inputs(player([CAPITOL_OPEN], [40, 20, 40, 20, 20, 20, 9000]))
	x["g_day"][0] = 6
	_, value = ft.build_features(x, t, ft.economy_features(x, t))
	prior = t.build_prior(x, SimpleNamespace(build_value=value, stock_threshold=ft.stock_thresholds(x, t)))
	(prior[0, 0, :] * torch.randn(c.N_BUILDINGS)).sum().backward()
	assert (t.week_weights.grad.abs() > 0).all(), t.week_weights.grad
	optimizer = make_optimizer(model, CampaignConfig())
	before = t.week_weights.detach().clone()
	optimizer.step()
	assert (t.week_weights.detach() != before).all()
	t.build_economy.fill_(0.0)
