"""Φ's heroes and the armies they lead (e5 for the user, 2026-11-15): a hero's own base (hero_base), its share with no
army (armyless_share) and the lead weight (lead_weight: an army at its worth x (1 + h x (its multiplier - 1))), today's
values by default; the ledger's (empire.py) and the model's replica alike (its phi_ buffers), the stakes on today's
constants; the learner's ramp (config.phi_*, learner.phi_now / _ramp_phi)."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from campaign_model import empire as E
from campaign_model import features as ft
from campaign_model.config import CampaignConfig
from campaign_model.learner import CampaignLearner, phi_now
from campaign_model.model import CampaignModel, with_added
from campaign_model.tests.test_empire import PIKEMAN, fresh_hero, player
from game_values import REFERENCE_KNOWLEDGE, REFERENCE_POWER, unled_share

# (the movement and the make-up's morale apart, as test_empire's ISOLATED: the rules of this test alone)
BASE = E.EmpireConfig(speed_power=0.0, army_morale=0.0)


def total(p, **values):
	return E.empire_value(p, replace(BASE, **values))["total"]


def armed_with_a_starting_hero():
	"""A player whose hero is Φ's unit hero (attack and defense 0, spell power 3, knowledge 2, its spellbook), armed."""
	before = player()
	before["heroes"][0]["army"] = [[PIKEMAN, 30]]
	before["heroes"][0].update(attack=0, defense=0, power=REFERENCE_POWER, knowledge=REFERENCE_KNOWLEDGE, spellbook=True)
	return before


@pytest.mark.parametrize("h", [1.0, 0.5])
def test_a_starting_heros_purchase_stays_neutral_at_any_lead_weight(h):
	before = armed_with_a_starting_hero()
	bought = copy.deepcopy(before)
	bought["resources"][6] -= 5 * E.vocab.creature_cost(PIKEMAN)
	bought["towns"][0]["available"][0][1] -= 5
	bought["heroes"][0]["army"] = [[PIKEMAN, 35]]
	assert E.hero_multiplier(before["heroes"][0]) == pytest.approx(1.0)
	assert total(bought, lead_weight=h) == pytest.approx(total(before, lead_weight=h), abs=1e-6)


@pytest.mark.parametrize("h", [1.0, 0.5])
def test_into_a_garrison_a_purchase_loses_h_times_what_it_lost_before(h):
	"""unled_share 0.953 at the spell step 0.02: -4.67% of the stack's worth at h = 1, -2.33% at 0.5."""
	before = armed_with_a_starting_hero()
	recruited = copy.deepcopy(before)
	recruited["resources"][6] -= 5 * E.vocab.creature_cost(PIKEMAN)
	recruited["towns"][0]["available"][0][1] -= 5
	recruited["towns"][0]["garrison"] = [[PIKEMAN, 5]]
	worth = E.army_worth([[PIKEMAN, 5]])
	loss = total(recruited, lead_weight=h) - total(before, lead_weight=h)
	assert loss == pytest.approx(h * (unled_share() - 1) * worth)
	assert loss / worth == pytest.approx({1.0: -0.0467, 0.5: -0.0233}[h], abs=5e-4)


def test_a_stronger_hero_makes_h_of_its_multiplier_of_an_army():
	"""Moving an army from a hero of attack and defense 1 to one of 10 and 10: its worth x h x the difference of their
	multipliers (test_empire's at h = 1)."""
	before = player()
	before["heroes"] = [fresh_hero(1, army=((PIKEMAN, 30),), attack=1, defense=1), fresh_hero(2, army=((PIKEMAN, 1),), attack=10, defense=10)]
	after = copy.deepcopy(before)
	after["heroes"][0]["army"], after["heroes"][1]["army"] = [[PIKEMAN, 1]], [[PIKEMAN, 30]]
	moved = E.army_worth(((PIKEMAN, 29),))
	difference = E.hero_multiplier(before["heroes"][1]) - E.hero_multiplier(before["heroes"][0])
	for h in (1.0, 0.5, 0.0):
		assert total(after, lead_weight=h) - total(before, lead_weight=h) == pytest.approx(h * moved * difference)


def test_at_the_targets_a_heros_base_is_825_whatever_its_army():
	config = replace(BASE, hero_base=825.0, armyless_share=1.0)
	assert E.hero_base_value(0.0, config) == E.hero_base_value(500.0, config) == E.hero_base_value(10 * config.armed_army, config) == 825.0


def test_the_evaluations_empire_keeps_todays_values_whatever_the_environment(monkeypatch):
	"""ALPHA_EMPIRE_HERO_BASE, _ARMYLESS, _LEAD_WEIGHT set a run's EmpireConfig() as ALPHA_EMPIRE_ALPHA does; EVAL_EMPIRE
	names today's values (its scores comparable across the change)."""
	monkeypatch.setenv("ALPHA_EMPIRE_HERO_BASE", "825")
	monkeypatch.setenv("ALPHA_EMPIRE_ARMYLESS", "1.0")
	monkeypatch.setenv("ALPHA_EMPIRE_LEAD_WEIGHT", "0.5")
	assert (E.EmpireConfig().hero_base, E.EmpireConfig().armyless_share, E.EmpireConfig().lead_weight) == (825.0, 1.0, 0.5)
	assert (E.EVAL_EMPIRE.hero_base, E.EVAL_EMPIRE.armyless_share, E.EVAL_EMPIRE.lead_weight) == (1500.0, 0.55, 1.0)
	monkeypatch.delenv("ALPHA_EMPIRE_HERO_BASE")
	monkeypatch.delenv("ALPHA_EMPIRE_ARMYLESS")
	monkeypatch.delenv("ALPHA_EMPIRE_LEAD_WEIGHT")
	assert (E.EmpireConfig().hero_base, E.EmpireConfig().armyless_share, E.EmpireConfig().lead_weight) == (E.HERO_BASE, E.ARMYLESS_SHARE, E.LEAD_WEIGHT) == (1500.0, 0.55, 1.0)


@pytest.fixture(scope="module")
def model():
	return CampaignModel(dim=32, cell_dim=16, layers=1).eval()


def test_the_models_lead_weight_and_the_stakes_constants(model):
	"""features.lead_factor / unled_lead / unled_loss at h: 1 + h x (m - 1); a hero's own worth (hero_worth, the stakes')
	keeps today's hero base and armyless share while Φ's buffers move."""
	t = model.features
	m = torch.tensor([0.9, 1.0, 1.3])
	try:
		for h in (1.0, 0.5):
			t.phi_lead_weight.fill_(h)
			assert torch.allclose(ft.lead_factor(m, t), 1 + h * (m - 1))
			assert float(ft.unled_lead(t)) == pytest.approx(1 + h * (unled_share() - 1))
			assert float(ft.unled_loss(t)) == pytest.approx(h * (unled_share() - 1))
		t.phi_lead_weight.fill_(1.0)
		assert torch.equal(ft.lead_factor(m, t), m)  # (the multiplier itself at 1, bit for bit)
		assert (t.hero_base, t.armyless_share) == (E.HERO_BASE, E.ARMYLESS_SHARE)
		x = {k: torch.from_numpy(v) for k, v in __import__("campaign_model.schema", fromlist=["random_inputs"]).random_inputs(batch=4, seed=2).items()}
		worth_today = ft.hero_worth(x, t)
		t.phi_hero_base.fill_(825.0)
		t.phi_armyless_share.fill_(1.0)
		assert torch.equal(ft.hero_worth(x, t), worth_today)
	finally:
		t.phi_hero_base.fill_(E.HERO_BASE)
		t.phi_armyless_share.fill_(E.ARMYLESS_SHARE)
		t.phi_lead_weight.fill_(E.LEAD_WEIGHT)


def test_the_ramp_moves_from_todays_values_to_the_targets_in_equal_steps():
	config = CampaignConfig(phi_hero_base=825.0, phi_armyless_share=1.0, phi_lead_weight=0.5, phi_from=100, phi_ramp=20)
	assert phi_now(config, 99) == {"hero_base": 1500.0, "armyless_share": 0.55, "lead_weight": 1.0}
	assert phi_now(config, 100) == pytest.approx({"hero_base": 1500.0 - 675.0 / 20, "armyless_share": 0.55 + 0.45 / 20, "lead_weight": 1.0 - 0.5 / 20})
	assert phi_now(config, 109) == pytest.approx({"hero_base": 1162.5, "armyless_share": 0.775, "lead_weight": 0.75})
	assert phi_now(config, 119) == phi_now(config, 500) == {"hero_base": 825.0, "armyless_share": 1.0, "lead_weight": 0.5}
	assert phi_now(CampaignConfig(), 10**4) == {"hero_base": 1500.0, "armyless_share": 0.55, "lead_weight": 1.0}


def test_the_learner_fills_the_labels_and_the_models_buffers_and_at_the_defaults_nothing(model):
	features_cpu = copy.deepcopy(model.features)
	empire = E.EmpireConfig()
	learner = SimpleNamespace(config=CampaignConfig(), version=5000, empire=empire, model=model, features_cpu=features_cpu)
	CampaignLearner._ramp_phi(learner)
	assert learner.empire is empire and float(model.features.phi_hero_base) == E.HERO_BASE
	learner.config = CampaignConfig(phi_hero_base=825.0, phi_armyless_share=1.0, phi_lead_weight=0.5, phi_from=4991, phi_ramp=20)
	try:
		CampaignLearner._ramp_phi(learner)  # (version 5000: the ramp's 10th step of 20)
		assert (learner.empire.hero_base, learner.empire.armyless_share, learner.empire.lead_weight) == pytest.approx((1162.5, 0.775, 0.75))
		assert learner.empire.alpha == empire.alpha and learner.empire.no_town_factor == empire.no_town_factor
		for features in (model.features, features_cpu):
			assert (float(features.phi_hero_base), float(features.phi_armyless_share), float(features.phi_lead_weight)) == pytest.approx((1162.5, 0.775, 0.75))
	finally:
		model.features.phi_hero_base.fill_(E.HERO_BASE)
		model.features.phi_armyless_share.fill_(E.ARMYLESS_SHARE)
		model.features.phi_lead_weight.fill_(E.LEAD_WEIGHT)


def test_a_checkpoint_from_before_loads_todays_values(model):
	old = {k: v for k, v in model.state_dict().items() if not k.startswith("features.phi_")}
	fresh = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	fresh.features.phi_hero_base.fill_(1.0)  # (anything: the loading model's own is what a checkpoint without them takes)
	fresh.load_state_dict(with_added(old, fresh.state_dict()))
	assert float(fresh.features.phi_hero_base) == 1.0
	again = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	again.load_state_dict(with_added(old, again.state_dict()))
	assert (float(again.features.phi_hero_base), float(again.features.phi_armyless_share), float(again.features.phi_lead_weight)) == pytest.approx((1500.0, 0.55, 1.0))
