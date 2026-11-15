"""The user's upgrade rule (2026-11-15, empire.py unit_own_form): u at a creature's own form. An upgrade gains ALPHA x the
cost difference a creature (at Φ's unit hero), a purchase stays neutral, a dwelling's upgrade gains it on every creature
for hire it converts; the model's Φ replica follows the ledger; off by default, the evaluation's Φ always off."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model import empire as E
from campaign_model import features as ft
from campaign_model import vocab
from campaign_model.config import CampaignConfig
from campaign_model.learner import CampaignLearner
from campaign_model.model import CampaignModel
from campaign_model.schema import empty_inputs
from campaign_model.tests.test_empire import CASTLE, DWELLING_1, HALBERDIER, PIKEMAN, VILLAGE_HALL, player
from game_values import REFERENCE_KNOWLEDGE, REFERENCE_POWER

DWELLING_1_UP = 37
BASE = E.EmpireConfig(speed_power=0.0, army_morale=0.0)  # (the movement and the make-up apart: the rule's own terms)
DIFF = vocab.creature_cost(HALBERDIER) - vocab.creature_cost(PIKEMAN)


def total(p, rule):
	return E.empire_value(p, replace(BASE, unit_own_form=rule))["total"]


def starting(army):
	p = player()
	p["heroes"][0]["army"] = [list(a) for a in army]
	p["heroes"][0].update(attack=0, defense=0, power=REFERENCE_POWER, knowledge=REFERENCE_KNOWLEDGE, spellbook=True)
	return p


def test_an_upgrade_gains_alpha_times_the_cost_difference_a_creature_and_nothing_without_the_rule():
	before = starting(((PIKEMAN, 30),))
	after = copy.deepcopy(before)
	after["heroes"][0]["army"] = [[HALBERDIER, 30]]
	after["resources"][6] -= 30 * DIFF
	assert DIFF > 0
	assert total(after, False) - total(before, False) == pytest.approx(0.0, abs=1e-6)
	assert total(after, True) - total(before, True) == pytest.approx(BASE.alpha * 30 * DIFF)


@pytest.mark.parametrize("rule", [False, True])
def test_a_purchase_stays_neutral_base_or_upgraded(rule):
	for creature in (PIKEMAN, HALBERDIER):
		before = starting(((creature, 30),))  # (armed: the hero's base stays whole, the purchase alone moves Φ)
		before["towns"][0]["available"][0] = [creature, 14]
		after = copy.deepcopy(before)
		after["towns"][0]["available"][0] = [creature, 9]
		after["heroes"][0]["army"] = [[creature, 35]]
		after["resources"][6] -= 5 * vocab.creature_cost(creature)
		assert total(after, rule) == pytest.approx(total(before, rule), abs=1e-6), (creature, rule)


def test_a_dwellings_upgrade_gains_it_on_every_creature_for_hire_it_converts():
	"""The user's "we improve existing un-bought creatures": 14 Pikemen for hire become Halberdiers."""
	before = player()
	before["towns"][0]["built"] = [VILLAGE_HALL, DWELLING_1]
	after = copy.deepcopy(before)
	after["towns"][0]["built"] = [VILLAGE_HALL, DWELLING_1, DWELLING_1_UP]
	after["towns"][0]["available"][0] = [HALBERDIER, 14]
	cost = E.resources_value(vocab.buildings()[(CASTLE, DWELLING_1_UP)]["cost"])
	after["resources"][6] -= cost
	assert E.building_value(CASTLE, DWELLING_1_UP, BASE) == pytest.approx(cost)  # (an upgrade's own worth: its cost)
	assert total(after, False) - total(before, False) == pytest.approx(0.0, abs=1e-6)
	assert total(after, True) - total(before, True) == pytest.approx(BASE.alpha * 14 * DIFF)


def test_the_models_phi_follows_the_ledger_with_the_rule_on_and_off():
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	x = empty_inputs(1)
	x["t_present"][0, 0], x["t_owner"][0, 0], x["t_faction"][0, 0] = True, c.OWNER_SELF, CASTLE
	x["t_built"][0, 0, [VILLAGE_HALL, DWELLING_1, DWELLING_1_UP]] = True
	x["t_available"][0, 0, 0], x["t_available_creature"][0, 0, 0] = 14, HALBERDIER
	x["h_present"][0, 0], x["h_owner"][0, 0], x["h_detailed"][0, 0] = True, c.OWNER_SELF, True
	x["h_primary"][0, 0] = x["h_primary_base"][0, 0] = [1, 1, 1, 2]
	x["h_skills"][0, 0, [0, 1]] = [1, 1]
	x["h_army_creature"][0, 0, 0], x["h_army_count"][0, 0, 0] = HALBERDIER, 20
	p = {"color": 0, "status": 0, "resources": [0] * 7, "income": [0] * 7, "explored": 0, "towns": [{"faction": CASTLE, "built": [VILLAGE_HALL, DWELLING_1, DWELLING_1_UP], "garrison": [], "available": [[HALBERDIER, 14]] + [[-1, 0]] * 6}], "heroes": [{"type": 0, "level": 1, "exp": 0, "mana": 0, "power": 1, "spell_cost": 0, "primary": [1, 1, 1, 2], "skills": [[0, 1], [1, 1]], "army": [[HALBERDIER, 20]], "artifacts": []}], "dwellings": [], "armies": []}
	t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in x.items()}
	try:
		for rule in (False, True):
			model.features.empire_unit_own.fill_(1.0 if rule else 0.0)
			expected = E.empire_value(p, replace(E.EmpireConfig(), unit_own_form=rule))
			with torch.no_grad():
				parts, _ = ft.empire_value(t, model.features)
			for name, value in zip(ft.PHI_PARTS, parts[0].tolist()):
				assert value == pytest.approx(expected[name], rel=1e-4, abs=1e-3), (name, rule)
		assert float(model.features.creature_unit[HALBERDIER]) > float(model.features.creature_unit[PIKEMAN])
	finally:
		model.features.empire_unit_own.fill_(0.0)
	assert float(model.features.creature_unit[HALBERDIER]) == pytest.approx(float(model.features.creature_unit[PIKEMAN]))  # (off: the base form's u)


def test_off_by_default_the_learner_fills_it_and_the_evaluation_never(monkeypatch):
	assert E.EmpireConfig().unit_own_form is False and E.EVAL_EMPIRE.unit_own_form is False
	monkeypatch.setenv("ALPHA_EMPIRE_UNIT_OWN", "1")
	assert E.EmpireConfig().unit_own_form is True and E.EVAL_EMPIRE.unit_own_form is False
	monkeypatch.delenv("ALPHA_EMPIRE_UNIT_OWN")
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	empire = E.EmpireConfig()
	learner = SimpleNamespace(config=CampaignConfig(), empire=empire, model=model, features_cpu=None)
	CampaignLearner._unit_rule(learner)
	assert learner.empire is empire and float(model.features.empire_unit_own) == 0.0
	learner.config = CampaignConfig(unit_own_form=True)
	CampaignLearner._unit_rule(learner)
	assert learner.empire.unit_own_form is True and float(model.features.empire_unit_own) == 1.0


def _x_one_hero(army_creature, count, gold):
	x = empty_inputs(1)
	x["g_resources"][0] = [10, 4, 10, 4, 4, 4, gold]
	x["t_present"][0, 0], x["t_owner"][0, 0], x["t_faction"][0, 0] = True, c.OWNER_SELF, CASTLE
	x["t_built"][0, 0, [VILLAGE_HALL, DWELLING_1]] = True
	x["h_present"][0, 0], x["h_owner"][0, 0], x["h_detailed"][0, 0] = True, c.OWNER_SELF, True
	x["h_primary"][0, 0] = x["h_primary_base"][0, 0] = [2, 1, 1, 1]
	x["h_skills"][0, 0, [0, 1]] = [1, 1]
	x["h_army_creature"][0, 0, 0], x["h_army_count"][0, 0, 0] = army_creature, count
	return x


def test_the_shown_terms_equal_the_replicas_change_the_rule_adds():
	"""UPGRADE's term (unit_rule_upgrades) and a dwelling upgrade's (unit_rule_build) against the replica's change of Φ with
	the rule on less with it off; both exactly 0 with it off."""
	from types import SimpleNamespace

	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	t = model.features
	tens = lambda x: {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in x.items()}  # noqa: E731

	def change(before, after):
		out = []
		for rule in (1.0, 0.0):
			t.empire_unit_own.fill_(rule)
			with torch.no_grad():
				out.append(float(ft.empire_value(after, t)[1][0] - ft.empire_value(before, t)[1][0]))
		return out[0] - out[1]

	try:
		before = tens(_x_one_hero(PIKEMAN, 20, 10000))
		after = tens(_x_one_hero(HALBERDIER, 20, 10000 - 20 * int(DIFF)))
		mult = ft.hero_multipliers(before["h_primary"][:, : c.OWN_HEROES], before["h_artifacts"][:, : c.OWN_HEROES], t)
		for rule in (0.0, 1.0):
			t.empire_unit_own.fill_(rule)
			with torch.no_grad():
				term = float(ft.unit_rule_upgrades(before, t, SimpleNamespace(hero_multiplier=mult))[0, 0, 0])
			assert term == (pytest.approx(change(before, after), rel=1e-5) if rule else 0.0)
		xb, xa = _x_one_hero(PIKEMAN, 20, 10000), _x_one_hero(PIKEMAN, 20, 10000)
		xb["t_available"][0, 0, 0], xb["t_available_creature"][0, 0, 0] = 14, PIKEMAN
		xa["t_built"][0, 0, DWELLING_1_UP] = True
		xa["t_available"][0, 0, 0], xa["t_available_creature"][0, 0, 0] = 14, HALBERDIER
		before, after = tens(xb), tens(xa)  # (the building's own worth and price apart: both arms of the rule alike)
		for rule in (0.0, 1.0):
			t.empire_unit_own.fill_(rule)
			with torch.no_grad():
				term = float(ft.unit_rule_build(before, t)[0, 0, DWELLING_1_UP])
			assert term == (pytest.approx(change(before, after), rel=1e-5) if rule else 0.0)
		assert change(before, after) == pytest.approx(float(t.empire_alpha) * 14 * DIFF, rel=1e-5)
	finally:
		t.empire_unit_own.fill_(0.0)
