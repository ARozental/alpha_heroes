"""B2, the no-town package (the user's design, 2026-11-13; e5's spec): a player's whole Φ x no_town_factor while it owns
no town, in the ledger's tables and the model's own phi alike, the evaluation on the old Φ; the prior's two terms from
that rule alone (taking a town while we have none pays (1 / f - 1) x our phi, losing our last takes (1 - f) x it more);
at f = 1 everything as before (schema 20's days input, read at 0: test_empty_visits)."""
import os
import subprocess
import sys
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model import features as ft
from campaign_model.empire import EVAL_EMPIRE, EmpireConfig, empire_value, shares
from campaign_model.model_v2 import CampaignModelV2
from campaign_model.schema import random_inputs

TO = c.OWN_TOWNS


def player(color, gold, towns=0, status=0):
	town = {"faction": 0, "built": [0, 10], "garrison": [], "available": []}
	return {"color": color, "status": status, "resources": [0] * 6 + [gold], "income": [0] * 7, "explored": 100, "towns": [dict(town) for _ in range(towns)], "heroes": [], "dwellings": [], "armies": []}


def test_the_ledger_rule_scales_every_part_of_a_townless_player_and_only_it():
	off, on = EmpireConfig(no_town_factor=1.0), EmpireConfig(no_town_factor=0.1)
	none, one = player(0, 5000), player(1, 5000, towns=1)
	before = empire_value(none, off)
	after = empire_value(none, on)
	assert set(after) == set(before) and all(after[k] == pytest.approx(0.1 * before[k]) for k in before)
	assert empire_value(one, on) == empire_value(one, off)  # (a town: as before)
	assert empire_value(player(2, 5000, status=1), on) == {"total": 0.0}  # (out of the game: 0 as before)
	assert shares({"players": [none, one]}, on)[0] < shares({"players": [none, one]}, off)[0]


def test_off_is_the_default_and_the_evaluation_keeps_the_old_phi_whatever_the_run_sets():
	assert EmpireConfig().no_town_factor == 1.0 and EVAL_EMPIRE.no_town_factor == 1.0  # (the test's environment sets none)
	code = "from campaign_model.empire import EmpireConfig, EVAL_EMPIRE; print(EmpireConfig().no_town_factor, EVAL_EMPIRE.no_town_factor)"
	env = {**os.environ, "ALPHA_EMPIRE_NO_TOWN": "0.1"}
	out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=os.getcwd(), check=True).stdout.split()
	assert out == ["0.1", "1.0"]


def inputs(n=3, seed=5):
	"""Row 0 owns no town, row 1 exactly one (slot 0), row 2 two (slots 0 and 1)."""
	x = random_inputs(n, seed=seed)
	for b, towns in enumerate((0, 1, 2)[:n]):
		x["t_present"][b, :TO] = False
		for s in range(towns):
			x["t_present"][b, s], x["t_owner"][b, s] = True, c.OWNER_SELF
	return x


def model():
	torch.manual_seed(3)
	return CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()


def known_at(m, x, factor):
	m.features.empire_no_town.fill_(factor)
	with torch.no_grad():
		return m.domain_knowledge({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in x.items()})


def test_the_model_phi_follows_the_ledger_rule():
	m, x = model(), inputs()
	off, on = known_at(m, x, 1.0), known_at(m, x, 0.1)
	assert float(on.phi_total[0]) == pytest.approx(0.1 * float(off.phi_total[0]), rel=1e-5)
	assert torch.equal(on.phi_total[1:], off.phi_total[1:])
	assert torch.allclose(on.phi_parts[0], 0.1 * off.phi_parts[0], rtol=1e-5)


def test_the_prior_terms_come_from_the_rule_and_are_nothing_at_one():
	m, x = model(), inputs()
	off, on = known_at(m, x, 1.0), known_at(m, x, 0.1)
	assert torch.equal(off.town_take_gain, torch.zeros_like(off.town_take_gain))
	base = ft.town_stakes({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in x.items()}, m.features, off.town_gold)
	assert torch.equal(off.town_stakes, base)  # (f = 1: the stakes as before, bit for bit)
	# no town: taking one pays 9 x our phi now (the rule's x0.1 already in it)
	assert float(on.town_take_gain[0]) == pytest.approx(9.0 * float(on.phi_total[0].clamp(min=0)), rel=1e-5)
	assert float(on.town_take_gain[1]) == 0.0 and float(on.town_take_gain[2]) == 0.0
	# the last town: its stake gains 0.9 x our phi; two towns: as before
	assert float(on.town_stakes[1, 0]) == pytest.approx(float(base[1, 0]) + 0.9 * float(on.phi_total[1].clamp(min=0)), rel=1e-5)
	assert torch.equal(on.town_stakes[1, 1:], base[1, 1:]) and torch.equal(on.town_stakes[2], base[2])


def test_taking_a_town_while_townless_raises_its_move_and_travel_prior():
	m, x = model(), inputs()
	off, on = known_at(m, x, 1.0), known_at(m, x, 0.1)
	# TRAVEL: an area holding a town not ours gains 9 x phi x its towns, discounted as the rest of its value
	assert (on.travel_gold[0] >= off.travel_gold[0] - 1e-3).all()
	contents, _ = ft.area_contents({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in x.items()}, off.object_counts)
	towns = contents[0, :, c.ObjectClass.TOWN]
	assert (towns > 0).any()  # (the inputs' map holds a town not ours: the check is not vacuous)
	assert torch.allclose(on.travel_gold[0] - off.travel_gold[0], towns * on.town_take_gain[0], rtol=1e-4, atol=1e-2)
	assert torch.equal(on.travel_gold[1:], off.travel_gold[1:])


def test_a_checkpoint_from_before_loads_with_its_process_factor_and_the_days_read_at_zero():
	m = model()
	old = {k: v for k, v in m.state_dict().items() if k != "features.empire_no_town" and not k.startswith("encode_global.no_town_days")}
	fresh = model()
	fresh.features.empire_no_town.fill_(0.1)  # (the loading process's factor, as ALPHA's)
	fresh.load_state_dict(old)
	assert float(fresh.features.empire_no_town) == pytest.approx(0.1)
	assert float(fresh.encode_global.no_town_days.weight.abs().sum()) == 0.0
	state = m.state_dict()
	m.features.empire_no_town.fill_(0.1)
	carried = model()
	carried.load_state_dict(m.state_dict())
	assert float(carried.features.empire_no_town) == pytest.approx(0.1)  # (the learner's, carried to an inference server)
	assert "features.empire_no_town" in state
