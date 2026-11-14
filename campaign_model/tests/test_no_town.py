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
from campaign_model.model import CORRECTION_STEP
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
	# the last town's stake term: off by default (config.no_town_last_town), the stakes as before; on, 0.9 x our phi more
	assert torch.equal(on.town_stakes, base)
	m.features.empire_last_town.fill_(1.0)
	last = known_at(m, x, 0.1)
	assert float(last.town_stakes[1, 0]) == pytest.approx(float(base[1, 0]) + 0.9 * float(last.phi_total[1].clamp(min=0)), rel=1e-5)
	assert torch.equal(last.town_stakes[1, 1:], base[1, 1:]) and torch.equal(last.town_stakes[2], base[2])


def chances_by_hand(m, x, known):
	"""town_take_chance one hero, town and area at a time: P(win) at MOVE's sharpness against the town's strength or a
	rival hero's standing in it, whichever is stronger; the best over the towns not ours in each area."""
	t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in x.items()}
	sharpness = ft.bounded(m.features.fight_sharpness, 0.5, 8.0, 4.0, CORRECTION_STEP).detach()
	areas = c.MAP_LEVELS * c.TRAVEL_GRID * c.TRAVEL_GRID
	index = ft.area_index(t, t["t_pos"])
	out = torch.zeros(x["t_pos"].shape[0], c.OWN_HEROES, areas)
	for b in range(out.shape[0]):
		for h in range(c.OWN_HEROES):
			if not x["h_present"][b, h]:
				continue
			for town in range(x["t_pos"].shape[1]):
				if not x["t_present"][b, town] or x["t_owner"][b, town] == c.OWNER_SELF or index[b, town] < 0:
					continue
				defence = float(known.town_strength[b, town])
				for other in range(x["h_pos"].shape[1]):
					if x["h_present"][b, other] and x["h_owner"][b, other] != c.OWNER_SELF and (x["h_pos"][b, other] == x["t_pos"][b, town]).all():
						defence = max(defence, float(known.hero_strength[b, other]))
				win = float(ft.win_chance(known.hero_strength[b, h] - defence, sharpness))
				out[b, h, index[b, town]] = max(float(out[b, h, index[b, town]]), win)
	return out


def test_taking_a_town_while_townless_raises_its_move_and_travel_prior():
	m, x = model(), inputs()
	rival = int(np.flatnonzero(x["t_present"][0] & (x["t_owner"][0] != c.OWNER_SELF))[0])
	visitor = c.OWN_HEROES  # (a rival hero standing in that town: its strength counts if stronger, as MOVE's fight reads it)
	x["h_present"][0, visitor], x["h_owner"][0, visitor], x["h_pos"][0, visitor] = True, c.OWNER_SELF + 1, x["t_pos"][0, rival]
	x["h_army_creature"][0, visitor], x["h_army_count"][0, visitor], x["h_detailed"][0, visitor] = x["t_garrison_creature"][0, rival], x["t_garrison_count"][0, rival] * 2000, True
	off, on = known_at(m, x, 1.0), known_at(m, x, 0.1)
	assert float(on.hero_strength[0, visitor]) > float(on.town_strength[0, rival])  # (the visitor is the defence that counts)
	# f = 1: every hero's area value the same, the class value alone (as before the per-hero term: bit for bit, the tree's check)
	assert torch.equal(off.travel_gold, off.travel_gold[:, :1].expand_as(off.travel_gold))
	# TRAVEL: each hero's area gains 9 x phi x its best chance of taking a town not ours there, discounted as the rest
	chance = chances_by_hand(m, x, on)
	assert (chance[0] > 0).any() and (chance[0] < 1).any()  # (not vacuous: towns not ours in reach, chances short of sure)
	assert torch.allclose(on.travel_gold[0] - off.travel_gold[0], on.town_take_gain[0] * chance[0], rtol=1e-4, atol=1e-2)
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


def test_the_factor_ramps_by_version_alone():
	from campaign_model.config import CampaignConfig
	from campaign_model.learner import no_town_now

	off = CampaignConfig()
	assert all(no_town_now(off, v) == 1.0 for v in (0, 3400, 10**6))
	on = CampaignConfig(no_town_factor=0.1, no_town_from=3400)
	assert no_town_now(on, 3399) == 1.0 and no_town_now(on, 3400) == pytest.approx(0.9)
	got = [no_town_now(on, 3400 + k) for k in range(30)]
	assert all(a >= b for a, b in zip(got, got[1:])) and got[-1] == 0.1 and got[20] > 0.1  # (about 22 updates down to 0.1)


def test_the_learner_ramps_its_labels_and_the_model_together():
	from campaign_model.config import CampaignConfig
	from campaign_model.learner import CampaignLearner, no_town_now

	m = model()
	config = CampaignConfig(no_town_factor=0.1, no_town_from=3400)
	learner = SimpleNamespace(config=config, version=3405, empire=EmpireConfig(), model=m, features_cpu=model().features)
	CampaignLearner._ramp_no_town(learner)
	f = no_town_now(config, 3405)
	assert learner.empire.no_town_factor == pytest.approx(f) and float(m.features.empire_no_town) == pytest.approx(f)
	assert float(learner.features_cpu.empire_no_town) == pytest.approx(f)
	untouched = SimpleNamespace(config=CampaignConfig(), version=3405, empire=EmpireConfig(), model=model(), features_cpu=None)
	CampaignLearner._ramp_no_town(untouched)
	assert untouched.empire == EmpireConfig()  # (the target at 1: nothing, bit for bit)


def test_the_last_town_switch_is_off_by_default_and_filled_from_the_config():
	from campaign_model.config import CampaignConfig
	from campaign_model.learner import set_advice_prior

	m = model()
	assert float(m.features.empire_last_town) == 0.0
	set_advice_prior(m, CampaignConfig(no_town_last_town=True))
	assert float(m.features.empire_last_town) == 1.0
	set_advice_prior(m, CampaignConfig())
	assert float(m.features.empire_last_town) == 0.0
