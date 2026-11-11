"""Schema 18 (2026-11-10): where today's walk toward each TRAVEL area stops (travel_stop) and the night there in TRAVEL's
logit prior (model.CustomFeatures.travel_prior, features.night_danger)."""
import copy
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model import features as ft
from campaign_model.model import CampaignModel
from campaign_model.records import _STARTS, IN_BYTES, SHORT_RECORDS, decode_inputs
from campaign_model.tests.test_model import custom_features, encoded, window_with

HO, TG = c.OWN_HEROES, c.TRAVEL_GRID


@pytest.fixture(scope="module")
def model():
	torch.manual_seed(0)
	return CampaignModel(dim=32, cell_dim=16, layers=1).eval()


def stop_code(pos):
	return (pos[2] * ft.STOP_RADIX + pos[1]) * ft.STOP_RADIX + pos[0] + 1


def area(pos):
	"""TRAVEL's flat area index of a tile (a 36 x 36 map: 2-tile areas)."""
	return (pos[2] * TG + pos[1] // 2) * TG + pos[0] // 2


def scene(hero_at, stops, town=True, enemy=True):
	"""Our hero (ten pikemen) at hero_at, a stronger enemy hero (500 angels) at [16, 10, 0], our town at [7, 10, 0]
	holding 1,000 angels behind a castle; TRAVEL legal to the areas of `stops` (1.5 days away), today's walk toward each
	stopping at that tile."""
	x, _ = window_with(-1)
	x["h_pos"][0, 0] = hero_at
	if enemy:
		x["h_present"][0, HO], x["h_owner"][0, HO] = True, c.OWNER_SELF + 1
		x["h_pos"][0, HO] = [16, 10, 0]
		x["h_army_creature"][0, HO, 0], x["h_army_count"][0, HO, 0], x["h_detailed"][0, HO] = 13, 500, True
	if town:
		x["t_present"][0, 0], x["t_owner"][0, 0], x["t_pos"][0, 0], x["t_fort"][0, 0] = True, c.OWNER_SELF, [7, 10, 0], 3
		x["t_garrison_creature"][0, 0, 0], x["t_garrison_count"][0, 0, 0], x["t_detailed"][0, 0] = 13, 1000, True
	days, mask, stop = x["travel_days"][0, 0].reshape(-1), x["travel_mask"][0, 0].reshape(-1), x["travel_stop"][0, 0].reshape(-1)
	for target, at in stops:
		mask[area(target)], days[area(target)], stop[area(target)] = True, 150, stop_code(at)
	return x


def night(f, target):
	"""What the night adds to an area's logit prior (the logits less the network's prior)."""
	return float(f.travel_logits[0, 0, area(target)] - f.travel_prior[0, 0, area(target)])


def test_a_record_from_before_reads_with_no_stop_known(model):
	"""A record of schema 17 ends where travel_stop starts: decoded, no stop is known, and TRAVEL's logits are its prior
	as before."""
	assert _STARTS["travel_stop"] in SHORT_RECORDS and _STARTS["travel_stop"] < IN_BYTES
	x = scene([10, 10, 0], [([30, 10, 0], [20, 10, 0])])
	full = zlib.decompress(encoded(x, 0))
	now = decode_inputs([SimpleNamespace(inputs=zlib.compress(full))])
	before = decode_inputs([SimpleNamespace(inputs=zlib.compress(full[: _STARTS["travel_stop"]]))])
	assert now["travel_stop"].reshape(-1)[area([30, 10, 0])] == stop_code([20, 10, 0]) and not before["travel_stop"].any()
	_, known = ft.travel_stops({k: torch.from_numpy(v) for k, v in before.items()})
	assert not known.any()
	f = custom_features(model, before)
	assert torch.equal(f.travel_logits, f.travel_prior)


def test_a_travel_that_stops_by_a_stronger_enemy_costs_and_one_away_pays(model):
	"""Our hero 6 tiles from the enemy (in its reach): a TRAVEL whose walk stops beside the enemy costs a little more
	danger than here; one that stops 25 tiles away pays what leaving its reach saves. The network's prior is the same as
	with no stop known (it reads travel_prior), and the stops decode back to their tiles."""
	toward, away = ([30, 10, 0], [15, 10, 0]), ([0, 34, 0], [0, 35, 0])
	x = scene([10, 10, 0], [toward, away], town=False)
	f = custom_features(model, x)
	stops, known = ft.travel_stops({k: torch.from_numpy(v) for k, v in x.items()})
	assert stops[0, 0, area(toward[0])].tolist() == toward[1] and bool(known[0, 0, area(away[0])]) and int(known.sum()) == 2
	assert night(f, toward[0]) < 0 and night(f, away[0]) > 2.0, (night(f, toward[0]), night(f, away[0]))
	blind = {k: v.copy() for k, v in x.items()}
	blind["travel_stop"][:] = 0
	g = custom_features(model, blind)
	assert torch.equal(g.travel_prior, f.travel_prior) and torch.equal(g.travel_logits, g.travel_prior)


def test_a_hero_in_a_town_of_ours_pays_to_leave_it_for_the_night_in_the_open(model):
	"""The hero stands in our town (1,000 angels and a castle: the enemy's 500 cannot take it): a TRAVEL whose walk stops
	in the open within the enemy's reach costs about what a lost battle takes; with no enemy in sight, nothing. The
	night term's weight (night_gain) scales it, and only the logits."""
	out = ([30, 10, 0], [12, 10, 0])
	f = custom_features(model, scene([7, 10, 0], [out]))
	calm = custom_features(model, scene([7, 10, 0], [out], enemy=False))
	assert night(f, out[0]) < -2.0 and night(calm, out[0]) == pytest.approx(0.0, abs=1e-6)
	weighted = copy.deepcopy(model)
	with torch.no_grad():
		weighted.features.night_gain.fill_(0.1)
	g = custom_features(weighted, scene([7, 10, 0], [out]))
	assert torch.equal(g.travel_prior, f.travel_prior) and night(g, out[0]) < night(f, out[0])


def test_the_night_in_another_town_of_ours_is_its_shelter(model):
	"""A walk that stops in a town of ours sleeps behind its walls: the same stop with the town there costs less than in
	the open (features.night_danger: the town's defence with the hero)."""
	x = scene([10, 10, 0], [([30, 10, 0], [12, 10, 0])], town=False)
	open_ground = night(custom_features(model, x), [30, 10, 0])
	x["t_present"][0, 1], x["t_owner"][0, 1], x["t_pos"][0, 1], x["t_fort"][0, 1] = True, c.OWNER_SELF, [12, 10, 0], 3
	x["t_garrison_creature"][0, 1, 0], x["t_garrison_count"][0, 1, 0], x["t_detailed"][0, 1] = 13, 1000, True
	assert night(custom_features(model, x), [30, 10, 0]) > open_ground + 1.0
