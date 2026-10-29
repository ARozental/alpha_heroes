"""Nullkiller2's brain as features (features.py): each op against Nullkiller2's own rules (VCMI's AI/Nullkiller2)."""
import math

import pytest
import torch

from campaign_model import constants as c
from campaign_model import features as ft
from campaign_model.model import CampaignModel
from campaign_model.schema import empty_inputs

SKILL = {name: i for i, name in enumerate(ft.SKILLS)}


@pytest.fixture(scope="module")
def tables():
	return CampaignModel(dim=32, cell_dim=16, layers=1).features


def tensors(x):
	return {k: torch.from_numpy(v) for k, v in x.items()}


def ours(x, heroes=2, towns=1):
	x["h_present"][0, :heroes], x["h_owner"][0, :heroes] = True, c.OWNER_SELF
	x["t_present"][0, :towns], x["t_owner"][0, :towns], x["t_faction"][0, :towns] = True, c.OWNER_SELF, 0  # Castle
	return x


def test_the_skill_ids_are_vcmis():
	from campaign_model import vocab

	assert [row["key"] for row in vocab.raw()["secondary_skills"]] == ["core:" + name for name in ft.SKILLS]


def test_main_heroes_are_the_strongest_one_per_town():
	"""HeroManager::update: by strength, max(towns, 1) main heroes (at most twice the towns); none without a town."""
	for towns, want in ((1, [False, True, False]), (2, [True, True, False]), (0, [False, False, False])):
		x = ours(empty_inputs(1), heroes=3, towns=towns)
		main = ft.hero_roles(tensors(x), torch.tensor([[1.0, 3.0, 0.5] + [0.0] * (c.N_HEROES - 3)]))
		assert main[0, :3].tolist() == want


def test_skill_scores_follow_nullkiller2s_evaluator(tables):
	"""A main hero with Offence: Logistics 2 + 1.5 (new, worth 2) = 3.5; Mysticism -1; Earth Magic 2 + 1.5 + 1 (its first
	school) = 4.5; Offence again 2 (not new); Wisdom at level 12 without it 1 + 1.5 + 1.5 (new: worth 1, one upgrade
	left: no) = 2.5. A scout: Estates 2 + 1.5 = 3.5, Offence 0."""
	x = ours(empty_inputs(1), heroes=2)
	x["h_skills"][0, 0, SKILL["offence"]] = 2  # one upgrade left
	x["h_exp"][0, 0] = 26000  # level 12 or so
	x["d_hero"][0] = 0
	names = ["logistics", "mysticism", "earthMagic", "offence", "wisdom"]
	x["o_kind"][0, : len(names)] = int(c.OptionKind.SKILL)
	x["o_type"][0, : len(names)] = [SKILL[n] for n in names]
	main = torch.zeros(1, c.OWN_HEROES, dtype=torch.bool)
	main[0, 0] = True
	scores = ft.skill_scores(tensors(x), main, tables)[0, : len(names)].tolist()
	level = math.floor(float(ft.fractional_level(torch.tensor(26000.0), tables.level_experience)))
	assert level > 10 and scores[:4] == pytest.approx([3.5, -1.0, 4.5, 2.0])
	assert scores[4] == pytest.approx(1 + 1.5 + 1.5)  # (one upgrade left: worth 1 counts as new)
	x["o_type"][0, :2] = [SKILL["estates"], SKILL["offence"]]
	scout = ft.skill_scores(tensors(x), torch.zeros_like(main), tables)[0, :2].tolist()
	assert scout == pytest.approx([3.5, 0.0])


def test_gold_pressure_and_the_city_hall_pause(tables):
	"""(army on offer + the next hall) / (1 + 2 gold + 7 income): 20 pikemen (60 each) and a Town Hall (2,500) against
	1,000 gold and 500 a day: 3,700 / 5,501; over 0.3 with no City Hall and nothing near: the pause."""
	x = ours(empty_inputs(1), heroes=1)
	x["t_available"][0, 0, 0], x["t_available_creature"][0, 0, 0] = 20, 0
	x["t_built"][0, 0, 10] = True
	x["g_resources"][0, 6], x["g_income"][0, 6] = 1000, 500
	t = tensors(x)
	pressure = ft.gold_pressure(t, tables)
	assert float(pressure[0]) == pytest.approx((20 * float(tables.creature_cost[0]) + 2500) / (1 + 2000 + 3500), rel=1e-4)
	assert float(ft.recruit_pause(t, pressure, torch.zeros(1, c.OWN_TOWNS))[0]) > 0.9
	x["t_built"][0, 0, ft.CITY_HALL] = True
	assert float(ft.recruit_pause(tensors(x), pressure, torch.zeros(1, c.OWN_TOWNS))[0]) == 0.0


def test_building_rewards_and_upgrade_gains(tables):
	"""BuildThis: a Town Hall's half week of the 500 it adds; Castle's first dwelling 1.5 x 14 pikemen's strength.
	makePossibleUpgrades: ten pikemen into halberdiers add ten times the difference in strength."""
	x = ours(empty_inputs(1), heroes=1)
	t = tensors(x)
	rewards = ft.build_rewards(t, tables)[0, 0]  # town 0, Castle (faction 0)
	assert float(rewards[11, 0]) == pytest.approx(math.log1p(3.5 * 500 / 1000), rel=1e-4)
	assert float(rewards[30, 1]) == pytest.approx(math.log1p(1.5 * 14 * float(tables.fight_value[0]) / 1000), rel=1e-4)
	x["h_army_creature"][0, 0, 0], x["h_army_count"][0, 0, 0] = 0, 10
	gain = ft.upgrade_gains(tensors(x), tables)[0, 0, 0]
	assert float(gain[0]) == pytest.approx(math.log1p(10 * float(tables.fight_value[1] - tables.fight_value[0]) / 1000), rel=1e-4)


def test_a_guard_is_credited_with_the_best_thing_it_blocks():
	"""blocked_by_guards: a monster whose zone is the only way into a pocket with a 5,000 artifact gets 5,000 on its
	tile; with the pocket reachable another way (not in the region), nothing."""
	W = 9
	value, region = torch.zeros(1, W, W), torch.zeros(1, W, W, dtype=torch.bool)
	monster, zone = torch.zeros(1, W, W), torch.zeros(1, W, W)
	monster[0, 4, 4] = 1
	zone[0, 3:6, 3:6] = 1
	region[0, 4, 6:] = True  # the pocket east of the zone
	value[0, 4, 8] = 5000.0
	assert float(ft.blocked_by_guards(value, region, zone, monster)[0, 4, 4]) == 5000.0
	assert float(ft.blocked_by_guards(value, torch.zeros_like(region), zone, monster).abs().max()) == 0.0


def test_the_pulls_are_weak_and_point_where_nullkiller2_would_go(tables):
	"""A main hero two tiles from a town of ours with an army to take: the town's MOVE prior rises by half a logit x
	tanh(gold / 5,000), no more; a threatened town pulls any hero; a scout with troops is pulled to the main hero. All
	three weights start at 0.5 and may fall to 0."""
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	x = ours(empty_inputs(1), heroes=1)
	x["g_kind"][0], x["g_map"][0] = int(c.DecisionKind.TURN), [36, 36, 1]
	x["h_pos"][0, 0], x["h_army_creature"][0, 0, 0], x["h_army_count"][0, 0, 0] = [10, 10, 0], 0, 10
	x["w_flags"][0, 0] = c.TILE_ON_MAP | c.TILE_REVEALED
	x["move_mask"][0, 0, c.WINDOW_RADIUS, c.WINDOW_RADIUS + 2] = True
	x["t_pos"][0, 0] = [12, 10, 0]  # two tiles east: window (R, R + 2)
	x["t_garrison_creature"][0, 0, 0], x["t_garrison_count"][0, 0, 0] = 0, 30
	windows = torch.tensor([0])
	f = model.features(tensors(x), windows)
	assert bool(f.main[0, 0]) and float(f.reinforcements[0, 0]) > 0
	gold = float(f.reinforcements[0, 0])
	tile = (c.WINDOW_RADIUS, c.WINDOW_RADIUS + 2)
	home = float(f.tile_prior[0][tile]) - float(ft.value_logit(f.tile_value[0][tile]))
	assert home == pytest.approx(0.5 * math.tanh(gold / ft.PULL_SCALE), abs=1e-4) and home <= 0.5
	weights = model.features.pull_weights()
	assert all(float(w) == pytest.approx(0.5) for w in weights)
	assert float(ft.bounded(torch.tensor(-10.0), 0.0, 2.0, 0.5, 10.0)) < 1e-6  # (a weight can fall to 0)


def test_hiring_trading_and_defenders_follow_nullkiller2(tables):
	"""hire_want: no hero yet: 1; recruit_prior: a threatened town buys into its garrison (the hero-target term gone at
	danger 1); trade_needs: a fort short of wood, with ore to spare, wants wood bought with ore."""
	x = ours(empty_inputs(1), heroes=0)
	t = tensors(x)
	want = ft.hire_want(t, torch.zeros(1), torch.zeros(1, c.OWN_TOWNS), torch.full((1, c.MAP_OBJECTS), -1))
	assert float(want[0]) == pytest.approx(1.0)
	danger = torch.zeros(1, c.OWN_TOWNS)
	danger[0, 0] = 1.0
	prior = tables.recruit_prior(danger)
	assert float(prior[0, 0, 0, 0, 1] - prior[0, 0, 0, 0, 0]) == pytest.approx(0.0) and float(prior[0, 1, 0, 0, 1] - prior[0, 1, 0, 0, 0]) == pytest.approx(2.0)
	x["build_short"][0, 0, 7] = True  # a fort: 20 wood, 20 ore, 5,000 gold
	x["g_resources"][0] = [5, 0, 60, 0, 0, 0, 9000]
	needs = ft.trade_needs(tensors(x), tables)[0]  # [R, 2]: lacking, spare
	wood, ore = 0, 2
	assert float(needs[wood, 0]) > 0 and float(needs[ore, 0]) == 0 and float(needs[ore, 1]) > 0

