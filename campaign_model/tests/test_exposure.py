"""Exposure tonight (e5 for the user, 2026-11-12): each _feature_ bounded and pointing the right way; the collector's
shape and range; a hero of ours beside a stronger enemy reads high, the same hero inside a town of ours with a strong
garrison low; 0 with the economy off."""
import torch

from campaign_model import constants as c
from campaign_model import features as ft
from campaign_model.model_v2 import CampaignModelV2
from campaign_model.tests.test_model_v2 import rich_inputs

HO, TO = c.OWN_HEROES, c.OWN_TOWNS
T = lambda *v: torch.tensor([[float(a) for a in v]])  # noqa: E731


def model():
	torch.manual_seed(3)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	m.features.build_economy.fill_(1.0)
	return m


def test_each_feature_is_bounded_and_points_the_right_way():
	assert float(ft._feature_exposure_heroes(T(0, 0), T(1000, 5000))) == 0.0
	assert 0.5 < float(ft._feature_exposure_heroes(T(1, 1), T(1000, 5000))) < 1.0
	assert float(ft._feature_exposure_empire(T(1), T(1e9), torch.tensor([10.0]))) == 1.0
	ours = torch.tensor([[True, True]])
	assert abs(float(ft._feature_exposure_main(T(0.2, 0.9), T(1.0, 3.0), ours)) - 0.9) < 1e-6  # (the strongest hero's danger)
	assert float(ft._feature_army_outside(T(3000, 1000), T(1, 0))) == 0.75 and float(ft._feature_army_outside(T(0, 0), T(1, 1))) == 0.0
	assert abs(float(ft._feature_town_exposure(T(0.3, 0.8), torch.tensor([[True, False]]))) - 0.3) < 1e-6  # (only our towns)
	assert float(ft._feature_town_exposure_stake(T(1.0), T(1e9), torch.tensor([[True]]), torch.tensor([0.0]))) == 1.0


def scene(m, in_town):
	"""Decision 0 of rich_inputs made one hero of ours with a small army at (10, 10) (or in a town of ours there, a strong
	garrison) and one enemy hero with a big army a tile away; nothing else on the map."""
	x = {k: torch.from_numpy(v) for k, v in rich_inputs(2, 4).items()}
	x["h_present"][0] = False
	x["t_present"][0] = False
	creature = int(m.features.dwelling_creature[0, 30])  # (a level-1 creature)
	for slot, owner, pos, count in ((0, c.OWNER_SELF, (10, 10, 0), 10), (HO, c.OWNER_SELF + 1, (11, 10, 0), 2000)):
		x["h_present"][0, slot], x["h_owner"][0, slot] = True, owner
		x["h_pos"][0, slot] = torch.tensor(pos)
		x["h_army_creature"][0, slot] = -1
		x["h_army_count"][0, slot] = 0
		x["h_army_creature"][0, slot, 0], x["h_army_count"][0, slot, 0] = creature, count
	if "h_garrisoned" in x:
		x["h_garrisoned"][0, 0] = False
	if in_town:
		x["t_present"][0, 0], x["t_owner"][0, 0], x["t_faction"][0, 0] = True, c.OWNER_SELF, 0
		x["t_pos"][0, 0] = torch.tensor((10, 10, 0))
		x["t_garrison_creature"][0, 0] = -1
		x["t_garrison_count"][0, 0] = 0
		x["t_garrison_creature"][0, 0, 0], x["t_garrison_count"][0, 0, 0] = int(m.features.dwelling_creature[0, 36]), 200  # (level 7s)
		x["t_fort"][0, 0] = 3
	with torch.no_grad():
		return m.domain_knowledge(x).danger[0]


def test_beside_a_stronger_enemy_high_in_a_strong_town_low():
	m = model()
	out, home = scene(m, False), scene(m, True)
	main, outside = ft.DANGER.index("exposure_main"), ft.DANGER.index("army_outside")
	assert float(out[main]) > 0.8 and float(out[outside]) == 1.0, out
	assert float(home[main]) < 0.2 and float(home[outside]) == 0.0, home
	assert float(out[ft.DANGER.index("exposure_heroes")]) > 0.5 > float(home[ft.DANGER.index("exposure_heroes")])


def test_the_collectors_range_and_off():
	m = model()
	x = {k: torch.from_numpy(v) for k, v in rich_inputs(6, 4).items()}
	with torch.no_grad():
		d = m.domain_knowledge(x).danger
	assert d.shape == (6, len(ft.DANGER)) and torch.isfinite(d).all() and float(d.min()) >= 0.0 and float(d.max()) <= 1.0
	m.features.build_economy.fill_(0.0)
	with torch.no_grad():
		assert float(m.domain_knowledge(x).danger.abs().max()) == 0.0
