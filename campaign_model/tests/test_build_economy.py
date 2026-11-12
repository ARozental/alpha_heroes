"""The economy and BUILD (e5 for the user, 2026-11-12): each _feature_ bounded and pointing the right way; the collectors'
shapes and bounds; BUILD's fixed prior ordering buildings as knowledge does; config.build_economy off = the model as
before whatever the features hold; a checkpoint from before loads with the new columns at 0 and plays as it did."""
import torch

from campaign_model import constants as c
from campaign_model import features as ft
from campaign_model.model_v2 import GROWN_INPUTS, CampaignModelV2
from campaign_model.tests.test_model_v2 import rich_inputs

TO, NB = c.OWN_TOWNS, c.N_BUILDINGS
NEW_COLUMNS = len(ft.ECONOMY) + len(ft.DANGER)  # (the grown layers' columns: the economy's and exposure's)
T = lambda *v: torch.tensor([float(a) for a in v])  # noqa: E731


def model():
	torch.manual_seed(3)
	return CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()


def inputs(b=6, seed=4):
	return {k: torch.from_numpy(v) for k, v in rich_inputs(b, seed).items()}


def test_each_economy_feature_is_bounded_and_points_the_right_way():
	assert torch.allclose(ft._feature_gold_over_pool(T(0, 9000), T(1000, 1000)), T(0, 9000 / 11000), atol=1e-6)
	assert float(ft._feature_pool_unbought(T(5000), T(1000))) == 0.0 and 0 < float(ft._feature_pool_unbought(T(0), T(4000))) < 1
	assert float(ft._feature_gold_idle(T(1000), T(2000), T(0))) == 0.0 and float(ft._feature_gold_idle(T(20000), T(0), T(0))) > 0.9
	assert float(ft._feature_flow_balance(T(0), T(5000))) == 0.0 and float(ft._feature_flow_balance(T(10000), T(0))) > 0.98
	w = ft._feature_production_week(T(0, 3000, 1e9), 3000.0)
	assert float(w[0]) == 0.0 and abs(float(w[1]) - 0.5) < 1e-6 and float(w[2]) < 1.0
	days = ft._feature_days_to_growth({"g_day": torch.tensor([1, 2, 7, 8])})
	assert torch.allclose(days, T(1.0, 6 / 7, 1 / 7, 1.0))


def test_each_entry_feature_is_bounded_and_points_the_right_way():
	assert float(ft._feature_income_add(T(0))) == 0.0 and abs(float(ft._feature_income_add(T(1000))) - 0.5) < 1e-6
	assert float(ft._feature_production_add(T(0))) == 0.0 and abs(float(ft._feature_production_add(T(2000))) - 0.5) < 1e-6
	payable = ft._feature_payable(T(0, 1000, 1000).reshape(1, 1, 3), T(5000), T(0), T(0, 0, 0).reshape(1, 1, 3), T(0), T(0))
	assert float(payable[0, 0, 0]) == 0.0 and float(payable[0, 0, 1]) == 1.0  # (no growth: 0; spare gold: all of it)
	assert float(ft._feature_payable(T(1000).reshape(1, 1, 1), T(0), T(0), T(5000).reshape(1, 1, 1), T(0), T(0))) == 0.0
	assert float(ft._feature_unlocks(T(0))) == 0.0 and float(ft._feature_unlocks(T(1e9))) <= 1.0
	assert float(ft._feature_cost_share(T(1e9).reshape(1, 1, 1), T(0), T(0))) == 1.0
	crowd = ft._feature_crowds_out(T(3000).reshape(1, 1, 1), T(3000), T(2000), T(0))
	assert 0 < float(crowd) <= 1.0  # (paying 3,000 of our 3,000 leaves the 2,000 for hire unpaid)


def test_the_collectors_shapes_and_bounds():
	m = model()
	m.features.build_economy.fill_(1.0)
	x = inputs()
	economy = ft.economy_features(x, m.features)
	entry, value = ft.build_features(x, m.features, economy)
	assert economy.shape == (6, len(ft.ECONOMY)) and entry.shape == (6, TO, NB, len(ft.BUILD_ENTRY)) and value.shape == (6, TO, NB, 3)
	for t in (economy, entry):
		assert torch.isfinite(t).all() and float(t.min()) >= 0.0 and float(t.max()) <= 1.0
	assert torch.isfinite(value).all() and float(value.min()) >= 0.0
	m.features.build_economy.fill_(0.0)
	assert float(ft.economy_features(x, m.features).abs().max()) == 0.0 and float(ft.build_features(x, m.features, economy)[0].abs().max()) == 0.0


def town(m, built=(), gold=50_000, faction=0, available=None):
	"""Decision 0 of rich_inputs made one town of ours (faction's), built as listed, gold as given, nothing for hire unless
	available ({level: count})."""
	x = inputs(2, 4)
	x["t_present"][0, :] = False
	x["t_present"][0, 0], x["t_owner"][0, 0], x["t_faction"][0, 0] = True, c.OWNER_SELF, faction
	x["t_built"][0, 0] = False
	for b in built:
		x["t_built"][0, 0, b] = True
	x["t_available"][0, 0] = 0
	x["t_available_creature"][0, 0] = -1
	for level, count in (available or {}).items():
		x["t_available"][0, 0, level] = count
		x["t_available_creature"][0, 0, level] = int(m.features.dwelling_creature[faction, 30 + level])
	x["t_growth"][0, 0] = 0
	x["g_resources"][0] = 0
	x["g_resources"][0, 6] = gold
	x["g_income"][0] = 0
	x["g_income"][0, 6] = 1000
	m.features.build_economy.fill_(1.0)
	economy = ft.economy_features(x, m.features)
	entry, value = ft.build_features(x, m.features, economy)
	f = type("F", (), {"build_value": value})()
	return m.features.build_prior(x, f)[0, 0], entry[0, 0]


def test_the_prior_orders_buildings_as_knowledge_does():
	m = model()
	gaps = {}
	for faction in range(9):  # (a town with a castle and gold to spare: the capitol against each level's base dwelling, none built)
		prior, _ = town(m, built=(7, 8, 9, 10, 11, 12, 5, 0, 14, 16), faction=faction, gold=200_000)
		gaps[faction] = [round(float(prior[13] - prior[30 + level]), 2) for level in range(7)]
	# (levels 1-6 at or under the capitol, a tie allowed: Necropolis L6 2 x 1,200 x 1.5 x 2 = 7,200 against 7,000; level 7
	# within 0.35: Castle's angels under a castle, 9,000 against the capitol's 7,000, a 20,000-gold dwelling that rarely
	# competes with a 10,000-gold capitol on the same day, e5)
	assert all(min(g[:6]) >= -0.05 and g[6] >= -0.35 for g in gaps.values()), f"capitol less each level's dwelling, by faction: {gaps}"
	base = (7, 10, 11, 5, 30, 31, 32, 34, 35)  # (a fort, halls to the town hall, a tavern, base dwellings 1-3, 5-6 built)
	broke, _ = town(m, built=base, gold=0, available={0: 200, 1: 100})
	assert float(broke[12]) > float(broke[33]), (float(broke[12]), float(broke[33]))  # (the pool unbought, no gold: a hall above a dwelling)
	prior2, entry2 = town(m, built=(7, 10, 30))  # (L1 built, no blacksmith: the blacksmith opens L4)
	assert float(entry2[16, ft.BUILD_ENTRY.index("unlocks")]) > 0 and float(prior2[16]) > 0
	prior3, _ = town(m, built=(7, 10, 30, 31, 32, 33))
	assert float(prior3[40]) < float(prior3[34]), (float(prior3[40]), float(prior3[34]))  # (L4's upgrade below L5's base dwelling)


def test_off_changes_nothing_whatever_the_features_hold():
	m = model()
	x = inputs()
	with torch.no_grad():
		base = m(x)
	real_e, real_b, real_d = ft.economy_features, ft.build_features, ft.danger_features
	try:
		ft.economy_features = lambda x_, t: torch.rand(x_["g_day"].shape[0], len(ft.ECONOMY)) * t.economy_on()
		ft.build_features = lambda x_, t, e: (torch.rand(x_["g_day"].shape[0], TO, NB, len(ft.BUILD_ENTRY)) * t.economy_on(), torch.rand(x_["g_day"].shape[0], TO, NB, 3) * 1e5)
		ft.danger_features = lambda x_, t, f: torch.rand(x_["g_day"].shape[0], len(ft.DANGER)) * t.economy_on()
		with torch.no_grad():
			out = m(x)
	finally:
		ft.economy_features, ft.build_features, ft.danger_features = real_e, real_b, real_d
	for name, value in base.items():
		if torch.is_tensor(value) and value.is_floating_point():
			assert torch.equal(torch.nan_to_num(value, nan=0.0, posinf=1e9, neginf=-1e9), torch.nan_to_num(out[name], nan=0.0, posinf=1e9, neginf=-1e9)), name


def test_a_checkpoint_from_before_loads_and_plays_as_it_did():
	m = model()
	x = inputs()
	with torch.no_grad():
		for name in GROWN_INPUTS[2:]:  # (a trained model's: its old columns as they are, the economy's at 0)
			p = dict(m.named_parameters())[name]
			p[:, -NEW_COLUMNS:] = 0.0
		base = m(x)
	old = {k: v for k, v in m.state_dict().items() if not k.startswith("build_situational") and k not in ("features.build_income_gain", "features.build_production_gain", "features.build_prior_gain", "features.build_faction_bias", "features.build_economy", "features.empire_alpha")}
	for name in GROWN_INPUTS[2:]:
		old[name] = old[name][:, : -NEW_COLUMNS].clone()
	fresh = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	fresh.load_state_dict(old)
	with torch.no_grad():
		out = fresh(x)
	assert float(fresh.features.empire_alpha) == 0.25  # (not ADDED's 0: deploy-check 2026-11-12, Φ off by 33k)
	for name, value in base.items():
		if torch.is_tensor(value) and value.is_floating_point():
			assert torch.allclose(torch.nan_to_num(value, nan=0.0, posinf=1e9, neginf=-1e9), torch.nan_to_num(out[name], nan=0.0, posinf=1e9, neginf=-1e9), atol=1e-6), name


def test_hall_income_counts_each_flag_once_and_the_silo_adds():
	m = model()
	x = inputs(2, 4)
	x["t_present"][0, :] = False
	x["t_present"][0, 0], x["t_owner"][0, 0], x["t_faction"][0, 0] = True, c.OWNER_SELF, 0
	x["t_built"][0, 0] = False
	for b in ft.HALLS:  # (a capitol town holds all four flags: 307 of 307 capitol towns in deploy 67's spool games)
		x["t_built"][0, 0, b] = True
	m.features.build_economy.fill_(1.0)
	own, faction = ft.own_towns(x), ft.vocab_index(x["t_faction"][:, : c.OWN_TOWNS], c.FACTION_VOCAB)
	assert float(ft._feature_hall_income_used(x, m.features, own, faction)[0]) == 1.0
	assert all(float(m.features.building_income[f, 15]) > 0 for f in range(9))  # (the silo's resources at their prices)


def test_alpha_rides_in_the_weights():
	from campaign_model.empire import EmpireConfig, building_value

	m = model()
	t = m.features
	assert float(t.empire_alpha) == 0.25
	cost = float(t.creature_unit_cost[12])
	assert float(t.creature_unit[12]) == 0.25 * cost
	for alpha in (0.25, 0.5):
		t.empire_alpha.fill_(alpha)
		assert float(t.creature_unit[12]) == alpha * cost
		assert abs(float(t.building_value[0, 31]) - building_value(0, 31, EmpireConfig(alpha=alpha))) < 1e-3
	fresh = model()
	state = m.state_dict()
	fresh.load_state_dict(state)
	assert float(fresh.features.empire_alpha) == 0.5  # (an inference server loading the learner's weights takes its ALPHA)


def test_a_castle_doubles_the_upgraded_and_horde_parts_too():
	m = model()
	x = inputs(2, 4)
	t = m.features
	faction = torch.zeros(2, TO, dtype=torch.long)
	plain = torch.zeros(2, TO, NB, dtype=torch.bool)
	plain[..., 30] = True
	castle = plain.clone()
	castle[..., (7, 8, 9)] = True
	a, b = ft._production_added(x, t, faction, plain), ft._production_added(x, t, faction, castle)
	assert float(a[0, 0, 37]) > 0 and abs(float(b[0, 0, 37]) - 2 * float(a[0, 0, 37])) < 1e-3
	horde = [k for k in range(NB) if int(t.horde_level[0, k]) == 0]
	if horde:
		assert abs(float(b[0, 0, horde[0]]) - 2 * float(a[0, 0, horde[0]])) < 1e-3 and float(a[0, 0, horde[0]]) > 0
