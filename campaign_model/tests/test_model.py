import math
import zlib

import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model import features as ft
from campaign_model import priors
from campaign_model.config import CampaignConfig
from campaign_model.empire import EmpireConfig, empire_value
from campaign_model.evaluate import paired
from campaign_model.labels import LabelConfig, horizon_advantages
from campaign_model.model import CampaignModel
from campaign_model.records import IN_LAYOUT, Decision, Game
from campaign_model.schema import empty_inputs, random_inputs
from campaign_model.train import make_optimizer

CASTLE, VILLAGE_HALL, DWELLING_1, PIKEMAN, HALBERDIER = 0, 10, 30, 0, 1
WINDMILL, GOLD_MINE, TAVERN, UTOPIA = 112, 53, 95, 25


def tensors(x):
	return {k: torch.from_numpy(v) for k, v in x.items()}


@pytest.fixture(scope="module")
def model():
	torch.manual_seed(0)
	return CampaignModel(dim=32, cell_dim=16, layers=1).eval()


def test_phi_in_the_graph_is_the_ledgers_value(model):
	"""The model's Φ from its inputs is empire.py's from the ledger, for the same empire."""
	x = empty_inputs(1)
	x["g_resources"][0] = [10, 4, 10, 4, 4, 4, 10000]
	x["g_object_income"][0] = [0, 0, 0, 0, 0, 0, 1000]
	x["g_explored"][0] = 100
	x["t_present"][0, 0], x["t_owner"][0, 0], x["t_faction"][0, 0] = True, c.OWNER_SELF, CASTLE
	x["t_built"][0, 0, [VILLAGE_HALL, DWELLING_1]] = True
	x["t_garrison_creature"][0, 0, 0], x["t_garrison_count"][0, 0, 0] = PIKEMAN, 5
	x["t_available"][0, 0, 0], x["t_available_creature"][0, 0, 0] = 14, PIKEMAN
	x["h_present"][0, 0], x["h_owner"][0, 0], x["h_detailed"][0, 0] = True, c.OWNER_SELF, True
	x["h_exp"][0, 0], x["h_mana"][0, 0], x["h_spell_cost"][0, 0] = 5000, 20, 10
	x["h_primary"][0, 0] = x["h_primary_base"][0, 0] = [3, 2, 2, 1]
	x["h_skills"][0, 0, [0, 1, 2]] = [2, 1, 1]
	x["h_army_creature"][0, 0, 0], x["h_army_count"][0, 0, 0] = PIKEMAN, 10
	x["h_artifacts"][0, 0, 0] = 7
	x["h_spells"][0, 0, [15, 21]] = True  # Magic Arrow, Fireball
	x["h_luck"][0, 0], x["h_battle_luck"][0, 0], x["h_morale"][0, 0], x["h_battle_morale"][0, 0] = 1, 1, 3, 2  # a Rally Flag's luck, +2 morale over +1
	# a scout: its skills count at SECONDARY_SHARE
	x["h_present"][0, 1], x["h_owner"][0, 1], x["h_detailed"][0, 1] = True, c.OWNER_SELF, True
	x["h_exp"][0, 1] = 1500
	x["h_primary"][0, 1] = x["h_primary_base"][0, 1] = [1, 1, 2, 2]
	x["h_skills"][0, 1, [3, 4, 5]] = [1, 1, 1]
	x["h_spells"][0, 1, 53] = True  # Haste
	x["h_morale"][0, 1], x["h_battle_morale"][0, 1] = -3, -3  # a Warrior's Tomb's, an army-less scout's: nothing
	x["g_owned_creature"][0, 0], x["g_owned_count"][0, 0] = PIKEMAN, 6  # left in a mine's garrison
	x["g_dwelling_creature"][0, 0], x["g_dwelling_count"][0, 0] = PIKEMAN, 9  # for hire at a dwelling we flagged
	player = {
		"color": 0, "status": 0, "resources": [10, 4, 10, 4, 4, 4, 10000], "income": [0, 0, 0, 0, 0, 0, 1000], "explored": 100,
		"towns": [{"faction": CASTLE, "built": [VILLAGE_HALL, DWELLING_1], "garrison": [[PIKEMAN, 5]], "available": [[PIKEMAN, 14]] + [[-1, 0]] * 6}],
		"heroes": [
			{
				"type": 0, "level": 5, "exp": 5000, "mana": 20, "power": 2, "spell_cost": 10, "primary": [3, 2, 2, 1], "skills": [[0, 2], [1, 1], [2, 1]], "army": [[PIKEMAN, 10]], "artifacts": [7],
				"spells": [15, 21], "luck": 1, "battle_luck": 1, "morale": 3, "battle_morale": 2,
			},
			{
				"type": 1, "level": 2, "exp": 1500, "mana": 0, "power": 2, "spell_cost": 0, "primary": [1, 1, 2, 2], "skills": [[3, 1], [4, 1], [5, 1]], "army": [], "artifacts": [],
				"spells": [53], "morale": -3, "battle_morale": -3,
			},
		],
		"dwellings": [{"available": [[PIKEMAN, 9]]}], "armies": [[PIKEMAN, 6]],
	}
	expected = empire_value(player)
	# Φ v3's parts: Magic Arrow and Fireball (levels 1 and 3) to the main hero, Haste to the scout at SECONDARY_SHARE, the
	# main hero's +1 luck and +2 morale (+1 -> +3) on its 10 pikemen; the scout's -3 morale lifts no army
	config, pikemen = EmpireConfig(), 10 * 75.0
	without = empire_value(dict(player, heroes=[dict(h, spells=[], battle_luck=0, battle_morale=0) for h in player["heroes"]]))
	assert expected["heroes"] - without["heroes"] == pytest.approx(config.spell_level * (4 + config.secondary_share) + config.boost * 3 * pikemen)
	with torch.no_grad():
		parts, total = ft.empire_value(tensors(x), model.features)
	assert float(total[0]) == pytest.approx(expected["total"], rel=1e-4)
	for name, value in zip(("resources", "buildings", "income", "army", "for_hire", "heroes", "artifacts", "mana", "hero_penalty", "explored"), parts[0].tolist()):
		assert value == pytest.approx(expected[name], rel=1e-4, abs=1e-3), name


def test_a_hire_in_the_graph_adds_what_empire_value_adds(model):
	"""features.tavern_worth (before the price and the crowding) is the change of empire.py's value when the hero for
	hire joins us, with its army, artifacts and skills (weighed against our strongest hero's)."""
	from campaign_model.empire import empire_value as ledger_value

	hero = {"type": 0, "level": 3, "exp": 2500, "mana": 0, "power": 1, "spell_cost": 0, "primary": [2, 2, 1, 1], "skills": [[0, 1], [1, 2]], "army": [[PIKEMAN, 10]], "artifacts": []}
	for tavern_skills, tavern_exp in (([[6, 1], [7, 1]], 0), ([[6, 3], [7, 3], [8, 2]], 9000)):  # weaker, then stronger than ours
		x = empty_inputs(1)
		x["h_present"][0, 0], x["h_owner"][0, 0], x["h_detailed"][0, 0] = True, c.OWNER_SELF, True
		x["h_exp"][0, 0] = 2500
		x["h_primary"][0, 0] = x["h_primary_base"][0, 0] = [2, 2, 1, 1]
		x["h_skills"][0, 0, [0, 1]] = [1, 2]
		x["h_army_creature"][0, 0, 0], x["h_army_count"][0, 0, 0] = PIKEMAN, 10
		x["g_tavern"][0, 0], x["g_tavern_exp"][0, 0] = 5, tavern_exp
		x["g_tavern_primary"][0, 0] = [1, 2, 1, 1]
		for skill, level in tavern_skills:
			x["g_tavern_skills"][0, 0, skill] = level
		x["g_tavern_army_creature"][0, 0, :2], x["g_tavern_army_count"][0, 0, :2] = [PIKEMAN, HALBERDIER], [12, 4]
		x["g_tavern_artifacts"][0, 0, 13] = 4  # a ballista
		x["g_tavern_spells"][0, 0, 15] = True  # Magic Arrow
		before = {"color": 0, "status": 0, "resources": [0] * 7, "income": [0] * 7, "explored": 0, "towns": [], "heroes": [hero], "dwellings": [], "armies": []}
		hired = {"type": 5, "level": 1, "exp": tavern_exp, "mana": 0, "power": 1, "spell_cost": 0, "primary": [1, 2, 1, 1], "skills": tavern_skills, "army": [[PIKEMAN, 12], [HALBERDIER, 4]], "artifacts": [4], "spells": [15], "morale": 1}  # (its morale: its army's make-up's, one faction)
		after = dict(before, heroes=[hero, hired])
		with torch.no_grad():
			worth = ft.tavern_worth(tensors(x), model.features)
		crowding = ledger_value(after)["hero_penalty"] - ledger_value(before)["hero_penalty"]
		assert float(worth[0, 0]) == pytest.approx(ledger_value(after)["total"] - ledger_value(before)["total"] - crowding, rel=1e-4)
		assert float(worth[0, 1]) == 0  # an empty slot


def window_with(obj, subtype=0, flags=0, owner=c.OWNER_NONE):
	"""One hero of ours, a revealed window, one object two tiles east of it."""
	x = empty_inputs(1)
	x["g_kind"][0], x["g_day"][0], x["g_map"][0] = c.DecisionKind.TURN, 1, [36, 36, 1]
	x["h_present"][0, 0], x["h_owner"][0, 0], x["h_detailed"][0, 0] = True, c.OWNER_SELF, True
	x["h_pos"][0, 0] = [10, 10, 0]
	x["h_primary"][0, 0] = x["h_primary_base"][0, 0] = [1, 1, 1, 2]  # a fresh hero's (empire.py: its skills worth 0)
	x["h_skills"][0, 0, [0, 1]] = [1, 1]
	x["h_move"][0, 0] = x["h_move_max"][0, 0] = 1500
	x["h_army_creature"][0, 0, 0], x["h_army_count"][0, 0, 0] = PIKEMAN, 10
	x["w_flags"][0, 0] = c.TILE_ON_MAP | c.TILE_REVEALED
	x["move_mask"][0, 0] = True
	x["move_mask"][0, 0, c.WINDOW_RADIUS, c.WINDOW_RADIUS] = False
	row, column = c.WINDOW_RADIUS, c.WINDOW_RADIUS + 2
	x["w_object"][0, 0, row, column], x["w_subtype"][0, 0, row, column], x["w_owner"][0, 0, row, column] = obj, subtype, owner
	x["w_flags"][0, 0, row, column] |= c.TILE_VISITABLE | flags
	return x, (row, column)


def custom_features(model, x):
	t = tensors(x)
	with torch.no_grad():
		windows, _ = model.windows(t)
		return model.features(t, windows)


def tile_worth(model, x):
	"""The gold of the object each window tile lets us take (0 where none may be taken)."""
	return custom_features(model, x).tile_object_gold[0]


def test_an_objects_worth_starts_at_nullkiller2s_and_counts_only_where_it_may_be_taken(model):
	types, subtypes = priors.object_values()
	assert float(types[WINDMILL]) == 2000 and float(types[TAVERN]) == 0  # (Φ's gold: 3-6 of a rare resource)
	logits = ft.value_logit(torch.tensor([100.0, 550.0, 1150.0, 4000.0, 10000.0, -4000.0]))
	assert 2 < float(logits[2]) < 3 and float(logits[4]) == pytest.approx(4.0, abs=0.05) and float(logits[5]) == pytest.approx(-float(logits[3]))
	assert float(ft.value_logit(torch.tensor(1.0))) == pytest.approx(-float(ft.value_logit(torch.tensor(-1.0)))) and abs(float(ft.value_logit(torch.tensor(1.0)))) < 0.01  # smooth through 0
	worth = model.features.object_gold(torch.tensor([WINDMILL, GOLD_MINE, GOLD_MINE, TAVERN]), torch.tensor([0, 6, 0, 0]))
	assert worth.tolist() == pytest.approx([2000.0, 5000.0, 2500.0, 0.0], abs=1e-3)
	x, tile = window_with(WINDMILL)
	worth = tile_worth(model, x)
	assert float(worth[tile]) == pytest.approx(2000.0, abs=1e-3) and float(worth.abs().sum()) == pytest.approx(float(worth[tile]))
	for flags, owner in ((c.TILE_VISITED_BY_PLAYER, c.OWNER_NONE), (c.TILE_VISITED_BY_HERO, c.OWNER_NONE), (0, c.OWNER_SELF)):
		x, tile = window_with(WINDMILL, flags=flags, owner=owner)
		assert float(tile_worth(model, x)[tile]) == 0  # visited (by us, by this hero) or ours: nothing to take
	x, tile = window_with(WINDMILL)
	x["w_flags"][0, 0, tile[0], tile[1]] &= ~c.TILE_VISITABLE  # a tile the object blocks, not its entrance
	assert float(tile_worth(model, x)[tile]) == 0


def test_the_optimizer_decays_only_the_prior_corrections():
	torch.manual_seed(1)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	config = CampaignConfig(learning_rate=1e-3, prior_decay=5.0)
	optimizer = make_optimizer(model, config)
	groups = {g["name"]: g for g in optimizer.param_groups}
	assert {name: g["weight_decay"] for name, g in groups.items()} == {name: 5.0 if name == "corrections" else 0.0 for name in groups}
	assert {id(p) for p in groups["corrections"]["params"]} == {id(p) for p in model.prior_corrections()}
	assert sum(len(g["params"]) for g in optimizer.param_groups) == len(list(model.parameters()))
	with torch.no_grad():
		model.features.object_type_delta.weight[WINDMILL] = 1.0
		model.features.object_gain.fill_(0.2)
		weights = [p.detach().clone() for g in optimizer.param_groups if g["name"] != "corrections" for p in g["params"]]
	for p in model.parameters():
		p.grad = torch.zeros_like(p)
	optimizer.step()  # no gradient: only the decay acts
	assert float(model.features.object_type_delta.weight[WINDMILL]) == pytest.approx(1.0 - 1e-3 * 5.0)
	assert float(model.features.object_gain) == pytest.approx(0.2 * (1 - 1e-3 * 5.0))
	assert all(torch.equal(a, b) for a, b in zip(weights, [p for g in optimizer.param_groups if g["name"] != "corrections" for p in g["params"]]))


def test_far_objects_reach_the_value_and_the_heroes():
	"""A Dragon Utopia far beyond every window reaches the value (through the global token's map summary): the value's
	gradient on the map's embedding of it is 0 without it and not with it; and the heroes' map read gets a gradient
	though it starts at 0. (At the start its effect is small, as every path's: 1e-6 of the value at one seed.)"""
	x = random_inputs(2, seed=4)
	without = {k: torch.from_numpy(v.copy()) for k, v in x.items()}  # (from_numpy shares the arrays changed below)
	x["m_object"][:, -1], x["m_owner"][:, -1], x["m_pos"][:, -1] = UTOPIA, c.OWNER_NONE, [35, 35, 1]
	grads = []
	for inputs in (without, tensors(x)):
		torch.manual_seed(0)
		trained = CampaignModel(dim=32, cell_dim=16, layers=1)
		with torch.no_grad():  # (the value heads start at 0, as the battle model's: trained a little, they read the trunk)
			trained.value_heads.weight.normal_()
		trained(inputs)["value"].sum().backward()
		grads.append(float(trained.map.object.weight.grad[UTOPIA].abs().sum()))
	assert grads[0] == 0 and grads[1] > 0
	assert float(trained.map_reader.out.weight.grad.abs().sum()) > 0


def decision(kind, today, values, color="RED", day=1):
	return Decision(color, day, kind, today, 1, 0, -1, -1, 0.0, values, 0, b"")


def test_the_policy_aims_at_the_longest_horizon():
	"""Trusted heads: a turn decision's step is V7 at the next turn decision that day (past a question), the day's last
	one bootstraps through V5 at the first turn decision two days on (2 + 5 = 7), and a quarter is the real outcome;
	untrusted: the real outcome and half the 2-day shortcut."""
	turn, select = int(c.DecisionKind.TURN), int(c.DecisionKind.SELECT)
	game = Game(decisions=[
		decision(turn, 0, (0.20, 0.25, 0.30)), decision(select, 0, (0.05, 0.08, 0.10)), decision(turn, 1, (0.30, 0.35, 0.40)),
		decision(turn, 2, (0.32, 0.40, 0.45)), decision(turn, 0, (0.40, 0.45, 0.50), day=2), decision(turn, 0, (0.50, 0.60, 0.55), day=3),
	])
	targets = np.array([[0.4, 0.5, 0.6]] * 6, dtype=np.float32)
	price = np.array([3e-4, 0, 2e-4, 1e-4, 1e-4, 1e-4], dtype=np.float32)
	config = LabelConfig(decision_price=1e-4)
	trusted = horizon_advantages(game, targets, price, 1.0, config, outcome_share_min=0.25, myopic_weight=0.5)
	mc = targets[:, 2] - np.array([0.30, 0.10, 0.40, 0.45, 0.50, 0.55]) - price
	td = np.array([0.40 - 0.30 - 1e-4, 0.40 - 0.10, 0.45 - 0.40 - 1e-4, 0.60 - 0.45 - 1e-4, 0.6 - 0.50 - 1e-4, 0.6 - 0.55 - 1e-4])
	assert trusted == pytest.approx(0.75 * td + 0.25 * mc, abs=1e-6)  # (days 2 and 3 have nothing two days on: the outcome)
	untrusted = horizon_advantages(game, targets, price, 0.0, config, outcome_share_min=0.25, myopic_weight=0.5)
	assert untrusted == pytest.approx(mc + 0.5 * (targets[:, 0] - np.array([0.20, 0.05, 0.30, 0.32, 0.40, 0.50])), abs=1e-6)
	# A failed credit check moves the credit to the real outcome, trusted heads or not, and never to the 2-day shortcut
	for trust in (0.0, 1.0):
		assert horizon_advantages(game, targets, price, trust, config, 0.25, 0.5, check=0.0) == pytest.approx(mc, abs=1e-6)
	_, error = horizon_advantages(game, targets, price, 1.0, config, parts=True)
	assert error[:4] == pytest.approx([0.40 - 0.6, 0.40 - 0.6, 0.45 - 0.6, 0.60 - 0.6], abs=1e-6) and np.isnan(error[4:]).all()


def test_paired_evals_compare_the_same_games():
	earlier = [{"seed": s, "map": "m", "color": "RED", "score": v} for s, v in ((1, 0.5), (2, 1.5), (3, 1.0))]
	now = [{"seed": s, "map": "m", "color": "RED", "score": v} for s, v in ((1, 0.6), (2, 1.6), (3, 1.2), (4, 9.0))]
	mean, se = paired(now, earlier)
	assert mean == pytest.approx((0.1 + 0.1 + 0.2) / 3) and se == pytest.approx(np.std([0.1, 0.1, 0.2], ddof=1) / np.sqrt(3))


def encoded(x, b):
	"""A decision's inputs as AlphaMapAI sends them (records.py)."""
	return zlib.compress(b"".join(np.ascontiguousarray(x[name][b]).astype(wire).tobytes() for name, _, wire, _, _ in IN_LAYOUT))


def test_an_update_in_its_own_process(tmp_path):
	from campaign_model.learner import UpdateProcess
	from campaign_model.tests.test_train import decisions_for

	torch.manual_seed(2)
	kwargs = {"dim": 32, "cell_dim": 16, "layers": 1}
	model = CampaignModel(**kwargs).eval()
	x = random_inputs(8, seed=6)
	decisions = [Decision(**{**vars(d), "inputs": encoded(x, b)}) for b, d in enumerate(decisions_for(model, x))]
	config = CampaignConfig(batch_size=4, passes=1, device="cpu", torch_threads=1)
	worker = UpdateProcess(config, kwargs, model, make_optimizer(model, config), 1.0, tmp_path / "checkpoint_latest.pt")
	try:
		state, metrics, lr_scale = worker.update(decisions, np.full((8, c.N_VALUE_OFFSETS), 0.5, np.float32), np.linspace(0.3, 0.7, 8).astype(np.float32), LabelConfig().weights, {"version": 2})
	finally:
		worker.close()
	assert metrics["steps"] == 2 and metrics["unsound"] == 0
	assert any(not torch.equal(state[k], v) for k, v in model.state_dict().items())
	saved = torch.load(tmp_path / "checkpoint_latest.pt")
	assert saved["version"] == 2 and "optimizer" in saved


def test_the_prior_corrections_are_learnable_within_a_run():
	"""Adam's first step moves each weight by lr: the windmill's worth moves CORRECTION_STEP times what plain weights
	would (its log-gain through the global and its type's gain, its gold through its type's and subtype's deltas), and
	however far the gain is pushed it stays within 5 times the prior."""
	from campaign_model.model import CORRECTION_STEP, MAX_LOG_GAIN

	torch.manual_seed(3)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	optimizer = make_optimizer(model, CampaignConfig(learning_rate=1e-4))
	obj, sub = torch.tensor([WINDMILL]), torch.tensor([0])
	before = float(model.features.object_gold(obj, sub))
	(-model.features.object_gold(obj, sub)).sum().backward()
	optimizer.step()
	after = float(model.features.object_gold(obj, sub))
	log_slope = 2 * MAX_LOG_GAIN * 0.25 * CORRECTION_STEP  # d log-gain / d weight at the start (features.bounded)
	gold_slope = 10000 * 0.25 * CORRECTION_STEP
	expected = before * (math.exp(2 * 1e-4 * log_slope) - 1) + 2 * 1e-4 * gold_slope
	assert after - before == pytest.approx(expected, rel=0.02)
	with torch.no_grad():
		model.features.object_gain.fill_(100.0)
		model.features.object_type_delta.weight.zero_()
		model.features.object_subtype_delta.weight.zero_()
	assert float(model.features.object_gold(obj, sub)) == pytest.approx(5 * before, rel=1e-3)


def test_phi_reaches_the_value_head_bounded_with_sane_gradients(model):
	"""Whatever the empire (a million gold, eight heroes with nothing else), Φ's features stay within -1..1 and the
	value's gradients stay as large as an ordinary empire's."""
	normal = random_inputs(4, seed=8)
	huge = {k: v.copy() for k, v in normal.items()}
	huge["g_resources"][:, 6] = 10**7
	huge["g_explored"][:] = 41472
	poor = empty_inputs(1)
	poor["h_present"][0, :8], poor["h_owner"][0, :8] = True, c.OWNER_SELF
	norms = []
	for x in (normal, huge, poor):
		t = tensors(x)
		with torch.no_grad():
			parts, total = ft.empire_value(t, model.features)
			features = ft.phi_features(parts, total)
		assert float(features.abs().max()) <= 1.0
		trained = CampaignModel(dim=32, cell_dim=16, layers=1)
		trained.load_state_dict(model.state_dict())
		trained(t)["value_logit"].sum().backward()
		norms.append(float(torch.sqrt(sum((p.grad ** 2).sum() for p in trained.parameters() if p.grad is not None))))
	assert all(math.isfinite(n) for n in norms) and max(norms) < 10 * min(norms)


def test_every_object_nullkiller2_values_has_a_class():
	import csv

	with (priors.PRIORS_DIR / "object_values.csv").open(newline="") as f:
		valued = {int(row["id"]) for row in csv.DictReader(f) if float(row["score"]) > 0}
	assert all(c.object_class(i) != c.ObjectClass.OTHER for i in valued)
	ids = [i for ids in c.OBJECT_CLASSES.values() for i in ids]
	assert len(ids) == len(set(ids))  # one class each
	assert c.object_class(53) == c.ObjectClass.MINE and c.object_class(98) == c.ObjectClass.TOWN and c.object_class(95) == c.ObjectClass.OTHER


def test_the_pair_bias_starts_at_zero_and_learns():
	from campaign_model.model import PairBias

	torch.manual_seed(4)
	bias = PairBias(heads=4)
	positions = torch.randint(0, 144, (2, 6, 3)).float()
	placed = torch.tensor([[True, True, False, True, True, False]] * 2)
	out = bias(positions, placed)
	assert out.shape == (2, 4, 6, 6) and float(out.abs().max()) == 0
	out.sum().backward()
	assert float(bias.net[-1].weight.grad.abs().sum()) > 0


def test_every_object_we_see_reaches_the_map_counted_by_class(model):
	"""Every object we see is listed (MAP_OBJECTS 8192: no engine ranking decides what the model sees, 2026-10-28) and
	torch counts them per block by class as the engine did (ours: the OUR_ form; visited: VISITED); one far down the
	list reaches the coarse blocks and TRAVEL's areas."""
	x = random_inputs(2, seed=9)
	with torch.no_grad():
		before = model.map(tensors(x), custom_features(model, x))
	slot = c.MAP_OBJECTS - 1  # the list's last slot
	x["m_object"][0, slot], x["m_subtype"][0, slot], x["m_owner"][0, slot], x["m_pos"][0, slot] = 16, 0, c.OWNER_NONE, [30, 30, 1]
	with torch.no_grad():
		after = model.map(tensors(x), custom_features(model, x))
	assert not torch.equal(before[0], after[0]) and not torch.equal(before[2], after[2])
	side = int(ft.block_side(tensors(x))[0])
	assert float(ft.object_counts(tensors(x))[0, 1, 30 // side, 30 // side, c.ObjectClass.BANK]) >= 1
	y = empty_inputs(1)
	y["m_object"][0, :3], y["m_owner"][0, :3] = [53, 79, 98], [c.OWNER_SELF, c.OWNER_NONE, c.OWNER_NONE]
	y["m_flags"][0, 1] = c.OBJECT_VISITED_BY_PLAYER
	assert ft.object_classes(tensors(y))[0, :4].tolist() == [c.ObjectClass.OUR_MINE, c.ObjectClass.VISITED, c.ObjectClass.TOWN, -1]


def test_a_bank_is_its_reward_plus_the_fight_with_its_guards(model):
	"""A Dragon Utopia (~81,000 gold of treasure behind ~87,000 of dragons, from VCMI's config) is a loss for a hero
	with ten pikemen: its treasure counts only as often as the fight is won (almost never), and the fight costs the
	army (10 x (60 + 15) gold) and the hero (a hire's 2,500); a mine next to the same hero is worth its gold."""
	x, tile = window_with(25)  # Dragon Utopia
	f = custom_features(model, x)
	from campaign_model.empire import EmpireConfig

	config = EmpireConfig()
	army = 10 * (60 + 15)
	# the army and the hero's replacement: its HERO_BASE at that army (empire.hero_base_value) and the rest of the price
	lost = army + config.hero_base * (config.armyless_share + (1 - config.armyless_share) * army / config.armed_army) + (2500 - config.hero_base)
	assert float(f.tile_object_gold[0][tile]) > 80000 and float(f.tile_fight_gold[0][tile]) == pytest.approx(-lost, abs=1.0)
	assert float(f.tile_value[0][tile]) < -3000 and float(f.tile_prior[0][tile]) < -3
	x, tile = window_with(GOLD_MINE, subtype=6)
	f = custom_features(model, x)
	assert float(f.tile_fight_gold[0][tile]) == 0 and float(f.tile_prior[0][tile]) == pytest.approx(float(ft.value_logit(torch.tensor(5000.0))), abs=1e-4)


def test_a_stronger_enemy_hero_makes_its_reach_cost_and_leaving_it_pay(model):
	"""With no enemy hero in sight the tiles are worth what they hold; with a far stronger one 6 tiles east, a tile
	toward it costs (its danger is higher than here) and a tile away from it pays (lower), both in proportion to what a
	lost battle takes."""
	x, _ = window_with(-1)
	calm = custom_features(model, x).tile_value[0]
	x["h_present"][0, c.OWN_HEROES], x["h_owner"][0, c.OWN_HEROES] = True, c.OWNER_SELF + 1
	x["h_pos"][0, c.OWN_HEROES] = [16, 10, 0]  # 6 tiles east of our hero
	x["h_army_creature"][0, c.OWN_HEROES, 0], x["h_army_count"][0, c.OWN_HEROES, 0], x["h_detailed"][0, c.OWN_HEROES] = 13, 500, True  # angels
	value = custom_features(model, x).tile_value[0]
	east, west = (c.WINDOW_RADIUS, c.WINDOW_RADIUS + 3), (c.WINDOW_RADIUS, c.WINDOW_RADIUS - 6)
	assert float(calm.abs().max()) < 1e-3
	assert float(value[east]) < -100 and float(value[west]) > 500  # (6 tiles away is already deep in its reach: closer adds little)


def test_each_value_head_is_its_own_readout_and_starts_at_its_prior():
	"""V2, V5, V7 and the growth heads each read the value trunk alone, none chained (a chained V7 starts as V2's
	answer to a shorter question): each head's loss reaches its own row of weights only. The shares start at 1 / the
	players in the game (2026-10-28 review: at 1/2, every decision of a 4-player game started a quarter below), the
	growth at none."""
	torch.manual_seed(6)
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	x = tensors(random_inputs(3, seed=8))
	with torch.no_grad():
		out = model(x)
	assert torch.equal(out["value"], torch.full_like(out["value"], 0.5)) and float(out["growth_logit"].abs().max()) == 0
	four = dict(x, g_players=torch.full_like(x["g_players"], 4))
	with torch.no_grad():
		assert torch.allclose(model(four)["value"], torch.full_like(out["value"], 0.25))  # a share of 1 / players
	with torch.no_grad():  # (trained a little: every head non-zero)
		model.value_heads.weight.normal_()
		model.growth_heads.weight.normal_()
	for name, key in (("value_heads", "value_logit"), ("growth_heads", "growth_logit")):
		for k in range(c.N_VALUE_OFFSETS):
			model.zero_grad()
			model(x)[key][:, k].sum().backward()
			rows = getattr(model, name).weight.grad.abs().sum(dim=1)
			assert [float(r) > 0 for r in rows] == [i == k for i in range(c.N_VALUE_OFFSETS)]


def test_dumb_decisions_are_counted():
	"""A turn ended with an unguarded pile in reach and no enemy hero in sight is a passed pickup; with an enemy hero in
	sight it is not. A dialog with a mine's garrison (the other army) that closes with more of our creatures there than
	when it opened left troops; one where a creature went in and came back out shuffled (2026-11-06: the moves into it
	were counted before, and every one of 17 in 39 games was taken back)."""
	from campaign_model.failures import failure_counts

	x, tile = window_with(79)  # a resource pile two tiles east, a legal move
	end = Decision("RED", 1, int(c.DecisionKind.TURN), 0, 1, int(c.ActionType.END), -1, -1, 0.0, (0.5,) * 3, 0, encoded(x, 0))
	x2 = {k: v.copy() for k, v in x.items()}
	x2["h_present"][0, c.OWN_HEROES], x2["h_owner"][0, c.OWN_HEROES] = True, c.OWNER_SELF + 1  # an enemy hero in sight
	risky = Decision("RED", 1, int(c.DecisionKind.TURN), 1, 1, int(c.ActionType.END), -1, -1, 0.0, (0.5,) * 3, 0, encoded(x2, 0))
	shape = (c.N_HOLDERS, c.ARMY_SLOTS, c.N_HOLDERS, c.N_ARMY_AMOUNTS)
	give, take = (int(np.ravel_multi_index(i, shape)) for i in ((0, 0, c.OTHER_ARMY, 0), (c.OTHER_ARMY, 0, 0, 0)))

	def exchange(action, index, garrison):  # an EXCHANGE decision whose garrison holds `garrison` creatures (slot 0)
		state = {k: v.copy() for k, v in x.items()}
		state["x_creature"][0, 0], state["x_count"][0, 0] = (5 if garrison else -1), garrison
		return Decision("RED", 2, int(c.DecisionKind.EXCHANGE), 0, 1, int(action), index, -1, 0.0, (0.5,) * 3, 0, encoded(state, 0))

	turn = Decision("RED", 2, int(c.DecisionKind.TURN), 0, 1, int(c.ActionType.END), -1, -1, 0.0, (0.5,) * 3, 0, encoded(x2, 0))
	shuffled = [exchange(c.ActionType.ARMY, give, 0), exchange(c.ActionType.ARMY, take, 5), exchange(c.ActionType.END, -1, 0)]
	left = [exchange(c.ActionType.ARMY, give, 0), exchange(c.ActionType.END, -1, 5)]
	counts = failure_counts([end, risky, *shuffled, turn, *left])
	assert counts["pickup_passed_end"] == (1, 3) and counts["pickup_passed_travel"] == (0, 0)
	assert counts["troops_left"] == (1, 2) and counts["exchange_shuffle"] == (1, 2)


def test_meetings_are_counted_by_moves_and_by_movement():
	"""Two heroes of ours (types 7 and 8) and a town. Day 1: hero 7 steps onto hero 8 twice (the second a repeat), then
	into the town, then ends the turn (the heroes' rows swapped: the hero found by its type). Day 2: onto hero 8 again (a
	new day: no repeat), hero 8 travels. A second game's day 1 (the day going back): onto hero 8, no repeat of the first
	game's. Movement spent: each hero's points less at the player's next decision that day."""
	from campaign_model.failures import failure_counts

	R, W, TG, L = c.WINDOW_RADIUS, c.WINDOW, c.TRAVEL_GRID, c.MAP_LEVELS

	def state(move7, move8, swapped=False):
		x = empty_inputs(1)
		x["g_kind"][0], x["g_map"][0] = c.DecisionKind.TURN, [36, 36, 1]
		rows = (1, 0) if swapped else (0, 1)
		for row, kind, pos, move in zip(rows, (7, 8), ([10, 10, 0], [12, 10, 0]), (move7, move8)):
			x["h_present"][0, row], x["h_owner"][0, row], x["h_type"][0, row] = True, c.OWNER_SELF, kind
			x["h_pos"][0, row], x["h_move"][0, row] = pos, move
		x["t_present"][0, 0], x["t_owner"][0, 0], x["t_pos"][0, 0] = True, c.OWNER_SELF, [10, 14, 0]
		return x

	def step(day, action, index, x):
		return Decision("RED", day, int(c.DecisionKind.TURN), 0, 1, int(action), index, -1, 0.0, (0.5,) * 3, 0, encoded(x, 0))

	onto_hero = (R + 0) * W + (R + 2)  # hero row 0's window: 2 east
	into_town = (R + 4) * W + R  # 4 south
	MOVE, TRAVEL, END = c.ActionType.MOVE, c.ActionType.TRAVEL, c.ActionType.END
	decisions = [
		step(1, MOVE, onto_hero, state(1000, 1500)),
		step(1, MOVE, onto_hero, state(800, 1500)),
		step(1, MOVE, into_town, state(600, 1500)),
		step(1, END, -1, state(300, 1500, swapped=True)),
		step(2, MOVE, onto_hero, state(1000, 1500)),
		step(2, TRAVEL, 1 * L * TG * TG, state(900, 1500)),
		step(2, END, -1, state(900, 1000)),
		step(1, MOVE, onto_hero, state(1000, 1500)),
		step(1, END, -1, state(950, 1500)),
	]
	counts = failure_counts(decisions)
	assert counts["own_hero_moves"] == (4, 5) and counts["repeat_meetings"] == (1, 4)
	assert counts["meeting_movement"] == (200 + 200 + 100 + 50, 1350) and counts["town_movement"] == (300, 1350)  # the travel: 500
	assert failure_counts(decisions, moves_limit=1)["own_hero_moves"] == (3, 4)  # whole games: the first only


def test_an_exchange_that_can_only_leave_troops_behind_closes(model):
	"""When every legal troop move puts our troops into the other army (a mine), ARMY is ~e^-4 as likely as before; a
	legal move out of the mine leaves its odds as they were."""
	x = random_inputs(1, seed=3)
	x["g_kind"][0] = c.DecisionKind.EXCHANGE
	x["army_mask"][:] = False
	x["army_mask"][0, 0, 0, c.OTHER_ARMY, 0] = True  # only: hero 0's first stack into the mine
	only_leave = model(tensors(x))["type_logp"][0]
	x["army_mask"][0, c.OTHER_ARMY, 0, 0, 0] = True  # also: the mine's troops to hero 0
	both = model(tensors(x))["type_logp"][0]
	p = lambda row: float(row[c.ActionType.ARMY].exp() / (row[c.ActionType.ARMY].exp() + row[c.ActionType.END].exp()))  # noqa: E731
	assert p(only_leave) < 0.05 and p(both) > 0.3


def test_the_credit_check_judges_each_action_type_against_the_others_by_game_player():
	"""The credit check (labels.credit_evidence, credit_gaps, credit_trust). A game-player's decisions share their days'
	luck: counted as decisions, honest heads looked several standard errors off in about half the updates (the check's
	first version failed on noise, and a failed check then raised the 2-day shortcut's weight). Judged by game-player,
	each type against the player's other decisions: honest heads pass update after update, an error common to all
	decisions passes (the advantages are centred: it costs the policy nothing), and heads that overprice what follows one
	type by 0.01 of a share (a quick gain that costs by day 7) fail."""
	from campaign_model.labels import credit_evidence, credit_gaps, credit_trust

	rng = np.random.default_rng(0)
	move, build = int(c.ActionType.MOVE), int(c.ActionType.BUILD)

	def update(bias=0.0, common=0.0):
		"""16 game-players' 7 days, each a BUILD and 4 MOVEs; the step's error: the player's and the day's luck (alike for
		all of a day's decisions), a little noise, an error common to all, and the bias after BUILD."""
		kinds, errors, players = [], [], []
		for player in range(16):
			luck = rng.normal(0, 0.05)
			for day in range(7):
				today = luck + rng.normal(0, 0.02)
				for kind in (build, move, move, move, move):
					kinds.append(kind)
					errors.append(today + rng.normal(0, 0.01) + common + (bias if kind == build else 0.0))
					players.append(player)
		decisions = [Decision("RED", 1, int(c.DecisionKind.TURN), 0, 1, k, 0, -1, 0.0, (0.5,) * 3, 0, b"") for k in kinds]
		return decisions, np.array(errors, dtype=np.float32), np.array(players)

	as_decisions = [e.mean() / (e.std(ddof=1) / np.sqrt(len(e))) for _, e, _ in (update() for _ in range(10))]
	assert sum(abs(t) > 3 for t in as_decisions) >= 3
	for common in (0.0, 0.05):
		evidence, worst = None, 0.0
		for _ in range(30):
			evidence = credit_evidence(*update(common=common), evidence)
			worst = max([worst] + [abs(t) for t in credit_gaps(evidence).values()])
		assert worst < 3 and credit_trust(credit_gaps(evidence)) == 1.0
	evidence = None
	for _ in range(3):
		evidence = credit_evidence(*update(bias=0.01), evidence)
	gaps = credit_gaps(evidence)
	assert gaps["BUILD"] > 6 and gaps["MOVE"] < -6 and credit_trust(gaps) == 0.0


def test_the_fight_prior_wins_a_two_to_one_fight_and_fears_an_even_one():
	"""P(win) at the start (sharpness 4): an even fight a coin toss, 2:1 won ~15 times in 16 (at 2, 1 in 5 was lost)."""
	features = CampaignModel(dim=32, cell_dim=16, layers=1).features
	sharpness = ft.bounded(features.fight_sharpness, 0.5, 8.0, 4.0, 10.0)
	win = ft.win_chance(torch.log(torch.tensor([1.0, 2.0, 0.5])), sharpness)
	assert win.tolist() == pytest.approx([0.5, 16 / 17, 1 / 17], abs=1e-4)


def test_a_town_is_priced_by_its_fort_and_defended_by_it():
	"""What taking a town we see adds to Φ at least: its village hall and its fort level's buildings at cost and the
	hall's income (5 days); its strength: the garrison's plus Nullkiller2's fort bonus (citadel 4,000, castle 10,000)
	while it is defended, none for an empty town (taken without a fight)."""
	features = CampaignModel(dim=32, cell_dim=16, layers=1).features
	x = empty_inputs(1)
	x["t_present"][0, :4], x["t_faction"][0, :4], x["t_fort"][0, :4] = True, 0, [0, 1, 2, 3]
	t = {k: torch.from_numpy(v) for k, v in x.items()}
	assert features.town_value(t)[0, :4].tolist() == pytest.approx([5000.0, 20000.0, 23750.0, 33750.0])
	garrison = torch.zeros(t["t_fort"].shape)
	garrison[0, :4] = 1000.0
	strength = features.town_strength(t, garrison)[0, :4]
	assert torch.allclose(strength, torch.log1p(torch.tensor([1000.0, 1000.0, 5000.0, 11000.0])))
	assert float(features.town_strength(t, torch.zeros_like(garrison)).abs().max()) == 0


def test_the_prior_tables_are_looked_up_by_object_and_subtype():
	"""The subtype rows (object_values.csv: a gold pile, a gold mine) and the banks' guards (bank_guards.csv) in one
	gather, as the CSVs say; another subtype gets its type's worth, nothing else gets guards."""
	features = CampaignModel(dim=32, cell_dim=16, layers=1).features
	types, subtypes = priors.object_values()
	obj, subtype = torch.tensor([79, 79, 53, 53, 101]), torch.tensor([6, 0, 6, 2, 0])
	extra = {(o, st): v for o, st, v in subtypes}
	want = [float(types[o]) + extra.get((o, st), 0.0) for o, st in zip(obj.tolist(), subtype.tolist())]
	score = features.object_score[obj] + features.subtype_score[obj, features.subtype_column(subtype)]
	assert score.tolist() == pytest.approx(want)
	(bank, kind), guard = next(iter(priors.bank_guards().items()))
	power, health = features.bank_guards(torch.tensor([bank, 79]), torch.tensor([kind, 6]))
	assert [float(power[0]), float(health[0])] == pytest.approx(list(guard)) and float(power[1]) == float(health[1]) == 0


def test_recruiting_starts_as_nullkiller2_buys_all_it_can_into_the_hero_best_first():
	"""RECRUIT's prior (Nullkiller2's BuyArmy and pickBestCreatures): with a hero visiting the town, buying all of the
	top level into the hero is the likeliest entry; each part of it is a learnable bounded parameter."""
	torch.manual_seed(0)
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	x = empty_inputs(1)
	x["g_kind"][0] = int(c.DecisionKind.TURN)
	x["t_present"][0, 0], x["t_owner"][0, 0] = True, c.OWNER_SELF
	x["recruit_mask"][0, 0, :2] = True  # levels 1 and 2, every amount, into the garrison or the hero
	with torch.no_grad():
		logp = model(tensors(x))["recruit_logp"][0, 0]
	best = np.unravel_index(int(logp.argmax()), tuple(logp.shape))
	assert best == (1, c.N_RECRUIT_AMOUNTS - 1, 1)  # level 2, all of it, into the hero
	p = logp.exp()
	assert float(p[:, :, 1].sum()) > 0.8 and float(p[:, -1, :].sum()) > 0.6


def test_army_prior_hands_a_towns_troops_to_its_hero_and_gathers_toward_the_stronger_hero():
	features = CampaignModel(dim=32, cell_dim=16, layers=1).features
	HO, TO = c.OWN_HEROES, c.OWN_TOWNS
	strength = torch.zeros(1, HO)
	strength[0, 1] = 2.0  # hero 1 is the stronger
	prior = features.army_prior(strength).expand(1, c.N_HOLDERS, c.ARMY_SLOTS, c.N_HOLDERS, c.N_ARMY_AMOUNTS)[0]
	ALL, ONE = int(c.ArmyAmount.ALL), int(c.ArmyAmount.ONE)
	assert float(prior[HO, 0, 0, ALL]) == pytest.approx(1.5)  # a whole stack from town 0's garrison into hero 0
	assert float(prior[HO, 0, 0, ONE]) == 0.0
	assert float(prior[0, 0, 1, ALL]) > 0.9 and float(prior[1, 0, 0, ALL]) < -0.9  # toward the stronger hero, not away
	assert float(prior[0, 0, c.OTHER_ARMY, ALL]) == pytest.approx(-4.0)  # troops left in a mine's garrison


def test_the_previous_decision_says_which_hero_and_town_it_was_about_and_starts_silent():
	"""features.last_actors reads our previous decision (the model's own choice, echoed): RECRUIT in town 2 marks town
	2; ARMY from hero 0 into town 1's garrison marks both; MOVE by hero 3 marks hero 3. Its weights start at 0: the
	outputs do not change until it is learned."""
	x = empty_inputs(3)
	W2 = c.WINDOW * c.WINDOW
	x["g_last_type"][:] = [int(c.ActionType.RECRUIT), int(c.ActionType.ARMY), int(c.ActionType.MOVE)]
	x["g_last_index"][0] = int(np.ravel_multi_index((2, 1, 3, 1), (c.OWN_TOWNS, c.N_LEVELS, c.N_RECRUIT_AMOUNTS, c.N_RECRUIT_TARGETS)))
	x["g_last_index"][1] = int(np.ravel_multi_index((0, 0, c.OWN_HEROES + 1, 0), (c.N_HOLDERS, c.ARMY_SLOTS, c.N_HOLDERS, c.N_ARMY_AMOUNTS)))
	x["g_last_index"][2] = 3 * W2 + 5
	heroes, towns = ft.last_actors(tensors(x))
	assert torch.nonzero(towns).tolist() == [[0, 2], [1, 1]] and torch.nonzero(heroes).tolist() == [[1, 0], [2, 3]]
	torch.manual_seed(0)
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	y = random_inputs(3, seed=2)
	with torch.no_grad():
		before = model(tensors(y))
		y.update({k: x[k] for k in ("g_last_type", "g_last_index")}, g_last_changed=np.ones_like(x["g_last_type"]))
		after = model(tensors(y))
	assert all(torch.equal(before[k], after[k]) for k in ("type_logp", "move_logp", "value"))


def test_travel_discounts_an_area_by_the_pathfinders_days():
	"""The same area, with a pile in it, one day away beats five days away (travel_days, the pathfinder's; Nullkiller2's
	discount by days)."""
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	x = random_inputs(1, seed=3)
	x["m_object"][0, 0], x["m_owner"][0, 0], x["m_pos"][0, 0] = 79, c.OWNER_NONE, [0, 0, 0]
	priors = []
	for days in (100, 500):
		x["travel_days"][0, 0, 0, 0, 0] = days
		priors.append(float(model.features(tensors(x), torch.zeros(0, dtype=torch.long)).travel_prior[0, 0, 0]))
	assert priors[0] > priors[1]

def test_the_meetings_counts_failing_leave_them_nan_and_the_rest():
	"""A diagnostic never ends an update: the meetings' counts failing (here a decision whose inputs cannot be decoded)
	leave their four shares nan, the other counts as they are."""
	from campaign_model.failures import failure_modes

	broken = Decision("RED", 1, int(c.DecisionKind.TURN), 0, 1, int(c.ActionType.MOVE), 5, -1, 0.0, (0.5,) * 3, 0, b"not zlib")
	shares = failure_modes([broken])
	assert all(math.isnan(shares[k]) for k in ("own_hero_moves", "repeat_meetings", "meeting_movement", "town_movement"))
