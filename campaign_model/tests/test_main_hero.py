"""M (the user's main-hero design, 2026-11-16; e5's spec 18:52; ALPHA_MAIN_HERO=1): a hero's own score without its army
(the type's tier + a learned correction, the town bonus where native, the level value / 4,000) ranks the mains; HIRE gets
relu(the tavern hero's score - our best's) + its level value / 4,000; the main hero's chest is experience whatever the gold
pressure. Off (unset): as before."""
import pytest
import torch

from campaign_model import constants as c
from campaign_model import features as ft

HO, TO = c.OWN_HEROES, c.OWN_TOWNS
ORRIN, VALESKA, SOLMYR = 0, 1, 45  # (Castle, Castle, Tower)
LEVEL = {1: 0, 2: 1000, 5: 4600, 10: 14700, 11: 17500}  # (VCMI's experience table)


def model(monkeypatch, on):
	from campaign_model.model_v2 import CampaignModelV2

	if on:
		monkeypatch.setenv("ALPHA_MAIN_HERO", "1")
	else:
		monkeypatch.delenv("ALPHA_MAIN_HERO", raising=False)
	torch.manual_seed(0)
	return CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()


def hero(x, h, kind, level, army=0):
	"""Our hero h: a fresh hero of `kind` at `level` (one skill level and one primary point a level past the first), with
	`army` angels (creature 13) or none."""
	x["h_present"][:, h], x["h_owner"][:, h], x["h_type"][:, h] = True, c.OWNER_SELF, kind
	x["h_exp"][:, h] = LEVEL[level]
	x["h_primary_base"][:, h] = torch.tensor([2, 2, 1, 1]) + torch.tensor([level - 1, 0, 0, 0])
	x["h_skills"][:, h] = 0
	x["h_skills"][:, h, 22] = min(1 + (level - 1), 3)  # (offence)
	x["h_skills"][:, h, 23] = max(min(level - 3, 3), 0) + 1  # (armorer)
	x["h_skills"][:, h, 2] = max(min(level - 6, 3), 0)  # (logistics)
	x["h_skills"][:, h, 7] = max(min(level - 9, 3), 0)  # (wisdom)
	x["h_army_creature"][:, h] = -1
	x["h_army_count"][:, h] = 0
	if army:
		x["h_army_creature"][:, h, 0], x["h_army_count"][:, h, 0] = 13, army


def tavern(x, s, kind, level):
	x["g_tavern"][:, s], x["g_tavern_level"][:, s], x["g_tavern_exp"][:, s] = kind, level, LEVEL[level]
	x["g_tavern_primary"][:, s] = torch.tensor([2, 2, 1, 1]) + torch.tensor([level - 1, 0, 0, 0])
	x["g_tavern_skills"][:, s] = 0
	x["g_tavern_skills"][:, s, 22] = min(level, 3)
	x["g_tavern_skills"][:, s, 23] = max(min(level - 3, 3), 0) + 1
	x["g_tavern_skills"][:, s, 2] = max(min(level - 6, 3), 0)
	x["g_tavern_skills"][:, s, 7] = max(min(level - 9, 3), 0)
	x["g_tavern_army_creature"][:, s] = -1
	x["g_tavern_army_count"][:, s] = 0


def inputs(n=2, seed=21):
	"""Day 1, one Castle town of ours, no heroes yet, the tavern empty, HIRE legal in town 0 for both slots."""
	from campaign_model.tests.test_model_v2 import rich_inputs, tensors

	x = tensors(rich_inputs(n, seed))
	x["g_kind"] = torch.zeros_like(x["g_kind"])
	x["g_day"][:] = 1
	x["h_present"][:] = False
	x["t_present"][:] = False
	x["t_present"][:, 0], x["t_owner"][:, 0], x["t_faction"][:, 0] = True, c.OWNER_SELF, 0
	x["g_tavern"][:] = -1
	mask = torch.zeros_like(x["hire_mask"]).reshape(n, TO, c.N_TAVERN)
	mask[:, 0, :2] = True
	x["hire_mask"] = mask.reshape(x["hire_mask"].shape)
	return x


def known(m, x):
	with torch.no_grad():
		return m.domain_knowledge(dict(x))


def test_the_tiers_and_tables():
	tiers, factions = ft.hero_tier_table(), ft.hero_faction_table()
	assert tiers[SOLMYR] == 8.0 and tiers[85] == 2.0 and tiers[78] == 2.0 and tiers[ORRIN] == 0.0 and int((tiers == 2.0).sum()) == 9
	assert factions[ORRIN] == 0 and factions[SOLMYR] == 2 and factions[-1] == c.FACTION_VOCAB


def test_solmyr_on_day_1_and_a_native_hero(monkeypatch):
	m = model(monkeypatch, True)
	x = inputs()
	hero(x, 0, ORRIN, 1)
	tavern(x, 0, SOLMYR, 1)
	tavern(x, 1, VALESKA, 1)
	f = known(m, x)
	assert torch.allclose(f.main_score[:, 0], torch.tensor(2.0), atol=0.05)  # (tier 0, native +2, a fresh hero's level value ~0)
	assert torch.allclose(f.hire_main[:, 0], torch.tensor(6.0), atol=0.1)  # (Solmyr 8, not native: relu(8 - 2))
	assert float(f.hire_main[:, 1].abs().max()) < 0.1  # (a native tier-0 hero: no better than ours)


def test_a_level_10_for_hire_with_our_level_11(monkeypatch):
	m = model(monkeypatch, True)
	x = inputs()
	hero(x, 0, ORRIN, 11)
	tavern(x, 0, VALESKA, 10)
	f = known(m, x)
	level10 = float(ft.level_value(x["g_tavern_primary"][:1, 0], x["g_tavern_skills"][:1, 0], x["g_tavern_exp"][:1, 0], m.features)) / ft.MAIN_LEVEL_GOLD
	assert level10 > 4.0  # (a level 10 hero's level value: ~4 logits and more)
	assert float(f.hire_main[0, 0]) == pytest.approx(level10, abs=1e-3)  # (not better than our level 11: the level term alone)


def test_the_main_by_its_own_score_not_its_army(monkeypatch):
	x = inputs()
	hero(x, 0, ORRIN, 5)  # (level 5, no army)
	hero(x, 1, VALESKA, 1, army=50)  # (level 1, 50 angels)
	off, on = known(model(monkeypatch, False), x), known(model(monkeypatch, True), x)
	assert bool(off.main[:, 1].all()) and not bool(off.main[:, 0].any())  # (as before: the army's)
	assert bool(on.main[:, 0].all()) and not bool(on.main[:, 1].any())
	x2 = inputs()
	hero(x2, 0, SOLMYR, 1)
	hero(x2, 1, ORRIN, 2, army=50)
	assert bool(known(model(monkeypatch, True), x2).main[:, 0].all())  # (Solmyr 8 against Orrin 2 + a level's value)


def test_the_chest(monkeypatch):
	from campaign_model.model import Features

	x = inputs()
	x["option_mask"][:] = False
	x["option_mask"][:, :2] = True
	x["o_kind"][:, :2] = int(c.OptionKind.COMPONENT)
	x["o_type"][:, 0], x["o_type"][:, 1] = int(c.ComponentType.RESOURCE), int(c.ComponentType.EXPERIENCE)
	x["d_hero"][:] = 0
	main = torch.zeros(x["g_day"].shape[0], HO, dtype=torch.bool)
	main[0, 0] = True  # (row 0: the main hero; row 1: another hero)
	f = Features(main=main, gold_pressure=torch.full((x["g_day"].shape[0],), 10.0))  # (far over Nullkiller2's MAX_GOLD_PRESSURE)
	off = model(monkeypatch, False).features.nk2_select(x, f)
	on = model(monkeypatch, True).features.nk2_select(x, f)
	assert float(off[0, 0]) > float(off[0, 1]) and float(off[1, 0]) > float(off[1, 1])  # (before: gold for everyone under pressure)
	assert float(on[0, 1]) > float(on[0, 0]) and float(on[1, 0]) > float(on[1, 1])  # (M: the main hero's experience; another's gold)


def test_the_new_keys_load_at_their_starts(monkeypatch):
	m = model(monkeypatch, True)
	old = {k: v for k, v in m.state_dict().items() if k not in ("features.hero_table", "features.town_bonus")}
	loading = model(monkeypatch, True)
	with torch.no_grad():
		loading.features.hero_table.fill_(3.0)
		loading.features.town_bonus.fill_(3.0)
	loading.load_state_dict(old)
	assert float(loading.features.hero_table_values().abs().max()) == 0.0 and float(loading.features.town_bonus_value()) == pytest.approx(2.0)
	assert any(p is loading.features.hero_table for p in loading.prior_corrections()) and any(p is loading.features.town_bonus for p in loading.prior_corrections())


def test_the_bonus_reaches_hires_logits(monkeypatch):
	m = model(monkeypatch, True)
	x = inputs()
	hero(x, 0, ORRIN, 1)
	tavern(x, 0, SOLMYR, 1)
	tavern(x, 1, VALESKA, 1)
	with torch.no_grad():
		a = m(dict(x))["hire_logp"].reshape(-1, TO, c.N_TAVERN)
		m.features.hero_table[SOLMYR] = 1000.0  # (Solmyr's correction at its top, +8: his bonus +8)
		b = m(dict(x))["hire_logp"].reshape(-1, TO, c.N_TAVERN)
	shift = (b[:, 0, 0] - b[:, 0, 1]) - (a[:, 0, 0] - a[:, 0, 1])
	assert torch.allclose(shift, torch.tensor(8.0), atol=1e-3)


def test_off_has_no_new_terms(monkeypatch):
	m = model(monkeypatch, False)
	x = inputs()
	hero(x, 0, ORRIN, 1)
	tavern(x, 0, SOLMYR, 1)
	f = known(m, x)
	assert "hire_main" not in f and "main_score" not in f
	with torch.no_grad():
		a = m(dict(x))
		m.features.hero_table.fill_(5.0)
		m.features.town_bonus.fill_(5.0)
		b = m(dict(x))
	assert all(torch.equal(a[k], b[k]) or (torch.isnan(a[k]).any() and torch.equal(torch.nan_to_num(a[k]), torch.nan_to_num(b[k]))) for k in a if torch.is_tensor(a[k]))


@pytest.mark.parametrize("bonus", [0.0, 5.0])
def test_the_actors_onnx_agrees_with_the_trainer(monkeypatch, tmp_path, bonus):
	import numpy as np
	import onnxruntime as ort

	from campaign_model.export import export

	m = model(monkeypatch, True)
	with torch.no_grad():
		for p in m.parameters():
			if float(p.abs().sum()) == 0:
				p.normal_(0, 0.05)
		m.features.hero_table[SOLMYR] = bonus
	path = export(m, tmp_path / "v2.onnx")
	x = inputs(2, 7)
	hero(x, 0, ORRIN, 1)
	hero(x, 1, VALESKA, 5, army=20)
	tavern(x, 0, SOLMYR, 1)
	tavern(x, 1, VALESKA, 10)
	with torch.no_grad():
		ours = {k: v.numpy() for k, v in m(dict(x)).items() if torch.is_tensor(v)}
	session = ort.InferenceSession(str(path))
	feed = {i.name: x[i.name].numpy() for i in session.get_inputs() if i.name in x}
	names = [o.name for o in session.get_outputs()]
	theirs = dict(zip(names, session.run(None, feed)))
	for name in ("hire_logp", "type_logp", "option_logp"):
		finite = np.isfinite(ours[name])
		assert np.array_equal(finite, np.isfinite(theirs[name])) and np.allclose(ours[name][finite], theirs[name][finite], atol=1e-4), name


def test_the_networks_inputs_keep_the_army_main(monkeypatch):
	"""e5's form (2026-11-16 19:31): the priors read the new main; the network's inputs (the hero's main flag, MOVE's and the
	tile's values and pulls, the views) keep the strongest army's hero, and the new main is a new column at 0 weights: with
	M on, where the two mains differ, the value heads and the foresight are unchanged."""
	x = inputs()
	hero(x, 0, ORRIN, 5)
	hero(x, 1, VALESKA, 1, army=50)
	off, on = model(monkeypatch, False), model(monkeypatch, True)
	with torch.no_grad():  # (the new column at 0, as a checkpoint from before loads it: test_the_new_column_starts_at_0)
		for m in (off, on):
			for layer in (m.join_hero.mlp.up, m.again_hero.mlp.up):
				layer.weight[:, -1] = 0.0
	assert not torch.equal(known(off, x).main, known(on, x).main) and torch.equal(known(off, x).main, known(on, x).army_main)
	with torch.no_grad():
		a, b = off(dict(x)), on(dict(x))
	for name in ("value", "value_logit", "value_hidden", "growth_logit", "end_logit", "fs_turn", "fs_hero", "fs_move", "fs_tiles", "fs_outcome", "memory"):
		if name in a:
			assert torch.equal(torch.nan_to_num(a[name]), torch.nan_to_num(b[name])), name


def test_the_new_column_starts_at_0(monkeypatch):
	"""The hero tokenizers' new last column (known.main) loads at 0 from a checkpoint without it (GROWN_INPUTS)."""
	from campaign_model.model_v2 import GROWN_INPUTS

	m = model(monkeypatch, True)
	state = m.state_dict()
	for key in ("join_hero.mlp.up.weight", "again_hero.mlp.up.weight"):
		assert key in GROWN_INPUTS
		state[key] = state[key][:, :-1].clone()
	loading = model(monkeypatch, True)
	loading.load_state_dict(state)
	for key in ("join_hero.mlp.up.weight", "again_hero.mlp.up.weight"):
		w = dict(loading.named_parameters())[key]
		assert float(w[:, -1].abs().max()) == 0.0 and torch.equal(w[:, :-1], m.state_dict()[key][:, :-1])
