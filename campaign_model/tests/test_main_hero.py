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
NEW_KEYS = ("buy_book_reset", "recruit_target_reset", "features.hero_table", "features.town_bonus", "features.chest_main_xp", "features.book_main", "features.book_other")
LEVEL = {1: 0, 2: 1000, 3: 2000, 4: 3200, 5: 4600, 10: 14700, 11: 17500}  # (VCMI's experience table)


def model(monkeypatch, on):
	from campaign_model.model_v2 import CampaignModelV2

	monkeypatch.setenv("ALPHA_MAIN_HERO", "1" if on else "0")  # (deploy 86: on by default; 0 turns it off)
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
	"""Nullkiller2's rule as before (gold for a hero not the main, or under gold pressure) + M's chest_main_xp (4.5 at the
	start) on the experience for the main by own score: under pressure experience leads by 4.5 - select_weight (+1.5 at
	the live 2.95), without by select_weight + 4.5; a hero not the main: gold as today; off: as before."""
	from campaign_model.model import Features

	x = inputs()
	x["option_mask"][:] = False
	x["option_mask"][:, :2] = True
	x["o_kind"][:, :2] = int(c.OptionKind.COMPONENT)
	x["o_type"][:, 0], x["o_type"][:, 1] = int(c.ComponentType.RESOURCE), int(c.ComponentType.EXPERIENCE)
	x["d_hero"][:] = 0
	main = torch.zeros(x["g_day"].shape[0], HO, dtype=torch.bool)
	main[0, 0] = True  # (row 0: the main hero; row 1: another hero)
	on_m, off_m = model(monkeypatch, True), model(monkeypatch, False)
	with torch.no_grad():
		for m in (on_m, off_m):
			m.features.select_pick.fill_(-0.0024)  # (the live select_weight, 2.95)
	sw = float(on_m.features.select_weight())
	assert sw == pytest.approx(2.95, abs=0.01) and float(on_m.features.chest_main_xp_value()) == pytest.approx(4.5)
	for pressure in (10.0, -10.0):
		f = Features(main=main, gold_pressure=torch.full((x["g_day"].shape[0],), pressure))
		on, off = on_m.features.nk2_select(x, f), off_m.features.nk2_select(x, f)
		lead = float(on[0, 1] - on[0, 0])
		assert lead == pytest.approx(4.5 - sw if pressure > 0 else sw + 4.5, abs=1e-3)
		assert torch.equal(on[1], off[1]) and float(on[1, 0]) > float(on[1, 1])  # (another hero: gold, as today)
	assert float(off[0, 1] - off[0, 0]) == pytest.approx(sw, abs=1e-3)  # (off, no pressure: experience by select_weight, as before)


def test_the_new_keys_load_at_their_starts(monkeypatch):
	m = model(monkeypatch, True)
	old = {k: v for k, v in m.state_dict().items() if k not in NEW_KEYS}
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


def levelup(x, skills):
	"""A level-up of our hero 0 offering `skills` (names)."""
	x["g_kind"][:] = int(c.DecisionKind.LEVEL_UP)
	x["d_hero"][:] = 0
	x["option_mask"][:] = False
	x["o_kind"][:] = 0
	x["o_type"][:] = -1
	for j, name in enumerate(skills):
		x["option_mask"][:, j] = True
		x["o_kind"][:, j] = int(c.OptionKind.SKILL)
		x["o_type"][:, j] = ft.SKILLS.index(name)
	return x


def test_the_level_ups_prior_is_nk2s_with_the_main_by_own_score(monkeypatch):
	"""Deploy 86 (the user 2026-11-16: "we can keep the NK2 numbers for now"): the level-up's prior is Nullkiller2's tables
	and extras at the learned skill_weight; only which hero takes the MAIN table is M's (by own score). Solmyr level 1 (the
	main by own score) beside Orrin level 2 with an army (the main by strength): Solmyr's level-up reads the MAIN table."""
	x = inputs()
	hero(x, 0, SOLMYR, 1)
	hero(x, 1, ORRIN, 2, army=30)
	x = levelup(x, ["estates", "logistics", "diplomacy"])
	on_m, off_m = model(monkeypatch, True), model(monkeypatch, False)
	on, off = known(on_m, x), known(off_m, x)
	assert torch.equal(on.skill_scores, ft.skill_scores(x, on.main, on_m.features)) and bool(on.main[0, 0]) and not bool(off.main[0, 0])
	main_est, scout_est = float(ft.MAIN_SKILL_SCORE[ft.SKILLS.index("estates")]), float(ft.SCOUT_SKILL_SCORE[ft.SKILLS.index("estates")])
	assert float(on.skill_scores[0, 0]) < float(off.skill_scores[0, 0]) and main_est < scout_est  # (estates: the MAIN table's -1 (+1.5 new) on, the scout's 2 off)
	assert torch.equal(on.skill_scores_input, off.skill_scores)  # (the network's input: by the army's main, as before)


BOOK = c.BUY_ARTIFACTS.index(ft.SPELLBOOK)
NEW_KEYS = ("buy_book_reset", "recruit_target_reset", "features.hero_table", "features.town_bonus", "features.chest_main_xp", "features.book_main", "features.book_other")


def trained_state(monkeypatch):
	"""A model's state as a checkpoint from before deploy 86: no marker, the book row away from 0."""
	m = model(monkeypatch, False)
	with torch.no_grad():
		m.buy_item.weight[BOOK] = 0.3
	return {k: v for k, v in m.state_dict().items() if k not in NEW_KEYS}


def test_the_book_row_resets_once(monkeypatch):
	old = trained_state(monkeypatch)
	off = model(monkeypatch, False)
	off.load_state_dict(old)
	assert float(off.buy_item.weight[BOOK].abs().max()) > 0 and float(off.buy_book_reset) == 0.0 and not off.book_row_reset_now  # (M off: no reset)
	m = model(monkeypatch, True)
	m.load_state_dict(old)
	assert float(m.buy_item.weight[BOOK].abs().max()) == 0.0 and float(m.buy_book_reset) == 1.0 and m.book_row_reset_now
	other = [r for r in range(m.buy_item.weight.shape[0]) if r != BOOK]
	assert torch.equal(m.buy_item.weight[other], old["buy_item.weight"][other])
	with torch.no_grad():  # (training moves the row)
		m.buy_item.weight[BOOK] = 0.07
	saved = m.state_dict()  # (a restart's checkpoint and a worker's reload: both carry the marker)
	restarted = model(monkeypatch, True)
	restarted.load_state_dict(saved)
	assert float(restarted.buy_item.weight[BOOK][0]) == pytest.approx(0.07) and not restarted.book_row_reset_now
	worker = model(monkeypatch, True)
	worker.load_state_dict({k: v.clone() for k, v in saved.items()})
	assert torch.equal(worker.buy_item.weight, m.buy_item.weight) and not worker.book_row_reset_now
	from campaign_model.model import with_added  # (the eval's and the servers' path: with_added first fills the marker at 0)
	served = model(monkeypatch, True)
	served.load_state_dict(with_added(old, served.state_dict()))
	assert float(served.buy_item.weight[BOOK].abs().max()) == 0.0 and served.book_row_reset_now
	again = model(monkeypatch, True)
	again.load_state_dict(with_added(saved, again.state_dict()))
	assert float(again.buy_item.weight[BOOK][0]) == pytest.approx(0.07) and not again.book_row_reset_now


def test_the_reset_row_moves_by_the_gradient_and_its_first_moment_is_0(monkeypatch):
	from campaign_model.train import make_optimizer, zero_book_moment

	m = model(monkeypatch, True).train()
	optimizer = make_optimizer(m)
	x = inputs()
	hero(x, 0, ORRIN, 3)
	mask = torch.zeros_like(x["buy_mask"]).reshape(-1, HO, c.N_BUY_ITEMS)
	mask[:, 0, BOOK] = True
	mask[:, 0, (BOOK + 1) % c.N_BUY_ITEMS] = True  # (two legal items: a head with one legal entry has no gradient on it)
	x["buy_mask"] = mask.reshape(x["buy_mask"].shape)

	def loss():  # (the book bought: BUY's type and its entry)
		out = m(dict(x))
		return -(out["type_logp"][:, c.ActionType.BUY] + out["buy_logp"].reshape(-1, HO, c.N_BUY_ITEMS)[:, 0, BOOK]).sum()

	for _ in range(2):  # (two updates with old moments, then the reset)
		optimizer.zero_grad()
		loss().backward()
		optimizer.step()
	m.load_state_dict({k: v for k, v in m.state_dict().items() if k != "buy_book_reset"})
	zero_book_moment(optimizer, m)
	assert float(m.buy_item.weight[BOOK].abs().max()) == 0.0 and float(optimizer.state[m.buy_item.weight]["exp_avg"][BOOK].abs().max()) == 0.0
	for _ in range(2):
		optimizer.zero_grad()
		loss().backward()
		optimizer.step()
	assert float(m.buy_item.weight[BOOK].abs().max()) > 0  # (the gradient alone moves it from 0)


def test_every_prior_between_two_heroes_sends_the_army_to_the_main(monkeypatch):
	"""e5 (2026-11-16 20:54): "swapping the main to some scout that should deliver creatures is bad". Our main (Solmyr,
	level 1, 1 angel, a weak multiplier) meets a scout (Orrin, level 1, 20 angels, a strong multiplier) two tiles east:
	with M, ARMY's moves between them (army_prior), GIVE_ALL's receiver (stronger_receiver) and the meeting's hero term in
	MOVE's prior (meeting_value with the own score) all favour the main; off, each the scout."""
	from campaign_model import features as ft_

	x = inputs(1, 5)
	hero(x, 0, SOLMYR, 1, army=1)
	hero(x, 1, ORRIN, 1, army=20)
	x["h_garrisoned"][:, :2] = 0
	x["h_level"][0, :2] = 1
	x["h_primary"][0, 0], x["h_primary"][0, 1], x["h_artifacts"][0, :2] = torch.tensor([0, 0, 2, 3]), torch.tensor([4, 4, 1, 1]), -1
	x["h_detailed"][0, :2] = True
	x["h_morale"][0, :2] = 1
	x["h_pos"][0, 0], x["h_pos"][0, 1] = torch.tensor([10, 10, 0]), torch.tensor([12, 10, 0])
	x["x_heroes"][0] = torch.tensor([1, 2])  # (GIVE_ALL's rows: hero 0 first, hero 1 second)
	on_m, off_m = model(monkeypatch, True), model(monkeypatch, False)
	on, off = known(on_m, x), known(off_m, x)
	assert bool(on.main[0, 0]) and not bool(on.main[0, 1]) and bool(off.main[0, 1])
	ALL = int(c.ArmyAmount.ALL)
	a_on, a_off = on_m.features.army_prior(on.main_score), off_m.features.army_prior(off.hero_strength[:, :HO])
	assert float(a_on[0, 1, 0, 0, ALL]) > float(a_on[0, 0, 0, 1, ALL]) and float(a_off[0, 1, 0, 0, ALL]) < float(a_off[0, 0, 0, 1, ALL])
	assert on_m.features.stronger_receiver(x, on)[0].tolist() == [1.0, 0.0] and off_m.features.stronger_receiver(x, off)[0].tolist() == [0.0, 1.0]
	offsets = torch.arange(c.WINDOW) - c.WINDOW_RADIUS
	dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
	tile_x, tile_y = 10 + dx, 10 + dy
	r = c.WINDOW_RADIUS

	def on_tile(pos, present):
		return (pos[:, None, None, :, 0] == tile_x[None, ..., None]) & (pos[:, None, None, :, 1] == tile_y[None, ..., None]) & (pos[:, None, None, :, 2] == 0) & present[:, None, None, :]

	t = on_m.features
	with torch.no_grad():
		by_score = float(t.meeting_value(x, on, torch.tensor([0]), torch.tensor([0]), on_tile, on.main, on.main_score)[0][r, r + 2])
		by_army = float(t.meeting_value(x, on, torch.tensor([0]), torch.tensor([0]), on_tile, on.army_main)[0][r, r + 2])

	def led(h, angels):
		rest = x["h_morale"][0, h].float() - ft_.make_up_morales(x["h_army_creature"][0, h], x["h_army_count"][0, h].float(), t)
		creatures = torch.tensor([13] + [-1] * 6)
		return float(ft_.led_values(creatures, torch.tensor([float(angels)] + [0.0] * 6), on.hero_multiplier[0, h], rest, t))

	apart = led(0, 1) + led(1, 20)
	gain = float(t.gain(t.meeting_gain))
	assert by_score == pytest.approx(max(led(0, 21) - apart, 0.0) * gain, rel=1e-4, abs=1e-3)  # (joined under the main)
	assert by_army == pytest.approx(max(led(1, 21) - apart, 0.0) * gain, rel=1e-4, abs=1e-3) and by_army > by_score  # (under the scout: its multiplier makes more of it)


def test_the_spellbook_prior(monkeypatch):
	"""The user (2026-11-16): the main hero's book +10, another hero's -1 (book_values); off: Nullkiller2's."""
	x = inputs()
	hero(x, 0, SOLMYR, 1)
	hero(x, 1, ORRIN, 1)
	on_m = model(monkeypatch, True)
	f = known(on_m, x)
	assert bool(f.main[0, 0]) and not bool(f.main[0, 1])
	assert on_m.features.book_prior(f.main)[0, :2].tolist() == pytest.approx([10.0, -1.0])


def recruit_state(monkeypatch):
	m = model(monkeypatch, False)
	with torch.no_grad():
		last = m.recruit_head[2]
		last.weight.normal_(0, 0.05)
		last.bias.normal_(0, 0.05)
		last.weight[1::2] -= 0.5  # (the hero rows far under the garrison's, as the teacher left them)
	return {k: v for k, v in m.state_dict().items() if k not in NEW_KEYS}


def recruit_parts(m, x):
	from campaign_model.model_v2 import recruit_rows

	cap = {}
	hook = m.recruit_head.register_forward_hook(lambda mod, i, o: cap.__setitem__("out", o))
	with torch.no_grad():
		m(dict(x))
	hook.remove()
	out = cap["out"].reshape(*cap["out"].shape[:-1], c.N_RECRUIT_AMOUNTS, c.N_RECRUIT_TARGETS)
	return out[..., 0], out[..., 1]


def test_the_recruit_rows_copy_once(monkeypatch):
	from campaign_model.model import with_added

	old = recruit_state(monkeypatch)
	x = inputs()
	hero(x, 0, ORRIN, 1)
	off = model(monkeypatch, False)
	off.load_state_dict(old)
	g, h = recruit_parts(off, x)
	assert float((g - h).abs().max()) > 0.1 and not off.recruit_rows_reset_now
	on = model(monkeypatch, True)
	on.load_state_dict(old)
	g, h = recruit_parts(on, x)
	assert float((g - h).abs().max()) < 1e-6 and on.recruit_rows_reset_now and float(on.recruit_target_reset) == 1.0
	with torch.no_grad():
		on.recruit_head[2].bias[1] += 0.3  # (training moves a hero row)
	saved = on.state_dict()
	for loader in (lambda m: m.load_state_dict(saved), lambda m: m.load_state_dict({k: v.clone() for k, v in saved.items()}), lambda m: m.load_state_dict(with_added(saved, m.state_dict()))):
		again = model(monkeypatch, True)
		loader(again)
		assert torch.equal(again.recruit_head[2].bias, on.recruit_head[2].bias) and not again.recruit_rows_reset_now
	served = model(monkeypatch, True)
	served.load_state_dict(with_added(old, served.state_dict()))
	assert served.recruit_rows_reset_now


def test_the_recruit_hero_rows_train_apart(monkeypatch):
	from campaign_model.train import make_optimizer, zero_recruit_moments

	m = model(monkeypatch, True).train()
	m.load_state_dict(recruit_state(monkeypatch))
	optimizer = make_optimizer(m)
	zero_recruit_moments(optimizer, m)
	x = inputs()
	hero(x, 0, ORRIN, 1)
	x["t_visiting"][:, 0] = 0
	mask = torch.zeros_like(x["recruit_mask"]).reshape(x["recruit_mask"].shape[0], -1)
	mask[:, :8] = True
	x["recruit_mask"] = mask.reshape(x["recruit_mask"].shape)
	for _ in range(2):
		optimizer.zero_grad()
		out = m(dict(x))
		(-out["recruit_logp"].reshape(out["recruit_logp"].shape[0], -1)[:, 1].sum() - out["type_logp"][:, c.ActionType.RECRUIT].sum()).backward()
		optimizer.step()
	w = m.recruit_head[2].weight.detach()
	assert float((w[1::2] - w[0::2]).abs().max()) > 0  # (the hero rows move apart from the garrison's)
