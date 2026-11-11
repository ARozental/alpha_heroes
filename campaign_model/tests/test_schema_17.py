"""Schema 17: a hero's further windows (w_extra, w_extra_pos) and the teleporter entrances no exit of which we have seen
(TILE_UNSEEN_EXIT, OBJECT_UNSEEN_EXIT).

Before it a hero's MOVE targets were the tiles of the 33 x 33 square around it on its own level: nothing past a gate or
a monolith was one, and TRAVEL's areas are a day or more away, so the only way through a gate was a MOVE onto the gate's
own tile, worth nothing by what stands on it. Now AlphaMapAI shows the tiles a hero reaches today outside its own
window in further windows, in the window slots no hero holds, and the model reads each window as its hero's."""
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model import features as ft
from campaign_model.failures import safe_pickups, window_hero
from campaign_model.model_v2 import CampaignModelV2, windows_of
from campaign_model.records import _STARTS, IN_BYTES, SHORT_RECORDS, decode_inputs, decode_wire, heroes_in
from campaign_model.schema import random_inputs
from campaign_model.tests.test_model import encoded, tensors
from campaign_model.tests.test_model_v2 import rich_inputs

HO, W, R = c.OWN_HEROES, c.WINDOW, c.WINDOW_RADIUS
WINDOW_INPUTS = ("w_terrain", "w_road", "w_flags", "w_object", "w_subtype", "w_owner", "w_creature", "w_count", "w_move_left", "w_turns", "move_mask", "w_arrive_left")
SPARE = 5  # a window slot no hero of random_inputs holds (it makes 1 to 3)
KNOWN = ("tile_value", "tile_prior", "move_features", "move_strength", "move_relative", "tile_fight_ratio", "tile_fights", "tile_stacks")


@pytest.fixture(scope="module")
def model():
	torch.manual_seed(0)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	g = torch.Generator().manual_seed(1)
	with torch.no_grad():
		for p in model.parameters():  # (the zero starts off 0: every path carries something)
			p.add_(0.05 * torch.randn(p.shape, generator=g))
	return model


def turns(batch, seed):
	"""rich_inputs' turn decisions alone (the ones with MOVE's tables), slot SPARE empty in each."""
	x = rich_inputs(4 * batch + 8, seed)
	keep = np.flatnonzero(x["g_kind"] == c.DecisionKind.TURN)[:batch]
	assert len(keep) >= 2 and not x["h_present"][:, SPARE].any()
	return {k: v[keep].copy() for k, v in x.items()}


def with_further(x, hero, middle, source=None):
	"""x with slot SPARE of every decision a further window of hero token `hero` around `middle` [3], its tiles those of
	window slot `source` (default: the hero's own)."""
	y = {k: v.copy() for k, v in x.items()}
	y["w_extra"][:, SPARE] = hero + 1
	y["w_extra_pos"][:, SPARE] = middle
	for name in WINDOW_INPUTS:
		y[name][:, SPARE] = x[name][:, hero if source is None else source]
	return y


def window_row(known, b, slot):
	"""The row of decision b's window slot among the windows the features were computed for."""
	found = torch.nonzero(known.windows[: known.n] == b * HO + slot).reshape(-1)
	assert len(found) == 1
	return int(found[0])


def test_no_further_window_is_every_slot_its_own_heros(model):
	"""w_extra 0 (as every record from before reads): each window its slot's own hero's around where it stands, the
	slots that hold a window our heroes', and the model's outputs those of inputs with no w_extra at all."""
	x = tensors(turns(8, seed=2))
	windows, n, real = windows_of(x)
	assert torch.equal(windows, torch.nonzero(x["h_present"][:, :HO].reshape(-1)).squeeze(-1)) and n == len(windows) and real.all()
	heroes, middles = ft.window_heroes(x, windows)
	assert torch.equal(heroes, windows) and torch.equal(middles, x["h_pos"][:, :HO].reshape(-1, 3)[windows])
	assert torch.equal(ft.slot_heroes(x), torch.arange(HO).expand(len(x["g_day"]), HO))
	bare = {k: v for k, v in x.items() if k not in ("w_extra", "w_extra_pos")}
	with torch.no_grad():
		full, without = model(x), model(bare)
	for name in full:
		if torch.is_tensor(full[name]):
			assert torch.equal(full[name], without[name]), name


def test_a_further_window_is_read_as_its_heros(model):
	"""A further window of hero 0 with its own window's middle and tiles: every feature of its tiles and every MOVE
	logit is the own window's (the hero's token, its act-now score, its strength, where it stands: each read through
	the window's hero, not through the slot, whose hero does not exist)."""
	x = turns(8, seed=3)
	# (schema 19: a candidate of Nullkiller2's names one entry, so two copies of a window cannot both hold it: this
	# decision is read without the advisor; its candidates in a further window: test_advice.py)
	x["g_nk_present"][:] = False
	y = tensors(with_further(x, 0, x["h_pos"][:, 0]))
	with torch.no_grad():
		known = model.domain_knowledge(y)
		out = model(y)
	for b in range(len(x["g_day"])):
		own, further = window_row(known, b, 0), window_row(known, b, SPARE)
		assert int(known.heroes[further]) == b * HO and torch.equal(known.middles[further], known.middles[own])
		for name in KNOWN:
			assert torch.equal(getattr(known, name)[further], getattr(known, name)[own]), name
		legal = y["move_mask"][b, 0]
		assert legal.any()
		assert torch.allclose(out["move_logp"][b, SPARE][legal], out["move_logp"][b, 0][legal], atol=1e-5)
		assert torch.allclose(out["fs_tiles"][b, SPARE], out["fs_tiles"][b, 0], atol=1e-5) and torch.allclose(out["fs_move"][b, SPARE], out["fs_move"][b, 0], atol=1e-5)
		assert abs(float(out["move_logp"][b].exp().sum()) - 1) < 1e-4
	# (and the slot without the window offers nothing: its logits masked, as an absent hero's are)
	with torch.no_grad():
		plain = model(tensors(x))
	assert torch.isneginf(plain["move_logp"][:, SPARE]).all()


def test_a_further_window_elsewhere_is_priced_as_if_its_hero_stood_there(model):
	"""A further window of hero 0 around a tile of the other level, against the same decision with hero 0 standing on
	that tile and those tiles its own window: the tiles' values, priors, fights and strengths are the same (no enemy
	hero is in sight, so where the hero stands changes no danger). The window's tiles are its middle's, not the
	hero's: an object, a town or a hero of ours on a tile is found by the tile's own place."""
	x = turns(8, seed=4)
	n = len(x["g_day"])
	middle = np.array([20, 9, 1])
	x["h_pos"][:, 1] = [21, 10, 1]  # (a hero of ours on a tile of that window, where there is one: a meeting's value)
	x["t_pos"][:, 0] = [19, 8, 1]  # (and a town of ours)
	there = dict(x)
	there["h_pos"] = x["h_pos"].copy()
	there["h_pos"][:, 0] = middle
	y = with_further(x, 0, middle)
	with torch.no_grad():
		far, stood = model.domain_knowledge(tensors(y)), model.domain_knowledge(tensors(there))
	differs = 0
	for b in range(n):
		further, own = window_row(far, b, SPARE), window_row(stood, b, 0)
		assert torch.equal(far.middles[further], torch.from_numpy(middle)) and int(far.heroes[further]) == b * HO
		for name in KNOWN:
			assert torch.allclose(getattr(far, name)[further], getattr(stood, name)[own], atol=1e-5), name
		differs += int(not torch.equal(far.tile_value[further], far.tile_value[window_row(far, b, 0)]))
	assert differs > 0  # (the same tiles around where the hero does stand are priced otherwise: the town and the hero are elsewhere)
	# the tile reader's cells of the window are the other level's
	tiles = far.window_tiles[window_row(far, 0, SPARE)]
	S = c.MAP_TILES
	assert int(tiles[R, R]) == (1 * S + 9) * S + 20


def test_the_reach_of_a_further_window_is_its_heros(model, monkeypatch):
	"""model_v2.see scatters each window's reach to its hero's map: a further window's tiles bias its hero's reading
	of the other level, not the reading of the slot's hero (there is none)."""
	x = turns(4, seed=5)
	x["h_pos"][:, :, 2] = 0
	y = with_further(x, 0, np.array([18, 18, 1]))
	y["w_turns"][:, SPARE], y["w_move_left"][:, SPARE] = 0, 700
	seen = []
	read = model.read_tiles.forward

	def spy(tokens, positions, present, flat, tile_positions, valid, reach=None, side=None):
		if reach is not None:
			seen.append(reach.detach().clone())
		return read(tokens, positions, present, flat, tile_positions, valid, reach, side=side) if reach is not None else read(tokens, positions, present, flat, tile_positions, valid, side=side)

	monkeypatch.setattr(model.read_tiles, "forward", spy)
	with torch.no_grad():
		model(tensors(y))
		model(tensors(x))
	with_window, without = seen
	S, L = c.MAP_TILES, c.MAP_LEVELS
	below = slice(S * S, L * S * S)  # the other level's tiles
	assert with_window.shape[1] == c.N_HEROES and with_window.shape[2] == L * S * S
	assert not without[:, 0, below].any() and with_window[:, 0, below].any()  # hero 0 reaches tiles of the other level through it
	assert not with_window[:, SPARE].any()  # (nothing scattered to the slot's hero)
	assert torch.equal(with_window[:, 1:HO], without[:, 1:HO])


def test_a_record_from_before_reads_with_no_further_window():
	"""records: a record of schema 16 ends where w_extra starts; decoded (both ways) its w_extra is 0, and the windows
	a batch needs are its heroes'. With a further window, one more."""
	x = turns(4, seed=6)
	assert _STARTS["w_extra"] in SHORT_RECORDS and _STARTS["w_extra"] < IN_BYTES
	y = with_further(x, 0, np.array([3, 4, 1]))
	full = zlib.decompress(encoded(y, 0))
	assert len(full) == IN_BYTES
	now = decode_inputs([SimpleNamespace(inputs=zlib.compress(full))])
	assert int(now["w_extra"][0, SPARE]) == 1 and now["w_extra_pos"][0, SPARE].tolist() == [3, 4, 1]
	old = [SimpleNamespace(inputs=zlib.compress(full[: _STARTS["w_extra"]]))]
	before = decode_inputs(old)
	assert not before["w_extra"].any() and not before["w_extra_pos"].any()
	assert np.array_equal(before["move_mask"], now["move_mask"])
	heroes = int(x["h_present"][0, :HO].sum())
	assert heroes_in(before).tolist() == [heroes] and heroes_in(now).tolist() == [heroes + 1]
	assert heroes_in(decode_wire(old)).tolist() == [heroes] and heroes_in(decode_wire([SimpleNamespace(inputs=zlib.compress(full))])).tolist() == [heroes + 1]


def test_the_window_budget_counts_further_windows():
	"""model_v2.windows_of with fixed windows and a budget: the slots that hold a window first, a further one among
	them; every slot's when none is set."""
	x = tensors(with_further(turns(4, seed=7), 0, np.array([3, 4, 1])))
	n = len(x["g_day"])
	present = ft.windows_present(x)
	assert present[:, SPARE].all() and int(present.sum()) == int(x["h_present"][:, :HO].sum()) + n
	windows, count, real = windows_of(x, fixed=True, budget=int(present.sum()))
	assert count == int(present.sum()) and real.all() and set(windows.tolist()) == set(torch.nonzero(present.reshape(-1)).reshape(-1).tolist())
	windows, count, real = windows_of(x, fixed=True)
	assert count == n * HO and torch.equal(real, present.reshape(-1))


def test_an_entrance_with_no_exit_seen_is_worth_the_land_it_shows(model):
	"""TILE_UNSEEN_EXIT on a tile whose surroundings are all revealed: the land a step there shows is a whole sight (1,
	as a tile deep in fog), and its value rises by it; a whirlpool's is not priced (it takes troops). OBJECT_UNSEEN_EXIT:
	its TRAVEL area counts as one we have not revealed."""
	x = turns(4, seed=8)
	explore = 5  # (model.MOVE_FEATURES: prior, object, fight, loss, win, explore, ...)
	row, column = R + 3, R - 2
	for name in ("w_object", "w_creature"):
		x[name][:, 0, row, column] = -1
	gate, pool = {k: v.copy() for k, v in x.items()}, {k: v.copy() for k, v in x.items()}
	for y, obj in ((gate, 103), (pool, ft.OBJ_WHIRLPOOL)):  # (103: a subterranean gate)
		y["w_object"][:, 0, row, column] = obj
		y["w_flags"][:, 0, row, column] |= c.TILE_VISITABLE | c.TILE_UNSEEN_EXIT
	seen = {k: v.copy() for k, v in gate.items()}
	seen["w_flags"][:, 0, row, column] -= c.TILE_UNSEEN_EXIT
	pool_seen = {k: v.copy() for k, v in pool.items()}  # (the same whirlpool unflagged: the flag adds nothing to it)
	pool_seen["w_flags"][:, 0, row, column] -= c.TILE_UNSEEN_EXIT
	with torch.no_grad():
		known = {name: model.domain_knowledge(tensors(y)) for name, y in (("gate", gate), ("pool", pool), ("seen", seen), ("pool_seen", pool_seen))}
	k = window_row(known["gate"], 0, 0)
	at = lambda name, feature: getattr(known[name], feature)[k, row, column]  # noqa: E731
	assert float(at("seen", "move_features")[explore]) == 0 and float(at("pool", "move_features")[explore]) == 0
	assert float(at("gate", "move_features")[explore]) == 1
	assert float(at("gate", "tile_value")) > float(at("seen", "tile_value")) and float(at("gate", "tile_prior")) > float(at("seen", "tile_prior"))
	assert torch.equal(known["pool"].tile_value[k, row, column], known["pool_seen"].tile_value[k, row, column])  # (2026-11-10: against the gate's it hung on the fixture's random object gains)
	# TRAVEL: the area of an object so flagged
	x["g_tiles"][:] = 0  # (no tile counted: no area hidden by its tiles)
	far = {k_: v.copy() for k_, v in x.items()}
	far["m_object"][:, 0], far["m_pos"][:, 0], far["m_flags"][:, 0] = 103, [30, 5, 1], c.OBJECT_UNSEEN_EXIT
	area = int(ft.area_index(tensors(far), tensors(far)["m_pos"][:, :1])[0, 0])
	hidden = ft.unseen_exits(tensors(far))
	assert hidden.shape == (len(x["g_day"]), c.MAP_LEVELS * c.TRAVEL_GRID**2) and float(hidden[0, area]) == 1 and float(hidden[0].sum()) == 1
	assert not ft.unseen_exits(tensors(x)).any()
	for y in (far, x):
		y["travel_days"][:, 0] = 150
		y["travel_mask"][:, 0] = True
	with torch.no_grad():
		with_gate, without = model.domain_knowledge(tensors(far)).travel_prior, model.domain_knowledge(tensors(x)).travel_prior
	assert float(with_gate[0, 0, area]) > float(without[0, 0, area])
	others = torch.ones_like(with_gate[0, 0], dtype=torch.bool)
	others[area] = False
	assert torch.equal(with_gate[0, 0][others], without[0, 0][others])
	pool_far = {k_: v.copy() for k_, v in far.items()}
	pool_far["m_object"][:, 0] = ft.OBJ_WHIRLPOOL
	assert not ft.unseen_exits(tensors(pool_far)).any()


def test_the_tile_encoder_reads_the_flag_only_once_trained():
	"""encode_tiles.unseen_exit starts at 0 (a checkpoint from before loads it so: model.ADDED): the flag changes no
	output of a fresh model but through the features that price it; a checkpoint without the parameter loads."""
	torch.manual_seed(0)
	fresh = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	assert not fresh.encode_tiles.unseen_exit.any()
	x = turns(4, seed=9)
	flagged = {k: v.copy() for k, v in x.items()}
	flagged["tile_flags"][:, 0, 3, 3] |= c.TILE_ON_MAP | c.TILE_UNSEEN_EXIT
	x["tile_flags"][:, 0, 3, 3] |= c.TILE_ON_MAP
	with torch.no_grad():
		a, b = fresh.encode_tiles(tensors(flagged)), fresh.encode_tiles(tensors(x))
	assert torch.equal(a, b)
	state = {k: v for k, v in fresh.state_dict().items() if k != "encode_tiles.unseen_exit"}
	other = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	with torch.no_grad():
		other.encode_tiles.unseen_exit.fill_(3.0)
	other.load_state_dict(state)
	assert not other.encode_tiles.unseen_exit.any()
	with torch.no_grad():
		other.encode_tiles.unseen_exit.fill_(0.5)
		assert not torch.equal(other.eval().encode_tiles(tensors(flagged)), other.encode_tiles(tensors(x)))


def test_the_metrics_read_a_move_through_a_further_window():
	"""failures: a MOVE's entry is (window slot, tile); a further window's slot is its hero's, its tile around the
	window's middle; a safe pickup in a further window is its hero's."""
	x = with_further(turns(4, seed=10), 1, np.array([7, 30, 1]))
	hero, middle = window_hero(x, 0, SPARE)
	assert hero == 1 and middle.tolist() == [7, 30, 1]
	hero, middle = window_hero(x, 0, 0)
	assert hero == 0 and middle.tolist() == x["h_pos"][0, 0].tolist()
	for name in ("w_object", "w_creature"):
		x[name][0] = -1
	x["move_mask"][0] = False
	x["h_present"][0, HO:] = False
	x["w_object"][0, SPARE, 4, 4], x["w_flags"][0, SPARE, 4, 4], x["move_mask"][0, SPARE, 4, 4] = 79, c.TILE_ON_MAP | c.TILE_REVEALED | c.TILE_VISITABLE, True  # (79: a resource)
	pickups = safe_pickups(x, 0)
	assert pickups.tolist() == [False, True] + [False] * (HO - 2)


def test_random_inputs_have_no_further_windows():
	assert not random_inputs(6, seed=1)["w_extra"].any()


def test_the_games_get_the_runs_window_settings():
	"""play.window_settings: the config's two switches as AlphaMapAI's environment, for every game of a run."""
	from campaign_model.config import CampaignConfig
	from campaign_model.play import window_settings

	assert dict(window_settings(CampaignConfig())) == {"ALPHA_MAP_FURTHER_WINDOWS": "1", "ALPHA_MAP_UNSEEN_EXITS": "1"}
	assert dict(window_settings(CampaignConfig(further_windows=0, unseen_exits=False))) == {"ALPHA_MAP_FURTHER_WINDOWS": "0", "ALPHA_MAP_UNSEEN_EXITS": "0"}
