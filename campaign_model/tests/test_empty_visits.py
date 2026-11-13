"""B1b (e5's rebuild of B1, 2026-11-13): a visit by a hero that would give nothing, by what we know, has its own bit
(TILE_EMPTY_FOR_HERO in w_flags, m_empty for the object list; VCMI's TILE_VISITED_BY_HERO and m_visitors stay its own).
The MOVE prior's gate closes on it; the learned layers read it only through reach_features' last column, which a
checkpoint from before loads at 0 (pad_new_inputs), so at deploy the prior's change is the only one. Schema 20's records
of 19 read it, m_empty and g_days_no_town as 0."""
import zlib
from types import SimpleNamespace

import numpy as np
import torch

from campaign_model import constants as c
from campaign_model.model_v2 import CampaignModelV2, reach_features
from campaign_model.schema import random_inputs

W = c.WINDOW


def model(seed=3):
	torch.manual_seed(seed)
	return CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()


def tensors(x):
	return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in x.items()}


def inputs(n=2, seed=5):
	x = random_inputs(n, seed=seed)
	x["w_flags"] &= ~np.int64(c.TILE_EMPTY_FOR_HERO)  # (none empty: the tests set their own)
	x["m_empty"][:] = 0
	return x


def gate_tiles(x):
	"""Our heroes' window tiles the MOVE prior's gate is open on (an object to take, not ours, not visited)."""
	flags, obj, owner = x["w_flags"][:, : c.OWN_HEROES], x["w_object"][:, : c.OWN_HEROES], x["w_owner"][:, : c.OWN_HEROES]
	return ((flags & c.TILE_VISITABLE) != 0) & (obj >= 0) & (obj != 34) & (obj != 54) & (owner != c.OWNER_SELF) & ((flags & (c.TILE_VISITED_BY_PLAYER | c.TILE_VISITED_BY_HERO)) == 0)


def test_the_bit_is_its_own_and_beyond_the_numbers_the_encoders_read():
	assert c.TILE_EMPTY_FOR_HERO == 1 << 10 and c.TILE_EMPTY_FOR_HERO >= 1 << c.N_TILE_FLAGS
	assert c.TILE_EMPTY_FOR_HERO not in (c.TILE_VISITED_BY_HERO, c.TILE_VISITED_BY_PLAYER, c.TILE_UNSEEN_EXIT)


def test_reach_features_carry_the_bit_last():
	x = inputs()
	x["w_flags"][0, 0, 3, 4] |= c.TILE_EMPTY_FOR_HERO
	r = reach_features(tensors(x), torch.tensor([0, 1]))
	assert r.shape[-1] == 12 and float(r[0, 3, 4, -1]) == 1.0 and float(r[..., -1].sum()) == 1.0


def test_the_prior_gate_closes_on_the_bit():
	m, x = model(), inputs()
	R = c.WINDOW_RADIUS
	for b in range(x["g_day"].shape[0]):  # (a gold pile beside each decision's first hero, ours: the gate open there)
		x["h_present"][b, 0], x["h_owner"][b, 0] = True, c.OWNER_SELF
		x["w_object"][b, 0, R, R + 1], x["w_subtype"][b, 0, R, R + 1], x["w_owner"][b, 0, R, R + 1] = 79, 6, c.OWNER_NONE
		x["w_flags"][b, 0, R, R + 1] = c.TILE_ON_MAP | c.TILE_REVEALED | c.TILE_VISITABLE
		x["w_creature"][b, 0, R, R + 1], x["w_turns"][b, 0, R, R + 1] = -1, 0
	open_ = gate_tiles(x)
	assert open_.any()
	with torch.no_grad():
		before = m.domain_knowledge(tensors(x))
		y = dict(x)
		y["w_flags"] = x["w_flags"].copy()
		y["w_flags"][:, : c.OWN_HEROES] |= np.where(open_, c.TILE_EMPTY_FOR_HERO, 0).astype(x["w_flags"].dtype)
		after = m.domain_knowledge(tensors(y))
	assert before.tile_object_gold.abs().sum() > 0 and float(after.tile_object_gold.abs().sum()) == 0.0  # (every object the gate priced: closed)


def test_a_checkpoint_from_before_reads_the_bit_through_zero_only_the_prior_changes():
	m = model()
	state = m.state_dict()
	old = dict(state)
	old["reach.mlp.up.weight"] = state["reach.mlp.up.weight"][:, :-1].clone()  # (the reach layer as it was: 11 columns)
	fresh = model(seed=9)
	fresh.load_state_dict(old)
	assert float(fresh.reach.mlp.up.weight[:, -1].abs().sum()) == 0.0
	x = inputs()
	closed = ~gate_tiles(x) & (x["w_object"][:, : c.OWN_HEROES] >= 0)  # (object tiles the prior's gate does not price)
	assert closed.any()
	y = dict(x)
	y["w_flags"] = x["w_flags"].copy()
	y["w_flags"][:, : c.OWN_HEROES] |= np.where(closed, c.TILE_EMPTY_FOR_HERO, 0).astype(x["w_flags"].dtype)
	y["m_empty"] = np.where(x["m_object"] >= 0, 1, 0).astype(x["m_empty"].dtype)
	with torch.no_grad():
		a, b = fresh(tensors(x)), fresh(tensors(y))
	for name, value in a.items():
		if torch.is_tensor(value) and value.is_floating_point():
			assert torch.equal(torch.nan_to_num(value), torch.nan_to_num(b[name])), name  # (the learned side: nothing at deploy)


def test_a_record_of_schema_19_reads_schema_20_as_none():
	from campaign_model.records import IN_BYTES, SHORT_RECORDS, _STARTS, decode_inputs
	from campaign_model.tests.test_model import encoded

	x = inputs(1)
	x["g_days_no_town"][:] = 3
	x["m_empty"][0, :4] = 5
	full = zlib.decompress(encoded(x, 0))
	assert len(full) == IN_BYTES and _STARTS["g_days_no_town"] in SHORT_RECORDS and _STARTS["m_empty"] > _STARTS["g_days_no_town"]
	got = decode_inputs([SimpleNamespace(inputs=zlib.compress(full[: _STARTS["g_days_no_town"]]))])
	assert int(got["g_days_no_town"][0]) == 0 and not got["m_empty"].any()
	whole = decode_inputs([SimpleNamespace(inputs=zlib.compress(full))])
	assert int(whole["g_days_no_town"][0]) == 3 and whole["m_empty"][0, :4].tolist() == [5, 5, 5, 5]


def test_the_days_input_starts_at_zero():
	m = model()
	x = inputs()
	assert float(m.encode_global.no_town_days.weight.detach().abs().sum()) == 0.0
	y = dict(x)
	y["g_days_no_town"] = np.full_like(x["g_days_no_town"], 5)
	with torch.no_grad():
		a, b = m(tensors(x)), m(tensors(y))
	for name, value in a.items():
		if torch.is_tensor(value) and value.is_floating_point():
			assert torch.equal(torch.nan_to_num(value), torch.nan_to_num(b[name])), name
