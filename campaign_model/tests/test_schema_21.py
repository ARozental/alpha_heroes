"""Schema 21 (e5, 2026-11-14: the soft route after the settled-pairs mark was closed): a pair of our heroes' last exchange
as facts the model may learn from, no rule acting on them. On the tile of a hero of ours in a window, TILE_PAIR_UNCHANGED
(both still hold what their last exchange left them) and TILE_PAIR_TODAY (it ended today), read only through
reach_features' two last columns; at an exchange, x_pair_today (one exchange of the pair ended today, two or more) onto
END's and MULTI's scores only, never the global token the value heads read. Every new weight starts at 0, so a model
loaded from a checkpoint of 20 plays as before, bit for bit; records of 20 read the new input as 0."""
import zlib
from types import SimpleNamespace

import numpy as np
import torch

from campaign_model import constants as c
from campaign_model.model import with_added
from campaign_model.model_v2 import GROWN_INPUTS, CampaignModelV2, pad_new_inputs, reach_features
from campaign_model.schema import random_inputs


def model(seed=3):
	torch.manual_seed(seed)
	return CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()


def tensors(x):
	return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in x.items()}


def outputs(m, x):
	with torch.no_grad():
		return {k: v for k, v in m(tensors(x)).items() if torch.is_tensor(v) and v.is_floating_point()}


def same(a, b):
	return all(torch.equal(torch.nan_to_num(a[k]), torch.nan_to_num(b[k])) for k in a)


def bare(x):
	"""x with schema 21's facts cleared: no pair bit on any tile, no exchange today."""
	y = dict(x)
	y["w_flags"] = x["w_flags"] & ~np.int64(c.TILE_PAIR_UNCHANGED | c.TILE_PAIR_TODAY)
	y["x_pair_today"] = np.zeros_like(x["x_pair_today"])
	return y


def test_the_bits_are_their_own_and_beyond_the_numbers_the_encoders_read():
	assert c.TILE_PAIR_UNCHANGED == 1 << 11 and c.TILE_PAIR_TODAY == 1 << 12 and c.TILE_PAIR_UNCHANGED >= 1 << c.N_TILE_FLAGS
	assert len({c.TILE_EMPTY_FOR_HERO, c.TILE_UNSEEN_EXIT, c.TILE_PAIR_UNCHANGED, c.TILE_PAIR_TODAY}) == 4


def test_random_inputs_hold_the_facts():
	x = random_inputs(64, seed=7)
	assert ((x["w_flags"] & c.TILE_PAIR_UNCHANGED) != 0).any() and ((x["w_flags"] & c.TILE_PAIR_TODAY) != 0).any()
	exchange = x["g_kind"] == c.DecisionKind.EXCHANGE
	assert x["x_pair_today"][exchange].any() and not x["x_pair_today"][~exchange].any()


def test_reach_features_carry_the_two_bits_last():
	x = bare(random_inputs(2, seed=5))
	x["w_flags"][0, 0, 3, 4] |= c.TILE_PAIR_UNCHANGED
	x["w_flags"][1, 0, 6, 2] |= c.TILE_PAIR_TODAY
	r = reach_features(tensors(x), torch.tensor([0, c.OWN_HEROES]))
	assert r.shape[-1] == 14
	assert float(r[0, 3, 4, 12]) == 1.0 and float(r[..., 12].sum()) == 1.0
	assert float(r[1, 6, 2, 13]) == 1.0 and float(r[..., 13].sum()) == 1.0


def test_at_zero_the_facts_change_nothing():
	m, x = model(), random_inputs(64, seed=11)
	assert float(m.pair_today.weight.abs().sum()) == 0.0
	assert ((x["w_flags"] & (c.TILE_PAIR_UNCHANGED | c.TILE_PAIR_TODAY)) != 0).any() and x["x_pair_today"].any()
	with torch.no_grad():
		m.reach.mlp.up.weight[:, -2:] = 0.0  # (as a checkpoint of 20 loads them: pad_new_inputs)
	assert same(outputs(m, x), outputs(m, bare(x)))


def test_the_facts_are_read_once_their_weights_move():
	m, x = model(), random_inputs(64, seed=11)
	with torch.no_grad():
		m.reach.mlp.up.weight[:, -2:] = 0.0
		m.pair_today.weight.fill_(0.5)
	a, b = outputs(m, x), outputs(m, bare(x))
	assert not torch.equal(torch.nan_to_num(a["type_logp"]), torch.nan_to_num(b["type_logp"]))  # (END and MULTI at the exchanges)
	turns = torch.from_numpy(x["g_kind"] == c.DecisionKind.TURN)
	assert torch.equal(torch.nan_to_num(a["type_logp"][turns]), torch.nan_to_num(b["type_logp"][turns]))  # (a turn decision's input is 0)
	with torch.no_grad():
		m.pair_today.weight.zero_()
		m.reach.mlp.up.weight[:, -2:] = 0.5
	assert not same(outputs(m, x), outputs(m, bare(x)))  # (the reach columns reach the windows)


def test_the_count_never_reaches_the_value_heads():
	m, x = model(), random_inputs(64, seed=11)
	with torch.no_grad():
		m.pair_today.weight.fill_(0.5)
	a, b = outputs(m, x), outputs(m, {**x, "x_pair_today": np.zeros_like(x["x_pair_today"])})
	for name in a:
		if name.startswith(("value", "share", "growth", "v_end", "phi")):
			assert torch.equal(torch.nan_to_num(a[name]), torch.nan_to_num(b[name])), name


def test_a_checkpoint_of_20_loads_with_the_new_weights_at_zero():
	old = model()
	state = {k: v for k, v in old.state_dict().items() if not k.startswith("pair_today")}
	state["reach.mlp.up.weight"] = state["reach.mlp.up.weight"][:, :-2].clone()  # (schema 20's 12 columns)
	fresh = model(seed=9)
	fresh.load_state_dict(with_added(pad_new_inputs(state, fresh.state_dict()), fresh.state_dict()))
	assert float(fresh.pair_today.weight.abs().sum()) == 0.0 and float(fresh.reach.mlp.up.weight[:, -2:].abs().sum()) == 0.0
	assert torch.equal(fresh.reach.mlp.up.weight[:, :-2], old.reach.mlp.up.weight[:, :-2])
	assert "reach.mlp.up.weight" in GROWN_INPUTS


def test_adams_moments_of_a_grown_layer_are_padded_not_reset():
	from campaign_model.train import fit_moments, make_optimizer

	m = model()
	opt = make_optimizer(m)
	p = m.reach.mlp.up.weight
	state = opt.state[p]
	state["step"] = torch.tensor(100.0)
	state["exp_avg"] = torch.full((p.shape[0], p.shape[1] - 2), 0.25)
	state["exp_avg_sq"] = torch.full((p.shape[0], p.shape[1] - 2), 0.5)
	fit_moments(opt, m)
	assert opt.state[p]["exp_avg"].shape == p.shape
	assert float(opt.state[p]["exp_avg"][:, :-2].min()) == 0.25 and float(opt.state[p]["exp_avg_sq"][:, :-2].min()) == 0.5
	assert float(opt.state[p]["exp_avg"][:, -2:].abs().sum()) == 0.0 and float(opt.state[p]["exp_avg_sq"][:, -2:].abs().sum()) == 0.0


def test_a_record_of_schema_20_reads_the_count_as_none():
	from campaign_model.records import IN_BYTES, SHORT_RECORDS, _STARTS, decode_inputs
	from campaign_model.tests.test_model import encoded

	x = random_inputs(1, seed=4)
	x["x_pair_today"][:] = (0, 1)
	full = zlib.decompress(encoded(x, 0))
	assert len(full) == IN_BYTES and _STARTS["x_pair_today"] in SHORT_RECORDS
	got = decode_inputs([SimpleNamespace(inputs=zlib.compress(full[: _STARTS["x_pair_today"]]))])
	assert not got["x_pair_today"].any()
	whole = decode_inputs([SimpleNamespace(inputs=zlib.compress(full))])
	assert whole["x_pair_today"][0].tolist() == [0, 1]
