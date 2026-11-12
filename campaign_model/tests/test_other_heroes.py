"""F1, schema 20 (e5 for the user, 2026-11-12): our other heroes' time to the same place. w_other_turns (1 + the fewest
days another hero of ours reaches a window tile, 0 none) at the end of the inputs: a record of 19 reads none. In the
graph, on MOVE and TRAVEL, each day discount's difference against the best other hero's and "another hero there today",
into view_pair and travel_view as zero columns at the end: a checkpoint from before loads padded and plays as it did."""
import zlib
from types import SimpleNamespace

import numpy as np
import torch

from campaign_model import constants as c
from campaign_model.model_v2 import OTHER_VIEWS, CampaignModelV2, other_views
from campaign_model.records import _STARTS, IN_BYTES, SHORT_RECORDS, decode_inputs
from campaign_model.schema import random_inputs
from campaign_model.tests.test_model import encoded
from campaign_model.tests.test_model_v2 import rich_inputs


def test_a_record_from_before_reads_with_no_other_hero():
	assert _STARTS["w_other_turns"] in SHORT_RECORDS and _STARTS["w_other_turns"] < IN_BYTES
	x = random_inputs(8, seed=3)
	b = int(np.flatnonzero(x["w_other_turns"].reshape(8, -1).any(1))[0])
	x = {k: v[b : b + 1] for k, v in x.items()}
	full = zlib.decompress(encoded(x, 0))
	now = decode_inputs([SimpleNamespace(inputs=zlib.compress(full))])
	before = decode_inputs([SimpleNamespace(inputs=zlib.compress(full[: _STARTS["w_other_turns"]]))])
	assert np.array_equal(now["w_other_turns"], x["w_other_turns"]) and not before["w_other_turns"].any()
	for name in x:
		if name != "w_other_turns":
			assert np.array_equal(now[name], before[name]), name


def test_the_views_are_bounded_and_signed_by_who_is_nearer():
	mine = torch.tensor([0.0, 2.0, 5.0, 30.0])
	other = torch.tensor([3.0, 2.0, 0.0, 30.0])
	v = other_views(mine, other)
	assert v.shape == (4, OTHER_VIEWS) and float(v.abs().max()) <= 1.0
	assert (v[0, :2] > 0).all() and (v[1, :2].abs() < 1e-6).all() and (v[2, :2] < 0).all()  # (nearer, even, farther)
	assert v[:, 2].tolist() == [0.0, 0.0, 1.0, 0.0]  # (the other there today)


def test_a_checkpoint_from_before_loads_padded_and_plays_as_it_did():
	torch.manual_seed(3)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	for layer in (model.view_pair, model.travel_view):  # (weights a trained model would have, on the old columns)
		with torch.no_grad():
			layer.weight[:, :-OTHER_VIEWS].normal_(0, 0.1)
	old = {k: (v[:, :-OTHER_VIEWS].clone() if k in ("view_pair.weight", "travel_view.weight") else v.clone()) for k, v in model.state_dict().items()}
	fresh = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	fresh.load_state_dict(old)
	for name in ("view_pair.weight", "travel_view.weight"):
		assert torch.equal(fresh.state_dict()[name][:, -OTHER_VIEWS:], torch.zeros_like(fresh.state_dict()[name][:, -OTHER_VIEWS:]))
	x = {k: torch.from_numpy(v) for k, v in rich_inputs(6, 4).items()}
	planted = dict(x)
	planted["w_other_turns"] = torch.randint(0, 5, x["w_other_turns"].shape)  # (at zero columns the field changes nothing)
	with torch.no_grad():
		a, b_ = fresh(x), fresh(planted)
	for name, value in a.items():
		if torch.is_tensor(value) and value.is_floating_point():
			assert torch.equal(torch.nan_to_num(value, nan=0.0, posinf=1e9, neginf=-1e9), torch.nan_to_num(b_[name], nan=0.0, posinf=1e9, neginf=-1e9)), name
	assert c.SCHEMA_VERSION == 20
