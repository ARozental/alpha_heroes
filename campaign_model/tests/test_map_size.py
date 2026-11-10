"""The map grid's blocks (docs/map_size_plan.md, 2026-11-08): every map up to 36 x 36 reads as before, bit for bit; a
bigger one a block of side x side tiles a cell, its positions and distances still in tiles."""
from pathlib import Path

import numpy as np
import torch

from campaign_model import constants as c
from campaign_model import features as ft
from campaign_model.model_v2 import CampaignModelV2, distance_buckets, map_index, window_tile_index
from campaign_model.tests.test_model import tensors
from campaign_model.tests.test_model_v2 import rich_inputs

GOLDEN = Path(__file__).parent / "data" / "map_size_golden.pt"
S, L, HO, W = c.MAP_TILES, c.MAP_LEVELS, c.OWN_HEROES, c.WINDOW


def golden_model():
	"""The model the golden outputs came from (scratchpad make_golden.py, the commit before the blocks): each parameter
	set by its name, the side embeddings (new since) left at their 0."""
	torch.manual_seed(0)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	with torch.no_grad():
		for name, p in sorted(model.named_parameters()):
			if name.startswith(("side_tile", "side_global")):
				continue
			g = torch.Generator().manual_seed(sum(ord(ch) * (i + 1) for i, ch in enumerate(name)) % (2**31))
			p.copy_(0.05 * torch.randn(p.shape, generator=g))
	return model


def test_a_small_map_gives_the_outputs_it_gave_before_the_blocks():
	"""At side 1 (every map up to 36 x 36) every output is the one the model gave before the change: bit for bit on the
	machine that wrote them (the Mac); on another CPU within float32 rounding (the PC's: 4.8e-7 at most, 2026-11-08), the
	same entries -inf."""
	import platform

	golden = torch.load(GOLDEN)
	here = f"{platform.system()}-{platform.machine()}" == golden.pop("_platform")
	with torch.no_grad():
		out = golden_model()(tensors(rich_inputs(3, 2)))
	for name, before in golden.items():
		now = out[name][:, :1] if name == "fs_move" else out[name]
		if here:
			assert torch.equal(now, before), name
		else:
			assert torch.equal(torch.isfinite(now), torch.isfinite(before)) and torch.allclose(now, before, rtol=1e-6, atol=1e-6), name


def test_block_indices_at_side_1_are_the_tile_indices():
	"""map_index, window_tile_index and distance_buckets with side 1 are their tile formulas, on random positions."""
	rng = np.random.default_rng(1)
	b = 4
	positions = torch.from_numpy(rng.integers(-3, 40, (b, 6, 3))).long()
	positions[..., 2] = positions[..., 2].clamp(0, 2)
	one = torch.ones(b, dtype=torch.long)
	assert all(torch.equal(a, e) for a, e in zip(map_index(positions, b, one), map_index(positions, b)))
	middles = (positions[:, :HO].clone() if positions.shape[1] >= HO else torch.cat([positions, positions], dim=1)[:, :HO]).reshape(-1, 3)
	windows = torch.arange(b * HO)
	assert torch.equal(window_tile_index(middles, windows, one), window_tile_index(middles, windows))
	delta = torch.from_numpy(rng.integers(-60, 60, (b, 5, 7, 3))).long()
	assert torch.equal(distance_buckets(delta, one[:, None, None]), distance_buckets(delta))


def test_a_144_map_maps_tiles_to_blocks_of_4():
	"""side 4 (144 x 144), positions in [0, 144): every index inside the grid, a hero's 33 x 33 window on at most 9 x 9
	blocks, each distance in tiles within one block side of the exact one."""
	rng = np.random.default_rng(2)
	b, side = 3, 4
	sides = torch.full((b,), side, dtype=torch.long)
	positions = torch.from_numpy(rng.integers(0, 144, (b, HO, 3))).long()
	positions[..., 2] = positions[..., 2] % L
	index, inside = map_index(positions, b, sides)
	assert inside.all() and (index >= 0).all() and (index < b * L * S * S).all()
	windows = torch.arange(b * HO)
	tiles = window_tile_index(positions.reshape(-1, 3), windows, sides)
	for k in range(b * HO):
		cells = tiles[k][tiles[k] >= 0]
		assert len(torch.unique(cells)) <= 9 * 9 and (cells < L * S * S).all()
	exact = torch.from_numpy(rng.integers(0, 144, (500, 2))).long()
	other = torch.from_numpy(rng.integers(0, 144, (500, 2))).long()
	true = (exact - other).abs().amax(dim=-1)
	blocks = (torch.div(exact, side, rounding_mode="floor") - torch.div(other, side, rounding_mode="floor")).abs().amax(dim=-1) * side
	assert ((blocks - true).abs() <= side).all()


def big_map(x, side=4):
	"""rich_inputs' decisions as on a map side times bigger: its size and every position (in tiles) scaled; the grid's
	arrays keep their shape."""
	x = {k: v.copy() for k, v in x.items()}
	x["g_map"][:, :2] = 36 * side
	for name in ("h_pos", "t_pos", "o_pos", "m_pos", "a_pos"):
		if name in x:
			placed = x[name][..., :2] >= 0
			x[name][..., :2] = np.where(placed, x[name][..., :2] * side + side // 2, x[name][..., :2])
	return x


def test_a_144_map_runs_through_the_model_with_the_same_shapes():
	"""A 144 x 144 map's decisions (side 4): the forward runs, every output finite where the 36 x 36 one is, the same
	shapes (the graphs capture one shape a batch)."""
	torch.manual_seed(3)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = rich_inputs(3, 2)
	with torch.no_grad():
		small, big = model(tensors(x)), model(tensors(big_map(x)))
	assert int(ft.block_side(tensors(big_map(x)))[0]) == 4
	for name, value in small.items():
		if not torch.is_tensor(value):
			continue
		assert big[name].shape == value.shape, name
		if value.dtype.is_floating_point:
			assert (torch.isfinite(big[name]) == torch.isfinite(value)).all(), name


def test_a_checkpoint_from_before_the_blocks_loads_with_them_at_0_and_side_1_never_learns():
	"""An older state dict (no side embeddings) loads (model.ADDED_HEADS: at 0), its outputs the full model's; a
	gradient at side 1 never reaches side 1's rows (padding_idx), a side-4 map's does reach its own."""
	torch.manual_seed(4)
	full = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	state = {k: v for k, v in full.state_dict().items() if not k.startswith(("side_tile", "side_global"))}
	old = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	old.load_state_dict(state)
	x = rich_inputs(2, 5)
	with torch.no_grad():
		a, b = full(tensors(x)), old(tensors(x))
	assert torch.equal(a["type_logp"], b["type_logp"]) and torch.equal(a["value"], b["value"])
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).train()
	with torch.no_grad():  # (the heads' zero outputs given weights, so a loss reaches the embeddings)
		for p in model.parameters():
			p.add_(0.02 * torch.randn_like(p))
		model.side_tile.weight.zero_()
		model.side_global.weight.zero_()
	for inputs, row, learns in ((x, 1, False), (big_map(x), 4, True)):
		model.zero_grad()
		out = model(tensors(inputs))
		(out["value_logit"].sum() + out["type_logp"].masked_fill(~torch.isfinite(out["type_logp"]), 0).sum()).backward()
		for table in (model.side_tile, model.side_global):
			grad = table.weight.grad
			assert grad is not None and (grad[row].abs().sum() > 0) == learns, (row, learns)
			assert grad[1].abs().sum() == 0
