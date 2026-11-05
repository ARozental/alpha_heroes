"""The subtype tables folded (2026-10-31): a blocked, not visitable tile embeds as nothing but blocked; a checkpoint with
the old SUBTYPE_HASH-row tables migrates to SUBTYPE_ROWS (train.migrate_subtypes)."""
import torch

from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.encoders import TileEncoder
from campaign_model.model_v2 import CampaignModelV2
from campaign_model.train import make_optimizer, migrate_subtypes

ON = c.TILE_ON_MAP | c.TILE_REVEALED


def test_a_blocked_tile_is_blocked_and_nothing_else():
	"""An obstacle (an object of no class) is an empty blocked tile; a town's blocked tiles keep the town but not their
	subtype; the town's visitable entrance stays itself."""
	torch.manual_seed(0)
	encoder = TileEncoder(8).eval()
	tiles = [  # (object, subtype, flags)
		(200, 5, ON | c.TILE_BLOCKED),  # an obstacle
		(-1, 0, ON | c.TILE_BLOCKED),  # a blocked tile with nothing on it
		(98, 3, ON | c.TILE_BLOCKED),  # a town's wall
		(98, 7, ON | c.TILE_BLOCKED),  # another town's wall
		(98, 3, ON | c.TILE_BLOCKED | c.TILE_VISITABLE),  # the town's entrance
	]
	shape = (1, 1, 1, len(tiles))
	x = {
		"tile_object": torch.tensor([t[0] for t in tiles]).reshape(shape), "tile_subtype": torch.tensor([t[1] for t in tiles]).reshape(shape),
		"tile_flags": torch.tensor([t[2] for t in tiles]).reshape(shape), "tile_terrain": torch.zeros(shape, dtype=torch.long),
		"tile_road": torch.zeros(shape, dtype=torch.long), "tile_creature": torch.full(shape, -1), "tile_count": torch.zeros(shape, dtype=torch.long),
	}
	with torch.no_grad():
		out = encoder(x).reshape(len(tiles), -1)
	assert torch.allclose(out[0], out[1])  # the obstacle: blocked, nothing more
	assert torch.allclose(out[2], out[3])  # a town's walls: the town, blocked; not their graphic
	assert not torch.allclose(out[2], out[1]) and not torch.allclose(out[2], out[4])


def test_an_old_checkpoint_folds_its_subtype_tables():
	"""Each new row is the mean of the old rows folded into it (a pair's row modulo SUBTYPE_BUCKETS) that a gradient
	reached; a row none reached keeps the fresh model's; Adam's moments fold alike. (v1's table: v2's options have no
	hashed rows since 2026-11-04, encoders.OptionEncoder.)"""
	from campaign_model.model import CampaignModel

	torch.manual_seed(1)
	model = CampaignModel(dim=16, cell_dim=8, layers=1)
	optimizer = make_optimizer(model, CampaignConfig())
	name = "subtype.weight"  # (still hashed: folded)
	state = {"model": {k: v.clone() for k, v in model.state_dict().items()}}
	old = torch.randn(c.SUBTYPE_HASH, state["model"][name].shape[1])
	state["model"][name] = old
	live = torch.zeros(c.SUBTYPE_HASH, dtype=torch.bool)
	live[[5, 5 + c.SUBTYPE_BUCKETS, 17]] = True  # two in bucket 5, one in bucket 17; bucket 17 + 1021 never trained
	order = [p for group in optimizer.param_groups for p in group["params"]]  # (the optimizer's numbering: its groups' order)
	position = next(i for i, p in enumerate(order) if p is dict(model.named_parameters())[name])
	saved = optimizer.state_dict()
	saved["state"] = {position: {"step": torch.tensor(3.0), "exp_avg": old.clone(), "exp_avg_sq": live.float().unsqueeze(1).expand_as(old).clone()}}
	state["optimizer"] = saved
	fresh = model.state_dict()[name].clone()
	assert name in migrate_subtypes(state, model)
	new = state["model"][name]
	assert new.shape[0] == c.SUBTYPE_ROWS
	assert torch.allclose(new[5], (old[5] + old[5 + c.SUBTYPE_BUCKETS]) / 2)
	assert torch.allclose(new[17], old[17])
	assert torch.allclose(new[18], fresh[18]) and torch.allclose(new[c.BLOCKED_SUBTYPE], fresh[c.BLOCKED_SUBTYPE])
	assert torch.allclose(state["optimizer"]["state"][position]["exp_avg"][5], (old[5] + old[5 + c.SUBTYPE_BUCKETS]) / 2)
	model.load_state_dict(state["model"])


def test_an_old_checkpoint_moves_each_object_pair_to_its_own_row():
	"""An (object, subtype) pair's new row (constants.OBJECT_PAIRS) is its old hashed row, exactly."""
	torch.manual_seed(2)
	model = CampaignModelV2(dim=16, cell_dim=8, layers=1)
	state = {"model": {k: v.clone() for k, v in model.state_dict().items()}}
	name = "encode_tiles.subtype.weight"
	old = torch.randn(c.SUBTYPE_HASH, state["model"][name].shape[1])
	state["model"][name] = old
	assert name in migrate_subtypes(state, model)
	new = state["model"][name]
	assert new.shape[0] == c.PAIR_ROWS
	for (obj, subtype) in ((98, 3), (53, 0), (5, 100)):  # a town, a mine, an artifact
		assert torch.equal(new[c.OBJECT_PAIRS[(obj, subtype)]], old[((obj + 1) * 7919 + (obj + 1) * 131 + subtype * 7) % c.SUBTYPE_HASH])
	model.load_state_dict(state["model"])
