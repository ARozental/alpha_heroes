"""The gain-per-movement view (2026-11-15, e5 for the user): a MOVE tile's value as gold a day (features.move_rate), its
two bounded views (features.rate_views) through model_v2's pair_rate, a layer at 0 to begin with."""
import numpy as np
import torch

from campaign_model import constants as c
from campaign_model import features as ft
from campaign_model.model import with_added
from campaign_model.model_v2 import CampaignModelV2
from campaign_model.tests.test_model_v2 import rich_inputs, tensors


def test_a_tiles_value_as_gold_a_day():
	"""500 gold at 300 points of a 1,500-point day: 2,500 a day; at 1,500 points: 500; two days away: 250; a step under
	MIN_MOVE_COST: priced at it; never reached: 0."""
	value = torch.full((1, 1, 5), 500.0)
	h_move, h_max = torch.tensor([1500]), torch.tensor([1500])
	left = torch.tensor([[[1200, 0, -1, 1450, -1]]])
	turns = torch.tensor([[[0, 0, 2, 0, -1]]])
	rate = ft.move_rate(value, h_move, h_max, left, turns)
	assert torch.allclose(rate, torch.tensor([[[2500.0, 500.0, 250.0, 500.0 * 1500 / ft.MIN_MOVE_COST, 0.0]]]))
	views = ft.rate_views(rate, torch.tensor([[[True, True, True, True, False]]]))
	assert views.shape == (1, 1, 5, 2)
	assert torch.allclose(views[..., 0], torch.tanh(ft.value_logit(rate) / 4))
	assert torch.allclose(views[0, 0, :, 1], torch.tensor([2500 / 7500, 500 / 7500, 250 / 7500, 1.0, 0.0]))
	# no legal tile with a positive rate: the share is 0 everywhere
	assert float(ft.rate_views(-rate, torch.ones_like(rate, dtype=torch.bool))[..., 1].abs().max()) == 0.0


def test_an_old_checkpoint_plays_bit_for_bit_and_the_pair_reads_the_views():
	"""A state dict without pair_rate loads the loading model's, at 0 (zero_output; with_added): every output as the model's with the layer at 0, on
	three seeds; and with the layer's weights not 0 the MOVE pair moves with the views."""
	torch.manual_seed(3)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	with torch.no_grad():
		for p in model.parameters():
			p.add_(0.05 * torch.randn_like(p))
		model.pair_rate.weight.zero_()
		model.pair_rate.bias.zero_()
	old = {k: v for k, v in model.state_dict().items() if not k.startswith("pair_rate")}
	fresh = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	assert float(fresh.pair_rate.weight.abs().max()) == 0.0  # (zero_output: a model starts with the layer at 0)
	fresh.load_state_dict(with_added(old, fresh.state_dict()))  # (ADDED_HEADS: a key the checkpoint lacks takes the loading model's)
	assert float(fresh.pair_rate.weight.abs().max()) == 0.0 and float(fresh.pair_rate.bias.abs().max()) == 0.0
	for seed in (1, 2, 3):
		x = tensors(rich_inputs(4, seed))
		with torch.no_grad():
			a, b = model(x), fresh(x)
		for name in a:
			if torch.is_tensor(a[name]) and a[name].is_floating_point():
				assert torch.equal(torch.nan_to_num(a[name]), torch.nan_to_num(b[name])), (seed, name)
	# the views reach the MOVE pair through the layer: with its weights not 0, other views make other pairs (the test
	# inputs hold few legal tiles with a value, so a loss's gradient there says little)
	captured = {}
	plain = fresh.pairs

	def capture(state, known):
		captured["args"] = (state, known)
		return plain(state, known)

	fresh.pairs = capture
	with torch.no_grad():
		fresh(tensors(rich_inputs(4, 5)))
		state, known = captured["args"]
		fresh.pair_rate.weight.fill_(0.1)
		before = plain(state, known)
		known.move_rate = torch.rand_like(known.move_rate)
		after = plain(state, known)
	assert not torch.allclose(before, after)
