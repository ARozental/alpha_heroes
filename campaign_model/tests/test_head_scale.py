"""config.head_scale: each value head's loss weighted inversely to its targets' variance, the heads' total at a constant
prediction kept, ramped in from the old weights (train.head_weights); micro_loss at the old weights is the old loss."""

import pytest
import torch

from campaign_model import constants as c
from campaign_model.train import head_weights, masked_variance

VAR = {"share": [0.0001, 0.0002, 0.0003], "growth": [0.0005, 0.0005, 0.0006], "end": [0.0094]}


def baseline(weights, var=VAR):
	return sum(w * v for name in var for w, v in zip(weights[name], var[name]))


def test_the_heads_get_equal_shares_and_the_total_is_kept():
	old = head_weights(VAR, 0.0, 0.5)
	assert old == {"share": [1.0] * 3, "growth": [0.5] * 3, "end": [1.0]}  # ramp 0: the old weights
	new = head_weights(VAR, 1.0, 0.5)
	assert baseline(new) == pytest.approx(baseline(old))  # the value loss keeps its size
	shares = {name: [w * v for w, v in zip(new[name], VAR[name])] for name in VAR}
	assert shares["share"][2] > 4 * VAR["share"][2] and new["end"][0] < 0.5  # the share heads up, V_end down
	assert shares["share"][2] / shares["end"][0] == pytest.approx((0.0003 / 0.0004) / (0.0094 / 0.0095), rel=1e-6)  # (eps keeps tiny ones below)
	half = head_weights(VAR, 0.5, 0.5)
	assert half["end"][0] == pytest.approx((old["end"][0] + new["end"][0]) / 2)


def test_a_missing_or_flat_head_keeps_its_place():
	w = head_weights({"share": [0.0, 0.0, 0.0]}, 1.0, 0.5)
	assert all(v == pytest.approx(0.0) or v > 0 for v in w["share"]) and w["growth"] == [0.5] * 3 and w["end"] == [1.0]
	assert head_weights({}, 1.0, 0.5) == {"share": [1.0] * 3, "growth": [0.5] * 3, "end": [1.0]}


def test_masked_variance():
	t = torch.tensor([[0.0, 1.0], [2.0, 5.0], [4.0, 9.0]])
	valid = torch.tensor([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
	assert torch.allclose(masked_variance(t, valid), torch.tensor([8 / 3, 16.0]))


def test_an_update_at_the_old_weights_is_the_old_update_and_a_scaled_one_runs():
	"""ppo_update with head_scale at ramp 0 (the old weights, as tensors in each micro-batch) moves the model exactly as
	without it; fully ramped it runs and returns the running variances and the weights it used."""
	import copy

	import numpy as np

	from campaign_model.config import CampaignConfig
	from campaign_model.model import build_model
	from campaign_model.tests.test_model import random_inputs
	from campaign_model.tests.test_train import decisions_for
	from campaign_model.train import ppo_update

	torch.manual_seed(3)
	model = build_model({"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"})
	x = random_inputs(8, seed=11)
	decisions = decisions_for(model.eval(), x)
	rng = np.random.default_rng(0)
	targets = (0.5 + 0.02 * rng.standard_normal((8, c.N_VALUE_OFFSETS))).astype(np.float32)
	growth = (10.0 + rng.standard_normal((8, c.N_VALUE_OFFSETS))).astype(np.float32)
	config = CampaignConfig(batch_size=4, passes=1, credit="schedule", share_target="change")
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731

	def run(head_scale):
		torch.manual_seed(5)
		m = copy.deepcopy(model)
		metrics, _ = ppo_update(m, torch.optim.Adam(m.parameters(), lr=1e-4), decisions, targets, np.zeros(8, dtype=np.float32), config, decode=decode, growth=growth, growth_real=growth, head_scale=head_scale)
		return m, metrics

	plain, _ = run(None)
	same, metrics = run({"var": None, "ramp": 0.0})
	for (name, a), (_, b) in zip(plain.state_dict().items(), same.state_dict().items()):
		assert torch.equal(a, b), name
	assert metrics["head_weights"]["share"] == [1.0] * c.N_VALUE_OFFSETS and metrics["head_var"]["share"][0] > 0
	_, scaled = run({"var": metrics["head_var"], "ramp": 1.0})
	assert np.isfinite(scaled["kl"]) and scaled["head_weights"]["share"][0] > 1.0
