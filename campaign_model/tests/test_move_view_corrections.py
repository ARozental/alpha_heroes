"""The MOVE views as bounded corrections (e5 for the user, 2026-11-15; features.rate_correction): each of the four
gain-per-movement and IRR views times its own weight (-2..2, 0 to begin with, the corrections' rate), into MOVE's tile
logits beside pair_rate; the learner logs the weights each update (correction_values)."""
import math

import pytest
import torch

from campaign_model import features as ft
from campaign_model.model import CORRECTION_STEP, CampaignModel


def test_zero_to_begin_with_and_each_view_at_its_weight():
	f = CampaignModel(dim=32, cell_dim=16, layers=1).eval().features
	assert any(p is f.rate_weights for p in f.corrections())
	views = torch.rand(3, 5, 5, 4)
	with torch.no_grad():
		assert float(ft.rate_correction(views, f).abs().max()) == 0.0
		raw = lambda v: math.log(((v + 2) / 4) / (1 - (v + 2) / 4)) / CORRECTION_STEP  # noqa: E731
		f.rate_weights.copy_(torch.tensor([raw(v) for v in (1.0, -1.0, 0.5, 0.0)]))
		assert torch.allclose(ft.rate_correction(views, f), views[..., 0] - views[..., 1] + 0.5 * views[..., 2], atol=1e-5)
	assert list(f.correction_values()) == [f"r{k}_move_view" for k in range(1, 5)]
	assert f.correction_values()["r1_move_view"] == pytest.approx(1.0, abs=1e-5)
