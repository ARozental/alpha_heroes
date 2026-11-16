"""ALPHA_RECRUIT_LIFT (e5, 2026-11-16: an eval arm's switch, never in training): model_v2.lift_recruits gives each legal
visiting-hero RECRUIT target its entry's garrison target's learned part, every other entry exactly as it was; unset, the
model never takes the branch."""
import torch

from campaign_model import constants as c
from campaign_model.model_v2 import lift_recruits


def test_a_legal_hero_target_gets_the_garrison_part_and_nothing_else_moves():
	torch.manual_seed(0)
	shape = (2, 3, c.N_LEVELS, c.N_RECRUIT_AMOUNTS, c.N_RECRUIT_TARGETS)
	learned = torch.randn(shape) * 10
	mask = torch.rand(shape) > 0.4
	out = lift_recruits(learned, mask)
	hero_legal = mask[..., 1]
	assert torch.equal(out[..., 1][hero_legal], learned[..., 0][hero_legal])
	assert torch.equal(out[..., 1][~hero_legal], learned[..., 1][~hero_legal])
	assert torch.equal(out[..., 0], learned[..., 0])


def test_the_override_is_read_at_construction_and_unset_is_none(monkeypatch):
	from campaign_model.model import CustomFeatures

	monkeypatch.delenv("ALPHA_RECRUIT_LIFT", raising=False)
	assert CustomFeatures().recruit_lift is None
	monkeypatch.setenv("ALPHA_RECRUIT_LIFT", "1")
	assert CustomFeatures().recruit_lift == 1.0
