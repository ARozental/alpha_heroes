"""ALPHA_TRAVEL_LIFT (e5, 2026-11-16: T, an eval arm's switch, never in training): TRAVEL's learned table is 0 for every
hero, so the prior decides; unset, the model never takes the branch."""
import torch

from campaign_model import constants as c


def test_the_override_is_read_at_construction_and_unset_is_none(monkeypatch):
	from campaign_model.model import CustomFeatures

	monkeypatch.delenv("ALPHA_TRAVEL_LIFT", raising=False)
	assert CustomFeatures().travel_lift is None
	monkeypatch.setenv("ALPHA_TRAVEL_LIFT", "1")
	assert CustomFeatures().travel_lift == 1.0


def test_travel_follows_its_prior_alone(monkeypatch):
	"""On a small v2 model with its zero-started parts moved: with the switch, TRAVEL's table is its prior's softmax (the
	learned part 0); without it, not; every output but travel_logp and type_logp unchanged."""
	from campaign_model.model_v2 import CampaignModelV2
	from campaign_model.tests.test_model_v2 import rich_inputs, tensors

	x = tensors(rich_inputs(3, 31))
	x["g_kind"] = torch.zeros_like(x["g_kind"])
	outs, caps = {}, {}
	for mode in (None, "1"):
		if mode is None:
			monkeypatch.delenv("ALPHA_TRAVEL_LIFT", raising=False)
		else:
			monkeypatch.setenv("ALPHA_TRAVEL_LIFT", mode)
		torch.manual_seed(0)
		model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
		with torch.no_grad():
			g = torch.Generator().manual_seed(1)
			for param in model.parameters():
				if not bool(param.any()):
					param.add_(0.05 * torch.randn(param.shape, generator=g))
			outs[mode] = model(dict(x))
	for name, value in outs[None].items():
		if name in ("travel_logp", "type_logp") or not torch.is_tensor(value):
			continue
		assert torch.equal(outs["1"][name], value), name
	assert not torch.equal(outs["1"]["travel_logp"], outs[None]["travel_logp"])
