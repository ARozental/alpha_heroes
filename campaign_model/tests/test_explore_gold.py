"""ALPHA_EXPLORE_GOLD (e5, 2026-11-16: X, an eval arm's switch, never in training): MOVE's explore term prices a full new
sight at the switch's gold; TRAVEL keeps explore_gold(); unset, the model never takes the branch."""
import torch


def test_the_override_is_read_at_construction_and_unset_is_none(monkeypatch):
	from campaign_model.model import CustomFeatures

	monkeypatch.delenv("ALPHA_EXPLORE_GOLD", raising=False)
	assert CustomFeatures().explore_override is None
	monkeypatch.setenv("ALPHA_EXPLORE_GOLD", "2100")
	assert CustomFeatures().explore_override == 2100.0


def test_only_move_and_the_types_move(monkeypatch):
	"""A small v2 model on test_model_v2's inputs: with the switch, TRAVEL's table is unchanged."""
	from campaign_model.model_v2 import CampaignModelV2
	from campaign_model.tests.test_model_v2 import rich_inputs, tensors

	x = tensors(rich_inputs(3, 41))
	x["g_kind"] = torch.zeros_like(x["g_kind"])
	from campaign_model import constants as c

	x["w_flags"] = x["w_flags"] | c.TILE_ON_MAP
	x["w_flags"][..., : c.WINDOW // 2, :] = x["w_flags"][..., : c.WINDOW // 2, :] & ~c.TILE_REVEALED  # (half of every window unrevealed: explore > 0 next to it)
	mm = torch.zeros_like(x["move_mask"]).reshape(3, c.OWN_HEROES, c.WINDOW, c.WINDOW)
	mm[:, 0, c.WINDOW // 2, :] = True  # (the row next to the hidden half: legal MOVEs that reveal it)
	x["move_mask"] = mm.reshape(x["move_mask"].shape).to(x["move_mask"].dtype)
	x["w_turns"][:, 0, c.WINDOW // 2, :] = 0
	outs = {}
	for mode in (None, "5000"):
		if mode is None:
			monkeypatch.delenv("ALPHA_EXPLORE_GOLD", raising=False)
		else:
			monkeypatch.setenv("ALPHA_EXPLORE_GOLD", mode)
		torch.manual_seed(0)
		model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
		with torch.no_grad():
			outs[mode] = model(dict(x))
	assert torch.equal(outs["5000"]["travel_logp"], outs[None]["travel_logp"])  # (TRAVEL keeps explore_gold(); that MOVE moves is the recorded decisions' gate, fl/gate1_x.sh: random inputs carry no windows to explore from)
