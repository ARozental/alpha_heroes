"""ALPHA_GARRISON_OUT (e5, 2026-11-16: G, an eval arm's switch, never in training): GARRISON_SWAP's start split by
direction; the OUT rows (a garrisoned hero of ours with movement left, no rival at least as strong within DANGER_RANGE of
the town) take the switch's logit, every other MULTI entry exactly as it was; unset, the model never takes the branch."""
import torch

from campaign_model import constants as c
from campaign_model import features as ft

HO, TO = c.OWN_HEROES, c.OWN_TOWNS
GS = c.MultiKind.GARRISON_SWAP


def inputs():
	from campaign_model.tests.test_model_v2 import rich_inputs, tensors

	x = tensors(rich_inputs(3, 21))
	x["g_kind"] = torch.zeros_like(x["g_kind"])
	x["h_present"][:, :2] = True
	x["h_owner"][:, :2] = c.OWNER_SELF
	x["h_garrisoned"][:, :HO] = 0
	x["h_garrisoned"][:, 0] = 1
	x["h_move"][:, 0] = 1500
	x["t_present"][:, 0] = True
	x["t_owner"][:, 0] = c.OWNER_SELF
	x["t_pos"][:, 0] = torch.tensor([10, 10, 0])
	x["t_garrison_hero"][:, :] = -1
	x["t_garrison_hero"][:, 0] = 0  # (row 0's town holds our hero 0)
	x["h_present"][:, HO:] = False
	x["h_present"][1, HO] = True  # (row 1: a rival within the range)
	x["h_owner"][1, HO] = c.OWNER_SELF + 1
	x["h_pos"][1, HO] = torch.tensor([14, 12, 0])
	x["h_move"][2, 0] = 0  # (row 2: no movement left)
	return x


def test_the_out_rows():
	from campaign_model.model import CustomFeatures

	t = CustomFeatures()

	class F:
		hero_strength = torch.zeros(3, c.N_HEROES if hasattr(c, "N_HEROES") else 16)

	x = inputs()
	F.hero_strength = torch.zeros(3, x["h_present"].shape[1])
	rows = t.garrison_out_rows(x, F)
	assert rows.shape == (3, TO)
	assert bool(rows[0, 0]) and not bool(rows[1, 0]) and not bool(rows[2, 0])  # (free / a rival near at 4 tiles / no movement)
	assert not bool(rows[:, 1:].any())  # (no garrison hero there)
	F.hero_strength[1, :] = 0.0
	F.hero_strength[1, 0] = 5.0  # (the rival weaker than ours: OUT again)
	assert bool(t.garrison_out_rows(x, F)[1, 0])


def test_the_override_is_read_at_construction_and_unset_is_none(monkeypatch):
	from campaign_model.model import CustomFeatures

	monkeypatch.delenv("ALPHA_GARRISON_OUT", raising=False)
	assert CustomFeatures().garrison_out is None
	monkeypatch.setenv("ALPHA_GARRISON_OUT", "0.5")
	assert CustomFeatures().garrison_out == 0.5


def test_the_model_moves_only_the_out_swap(monkeypatch):
	from campaign_model.model_v2 import CampaignModelV2

	x = inputs()
	mask = torch.zeros_like(x["multi_mask"]).reshape(3, TO, c.N_MULTI)
	mask[:, 0, GS] = True
	mask[:, 1, c.MultiKind.RECRUIT_ALL] = True
	x["multi_mask"] = mask.reshape(x["multi_mask"].shape).to(x["multi_mask"].dtype)
	x["t_garrison_hero"][1, 0] = -1  # (row 1: no garrison hero; row 2: no movement left; the rival rule is test_the_out_rows')
	outs = {}
	for mode in (None, "1.0"):
		if mode is None:
			monkeypatch.delenv("ALPHA_GARRISON_OUT", raising=False)
		else:
			monkeypatch.setenv("ALPHA_GARRISON_OUT", mode)
		torch.manual_seed(0)
		model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
		with torch.no_grad():
			outs[mode] = model(dict(x))
	a, b = outs[None]["multi_logp"].reshape(3, TO, c.N_MULTI), outs["1.0"]["multi_logp"].reshape(3, TO, c.N_MULTI)
	assert b[0, 0, GS] > a[0, 0, GS]  # (row 0: OUT, lifted from -4.4 to 1.0)
	assert torch.equal(outs["1.0"]["multi_logp"][1:], outs[None]["multi_logp"][1:])  # (rows 1-2: no garrison hero / no movement: not OUT, as they were)
