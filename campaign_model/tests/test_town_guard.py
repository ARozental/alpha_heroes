"""The town guard (bundle 60, config.town_guard): a hero of ours visiting a threatened town of ours pays, on each of its
MOVE and TRAVEL entries, what its leaving adds to the town's expected loss as value_logit prices it; exactly 0 unless
Nullkiller2's threat reaches the town by tomorrow, and 0 at size 0."""
from types import SimpleNamespace

import torch

from campaign_model import constants as c
from campaign_model import features as ft
from campaign_model.model_v2 import CampaignModelV2
from campaign_model.tests.test_model_v2 import rich_inputs

TO, HO = c.OWN_TOWNS, c.OWN_HEROES


def state(b=2):
	x = {"t_nk_threat": torch.zeros(b, TO, 3, dtype=torch.long), "t_visiting": torch.full((b, TO), -1, dtype=torch.long)}
	f = SimpleNamespace(left_danger=torch.zeros(b, TO), town_danger=torch.zeros(b, TO + 2), town_stakes=torch.full((b, TO + 2), 5000.0))
	return x, f


def test_the_cost_is_the_added_loss_on_the_visitor_only_where_the_threat_comes_by_tomorrow():
	torch.manual_seed(3)
	features = CampaignModelV2(dim=32, cell_dim=8, layers=1).features
	x, f = state()
	x["t_visiting"][0, 1] = 2  # (hero 2 visits town 1)
	f.left_danger[0, 1], f.town_danger[0, 1] = 0.6, 0.1
	x["t_nk_threat"][0, 1] = torch.tensor([1, 1, 1500])
	assert float(features.town_guard_cost(x, f).abs().max()) == 0.0  # (size 0: off)
	features.town_guard.fill_(1.0)
	cost = features.town_guard_cost(x, f)
	expect = float(ft.value_logit(torch.tensor(0.5 * 5000.0)))
	assert abs(float(cost[0, 2]) - expect) < 1e-4 and float(cost.abs().sum()) == float(cost[0, 2])  # (only hero 2 of decision 0)
	x["t_nk_threat"][0, 1, 1] = 2  # (the threat two days away: none)
	assert float(features.town_guard_cost(x, f).abs().max()) == 0.0
	x["t_nk_threat"][0, 1, 1] = 0
	f.left_danger[0, 1] = 0.05  # (what stays is enough: leaving adds nothing)
	assert float(features.town_guard_cost(x, f).abs().max()) == 0.0


def test_a_cost_lowers_its_heros_move_and_travel_and_nothing_else():
	torch.manual_seed(3)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = {k: torch.from_numpy(v) for k, v in rich_inputs(6, 4).items()}
	with torch.no_grad():
		base = model(x)
		planted = torch.zeros(6, HO)
		planted[:, 0] = 2.0
		model.features.town_guard_cost = lambda x, f: planted
		out = model(x)
	for name in ("move_logp", "travel_logp"):  # (within its tables: every entry of hero 0's down alike)
		b = next(r for r in range(6) if torch.isfinite(base[name][r, 0]).sum() >= 1 and torch.isfinite(base[name][r, 1:]).sum() >= 1)
		a0, a1 = base[name][b].reshape(HO, -1), out[name][b].reshape(HO, -1)
		hero0 = torch.isfinite(a0[0])
		others = torch.isfinite(a0[1:])
		shift0 = (a1[0][hero0] - a0[0][hero0])
		shift1 = (a1[1:][others] - a0[1:][others])
		assert torch.allclose(shift0, shift0[0].expand_as(shift0), atol=1e-4) and torch.allclose(shift1, shift1[0].expand_as(shift1), atol=1e-4)
		assert float(shift0[0] - shift1[0]) < -1.9 and float(shift0[0] - shift1[0]) > -2.1, (name, float(shift0[0] - shift1[0]))
	for name in ("build_logp", "recruit_logp"):  # (other tables as they were)
		assert torch.allclose(out[name], base[name], atol=1e-6, equal_nan=True), name
