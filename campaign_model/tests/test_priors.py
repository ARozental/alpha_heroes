import pytest
import torch

from campaign_model import constants as c
from campaign_model import priors
from campaign_model.model import CampaignModel
from campaign_model.schema import random_inputs


def test_the_priors_file_names_every_action_type_once():
	rows = priors.read("action_types.csv")
	assert sorted(rows) == [int(a) for a in c.ActionType]
	assert all(rows[int(a)][0] == a.name for a in c.ActionType)


def no_habits(x):
	"""The inputs with none of Nullkiller2's habits at work: no upgrade or spellbook to buy, a City Hall (no saving)."""
	x = {k: v.copy() for k, v in x.items()}
	x["upgrade_mask"][:] = x["buy_mask"][:] = x["hire_mask"][:] = False
	x["t_built"][:, :, 12] = True
	return x


def nothing_on_offer(x):
	"""The inputs with nothing worth taking in reach (no objects or monsters in the windows, no TRAVEL), and no habits."""
	x = no_habits(x)
	x["w_object"][:] = x["w_creature"][:] = -1
	x["travel_mask"][:] = False
	return x


def test_untrained_types_follow_the_priors_when_nothing_is_on_offer_and_the_bias_is_learned():
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	x = {k: torch.from_numpy(v) for k, v in nothing_on_offer(random_inputs(8, seed=2)).items()}
	with torch.no_grad():
		type_logp = model(x)["type_logp"]
	bias = priors.action_types()
	for row in type_logp:
		legal = torch.isfinite(row)
		if legal.sum() > 1:
			assert torch.allclose(row[legal], torch.log_softmax(bias[legal], dim=0), atol=1e-5)
	assert model.type_head.bias.requires_grad


def test_what_is_on_offer_moves_only_end_and_travel():
	"""The couplings hold END (and TRAVEL, for what is in reach) back, and leave the odds among the other types alone."""
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	x = random_inputs(8, seed=2)
	with torch.no_grad():
		offered, empty = (model({k: torch.from_numpy(v) for k, v in inputs.items()})["type_logp"] for inputs in (no_habits(x), nothing_on_offer(x)))
	others = [int(a) for a in c.ActionType if a not in (c.ActionType.END, c.ActionType.TRAVEL, c.ActionType.MOVE)] + [int(c.ActionType.MOVE)]
	moved = 0
	for a, b in zip(offered, empty):
		legal = torch.isfinite(a) & torch.isfinite(b)
		rest = [t for t in others if legal[t]]
		if len(rest) > 1:  # the odds among the types that are neither END nor TRAVEL are the same
			assert torch.allclose(a[rest] - a[rest].logsumexp(0), b[rest] - b[rest].logsumexp(0), atol=1e-5)
		if legal[c.ActionType.END] and legal.sum() > 1:
			moved += float(a[c.ActionType.END]) < float(b[c.ActionType.END]) - 1e-3
	assert moved > 0



def test_nullkiller2s_habits_move_upgrade_buy_and_recruit():
	"""An upgrade that makes a stack stronger lifts UPGRADE by its bias (1.5); a spellbook a main hero lacks lifts BUY
	(2); short of gold before any City Hall, with nothing threatening, RECRUIT drops (by 2 x the pause)."""
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	x = nothing_on_offer(random_inputs(1, seed=4))
	x["g_kind"][0] = int(c.DecisionKind.TURN)
	x["h_army_creature"][0, 0, 0], x["h_army_count"][0, 0, 0] = 0, 10  # ten pikemen (halberdiers are stronger)
	x["upgrade_mask"][0, 0, 0] = True
	x["recruit_mask"][0, 0, 0] = True
	tensor = lambda x: {k: torch.from_numpy(v) for k, v in x.items()}  # noqa: E731
	with torch.no_grad():
		base = model(tensor(x))["type_logp"][0]
	bias = priors.action_types()
	assert torch.isclose(base[c.ActionType.UPGRADE] - base[c.ActionType.RECRUIT], bias[c.ActionType.UPGRADE] - bias[c.ActionType.RECRUIT] + 1.5, atol=1e-4)
	x["t_built"][:, :, 12] = False  # no City Hall, and short of gold
	x["g_resources"][0] = 0
	x["t_available"][0, 0, 0], x["t_available_creature"][0, 0, 0] = 50, 0
	f = model.features(tensor(x), torch.zeros(0, dtype=torch.long))
	assert float(f.recruit_pause[0]) > 0.99
	with torch.no_grad():
		paused = model(tensor(x))["type_logp"][0]
	assert float((paused[c.ActionType.RECRUIT] - paused[c.ActionType.UPGRADE]) - (base[c.ActionType.RECRUIT] - base[c.ActionType.UPGRADE])) == pytest.approx(-2.0, abs=0.05)
