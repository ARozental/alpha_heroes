"""Deploy 85 (the user via e5, 2026-11-16: "bugs should be fixed right away"): GARRISON_SWAP's OUT rows (our garrisoned hero
with movement left, no rival at least as strong within DANGER_RANGE of the town) take a start ramped from the learned
multi_garrison to config.garrison_out, by version, through two buffers; off (weight 0) the prior is exactly as before; IN
and a threatened OUT keep the learned start; the actor's ONNX agrees with the trainer's forward on such a row along the ramp."""
import pytest
import torch

from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.learner import garrison_now, set_advice_prior

HO, TO = c.OWN_HEROES, c.OWN_TOWNS
GS = c.MultiKind.GARRISON_SWAP


def inputs(n=3, seed=21):
	from campaign_model.tests.test_model_v2 import rich_inputs, tensors

	x = tensors(rich_inputs(n, seed))
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
	x["t_garrison_hero"][:, 0] = 0  # (town 0 holds our hero 0: an OUT row unless a rival is near)
	x["h_present"][:, HO:] = False
	mask = torch.zeros(n, TO, c.N_MULTI, dtype=torch.bool)
	mask[:, 0, GS] = True
	mask[:, 1, c.MultiKind.RECRUIT_ALL] = True
	x["multi_mask"] = mask.reshape(x["multi_mask"].shape).to(x["multi_mask"].dtype)
	return x


def test_the_ramp_by_version():
	assert garrison_now(CampaignConfig(), 5000) == (0.0, 0.0)
	on = CampaignConfig(garrison_out_on=True, garrison_out=0.0, garrison_out_from=100, garrison_out_ramp=20)
	assert garrison_now(on, 99)[0] == 0.0 and garrison_now(on, 100)[0] == pytest.approx(0.05) and garrison_now(on, 109)[0] == pytest.approx(0.5) and garrison_now(on, 400)[0] == 1.0
	assert garrison_now(on, None)[0] == 0.0


def test_set_advice_prior_is_the_one_source():
	from campaign_model.model_v2 import CampaignModelV2

	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	on = CampaignConfig(garrison_out_on=True, garrison_out=0.5, garrison_out_from=100, garrison_out_ramp=20)
	set_advice_prior(model, on, 109)
	assert float(model.features.garrison_out_weight) == pytest.approx(0.5) and float(model.features.garrison_out_target) == 0.5
	set_advice_prior(model, CampaignConfig(), 109)
	assert float(model.features.garrison_out_weight) == 0.0


def test_off_is_the_learned_start_and_on_moves_only_the_out_rows():
	from campaign_model.model_v2 import CampaignModelV2

	torch.manual_seed(0)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = inputs()
	with torch.no_grad():
		known = model.domain_knowledge(dict(x))
		off = known.multi_prior.clone()
		start = float(model.features.multi_priors_16()[2])
		assert float(off[0, 0, GS]) == pytest.approx(start + float(off[0, 0, GS] - start))  # (as built)
		model.features.garrison_out_weight.fill_(1.0)
		model.features.garrison_out_target.fill_(0.0)
		on = model.domain_knowledge(dict(x)).multi_prior
	assert float(on[0, 0, GS] - off[0, 0, GS]) == pytest.approx(-start, abs=1e-4)  # (the OUT row's start -4 -> 0)
	other = torch.ones_like(off, dtype=torch.bool)
	other[:, 0, GS] = False
	assert torch.equal(on[other], off[other])


@pytest.mark.parametrize("weight", [0.0, 0.05, 1.0])
def test_the_actors_onnx_agrees_with_the_trainer_on_an_out_row(tmp_path, weight):
	import numpy as np
	import onnxruntime as ort

	from campaign_model.export import export
	from campaign_model.model_v2 import CampaignModelV2

	torch.manual_seed(2)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	with torch.no_grad():
		for p in model.parameters():
			if float(p.abs().sum()) == 0:
				p.normal_(0, 0.05)
		model.features.garrison_out_weight.fill_(weight)
		model.features.garrison_out_target.fill_(0.0)
	path = export(model, tmp_path / "v2.onnx")
	x = inputs(2, 7)
	with torch.no_grad():
		ours = model(dict(x))["multi_logp"].numpy()
	session = ort.InferenceSession(str(path))
	feed = {i.name: x[i.name].numpy() for i in session.get_inputs() if i.name in x}
	names = [o.name for o in session.get_outputs()]
	theirs = session.run(None, feed)[names.index("multi_logp")]
	finite = np.isfinite(ours)
	assert np.array_equal(finite, np.isfinite(theirs)) and np.allclose(ours[finite], theirs[finite], atol=1e-4)
