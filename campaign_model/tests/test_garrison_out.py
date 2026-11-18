"""Deploy 85 (the user via e5, 2026-11-16 ~18:45: "going in with a minus and going out is not (trainable bias from 0?)", "no
real need to ramp it up, this is a bug fix"): GARRISON_SWAP's pure OUT rows (our garrisoned hero with movement left, the
gate empty) take their own learned start features.multi_garrison_out, 0 at the start; going in and a swap with a visitor at
the gate keep the learned multi_garrison; a checkpoint without the key loads it at 0; the actor's ONNX agrees with the
trainer's forward on such rows."""
import pytest
import torch

from campaign_model import constants as c

HO, TO = c.OWN_HEROES, c.OWN_TOWNS
GS = c.MultiKind.GARRISON_SWAP


def inputs(n=3, seed=21):
	"""Town 0: our hero 0 in the garrison, movement left, the gate empty (OUT). Town 1: our hero 1 at the gate, the garrison
	empty (IN). Town 2: our hero 2 in the garrison and our hero 3 at the gate (a swap). Town 3: our hero 4 in the garrison
	with no movement left (not OUT). Each town's swap legal."""
	from campaign_model.tests.test_model_v2 import rich_inputs, tensors

	x = tensors(rich_inputs(n, seed))
	x["g_kind"] = torch.zeros_like(x["g_kind"])
	x["h_present"][:, :5] = True
	x["h_owner"][:, :5] = c.OWNER_SELF
	x["h_garrisoned"][:, :HO] = 0
	x["h_garrisoned"][:, [0, 2, 4]] = 1
	x["h_move"][:, :5] = 1500
	x["h_move"][:, 4] = 0
	x["t_present"][:, :4] = True
	x["t_owner"][:, :4] = c.OWNER_SELF
	x["t_garrison_hero"][:, :] = -1
	x["t_visiting"][:, :] = -1
	x["t_garrison_hero"][:, 0] = 0
	x["t_visiting"][:, 1] = 1
	x["t_garrison_hero"][:, 2], x["t_visiting"][:, 2] = 2, 3
	x["t_garrison_hero"][:, 3] = 4
	mask = torch.zeros(n, TO, c.N_MULTI, dtype=torch.bool)
	mask[:, :4, GS] = True
	x["multi_mask"] = mask.reshape(x["multi_mask"].shape).to(x["multi_mask"].dtype)
	return x


def model():
	from campaign_model.model_v2 import CampaignModelV2

	torch.manual_seed(0)
	return CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()


def test_the_out_rows():
	rows = model().features.garrison_out_rows(inputs())
	assert rows[:, 0].all() and not rows[:, 1:].any()


def test_out_starts_at_0_and_only_the_out_rows_move():
	m = model()
	x = inputs()
	with torch.no_grad():
		learned = float(m.features.multi_priors_16()[2])
		assert float(m.features.multi_garrison_out_start()) == 0.0
		prior = m.domain_knowledge(dict(x)).multi_prior
		m.features.multi_garrison_out.fill_(-1000.0)  # (bounded: the start at its low end, -10)
		low = m.domain_knowledge(dict(x)).multi_prior
		m.features.multi_garrison_out.zero_()
		m.features.multi_garrison.fill_(1000.0)  # (the learned start at its high end, 4: the OUT rows do not follow it)
		high_in = m.domain_knowledge(dict(x)).multi_prior
	assert float(low[0, 0, GS] - prior[0, 0, GS]) == pytest.approx(-10.0, abs=1e-3)
	other = torch.ones_like(prior, dtype=torch.bool)
	other[:, 0, GS] = False
	assert torch.equal(low[other], prior[other])
	assert torch.equal(high_in[:, 0, GS], prior[:, 0, GS])
	for town in (1, 2, 3):  # (IN, a swap at the gate, no movement: the learned start, -4 at raw 0 -> 4 at the top)
		assert float(high_in[0, town, GS] - prior[0, town, GS]) == pytest.approx(4.0 - learned, abs=1e-3)


def test_a_checkpoint_without_the_key_loads_it_at_0():
	m = model()
	old = {k: v for k, v in m.state_dict().items() if k != "features.multi_garrison_out"}
	loading = model()
	with torch.no_grad():
		loading.features.multi_garrison_out.fill_(3.0)
	loading.load_state_dict(old)  # (the model's own load, strict: the ADDED path fills the key at 0, not the loading model's)
	assert float(loading.features.multi_garrison_out) == 0.0 and float(loading.features.multi_garrison_out_start()) == 0.0
	assert "features.multi_garrison_out" in loading.state_dict()


def test_the_parameter_trains():
	m = model().train()
	x = inputs()
	out = m(dict(x))
	out["multi_logp"][..., 0, GS].sum().backward()
	assert m.features.multi_garrison_out.grad is not None and float(m.features.multi_garrison_out.grad.abs()) > 0
	assert any(p is m.features.multi_garrison_out for p in m.prior_corrections())  # (the prior scalars' step and group)


@pytest.mark.parametrize("raw", [0.0, -5.0, 5.0])
def test_the_actors_onnx_agrees_with_the_trainer_on_the_out_rows(tmp_path, raw):
	import numpy as np
	import onnxruntime as ort

	from campaign_model.export import export

	m = model()
	with torch.no_grad():
		for p in m.parameters():
			if float(p.abs().sum()) == 0:
				p.normal_(0, 0.05)
		m.features.multi_garrison_out.fill_(raw)
	path = export(m, tmp_path / "v2.onnx")
	x = inputs(2, 7)
	with torch.no_grad():
		ours = m(dict(x))["multi_logp"].numpy()
	session = ort.InferenceSession(str(path))
	feed = {i.name: x[i.name].numpy() for i in session.get_inputs() if i.name in x}
	names = [o.name for o in session.get_outputs()]
	theirs = session.run(None, feed)[names.index("multi_logp")]
	finite = np.isfinite(ours)
	assert np.array_equal(finite, np.isfinite(theirs)) and np.allclose(ours[finite], theirs[finite], atol=1e-4)
