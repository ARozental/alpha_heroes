"""The Φ settings' consumers on the run's real path (e5, 2026-11-15, deploy-safety: every consumer). A learner resumed
as the unit starts it (CampaignLearner, resume, the learner "turns"), at phi_from + 5, the run's line carrying the phi_
targets, the stock floor and campaign.unit_own_form, and no ALPHA_EMPIRE_* in the environment (the PC's) or wrong ones:
(a) the model the update's forward runs on, (b) the weights the inference server gets (the remote workers' come from
it: inference.set_model sends the whole state_dict), (c) the labeler's and the learner's EmpireConfig and the labeler's
features, (d) the checkpoint written, all read the config's ramp at the version; set_advice_prior after a load (the
update process's path) too. The checkpoint resumed from is deploy 81's: none of the new buffers."""
from types import SimpleNamespace

import pytest
import torch

from campaign_model.config import CampaignConfig
from campaign_model.learner import PHI_FIELDS, CampaignLearner, phi_now, set_advice_prior
from campaign_model.model import build_model
from campaign_model.train import make_optimizer, model_kwargs

FROM, RESUMED = 3480, 3485
NEW_BUFFERS = ("features.phi_hero_base", "features.phi_armyless_share", "features.phi_lead_weight", "features.phi_stock_floor", "features.empire_unit_own", "features.give_all_exact")
ENVS = {
	"none": {},
	"wrong": {"ALPHA_EMPIRE_HERO_BASE": "2000", "ALPHA_EMPIRE_ARMYLESS": "0.3", "ALPHA_EMPIRE_LEAD_WEIGHT": "0.9", "ALPHA_EMPIRE_UNIT_OWN": "0", "ALPHA_EMPIRE_STOCK_FLOOR": "0.9"},
}


def run_config():
	return CampaignConfig(
		learner="turns", device="cpu", model_architecture="v2", model_dim=32, model_cell_dim=8, model_layers=1,
		phi_hero_base=825.0, phi_armyless_share=1.0, phi_lead_weight=0.5, phi_stock_floor=1.0 / 3.0, phi_from=FROM, phi_ramp=20,
		unit_own_form=True, give_all_exact=True,
	)


def expected(version):
	return {**phi_now(run_config(), version), "unit_own_form": True}


def read(state):
	"""{EmpireConfig field: value} from a state_dict's buffers (features.)."""
	out = {field_: float(state[f"features.{name}"]) for name, field_, _ in PHI_FIELDS}
	out["unit_own_form"] = bool(float(state["features.empire_unit_own"]) > 0)
	return out


def of_empire(empire):
	return {**{field_: float(getattr(empire, field_)) for _, field_, _ in PHI_FIELDS}, "unit_own_form": bool(empire.unit_own_form)}


@pytest.fixture
def resumed(tmp_path, monkeypatch, request):
	for name in ENVS["wrong"]:
		monkeypatch.delenv(name, raising=False)
	for name, value in ENVS[request.param].items():
		monkeypatch.setenv(name, value)
	config = run_config()
	model = build_model(model_kwargs(config))
	state = {key: value for key, value in model.state_dict().items() if key not in NEW_BUFFERS}  # (deploy 81's checkpoint)
	(tmp_path / "campaign").mkdir()
	torch.save({"model": state, "optimizer": make_optimizer(model, config).state_dict(), "version": RESUMED, "model_kwargs": model_kwargs(config)}, tmp_path / "campaign" / "checkpoint_latest.pt")
	served = []
	server = SimpleNamespace(set_model=lambda name, version, m: served.append((name, version, {k: v.detach().clone() for k, v in m.state_dict().items() if k.startswith("features.")})), socket_path="/nonexistent", close=lambda: None, stop=lambda: None)
	learner = CampaignLearner(config, tmp_path, torch.device("cpu"), battle=SimpleNamespace(config=SimpleNamespace(remote_port=0)), resume=True, server=server)
	try:
		yield learner, served, tmp_path
	finally:
		learner.labeler.close()
		learner.writer.close()


@pytest.mark.parametrize("resumed", list(ENVS), indirect=True)
def test_every_consumer_reads_the_configs_ramp_at_the_resume_and_after_an_update(resumed):
	learner, served, tmp_path = resumed
	want = expected(RESUMED)
	assert want["hero_base"] == pytest.approx(1500 + (825 - 1500) * 6 / 20) and want["stock_floor"] == pytest.approx(1 - (2 / 3) * 6 / 20)
	assert learner.version == RESUMED
	assert read(learner.model.state_dict()) == pytest.approx(want)  # (a) the model the update's forward runs on
	assert served and served[-1][1] == RESUMED and read({k: v for k, v in served[-1][2].items()}) == pytest.approx(want)  # (b) the inference server's (the workers')
	assert of_empire(learner.empire) == pytest.approx(want) and of_empire(learner.labeler.empire) == pytest.approx(want)  # (c) the labels'
	assert read({f"features.{k}": v for k, v in learner.features_cpu.state_dict().items()}) == pytest.approx(want)  # (c) the labeler's features (foresight)
	assert float(learner.model.features.give_all_exact) == 1.0
	# An update: step's ramp at the version, the update, the version + 1 served and saved (the ramp's next at the next step)
	CampaignLearner._ramp_phi(learner)
	CampaignLearner._unit_rule(learner)
	learner.version += 1
	learner.server.set_model("current", learner.version, learner.model)
	learner._save(tmp_path / "campaign" / "checkpoint_latest.pt")
	saved = torch.load(tmp_path / "campaign" / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
	assert read(saved["model"]) == pytest.approx(want) and read(served[-1][2]) == pytest.approx(want)  # (d) the checkpoint written
	CampaignLearner._ramp_phi(learner)
	CampaignLearner._unit_rule(learner)
	later = expected(RESUMED + 1)
	assert read(learner.model.state_dict()) == pytest.approx(later) and of_empire(learner.empire) == pytest.approx(later)
	assert read({f"features.{k}": v for k, v in learner.features_cpu.state_dict().items()}) == pytest.approx(later)
	# The update process's path: a model loaded from the checkpoint, then set_advice_prior with its version
	model = build_model(saved["model_kwargs"])
	model.load_state_dict(saved["model"])
	set_advice_prior(model, learner.config, saved["version"])
	assert read(model.state_dict()) == pytest.approx(later)


def test_set_advice_prior_without_targets_keeps_the_environments_for_a_paired_test(monkeypatch):
	"""A line without the phi_ targets or unit_own_form (deploy 81's, the Mac arms'): the environment's, as before."""
	for name, value in ENVS["wrong"].items():
		monkeypatch.setenv(name, value)
	monkeypatch.setenv("ALPHA_EMPIRE_UNIT_OWN", "1")
	model = build_model({"dim": 32, "cell_dim": 8, "heads": 4, "layers": 1, "architecture": "v2"})
	set_advice_prior(model, CampaignConfig(), 3485)
	assert read(model.state_dict()) == pytest.approx({"hero_base": 2000.0, "armyless_share": 0.3, "lead_weight": 0.9, "stock_floor": 0.9, "unit_own_form": True})
