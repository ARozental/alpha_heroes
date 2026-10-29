import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model.codegen import HEADER_PATH, render
from campaign_model.export import export, verify
from campaign_model.model import OUTPUT_NAMES, CampaignModel
from campaign_model.schema import HEADS, OUTPUTS, SPECS, random_inputs


@pytest.fixture(scope="module")
def model():
	torch.manual_seed(0)
	return CampaignModel(dim=32, cell_dim=16, layers=1).eval()


def run(model, batch=4, seed=0):
	x = random_inputs(batch, seed)
	with torch.no_grad():
		return x, model({k: torch.from_numpy(v) for k, v in x.items()})


def test_generated_header_is_up_to_date():
	assert HEADER_PATH.read_text() == render(), "run: .venv/bin/python -m campaign_model.codegen"


def test_onnx_export_matches_torch(model, tmp_path):
	assert verify(model, export(model, tmp_path / "campaign.onnx")) < 1e-4


def test_outputs_have_the_contract_shapes(model):
	_, out = run(model)
	for spec in OUTPUTS:
		assert tuple(out[spec.name].shape[1:]) == spec.shape, spec.name


@pytest.mark.parametrize("seed", range(4))
def test_tables_are_distributions_over_exactly_the_legal_entries(model, seed):
	x, out = run(model, seed=seed)
	for action, (output, mask) in HEADS.items():
		legal = torch.from_numpy(x[mask]).reshape(4, -1)
		logp = out[output].reshape(4, -1)
		for b in range(4):
			if not legal[b].any():
				continue
			assert torch.isfinite(logp[b][legal[b]]).all() and torch.isneginf(logp[b][~legal[b]]).all(), output
			if output != "cast_cell_logp":
				assert torch.isclose(logp[b][legal[b]].exp().sum(), torch.tensor(1.0), atol=1e-5), output
	kinds = x["g_kind"]
	for b in range(4):
		legal_types = torch.isfinite(out["type_logp"][b])
		turn = kinds[b] in (c.DecisionKind.TURN, c.DecisionKind.EXCHANGE)
		assert bool(legal_types[c.ActionType.END]) == turn
		for action, (_, mask) in HEADS.items():
			assert bool(legal_types[action]) == bool(x[mask][b].any()), action


def test_values_are_shares(model):
	_, out = run(model)
	assert ((out["value"] > 0) & (out["value"] < 1)).all()
	assert set(OUTPUT_NAMES) == {s.name for s in OUTPUTS} and "move_mask" in SPECS
