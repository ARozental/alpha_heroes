import torch

from battle_model.codegen import HEADER_PATH, render
from battle_model.export import export, verify
from battle_model.model import BattleModel


def test_generated_header_is_up_to_date():
	assert HEADER_PATH.read_text() == render(), "run: .venv/bin/python -m battle_model.codegen"


def test_onnx_export_matches_torch_and_contract(tmp_path):
	torch.manual_seed(0)
	model = BattleModel(dim=32, layers=1)
	path = export(model, tmp_path / "model.onnx")
	assert verify(model, path) < 1e-4


def test_any_architecture_exports_to_the_same_contract(tmp_path):
	"""A different (bigger) model must expose exactly the same interface."""
	torch.manual_seed(1)
	model = BattleModel(dim=96, layers=3, heads=6)
	assert verify(model, export(model, tmp_path / "big.onnx")) < 1e-4
