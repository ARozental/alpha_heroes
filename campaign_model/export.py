"""Export the campaign model to ONNX and check it against the torch model.

	.venv/bin/python -m campaign_model.export [--checkpoint path] [--out path]

Training games ask the inference server (battle_model/inference.py, family "campaign"); the ONNX file is the
finished product, and the check that every in-graph feature exports.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from .constants import SCHEMA_VERSION
from .model import INPUT_NAMES, OUTPUT_NAMES, build_model
from .schema import contract_hash, random_inputs

DEFAULT_PATH = Path(__file__).resolve().parent / "artifacts" / "campaign_model.onnx"


class OnnxWrapper(nn.Module):
	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, *tensors):
		outputs = self.model(dict(zip(INPUT_NAMES, tensors)))
		return tuple(outputs[name] for name in OUTPUT_NAMES)


def export(model, path=DEFAULT_PATH, opset=17):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	model = model.eval()
	sample = random_inputs(batch=2, seed=0)
	args = tuple(torch.from_numpy(sample[name]) for name in INPUT_NAMES)
	torch.onnx.export(
		OnnxWrapper(model),
		args,
		str(path),
		input_names=list(INPUT_NAMES),
		output_names=list(OUTPUT_NAMES),
		dynamic_axes={name: {0: "batch"} for name in INPUT_NAMES + OUTPUT_NAMES},
		opset_version=opset,
		dynamo=False,
	)
	proto = onnx.load(str(path))
	for key, value in {"schema_version": str(SCHEMA_VERSION), "contract_hash": contract_hash()}.items():
		entry = proto.metadata_props.add()
		entry.key, entry.value = key, value
	onnx.checker.check_model(proto)
	onnx.save(proto, str(path))
	return path


def verify(model, path, seeds=(1, 2), batch_sizes=(1, 3)):
	"""The largest difference between the ONNX and torch outputs (finite entries; -inf where both are -inf)."""
	session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
	names = [i.name for i in session.get_inputs()]
	worst = 0.0
	for seed in seeds:
		for batch in batch_sizes:
			sample = random_inputs(batch=batch, seed=seed)
			with torch.no_grad():
				expected = model.eval()({k: torch.from_numpy(v) for k, v in sample.items()})
			got = session.run(list(OUTPUT_NAMES), {name: sample[name] for name in names})
			for name, value in zip(OUTPUT_NAMES, got):
				want = expected[name].numpy()
				assert np.array_equal(np.isneginf(want), np.isneginf(value)), f"{name}: -inf entries differ"
				finite = np.isfinite(want)
				if finite.any():
					worst = max(worst, float(np.abs(want[finite] - value[finite]).max()))
	return worst


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--checkpoint", default="")
	parser.add_argument("--out", default=str(DEFAULT_PATH))
	args = parser.parse_args()
	model = build_model()
	if args.checkpoint:
		state = torch.load(args.checkpoint, map_location="cpu")
		model = build_model(state.get("model_kwargs", {}))
		model.load_state_dict(state["model"])
	path = export(model, args.out)
	print(f"wrote {path}; largest difference from torch {verify(model, path):.2e}")


if __name__ == "__main__":
	main()
