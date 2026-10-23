"""Export the battle model to ONNX and verify the exported file against the contract.

	.venv/bin/python -m battle_model.export [--out PATH] [--seed N] [--dim D] [--layers L]

The ONNX metadata carries schema_version and contract_hash; the C++ side refuses models
whose values differ from the contract it was built with.
"""
import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .constants import SCHEMA_VERSION
from .model import BattleModel, OnnxWrapper
from .schema import INPUT_NAMES, INPUTS, OUTPUT_NAMES, OUTPUTS, contract_hash, random_inputs

DEFAULT_PATH = Path(__file__).resolve().parent / "artifacts" / "battle_model.onnx"
ORT_TYPES = {"int64": "tensor(int64)", "bool": "tensor(bool)", "float32": "tensor(float)"}


def export(model, path, opset=17):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	model = model.eval()
	sample = random_inputs(batch=2, seed=0)
	args = tuple(torch.from_numpy(sample[name]) for name in INPUT_NAMES)
	dynamic_axes = {name: {0: "batch"} for name in INPUT_NAMES + OUTPUT_NAMES}
	torch.onnx.export(
		OnnxWrapper(model),
		args,
		str(path),
		input_names=list(INPUT_NAMES),
		output_names=list(OUTPUT_NAMES),
		dynamic_axes=dynamic_axes,
		opset_version=opset,
		dynamo=False,
	)
	proto = onnx.load(str(path))
	metadata = {"schema_version": str(SCHEMA_VERSION), "contract_hash": contract_hash(), "parameters": str(sum(p.numel() for p in model.parameters()))}
	for key, value in metadata.items():
		entry = proto.metadata_props.add()
		entry.key, entry.value = key, value
	onnx.checker.check_model(proto)
	onnx.save(proto, str(path))
	return path


def verify(model, path, seeds=(1, 2, 3), batch_sizes=(1, 4)):
	"""Check the ONNX interface against the schema and its outputs against the torch model."""
	session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
	meta = session.get_modelmeta().custom_metadata_map
	assert meta.get("schema_version") == str(SCHEMA_VERSION), meta
	assert meta.get("contract_hash") == contract_hash(), meta

	# The model may leave inputs out that it does not read (the exporter drops them; the C++ side passes only those
	# a model declares), but the ones it has come in the contract's order
	inputs = {i.name: i for i in session.get_inputs()}
	assert tuple(inputs) == tuple(name for name in INPUT_NAMES if name in inputs), f"input names/order differ: {tuple(inputs)}"
	for spec in INPUTS:
		if spec.name not in inputs:
			continue
		got = inputs[spec.name]
		assert got.type == ORT_TYPES[spec.dtype], (spec.name, got.type)
		assert tuple(got.shape[1:]) == spec.shape, (spec.name, got.shape)
	outputs = {o.name: o for o in session.get_outputs()}
	assert tuple(outputs) == OUTPUT_NAMES, f"output names/order differ: {tuple(outputs)}"
	for spec in OUTPUTS:
		assert outputs[spec.name].type == ORT_TYPES[spec.dtype], (spec.name, outputs[spec.name].type)

	model = model.eval()
	worst = 0.0
	for seed in seeds:
		for batch in batch_sizes:
			x = random_inputs(batch=batch, seed=seed)
			ort_out = dict(zip(OUTPUT_NAMES, session.run(list(OUTPUT_NAMES), {k: v for k, v in x.items() if k in inputs})))
			with torch.no_grad():
				torch_out = model({k: torch.from_numpy(v) for k, v in x.items()})
			for spec in OUTPUTS:
				a, b = ort_out[spec.name], torch_out[spec.name].numpy()
				assert a.shape == (batch,) + spec.shape, (spec.name, a.shape)
				assert np.array_equal(np.isneginf(a), np.isneginf(b)), f"{spec.name}: illegal entries differ"
				finite = np.isfinite(a)
				assert np.isfinite(a[finite]).all() and not np.isnan(a).any(), f"{spec.name}: NaN"
				worst = max(worst, float(np.abs(a[finite] - b[finite]).max(initial=0.0)))
	assert worst < 1e-4, f"ONNX and torch outputs differ by {worst}"
	return worst


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--out", default=str(DEFAULT_PATH))
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--dim", type=int, default=64)
	parser.add_argument("--layers", type=int, default=2)
	args = parser.parse_args()
	torch.manual_seed(args.seed)
	model = BattleModel(dim=args.dim, layers=args.layers)
	path = export(model, args.out)
	worst = verify(model, path)
	print(f"exported {path} ({path.stat().st_size / 1e6:.1f} MB), contract {contract_hash()}, max |onnx - torch| = {worst:.2e}")


if __name__ == "__main__":
	main()
