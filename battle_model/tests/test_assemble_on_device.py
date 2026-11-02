"""supply.assemble with device: the sparse inputs made dense on the device give exactly the batch the CPU made."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from battle_model.data import Sparse
from battle_model.losses import AdvantageConfig
from battle_model.supply import assemble


def item(rng, n, sparse_dense=True, source="generated"):
	big = rng.normal(size=(n, 40, 30)).astype(np.float32) * (rng.random((n, 40, 30)) < 0.05)
	mask = rng.random((n, 25, 50)) < 0.03
	ids = (rng.integers(0, 9, size=(n, 12, 100)) * (rng.random((n, 12, 100)) < 0.1)).astype(np.int32)
	return {
		"data": {
			"inputs": {"big": Sparse(big) if sparse_dense else big, "mask": Sparse(mask), "ids": Sparse(ids), "small": rng.normal(size=(n, 7)).astype(np.float32)},
			"advantage": rng.normal(size=n).astype(np.float32),
			"old_log_prob": rng.normal(size=n).astype(np.float32),
		},
		"decisions": n,
		"summary": SimpleNamespace(our_side=int(rng.integers(0, 2))),
		"source": source,
	}


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_the_device_built_batch_is_the_cpu_one(device):
	rng = np.random.default_rng(3)
	items = [item(rng, 5), item(rng, 1, source="real"), item(rng, 7, sparse_dense=False), item(rng, 3), {"data": None, "summary": SimpleNamespace(our_side=0)}]
	items[3]["data"]["inputs"]["mask"] = Sparse(np.zeros((3, 25, 50), bool))  # (no nonzero entry at all)
	cpu, summaries = assemble(items, AdvantageConfig())
	built, summaries_too = assemble(items, AdvantageConfig(), device=torch.device(device))
	assert len(summaries) == len(summaries_too) == 5

	def compare(a, b):
		if isinstance(a, dict):
			assert a.keys() == b.keys()
			for k in a:
				compare(a[k], b[k])
			return
		assert a.dtype == b.dtype and a.shape == b.shape
		assert torch.equal(a, b.cpu())

	compare(cpu, built)
	assert built["inputs"]["big"].device.type == device and built["inputs"]["small"].device.type == "cpu"  # (moved by the caller)
