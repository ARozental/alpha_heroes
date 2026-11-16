"""A CUDA graph's capture refuses any copy from host memory (deploy 70's first gate, 2026-11-12: list indexes and an index
tensor built in the forward of the economy's features). Two checks before a deploy meets the PC's gate:
(1) the forward's source files index nothing with a list (a list index becomes a host index tensor on CUDA; 12881e9's
    code had none);
(1b) on any device, the forward makes no tensor from host data but through battle_model.device's per-device caches
    (constant, moved): torch.tensor, as_tensor, from_numpy or Tensor.new_tensor called anew each forward is a host copy a
    CUDA graph refuses, even of a scalar (deploy 82's first gate, 2026-11-15: phi-hero's new_tensor(unled_share) and
    the stockpile's torch.tensor(STOCK_FLOORS); (2) below lets 0-dim host tensors and factories through);
(2) on a GPU (CUDA or Apple's), the model's whole forward with the economy on gives no op a host tensor of one or more
    dimensions, but for the tables moved once by battle_model.device.moved (a cache copied before any capture: the same
    three as 12881e9's forward). Skipped where no GPU exists."""
import ast
import traceback
from pathlib import Path

import pytest
import torch

from campaign_model.model_v2 import CampaignModelV2
from campaign_model.tests.test_model_v2 import rich_inputs

FORWARD_FILES = ("features.py", "model.py", "model_v2.py", "encoders.py", "foresight.py")


def test_no_list_index_in_the_forward_files():
	root = Path(__file__).resolve().parent.parent
	hits = []
	for name in FORWARD_FILES:
		path = root / name
		if not path.exists():
			continue
		for node in ast.walk(ast.parse(path.read_text())):
			if isinstance(node, ast.Subscript) and any(isinstance(s, ast.List) or (isinstance(s, ast.Call) and getattr(s.func, "id", "") == "list") for s in ast.walk(node.slice)):
				hits.append(f"{name}:{node.lineno}")
	assert not hits, f"list indexes (a host index tensor on CUDA): {hits}"


def gpu():
	if torch.cuda.is_available():
		return torch.device("cuda")
	if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
		return torch.device("mps")
	return None


@pytest.mark.skipif(gpu() is None, reason="no GPU here")
def test_the_forward_gives_no_op_a_host_tensor():
	from torch.utils._python_dispatch import TorchDispatchMode
	from torch.utils._pytree import tree_flatten

	device = gpu()
	torch.manual_seed(3)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	model.features.build_economy.fill_(1.0)
	model = model.to(device)
	x = {k: torch.from_numpy(v).to(device) for k, v in rich_inputs(4, 4).items()}
	found = []

	class Host(TorchDispatchMode):
		def __torch_dispatch__(self, func, types, args=(), kwargs=None):
			flat, _ = tree_flatten((args, kwargs or {}))
			if any(isinstance(a, torch.Tensor) and a.device.type == "cpu" and a.dim() > 0 for a in flat):
				frames = [fr for fr in traceback.extract_stack() if "campaign_model/" in fr.filename and "tests/" not in fr.filename]
				line = frames[-1].line if frames else ""
				if "moved(" not in (line or ""):
					found.append(f"{frames[-1].filename.split('campaign_model/')[-1]}:{frames[-1].lineno} {func}" if frames else str(func))
			return func(*args, **(kwargs or {}))

	with torch.no_grad(), Host():
		model(x)
	assert not found, f"ops given a host tensor in the forward (a CUDA graph refuses them): {sorted(set(found))}"


def test_the_forward_makes_no_tensor_from_host_data_but_through_the_caches():
	from campaign_model.model_v2 import CampaignModelV2

	torch.manual_seed(3)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	model.features.build_economy.fill_(1.0)
	model.features.phi_stock_floor.fill_(1.0 / 3.0)  # (the stockpile part's own paths)
	model.features.empire_unit_own.fill_(1.0)
	model.features.give_all_exact.fill_(1.0)
	x = {k: torch.from_numpy(v) for k, v in rich_inputs(4, 4).items()}
	found = []

	def wrap(owner, attr):
		real = getattr(owner, attr)

		def probe(*args, **kwargs):
			stack = traceback.extract_stack()[:-1]
			ours = [fr for fr in stack if "campaign_model/" in fr.filename and "tests/" not in fr.filename]
			if ours and not stack[-1].filename.endswith("battle_model/device.py"):  # (the caches: made once per device)
				found.append(f"{ours[-1].filename.split('campaign_model/')[-1]}:{ours[-1].lineno} {attr}")
			return real(*args, **kwargs)

		setattr(owner, attr, probe)
		return owner, attr, real

	saved = [wrap(o, a) for o, a in ((torch, "tensor"), (torch, "as_tensor"), (torch, "from_numpy"), (torch.Tensor, "new_tensor"))]
	try:
		with torch.no_grad():
			model(x)
	finally:
		for owner, attr, real in saved:
			setattr(owner, attr, real)
	assert not found, f"tensors made from host data in the forward (a CUDA graph refuses the copy): {sorted(set(found))}"
