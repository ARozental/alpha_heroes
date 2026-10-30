"""Fixed tensors on a model's device, made once per device instead of once per forward.

torch.tensor(values, device=...) or a CPU tensor's .to(device) inside a forward is a copy from the CPU, and on Apple's
GPU such a copy waits for every kernel queued before it: 21 of them were three quarters of a battle training step's
time (2026-10-29: a step of 256 decisions 508 -> 376 ms with them made once).
"""
import functools
import os

import torch

# PyTorch's allocator on Apple's GPU frees the buffers it caches only above its low watermark, by default 1.4 x the
# recommended working set (24.9 GB on the 24 GB M4): a joint run's two learners kept 10 and 4.3 GB and the machine
# compressed memory at 2 GB/s. At 0.2 (3.6 GB, a soft limit: a process may use more) the same run's swap-in fell 12.4 ->
# 0.6 MB/s and its campaign updates took 16% less a decision, the battle updates as fast (2026-10-30,
# docs/performance_2026-10-29.md). Read when a process first uses the GPU: the processes it spawns inherit it; a value
# already in the environment is kept.
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.2")


@functools.lru_cache(maxsize=None)
def _constant(values, device, dtype):
	return torch.tensor(values, dtype=dtype, device=device)


def constant(values, device, dtype=None):
	"""torch.tensor(values, dtype=dtype, device=device) for a flat sequence of fixed values, made once per device."""
	return _constant(tuple(values), torch.device(device), dtype)


@functools.lru_cache(maxsize=None)
def moved(tensor, device):
	"""tensor.to(device) for a module-level constant tensor (cached by identity), moved once per device."""
	return tensor.to(device)


def use_fused(optimizer):
	"""Adam's (AdamW's) fused kernel on a GPU: one kernel for all the parameters instead of a few per parameter tensor
	(the battle model's 185: a step 1.7 ms instead of 21 ms on Apple's GPU, 2026-10-29). Called after load_state_dict
	too, which restores the saved groups' settings (fused off); the fused kernel keeps its step counts on the device."""
	params = [p for group in optimizer.param_groups for p in group["params"]]
	if not params or params[0].device.type not in ("mps", "cuda"):
		return optimizer
	for group in optimizer.param_groups:
		group["fused"], group["foreach"] = True, None
	for state in optimizer.state.values():
		if torch.is_tensor(state.get("step")):
			state["step"] = state["step"].to(device=params[0].device, dtype=torch.float32)
	return optimizer


def sync(device):
	"""Waits for the work queued on device (a GPU runs asynchronously): for timing its parts."""
	if device.type == "mps":
		torch.mps.synchronize()
	elif device.type == "cuda":
		torch.cuda.synchronize()
