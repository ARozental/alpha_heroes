"""CUDA graphs for the campaign learner's update (train.ppo_update): its forward and backward recorded once a batch
size and replayed. Under WSL every op's launch costs: a 64-decision micro-batch's forward and backward were ~2.3 s of
launches against 60 ms of GPU work (2026-10-30, the PC), the learner ~1/8 as fast as both machines' games. Recorded
with the model's fixed windows (model_v2.windows_of: one shape a batch size, no sync with the host; the same outputs),
each replay is a launch or two. The optimizer updates the weights in place, so the graphs always read the current ones.

	Graphs(model).train(inputs)   the outputs, the backward recorded too (torch.cuda.make_graphed_callables)
	Graphs(model).eval(inputs)    the outputs without gradients (the update's first pass)

A batch of another size than the first one recorded, or a model without fixed windows, runs eagerly as before.
"""
import torch
from torch import nn


class _Flat(nn.Module):
	"""The model with its inputs and outputs as tuples in a fixed order (what a graph's static buffers need)."""

	def __init__(self, model, names, outputs):
		super().__init__()
		self.model, self.names, self.outputs = model, names, outputs

	def forward(self, *tensors):
		out = self.model(dict(zip(self.names, tensors)))
		return tuple(out[name] for name in self.outputs)


class Graphs:
	def __init__(self, model):
		self.model = model
		self.names = self.outputs = None
		self.trained = {}  # rows -> the graphed module (forward and backward), or None: its recording failed
		self.evaluated = {}  # rows -> (static inputs, static outputs, graph), or None
		self.pool = None

	@staticmethod
	def usable(model, device):
		return device.type == "cuda" and hasattr(model, "fixed_windows")

	def _order(self, inputs):
		if self.names is None:
			self.names = tuple(sorted(inputs))
			was = self.model.training
			self.model.eval()
			with torch.no_grad():
				self.outputs = tuple(self.model(inputs))  # (the model's output names, in its order)
			self.model.train(was)
		return tuple(inputs[name] for name in self.names)

	def train(self, inputs):
		"""The model's outputs for inputs (a dict on the GPU), its backward through the recorded graph."""
		self.model.fixed_windows = True
		args = self._order(inputs)
		rows = args[0].shape[0]
		if rows not in self.trained:
			try:
				flat = _Flat(self.model, self.names, self.outputs)
				self.trained[rows] = torch.cuda.make_graphed_callables(flat, tuple(a.detach().clone() for a in args), num_warmup_iters=3)
			except Exception as error:  # (eager from then on, at this size)
				self.trained[rows] = None
				print(f"campaign graphs: no training graph at {rows} rows ({type(error).__name__}: {str(error)[:200]}): eager", flush=True)
		graphed = self.trained[rows]
		if graphed is None:
			return self.model(inputs)
		return dict(zip(self.outputs, graphed(*args)))

	def eval(self, inputs):
		"""The model's outputs for inputs without gradients (copies: the next replay overwrites the graph's)."""
		self.model.fixed_windows = True
		args = self._order(inputs)
		rows = args[0].shape[0]
		if rows not in self.evaluated:
			try:
				static = tuple(a.detach().clone() for a in args)
				flat = _Flat(self.model, self.names, self.outputs)
				side = torch.cuda.Stream()
				side.wait_stream(torch.cuda.current_stream())
				with torch.cuda.stream(side), torch.no_grad():
					for _ in range(2):
						flat(*static)
				torch.cuda.current_stream().wait_stream(side)
				graph = torch.cuda.CUDAGraph()
				with torch.no_grad(), torch.cuda.graph(graph, pool=self.pool):
					out = flat(*static)
				self.pool = graph.pool()
				self.evaluated[rows] = (static, out, graph)
			except Exception as error:
				self.evaluated[rows] = None
				print(f"campaign graphs: no first-pass graph at {rows} rows ({type(error).__name__}: {str(error)[:200]}): eager", flush=True)
		entry = self.evaluated[rows]
		if entry is None:
			with torch.no_grad():
				return self.model(inputs)
		static, out, graph = entry
		for buffer, value in zip(static, args):
			buffer.copy_(value)
		graph.replay()
		return {name: value.clone() for name, value in zip(self.outputs, out)}
