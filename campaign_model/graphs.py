"""CUDA graphs for the campaign learner's update (train.ppo_update): its forward and backward recorded once a batch
size and replayed. Under WSL every op's launch costs: a 64-decision micro-batch's forward and backward were ~2.3 s of
launches against 60 ms of GPU work (2026-10-30, the PC), the learner ~1/8 as fast as both machines' games. Recorded
with the model's fixed windows (model_v2.windows_of: one shape a batch size, no sync with the host; the same outputs),
each replay is a launch or two. The optimizer updates the weights in place, so the graphs always read the current ones.

	Graphs(model).train(inputs)   the outputs, the backward recorded too (torch.cuda.make_graphed_callables)
	Graphs(model).eval(inputs)    the outputs without gradients (the update's first pass)

A batch of another size than the first one recorded, or a model without fixed windows, runs eagerly as before.

A graph may compute fewer windows than every hero slot (window_budget: model_v2.windows_of's budget), the caller having
counted the batch's heroes; its own graph, checked as a step's (StepGraph: against the eager step over every slot)
before the update's first pass uses it (StepGraph.budget_ok).
"""
import contextlib

import torch
from torch import nn


@contextlib.contextmanager
def window_budget(model, budget):
	"""The model's fixed windows (model_v2.windows_of), `budget` of them a batch (None: every hero slot), while a graph
	is recorded or a step runs eagerly: never left set (an eager forward elsewhere, on a batch with more of our heroes,
	would lose some of their windows)."""
	model.fixed_windows, model.window_budget = True, budget
	try:
		yield
	finally:
		model.window_budget = None


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
				# (allow_unused_input: parameters no output reaches, e.g. a head's the loss leaves out, failed the capture: "the
				# differentiated Tensor ... appears to not have been used", 2026-10-30)
				self.trained[rows] = torch.cuda.make_graphed_callables(flat, tuple(a.detach().clone() for a in args), num_warmup_iters=3, allow_unused_input=True)
			except Exception as error:  # (eager from then on, at this size)
				self.trained[rows] = None
				print(f"campaign graphs: no training graph at {rows} rows ({type(error).__name__}: {str(error)[:200]}): eager", flush=True)
		graphed = self.trained[rows]
		if graphed is None:
			return self.model(inputs)
		return dict(zip(self.outputs, graphed(*args)))

	def eval(self, inputs, budget=None):
		"""The model's outputs for inputs without gradients (copies: the next replay overwrites the graph's). budget: the
		windows a batch (window_budget), a graph of its own; every slot's if that one could not be recorded."""
		self.model.fixed_windows = True
		args = self._order(inputs)
		key = (args[0].shape[0], budget)
		if key not in self.evaluated:
			try:
				static = tuple(a.detach().clone() for a in args)
				flat = _Flat(self.model, self.names, self.outputs)
				with window_budget(self.model, budget):
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
				self.evaluated[key] = (static, out, graph)
			except Exception as error:
				self.evaluated[key] = None
				print(f"campaign graphs: no first-pass graph at {key[0]} rows, {budget or 'all'} windows ({type(error).__name__}: {str(error)[:200]}): {'every slot' if budget else 'eager'}", flush=True)
		entry = self.evaluated[key]
		if entry is None:
			if budget is not None:
				return self.eval(inputs)
			with torch.no_grad():
				return self.model(inputs)
		static, out, graph = entry
		for buffer, value in zip(static, args):
			buffer.copy_(value)
		graph.replay()
		return {name: value.clone() for name, value in zip(self.outputs, out)}


class StepGraph:
	"""A micro-batch's whole training step recorded as one CUDA graph: the model's forward, the loss (fn) and the backward,
	its gradient added into the parameters' .grad (the tensors there when it was recorded: the caller zeroes them in
	place, never to None). One replay instead of the ~1,000 kernels the loss and its backward launched outside the
	model's graph (Graphs.train), ~45 of a 76 s update on the PC (2026-10-31).

		StepGraph(model, fn).run(inputs, batch)  fn's outputs past the loss (copies), or None: not recorded (eager then)

	fn(out, batch) -> (loss, *outputs): tensors only, nothing read to the host. A graph is recorded per shape of the
	inputs and the batch, and checked the first time against the eager step on the same micro-batch (its gradient and
	outputs); one that differs, that could not be recorded or whose check found no memory, is dropped for good and the
	caller steps eagerly."""

	TOLERANCE = 1e-3  # the largest gradient difference, relative to the largest gradient (TF32 matmuls: ~1e-3 each)

	def __init__(self, model, fn):
		self.model, self.fn = model, fn
		self.entries = {}  # (shape signature, window budget) -> {"inputs", "batch", "outputs", "graph", "checked"} or None
		# A budget graph's check (against the eager step over every slot): None before the first, then whether it held;
		# the update uses budgets while not False, and in its first pass (no check there) once True
		self.budget_ok = None

	@staticmethod
	def signature(*dicts):
		return tuple(tuple(sorted((name, tuple(value.shape), value.dtype) for name, value in d.items())) for d in dicts)

	def run(self, inputs, batch, budget=None):
		"""budget: the windows of the step (window_budget), a graph of its own; every slot's (the graph without a budget)
		once one with a budget failed its check or could not be recorded."""
		self.model.fixed_windows = True
		if self.budget_ok is False:
			budget = None
		key = self.signature(inputs, batch) + (budget,)
		if key not in self.entries:
			self.entries[key] = self._record(inputs, batch, budget)
		entry = self.entries[key]
		if entry is None:
			return None if budget is None else self.run(inputs, batch)
		for name, value in inputs.items():
			entry["inputs"][name].copy_(value)
		for name, value in batch.items():
			entry["batch"][name].copy_(value)
		if not entry["checked"]:
			return self._first(entry, key, inputs, batch, budget)
		entry["graph"].replay()
		return tuple(None if o is None else o.clone() for o in entry["outputs"])

	def _params(self):
		return [p for p in self.model.parameters() if p.requires_grad]

	def _record(self, inputs, batch, budget=None):
		params, before = self._params(), None
		try:
			for p in params:
				if p.grad is None:
					p.grad = torch.zeros_like(p)
			before = [p.grad.clone() for p in params]  # (the minibatch's gradient so far: the warm-up adds to it)
			with window_budget(self.model, budget):
				static_inputs = {name: value.detach().clone() for name, value in inputs.items()}
				static_batch = {name: value.detach().clone() for name, value in batch.items()}
				side = torch.cuda.Stream()
				side.wait_stream(torch.cuda.current_stream())
				with torch.cuda.stream(side):
					for _ in range(2):  # (the allocator and the libraries' choices settle before the recording)
						loss, *_ = self.fn(self.model(static_inputs), static_batch)
						loss.backward()
						del loss
				torch.cuda.current_stream().wait_stream(side)
				for p, g in zip(params, before):
					p.grad.copy_(g)
				graph = torch.cuda.CUDAGraph()
				with torch.cuda.graph(graph):
					loss, *outputs = self.fn(self.model(static_inputs), static_batch)
					loss.backward()
			return {"inputs": static_inputs, "batch": static_batch, "outputs": tuple(None if o is None else o.detach() for o in outputs), "graph": graph, "checked": False}
		except Exception as error:
			if before is not None:  # (a warm-up cut short left its part added, maybe still running on its stream)
				torch.cuda.synchronize()
				for p, g in zip(params, before):
					p.grad.copy_(g)
			print(f"campaign step graph: not recorded{f' at {budget} windows' if budget else ''} ({type(error).__name__}: {str(error)[:300]}): {'every slot' if budget else 'eager steps'}", flush=True)
			return None

	def _first(self, entry, key, inputs, batch, budget=None):
		"""The first replay, checked against the eager step on the same micro-batch (over every hero slot: a budget is
		set only while recording); the gradient added either way."""
		params = self._params()
		try:
			loss, *eager_outputs = self.fn(self.model(inputs), batch)
			eager = torch.autograd.grad(loss, params, allow_unused=True)
			del loss
		except torch.OutOfMemoryError as error:
			# The eager step beside the recorded one's pool did not fit (2026-11-01: joint7's first update, under the 27 GB
			# cap, a run's end): the graph is dropped, its pool freed once the caller lets go of it, and the caller steps
			# eagerly (nothing was added to .grad)
			self.entries[key] = None
			print(f"campaign step graph: no memory for its check ({str(error)[:160]}): dropped, eager steps", flush=True)
			return None
		before = [p.grad.clone() for p in params]
		entry["graph"].replay()
		recorded = [p.grad - b for p, b in zip(params, before)]
		scale = max(float(g.abs().max()) for g in eager if g is not None) + 1e-12
		worst = max(float(((r - (g if g is not None else torch.zeros_like(r))).abs().max())) for r, g in zip(recorded, eager)) / scale
		outputs = tuple(None if o is None else o.clone() for o in entry["outputs"])
		output_diff = max((float((o - e.detach()).abs().max()) for o, e in zip(outputs, eager_outputs) if o is not None and e is not None), default=0.0)
		windows = f" at {budget} windows" if budget else ""
		if worst > self.TOLERANCE or output_diff > self.TOLERANCE or not all(torch.isfinite(r).all() for r in recorded):
			for p, b, g in zip(params, before, eager):  # (this micro-batch's gradient: the eager one)
				p.grad.copy_(b if g is None else b + g)
			self.entries[key] = None
			if budget:
				self.budget_ok = False
			print(f"campaign step graph{windows}: differs from the eager step (gradient {worst:.1e}, outputs {output_diff:.1e} of the largest): {'every slot from now on' if budget else 'eager steps'}", flush=True)
			return tuple(None if e is None else e.detach() for e in eager_outputs)
		entry["checked"] = True
		if budget:
			self.budget_ok = True
		print(f"campaign step graph{windows}: recorded and checked (gradient within {worst:.1e} of the eager step's, outputs {output_diff:.1e})", flush=True)
		return outputs
