"""Central inference: one process runs the model for every game, batched, on the GPU (or the CPU), with the trainer's
live weights. The games ask it over a Unix socket instead of each loading an ONNX file (1.1 s of CPU per battle, to
create the session) and running it on a CPU core; the trainer sends new weights over a pipe instead of exporting them
(1.3 s an update). The ONNX path stays for evals and the finished product.

Generic over model families (a network and its contract): "battle", and "campaign" (the adventure map,
campaign_model/). A client says which family it speaks at the start (the contract hash must match) and names the model
of each request: "current" (the trainer's newest weights) or any other the trainer set, e.g. a past version for the
league ("v350").

Wire protocol, little endian, one request at a time per connection; every message is a u32 length and that many bytes:
  hello    client: json {"family": ..., "contract_hash": ...}          server: json {"ok": true} or {"error": ...}
           (+ "compress": "zlib": every later message both ways is u32 size + zlib (level 1) of the message: a client on
           another machine, over TCP; a battle decision's ~400 KB both ways become ~4 KB, being mostly 0 and -inf)
  request  client: u8 name length, name (utf-8), u32 decisions (0: only the version), then each decision's inputs in
           contract order: a bool as one byte, an int64 as an int32
  reply    server: u32 version, then each decision's outputs in contract order as float32; version 0xFFFFFFFF is an
           error, the rest of the message its text
"""
import json
import os
import queue
import socket
import struct
import threading
import time
import zlib
from dataclasses import dataclass

import numpy as np

ERROR = 0xFFFFFFFF
MAX_BATCH = 256  # decisions per forward pass
# CUDA graphs (on CUDA): each model's forward recorded once a batch size, a batch padded up to the next. A forward is
# ~35 ms of launches at any batch up to 64 under WSL (the campaign model's; 0.6 ms a decision at 64, 34 ms at one,
# measured 2026-10-30 on the PC), and the games send one decision at a time
GRAPH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
# Seconds a request for a model not served yet waits for its weights: set_model returns once the pipe has taken them,
# before the server has unpickled them, and a request sent right after it found no model (every time on Linux). Such a
# request is parked (the worker goes on with the others, looking again every PARK_POLL seconds) and answered when the
# weights come, or with an error after MODEL_WAIT. Until 2026-11-10 the worker itself waited, MODEL_WAIT a request,
# and every other request of its process waited behind it (at a start: the games' version probes before the
# campaign's weights were set, on 11 of 12 processes at once)
MODEL_WAIT = 5.0
PARK_POLL = 0.02
# Seconds a batch waits for more requests after its first (0: none): fewer, larger forwards. Beside a trainer on a
# Windows GPU (WSL) every submission to the GPU is a context switch away from the trainer's (2026-10-30: a server
# answering one-decision requests as they came cost the trainer 54% of its step rate)
BATCH_WAIT = float(os.environ.get("ALPHA_INFERENCE_BATCH_WAIT_MS", "0")) / 1000
# Seconds a request waits for its reply before the server says so in the run's log, with what its worker is at (then
# again every so many). 2026-11-10: of the games their stall watchdog killed (no log output for 60 s), 26 kept on the
# PC had their AI waiting in the socket read for a reply (a client's wait has no limit), and nothing here said why
SLOW = float(os.environ.get("ALPHA_INFERENCE_SLOW_SECONDS", "10"))


@dataclass(frozen=True)
class Family:
	inputs: tuple  # schema.TensorSpec, in contract order
	outputs: tuple
	contract_hash: str
	build: object  # model_kwargs -> torch.nn.Module


def families():
	from campaign_model import schema as campaign_schema
	from campaign_model.model import build_model as build_campaign_model

	from . import schema
	from .model import build_model

	return {
		"battle": Family(schema.INPUTS, schema.OUTPUTS, schema.contract_hash(), build_model),
		"campaign": Family(campaign_schema.INPUTS, campaign_schema.OUTPUTS, campaign_schema.contract_hash(), build_campaign_model),
	}


def wire_layout(specs, inputs):
	"""[(name, shape, wire dtype, model dtype, bytes per decision)] in contract order."""
	layout = []
	for spec in specs:
		shape = tuple(spec.shape)
		wire = (np.uint8 if spec.dtype == "bool" else np.int32) if inputs else np.float32
		model = {"bool": np.bool_, "int64": np.int64, "float32": np.float32}[spec.dtype]
		layout.append((spec.name, shape, wire, model, int(np.prod(shape, dtype=np.int64)) * np.dtype(wire).itemsize))
	return layout


def _read_exactly(conn, n):
	data = bytearray(n)
	view, got = memoryview(data), 0
	while got < n:
		k = conn.recv_into(view[got:], n - got)
		if not k:
			raise ConnectionError("closed")
		got += k
	return data


def read_message(conn):
	(length,) = struct.unpack("<I", _read_exactly(conn, 4))
	return _read_exactly(conn, length)


def send_message(conn, payload):
	conn.sendall(struct.pack("<I", len(payload)) + payload)


def pack(message):
	"""A message as a compressed connection sends it (the hello's "compress": "zlib")."""
	return struct.pack("<I", len(message)) + zlib.compress(message, 1)


def unpack(packed):
	(size,) = struct.unpack_from("<I", packed)
	message = zlib.decompress(memoryview(packed)[4:])
	if len(message) != size:
		raise ConnectionError(f"a compressed message of {len(message)} bytes, {size} announced")
	return message


# ---- the server process ----


def _serve(socket_path, device_name, family_names, model_kwargs, control, threads, tcp=None):
	"""One process serving the models of family_names (model_kwargs: {family: kwargs}; a client says which it speaks in
	its hello). tcp: (host, port) to listen on for clients on other machines too, or None. One process for every family
	(2026-10-30): beside a trainer on a Windows GPU each process is a context the GPU is time-sliced between."""
	import torch

	torch.set_num_threads(threads)
	torch_dtype = {np.dtype(t): getattr(torch, name) for t, name in ((np.uint8, "uint8"), (np.int32, "int32"), (np.float32, "float32"), (np.bool_, "bool"), (np.int64, "int64"))}
	device = torch.device(device_name)
	all_families = families()
	served = {}  # family name -> (Family, in_layout, out_layout, in_bytes, offsets)
	for name in family_names:
		family = all_families[name]
		in_layout, out_layout = wire_layout(family.inputs, True), wire_layout(family.outputs, False)
		served[name] = (family, in_layout, out_layout, sum(entry[-1] for entry in in_layout), np.cumsum([0] + [entry[-1] for entry in in_layout])[:-1])
	models = {}  # (family, name) -> [module, version]
	requests = queue.Queue()  # (family, name, decisions, payload, reply): the worker puts its reply's bytes in reply[0], sets reply[1]
	commands = queue.Queue()  # from the trainer, applied by the worker (the only thread that touches the device)
	doing = ["starting", time.monotonic()]  # what the worker is at and since when (said beside a request that waits long)
	said = [0.0]  # when a waiting request was last said (a line per SLOW seconds a process at most)

	def say(text):
		"""A line in the run's log; a diagnostic, never an error here."""
		try:
			print(f"inference server {os.getpid()}: {text}", flush=True)
		except Exception:  # noqa: BLE001
			pass

	def control_reader():  # drains the pipe at once: a weights message (~1 MB) would block the trainer until read
		while True:
			try:
				commands.put(control.recv())
			except (EOFError, OSError):
				return

	def reader(conn):
		"""One client's connection: its requests to the worker, the replies back (compressed here, not in the worker)."""
		try:
			hello = json.loads(read_message(conn))
			family_name = hello.get("family")
			if family_name not in served or hello.get("contract_hash") != served[family_name][0].contract_hash:
				send_message(conn, json.dumps({"error": f"this server runs {', '.join(f'{n} {f[0].contract_hash}' for n, f in served.items())}, the client speaks {hello}"}).encode())
				return
			in_bytes = served[family_name][3]
			packed = hello.get("compress") == "zlib"
			send_message(conn, json.dumps({"ok": True}).encode())
			while True:
				message = read_message(conn)
				if packed:
					message = unpack(message)
				name = message[1 : 1 + message[0]].decode()
				(decisions,) = struct.unpack_from("<I", message, 1 + message[0])
				payload = memoryview(message)[5 + message[0] :]
				if len(payload) != decisions * in_bytes:
					reply = struct.pack("<I", ERROR) + f"{decisions} decisions need {decisions * in_bytes} bytes, got {len(payload)}".encode()
				else:
					box = [None, threading.Event()]
					asked = time.monotonic()
					requests.put((family_name, name, decisions, payload, box))
					# one request at a time per connection: the reply goes out before the next is read. A request still
					# waiting after SLOW seconds is said, with what the worker is at
					while not box[1].wait(SLOW):
						now = time.monotonic()
						if now - said[0] >= SLOW:
							said[0] = now
							say(f"a request for {family_name}/{name} ({decisions} decisions) has waited {now - asked:.0f} s for its reply; the worker: {doing[0]}, for {now - doing[1]:.0f} s; {requests.qsize()} requests queued")
					reply = box[0]
				send_message(conn, pack(reply) if packed else reply)
		except (ConnectionError, OSError, zlib.error):
			pass
		finally:
			conn.close()

	def apply_control():
		"""Applies the trainer's queued commands."""
		while True:
			try:
				command = commands.get_nowait()
			except queue.Empty:
				return
			if command[0] == "set":
				_, family_name, name, version, state = command
				key = (family_name, name)
				if key not in models:
					models[key] = [served[family_name][0].build(model_kwargs[family_name]).to(device).eval(), version]
					if device.type == "cuda" and hasattr(models[key][0], "fixed_windows"):
						models[key][0].fixed_windows = True  # (the same outputs in fixed shapes: model_v2.windows_of)
				models[key][0].load_state_dict(state)
				models[key][1] = version
			elif command[0] == "drop":
				key = (command[1], command[2])
				models.pop(key, None)
				for graph_key in [graph_key for graph_key in graphs if graph_key[0] == key]:
					del graphs[graph_key]

	def decode_on_device(family_name, on_device):
		"""The inputs of a batch's bytes [N, in_bytes] on the GPU, its columns cut and converted there."""
		_, in_layout, _, _, offsets = served[family_name]
		inputs, n = {}, on_device.shape[0]
		for (input_name, shape, wire, dtype, nbytes), offset in zip(in_layout, offsets):
			# (a copy always: one decision's slice counts as contiguous, so .contiguous() kept its odd row stride, the
			# record's byte count, and the view refused it: "stride(0) must be divisible by 4", 2026-10-30, the PC)
			column = on_device[:, offset : offset + nbytes].clone(memory_format=torch.contiguous_format).view(torch_dtype[np.dtype(wire)])
			inputs[input_name] = column.to(torch_dtype[np.dtype(dtype)]).reshape((n,) + shape)
		return inputs

	def forward_flat(family_name, model, inputs, n):
		out = model(inputs)
		return torch.cat([out[o_name].float().reshape(n, -1) for o_name, *_ in served[family_name][2]], dim=1)

	graphs = {}  # ((family, name), size) -> (flat outputs, graph), or None: eager (its capture failed)
	# A size's input bytes, pinned on the host and on the device, shared by every model's graph at that size (they run
	# one at a time): one each a model was ~77 MB of both for the battle server's ~50 league models (2026-10-30)
	graph_inputs = {}  # (family, size) -> (pinned host bytes, device bytes)
	graph_pool = [None]  # one memory pool for all, likewise

	def graphed(key, model, data):
		"""flat [N, out] on the host for the bytes data [N, in_bytes] of model key = (family, name), through its graph at
		the next size up (recorded on first use), or None when there is none (too big, or its capture failed)."""
		family_name, in_bytes = key[0], served[key[0]][3]
		total = data.shape[0]
		size = next((size for size in GRAPH_SIZES if size >= total), None)
		graph_key = (key, size)
		if size is None or (graph_key in graphs and graphs[graph_key] is None):
			return None
		if graph_key not in graphs:
			try:
				if (family_name, size) not in graph_inputs:
					host = torch.empty((size, in_bytes), dtype=torch.uint8).pin_memory()
					graph_inputs[(family_name, size)] = (host, torch.empty((size, in_bytes), dtype=torch.uint8, device=device))
				host, on_device = graph_inputs[(family_name, size)]
				host[:total] = torch.from_numpy(data)
				host[total:] = host[:1]  # (the padding rows a real decision: their outputs are dropped)
				on_device.copy_(host)
				side = torch.cuda.Stream()
				side.wait_stream(torch.cuda.current_stream())
				with torch.cuda.stream(side), torch.inference_mode():
					for _ in range(2):
						forward_flat(family_name, model, decode_on_device(family_name, on_device), size)
				torch.cuda.current_stream().wait_stream(side)
				graph = torch.cuda.CUDAGraph()
				with torch.inference_mode(), torch.cuda.graph(graph, pool=graph_pool[0]):
					flat = forward_flat(family_name, model, decode_on_device(family_name, on_device), size)
				graph_pool[0] = graph.pool()
				graphs[graph_key] = (flat, graph)
			except Exception as error:  # (eager from then on, for this model and size)
				graphs[graph_key] = None
				print(f"inference: no CUDA graph for {key} at {size} ({type(error).__name__}: {str(error)[:200]}): eager", flush=True)
				return None
		flat, graph = graphs[graph_key]
		host, on_device = graph_inputs[(family_name, size)]
		host[:total] = torch.from_numpy(data)  # (the rows past total: an earlier batch's, real decisions)
		on_device.copy_(host, non_blocking=True)
		graph.replay()
		return flat[:total].cpu().numpy()  # (one copy back, one sync)

	def run(key, batch):
		"""The replies of one model's requests, run as one batch."""
		family_name = key[0]
		_, in_layout, _, in_bytes, offsets = served[family_name]
		model, version = models[key]
		total = sum(decisions for _, decisions, _, _ in batch)
		data = np.concatenate([np.frombuffer(payload, dtype=np.uint8).reshape(decisions, in_bytes) for _, decisions, payload, _ in batch])
		outputs = graphed(key, model, data) if device.type == "cuda" else None
		if outputs is None:
			inputs = {}
			if device.type == "cpu":
				for (input_name, shape, wire, dtype, nbytes), offset in zip(in_layout, offsets):
					column = np.ascontiguousarray(data[:, offset : offset + nbytes]).view(wire)
					inputs[input_name] = torch.from_numpy(column.astype(dtype).reshape((total,) + shape))
			else:  # the whole batch to the GPU in one copy, its columns cut and converted there (not one copy an input)
				inputs = decode_on_device(family_name, torch.from_numpy(data).to(device))
			with torch.inference_mode():
				flat = forward_flat(family_name, model, inputs, total)
			outputs = flat.cpu().numpy()  # (one copy back, one sync)
		replies, first = [], 0
		for _, decisions, _, box in batch:
			replies.append((box, struct.pack("<I", version) + outputs[first : first + decisions].astype(np.float32).tobytes()))
			first += decisions
		return replies

	def worker():
		parked = []  # (deadline, request): those for a model not served yet (MODEL_WAIT); they hold no other back
		while True:
			doing[:] = [f"idle, {len(parked)} requests parked for a model not set" if parked else "idle", time.monotonic()]
			pending = []
			try:
				pending.append(requests.get(timeout=PARK_POLL) if parked else requests.get())
			except queue.Empty:
				pass
			deadline = time.monotonic() + BATCH_WAIT
			while pending and len(pending) < MAX_BATCH:
				try:
					pending.append(requests.get(timeout=max(0.0, deadline - time.monotonic())) if BATCH_WAIT else requests.get_nowait())
				except queue.Empty:
					break
			doing[:] = ["the trainer's commands", time.monotonic()]
			try:
				apply_control()
			except Exception as error:  # noqa: BLE001
				# (never the worker's end: every request after it would wait for ever. The commands left are applied
				# before the next batch)
				say(f"a command of the trainer's failed: {type(error).__name__}: {str(error)[:300]}")
			now = time.monotonic()
			waiting, parked = parked, []
			for until, item in waiting:  # the parked: into this batch once their model is served, an error at their deadline
				key = (item[0], item[1])
				if key in models:
					pending.append(item)
				elif now >= until:
					item[4][0] = struct.pack("<I", ERROR) + f"no model named {key[1]!r} of family {key[0]} (the server has {sorted(models)})".encode()
					item[4][1].set()
				else:
					parked.append((until, item))
			by_key = {}
			for item in pending:
				key = (item[0], item[1])
				if key in models:
					by_key.setdefault(key, []).append(item[1:])
				else:
					parked.append((now + MODEL_WAIT, item))
			for key, batch in by_key.items():
				doing[:] = [f"{key[0]}/{key[1]}, {len(batch)} of {len(pending)} requests", time.monotonic()]
				try:
					replies = [(box, struct.pack("<I", models[key][1])) for _, decisions, _, box in batch if decisions == 0]
					work = [item for item in batch if item[1] > 0]
					replies += run(key, work) if work else []
				except Exception as error:
					replies = [(box, struct.pack("<I", ERROR) + str(error).encode()) for _, _, _, box in batch]
				for box, payload in replies:  # (the connections' threads send them)
					box[0] = payload
					box[1].set()

	server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
	server.bind(socket_path)
	server.listen(256)
	listeners = [server]
	if tcp:  # clients on other machines (battle_model/worker.py)
		remote = socket.create_server(tuple(tcp), backlog=256, reuse_port=False)
		listeners.append(remote)
	control.send(("ready",))
	threading.Thread(target=control_reader, daemon=True).start()
	threading.Thread(target=worker, daemon=True).start()
	threading.Thread(target=_watch_parent, args=(listeners,), daemon=True).start()

	def accept(listener):
		while True:
			try:
				conn, _ = listener.accept()
			except OSError:
				return
			if listener.family != socket.AF_UNIX:
				conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # (one small message after another)
			threading.Thread(target=reader, args=(conn,), daemon=True).start()

	for listener in listeners[1:]:
		threading.Thread(target=accept, args=(listener,), daemon=True).start()
	accept(server)


def _watch_parent(listeners):
	"""The server ends with its trainer, so no server outlives a run."""
	import time

	while True:
		try:
			if os.getppid() == 1:
				break
			time.sleep(1)
		except Exception:
			break
	for listener in listeners:
		listener.close()
	os._exit(0)


# ---- the trainer's handle ----


class InferenceServer:
	"""Starts the server processes, each on its own socket (socket_paths; socket_path is the first); set_model gives
	them all weights (state_dict) under a name. On the CPU one process does not scale past a core or two (its Python
	and the model's many small operations), so several share the games; on a GPU one process batches them all.

	One server can serve several families (families=("battle", "campaign"), model_kwargs {family: kwargs}): the
	trainer's two models in one process, one GPU context (2026-10-30: a Windows GPU is time-sliced between processes).
	view(family) is the handle one family's owner uses (its set_model, serving, socket_for)."""

	def __init__(self, socket_path, device="cpu", family="battle", model_kwargs=None, threads=2, processes=1, tcp_host=None, tcp_port=0, families=None):
		"""tcp_host, tcp_port: also listen there for clients on other machines (worker.py), process i on tcp_port + i."""
		import multiprocessing

		base = str(socket_path)
		self.families = tuple(families) if families else (family,)
		kwargs = dict(model_kwargs or {})
		self.kwargs = kwargs if families else {family: kwargs}  # family -> what builds its model
		self.kwargs = {f: dict(self.kwargs.get(f) or {}) for f in self.families}
		self.family = self.families[0]
		self.model_kwargs = self.kwargs[self.family]  # (one family's handle: what workers that run their own servers build)
		# What it serves, for workers that run their own servers (remote.Hub sends them the weights): (family, name) ->
		# (version, state)
		self.served = {}
		self.socket_paths = [base] if processes == 1 else [base.replace(".sock", f"-{i}.sock") for i in range(processes)]
		self.socket_path = self.socket_paths[0]
		self.tcp_ports = [tcp_port + i for i in range(processes)] if tcp_port else []
		context = multiprocessing.get_context("spawn")  # no fork of a process with threads and a GPU context
		self.lock = threading.Lock()
		self.controls, self.processes = [], []
		for i, path in enumerate(self.socket_paths):
			if os.path.exists(path):
				os.unlink(path)
			control, child = context.Pipe()
			tcp = (tcp_host or "0.0.0.0", self.tcp_ports[i]) if self.tcp_ports else None
			process = context.Process(target=_serve, args=(path, device, self.families, self.kwargs, child, threads, tcp), daemon=True)
			process.start()
			self.controls.append(control)
			self.processes.append(process)
		for control in self.controls:
			if not control.poll(120) or control.recv() != ("ready",):
				raise RuntimeError("the inference server did not start")

	def view(self, family):
		"""This server as one family's owner sees it."""
		return _FamilyView(self, family)

	def set_model(self, name, version, model, family=None):
		"""Serve model's current weights (a module or a state_dict) under name, as version, for family (the first)."""
		family = family or self.family
		state = model.state_dict() if hasattr(model, "state_dict") else model
		state = {key: value.detach().cpu() for key, value in state.items()}
		with self.lock:
			for control in self.controls:
				control.send(("set", family, name, int(version), state))
			self.served[(family, name)] = (int(version), state)

	def drop(self, name, family=None):
		family = family or self.family
		with self.lock:
			for control in self.controls:
				control.send(("drop", family, name))
			self.served.pop((family, name), None)

	def serving(self, family=None):
		"""{name: (version, state)} of a family (the first) as now served (a copy)."""
		family = family or self.family
		with self.lock:
			return {name: item for (f, name), item in self.served.items() if f == family}

	def socket_for(self, index):
		"""The socket of the index-th client (actor): the clients spread evenly over the processes."""
		return self.socket_paths[index % len(self.socket_paths)]

	def endpoints(self, host):
		"""The addresses clients on other machines use, "tcp:<host>:<port>" a process (host: this machine's, as they reach it)."""
		return [f"tcp:{host}:{port}" for port in self.tcp_ports]

	def close(self):
		for process in self.processes:
			process.terminate()
		for process in self.processes:
			process.join(timeout=10)
		for path in self.socket_paths:
			if os.path.exists(path):
				os.unlink(path)


class ModelStore:
	"""In place of an InferenceServer for a family whose inference other machines serve (TrainConfig.inference_endpoints;
	2026-10-30: on the Windows PC every process on the GPU costs the trainer ~40% of its steps, while the Mac's cores idle
	at 9-12 ms a campaign decision): keeps what is served, which remote.Hub sends to the workers' own servers
	(--local --serve), and the endpoints ("tcp:<host>:<port>") the games here ask."""

	def __init__(self, family, endpoints, model_kwargs=None):
		self.family, self.model_kwargs = family, dict(model_kwargs or {})
		self.socket_paths = list(endpoints)
		self.socket_path, self.tcp_ports = self.socket_paths[0], []
		self.served, self.lock = {}, threading.Lock()

	def set_model(self, name, version, model):
		state = model.state_dict() if hasattr(model, "state_dict") else model
		with self.lock:
			self.served[name] = (int(version), {key: value.detach().cpu() for key, value in state.items()})

	def drop(self, name):
		with self.lock:
			self.served.pop(name, None)

	def serving(self):
		with self.lock:
			return dict(self.served)

	def socket_for(self, index):
		return self.socket_paths[index % len(self.socket_paths)]

	def endpoints(self, host):
		return list(self.socket_paths)

	def close(self):
		pass


def parse_endpoints(text):
	"""TrainConfig.inference_endpoints, "battle=tcp:host:7901,tcp:host:7902;campaign=tcp:host:7951", as {family: [endpoints]}."""
	out = {}
	for part in filter(None, (p.strip() for p in text.split(";"))):
		family, _, addresses = part.partition("=")
		out[family.strip()] = [a.strip() for a in addresses.split(",") if a.strip()]
	return out


class _FamilyView:
	"""One family's handle on a shared InferenceServer: the same calls, this family's models."""

	def __init__(self, server, family):
		self.server, self.family = server, family
		self.model_kwargs = server.kwargs[family]
		self.socket_paths, self.socket_path, self.tcp_ports = server.socket_paths, server.socket_path, server.tcp_ports

	def set_model(self, name, version, model):
		self.server.set_model(name, version, model, family=self.family)

	def drop(self, name):
		self.server.drop(name, family=self.family)

	def serving(self):
		return self.server.serving(self.family)

	def socket_for(self, index):
		return self.server.socket_for(index)

	def endpoints(self, host):
		return self.server.endpoints(host)

	def close(self):
		pass  # (the server's owner closes it)


class InferenceClient:
	"""The protocol from Python (tests, tools): run(name, inputs) with a batch of contract inputs."""

	def __init__(self, socket_path, family="battle"):
		"""socket_path: a Unix socket, or "tcp:<host>:<port>" (compressed, as AlphaBattleAI and AlphaMapAI do over TCP)."""
		self.family = families()[family]
		self.in_layout, self.out_layout = wire_layout(self.family.inputs, True), wire_layout(self.family.outputs, False)
		self.packed = str(socket_path).startswith("tcp:")
		if self.packed:
			host, port = str(socket_path)[4:].rsplit(":", 1)
			self.conn = socket.create_connection((host, int(port)))
			self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		else:
			self.conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
			self.conn.connect(str(socket_path))
		hello = {"family": family, "contract_hash": self.family.contract_hash, **({"compress": "zlib"} if self.packed else {})}
		send_message(self.conn, json.dumps(hello).encode())
		reply = json.loads(read_message(self.conn))
		if not reply.get("ok"):
			raise RuntimeError(reply.get("error"))

	def run(self, name, inputs=None):
		"""(version, {output: array [decisions, ...]}); inputs: {name: array [decisions, ...]}, None for the version only."""
		decisions = 0 if inputs is None else len(next(iter(inputs.values())))
		rows = [b"".join(np.ascontiguousarray(np.asarray(inputs[n][d]).astype(wire)).tobytes() for n, _, wire, _, _ in self.in_layout) for d in range(decisions)]
		encoded = name.encode()
		message = bytes([len(encoded)]) + encoded + struct.pack("<I", decisions) + b"".join(rows)
		send_message(self.conn, pack(message) if self.packed else message)
		reply = read_message(self.conn)
		if self.packed:
			reply = unpack(reply)
		(version,) = struct.unpack_from("<I", reply)
		if version == ERROR:
			raise RuntimeError(reply[4:].decode())
		flat = np.frombuffer(reply[4:], dtype=np.float32).reshape(decisions, -1) if decisions else np.zeros((0, 0), np.float32)
		outputs, first = {}, 0
		for name_, shape, _, _, nbytes in self.out_layout:
			n = nbytes // 4
			outputs[name_] = flat[:, first : first + n].reshape((decisions,) + shape)
			first += n
		return version, outputs

	def close(self):
		self.conn.close()
