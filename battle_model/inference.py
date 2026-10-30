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
MODEL_WAIT = 5.0  # seconds a request for a model not served yet waits for its weights (apply_control)


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


def _serve(socket_path, device_name, family_name, model_kwargs, control, threads, tcp=None):
	"""tcp: (host, port) to listen on for clients on other machines too, or None."""
	import torch

	torch.set_num_threads(threads)
	torch_dtype = {np.dtype(t): getattr(torch, name) for t, name in ((np.uint8, "uint8"), (np.int32, "int32"), (np.float32, "float32"), (np.bool_, "bool"), (np.int64, "int64"))}
	family = families()[family_name]
	device = torch.device(device_name)
	in_layout, out_layout = wire_layout(family.inputs, True), wire_layout(family.outputs, False)
	in_bytes = sum(entry[-1] for entry in in_layout)
	offsets = np.cumsum([0] + [entry[-1] for entry in in_layout])[:-1]
	models = {}  # name -> [module, version]
	requests = queue.Queue()  # (name, decisions, payload, reply): the worker puts its reply's bytes in reply[0], sets reply[1]
	commands = queue.Queue()  # from the trainer, applied by the worker (the only thread that touches the device)

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
			if hello.get("family") != family_name or hello.get("contract_hash") != family.contract_hash:
				send_message(conn, json.dumps({"error": f"this server runs {family_name} {family.contract_hash}, the client speaks {hello}"}).encode())
				return
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
					requests.put((name, decisions, payload, box))
					box[1].wait()  # one request at a time per connection: the reply goes out before the next is read
					reply = box[0]
				send_message(conn, pack(reply) if packed else reply)
		except (ConnectionError, OSError, zlib.error):
			pass
		finally:
			conn.close()

	def apply_control(wait_for=None):
		"""Applies the trainer's queued commands. wait_for: a model name not served yet, waited for up to MODEL_WAIT
		seconds: set_model returns once the pipe has taken the weights, before control_reader has unpickled them, and a
		request sent right after it found no model (every time on Linux)."""
		deadline = time.monotonic() + MODEL_WAIT
		while True:
			try:
				if wait_for is None or wait_for in models:
					command = commands.get_nowait()
				else:
					command = commands.get(timeout=max(0.0, deadline - time.monotonic()))
			except queue.Empty:
				return
			if command[0] == "set":
				_, name, version, state = command
				if name not in models:
					models[name] = [family.build(model_kwargs).to(device).eval(), version]
				models[name][0].load_state_dict(state)
				models[name][1] = version
			elif command[0] == "drop":
				models.pop(command[1], None)

	def run(name, batch):
		"""The replies of one model's requests, run as one batch."""
		model, version = models[name]
		total = sum(decisions for _, decisions, _, _ in batch)
		data = np.concatenate([np.frombuffer(payload, dtype=np.uint8).reshape(decisions, in_bytes) for _, decisions, payload, _ in batch])
		inputs = {}
		if device.type == "cpu":
			for (input_name, shape, wire, dtype, nbytes), offset in zip(in_layout, offsets):
				column = np.ascontiguousarray(data[:, offset : offset + nbytes]).view(wire)
				inputs[input_name] = torch.from_numpy(column.astype(dtype).reshape((total,) + shape))
		else:  # the whole batch to the GPU in one copy, its columns cut and converted there (not one copy an input)
			on_device = torch.from_numpy(data).to(device)
			for (input_name, shape, wire, dtype, nbytes), offset in zip(in_layout, offsets):
				# (a copy always: one decision's slice counts as contiguous, so .contiguous() kept its odd row stride, the
				# record's byte count, and the view refused it: "stride(0) must be divisible by 4", 2026-10-30, the PC)
				column = on_device[:, offset : offset + nbytes].clone(memory_format=torch.contiguous_format).view(torch_dtype[np.dtype(wire)])
				inputs[input_name] = column.to(torch_dtype[np.dtype(dtype)]).reshape((total,) + shape)
		with torch.inference_mode():
			out = model(inputs)
			flat = torch.cat([out[o_name].float().reshape(total, -1) for o_name, *_ in out_layout], dim=1)
		outputs = flat.cpu().numpy()  # (one copy back, one sync)
		replies, first = [], 0
		for _, decisions, _, box in batch:
			replies.append((box, struct.pack("<I", version) + outputs[first : first + decisions].astype(np.float32).tobytes()))
			first += decisions
		return replies

	def worker():
		while True:
			pending = [requests.get()]
			while len(pending) < MAX_BATCH:
				try:
					pending.append(requests.get_nowait())
				except queue.Empty:
					break
			apply_control()
			by_name = {}
			for item in pending:
				by_name.setdefault(item[0], []).append(item)
			for name, batch in by_name.items():
				try:
					if name not in models:
						apply_control(wait_for=name)
					if name not in models:
						raise KeyError(f"no model named {name!r} (the server has {sorted(models)})")
					replies = [(box, struct.pack("<I", models[name][1])) for _, decisions, _, box in batch if decisions == 0]
					work = [item for item in batch if item[1] > 0]
					replies += run(name, work) if work else []
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
	and the model's many small operations), so several share the games; on a GPU one process batches them all."""

	def __init__(self, socket_path, device="cpu", family="battle", model_kwargs=None, threads=2, processes=1, tcp_host=None, tcp_port=0):
		"""tcp_host, tcp_port: also listen there for clients on other machines (worker.py), process i on tcp_port + i."""
		import multiprocessing

		base = str(socket_path)
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
			process = context.Process(target=_serve, args=(path, device, family, dict(model_kwargs or {}), child, threads, tcp), daemon=True)
			process.start()
			self.controls.append(control)
			self.processes.append(process)
		for control in self.controls:
			if not control.poll(120) or control.recv() != ("ready",):
				raise RuntimeError("the inference server did not start")

	def set_model(self, name, version, model):
		"""Serve model's current weights (a module or a state_dict) under name, as version."""
		state = model.state_dict() if hasattr(model, "state_dict") else model
		state = {key: value.detach().cpu() for key, value in state.items()}
		with self.lock:
			for control in self.controls:
				control.send(("set", name, int(version), state))

	def drop(self, name):
		with self.lock:
			for control in self.controls:
				control.send(("drop", name))

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
