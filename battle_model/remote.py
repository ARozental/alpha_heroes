"""Training on several machines (2026-10-29): the master (the machine with the GPU: the learners, the inference servers,
the evals) also trains on what workers on other machines play (worker.py). A worker runs only games: generated battles
and campaign games, every decision asked of the master's inference servers over the network (no model and no GPU on
the worker; a decision's request and reply are ~4 KB compressed), its records sent here. So the workers' cores go to
VCMI alone, the work the master's own cores are too few for once the GPU trains faster than they play.

	the master   battle_model.train / campaign_model.joint with remote_port (e.g. 7700): this Hub listens there, the
	             battle inference servers on remote_port + 1 + i, the campaign's on remote_port + 51 + i
	a worker     .venv/bin/python -m battle_model.worker --master <host>   (worker.py: its sizes, its token)

The hub's protocol, over TCP, every frame a u64 length and that many bytes:
	hello     worker: json {"protocol", "token", "name", "contracts": {family: hash}, "models": whether it serves its
	          own games}; hub: json {"ok", "number", "name", "config" (the run's TrainConfig), "battle_ports",
	          "campaign_ports", "model_kwargs": {family: kwargs}} or {"error"}. Only then pickles: a worker without the
	          run's token never gets one unpickled
	then      zlib (level 1) of a pickle, a tuple each:
	          worker: ("battle", records, source, game)   checked and compacted records (ActorPool._check), for the buffer
	                  ("campaign", key, map, players, game, info, game_map)   a finished campaign game (CampaignLearner)
	                  ("failed", kind, message)   a battle or a game that failed (the actors' failure count)
	          hub:    ("state", {...})   every STATE_SECONDS: the model's version, the league, whether to play (evals
	                  pause generated battles), whether the buffer has room for generated battles (backpressure)
	                  ("models", family, {name: (version, state) or None})   to a worker with "models": the weights
	                  the master's servers serve that changed since the last (None: dropped), before the state that
	                  names them. Its games then ask its own servers on its own cores (2026-10-30: the master's CPU
	                  servers were the limit, the Mac's cores idle at 16 ms a round trip)

The token: TrainConfig.remote_token, else runtime/remote-token (made on the master's first remote run; copy it to each
worker's runtime/). The records are the master's model versions, so the buffer's staleness rules hold for them alike.
"""
import json
import pickle
import secrets
import socket
import struct
import threading
import time
import zlib
from dataclasses import asdict
from pathlib import Path

PROTOCOL = 1
BATTLE_PORT, CAMPAIGN_PORT = 1, 51  # the inference servers' ports past the hub's: remote_port + these + process
STATE_SECONDS = 1.0
ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = ROOT / "runtime" / "remote-token"


def read_exactly(conn, n):
	data = bytearray(n)
	view, got = memoryview(data), 0
	while got < n:
		k = conn.recv_into(view[got:], n - got)
		if not k:
			raise ConnectionError("closed")
		got += k
	return data


def send_frame(conn, payload, lock=None):
	frame = struct.pack("<Q", len(payload)) + payload
	if lock is None:
		conn.sendall(frame)
	else:
		with lock:
			conn.sendall(frame)


def read_frame(conn):
	(n,) = struct.unpack("<Q", read_exactly(conn, 8))
	return read_exactly(conn, n)


def send_message(conn, message, lock=None):
	send_frame(conn, zlib.compress(pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL), 1), lock)


def read_message(conn):
	return pickle.loads(zlib.decompress(read_frame(conn)))


def token(config):
	"""The run's shared secret: config.remote_token, else TOKEN_FILE (made here if there is none)."""
	if config.remote_token:
		return config.remote_token
	if not TOKEN_FILE.exists():
		TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
		TOKEN_FILE.write_text(secrets.token_hex(16) + "\n")
		TOKEN_FILE.chmod(0o600)
	return TOKEN_FILE.read_text().strip()


def contracts():
	"""{family: contract hash} of this machine's code: a worker on other code than the master's would play wrong."""
	from campaign_model import schema as campaign_schema

	from . import schema

	return {"battle": schema.contract_hash(), "campaign": campaign_schema.contract_hash()}


class Hub:
	"""The master's side: workers connect, get the run's settings and the inference servers' ports, and send what they
	played; their battle records go to the actors' buffer (pool, actors.ActorPool) and their campaign games to the
	campaign learner (campaign, or None)."""

	def __init__(self, config, pool, campaign=None):
		self.config, self.pool, self.campaign = config, pool, campaign
		self.token = token(config)
		self.listener = socket.create_server(("0.0.0.0", config.remote_port), backlog=16)
		self.stop = threading.Event()
		self.lock = threading.Lock()
		self.workers = {}  # name -> {"address", "since", "battles", "decisions", "games"}
		self.numbers = iter(range(1, 10**6))
		threading.Thread(target=self._accept, daemon=True).start()
		print(f"remote: workers connect to port {config.remote_port} (battle inference {self.battle_ports()}, campaign {self.campaign_ports()}; token in {TOKEN_FILE if not config.remote_token else 'remote_token'})", flush=True)

	def battle_ports(self):
		return list(self.pool.server.tcp_ports) if self.pool.server else []

	def campaign_ports(self):
		return list(self.campaign.server.tcp_ports) if self.campaign else []

	def servers(self):
		"""{family: the master's inference server} (those a worker can mirror)."""
		found = {"battle": self.pool.server, "campaign": self.campaign.server if self.campaign else None}
		return {family: server for family, server in found.items() if server is not None and hasattr(server, "serving")}

	def close(self):
		self.stop.set()
		self.listener.close()

	def state(self):
		pool, campaign = self.pool, self.campaign
		with pool.lock:
			room = pool.generated_waiting() < pool.generated_room()
		return {
			"version": pool.version,
			"running": pool.running.is_set(),
			"league": pool.past_versions(),
			"room": room,
			"campaign_version": campaign.version if campaign else 0,
			"campaign_running": campaign.running.is_set() if campaign else False,
		}

	def _accept(self):
		while not self.stop.is_set():
			try:
				conn, address = self.listener.accept()
			except OSError:
				return
			conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
			threading.Thread(target=self._serve, args=(conn, address), daemon=True).start()

	def _serve(self, conn, address):
		name, lock = None, threading.Lock()
		try:
			conn.settimeout(30)
			hello = json.loads(read_frame(conn))
			problem = None
			if hello.get("protocol") != PROTOCOL:
				problem = f"protocol {hello.get('protocol')}, the master speaks {PROTOCOL}"
			elif not secrets.compare_digest(str(hello.get("token", "")), self.token):
				problem = "a wrong token (the master's runtime/remote-token)"
			elif hello.get("contracts") != contracts():
				problem = f"other models' contracts than the master's (pull the master's code): {hello.get('contracts')} vs {contracts()}"
			if problem:
				send_frame(conn, json.dumps({"error": problem}).encode())
				print(f"remote: refused {address[0]}: {problem}", flush=True)
				return
			name = f"w{next(self.numbers)}-{str(hello.get('name', address[0]))[:24]}"
			model_kwargs = {family: server.model_kwargs for family, server in self.servers().items()}
			send_frame(conn, json.dumps({"ok": True, "number": int(name[1 : name.index("-")]), "name": name, "config": asdict(self.config), "battle_ports": self.battle_ports(), "campaign_ports": self.campaign_ports(), "model_kwargs": model_kwargs}).encode())
			conn.settimeout(None)
			with self.lock:
				self.workers[name] = {"address": address[0], "since": time.time(), "battles": 0, "decisions": 0, "games": 0}
			print(f"remote: worker {name} ({address[0]}) joined: {hello.get('sizes', {})}", flush=True)
			threading.Thread(target=self._send_states, args=(conn, lock, bool(hello.get("models"))), daemon=True).start()
			while not self.stop.is_set():
				message = read_message(conn)
				kind = message[0]
				if kind == "battle":
					_, records, source, game = message
					self.pool._add(records, source, f"{name}:{game}", checked=True)
					with self.lock:
						self.workers[name]["battles"] += 1
						self.workers[name]["decisions"] += sum(r.decisions for r in records)
				elif kind == "campaign" and self.campaign:
					_, key, map_name, players, game, info, game_map = message
					self.campaign._game_done(key, map_name, players, game, info, game_map)
					with self.lock:
						self.workers[name]["games"] += 1
				elif kind == "failed":
					_, what, text = message
					if what == "campaign" and self.campaign:
						self.campaign._failed(f"{name}: {text}")
					else:
						self.pool._failed(f"{name}: {text}")
		except (ConnectionError, OSError, EOFError, ValueError, pickle.UnpicklingError, zlib.error) as error:
			if name:
				with self.lock:
					done = self.workers.get(name, {})
				hours = (time.time() - done.get("since", time.time())) / 3600
				print(f"remote: worker {name} left ({str(error)[:200]}) after {hours:.2f} h: {done.get('battles', 0)} battles, {done.get('decisions', 0)} decisions, {done.get('games', 0)} campaign games", flush=True)
		finally:
			conn.close()
			if name:
				with self.lock:
					self.workers.pop(name, None)

	def _send_states(self, conn, lock, models=False):
		sent = {}  # (family, name) -> the version this worker has
		while not self.stop.is_set():
			try:
				if models:
					for family, server in self.servers().items():
						serving = server.serving()
						changes = {name: (version, state) for name, (version, state) in serving.items() if sent.get((family, name)) != version}
						changes.update({name: None for f, name in list(sent) if f == family and name not in serving})
						if changes:
							send_message(conn, ("models", family, changes), lock)
							for name, item in changes.items():
								if item is None:
									sent.pop((family, name), None)
								else:
									sent[(family, name)] = item[0]
				send_message(conn, ("state", self.state()), lock)
			except OSError:
				return
			time.sleep(STATE_SECONDS)
