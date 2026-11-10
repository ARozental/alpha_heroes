"""Training data as a DataLoader makes it (the user, 2026-10-30: "all map creations triggered by workers from an
IterableDataLoader ... the PyTorch way to ensure maximum efficiency for the GPU"): the games are the dataset, a
DataLoader's worker processes play them and turn their records into training arrays, and the trainer pulls what it
will train on, so no game is played to be dropped and none of this work shares the trainer's interpreter lock
(2026-10-30, the PC: threads beside the trainer's eager launches slowed its step 12x).

	Games          an IterableDataset of one kind: generated battles ("battle") or campaign games ("campaign"). Each
	               worker (num_workers of them) plays one game at a time and yields what it made: for a battle, each
	               side's item (its arrays, its summary, its failure counts); for a campaign game, every battle's item
	               as it ends (source "real": the battle learner trains on them fresh) and then the game itself
	where          before each game a worker decides where it plays: here while this machine's games of the kind are
	               under its ceiling (the machine config's count) and memory is left (MEMORY_FLOOR_GB), else on a
	               remote machine's slot (a worker there holds a connection open on remote_port + SLOT_PORT until it is
	               given a game: worker.py --slots), else it waits. So the local cores are used first and the remote
	               ones for the shortfall, and the remote plays exactly the games asked of it
	throttle       generated battles: the loader's prefetch (prefetch_factor 1: a worker that has yielded one battle
	               plays no other until the trainer took it) and the trainer's room for them (ActorPool.generated_room:
	               what the next `ahead` updates take of generated battles, less than that while campaign battles are
	               plenty). Campaign games cannot wait mid-game (their battles must flow), so they are throttled at
	               the start: a worker starts one with a permit, which the trainer gives while the games finished and
	               in flight hold less than `ahead` updates' worth of decisions (Supply.allow)
	Supply         the trainer's side: the loaders, their threads that pull items into the learners (ActorPool's
	               buffer, CampaignLearner's finished games), the shared counters, the listener for remote slots

Items are plain dicts of numpy arrays and numbers (pickled by value between the processes; a battle's is a few MB).
"""
import json
import math
import multiprocessing
import os
import random
import secrets
import shutil
import signal
import socket
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from . import forkserver, generator, play, remote
from .constants import MAX_UNITS, N_HEXES, N_VALUE_HORIZONS, ActionType
from .data import SPARSE_INPUTS, Sparse, battle_examples, check_record, collate, compact_record, version_of

KEPT_RUNS = deque(maxlen=9)  # (the VCMI logs of clean games with actions that changed nothing: the newest 8 a process)
from .losses import normalize_advantages

MEMORY_FLOOR_GB = 4.5  # no new game here under this much available memory (Linux; elsewhere unknown: no limit). Games grow after they start (a VCMI client, the worker's arrays), and runtime/memwatch.sh stops a run under 1.5 GB: it did, 2026-10-30 23:27. The trainer's own swing is kept free besides (memory_allows); what still runs past the floor the guard sheds a game at a time (runtime/memwatch.sh). (6 GB on 2026-11-01 13:38-15:00 CST deadlocked joint7: the trainer at its peak, 5.6 GB left, no game could start, and it waited for games)
LAST_GAME_GB = 3.5  # one game of a kind plays while this much is available, whatever the floor says: the run never waits for memory only games would free (over the guard's shedding level, 3 GB)
SLOT_HELLO_SECONDS = 30
NEUTRAL_AI = "BattleAI"  # the wandering monsters' battle AI in generated battles, as in a campaign game
TRACE_EVERY = 20  # worker 0 keeps the compact record of one battle in this many, for the run's traces


def available_gb():
	"""Memory available for new processes (GB; MemAvailable on Linux), or None where unknown."""
	try:
		with open("/proc/meminfo") as f:
			for line in f:
				if line.startswith("MemAvailable:"):
					return int(line.split()[1]) / 2**20
	except OSError:
		pass
	return None


def inference_ready(address, family):
	"""Whether the inference server at address (a Unix socket or "tcp:<host>:<port>") answers for its current model.
	A campaign game asking a server that was down ended every turn unasked and reached its day cap with no decision:
	2,042 of joint5's games before 2026-10-31, each time the Mac's servers were down; and starting one in its place every
	few seconds made thousands."""
	from .inference import InferenceClient

	before = socket.getdefaulttimeout()
	socket.setdefaulttimeout(5)
	try:
		client = InferenceClient(address, family=family)
		try:
			client.run("current")
		finally:
			client.close()
		return True
	except Exception:
		return False
	finally:
		socket.setdefaulttimeout(before)


def process_gb(pid, field):
	"""A process's VmRSS or VmHWM (its peak resident memory) in GB on Linux, else None."""
	try:
		with open(f"/proc/{pid}/status") as status:
			return next(int(line.split()[1]) for line in status if line.startswith(field + ":")) / 2**20
	except (OSError, StopIteration):
		return None


SHARED = None  # the trainer's Shared (Supply sets it): memory_allows() in the trainer's process (the campaign eval)
PEAK_WINDOW = 20 * 60  # s: the trainer's recent peak, over 2-3 campaign updates (each ~5-9 minutes apart)
PEAK_EVERY = 5  # s between its samples (an update's peak lasts minutes)
PERMITS_EVERY = 10  # s: the campaign permits counted again at least this often (Supply._keep_allowing)
CENSUS_EVERY = 120  # s between the trainer's memory census lines (Supply.memory_census)
CENSUS_TYPES_EVERY = 600  # s between its counts of objects by type (gc.get_objects: seconds on a big heap)


def held_bytes(obj, depth=0):
	"""The bytes of the arrays and tensors in obj (dicts, lists, tuples, objects' attributes and slots, a few levels)."""
	if obj is None or depth > 6:
		return 0
	if isinstance(obj, np.ndarray):
		return obj.nbytes
	if isinstance(obj, torch.Tensor):
		return obj.element_size() * obj.nelement()
	if isinstance(obj, (bytes, bytearray)):
		return len(obj)
	if isinstance(obj, dict):
		return sum(held_bytes(v, depth + 1) for v in obj.values())
	if isinstance(obj, (list, tuple)):
		return sum(held_bytes(v, depth + 1) for v in obj)
	slots = getattr(type(obj), "__slots__", ())
	if slots:
		return sum(held_bytes(getattr(obj, s, None), depth + 1) for s in slots)
	if hasattr(obj, "__dict__"):
		return sum(held_bytes(v, depth + 1) for v in vars(obj).values())
	return 0


def memory_allows(shared=None, playing=None):
	"""Whether a game may start here: the memory available, less the trainer's way back to its recent peak, over the
	floor. The trainer's memory swings ~5 GB each campaign update (it frees the update's buffers after it, takes them
	back in the next); games let in at its low point took what the next update needed, and the guard stopped joint7 at
	1.06 GB (2026-11-01 12:21 CST: the trainer 9.6 -> 13.2 GB in update 49, the unit 18.9 -> 25.1 GB in 4.5 minutes).
	Its recent peak (Supply._watch_trainer: the highest of the last PEAK_WINDOW), not its highest ever: much of that is
	the finished games it holds, which only games played bring back; kept free after a restart (14.3 GB, the trainer
	at 3.9), it let 1 game of 14 play (13:38-13:55 CST). playing: games of the kind playing here now (none: LAST_GAME_GB
	is enough)."""
	free = available_gb()
	if free is None:
		return True
	if playing == 0 and free >= LAST_GAME_GB:
		return True
	shared = shared or SHARED
	rss = process_gb(shared.trainer, "VmRSS") if shared is not None else None
	if rss is None:
		return free >= MEMORY_FLOOR_GB
	return free - max(0.0, shared.peak.value - rss) >= MEMORY_FLOOR_GB


def peer_closed(conn):
	"""Whether the other end of a connection has closed it (nothing more to read, and it said so). Without waiting: a
	socket with a timeout would wait that long for data first."""
	timeout = conn.gettimeout()
	conn.setblocking(False)
	try:
		return conn.recv(1, socket.MSG_PEEK) == b""
	except BlockingIOError:
		return False
	except OSError:
		return True
	finally:
		conn.settimeout(timeout)


# ---- what a worker makes of a battle ----


def prepare(record, source, key, config, keep_record=False):
	"""(item, None) for one side's record of a battle, or (None, why) for one that is never trained on (an error,
	an unsound record). The item: "data", the battle's training arrays as data.collate(widen=False) gives them (numpy,
	the forced decisions left out), "summary" (data.BattleSummary), "failures" (train.failure_counts), "actions" (the
	count of each action type), "version" (the model that played), "record" (compact, for traces, when asked)."""
	from .train import failure_counts

	error = next((e for e in record.events if e.get("event") == "error"), None)
	if error:
		return None, f"FAILURE: a {source} battle had an error ({key}): {error.get('message', '')[:300]}"
	problem = check_record(record)
	if problem:
		return None, f"FAILURE: a {source} battle's record is unsound ({key}): {problem[:300]}"
	examples, summary = battle_examples(record, config.reward, config.advantage)
	chosen = [e for e in examples if not e["forced"]]
	if config.examples_per_battle and len(chosen) > config.examples_per_battle:  # (TrainConfig.examples_per_battle)
		chosen = [chosen[i] for i in sorted(random.sample(range(len(chosen)), config.examples_per_battle))]
	item = {
		"kind": "battle", "source": source, "game": key, "version": version_of(record), "decisions": len(chosen), "summary": summary,
		"failures": failure_counts([record]), "actions": Counter(ActionType(a["type"]).name for a in record.actions), "data": None, "record": None,
	}
	if chosen:
		batch = collate(chosen, widen=False)
		if "foresight" not in batch:  # (a battle of one decision: no next one; the same keys as the others')
			n = len(chosen)
			batch["foresight"] = {"dies": torch.zeros(n, MAX_UNITS), "lost": torch.zeros(n, MAX_UNITS), "dealt": torch.zeros(n, MAX_UNITS), "unit_mask": torch.zeros(n, MAX_UNITS), "hex_enemy": torch.zeros(n, N_HEXES), "valid": torch.zeros(n)}
		batch.pop("search", None)
		data = _numpy(batch)
		for name in SPARSE_INPUTS:  # the big, mostly zero tables kept sparse while the item waits (data.compact_record's lesson: ~10x smaller)
			if name in data["inputs"]:
				data["inputs"][name] = Sparse(data["inputs"][name])
		item["data"] = data
	if keep_record:
		item["record"] = compact_record(record)
	return item, None


def _numpy(value):
	if isinstance(value, dict):
		return {k: _numpy(v) for k, v in value.items()}
	return value.numpy() if torch.is_tensor(value) else value


def _tensors(value):
	if isinstance(value, dict):
		return {k: _tensors(v) for k, v in value.items()}
	if isinstance(value, torch.Tensor):  # (dense_on's, already on its device)
		return value
	return torch.from_numpy(np.ascontiguousarray(value))


def dense_on(values, device):
	"""The items' arrays of one input (Sparse or not) joined along their rows, built on device: a Sparse one's nonzero
	entries go there and are placed in the zeros there, so the dense batch never exists on the CPU (2026-11-01: an
	update's 6,400 battle decisions made densely on the CPU were a ~3-4 GB swing of the trainer every epoch, 0.3 GB sparse)."""
	first = values[0]
	rest = tuple(first.shape[1:])
	rows = [v.shape[0] for v in values]
	dtype = torch.from_numpy(np.zeros(0, first.dtype)).dtype
	out = torch.zeros((sum(rows),) + rest, dtype=dtype, device=device)
	flat, per_row, offset = out.view(-1), int(np.prod(rest)) if rest else 1, 0
	for v, n in zip(values, rows):
		if isinstance(v, Sparse):
			if len(v.index):
				where = torch.from_numpy(v.index.astype(np.int64)).to(device) + offset * per_row
				flat[where] = True if v.values is None else torch.from_numpy(np.ascontiguousarray(v.values)).to(device)
		else:
			out[offset : offset + n] = torch.from_numpy(np.ascontiguousarray(v)).to(device)
		offset += n
	return out


def assemble(items, advantage_config, device=None):
	"""One update's batch (as data.collate(widen=False) gives it, the int64 inputs int32) from the battle items taken, the
	advantages normalised per side and source over the update (data.make_dataloader: the campaign games' battles got 50%
	larger advantages than the generated ones' when normalised together); and the items' summaries, in order. device:
	the sparse inputs made dense there (dense_on), not on the CPU; the rest as before (moved by the caller)."""
	parts = [item["data"] for item in items if item["data"]]
	if not parts:
		return None, [item["summary"] for item in items]

	def join(values):
		if isinstance(values[0], dict):
			return {k: join([v[k] for v in values]) for k in values[0]}
		if device is not None and any(isinstance(v, Sparse) for v in values):
			return dense_on(values, device)
		return np.concatenate([v.dense() if isinstance(v, Sparse) else v for v in values])

	batch = _tensors(join(parts))
	rows = [item["decisions"] for item in items if item["data"]]
	groups = torch.cat([torch.full((n,), item["summary"].our_side + 2 * (item["source"] == "real")) for n, item in zip(rows, (i for i in items if i["data"]))])
	battles = torch.cat([torch.full((n,), i) for i, n in enumerate(rows)])
	batch["advantage"] = normalize_advantages(batch["advantage"], groups, battles, advantage_config)
	batch["real"] = (groups >= 2).float()  # (each decision's source, for the fresh value loss by source)
	return batch, [item["summary"] for item in items]


# ---- the games, played by the loader's workers ----


class Shared:
	"""Counters the workers and the trainer share (the loader's context: inherited by the workers)."""

	def __init__(self, ctx, ceilings, servers=1):
		self.lock = ctx.Lock()
		self.trainer = os.getpid()  # (memory_allows: its recent peak is kept free)
		self.peak = ctx.Value("d", 0.0, lock=False)  # the trainer's highest memory in the last PEAK_WINDOW (GB)
		# Eval games waiting for memory (evaluate.py): no training game starts here meanwhile, or the training's took each
		# freed GB first and eval 53's 48 games waited 17 minutes without one starting (2026-11-01 15:20-15:37 CST)
		self.eval_waiting = ctx.Value("i", 0, lock=False)
		self.ceilings = dict(ceilings)  # kind -> games of it at once on this machine
		self.local = {kind: ctx.Value("i", 0, lock=False) for kind in ceilings}  # playing here now
		self.remote = {kind: ctx.Value("i", 0, lock=False) for kind in ceilings}  # playing on remote slots now
		self.permits = ctx.Value("i", 0, lock=False)  # campaign games the trainer lets start (Supply.allow)
		self.started = ctx.Value("i", 0, lock=False)  # campaign games started (in flight = this - what the trainer got)
		self.version = ctx.Value("i", 0, lock=False)  # the battle model's version (the league's draw: no snapshot of it)
		# The credit the trainer set last (Supply.publish): the window credit's outcome share, then the policy heads' weights,
		# both following the value head's quality (train.py); unset (< 0) before its first update: the config's own
		self.credit = ctx.Array("d", [-1.0] * (2 + N_VALUE_HORIZONS), lock=False)
		self.server_games = ctx.Array("i", max(servers, 1), lock=False)  # this machine's games on each inference process now (Games.take_server)
		self.stop = ctx.Event()


class Games(IterableDataset):
	"""One kind of game ("battle": generated battles; "campaign": campaign games), each worker playing one at a time,
	here or on a remote slot (the module's docstring). What a worker yields: items (prepare), campaign games
	({"kind": "campaign", ...}), and {"kind": "failed", ...} for a game that failed (counted, never trained on)."""

	def __init__(self, kind, config, run_dir, battle_sockets, campaign_sockets, shared, token, listener=None):
		"""battle_sockets, campaign_sockets: the inference servers' addresses the games here ask (a Unix socket a server
		process, or "tcp:<host>:<port>" of another machine's: inference.ModelStore), spread over the workers."""
		self.kind, self.config, self.run_dir = kind, config, Path(run_dir)
		self.battle_sockets, self.campaign_sockets = [str(s) for s in battle_sockets], [str(s) for s in campaign_sockets or ()]
		self.shared, self.token, self.listener = shared, token, listener
		self.link = self.run_dir / "models" / "current.onnx"  # (named in the env, never read: the servers play)

	# -- where a game plays --

	def place(self):
		"""("here", None), ("remote", connection) or None (nothing free now): local first, then a remote slot."""
		shared, kind = self.shared, self.kind
		if (self.run_dir / "drain").exists():  # (runtime/drain_restart.sh: the games playing finish, no new one starts)
			return None
		with shared.lock:
			# (while an eval game waits for memory only a kind's last game starts: memory_allows' LAST_GAME_GB, so the
			# trainer, waiting for games, never waits on the eval)
			playing = shared.local[kind].value + shared.remote[kind].value
			if shared.local[kind].value < shared.ceilings[kind] and (shared.eval_waiting.value == 0 or playing == 0) and memory_allows(shared, playing):
				shared.local[kind].value += 1
				return "here", None
		# A remote slot's game takes little memory here (its worker exists already and holds only what it played, 20-130
		# MB), so it waits only for LAST_GAME_GB, not the floor with the trainer's peak kept free: under that the Mac's 4
		# slots played ~6 games in 15 minutes while 11 permits were open (2026-11-02 03:10-03:26 CST)
		free = available_gb()
		if self.listener is not None and (free is None or free >= LAST_GAME_GB):
			conn = self._accept_slot()
			if conn is not None:
				with shared.lock:
					shared.remote[kind].value += 1
				return "remote", conn
		return None

	def release(self, where, conn):
		with self.shared.lock:
			(self.shared.local if where == "here" else self.shared.remote)[self.kind].value -= 1
		if conn is not None:
			conn.close()

	def _accept_slot(self):
		"""A remote slot's connection (its hello checked), or None when none waits."""
		while True:
			try:
				self.listener.settimeout(0.5)
				conn, address = self.listener.accept()
			except (socket.timeout, OSError):
				return None
			conn = self._slot_hello(conn)
			if conn is not None:
				return conn

	def _slot_hello(self, conn):
		"""The connection, its hello checked and answered; None if it was refused or its slot is gone."""
		try:
			conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
			conn.settimeout(SLOT_HELLO_SECONDS)
			hello = json.loads(remote.read_frame(conn))
			if peer_closed(conn):  # a slot that gave up waiting, still queued here with its hello: the next one
				conn.close()
				return None
			problem = None
			if hello.get("protocol") != remote.PROTOCOL:
				problem = f"protocol {hello.get('protocol')}, the master speaks {remote.PROTOCOL}"
			elif not secrets.compare_digest(str(hello.get("token", "")), self.token):
				problem = "a wrong token (the master's runtime/remote-token)"
			elif hello.get("contracts") != remote.contracts():
				problem = f"other models' contracts than the master's (pull the master's code): {hello.get('contracts')} vs {remote.contracts()}"
			if problem:
				remote.send_frame(conn, json.dumps({"error": problem}).encode())
				conn.close()
				return None
			remote.send_frame(conn, json.dumps({"ok": True}).encode())
			conn.settimeout(None)
			return conn
		except (OSError, ValueError):
			conn.close()
			return None

	def permit(self):
		"""Campaign games start with a permit (Supply.allow gives them); generated battles always (the loader's prefetch
		is their throttle)."""
		if self.kind != "campaign":
			return True
		with self.shared.lock:
			if self.shared.permits.value <= 0:
				return False
			self.shared.permits.value -= 1
			self.shared.started.value += 1
			return True

	# -- the games --

	def __iter__(self):
		info = get_worker_info()
		index = info.id if info else 0
		# (and the battle model's version at the worker's start: seeded by its index alone, each worker played the same
		# battles and games again after every restart, w2_g2_11 among them five times in a day, 2026-11-05)
		rng = random.Random(1_000_003 * (self.config.seed + 1) + 7919 * index + (104_729 * self.config.campaign.seed if self.kind == "campaign" else 0) + 15_485_863 * self.shared.version.value)
		runner = Runner(self.config, self.run_dir, self.battle_sockets[index % len(self.battle_sockets)], self.campaign_sockets[index % len(self.campaign_sockets)] if self.campaign_sockets else None, self.link, f"w{index}", self.shared.version, self.shared.credit)
		stop = self.shared.stop
		n = 0
		unreachable = 0.0  # (when this worker last said its campaign inference was unreachable)
		try:
			while not stop.is_set():
				if self.kind == "campaign" and not self.permits_left():
					time.sleep(0.5)
					continue
				if self.kind == "campaign" and runner.campaign_socket and not inference_ready(runner.campaign_socket, "campaign"):
					if time.monotonic() - unreachable > 60:
						unreachable = time.monotonic()
						print(f"supply: FAILURE: campaign inference unreachable at {runner.campaign_socket}: no campaign game starts until it answers", flush=True)
					time.sleep(5)
					continue
				placed = self.place()
				if placed is None:
					time.sleep(0.5)
					continue
				where, conn = placed
				if not self.permit():  # (taken once a place is found: a permit held while looking counted as a game in flight)
					self.release(where, conn)
					time.sleep(0.5)
					continue
				n += 1
				spec = runner.sample_battle(rng, index, n) if self.kind == "battle" else runner.sample_campaign(rng, index, n)
				slot = self.take_server() if where == "here" else None
				if slot is not None:  # (both models' processes are the same: a game's battles go where its decisions do)
					runner.battle_socket = self.battle_sockets[slot % len(self.battle_sockets)]
					if self.campaign_sockets:
						runner.campaign_socket = self.campaign_sockets[slot % len(self.campaign_sockets)]
				try:
					if where == "here":
						yield from (runner.battle(spec, keep_record=index == 0 and n % TRACE_EVERY == 0) if self.kind == "battle" else runner.campaign(spec, stop))
					else:
						yield from relay(conn, spec)
				except Exception as error:  # one failed game does not stop the worker
					# (a campaign game lost to an error yields no game item: "lost" lets the trainer count it as over)
					yield {"kind": "failed", "what": self.kind, "message": f"{spec.get('key', '?')}: {str(error)[:400]}", "lost": self.kind == "campaign"}
				finally:
					if slot is not None:
						self.release_server(slot)
					self.release(where, conn)
					trim_memory()
		finally:
			runner.close()

	def take_server(self):
		"""The inference process with the fewest of this machine's games now (Shared.server_games), counted one more
		until release_server. By the worker's own index (index % processes) two games often shared a process while others
		idled: 2026-11-05, 14 games and 12 processes, 9 busy 76-100% and 3 idle, the games waiting on them."""
		with self.shared.lock:
			games = self.shared.server_games
			slot = min(range(len(games)), key=games.__getitem__)
			games[slot] += 1
		return slot

	def release_server(self, slot):
		with self.shared.lock:
			self.shared.server_games[slot] -= 1

	def permits_left(self):
		with self.shared.lock:
			return self.shared.permits.value > 0


def relay(conn, spec):
	"""A game on a remote slot: the spec sent, its items read back until the slot says it is done."""
	remote.send_message(conn, ("play", spec))
	while True:
		message = remote.read_message(conn)
		if message[0] == "done":
			return
		if message[0] == "item":
			yield message[1]
		elif message[0] == "failed":
			raise RuntimeError(f"on the remote slot: {message[1]}")


class Runner:
	"""Plays games on this machine (a worker's, or a remote slot's: worker.py) and yields their items: one battle
	process for the generated battles (play.BattleProcess), one record server for the campaign games' battles."""

	def __init__(self, config, run_dir, battle_socket, campaign_socket, link, name, version=None, credit=None):
		self.config, self.run_dir, self.name = config, Path(run_dir), name
		self.battle_socket, self.campaign_socket, self.link = battle_socket, campaign_socket, link
		self.version = version  # a shared Value (the battle model's), or None
		self.credit = credit  # Shared.credit, or None (a remote slot: the config's own)
		self.process = None  # play.BattleProcess, made on the first battle
		self.records = None  # play.RecordServer for the campaign games' battles
		self.game = None  # the campaign game running (play.GameRun), for close()

	def advantage_config(self):
		"""The config with the credit the trainer uses now (Shared.credit). Until 2026-11-05 the workers kept the config's
		own: the outcome share the trainer logged as following the value head stayed at its floor here."""
		credit = list(self.credit) if self.credit is not None else [-1.0]
		if credit[0] < 0:
			return self.config
		return replace(self.config, advantage=replace(self.config.advantage, outcome_share=credit[0], policy_heads=tuple(credit[1:])))

	def close(self):
		if self.game is not None:
			self.game.terminate()
		if self.process is not None:
			self.process.close()
		if self.records is not None:
			self.records.__exit__(None, None, None)

	def past_versions(self):
		"""The league's past versions (server model names, "v350"): the run's snapshots, not the one being trained."""
		version = self.version.value if self.version is not None else 10**9
		return [f"v{int(p.stem.split('_')[1])}" for p in sorted((self.run_dir / "models").glob("snapshot_*.pt")) if 0 < int(p.stem.split("_")[1]) < version]

	def sample_battle(self, rng, index, n):
		"""One generated battle's spec: its scenario, the league's draw (actors.py: self-play, a past version, or a fixed
		AI), our side, its seeds."""
		config = self.config
		scenario = generator.sample(rng)
		self_play, past, fixed = config.league
		roll, past_versions = rng.random() * (self_play + past + fixed), self.past_versions()
		if roll < self_play or (roll < self_play + past and not past_versions):
			opponent, past_model = play.SELF_PLAY, None
		elif roll < self_play + past:
			opponent, past_model = "AlphaBattleAI", rng.choice(past_versions)
		else:
			opponent, past_model = rng.choice(config.league_ais), None
		our_side = 0 if opponent == play.SELF_PLAY else rng.randrange(2)
		if "neutrals" in scenario:  # wandering monsters are the game's AI, never ours (campaign_model/play.py: neutral_ai)
			opponent, past_model, our_side = NEUTRAL_AI, None, 0
		return {"kind": "battle", "key": f"{self.name}_g{index}_{n}", "scenario": scenario, "opponent": opponent, "past": past_model, "our_side": our_side, "sampler_seed": rng.randrange(2**31)}

	def sample_campaign(self, rng, index, n):
		"""One campaign game's spec: its players, size and seed, one of the game's own maps or a generated one, its day
		cap (learner.draw_days), its seeds. Self-play only: never against Nullkiller2, the evaluation's opponent."""
		from campaign_model.learner import draw_days
		from campaign_model.maps import GAME_MAPS

		config = self.config.campaign
		players, size, seed = rng.choice(config.players), rng.choice(config.sizes), rng.randrange(1, 2**31)
		game_map, days = rng.random() < config.game_map_share, draw_days(rng, config)
		if config.big_game_every > 0 and n % config.big_game_every == 0:  # (a big map now and then: docs/map_size_plan.md)
			size, game_map = config.big_game_size, False
		spec = {"kind": "campaign", "key": f"{self.name}_c{index}_{n}", "players": players, "size": size, "seed": seed, "days": days, "map": None}
		if game_map:
			spec["map"], spec["players"] = rng.choice(sorted(GAME_MAPS.items()))
			spec["size"] = 36
		return spec

	def battle(self, spec, keep_record=False):
		"""Plays one generated battle here; yields its items (one a recorded side)."""
		from .train import generated_games

		config = self.advantage_config()
		if self.process is None:
			self.process = play.BattleProcess()
		scenario, opponent, our_side = spec["scenario"], spec["opponent"], spec["our_side"]
		games, files = generated_games([scenario], spec["key"], config, our_sides=(our_side,), opponent=opponent, temperature=1.0)
		try:
			game = replace(games[0], settings=games[0].settings + (("ALPHA_BATTLE_INFERENCE_SOCKET", self.battle_socket),))
			if spec["past"] is not None:  # the other side is our AI with an older model, not recorded
				game = replace(game, settings=game.settings + ((f"ALPHA_BATTLE_INFERENCE_MODEL_{game.colors[1 - our_side]}", str(spec["past"])),))
			game = replace(game, sampler_seed=spec["sampler_seed"])
			try:
				records = self.process.play(game, self.link)
			except RuntimeError:  # its scenario kept beside the client's folder, which play kept (BattleProcess.close_process)
				keep_scenario(files[0], spec["key"])
				raise
		finally:
			for path in files:
				path.unlink(missing_ok=True)
		for record in records:
			item, problem = prepare(record, "generated", spec["key"], config, keep_record)
			yield item if item else {"kind": "failed", "what": "battle", "message": problem}

	def campaign_settings(self, key, seed):
		"""The env of a campaign game's battles: every player's battle played by the current battle model and recorded
		here as a real game's (under "<key>:<COLOR>"; the wandering monsters are BattleAI)."""
		from campaign_model.play import COLORS

		settings = {
			"ALPHA_BATTLE_INFERENCE_SOCKET": self.battle_socket,
			"ALPHA_BATTLE_MODEL": str(Path(self.link).absolute()),
			"ALPHA_BATTLE_TEMPERATURE": "1",
			"ALPHA_BATTLE_RECORD_SOCKET": str(self.records.path),
			"ALPHA_BATTLE_REQUIRE_MODEL": "1",  # a battle without its model fails the game, never plays defending
		}
		for i, color in enumerate(COLORS):
			settings[f"ALPHA_BATTLE_RECORD_KEY_{color}"] = f"{key}:{color}"
			settings[f"ALPHA_BATTLE_SEED_{color}"] = str(seed * 101 + i)
		if self.config.campaign.nk2_advice > 0:  # (a training game: Nullkiller2's advice on some of our decisions)
			settings["ALPHA_MAP_NK2_ADVICE"] = str(self.config.campaign.nk2_advice)
			if self.config.campaign.nk2_gather_weight > 0:  # (its visits again of a town as our GATHER's label: Advice.cpp visitAgain)
				settings["ALPHA_MAP_GATHER_ADVICE"] = "1"
		return tuple(settings.items())

	def campaign(self, spec, stop=None):
		"""Plays one campaign game here (self-play, every player's battle by the battle model); yields each battle's
		item as it ends, then the game: {"kind": "campaign", "key", "map", "players", "game" (records.Game), "info"
		(play.GameInfo), "game_map"}."""
		from campaign_model.play import GameRun, make_map, time_scale
		from battle_model import play as battle_play
		import shutil

		config = self.config.campaign
		if self.records is None:
			self.records = play.RecordServer(concurrent=True)
		key, players, seed = spec["key"], spec["players"], spec["seed"]
		game_map = spec["map"] is not None
		map_name = spec["map"] if game_map else make_map(seed, players, spec["size"], config.levels, config.water)[0]
		try:
			self.game = GameRun(
				self.campaign_socket, map_name, spec["days"], seed, max_seconds=config.game_seconds * time_scale(spec["size"]), battle_ai="AlphaBattleAI", game_key=key,
				settings=self.campaign_settings(key, seed), niceness=config.niceness, temperature=config.temperature,
			)
			try:
				while not self.game.poll(stop):
					yield from self._battles(key)
					time.sleep(0.25)
				game, info = self.game.result()
			finally:
				# Ended first if still running: an exception here (the worker's SIGTERM in the sleep, a remote slot's
				# send failing) left the game unreferenced, so close() could not end its VCMI (2026-10-31 review)
				self.game.terminate()
				self.game = None
			yield from self._battles(key)
		finally:
			if not game_map:
				(battle_play.USER_DIR / (map_name + ".vmap")).unlink(missing_ok=True)
		game.days = spec["days"]
		ok = info.status.startswith("stopped") or "exit code 0" in info.status  # (at the cap, or training is over)
		if ok and any(done.get("failed_actions", 0) or done.get("errors", 0) for done in game.done.values()):
			# A clean game whose AI made an action that changed nothing (or logged an error): its VCMI logs kept, the
			# server's reason in them (2026-11-05: CAST changed nothing 2,110 times in joint7, BUILD 569, reasons lost
			# with the logs), the newest KEPT_RUNS of this process's
			KEPT_RUNS.append(info.run_dir)
			while len(KEPT_RUNS) > KEPT_RUNS.maxlen - 1:
				shutil.rmtree(KEPT_RUNS.popleft(), ignore_errors=True)
		elif ok:
			shutil.rmtree(info.run_dir, ignore_errors=True)
		else:  # its records up to then are still used: the labels take the last day it reached as its end
			yield {"kind": "failed", "what": "campaign", "message": f"game {key} ({map_name}, {players} players): {info.status} (run {info.run_dir})", "run_dir": info.run_dir}
		yield {"kind": "campaign", "key": key, "map": map_name, "players": players, "game": game, "info": info, "game_map": game_map}

	def _battles(self, key):
		"""The campaign game's battles recorded so far, each as an item."""
		from .data import parse_record

		for record_key, records in self.records.drain().items():
			for data in records:
				# config.real_keep: the battle learner gets this share of the campaign's battles, chosen at random before
				# anything is done with them (2026-11-02, the user: 90% fewer battles, the campaign's share of them fixed)
				if self.config.real_keep < 1.0 and random.random() >= self.config.real_keep:
					continue
				try:
					record = parse_record(data)
				except Exception as error:
					yield {"kind": "failed", "what": "battle", "message": f"FAILURE: unreadable record from {record_key}: {error}"}
					continue
				item, problem = prepare(record, "real", record_key.split(":", 1)[0], self.advantage_config())
				yield item if item else {"kind": "failed", "what": "battle", "message": problem}


STALLED_SCENARIOS = Path(__file__).resolve().parent.parent / "runtime" / "stalled_scenarios"
STALLED_SCENARIOS_KEPT = 20


def keep_scenario(path, key):
	"""A failed generated battle's scenario file into STALLED_SCENARIOS (the newest STALLED_SCENARIOS_KEPT): the battle
	that stalled can be played again from it (2026-11-05: one stalled at its start every time it came, its scenario deleted
	with the battle)."""
	try:
		STALLED_SCENARIOS.mkdir(parents=True, exist_ok=True)
		shutil.copy2(path, STALLED_SCENARIOS / f"{time.strftime('%Y%m%d-%H%M%S')}-{key}.json")
		for old in sorted(STALLED_SCENARIOS.glob("*.json"))[:-STALLED_SCENARIOS_KEPT]:
			old.unlink(missing_ok=True)
	except OSError as error:
		print(f"supply: a stalled battle's scenario not kept: {error}", flush=True)


def _worker_init(_index):
	"""Each loader worker: on SIGTERM (the loader shutting it down) end the games it runs, so no VCMI outlives a run.
	What it inherits from the forkserver is already frozen there (fork_preload): no collection here first, which would
	walk every inherited object and copy its page (49 MB a worker on the PC, 2026-11-02); what the fork itself made is
	frozen too."""
	import gc

	def stop(signum, frame):
		raise SystemExit(0)  # (the generator's finally closes the runner)

	signal.signal(signal.SIGTERM, stop)
	signal.signal(signal.SIGINT, signal.SIG_IGN)  # (the trainer's Ctrl-C is the trainer's to handle)
	gc.freeze()


def trim_memory():
	"""The heap's freed pages back to the system (glibc keeps them: a worker's RSS stayed at its peak game, 0.7 GB)."""
	import gc

	gc.collect()
	if sys.platform.startswith("linux"):
		import ctypes

		try:
			ctypes.CDLL("libc.so.6").malloc_trim(0)
		except OSError:
			pass


def _identity(item):
	return item


# ---- the trainer's side ----


class Supply:
	"""The loaders and what pulls from them: battle items into pool (actors.ActorPool: its buffer and take()), campaign
	games into campaign (campaign_model.learner.CampaignLearner: _game_done) with their battles into the pool. start()
	after both exist; close() ends the workers and their games."""

	def __init__(self, config, out, pool, campaign=None):
		self.config, self.out, self.pool, self.campaign = config, Path(out), pool, campaign
		self.ctx = forkserver.context()  # (fork_preload: every module a worker imports, once, frozen: inherited)
		ceilings = {"battle": config.generated_actors, **({"campaign": config.campaign.games} if campaign else {})}
		self.shared = Shared(self.ctx, ceilings, servers=len(pool.server.socket_paths))
		self.shared.version.value = pool.version
		global SHARED
		SHARED = self.shared
		self.token = remote.token(config)
		self.listener = socket.create_server(("0.0.0.0", config.remote_port + remote.SLOT_PORT), backlog=64) if config.remote_port and config.remote_slots else None
		battle_sockets = list(pool.server.socket_paths)
		campaign_sockets = list(campaign.server.socket_paths) if campaign else None
		self.datasets = {kind: Games(kind, config, out, battle_sockets, campaign_sockets, self.shared, self.token, self.listener) for kind in ceilings}
		self.loaders, self.threads = {}, []
		self.stop = threading.Event()
		threading.Thread(target=self._watch_trainer, daemon=True).start()
		self.stats = Counter()
		self.game_decisions = 1500.0  # a campaign game's decisions, a running mean (the permits' estimate)
		# From the finished games a resumed run took back (learner.py's spool), when there are: at 1,500 against 14 of
		# ~7,900 decisions the waiting looked like 3 updates' worth, the permits stayed 0 and no game played until those
		# were trained (2026-11-01 15:46-16:00 CST)
		waiting = [len(game.decisions) for game, _ in getattr(campaign, "finished", ())]
		if waiting:
			self.game_decisions = float(np.mean(waiting))
		self.received = 0  # campaign games got (in flight: started - this)

	def _watch_trainer(self):
		"""shared.peak: this process's (the trainer's) highest resident memory over the last PEAK_WINDOW, sampled every
		PEAK_EVERY (memory_allows keeps the way back to it free)."""
		samples, census, types = deque(), 0.0, 0.0
		while not self.stop.is_set():
			rss = process_gb(os.getpid(), "VmRSS")
			if rss is not None:
				now = time.time()
				samples.append((now, rss))
				while samples[0][0] < now - PEAK_WINDOW:
					samples.popleft()
				self.shared.peak.value = max(value for _, value in samples)
				if now - census >= CENSUS_EVERY:
					census = now
					try:
						print(self.memory_census(rss, by_type=now - types >= CENSUS_TYPES_EVERY), flush=True)
						types = now if now - types >= CENSUS_TYPES_EVERY else types
					except Exception as error:  # (a diagnostic: never the run's end)
						print(f"trainer memory: census failed: {str(error)[:200]}", flush=True)
					trim_memory()  # (what the trainer freed meanwhile, the waiting games' inputs (strip_inputs) among it, back to the system)
			self.stop.wait(PEAK_EVERY)

	def memory_census(self, rss, by_type=False):
		"""One line of what the trainer holds (2026-11-01: it grew from ~2 GB after each start to ~12 GB within ~20
		minutes and kept it after its updates, and the memory guard stopped joint7: what it was, nothing logged said)."""
		import gc

		parts = [f"trainer memory: rss {rss:.2f} GB"]
		campaign = self.campaign
		if campaign is not None:
			with campaign.lock:
				waiting = list(campaign.finished)
			futures = list(getattr(campaign, "label_futures", {}).values())
			labels = sum(held_bytes(f.result()) for f in futures if f.done() and not f.cancelled() and f.exception() is None)
			held = sum(len(d.inputs) for g, _ in waiting for d in g.decisions if d.inputs is not None)  # (strip_inputs: none, mostly)
			parts.append(f"campaign waiting {len(waiting)} games, {sum(len(g.decisions) for g, _ in waiting)} decisions, inputs {held / 2**20:.0f} MB; label results {len(futures)} ({labels / 2**20:.0f} MB)")
		with self.pool.lock:
			buffer = list(self.pool.buffer)
		parts.append(f"battle buffer {len(buffer)} battles, {sum(b['decisions'] for b in buffer)} decisions, {sum(held_bytes(b.get('data')) for b in buffer) / 2**20:.0f} MB")
		replay = getattr(self.pool, "replay", None)  # (train.py: config.aux_value's replay, aux_value.ValueReplay)
		if replay is not None:
			parts.append(f"value replay {len(replay.entries)} battles, {replay.decisions()} decisions, {replay.bytes / 2**20:.0f} MB")
		if torch.cuda.is_available():
			host = torch.cuda.host_memory_stats()
			parts.append(f"pinned host {host.get('reserved_bytes.current', host.get('allocated_bytes.current', 0)) / 2**30:.2f} GB")
		if by_type:
			started = time.perf_counter()
			counts = Counter(type(o).__name__ for o in gc.get_objects())
			parts.append("objects " + ", ".join(f"{name} {n}" for name, n in counts.most_common(6)) + f" ({time.perf_counter() - started:.1f}s)")
		return "; ".join(parts)

	def workers(self, kind):
		"""The loader's workers for a kind: this machine's ceiling, plus workers for the remote slots (the campaign's
		all of them, the battles' half: the remote machines' slots are shared by both kinds, and a campaign game is the
		longer use of one)."""
		remote_workers = (self.config.remote_slots if kind == "campaign" else max(1, self.config.remote_slots // 2)) if self.listener is not None else 0
		return self.shared.ceilings[kind] + remote_workers

	def start(self):
		for kind, dataset in self.datasets.items():
			loader = DataLoader(
				dataset, batch_size=None, num_workers=self.workers(kind), prefetch_factor=1 if kind == "battle" else 8, in_order=False, persistent_workers=True,
				multiprocessing_context=self.ctx, timeout=60, collate_fn=_identity, worker_init_fn=_worker_init,
			)
			self.loaders[kind] = loader
			thread = threading.Thread(target=self._pull, args=(kind, loader), daemon=True, name=f"supply-{kind}")
			self.threads.append(thread)
			thread.start()
		if self.campaign:
			self.allow()
			threading.Thread(target=self._keep_allowing, daemon=True, name="supply-permits").start()
		print(f"supply: {self.workers('battle')} battle workers" + (f", {self.workers('campaign')} campaign workers" if self.campaign else "") + f" ({self.ctx.get_start_method()}); ceilings here {self.shared.ceilings}" + (f"; remote slots on port {self.config.remote_port + remote.SLOT_PORT}" if self.listener else ""), flush=True)

	def close(self):
		self.stop.set()
		self.shared.stop.set()
		if self.listener is not None:
			self.listener.close()
		for thread in self.threads:
			thread.join(timeout=90)
		for loader in self.loaders.values():
			iterator = getattr(loader, "_iterator", None)
			if iterator is not None:
				iterator._shutdown_workers()

	def publish(self, version, advantage=None):
		"""The battle model's new version (the league's draw leaves its snapshot out), and the credit its update used
		(advantage: an AdvantageConfig; the workers prepare the next battles with it)."""
		self.shared.version.value = int(version)
		if advantage is not None:
			self.shared.credit[:] = [float(advantage.outcome_share), *(float(w) for w in advantage.policy_heads)]

	def _pull(self, kind, loader):
		iterator = iter(loader)
		loader._iterator = iterator
		pool = self.pool
		while not self.stop.is_set():
			if kind == "battle":  # (the loader's prefetch then parks the workers: each has one battle waiting)
				with pool.lock:
					while not self.stop.is_set() and pool.generated_waiting() >= pool.generated_room():
						pool.lock.wait(timeout=1)
				if self.stop.is_set():
					return
			try:
				item = next(iterator)
			except RuntimeError as error:  # the loader's timeout (nothing finished in a minute), or a worker died
				if "timed out" not in str(error):
					print(f"supply: the {kind} loader failed: {str(error)[:300]}", flush=True)
					time.sleep(5)
				continue
			except StopIteration:
				return
			self._route(item)

	def _route(self, item):
		kind = item["kind"]
		if kind == "battle":
			self.pool.add_item(item)
			self.stats["battles"] += 1
		elif kind == "campaign":
			self.received += 1
			n = len(item["game"].decisions)
			self.game_decisions += (n - self.game_decisions) * 0.1
			self.campaign._game_done(item["key"], item["map"], item["players"], item["game"], item["info"], item["game_map"])
			self.allow()
		elif kind == "failed":
			if item.get("lost"):  # a campaign game that never came: over, for the permits
				self.received += 1
				self.allow()
			if item["what"] == "campaign" and self.campaign:
				self.campaign._failed(item["message"], item.get("run_dir"))
			else:
				self.pool._failed(item["message"])

	def _keep_allowing(self):
		"""allow() every PERMITS_EVERY: it ran only as a game finished, so an update that took the waiting games while
		none was playing left the permits at 0 for good (2026-11-01 15:58-16:16 CST: no campaign game for 18 minutes)."""
		while not self.stop.wait(PERMITS_EVERY):
			try:
				self.allow()
			except Exception as error:  # (the next one tries again)
				print(f"supply: the permits not counted: {str(error)[:200]}", flush=True)

	def allow(self):
		"""The campaign games that may start now: what `ahead` updates take, less the decisions finished and waiting and
		the games in flight (at the mean game's decisions)."""
		if not self.campaign:
			return
		config = self.config.campaign
		waiting = self.campaign.waiting_decisions()
		games = self.campaign.waiting_games() if hasattr(self.campaign, "waiting_games") else 0
		with self.shared.lock:
			in_flight = max(0, self.shared.started.value - self.received)
			# An update takes decisions_per_update decisions and min_games_per_update games, whichever is more: with long
			# games (joint7's ~12,000 decisions) `ahead` updates of decisions alone were 8 games, fewer than one update's
			# 20, so once 20 were in or playing none started, the slots emptied while the update waited for the slowest
			# and stayed empty through it (2026-11-01: 0 games for the 15 minutes of an update)
			update = max(config.decisions_per_update, config.min_games_per_update * self.game_decisions)
			want = (self.config.ahead * update - waiting) / max(self.game_decisions, 1.0) - in_flight
			# And enough games for an update: with long games (joint7's ~7,000 decisions) a few were the decisions of
			# `ahead` updates, the permits fell to 0, and the learner waited for its min_games_per_update forever
			want = max(want, config.min_games_per_update - games - in_flight)
			self.shared.permits.value = max(0, int(math.ceil(want)))

	def status(self):
		"""For the logs: where games play now."""
		s = self.shared
		with s.lock:
			here = {k: v.value for k, v in s.local.items()}
			there = {k: v.value for k, v in s.remote.items()}
			permits, started = s.permits.value, s.started.value
		return {"here": here, "remote": there, "permits": permits, "in_flight": max(0, started - self.received)}
