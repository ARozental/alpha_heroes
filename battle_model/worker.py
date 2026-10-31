"""A worker: this machine plays games for a training run on another machine (the master: remote.py, supply.py), as the
master's data loader asks for them. The master's code (git pull) and game data (runtime/vcmi-user, as on any machine:
README) are needed here, and its token (runtime/remote-token, copied).

	.venv/bin/python -m battle_model.worker --master 192.168.1.20 --slots 12 --local
	.venv/bin/python -m battle_model.worker --master pc.local --machine configs/machines/mac-m4/worker.toml --local

	--slots N   N games at once, at most: N connections held open on the master's slot port (remote_port + SLOT_PORT),
	            each given a game to play when the master has one to place here (its own machine's games first), then
	            closed and opened again. What a slot plays is what supply.Runner plays on the master: a generated battle
	            or a campaign game, its records turned into training arrays here and sent back as they are made
	--local     the games ask inference servers on this machine's CPU (inference_processes, campaign.inference_processes:
	            its sizes; the master sends the weights as they change, remote.py) instead of the master's over the
	            network: its cores play, not only VCMI (2026-10-30: the master's CPU servers were the limit while the
	            Mac's cores idled)

It reconnects when the master goes away (a resumed run), until stopped (Ctrl-C, kill -TERM).
"""
import argparse
import dataclasses
import json
import os
import signal
import socket
import sys
import threading
import time
from collections import Counter
from pathlib import Path

from . import play, remote
from .train import TrainConfig, machine_overrides, with_overrides

DEFAULT_PORT = 7700
RETRY_SECONDS = 10


def config_from_dict(values, template=None):
	"""A dataclass (a TrainConfig by default) from asdict's dict of it as JSON carried it: lists back to tuples where
	the field holds a tuple, the nested configs (reward, advantage, ppo, campaign) rebuilt alike; fields this code does
	not have are left out."""
	template = template if template is not None else TrainConfig()
	changes = {}
	for field in dataclasses.fields(template):
		if field.name not in values:
			continue
		default, value = getattr(template, field.name), values[field.name]
		if dataclasses.is_dataclass(default):
			value = config_from_dict(value, default)
		elif isinstance(default, tuple) and isinstance(value, list):
			value = tuple(tuple(v) if isinstance(v, list) else v for v in value)
		changes[field.name] = value
	return dataclasses.replace(template, **changes)


class Endpoints:
	"""In place of an inference server (the master's, over TCP): socket_for gives its address."""

	def __init__(self, addresses):
		self.addresses, self.tcp_ports = list(addresses), []

	def socket_for(self, index):
		return self.addresses[index % len(self.addresses)]

	def close(self):
		pass


class Channel:
	"""The connection to the master's hub, shared by every thread: a message whole or not at all, and a lost connection
	only noted (the main loop sees it and starts again)."""

	def __init__(self, conn):
		self.conn, self.lock, self.lost = conn, threading.Lock(), threading.Event()
		self.sent = Counter()

	def send(self, message):
		if self.lost.is_set():
			return
		try:
			remote.send_message(self.conn, message, self.lock)
			self.sent[message[0]] += 1
		except OSError:
			self.lost.set()


def connect(host, port, token, sizes, models=False):
	"""(the connection, the hub's welcome); raises SystemExit when the master refuses this worker for good."""
	conn = socket.create_connection((host, port), timeout=30)
	conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
	remote.send_frame(conn, json.dumps({"protocol": remote.PROTOCOL, "token": token, "name": socket.gethostname(), "contracts": remote.contracts(), "sizes": sizes, "models": models}).encode())
	welcome = json.loads(remote.read_frame(conn))
	if "error" in welcome:
		conn.close()
		raise SystemExit(f"the master refused this worker: {welcome['error']}")
	conn.settimeout(None)
	return conn, welcome


SERVE_PORTS = {"battle": 1, "campaign": 51}  # --serve: a family's first TCP port past the given one (process i on + i)


def local_servers(config, welcome, serve=0):
	"""{family: an inference server on this machine's CPU} for --local, its sizes the worker's (inference_processes,
	campaign.inference_processes; 0: a fifth of the cores), the models the master's (welcome's model_kwargs). serve: a
	port; the servers then listen on it + SERVE_PORTS[family] + i too, for the master's own games (its
	inference_endpoints)."""
	from .inference import InferenceServer

	play.SOCKET_DIR.mkdir(parents=True, exist_ok=True)
	sizes = {"battle": (config.inference_processes, config.inference_threads), "campaign": (config.campaign.inference_processes, config.campaign.inference_threads)}
	servers = {}
	for family, kwargs in welcome.get("model_kwargs", {}).items():
		processes, threads = sizes[family]
		processes = processes or max(1, (os.cpu_count() or 4) // 5)
		tcp_port = serve + SERVE_PORTS[family] if serve else 0
		servers[family] = InferenceServer(play.SOCKET_DIR / f"worker-{family}-{os.getpid()}.sock", device="cpu", family=family, model_kwargs=kwargs, threads=threads, processes=processes, tcp_host="0.0.0.0" if serve else None, tcp_port=tcp_port)
	return servers


def apply_models(servers, family, changes):
	"""A hub's ("models", family, changes): the weights this machine's servers serve from now on."""
	server = servers.get(family)
	for model_name, item in changes.items():
		if server is None:
			continue
		if item is None:
			server.drop(model_name)
		else:
			server.set_model(model_name, item[0], item[1])


class Slot:
	"""One game at a time for the master: a connection held open on its slot port until a game is placed here, played
	(supply.Runner, this machine's servers), its items sent back; then a new connection. The master closes the
	connection after a game, or when it has nothing for us (its own games suffice)."""

	def __init__(self, index, host, port, token, config, run_dir, servers, endpoints, channel, name, version):
		self.index, self.host, self.port, self.token, self.channel, self.name = index, host, port, token, channel, name
		battle_socket = servers["battle"].socket_for(index) if "battle" in servers else endpoints["battle"].socket_for(index)
		campaign_socket = (servers["campaign"].socket_for(index) if "campaign" in servers else endpoints["campaign"].socket_for(index)) if config.campaign.games else None
		from .supply import Runner

		self.runner = Runner(config, run_dir, battle_socket, campaign_socket, Path(run_dir) / "models" / "current.onnx", f"{name}-s{index}", version)
		self.stop = threading.Event()
		self.played = Counter()
		self.thread = threading.Thread(target=self._loop, daemon=True, name=f"slot-{index}")

	def start(self):
		self.thread.start()

	def close(self):
		self.stop.set()
		self.runner.close()
		self.thread.join(timeout=30)

	def _loop(self):
		while not self.stop.is_set() and not self.channel.lost.is_set():
			try:
				conn = socket.create_connection((self.host, self.port), timeout=30)
				conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
				remote.send_frame(conn, json.dumps({"protocol": remote.PROTOCOL, "token": self.token, "name": self.name, "contracts": remote.contracts()}).encode())
				welcome = json.loads(remote.read_frame(conn))
				if "error" in welcome:
					print(f"slot {self.index}: refused: {welcome['error']}", flush=True)
					conn.close()
					time.sleep(RETRY_SECONDS)
					continue
				conn.settimeout(None)
				self._serve(conn)
			except (ConnectionError, OSError, EOFError, ValueError) as error:
				if not self.stop.is_set():
					time.sleep(min(RETRY_SECONDS, 2.0))  # (the master has nothing to place here yet, or is away)
					_ = error

	def _serve(self, conn):
		"""The games this connection is given (one, as supply.Games.relay does), until the master closes it."""
		try:
			while not self.stop.is_set():
				kind, spec = remote.read_message(conn)
				if kind != "play":
					continue
				decisions = 0
				try:
					items = self.runner.battle(spec) if spec["kind"] == "battle" else self.runner.campaign(spec, self.stop)
					for item in items:
						remote.send_message(conn, ("item", item))
						if item["kind"] == "battle":
							decisions += item["decisions"]
						elif item["kind"] == "campaign":
							decisions += len(item["game"].decisions)
					remote.send_message(conn, ("done",))
				except Exception as error:  # the game failed: the master counts it
					remote.send_message(conn, ("failed", f"{spec.get('key', '?')}: {str(error)[:400]}"))
				self.played[spec["kind"]] += 1
				self.channel.send(("played", spec["kind"], decisions))
		finally:
			conn.close()


def serve(host, port, token, overrides, sizes, slots, local=False, serve_port=0, servers=None):
	"""One connection's work: until the master goes away (or this one is stopped). serve_port: --serve. servers: this
	worker's inference servers, kept across the master's connections (a run's restart must not kill the games that ask
	them: it did, 2026-10-30, every PC game for a minute) and filled here on the first; their models are set again from
	each new master."""
	conn, welcome = connect(host, port, token, sizes, models=local)
	running = []
	try:
		name, number = welcome["name"], welcome["number"]
		config = with_overrides(config_from_dict(welcome["config"]), overrides)
		# Other battles and games than the master's and the other workers': each worker its own seeds
		config = dataclasses.replace(config, seed=config.seed + 1000 * number, campaign=dataclasses.replace(config.campaign, seed=config.campaign.seed + 1000 * number))
		if local and servers is not None and not servers:
			servers.update(local_servers(config, welcome, serve_port))
			if serve_port:
				print("worker: serving inference on " + ", ".join(f"{family} {', '.join(str(p) for p in server.tcp_ports)}" for family, server in servers.items()) + f" (the master's inference_endpoints, with this machine's address)", flush=True)
		servers = servers if local else {}
		if local and not servers:
			raise SystemExit("--local: the master sent no models to serve (an older master?)")
		pending = []
		while not all("current" in server.serving() for server in servers.values()):  # (--local: no game before its servers have the master's current models)
			kind, payload, *rest = remote.read_message(conn)
			if kind == "models":
				apply_models(servers, payload, rest[0])
			else:
				pending.append((kind, payload))
		endpoints = {"battle": Endpoints([f"tcp:{host}:{p}" for p in welcome["battle_ports"]]), "campaign": Endpoints([f"tcp:{host}:{p}" for p in welcome["campaign_ports"] or welcome["battle_ports"]])}
		channel = Channel(conn)
		run_dir = play.ROOT / "runtime" / "worker" / config.run  # (the league's snapshots are not here: the master names them in each spec)
		(run_dir / "models").mkdir(parents=True, exist_ok=True)
		version = _Version()
		running = [Slot(i, host, port + remote.SLOT_PORT, token, config, run_dir, servers, endpoints, channel, name, version) for i in range(slots)]
		where = f"its own servers ({', '.join(f'{f} {len(s.socket_paths)}' for f, s in servers.items())} processes)" if servers else "the master's servers"
		print(f"worker {name}: {slots} slots for run {config.run} at {host}:{port}, on {where}", flush=True)
		for slot in running:
			slot.start()
		started, last_report = time.monotonic(), time.monotonic()
		while not channel.lost.is_set():
			if pending:
				kind, state = pending.pop(0)
			else:
				kind, state, *rest = remote.read_message(conn)
			if kind == "models":
				apply_models(servers, state, rest[0])
				continue
			if kind != "state":
				continue
			version.value = state["version"]
			if time.monotonic() - last_report >= 300:
				last_report = time.monotonic()
				played = sum((slot.played for slot in running), Counter())
				print(f"worker {name}: {played['battle']} generated battles, {played['campaign']} campaign games in {(last_report - started) / 3600:.2f} h; model version {state['version']}", flush=True)
	except (ConnectionError, OSError, EOFError) as error:
		print(f"worker: lost the master ({str(error)[:200]})", flush=True)
	finally:
		conn.close()
		close_slots(running)


def close_slots(slots, timeout=60):
	"""Every slot's games ended at once (one after another took up to 40 s a slot, and a worker killed meanwhile left
	its VCMI processes behind, 2026-10-30), waited for up to timeout seconds in all."""
	for slot in slots:
		slot.stop.set()
	threads = [threading.Thread(target=slot.runner.close, daemon=True) for slot in slots]
	for thread in threads:
		thread.start()
	deadline = time.monotonic() + timeout
	for thread in threads + [slot.thread for slot in slots]:
		thread.join(timeout=max(0.0, deadline - time.monotonic()))


class _Version:
	"""The battle model's version as the hub's state says (what supply.Runner reads for the league's draw)."""

	value = 0


def main():
	def stop(signum, frame):
		raise KeyboardInterrupt(f"signal {signum}")

	signal.signal(signal.SIGTERM, stop)
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--master", required=True, metavar="HOST[:PORT]", help=f"the master's address (port {DEFAULT_PORT}: its remote_port)")
	parser.add_argument("--token", default="", help=f"the run's shared secret (default: {remote.TOKEN_FILE}, copied from the master)")
	parser.add_argument("--machine", metavar="TOML", help="this machine's sizes (remote_slots: its slots; inference_processes...), applied over the run's")
	parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="more of this machine's sizes, e.g. inference_processes=2")
	parser.add_argument("--slots", type=int, default=0, help="games at once, at most (default: the machine config's remote_slots, else 4)")
	parser.add_argument("--once", action="store_true", help="end when the master goes away instead of waiting for it")
	parser.add_argument("--local", action="store_true", help="the games ask inference servers on this machine's CPU, the master's weights (sizes: inference_processes, campaign.inference_processes)")
	parser.add_argument("--serve", type=int, default=0, metavar="PORT", help="with --local: the servers also listen on PORT+1.. (battle) and PORT+51.. (campaign) for the master's own games (its inference_endpoints)")
	args = parser.parse_args()
	host, _, port = args.master.partition(":")
	token = args.token or (remote.TOKEN_FILE.read_text().strip() if remote.TOKEN_FILE.exists() else "")
	if not token:
		raise SystemExit(f"no token: copy the master's {remote.TOKEN_FILE} here, or pass --token")
	overrides = tuple((machine_overrides(args.machine) if args.machine else []) + args.set)
	sizes = dict(o.split("=", 1) for o in overrides)
	slots = args.slots or int(sizes.get("remote_slots", 0)) or 4
	servers = {}  # (this machine's inference servers, kept across the master's connections)
	try:
		while True:
			try:
				serve(host, int(port or DEFAULT_PORT), token, overrides, sizes, slots, local=args.local, serve_port=args.serve, servers=servers)
			except (ConnectionError, OSError) as error:
				print(f"worker: no master at {args.master} ({str(error)[:200]}); trying again in {RETRY_SECONDS}s", flush=True)
			except KeyboardInterrupt:
				print("worker: stopped", flush=True)
				return
			if args.once:
				return
			try:
				time.sleep(RETRY_SECONDS)
			except KeyboardInterrupt:
				return
	finally:
		for server in servers.values():
			server.close()


if __name__ == "__main__":
	sys.exit(main())
