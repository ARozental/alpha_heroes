"""A worker: this machine plays for a training run on another machine (the master: remote.py), with the master's
models over the network. Its generated battles and campaign games run here (VCMI only: no model, no GPU needed), every
decision asked of the master's inference servers, the records sent there. The master's code (git pull) and game data
(runtime/vcmi-user, as on any machine: README) are needed here, and its token (runtime/remote-token, copied).

	.venv/bin/python -m battle_model.worker --master 192.168.1.20                 # the run's sizes
	.venv/bin/python -m battle_model.worker --master pc.local --machine configs/machines/mac-m4/worker.toml
	.venv/bin/python -m battle_model.worker --master pc.local --set generated_actors=8 campaign.games=4

	.venv/bin/python -m battle_model.worker --master pc.local --local             # its own inference servers

--local: the games ask inference servers on this machine's CPU (inference_processes, campaign.inference_processes:
its sizes; the master sends the weights as they change, remote.py) instead of the master's over the network: its
cores play, not only VCMI (2026-10-30: the master's CPU servers were the limit while the Mac's cores idled).

It reconnects when the master goes away (a resumed run), until stopped (Ctrl-C, kill -TERM).
"""
import argparse
import dataclasses
import json
import signal
import socket
import sys
import threading
import time
import os
from collections import Counter

from . import play, remote
from .actors import ActorPool
from .train import TrainConfig, generated_games, machine_overrides, with_overrides

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
	"""In place of an inference server (actors.ActorPool.server, CampaignLearner.server): the master's, over TCP."""

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


class RemoteActors(ActorPool):
	"""ActorPool's generated actors (and its collector of the campaign games' battles) on a worker: the master's model
	versions and league, its backpressure (state "room"), the records checked and compacted here, sent there."""

	def __init__(self, config, endpoints, channel, name, server=None):
		self.config, self.channel = config, channel
		# (its scenario and map files named apart from the master's and the other workers': one disk may hold them all)
		self.generated_games = lambda scenarios, label, config, **game: generated_games(scenarios, f"{name}_{label}", config, **game)
		self.link = play.ROOT / "runtime" / "remote-model.onnx"  # (named in the env, never read: the master's servers play)
		self.server = server or Endpoints(endpoints)
		self.version, self.league, self.room = -1, [], False
		self.buffer = []
		self.lock = threading.Condition()
		self.stop = threading.Event()
		self.running = threading.Event()
		self.stats = {"failures": 0, "dropped_stale": 0, "dropped_overflow": 0, "battles": {"generated": 0, "real": 0}, "last_failure": "", "decisions": 0}
		self.records = play.RecordServer(concurrent=True)
		self.threads = []

	def past_versions(self):
		return list(self.league)

	def generated_waiting(self):
		return 0

	def generated_room(self):
		return 1 if self.room else 0

	def _add(self, records, source, game, checked=False):
		sound, failures = self._check(records, source, game)
		if sound:
			self.channel.send(("battle", sound, source, game))
			with self.lock:
				self.stats["battles"][source] += len(sound)
				self.stats["decisions"] += sum(r.decisions for r in sound)
		for failure in failures:
			self.channel.send(("failed", "battle", failure))

	def _failed(self, error):
		super()._failed(error)
		self.channel.send(("failed", "battle", str(error)[:300]))


def remote_campaign(config, actors, endpoints, name, channel, server=None):
	"""CampaignLearner's self-play games (its _game_actor) on a worker: on the master's campaign server, every finished
	game sent to the master (remote.Hub hands it to the learner's _game_done)."""
	from campaign_model.learner import CampaignLearner

	class RemoteCampaign(CampaignLearner):
		def __init__(self):
			self.config, self.battle, self.server = config, actors, server or Endpoints(endpoints)
			self.version, self.key_prefix = 1, f"{name}-"
			self.lock = threading.Condition()
			self.stop, self.running = threading.Event(), threading.Event()
			self.stats, self.failed_runs, self.threads = Counter(), [], []

		def start(self):
			for i in range(self.config.games):
				self.threads.append(threading.Thread(target=self._game_actor, args=(i,), daemon=True))
			for thread in self.threads:
				thread.start()

		def close(self):
			self.stop.set()
			for thread in self.threads:
				thread.join(timeout=150)

		def _game_done(self, key, map_name, players, game, info, game_map):
			channel.send(("campaign", key, map_name, players, game, info, game_map))
			with self.lock:
				self.stats["games"] += 1
				self.stats["decisions"] += len(game.decisions)

		def _failed(self, message, run_dir=None):
			super()._failed(message, run_dir)
			channel.send(("failed", "campaign", message[:400]))

	return RemoteCampaign()


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


def local_servers(config, welcome):
	"""{family: an inference server on this machine's CPU} for --local, its sizes the worker's (inference_processes,
	campaign.inference_processes; 0: a fifth of the cores), the models the master's (welcome's model_kwargs)."""
	from .inference import InferenceServer

	play.SOCKET_DIR.mkdir(parents=True, exist_ok=True)
	sizes = {"battle": (config.inference_processes, config.inference_threads), "campaign": (config.campaign.inference_processes, config.campaign.inference_threads)}
	servers = {}
	for family, kwargs in welcome.get("model_kwargs", {}).items():
		processes, threads = sizes[family]
		processes = processes or max(1, (os.cpu_count() or 4) // 5)
		servers[family] = InferenceServer(play.SOCKET_DIR / f"worker-{family}-{os.getpid()}.sock", device="cpu", family=family, model_kwargs=kwargs, threads=threads, processes=processes)
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


def serve(host, port, token, overrides, sizes, local=False):
	"""One connection's work: until the master goes away (or this one is stopped)."""
	conn, welcome = connect(host, port, token, sizes, models=local)
	servers = {}
	try:
		name, number = welcome["name"], welcome["number"]
		config = with_overrides(config_from_dict(welcome["config"]), overrides)
		# Other battles and games than the master's and the other workers': each worker its own seeds
		config = dataclasses.replace(config, seed=config.seed + 1000 * number, campaign=dataclasses.replace(config.campaign, seed=config.campaign.seed + 1000 * number))
		servers = local_servers(config, welcome) if local else {}
		if local and not servers:
			raise SystemExit("--local: the master sent no models to serve (an older master?)")
		pending = []
		while not all("current" in server.serving() for server in servers.values()):  # (--local: no game before its servers have the master's current models)
			kind, payload, *rest = remote.read_message(conn)
			if kind == "models":
				apply_models(servers, payload, rest[0])
			else:
				pending.append((kind, payload))
		_play(conn, welcome, config, name, host, port, servers, pending)
	except (ConnectionError, OSError, EOFError) as error:
		print(f"worker: lost the master before playing ({str(error)[:200]})", flush=True)
	finally:
		conn.close()
		for server in servers.values():
			server.close()


def _play(conn, welcome, config, name, host, port, servers, pending):
	"""serve's games, on servers (this machine's, --local) or the master's, until the master goes away."""
	channel = Channel(conn)
	actors = RemoteActors(config, [f"tcp:{host}:{p}" for p in welcome["battle_ports"]], channel, name, server=servers.get("battle"))
	campaign_ok = config.campaign.games and (welcome["campaign_ports"] or "campaign" in servers)
	campaign = remote_campaign(config.campaign, actors, [f"tcp:{host}:{p}" for p in welcome["campaign_ports"]], name, channel, server=servers.get("campaign")) if campaign_ok else None
	where = f"its own servers ({', '.join(f'{f} {len(s.socket_paths)}' for f, s in servers.items())} processes)" if servers else "the master's servers"
	print(f"worker {name}: {config.generated_actors} generated actors, {config.campaign.games if campaign else 0} campaign games, for run {config.run} at {host}:{port}, on {where}", flush=True)
	actors.start()
	if campaign:
		campaign.start()
	started, last_report = time.monotonic(), time.monotonic()
	try:
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
			with actors.lock:
				actors.version, actors.league, actors.room = state["version"], state["league"], state["room"]
				actors.lock.notify_all()
			(actors.running.set if state["running"] else actors.running.clear)()
			if campaign:
				campaign.version = state["campaign_version"]
				(campaign.running.set if state["campaign_running"] else campaign.running.clear)()
			if time.monotonic() - last_report >= 300:
				last_report = time.monotonic()
				hours = (last_report - started) / 3600
				print(f"worker {name}: {actors.stats['battles']['generated']} generated battles, {actors.stats['battles']['real']} campaign battles, {actors.stats['decisions']} decisions ({actors.stats['decisions'] / max(hours, 1e-9) / 3600:.1f}/s), {campaign.stats['games'] if campaign else 0} campaign games, {actors.stats['failures']} failures; model version {state['version']}", flush=True)
	except (ConnectionError, OSError, EOFError) as error:
		print(f"worker {name}: lost the master ({str(error)[:200]})", flush=True)
	finally:
		channel.lost.set()
		if campaign:
			campaign.close()
		actors.close()


def main():
	def stop(signum, frame):
		raise KeyboardInterrupt(f"signal {signum}")

	signal.signal(signal.SIGTERM, stop)
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--master", required=True, metavar="HOST[:PORT]", help=f"the master's address (port {DEFAULT_PORT}: its remote_port)")
	parser.add_argument("--token", default="", help=f"the run's shared secret (default: {remote.TOKEN_FILE}, copied from the master)")
	parser.add_argument("--machine", metavar="TOML", help="this machine's sizes (generated_actors, [campaign] games), applied over the run's")
	parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="more of this machine's sizes, e.g. generated_actors=8 campaign.games=4")
	parser.add_argument("--once", action="store_true", help="end when the master goes away instead of waiting for it")
	parser.add_argument("--local", action="store_true", help="the games ask inference servers on this machine's CPU, the master's weights (sizes: inference_processes, campaign.inference_processes)")
	args = parser.parse_args()
	host, _, port = args.master.partition(":")
	token = args.token or (remote.TOKEN_FILE.read_text().strip() if remote.TOKEN_FILE.exists() else "")
	if not token:
		raise SystemExit(f"no token: copy the master's {remote.TOKEN_FILE} here, or pass --token")
	overrides = tuple((machine_overrides(args.machine) if args.machine else []) + args.set)
	sizes = dict(o.split("=", 1) for o in overrides)
	while True:
		try:
			serve(host, int(port or DEFAULT_PORT), token, overrides, sizes, local=args.local)
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


if __name__ == "__main__":
	sys.exit(main())
