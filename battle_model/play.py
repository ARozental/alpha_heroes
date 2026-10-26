"""Play recorded battles in VCMI: our model on one side, another battle AI on the other, or our model on
both sides (opponent SELF_PLAY: each side has its own AlphaBattleAI with its own record and sampling seed).

A scenario (battle_model/scenarios/*.json) becomes a VCMI "battle only" map: the game starts with the
battle between its two heroes and ends right after it. sides[0] attacks from the left (red), sides[1]
defends on the right (blue). AlphaBattleAI writes the record (every decision plus events.jsonl) that
data.py turns into training examples.

Each game runs through scripts/headless.sh, so it has a time limit, a disk limit and stall detection.
"""
import functools
import itertools
import json
import os
import select
import shutil
import socket
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER_DIR = ROOT / "runtime" / "vcmi-user"
COLORS = ("RED", "BLUE")  # battle side 0 (attacker) and 1 (defender)
SELF_PLAY = "self"  # Game.opponent for our model against itself
SOCKET_DIR = ROOT / "runtime" / "sockets"  # RecordServer sockets: only addresses, no data on disk
PREFIX_DIR = ROOT / "runtime" / "prefixes"  # Game.prefix files, deleted after each battle
MAP_BUILDER_DIR = ROOT / "runtime" / "map-builders"  # the map builder's user data folders, one per process and thread
_server_numbers = itertools.count()


@dataclass(frozen=True)
class Game:
	scenario: str  # path to the scenario json
	our_side: int  # 0 = we play sides[0] (attacker), 1 = sides[1] (defender)
	opponent: str = "BattleAI"  # the other side's battle AI, or SELF_PLAY
	game_seed: int = 1  # VCMI's random seed: dice, obstacles, the opponent's choices
	sampler_seed: int = 0  # our model's sampling
	temperature: float = 1.0
	ai: str = "AlphaBattleAI"  # our side's battle AI library; AlphaBattleAIDev to try C++ changes during training runs
	# Map to play, as returned by build_map; empty = the scenario's shared map (map_name). Training runs use
	# their own copy, so rebuilding a map (e.g. after a map tool fix) never changes a running experiment.
	map: str = ""
	# Actions (dicts as in a record's action.json) our side plays before the model chooses: battles are
	# deterministic given game_seed and our actions, so this restores a state of a recorded battle
	prefix: tuple = ()
	# Extra environment settings for this battle's AIs, (name, value) pairs, e.g. (("ALPHA_BATTLE_DELEGATE", "BattleAI"),)
	# (agents/AlphaBattleAI/AlphaBattleAI.h lists them)
	settings: tuple = ()

	@property
	def neutrals(self):
		"""Wandering monsters defend (the scenario's "neutrals"): side 1 is VCMI's neutral player, whose battle AI is the
		setting combatNeutralAI (our AI in self-play and when we play the monsters, else the opponent)."""
		return has_neutrals(self.scenario)

	@property
	def colors(self):
		"""The player colour of battle side 0 and side 1."""
		return (COLORS[0], NEUTRAL) if self.neutrals else COLORS

	@property
	def recorded_sides(self):
		"""The sides our model plays, each with its own record."""
		return (0, 1) if self.opponent == SELF_PLAY else (self.our_side,)


NEUTRAL = "NEUTRAL"  # the wandering monsters' player (AlphaBattleAI reads ALPHA_BATTLE_<NAME>_NEUTRAL)


@functools.lru_cache(maxsize=4096)
def scenario_of(scenario_path):
	return json.loads(Path(scenario_path).read_text())


def has_neutrals(scenario_path):
	return "neutrals" in scenario_of(scenario_path)


def record_dirs(game, record_dir):
	"""The record folder of each side in game.recorded_sides."""
	record_dir = Path(record_dir).resolve()
	return [record_dir / f"side_{side}" for side in game.recorded_sides] if game.opponent == SELF_PLAY else [record_dir]


def load_scenario(path):
	return json.loads(Path(path).read_text())


def map_name(scenario_path, variant=""):
	return "Maps/alpha/" + Path(scenario_path).stem + (f"-{variant}" if variant else "")


def active_mods():
	"""The mods every game starts with (the shared user data folder's preset; headless.sh copies it into each run)."""
	settings = json.loads((USER_DIR / "config" / "modSettings.json").read_text())
	return settings["presets"][settings.get("activePreset", "default")]["mods"]


def _map_builder_dir():
	"""This thread's own VCMI user data folder for alpha_make_battle_map (the shared one's game data, linked). VCMI
	rewrites config/modSettings.json at every start (truncates it, then writes), so map builds running in parallel on
	the shared folder now and then read it half-written, fell back to the default mods and saved that: the MMAI mod was
	switched off for every game after (the eval against MMAI was played by BattleAI, MMAI's fallback)."""
	path = MAP_BUILDER_DIR / f"{os.getpid()}-{threading.get_ident()}"
	if not (path / "config" / "modSettings.json").exists():
		_remove_dead_map_builder_dirs()
		(path / "config").mkdir(parents=True, exist_ok=True)
		for name in ("Data", "Mods"):
			if not (path / name).is_symlink():
				(path / name).symlink_to(USER_DIR / name)
		(path / "config" / "settings.json").write_text(json.dumps({"logging": {"loggers": [{"domain": "global", "level": "info"}]}}))  # trace logs: 1 MB a map
		shutil.copy(USER_DIR / "config" / "modSettings.json", path / "config" / "modSettings.json")
		(path / "config" / "modSettings.json").chmod(0o644)
	return path


def _remove_dead_map_builder_dirs():
	for path in MAP_BUILDER_DIR.glob("*-*"):
		try:
			os.kill(int(path.name.split("-")[0]), 0)
		except ProcessLookupError:
			shutil.rmtree(path, ignore_errors=True)
		except (ValueError, PermissionError):
			pass


def build_map(scenario_path, variant=""):
	"""Write the scenario's battle-only map into the VCMI user data folder (overwrites); returns its map name.

	variant: a suffix for a private copy (e.g. the training run's name).
	"""
	name = map_name(scenario_path, variant)
	out = USER_DIR / (name + ".vmap")
	out.parent.mkdir(parents=True, exist_ok=True)
	env = {**os.environ, "VCMI_USER_DIR": str(_map_builder_dir())}
	result = subprocess.run([str(ROOT / "build" / "bin" / "alpha_make_battle_map"), str(scenario_path), str(out)], env=env, capture_output=True, text=True)
	if result.returncode != 0 or not out.exists():
		raise RuntimeError(f"building the map failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
	return name


class RecordServer:
	"""Receives battle records from AlphaBattleAI over a Unix socket, so training data never touches the disk.

	Each AI connects at the end of its battle and sends a key (u32 length + bytes) and the record
	(agents/AlphaBattleAI/Dump.h BattleRecord); records: {key: [bytes, ...]}, every battle sent with the key in the
	order received (a battle-only game sends one, a real game one per battle). Use as a context manager: leaving it
	reads what is still queued, then closes the socket.
	"""

	def __init__(self):
		SOCKET_DIR.mkdir(parents=True, exist_ok=True)
		self.path = SOCKET_DIR / f"{os.getpid()}-{next(_server_numbers)}.sock"
		if len(str(self.path)) >= 100:  # sockaddr_un.sun_path is 104 bytes on macOS
			raise RuntimeError(f"socket path too long: {self.path}")
		self.path.unlink(missing_ok=True)
		self.records = {}
		self._lock = threading.Lock()
		self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
		self._socket.bind(str(self.path))
		self._socket.listen(128)
		self._stop = threading.Event()
		self._thread = threading.Thread(target=self._serve, daemon=True)
		self._thread.start()

	def _serve(self):
		while True:
			ready, _, _ = select.select([self._socket], [], [], 0.1)
			if ready:
				connection, _ = self._socket.accept()
				with connection:
					data = bytearray()
					while chunk := connection.recv(1 << 20):
						data += chunk
				(key_length,) = struct.unpack_from("<I", data, 0)
				key = bytes(data[4 : 4 + key_length]).decode().split("#", 1)[0]  # AlphaBattleAI numbers each battle: key#n
				with self._lock:
					self.records.setdefault(key, []).append(bytes(data[4 + key_length :]))
			elif self._stop.is_set():
				return

	def drain(self):
		"""The records received since the last drain ({key: [bytes, ...]}), removed from records."""
		with self._lock:
			records, self.records = self.records, {}
		return records

	def __enter__(self):
		return self

	def __exit__(self, *exc):
		self._stop.set()
		self._thread.join()
		self._socket.close()
		self.path.unlink(missing_ok=True)


def game_setup(game, model_path, records):
	"""(map name, env, VCMI settings, prefix file or None) of one battle: what start() gives a new VCMI process, and a
	BattleProcess sends its running one as the next game.

	records: per side in game.recorded_sides, where its record goes: "socket" and "key" (a RecordServer: sent
	at the end of the battle, no files) and/or "dir" (written there for debugging; must not exist yet). Each
	side our model plays gets its own settings (AlphaBattleAI reads ALPHA_BATTLE_<NAME>_<COLOR> before
	ALPHA_BATTLE_<NAME>): its record and its sampling seed.
	"""
	env = {"ALPHA_BATTLE_MODEL": str(Path(model_path).resolve()), "ALPHA_BATTLE_TEMPERATURE": str(game.temperature)}
	settings = {"server": {"seed": game.game_seed}, **({"ai": {"combatNeutralAI": game.ai if 1 in game.recorded_sides else game.opponent}} if game.neutrals else {})}
	env.update(dict(game.settings))
	for side, color in enumerate(game.colors):
		env[f"VCMI_BATTLE_AI_{color}"] = game.ai if side in game.recorded_sides else game.opponent
	for side, setup in zip(game.colors, scenario_of(game.scenario).get("sides", [])):  # what a map cannot set
		if "gold" in setup:
			env[f"VCMI_BATTLE_GOLD_{side}"] = str(int(setup["gold"]))
		if "mana_fraction" in setup:
			env[f"VCMI_BATTLE_MANA_{side}"] = str(setup["mana_fraction"])
	for i, (side, target) in enumerate(zip(game.recorded_sides, records)):
		color = game.colors[side]
		if target.get("socket"):
			env[f"ALPHA_BATTLE_RECORD_SOCKET_{color}"] = str(target["socket"])
			env[f"ALPHA_BATTLE_RECORD_KEY_{color}"] = target["key"]
		if target.get("dir"):
			env[f"ALPHA_BATTLE_RECORD_DIR_{color}"] = str(Path(target["dir"]).resolve())
		env[f"ALPHA_BATTLE_SEED_{color}"] = str(game.sampler_seed + 1_000_003 * i)  # the two sides sample independently
	prefix = None
	if game.prefix:
		PREFIX_DIR.mkdir(parents=True, exist_ok=True)
		prefix = PREFIX_DIR / f"{os.getpid()}-{next(_server_numbers)}.jsonl"  # a few KB, deleted by finish()
		prefix.write_text("".join(json.dumps(action) + "\n" for action in game.prefix))
		env[f"ALPHA_BATTLE_PREFIX_{game.colors[game.our_side]}"] = str(prefix)
	return game.map or map_name(game.scenario), env, settings, prefix


def start(game, model_path, records, max_seconds=300, max_mb=200):
	"""Start one battle in a new VCMI process in the background; returns the process (records: as game_setup)."""
	name, setup, settings, prefix = game_setup(game, model_path, records)
	env = {**os.environ, **setup, "SETTINGS": json.dumps(settings)}
	# A big late-game battle on a busy machine can go a while without a log line: a stall is 60s of silence
	command = [str(ROOT / "scripts" / "headless.sh"), name, str(max_seconds), str(max_mb), "60"]
	process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
	process.prefix_file = prefix
	return process


QUEUE_ENV_PREFIXES = ("ALPHA_BATTLE_", "VCMI_BATTLE_")  # a queued game's env replaces the previous game's
BATTLE_NICENESS = 10  # BattleProcess's (the actors' generated battles; real games run at 15)


class BattleProcess:
	"""One VCMI client that plays battle after battle (the client's test queue, vcmi client/TestQueue.h): its start-up
	(~0.6 s of CPU, more than the rest of a short battle) once every max_battles, not once a battle. play() is play()
	for one game; a battle that fails or stalls kills the process (the next play starts a new one) and raises."""

	def __init__(self, battle_seconds=300, max_battles=300, max_mb=300):
		self.battle_seconds, self.max_battles, self.max_mb = battle_seconds, max_battles, max_mb
		self.records = RecordServer()
		self.process = self.conn = None
		self.battles = 0
		self.pending = b""

	def _start(self):
		self.close_process()
		SOCKET_DIR.mkdir(parents=True, exist_ok=True)
		path = SOCKET_DIR / f"queue-{os.getpid()}-{next(_server_numbers)}.sock"
		listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
		listener.bind(str(path))
		listener.listen(1)
		env = {**os.environ, "VCMI_TEST_QUEUE": str(path)}
		# The process lives for many battles: its watchdog's limits are the whole run's; a stall (60 s without a log
		# line while idle, e.g. waiting for its next battle during an eval) ends it, and the next battle starts another
		# At low priority: the actors make more battles than the learner takes, and the learner (and the inference
		# server) starved for CPU (a learner update took 15 s in a run, 9.4 s alone)
		command = ["nice", "-n", str(BATTLE_NICENESS), str(ROOT / "scripts" / "headless.sh"), "queue", str(24 * 3600), str(self.max_mb), "60"]
		self.process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
		listener.settimeout(120)
		try:
			self.conn, _ = listener.accept()
		except socket.timeout:
			self.close_process()
			raise RuntimeError("the VCMI battle process did not connect to its queue")
		finally:
			listener.close()
			path.unlink(missing_ok=True)
		self.battles, self.pending = 0, b""

	def _line(self, timeout):
		"""The client's next line ("ready" for the next game), or None if it did not come in time or the client ended."""
		deadline = time.monotonic() + timeout  # monotonic: it stands still while the machine sleeps
		while b"\n" not in self.pending:
			left = deadline - time.monotonic()
			if left <= 0 or self.process.poll() is not None:
				return None
			ready, _, _ = select.select([self.conn], [], [], min(left, 1.0))
			if ready:
				chunk = self.conn.recv(4096)
				if not chunk:
					return None
				self.pending += chunk
		line, self.pending = self.pending.split(b"\n", 1)
		return line

	def play(self, game, model_path):
		"""Play one game; its records (data.Record), one per side in game.recorded_sides."""
		from .data import parse_record

		if self.process is None or self.process.poll() is not None or self.battles >= self.max_battles:
			self._start()
		if self._line(120) is None:  # the client asks for its next game when the last one is over
			self._start()
			if self._line(120) is None:
				self.close_process()
				raise RuntimeError("the VCMI battle process did not ask for a game")
		self.battles += 1
		keys = [f"{os.getpid()}-{next(_server_numbers)}:{side}" for side in game.recorded_sides]
		name, env, settings, prefix = game_setup(game, model_path, [{"socket": self.records.path, "key": key} for key in keys])
		try:
			job = {"map": name, "env": env, "settings": settings, "clear_env": list(QUEUE_ENV_PREFIXES)}
			self.conn.sendall((json.dumps(job) + "\n").encode())
			deadline = time.monotonic() + self.battle_seconds
			while True:
				with self.records._lock:
					done = all(key.split("#")[0] in self.records.records for key in keys)
				if done:
					break
				if time.monotonic() > deadline or self.process.poll() is not None:
					status = "stalled" if self.process.poll() is None else f"exited ({self.process.poll()})"
					self.close_process()
					raise RuntimeError(f"battle did not finish cleanly ({status} in a battle process, map {name})")
				time.sleep(0.02)
		finally:
			if prefix:
				prefix.unlink(missing_ok=True)
		with self.records._lock:
			received = {key: self.records.records.pop(key) for key in keys}
		return [parse_record(received[key][-1]) for key in keys]

	def close_process(self):
		if self.conn:
			try:
				self.conn.sendall(b"{}\n")  # no more games
			except OSError:
				pass
			self.conn.close()
			self.conn = None
		if self.process and self.process.poll() is None:
			try:
				self.process.wait(timeout=10)
			except subprocess.TimeoutExpired:
				self.process.terminate()
				self.process.communicate()
		if self.process:
			output = self.process.communicate()[0] if self.process.stdout and not self.process.stdout.closed else ""
			run = next((line.split(":", 1)[1].strip() for line in (output or "").splitlines() if line.startswith("run:")), "")
			if run:
				shutil.rmtree(run, ignore_errors=True)
		self.process = None

	def close(self):
		self.close_process()
		self.records.__exit__(None, None, None)


def finish(process, keep_run=False):
	"""Wait for a battle started with start(); returns the runner's report (run dir, status, size).

	The runner's folder (VCMI logs and settings, ~1 MB) is deleted after a clean finish unless keep_run;
	it is kept when the battle failed, for debugging.
	"""
	output, _ = process.communicate()
	if getattr(process, "prefix_file", None):
		process.prefix_file.unlink(missing_ok=True)
	report = dict(line.split(":", 1) for line in output.splitlines() if ":" in line)
	report = {key.strip(): value.strip() for key, value in report.items()}
	status = report.get("status", "")
	if not status.startswith("finished") or not status.endswith("(exit code 0)"):  # killed by a limit, or crashed
		raise RuntimeError(f"battle did not finish cleanly ({status}; run {report.get('run')}):\n{output[-2000:]}")
	if not keep_run and report.get("run"):
		shutil.rmtree(report["run"], ignore_errors=True)
	return report


def play(games, model_path, record_root=None):
	"""Play several battles in parallel. Returns every battle's records (data.Record, in memory): per game in the
	order of `games`, one per side in game.recorded_sides (each record says which side it played).

	The records come over a RecordServer socket, never through files. With record_root they are also written
	there (game_<i>[/side_<s>] folders) for debugging.
	"""
	from .data import parse_record

	model_path = Path(model_path).resolve()  # VCMI runs in build/bin: paths must be absolute
	game_dirs = [Path(record_root).resolve() / f"game_{i}" for i in range(len(games))] if record_root else [None] * len(games)
	with RecordServer() as server:
		targets = [
			[{"socket": server.path, "key": f"{i}:{side}", "dir": folder} for side, folder in zip(game.recorded_sides, record_dirs(game, d) if d else [None] * len(game.recorded_sides))]
			for i, (game, d) in enumerate(zip(games, game_dirs))
		]
		processes = [start(game, model_path, t) for game, t in zip(games, targets)]
		reports, failures = [], []
		for process in processes:  # wait for every battle before raising, so none is left sending or writing its record
			try:
				reports.append(finish(process, keep_run=True))
			except RuntimeError as error:
				reports.append({})
				failures.append(str(error))
	received = server.records
	recorded = [all(t["key"] in received for t in game_targets) for game_targets in targets]
	for report, ok in zip(reports, recorded):
		if report.get("run") and ok:
			shutil.rmtree(report["run"], ignore_errors=True)
	# Failed battles keep their runner folders (VCMI logs) for debugging
	if failures:
		raise RuntimeError(f"{len(failures)} of {len(processes)} battles failed; first: {failures[0]}")
	missing = [(i, report) for i, (report, ok) in enumerate(zip(reports, recorded)) if report and not ok]
	if missing:
		raise RuntimeError("no record received from " + ", ".join(f"game {i} (run {report.get('run')})" for i, report in missing))
	return [parse_record(received[t["key"]][-1]) for game_targets in targets for t in game_targets]
