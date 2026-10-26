"""Training data from many battles at once: actors play all the time, the learner trains in updates (train.py with
actors=true), so the examples of an update come from many games instead of one game's consecutive battles.

	generated actors   each plays one generated battle after another (generator.py): self-play, or against a past
	                   version of the model or a fixed AI (the league: config.league)
	real-game actors   each plays whole real games (real_games.py), every battle of them by the current model; their
	                   records stream in while the game goes on
	the buffer         every record waits there until an update takes it, at most max_record_age model versions old
	take()             an update's battles: real_share of its decisions from real games, the rest generated (filled
	                   from the other source when one is short), at most game_share of them from any one game, newest
	                   first; each battle is used by one update (its PPO passes), then dropped

TODO(hard cases): battles where a person beat the model (exported as scenarios by AlphaBattleAI's "scenario" event)
kept in a pool of their own, a small share of every update's generated battles and of every eval, so an exploit found
by playing against the model is trained away and checked from then on.

Inference (config.inference):
	"server"   the default: one inference process (inference.py) runs the model for every game, batched, with the
	           learner's weights sent in memory after each update (no export); a game creates no ONNX session (1.1 s of
	           CPU a battle) and spends no CPU core on the model. Past versions for the league are kept by it as "v<n>"
	           (and saved as models/snapshot_<n>.pt, so a resumed run has them).
	"onnx"     each version exported (models/model_<v>.onnx) behind the link models/current.onnx, which a battle loads
	           at its start.
Either way a record says which version played (the start event's "model"), so stale records can be dropped.
"""
import json
import os
import random
import re
import shutil
import threading
import time
from pathlib import Path

from . import generator, play, real_games
from .data import parse_record
from .inference import InferenceServer

KEEP_VERSIONS = 6  # the newest model versions kept for the actors (older ones are deleted unless a league snapshot)
LEAGUE_EVERY = 50  # every this many versions one is kept for good, as a past opponent


class ActorPool:
	def __init__(self, config, out, generated_games, model_kwargs=None):
		self.config = config
		self.models = out / "models"
		self.models.mkdir(parents=True, exist_ok=True)
		self.link = self.models / "current.onnx"
		self.generated_games = generated_games  # train.generated_games
		# A resumed run goes on numbering from its newest model (else it would overwrite its league snapshots)
		self.version = max((int(p.stem.split("_")[1]) for p in [*self.models.glob("model_*.onnx"), *self.models.glob("snapshot_*.pt")]), default=-1)
		self.server = None
		if config.inference == "server":
			play.SOCKET_DIR.mkdir(parents=True, exist_ok=True)
			self.socket = play.SOCKET_DIR / f"infer-{os.getpid()}.sock"
			device = inference_device(config.inference_device)
			processes = config.inference_processes or (1 if device != "cpu" else max(1, (os.cpu_count() or 4) // 5))
			self.server = InferenceServer(self.socket, device=device, model_kwargs=model_kwargs, threads=config.inference_threads, processes=processes)
			import torch

			for path in sorted(self.models.glob("snapshot_*.pt")):  # the league of a resumed run
				self.server.set_model(path.stem.replace("snapshot_", "v"), int(path.stem.split("_")[1]), torch.load(path, map_location="cpu"))
		self.buffer = []  # dicts: record, source, game, version, decisions
		self.lock = threading.Condition()
		self.stop = threading.Event()
		self.running = threading.Event()  # cleared while the learner evaluates: generated actors start no new battle
		self.running.set()
		self.stats = {"failures": 0, "dropped_stale": 0, "battles": {"generated": 0, "real": 0}, "last_failure": ""}
		self.records = play.RecordServer()
		self.threads = []

	# ---- models ----

	def publish(self, model, export_to):
		"""A new model version: the server gets the weights of model, or export_to(path) writes it and the link is
		switched to it atomically."""
		self.version += 1
		if self.server:
			self.server.set_model("current", self.version, model)
			if self.version % LEAGUE_EVERY == 0:  # kept for good, as a past opponent
				import torch

				state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
				torch.save(state, self.models / f"snapshot_{self.version:05d}.pt")
				self.server.set_model(f"v{self.version}", self.version, state)
			with self.lock:
				self.lock.notify_all()
			return
		path = self.models / f"model_{self.version:05d}.onnx"
		export_to(path)
		temporary = self.models / f".current-{self.version}.onnx"
		temporary.unlink(missing_ok=True)
		temporary.symlink_to(path.name)
		os.replace(temporary, self.link)
		for old in self.models.glob("model_*.onnx"):  # keep the recent ones and the league's snapshots
			v = int(old.stem.split("_")[1])
			if v < self.version - KEEP_VERSIONS and v % LEAGUE_EVERY:
				old.unlink(missing_ok=True)
		with self.lock:
			self.lock.notify_all()

	def past_versions(self):
		"""The league's past versions: server model names ("v350") or ONNX files."""
		if self.server:
			return [f"v{int(p.stem.split('_')[1])}" for p in sorted(self.models.glob("snapshot_*.pt")) if int(p.stem.split("_")[1]) < self.version]
		return sorted(p for p in self.models.glob("model_*.onnx") if int(p.stem.split("_")[1]) % LEAGUE_EVERY == 0 and int(p.stem.split("_")[1]) < self.version)

	@staticmethod
	def version_of(record):
		"""The version that played: "model_<v>.onnx" (ONNX) or "remote:<name>@<v>" (the inference server)."""
		start = next((e for e in record.events if e.get("event") == "start"), {})
		match = re.search(r"model_(\d+)\.onnx|@(\d+)$", start.get("model", ""))
		return int(match.group(1) or match.group(2)) if match else -1

	def inference_settings(self, actor):
		"""The settings (env) that send an actor's battles to the inference server (one of its processes), if any."""
		return (("ALPHA_BATTLE_INFERENCE_SOCKET", self.server.socket_for(actor)),) if self.server else ()

	# ---- actors ----

	def start(self):
		for i in range(self.config.generated_actors):
			self.threads.append(threading.Thread(target=self._generated_actor, args=(i,), daemon=True))
		for i in range(self.config.real_games):
			self.threads.append(threading.Thread(target=self._real_actor, args=(i,), daemon=True))
		if self.config.real_games:
			self.threads.append(threading.Thread(target=self._collector, daemon=True))
		for t in self.threads:
			t.start()

	def close(self):
		self.stop.set()
		for t in self.threads:
			t.join(timeout=60)
		self.records.__exit__(None, None, None)
		if self.server:
			self.server.close()

	def _add(self, records, source, game):
		with self.lock:
			for r in records:
				if any(e.get("event") == "error" for e in r.events):  # a battle the model did not play: never trained on
					self.stats["failures"] += 1
					self.stats["last_failure"] = message = next(e for e in r.events if e.get("event") == "error").get("message", "")[:300]
					print(f"actors: a {source} battle had an error ({game}): {message}", flush=True)
					continue
				self.buffer.append({"record": r, "source": source, "game": game, "version": self.version_of(r), "decisions": r.decisions})
				self.stats["battles"][source] += 1
			self.lock.notify_all()

	def _failed(self, error):
		with self.lock:
			self.stats["failures"] += 1
			self.stats["last_failure"] = str(error)[:300]
		print(f"actors: a battle failed: {str(error)[:300]}", flush=True)

	def _opponent(self, rng):
		"""(opponent, settings for it): the league's draw for one generated battle."""
		self_play, past, fixed = self.config.league
		roll, past_versions = rng.random() * (self_play + past + fixed), self.past_versions()
		if roll < self_play or (roll < self_play + past and not past_versions):
			return play.SELF_PLAY, None
		if roll < self_play + past:
			return "AlphaBattleAI", rng.choice(past_versions)
		return rng.choice(self.config.league_ais), None

	def pause(self):
		"""No new generated battles (an eval gets the machine; real games run on at low priority)."""
		self.running.clear()

	def resume(self):
		self.running.set()

	def _generated_actor(self, index):
		rng = random.Random(1_000_003 * (self.config.seed + 1) + 7919 * index)
		n = 0
		process = play.BattleProcess() if self.config.persistent_battles else None  # one VCMI client for its battles
		while not self.stop.is_set():
			if self.version < 0 or not self.running.wait(timeout=1):
				continue
			n += 1
			scenario = generator.sample(rng)
			opponent, past_model = self._opponent(rng)
			our_side = 0 if opponent == play.SELF_PLAY else rng.randrange(2)
			files = []
			try:
				games, files = self.generated_games([scenario], f"actor{index}_{n}", self.config, our_sides=(our_side,), opponent=opponent, temperature=1.0)
				game = play.replace(games[0], settings=games[0].settings + self.inference_settings(index))
				if past_model is not None:  # the other side is our AI with an older model, not recorded
					key = "ALPHA_BATTLE_INFERENCE_MODEL" if self.server else "ALPHA_BATTLE_MODEL"
					game = play.replace(game, settings=game.settings + ((f"{key}_{game.colors[1 - our_side]}", str(past_model)),))
				game = play.replace(game, sampler_seed=rng.randrange(2**31))
				records = process.play(game, self.link) if process else play.play([game], self.link)
				self._add(records, "generated", f"g{index}_{n}")
			except Exception as error:  # one failed battle (its runner folder is kept) does not stop training
				self._failed(error)
			finally:
				for path in files:
					path.unlink(missing_ok=True)
		if process:
			process.close()

	def _real_actor(self, index):
		rng = random.Random(7 * (self.config.seed + 1) + 104729 * index)
		maps = real_games.maps()
		n = 0
		while not self.stop.is_set():
			if self.version < 0:
				time.sleep(1)
				continue
			n += 1
			process = real_games.start_game(rng.choice(maps), self.link, self.records.path, f"r{index}_{n}", rng.randrange(2**31), max_seconds=self.config.real_game_seconds, settings=self.inference_settings(self.config.generated_actors + index))
			while process.poll() is None and not self.stop.is_set():
				time.sleep(1)
			if process.poll() is None:  # training is over
				process.terminate()
				process.communicate()
				return
			output = process.communicate()[0] or ""
			status = next((line for line in output.splitlines() if line.startswith("status:")), "")
			run = next((line.split(":", 1)[1].strip() for line in output.splitlines() if line.startswith("run:")), "")
			if "exit code 0" in status or "time limit" in status:  # a game cut at its time limit is normal
				shutil.rmtree(run, ignore_errors=True)
			else:
				self._failed(f"real game {index}/{n}: {status} (run {run})")

	def _collector(self):
		while not self.stop.is_set():
			time.sleep(1)
			for key, records in self.records.drain().items():
				parsed = []
				for data in records:
					try:
						parsed.append(parse_record(data))
					except Exception as error:
						self._failed(f"unreadable record from {key}: {error}")
				self._add(parsed, "real", key.split(":", 1)[0])

	# ---- the learner's side ----

	def take(self, decisions, timeout=3600):
		"""The battles of the next update (records), and its statistics; waits until an update can be filled."""
		config = self.config
		deadline = time.time() + timeout
		with self.lock:
			while True:
				stale = [b for b in self.buffer if b["version"] < self.version - config.max_record_age]
				self.stats["dropped_stale"] += len(stale)
				self.buffer = [b for b in self.buffer if b["version"] >= self.version - config.max_record_age]
				chosen, per_game, taken = self._select(decisions)
				if sum(taken.values()) >= decisions:
					break
				if time.time() > deadline or self.stop.is_set():
					raise RuntimeError(f"actors: an update could not be filled in {timeout}s ({sum(taken.values())} decisions; {self.stats})")
				self.lock.wait(timeout=5)
			ids = {id(b) for b in chosen}
			self.buffer = [b for b in self.buffer if id(b) not in ids]
			overflow = sum(b["decisions"] for b in self.buffer) - config.buffer_decisions
			while overflow > 0 and self.buffer:  # the oldest go first
				overflow -= self.buffer.pop(0)["decisions"]
			total = max(1, sum(taken.values()))
			stats = {
				"real_share": round(taken["real"] / total, 3),
				"games_in_update": len(per_game),
				"buffer_decisions": sum(b["decisions"] for b in self.buffer),
				"record_age": round(sum(self.version - b["version"] for b in chosen) / max(1, len(chosen)), 2),
				"actor_failures": self.stats["failures"],
			}
		return [b["record"] for b in chosen], stats

	def _select(self, decisions):
		"""(battles, decisions per game, decisions per source) for an update, newest first: real_share from real games,
		the rest generated, the other source filling when one is short; no real game over game_share of the update
		(a generated battle is a game of its own, both of its sides taken together)."""
		config = self.config
		cap = max(1, int(config.game_share * decisions))
		chosen, per_game, taken = [], {}, {"generated": 0, "real": 0}
		chosen_ids = set()

		def pick(source, goal):
			for b in sorted((b for b in self.buffer if b["source"] == source and id(b) not in chosen_ids), key=lambda b: -b["version"]):
				if taken[source] >= goal or sum(taken.values()) >= decisions:
					return
				if source == "real" and per_game.get(b["game"], 0) > 0 and per_game[b["game"]] + b["decisions"] > cap:
					continue
				chosen.append(b)
				chosen_ids.add(id(b))
				per_game[b["game"]] = per_game.get(b["game"], 0) + b["decisions"]
				taken[source] += b["decisions"]

		real_goal = config.real_share * decisions if config.real_games else 0.0
		pick("real", real_goal)
		pick("generated", decisions - real_goal)
		pick("real", decisions)  # one source short: the other fills the update
		pick("generated", decisions)
		return chosen, per_game, taken


def inference_device(name):
	"""The inference server's device: "auto" is CUDA when there is one, else the CPU (on Apple's GPU this model's many
	small operations run slower than on a few CPU threads, and it cannot be shared with the learner's thread)."""
	if name != "auto":
		return name
	import torch

	return "cuda" if torch.cuda.is_available() else "cpu"
