"""The battle learner's side of the training data: its model versions (the inference server's "current", the
league's snapshots), its buffer of battles waiting for an update, and take(), which fills an update from it.

The battles themselves are played by supply.py's loader workers (generated battles; the campaign games' battles come
with the games), here or on other machines, as the learner takes them: no actor threads here since 2026-10-30.

	the buffer         every battle item (supply.prepare) waits there until an update takes it, at most max_record_age
	                   model versions old
	take()             an update's battles: real_share of its decisions from the campaign games' battles ("real"), the
	                   rest generated (filled from the other source when one is short), at most game_share of them from
	                   any one game, oldest first; each battle is used by one update (its PPO passes), then dropped
	generated_room()   what the generated battles' workers may still make: the next `ahead` updates' generated share
	                   (more when the campaign battles waiting fall short of theirs): supply.Supply pulls no more than
	                   that, and the loader's prefetch parks the workers

Inference (config.inference): "server", one process (inference.py) runs the model for every game, batched, with the
learner's weights sent in memory after each update; past versions for the league are kept by it as "v<n>" (and saved
as models/snapshot_<n>.pt, so a resumed run has them). "onnx" (each version exported behind models/current.onnx) is
not used with the supply.
"""
import os
import threading
import time
from pathlib import Path

from . import remote
from .data import expand_record
from .inference import InferenceServer

KEEP_VERSIONS = 6  # the newest model versions kept for the actors (older ones are deleted unless a league snapshot)
LEAGUE_EVERY = 50  # every this many versions one is kept for good, as a past opponent


class ActorPool:
	def __init__(self, config, out, generated_games=None, model_kwargs=None, trained_epochs=0, server=None):
		"""server: a view of an inference server shared with the campaign model (inference.InferenceServer.view), or None
		for one of this pool's own."""
		self.config = config
		self.models = out / "models"
		self.models.mkdir(parents=True, exist_ok=True)
		self.link = self.models / "current.onnx"
		# A version is the epoch that trained it (publish adds 1: the first is the model as it starts). A resumed run
		# goes on from its epoch, and past its newest model (else it would overwrite its league snapshots): from the
		# newest model alone it fell behind the epochs by up to LEAGUE_EVERY each resume
		newest = max((int(p.stem.split("_")[1]) for p in [*self.models.glob("model_*.onnx"), *self.models.glob("snapshot_*.pt")]), default=-1)
		self.version = max(newest, trained_epochs - 1)
		self.server = server
		self.current_state = None  # the published weights (a CPU state_dict), pinned as "eval" for a campaign evaluation
		if config.inference == "server" and server is None:
			from . import play

			play.SOCKET_DIR.mkdir(parents=True, exist_ok=True)
			self.socket = play.SOCKET_DIR / f"infer-{os.getpid()}.sock"
			device = inference_device(config.inference_device)
			processes = config.inference_processes or (1 if device != "cpu" else max(1, (os.cpu_count() or 4) // 5))
			# (and over TCP for workers on other machines: remote.py)
			tcp_port = config.remote_port + remote.BATTLE_PORT if config.remote_port else 0
			self.server = InferenceServer(self.socket, device=device, model_kwargs=model_kwargs, threads=config.inference_threads, processes=processes, tcp_port=tcp_port)
		if self.server is not None:
			import torch

			for path in sorted(self.models.glob("snapshot_*.pt")):  # the league of a resumed run, by the names past_versions gives
				version = int(path.stem.split("_")[1])
				self.server.set_model(f"v{version}", version, torch.load(path, map_location="cpu"))
		self.buffer = []  # supply.prepare items: data, source, game, version, decisions, summary, ...
		self.lock = threading.Condition()
		self.stats = {"failures": 0, "dropped_stale": 0, "dropped_overflow": 0, "battles": {"generated": 0, "real": 0}, "last_failure": ""}

	# ---- models ----

	def publish(self, model, export_to=None):
		"""A new model version: the server gets the weights of model (export_to, for the ONNX mode, is not used)."""
		import torch

		self.version += 1
		state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
		self.current_state = state
		if self.server:
			self.server.set_model("current", self.version, state)
			if self.version % LEAGUE_EVERY == 0:  # kept for good, as a past opponent
				torch.save(state, self.models / f"snapshot_{self.version:05d}.pt")
				self.server.set_model(f"v{self.version}", self.version, state)
		with self.lock:
			self.lock.notify_all()

	def past_versions(self):
		"""The league's past versions: server model names ("v350")."""
		# Not version 0: the untrained model, a free win that teaches nothing
		return [f"v{int(p.stem.split('_')[1])}" for p in sorted(self.models.glob("snapshot_*.pt")) if 0 < int(p.stem.split("_")[1]) < self.version]

	def pin_eval(self):
		"""The current weights kept as the server's "eval" model, for an evaluation that must not see them change
		under it (a campaign evaluation's battles: 2-3 minutes, several updates); returns its version."""
		if self.server and self.current_state is not None:
			self.server.set_model("eval", self.version, self.current_state)
		return self.version

	def eval_settings(self, game):
		"""The env of our battles in a campaign evaluation game, greedy, not recorded: the weights pinned by pin_eval
		(2026-10-28 review: the two evals measured different battle models; 2026-10-30: the model changing during the
		evaluation)."""
		settings = {"ALPHA_BATTLE_MODEL": str(self.link.absolute()), "ALPHA_BATTLE_TEMPERATURE": "0", "ALPHA_BATTLE_REQUIRE_MODEL": "1"}
		if self.server:
			settings.update(ALPHA_BATTLE_INFERENCE_SOCKET=self.server.socket_for(game), ALPHA_BATTLE_INFERENCE_MODEL="eval")
		return tuple(settings.items())

	def close(self):
		if self.server:
			self.server.close()

	# ---- the buffer ----

	def save_buffer(self, path):
		"""The battles not trained on yet, to path (at the run's stop: load_buffer takes them back at its resume)."""
		import torch

		with self.lock:
			buffer = list(self.buffer)
		if buffer:
			torch.save(buffer, str(path) + ".tmp")
			os.replace(str(path) + ".tmp", path)
			print(f"actors: {len(buffer)} battles not trained on yet saved ({sum(b['decisions'] for b in buffer)} decisions)", flush=True)

	def load_buffer(self, path):
		import torch

		if not Path(path).exists():
			return
		try:
			buffer = torch.load(path, map_location="cpu", weights_only=False)
		except Exception as error:
			print(f"actors: FAILURE: {path} unreadable: {str(error)[:200]}", flush=True)
			buffer = []
		with self.lock:
			self.buffer = buffer + self.buffer
		Path(path).unlink()
		print(f"actors: {len(buffer)} battles from before the stop back in the buffer", flush=True)

	def add_item(self, item):
		"""A battle from the supply (supply.prepare's item), into the buffer. The campaign games' battles come whatever
		the learner takes (a game cannot wait mid-way), so beyond buffer_decisions of them the oldest go at once, as
		take() would drop them (2026-10-30: they piled up through a long campaign update, 5 GB in the trainer)."""
		with self.lock:
			if item["data"] is not None:
				self.buffer.append(item)
				if item["source"] == "real":
					real = sum(b["decisions"] for b in self.buffer if b["source"] == "real")
					for b in list(self.buffer):
						if real <= self.config.buffer_decisions:
							break
						if b["source"] == "real":
							self.buffer.remove(b)
							real -= b["decisions"]
							self.stats["dropped_overflow"] += b["decisions"]
			self.stats["battles"][item["source"]] += 1
			self.lock.notify_all()

	def _failed(self, error):
		with self.lock:
			self.stats["failures"] += 1
			self.stats["last_failure"] = str(error)[:300]
		print(f"actors: a battle failed: {str(error)[:300]}", flush=True)

	def generated_room(self):
		"""The generated decisions the workers may still make (call with self.lock held): the next `ahead` updates'
		share of them, or more when the real games' battles waiting fall short of theirs, as _select then fills the
		update with generated ones (2026-10-28: campaign battles were ~20% of each update, not real_share's 40%, and with
		the generated actors held at their 60% the learner waited 13-50 s an update for the campaign's battles)."""
		config = self.config
		real = sum(b["decisions"] for b in self.buffer if b["source"] == "real" and b["version"] >= self.version - config.max_record_age)
		room = config.ahead * max((1 - config.real_share) * config.decisions_per_update, config.decisions_per_update - real / config.ahead)
		# No more than the buffer keeps: take() drops whatever is over buffer_decisions, oldest first, so a room past it
		# had battles played to be dropped (2026-10-31 review: the PC's 3 x 6,400 against a buffer of 8,000)
		waiting_real = sum(b["decisions"] for b in self.buffer if b["source"] == "real")
		return min(room, max(0, config.buffer_decisions - waiting_real))

	def generated_waiting(self):
		"""Decisions of generated battles in the buffer, young enough to be used (call with self.lock held)."""
		return sum(b["decisions"] for b in self.buffer if b["source"] == "generated" and b["version"] >= self.version - self.config.max_record_age)

	# ---- the learner's side ----

	def ready(self, decisions):
		"""Whether an update of this many decisions can be filled now."""
		with self.lock:
			self._drop_stale()
			_, _, taken = self._select(decisions)
			return sum(taken.values()) >= decisions

	def _drop_stale(self):
		stale = [b for b in self.buffer if b["version"] < self.version - self.config.max_record_age]
		self.stats["dropped_stale"] += len(stale)
		if stale:
			self.buffer = [b for b in self.buffer if b["version"] >= self.version - self.config.max_record_age]

	def take(self, decisions, timeout=3600, stop=None):
		"""The battles of the next update (supply items), and its statistics; waits until an update can be filled (stop,
		a callable: give up early when it returns True; then None)."""
		config = self.config
		deadline = time.time() + timeout
		with self.lock:
			while True:
				self._drop_stale()
				chosen, per_game, taken = self._select(decisions)
				if sum(taken.values()) >= decisions:
					break
				if time.time() > deadline:
					raise RuntimeError(f"actors: an update could not be filled in {timeout}s ({sum(taken.values())} decisions; {self.stats})")
				if stop is not None and stop():
					return None, {}
				self.lock.wait(timeout=1 if stop is not None else 5)
			ids = {id(b) for b in chosen}
			self.buffer = [b for b in self.buffer if id(b) not in ids]
			overflow = sum(b["decisions"] for b in self.buffer) - config.buffer_decisions
			while overflow > 0 and self.buffer:  # the oldest go first
				dropped = self.buffer.pop(0)["decisions"]
				overflow -= dropped
				self.stats["dropped_overflow"] += dropped
			self.lock.notify_all()  # the supply, waiting for room, goes on
			total = max(1, sum(taken.values()))
			stats = {
				"real_share": round(taken["real"] / total, 3),
				"games_in_update": len(per_game),
				"buffer_decisions": sum(b["decisions"] for b in self.buffer),
				"record_age": round(sum(self.version - b["version"] for b in chosen) / max(1, len(chosen)), 2),
				"actor_failures": self.stats["failures"],
				"dropped_stale": self.stats["dropped_stale"],
				"dropped_overflow": self.stats["dropped_overflow"],
			}
		return chosen, stats

	def _select(self, decisions):
		"""(battles, decisions per game, decisions per source) for an update, oldest first (within max_record_age: newest
		first left the long battles, which start under an older version, to age out): real_share from real games,
		the rest generated, the other source filling when one is short; no real game over game_share of the update
		(a generated battle is a game of its own, both of its sides taken together)."""
		config = self.config
		cap = max(1, int(config.game_share * decisions))
		chosen, per_game, taken = [], {}, {"generated": 0, "real": 0}
		chosen_ids = set()

		def pick(source, goal):
			for b in sorted((b for b in self.buffer if b["source"] == source and id(b) not in chosen_ids), key=lambda b: b["version"]):
				if taken[source] >= goal or sum(taken.values()) >= decisions:
					return
				if source == "real" and per_game.get(b["game"], 0) > 0 and per_game[b["game"]] + b["decisions"] > cap:
					continue
				chosen.append(b)
				chosen_ids.add(id(b))
				per_game[b["game"]] = per_game.get(b["game"], 0) + b["decisions"]
				taken[source] += b["decisions"]

		real_goal = config.real_share * decisions if config.campaign.games else 0.0
		pick("real", real_goal)
		pick("generated", decisions - real_goal)
		pick("real", decisions)  # one source short: the other fills the update
		pick("generated", decisions)
		return chosen, per_game, taken


def records_of(items):
	"""The compact records some items carry (supply.TRACE_EVERY), expanded: for the run's traces."""
	return [expand_record(item["record"]) for item in items if item.get("record") is not None]


def inference_device(name):
	"""The inference server's device: "auto" is CUDA when there is one, else the CPU (on Apple's GPU this model's many
	small operations run slower than on a few CPU threads, and it cannot be shared with the learner's thread)."""
	if name != "auto":
		return name
	import torch

	return "cuda" if torch.cuda.is_available() else "cpu"


Path  # (the models folder is a Path from the run's folder)
