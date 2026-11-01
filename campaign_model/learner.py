"""The campaign model's learner in a joint run (the user, 2026-10-27: "training the campaign comes alongside training
the battle model (that has its own optimizer etc...)"): battle_model/train.py runs it when campaign.games > 0.

	games        self-play games, each on a new random map (2-4 players, water and the underground on, a day cap drawn
	             per game: draw_days), played by battle_model/supply.py's loader workers as this learner takes them (up
	             to config.games at once on this machine): every player our campaign model (the inference server,
	             family "campaign"), every player's battle our battle model (the wandering monsters BattleAI's). The
	             battles go to the battle learner as real games' (actors.py: its real_share, with the generated battles,
	             hard cases among them, as the rest, so it does not forget them); the finished games come here (_game_done)
	update       once decisions_per_update decisions of at least min_games_per_update finished games are in: their
	             labels (labels.py: the real shares and our own growth, the longer heads graded against the shorter
	             ones' forecasts at each player's day starts, made by the model as it is now: day_starts; the share
	             at the game's end, V_end graded against itself a day on), the policy's credit (the 7-day one, moving
	             to the end's as V_end proves itself: end_credit), one PPO
	             update (train.ppo_update) with this learner's own optimizer, in a process
	             of its own (UpdateProcess, learner "process": on Apple's GPU or CUDA next to the battle learner, which
	             two threads of one process cannot share), or in a thread; the new weights go to the campaign server
	             at once
	evaluation   every eval_minutes, games on fixed maps against Nullkiller2 (evaluate.py)

Logs: <run>/campaign/progress.csv and eval.csv, TensorBoard <run>/tensorboard/campaign (campaign/..., campaign_eval/...),
checkpoints <run>/campaign/checkpoint_latest.pt (and one per evaluation).
"""
import copy
import csv
import json
import math
import multiprocessing
import os
import shutil
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from battle_model import play as battle_play
from battle_model.device import use_fused
from battle_model.inference import InferenceServer

from . import constants as c
from . import evaluate
from .empire import EmpireConfig, empire_value
from .failures import NAMES as FAILURES, failure_modes
from .features import SKILLS
from .labels import (
	COLORS, GROWTH_CAP, PHI_FLOOR, LabelConfig, attributed_labels, credit_evidence, credit_gaps, credit_trust, end_labels, end_steps, end_targets, growth_labels, horizon_advantages,
	label, ladder_targets, pooled_ev, residual_variances, share_change_labels,
)
from .foresight import labels_of
from .labeler import Labeler
from .model import build_model
from .records import NO_EFFECT, TEACHER, decode_fields, decode_inputs
from .train import _prefetched, load_optimizer, make_optimizer, migrate_subtypes, model_kwargs, ppo_update, usable

TYPES = [a.name.lower() for a in c.ActionType]
FORESIGHT_COLUMNS = tuple(f"foresight_{head}_{part}" for head in ("move", "answer") for part in ("battle", "score", "parts")) + ("foresight_hero_alive", "foresight_hero_army", "foresight_turn")
COLUMNS = [
	"version", "epoch", "games", "decisions", "dropped_old", "unsound", "credited", "days", "players", "phi_end", "heroes", "label_mean", "price_mean",
	"trust", "credit_trust", "end_credit", "policy_loss", "value_loss", "value_loss_fresh", "entropy", "below_floor", "kl", "last_kl", "teacher_kl", "teacher_weight", "advised", "teacher_agree", "teacher_ce", "clip_fraction", "grad_norm", "steps", "learning_rate",
	*(f"value_ev_d{k}" for k in c.VALUE_OFFSETS), *(f"head_ev_{kind}{k}" for kind in "vg" for k in c.VALUE_OFFSETS), "head_ev_vend", "no_effect", "ai_errors", "failed_actions", "game_failures", "game_maps", "dropped_games",
	"game_seconds", "record_mb", "seconds", "gpu_mb", "label_staleness", *(f"share_{t}" for t in TYPES), *(f"failure_{k}" for k in FAILURES), "no_effect_types", "kinds",
	# v2's foresight losses (foresight.py) and the gradient check: the shared layers' gradient from the PPO losses and the
	# foresight's, their cosine, the share of parameters no loss reaches
	*FORESIGHT_COLUMNS, "grad_shared_main", "grad_shared_foresight", "grad_shared_cosine", "grad_dead_share",
]
KEEP_FAILED_RUNS = 10  # run folders (logs) of failed games kept for debugging; the rest are deleted
DROP_BEYOND = 4  # updates' worth of finished games beyond which the oldest are dropped (_take)
MAX_LABEL_STALENESS = 2  # model versions: older precomputed labels (labeler.py) are recomputed with the current tables


def campaign_model_kwargs(config, out, resume=False):
	"""What builds the campaign model this run trains: a resumed run's checkpoint says, else the config (model_kwargs);
	for an inference server made before the learner (battle_model/train.py: one server for both models)."""
	latest = Path(out) / "campaign" / "checkpoint_latest.pt"
	if resume and latest.exists():
		return torch.load(latest, map_location="cpu", weights_only=False)["model_kwargs"]
	if config.init_from:
		return torch.load(config.init_from, map_location="cpu", weights_only=False).get("model_kwargs") or model_kwargs(config)
	return model_kwargs(config)


def game_problem(game, players):
	"""Why a finished game must not be trained on (CampaignLearner._game_done), or None: an AI error, or fewer turn
	decisions than half a decision a player and day."""
	if not game.decisions:  # (a game that ended before its first decision: ALPHA_MAP_REQUIRE_MODEL's abort among them)
		return "no decision"
	if any(d.flags & TEACHER for d in game.decisions):  # (its choices are Nullkiller2's: no policy probability to train on)
		return "a teacher's game (AlphaTeacherAI)"
	errors = {color: summary.get("errors", 0) for color, summary in game.done.items() if summary.get("errors", 0)}
	if errors:
		return "AI errors " + ", ".join(f"{color} {n}" for color, n in sorted(errors.items()))
	days = max(game.ledgers) if game.ledgers else 0
	turns = sum(1 for d in game.decisions if d.kind == c.DecisionKind.TURN)
	if days and turns < 0.5 * players * days:
		return f"{turns} turn decisions in {days} days of {players} players"
	return None


class CampaignLearner:
	def __init__(self, config, out, device, battle, resume=False, server=None):
		"""out: the run's folder; battle: the battle learner's actors (actors.ActorPool: the evaluation's battle settings);
		server: an inference server view for the family "campaign" (a server shared with the battle model's), or None for
		one of its own."""
		self.config = config
		self.run = Path(out)
		self.out = self.run / "campaign"
		self.out.mkdir(parents=True, exist_ok=True)
		self.device = device
		self.battle = battle
		self.labels = LabelConfig(weights=tuple(config.label_weights), decision_price=config.decision_price)
		self.empire = EmpireConfig()
		torch.manual_seed(config.seed)
		self.kwargs = model_kwargs(config)
		self.worker = None
		if config.learner == "process":  # the updates' device is the process's; here the model only serves and evaluates
			device = torch.device("cpu")
			self.device = device
		self.model = build_model(self.kwargs).to(device).eval()
		self.optimizer = make_optimizer(self.model, config)
		self.version, self.lr_scale, self.games_used = 1, 1.0, 0  # version: the served model's (1: as the run starts)
		self.teacher_weight = config.teacher_kl  # (the teacher KL's weight, adapted after each update: config.teacher_kl_target)
		self.value_ev = None  # the value heads' explained variance, smoothed over ~10 updates: how far the credit trusts them
		self.credit_evidence = {}  # the credit check's (labels.credit_evidence): the step's errors by action type, kept
		self.credit_check = 1.0  # and its verdict, how far the step's estimate may be trusted (labels.credit_trust)
		self.end_evidence = {}  # the same for V_end's steps, the end credit's (campaign_end_credit/*)
		self.end_check = 1.0  # and its verdict (end_credit: config.end_credit_check)
		self.head_ev = {}  # each head's explained variance at the day starts, smoothed: how far it grades the next (grade_trust)
		self.value_pool, self.head_pool = None, {}  # their parts (labels.pooled_ev): the residual variance and the variance, smoothed apart
		latest = self.out / "checkpoint_latest.pt"
		if resume and latest.exists():
			state = torch.load(latest, map_location=device)
			self.kwargs = state["model_kwargs"]
			self.model = build_model(self.kwargs).to(device).eval()
			self.optimizer = make_optimizer(self.model, config)
			if migrated := migrate_subtypes(state, self.model):
				print(f"campaign: subtype tables folded to {c.SUBTYPE_ROWS} rows (train.migrate_subtypes): {', '.join(migrated)}", flush=True)
			self.model.load_state_dict(state["model"])
			if "optimizer" in state:
				load_optimizer(self.optimizer, self.model, state["optimizer"], state["model"].keys())
			self.version, self.lr_scale, self.games_used = state["version"], state.get("lr_scale", 1.0), state.get("games_used", 0)
			self.teacher_weight = state.get("teacher_weight", config.teacher_kl)
			self.value_ev = state.get("value_ev")
			self.credit_evidence = state.get("credit_evidence", {})
			self.credit_check = credit_trust(credit_gaps(self.credit_evidence))
			self.end_evidence = state.get("end_evidence", {})
			self.end_check = credit_trust(credit_gaps(self.end_evidence))
			self.head_ev = state.get("head_ev", {})
			self.value_pool, self.head_pool = state.get("value_pool"), state.get("head_pool", {})
			print(f"campaign: resuming at version {self.version}", flush=True)
		elif config.init_from:
			state = torch.load(config.init_from, map_location=device)
			self.kwargs = state.get("model_kwargs") or self.kwargs
			self.model = build_model(self.kwargs).to(device).eval()
			self.optimizer = make_optimizer(self.model, config)
			migrate_subtypes(state, self.model)
			self.model.load_state_dict(state["model"])
			print(f"campaign: starting from {config.init_from}", flush=True)
		# The teacher (config.teacher_kl): a frozen imitation of Nullkiller2 whose KL the update adds
		self.teacher = None
		if config.teacher_kl > 0 and config.learner == "process":
			raise SystemExit("campaign.teacher_kl: the 'process' learner does not take the teacher (use 'turns' or 'thread')")
		if config.teacher_kl > 0 and not config.teacher_checkpoint and config.nk2_advice <= 0:
			raise SystemExit("campaign.teacher_kl needs a teacher: campaign.teacher_checkpoint (a model's KL) or campaign.nk2_advice (Nullkiller2's own)")
		if config.teacher_kl > 0 and config.teacher_checkpoint:  # (with nk2_advice alone, teacher_kl is only the teacher's weight)
			state = torch.load(config.teacher_checkpoint, map_location="cpu")
			self.teacher = build_model(state.get("model_kwargs") or self.kwargs)
			migrate_subtypes(state, self.teacher)
			self.teacher.load_state_dict(state["model"])
			self.teacher = self.teacher.to(device).eval().requires_grad_(False)
			print(f"campaign: the teacher's KL at {config.teacher_kl}, the teacher {config.teacher_checkpoint}", flush=True)
		# The features module on the CPU (its priors' corrections are learned: refreshed after each update) for the
		# foresight's labels, computed from decoded inputs on the CPU while the model trains on the GPU (learner "turns")
		self.features_cpu = copy.deepcopy(self.model.features).cpu()
		# The labels of each game as it arrives, in processes of their own (labeler.py): the trainer's thread never waits
		self.labeler = Labeler(self.labels, self.empire, foresight=self.kwargs.get("architecture") == "v2", share_target=config.share_target)
		if config.credit == "schedule" and config.share_target != "change":
			raise SystemExit("campaign.credit 'schedule' reads the share heads as changes: it needs campaign.share_target 'change'")
		if config.share_target == "change" and config.credit in ("horizon", "mix"):
			raise SystemExit(f"campaign.share_target 'change' with credit '{config.credit}': that credit reads the heads as shares (use 'attributed')")
		self.label_futures = {}  # id(game) -> its Future (precomputed takes it)
		if config.learner == "process":
			self.worker = UpdateProcess(config, self.kwargs, self.model, self.optimizer, self.lr_scale, latest)
		battle_play.SOCKET_DIR.mkdir(parents=True, exist_ok=True)
		from battle_model.actors import inference_device

		from battle_model import remote

		port = battle.config.remote_port  # (workers on other machines: remote.py)
		self.own_server = server is None
		self.server = server or InferenceServer(battle_play.SOCKET_DIR / f"campaign-{os.getpid()}.sock", device=inference_device(config.inference_device), family="campaign", model_kwargs=self.kwargs, threads=config.inference_threads, processes=config.inference_processes, tcp_port=port + remote.CAMPAIGN_PORT if port else 0)
		self.server.set_model("current", self.version, self.model)
		if self.device.type != "cpu":
			use_fused(self.optimizer)
		self.finished = []  # finished games: (records.Game, play.GameInfo), from supply.Supply
		self.lock = threading.Condition()
		self.stop = threading.Event()
		self.running = threading.Event()  # (kept for workers on the old protocol: always set, evaluations pause nothing)
		self.running.set()
		self.evaluation = None  # the evaluation's thread while one plays (start_eval)
		self.eval_error = None
		self.stats = Counter()
		self.model_lock = threading.Lock()  # the learner thread's update against a checkpoint taken by an evaluation
		self.battle_epoch = 0
		self.key_prefix = ""  # (a worker's games: its name first, remote.Hub)
		self.failed_runs = []
		self.threads = []
		self.writer = SummaryWriter(self.run / "tensorboard" / "campaign")
		self.progress = self.out / "progress.csv"
		if not self.progress.exists() or not resume:
			with self.progress.open("w", newline="") as f:
				csv.writer(f).writerow(COLUMNS)
		else:  # a resume after columns were added: the old rows get them, empty (rows were misaligned without this)
			from battle_model.train import upgrade_csv

			upgrade_csv(self.progress, COLUMNS)
		# A new run evaluates its starting model first; a resumed one when eval_minutes have passed since its last eval
		# (by eval.csv's time: restarts must not keep putting evals off)
		evals = self.out / "eval.csv"
		if resume and evals.exists():  # (its columns grew: vs_previous, vs_first)
			from battle_model.train import upgrade_csv

			upgrade_csv(evals, evaluate.COLUMNS)
		self.last_eval = time.monotonic() - (time.time() - evals.stat().st_mtime) if resume and evals.exists() else None

	# ---- games ----

	def start(self):
		"""The games are the supply's (battle_model/supply.py); here only the learner thread, in its modes."""
		if self.config.learner in ("thread", "process"):
			self.threads.append(threading.Thread(target=self._learn, daemon=True))
		for thread in self.threads:
			thread.start()

	def close(self):
		self.stop.set()
		with self.lock:
			self.lock.notify_all()
		for thread in self.threads:
			thread.join(timeout=150)
		if self.evaluation is not None:
			self.evaluation.join(timeout=600)
		if getattr(self, "labeler", None):
			self.labeler.close()
		if self.worker:
			self.worker.close()
		if self.own_server:
			self.server.close()
		self.writer.close()

	def note_epoch(self, epoch):
		"""The battle learner's latest update, for the logs (learner "thread")."""
		self.battle_epoch = epoch

	def _learn(self):
		"""The campaign learner's own thread: an update whenever its games are in."""
		if self.device.type == "cpu" and self.config.torch_threads > 0 and not self.worker:  # (the process sets its own)
			torch.set_num_threads(self.config.torch_threads)
		while not self.stop.is_set():
			try:
				ran = self.step(self.battle_epoch)
			except Exception as error:  # an update that fails is reported, and the next one tried
				import traceback

				print(f"campaign: an update failed: {error}\n{traceback.format_exc()}", flush=True)
				self.stats["update_failures"] += 1
				ran = False
			if not ran:
				with self.lock:
					self.lock.wait(timeout=5)

	def _game_done(self, key, map_name, players, game, info, game_map):
		"""A finished game (one of this machine's, or a worker's: supply.Supply) for the next updates; its labels start now
		(labeler.py), with the features as they are."""
		self._log_game(key, map_name, players, game, info)
		# A game is checked before it is trained on, and one that fails the check is left out with a FAILURE: a sign of a
		# bug elsewhere. Any AI error (a good game reports none: 8,118 of joint5's 10,206 games had none, 46 one or two),
		# or fewer turn decisions than half a player-day's (a player in the game decides each day, if only to end its turn).
		# 2,042 games ran with the inference servers down, ended every turn unasked and reached their day caps with no
		# decision, and were taken as finished (2026-10-31)
		problem = game_problem(game, players)
		if problem:
			with self.lock:
				self.stats["rejected_games"] = left = self.stats.get("rejected_games", 0) + 1
			if left == 1 or left % 100 == 0:
				print(f"campaign: FAILURE: game {key} left out ({problem}); {left} games left out so far", flush=True)
			return
		if getattr(self, "labeler", None):
			game.labels_version = self.version
			self.label_futures[id(game)] = self.labeler.submit(game, self.features_cpu)  # (not on the game: it is pickled to the labeler after this returns)
		with self.lock:
			self.finished.append((game, info))
			self.stats["games"] += 1
			self.stats["game_maps"] += game_map
			self.lock.notify_all()

	def _log_game(self, key, map_name, players, game, info):
		"""One line per finished game in <run>/campaign/games.jsonl: how and when it ended, each player's end."""
		snapshots = sorted([*game.ledgers.values(), *game.final], key=lambda s: (s["day"], bool(s.get("final"))))
		end = snapshots[-1] if snapshots else {"players": [], "day": 0}
		line = {
			"key": key, "map": map_name, "players": players, "status": info.status, "seconds": round(info.seconds, 1), "version": self.version,
			"days": game.days, "last_day": max(game.ledgers) if game.ledgers else 0, "end_day": end["day"], "decisions": len(game.decisions),
			"end": {COLORS[p["color"]]: {"status": p["status"], "towns": len(p["towns"]), "heroes": len(p["heroes"])} for p in end["players"]},
			"ai": game.done,
		}
		with (self.out / "games.jsonl").open("a") as f:
			f.write(json.dumps(line) + "\n")

	def _failed(self, message, run_dir=None):
		with self.lock:
			self.stats["game_failures"] += 1
		print(f"campaign: {message[:400]}", flush=True)
		if run_dir:
			self.failed_runs.append(run_dir)
			while len(self.failed_runs) > KEEP_FAILED_RUNS:
				shutil.rmtree(self.failed_runs.pop(0), ignore_errors=True)

	# ---- updates ----

	def waiting_decisions(self):
		"""Decisions of the finished games no update has taken (the supply's permits keep them under `ahead` updates' worth)."""
		with self.lock:
			return sum(len(g.decisions) for g, _ in self.finished)

	def waiting_games(self):
		"""Finished games no update has taken (the supply's permits keep at least min_games_per_update coming)."""
		with self.lock:
			return len(self.finished)

	def ready(self):
		"""Whether an update's games are in (what _take would take)."""
		config = self.config
		with self.lock:
			return len(self.finished) >= config.min_games_per_update and sum(len(g.decisions) for g, _ in self.finished) >= config.decisions_per_update

	def _take(self):
		"""The games of the next update (oldest first), or None while too few have finished. The supply starts games only
		while the finished and running ones hold under `ahead` updates' worth (supply.Supply.allow), so nothing is played
		to be dropped; beyond DROP_BEYOND updates' worth (a safety: games already in flight when the learner stalled) the
		oldest are dropped (counted)."""
		config = self.config
		with self.lock:
			# (an update's worth: its decisions or its games' as the permits count it, whichever is more)
			mean = sum(len(g.decisions) for g, _ in self.finished) / max(1, len(self.finished))
			update = max(config.decisions_per_update, config.min_games_per_update * mean)
			while len(self.finished) > config.min_games_per_update and sum(len(g.decisions) for g, _ in self.finished[1:]) >= DROP_BEYOND * update:
				self.stats["dropped_games"] += 1
				dropped, _ = self.finished.pop(0)
				getattr(self, "label_futures", {}).pop(id(dropped), None)
			taken, decisions = [], 0
			for game, info in self.finished:
				if len(taken) >= config.min_games_per_update and decisions >= config.decisions_per_update:
					break
				taken.append((game, info))
				decisions += len(game.decisions)
			if len(taken) < config.min_games_per_update or decisions < config.decisions_per_update:
				return None
			self.finished = self.finished[len(taken) :]
		return taken

	def step(self, epoch):
		"""One update if its games are in (the battle learner calls this between its updates); True if it ran."""
		games = self._take()
		if games is None:
			return False
		started, memory = time.time(), [("taken", rss_gb())]
		played = []
		for game, _ in games:
			game.all_decisions = game.decisions  # (the foresight's labels count record serials: the battles' point at them)
			game.decisions = usable(game.decisions)
			if game.decisions and game.ledgers:
				played.append(game)
		if not played:
			return False
		starts, trust, (weight, outcome) = day_starts(self.model, played, self.model_lock), self.grade_trust(), self.end_credit()
		memory.append(("starts", rss_gb()))
		foresight = [] if self.kwargs.get("architecture") == "v2" else None  # (v2's foresight labels, each game's)
		decisions, targets, labels, growth, returns, advantages, errors, players, forecasts, row = [], [], [], [], [], [], [], [], [], Counter()
		growth_real, attributed = [], {"share_now": [], "taken": []}
		end = {"targets": [], "real": [], "step": [], "week": [], "price": []}
		for g, game in enumerate(played):
			# The labeler's labels, as the game arrived (b1, 2026-10-30: a game's labels from one table version, one update
			# old, act like a target network); past MAX_LABEL_STALENESS updates they are computed here with today's tables
			staleness = self.version - getattr(game, "labels_version", self.version)
			row["label_staleness"] = max(row["label_staleness"], staleness)
			future = getattr(self, "label_futures", {}).pop(id(game), None)
			pre = precomputed(future, game) if staleness <= MAX_LABEL_STALENESS and future is not None else None
			if pre is None:
				t, mixed, price = label(game, self.labels, self.empire)
				if self.config.share_target == "change":
					t = share_change_labels(game, self.empire)
				z = growth_labels(game, self.empire)
				real = end_labels(game, self.empire)
				steps = end_steps(game)
			else:
				t, mixed, price, z, real, steps = pre["t"], pre["mixed"], pre["price"], pre["z"], pre["real"], (pre["step"], pre["week"])
			# (the change heads are graded on real outcomes only: the ladder mixes a forecast in as a share)
			ladder_trust = trust if self.config.share_target == "share" else {k: v for k, v in trust.items() if not k.startswith("V")}
			graded, grown = ladder_targets(game, t, z, starts[g], ladder_trust, self.config.outcome_share_min)
			forecasts.append(start_forecasts(starts[g], t, z, real))
			fresh = np.array([d.version >= self.version - self.config.max_record_age for d in game.decisions])
			row["dropped_old"] += int((~fresh).sum())
			# The end credit's steps, as rows of this update (a decision after a fresh one is fresh: versions only grow)
			rows = np.where(fresh, len(decisions) + np.cumsum(fresh) - 1, -1)
			for name, index in zip(("step", "week"), steps):
				end[name].append(np.where(index >= 0, rows[np.maximum(index, 0)], -1)[fresh])
			end["targets"].append(end_targets(game, real, starts[g], trust.get("Vend", 0.0), self.config.outcome_share_min)[fresh])
			end["real"].append(real[fresh])
			end["price"].append(price[fresh])
			decisions += [d for d, keep in zip(game.decisions, fresh) if keep]
			if foresight is not None:
				rows = pre["foresight"] if pre is not None and "foresight" in pre else labels_of(game, self.features_cpu)
				foresight.append({name: value[fresh] for name, value in rows.items()})
			targets.append(graded[fresh])
			labels.append(t[fresh])
			growth.append(grown[fresh])
			growth_real.append(z[fresh])  # (the real log values, for loss_noise's growth credit)
			returns.append((mixed - price)[fresh])
			if self.config.credit == "attributed":  # (our share as the day starts, what we took: ppo_update makes the credit)
				try:
					now, took = attributed_labels(game, self.empire)
				except Exception as error:  # (a game whose labels fail is no credit, never the trainer's end: a FAILURE to look into)
					print(f"campaign: FAILURE: attributed labels of a game failed ({type(error).__name__}: {str(error)[:200]}): no credit from it", flush=True)
					now, took = np.full(len(game.decisions), np.nan, np.float32), np.full(len(game.decisions), np.nan, np.float32)
				attributed["share_now"].append(now[fresh])
				attributed["taken"].append(took[fresh])
			if self.config.credit == "schedule":  # (ppo_update makes the credit, from these labels and its prepass)
				advantages.append(np.zeros(int(fresh.sum()), dtype=np.float32))
			elif self.config.credit in ("horizon", "attributed"):  # (its errors feed the credit check, kept for the record)
				a, error = horizon_advantages(game, t, price, self.trust(), self.labels, self.config.outcome_share_min, self.config.myopic_weight, self.credit_check, parts=True)
				advantages.append(a[fresh])
				errors.append(error[fresh])
				players.append(np.array([g * len(COLORS) + COLORS.index(d.color) for d in game.decisions])[fresh])
			else:  # "mix": the label mix minus the heads' mix
				advantages.append((mixed - price - np.array([np.dot(d.value, self.labels.weights) for d in game.decisions], dtype=np.float32))[fresh])
			row["label_sum"] += float(np.nansum(mixed[fresh]))
			row["price_sum"] += float(price[fresh].sum())
		if not decisions:
			return False
		memory.append(("labels", rss_gb()))
		self._note_head_ev(forecasts)  # (for the next update's grades: these were graded by the trust before them)
		targets, labels, growth = np.concatenate(targets), np.concatenate(labels), np.concatenate(growth)
		returns, advantages = np.concatenate(returns), np.concatenate(advantages)
		end = {name: np.concatenate(parts) for name, parts in end.items()}
		if foresight is not None:
			foresight = {name: np.concatenate([rows[name] for rows in foresight]) for name in foresight[0]}
		row["trust"] = self.trust()
		end["outcome"], end["weight"] = outcome, weight if self.config.credit == "horizon" else (self.config.attributed_end_weight if weight > 0 else 0.0) if self.config.credit in ("attributed", "schedule") else 0.0
		attributed = {name: np.concatenate(parts) for name, parts in attributed.items()} if self.config.credit == "attributed" else None
		row["end_credit"] = end["weight"]
		if errors:  # the credit check: the step's estimate against what then happened, by action type (labels.credit_evidence)
			self.credit_evidence = credit_evidence(decisions, np.concatenate(errors), np.concatenate(players), self.credit_evidence)
			gaps = credit_gaps(self.credit_evidence)
			self.credit_check = credit_trust(gaps)
			row["credit_trust"] = self.credit_check
			row["credit_gaps"] = gaps
		if self.worker:  # (the process saves checkpoint_latest.pt, with its optimizer)
			state, metrics, lr_scale = self.worker.update(decisions, targets, returns, self.labels.weights, {"version": self.version + 1, "games_used": self.games_used + len(games), "value_ev": self.value_ev, "credit_evidence": self.credit_evidence, "head_ev": self.head_ev, "value_pool": self.value_pool, "head_pool": self.head_pool}, advantages, growth, labels, end, foresight)
			with self.model_lock:
				self.model.load_state_dict(state)
				self.lr_scale = lr_scale
				self.version += 1
				self.server.set_model("current", self.version, self.model)
				self.games_used += len(games)
				self._refresh_features()
		else:
			out_of_memory = None
			with self.model_lock:
				try:
					metrics, self.lr_scale = ppo_update(self.model, self.optimizer, decisions, targets, returns, self.config, device=self.device, lr_scale=self.lr_scale, weights=self.labels.weights, advantages=advantages, growth=growth, labels=labels, end=end, foresight=foresight, growth_real=np.concatenate(growth_real), attributed=attributed, teacher=self.teacher, teacher_weight=self.teacher_weight, measure_noise=bool(self.config.noise_every) and self.version % self.config.noise_every == 0)
				except torch.OutOfMemoryError as error:  # (handled below, once the error's frames, and their tensors, are gone)
					out_of_memory = str(error)[:200]
			if out_of_memory is not None:
				# The run goes on (2026-11-01: two of joint7's updates ran out of the GPU's 27 GB and ended it, and their
				# games with it): this update's games are dropped, as games with no decisions are; the weights keep the
				# steps it took, served with the next update
				torch.cuda.empty_cache()
				print(f"campaign update {self.version + 1}: out of GPU memory, its {len(games)} games dropped: {out_of_memory}", flush=True)
				return False
			with self.model_lock:
				if self.device.type == "mps":
					torch.mps.empty_cache()  # (the battle learner's lesson: MPS keeps blocks of every size it saw)
				self.version += 1
				self.server.set_model("current", self.version, self.model)
				self.games_used += len(games)
				self._refresh_features()
				self._note_value_ev(metrics)
				self.teacher_weight = adapted_teacher_weight(self.teacher_weight, metrics.get("teacher_kl"), self.config.teacher_kl_target)
				self._save(self.out / "checkpoint_latest.pt")
		if self.worker:  # (the process saved its checkpoint with the trust before this update: one update behind)
			self._note_value_ev(metrics)
		end_errors = metrics.pop("end_step_error", None)
		if end_errors is not None and errors:  # the end credit's check, the 7-day one's for V_end's steps (logged only)
			self.end_evidence = credit_evidence(decisions, end_errors, np.concatenate(players), getattr(self, "end_evidence", None))
			row["end_credit_gaps"] = credit_gaps(self.end_evidence)
			self.end_check = credit_trust(row["end_credit_gaps"])  # (for the next update's end credit: end_credit)
			row["end_check"] = self.end_check
		self._relabel_waiting()
		memory.append(("updated", rss_gb()))
		self._log(epoch, games, decisions, row, metrics, time.time() - started)
		memory.append(("logged", rss_gb()))
		if all(value == value for _, value in memory):
			print("campaign memory (GB, this process): " + "  ".join(f"{stage} {value:.1f}" for stage, value in memory), flush=True)
		if self.device.type == "cuda":  # (the allocator's live tensors and its reserve, and what the card has left)
			free, total = torch.cuda.mem_get_info(self.device)
			print(f"campaign gpu memory (GB): allocated {torch.cuda.memory_allocated(self.device) / 2**30:.1f}  reserved {torch.cuda.memory_reserved(self.device) / 2**30:.1f}  card free {free / 2**30:.1f} of {total / 2**30:.1f}", flush=True)
		return True

	def _relabel_waiting(self):
		"""The games still waiting whose labels came from tables MAX_LABEL_STALENESS versions back or more: labelled again
		now with the new tables, in the labeler (ready by their update; computed in the trainer's thread they were ~27 s
		of GPU idle an update, 2026-10-31: with ahead=3 and back-to-back updates a game waits ~2 versions)."""
		if not getattr(self, "labeler", None):
			return
		with self.lock:
			waiting = [game for game, _ in self.finished if self.version - getattr(game, "labels_version", self.version) >= MAX_LABEL_STALENESS]
		for game in waiting:
			game.labels_version = self.version
			self.label_futures[id(game)] = self.labeler.submit(game, self.features_cpu)

	def _refresh_features(self):
		"""The CPU features' weights as the model's are now (call with model_lock held)."""
		self.features_cpu.load_state_dict({key: value.detach().cpu() for key, value in self.model.features.state_dict().items()})

	def _note_value_ev(self, metrics):
		"""The heads the step's estimate uses (V5, V7): their explained variance, its parts pooled over ~10 updates
		(labels.pooled_ev)."""
		pools, evs = [], []
		for i, k in enumerate(c.VALUE_OFFSETS[1:]):
			pool, ev = pooled_ev(self.value_pool[i] if self.value_pool else None, metrics[f"value_parts_d{k}"])
			pools.append(pool)
			evs.append(ev)
		if any(pool is None for pool in pools):
			return  # (no real shares measured yet)
		self.value_pool, self.value_ev = pools, evs

	def trust(self):
		"""How far the credit trusts the value heads, 0..1: their smoothed explained variance over the share prior's (the
		worse of V5 and V7: ppo_update's value_ev) over trust_full_ev. The credit check (credit_check) applies apart, in labels.horizon_advantages: a fit this good
		on the whole can still price a quick gain that costs by day 7, which shows as the step's error by action type."""
		if self.value_ev is None:
			return 0.0
		return float(min(1.0, max(0.0, min(self.value_ev) / self.config.trust_full_ev)))

	def _note_head_ev(self, forecasts):
		"""Each head's explained variance at the day starts (start_forecasts), its parts pooled over ~10 updates
		(labels.pooled_ev). The share heads'
		(V2, V5, V7, V_end: "Vend") over the share prior's, 1 / the players still in: the player count alone explains
		much of a share (1/2 in a 2-player game, 1/4 in a 4-player one), and the heads start there, so their plain
		explained variance trusted them before they had learned anything (the growth heads start at a constant: 0)."""
		shares, real, grown, real_growth, end, real_end, prior = (np.concatenate(part) for part in zip(*forecasts))
		start = prior if self.config.share_target == "share" else np.full_like(prior, 0.5)  # (the change heads' start: no change)
		pairs = [(f"V{k}", shares[:, i] - start, real[:, i] - start) for i, k in enumerate(c.VALUE_OFFSETS)] + [(f"G{k}", grown[:, i], real_growth[:, i]) for i, k in enumerate(c.VALUE_OFFSETS)]
		for name, predicted, actual in pairs + [("Vend", end - prior, real_end - prior)]:
			pool, ev = pooled_ev(self.head_pool.get(name), residual_variances(predicted, actual))
			if pool is not None:
				self.head_pool[name], self.head_ev[name] = pool, ev

	def grade_trust(self):
		"""{head: 0..1} how far each head is trusted to grade the next longer one (labels.ladder_targets), V_end itself
		(labels.end_targets): its smoothed explained variance at the day starts over trust_full_ev; none measured yet, 0
		(the real outcome alone)."""
		return {name: float(min(1.0, max(0.0, ev / self.config.trust_full_ev))) for name, ev in self.head_ev.items()}

	def end_credit(self):
		"""(the end credit's share of the policy's advantage, the week's share in the end credit): end_credit_max x the
		trust in V_end, and 1 - (1 - outcome_share_min) x that trust (labels.end_advantages)."""
		trust = self.grade_trust().get("Vend", 0.0)
		step = min(trust, getattr(self, "end_check", 1.0)) if self.config.end_credit_check else trust  # (its check moves the credit from V_end's steps to its weeks)
		return self.config.end_credit_max * trust, 1 - (1 - self.config.outcome_share_min) * step

	def _log(self, epoch, games, decisions, sums, metrics, seconds):
		n = len(decisions)
		turn = [d for d in decisions if d.action_type >= 0]
		types = Counter(c.ActionType(d.action_type).name.lower() for d in turn)
		phi, heroes = [], []
		for game, _ in games:
			if game.ledgers:
				last = game.ledgers[max(game.ledgers)]
				phi += [empire_value(p, self.empire)["total"] for p in last["players"]]
			heroes += [len(p["heroes"]) for snapshot in game.ledgers.values() for p in snapshot["players"] if p["status"] == 0]
		done = [summary for game, _ in games for summary in game.done.values()]
		mean = lambda values: float(np.mean(values)) if len(values) else float("nan")  # noqa: E731
		row = {
			"version": self.version,
			"epoch": epoch,
			"games": len(games),
			"decisions": n,
			"dropped_old": sums["dropped_old"],
			"trust": sums["trust"],
			"credit_trust": sums.get("credit_trust", float("nan")),
			"end_credit": sums["end_credit"],
			"unsound": metrics["unsound"],
			"credited": metrics.get("credited", n),
			"days": mean([max(game.ledgers) - 1 for game, _ in games if game.ledgers]),
			"players": mean([len(game.ledgers[min(game.ledgers)]["players"]) for game, _ in games if game.ledgers]),
			"phi_end": mean(phi),
			"heroes": mean(heroes),  # the one number for heroes (the user): mean heroes per player still in, over days
			"label_mean": sums["label_sum"] / max(n, 1),
			"price_mean": sums["price_sum"] / max(n, 1),
			"policy_loss": metrics.get("policy", float("nan")),
			"value_loss": metrics.get("value", float("nan")),
			"value_loss_fresh": metrics.get("value_fresh", float("nan")),  # train.ppo_update: before the update, above the targets' entropy
			"entropy": metrics.get("entropy", float("nan")),
			"below_floor": metrics.get("below_floor", float("nan")),
			"kl": metrics.get("kl", float("nan")),
			"teacher_kl": metrics.get("teacher_kl", float("nan")),  # (KL(teacher || model) a decision; 0 without a teacher)
			"teacher_weight": self.teacher_weight,
			# Nullkiller2's advice (config.nk2_advice): the share of turn and exchange decisions it advised, how often our
			# choice was its, and the cross-entropy toward it
			"advised": sum(d.advice is not None for d in decisions) / max(1, sum(d.kind in (c.DecisionKind.TURN, c.DecisionKind.EXCHANGE) for d in decisions)),
			"teacher_agree": float(np.mean([agrees(d) for d in decisions if d.advice is not None])) if any(d.advice is not None for d in decisions) else float("nan"),
			"teacher_ce": metrics.get("teacher_ce", float("nan")),
			"last_kl": metrics["last_kl"],
			"clip_fraction": metrics.get("clip_fraction", float("nan")),
			"grad_norm": metrics.get("grad_norm", float("nan")),
			"steps": metrics["steps"],
			"learning_rate": metrics["learning_rate"],
			**{f"value_ev_d{k}": metrics[f"value_ev_d{k}"] for k in c.VALUE_OFFSETS},
			**{f"head_ev_{kind}{k}": self.head_ev.get(f"{kind.upper()}{k}", float("nan")) for kind in "vg" for k in c.VALUE_OFFSETS},
			"head_ev_vend": self.head_ev.get("Vend", float("nan")),
			"no_effect": sum(1 for d in decisions if d.flags & NO_EFFECT) / max(n, 1),
			"ai_errors": sum(s.get("errors", 0) for s in done),
			"failed_actions": sum(s.get("failed_actions", 0) for s in done),
			"game_failures": self.stats["game_failures"],
			"game_maps": sum(1 for _, info in games if not info.map_name.startswith("Maps/alpha_random/")),
			"dropped_games": self.stats["dropped_games"],
			"game_seconds": mean([info.seconds for _, info in games]),
			"record_mb": sum(game.record_bytes for game, _ in games) / 2**20,
			"seconds": seconds,
			"gpu_mb": metrics.get("gpu_mb", float("nan")),  # the update's peak on the GPU (train.gpu_held)
			"label_staleness": sums.get("label_staleness", 0),  # the oldest precomputed labels used, in model versions (labeler.py)
			**{f"share_{t}": types[t] / max(len(turn), 1) for t in TYPES},
			**{f"failure_{k}": v for k, v in failure_modes(decisions).items()},  # dumb decisions (failures.py)
			"kinds": " ".join(f"{k}:{n}" for k, n in sorted(Counter(c.DecisionKind(d.kind).name for d in decisions).items())),
			"no_effect_types": " ".join(f"{t}:{n}" for t, n in sorted(Counter(c.ActionType(d.action_type).name for d in decisions if d.flags & NO_EFFECT).items())),
			**{name: metrics.get(name, float("nan")) for name in (*FORESIGHT_COLUMNS, "grad_shared_main", "grad_shared_foresight", "grad_shared_cosine", "grad_dead_share")},
		}
		with self.progress.open("a", newline="") as f:
			csv.writer(f).writerow([round(row[k], 6) if isinstance(row[k], float) else row[k] for k in COLUMNS])
		if row["value_loss_fresh"] == row["value_loss_fresh"]:  # the eval tab's (battle_model/train.py VALUE_LOSS_TAB)
			self.writer.add_scalar("eval/campaign_value_loss", row["value_loss_fresh"], self.version)
		for name in ("value_corr", "value_spread", "label_spread", "gpu_anchor_mb", "update_seconds", "prepass_seconds", "wait_seconds", "sync_seconds", "cpu_share", "stopped_pass", "signal_ratio", "attributed_taken_share", "attributed_taken_mean", "attributed_own_std", "attributed_taken_std", "forward_seconds", "losses_seconds", "backward_seconds", "optimizer_seconds"):
			if metrics.get(name, float("nan")) == metrics.get(name, float("nan")):
				self.writer.add_scalar(f"campaign/{name}", metrics[name], self.version)
		for kind, gap in sums.get("credit_gaps", {}).items():  # (the credit check's t-statistic per action type)
			self.writer.add_scalar(f"campaign_credit/{kind.lower()}", gap, self.version)
		for kind, gap in sums.get("end_credit_gaps", {}).items():
			self.writer.add_scalar(f"campaign_end_credit/{kind.lower()}", gap, self.version)
		noise = metrics.get("noise") or {}
		for name, scale in noise.items():  # (train.loss_noise: each loss term's gradient noise scale, in decisions)
			self.writer.add_scalar(f"campaign_noise/{name.replace(' ', '_')}", min(scale, 1e9), self.version)
		if noise:
			print("campaign noise scale (decisions, the batch where a term's gradient noise equals its signal): " + "  ".join(f"{name} {scale:.0f}" for name, scale in noise.items()), flush=True)
		if "end_check" in sums:  # (how far the end credit's step part is used: 1 all of it, 0 its week part alone)
			self.writer.add_scalar("campaign/end_check", sums["end_check"], self.version)
		for skill, (picked, offered) in skill_picks(decisions).items():  # (how often each skill is taken when offered)
			self.writer.add_scalar(f"campaign_skills/{skill}", picked / offered, self.version)
		turns = max(1, sum(1 for d in decisions if d.action_type >= 0))
		for what, (count, rise) in cast_report(game for game, _ in games).items():  # (each spell's share of the turn decisions, V7's rise after it)
			self.writer.add_scalar(f"campaign_casts/{what}", count / turns, self.version)
			if rise == rise:
				self.writer.add_scalar(f"campaign_casts/{what}_v7_rise", rise, self.version)
		for key in COLUMNS[2:]:
			value = row[key]
			if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value:
				group = "campaign_actions/" + key[6:] if key.startswith("share_") else "campaign_failures/" + key[8:] if key.startswith("failure_") else "campaign/" + key
				self.writer.add_scalar(group, value, self.version)
		self.writer.flush()
		print(
			f"campaign {self.version:4d}  games {row['games']}  decisions {n}  days {row['days']:.1f}  Φ at the end {row['phi_end']:.0f}  heroes {row['heroes']:.2f}"
			# (pooled over ~10 updates like the growth and end heads': this update's alone swings with its few games' spread,
			# -0.65 at a correlation of +0.22 and a spread as wide as the outcomes', 2026-10-30, while the pooled rose)
			"  value ev " + "/".join(f"{row[f'head_ev_v{k}']:+.2f}" for k in c.VALUE_OFFSETS)
			+ f" (this update {row[f'value_ev_d{c.VALUE_OFFSETS[-1]}']:+.2f}: corr {metrics.get('value_corr', float('nan')):+.2f}, spread {metrics.get('value_spread', float('nan')):.3f} vs {metrics.get('label_spread', float('nan')):.3f})"
			+ "  growth ev " + "/".join(f"{row[f'head_ev_g{k}']:+.2f}" for k in c.VALUE_OFFSETS) + f"  end ev {row['head_ev_vend']:+.2f} (credit {row['end_credit']:.2f})"
			+ f"  kl {row['last_kl']:.4f} (target {metrics.get('kl_target', float('nan')):.4f})" + (f"  teacher kl {row['teacher_kl']:.3f} (weight {row['teacher_weight']:.2f})" if self.teacher is not None else "") + (f"  advice {row['advised']:.0%} of turns and exchanges, ours its {row['teacher_agree']:.0%}, teacher ce {row['teacher_ce']:.3f} (weight {row['teacher_weight']:.2f})" if self.config.nk2_advice > 0 else "") + f"  signal {metrics.get('signal_ratio', float('nan')):.3f}  steps {row['steps']}" + (f" (stopped in pass {metrics['stopped_pass']})" if metrics.get("stopped_pass") else "") + f"  {seconds:.1f}s (update {metrics.get('update_seconds', float('nan')):.0f}s: prepass {metrics.get('prepass_seconds', float('nan')):.0f}, decode waits {metrics.get('wait_seconds', float('nan')):.0f}, gpu syncs {metrics.get('sync_seconds', float('nan')):.0f}, cpu {metrics.get('cpu_share', float('nan')):.0%}; launching forward {metrics.get('forward_seconds', float('nan')):.0f}, losses {metrics.get('losses_seconds', float('nan')):.0f}, backward {metrics.get('backward_seconds', float('nan')):.0f}, optimizer {metrics.get('optimizer_seconds', float('nan')):.0f})  gpu {row['gpu_mb']:.0f} MB (anchor {metrics.get('gpu_anchor_mb', float('nan')):.0f})  labels {row['label_staleness']} old",
			flush=True,
		)

	def save(self, path):
		with self.model_lock:
			self._save(path)

	def _save(self, path):
		state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
		optimizer = {} if self.worker else {"optimizer": self.optimizer.state_dict()}  # (the process's is in checkpoint_latest.pt)
		torch.save({"model": state, **optimizer, "version": self.version, "lr_scale": self.lr_scale, "teacher_weight": self.teacher_weight, "games_used": self.games_used, "value_ev": self.value_ev, "credit_evidence": self.credit_evidence, "end_evidence": getattr(self, "end_evidence", {}), "head_ev": self.head_ev, "value_pool": self.value_pool, "head_pool": self.head_pool, "model_kwargs": self.kwargs}, path)

	# ---- evaluation ----

	def eval_due(self):
		if not self.config.eval_minutes or not self.config.eval_games or self.evaluating():
			return False
		return self.last_eval is None or time.monotonic() - self.last_eval >= self.config.eval_minutes * 60

	def evaluating(self):
		return self.evaluation is not None and self.evaluation.is_alive()

	def evaluate(self, epoch):
		"""Games against Nullkiller2 (evaluate.py) with the models as they are now, pinned as "eval" on both servers so
		the updates that land meanwhile do not change them under the evaluation (2026-10-30). Training goes on."""
		with self.model_lock:
			version = self.version
			self.server.set_model("eval", version, self.model)
			self.battle.pin_eval()
		try:
			row = evaluate.evaluate(self.config, self.server, self.battle, version=version, epoch=epoch, stop=self.stop, model="eval")
		finally:
			self.last_eval = time.monotonic()
		evaluate.log(row, self.out / "eval.csv", self.writer)
		if self.config.teacher_mix == "score" and (self.teacher is not None or self.config.nk2_advice > 0) and np.isfinite(row.get("score", np.nan)):
			self.teacher_weight = teacher_weight_for(row["score"])  # (config.teacher_mix: the teacher's share by our strength)
			print(f"campaign: the teacher's weight {self.teacher_weight:.2f} (its share of the loss {self.teacher_weight / (1 + self.teacher_weight):.2f}) for an eval score of {row['score']:.3f}", flush=True)
		self.save(self.out / f"checkpoint_{version:05d}.pt")
		return row

	def start_eval(self, epoch):
		"""evaluate() in a thread of its own: training never waits for it (its error is raised by check())."""
		def run():
			try:
				self.evaluate(epoch)
			except BaseException as error:  # noqa: BLE001 (raised in the training thread)
				self.eval_error = error

		self.evaluation = threading.Thread(target=run, daemon=True, name="campaign-eval")
		self.evaluation.start()

	def check(self):
		"""Raises the background evaluation's error, if one failed."""
		if self.eval_error is not None:
			error, self.eval_error = self.eval_error, None
			raise error


def precomputed(future, game=None):
	"""The labeler's result (its Future, learner._game_done started it), or None: it failed (noted), or it does not match
	the game's usable decisions (game: checked when given)."""
	try:
		pre = future.result(timeout=300)
	except Exception as error:  # noqa: BLE001 (computed in the trainer's thread instead)
		print(f"campaign: a game's labels failed in the labeler ({str(error)[:200]}): computed here", flush=True)
		return None
	if pre is None or (game is not None and pre["n"] != len(game.decisions)):
		return None
	return pre


def rss_gb():
	"""This process's resident memory (GB) on Linux, else nan: the update's stages are logged with it (2026-10-30: the
	PC's main process grew 7 GB in 20 s around an update, and the memory guard stopped the run)."""
	try:
		with open("/proc/self/status") as status:
			return next(int(line.split()[1]) for line in status if line.startswith("VmRSS:")) / 2**20
	except (OSError, StopIteration):
		return float("nan")


SPELL_NAMES = ("summon_boat", "scuttle_boat", "visions", "view_earth", "disguise", "view_air", "fly", "water_walk", "dimension_door", "town_portal")


def cast_report(games):
	"""{spell or "move": (decisions, the mean rise of V7 from it to the player's next turn decision that day)} over the
	games' casts and moves (the recorded heads, as they played). Casting rose from 2% to 10% of joint5's turn decisions
	while the credit check priced casts above what followed them (campaign_credit/cast ~+4, 2026-10-31): a spell that
	only shows the map (View Air, View Earth, Visions) and lifts the heads would be credited for nothing."""
	rises = {}
	for game in games:
		ds, latest = game.decisions, {}
		following = [-1] * len(ds)
		for i in reversed(range(len(ds))):
			following[i] = latest.get((ds[i].color, ds[i].day), -1)
			if ds[i].kind == c.DecisionKind.TURN:
				latest[(ds[i].color, ds[i].day)] = i
		for i, d in enumerate(ds):
			if d.kind != c.DecisionKind.TURN or d.action_type not in (c.ActionType.CAST, c.ActionType.MOVE):
				continue
			what = SPELL_NAMES[d.index % c.N_ADVENTURE_SPELLS] if d.action_type == c.ActionType.CAST else "move"
			j = following[i]
			rises.setdefault(what, []).append(ds[j].value[-1] - d.value[-1] if j >= 0 else float("nan"))
	return {what: (len(v), float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")) for what, v in rises.items()}


def skill_picks(decisions):
	"""{skill: (times taken, times offered)} over the level-ups among the decisions: the user's worry (2026-10-28), that a
	7-day horizon takes Estates too often and Offence, Earth Magic or Wisdom (high-level spells weeks on) too rarely,
	which the end head is there to change."""
	ups = [d for d in decisions if d.kind == c.DecisionKind.LEVEL_UP and 0 <= d.index < c.N_OPTIONS]
	if not ups:
		return {}
	x, counts = decode_fields(ups, ("o_kind", "o_type")), {}  # (two fields, not every decision's 1.7 MB decoded)
	for b, d in enumerate(ups):
		skill = x["o_kind"][b] == c.OptionKind.SKILL
		for s in x["o_type"][b][skill]:
			if 0 <= s < len(SKILLS):
				counts.setdefault(SKILLS[s], [0, 0])[1] += 1
		if skill[d.index] and 0 <= x["o_type"][b, d.index] < len(SKILLS):
			counts[SKILLS[x["o_type"][b, d.index]]][0] += 1
	return {name: tuple(v) for name, v in counts.items()}


def day_starts(model, games, lock, batch=64):
	"""Per game, {(color, day): (decision index, shares [N_VALUE_OFFSETS], log growth [N_VALUE_OFFSETS], log of our value,
	V_end, the share prior 1 / players)}: the model as it is now at each player's first turn decision of each day, one
	forward pass a player-day. Its forecasts there grade the longer heads (labels.ladder_targets) and V_end
	(labels.end_targets)."""
	rows = []
	for g, game in enumerate(games):
		seen = set()
		for i, d in enumerate(game.decisions):
			if d.kind == c.DecisionKind.TURN and (d.color, d.day) not in seen:
				seen.add((d.color, d.day))
				rows.append((g, i))
	starts, device = [{} for _ in games], next(model.parameters()).device
	chunks = [rows[k : k + batch] for k in range(0, len(rows), batch)]
	# (decoded in threads ahead of the forward, as the update's batches are: ~2,700 day starts an update on the PC, 2.3 GB
	# of zlib in one thread was several seconds of the trainer's thread, 2026-10-30)
	for chunk, inputs in _prefetched(chunks, lambda rows_: decode_inputs([games[g].decisions[i] for g, i in rows_]), device):
		with lock, torch.no_grad():
			out = model(inputs)
		shares = out["value"].float().cpu().numpy()
		grown = ((2 * torch.sigmoid(out["growth_logit"]) - 1) * GROWTH_CAP).float().cpu().numpy()
		log_value = np.log(np.maximum(out["phi"].float().cpu().numpy(), PHI_FLOOR))
		end = torch.sigmoid(out["end_logit"]).float().cpu().numpy()
		prior = 1.0 / np.maximum(inputs["g_players"].float().cpu().numpy(), 2.0)
		for j, (g, i) in enumerate(chunk):
			d = games[g].decisions[i]
			starts[g][(d.color, d.day)] = (i, shares[j], grown[j], float(log_value[j]), float(end[j]), float(prior[j]))
	return starts


def start_forecasts(starts, real, growth, real_end):
	"""The forecasts at a game's day starts (day_starts) and what then happened: shares [K, N_VALUE_OFFSETS], the real
	shares (label's), log growth, the real log growth (from growth_labels', within the heads' range), V_end [K], the
	real end (end_labels') and the share prior."""
	if not starts:
		empty, none = np.zeros((0, c.N_VALUE_OFFSETS), dtype=np.float32), np.zeros(0, dtype=np.float32)
		return empty, empty, empty, empty, none, none, none
	index, shares, grown, log_value, end, prior = zip(*starts.values())
	index = np.array(index)
	clipped = np.clip(growth[index] - np.array(log_value)[:, None], -GROWTH_CAP, GROWTH_CAP)
	return np.stack(shares), real[index], np.stack(grown), clipped, np.array(end, dtype=np.float32), real_end[index], np.array(prior, dtype=np.float32)


def agrees(decision):
	"""Our choice is Nullkiller2's advice (Decision.advice): the same type, and the same entry where the type has a table
	(an end is only its type: ours has no index, -1, the advice's 0)."""
	action, index = decision.advice[:2]
	return decision.action_type == action and (action <= c.ActionType.END or decision.index == index)


def teacher_weight_for(score):
	"""The teacher KL's weight for an evaluation score against Nullkiller2 (config.teacher_mix "score"; 1 = an average
	player of its game): its share of the loss lambda = clip(1.2 - score, 0.05, 0.95), the weight lambda / (1 - lambda)
	against PPO's 1: 19 at a score of 0.25 or under, 2.3 at 0.5, 0.25 at 1 (as good as the teacher), 0.05 from 1.15 up."""
	share = float(np.clip(1.2 - score, 0.05, 0.95))
	return share / (1.0 - share)


def adapted_teacher_weight(weight, kl, target):
	"""The teacher KL's weight after an update whose KL from the teacher was kl (config.teacher_kl_target): x1.5 over
	1.5 x target, /1.5 under target / 1.5, within 0.01..10; unchanged with no target or no KL measured."""
	if not target or kl is None or not np.isfinite(kl):
		return weight
	if kl > 1.5 * target:
		weight *= 1.5
	elif kl < target / 1.5:
		weight /= 1.5
	return float(min(10.0, max(0.01, weight)))


def draw_days(rng, config):
	"""A game's day cap (config.days_min, days_tail, days_max): days_min plus a geometric number of days, mean days_tail:
	past days_min each day ends the game with the same chance, 1 / (days_tail + 1), whatever came before."""
	p = 1.0 / (config.days_tail + 1.0)
	extra = int(math.log(1.0 - rng.random()) / math.log(1.0 - p))
	return min(config.days_min + extra, config.days_max)


def auto_device():
	import torch

	return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# MPS's allocator frees its cached blocks once it holds this fraction of the GPU's recommended working set (~16 GB
# here: ~5 GB); its default (1.4) let the update hold ~8 GB next to the battle learner's (train.gpu_held)
MPS_LOW_WATERMARK = "0.3"


def _update_worker(config, kwargs, device_name, state, requests, results, latest):
	"""The update process's loop (UpdateProcess): one PPO update per request, the new weights back, checkpoint_latest.pt
	written with its optimizer."""
	os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", MPS_LOW_WATERMARK)  # (read when MPS is first used: below)
	if config.torch_threads > 0:
		torch.set_num_threads(config.torch_threads)
	device = torch.device(device_name)
	model = build_model(kwargs).to(device).eval()
	optimizer = make_optimizer(model, config)
	migrate_subtypes(state, model)
	model.load_state_dict(state["model"])
	if state.get("optimizer"):
		optimizer.load_state_dict(state["optimizer"])
	use_fused(optimizer)
	lr_scale = state["lr_scale"]
	while (request := requests.get()) is not None:
		decisions, targets, returns, weights, about, advantages, growth, labels, end, foresight = request
		try:
			metrics, lr_scale = ppo_update(model, optimizer, decisions, targets, returns, config, device=device, lr_scale=lr_scale, weights=weights, advantages=advantages, growth=growth, labels=labels, end=end, foresight=foresight)
			if device.type == "mps":
				torch.mps.empty_cache()  # (the battle learner's lesson: MPS keeps blocks of every size it saw)
			cpu = {key: value.detach().cpu() for key, value in model.state_dict().items()}
			torch.save({"model": cpu, "optimizer": optimizer.state_dict(), "lr_scale": lr_scale, "model_kwargs": kwargs, **about}, latest)
			results.put(("ok", cpu, metrics, lr_scale))
		except Exception:
			results.put(("error", traceback.format_exc(), None, lr_scale))


class UpdateProcess:
	"""The campaign updates in a process of their own (learner "process"), on config.device ("auto": CUDA, else Apple's
	GPU, else the CPU): the battle learner keeps its own process's GPU, and the campaign's games go on meanwhile. Only
	the update's decisions go in (compressed inputs: ~3 KB each) and the new weights come out; the learner here keeps a
	CPU copy for the inference server and the evaluations."""

	def __init__(self, config, kwargs, model, optimizer, lr_scale, latest):
		context = multiprocessing.get_context("spawn")
		self.requests, self.results = context.Queue(), context.Queue()
		device = auto_device() if config.device == "auto" else config.device
		state = {"model": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "optimizer": optimizer.state_dict(), "lr_scale": lr_scale}
		self.process = context.Process(target=_update_worker, args=(config, kwargs, device, state, self.requests, self.results, str(latest)), daemon=True)
		self.process.start()
		print(f"campaign: updates in process {self.process.pid} on {device}", flush=True)

	def update(self, decisions, targets, returns, weights, about, advantages=None, growth=None, labels=None, end=None, foresight=None):
		"""(the new weights on the CPU, the update's metrics, the KL control's lr_scale); raises if the update failed."""
		self.requests.put((decisions, targets, returns, weights, about, advantages, growth, labels, end, foresight))
		while True:
			try:
				status, state, metrics, lr_scale = self.results.get(timeout=60)
				break
			except Exception:  # (queue.Empty) still updating, unless the process is gone
				if not self.process.is_alive():
					raise RuntimeError(f"the campaign update process died (exit code {self.process.exitcode})")
		if status != "ok":
			raise RuntimeError(f"a campaign update failed in its process:\n{state}")
		return state, metrics, lr_scale

	def close(self):
		self.requests.put(None)
		self.process.join(timeout=60)
		if self.process.is_alive():
			self.process.kill()
