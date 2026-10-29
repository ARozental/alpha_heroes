"""The campaign model's learner in a joint run (the user, 2026-10-27: "training the campaign comes alongside training
the battle model (that has its own optimizer etc...)"): battle_model/train.py runs it when campaign.games > 0.

	games        config.games self-play games at once, each on a new random map (2-4 players, water and the
	             underground on, a day cap drawn per game: draw_days): every player our campaign model (this learner's inference server, family
	             "campaign"), every player's battle our battle model (the battle learner's server; the wandering monsters BattleAI's).
	             The battles' records go to the battle learner as real games' do (actors.py: its real_share, with the
	             generated battles, hard cases among them, as the rest, so it does not forget them); the campaign
	             decisions and the daily ledgers come here
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
import csv
import json
import math
import multiprocessing
import os
import random
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
from battle_model.inference import InferenceServer

from . import constants as c
from . import evaluate
from .empire import EmpireConfig, empire_value
from .failures import NAMES as FAILURES, failure_modes
from .features import SKILLS
from .labels import (
	COLORS, GROWTH_CAP, PHI_FLOOR, LabelConfig, credit_evidence, credit_gaps, credit_trust, end_labels, end_steps, end_targets, growth_labels, horizon_advantages,
	label, ladder_targets, pooled_ev, residual_variances,
)
from .maps import GAME_MAPS
from .model import CampaignModel
from .play import make_map, play
from .records import NO_EFFECT, decode_inputs
from .train import make_optimizer, model_kwargs, ppo_update, usable

TYPES = [a.name.lower() for a in c.ActionType]
COLUMNS = [
	"version", "epoch", "games", "decisions", "dropped_old", "unsound", "credited", "days", "players", "phi_end", "heroes", "label_mean", "price_mean",
	"trust", "credit_trust", "end_credit", "policy_loss", "value_loss", "entropy", "below_floor", "kl", "last_kl", "clip_fraction", "grad_norm", "steps", "learning_rate",
	*(f"value_ev_d{k}" for k in c.VALUE_OFFSETS), *(f"head_ev_{kind}{k}" for kind in "vg" for k in c.VALUE_OFFSETS), "head_ev_vend", "no_effect", "ai_errors", "failed_actions", "game_failures", "game_maps", "dropped_games",
	"game_seconds", "record_mb", "seconds", "gpu_mb", *(f"share_{t}" for t in TYPES), *(f"failure_{k}" for k in FAILURES), "no_effect_types", "kinds",
]
KEEP_FAILED_RUNS = 10  # run folders (logs) of failed games kept for debugging; the rest are deleted


class CampaignLearner:
	def __init__(self, config, out, device, battle, resume=False):
		"""out: the run's folder; battle: the battle learner's actors (actors.ActorPool) the games' battles go to."""
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
		self.model = CampaignModel(**self.kwargs).to(device).eval()
		self.optimizer = make_optimizer(self.model, config)
		self.version, self.lr_scale, self.games_used = 1, 1.0, 0  # version: the served model's (1: as the run starts)
		self.value_ev = None  # the value heads' explained variance, smoothed over ~10 updates: how far the credit trusts them
		self.credit_evidence = {}  # the credit check's (labels.credit_evidence): the step's errors by action type, kept
		self.credit_check = 1.0  # and its verdict, how far the step's estimate may be trusted (labels.credit_trust)
		self.head_ev = {}  # each head's explained variance at the day starts, smoothed: how far it grades the next (grade_trust)
		self.value_pool, self.head_pool = None, {}  # their parts (labels.pooled_ev): the residual variance and the variance, smoothed apart
		latest = self.out / "checkpoint_latest.pt"
		if resume and latest.exists():
			state = torch.load(latest, map_location=device)
			self.kwargs = state["model_kwargs"]
			self.model = CampaignModel(**self.kwargs).to(device).eval()
			self.optimizer = make_optimizer(self.model, config)
			self.model.load_state_dict(state["model"])
			if "optimizer" in state:
				self.optimizer.load_state_dict(state["optimizer"])
			self.version, self.lr_scale, self.games_used = state["version"], state.get("lr_scale", 1.0), state.get("games_used", 0)
			self.value_ev = state.get("value_ev")
			self.credit_evidence = state.get("credit_evidence", {})
			self.credit_check = credit_trust(credit_gaps(self.credit_evidence))
			self.head_ev = state.get("head_ev", {})
			self.value_pool, self.head_pool = state.get("value_pool"), state.get("head_pool", {})
			print(f"campaign: resuming at version {self.version}", flush=True)
		elif config.init_from:
			state = torch.load(config.init_from, map_location=device)
			self.kwargs = state.get("model_kwargs") or self.kwargs
			self.model = CampaignModel(**self.kwargs).to(device).eval()
			self.optimizer = make_optimizer(self.model, config)
			self.model.load_state_dict(state["model"])
			print(f"campaign: starting from {config.init_from}", flush=True)
		if config.learner == "process":
			self.worker = UpdateProcess(config, self.kwargs, self.model, self.optimizer, self.lr_scale, latest)
		battle_play.SOCKET_DIR.mkdir(parents=True, exist_ok=True)
		self.server = InferenceServer(battle_play.SOCKET_DIR / f"campaign-{os.getpid()}.sock", device="cpu", family="campaign", model_kwargs=self.kwargs, threads=config.inference_threads, processes=config.inference_processes)
		self.server.set_model("current", self.version, self.model)
		self.finished = []  # finished games: (records.Game, play.GameInfo)
		self.lock = threading.Condition()
		self.stop = threading.Event()
		self.running = threading.Event()  # cleared while an evaluation plays: no new self-play game starts
		self.running.set()
		self.stats = Counter()
		self.model_lock = threading.Lock()  # the learner thread's update against a checkpoint taken by an evaluation
		self.battle_epoch = 0
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
		for i in range(self.config.games):
			self.threads.append(threading.Thread(target=self._game_actor, args=(i,), daemon=True))
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
		if self.worker:
			self.worker.close()
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

	def _game_actor(self, index):
		config = self.config
		rng = random.Random(1_000_033 * (config.seed + 1) + 7919 * index + 104_729 * self.version)
		n = 0
		while not self.stop.is_set():
			if not self.running.wait(timeout=1):
				continue
			n += 1
			key = f"c{index}_{n}"
			players, size, seed = rng.choice(config.players), rng.choice(config.sizes), rng.randrange(1, 2**31)
			game_map, days = rng.random() < config.game_map_share, draw_days(rng, config)
			map_name = None
			try:
				if game_map:  # one of the game's own maps (kept), else a new generated one (deleted after its game)
					map_name, players = rng.choice(sorted(GAME_MAPS.items()))
					size = 36
				else:
					map_name, _ = make_map(seed, players, size, config.levels, config.water)
				game, info = play(
					self.server.socket_for(index), map_name, days, seed, max_seconds=config.game_seconds, battle_ai="AlphaBattleAI", game_key=key,
					settings=self.battle.campaign_settings(index, key, seed), niceness=config.niceness, temperature=config.temperature, stop=self.stop,
				)
			except Exception as error:  # one failed game does not stop training
				self._failed(f"game {key}: {error}")
				continue
			finally:
				if map_name and not game_map:
					(battle_play.USER_DIR / (map_name + ".vmap")).unlink(missing_ok=True)
			game.days = days
			ok = info.status.startswith("stopped") or "exit code 0" in info.status  # (at the cap, or training is over)
			if self.stop.is_set():
				shutil.rmtree(info.run_dir, ignore_errors=True)
				return
			if ok:
				shutil.rmtree(info.run_dir, ignore_errors=True)
			else:  # its records up to then are still used: the labels take the last day it reached as its end
				self._failed(f"game {key} ({map_name}, {players} players): {info.status} (run {info.run_dir})", info.run_dir)
			self._log_game(key, map_name, players, game, info)
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

	def _take(self):
		"""The games of the next update (oldest first), or None while too few have finished. Games beyond two updates'
		worth wait for none: the oldest are dropped (counted), so a learner slower than its games trains on recent
		play, not on an ever older backlog."""
		config = self.config
		with self.lock:
			while len(self.finished) > config.min_games_per_update and sum(len(g.decisions) for g, _ in self.finished[1:]) >= 2 * config.decisions_per_update:
				self.stats["dropped_games"] += 1
				self.finished.pop(0)
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
		started = time.time()
		played = []
		for game, _ in games:
			game.decisions = usable(game.decisions)
			if game.decisions and game.ledgers:
				played.append(game)
		if not played:
			return False
		starts, trust, (weight, outcome) = day_starts(self.model, played, self.model_lock), self.grade_trust(), self.end_credit()
		decisions, targets, labels, growth, returns, advantages, errors, players, forecasts, row = [], [], [], [], [], [], [], [], [], Counter()
		end = {"targets": [], "real": [], "step": [], "week": [], "price": []}
		for g, game in enumerate(played):
			t, mixed, price = label(game, self.labels, self.empire)
			z = growth_labels(game, self.empire)
			graded, grown = ladder_targets(game, t, z, starts[g], trust, self.config.outcome_share_min)
			real = end_labels(game, self.empire)
			forecasts.append(start_forecasts(starts[g], t, z, real))
			fresh = np.array([d.version >= self.version - self.config.max_record_age for d in game.decisions])
			row["dropped_old"] += int((~fresh).sum())
			# The end credit's steps, as rows of this update (a decision after a fresh one is fresh: versions only grow)
			rows = np.where(fresh, len(decisions) + np.cumsum(fresh) - 1, -1)
			for name, index in zip(("step", "week"), end_steps(game)):
				end[name].append(np.where(index >= 0, rows[np.maximum(index, 0)], -1)[fresh])
			end["targets"].append(end_targets(game, real, starts[g], trust.get("Vend", 0.0), self.config.outcome_share_min)[fresh])
			end["real"].append(real[fresh])
			end["price"].append(price[fresh])
			decisions += [d for d, keep in zip(game.decisions, fresh) if keep]
			targets.append(graded[fresh])
			labels.append(t[fresh])
			growth.append(grown[fresh])
			returns.append((mixed - price)[fresh])
			if self.config.credit == "horizon":
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
		self._note_head_ev(forecasts)  # (for the next update's grades: these were graded by the trust before them)
		targets, labels, growth = np.concatenate(targets), np.concatenate(labels), np.concatenate(growth)
		returns, advantages = np.concatenate(returns), np.concatenate(advantages)
		end = {name: np.concatenate(parts) for name, parts in end.items()}
		row["trust"] = self.trust()
		end["outcome"], end["weight"] = outcome, weight if self.config.credit == "horizon" else 0.0
		row["end_credit"] = end["weight"]
		if errors:  # the credit check: the step's estimate against what then happened, by action type (labels.credit_evidence)
			self.credit_evidence = credit_evidence(decisions, np.concatenate(errors), np.concatenate(players), self.credit_evidence)
			gaps = credit_gaps(self.credit_evidence)
			self.credit_check = credit_trust(gaps)
			row["credit_trust"] = self.credit_check
			row["credit_gaps"] = gaps
		if self.worker:  # (the process saves checkpoint_latest.pt, with its optimizer)
			state, metrics, lr_scale = self.worker.update(decisions, targets, returns, self.labels.weights, {"version": self.version + 1, "games_used": self.games_used + len(games), "value_ev": self.value_ev, "credit_evidence": self.credit_evidence, "head_ev": self.head_ev, "value_pool": self.value_pool, "head_pool": self.head_pool}, advantages, growth, labels, end)
			with self.model_lock:
				self.model.load_state_dict(state)
				self.lr_scale = lr_scale
				self.version += 1
				self.server.set_model("current", self.version, self.model)
				self.games_used += len(games)
		else:
			with self.model_lock:
				metrics, self.lr_scale = ppo_update(self.model, self.optimizer, decisions, targets, returns, self.config, device=self.device, lr_scale=self.lr_scale, weights=self.labels.weights, advantages=advantages, growth=growth, labels=labels, end=end)
				if self.device.type == "mps":
					torch.mps.empty_cache()  # (the battle learner's lesson: MPS keeps blocks of every size it saw)
				self.version += 1
				self.server.set_model("current", self.version, self.model)
				self.games_used += len(games)
				self._note_value_ev(metrics)
				self._save(self.out / "checkpoint_latest.pt")
		if self.worker:  # (the process saved its checkpoint with the trust before this update: one update behind)
			self._note_value_ev(metrics)
		self._log(epoch, games, decisions, row, metrics, time.time() - started)
		return True

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
		pairs = [(f"V{k}", shares[:, i] - prior, real[:, i] - prior) for i, k in enumerate(c.VALUE_OFFSETS)] + [(f"G{k}", grown[:, i], real_growth[:, i]) for i, k in enumerate(c.VALUE_OFFSETS)]
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
		return self.config.end_credit_max * trust, 1 - (1 - self.config.outcome_share_min) * trust

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
			"entropy": metrics.get("entropy", float("nan")),
			"below_floor": metrics.get("below_floor", float("nan")),
			"kl": metrics.get("kl", float("nan")),
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
			"gpu_mb": metrics.get("gpu_mb", float("nan")),  # the update's peak on Apple's GPU (train.gpu_held)
			**{f"share_{t}": types[t] / max(len(turn), 1) for t in TYPES},
			**{f"failure_{k}": v for k, v in failure_modes(decisions).items()},  # dumb decisions (failures.py)
			"kinds": " ".join(f"{k}:{n}" for k, n in sorted(Counter(c.DecisionKind(d.kind).name for d in decisions).items())),
			"no_effect_types": " ".join(f"{t}:{n}" for t, n in sorted(Counter(c.ActionType(d.action_type).name for d in decisions if d.flags & NO_EFFECT).items())),
		}
		with self.progress.open("a", newline="") as f:
			csv.writer(f).writerow([round(row[k], 6) if isinstance(row[k], float) else row[k] for k in COLUMNS])
		for name in ("value_corr", "value_spread", "label_spread", "gpu_anchor_mb", "update_seconds"):
			if metrics.get(name, float("nan")) == metrics.get(name, float("nan")):
				self.writer.add_scalar(f"campaign/{name}", metrics[name], self.version)
		for kind, gap in sums.get("credit_gaps", {}).items():  # (the credit check's t-statistic per action type)
			self.writer.add_scalar(f"campaign_credit/{kind.lower()}", gap, self.version)
		for skill, (picked, offered) in skill_picks(decisions).items():  # (how often each skill is taken when offered)
			self.writer.add_scalar(f"campaign_skills/{skill}", picked / offered, self.version)
		for key in COLUMNS[2:]:
			value = row[key]
			if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value:
				group = "campaign_actions/" + key[6:] if key.startswith("share_") else "campaign_failures/" + key[8:] if key.startswith("failure_") else "campaign/" + key
				self.writer.add_scalar(group, value, self.version)
		self.writer.flush()
		print(
			f"campaign {self.version:4d}  games {row['games']}  decisions {n}  days {row['days']:.1f}  Φ at the end {row['phi_end']:.0f}  heroes {row['heroes']:.2f}"
			"  value ev " + "/".join(f"{row[f'value_ev_d{k}']:+.2f}" for k in c.VALUE_OFFSETS)
			+ f" (corr {metrics.get('value_corr', float('nan')):+.2f}, spread {metrics.get('value_spread', float('nan')):.3f} vs {metrics.get('label_spread', float('nan')):.3f})"
			+ "  growth ev " + "/".join(f"{row[f'head_ev_g{k}']:+.2f}" for k in c.VALUE_OFFSETS) + f"  end ev {row['head_ev_vend']:+.2f} (credit {row['end_credit']:.2f})"
			+ f"  kl {row['last_kl']:.4f}  steps {row['steps']}  {seconds:.1f}s (update {metrics.get('update_seconds', float('nan')):.0f}s)  gpu {row['gpu_mb']:.0f} MB (anchor {metrics.get('gpu_anchor_mb', float('nan')):.0f})",
			flush=True,
		)

	def save(self, path):
		with self.model_lock:
			self._save(path)

	def _save(self, path):
		state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
		optimizer = {} if self.worker else {"optimizer": self.optimizer.state_dict()}  # (the process's is in checkpoint_latest.pt)
		torch.save({"model": state, **optimizer, "version": self.version, "lr_scale": self.lr_scale, "games_used": self.games_used, "value_ev": self.value_ev, "credit_evidence": self.credit_evidence, "head_ev": self.head_ev, "value_pool": self.value_pool, "head_pool": self.head_pool, "model_kwargs": self.kwargs}, path)

	# ---- evaluation ----

	def eval_due(self):
		if not self.config.eval_minutes or not self.config.eval_games:
			return False
		return self.last_eval is None or time.monotonic() - self.last_eval >= self.config.eval_minutes * 60

	def evaluate(self, epoch):
		"""Games against Nullkiller2 (evaluate.py) with the current models; no new self-play game starts meanwhile."""
		self.running.clear()
		try:
			row = evaluate.evaluate(self.config, self.server, self.battle, version=self.version, epoch=epoch, stop=self.stop)
		finally:
			self.running.set()
			self.last_eval = time.monotonic()
		evaluate.log(row, self.out / "eval.csv", self.writer)
		self.save(self.out / f"checkpoint_{self.version:05d}.pt")
		return row


def skill_picks(decisions):
	"""{skill: (times taken, times offered)} over the level-ups among the decisions: the user's worry (2026-10-28), that a
	7-day horizon takes Estates too often and Offence, Earth Magic or Wisdom (high-level spells weeks on) too rarely,
	which the end head is there to change."""
	ups = [d for d in decisions if d.kind == c.DecisionKind.LEVEL_UP and 0 <= d.index < c.N_OPTIONS]
	if not ups:
		return {}
	x, counts = decode_inputs(ups), {}
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
	for k in range(0, len(rows), batch):
		chunk = rows[k : k + batch]
		x = decode_inputs([games[g].decisions[i] for g, i in chunk])
		with lock, torch.no_grad():
			out = model({name: torch.from_numpy(v).to(device) for name, v in x.items()})
		shares = out["value"].float().cpu().numpy()
		grown = ((2 * torch.sigmoid(out["growth_logit"]) - 1) * GROWTH_CAP).float().cpu().numpy()
		log_value = np.log(np.maximum(out["phi"].float().cpu().numpy(), PHI_FLOOR))
		end = torch.sigmoid(out["end_logit"]).float().cpu().numpy()
		prior = 1.0 / np.maximum(x["g_players"].astype(np.float32), 2.0)
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
	model = CampaignModel(**kwargs).to(device).eval()
	optimizer = make_optimizer(model, config)
	model.load_state_dict(state["model"])
	if state.get("optimizer"):
		optimizer.load_state_dict(state["optimizer"])
	lr_scale = state["lr_scale"]
	while (request := requests.get()) is not None:
		decisions, targets, returns, weights, about, advantages, growth, labels, end = request
		try:
			metrics, lr_scale = ppo_update(model, optimizer, decisions, targets, returns, config, device=device, lr_scale=lr_scale, weights=weights, advantages=advantages, growth=growth, labels=labels, end=end)
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

	def update(self, decisions, targets, returns, weights, about, advantages=None, growth=None, labels=None, end=None):
		"""(the new weights on the CPU, the update's metrics, the KL control's lr_scale); raises if the update failed."""
		self.requests.put((decisions, targets, returns, weights, about, advantages, growth, labels, end))
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
