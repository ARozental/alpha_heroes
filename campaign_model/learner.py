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
import bisect
import math
import multiprocessing
import os
import pickle
import shutil
import threading
import time
import traceback
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from battle_model import play as battle_play
from battle_model.device import use_fused
from battle_model.optim import calibrated_clips, clip_settings, stale_clips
from battle_model.inference import InferenceServer
from battle_model.supply import trim_memory

from . import constants as c
from . import evaluate
from .empire import STATUS_INGAME, EmpireConfig, empire_value
from .failures import NAMES as FAILURES, failure_modes
from .features import SKILLS
from .labels import (
	COLORS, CREDIT_MEMORY, CREDIT_MIN_PLAYERS, GROWTH_CAP, PHI_FLOOR, LabelConfig, actions_ahead, attributed_labels, credit_evidence, credit_gaps, credit_trust, end_labels, end_steps, end_targets, growth_labels, horizon_advantages,
	label, ladder_targets, pooled_ev, residual_variances, share_change_labels, share_logit, share_now, tables,
)
from .advice import teacher_sets
from .foresight import LABEL_NAMES, labels_of
from .labeler import Labeler
from .model import build_model
from .play import choice_hidden
from .records import NO_EFFECT, TEACHER, decode_fields, decode_inputs
from .train import GRAD_GROUPS, HEAD_SCALE_RAMP, _prefetched, fit_moments, load_optimizer, make_optimizer, migrate_subtypes, model_kwargs, pooled_evidence, ppo_update, segment_shrinkage, usable

TYPES = [a.name.lower() for a in c.ActionType]
FORESIGHT_COLUMNS = tuple(f"foresight_{head}_{part}" for head in ("move", "answer") for part in ("battle", "score", "parts")) + ("foresight_hero_alive", "foresight_hero_army", "foresight_turn", "foresight_plan", "foresight_move_tiles", "foresight_move_win", "foresight_move_stake")
# The shared layers' gradient norm from each term of the loss as it enters it (foresight.gradient_check at weight 1,
# times the term's weight; grad_shared_foresight is the foresight's with its plan, these the plan's alone and the rest's),
# and the loss shares' weights (config.loss_shares)
GRADIENT_COLUMNS = tuple(f"grad_shared_{term}" for term in ("policy", "value", "entropy", "teacher", "plan")) + ("loss_weight_value", "loss_weight_teacher", "loss_weight_foresight", "teacher_share", "teacher_set_agree")
# (2026-11-10) each policy term's push on the watched prior corrections (foresight.WATCHED): minus its gradient there
# times its weight in the loss, on one micro-batch an update (the gradient check's); below 0 drives the correction down
PUSH_COLUMNS = tuple(f"push_{label}_{term}" for label in ("night_gain", "danger_gain") for term in ("policy", "entropy", "teacher"))
GRADIENT_COLUMNS = GRADIENT_COLUMNS + PUSH_COLUMNS
# The loss shares (config.loss_shares; the user, 2026-11-02): each term of the loss gets a share of the shared layers'
# gradient, its weight set from its gradient's running norm (GradNorm's balance, Chen et al. 2018, in closed form: the
# weight that gives a term its share against the policy's, whose weight stays 1), each weight moving at most
# LOSS_SCALE_STEP times an update. The teacher's share is how good we are (teacher_share_for), the rest policy : value :
# foresight 4 : 3 : 1: policy and value never under 15% each
LOSS_TERMS = ("policy", "value", "teacher", "foresight")
TEACHER_SHARE_MAX = 0.6
LOSS_SCALE_STEP = 1.25
LOSS_SCALE_MEMORY = 0.9  # (each term's log gradient norm, averaged over ~10 updates)
# The value term's share is a floor (the user's "never under 15%", 2026-11-02), not a target: its weight is raised while
# its share is below its target, and cut only past VALUE_SHARE_CAP times it. Held at the target, the balancer cut the
# weight exactly when the heads were most wrong: after schema 14 (v1138-v1145) the value's gradient rose to a share of
# 0.33-0.47 against 0.15 and its weight fell 356 -> 146; the 7-day growth heads' explained variance 0.40-0.45 -> 0.19-0.23,
# and ~100 updates later it had won back half (2026-11-05)
VALUE_SHARE_CAP = 3.0
COLUMNS = [
	"version", "epoch", "games", "decisions", "dropped_old", "unsound", "credited", "days", "players", "phi_end", "heroes", "label_mean", "price_mean",
	"trust", "credit_trust", "end_credit", "policy_loss", "value_loss", "value_loss_weighted", "value_loss_fresh", "entropy", "below_floor", "kl", "last_kl", "teacher_kl", "teacher_weight", "advised", "teacher_agree", "teacher_ce", "teacher_outside", "clip_fraction", "grad_norm", "steps", "learning_rate",
	*(f"value_ev_d{k}" for k in c.VALUE_OFFSETS), *(f"head_ev_{kind}{k}" for kind in "vg" for k in c.VALUE_OFFSETS), "head_ev_vend", "no_effect", "ai_errors", "failed_actions", "game_failures", "game_maps", "dropped_games",
	"game_seconds", "record_mb", "seconds", "gpu_mb", "label_staleness", *(f"share_{t}" for t in TYPES), *(f"failure_{k}" for k in FAILURES), "no_effect_types", "kinds",
	# v2's foresight losses (foresight.py) and the gradient check: the shared layers' gradient from the PPO losses and the
	# foresight's, their cosine, the share of parameters no loss reaches
	*FORESIGHT_COLUMNS, "grad_shared_main", "grad_shared_foresight", "grad_shared_cosine", "grad_dead_share", *GRADIENT_COLUMNS,
	# (2026-11-11, GATHER's return) its label's weight this update (config.nk2_gather_ramp), and the readings that saw the
	# 2026-11-10 collapse first: our armies' gold on heroes against all of it (heroes, garrisons, armies no hero leads),
	# players out of their games, the spread of the value heads' targets
	"gather_weight", "army_on_heroes", "eliminated", "value_target_std",
	"night_dropped",  # (config.nk2_night_drop: the share of Nullkiller2's advised labels left out as into the night)
	"valve_r", "valve_s", "kl_target",  # (the value valves: r_v after this update's reading, s and the KL target this update used)
	# (config.adv_weight, the advantage head: its loss, its explained variance of the update's advantage before the
	# update's steps, apart on the choices that were Nullkiller2's own and on those that left one it had, their share,
	# the advantage's spread before it is standardized)
	"adv_loss", "adv_ev", "adv_ev_followed", "adv_ev_deviating", "adv_followed", "adv_raw_sd",
	# (config.nk2_choice_games: the share of the update's games that hid Nullkiller2's own choice; in those and in the rest,
	# how often our choice was its advice, the advantage head's explained variance, players out of their game)
	"choice_hidden", "teacher_agree_shown", "teacher_agree_hidden", "adv_ev_shown", "adv_ev_hidden", "eliminated_shown", "eliminated_hidden",
	"adv_raw_sd_shown", "adv_raw_sd_hidden",
	# (config.adv_gain: the KL the head's term adds to our own policy a decision and the mean |gain x A_hat|, all and apart)
	"adv_kl", "adv_read", "adv_kl_shown", "adv_kl_hidden", "adv_read_shown", "adv_read_hidden",
	# (config.schedule_action_price: the mean charge a decision and its share of the 7-day credit's spread; what it is judged
	# by: our heroes' idle movement, and how often a day's first action is END, apart for the two kinds of games)
	"action_price_mean", "action_price_share", "idle_movement_shown", "idle_movement_hidden", "end_first", "end_first_shown", "end_first_hidden",
	# (each kind of game's advantage as the policy and the head take it: its mean, its spread, its share of sum |A| x credit;
	# config.adv_by_kind standardizes them apart)
	"adv_mean_shown", "adv_mean_hidden", "adv_sd_shown", "adv_sd_hidden", "adv_weight_shown", "adv_weight_hidden",
	"loops_cut", "loops_cut_move", "loops_cut_multi", "probes_no_way",  # (bundle 60's loop guard, a game: summed over its players)
	"loops_per_player_day",
	# (config.nk_predict_weight, nk_predict.py: the head's loss on its blind rows, and its fit on the prepass's rows before
	# the steps: a's AUC and its positives' share, b's explained variance per form, c's cross-entropy against the uniform's,
	# its top-1 agreement with the hero's best candidate and with Nullkiller2's recorded own choice)
	"nkp_loss", "nkp_auc", "nkp_positives", *(f"nkp_ev_{form}" for form in ("preference_all", "preference_hero", "soon", "later", "rank")), "nkp_ce", "nkp_ce_uniform", "nkp_top1_best", "nkp_top1_choice",
	# (config.credit_segments: the k's this update's credit used, each segment's F (its action types' between-type spread
	# of gaps over their sampling variance) and the number of types that entered it, the credit's spread with the k's and
	# the 7-day credit's on the same rows: the noise the k's leave out)
	"segment_k5", "segment_k7", "segment_f2", "segment_f5", "segment_f7", "segment_types2", "segment_types5", "segment_types7", "segment_credit_std", "segment_full_std",
	# (2026-11-12: F stratified by the kind of game, shown or hidden; the same evidence's F with the kinds merged, the
	# statistic before, for comparison)
	"segment_f2_pooled", "segment_f5_pooled", "segment_f7_pooled",
	# (e5's learning-rate readings, 2026-11-11: each optimizer group's signal fraction (train.signal_fraction, in
	# TensorBoard before), the trunk group's shared layers (body) and policy heads apart with their mean |m| / sqrt(v)
	# (train.trunk_moments), the KL control's lr scale; kl, last_kl and learning_rate are above)
	*(f"signal_{name}" for name in ("all", *GRAD_GROUPS)), "signal_body", "signal_heads", "ratio_body", "ratio_heads", "lr_scale",
]
SEGMENT_RATE = 0.05  # config.credit_segments: each long segment's k moves this far toward its measured max(0, 1 - 1/F) an update
KEEP_FAILED_RUNS = 10  # run folders (logs) of failed games kept for debugging; the rest are deleted
DROP_BEYOND = 4  # updates' worth of finished games beyond which the oldest are dropped (_take)
# The value-loss guard (config.value_jump_stop; the user, 2026-11-10: a jump of the campaign's value loss is the first
# sign that something has gone wrong, and the learning must stop at once: it does not say the change is bad). The fresh
# value loss (value_loss_fresh) of the last VALUE_JUMP_WINDOW updates, its median, against its calm level: the lowest
# median of VALUE_JUMP_BASELINE readings in a row among the VALUE_JUMP_MEMORY before them. On joint7's 2,837 updates,
# at 3 times the level, it fires twice: v1137 (2026-11-05, the first GATHER advice: the loss rose to 29 times its
# level, the evals fell to 0.25) and v2749 (2026-11-10, the second: 14 updates after the deploy, 87 updates and five
# evals before anybody looked); single readings 2-5 times the level follow every restart and do not fire it
VALUE_JUMP_WINDOW, VALUE_JUMP_BASELINE, VALUE_JUMP_MEMORY = 7, 50, 300
# The value valves (config.value_valve, kl_valve; valve_step): r_v's window and its reference's horizon (updates), the
# floors of the shared layers' share (s) and of the KL target's factor (k), the most either moves an update (times)
VALVE_WINDOW, VALVE_HORIZON, VALVE_S_MIN, VALVE_K_MIN, VALVE_RATE = 7, 50, 0.1, 0.25, 1.25
MAX_LABEL_STALENESS = 2  # model versions: older precomputed labels (labeler.py) are recomputed with the current tables


def set_advice_prior(model, config):
	"""config.nk2_prior and nk2_prior_fallback into the model's buffers (model.CustomFeatures.nk_beta, nk_beta_fallback):
	the size of Nullkiller2's preference as a prior on MOVE's and TRAVEL's entries. At every start, after the weights
	are loaded: the run's line decides it, the checkpoints and the inference servers carry it (a model without the
	buffers, v1: nothing)."""
	features = getattr(model, "features", None)
	if features is None or not hasattr(features, "nk_beta"):
		return
	with torch.no_grad():
		features.nk_beta.fill_(float(getattr(config, "nk2_prior", 0.0)))
		features.nk_beta_fallback.fill_(float(getattr(config, "nk2_prior_fallback", 0.0)))
		features.nk_label.fill_(float(getattr(config, "nk2_label_prior", 0.0)))
		features.nk_follow.fill_(float(getattr(config, "nk2_follow", 0.0)))
		if hasattr(features, "adv_gain"):  # (the advantage head read by our own policy: config.adv_gain)
			features.adv_gain.fill_(float(getattr(config, "adv_gain", 0.0)))
		if hasattr(features, "town_guard"):  # (bundle 60's town guard: config.town_guard)
			features.town_guard.fill_(float(getattr(config, "town_guard", 0.0)))


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
		self.segment_evidence = [{}, {}, {}]  # (config.credit_segments: credit_evidence of each segment's surprise, 0-2 2-5 5-7 days)
		self.segment_k = (0.0, 0.0)  # (and the 2-5 and 5-7 segments' weights in the credit, from 0: the 2-day credit)
		self.head_ev = {}  # each head's explained variance at the day starts, smoothed: how far it grades the next (grade_trust)
		self.value_pool, self.head_pool = None, {}  # their parts (labels.pooled_ev): the residual variance and the variance, smoothed apart
		self.head_var, self.head_scale_updates = None, 0  # (config.head_scale) the heads' running target variances, the updates since it began
		self.loss_scale, self.teacher_share = None, None  # (config.loss_shares) {"log_norm", "weight"} of the terms; the teacher's share
		self.trust_ema = None  # (config.trust_memory) trust's slow average, trust() floors it at config.trust_floor
		self.head_residual, self.head_var_slow, self.head_residual_updates = None, None, 0  # (config.head_scale_by "residual") the heads' running fresh excess, the variances that scale it, the updates since it began
		# (config.value_jump_stop) the value-loss guard: why the updates are held (None: they run), the updates' games
		# dropped since, the fresh value loss's readings (progress.csv's on a resume, _judge_value's after), this process's own
		self.held, self.held_updates, self.value_readings, self.value_own = None, 0, [], 0
		self.grad_clips, self.clip_samples = None, []  # (config.grad_clip_by_group) the groups' clips once calibrated, the calibration's steps' norms till then
		self.gather_ramp_start = None  # (config.nk2_gather_ramp) the version at the ramp's first update
		self.valve = None  # (config.value_valve, kl_valve) valve_step's state: r_v's reference and window, s and k
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
			self.segment_evidence, self.segment_k = state.get("segment_evidence", [{}, {}, {}]), tuple(state.get("segment_k", (0.0, 0.0)))
			self.head_ev = state.get("head_ev", {})
			self.value_pool, self.head_pool = state.get("value_pool"), state.get("head_pool", {})
			self.head_var, self.head_scale_updates = state.get("head_var"), state.get("head_scale_updates", 0)
			self.loss_scale, self.teacher_share = state.get("loss_scale"), state.get("teacher_share")
			self.trust_ema = state.get("trust_ema")
			self.head_residual, self.head_var_slow, self.head_residual_updates = state.get("head_residual"), state.get("head_var_slow"), state.get("head_residual_updates", 0)
			self.grad_clips, self.clip_samples = state.get("grad_clips"), state.get("clip_samples", [])
			self.gather_ramp_start = state.get("gather_ramp_start")
			self.valve = state.get("valve")
			print(f"campaign: resuming at version {self.version}", flush=True)
		elif config.init_from:
			state = torch.load(config.init_from, map_location=device)
			self.kwargs = state.get("model_kwargs") or self.kwargs
			self.model = build_model(self.kwargs).to(device).eval()
			self.optimizer = make_optimizer(self.model, config)
			migrate_subtypes(state, self.model)
			self.model.load_state_dict(state["model"])
			print(f"campaign: starting from {config.init_from}", flush=True)
		set_advice_prior(self.model, config)
		if getattr(config, "nk2_inputs", False) or config.nk2_prior or config.nk2_prior_fallback or config.nk2_label_prior or config.nk2_follow or not getattr(config, "nk2_teacher", True):
			print(f"campaign: schema 19: Nullkiller2's view as inputs {'on' if getattr(config, 'nk2_inputs', False) else 'off'} (off in {getattr(config, 'nk2_input_dropout', 0.0):g} of the training games), its own choice followed at a chance of {config.nk2_follow:g}, the prior on it {config.nk2_label_prior:g}, on its best candidate {config.nk2_prior:g} (fallback {config.nk2_prior_fallback:g}), the teacher's loss {'on' if getattr(config, 'nk2_teacher', True) else 'OFF'}", flush=True)
		if stale_clips(self.grad_clips, config.grad_clip_since):  # (a recalibration: config.grad_clip_since)
			print(f"campaign: the gradient clips of v{self.grad_clips.get('version', '?')} measured again (grad_clip_since {config.grad_clip_since})", flush=True)
			self.grad_clips, self.clip_samples = None, []
		if clip_settings(config.grad_clip_groups, "campaign.grad_clip_groups"):  # (the groups' clips by hand, over any calibrated ones)
			self.grad_clips = clip_settings(config.grad_clip_groups, "campaign.grad_clip_groups")
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
			# (UpdateProcess alone is tested; with the games supply.py hands this process since 2026-10-30 it never ran)
			raise SystemExit("campaign.learner 'process' has not run since the supply's rebuild (2026-10-30): use 'turns' (or 'thread')")
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
			self.value_readings = fresh_readings(self.progress, since=int(getattr(config, "value_jump_from", 0) or 0))
		(self.out / "held").unlink(missing_ok=True)  # (the value-loss guard's note of a hold: a restart lifts it)
		# A new run evaluates its starting model first; a resumed one when eval_minutes have passed since its last eval
		# (by eval.csv's time: restarts must not keep putting evals off)
		evals = self.out / "eval.csv"
		if resume and evals.exists():  # (its columns grew: vs_previous, vs_first)
			from battle_model.train import upgrade_csv

			upgrade_csv(evals, evaluate.COLUMNS)
		self.last_eval = time.monotonic() - (time.time() - evals.stat().st_mtime) if resume and evals.exists() else None
		if resume and evals.exists() and config.teacher_mix == "score" and (getattr(self, "teacher", None) is not None or config.nk2_advice > 0):
			# The teacher's weight from the last evaluation's score, its source (evaluate): the checkpoint is the updates',
			# and a weight an evaluation set after the last update came back unchanged (2026-11-01 16:16 CST: 19, not eval
			# 56's 13.76)
			with evals.open(newline="") as f:
				rows = list(csv.DictReader(f))
			score = evaluate.eval_score(rows[-1]) if rows else float("nan")  # (OWN's in a split evaluation)
			if np.isfinite(score):
				self.teacher_weight, self.teacher_share = teacher_weight_for(score), teacher_share_for(score)
				print(f"campaign: the teacher's weight {self.teacher_weight:.2f} (its gradient share {self.teacher_share:.2f} with config.loss_shares) from the last eval's score {score:.3f}", flush=True)
		# Each finished game is on disk from its arrival until the checkpoint of the update that trained on it is saved,
		# and a resumed run takes them back: a stop (a crash, the memory guard, a restart) loses no finished game (until
		# 2026-11-01 each restart lost the ~20-40 finished games waiting for their update)
		self.spool = self.out / "finished"
		self.spool.mkdir(exist_ok=True)
		if resume:
			self._reload_spool()
		else:
			for path in self.spool.glob("*.pkl"):
				path.unlink()

	# ---- games ----

	def _spool_game(self, game, info):
		"""Writes the game to the spool (whole, or not at all: a stop while it is written leaves only its .tmp)."""
		path = self.spool / f"{time.time_ns()}-{id(game)}.pkl"
		tmp = path.with_suffix(".tmp")
		try:
			with tmp.open("wb") as f:
				pickle.dump((game, info), f, protocol=pickle.HIGHEST_PROTOCOL)
			tmp.rename(path)
			game.spool_path = str(path)
		except OSError as error:
			print(f"campaign: FAILURE: a finished game not written to {self.spool}: {error}", flush=True)

	def _reload_spool(self):
		"""The finished games a stopped run had not trained on, back in the queue (oldest first), labeled again."""
		for tmp in self.spool.glob("*.tmp"):  # (cut off as it was written: the game's in-memory copy was its only one)
			tmp.unlink()
		paths = sorted(self.spool.glob("*.pkl"))
		for path in paths:
			try:
				with path.open("rb") as f:
					game, info = pickle.load(f)
			except Exception as error:
				print(f"campaign: FAILURE: {path.name} unreadable, left out: {str(error)[:200]}", flush=True)
				path.unlink()
				continue
			game.spool_path = str(path)
			if getattr(self, "labeler", None):
				game.labels_version = self.version
				self.label_futures[id(game)] = self.labeler.submit(label_source(game), self.features_cpu)
			strip_inputs(game)
			self.finished.append((game, info))
		if paths:
			print(f"campaign: {len(self.finished)} finished games from before the stop back in the queue ({sum(len(g.decisions) for g, _ in self.finished)} decisions)", flush=True)

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
		self._spool_game(game, info)
		if getattr(self, "labeler", None):
			game.labels_version = self.version
			self.label_futures[id(game)] = self.labeler.submit(label_source(game), self.features_cpu)  # (not on the game: it is pickled to the labeler after this returns)
		strip_inputs(game)
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
				unspool([dropped])
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
		if getattr(self, "held", None) is not None:
			# The value-loss guard holds the updates (_judge_value): the model stays as it is, served and evaluated as
			# before, and the games that would have made this update are dropped (their battles went to the battle
			# learner as they were played). A restart lifts it: somebody has looked
			with self.lock:
				for game, _ in games:
					getattr(self, "label_futures", {}).pop(id(game), None)
				self.stats["dropped_games"] += len(games)
			unspool(game for game, _ in games)
			self.held_updates += 1
			if self.held_updates % 20 == 1:
				print(f"campaign: FAILURE: the updates are HELD at version {self.version} ({self.held}); {self.held_updates} updates' games dropped since. A restart lifts the hold", flush=True)
			return False
		started, memory = time.time(), [("waited", rss_gb())]
		unread = [game for game, _ in games if not restore_inputs(game)]  # (strip_inputs: only their spool files had them)
		if unread:
			games = [(game, info) for game, info in games if not any(game is u for u in unread)]
			unspool(unread)
		memory.append(("taken", rss_gb()))
		played = []
		for game, _ in games:
			game.all_decisions = game.decisions  # (the foresight's labels count record serials: the battles' point at them)
			game.decisions = usable(game.decisions)
			if game.decisions and game.ledgers:
				played.append(game)
		if not played:
			unspool(game for game, _ in games)
			return False
		starts, trust, (weight, outcome) = day_starts(self.model, played, self.model_lock), self.grade_trust(), self.end_credit()
		memory.append(("starts", rss_gb()))
		foresight = [] if self.kwargs.get("architecture") == "v2" else None  # (v2's foresight labels, each game's)
		sets = [] if getattr(self.config, "nk2_advice", 0) > 0 else None  # (the teacher's entries beyond the advice's: advice.teacher_sets)
		decisions, targets, labels, growth, returns, advantages, errors, players, forecasts, row = [], [], [], [], [], [], [], [], [], Counter()
		growth_real, attributed = [], {"share_now": [], "taken": []}
		privileged = [] if self.config.privileged_critic and getattr(self.model, "privileged", None) is not None else None  # (privileged_rivals, each game's)
		end = {"targets": [], "real": [], "step": [], "week": [], "price": []}
		if self.config.schedule_action_price > 0:  # (each decision's actions through its 7-day window: labels.actions_ahead)
			end["actions"] = []
		streams = ([], [])  # each decision's game and player, and its place among the player's records (config.memory_steps)
		hidden = []  # (each decision's game hid Nullkiller2's own choice: config.nk2_choice_games, play.choice_hidden)
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
			if "actions" in end:
				end["actions"].append(actions_ahead(game)[fresh])
			if privileged is not None:
				privileged.append(privileged_rivals(game, fresh, len(decisions), self.empire))
			decisions += [d for d, keep in zip(game.decisions, fresh) if keep]
			hidden.append(np.full(int(fresh.sum()), choice_hidden(game), dtype=bool))
			place, count = {}, Counter()  # (all_decisions: every record as sent, unusable ones too: a gap breaks a sequence)
			for d in getattr(game, "all_decisions", game.decisions):
				place[id(d)], count[d.color] = count[d.color], count[d.color] + 1
			streams[0].append(np.array([g * len(COLORS) + COLORS.index(d.color) for d in game.decisions], dtype=np.int64)[fresh])
			streams[1].append(np.array([place[id(d)] for d in game.decisions], dtype=np.int64)[fresh])
			if foresight is not None:
				# (labels precomputed before a label was added, a spooled game's at a resume: computed again here)
				rows = pre["foresight"] if pre is not None and "foresight" in pre and LABEL_NAMES <= pre["foresight"].keys() else labels_of(game, self.features_cpu)
				foresight.append({name: value[fresh] for name, value in rows.items()})
			if sets is not None:  # (over all the game's decisions: what we did counts, usable or not)
				full = getattr(game, "all_decisions", game.decisions)
				kept = {id(d) for d in game.decisions}
				used = np.array([id(d) in kept for d in full], dtype=bool)
				sets.append(tuple(part[used][fresh] for part in teacher_sets(full)))
				row["set_agree"] += set_agreement([d for d, keep in zip(game.decisions, fresh) if keep], *sets[-1])
				row["set_advised"] += sum(1 for d, keep in zip(game.decisions, fresh) if keep and d.advice is not None)
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
		gather_weight = self.gather_weight_now()
		row["gather_weight"] = gather_weight
		valve_s, valve_k = self.valve_now()
		row["valve_s"], row["kl_target"] = valve_s, self.config.target_kl * valve_k
		if hasattr(self.model, "value_valve"):
			with self.model_lock:
				self.model.value_valve.fill_(valve_s)
		with np.errstate(all="ignore"):
			spread = np.nanstd(np.where(np.isfinite(targets), targets, np.nan), axis=0)
		row["value_target_std"] = float(np.nanmean(spread)) if np.isfinite(spread).any() else float("nan")
		returns, advantages = np.concatenate(returns), np.concatenate(advantages)
		end = {name: np.concatenate(parts) for name, parts in end.items()}
		if foresight is not None:
			foresight = {name: np.concatenate([rows[name] for rows in foresight]) for name in foresight[0]}
		if sets is not None:
			sets = tuple(np.concatenate([game_sets[k] for game_sets in sets]) for k in range(3))
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
			state, metrics, lr_scale = self.worker.update(decisions, targets, returns, self.labels.weights, {"version": self.version + 1, "nk2_gather_weight": gather_weight, "gather_ramp_start": self.gather_ramp_start, "value_valve": valve_s, "target_kl": self.config.target_kl * valve_k, "valve": self.valve, "games_used": self.games_used + len(games), "value_ev": self.value_ev, "credit_evidence": self.credit_evidence, "head_ev": self.head_ev, "value_pool": self.value_pool, "head_pool": self.head_pool}, advantages, growth, labels, end, foresight)
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
					metrics, self.lr_scale = ppo_update(self.model, self.optimizer, decisions, targets, returns, replace(self.config, nk2_gather_weight=gather_weight, target_kl=self.config.target_kl * valve_k), device=self.device, lr_scale=self.lr_scale, weights=self.labels.weights, advantages=advantages, growth=growth, labels=labels, end=end, foresight=foresight, growth_real=np.concatenate(growth_real), attributed=attributed, teacher=self.teacher, teacher_weight=self.loss_weights()["teacher"], share_trust=self.trust(), growth_trust=self.grade_trust().get("G7", 0.0), teacher_sets=sets, loss_weights=self.loss_weights() if getattr(self.config, "loss_shares", False) else None, head_scale=self._head_scale() if getattr(self.config, "head_scale", False) else None, measure_noise=bool(self.config.noise_every) and self.version % self.config.noise_every == 0, streams=(np.concatenate(streams[0]), np.concatenate(streams[1])), clips=self.grad_clips if self.config.grad_clip_by_group else None, privileged={name: np.concatenate([part[name] for part in privileged]) for name in privileged[0]} if privileged else None, hidden=np.concatenate(hidden) if hidden else None, segment_k=self.segment_k if getattr(self.config, "credit_segments", False) else None)
				except torch.OutOfMemoryError as error:  # (handled below, once the error's frames, and their tensors, are gone)
					out_of_memory = str(error)[:200]
			if out_of_memory is not None:
				# The run goes on (2026-11-01: two of joint7's updates ran out of the GPU's 27 GB and ended it, and their
				# games with it): this update's games are dropped, as games with no decisions are; the weights keep the
				# steps it took, served with the next update
				torch.cuda.empty_cache()
				print(f"campaign update {self.version + 1}: out of GPU memory, its {len(games)} games dropped: {out_of_memory}", flush=True)
				unspool(game for game, _ in games)
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
				self._note_gradients(metrics)
				self._calibrate_clips(metrics)
				if metrics.get("head_residual") is not None:  # (config.head_scale_by "residual": for the next update; its ramp goes on)
					self.head_residual, self.head_var_slow, self.head_residual_updates = metrics["head_residual"], metrics["head_var_slow"], self.head_residual_updates + 1
				if metrics.get("head_var") is not None:  # (config.head_scale: for the next update; the ramp goes on)
					self.head_var, self.head_scale_updates = metrics["head_var"], getattr(self, "head_scale_updates", 0) + 1
					weights = metrics.get("head_weights") or {}
					print("campaign value heads' weights (ramp %.2f): " % min(1.0, self.head_scale_updates / HEAD_SCALE_RAMP) + "  ".join(f"{name} " + "/".join(f"{w:.2f}" for w in values) for name, values in weights.items()) + "  target std " + "  ".join(f"{name} " + "/".join(f"{v ** 0.5:.3f}" for v in values) for name, values in self.head_var.items()), flush=True)
				self._save(self.out / "checkpoint_latest.pt")
		unspool(game for game, _ in games)  # (in the checkpoint now: this process's, or the update process's)
		if self.worker:  # (the process saved its checkpoint with the trust before this update: one update behind)
			self._note_value_ev(metrics)
		end_errors = metrics.pop("end_step_error", None)
		if end_errors is not None and errors:  # the end credit's check, the 7-day one's for V_end's steps (logged only)
			self.end_evidence = credit_evidence(decisions, end_errors, np.concatenate(players), getattr(self, "end_evidence", None))
			row["end_credit_gaps"] = credit_gaps(self.end_evidence)
			self.end_check = credit_trust(row["end_credit_gaps"])  # (for the next update's end credit: end_credit)
			row["end_check"] = self.end_check
		surprises = metrics.pop("segment_surprise", None)
		if surprises is not None:  # (config.credit_segments: each segment's evidence, and the k's for the next update)
			row.update(self._note_segments(decisions, surprises, np.concatenate(streams[0]), np.concatenate(hidden) if hidden else None))
		self._relabel_waiting()
		memory.append(("updated", rss_gb()))
		self._log(epoch, games, decisions, row, metrics, time.time() - started)
		memory.append(("logged", rss_gb()))
		# The update's freed memory back to the system: glibc kept it, and the trainer climbed from ~4 GB after each
		# restart to 11-14 GB within ~4 updates (2026-11-01), memory the games here could not have
		del games, decisions
		trim_memory()
		if self.device.type == "cuda" and hasattr(torch._C, "_host_emptyCache"):
			torch._C._host_emptyCache()  # (the pinned staging's cached blocks: the trainer held 1.85 GB of such mappings)
		memory.append(("trimmed", rss_gb()))
		if all(value == value for _, value in memory):
			print("campaign memory (GB, this process): " + "  ".join(f"{stage} {value:.1f}" for stage, value in memory), flush=True)
		if self.device.type == "cuda":  # (the allocator's live tensors and its reserve, and what the card has left)
			free, total = torch.cuda.mem_get_info(self.device)
			print(f"campaign gpu memory (GB): allocated {torch.cuda.memory_allocated(self.device) / 2**30:.1f}  reserved {torch.cuda.memory_reserved(self.device) / 2**30:.1f}  card free {free / 2**30:.1f} of {total / 2**30:.1f}", flush=True)
		return True

	def _note_segments(self, decisions, surprises, players, hidden=None):
		"""config.credit_segments: each segment's surprise (ppo_update's, [n, 3]) into its evidence (labels.credit_evidence:
		by action type, against the player's other decisions, faded); each long segment's k one step of SEGMENT_RATE toward
		train.segment_shrinkage's max(0, 1 - 1/F), at most 1; the next update's credit uses them (one update's lag). The
		row's readings: the k's this update used, each segment's F and its types, and its F with the kinds merged.
		hidden [n] (each decision's game hid Nullkiller2's own choice), where the update has both kinds: the evidence kept
		by kind ("hidden/TYPE", "shown/TYPE"), so that F reads the types' spread within each kind and not the gap between
		the kinds (e5, 2026-11-12); one kind, or none known: as before. The whole evidence fades once an update."""
		out = {"segment_k5": self.segment_k[0], "segment_k7": self.segment_k[1]}
		ks = list(self.segment_k)
		hidden = None if hidden is None or len(hidden) != len(decisions) else np.asarray(hidden, dtype=bool)
		strata = [("hidden/", hidden), ("shown/", ~hidden)] if hidden is not None and hidden.any() and (~hidden).any() else [("", None)]
		for j, offset in enumerate(c.VALUE_OFFSETS):
			for i, (prefix, rows) in enumerate(strata):
				errors = surprises[:, j] if rows is None else np.where(rows, surprises[:, j], np.nan)
				self.segment_evidence[j] = credit_evidence(decisions, errors, players, self.segment_evidence[j], memory=CREDIT_MEMORY if i == 0 else 1.0, prefix=prefix)
			k, f, types = segment_shrinkage(self.segment_evidence[j], CREDIT_MIN_PLAYERS)
			out[f"segment_f{offset}"], out[f"segment_types{offset}"] = f, types
			out[f"segment_f{offset}_pooled"] = segment_shrinkage(pooled_evidence(self.segment_evidence[j]), CREDIT_MIN_PLAYERS)[1]
			if j > 0 and np.isfinite(k):
				ks[j - 1] = min(1.0, ks[j - 1] + SEGMENT_RATE * (k - ks[j - 1]))
		self.segment_k = (float(ks[0]), float(ks[1]))
		return out

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
			self.label_futures[id(game)] = self.labeler.submit(label_source(game), self.features_cpu)

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
		today, memory = self.trust_today(), self.config.trust_memory
		self.trust_ema = today if getattr(self, "trust_ema", None) is None else memory * self.trust_ema + (1 - memory) * today

	def trust_today(self):
		"""How far the value heads' fit would have the credit trust them, 0..1: their smoothed explained variance over the
		share prior's (the worse of V5 and V7: ppo_update's value_ev) over trust_full_ev."""
		if self.value_ev is None:
			return 0.0
		return float(min(1.0, max(0.0, min(self.value_ev) / self.config.trust_full_ev)))

	def trust(self):
		"""How far the credit trusts the value heads, 0..1 (the schedule credit's share baseline, labels.horizon_advantages):
		trust_today, since 2026-11-06 as a floored slow average (config.trust_floor, trust_memory; docs/optimizer_plan.md
		step 4): max(trust_floor, its average at trust_memory an update). As a feedback variable a re-fit cut it 0.40 ->
		0.25 and the credit went three quarters Monte Carlo, the noise slowing the re-fit's own signal. The credit check
		(credit_check) applies apart, in labels.horizon_advantages: a fit this good on the whole can still price a quick
		gain that costs by day 7, which shows as the step's error by action type."""
		if self.value_ev is None:
			return 0.0  # (nothing measured yet: no ground for the floor either, the credit leans on the real outcome)
		average = getattr(self, "trust_ema", None)
		value = average if self.config.trust_memory > 0 and average is not None else self.trust_today()
		return float(max(self.config.trust_floor, value))

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
			"gather_weight": sums.get("gather_weight", float("nan")),
			"army_on_heroes": reading(army_on_heroes, games, self.empire),
			"eliminated": reading(eliminated_share, games),
			"value_target_std": sums.get("value_target_std", float("nan")),
			"night_dropped": metrics.get("night_dropped", float("nan")),
			"valve_r": self._valve_reading(metrics.get("value_fresh", float("nan"))),
			"valve_s": sums.get("valve_s", float("nan")),
			"kl_target": sums.get("kl_target", float("nan")),
			# (config.nk2_choice_games: the games that hid Nullkiller2's own choice, and the rest: how often ours was its
			# choice there, the advantage head's fit, players out of their game)
			**choice_readings(games, metrics),
			**{name: metrics.get(name, float("nan")) for name in ("adv_loss", "adv_ev", "adv_ev_followed", "adv_ev_deviating", "adv_followed", "adv_raw_sd", "adv_kl", "adv_read")},
			**{name: sums.get(name, float("nan")) for name in ("segment_k5", "segment_k7", "segment_f2", "segment_f5", "segment_f7", "segment_types2", "segment_types5", "segment_types7", "segment_f2_pooled", "segment_f5_pooled", "segment_f7_pooled")},
			**{name: metrics.get(name, float("nan")) for name in ("segment_credit_std", "segment_full_std")},
			**{name: metrics.get(name, float("nan")) for name in ("nkp_loss", "nkp_auc", "nkp_positives", *(f"nkp_ev_{form}" for form in ("preference_all", "preference_hero", "soon", "later", "rank")), "nkp_ce", "nkp_ce_uniform", "nkp_top1_best", "nkp_top1_choice")},
			"action_price_mean": metrics.get("schedule_action_price_mean", float("nan")),
			"action_price_share": metrics.get("schedule_action_price_share", float("nan")),
			"end_first": reading(end_first, [d for game, _ in games for d in game.decisions]),
			"label_mean": sums["label_sum"] / max(n, 1),
			"price_mean": sums["price_sum"] / max(n, 1),
			"policy_loss": metrics.get("policy", float("nan")),
			"value_loss": metrics.get("value", float("nan")),
			"value_loss_weighted": metrics.get("value_weighted", float("nan")),  # (config.head_scale's weights: what the update takes)
			"value_loss_fresh": metrics.get("value_fresh", float("nan")),  # train.ppo_update: before the update, above the targets' entropy
			"entropy": metrics.get("entropy", float("nan")),
			"below_floor": metrics.get("below_floor", float("nan")),
			"kl": metrics.get("kl", float("nan")),
			"teacher_kl": metrics.get("teacher_kl", float("nan")),  # (KL(teacher || model) a decision; 0 without a teacher)
			"teacher_weight": metrics.get("loss_weight_teacher", self.teacher_weight),  # (the update's: the loss shares' when on)
			# Nullkiller2's advice (config.nk2_advice): the share of turn and exchange decisions it advised, how often our
			# choice was its, and the cross-entropy toward it
			"advised": sum(d.advice is not None for d in decisions) / max(1, sum(d.kind in (c.DecisionKind.TURN, c.DecisionKind.EXCHANGE) for d in decisions)),
			"teacher_agree": float(np.mean([agrees(d) for d in decisions if d.advice is not None])) if any(d.advice is not None for d in decisions) else float("nan"),
			"teacher_ce": metrics.get("teacher_ce", float("nan")),
			"teacher_outside": metrics.get("teacher_outside", float("nan")),
			"last_kl": metrics["last_kl"],
			"clip_fraction": metrics.get("clip_fraction", float("nan")),
			"grad_norm": metrics.get("grad_norm", float("nan")),
			"steps": metrics["steps"],
			"learning_rate": metrics["learning_rate"],
			**{f"signal_{name}": (metrics.get("signal_fraction") or {}).get(name, float("nan")) for name in ("all", *GRAD_GROUPS)},
			**{name: metrics.get(name, float("nan")) for name in ("signal_body", "signal_heads", "ratio_body", "ratio_heads")},
			"lr_scale": self.lr_scale,
			**{f"value_ev_d{k}": metrics[f"value_ev_d{k}"] for k in c.VALUE_OFFSETS},
			**{f"head_ev_{kind}{k}": self.head_ev.get(f"{kind.upper()}{k}", float("nan")) for kind in "vg" for k in c.VALUE_OFFSETS},
			"head_ev_vend": self.head_ev.get("Vend", float("nan")),
			"no_effect": sum(1 for d in decisions if d.flags & NO_EFFECT) / max(n, 1),
			"ai_errors": sum(s.get("errors", 0) for s in done),
			"failed_actions": sum(s.get("failed_actions", 0) for s in done),
			# (bundle 60's loop guard: actions dropped for the day for bringing back a state of today's, a game, by type; it
			# should fall as the policy learns out of the cycles; and probes of an unknown entrance that found no way)
			"loops_cut": sum(s.get("loops_cut", 0) for s in done) / max(len(games), 1),
			"loops_cut_move": sum((s.get("loops_cut_types") or {}).get(str(int(c.ActionType.MOVE)), 0) for s in done) / max(len(games), 1),
			"loops_cut_multi": sum((s.get("loops_cut_types") or {}).get(str(int(c.ActionType.MULTI)), 0) for s in done) / max(len(games), 1),
			"probes_no_way": sum(s.get("probes_no_way", 0) for s in done) / max(len(games), 1),
			# (e5, 2026-11-11: loops_cut is the sum over a game's players, a mean over the update's games; this one is per
			# player-day, comparable across player counts and game lengths: the cuts over the players' days played)
			"loops_per_player_day": sum(s.get("loops_cut", 0) for s in done) / max(sum(len(game.ledgers[min(game.ledgers)]["players"]) * (max(game.ledgers) - 1) for game, _ in games if game.ledgers), 1),
			"game_failures": self.stats["game_failures"],
			"game_maps": sum(1 for _, info in games if not info.map_name.startswith("Maps/alpha_random/")),
			"dropped_games": self.stats["dropped_games"],
			"game_seconds": mean([info.seconds for _, info in games]),
			"record_mb": sum(game.record_bytes for game, _ in games) / 2**20,
			"seconds": seconds,
			"gpu_mb": metrics.get("gpu_mb", float("nan")),  # the update's peak on the GPU (train.gpu_held)
			"label_staleness": sums.get("label_staleness", 0),  # the oldest precomputed labels used, in model versions (labeler.py)
			**{f"share_{t}": types[t] / max(len(turn), 1) for t in TYPES},
			**{f"failure_{k}": v for k, v in failure_modes(decisions, moves_limit=3000).items()},  # dumb decisions (failures.py; the meetings' over ~3k moves)
			"kinds": " ".join(f"{k}:{n}" for k, n in sorted(Counter(c.DecisionKind(d.kind).name for d in decisions).items())),
			"no_effect_types": " ".join(f"{t}:{n}" for t, n in sorted(Counter(c.ActionType(d.action_type).name for d in decisions if d.flags & NO_EFFECT).items())),
			**{name: metrics.get(name, float("nan")) for name in (*FORESIGHT_COLUMNS, "grad_shared_main", "grad_shared_foresight", "grad_shared_cosine", "grad_dead_share", *GRADIENT_COLUMNS) if name != "teacher_set_agree"},
			"teacher_set_agree": sums["set_agree"] / sums["set_advised"] if sums.get("set_advised") else float("nan"),  # (ours one of the teacher's entries: its advice's or a set's)
		}
		with self.progress.open("a", newline="") as f:
			csv.writer(f).writerow([round(row[k], 6) if isinstance(row[k], float) else row[k] for k in COLUMNS])
		if row["value_loss_fresh"] == row["value_loss_fresh"]:  # the eval tab's (battle_model/train.py VALUE_LOSS_TAB)
			self.writer.add_scalar("eval/campaign_value_loss", row["value_loss_fresh"], self.version)
		self._judge_value(row["value_loss_fresh"])
		if metrics.get("value_unexplained", float("nan")) == metrics.get("value_unexplained", float("nan")):  # (train.ppo_update: not a column)
			self.writer.add_scalar("eval/campaign_value_unexplained", metrics["value_unexplained"], self.version)
		# The privileged critic (config.privileged_critic): its fit beside the critic's, how large its corrections have
		# grown, how many rivals' states a decision had, its loss, each head's fresh excess with it
		if metrics.get("value_unexplained_privileged", float("nan")) == metrics.get("value_unexplained_privileged", float("nan")):
			self.writer.add_scalar("eval/campaign_value_unexplained_privileged", metrics["value_unexplained_privileged"], self.version)
		for name in ("privileged_correction_rms", "privileged_rivals", "privileged_present", "value_privileged"):
			if metrics.get(name, float("nan")) == metrics.get(name, float("nan")):
				self.writer.add_scalar(f"campaign/{name}", metrics[name], self.version)
		for name, values in (metrics.get("privileged_head_fresh") or {}).items():
			for k, value in enumerate(values):
				head = "Vend" if name == "end" else f"{'V' if name == 'share' else 'G'}{c.VALUE_OFFSETS[k]}"
				if value == value:
					self.writer.add_scalar(f"campaign_heads/fresh_privileged_{head}", value, self.version)
		for name in ("value_corr", "value_spread", "label_spread", "gpu_anchor_mb", "update_seconds", "prepass_seconds", "wait_seconds", "sync_seconds", "cpu_share", "stopped_pass", "signal_ratio", "attributed_taken_share", "attributed_taken_mean", "attributed_own_std", "attributed_taken_std", "forward_seconds", "losses_seconds", "backward_seconds", "optimizer_seconds",
				"value_learning_rate", "value_readout_rms", *(f"grad_norm_{g}" for g in GRAD_GROUPS), *(f"clip_bound_{g}" for g in GRAD_GROUPS), "kl_first", "kl_growth"):
			if metrics.get(name, float("nan")) == metrics.get(name, float("nan")):
				self.writer.add_scalar(f"campaign/{name}", metrics[name], self.version)
		# config.step_credit's readings: the share forecast's trust for its step, the growth step's spread, each action
		# type's mean own step (train.step_report: what the step says of the type before the week's outcome corrects it)
		for name, value in metrics.items():
			if name.startswith(("step_", "schedule_step_")) and isinstance(value, float) and value == value:
				self.writer.add_scalar(f"campaign_step/{name.replace('schedule_step_', '').replace('step_', '')}", value, self.version)
		# docs/optimizer_plan.md's readings: the trust before its floor and average, the share of each group's gradient that
		# is signal (train.signal_fraction), each value head's fresh excess loss and its weight (config.head_scale)
		self.writer.add_scalar("campaign/trust_today", self.trust_today(), self.version)
		self.writer.add_scalar("campaign/clipped_by_group", float(bool(metrics.get("clipped_by_group"))), self.version)
		for group, fraction in (metrics.get("signal_fraction") or {}).items():
			self.writer.add_scalar("campaign/signal_fraction" if group == "all" else f"campaign_signal/{group}", fraction, self.version)
		for table, prefix in ((metrics.get("head_fresh") or {}, "fresh"), (metrics.get("head_weights") or {}, "weight"), (metrics.get("head_residual") or {}, "residual")):
			for name, values in table.items():
				for k, value in enumerate(values):
					head = "Vend" if name == "end" else f"{'V' if name == 'share' else 'G'}{c.VALUE_OFFSETS[k]}"
					if value == value:
						self.writer.add_scalar(f"campaign_heads/{prefix}_{head}", value, self.version)
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
		privileged_note = f", privileged {metrics['value_unexplained_privileged']:.2f}" if "value_unexplained_privileged" in metrics else ""
		print(
			f"campaign {self.version:4d}  games {row['games']}  decisions {n}  days {row['days']:.1f}  Φ at the end {row['phi_end']:.0f}  heroes {row['heroes']:.2f}"
			# (pooled over ~10 updates like the growth and end heads': this update's alone swings with its few games' spread,
			# -0.65 at a correlation of +0.22 and a spread as wide as the outcomes', 2026-10-30, while the pooled rose)
			"  value ev " + "/".join(f"{row[f'head_ev_v{k}']:+.2f}" for k in c.VALUE_OFFSETS)
			+ f" (this update {row[f'value_ev_d{c.VALUE_OFFSETS[-1]}']:+.2f}: corr {metrics.get('value_corr', float('nan')):+.2f}, spread {metrics.get('value_spread', float('nan')):.3f} vs {metrics.get('label_spread', float('nan')):.3f}, unexplained {metrics.get('value_unexplained', float('nan')):.2f}{privileged_note})"
			+ "  growth ev " + "/".join(f"{row[f'head_ev_g{k}']:+.2f}" for k in c.VALUE_OFFSETS) + f"  end ev {row['head_ev_vend']:+.2f} (credit {row['end_credit']:.2f})  trust {row['trust']:.2f} (today {self.trust_today():.2f})"
			+ f"  kl {row['last_kl']:.4f} (target {metrics.get('kl_target', float('nan')):.4f})" + (f"  teacher kl {row['teacher_kl']:.3f} (weight {row['teacher_weight']:.2f})" if self.teacher is not None else "") + (f"  advice {row['advised']:.0%} of turns and exchanges, ours its {row['teacher_agree']:.0%}, teacher ce {row['teacher_ce']:.3f} (weight {row['teacher_weight']:.2f})" if self.config.nk2_advice > 0 else "") + (f"  action price {row['action_price_mean']:.2e} a decision ({row['action_price_share']:.2%} of the 7-day credit's spread), a day begun with END {row['end_first']:.3f}" if self.config.schedule_action_price > 0 else "") + (f"  credit by segments: k5 {row['segment_k5']:.3f} k7 {row['segment_k7']:.3f}, F {row['segment_f2']:.2f}/{row['segment_f5']:.2f}/{row['segment_f7']:.2f} over {row['segment_types2']}/{row['segment_types5']}/{row['segment_types7']} types (0-2/2-5/5-7 days), the credit's sd {row['segment_credit_std']:.4f} against the 7-day credit's {row['segment_full_std']:.4f}" if getattr(self.config, "credit_segments", False) else "") + (f"  its choice hidden in {row['choice_hidden']:.0%} of the games: ours its advice {row['teacher_agree_shown']:.0%} shown, {row['teacher_agree_hidden']:.0%} hidden; the head's ev {row['adv_ev_shown']:+.3f} shown, {row['adv_ev_hidden']:+.3f} hidden, the advantage's sd {row['adv_raw_sd_shown']:.3g} shown, {row['adv_raw_sd_hidden']:.3g} hidden; idle movement {row['idle_movement_shown']:.3f} shown, {row['idle_movement_hidden']:.3f} hidden; a day begun with END {row['end_first_shown']:.3f} shown, {row['end_first_hidden']:.3f} hidden; eliminated {row['eliminated_shown']:.3f} shown, {row['eliminated_hidden']:.3f} hidden" if getattr(self.config, "nk2_choice_games", 1.0) < 1.0 else "") + (f"  advantage head: loss {row['adv_loss']:.4f}, ev {row['adv_ev']:+.3f} (its own choice {row['adv_ev_followed']:+.3f} at {row['adv_followed']:.0%}, leaving it {row['adv_ev_deviating']:+.3f}), the advantage's sd {row['adv_raw_sd']:.3g}" + (f"; read by our policy at {float(self.model.features.adv_gain_weight().detach()):.3f} a unit: |term| {row['adv_read']:.4f}, its KL {row['adv_kl']:.5f} (shown {row['adv_kl_shown']:.5f}, hidden {row['adv_kl_hidden']:.5f})" if getattr(self.config, "adv_gain", 0.0) > 0 else "") if self.config.adv_weight > 0 else "") + (f"  planner prediction: loss {row['nkp_loss']:.4f}, AUC {row['nkp_auc']:.3f} (tasks at {row['nkp_positives']:.3f} of the entries), ev " + "/".join(f"{row[f'nkp_ev_{form}']:+.2f}" for form in ("preference_all", "preference_hero", "soon", "later", "rank")) + f", plan ce {row['nkp_ce']:.3f} against the uniform's {row['nkp_ce_uniform']:.3f}, top-1 {row['nkp_top1_best']:.3f} its best, {row['nkp_top1_choice']:.3f} its own choice" if getattr(self.config, "nk_predict_weight", 0.0) > 0 else "") + f"  signal {metrics.get('signal_ratio', float('nan')):.3f} (fraction {(metrics.get('signal_fraction') or {}).get('all', float('nan')):.3f})  steps {row['steps']}" + (f" (stopped in pass {metrics['stopped_pass']})" if metrics.get("stopped_pass") else "") + f"  {seconds:.1f}s (update {metrics.get('update_seconds', float('nan')):.0f}s: prepass {metrics.get('prepass_seconds', float('nan')):.0f}, decode waits {metrics.get('wait_seconds', float('nan')):.0f}, gpu syncs {metrics.get('sync_seconds', float('nan')):.0f}, cpu {metrics.get('cpu_share', float('nan')):.0%}; launching forward {metrics.get('forward_seconds', float('nan')):.0f}, losses {metrics.get('losses_seconds', float('nan')):.0f}, backward {metrics.get('backward_seconds', float('nan')):.0f}, optimizer {metrics.get('optimizer_seconds', float('nan')):.0f}" + (f", every hero slot in {metrics['full_windows']} steps" if metrics.get("full_windows") else "") + (", window budget refused" if metrics.get("window_budget_ok") is False else "") + f")  gpu {row['gpu_mb']:.0f} MB (anchor {metrics.get('gpu_anchor_mb', float('nan')):.0f})  labels {row['label_staleness']} old",
			flush=True,
		)

	def _judge_value(self, reading):
		"""The value-loss guard (config.value_jump_stop, 0: off; the user, 2026-11-10): an update's fresh value loss into
		the readings; once this process has VALUE_JUMP_WINDOW of its own (a restart's first updates are not judged by the
		readings from before it: after a rollback those are the abandoned updates'), value_jump of them at or above
		the factor holds the updates (step) and says so: one FAILURE line, the file `held` beside progress.csv, the
		TensorBoard series campaign/value_jump every update. An error here is a FAILURE line, never the run's end."""
		try:
			factor = float(getattr(self.config, "value_jump_stop", 0.0) or 0.0)
			if not factor or reading != reading:
				return
			self.value_readings = (self.value_readings + [float(reading)])[-(VALUE_JUMP_MEMORY + VALUE_JUMP_WINDOW) :]
			self.value_own += 1
			ratio, now, level = value_jump(self.value_readings)
			if ratio == ratio:
				self.writer.add_scalar("campaign/value_jump", ratio, self.version)
			if self.held is not None or self.value_own < VALUE_JUMP_WINDOW or not ratio >= factor:
				return
			self.held = f"the fresh value loss of the last {VALUE_JUMP_WINDOW} updates, median {now:.5f}, is {ratio:.1f} times its calm level {level:.5f}; config.value_jump_stop {factor:g}"
			print(f"campaign: FAILURE: VALUE LOSS JUMP at version {self.version}: {self.held}. The campaign's updates are HELD from here: its model stays as it is, its games are dropped, until a restart", flush=True)
			(self.out / "held").write_text(f"version {self.version}: {self.held}\n")
		except Exception as error:
			print(f"campaign: FAILURE: the value-loss guard failed ({type(error).__name__}: {str(error)[:200]}): this update is not judged", flush=True)

	def save(self, path):
		with self.model_lock:
			self._save(path)

	def _save(self, path):
		from battle_model.train import save_atomically  # (whole or not at all: the run resumes from it)

		state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
		optimizer = {} if self.worker else {"optimizer": self.optimizer.state_dict()}  # (the process's is in checkpoint_latest.pt)
		save_atomically({"model": state, **optimizer, "version": self.version, "lr_scale": self.lr_scale, "teacher_weight": self.teacher_weight, "games_used": self.games_used, "value_ev": self.value_ev, "credit_evidence": self.credit_evidence, "end_evidence": getattr(self, "end_evidence", {}), "segment_evidence": getattr(self, "segment_evidence", [{}, {}, {}]), "segment_k": getattr(self, "segment_k", (0.0, 0.0)), "head_ev": self.head_ev, "value_pool": self.value_pool, "head_pool": self.head_pool, "head_var": getattr(self, "head_var", None), "head_scale_updates": getattr(self, "head_scale_updates", 0), "loss_scale": getattr(self, "loss_scale", None), "teacher_share": getattr(self, "teacher_share", None), "trust_ema": self.trust_ema, "head_residual": self.head_residual, "head_var_slow": self.head_var_slow, "head_residual_updates": self.head_residual_updates, "grad_clips": self.grad_clips, "clip_samples": self.clip_samples, "gather_ramp_start": self.gather_ramp_start, "valve": self.valve, "model_kwargs": self.kwargs}, path)

	# ---- the loss shares (config.loss_shares) ----

	def loss_shares(self):
		"""Each term's share of the shared layers' gradient: the teacher's by how good we are (teacher_share), the rest
		policy : value : foresight 4 : 3 : 1."""
		teacher = getattr(self, "teacher_share", None) if getattr(self, "teacher_share", None) is not None else min(TEACHER_SHARE_MAX, self.teacher_weight / (1.0 + self.teacher_weight))
		if not getattr(self.config, "nk2_teacher", True):
			teacher = 0.0
		rest = 1.0 - teacher
		return {"policy": rest * 4 / 8, "value": rest * 3 / 8, "teacher": teacher, "foresight": rest / 8}

	def loss_weights(self):
		"""The terms' weights for the next update (ppo_update's loss_weights and teacher_weight): the loss shares' (their
		start: as without them, the teacher at its weight), else the teacher's weight alone; kl_teacher the teacher's share
		against the policy's (the KL target's)."""
		taught = getattr(self.config, "nk2_teacher", True)  # (False: the teacher's loss off, config.nk2_teacher)
		if not getattr(self.config, "loss_shares", False):
			return {"value": 1.0, "foresight": 1.0, "teacher": self.teacher_weight if taught else 0.0}
		if getattr(self, "loss_scale", None) is None:
			self.loss_scale = {"log_norm": {}, "weight": {"value": 1.0, "teacher": self.teacher_weight, "foresight": 1.0}}
		shares = self.loss_shares()
		if not taught:  # (the others' weights as they stand: the frozen ones, or the balancer's with the teacher's share 0)
			return {**self.loss_scale["weight"], "teacher": 0.0, "kl_teacher": 0.0}
		return {**self.loss_scale["weight"], "kl_teacher": shares["teacher"] / shares["policy"]}

	def _head_scale(self):
		"""ppo_update's head_scale (config.head_scale): the heads' running target variances and the weights' ramp; with
		head_scale_by "residual" also their running fresh excess, the slow variances that scale it, and how far the
		residual weights have ramped in (HEAD_SCALE_RAMP updates from the first)."""
		scale = {"var": getattr(self, "head_var", None), "ramp": min(1.0, getattr(self, "head_scale_updates", 0) / HEAD_SCALE_RAMP)}
		if self.config.head_scale_by == "residual":
			scale.update(residual=self.head_residual, var_slow=self.head_var_slow, residual_ramp=min(1.0, self.head_residual_updates / HEAD_SCALE_RAMP))
		return scale

	def _calibrate_clips(self, metrics):
		"""(config.grad_clip_by_group) The update's steps' gradient norms (ppo_update's grad_norms) into the calibration
		until it holds grad_clip_calibration steps; then each group's clip is its median and the scale the total's
		(train.calibrated_clips), kept for the run (re-measured at a deploy only: no controller)."""
		norms = metrics.pop("grad_norms", None) or []
		if not self.config.grad_clip_by_group or self.grad_clips is not None:
			return
		self.clip_samples = self.clip_samples + [[float(x) for x in step] for step in norms]
		if len(self.clip_samples) >= self.config.grad_clip_calibration:
			self.grad_clips = {**calibrated_clips(self.clip_samples, GRAD_GROUPS), "version": self.version}
			print(f"campaign: the gradient clipped by group from the next update, at the medians of {len(self.clip_samples)} steps' norms: " + "  ".join(f"{name} {self.grad_clips[name]:.4g}" for name in GRAD_GROUPS) + f"; all then x {self.config.grad_clip} / {self.grad_clips['typical']:.4g} (the total's median)", flush=True)

	def gather_weight_now(self):
		"""GATHER's label weight for the update to come (config.nk2_gather_ramp): nk2_gather_weight x min(1, updates since
		the ramp's start / its length), the start this version at its first update; nk2_gather_weight itself without a
		ramp (or with no weight)."""
		target, ramp = self.config.nk2_gather_weight, getattr(self.config, "nk2_gather_ramp", 0)
		if ramp <= 0 or target <= 0:
			return target
		if self.gather_ramp_start is None:
			self.gather_ramp_start = self.version
			print(f"campaign: GATHER's label ramped from version {self.version}: {target} over {ramp} updates", flush=True)
		return target * min(1.0, (self.version + 1 - self.gather_ramp_start) / ramp)

	def valve_now(self):
		"""(s, k) for the update to come (config.value_valve, kl_valve; 1 for a valve off): the valves' state, started at the
		first update with one on from progress.csv's fresh readings (valve_reference), so a deploy during a jump does not
		take the jump for the level."""
		if not (getattr(self.config, "value_valve", False) or getattr(self.config, "kl_valve", False)):
			return 1.0, 1.0
		if self.valve is None:
			self.valve = valve_reference(fresh_readings(self.progress, keep=VALVE_HORIZON))
			if self.valve is not None:
				print(f"campaign: the value valves from version {self.version}: r_v's reference {self.valve['reference']:.5f} (the median of progress.csv's last {VALVE_HORIZON} fresh readings)", flush=True)
		if self.valve is None:
			return 1.0, 1.0
		return (self.valve["s"] if self.config.value_valve else 1.0), (self.valve["k"] if self.config.kl_valve else 1.0)

	def _valve_reading(self, reading):
		"""The update's fresh value loss into the valves, r_v after it (nan with the valves off) for the log row."""
		self._note_valve(reading)
		return self.valve["r"] if self.valve is not None else float("nan")

	def _note_valve(self, reading):
		"""An update's fresh value loss into the valves (valve_step); an error is a FAILURE line and the valves open."""
		if self.valve is None:
			return
		try:
			self.valve = valve_step(self.valve, reading)
		except Exception as error:
			print(f"campaign: FAILURE: the value valves failed ({type(error).__name__}: {str(error)[:200]}): open from here", flush=True)
			self.valve = None

	def _note_gradients(self, metrics):
		"""After an update: each term's gradient on the shared layers as it entered the loss (grad_shared_*: its norm at
		weight 1, grad_raw_*, times its weight) into metrics; with the loss shares, each term's log norm into its running
		mean and each weight a step toward the one that gives its term its share against the policy's (GradNorm's balance:
		weight = share / share_policy x norm_policy / norm over the running means), at most LOSS_SCALE_STEP times."""
		weights = self.loss_weights()
		scale = {"policy": 1.0, "entropy": 1.0, "value": weights["value"], "teacher": weights["teacher"], "foresight": weights["foresight"], "plan": weights["foresight"]}
		for term, weight in scale.items():
			metrics[f"grad_shared_{term}"] = metrics.get(f"grad_raw_{term}", float("nan")) * weight
			for label in ("night_gain", "danger_gain"):
				metrics[f"push_{label}_{term}"] = -metrics.get(f"grad_raw_{label}_{term}", float("nan")) * weight
		metrics.update({f"loss_weight_{term}": weights[term] for term in ("value", "teacher", "foresight")}, teacher_share=self.loss_shares()["teacher"])
		if not getattr(self.config, "loss_shares", False):
			return
		state = self.loss_scale
		for term in LOSS_TERMS:
			norm = metrics.get(f"grad_raw_{term}", float("nan"))
			if np.isfinite(norm) and norm > 0:
				old = state["log_norm"].get(term)
				state["log_norm"][term] = math.log(norm) if old is None else LOSS_SCALE_MEMORY * old + (1 - LOSS_SCALE_MEMORY) * math.log(norm)
		shares, norms = self.loss_shares(), state["log_norm"]
		frozen = bool(self.config.loss_shares_freeze) and self.version >= self.config.loss_shares_freeze  # (step 3: measured, not moved)
		if "policy" in norms and not frozen:
			for term in ("value", "teacher", "foresight"):
				if term in norms:
					target = shares[term] / shares["policy"] * math.exp(norms["policy"] - norms[term])
					weight = state["weight"][term]
					if term == "value":  # (a floor: between its share and VALUE_SHARE_CAP times it, the weight holds)
						target = float(np.clip(weight, target, VALUE_SHARE_CAP * target))
					state["weight"][term] = float(np.clip(np.clip(target, weight / LOSS_SCALE_STEP, weight * LOSS_SCALE_STEP), 1e-4, 1e6))
		now = {term: metrics[f"grad_shared_{term}"] for term in LOSS_TERMS}
		total = sum(v for v in now.values() if np.isfinite(v))
		print("campaign loss shares (target / this update's): " + "  ".join(f"{term} {shares[term]:.2f}/{now[term] / total if total > 0 and np.isfinite(now[term]) else float('nan'):.2f}" for term in LOSS_TERMS) + "  weights next: " + "  ".join(f"{term} {state['weight'][term]:.3g}" for term in ("value", "teacher", "foresight")) + (f" (frozen since v{self.config.loss_shares_freeze})" if frozen else ""), flush=True)

	# ---- evaluation ----

	def eval_due(self):
		if not self.config.eval_minutes or not self.config.eval_games or self.evaluating():
			return False
		if (self.out.parent / "drain").exists():  # (runtime/drain_restart.sh: the run restarts once its games end, an eval
			return False  # begun now would be cut there; the resumed run evaluates first, eval.csv's time past due)
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
			# The weights this eval scores (saved once it ended, they were the updates' since: the rollback to "v1288"
			# restored v1293's, 2026-11-06). _save: the lock is held
			self._save(self.out / f"checkpoint_{version:05d}.pt")
		try:
			row = evaluate.evaluate(self.config, self.server, self.battle, version=version, epoch=epoch, stop=self.stop, model="eval")
		finally:
			self.last_eval = time.monotonic()
		if row is None:  # (stopped with the run)
			return None
		if self.config.eval_size != evaluate.EVAL_SIZE:  # (another map size: its own file, the run's scores and teacher left alone)
			evaluate.log(row, self.out / f"eval_{self.config.eval_size}.csv", None)
			return row
		evaluate.log(row, self.out / "eval.csv", self.writer)
		score = evaluate.eval_score(row)  # (our own policy's: OWN's in a split evaluation)
		if self.config.teacher_mix == "score" and (self.teacher is not None or self.config.nk2_advice > 0) and np.isfinite(score):
			self.teacher_weight = teacher_weight_for(score)  # (config.teacher_mix: the teacher's share by our strength)
			self.teacher_share = teacher_share_for(score)  # (config.loss_shares: its share of the gradient)
			print(f"campaign: the teacher's weight {self.teacher_weight:.2f} (its share of the loss {self.teacher_weight / (1 + self.teacher_weight):.2f}) for an eval score of {score:.3f}", flush=True)
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


def privileged_rivals(game, fresh, first_row, empire=EmpireConfig(), rivals=None):
	"""The privileged critic's inputs (model_v2.PrivilegedCritic) for a game's fresh decisions, rows first_row.. of the
	update: for each, every other player alive as its day began (the ledger's snapshot opening the day) and that
	player's last fresh decision on a day before (its row; -1 none: before the day, so nothing in it follows this
	decision), and our true share's logit as the day began (labels.share_now, the change labels' anchor). {"rivals" [N, R]
	int64, "present" [N, R] (1: a rival alive in the game then), "share_logit" [N]}; R: model_v2.PRIVILEGED_RIVALS."""
	from .model_v2 import PRIVILEGED_RIVALS

	rivals = rivals or PRIVILEGED_RIVALS
	table, _, last, _ = tables(game, empire)
	kept = [d for d, keep in zip(game.decisions, fresh) if keep]
	days, rows = {}, {}
	for row, d in enumerate(kept, start=first_row):
		days.setdefault(d.color, []).append(d.day)
		rows.setdefault(d.color, []).append(row)
	ledger_days, living, anchors = sorted(game.ledgers), {}, {}

	def alive(day):  # (the colors in the game as the day began: its snapshot, else the latest before it)
		if day not in living:
			i = max(bisect.bisect_right(ledger_days, day) - 1, 0)
			snapshot = game.ledgers[ledger_days[i]] if ledger_days else {"players": []}
			living[day] = {COLORS[p["color"]] for p in snapshot["players"] if p.get("status", STATUS_INGAME) == STATUS_INGAME}
		return living[day]

	players = [color for color in COLORS if color in table]
	n = len(kept)
	out = {"rivals": np.full((n, rivals), -1, np.int64), "present": np.zeros((n, rivals), np.float32), "share_logit": np.zeros(n, np.float32)}
	for i, d in enumerate(kept):
		if (d.color, d.day) not in anchors:
			anchors[d.color, d.day] = share_logit(share_now(table, d.color, d.day, last))
		out["share_logit"][i] = anchors[d.color, d.day]
		now = alive(d.day)
		for slot, color in enumerate([other for other in players if other != d.color][:rivals]):
			if color not in now:
				continue
			out["present"][i, slot] = 1.0
			j = bisect.bisect_left(days.get(color, []), d.day) - 1  # (a player's decisions are in its order: days only grow)
			if j >= 0:
				out["rivals"][i, slot] = rows[color][j]
	return out


def label_source(game):
	"""What the labeler reads a game from: its spool file (the trainer keeps no inputs of a waiting game), else the game."""
	return getattr(game, "spool_path", None) or game


def strip_inputs(game):
	"""A waiting game's inputs out of the trainer's memory, when its spool file has them (restore_inputs takes them back
	when an update takes it): 9 waiting games held 1.16 GB of inputs (2026-11-01 17:20 CST, ~14 KB a decision), and the
	trainer grew from ~2 GB after each start to ~12 GB as up to ~30 waited."""
	if getattr(game, "spool_path", None):
		for d in game.decisions:
			d.inputs = None
		game.inputs_stripped = True


def restore_inputs(game):
	"""The game's inputs back from its spool file (strip_inputs); False when it cannot be read (noted)."""
	if not getattr(game, "inputs_stripped", False):
		return True
	try:
		with open(game.spool_path, "rb") as f:
			saved = pickle.load(f)[0]
	except Exception as error:  # noqa: BLE001
		print(f"campaign: FAILURE: a game's inputs not read back from {game.spool_path}: {str(error)[:200]}", flush=True)
		return False
	if len(saved.decisions) != len(game.decisions):
		print(f"campaign: FAILURE: {game.spool_path} has {len(saved.decisions)} decisions, the game {len(game.decisions)}", flush=True)
		return False
	for d, s in zip(game.decisions, saved.decisions):
		d.inputs = s.inputs
	game.inputs_stripped = False
	return True


def value_jump(readings, window=VALUE_JUMP_WINDOW, baseline=VALUE_JUMP_BASELINE, memory=VALUE_JUMP_MEMORY):
	"""How far the fresh value loss stands above its calm level (the value-loss guard): (ratio, now, level), now the
	median of the last `window` readings, level the lowest median of `baseline` readings in a row among the `memory`
	before them (the lowest: after a rollback the abandoned updates' readings are still in progress.csv and must not
	become the level; a median: one game's outcome can make a reading several times its neighbours'). nan with fewer
	than window + baseline readings or a level that is not above 0."""
	nan = float("nan")
	readings = [float(r) for r in readings if r == r]
	if len(readings) < window + baseline:
		return nan, nan, nan
	last, before = readings[-window:], readings[-window - memory : -window]
	level = min(float(np.median(before[i : i + baseline])) for i in range(len(before) - baseline + 1))
	now = float(np.median(last))
	return (now / level if level > 0 else nan), now, level


def fresh_readings(progress, keep=VALUE_JUMP_MEMORY + VALUE_JUMP_WINDOW, since=0):
	"""progress.csv's value_loss_fresh readings (the finite ones, the last `keep`): the value-loss guard's memory across
	a restart. Only the line the run is on: the rows after the last place the version went down in file order (a
	rollback's: before 2026-11-12 the abandoned line's rows were read as the guard's calm level and held the mixed line at
	v3261, deploy 68). since (config.value_jump_from): of those, only the rows of that version and later, the guard's
	memory after a hold somebody looked into and let go on. A file that cannot be read is no memory (the guard then waits
	for its own readings), never an error."""
	try:
		with open(progress, newline="") as f:
			rows = list(csv.DictReader(f))
		versions = [float(row.get("version") or 0) for row in rows]
		start = max((i for i in range(1, len(versions)) if versions[i] < versions[i - 1]), default=0)
		rows = [row for row in rows[start:] if not since or float(row.get("version") or 0) >= since]
		values = [row.get("value_loss_fresh") for row in rows]
		values = [float(v) for v in values if v not in (None, "", "nan")]
		return [v for v in values if np.isfinite(v)][-keep:]
	except Exception as error:
		print(f"campaign: FAILURE: the value-loss guard could not read {progress} ({type(error).__name__}: {str(error)[:120]}): it starts with no memory", flush=True)
		return []


def unspool(games):
	"""Removes the games' files from the learner's spool: trained on (in a saved checkpoint), or dropped on purpose."""
	for game in games:
		path = getattr(game, "spool_path", None)
		if path:
			Path(path).unlink(missing_ok=True)


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
	if action < 0:  # (a question's advice, bundle 60's shipyard YES: the option)
		return decision.action_type < 0 and decision.index == index
	return decision.action_type == action and (action <= c.ActionType.END or decision.index == index)


def teacher_share_for(score):
	"""The teacher's share of the shared layers' gradient for an evaluation score (config.loss_shares): teacher_weight_for's
	share of the loss, at most TEACHER_SHARE_MAX: 0.6 up to a score of 0.6, 0.2 at 1 (as good as the teacher)."""
	return float(min(TEACHER_SHARE_MAX, np.clip(1.2 - score, 0.05, 0.95)))


def set_agreement(decisions, types, indices, valid):
	"""How many of the decisions with an advice chose one of the teacher's entries: the advice's own (agrees: an end only
	its type, ours -1 and the advice's 0, which an entry's comparison never matched, 2026-11-10) or one of its set."""
	count = 0
	for i, d in enumerate(decisions):
		if d.advice is None:
			continue
		count += agrees(d) or any(valid[i] & (types[i] == d.action_type) & (indices[i] == d.index))
	return count


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
	set_advice_prior(model, config)
	if state.get("optimizer"):
		optimizer.load_state_dict(state["optimizer"])
		fit_moments(optimizer, model)
	use_fused(optimizer)
	lr_scale = state["lr_scale"]
	while (request := requests.get()) is not None:
		decisions, targets, returns, weights, about, advantages, growth, labels, end, foresight = request
		if hasattr(model, "value_valve"):  # (the learner's value valve for this update)
			model.value_valve.fill_(about.get("value_valve", 1.0))
		try:
			metrics, lr_scale = ppo_update(model, optimizer, decisions, targets, returns, replace(config, nk2_gather_weight=about.get("nk2_gather_weight", config.nk2_gather_weight), target_kl=about.get("target_kl", config.target_kl)), device=device, lr_scale=lr_scale, weights=weights, advantages=advantages, growth=growth, labels=labels, end=end, foresight=foresight)
			if device.type == "mps":
				torch.mps.empty_cache()  # (the battle learner's lesson: MPS keeps blocks of every size it saw)
			cpu = {key: value.detach().cpu() for key, value in model.state_dict().items()}
			from battle_model.train import save_atomically

			save_atomically({"model": cpu, "optimizer": optimizer.state_dict(), "lr_scale": lr_scale, "model_kwargs": kwargs, **about}, latest)
			results.put(("ok", cpu, metrics, lr_scale))
		except Exception:
			results.put(("error", traceback.format_exc(), None, lr_scale))


def reading(function, *args):
	"""A diagnostic's value for the update's row: nan when it fails (a reading never ends an update)."""
	try:
		return function(*args)
	except Exception as error:  # noqa: BLE001
		print(f"FAILURE: campaign reading {function.__name__} failed: {error!r}", flush=True)
		return float("nan")


def valve_reference(readings):
	"""The value valves' state from fresh value-loss readings (progress.csv's: their first update, or a run resumed from a
	checkpoint without them): r_v's reference the median of the last VALVE_HORIZON, its window the last VALVE_WINDOW, both
	valves open (s = k = 1); None with no reading."""
	readings = [float(r) for r in readings if r == r]
	if not readings:
		return None
	return {"reference": float(np.median(readings[-VALVE_HORIZON:])), "window": readings[-VALVE_WINDOW:], "s": 1.0, "k": 1.0, "r": float("nan")}


def valve_step(state, reading):
	"""The value valves after an update's fresh value loss (reading; nan: as they were): r_v = the median of the last
	VALVE_WINDOW readings over the reference as it stood, then the reference an EMA step toward the reading (a VALVE_HORIZON
	update horizon: a jump is measured, a level that stays is absorbed); s toward clip(1 / r_v, VALVE_S_MIN, 1) and k
	toward clip(1 / r_v, VALVE_K_MIN, 1), each at most VALVE_RATE times or over an update (Adam's second moments lag: a
	valve that snapped open would overshoot)."""
	if reading != reading:
		return state
	state = dict(state)
	window = (list(state["window"]) + [float(reading)])[-VALVE_WINDOW:]
	r = float(np.median(window)) / state["reference"] if state["reference"] > 0 else float("nan")
	if r == r:
		for key, floor in (("s", VALVE_S_MIN), ("k", VALVE_K_MIN)):
			target = min(1.0, max(floor, 1.0 / r))
			state[key] = min(state[key] * VALVE_RATE, max(state[key] / VALVE_RATE, target))
	state["reference"] += (float(reading) - state["reference"]) / VALVE_HORIZON
	state.update(window=window, r=r)
	return state


def army_on_heroes(games, empire):
	"""Our armies' worth on heroes against all of it (on heroes, in towns' garrisons, armies no hero leads: empire.army_worth
	at cost), over every player still in at each day's snapshot of the games; nan with none. The 2026-11-10 collapse:
	GATHER put the garrisons on heroes (half and more), which were then killed in the field."""
	from .empire import STATUS_INGAME, army_worth

	on, total = 0.0, 0.0
	for game, _ in games:
		for snapshot in game.ledgers.values():
			for p in snapshot["players"]:
				if p.get("status", STATUS_INGAME) != STATUS_INGAME:
					continue
				led = sum(army_worth(hero.get("army", []), empire) for hero in p.get("heroes", []))
				on += led
				total += led + sum(army_worth(town.get("garrison", []), empire) for town in p.get("towns", [])) + army_worth(p.get("armies", []), empire)
	return on / total if total > 0 else float("nan")


def choice_readings(games, metrics):
	"""The update's readings apart for the games that hid Nullkiller2's own choice (play.choice_hidden) and the rest:
	their share, how often our choice was its advice (agrees), the advantage head's explained variance (train.ppo_update's
	adv_ev_shown, adv_ev_hidden), players out of their game (eliminated_share); nan where a kind has none."""
	kinds = {True: [], False: []}
	for game, info in games:
		kinds[choice_hidden(game)].append((game, info))
	out = {"choice_hidden": len(kinds[True]) / len(games) if games else float("nan")}
	for name, hidden in (("shown", False), ("hidden", True)):
		advised = [d for game, _ in kinds[hidden] for d in game.decisions if d.advice is not None]
		out[f"teacher_agree_{name}"] = float(np.mean([agrees(d) for d in advised])) if advised else float("nan")
		out[f"adv_ev_{name}"] = metrics.get(f"adv_ev_{name}", float("nan"))
		out[f"adv_raw_sd_{name}"] = metrics.get(f"adv_raw_sd_{name}", float("nan"))
		for what in ("mean", "sd", "weight"):
			out[f"adv_{what}_{name}"] = metrics.get(f"adv_{what}_{name}", float("nan"))
		out[f"adv_kl_{name}"] = metrics.get(f"adv_kl_{name}", float("nan"))
		out[f"adv_read_{name}"] = metrics.get(f"adv_read_{name}", float("nan"))
		ours = [d for game, _ in kinds[hidden] for d in game.decisions]
		out[f"idle_movement_{name}"] = reading(idle_movement, ours) if ours else float("nan")
		out[f"end_first_{name}"] = reading(end_first, ours) if ours else float("nan")
		out[f"eliminated_{name}"] = reading(eliminated_share, kinds[hidden]) if kinds[hidden] else float("nan")
	return out


def idle_movement(decisions):
	"""failures.py's idle_movement over these decisions alone (of the movement our heroes have in a day, the share left
	when the turn ends); nan with none."""
	from .failures import NAMES as names, idle_counts

	counts = {name: [0, 0] for name in names}
	idle_counts(decisions, counts, 3000)
	hit, total = counts["idle_movement"]
	return hit / total if total else float("nan")


def end_first(decisions):
	"""The share of the days' first turn decisions (Decision.today 0) that ended the turn at once; nan with none."""
	first = [d for d in decisions if d.kind == c.DecisionKind.TURN and d.today == 0 and d.action_type >= 0]
	return sum(d.action_type == c.ActionType.END for d in first) / len(first) if first else float("nan")


def eliminated_share(games):
	"""The share of the games' players out of their game by its end (EPlayerStatus LOSER at its last snapshot, the final
	one if any); nan with none. Self-play's: 0.2% before the 2026-11-10 collapse, 6-8% in it."""
	from .empire import STATUS_LOST

	out, players = 0, 0
	for game, _ in games:
		snapshots = sorted([*game.ledgers.values(), *getattr(game, "final", [])], key=lambda s: (s["day"], bool(s.get("final"))))
		if snapshots:
			players += len(snapshots[-1]["players"])
			out += sum(1 for p in snapshots[-1]["players"] if p.get("status") == STATUS_LOST)
	return out / players if players else float("nan")


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
