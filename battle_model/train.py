"""Training loop: play battles against a fixed opponent, learn from them with PPO (or from a search, search_simulations), repeat.

	.venv/bin/python -m battle_model.train [--epochs N] [--run NAME] [--resume] [--set games_per_side=8 ppo.entropy_weight=0.02 ...]
	.venv/bin/python -m battle_model.train --follow NAME   # mirror a running run's progress.csv into TensorBoard
	.venv/bin/tensorboard --logdir runtime/train            # then open http://localhost:6006

Each epoch:
  1. the model, exported to ONNX, plays the epoch's battles in VCMI on the CPU (C++ battle AI), in a background
     thread (Producer) while the previous epoch trains: playing and training overlap (prefetch),
  2. play games_per_side battles per scenario side, all in parallel (we attack with sides[0], we defend
     with sides[1]), with VCMI's seed fixed per side, so only our own (sampled) choices differ between
     the battles of a side and between epochs. With opponent=self (self-play) our model plays both sides
     of games_per_side battles, half with each side's seed, and learns from both sides' records,
  3. turn the records into training examples (data.py) and update the model with PPO (losses.py), on the GPU
     when there is one (device),
  4. log both battles' rewards and outcomes to <run dir>/progress.csv and TensorBoard
     (<run dir>/tensorboard/<start time>: one folder per start, so a restarted run never mixes with old
     data a running TensorBoard still holds in memory). The records only ever exist in memory
     (play.RecordServer); keep_records also writes them to <run dir>/records for debugging.
  5. every eval_every epochs (and before the first), an eval round, not trained on: greedy battles (the most
     likely action, no sampling) against eval_opponent (BattleAI), one per side on each of eval_seeds fixed
     random VCMI seeds (the same in every round and every run). Next to each, the eval opponent's own result
     on the same seed and side (it plays both sides, once per run): vs_baseline = our reward minus its
     reward, seed by seed, the fairest measure of strength (the scenario favours the attacker: BattleAI
     never wins its defence against itself). Logged to <run dir>/eval.csv and under eval/ in TensorBoard.

Rewards are logged on the -1..1 scale (reward.to_scale: -1 = worst possible, +1 = best possible), the
raw reward next to them.

Warning signs, logged every epoch. The lr 3e-4 run learned to win the attack, then forgot it; these
showed it coming:
  signal_ratio     median |m| / sqrt(v) in Adam's state: near 0 the recent gradients cancel out, and
                   Adam's normalised steps are mostly noise. It halved around each collapse.
  weight_change    how far the weights moved this epoch. Adam keeps it ~constant even when nothing is
                   left to learn: that drift is what forgets.
  approx_kl, clip_fraction   how far each epoch's update moved the policy (PPO's usual checks)
  advantage_std, value_error per side: how much there is to learn, how wrong V is

The first test: the untrained model should lose in epoch 1, and the model should overfit this fixed
scenario and win by epoch 1000. If it needs more, something is probably wrong. Nothing here may depend
on the seeds being fixed: the same training has to work on real, ever-different battles.
"""
import argparse
import copy
import functools
import itertools
import csv
import json
import os
import random
import shutil
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from campaign_model.config import CampaignConfig

from . import generator, play, rules
from .data import collate, credit_check, make_dataloader, summarize, widened
from .distributions import log_prob
from .export import export, verify
from .device import sync, use_fused
from .losses import FORESIGHT_PARTS, AdvantageConfig, PPOConfig, foresight_loss, ppo_loss, search_loss, value_excess, weighted_mean
from .constants import N_ACTION_TYPES, ActionType
from .model import VALUE_PRIOR_GAIN, VALUE_PRIOR_KINDS, VALUE_PRIOR_STEP, build_model
from .trace import trace
from .reward import Outcome, RewardConfig, to_scale

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TrainConfig:
	scenario: str = str(ROOT / "battle_model" / "scenarios" / "castle_vs_inferno.json")
	# Random battles instead of `scenario`: every training battle is a new one drawn by battle_model/generator.py, and
	# the eval battles are a fixed set of eval_scenarios generated two-sided battles (hero vs hero, sieges), each
	# played from both sides by us and by the eval opponent (the eval opponent's own results: eval_baseline)
	generator: bool = False
	eval_scenarios: int = 10
	eval_neutral_scenarios: int = 0  # generator evals: also this many battles against wandering monsters, from both sides
	eval_hard_scenarios: int = 8  # generator evals: also this many hard cases (generator.HARD_KINDS in turn), both sides
	eval_real_scenarios: int = 0  # generator evals: also the first this many battles of real games (REAL_EVAL), both sides
	eval_castle_seeds: int = 0  # generator evals: also castle_vs_inferno (`scenario`) on this many EVAL_SEEDS: regressions
	eval_extra_opponents: tuple = ("MMAI",)  # also evaluated against these (eval_<name>.csv, TensorBoard eval_<name>/)
	eval_workers: int = 12  # battle processes of an eval (play_eval; the learner and the generated actors wait meanwhile)
	# Training in updates from many battles at once (actors.py) instead of epochs of games_per_side battles: an
	# "epoch" is then one update of decisions_per_update decisions, taken from the actors' buffer
	actors: bool = False
	generated_actors: int = 10  # generated battles played at once (on 14 cores: 16 starved the learner, 6 left it waiting)
	real_games: int = 0  # real games played at once (real_games.py); their battles are real_share of each update
	real_share: float = 0.5
	real_game_seconds: int = 900
	game_share: float = 0.125  # at most this share of an update from any one game
	max_record_age: int = 3  # model versions: older records are dropped
	buffer_decisions: int = 16000  # at most this many decisions wait in the buffer (the oldest go first)
	decisions_per_update: int = 3200  # ~60 battles: each gradient over many outcomes (800, ~15 battles, was the noisy v14-11h)
	# Time instead of epochs: stop after run_hours (0: after `epochs`), the linear learning rate following the time;
	# an eval every eval_minutes (0: every eval_every epochs); the rate ends at lr_final_fraction of learning_rate
	run_hours: float = 0.0
	eval_minutes: float = 0.0
	lr_final_fraction: float = 0.0
	# Generated battles: our current model against itself, against a past version of itself (a self-play blind spot that
	# only the current version shares cannot last), and against a fixed AI (league_ais; none: we train against ourselves)
	league: tuple = (0.85, 0.15, 0.0)
	league_ais: tuple = ("BattleAI",)
	inference: str = "server"  # actors: "server" (one process runs the model for every game, live weights) or "onnx" (actors.py)
	inference_device: str = "auto"  # the server's: "auto" = CUDA if there is one, else the CPU; or e.g. "cpu", "cuda"
	inference_threads: int = 1  # each server process's torch threads on the CPU (more threads barely help at small batches)
	inference_processes: int = 0  # server processes; 0: one on a GPU, a fifth of the cores on the CPU
	persistent_battles: bool = True  # each generated actor plays its battles in one VCMI process (play.BattleProcess)
	# Workers on other machines (remote.py, worker.py): they connect to this port (0: none), play with this machine's
	# inference servers over the network and send their records here; remote_token: their shared secret (empty: the
	# file runtime/remote-token, made here on the first run)
	remote_port: int = 0
	remote_token: str = ""
	run: str = "castle_vs_inferno"
	epochs: int = 1000
	opponent: str = "BattleAI"  # who we train against; "self" (play.SELF_PLAY): our model on both sides, learning from both
	ai: str = "AlphaBattleAI"  # play.Game.ai
	eval_every: int = 10  # epochs between eval rounds against eval_opponent; 0 = none
	eval_opponent: str = "BattleAI"
	eval_seeds: int = 16  # greedy eval battles per side, each on its own fixed random seed (EVAL_SEEDS)
	game_seeds: tuple = (1, 2)  # VCMI seed of the attack battles and of the defence battles: fixed, so the test is about learning, not luck
	games_per_side: int = 8  # battles per side per epoch, played in parallel; 1 (~40 decisions an update) was far too noisy
	# Give every battle its own random VCMI seed instead of game_seeds (dice, obstacles and the opponent's
	# choices differ): learning a strategy rather than one battle.
	random_seeds: bool = False
	learning_rate: float = 1e-4  # Adam; 3e-4 (the usual PPO default) learned and then forgot this scenario
	lr_schedule: str = "constant"  # "linear": decay to 0 at the last epoch, as standard PPO does; Adam's steps stay ~lr even when the gradients are only noise
	ppo_passes: int = 2  # optimisation passes over each epoch's decisions; PPO reuses its data a few times
	batch_size: int = 256  # minibatch of decisions: on the GPU 1.15 ms a decision at 512, 1.8 at 64 (2026-10-26)
	grad_clip: float = 0.5  # max gradient norm, the usual PPO value
	# KL control, so the step size does not hang on a hand-picked learning rate: an update stops its steps once its KL
	# (old against new policy) passes 1.5x target_kl; then the learning rate's scale is cut by 1.5, and raised by 1.5
	# after an update that took all its steps and ended under half of target_kl (within 0.1..lr_scale_max). The KL is the mean
	# of the last KL_WINDOW minibatches' (k3), nearest the update's end: a mean over all its steps counts the first ones,
	# where the policy has hardly moved, and kept raising the rate (to 7.5x within 11 updates, ending in 1-2-step updates);
	# one minibatch's is noise (2026-10-28 review: 15% of v15-window's updates stopped early, 99% of them with a mean KL
	# under half the stop line, and each stop cut the rate). 0 = off. With a fixed
	# rate, v15-base-lr2 (2e-4, 3200-decision updates) went from KL 0.008 to 0.03-0.09 as its policy sharpened, its
	# entropy rose and its evals fell.
	target_kl: float = 0.015
	# The user's rule (2026-10-26): stability over speed, never above learning_rate (1e-4: 2e-4 went unstable), so the
	# KL control only lowers the rate and raises it back up to where it started
	lr_scale_max: float = 1.0
	checkpoint_every: int = 0  # also keep checkpoint_<epoch>.pt every this many epochs; 0 = only checkpoint_latest.pt
	# The evals (and the checkpoints' "average") use a running average of the weights over updates (WeightAverage),
	# decaying by this per update (0.99: ~100 updates, ~1 h); 0: the evals use the trained weights. The battle policy's
	# gradient is noise-level (gradient_study.py, 2026-10-28: an update's direction no more alike than pure noise's), so
	# each update's step is mostly noise and the weights walk; their average cancels the walk and keeps what the signal
	# moves. eval_live: also evaluate the trained weights against eval_opponent (eval_live.csv) and log the paired
	# difference average - live on the same battles (eval/average_minus_live): the test of the average
	# Off since 2026-10-29: in joint2 the average scored below the trained weights in all 5 paired evals (pooled -0.0075
	# +- 0.0025) and idled shooters 2-3x as often (3315: 0.25/0.49 against 0.15): averaging the weights of a walk is not
	# the walk's average policy
	weight_average: float = 0.0
	eval_live: bool = True
	attempts: int = 3  # tries per epoch when a battle fails or stalls before the run stops
	keep_records: bool = False  # also write every battle record to <run dir>/records (debugging; ~0.3 MB a decision on disk)
	device: str = "auto"  # where torch trains: "auto" = the GPU (cuda, Apple mps) if there is one, else "cpu"; the CPU plays the battles
	# Play the next epoch's battles while this epoch trains (Producer): the battles are then one update older than the
	# model they train, which PPO handles (each record holds the probabilities of the model that played it)
	prefetch: bool = True
	# > 0: the training battles search every decision with this many simulations (Gumbel AlphaZero, C++) and the
	# policy learns the search's improved policy (losses.search_loss) instead of PPO's; eval stays greedy, no search
	search_simulations: int = 0
	model_architecture: str = "v3"  # "v3": model.BattleModel; "v2": model_v2.BattleModelV2; "v1": model_v1.BattleModelV1
	model_dim: int = 64
	model_layers: int = 2
	model_mask_features: bool = False  # v1 only: BattleModelV1(mask_features=...)
	model_chain_final_value: bool = False  # v1 only: BattleModelV1(chain_final_value=...)
	model_kill_rule: bool = True  # v1 only: BattleModelV1(kill_rule=...)
	seed: int = 0  # the model's initial weights
	# A new run that goes on from another run's checkpoint (its weights, optimizer, epoch numbering, KL control and
	# league snapshots; its eval baselines when the eval sets are the same), evaluated first
	init_from: str = ""
	reward: RewardConfig = field(default_factory=RewardConfig)
	advantage: AdvantageConfig = field(default_factory=AdvantageConfig)
	ppo: PPOConfig = field(default_factory=PPOConfig)
	# The campaign model trained alongside (campaign_model/learner.py; campaign.games > 0, with actors): its games'
	# battles are this run's real battles (real_share), its updates take turns with ours, its own optimizer
	campaign: CampaignConfig = field(default_factory=CampaignConfig)


def _parse_item(text):
	"""A tuple item of an override: an int, a float, else the text (e.g. an AI's name)."""
	for kind in (int, float):
		try:
			return kind(text)
		except ValueError:
			pass
	return text


def with_overrides(config, assignments):
	"""config with "name=value" assignments applied; "ppo.name", "advantage.name", "reward.name", "campaign.name" set
	the sub-configs."""
	for assignment in assignments:
		key, value = assignment.split("=", 1)
		target, name = key.split(".", 1) if "." in key else (None, key)
		owner = getattr(config, target) if target else config
		kind = {f.name: f.type for f in fields(owner)}[name]
		kind = {"int": int, "float": float, "str": str, "bool": lambda v: v.lower() in ("1", "true", "yes"), "tuple": lambda v: tuple(_parse_item(x) for x in v.split(",") if x)}.get(getattr(kind, "__name__", kind), kind)
		owner = replace(owner, **{name: kind(value)})
		config = replace(config, **{target: owner}) if target else owner
	return config


def machine_overrides(path):
	"""A machine config's settings (configs/machines/<machine>/<case>.toml: how many games, actors and servers this
	machine runs, never what is learned) as with_overrides assignments: top-level keys set TrainConfig fields, a table its
	sub-config ([campaign] -> campaign.NAME)."""
	import tomllib

	def text(value):
		if isinstance(value, bool):
			return str(value).lower()
		if isinstance(value, list):
			return ",".join(map(str, value))
		return str(value)

	assignments = []
	for key, value in tomllib.loads(Path(path).read_text()).items():
		if isinstance(value, dict):
			assignments += [f"{key}.{name}={text(item)}" for name, item in value.items()]
		else:
			assignments.append(f"{key}={text(value)}")
	return assignments


TRAIN_ROOT = ROOT / "runtime" / "train"


def run_dir(config, new=False):
	"""The folder of run config.run: <start time>-<name> (e.g. 20261023-1412-selfplay), so that sorting runs by name
	(TensorBoard, ls) sorts them by time. An existing run is the newest folder with that name (or exactly that
	folder name); new=True makes a new one."""
	if not new:
		if (TRAIN_ROOT / config.run).is_dir():
			return TRAIN_ROOT / config.run
		matches = sorted(TRAIN_ROOT.glob(f"????????-????-{config.run}"))
		if matches:
			return matches[-1]
	return TRAIN_ROOT / f"{time.strftime('%Y%m%d-%H%M')}-{config.run}"


# Action types whose share of each side's decisions is logged every epoch (actions_<side>/<type> in TensorBoard)
ACTION_KINDS = ("WAIT", "DEFEND", "MOVE", "MELEE", "SHOOT", "HERO_SPELL", "CREATURE_SPELL", "RETREAT", "SURRENDER")


COLUMNS = [
	"epoch", "attack_reward", "attack_outcome", "defence_reward", "defence_outcome", "mean_reward", "attack_raw_reward", "defence_raw_reward", "attack_win_rate", "defence_win_rate",
	"decisions", "rounds", "policy_loss", "value_loss", "value_loss_fresh", "entropy", "below_floor", "approx_kl", "clip_fraction", "signal_ratio", "weight_change",
	"attack_advantage_std", "defence_advantage_std", "attack_value_error", "defence_value_error", "seconds", "wait_seconds", "train_seconds", "data_seconds", "prepass_seconds", "steps_seconds", "publish_seconds",
] + [f"{side}_share_{kind.lower()}" for side in ("attack", "defence") for kind in ACTION_KINDS] + [
	f"{side}_{name}" for side in ("attack", "defence") for name in ("value_ev", "credit_own_lost", "credit_killed")
] + ["real_share", "games_in_update", "buffer_decisions", "record_age", "actor_failures", "dropped_stale", "dropped_overflow", "learning_rate", "ppo_steps", "update_kl", "outcome_share", "round_head_weight"] + [
	f"train_{name}" for name in ("escape_while_ahead", "shooter_not_shooting", "worthless_cast", "cast_skipped_on_loss")
] + [f"value_gain_{kind}" for kind in VALUE_PRIOR_KINDS] + [f"foresight_{k}" for k in FORESIGHT_PARTS] + [
	"grad_trunk_main", "grad_trunk_foresight", "grad_trunk_cosine", "grad_dead_share",
]  # progress.csv. Per side, over its games_per_side battles: mean reward (-1..1 scale), mean raw reward (reward
# units), win rate, outcome (its name for one battle, else counts like "WIN:3 LOSS:5"), mean advantage std and value
# error. decisions: all decisions of the epoch; rounds: attack + defence battle rounds, averaged over the games.

# progress.csv column -> TensorBoard tag
# The value loss in the eval tab too (the user, 2026-10-29: it should go down over time), though it is training's: every
# update's battles under the model as the update starts, before it trains on them, all the heads weighted as the loss
# weights them, above the labels' own entropy (losses.value_excess: 0 = V predicts every label exactly)
VALUE_LOSS_TAB = "eval/battle_value_loss"
TENSORBOARD_TAGS = {
	**{f"train_{name}": f"train_failures/{name}" for name in ("escape_while_ahead", "shooter_not_shooting", "worthless_cast", "cast_skipped_on_loss")},
	**{name: f"actors/{name}" for name in ("real_share", "games_in_update", "buffer_decisions", "record_age", "actor_failures", "dropped_stale", "dropped_overflow")},
	"learning_rate": "stability/learning_rate",
	"ppo_steps": "stability/ppo_steps",  # fewer than the passes' when the KL control stopped the update
	"update_kl": "stability/update_kl",
	"outcome_share": "stability/outcome_share",  # the window credit's share of the real outcome (by V's quality)
	"round_head_weight": "stability/round_head_weight",  # V2's share of what the policy maximizes (by V's quality)
	**{f"value_gain_{kind}": f"priors/value_gain_{kind}" for kind in VALUE_PRIOR_KINDS},  # each value prior's learned gain (model.ActionHeads.value_prior)
	**{f"foresight_{k}": f"foresight/{k}" for k in FORESIGHT_PARTS},  # model.Foresight's losses (BCE): falling = it learns
	# Gradient health (trunk_gradients): the trunk's gradient norm from the PPO losses and from the foresight's, how much
	# they agree (cosine; strongly negative = they fight), and the share of parameters no loss reaches
	**{name: f"gradients/{name[5:]}" for name in ("grad_trunk_main", "grad_trunk_foresight", "grad_trunk_cosine", "grad_dead_share")},
	**{f"{side}_share_{kind.lower()}": f"actions_{side}/{kind.lower()}" for side in ("attack", "defence") for kind in ACTION_KINDS},
	**{f"{side}_{name}": f"credit_{side}/{name}" for side in ("attack", "defence") for name in ("value_ev", "credit_own_lost", "credit_killed")},
	"mean_reward": "reward/mean", "attack_reward": "reward/attack", "defence_reward": "reward/defence",
	"attack_raw_reward": "raw_reward/attack", "defence_raw_reward": "raw_reward/defence",
	"attack_win_rate": "won/attack", "defence_win_rate": "won/defence",
	"policy_loss": "loss/policy", "value_loss": "loss/value", "value_loss_fresh": VALUE_LOSS_TAB, "entropy": "loss/entropy", "below_floor": "loss/entropy_below_floor",
	"signal_ratio": "stability/signal_ratio", "weight_change": "stability/weight_change", "approx_kl": "stability/approx_kl", "clip_fraction": "stability/clip_fraction",
	"attack_advantage_std": "advantage_std/attack", "defence_advantage_std": "advantage_std/defence",
	"attack_value_error": "value_error/attack", "defence_value_error": "value_error/defence",
	"decisions": "battle/decisions", "rounds": "battle/rounds", "seconds": "time/epoch_seconds",
	"wait_seconds": "time/waiting_for_battles_seconds", "train_seconds": "time/training_seconds",
	# training_seconds' parts: the update's decisions collated onto the device, the passes before the steps (facts, the
	# KL anchor, the fresh value loss), the PPO steps, the new weights published to the actors
	**{f"{part}_seconds": f"time/{part}_seconds" for part in ("data", "prepass", "steps", "publish")},
}


EVAL_COLUMNS = [
	"epoch", "total_score", "attack_win_rate", "defence_win_rate", "attack_reward", "defence_reward", "mean_reward", "attack_outcome", "defence_outcome",
	"attack_vs_baseline", "defence_vs_baseline", "mean_vs_baseline", "attack_baseline_reward", "defence_baseline_reward", "seconds",
	"attack_army_kept", "attack_army_killed", "defence_army_kept", "defence_army_killed",
	"attack_baseline_army_kept", "attack_baseline_army_killed", "defence_baseline_army_kept", "defence_baseline_army_killed",
	"escape_while_ahead", "shooter_not_shooting", "worthless_cast", "cast_skipped_on_loss",
	"score_generated", "score_neutrals", "score_real", "score_castle", "score_hard",  # one a set of EVAL_SETS
	"score_balanced", "score_worst",  # the sets' mean (each alike) and the worst set
	"score_generated_se", "score_neutrals_se", "score_real_se", "score_castle_se", "score_hard_se",  # their standard errors
	"failed_battles",  # left out: they failed every attempt (play_eval)
]  # eval.csv, greedy battles per side: win rate, mean reward -1..1, outcomes; and minus the eval opponent's own reward on
# the same seeds. army_kept / army_killed: shares of our army kept and of the enemy's lost (data.BattleSummary), ours
# and the eval opponent's on the same seeds: the same whatever the reward's definition. total_score: THE measure of a
# model, the mean zero-sum reward (reward.py) of all eval battles, attack and defence: -1..1, higher is better

# The eval battles' VCMI seeds: fixed random numbers, the same for every eval round of every run
_eval_rng = random.Random(20261023)
EVAL_SEEDS = tuple(_eval_rng.randrange(1, 2**31) for _ in range(256))


# TensorBoard's eval tab holds the top-level scores only (the user, 2026-10-29): eval/battle (here, evaluate_now: the
# battle model against the eval opponents) and eval/campaign (campaign_model/evaluate.py: against Nullkiller2), and
# each model's training value loss (VALUE_LOSS_TAB, campaign_model/learner.py); the rest of an eval goes under
# eval_<opponent>/ and eval_details/
EVAL_TAB = "eval/battle"


def log_eval_tensorboard(writer, row, opponent=""):
	"""An eval row as TensorBoard scalars under eval_<opponent>/ (EVAL_TAB is the mean over the opponents, evaluate_now),
	or eval_details/ for a run with a single, unnamed opponent (whose total is EVAL_TAB)."""
	epoch = int(row["epoch"])
	prefix = f"eval_{opponent}" if opponent else "eval_details"
	if opponent:
		writer.add_scalar(f"{prefix}/total_score", float(row["total_score"]), epoch)
	else:  # (one opponent: its score is the total)
		writer.add_scalar(EVAL_TAB, float(row["total_score"]), epoch)
	for kind, score in row.get("hard_kinds", {}).items():
		writer.add_scalar(f"{prefix}/hard/{kind}", float(score), epoch)
	for column in (*(f"score_{k}" for k in EVAL_SETS), *(f"score_{k}_se" for k in EVAL_SETS), "score_balanced", "score_worst", "escape_while_ahead", "shooter_not_shooting", "worthless_cast", "cast_skipped_on_loss"):
		if row.get(column) not in (None, "") and row[column] == row[column]:  # skip nan: a set or a case that never came up
			writer.add_scalar(f"{prefix}/{column}", float(row[column]), epoch)
	for name in ("attack", "defence", "mean"):
		if name != "mean":
			writer.add_scalar(f"{prefix}/won/{name}", float(row[f"{name}_win_rate"]), epoch)
		writer.add_scalar(f"{prefix}/reward/{name}", float(row[f"{name}_reward"]), epoch)
		writer.add_scalar(f"{prefix}/vs_baseline/{name}", float(row[f"{name}_vs_baseline"]), epoch)
		if name != "mean":
			for measure in ("army_kept", "army_killed"):
				writer.add_scalar(f"{prefix}/{measure}/{name}", float(row[f"{name}_{measure}"]), epoch)
				writer.add_scalar(f"{prefix}/{measure}_vs_baseline/{name}", float(row[f"{name}_{measure}"]) - float(row[f"{name}_baseline_{measure}"]), epoch)
	writer.flush()


def log_tensorboard(writer, row, self_play=False):
	"""One epoch's progress row as TensorBoard scalars; empty columns are skipped. In self-play the mean reward of
	the two sides is left out: one side's win is the other's loss, so it is flat whatever the model learns."""
	epoch = int(row["epoch"])
	for column, tag in TENSORBOARD_TAGS.items():
		if self_play and column == "mean_reward":
			continue
		if row.get(column) not in (None, ""):
			value = float(row[column])
			if value == value:  # not nan: a case that never came up in this update
				writer.add_scalar(tag, value, epoch)
	for name in ("attack", "defence"):
		if row.get(f"{name}_win_rate") in (None, ""):  # runs from before win rates were logged
			writer.add_scalar(f"won/{name}", float(row[f"{name}_outcome"] == "WIN"), epoch)
	writer.flush()


def outcome_text(summaries):
	"""One battle: its outcome's name; several: counts, e.g. "WIN:3 LOSS:5"."""
	if len(summaries) == 1:
		return summaries[0].outcome.name
	counts = Counter(s.outcome.name for s in summaries)
	return " ".join(f"{name}:{n}" for name, n in sorted(counts.items()))


def signal_ratio(optimizer):
	"""Median over all weights of |m| / sqrt(v) in Adam's state (1 = the gradients always agree, 0 = they cancel out).

	m and v are bias-corrected as in Adam's step; without it the ratio is ~3 on the first step.
	"""
	beta1, beta2 = optimizer.param_groups[0]["betas"]
	ratios = []
	for state in optimizer.state.values():
		if "exp_avg" in state:
			step = float(state["step"])
			m = state["exp_avg"].flatten() / (1 - beta1**step)
			v = state["exp_avg_sq"].flatten() / (1 - beta2**step)
			ratios.append(m.abs() / (v.sqrt() + 1e-12))
	return float(torch.cat(ratios).median()) if ratios else float("nan")


def flat_weights(model):
	return torch.cat([p.detach().flatten() for p in model.parameters()])


def follow(run, poll_seconds=10, idle_minutes=10):
	"""Mirror a run's progress.csv into its TensorBoard folder until no new epoch arrives for idle_minutes."""
	out = run_dir(TrainConfig(run=run))
	writer = SummaryWriter(out / "tensorboard" / ("follow-" + time.strftime("%Y%m%d-%H%M%S")))
	logged, idle_since = 0, time.time()
	while time.time() - idle_since < idle_minutes * 60:
		rows = list(csv.DictReader((out / "progress.csv").open()))
		for row in rows[logged:]:
			log_tensorboard(writer, row)
		if len(rows) > logged:
			logged, idle_since = len(rows), time.time()
		time.sleep(poll_seconds)
	writer.close()


def play_with_retries(games, model_path, record_root, config, label):
	"""play.play with up to config.attempts tries (a battle can fail or stall); the records (in memory).

	record_root: where to also write them for debugging (config.keep_records), or None."""
	for attempt in range(1, config.attempts + 1):
		try:
			if record_root:
				shutil.rmtree(record_root, ignore_errors=True)
			return play.play(games, model_path, record_root)
		except RuntimeError as error:  # a battle failed or stalled; its runner folder is kept for debugging
			print(f"{label}: attempt {attempt} failed: {str(error)[:300]}", flush=True)
			if attempt == config.attempts:
				raise


def play_in_groups(games, model_path, config, label):
	"""play_with_retries for many battles, games_per_side at a time (as many VCMI processes as a training epoch)."""
	n = config.games_per_side
	return [r for i in range(0, len(games), n) for r in play_with_retries(games[i : i + n], model_path, None, config, label)]


def play_eval(games, model_path, config, label):
	"""The records of the eval battles, played on config.eval_workers persistent battle processes (play.BattleProcess:
	a VCMI start-up per 50 battles, not one per battle as play_in_groups), new ones each eval. Game i goes to worker
	i % eval_workers, so every eval plays the same battles in the same processes in the same order. A battle that
	fails or stalls is retried once in a new process, then left out (an eval stopped the run when
	one battle stalled every time). Returns (the games played, their records)."""
	model_path = Path(model_path).resolve()  # VCMI runs in build/bin: paths must be absolute
	n = min(config.eval_workers, len(games))
	results, errors = [None] * len(games), []

	def work(worker):
		process = play.BattleProcess()
		try:
			for i in range(worker, len(games), n):
				for attempt in range(1, min(config.attempts, 2) + 1):  # greedy: a battle that stalls once mostly stalls again
					try:
						results[i] = process.play(games[i], model_path)
						break
					except RuntimeError as error:
						print(f"{label}: battle {i}, attempt {attempt} failed: {str(error)[:300]}", flush=True)
		except BaseException as error:  # re-raised below, in the calling thread
			errors.append(error)
		finally:
			process.close()

	threads = [threading.Thread(target=work, args=(w,), daemon=True) for w in range(n)]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()
	if errors:
		raise errors[0]
	played = [i for i, records in enumerate(results) if records is not None]
	if len(played) < len(games):
		print(f"{label}: left out {len(games) - len(played)} battles that failed every attempt: {[games[i].map for i in range(len(games)) if results[i] is None]}", flush=True)
	return [games[i] for i in played], [r for i in played for r in results[i]]


GENERATED_ROOT = ROOT / "runtime" / "generated"  # generated scenarios of running training epochs, deleted after each
EVAL_GENERATOR_SEED = 20261025  # the generated eval battles: the same for every run


def generated_games(scenarios, label, config, our_sides=(0,), **game):
	"""Games of generated scenarios: each written to GENERATED_ROOT/<run>/<label>_<i>.json and built into its map.
	our_sides: the sides we play in each (with SELF_PLAY: (0,), both are ours). Returns (games, files to delete)."""
	folder = GENERATED_ROOT / config.run
	folder.mkdir(parents=True, exist_ok=True)
	games, files = [], []
	for i, scenario in enumerate(scenarios):
		path = folder / f"{label}_{i}.json"
		path.write_text(json.dumps(scenario))
		name = play.build_map(path, variant=config.run)
		files += [path, play.USER_DIR / (name + ".vmap")]
		games += [play.Game(str(path), our_side=side, game_seed=scenario["seed"], ai=config.ai, map=name, **game) for side in our_sides]
	return games, files


def eval_scenarios(config):
	"""The generated eval battles, drawn from EVAL_GENERATOR_SEED: the first config.eval_scenarios two-sided ones,
	then the first config.eval_neutral_scenarios against wandering monsters (both ordinary battles: the same as before
	the generator had hard cases), then config.eval_hard_scenarios hard cases; each is played from both sides."""
	cfg, w = generator.load_config(), generator.world()
	rng, two_sided, neutral = random.Random(EVAL_GENERATOR_SEED), [], []
	while len(two_sided) < config.eval_scenarios or len(neutral) < config.eval_neutral_scenarios:
		scenario = generator.sample_type(rng, cfg, w)
		if "neutrals" in scenario and len(neutral) < config.eval_neutral_scenarios:
			neutral.append(scenario)
		elif "neutrals" not in scenario and len(two_sided) < config.eval_scenarios:
			two_sided.append(scenario)
	kinds, hard_rng = sorted(generator.HARD_KINDS), random.Random(EVAL_GENERATOR_SEED + 1)
	hard = [generator.HARD_KINDS[kinds[i % len(kinds)]](hard_rng, cfg, w) for i in range(config.eval_hard_scenarios)]
	return two_sided + neutral + hard


REAL_EVAL = ROOT / "battle_model" / "scenarios" / "real_eval.json"  # battles of real games (real_games.py --collect)
_generated_eval = {}  # run -> its eval games (the maps are built once a run)


def _generated_eval_games(config):
	key = (config.run, config.eval_opponent)
	if key not in _generated_eval:
		real = json.loads(REAL_EVAL.read_text())[: config.eval_real_scenarios] if config.eval_real_scenarios else []
		castle = [{**json.loads(Path(config.scenario).read_text()), "seed": seed, "_about": "castle_vs_inferno"} for seed in EVAL_SEEDS[: config.eval_castle_seeds]]
		_generated_eval[key] = generated_games(eval_scenarios(config) + real + castle, f"eval_{config.eval_opponent}", config, our_sides=(0, 1), opponent=config.eval_opponent, temperature=0.0)[0]
	return _generated_eval[key]


def eval_name(config, main_opponent):
	"""The eval files' suffix of an opponent: none for the main one."""
	return "" if config.eval_opponent == main_opponent else f"_{config.eval_opponent}"


EVAL_SETS = ("generated", "neutrals", "real", "castle", "hard")
# Greedy evals: a shooter that could shoot at full damage and did not is a failure of the model or of the metric
# above this share (the user, 2026-10-28: 0.99+ shooting is the aim with no sampling noise; 0.95 is fine in training)
SHOOTER_IDLE_LIMIT = 0.01


def eval_set(game):
	"""Which eval set a game is of: real battles, castle_vs_inferno, generated against neutrals, other generated."""
	about = play.scenario_of(game.scenario).get("_about", "")
	return "real" if about.startswith("real") else "castle" if about.startswith("castle") else "hard" if about.startswith("hard") else "neutrals" if about.startswith("hero_vs_neutrals") else "generated"


@functools.lru_cache(maxsize=1)
def _creature_tables():
	return rules.creature_tables()


def _no_distance_penalty():
	return _creature_tables()["no_distance_penalty"].numpy()


def full_shot(t, i):
	"""Whether the acting unit of decision i (record tensors t) can shoot something at full damage: a legal target
	within rules.SHOOTING_PENALTY_DISTANCE hexes (either of its hexes), or it has no distance penalty."""
	shooter = int(t["g_active_unit"][i])
	creature = int(t["u_creature"][i, shooter])
	if 0 <= creature < len(_no_distance_penalty()) - 1 and _no_distance_penalty()[creature]:
		return True
	legal = np.asarray(t["m_shoot_target"][i], dtype=bool)
	if not legal.any():
		return False
	here = torch.as_tensor(int(t["u_hex"][i, shooter]))
	hexes, rears = torch.as_tensor(np.asarray(t["u_hex"][i], dtype=np.int64)), torch.as_tensor(np.asarray(t["u_rear_hex"][i], dtype=np.int64))
	near = (rules.hex_distance(here, hexes) <= rules.SHOOTING_PENALTY_DISTANCE) | ((rears >= 0) & (rules.hex_distance(here, rears) <= rules.SHOOTING_PENALTY_DISTANCE))
	return bool((near.numpy() & legal).any())


def melee_escapes(t, rows, actions):
	"""[len(rows)] for MOVE decisions rows of a record (tensors t): the move takes the acting unit to a hex where less
	of the enemy army could attack it in melee in its coming turn than where it stood (features.threat_maps'
	melee_threat_value), the hex it left being within some enemy's melee reach: a shooter kiting."""
	from . import features as F
	from .schema import INPUTS

	index = np.asarray(rows)
	x = {s.name: torch.from_numpy(np.asarray(t[s.name])[index].astype(np.int64) if s.dtype == "int64" else np.asarray(t[s.name])[index]) for s in INPUTS}
	with torch.no_grad():
		b = F.read_battle(x, _creature_tables())
		maps, _ = F.threat_maps(b, *F.distance_maps(b))
	threat = maps[..., F.HEX_FEATURES.index("melee_threat_value")].numpy()  # [n, H]
	here = threat[np.arange(len(index)), np.asarray(t["u_hex"])[index, np.asarray(t["g_active_unit"])[index]].clip(min=0)]
	there = np.array([threat[k, a["hex"]] if a.get("hex", -1) >= 0 else here[k] for k, a in enumerate(actions)])
	return (here > 0) & (there < here)


def failure_modes(records):
	"""Signs of dumb play in a set of battle records (eval), each a share 0..1 (nan when it never came up):
	escape_while_ahead    retreats and surrenders while our value head thought we were winning (V > 0)
	shooter_not_shooting  of a shooter's decisions where it can shoot at full damage (a legal target within 10 hexes, or
	                      no distance penalty), those that waste it (2026-10-29, the user): it defends; it moves without
	                      escaping melee (a move off a hex enemies could reach in melee next turn to one fewer of them can
	                      is kiting: 38% of the idle moves at 3792); it waits and its next decision that round can no
	                      longer shoot at full damage (waiting, then shooting at full damage, is common and fine; a wait
	                      when only a half-damage shot was there is no failure). Defending while blocked is none (it
	                      cannot shoot then). Before, a unit's whole round counted, kiting included
	worthless_cast        hero casts at a unit or hex that lose us value (predicted value < 0: e.g. a Frost Ring on our own
	                      stacks); 0 is no sign: buffs, debuffs and summons change no health when cast (Slow, Shield)
	cast_skipped_on_loss  in battles we lost, the share of the rounds our hero could cast in (one of our decisions had a
	                      hero spell legal) where it cast nothing: a hero that died early counts only the rounds it was
	                      there for (the share of mana left unspent counted those too)"""
	from .data import summarize
	counts = {name: [0, 0] for name in ("escape_while_ahead", "shooter_not_shooting", "worthless_cast", "cast_skipped_on_loss")}
	for r in records:
		t = r.tensors
		full = [bool(t["m_type"][i, ActionType.SHOOT]) and full_shot(t, i) for i in range(len(r.actions))]
		chances = [i for i in range(len(r.actions)) if full[i]]
		moves = [i for i in chances if r.actions[i]["type"] == ActionType.MOVE]
		escaped = dict(zip(moves, melee_escapes(t, moves, [r.actions[i] for i in moves]))) if moves else {}
		for i in chances:
			kind = r.actions[i]["type"]
			if kind == ActionType.WAIT:  # judged by what it left: the unit's next decision this round
				later = next((j for j in range(i + 1, len(r.actions)) if int(t["g_round"][j]) == int(t["g_round"][i]) and int(t["g_active_unit"][j]) == int(t["g_active_unit"][i])), None)
				wasted = later is not None and not full[later]
			else:
				wasted = kind == ActionType.DEFEND or (kind == ActionType.MOVE and not escaped[i])
			counts["shooter_not_shooting"][0] += wasted
			counts["shooter_not_shooting"][1] += 1
		for i, a in enumerate(r.actions):
			kind = a["type"]
			if kind in (ActionType.RETREAT, ActionType.SURRENDER):
				counts["escape_while_ahead"][0] += float(t["value"][i, -1]) > 0
				counts["escape_while_ahead"][1] += 1
			if kind == ActionType.HERO_SPELL and a["spell"] >= 0:
				sign = 1.0 if int(t["g_our_side"][i]) == 0 else -1.0  # the values are the attacker's gain minus the defender's
				chosen = sign * float(t["s_unit_value"][i, a["spell"], a["target_unit"]]) if a["target_unit"] >= 0 else sign * float(t["s_hex_value"][i, a["spell"], a["target_hex"]]) if a["target_hex"] >= 0 else None
				if chosen is not None:
					counts["worthless_cast"][0] += chosen < 0
					counts["worthless_cast"][1] += 1
		if summarize(r.events).outcome == Outcome.LOSS:
			could, cast = set(), set()
			for i, a in enumerate(r.actions):
				if t["m_type"][i, ActionType.HERO_SPELL]:
					could.add(int(t["g_round"][i]))
				if a["type"] == ActionType.HERO_SPELL:
					cast.add(int(t["g_round"][i]))
			counts["cast_skipped_on_loss"][0] += len(could - cast)
			counts["cast_skipped_on_loss"][1] += len(could)
	return {name: round(hit / n, 4) if n else float("nan") for name, (hit, n) in counts.items()}


def eval_games(config, battle_map):
	"""The eval battles: greedy, one per side on each of the first config.eval_seeds EVAL_SEEDS (generator: each
	eval scenario from both sides)."""
	if config.generator:
		return list(_generated_eval_games(config))
	return [
		play.Game(config.scenario, our_side=side, opponent=config.eval_opponent, game_seed=seed, temperature=0.0, ai=config.ai, map=battle_map)
		for side in (0, 1)
		for seed in EVAL_SEEDS[: config.eval_seeds]
	]


def eval_baseline(model, out, battle_map, config, suffix=""):
	"""The eval opponent's own results on the eval battles (it plays our side too: ALPHA_BATTLE_DELEGATE),
	{(side, seed): (reward, army kept, army killed)}; computed once per run, kept in <run dir>/eval_baseline<suffix>.csv."""
	path = out / f"eval_baseline{suffix}.csv"
	games = eval_games(config, battle_map)
	if path.exists():
		rows = {(int(r["side"]), int(r["seed"])): (float(r["reward"]), float(r.get("army_kept") or "nan"), float(r.get("army_killed") or "nan")) for r in csv.DictReader(path.open())}
		if all((g.our_side, g.game_seed) in rows for g in games):
			return rows
	model_path = out / "eval.onnx"
	export(cpu_copy(model), model_path)  # loaded but not used: the delegate plays
	games, records = play_eval([replace(g, settings=(("ALPHA_BATTLE_DELEGATE", config.eval_opponent),)) for g in games], model_path, config, "eval baseline")
	summaries = {(g.our_side, g.game_seed): summarize(r.events) for g, r in zip(games, records)}
	rows = {key: (s.reward, s.army_kept, s.army_killed) for key, s in summaries.items()}
	with path.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["side", "seed", "reward", "army_kept", "army_killed"])
		writer.writerows([side, seed, *(round(v, 6) for v in values)] for (side, seed), values in rows.items())
	return rows


KL_WINDOW = 4  # minibatches whose mean KL the KL control judges an update by (TrainConfig.target_kl)


def eval_round(model, epoch, out, battle_map, config, baseline, name=""):
	"""Play the eval battles against config.eval_opponent with the current model; the eval.csv row."""
	started = time.time()
	model_path = out / "eval.onnx"
	cpu = cpu_copy(model)
	export(cpu, model_path)
	verify(cpu, model_path, seeds=(epoch,), batch_sizes=(2,))  # the eval plays this file, training the torch model: the same?
	games = [g for g in eval_games(config, battle_map) if (g.our_side, g.game_seed) in baseline]  # (one the baseline failed has none)
	played, records = play_eval(games, model_path, config, f"eval {epoch}")
	failed, games = len(games) - len(played), played
	summaries = [summarize(r.events) for r in records]
	write_traces(out, epoch, f"vs_{config.eval_opponent}_greedy{name}", [next(r for g, r in zip(games, records) if g.our_side == side) for side in (0, 1)])  # the first seed's
	row = {"epoch": epoch}
	rewards, baselines = [], []
	for side, name in ((0, "attack"), (1, "defence")):
		ours = [(g, s) for g, s in zip(games, summaries) if g.our_side == side]
		reward = [s.reward for _, s in ours]
		base = [baseline[(side, g.game_seed)][0] for g, _ in ours]
		mean = lambda values: round(sum(values) / len(values), 4)
		row[f"{name}_army_kept"], row[f"{name}_army_killed"] = mean([s.army_kept for _, s in ours]), mean([s.army_killed for _, s in ours])
		row[f"{name}_baseline_army_kept"] = mean([baseline[(side, g.game_seed)][1] for g, _ in ours])
		row[f"{name}_baseline_army_killed"] = mean([baseline[(side, g.game_seed)][2] for g, _ in ours])
		rewards += reward
		baselines += base
		row[f"{name}_win_rate"] = round(sum(s.outcome == Outcome.WIN for _, s in ours) / len(ours), 4)
		row[f"{name}_reward"] = round(sum(reward) / len(reward), 4)
		row[f"{name}_outcome"] = outcome_text([s for _, s in ours])
		row[f"{name}_baseline_reward"] = round(sum(base) / len(base), 4)
		row[f"{name}_vs_baseline"] = round((sum(reward) - sum(base)) / len(reward), 4)
	row["mean_reward"] = round(sum(rewards) / len(rewards), 4)
	row["total_score"] = row["mean_reward"]
	# The score of each eval set (our reward minus the eval opponent's own on the same battles): a gain on one must not
	# hide a loss on another
	by_set = {}
	for g, s in zip(games, summaries):
		key = eval_set(g)
		by_set.setdefault(key, []).append(s.reward - baseline[(g.our_side, g.game_seed)][0])
	for key in EVAL_SETS:
		row[f"score_{key}"] = round(sum(by_set[key]) / len(by_set[key]), 4) if key in by_set else float("nan")
		row[f"score_{key}_se"] = round(float(np.std(by_set[key], ddof=1) / np.sqrt(len(by_set[key]))), 4) if len(by_set.get(key, ())) > 1 else float("nan")  # its noise
	# The goal is to win every set, not a good average (the user, 2026-10-27): each set counted alike, and the worst
	sets = [row[f"score_{key}"] for key in EVAL_SETS if row[f"score_{key}"] == row[f"score_{key}"]]
	row["score_balanced"] = round(sum(sets) / len(sets), 4) if sets else float("nan")
	# Each trick of the hard set on its own (generator.HARD_KINDS; TensorBoard eval_<opponent>/hard/<kind>, not a column)
	kinds = {}
	for g, s in zip(games, summaries):
		about = play.scenario_of(g.scenario).get("_about", "")
		if about.startswith("hard"):
			kinds.setdefault(about.split()[1].rstrip(",:"), []).append(s.reward - baseline[(g.our_side, g.game_seed)][0])
	row["hard_kinds"] = {k: round(sum(v) / len(v), 4) for k, v in sorted(kinds.items())}
	row["score_worst"] = round(min(sets), 4) if sets else float("nan")
	row["mean_vs_baseline"] = round((sum(rewards) - sum(baselines)) / len(rewards), 4)
	row.update(failure_modes(records))
	row["failed_battles"] = failed
	row["seconds"] = round(time.time() - started, 1)
	row["rewards"] = {(g.our_side, g.game_seed, eval_set(g)): s.reward for g, s in zip(games, summaries)}  # (not a column: for paired comparisons)
	return row


def upgrade_csv(path, columns):
	"""Rewrite a log written by an older version with the current columns (missing ones empty), so a resumed run appends cleanly."""
	with path.open(newline="") as f:
		reader = csv.DictReader(f)
		if reader.fieldnames == columns:
			return
		rows = list(reader)
	with path.open("w", newline="") as f:
		writer = csv.DictWriter(f, columns, extrasaction="ignore")
		writer.writeheader()
		writer.writerows({c: row.get(c, "") for c in columns} for row in rows)


def write_traces(out, epoch, label, records):
	"""Readable traces (trace.py) of a few battles, to see what the model does: <run>/traces/epoch_<epoch>_<label>.txt
	(a few KB each)."""
	folder = out / "traces"
	folder.mkdir(exist_ok=True)
	(folder / f"epoch_{epoch:05d}_{label}.txt").write_text("\n\n".join(trace(r) for r in records) + "\n")


def resolve_device(name):
	if name != "auto":
		return torch.device(name)
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def to_device(batch, device):
	"""A collated batch (nested dicts of tensors) on device."""
	if isinstance(batch, dict):
		return {key: to_device(value, device) for key, value in batch.items()}
	return batch.to(device)


def select(batch, index):
	"""The rows `index` of a collated batch (nested dicts of tensors)."""
	if isinstance(batch, dict):
		return {key: select(value, index) for key, value in batch.items()}
	return batch[index]


def cpu_copy(model):
	"""A CPU copy of the model, for the ONNX export (the battles run the model on the CPU)."""
	return copy.deepcopy(model).cpu()


class Producer:
	"""The source of training data: plays an epoch's battles in a background thread, so that playing (VCMI, on the
	CPU) and training (torch, on the GPU) overlap. start(epoch, model_file) begins; take() waits for the records."""

	def __init__(self, play_epoch):
		self._play_epoch = play_epoch
		self._thread, self._result = None, None

	def start(self, epoch, model_file):
		def run():
			try:
				self._result = (self._play_epoch(epoch, model_file), None)
			except BaseException as error:  # re-raised by take() in the training thread
				self._result = (None, error)

		self._thread = threading.Thread(target=run, daemon=True)
		self._thread.start()

	def take(self):
		self._thread.join()
		records, error = self._result
		self._thread, self._result = None, None
		if error:
			raise error
		return records


def model_kwargs(config):
	"""The model's constructor arguments, saved in checkpoints (model.build_model rebuilds the model from them)."""
	if config.model_architecture in ("v2", "v3"):
		return {"architecture": config.model_architecture, "dim": config.model_dim, "layers": config.model_layers}
	return {"architecture": "v1", "dim": config.model_dim, "layers": config.model_layers, "mask_features": config.model_mask_features, "chain_final_value": config.model_chain_final_value, "kill_rule": config.model_kill_rule}


class WeightAverage:
	"""One running average of a model's weights, updated after each update: average = d * average + (1 - d) * weights,
	d = min(decay, (1 + n) / (10 + n)) after n updates (a short average while there are few, as in TensorFlow's
	ExponentialMovingAverage). A single copy of the parameters on the model's device, one fused op per update; the
	buffers are constants, copied with the model."""

	def __init__(self, model, decay, state=None, updates=0):
		self.decay, self.updates = decay, updates
		self.model = copy.deepcopy(model).eval()
		for p in self.model.parameters():
			p.requires_grad_(False)
		if state is not None:
			self.model.load_state_dict(state)

	@torch.no_grad()
	def update(self, model):
		d = min(self.decay, (1 + self.updates) / (10 + self.updates))
		torch._foreach_lerp_(list(self.model.parameters()), [p.detach() for p in model.parameters()], 1 - d)
		self.updates += 1


def value_gains(model):
	"""{value_gain_<kind>: its gain} of the model's value priors (model.VALUE_PRIOR_KINDS; none for older architectures)."""
	heads = getattr(model, "actions", None)
	if not hasattr(heads, "value_gains"):
		return {}
	with torch.no_grad():
		gains = VALUE_PRIOR_GAIN ** torch.tanh(VALUE_PRIOR_STEP * heads.value_gains.detach().float().cpu())
	return {f"value_gain_{kind}": round(float(g), 4) for kind, g in zip(VALUE_PRIOR_KINDS, gains)}


GRADIENT_CHECK_EVERY = 5  # updates between gradient health checks (trunk_gradients), on an update's first minibatch


def trunk_gradients(model, main, foresight):
	"""Gradient health on one minibatch, without touching .grad: the trunk's gradient norm from the PPO losses (main) and
	from the foresight's, their cosine, and the share of the model's parameters neither loss reaches (an exactly zero
	gradient: a path cut off). The foresight's gradient should be a fraction of the main one's, not dominate it."""
	trunk = [p for p in model.trunk.parameters() if p.requires_grad]
	params = [p for p in model.parameters() if p.requires_grad]
	g_main = torch.autograd.grad(main, params, retain_graph=True, allow_unused=True)
	g_fore = torch.autograd.grad(foresight, params, retain_graph=True, allow_unused=True)
	index = {id(p): i for i, p in enumerate(params)}
	flat = lambda grads: torch.cat([(grads[index[id(p)]] if grads[index[id(p)]] is not None else torch.zeros_like(p)).flatten() for p in trunk])  # noqa: E731
	a, b = flat(g_main), flat(g_fore)
	names = {id(p): name for name, p in model.named_parameters()}
	dead = [names[id(p)] for p, x, y in zip(params, g_main, g_fore) if (x is None or not bool(x.any())) and (y is None or not bool(y.any()))]
	if dead:  # (heads the minibatch never used, embeddings of ids it never saw; a part always here is a path cut off)
		print("gradients: none reach " + ", ".join(sorted({".".join(n.split(".")[:3]) for n in dead})), flush=True)
	dead = len(dead)
	return {
		"grad_trunk_main": round(float(a.norm()), 6),
		"grad_trunk_foresight": round(float(b.norm()), 6),
		"grad_trunk_cosine": round(float(torch.nn.functional.cosine_similarity(a, b, dim=0)), 4),
		"grad_dead_share": round(dead / max(len(params), 1), 4),
	}


def load_optimizer(optimizer, model, state):
	"""The checkpoint's optimizer state. A checkpoint from before parameters were added (2026-10-29: the value priors,
	model.VALUE_PRIOR_KINDS) keeps Adam's moments for the parameters it had, matched by name in the model's order; the
	new ones start fresh."""
	saved = state["optimizer"]
	names = [name for name, _ in model.named_parameters()]
	if len(saved["param_groups"][0]["params"]) == len(names):
		optimizer.load_state_dict(saved)
		return
	had = [i for i, name in enumerate(names) if name in state["model"]]  # the checkpoint's parameters, in its order
	if len(had) != len(saved["param_groups"][0]["params"]):
		raise ValueError(f"the checkpoint's optimizer has {len(saved['param_groups'][0]['params'])} parameters, its model {len(had)}")
	shapes = [p.shape for _, p in model.named_parameters()]
	moments = {new: saved["state"][old] for old, new in enumerate(had) if old in saved["state"] and saved["state"][old]["exp_avg"].shape == shapes[new]}
	reshaped = [names[new] for old, new in enumerate(had) if old in saved["state"] and new not in moments]
	if reshaped:
		print("optimizer: fresh moments for the parameters whose shape changed: " + ", ".join(reshaped), flush=True)
	optimizer.load_state_dict({"state": moments, "param_groups": [{**saved["param_groups"][0], "params": list(range(len(names)))}]})
	print(f"optimizer: Adam's moments kept for {len(had)} parameters, fresh for {len(names) - len(had)} new: " + ", ".join(n for i, n in enumerate(names) if i not in set(had)), flush=True)


def save_checkpoint(path, model, optimizer, epoch, config, elapsed=0.0, lr_scale=1.0, value_ev=None, average=None):
	"""elapsed: the training's seconds so far (run_hours counts them across resumes); lr_scale: the KL control's;
	value_ev: the smoothed explained variance the window credit's outcome share follows; average: the WeightAverage
	(saved as "average", the weights the evals played; "model" is the trained weights the optimizer goes on from)."""
	extra = {"average": average.model.state_dict(), "average_updates": average.updates} if average else {}
	torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "model_kwargs": model_kwargs(config), "elapsed": elapsed, "lr_scale": lr_scale, "value_ev": value_ev, **extra}, path)


def train(config, resume=False):
	out = run_dir(config, new=not resume)
	out.mkdir(parents=True, exist_ok=True)
	(out / "config.txt").write_text("\n".join(f"{k} = {v}" for k, v in asdict(config).items()) + "\n")
	torch.manual_seed(config.seed)
	device = resolve_device(config.device)
	architecture = model_kwargs(config)
	model = build_model(architecture).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
	first_epoch = 1
	elapsed_before = 0.0
	lr_scale = 1.0  # the KL control's (target_kl)
	value_ev = None  # the value head's explained variance (credit_check), smoothed: sets the window credit's outcome share
	latest = out / "checkpoint_latest.pt"
	state = {}
	if resume and latest.exists():
		state = torch.load(latest, map_location=device)
		architecture = state.get("model_kwargs", {})
		model = build_model(architecture).to(device)  # the architecture the run was started with
		optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
		model.load_state_dict(state["model"])
		load_optimizer(optimizer, model, state)
		first_epoch = state["epoch"] + 1
		elapsed_before = state.get("elapsed", 0.0)
		lr_scale = state.get("lr_scale", 1.0)
		value_ev = state.get("value_ev")
		print(f"resuming after epoch {state['epoch']} ({elapsed_before / 3600:.2f} h)")
	elif config.init_from:  # a new run going on from another's checkpoint
		source = Path(config.init_from)
		state = torch.load(source, map_location=device)
		architecture = state.get("model_kwargs", {})
		model = build_model(architecture).to(device)
		optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
		model.load_state_dict(state["model"])
		load_optimizer(optimizer, model, state)
		first_epoch = state["epoch"] + 1  # its numbering goes on, so its league snapshots keep their versions
		lr_scale = state.get("lr_scale", 1.0)
		value_ev = state.get("value_ev")
		(out / "models").mkdir(parents=True, exist_ok=True)
		for snapshot in (source.parent / "models").glob("snapshot_*.pt"):  # the league's past versions
			if int(snapshot.stem.split("_")[1]) <= state["epoch"] and not (out / "models" / snapshot.name).exists():
				shutil.copy2(snapshot, out / "models" / snapshot.name)
		for baseline in source.parent.glob("eval_baseline*.csv"):  # (eval_baseline recomputes them if the sets differ)
			if not (out / baseline.name).exists():
				shutil.copy2(baseline, out / baseline.name)
		print(f"starting from {source} (epoch {state['epoch']})")
	use_fused(optimizer)
	# (a run from another's checkpoint keeps its average only on a resume: a new run averages from its start)
	average = WeightAverage(model, config.weight_average, *((state["average"], state["average_updates"]) if resume and "average" in state else ())) if config.weight_average else None

	battle_map = None if config.generator else play.build_map(config.scenario, variant=config.run)
	reward_config, advantage_config, ppo_config = config.reward, config.advantage, config.ppo
	progress = out / "progress.csv"
	if not progress.exists() or not resume:
		with progress.open("w", newline="") as f:
			csv.writer(f).writerow(COLUMNS)
	else:
		upgrade_csv(progress, COLUMNS)
		if value_ev is None:  # a checkpoint from before value_ev was kept: the last updates' explained variance
			evs = [(float(r["attack_value_ev"]) + float(r["defence_value_ev"])) / 2 for r in list(csv.DictReader(progress.open()))[-10:] if r.get("attack_value_ev") not in (None, "", "nan") and r.get("defence_value_ev") not in (None, "", "nan")]
			value_ev = sum(evs) / len(evs) if evs else None
	writer = SummaryWriter(out / "tensorboard" / "run")  # one folder for the whole run: a resume goes on with the same line
	eval_csv = out / "eval.csv"
	if not eval_csv.exists() or not resume:
		with eval_csv.open("w", newline="") as f:
			csv.writer(f).writerow(EVAL_COLUMNS)
	else:
		upgrade_csv(eval_csv, EVAL_COLUMNS)

	opponents = [replace(config, eval_opponent=name) for name in (config.eval_opponent, *config.eval_extra_opponents)]
	baselines = [eval_baseline(model, out, battle_map, c, eval_name(c, config.eval_opponent)) for c in opponents] if config.eval_every else []
	for c in opponents[1:]:
		path = out / f"eval{eval_name(c, config.eval_opponent)}.csv"
		if not path.exists() or not resume:
			with path.open("w", newline="") as f:
				csv.writer(f).writerow(EVAL_COLUMNS)
		else:  # (it was not: the rows after a resume that added columns did not fit its header)
			upgrade_csv(path, EVAL_COLUMNS)
	if resume and (out / "eval_live.csv").exists():  # (the same for the live weights' evals: 2026-10-28 review)
		upgrade_csv(out / "eval_live.csv", EVAL_COLUMNS)

	def evaluate_now(epoch):
		"""Every eval opponent's eval round; the score of the model is their mean, each opponent counting the same
		(eval/total_score in TensorBoard, eval_total.csv; each opponent's own rows under eval_<opponent>/)."""
		if any(c.eval_opponent == "MMAI" for c in opponents) and "mmai" not in play.active_mods():  # it would play as BattleAI
			raise RuntimeError(f"the MMAI mod is not active ({play.USER_DIR / 'config' / 'modSettings.json'}): MMAI would fall back to BattleAI")
		rows = {c.eval_opponent: evaluate_against(epoch, c, baseline) for c, baseline in zip(opponents, baselines)}
		if average and average.updates and config.eval_live:  # the trained weights on the same battles: the average's test
			live = eval_round(model, epoch, out, battle_map, opponents[0], baselines[0], "_live")  # (its own traces)
			with (out / "eval_live.csv").open("a", newline="") as f:
				if f.tell() == 0:
					csv.writer(f).writerow(EVAL_COLUMNS)
				csv.writer(f).writerow([live[c] for c in EVAL_COLUMNS])
			log_eval_tensorboard(writer, live, "live")
			ours, theirs = rows[opponents[0].eval_opponent]["rewards"], live["rewards"]
			diff = np.array([ours[k] - theirs[k] for k in ours if k in theirs])
			if len(diff) > 1:
				writer.add_scalar("eval_details/average_minus_live", float(diff.mean()), epoch)
				print(f"eval  {epoch:4d}  average - live weights vs {opponents[0].eval_opponent}: {diff.mean():+.4f} +- {diff.std(ddof=1) / np.sqrt(len(diff)):.4f} over {len(diff)} battles (live: total score {live['total_score']:+.3f})", flush=True)
		scores = {name: row["total_score"] for name, row in rows.items()}
		total = round(sum(scores.values()) / len(scores), 4)
		balanced = round(sum(row["score_balanced"] for row in rows.values()) / len(rows), 4)
		worst = round(min(row["score_worst"] for row in rows.values()), 4)  # the worst set against any opponent
		writer.add_scalar("eval_details/balanced_score", balanced, epoch)
		writer.add_scalar("eval_details/worst_set", worst, epoch)
		path = out / "eval_total.csv"
		new = not path.exists()
		with path.open("a", newline="") as f:
			if new:
				csv.writer(f).writerow(["epoch", "total_score", *(f"vs_{name}" for name in scores)])
			csv.writer(f).writerow([epoch, total, *scores.values()])
		writer.add_scalar(EVAL_TAB, total, epoch)
		writer.flush()
		print(f"eval  {epoch:4d}  TOTAL SCORE {total:+.3f}  (" + ", ".join(f"vs {name} {score:+.3f}" for name, score in scores.items()) + f")  balanced {balanced:+.3f}  worst set {worst:+.3f}", flush=True)

	def evaluate_against(epoch, config_for, baseline):
		row = eval_round(average.model if average else model, epoch, out, battle_map, config_for, baseline)
		if average and pool:  # the weights it just scored (eval.onnx) play the campaign evals' battles too
			staged = pool.models / "average.onnx.tmp"
			shutil.copyfile(out / "eval.onnx", staged)
			os.replace(staged, pool.models / "average.onnx")
			pool.eval_model = pool.models / "average.onnx"
		with (out / f"eval{eval_name(config_for, config.eval_opponent)}.csv").open("a", newline="") as f:
			csv.writer(f).writerow([row[c] for c in EVAL_COLUMNS])
		log_eval_tensorboard(writer, row, config_for.eval_opponent)
		if row["shooter_not_shooting"] > SHOOTER_IDLE_LIMIT:  # (nan compares false)
			print(f"eval  {epoch:4d}  FAILURE vs {config_for.eval_opponent}: shooters idle in {row['shooter_not_shooting']:.3f} of their full-damage chances, over {SHOOTER_IDLE_LIMIT} (greedy: the model or the metric is wrong)", flush=True)
		print(f"eval  {epoch:4d}  total score {row['total_score']:+.3f}  vs {config_for.eval_opponent}, greedy: attack won {row['attack_win_rate']:.2f} {row['attack_reward']:+.3f} ({row['attack_vs_baseline']:+.3f} vs its own)  defence won {row['defence_win_rate']:.2f} {row['defence_reward']:+.3f} ({row['defence_vs_baseline']:+.3f} vs its own)  {row['seconds']}s"
			f"  | sets: " + ", ".join(f"{k} {row[f'score_{k}']:+.3f}" for k in EVAL_SETS if row[f"score_{k}"] == row[f"score_{k}"]) +
			f"  | failures: escape ahead {row['escape_while_ahead']}, shooter idle {row['shooter_not_shooting']}, worthless cast {row['worthless_cast']}, cast skipped on loss {row['cast_skipped_on_loss']}"
			+ ("  | tricks: " + ", ".join(f"{k} {v:+.3f}" for k, v in row["hard_kinds"].items()) if row.get("hard_kinds") else ""), flush=True)
		return row

	pool = None  # (the actors start after the first eval: evaluate_against reads it)
	if config.eval_every and (first_epoch == 1 or (config.init_from and not resume)):
		evaluate_now(first_epoch - 1)  # the model as the run starts
	settings = (("ALPHA_BATTLE_SEARCH_SIMULATIONS", str(config.search_simulations)),) if config.search_simulations > 0 else ()

	def epoch_games(epoch):
		"""(the epoch's games, files to delete after they are played)"""
		if config.generator:  # new random battles every epoch, the same for a given (seed, epoch)
			rng = random.Random(1_000_003 * (config.seed + 1) + epoch)
			scenarios = [generator.sample(rng) for _ in range(config.games_per_side)]
			sides = (0,) if config.opponent == play.SELF_PLAY else (0, 1)
			games, files = generated_games(scenarios, f"epoch_{epoch}", config, our_sides=sides, opponent=config.opponent, settings=settings)
			return [replace(g, sampler_seed=2 * epoch * config.games_per_side + i) for i, g in enumerate(games)], files
		seeds = random.Random(epoch)
		seed_for = lambda side: seeds.randrange(1, 2**31) if config.random_seeds else config.game_seeds[side]
		if config.opponent == play.SELF_PLAY:  # each battle records both sides: half the battles with each side's seed
			return [
				play.Game(config.scenario, our_side=0, opponent=play.SELF_PLAY, game_seed=seed_for(g % 2), sampler_seed=2 * epoch * config.games_per_side + g, ai=config.ai, map=battle_map, settings=settings)
				for g in range(config.games_per_side)
			], []
		return [
			play.Game(config.scenario, our_side=side, opponent=config.opponent, game_seed=seed_for(side), sampler_seed=(2 * epoch + side) * config.games_per_side + g, ai=config.ai, map=battle_map, settings=settings)
			for side in (0, 1)
			for g in range(config.games_per_side)
		], []

	def play_epoch(epoch, model_path):
		record_root = out / "records" / f"epoch_{epoch}" if config.keep_records else None
		games, files = epoch_games(epoch)
		try:
			return play_with_retries(games, model_path, record_root, config, f"epoch {epoch}")
		finally:
			for path in files:
				path.unlink(missing_ok=True)

	# The model plays from one of two ONNX files: while battles run with one, the next export goes to the other
	model_files = [out / "model_a.onnx", out / "model_b.onnx"]
	current = 0
	pool = None
	if config.actors:  # actors play all the time; each update takes its battles from their buffer
		from .actors import ActorPool
		pool = ActorPool(config, out, generated_games, architecture, trained_epochs=first_epoch - 1)
		pool.publish(model, lambda path: export(cpu_copy(model), path))
		pool.start()
	campaign = None
	if config.campaign.games:  # the campaign model, trained alongside: its games' battles are our real battles
		if not pool or not pool.server:
			raise ValueError("campaign training needs actors=true and inference=server")
		from campaign_model.learner import CampaignLearner

		try:
			campaign = CampaignLearner(config.campaign, out, device if config.campaign.device == "auto" else torch.device(config.campaign.device), pool, resume=resume)
			if campaign.eval_due():  # the campaign model as the run starts, against Nullkiller2
				pool.pause()
				campaign.evaluate(first_epoch - 1)
				pool.resume()
			campaign.start()
		except BaseException:
			if campaign:
				campaign.close()
			pool.close()
			raise
	hub = None
	if config.remote_port and pool:  # workers on other machines
		from .remote import Hub

		hub = Hub(config, pool, campaign)
	if not pool:
		export(cpu_copy(model), model_files[current])
		producer = Producer(play_epoch)
		producer.start(first_epoch, model_files[current])
	# The run's time (run_hours, the learning rate's schedule, eval_minutes) on the monotonic clock: it stands still
	# while the machine sleeps (a closed lid stole 2.3 of v15-window's 11 hours and decayed its rate as if they ran)
	run_started, last_eval = time.monotonic(), time.monotonic()
	if resume and (out / "eval_total.csv").exists():  # the next eval eval_minutes after the last one, not after the resume
		last_eval -= time.time() - (out / "eval_total.csv").stat().st_mtime
	elapsed = lambda: elapsed_before + time.monotonic() - run_started
	try:
		for epoch in itertools.count(first_epoch):
			if (config.run_hours and elapsed() >= config.run_hours * 3600) or (not config.run_hours and epoch > config.epochs):
				break
			started = time.time()
			if config.lr_schedule == "linear":
				done = min(1.0, elapsed() / (config.run_hours * 3600)) if config.run_hours else (epoch - 1) / config.epochs
				for group in optimizer.param_groups:
					group["lr"] = config.learning_rate * (1 - (1 - config.lr_final_fraction) * done) * lr_scale
			elif config.target_kl:
				for group in optimizer.param_groups:
					group["lr"] = config.learning_rate * lr_scale
			actor_stats = {}
			if pool:
				records, actor_stats = pool.take(config.decisions_per_update)
			else:
				records = producer.take()
			waited = time.time() - started
			if not pool and config.prefetch and epoch < config.epochs:
				producer.start(epoch + 1, model_files[current])  # with the model before this epoch's update
			# How far the credit trusts the value head (its smoothed explained variance): the outcome's share falls with it,
			# and so does the round heads' share of what the policy maximizes (AdvantageConfig.policy_heads)
			a = config.advantage
			trust = min(1.0, max(0.0, (value_ev if value_ev is not None else 0.0) / a.outcome_share_full_ev))
			if advantage_config.credit == "window" and a.outcome_share_max > a.outcome_share:
				advantage_config = replace(advantage_config, outcome_share=a.outcome_share_max - (a.outcome_share_max - a.outcome_share) * trust)
			if a.fade_round_heads:
				advantage_config = replace(advantage_config, policy_heads=tuple(w * (1 - trust) for w in a.policy_heads[:-1]) + (a.policy_heads[-1],))
			loader, summaries, _ = make_dataloader(records, config.batch_size, True, reward_config, advantage_config)

			weights_before = flat_weights(model)
			parts_sum, batches = {key: torch.zeros((), device=device) for key in ("policy", "value", "entropy", "below_floor", "approx_kl", "clip_fraction", *(f"foresight_{k}" for k in FORESIGHT_PARTS))}, 0
			gradient_check = {}
			trained = time.time()
			model.train()
			examples = loader.dataset
			# Every decision of the epoch, moved to the GPU once, its int64 inputs as int32 (half the bytes to stack and to move:
			# 1.9 GB of a 3,200-decision update in int64), widened a minibatch at a time
			epoch_data = to_device(collate(examples, widen=False), device) if examples else None
			marks = [time.time()]
			# Every tensor shape on the GPU is fixed: MPS keeps a compiled graph per shape (~100 MB each, never freed), so
			# a last batch of a new size every epoch filled memory and swap. The last chunk repeats its last decision;
			# each pass leaves out its last partial batch (a different one each pass).
			if epoch_data is not None and hasattr(model, "facts"):  # the features: once per decision, not once per pass
				# Filled in place, a chunk at a time: a torch.cat of the chunks held the features twice at its peak (several
				# GB on Apple's GPU at ~3,200 decisions: the perf session's measurement, 2026-10-30)
				facts = None
				for i in range(0, len(examples), 256):
					chunk = model.facts(widened(select(epoch_data["inputs"], torch.arange(i, i + 256, device=device).clamp(max=len(examples) - 1))))
					if facts is None:
						facts = {key: value.new_empty((len(examples),) + tuple(value.shape[1:])) for key, value in chunk.items()}
					rows = min(256, len(examples) - i)
					for key, value in chunk.items():
						facts[key][i : i + rows] = value[:rows]
					del chunk
				epoch_data["facts"] = facts
			fresh_value = float("nan")
			if epoch_data is not None:
				# The KL control's anchor: this update's actions under the model as the update starts. Measured against the
				# policy that played (old_log_prob, about one version back), the previous update's own move counted as
				# this one's: 44 of v15-window's updates stopped after one step, most right after a full update
				# The same pass measures the value loss on these battles before the update learns from them (VALUE_LOSS_TAB)
				with torch.no_grad():
					anchors, excess = [], []
					for i in range(0, len(examples), 256):
						index = torch.arange(i, i + 256, device=device).clamp(max=len(examples) - 1)
						rows = select(epoch_data, index)
						out_rows = model(widened(rows["inputs"]), rows["facts"]) if "facts" in rows else model(widened(rows["inputs"]))
						anchors.append(log_prob(out_rows, rows["action"]))
						excess.append(value_excess(out_rows, rows["value_labels"], ppo_config))
					epoch_data["anchor_log_prob"] = torch.cat(anchors)[: len(examples)]
					fresh_value = float(weighted_mean(torch.cat(excess)[: len(examples)], epoch_data.get("weight")))
					del anchors, excess, out_rows
			sync(device)
			marks.append(time.time())
			stopped, update_kl, kl_sum, recent = False, 0.0, 0.0, deque(maxlen=KL_WINDOW)
			kl_of = lambda out, rows: (lambda d: (d.exp() - 1 - d).mean())(log_prob(out, rows["action"]).detach() - rows["anchor_log_prob"])  # k3: KL(anchor || now)
			for _ in range(config.ppo_passes if examples else 0):
				if stopped:
					break
				order = torch.randperm(len(examples)).to(device)
				if len(examples) >= config.batch_size:
					order = order[: len(examples) - len(examples) % config.batch_size]
				for first in range(0, len(order), config.batch_size):
					batch = select(epoch_data, order[first : first + config.batch_size])
					# The minibatch's actions as the GPU holds them, before any use: on Apple's GPU they were twice garbage
					# (action types of 30570, 10640) in a heavily swapping run, though valid on the CPU (data.check_record),
					# and indexing with them stopped the run; such an update is now dropped with a note instead
					if bool(((batch["action"]["type"] < 0) | (batch["action"]["type"] >= N_ACTION_TYPES)).any()):
						print(f"update {epoch}: a minibatch's actions are corrupt on the {device} (step {batches}): the rest of this update is dropped", flush=True)
						stopped = True
						break
					out_batch = model(widened(batch["inputs"]), batch["facts"]) if "facts" in batch else model(widened(batch["inputs"]))
					if config.search_simulations > 0 and "search" in batch:
						loss, parts = search_loss(out_batch, batch["search"], batch["value_labels"], ppo_config)
					else:
						loss, parts = ppo_loss(out_batch, batch["action"], batch["old_log_prob"], batch["advantage"], batch["value_labels"], ppo_config, batch.get("weight"))
					if "foresight" in batch and "foresight_unit" in out_batch:
						foresight, foresight_parts = foresight_loss(out_batch, batch["foresight"], ppo_config, batch.get("weight"))
						loss = loss + foresight
						parts.update({f"foresight_{k}": v for k, v in foresight_parts.items()})
						if batches == 0 and epoch % GRADIENT_CHECK_EVERY == 0:
							gradient_check = trunk_gradients(model, loss - foresight, foresight)
					optimizer.zero_grad()
					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
					optimizer.step()
					for key in parts_sum:
						if key in parts:
							parts_sum[key] += parts[key]  # summed on the device: no sync each step
					batches += 1
					if config.target_kl:
						kl = float(kl_of(out_batch, batch))  # before this step: how far the update has moved so far
						kl_sum += kl
						recent.append(kl)
						update_kl = sum(recent) / len(recent)  # the last KL_WINDOW minibatches': one's is noise
						if update_kl > 1.5 * config.target_kl:
							stopped = True  # this update has moved the policy far enough
							break
			model.eval()
			sync(device)
			marks.append(time.time())
			if average and batches:
				average.update(model)
			if config.target_kl and batches:
				if stopped:
					lr_scale = max(0.1, lr_scale / 1.5)
				elif update_kl < config.target_kl / 2:  # the update's last minibatches: nearest where it ended
					lr_scale = min(config.lr_scale_max, lr_scale * 1.5)
			epoch_data = batch = out_batch = loss = None  # (none of them may exist: an update with no minibatch)
			if device.type == "mps":
				# The update's tensors differ in size every update (its battles' decisions): MPS's caching allocator kept
				# blocks of every size, a 22 GB footprint on a 26 GB machine, which then swapped heavily (and twice
				# handed a training step garbage); give them back after each update
				torch.mps.empty_cache()
			if pool:
				pool.publish(model, lambda path: export(cpu_copy(model), path))
			else:
				current = 1 - current
				export(cpu_copy(model), model_files[current])
				if not config.prefetch and epoch < config.epochs:
					producer.start(epoch + 1, model_files[current])
			train_seconds = time.time() - trained
			marks.append(time.time())

			by_side = [[s for s in summaries if s.our_side == side] for side in (0, 1)]
			mean_of = lambda values: sum(values) / len(values) if values else float("nan")  # a side can be absent (generator: all battles against neutrals)
			mean = {key: float(value) / max(batches, 1) for key, value in parts_sum.items()}
			row = {
				"epoch": epoch,
				"mean_reward": round(to_scale(mean_of([s.reward for s in by_side[0] + by_side[1]]), reward_config), 4),
				"decisions": sum(s.decisions for s in summaries),
				"rounds": round(sum(mean_of([s.rounds for s in side]) for side in by_side), 2),
				"policy_loss": round(mean["policy"], 5),
				"value_loss": round(mean["value"], 5),
				"value_loss_fresh": round(fresh_value, 5),
				"entropy": round(mean["entropy"], 4),
				"below_floor": round(mean["below_floor"], 4),
				"approx_kl": round(mean["approx_kl"], 5),
				"clip_fraction": round(mean["clip_fraction"], 4),
				"signal_ratio": round(signal_ratio(optimizer), 4),
				"weight_change": round(float((flat_weights(model) - weights_before).norm()), 4),
				"seconds": round(time.time() - started, 1),
				"wait_seconds": round(waited, 1),
				"train_seconds": round(train_seconds, 1),
				**{f"{part}_seconds": round(b - a, 2) for part, a, b in zip(("data", "prepass", "steps", "publish"), [trained] + marks, marks)},
				**{k: actor_stats.get(k, "") for k in ("real_share", "games_in_update", "buffer_decisions", "record_age", "actor_failures", "dropped_stale", "dropped_overflow")},
				"learning_rate": round(optimizer.param_groups[0]["lr"], 8),
				"ppo_steps": batches,
				"update_kl": round(kl_sum / max(batches, 1), 5),  # KL(start of the update || during it), what the KL control uses
				"outcome_share": round(advantage_config.outcome_share, 3),
				"round_head_weight": round(float(advantage_config.policy_heads[2]), 3),
				**value_gains(model),
				**{f"train_{k}": v for k, v in failure_modes(records).items()},  # the same signs of dumb play, in training
				**{f"foresight_{k}": round(mean[f"foresight_{k}"], 5) for k in FORESIGHT_PARTS},
				**gradient_check,
			}
			for side, name in ((0, "attack"), (1, "defence")):
				kinds = Counter(ActionType(a["type"]).name for r in records if r.events[0]["our_side"] == side for a in r.actions)
				total = max(sum(kinds.values()), 1)
				for kind in ACTION_KINDS:
					row[f"{name}_share_{kind.lower()}"] = round(kinds[kind] / total, 4)
			if config.eval_every and epoch % config.eval_every == 0 and config.opponent == play.SELF_PLAY:
				write_traces(out, epoch, "selfplay", records[:2])  # one self-play battle, seen from both sides
			for name, side_summaries in (("attack", by_side[0]), ("defence", by_side[1])):
				raw = mean_of([s.reward for s in side_summaries])
				row[f"{name}_reward"] = round(to_scale(raw, reward_config), 4)
				row[f"{name}_raw_reward"] = round(raw, 4)
				row[f"{name}_win_rate"] = round(mean_of([float(s.outcome == Outcome.WIN) for s in side_summaries]), 4)
				row[f"{name}_outcome"] = outcome_text(side_summaries)
				row[f"{name}_advantage_std"] = round(mean_of([s.advantage_std for s in side_summaries]), 4)
				row[f"{name}_value_error"] = round(mean_of([s.value_error for s in side_summaries]), 4)
				ev, own_lost, killed = credit_check(side_summaries)
				row[f"{name}_value_ev"], row[f"{name}_credit_own_lost"], row[f"{name}_credit_killed"] = round(ev, 3), round(own_lost, 3), round(killed, 3)
			evs = [row[f"{name}_value_ev"] for name in ("attack", "defence") if row[f"{name}_value_ev"] == row[f"{name}_value_ev"]]
			if evs:  # the window credit's outcome share follows it (smoothed over ~10 updates)
				now = max(-1.0, min(1.0, sum(evs) / len(evs)))
				value_ev = now if value_ev is None else 0.9 * value_ev + 0.1 * now
			with progress.open("a", newline="") as f:
				csv.writer(f).writerow([row.get(c, "") for c in COLUMNS])  # (the gradient check's columns: every GRADIENT_CHECK_EVERY updates)
			log_tensorboard(writer, row, self_play=config.opponent == play.SELF_PLAY)
			print(
				f"epoch {epoch:4d}  attack {row['attack_outcome']:9s} {row['attack_reward']:+.3f}  defence {row['defence_outcome']:9s} {row['defence_reward']:+.3f}  won {row['attack_win_rate']:.2f} / {row['defence_win_rate']:.2f}"
				f"  mean {row['mean_reward']:+.3f}  decisions {row['decisions']:3d}"
				f"  signal {row['signal_ratio']}  {row['seconds']}s",
				flush=True,
			)

			if campaign and config.campaign.learner == "turns":
				campaign.step(epoch)  # an update of the campaign model when its games are in (its own optimizer)
			elif campaign:
				campaign.note_epoch(epoch)  # (its updates run in its own thread)
			due = time.monotonic() - last_eval >= config.eval_minutes * 60 if config.eval_minutes else config.eval_every and epoch % config.eval_every == 0
			if due:
				if pool:
					pool.pause()
				evaluate_now(epoch)
				if pool:
					pool.resume()
				last_eval = time.monotonic()
				save_checkpoint(out / f"checkpoint_{epoch:05d}.pt", model, optimizer, epoch, config, elapsed(), lr_scale, value_ev, average)  # every evaluated model
			if campaign and campaign.eval_due():  # against Nullkiller2, with the battle actors paused
				pool.pause()
				campaign.evaluate(epoch)
				pool.resume()
			save_checkpoint(latest, model, optimizer, epoch, config, elapsed(), lr_scale, value_ev, average)
			if config.checkpoint_every and epoch % config.checkpoint_every == 0:
				save_checkpoint(out / f"checkpoint_{epoch:04d}.pt", model, optimizer, epoch, config, elapsed(), lr_scale, value_ev, average)
	finally:
		if hub:
			hub.close()
		if campaign:
			campaign.close()
		if pool:
			pool.close()


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
	parser.add_argument("--run", default=TrainConfig.run)
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--follow", metavar="RUN", help="mirror a running run's progress.csv into TensorBoard instead of training")
	parser.add_argument("--machine", metavar="TOML", help="this machine's sizes (configs/machines/<machine>/<case>.toml), applied before --set")
	parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="override TrainConfig fields; ppo.NAME, advantage.NAME, reward.NAME, campaign.NAME for the sub-configs")
	args = parser.parse_args()
	if args.follow:
		follow(args.follow)
	else:
		machine = machine_overrides(args.machine) if args.machine else []
		train(with_overrides(TrainConfig(epochs=args.epochs, run=args.run), machine + args.set), resume=args.resume)


if __name__ == "__main__":
	main()
