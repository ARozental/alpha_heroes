"""Training loop: play battles against a fixed opponent, learn from them with PPO, repeat.

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
  5. every eval_every epochs (and before the first), an eval round, not trained on: games_per_side
     sampled battles per side plus one greedy battle per side against eval_opponent (BattleAI) on the
     fixed game_seeds, logged to <run dir>/eval.csv and under eval/ in TensorBoard.

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
import csv
import random
import shutil
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from . import play
from .data import collate, make_dataloader, summarize
from .export import export
from .losses import AdvantageConfig, PPOConfig, ppo_loss
from .model import BattleModel
from .reward import Outcome, RewardConfig, to_scale

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TrainConfig:
	scenario: str = str(ROOT / "battle_model" / "scenarios" / "castle_vs_inferno.json")
	run: str = "castle_vs_inferno"
	epochs: int = 1000
	opponent: str = "BattleAI"  # who we train against; "self" (play.SELF_PLAY): our model on both sides, learning from both
	ai: str = "AlphaBattleAI"  # play.Game.ai
	eval_every: int = 10  # epochs between eval rounds against eval_opponent; 0 = none
	eval_opponent: str = "BattleAI"
	game_seeds: tuple = (1, 2)  # VCMI seed of the attack battles and of the defence battles: fixed, so the test is about learning, not luck
	games_per_side: int = 8  # battles per side per epoch, played in parallel; 1 (~40 decisions an update) was far too noisy
	# Give every battle its own random VCMI seed instead of game_seeds (dice, obstacles and the opponent's
	# choices differ): learning a strategy rather than one battle.
	random_seeds: bool = False
	learning_rate: float = 1e-4  # Adam; 3e-4 (the usual PPO default) learned and then forgot this scenario
	lr_schedule: str = "constant"  # "linear": decay to 0 at the last epoch, as standard PPO does; Adam's steps stay ~lr even when the gradients are only noise
	ppo_passes: int = 4  # optimisation passes over each epoch's decisions; PPO reuses its data a few times
	batch_size: int = 64  # minibatch of decisions; an epoch of 16 battles is ~300 decisions
	grad_clip: float = 0.5  # max gradient norm, the usual PPO value
	checkpoint_every: int = 50
	attempts: int = 3  # tries per epoch when a battle fails or stalls before the run stops
	keep_records: bool = False  # also write every battle record to <run dir>/records (debugging; ~0.3 MB a decision on disk)
	device: str = "auto"  # where torch trains: "auto" = the GPU (cuda, Apple mps) if there is one, else "cpu"; the CPU plays the battles
	# Play the next epoch's battles while this epoch trains (Producer): the battles are then one update older than the
	# model they train, which PPO handles (each record holds the probabilities of the model that played it)
	prefetch: bool = True
	model_dim: int = 64
	model_layers: int = 2
	model_mask_features: bool = False  # model.BattleModel(mask_features=...)
	seed: int = 0  # the model's initial weights
	reward: RewardConfig = field(default_factory=RewardConfig)
	advantage: AdvantageConfig = field(default_factory=AdvantageConfig)
	ppo: PPOConfig = field(default_factory=PPOConfig)


def with_overrides(config, assignments):
	"""config with "name=value" assignments applied; "ppo.name", "advantage.name", "reward.name" set the sub-configs."""
	for assignment in assignments:
		key, value = assignment.split("=", 1)
		target, name = key.split(".", 1) if "." in key else (None, key)
		owner = getattr(config, target) if target else config
		kind = {f.name: f.type for f in fields(owner)}[name]
		kind = {"int": int, "float": float, "str": str, "bool": lambda v: v.lower() in ("1", "true", "yes"), "tuple": lambda v: tuple(int(x) if x.lstrip("-").isdigit() else float(x) for x in v.split(","))}.get(getattr(kind, "__name__", kind), kind)
		owner = replace(owner, **{name: kind(value)})
		config = replace(config, **{target: owner}) if target else owner
	return config


def run_dir(config):
	return ROOT / "runtime" / "train" / config.run


COLUMNS = [
	"epoch", "attack_reward", "attack_outcome", "defence_reward", "defence_outcome", "mean_reward", "attack_raw_reward", "defence_raw_reward", "attack_win_rate", "defence_win_rate",
	"decisions", "rounds", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "signal_ratio", "weight_change",
	"attack_advantage_std", "defence_advantage_std", "attack_value_error", "defence_value_error", "seconds", "wait_seconds", "train_seconds",
]  # progress.csv. Per side, over its games_per_side battles: mean reward (-1..1 scale), mean raw reward (reward
# units), win rate, outcome (its name for one battle, else counts like "WIN:3 LOSS:5"), mean advantage std and value
# error. decisions: all decisions of the epoch; rounds: attack + defence battle rounds, averaged over the games.

# progress.csv column -> TensorBoard tag
TENSORBOARD_TAGS = {
	"mean_reward": "reward/mean", "attack_reward": "reward/attack", "defence_reward": "reward/defence",
	"attack_raw_reward": "raw_reward/attack", "defence_raw_reward": "raw_reward/defence",
	"attack_win_rate": "won/attack", "defence_win_rate": "won/defence",
	"policy_loss": "loss/policy", "value_loss": "loss/value", "entropy": "loss/entropy",
	"signal_ratio": "stability/signal_ratio", "weight_change": "stability/weight_change", "approx_kl": "stability/approx_kl", "clip_fraction": "stability/clip_fraction",
	"attack_advantage_std": "advantage_std/attack", "defence_advantage_std": "advantage_std/defence",
	"attack_value_error": "value_error/attack", "defence_value_error": "value_error/defence",
	"decisions": "battle/decisions", "rounds": "battle/rounds", "seconds": "time/epoch_seconds",
	"wait_seconds": "time/waiting_for_battles_seconds", "train_seconds": "time/training_seconds",
}


EVAL_COLUMNS = [
	"epoch", "attack_win_rate", "defence_win_rate", "attack_reward", "defence_reward", "mean_reward", "attack_outcome", "defence_outcome",
	"attack_greedy_outcome", "defence_greedy_outcome", "attack_greedy_reward", "defence_greedy_reward", "seconds",
]  # eval.csv: sampled battles (win rate, mean reward -1..1, outcomes) and the one greedy battle, per side


def log_eval_tensorboard(writer, row):
	epoch = int(row["epoch"])
	for name in ("attack", "defence"):
		writer.add_scalar(f"eval/won/{name}", float(row[f"{name}_win_rate"]), epoch)
		writer.add_scalar(f"eval/reward/{name}", float(row[f"{name}_reward"]), epoch)
		writer.add_scalar(f"eval/won_greedy/{name}", float(row[f"{name}_greedy_outcome"] == "WIN"), epoch)
		writer.add_scalar(f"eval/reward_greedy/{name}", float(row[f"{name}_greedy_reward"]), epoch)
	writer.add_scalar("eval/reward/mean", float(row["mean_reward"]), epoch)
	writer.flush()


def log_tensorboard(writer, row):
	"""One epoch's progress row as TensorBoard scalars; empty columns are skipped."""
	epoch = int(row["epoch"])
	for column, tag in TENSORBOARD_TAGS.items():
		if row.get(column) not in (None, ""):
			writer.add_scalar(tag, float(row[column]), epoch)
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


def eval_round(model, epoch, out, battle_map, config):
	"""Play the eval battles against config.eval_opponent with the current model; the eval.csv row."""
	started = time.time()
	model_path = out / "eval.onnx"
	export(cpu_copy(model), model_path)
	n = config.games_per_side
	games = [
		play.Game(config.scenario, our_side=side, opponent=config.eval_opponent, game_seed=config.game_seeds[side], sampler_seed=10**7 + (2 * epoch + side) * (n + 1) + g, temperature=0.0 if g == n else 1.0, ai=config.ai, map=battle_map)
		for side in (0, 1)
		for g in range(n + 1)
	]  # g == n: the greedy battle
	record_root = out / "records" / f"eval_{epoch}" if config.keep_records else None
	summaries = [summarize(r.events) for r in play_with_retries(games, model_path, record_root, config, f"eval {epoch}")]
	row = {"epoch": epoch}
	sampled_all = []
	for side, name in ((0, "attack"), (1, "defence")):
		sampled = [s for game, s in zip(games, summaries) if game.our_side == side and game.temperature > 0]
		greedy = next(s for game, s in zip(games, summaries) if game.our_side == side and game.temperature == 0)
		sampled_all += sampled
		row[f"{name}_win_rate"] = round(sum(s.outcome == Outcome.WIN for s in sampled) / len(sampled), 4)
		row[f"{name}_reward"] = round(to_scale(sum(s.reward for s in sampled) / len(sampled), config.reward), 4)
		row[f"{name}_outcome"] = outcome_text(sampled)
		row[f"{name}_greedy_outcome"] = greedy.outcome.name
		row[f"{name}_greedy_reward"] = round(to_scale(greedy.reward, config.reward), 4)
	row["mean_reward"] = round(to_scale(sum(s.reward for s in sampled_all) / len(sampled_all), config.reward), 4)
	row["seconds"] = round(time.time() - started, 1)
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
	return {"dim": config.model_dim, "layers": config.model_layers, "mask_features": config.model_mask_features}


def save_checkpoint(path, model, optimizer, epoch, config):
	torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "model_kwargs": model_kwargs(config)}, path)


def train(config, resume=False):
	out = run_dir(config)
	out.mkdir(parents=True, exist_ok=True)
	(out / "config.txt").write_text("\n".join(f"{k} = {v}" for k, v in asdict(config).items()) + "\n")
	torch.manual_seed(config.seed)
	device = resolve_device(config.device)
	model = BattleModel(**model_kwargs(config)).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
	first_epoch = 1
	latest = out / "checkpoint_latest.pt"
	if resume and latest.exists():
		state = torch.load(latest, map_location=device)
		model.load_state_dict(state["model"])
		optimizer.load_state_dict(state["optimizer"])
		first_epoch = state["epoch"] + 1
		print(f"resuming after epoch {state['epoch']}")

	battle_map = play.build_map(config.scenario, variant=config.run)
	reward_config, advantage_config, ppo_config = config.reward, config.advantage, config.ppo
	progress = out / "progress.csv"
	if not progress.exists() or not resume:
		with progress.open("w", newline="") as f:
			csv.writer(f).writerow(COLUMNS)
	else:
		upgrade_csv(progress, COLUMNS)
	writer = SummaryWriter(out / "tensorboard" / time.strftime("%Y%m%d-%H%M%S"))
	eval_csv = out / "eval.csv"
	if not eval_csv.exists() or not resume:
		with eval_csv.open("w", newline="") as f:
			csv.writer(f).writerow(EVAL_COLUMNS)
	else:
		upgrade_csv(eval_csv, EVAL_COLUMNS)

	def evaluate_now(epoch):
		row = eval_round(model, epoch, out, battle_map, config)
		with eval_csv.open("a", newline="") as f:
			csv.writer(f).writerow([row[c] for c in EVAL_COLUMNS])
		log_eval_tensorboard(writer, row)
		print(f"eval  {epoch:4d}  vs {config.eval_opponent}: attack won {row['attack_win_rate']:.2f} (greedy {row['attack_greedy_outcome']})  defence won {row['defence_win_rate']:.2f} (greedy {row['defence_greedy_outcome']})  {row['seconds']}s", flush=True)

	if config.eval_every and first_epoch == 1:
		evaluate_now(0)  # the untrained model
	def epoch_games(epoch):
		seeds = random.Random(epoch)
		seed_for = lambda side: seeds.randrange(1, 2**31) if config.random_seeds else config.game_seeds[side]
		if config.opponent == play.SELF_PLAY:  # each battle records both sides: half the battles with each side's seed
			return [
				play.Game(config.scenario, our_side=0, opponent=play.SELF_PLAY, game_seed=seed_for(g % 2), sampler_seed=2 * epoch * config.games_per_side + g, ai=config.ai, map=battle_map)
				for g in range(config.games_per_side)
			]
		return [
			play.Game(config.scenario, our_side=side, opponent=config.opponent, game_seed=seed_for(side), sampler_seed=(2 * epoch + side) * config.games_per_side + g, ai=config.ai, map=battle_map)
			for side in (0, 1)
			for g in range(config.games_per_side)
		]

	def play_epoch(epoch, model_path):
		record_root = out / "records" / f"epoch_{epoch}" if config.keep_records else None
		return play_with_retries(epoch_games(epoch), model_path, record_root, config, f"epoch {epoch}")

	# The model plays from one of two ONNX files: while battles run with one, the next export goes to the other
	model_files = [out / "model_a.onnx", out / "model_b.onnx"]
	current = 0
	export(cpu_copy(model), model_files[current])
	producer = Producer(play_epoch)
	producer.start(first_epoch, model_files[current])
	for epoch in range(first_epoch, config.epochs + 1):
		started = time.time()
		if config.lr_schedule == "linear":
			for group in optimizer.param_groups:
				group["lr"] = config.learning_rate * (1 - (epoch - 1) / config.epochs)
		records = producer.take()
		waited = time.time() - started
		if config.prefetch and epoch < config.epochs:
			producer.start(epoch + 1, model_files[current])  # with the model before this epoch's update
		loader, summaries, _ = make_dataloader(records, config.batch_size, True, reward_config, advantage_config)

		weights_before = flat_weights(model)
		parts_sum, batches = {key: torch.zeros((), device=device) for key in ("policy", "value", "entropy", "approx_kl", "clip_fraction")}, 0
		trained = time.time()
		model.train()
		examples = loader.dataset
		epoch_data = to_device(collate(examples), device) if examples else None  # every decision of the epoch, moved to the GPU once
		for _ in range(config.ppo_passes if examples else 0):
			order = torch.randperm(len(examples)).to(device)
			for first in range(0, len(examples), config.batch_size):
				batch = select(epoch_data, order[first : first + config.batch_size])
				out_batch = model(batch["inputs"])
				loss, parts = ppo_loss(out_batch, batch["action"], batch["old_log_prob"], batch["advantage"], batch["value_labels"], ppo_config)
				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
				optimizer.step()
				for key in parts_sum:
					parts_sum[key] += parts[key]  # summed on the device: no sync each step
				batches += 1
		model.eval()
		current = 1 - current
		export(cpu_copy(model), model_files[current])
		if not config.prefetch and epoch < config.epochs:
			producer.start(epoch + 1, model_files[current])
		train_seconds = time.time() - trained

		by_side = [[s for s in summaries if s.our_side == side] for side in (0, 1)]
		average = lambda values: sum(values) / len(values)
		mean = {key: float(value) / max(batches, 1) for key, value in parts_sum.items()}
		row = {
			"epoch": epoch,
			"mean_reward": round(to_scale(average([s.reward for s in by_side[0] + by_side[1]]), reward_config), 4),
			"decisions": sum(s.decisions for s in summaries),
			"rounds": round(sum(average([s.rounds for s in side]) for side in by_side), 2),
			"policy_loss": round(mean["policy"], 5),
			"value_loss": round(mean["value"], 5),
			"entropy": round(mean["entropy"], 4),
			"approx_kl": round(mean["approx_kl"], 5),
			"clip_fraction": round(mean["clip_fraction"], 4),
			"signal_ratio": round(signal_ratio(optimizer), 4),
			"weight_change": round(float((flat_weights(model) - weights_before).norm()), 4),
			"seconds": round(time.time() - started, 1),
			"wait_seconds": round(waited, 1),
			"train_seconds": round(train_seconds, 1),
		}
		for name, side_summaries in (("attack", by_side[0]), ("defence", by_side[1])):
			raw = average([s.reward for s in side_summaries])
			row[f"{name}_reward"] = round(to_scale(raw, reward_config), 4)
			row[f"{name}_raw_reward"] = round(raw, 4)
			row[f"{name}_win_rate"] = round(average([float(s.outcome == Outcome.WIN) for s in side_summaries]), 4)
			row[f"{name}_outcome"] = outcome_text(side_summaries)
			row[f"{name}_advantage_std"] = round(average([s.advantage_std for s in side_summaries]), 4)
			row[f"{name}_value_error"] = round(average([s.value_error for s in side_summaries]), 4)
		with progress.open("a", newline="") as f:
			csv.writer(f).writerow([row[c] for c in COLUMNS])
		log_tensorboard(writer, row)
		print(
			f"epoch {epoch:4d}  attack {row['attack_outcome']:9s} {row['attack_reward']:+.3f}  defence {row['defence_outcome']:9s} {row['defence_reward']:+.3f}  won {row['attack_win_rate']:.2f} / {row['defence_win_rate']:.2f}"
			f"  mean {row['mean_reward']:+.3f}  decisions {row['decisions']:3d}"
			f"  signal {row['signal_ratio']}  {row['seconds']}s",
			flush=True,
		)

		if config.eval_every and epoch % config.eval_every == 0:
			evaluate_now(epoch)
		save_checkpoint(latest, model, optimizer, epoch, config)
		if epoch % config.checkpoint_every == 0:
			save_checkpoint(out / f"checkpoint_{epoch:04d}.pt", model, optimizer, epoch, config)


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
	parser.add_argument("--run", default=TrainConfig.run)
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--follow", metavar="RUN", help="mirror a running run's progress.csv into TensorBoard instead of training")
	parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="override TrainConfig fields; ppo.NAME, advantage.NAME, reward.NAME for the sub-configs")
	args = parser.parse_args()
	if args.follow:
		follow(args.follow)
	else:
		train(with_overrides(TrainConfig(epochs=args.epochs, run=args.run), args.set), resume=args.resume)


if __name__ == "__main__":
	main()
