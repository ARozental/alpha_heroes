"""Training loop: play battles against a fixed opponent, learn from them with PPO, repeat.

	.venv/bin/python -m battle_model.train [--epochs N] [--run NAME] [--resume] [--set games_per_side=8 ppo.entropy_weight=0.02 ...]
	.venv/bin/python -m battle_model.train --follow NAME   # mirror a running run's progress.csv into TensorBoard
	.venv/bin/tensorboard --logdir runtime/train            # then open http://localhost:6006

Each epoch:
  1. export the current model to ONNX (the C++ battle AI plays with it),
  2. play games_per_side battles per scenario side, all in parallel (we attack with sides[0], we defend
     with sides[1]), with VCMI's seed fixed per side, so only our own (sampled) choices differ between
     the battles of a side and between epochs,
  3. turn the records into training examples (data.py) and update the model with PPO (losses.py),
  4. log both battles' rewards and outcomes to <run dir>/progress.csv and TensorBoard
     (<run dir>/tensorboard/<start time>: one folder per start, so a restarted run never mixes with old
     data a running TensorBoard still holds in memory), delete the records.

Rewards are logged on the -1..1 scale (reward.to_scale: -1 = worst possible, +1 = best possible), the
raw reward next to them.

Warning signs, logged every epoch. The lr 3e-4 run learned to win the attack, then forgot it; these
showed it coming:
  best_line_logp   log-probability that the current model replays each side's best battle so far
                   (the battles are deterministic given our actions). It fell for ~100 epochs
                   before the attack wins collapsed.
  signal_ratio     median |m| / sqrt(v) in Adam's state: near 0 the recent gradients cancel out, and
                   Adam's normalised steps are mostly noise. It halved around each collapse.
  weight_change    how far the weights moved this epoch. Adam keeps it ~constant even when nothing is
                   left to learn: that drift is what forgets.
  approx_kl, clip_fraction   how far each epoch's update moved the policy (PPO's usual checks)
  advantage_std, value_error per side: how much there is to learn, how wrong V is

The first test: the untrained model should lose in epoch 1, and the model should overfit this fixed
scenario and win by epoch 1000. If it needs more, something is probably wrong.
"""
import argparse
import csv
import random
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from . import play
from .data import collate, make_dataloader
from .distributions import log_prob
from .export import export
from .losses import AdvantageConfig, PPOConfig, ppo_loss, self_imitation_loss
from .model import BattleModel
from .reward import Outcome, RewardConfig, to_scale

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TrainConfig:
	scenario: str = str(ROOT / "battle_model" / "scenarios" / "castle_vs_inferno.json")
	run: str = "castle_vs_inferno"
	epochs: int = 1000
	opponent: str = "BattleAI"
	ai: str = "AlphaBattleAI"  # play.Game.ai
	game_seeds: tuple = (1, 2)  # VCMI seed of the attack battles and of the defence battles: fixed, so the test is about learning, not luck
	games_per_side: int = 1  # battles per side per epoch; they differ only by our sampling. More = less noisy updates
	# Give every battle its own random VCMI seed instead of game_seeds (dice, obstacles and the opponent's
	# choices differ): learning a strategy rather than one battle. Not with explore_from_best, whose replayed
	# openings are only valid for the seed they were recorded with.
	random_seeds: bool = False
	# Of each side's games, how many first replay a random-length opening of that side's best battle so far and
	# play on from there (Go-Explore: battles are deterministic, so this restores the state). They are trained
	# on only from where the model takes over, and are not counted in the logged rewards and win rates.
	explore_from_best: int = 0
	learning_rate: float = 1e-4  # Adam; 3e-4 (the usual PPO default) learned and then forgot this scenario
	lr_schedule: str = "constant"  # "linear": decay to 0 at the last epoch, as standard PPO does; Adam's steps stay ~lr even when the gradients are only noise
	ppo_passes: int = 4  # optimisation passes over each epoch's decisions; PPO reuses its data a few times
	batch_size: int = 64  # an epoch is only ~20-80 decisions, so usually one batch
	grad_clip: float = 0.5  # max gradient norm, the usual PPO value
	checkpoint_every: int = 50
	attempts: int = 3  # tries per epoch when a battle fails or stalls before the run stops
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
	"attack_advantage_std", "defence_advantage_std", "attack_value_error", "defence_value_error", "attack_best_line_logp", "defence_best_line_logp", "seconds",
	"attack_restart_win_rate", "defence_restart_win_rate", "attack_best_reward", "defence_best_reward",
]  # progress.csv. Per side, over its games_per_side battles: mean reward (-1..1 scale), mean raw reward (reward
# units), win rate, outcome (its name for one battle, else counts like "WIN:3 LOSS:5"), mean advantage std and value
# error. decisions: all decisions of the epoch; rounds: attack + defence battle rounds, averaged over the games.

# progress.csv column -> TensorBoard tag
TENSORBOARD_TAGS = {
	"mean_reward": "reward/mean", "attack_reward": "reward/attack", "defence_reward": "reward/defence",
	"attack_raw_reward": "raw_reward/attack", "defence_raw_reward": "raw_reward/defence",
	"attack_win_rate": "won/attack", "defence_win_rate": "won/defence",
	"attack_restart_win_rate": "won_after_restart/attack", "defence_restart_win_rate": "won_after_restart/defence",
	"attack_best_reward": "best_reward/attack", "defence_best_reward": "best_reward/defence",
	"policy_loss": "loss/policy", "value_loss": "loss/value", "entropy": "loss/entropy",
	"attack_best_line_logp": "best_line_logp/attack", "defence_best_line_logp": "best_line_logp/defence",
	"signal_ratio": "stability/signal_ratio", "weight_change": "stability/weight_change", "approx_kl": "stability/approx_kl", "clip_fraction": "stability/clip_fraction",
	"attack_advantage_std": "advantage_std/attack", "defence_advantage_std": "advantage_std/defence",
	"attack_value_error": "value_error/attack", "defence_value_error": "value_error/defence",
	"decisions": "battle/decisions", "rounds": "battle/rounds", "seconds": "battle/seconds",
}


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


def line_log_prob(model, line):
	"""Log-probability that the model makes all the decisions of a recorded battle (a collated batch)."""
	with torch.no_grad():
		return float(log_prob(model(line["inputs"]), line["action"]).sum())


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


def model_kwargs(config):
	return {"dim": config.model_dim, "layers": config.model_layers, "mask_features": config.model_mask_features}


def save_checkpoint(path, model, optimizer, epoch, config):
	torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "model_kwargs": model_kwargs(config)}, path)


def train(config, resume=False):
	if config.explore_from_best >= config.games_per_side:
		raise ValueError("explore_from_best must leave at least one game per side that starts from the beginning")
	if config.random_seeds and config.explore_from_best:
		raise ValueError("explore_from_best replays openings recorded with one VCMI seed: it needs fixed game_seeds")
	out = run_dir(config)
	out.mkdir(parents=True, exist_ok=True)
	(out / "config.txt").write_text("\n".join(f"{k} = {v}" for k, v in asdict(config).items()) + "\n")
	torch.manual_seed(config.seed)
	model = BattleModel(**model_kwargs(config))
	optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
	first_epoch = 1
	latest = out / "checkpoint_latest.pt"
	if resume and latest.exists():
		state = torch.load(latest)
		model.load_state_dict(state["model"])
		optimizer.load_state_dict(state["optimizer"])
		first_epoch = state["epoch"] + 1
		print(f"resuming after epoch {state['epoch']}")

	# Each side's best battle so far: {"reward", "epoch", "line"} (line = its decisions, collated).
	# Saved apart from the checkpoints, only when it changes (it holds the dense inputs, a few MB).
	best_path = out / "best_lines.pt"
	best = torch.load(best_path) if resume and best_path.exists() else [None, None]

	battle_map = play.build_map(config.scenario, variant=config.run)
	reward_config, advantage_config, ppo_config = config.reward, config.advantage, config.ppo
	progress = out / "progress.csv"
	if not progress.exists() or not resume:
		with progress.open("w", newline="") as f:
			csv.writer(f).writerow(COLUMNS)
	writer = SummaryWriter(out / "tensorboard" / time.strftime("%Y%m%d-%H%M%S"))

	for epoch in range(first_epoch, config.epochs + 1):
		started = time.time()
		if config.lr_schedule == "linear":
			for group in optimizer.param_groups:
				group["lr"] = config.learning_rate * (1 - (epoch - 1) / config.epochs)
		model_path = out / "model.onnx"
		export(model, model_path)

		openings = random.Random(epoch)
		games, restarted = [], []
		for side in (0, 1):
			for g in range(config.games_per_side):
				prefix = ()
				if g < config.explore_from_best and best[side] is not None and len(best[side].get("actions", ())) > 1:
					prefix = tuple(best[side]["actions"][: openings.randrange(1, len(best[side]["actions"]))])
				game_seed = openings.randrange(1, 2**31) if config.random_seeds else config.game_seeds[side]
				games.append(play.Game(config.scenario, our_side=side, opponent=config.opponent, game_seed=game_seed, sampler_seed=(2 * epoch + side) * config.games_per_side + g, ai=config.ai, prefix=prefix, map=battle_map))
				restarted.append(bool(prefix))
		records = out / "records" / f"epoch_{epoch}"
		for attempt in range(1, config.attempts + 1):
			try:
				shutil.rmtree(records, ignore_errors=True)
				dirs = play.play(games, model_path, records)
				loader, summaries, lines = make_dataloader(dirs, config.batch_size, True, reward_config, advantage_config)
				break
			except RuntimeError as error:  # a battle failed or stalled; its runner folder is kept for debugging
				print(f"epoch {epoch}: attempt {attempt} failed: {str(error)[:300]}", flush=True)
				if attempt == config.attempts:
					raise

		for game, summary, decisions in zip(games, summaries, lines):
			side = game.our_side
			if decisions and (best[side] is None or summary.reward > best[side]["reward"]):
				actions = [{k: v for k, v in e["action"].items() if k != "forced"} for e in decisions]
				best[side] = {"reward": summary.reward, "epoch": epoch, "line": collate(decisions), "actions": actions}
				torch.save(best, best_path)

		weights_before = flat_weights(model)
		parts_sum, batches = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}, 0
		model.train()
		for _ in range(config.ppo_passes):
			for batch in loader:
				out_batch = model(batch["inputs"])
				loss, parts = ppo_loss(out_batch, batch["action"], batch["old_log_prob"], batch["advantage"], batch["value_labels"], ppo_config)
				if ppo_config.self_imitation_weight > 0:
					for line in best:
						if line:
							imitation, _ = self_imitation_loss(model(line["line"]["inputs"]), line["line"]["action"], to_scale(line["reward"], reward_config), ppo_config)
							loss = loss + imitation
				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
				optimizer.step()
				for key in parts_sum:
					parts_sum[key] += float(parts[key])
				batches += 1
		model.eval()

		by_side = [[s for game, s, r in zip(games, summaries, restarted) if game.our_side == side and not r] for side in (0, 1)]
		restarted_by_side = [[s for game, s, r in zip(games, summaries, restarted) if game.our_side == side and r] for side in (0, 1)]
		average = lambda values: sum(values) / len(values)
		mean = {key: value / max(batches, 1) for key, value in parts_sum.items()}
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
		}
		for name, side_summaries, line in (("attack", by_side[0], best[0]), ("defence", by_side[1], best[1])):
			raw = average([s.reward for s in side_summaries])
			row[f"{name}_reward"] = round(to_scale(raw, reward_config), 4)
			row[f"{name}_raw_reward"] = round(raw, 4)
			row[f"{name}_win_rate"] = round(average([float(s.outcome == Outcome.WIN) for s in side_summaries]), 4)
			row[f"{name}_outcome"] = outcome_text(side_summaries)
			row[f"{name}_advantage_std"] = round(average([s.advantage_std for s in side_summaries]), 4)
			row[f"{name}_value_error"] = round(average([s.value_error for s in side_summaries]), 4)
			row[f"{name}_best_line_logp"] = round(line_log_prob(model, line["line"]), 3) if line else ""
			row[f"{name}_best_reward"] = round(to_scale(line["reward"], reward_config), 4) if line else ""
		for name, side_summaries in (("attack", restarted_by_side[0]), ("defence", restarted_by_side[1])):
			row[f"{name}_restart_win_rate"] = round(average([float(s.outcome == Outcome.WIN) for s in side_summaries]), 4) if side_summaries else ""
		with progress.open("a", newline="") as f:
			csv.writer(f).writerow([row[c] for c in COLUMNS])
		log_tensorboard(writer, row)
		print(
			f"epoch {epoch:4d}  attack {row['attack_outcome']:9s} {row['attack_reward']:+.3f}  defence {row['defence_outcome']:9s} {row['defence_reward']:+.3f}  won {row['attack_win_rate']:.2f} / {row['defence_win_rate']:.2f}"
			f"  mean {row['mean_reward']:+.3f}  decisions {row['decisions']:3d}  best-line logp {row['attack_best_line_logp']} / {row['defence_best_line_logp']}"
			f"  signal {row['signal_ratio']}  {row['seconds']}s",
			flush=True,
		)

		shutil.rmtree(records, ignore_errors=True)
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
