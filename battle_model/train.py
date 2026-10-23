"""Training loop: play battles against a fixed opponent, learn from them with PPO, repeat.

	.venv/bin/python -m battle_model.train [--epochs N] [--run NAME] [--resume]
	.venv/bin/python -m battle_model.train --follow NAME   # mirror a running run's progress.csv into TensorBoard
	.venv/bin/tensorboard --logdir runtime/train            # then open http://localhost:6006

Each epoch:
  1. export the current model to ONNX (the C++ battle AI plays with it),
  2. play one battle per scenario side in parallel (we attack with sides[0], we defend with sides[1]),
     with VCMI's seed fixed per side, so only our own choices change between epochs,
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
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from . import play
from .data import collate, make_dataloader
from .distributions import log_prob
from .export import export
from .losses import AdvantageConfig, PPOConfig, ppo_loss
from .model import BattleModel
from .reward import RewardConfig, to_scale

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TrainConfig:
	scenario: str = str(ROOT / "battle_model" / "scenarios" / "castle_vs_inferno.json")
	run: str = "castle_vs_inferno"
	epochs: int = 1000
	opponent: str = "BattleAI"
	game_seeds: tuple = (1, 2)  # VCMI seed of the attack battle and of the defence battle: fixed, so the test is about learning, not luck
	learning_rate: float = 1e-4  # Adam; 3e-4 (the usual PPO default) learned and then forgot this scenario
	ppo_passes: int = 4  # optimisation passes over each epoch's decisions; PPO reuses its data a few times
	batch_size: int = 64  # an epoch is only ~20-80 decisions, so usually one batch
	grad_clip: float = 0.5  # max gradient norm, the usual PPO value
	checkpoint_every: int = 50
	attempts: int = 3  # tries per epoch when a battle fails or stalls before the run stops
	model_dim: int = 64
	model_layers: int = 2
	seed: int = 0  # the model's initial weights


def run_dir(config):
	return ROOT / "runtime" / "train" / config.run


COLUMNS = [
	"epoch", "attack_reward", "attack_outcome", "defence_reward", "defence_outcome", "mean_reward", "attack_raw_reward", "defence_raw_reward",
	"decisions", "rounds", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "signal_ratio", "weight_change",
	"attack_advantage_std", "defence_advantage_std", "attack_value_error", "defence_value_error", "attack_best_line_logp", "defence_best_line_logp", "seconds",
]  # progress.csv; rewards on the -1..1 scale, raw_reward in reward units

# progress.csv column -> TensorBoard tag
TENSORBOARD_TAGS = {
	"mean_reward": "reward/mean", "attack_reward": "reward/attack", "defence_reward": "reward/defence",
	"attack_raw_reward": "raw_reward/attack", "defence_raw_reward": "raw_reward/defence",
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
		writer.add_scalar(f"won/{name}", float(row[f"{name}_outcome"] == "WIN"), epoch)
	writer.flush()


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


def save_checkpoint(path, model, optimizer, epoch):
	torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, path)


def train(config, resume=False):
	out = run_dir(config)
	out.mkdir(parents=True, exist_ok=True)
	(out / "config.txt").write_text("\n".join(f"{k} = {v}" for k, v in asdict(config).items()) + "\n")
	torch.manual_seed(config.seed)
	model = BattleModel(dim=config.model_dim, layers=config.model_layers)
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

	play.build_map(config.scenario)
	reward_config, advantage_config, ppo_config = RewardConfig(), AdvantageConfig(), PPOConfig()
	progress = out / "progress.csv"
	if not progress.exists() or not resume:
		with progress.open("w", newline="") as f:
			csv.writer(f).writerow(COLUMNS)
	writer = SummaryWriter(out / "tensorboard" / time.strftime("%Y%m%d-%H%M%S"))

	for epoch in range(first_epoch, config.epochs + 1):
		started = time.time()
		model_path = out / "model.onnx"
		export(model, model_path)

		games = [play.Game(config.scenario, our_side=side, opponent=config.opponent, game_seed=config.game_seeds[side], sampler_seed=2 * epoch + side) for side in (0, 1)]
		records = out / "records" / f"epoch_{epoch}"
		for attempt in range(1, config.attempts + 1):
			try:
				shutil.rmtree(records, ignore_errors=True)
				dirs = play.play(games, model_path, records)
				loader, summaries = make_dataloader(dirs, config.batch_size, True, reward_config, advantage_config)
				break
			except RuntimeError as error:  # a battle failed or stalled; its runner folder is kept for debugging
				print(f"epoch {epoch}: attempt {attempt} failed: {str(error)[:300]}", flush=True)
				if attempt == config.attempts:
					raise

		examples = loader.dataset  # the attack battle's decisions, then the defence battle's
		split = summaries[0].decisions
		for side, (summary, decisions) in enumerate(zip(summaries, (examples[:split], examples[split:]))):
			if decisions and (best[side] is None or summary.reward > best[side]["reward"]):
				best[side] = {"reward": summary.reward, "epoch": epoch, "line": collate(decisions)}
				torch.save(best, best_path)

		weights_before = flat_weights(model)
		parts_sum, batches = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}, 0
		model.train()
		for _ in range(config.ppo_passes):
			for batch in loader:
				out_batch = model(batch["inputs"])
				loss, parts = ppo_loss(out_batch, batch["action"], batch["old_log_prob"], batch["advantage"], batch["value_labels"], ppo_config)
				optimizer.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
				optimizer.step()
				for key in parts_sum:
					parts_sum[key] += float(parts[key])
				batches += 1
		model.eval()

		attack, defence = summaries
		mean = {key: value / max(batches, 1) for key, value in parts_sum.items()}
		row = {
			"epoch": epoch,
			"mean_reward": round(to_scale((attack.reward + defence.reward) / 2, reward_config), 4),
			"decisions": attack.decisions + defence.decisions,
			"rounds": attack.rounds + defence.rounds,
			"policy_loss": round(mean["policy"], 5),
			"value_loss": round(mean["value"], 5),
			"entropy": round(mean["entropy"], 4),
			"approx_kl": round(mean["approx_kl"], 5),
			"clip_fraction": round(mean["clip_fraction"], 4),
			"signal_ratio": round(signal_ratio(optimizer), 4),
			"weight_change": round(float((flat_weights(model) - weights_before).norm()), 4),
			"seconds": round(time.time() - started, 1),
		}
		for name, summary, line in (("attack", attack, best[0]), ("defence", defence, best[1])):
			row[f"{name}_reward"] = round(to_scale(summary.reward, reward_config), 4)
			row[f"{name}_raw_reward"] = round(summary.reward, 4)
			row[f"{name}_outcome"] = summary.outcome.name
			row[f"{name}_advantage_std"] = round(summary.advantage_std, 4)
			row[f"{name}_value_error"] = round(summary.value_error, 4)
			row[f"{name}_best_line_logp"] = round(line_log_prob(model, line["line"]), 3) if line else ""
		with progress.open("a", newline="") as f:
			csv.writer(f).writerow([row[c] for c in COLUMNS])
		log_tensorboard(writer, row)
		print(
			f"epoch {epoch:4d}  attack {row['attack_outcome']:9s} {row['attack_reward']:+.3f}  defence {row['defence_outcome']:9s} {row['defence_reward']:+.3f}"
			f"  mean {row['mean_reward']:+.3f}  decisions {row['decisions']:3d}  best-line logp {row['attack_best_line_logp']} / {row['defence_best_line_logp']}"
			f"  signal {row['signal_ratio']}  {row['seconds']}s",
			flush=True,
		)

		shutil.rmtree(records, ignore_errors=True)
		save_checkpoint(latest, model, optimizer, epoch)
		if epoch % config.checkpoint_every == 0:
			save_checkpoint(out / f"checkpoint_{epoch:04d}.pt", model, optimizer, epoch)


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
	parser.add_argument("--run", default=TrainConfig.run)
	parser.add_argument("--resume", action="store_true")
	parser.add_argument("--follow", metavar="RUN", help="mirror a running run's progress.csv into TensorBoard instead of training")
	args = parser.parse_args()
	if args.follow:
		follow(args.follow)
	else:
		train(TrainConfig(epochs=args.epochs, run=args.run), resume=args.resume)


if __name__ == "__main__":
	main()
