"""Training loop: play battles against a fixed opponent, learn from them with PPO, repeat.

	.venv/bin/python -m battle_model.train [--epochs N] [--run NAME] [--resume]

Each epoch:
  1. export the current model to ONNX (the C++ battle AI plays with it),
  2. play one battle per scenario side in parallel (we attack with sides[0], we defend with sides[1]),
     with VCMI's seed fixed per side, so only our own choices change between epochs,
  3. turn the records into training examples (data.py) and update the model with PPO (losses.py),
  4. log both battles' rewards and outcomes to <run dir>/progress.csv, delete the records.

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

from . import play
from .data import make_dataloader
from .export import export
from .losses import AdvantageConfig, PPOConfig, ppo_loss
from .model import BattleModel
from .reward import RewardConfig

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class TrainConfig:
	scenario: str = str(ROOT / "battle_model" / "scenarios" / "castle_vs_inferno.json")
	run: str = "castle_vs_inferno"
	epochs: int = 1000
	opponent: str = "BattleAI"
	game_seeds: tuple = (1, 2)  # VCMI seed of the attack battle and of the defence battle: fixed, so the test is about learning, not luck
	learning_rate: float = 3e-4  # Adam's usual default for PPO
	ppo_passes: int = 4  # optimisation passes over each epoch's decisions; PPO reuses its data a few times
	batch_size: int = 64  # an epoch is only ~20-80 decisions, so usually one batch
	grad_clip: float = 0.5  # max gradient norm, the usual PPO value
	checkpoint_every: int = 50
	attempts: int = 3  # tries per epoch when a battle fails or stalls before the run stops
	model_dim: int = 64
	model_layers: int = 2


def run_dir(config):
	return ROOT / "runtime" / "train" / config.run


def save_checkpoint(path, model, optimizer, epoch):
	torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, path)


def train(config, resume=False):
	out = run_dir(config)
	out.mkdir(parents=True, exist_ok=True)
	(out / "config.txt").write_text("\n".join(f"{k} = {v}" for k, v in asdict(config).items()) + "\n")
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

	play.build_map(config.scenario)
	reward_config, advantage_config, ppo_config = RewardConfig(), AdvantageConfig(), PPOConfig()
	progress = out / "progress.csv"
	columns = ["epoch", "attack_reward", "attack_outcome", "defence_reward", "defence_outcome", "mean_reward", "decisions", "rounds", "policy_loss", "value_loss", "entropy", "seconds"]
	if not progress.exists() or not resume:
		with progress.open("w", newline="") as f:
			csv.writer(f).writerow(columns)

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

		parts_sum, batches = {"policy": 0.0, "value": 0.0, "entropy": 0.0}, 0
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
		row = {
			"epoch": epoch,
			"attack_reward": round(attack.reward, 4),
			"attack_outcome": attack.outcome.name,
			"defence_reward": round(defence.reward, 4),
			"defence_outcome": defence.outcome.name,
			"mean_reward": round((attack.reward + defence.reward) / 2, 4),
			"decisions": attack.decisions + defence.decisions,
			"rounds": attack.rounds + defence.rounds,
			"policy_loss": round(parts_sum["policy"] / max(batches, 1), 5),
			"value_loss": round(parts_sum["value"] / max(batches, 1), 5),
			"entropy": round(parts_sum["entropy"] / max(batches, 1), 4),
			"seconds": round(time.time() - started, 1),
		}
		with progress.open("a", newline="") as f:
			csv.writer(f).writerow([row[c] for c in columns])
		print(f"epoch {epoch:4d}  attack {row['attack_outcome']:9s} {row['attack_reward']:+.3f}  defence {row['defence_outcome']:9s} {row['defence_reward']:+.3f}  mean {row['mean_reward']:+.3f}  decisions {row['decisions']:3d}  {row['seconds']}s", flush=True)

		shutil.rmtree(records, ignore_errors=True)
		save_checkpoint(latest, model, optimizer, epoch)
		if epoch % config.checkpoint_every == 0:
			save_checkpoint(out / f"checkpoint_{epoch:04d}.pt", model, optimizer, epoch)


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
	parser.add_argument("--run", default=TrainConfig.run)
	parser.add_argument("--resume", action="store_true")
	args = parser.parse_args()
	train(TrainConfig(epochs=args.epochs, run=args.run), resume=args.resume)


if __name__ == "__main__":
	main()
