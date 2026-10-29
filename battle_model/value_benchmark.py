"""Shared or separate value network, offline: does the value head predict better when it does not share its trunk with
the policy?

From one trained model and one fixed set of self-play battles (played by it), two copies train on the same batches:
  shared     the full PPO loss (policy and value through one trunk), as training does
  separate   the value loss alone (a value network of its own: the same architecture, no policy gradient in it)
Both are scored by the value head's explained variance on held-out battles (V, the final-reward head), after every
pass. If the separate one is clearly better, the policy's gradients hurt the value head and a split is worth a run.

Usage: .venv/bin/python -m battle_model.value_benchmark --checkpoint <checkpoint_*.pt> --model <its onnx> [--battles 200]

KNOWN ISSUE (docs/adversarial_review_2026-10-27.md #8, open): "separate" starts from the shared trunk, which the policy
loss trained, so a few value-only passes measure "stop adding policy gradients", not "a value network of its own"
(the fix: train it from initialization, with more passes and battles). Do not read a split's worth from it yet.
"""
import argparse
import copy
import random

import numpy as np
import torch

from . import generator, play


def explained_variance(model, data, device, chunk=256):
	import battle_model.train as T

	model.eval()
	values = []
	with torch.no_grad():
		for i in range(0, len(data["action"]["type"]), chunk):
			index = torch.arange(i, min(i + chunk, len(data["action"]["type"])), device=device)
			values.append(model(T.select(data, index)["inputs"])["value"][:, -1].float())
	value = torch.cat(values)
	target = data["value_labels"][:, -1].float()
	return float(1 - ((target - value) ** 2).mean() / target.var())


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--checkpoint", required=True)
	parser.add_argument("--model", required=True, help="the checkpoint's model as ONNX, to play the battles")
	parser.add_argument("--battles", type=int, default=200)  # ~10k decisions: ~6 GB in memory and as much on the GPU
	parser.add_argument("--passes", type=int, default=8)
	parser.add_argument("--learning-rate", type=float, default=1e-4)  # the user's rule: never above 1e-4
	parser.add_argument("--seed", type=int, default=1)
	args = parser.parse_args()
	import battle_model.train as T

	device = T.resolve_device("auto")
	state = torch.load(args.checkpoint, map_location="cpu")
	config = T.TrainConfig(generator=True, run="valuebench", opponent=play.SELF_PLAY)
	rng = random.Random(args.seed)
	scenarios = [generator.sample(rng) for _ in range(args.battles)]
	games, files = T.generated_games(scenarios, "valuebench", config, our_sides=(0,), opponent=play.SELF_PLAY, temperature=1.0)
	records = T.play_in_groups([play.replace(g, sampler_seed=rng.randrange(2**31)) for g in games], args.model, config, "valuebench")
	for path in files:
		path.unlink(missing_ok=True)
	held_out = set(range(0, len(games), 5))  # every fifth battle, both of its records
	split = lambda keep: [r for i, r in enumerate(records) if (i // 2 in held_out) == keep]
	sets = {}
	for name, part in (("train", split(False)), ("test", split(True))):
		loader, _, _ = T.make_dataloader(part, config.batch_size, True, config.reward, config.advantage)
		sets[name] = T.to_device(T.collate(loader.dataset), device)
	n = len(sets["train"]["action"]["type"])
	print(f"{len(records)} records: {n} training decisions, {len(sets['test']['action']['type'])} held out")

	base = T.build_model(state["model_kwargs"]).to(device)
	base.load_state_dict(state["model"])
	models = {"shared": base, "separate": copy.deepcopy(base)}
	optimizers = {name: torch.optim.Adam(m.parameters(), lr=args.learning_rate) for name, m in models.items()}
	print("before:", {name: round(explained_variance(m, sets["test"], device), 4) for name, m in models.items()})
	generator_ = torch.Generator().manual_seed(args.seed)
	for p in range(1, args.passes + 1):
		order = torch.randperm(n, generator=generator_).to(device)
		for first in range(0, n - config.batch_size + 1, config.batch_size):
			batch = T.select(sets["train"], order[first : first + config.batch_size])
			for name, model in models.items():
				model.train()
				out = model(batch["inputs"])
				if name == "shared":
					loss, _ = T.ppo_loss(out, batch["action"], batch["old_log_prob"], batch["advantage"], batch["value_labels"], config.ppo, batch.get("weight"))
				else:
					from .losses import value_loss

					loss, _ = value_loss(out, batch["value_labels"], config.ppo, batch.get("weight"))
				optimizers[name].zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
				optimizers[name].step()
		print(f"pass {p}: held-out explained variance", {name: round(explained_variance(m, sets["test"], device), 4) for name, m in models.items()}, flush=True)


if __name__ == "__main__":
	main()
