"""Imitation of the teacher: the campaign model trained to choose Nullkiller2's decisions (teacher.py's games), and the
replication test (the user, 2026-10-31: "a model that ... can replicate/generalize the existing solution easily"): on
games it never trained on, how often its most likely action is the teacher's, and its loss, by action type.

	.venv/bin/python -m campaign_model.imitate --data runtime/teacher/t1 --out runtime/teacher/bc1 --minutes 60

The loss is the teacher's choice's negative log-probability under the model (train.chosen_logp: the type's and its
table entry's), Adam at --lr (never above 1e-4: the user's rule). One game in --holdout is never trained on (so a
map's decisions are not on both sides). Every --eval-every steps: the held-out games' loss and top-1 agreement, overall
and by the teacher's action type, and a checkpoint (bc_latest.pt: "model", "model_kwargs", the step), which the
learner can start from (campaign.init_from)."""
import argparse
import json
import pickle
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import constants as c
from .config import CampaignConfig
from .model import build_model
from .records import TEACHER, decode_inputs
from .schema import HEADS
from .train import _prefetched, chosen_logp, migrate_subtypes, model_kwargs


def teacher_decisions(paths):
	"""The teacher's turn decisions of each game file: [list of records.Decision]."""
	games = []
	for path in paths:
		with open(path, "rb") as f:
			game = pickle.load(f)
		games.append([d for d in game.decisions if d.flags & TEACHER and d.kind == c.DecisionKind.TURN])
	return games


def choice_tensors(decisions, device):
	return {
		"kind": torch.tensor([d.kind for d in decisions], device=device),
		"action": torch.tensor([d.action_type for d in decisions], device=device),
		"index": torch.tensor([d.index for d in decisions], device=device),
		"cell": torch.tensor([d.cell for d in decisions], device=device),
	}


def mode_logp(out, n):
	"""[N, N_ACTION_TYPES] the log-probability of each type's most likely entry (the type's own and its table's best);
	its max over types is the model's most likely action's (a targeted spell's tile left out)."""
	best = out["type_logp"].clone()
	for a, (output, _) in HEADS.items():
		best[:, a] = best[:, a] + out[output].reshape(n, -1).max(dim=1).values
	return best


def evaluate(model, decisions, device, batch):
	"""The held-out decisions' mean loss and top-1 agreement, overall and by the teacher's action type."""
	sums = defaultdict(lambda: np.zeros(3))  # type -> (decisions, loss, agreed)
	model.eval()
	with torch.no_grad():
		chunks = [list(range(i, min(i + batch, len(decisions)))) for i in range(0, len(decisions), batch)]
		for rows, inputs in _prefetched(chunks, lambda rows: decode_inputs([decisions[r] for r in rows]), device):
			chosen = [decisions[r] for r in rows]
			out = model(inputs)
			choice = choice_tensors(chosen, device)
			logp, _ = chosen_logp(out, choice)
			best = mode_logp(out, len(rows)).max(dim=1).values
			agreed = (logp >= best - 1e-4).float()
			for d, loss, ok in zip(chosen, (-logp).cpu().numpy(), agreed.cpu().numpy()):
				for key in ("all", c.ActionType(d.action_type).name):
					sums[key] += (1.0, float(loss), float(ok))
	model.train()
	return {key: {"decisions": int(v[0]), "loss": v[1] / v[0], "top1": v[2] / v[0]} for key, v in sorted(sums.items(), key=lambda kv: -kv[1][0])}


def main():
	config = CampaignConfig()
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--data", required=True, help="teacher.py's folder of game_*.pkl")
	parser.add_argument("--out", required=True)
	parser.add_argument("--init", default="", help="a campaign checkpoint to start from (none: a fresh model)")
	parser.add_argument("--architecture", default="v2")
	parser.add_argument("--cell-dim", type=int, default=16)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--batch", type=int, default=128)
	parser.add_argument("--holdout", type=float, default=0.1)
	parser.add_argument("--minutes", type=float, default=60.0)
	parser.add_argument("--epochs", type=float, default=10.0)
	parser.add_argument("--eval-every", type=int, default=500)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--device", default="")
	args = parser.parse_args()
	assert args.lr <= 1e-4, "the user's rule: never above 1e-4"
	from .learner import auto_device

	device = torch.device(args.device or auto_device())
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)
	paths = sorted(Path(args.data).glob("game_*.pkl"))
	rng = random.Random(args.seed)
	rng.shuffle(paths)
	held = max(1, int(round(args.holdout * len(paths))))
	val_games, train_games = teacher_decisions(paths[:held]), teacher_decisions(paths[held:])
	val = [d for g in val_games for d in g]
	train = [d for g in train_games for d in g]
	print(f"imitate: {len(train)} decisions of {len(train_games)} games to train on, {len(val)} of {len(val_games)} held out; {device}", flush=True)
	if args.init:
		state = torch.load(args.init, map_location="cpu")
		kwargs = state.get("model_kwargs", {})
		model = build_model(kwargs)
		migrate_subtypes(state, model)
		model.load_state_dict(state["model"])
	else:
		import dataclasses

		kwargs = model_kwargs(dataclasses.replace(config, model_architecture=args.architecture, model_cell_dim=args.cell_dim))
		torch.manual_seed(args.seed)
		model = build_model(kwargs)
	model = model.to(device).train()
	optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
	history = []
	report = evaluate(model, val, device, args.batch)
	history.append({"step": 0, "decisions": 0, "val": report})
	print(f"imitate: step 0 held-out loss {report['all']['loss']:.3f} top-1 {report['all']['top1']:.3f}", flush=True)
	started, step, seen = time.time(), 0, 0
	deadline = started + 60 * args.minutes
	order = list(range(len(train)))
	done = False
	while not done:
		rng.shuffle(order)
		chunks = [order[i : i + args.batch] for i in range(0, len(order) - args.batch + 1, args.batch)]
		running = []
		for rows, inputs in _prefetched(chunks, lambda rows: decode_inputs([train[r] for r in rows]), device):
			chosen = [train[r] for r in rows]
			logp, _ = chosen_logp(model(inputs), choice_tensors(chosen, device))
			loss = -logp.mean()
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			step, seen = step + 1, seen + len(rows)
			running.append(float(loss.detach()))
			if step % args.eval_every == 0 or time.time() > deadline or seen >= args.epochs * len(train):
				report = evaluate(model, val, device, args.batch)
				history.append({"step": step, "decisions": seen, "train_loss": float(np.mean(running)), "val": report})
				by_type = ", ".join(f"{k} {v['top1']:.2f}" for k, v in report.items() if k != "all")
				print(f"imitate: step {step} ({seen / max(1, len(train)):.2f} epochs, {(time.time() - started) / 60:.0f} min): train loss {np.mean(running):.3f}, held-out loss {report['all']['loss']:.3f} top-1 {report['all']['top1']:.3f} ({by_type})", flush=True)
				running = []
				torch.save({"model": model.state_dict(), "model_kwargs": kwargs, "step": step, "version": 0}, out / "bc_latest.pt")
				(out / "history.json").write_text(json.dumps(history, indent=1))
				if time.time() > deadline or seen >= args.epochs * len(train):
					done = True
					break


if __name__ == "__main__":
	main()
