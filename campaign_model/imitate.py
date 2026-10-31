"""Imitation of the teacher: the campaign model trained to choose Nullkiller2's decisions (teacher.py's games), and the
replication test (the user, 2026-10-31: "a model that ... can replicate/generalize the existing solution easily"): on
games it never trained on, how often its most likely action is the teacher's, and its loss, by action type.

	.venv/bin/python -m campaign_model.imitate --data runtime/teacher/t1 --out runtime/teacher/bc1 --minutes 60

The loss is the teacher's choice's negative log-probability under the model (train.chosen_logp: the type's and its
table entry's), Adam at --lr (never above 1e-4: the user's rule). One game in --holdout is never trained on (so a
map's decisions are not on both sides). Each player's decisions are taken in their order, with the model's memory
(model_v2.Memory) carried from one to the next: --lanes players at once, --steps decisions of each a chunk, the
gradient through the chunk's memory (truncated backpropagation through time; each step's activations recomputed in
the backward pass, so a chunk fits), the memory carried on into the next chunk without its gradient, and started at 0
with each player's game. A variant "<dim>x<layers>-nomem" has its memory held at 0: the check of what the memory adds,
on the same decisions. Every --eval-every steps: the held-out games' loss and top-1 agreement, overall
and by the teacher's action type, and a checkpoint (bc_latest.pt: "model", "model_kwargs", the step), which the
learner can start from (campaign.init_from).

The value heads learn alongside (--value-weight, the update's value_weight): the share heads and the growth heads
against the real outcomes the teacher's games had (labels.label, labels.growth_labels: every player's ledger is in
them), so a run started from here credits its first updates against forecasts that already mean something. V_end is
left to the run: these games end at one day cap (--days), the run's games at caps drawn per game."""
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
from .labels import GROWTH_CAP, PHI_FLOOR, growth_labels, label
from .records import TEACHER, decode_inputs
from .schema import HEADS
from .train import _prefetched, chosen_logp, head_loss, migrate_subtypes, model_kwargs


def teacher_decisions(paths):
	"""The teacher's turn decisions of each game file, with their value labels: [(decisions, shares [n, N_VALUE_OFFSETS],
	log growth [n, N_VALUE_OFFSETS])] (labels.label's real shares and labels.growth_labels', nan where there is none)."""
	games = []
	for path in paths:
		with open(path, "rb") as f:
			game = pickle.load(f)
		shares, growth = label(game)[0], growth_labels(game)
		keep = [i for i, d in enumerate(game.decisions) if d.flags & TEACHER and d.kind == c.DecisionKind.TURN]
		games.append(([game.decisions[i] for i in keep], shares[keep], growth[keep]))
	return games


def value_loss(out, shares, growth, device, growth_weight):
	"""The share and growth heads' loss on a batch (train.micro_loss's, against the real outcomes), each head as a share
	head weighs, growth at growth_weight; and its heads' count."""
	shares, growth = torch.from_numpy(shares).to(device), torch.from_numpy(growth).to(device)
	share_valid, growth_valid = torch.isfinite(shares).float(), torch.isfinite(growth).float()
	value = head_loss(out["value_logit"], shares.nan_to_num(0.0), share_valid).sum()
	now = out["phi"].detach().clamp(min=PHI_FLOOR).log().unsqueeze(-1)
	scaled = ((growth.nan_to_num(0.0) - now).clamp(-GROWTH_CAP, GROWTH_CAP) + GROWTH_CAP) / (2 * GROWTH_CAP)
	value = value + growth_weight * head_loss(out["growth_logit"], scaled, growth_valid).sum()
	return value / (c.N_VALUE_OFFSETS * (1 + growth_weight))


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


def player_streams(games):
	"""Each player's decisions of each game, in their order, with their labels: [(decisions, shares, growth)]."""
	streams = []
	for decisions, shares, growth in games:
		for color in sorted({d.color for d in decisions}):
			rows = [i for i, d in enumerate(decisions) if d.color == color]
			streams.append(([decisions[i] for i in rows], shares[rows], growth[rows]))
	return streams


def chunks(streams, lanes, steps, rng=None):
	"""The chunks of the streams, `lanes` players at once: each (steps lists of the lanes' (stream, position), None for a
	lane past its stream's end; the lanes whose memory starts afresh). A lane takes the next stream at a chunk's start."""
	order = list(range(len(streams)))
	if rng is not None:
		rng.shuffle(order)
	queue = iter(order)
	stream, position, fresh = [next(queue, None) for _ in range(lanes)], [0] * lanes, [True] * lanes
	while any(k is not None for k in stream):
		yield [[(k, position[i] + t) if k is not None and position[i] + t < len(streams[k][0]) else None for i, k in enumerate(stream)] for t in range(steps)], fresh
		fresh = [False] * lanes
		for i, k in enumerate(stream):
			if k is not None:
				position[i] += steps
				if position[i] >= len(streams[k][0]):
					stream[i], position[i], fresh[i] = next(queue, None), 0, True


def step_batches(streams, schedule):
	"""(step, its lanes' decisions (a lane past its end given the step's first decision, masked out), its valid mask
	[lanes], its labels) of each step of each chunk, and the chunk's fresh lanes at its first step."""
	for chunk, fresh in schedule:
		for t, lanes in enumerate(chunk):
			known = [entry for entry in lanes if entry is not None]
			if not known:
				continue
			filled = [entry or known[0] for entry in lanes]
			yield t, [streams[k][0][p] for k, p in filled], np.array([entry is not None for entry in lanes]), np.stack([streams[k][1][p] for k, p in filled]), np.stack([streams[k][2][p] for k, p in filled]), fresh if t == 0 else None


def evaluate(model, streams, device, lanes, steps, growth_weight, memory_on=True):
	"""The held-out players' mean loss and top-1 agreement, overall and by the teacher's action type, each player's
	decisions in order with the memory carried; and the value heads' loss ("value")."""
	sums = defaultdict(lambda: np.zeros(3))  # type -> (decisions, loss, agreed)
	values = []
	model.eval()
	memory = torch.zeros(lanes, c.MEMORY_SLOTS, c.MEMORY_WIDTH, device=device)
	with torch.no_grad():
		for (t, chosen, valid, shares, growth, fresh), inputs in _prefetched(step_batches(streams, chunks(streams, lanes, steps)), lambda batch: decode_inputs(batch[1]), device):
			if fresh is not None:
				memory = memory * torch.tensor([not f for f in fresh], device=device).float()[:, None, None]
			out = model({**inputs, "mem_float": memory if memory_on else torch.zeros_like(memory)})
			memory = out["memory"]
			logp, _ = chosen_logp(out, choice_tensors(chosen, device))
			best = mode_logp(out, len(chosen)).max(dim=1).values
			agreed = (logp >= best - 1e-4).float().cpu().numpy()
			losses = (-logp).cpu().numpy()
			for d, loss, ok, v in zip(chosen, losses, agreed, valid):
				if v:
					for key in ("all", c.ActionType(d.action_type).name):
						sums[key] += (1.0, float(loss), float(ok))
			rows = np.flatnonzero(valid)
			values.append((len(rows), float(value_loss({k: v[torch.from_numpy(rows).to(device)] for k, v in out.items() if k in ("value_logit", "growth_logit", "phi")}, shares[rows], growth[rows], device, growth_weight))))
	model.train()
	report = {key: {"decisions": int(v[0]), "loss": v[1] / v[0], "top1": v[2] / v[0]} for key, v in sorted(sums.items(), key=lambda kv: -kv[1][0])}
	report["value"] = sum(n * v for n, v in values) / max(1, sum(n for n, _ in values))
	return report


def main():
	config = CampaignConfig()
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--data", required=True, help="teacher.py's folder of game_*.pkl")
	parser.add_argument("--out", required=True)
	parser.add_argument("--init", default="", help="a campaign checkpoint to start from (none: a fresh model)")
	parser.add_argument("--architecture", default="v2")
	parser.add_argument("--variants", default="64x2", help="the models trained side by side on the same batches, <dim>x<layers> each (a fresh model's); their checkpoints bc_latest_<variant>.pt when more than one")
	parser.add_argument("--cell-dim", type=int, default=16)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--value-weight", type=float, default=config.value_weight, help="the value heads' loss weight (0: the policy only)")
	parser.add_argument("--lanes", type=int, default=32, help="players taken at once (each decision of a step one of theirs)")
	parser.add_argument("--steps", type=int, default=8, help="each player's decisions a chunk: the memory's gradient goes back this far")
	parser.add_argument("--holdout", type=float, default=0.1)
	parser.add_argument("--max-games", type=int, default=0, help="at most this many of the folder's games (0: all; they are held in memory, ~35 MB each)")
	parser.add_argument("--minutes", type=float, default=60.0)
	parser.add_argument("--epochs", type=float, default=10.0)
	parser.add_argument("--eval-every", type=int, default=200, help="chunks (optimizer steps) between the held-out evaluations")
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--device", default="")
	parser.add_argument("--decode-threads", type=int, default=3, help="batches decoded at once (zlib: ~0.3 s a 128-decision batch a thread)")
	args = parser.parse_args()
	assert args.lr <= 1e-4, "the user's rule: never above 1e-4"
	from .learner import auto_device

	device = torch.device(args.device or auto_device())
	if device.type == "cuda":  # (as the trainer's tf32 setting: float32 matmuls as TF32)
		torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)
	paths = sorted(Path(args.data).glob("game_*.pkl"))
	rng = random.Random(args.seed)
	rng.shuffle(paths)
	paths = paths[: args.max_games] if args.max_games else paths
	held = max(1, int(round(args.holdout * len(paths))))
	val_games, train_games = teacher_decisions(paths[:held]), teacher_decisions(paths[held:])
	print(f"imitate: {sum(len(g[0]) for g in train_games)} decisions of {len(train_games)} games to train on, {sum(len(g[0]) for g in val_games)} of {len(val_games)} held out; {device}", flush=True)
	from torch.utils.checkpoint import checkpoint

	models = {}  # name -> (model, optimizer, kwargs, memory on): each trained on the same decisions, decoded once
	if args.init:
		state = torch.load(args.init, map_location="cpu")
		kwargs = state.get("model_kwargs", {})
		model = build_model(kwargs)
		migrate_subtypes(state, model)
		model.load_state_dict(state["model"], strict=False)  # (a checkpoint from before the memory: its parts fresh)
		models["init"] = (model, kwargs, True)
	else:
		import dataclasses

		for variant in args.variants.split(","):  # "<dim>x<layers>", "-nomem": its memory held at 0
			size, _, flag = variant.partition("-")
			dim, layers = (int(v) for v in size.split("x"))
			kwargs = model_kwargs(dataclasses.replace(config, model_architecture=args.architecture, model_cell_dim=args.cell_dim, model_dim=dim, model_layers=layers))
			torch.manual_seed(args.seed)
			models[variant] = (build_model(kwargs), kwargs, flag != "nomem")
	models = {name: (model.to(device).train(), torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr), kwargs, on) for name, (model, kwargs, on) in models.items()}
	train_streams, val_streams = player_streams(train_games), player_streams(val_games)
	history = {name: [] for name in models}
	evaluation = lambda model, on: evaluate(model, val_streams, device, args.lanes, args.steps, config.growth_weight, on)  # noqa: E731
	for name, (model, _, _, on) in models.items():
		report = evaluation(model, on)
		history[name].append({"step": 0, "decisions": 0, "val": report})
		print(f"imitate: {name} ({sum(p.numel() for p in model.parameters()):,} weights) step 0 held-out loss {report['all']['loss']:.3f} top-1 {report['all']['top1']:.3f} value {report['value']:.4f}", flush=True)

	def step(model, inputs, memory, choice, shares, growth, valid):
		"""One decision of each lane: (its loss summed over the valid lanes, the memory it leaves)."""
		out = model({**inputs, "mem_float": memory})
		logp, _ = chosen_logp(out, choice)
		loss = -(logp * valid).sum()
		if args.value_weight > 0:
			rows = valid.nonzero().squeeze(-1)
			loss = loss + args.value_weight * valid.sum() * value_loss({k: out[k][rows] for k in ("value_logit", "growth_logit", "phi")}, shares, growth, device, config.growth_weight)
		return loss, out["memory"]

	started, steps_done, seen, epoch = time.time(), 0, 0, 0
	deadline = started + 60 * args.minutes
	memories = {name: torch.zeros(args.lanes, c.MEMORY_SLOTS, c.MEMORY_WIDTH, device=device) for name in models}
	total = sum(len(st[0]) for st in train_streams)
	done = False
	while not done:
		pending = {name: [] for name in models}  # the chunk's steps: (loss, valid count)
		running = {name: [] for name in models}
		schedule = chunks(train_streams, args.lanes, args.steps, random.Random(args.seed + epoch))
		for (t, chosen, valid, shares, growth, fresh), inputs in _prefetched(step_batches(train_streams, schedule), lambda batch: decode_inputs(batch[1]), device, threads=args.decode_threads):
			valid_t = torch.from_numpy(valid).to(device).float()
			rows = np.flatnonzero(valid)
			choice = choice_tensors(chosen, device)
			if t == 0 and any(pending[name] for name in models):  # (a new chunk: the last one's gradient first)
				for name, (model, optimizer, _, _) in models.items():
					count = sum(n for _, n in pending[name])
					loss = sum(l for l, _ in pending[name]) / max(count, 1.0)
					optimizer.zero_grad(set_to_none=True)
					loss.backward()
					torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
					optimizer.step()
					running[name].append(float(loss.detach()))
					pending[name] = []
					memories[name] = memories[name].detach()
				steps_done += 1
			for name, (model, optimizer, _, on) in models.items():
				memory = memories[name]
				if fresh is not None:
					memory = memory * torch.tensor([not f for f in fresh], device=device).float()[:, None, None]
				if not on:
					memory = torch.zeros_like(memory)
				loss, after = checkpoint(step, model, inputs, memory, choice, shares[rows], growth[rows], valid_t, use_reentrant=False)
				pending[name].append((loss, float(valid.sum())))
				memories[name] = torch.where(valid_t.bool()[:, None, None], after, memory)
			seen += int(valid.sum())
			if t == 0 and (steps_done % args.eval_every == 0 and steps_done > 0 or time.time() > deadline or seen >= args.epochs * total):
				for name, (model, _, kwargs, on) in models.items():
					report = evaluation(model, on)
					history[name].append({"step": steps_done, "decisions": seen, "train_loss": float(np.mean(running[name])) if running[name] else float("nan"), "val": report})
					by_type = ", ".join(f"{k} {v['top1']:.2f}" for k, v in report.items() if k not in ("all", "value"))
					print(f"imitate: {name} step {steps_done} ({seen / max(1, total):.2f} epochs, {(time.time() - started) / 60:.0f} min): train loss {np.mean(running[name]) if running[name] else float('nan'):.3f}, held-out loss {report['all']['loss']:.3f} top-1 {report['all']['top1']:.3f} value {report['value']:.4f} ({by_type})", flush=True)
					running[name] = []
					suffix = "" if len(models) == 1 else "_" + name
					torch.save({"model": model.state_dict(), "model_kwargs": kwargs, "step": steps_done, "version": 0}, out / f"bc_latest{suffix}.pt")
				(out / "history.json").write_text(json.dumps(history, indent=1))
				if time.time() > deadline or seen >= args.epochs * total:
					done = True
					break
		epoch += 1

if __name__ == "__main__":
	main()
