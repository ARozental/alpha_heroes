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
import itertools
import json
import pickle
import random
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import constants as c
from .config import CampaignConfig
from .model import build_model
from .labels import GROWTH_CAP, PHI_FLOOR, growth_labels, label, share_change_labels
from .records import TEACHER, decision_name, decode_inputs
from .schema import HEADS
from .train import _prefetched, chosen_logp, head_loss, migrate_subtypes, model_kwargs


def teacher_decisions(paths, share_target="change"):
	"""The teacher's turn decisions of each game file, with their value labels: [(decisions, shares [n, N_VALUE_OFFSETS],
	log growth [n, N_VALUE_OFFSETS])] (labels.label's real shares and labels.growth_labels', nan where there is none)."""
	games = []
	for path in paths:
		with open(path, "rb") as f:
			games.append(labelled(pickle.load(f), share_target))
	return games


def labelled(game, share_target="change"):
	"""A game's teacher decisions (turns and its answers: records.TEACHER_QUESTIONS) with their value labels."""
	shares, growth = (share_change_labels(game) if share_target == "change" else label(game)[0]), growth_labels(game)
	keep = [i for i, d in enumerate(game.decisions) if d.flags & TEACHER]
	return [game.decisions[i] for i in keep], shares[keep], growth[keep]


class TeacherPlay(torch.utils.data.IterableDataset):
	"""Nullkiller2's games played as they are needed (teacher.py's, nothing written to disk): each DataLoader worker plays
	one at a time, on the training games' maps and day caps (config.CampaignConfig, learner.draw_days), and gives its
	players' streams (player_streams') as it ends. A worker stopping ends its game (no VCMI outlives the run)."""

	def __init__(self, seed, share_target="change", game_seconds=1800):
		self.seed, self.share_target, self.game_seconds = seed, share_target, game_seconds

	def __iter__(self):
		from battle_model import play as battle_play

		from .learner import draw_days
		from .play import GameRun, make_map

		info = torch.utils.data.get_worker_info()
		worker = info.id if info is not None else 0
		config, rng = CampaignConfig(), random.Random(self.seed * 1000 + worker)
		for n in itertools.count():
			seed = (self.seed * 100 + worker) * 100000 + n
			players, size, days = rng.choice(config.players), rng.choice(config.sizes), draw_days(rng, config)
			map_name = make_map(seed, players, size, config.levels, config.water)[0]
			run = GameRun(battle_play.SOCKET_DIR / "teacher-none.sock", map_name, days, seed, max_seconds=self.game_seconds, battle_ai="BattleAI", game_key=f"s{seed}", teacher=True)
			try:
				while not run.poll():
					time.sleep(0.5)
				game, _ = run.result()
			finally:
				run.terminate()
				(battle_play.USER_DIR / (map_name + ".vmap")).unlink(missing_ok=True)
			game.days = days
			for stream in player_streams([labelled(game, self.share_target)]):
				if len(stream[0]):
					yield stream


def _worker_init(_):
	import signal

	signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # (the generator's finally ends the game being played)


class StreamPool:
	"""Players' streams as the workers end their games, taken in random order (so a step's lanes are mostly of different
	games); filled by a thread, at most `size` waiting (the workers then wait too)."""

	def __init__(self, loader, size, rng):
		self.items, self.size, self.rng, self.lock = [], size, rng, threading.Condition()
		self.loader, self.closed = loader, False
		threading.Thread(target=self._fill, args=(loader,), daemon=True).start()

	def _fill(self, loader):
		for stream in loader:
			with self.lock:
				while len(self.items) >= self.size and not self.closed:
					self.lock.wait()
				if self.closed:
					return
				self.items.append(stream)
				self.lock.notify_all()

	def close(self):
		"""Ends the workers, and with them the games they play (TeacherPlay's finally, on their SIGTERM): the iterator
		holds them until it is collected, and a thread waiting on it kept that from happening (the process never ended,
		2026-10-31). As supply.Supply.close does."""
		with self.lock:
			self.closed = True
			self.lock.notify_all()
		iterator = getattr(self.loader, "_iterator", None)
		if iterator is not None:
			iterator._shutdown_workers()

	def take(self):
		with self.lock:
			while not self.items:
				self.lock.wait()
			stream = self.items.pop(self.rng.randrange(len(self.items)))
			self.lock.notify_all()
			return stream


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


def mode_logp(out, n, kind):
	"""[N] the log-probability of the model's most likely choice: a turn's, the best over types of the type's own and its
	table's best entry (a targeted spell's tile left out); a question's, its best option's."""
	best = out["type_logp"].clone()
	for a, (output, _) in HEADS.items():
		best[:, a] = best[:, a] + out[output].reshape(n, -1).max(dim=1).values
	turn = (kind == c.DecisionKind.TURN) | (kind == c.DecisionKind.EXCHANGE)
	return torch.where(turn, best.max(dim=1).values, out["option_logp"].max(dim=1).values)


def player_streams(games):
	"""Each player's decisions of each game, in their order, with their labels: [(decisions, shares, growth)]."""
	streams = []
	for decisions, shares, growth in games:
		for color in sorted({d.color for d in decisions}):
			rows = [i for i, d in enumerate(decisions) if d.color == color]
			streams.append(([decisions[i] for i in rows], shares[rows], growth[rows]))
	return streams


class GameStreams:
	"""Each player's stream of each game file (player_streams'), loaded only while one of its players is used: an index
	of (file, color, decisions) made once (index.json in each folder), the games kept in a cache of the last `keep`
	(~35 MB each); streams[k] as player_streams' k-th, length(k) without loading it."""

	def __init__(self, paths, share_target="change", keep=64):
		self.share_target, self.keep, self.cache = share_target, keep, {}
		self.index = []
		folders = {}
		for path in paths:
			folders.setdefault(path.parent, []).append(path)
		for folder, files in folders.items():
			known_path = folder / "index.json"
			known = json.loads(known_path.read_text()) if known_path.exists() else {}
			for path in files:
				if path.name not in known:
					with open(path, "rb") as f:
						game = pickle.load(f)
					counts = {}
					for d in game.decisions:
						if d.flags & TEACHER:
							counts[d.color] = counts.get(d.color, 0) + 1
					known[path.name] = counts
					del game
				self.index += [(path, color, n) for color, n in sorted(known[path.name].items())]
			known_path.write_text(json.dumps(known))

	def __len__(self):
		return len(self.index)

	def length(self, k):
		return self.index[k][2]

	def __getitem__(self, k):
		path, color, _ = self.index[k]
		if path not in self.cache:
			if len(self.cache) >= self.keep:
				self.cache.pop(next(iter(self.cache)))  # (the oldest loaded)
			(decisions, shares, growth), = teacher_decisions([path], self.share_target)
			self.cache[path] = {c_: ([decisions[i] for i in rows], shares[rows], growth[rows]) for c_ in sorted({d.color for d in decisions}) for rows in [[i for i, d in enumerate(decisions) if d.color == c_]]}
		return self.cache[path][color]


def ordered(streams, rng=None):
	"""next_stream for chunks over a folder's streams (a list, or GameStreams: each loaded as a lane takes it): each
	once, in random order with rng; None once all are taken."""
	order = list(range(len(streams)))
	if rng is not None:
		rng.shuffle(order)
	queue = iter(order)
	return lambda: (lambda k: None if k is None else streams[k])(next(queue, None))


def chunks(next_stream, lanes, steps):
	"""The chunks of the streams next_stream gives (a folder's: ordered; live ones: StreamPool.take), `lanes` players at
	once: each (steps lists of the lanes' (stream, position), None for a lane past its stream's end or without one;
	the lanes whose memory starts afresh). A lane takes the next stream at a chunk's start."""
	stream, position, fresh = [next_stream() for _ in range(lanes)], [0] * lanes, [True] * lanes
	while any(s is not None for s in stream):
		yield [[(s, position[i] + t) if s is not None and position[i] + t < len(s[0]) else None for i, s in enumerate(stream)] for t in range(steps)], fresh
		fresh = [False] * lanes
		for i, s in enumerate(stream):
			if s is not None:
				position[i] += steps
				if position[i] >= len(s[0]):
					stream[i], position[i], fresh[i] = next_stream(), 0, True


def step_batches(schedule):
	"""(step, its lanes' decisions (a lane past its end given the step's first decision, masked out), its valid mask
	[lanes], its labels) of each step of each chunk, and the chunk's fresh lanes at its first step."""
	for chunk, fresh in schedule:
		for t, lanes in enumerate(chunk):
			known = [entry for entry in lanes if entry is not None]
			if not known:
				continue
			filled = [entry or known[0] for entry in lanes]
			yield t, [s[0][p] for s, p in filled], np.array([entry is not None for entry in lanes]), np.stack([s[1][p] for s, p in filled]), np.stack([s[2][p] for s, p in filled]), fresh if t == 0 else None


def evaluate(model, streams, device, lanes, steps, growth_weight, memory_on=True):
	"""The held-out players' mean loss and top-1 agreement, overall and by the teacher's action type, each player's
	decisions in order with the memory carried; and the value heads' loss ("value")."""
	sums = defaultdict(lambda: np.zeros(3))  # type -> (decisions, loss, agreed)
	values = []
	model.eval()
	memory = torch.zeros(lanes, c.MEMORY_SLOTS, c.MEMORY_WIDTH, device=device)
	with torch.no_grad():
		for (t, chosen, valid, shares, growth, fresh), inputs in _prefetched(step_batches(chunks(ordered(streams), lanes, steps)), lambda batch: decode_inputs(batch[1]), device):
			if fresh is not None:
				memory = memory * torch.tensor([not f for f in fresh], device=device).float()[:, None, None]
			out = model({**inputs, "mem_float": memory if memory_on else torch.zeros_like(memory)})
			choice = choice_tensors(chosen, device)
			memory = torch.where((choice["kind"] == c.DecisionKind.TURN)[:, None, None], out["memory"], memory)  # (turn decisions write it: AlphaMapAI's rule)
			logp, _ = chosen_logp(out, choice)
			best = mode_logp(out, len(chosen), choice["kind"])
			agreed = (logp >= best - 1e-4).float().cpu().numpy()
			losses = (-logp).cpu().numpy()
			for d, loss, ok, v in zip(chosen, losses, agreed, valid):
				if v:
					for key in ("all", decision_name(d)):
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
	parser.add_argument("--share-target", default="change", help="what the share heads learn (config.share_target): change (the logit's change) or share")
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
	parser.add_argument("--stream", type=int, default=0, help="Nullkiller2's games played as needed by this many workers (TeacherPlay), not read from --data: each decision seen once, its loss before the update the held-out measure")
	parser.add_argument("--until-plateau", type=float, default=0.0, help="streaming: stop once the last --plateau-periods periods' loss is under the as many before by less than this fraction (0: never)")
	parser.add_argument("--plateau-periods", type=int, default=5, help="the periods each side of the plateau's comparison (a period's loss varies from game to game)")
	args = parser.parse_args()
	assert args.lr <= 1e-4, "the user's rule: never above 1e-4"
	import signal

	def stop(signum, _frame):  # (SIGTERM as SIGINT: the workers and their games end in the finally; a background job of
		raise KeyboardInterrupt(f"signal {signum}")  # a script ignores SIGINT, so runtime/full_run.sh sends SIGTERM)

	signal.signal(signal.SIGTERM, stop)
	from .learner import auto_device

	device = torch.device(args.device or auto_device())
	if device.type == "cuda":  # (as the trainer's tf32 setting: float32 matmuls as TF32)
		torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)
	rng = random.Random(args.seed)
	pool = None
	if args.stream:
		import multiprocessing

		loader = torch.utils.data.DataLoader(TeacherPlay(args.seed, args.share_target), batch_size=None, num_workers=args.stream, worker_init_fn=_worker_init, persistent_workers=True, prefetch_factor=1, multiprocessing_context=multiprocessing.get_context("forkserver" if sys.platform != "win32" else "spawn"))
		pool = StreamPool(loader, 2 * args.lanes, rng)
		val_streams, train_streams, total = None, None, float("inf")
		print(f"imitate: Nullkiller2's games played by {args.stream} workers, each decision seen once; {device}", flush=True)
	else:
		paths = sorted(p for folder in args.data.split(",") for p in Path(folder).glob("game_*.pkl"))  # (folders, comma-separated)
		rng.shuffle(paths)
		paths = paths[: args.max_games] if args.max_games else paths
		held = max(1, int(round(args.holdout * len(paths))))
		val_streams, train_streams = GameStreams(paths[:held], args.share_target), GameStreams(paths[held:], args.share_target, keep=2 * args.lanes)
		total = sum(n for *_, n in train_streams.index)
		print(f"imitate: {total} decisions of {len(paths) - held} games to train on, {sum(n for *_, n in val_streams.index)} of {held} held out; {device}", flush=True)
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
			kwargs = model_kwargs(dataclasses.replace(config, model_architecture=args.architecture, model_cell_dim=args.cell_dim, model_dim=dim, model_layers=layers, share_target=args.share_target))
			torch.manual_seed(args.seed)
			models[variant] = (build_model(kwargs), kwargs, flag != "nomem")
	models = {name: (model.to(device).train(), torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr), kwargs, on) for name, (model, kwargs, on) in models.items()}
	history = {name: [] for name in models}
	online = {name: defaultdict(lambda: np.zeros(3)) for name in models}  # (streaming: each decision's loss and top-1 before the update)
	def evaluation(model, on, name):
		if val_streams is not None:
			return evaluate(model, val_streams, device, args.lanes, args.steps, config.growth_weight, on)
		sums, online[name] = online[name], defaultdict(lambda: np.zeros(3))
		return {key: {"decisions": int(v[0]), "loss": v[1] / max(v[0], 1), "top1": v[2] / max(v[0], 1)} for key, v in sorted(sums.items(), key=lambda kv: -kv[1][0])} | {"value": float("nan")}

	if val_streams is not None:
		for name, (model, _, _, on) in models.items():
			report = evaluation(model, on, name)
			history[name].append({"step": 0, "decisions": 0, "val": report})
			print(f"imitate: {name} ({sum(p.numel() for p in model.parameters()):,} weights) step 0 held-out loss {report['all']['loss']:.3f} top-1 {report['all']['top1']:.3f} value {report['value']:.4f}", flush=True)

	def step(model, inputs, memory, choice, shares, growth, valid):
		"""One decision of each lane: (its loss summed over the valid lanes, the memory it leaves, each lane's logp of the
		teacher's choice and of the model's most likely one, without gradients)."""
		# (the inputs come on the CPU and go to the device here: the checkpoint keeps a chunk's steps' inputs to recompute
		# them, ~0.9 MB a decision, which on the GPU was 7 GB at 128 lanes x 64 steps)
		out = model({**{k: v.to(device, non_blocking=True) for k, v in inputs.items()}, "mem_float": memory})
		logp, _ = chosen_logp(out, choice)
		loss = -(logp * valid).sum()
		if args.value_weight > 0:
			rows = valid.nonzero().squeeze(-1)
			loss = loss + args.value_weight * valid.sum() * value_loss({k: out[k][rows] for k in ("value_logit", "growth_logit", "phi")}, shares, growth, device, config.growth_weight)
		return loss, out["memory"], logp.detach(), mode_logp(out, logp.shape[0], choice["kind"]).detach()

	started = time.time()
	deadline = started + 60 * args.minutes
	memories = {name: torch.zeros(args.lanes, c.MEMORY_SLOTS, c.MEMORY_WIDTH, device=device) for name in models}
	try:
		train_loop(args, models, memories, pool, train_streams, total, history, evaluation, step, checkpoint, online, out, device, started, deadline)
	finally:
		if pool is not None:
			pool.close()


def train_loop(args, models, memories, pool, train_streams, total, history, evaluation, step, checkpoint, online, out, device, started, deadline):
	"""imitate's training: chunk after chunk of each lane's decisions, until the deadline, the epochs or a plateau."""
	steps_done, seen, epoch = 0, 0, 0
	done = False
	while not done:
		pending = {name: [] for name in models}  # the chunk's steps: (loss, valid count)
		running = {name: [] for name in models}
		schedule = chunks(pool.take if pool is not None else ordered(train_streams, random.Random(args.seed + epoch)), args.lanes, args.steps)
		for (t, chosen, valid, shares, growth, fresh), inputs in _prefetched(step_batches(schedule), lambda batch: decode_inputs(batch[1]), torch.device("cpu"), threads=args.decode_threads):
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
				loss, after, logp, best = checkpoint(step, model, inputs, memory, choice, shares[rows], growth[rows], valid_t, use_reentrant=False)
				pending[name].append((loss, float(valid.sum())))
				if pool is not None:  # (streaming: this decision is new to the model, its loss before the update a held-out one)
					agreed, losses = (logp >= best - 1e-4).float().cpu().numpy(), (-logp).cpu().numpy()
					for d, l_, ok, v in zip(chosen, losses, agreed, valid):
						if v:
							for key in ("all", decision_name(d)):
								online[name][key] += (1.0, float(l_), float(ok))
				writes = valid_t.bool() & (choice["kind"] == c.DecisionKind.TURN)  # (turn decisions write the memory: AlphaMapAI's rule)
				memories[name] = torch.where(writes[:, None, None], after, memory)
			seen += int(valid.sum())
			if t == 0 and (steps_done % args.eval_every == 0 and steps_done > 0 or time.time() > deadline or seen >= args.epochs * total):
				for name, (model, _, kwargs, on) in models.items():
					report = evaluation(model, on, name)
					history[name].append({"step": steps_done, "decisions": seen, "train_loss": float(np.mean(running[name])) if running[name] else float("nan"), "val": report})
					by_type = ", ".join(f"{k} {v['top1']:.2f}" for k, v in report.items() if k not in ("all", "value"))
					print(f"imitate: {name} step {steps_done} ({seen / max(1, total):.2f} epochs, {(time.time() - started) / 60:.0f} min): train loss {np.mean(running[name]) if running[name] else float('nan'):.3f}, held-out loss {report['all']['loss']:.3f} top-1 {report['all']['top1']:.3f} value {report['value']:.4f} ({by_type})", flush=True)
					running[name] = []
					suffix = "" if len(models) == 1 else "_" + name
					torch.save({"model": model.state_dict(), "model_kwargs": kwargs, "step": steps_done, "version": 0}, out / f"bc_latest{suffix}.pt")
				(out / "history.json").write_text(json.dumps(history, indent=1))
				losses = [h["val"]["all"]["loss"] for h in next(iter(history.values())) if "all" in h["val"]]
				k = args.plateau_periods
				plateau = bool(args.until_plateau) and len(losses) >= 2 * k and np.mean(losses[-k:]) > (1 - args.until_plateau) * np.mean(losses[-2 * k : -k])
				if plateau:
					print(f"imitate: the loss is on a plateau ({np.mean(losses[-2 * k : -k]):.4f} -> {np.mean(losses[-k:]):.4f}): done", flush=True)
				if time.time() > deadline or seen >= args.epochs * total or plateau:
					done = True
					break
		epoch += 1

if __name__ == "__main__":
	main()
