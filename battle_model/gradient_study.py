"""Why the battle model drifted in joint1 (2026-10-28): gradient diagnostics on a fixed dataset, to understand the
drift before changing the learner (the user: "not random experiments but actually understand the issue").

In joint1 the battle model's updates moved the weights as far as v15-window's (the same KL, weight change and Adam
signal ratio, ~0.11), but each update's surrogate gain was ~10x smaller, raw advantages varied ~50% more, battles grew
longer, and the policy drifted toward moving and waiting (shooters that could shoot at full damage idle in 5% -> 30% of
their turns, in training). The questions:
  1. Is the update direction mostly noise? Per source, how much do the minibatch gradients agree (coherence: the
     norm of their mean over their mean norm; 1 = all alike, ~1/sqrt(K) for K that are pure noise)?
  2. Do the two sources pull the same way (the cosine between their mean gradients)?
  3. What pushes shooters off shooting? Along each loss term's descent direction (policy, value, entropy) of each
     source, the change of mean log p(SHOOT) over decisions where shooting is legal: raw and through Adam's own
     preconditioning (the checkpoint's second moments), which is what an update actually does. The mean says how
     big each term's push is; the t-statistic only how consistent it is (a fixed regularizer's is large by
     construction, the policy gradient's never): compare the terms by their means, with the policy gradient's
     uncertainty (the 2026-10-28 review). The terms: the bonus joint1 had (entropy) and the floor that replaced it.
  4. Does the joint normalization (per side over the whole update, as training does) shrink one source's signal?

	.venv/bin/python -m battle_model.gradient_study run --out runtime/studies/drift        # collect and analyze, in memory
	.venv/bin/python -m battle_model.gradient_study collect --out ... / analyze --data ...  # through records.pkl

A battle decision's record is ~0.4 MB (mostly the spell-by-hex inputs and outputs): 10k decisions are ~4 GB, so run
keeps them in memory only.

collect plays generated self-play battles (battle_model/generator.py, both sides recorded) and the battles of
campaign self-play games (every side ours, the monsters' too: the joint run's real battles), all with the same
battle checkpoint, and pickles the records with their source.
"""
import argparse
import json
import os
import pickle
import random
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import generator, play, real_games
from .constants import ActionType
from .data import collate, make_dataloader, parse_record, summarize
from .distributions import entropy as action_entropy
from .distributions import log_prob
from .export import export
from .losses import AdvantageConfig, PPOConfig, value_loss, weighted_mean
from .model import build_model

ROOT = Path(__file__).resolve().parent.parent
BEST = ROOT / "runtime" / "train" / "20261027-2240-joint1" / "checkpoint_02840.pt"  # the last good one of joint1
CAMPAIGN = ROOT / "runtime" / "train" / "20261027-2240-joint1" / "campaign" / "checkpoint_latest.pt"


def load_model(checkpoint):
	state = torch.load(checkpoint, map_location="cpu")
	model = build_model(state["model_kwargs"])
	model.load_state_dict(state["model"])
	return model.eval(), state


# ---- collect ----


def collect_generated(onnx, count, workers, seed):
	from .train import TrainConfig, generated_games

	config = TrainConfig(run="gradient-study", ai="AlphaBattleAI")
	rng = random.Random(seed)
	scenarios = [generator.sample(rng) for _ in range(count)]
	results = [[] for _ in range(workers)]

	def work(w):
		process = play.BattleProcess()
		try:
			for i in range(w, count, workers):
				games, files = generated_games([scenarios[i]], f"study{i}", config, our_sides=(0,), opponent=play.SELF_PLAY, temperature=1.0)
				try:
					results[w] += process.play(play.replace(games[0], sampler_seed=seed * 1000 + i), onnx)
				except RuntimeError as error:
					print(f"generated battle {i} failed: {str(error)[:200]}", flush=True)
				finally:
					for path in files:
						path.unlink(missing_ok=True)
		finally:
			process.close()

	threads = [threading.Thread(target=work, args=(w,)) for w in range(workers)]
	for t in threads:
		t.start()
	for t in threads:
		t.join()
	return [r for rs in results for r in rs]


def collect_campaign(onnx, games, workers, seed):
	from campaign_model.play import make_map
	from campaign_model.play import play as play_campaign
	from campaign_model.train import serve_model

	records = []
	lock = threading.Lock()
	with serve_model(str(CAMPAIGN)) as (_, campaign_server), play.RecordServer() as battles:
		def work(w):
			rng = random.Random(seed * 7919 + w)
			for g in range(w, games, workers):
				game_seed = rng.randrange(1, 2**31)
				name, _ = make_map(game_seed, rng.choice((2, 3, 4)), 36)
				key = f"s{g}"
				settings = {"ALPHA_BATTLE_MODEL": str(onnx), "ALPHA_BATTLE_TEMPERATURE": "1", "ALPHA_BATTLE_RECORD_SOCKET": str(battles.path), "ALPHA_BATTLE_REQUIRE_MODEL": "1"}
				for i, color in enumerate(real_games.PLAYER_COLORS):
					settings[f"ALPHA_BATTLE_RECORD_KEY_{color}"] = f"{key}:{color}"
					settings[f"ALPHA_BATTLE_SEED_{color}"] = str(game_seed * 101 + i)
				try:
					_, info = play_campaign(campaign_server.socket_path, name, 56, game_seed, max_seconds=300, battle_ai="AlphaBattleAI", game_key=key, settings=tuple(settings.items()))
					import shutil

					shutil.rmtree(info.run_dir, ignore_errors=True)
				finally:
					(play.USER_DIR / (name + ".vmap")).unlink(missing_ok=True)

		threads = [threading.Thread(target=work, args=(w,)) for w in range(workers)]
		for t in threads:
			t.start()
		for t in threads:
			t.join()
		time.sleep(1)
		for key, payloads in battles.drain().items():
			for data in payloads:
				try:
					records.append(parse_record(data))
				except Exception as error:
					print(f"unreadable record {key}: {error}", flush=True)
	return records


def collect(args):
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)
	model, state = load_model(args.checkpoint)
	onnx = (out / "battle.onnx").resolve()
	export(model, onnx)
	started = time.time()
	generated = collect_generated(onnx, args.generated, args.workers, args.seed)
	print(f"generated: {len(generated)} records, {sum(r.decisions for r in generated)} decisions, {time.time() - started:.0f}s", flush=True)
	started = time.time()
	campaign = collect_campaign(onnx, args.campaign_games, min(args.workers, 6), args.seed)
	print(f"campaign: {len(campaign)} records, {sum(r.decisions for r in campaign)} decisions, {time.time() - started:.0f}s", flush=True)
	with (out / "records.pkl").open("wb") as f:
		pickle.dump({"checkpoint": str(args.checkpoint), "generated": generated, "campaign": campaign}, f)
	print(f"wrote {out / 'records.pkl'} ({(out / 'records.pkl').stat().st_size / 2**20:.0f} MB)")


# ---- analyze ----


def describe(records):
	summaries = [summarize(r.events) for r in records]
	kinds = Counter(ActionType(a["type"]).name for r in records for a in r.actions)
	total = max(1, sum(kinds.values()))
	return {
		"battles": len(records),
		"decisions": sum(r.decisions for r in records),
		"decisions_per_battle": round(float(np.mean([r.decisions for r in records])), 1),
		"rounds": round(float(np.mean([s.rounds for s in summaries])), 1),
		"outcomes": dict(Counter(s.outcome.name for s in summaries)),
		"actions": {k: round(v / total, 3) for k, v in kinds.most_common(6)},
	}


BONUS_WEIGHT = 0.01  # the constant entropy bonus joint1 trained with (PPOConfig.entropy_weight then; 0 since 2026-10-28)


def term_grads(model, batch, params, ppo=PPOConfig()):
	"""The gradient of each loss term on one batch (flattened), as training weighs them: policy, value_weight x value,
	the entropy bonus joint1 had (-BONUS_WEIGHT x entropy: PPOConfig's is 0 now, which made this term 0 on a rerun),
	and the entropy floor that replaced it (entropy_floor_weight x mean max(0, floor - H)); descent on the total is
	minus their sum."""
	out = model(batch["inputs"])
	weight = batch["weight"]
	ratio = torch.exp(log_prob(out, batch["action"]) - batch["old_log_prob"])
	advantage = batch["advantage"]
	clipped = ratio.clamp(1 - ppo.clip_range, 1 + ppo.clip_range)
	terms = {
		"policy": -weighted_mean(torch.min(ratio * advantage, clipped * advantage), weight),
		"value": ppo.value_weight * value_loss(out, batch["value_labels"], ppo, weight)[0],
		"entropy": -BONUS_WEIGHT * weighted_mean(action_entropy(out), weight),
		"floor": ppo.entropy_floor_weight * weighted_mean(torch.relu(ppo.entropy_floor - action_entropy(out)), weight),
	}
	grads = {}
	for name, term in terms.items():
		g = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
		grads[name] = torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x, p in zip(g, params)])
	return grads


def shoot_direction(model, examples, params):
	"""The gradient of mean log p(SHOOT) over decisions where shooting is legal, and how many there are."""
	shooters = [e for e in examples if bool(e["inputs"]["m_type"][ActionType.SHOOT])]
	if not shooters:
		return None, 0
	total = None
	for i in range(0, len(shooters), 256):
		batch = collate(shooters[i : i + 256])
		out = model(batch["inputs"])
		value = out["logp_type"][:, ActionType.SHOOT].sum() / len(shooters)
		g = torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x, p in zip(torch.autograd.grad(value, params, allow_unused=True), params)])
		total = g if total is None else total + g
	return total, len(shooters)


def analyze(args, data=None):
	torch.manual_seed(0)
	data = data or pickle.load(Path(args.data, "records.pkl").open("rb"))
	model, state = load_model(data["checkpoint"])
	params = [p for p in model.parameters() if p.requires_grad]
	# Adam's preconditioner from the checkpoint: an update moves each weight by ~ lr * m / sqrt(v)
	adam = state["optimizer"]["state"]
	v = torch.cat([adam[i]["exp_avg_sq"].flatten() if i in adam else torch.zeros(p.numel()) for i, p in enumerate(params)])
	step = max(float(adam[i]["step"]) for i in adam)
	precondition = 1.0 / ((v / (1 - 0.999**step)).sqrt() + 1e-8)
	advantage_config = AdvantageConfig()
	report = {"checkpoint": data["checkpoint"]}
	sources = {"generated": data["generated"], "campaign": data["campaign"]}
	for name, records in sources.items():
		report[name] = describe(records)
	# 4. The joint normalization (as training: per side over all the update's decisions) against per source
	joint_loader, _, _ = make_dataloader(sources["generated"] + sources["campaign"], 256, False, advantage_config=advantage_config)
	n_generated = sum(1 for r in sources["generated"] for a in r.actions if not a.get("forced", False))
	joint = [e["advantage"].abs().item() for e in joint_loader.dataset]
	report["joint_normalization_mean_abs_advantage"] = {"generated": round(float(np.mean(joint[:n_generated])), 3), "campaign": round(float(np.mean(joint[n_generated:])), 3)}
	# 1-3 per source (normalized within the source, as if trained alone) and joint
	means, per = {}, {}
	examples_by = {name: make_dataloader(records, 256, False, advantage_config=advantage_config)[0].dataset for name, records in sources.items()}
	examples_by["joint"] = joint_loader.dataset
	shoot, shooters = shoot_direction(model, examples_by["generated"], params)
	report["shooter_decisions_probed"] = shooters
	for name, examples in examples_by.items():
		order = np.random.default_rng(0).permutation(len(examples))
		batches = [order[i : i + 256] for i in range(0, len(order) - 255, 256)][: args.batches]
		collected = {"policy": [], "value": [], "entropy": [], "floor": []}
		for idx in batches:
			g = term_grads(model, collate([examples[i] for i in idx]), params)
			for term in collected:
				collected[term].append(g[term])
		row = {"batches": len(batches)}
		for term, gs in collected.items():
			stack = torch.stack(gs)
			mean = stack.mean(dim=0)
			means[(name, term)] = mean
			coherence = float(mean.norm() / stack.norm(dim=1).mean())
			# Each minibatch's step's effect on mean log p(SHOOT) through Adam: its mean and t-statistic (is it real?)
			effects = (-(stack * precondition) @ shoot).numpy() if shoot is not None else np.zeros(1)
			row[term] = {
				"norm": float(mean.norm()),
				"coherence": round(coherence, 3),
				"noise_coherence": round(1 / np.sqrt(len(gs)), 3),
				"shoot_adam_mean": round(float(effects.mean()), 2),
				"shoot_adam_t": round(float(effects.mean() / (effects.std(ddof=1) / np.sqrt(len(effects)) + 1e-12)), 2),
			}
		per[name] = row
	report["gradients"] = per
	# The battle advantage's parts on the generated battles, each alone (normalized as training does): the window
	# credit on V (policy_heads (0, 0, 0, 1)) and the 2-round score's label minus its prediction ((0, 0, 1, 0))
	from dataclasses import replace as dc_replace

	parts = {}
	for part, heads in (("window_V_only", (0, 0, 0, 1)), ("round2_score_only", (0, 0, 1, 0)), ("as_trained", advantage_config.policy_heads)):
		config = dc_replace(advantage_config, policy_heads=heads)
		examples = make_dataloader(sources["generated"], 256, False, advantage_config=config)[0].dataset
		order = np.random.default_rng(1).permutation(len(examples))
		gs = [term_grads(model, collate([examples[i] for i in order[k : k + 256]]), params)["policy"] for k in range(0, len(order) - 255, 256)][: args.batches]
		stack = torch.stack(gs)
		effects = (-(stack * precondition) @ shoot).numpy()
		parts[part] = {"coherence": round(float(stack.mean(dim=0).norm() / stack.norm(dim=1).mean()), 3), "shoot_adam_t": round(float(effects.mean() / (effects.std(ddof=1) / np.sqrt(len(effects)))), 2)}
	report["advantage_parts_generated"] = parts
	report["cosine_policy_generated_campaign"] = float(torch.nn.functional.cosine_similarity(means[("generated", "policy")], means[("campaign", "policy")], dim=0))
	report["cosine_policy_generated_joint"] = float(torch.nn.functional.cosine_similarity(means[("generated", "policy")], means[("joint", "policy")], dim=0))
	text = json.dumps(report, indent=1, default=str)
	Path(args.data, "analysis.json").write_text(text)
	print(text)


def run(args):
	"""collect and analyze in one process: the records stay in memory."""
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)
	model, _ = load_model(args.checkpoint)
	onnx = (out / "battle.onnx").resolve()
	export(model, onnx)
	started = time.time()
	generated = collect_generated(onnx, args.generated, args.workers, args.seed)
	print(f"generated: {len(generated)} records, {sum(r.decisions for r in generated)} decisions, {time.time() - started:.0f}s", flush=True)
	started = time.time()
	campaign = collect_campaign(onnx, args.campaign_games, min(args.workers, 6), args.seed)
	print(f"campaign: {len(campaign)} records, {sum(r.decisions for r in campaign)} decisions, {time.time() - started:.0f}s", flush=True)
	args.data = str(out)
	analyze(args, {"checkpoint": str(args.checkpoint), "generated": generated, "campaign": campaign})


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	sub = parser.add_subparsers(dest="command", required=True)
	c = sub.add_parser("collect")
	c.add_argument("--checkpoint", default=str(BEST))
	c.add_argument("--out", required=True)
	c.add_argument("--generated", type=int, default=400)
	c.add_argument("--campaign-games", type=int, default=24)
	c.add_argument("--workers", type=int, default=10)
	c.add_argument("--seed", type=int, default=1)
	a = sub.add_parser("analyze")
	a.add_argument("--data", required=True)
	a.add_argument("--batches", type=int, default=24)
	r = sub.add_parser("run")
	r.add_argument("--checkpoint", default=str(BEST))
	r.add_argument("--out", required=True)
	r.add_argument("--generated", type=int, default=130)
	r.add_argument("--campaign-games", type=int, default=18)
	r.add_argument("--workers", type=int, default=10)
	r.add_argument("--seed", type=int, default=1)
	r.add_argument("--batches", type=int, default=20)
	args = parser.parse_args()
	{"collect": collect, "analyze": analyze, "run": run}[args.command](args)


if __name__ == "__main__":
	main()
