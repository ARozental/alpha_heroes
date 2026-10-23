"""Win rates of a training checkpoint: play the scenario from both sides at a given temperature.

	.venv/bin/python -m battle_model.evaluate runtime/train/<run>/checkpoint_latest.pt [--games 8] [--temperature 0] [--seeds 1,2]

Temperature 0 plays greedily (the most likely action every time): with VCMI's seed fixed the battle is then
deterministic, so one game per side and seed says everything. Records stay in memory.
"""
import argparse
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import torch

from . import play
from .data import summarize
from .export import export
from .model import build_model
from .reward import to_scale
from .train import ROOT, TrainConfig


def load(checkpoint):
	state = torch.load(checkpoint, map_location="cpu")
	model = build_model(state.get("model_kwargs", {}))  # without the key: the first model (model_v1)
	model.load_state_dict(state["model"])
	return model.eval(), state["epoch"]


def evaluate(model, scenario, seeds=(1, 2), games=8, temperature=1.0, sampler_seed=10_000, ai="AlphaBattleAI", opponent="BattleAI", battle_map=""):
	"""{side: [BattleSummary]} for `games` battles per side (one when greedy), side s played with VCMI seed seeds[s]."""
	work = Path(tempfile.mkdtemp(prefix="evaluate-", dir=ROOT / "runtime"))
	try:
		export(model, work / "model.onnx")
		n = 1 if temperature <= 0 else games
		battles = [play.Game(scenario, our_side=side, opponent=opponent, game_seed=seeds[side], sampler_seed=sampler_seed + i, temperature=temperature, ai=ai, map=battle_map) for side in (0, 1) for i in range(n)]
		results = {0: [], 1: []}
		for record in play.play(battles, work / "model.onnx"):
			summary = summarize(record.events)
			results[summary.our_side].append(summary)
		return results
	finally:
		shutil.rmtree(work, ignore_errors=True)


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("checkpoint")
	parser.add_argument("--games", type=int, default=8)
	parser.add_argument("--temperature", type=float, default=1.0)
	parser.add_argument("--seeds", default="1,2", help="VCMI seed of the attack battles, of the defence battles")
	parser.add_argument("--scenario", default=TrainConfig.scenario)
	parser.add_argument("--ai", default="AlphaBattleAI")
	parser.add_argument("--opponent", default="BattleAI")
	parser.add_argument("--map", default="", help="play this existing map (e.g. Maps/alpha/castle_vs_inferno-<run>, the one a run trained on) instead of building one")
	args = parser.parse_args()
	model, epoch = load(args.checkpoint)
	battle_map = args.map or play.build_map(args.scenario, variant=f"evaluate-{os.getpid()}")
	try:
		results = evaluate(model, args.scenario, tuple(int(s) for s in args.seeds.split(",")), args.games, args.temperature, ai=args.ai, opponent=args.opponent, battle_map=battle_map)
	finally:
		if not args.map:
			(play.USER_DIR / (battle_map + ".vmap")).unlink(missing_ok=True)
	for side, name in ((0, "attack"), (1, "defence")):
		summaries = results[side]
		wins = sum(s.outcome.name == "WIN" for s in summaries)
		outcomes = " ".join(f"{k}:{v}" for k, v in sorted(Counter(s.outcome.name for s in summaries).items()))
		mean = sum(to_scale(s.reward) for s in summaries) / len(summaries)
		print(f"epoch {epoch} {name}: won {wins}/{len(summaries)}  mean reward {mean:+.3f}  ({outcomes})")


if __name__ == "__main__":
	main()
