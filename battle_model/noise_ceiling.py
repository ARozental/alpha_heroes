"""How much of a battle's outcome no value head can predict: the same position played on several times.

Generated self-play battles are played once; each is cut at a round, both sides replay their recorded actions before
it (the same VCMI seed: the same dice, so the same position), then go on sampling with new seeds, `continuations`
times. Outcome noise given the position is the variance across continuations of one position; the best explained
variance any value head could reach on these positions is 1 - (that) / (the outcomes' total variance). Our value head's
own (its V at the cut against the continuations' outcomes) shows how far from that it is.

Usage: .venv/bin/python -m battle_model.noise_ceiling --model <onnx> [--battles 30] [--continuations 6]

KNOWN ISSUES (docs/adversarial_review_2026-10-27.md #7, open): a battle with a tactics phase breaks the replay (the
placements are sampled again, then a TACTICS_* prefix action is "not legal here": the run stops); the continuations
share VCMI's dice after the cut (the same server seed), so the noise is underestimated and the ceiling overstated
(the fix: reseed at the cut, the IBattleRandomizer hook); V is taken at side 0's first decision of the cut round,
which may follow side 1's moves. Do not trust its numbers until these are fixed.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

from . import generator, play
from .data import summarize


def cut_round(record, rng):
	"""A round in the middle of the battle, where our side has a decision (None for a battle too short)."""
	rounds = sorted({int(r) for r in record.tensors["g_round"]})
	return rng.choice(rounds[1:-1]) if len(rounds) >= 3 else None


def prefix_before(record, round_):
	"""Our side's recorded actions before round_ (play.Game.prefix entries)."""
	return [dict(a) for a, r in zip(record.actions, record.tensors["g_round"]) if int(r) < round_]


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--model", required=True)
	parser.add_argument("--battles", type=int, default=30)
	parser.add_argument("--continuations", type=int, default=6)
	parser.add_argument("--seed", type=int, default=1)
	args = parser.parse_args()
	from . import train

	rng = random.Random(args.seed)
	config = train.TrainConfig(generator=True, run="noise", opponent=play.SELF_PLAY)
	scenarios = [generator.sample_type(rng, generator.load_config(), generator.world()) for _ in range(args.battles)]
	games, files = train.generated_games(scenarios, "noise", config, our_sides=(0,), opponent=play.SELF_PLAY, temperature=1.0)
	games = [play.replace(g, sampler_seed=rng.randrange(2**31)) for g in games]
	first = train.play_in_groups(games, args.model, config, "noise first")  # two records a battle (self-play)
	play.PREFIX_DIR.mkdir(parents=True, exist_ok=True)
	positions, continued = [], []
	for i, game in enumerate(games):
		records = first[2 * i : 2 * i + 2]
		round_ = cut_round(records[0], rng)
		if round_ is None:
			continue
		prefixes = []
		for side, record in enumerate(records):
			path = play.PREFIX_DIR / f"noise-{i}-{side}.jsonl"
			path.write_text("".join(json.dumps(a) + "\n" for a in prefix_before(record, round_)))
			prefixes.append((f"ALPHA_BATTLE_PREFIX_{game.colors[side]}", str(path)))
			files.append(path)
		at_cut = [k for k, r in enumerate(records[0].tensors["g_round"]) if int(r) == round_]
		value = float(records[0].tensors["value"][at_cut[0], -1]) if at_cut else float("nan")
		positions.append((i, round_, value))
		continued += [play.replace(game, settings=game.settings + tuple(prefixes), sampler_seed=rng.randrange(2**31)) for _ in range(args.continuations)]
	results = train.play_in_groups(continued, args.model, config, "noise continuations")
	rewards = np.array([summarize(r.events).reward for r in results[0::2]]).reshape(len(positions), args.continuations)  # side 0's
	values = np.array([v for _, _, v in positions])
	within = rewards.var(axis=1, ddof=1).mean()
	total = rewards.var(ddof=1)
	ceiling = 1 - within / total
	ours = 1 - np.nanmean((rewards - values[:, None]) ** 2) / total
	print(f"{len(positions)} positions x {args.continuations} continuations: outcome variance {total:.4f}, of it noise given the position {within:.4f}")
	print(f"the best explained variance any value head could reach here: {ceiling:.3f}; ours (V at the cut): {ours:.3f}")
	for path in files:
		Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
	main()
