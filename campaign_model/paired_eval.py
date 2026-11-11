"""A behaviour change's paired test before it goes to the PC (2026-11-10, run-away): the campaign evaluation's games
(evaluate.setups: seeds, players, our seat; evaluate.eval_map: the kept maps, the rest made on this machine once and
then the same), 56 days, temperature 1, scored as evaluate.score scores them, every battle BattleAI's (the PC's eval
plays ours with the battle model). Two arms on the same games, then compare: the score's paired difference and its
standard error, the eliminations.

Why paired: one 48-game evaluation against one baseline resolves about +-0.1 at two standard errors (run-away: the
same arms' halves read +0.15 and 0.00), and make_map gives a different map from a seed on every call, so two runs of
"the same seeds" are not the same games unless the maps are made first. The PC's evals after a deploy watch for harm.

	python -m campaign_model.paired_eval maps [games]                     the maps, before the arms (once a machine)
	python -m campaign_model.paired_eval play <checkpoint> <out.pkl> [--games 48] [--parallel 3] [--env NAME=VALUE ...] [--keep-games]
	python -m campaign_model.paired_eval compare <control.pkl> <new.pkl>

An arm plays the code it is run from (another tree: run it from there; its maps are its runtime's, so make them
there too) and the switches it is given (--env: the game's settings, e.g. ALPHA_MAP_FURTHER_WINDOWS=1; default both
schema 17 switches off)."""
import argparse
import os
import pickle
import sys
import threading

import numpy as np

SWITCHES = {"ALPHA_MAP_FURTHER_WINDOWS": "0", "ALPHA_MAP_UNSEEN_EXITS": "0"}


def play_arm(checkpoint, out, games, parallel, env, keep_games=False):
	from .config import CampaignConfig
	from .evaluate import eval_map, score, setups
	from .play import play
	from .train import serve_model

	os.environ.update({**SWITCHES, **env})  # (the games inherit them)
	config, results, kept, gate = CampaignConfig(), {}, {}, threading.Semaphore(parallel)
	with serve_model(checkpoint, processes=parallel) as (_, server):
		def one(i, seed, players, color):
			name = eval_map(seed, players, config)
			with gate:
				game, info = play(server.socket_path, name, 56, seed, max_seconds=3600, max_mb=300, battle_ai="BattleAI", our_battle_ai="BattleAI", ours=(color,), temperature=1.0, game_key=f"p{os.getpid()}-{i}")
			results[i] = {**score(game, color, players, 56), "seed": seed, "players": players, "color": color, "map": name, "status": info.status}
			if keep_games:  # (the records, for what happened in them: <out>.games.pkl)
				kept[i] = game
			print(f"  {i} {seed} {players}p {color} {info.status[:40]} score {results[i]['score']:.3f}", flush=True)

		threads = [threading.Thread(target=one, args=(i, *g)) for i, g in enumerate(setups(games))]
		for t in threads:
			t.start()
		for t in threads:
			t.join()
	pickle.dump({"results": results, "checkpoint": checkpoint, "env": {**SWITCHES, **env}}, open(out, "wb"))
	if keep_games:
		pickle.dump(kept, open(out + ".games.pkl", "wb"))
	print(f"saved {out}")


def compare(control, new):
	"""(text lines) the arms' scores on their common games: means, the paired difference and its standard error, day 28's,
	the eliminations."""
	a, b = (pickle.load(open(path, "rb")) for path in (control, new))
	a, b = a.get("results", a), b.get("results", b)
	keys = sorted(set(a) & set(b))
	d = np.array([b[k]["score"] - a[k]["score"] for k in keys])
	d28 = np.array([b[k]["score_d28"] - a[k]["score_d28"] for k in keys])
	se = lambda v: v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")  # noqa: E731
	eliminated = lambda r: sum(bool(r[k]["eliminated"]) for k in keys)  # noqa: E731
	return [
		f"{len(keys)} common games: score {np.mean([a[k]['score'] for k in keys]):.3f} -> {np.mean([b[k]['score'] for k in keys]):.3f}, paired {d.mean():+.3f} +- {se(d):.3f} (se); day 28 {d28.mean():+.3f} +- {se(d28):.3f}",
		f"eliminated {eliminated(a)} -> {eliminated(b)}; better on {int((d > 0).sum())} games, worse on {int((d < 0).sum())}",
	]


def main():
	parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
	sub = parser.add_subparsers(dest="command", required=True)
	maps = sub.add_parser("maps")
	maps.add_argument("games", nargs="?", type=int, default=48)
	arm = sub.add_parser("play")
	arm.add_argument("checkpoint")
	arm.add_argument("out")
	arm.add_argument("--games", type=int, default=48)
	arm.add_argument("--parallel", type=int, default=3)
	arm.add_argument("--env", nargs="*", default=[], metavar="NAME=VALUE")
	arm.add_argument("--keep-games", action="store_true", help="also <out>.games.pkl: each game's records by its index")
	both = sub.add_parser("compare")
	both.add_argument("control")
	both.add_argument("new")
	args = parser.parse_args()
	if args.command == "maps":
		from .config import CampaignConfig
		from .evaluate import eval_map, setups

		for seed, players, color in setups(args.games):
			print(eval_map(seed, players, CampaignConfig()), color, flush=True)
	elif args.command == "play":
		play_arm(args.checkpoint, args.out, args.games, args.parallel, dict(item.split("=", 1) for item in args.env), args.keep_games)
	else:
		print("\n".join(compare(args.control, args.new)))


if __name__ == "__main__":
	sys.exit(main())
