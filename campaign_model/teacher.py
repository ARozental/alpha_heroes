"""Nullkiller2's games as the campaign model's teacher: every player Nullkiller2, each of its turn requests recorded as
our turn decision doing the same (AlphaTeacherAI: agents/AlphaMapAI/Teacher.cpp), saved for imitation (imitate.py).

	.venv/bin/python -m campaign_model.teacher --games 4 --parallel 4 --days 14 --out runtime/teacher/smoke

The training games' maps (random, 36 a side, 2-4 players, underground and water as config.CampaignConfig). Each game is
saved as soon as it ends (game_<seed>.pkl: records.Game); the run stops at --games, --minutes or --max-gb, whichever
comes first, and ends its games on the way out. The audit (audit.json, printed): what the recorder matched, what only
approximately (our nearest amount or area), and what has no equal among our actions, by kind; and MB, MB a minute."""
import argparse
import json
import pickle
import random
import signal
import sys
import time
from collections import Counter
from pathlib import Path

from battle_model import play as battle_play

from . import constants as c
from .config import CampaignConfig
from .play import GameRun, make_map
from .records import TEACHER


def tally(game, totals=None):
	"""What the recorder made of a game's teacher requests, added to totals (a Counter of counts; "type:" and
	"unmatched:" keys by kind); returns totals. The games themselves are not kept (each ~35 MB)."""
	totals = Counter() if totals is None else totals
	totals["games"] += 1
	for summary in game.done.values():
		teacher = summary.get("teacher", {})
		totals["matched"] += teacher.get("matched", 0)
		totals["approximate"] += teacher.get("approximate", 0)
		totals["errors"] += summary.get("errors", 0)
		for kind, count in teacher.get("unmatched", {}).items():
			totals["unmatched:" + kind] += count
	for d in game.decisions:
		if d.flags & TEACHER:
			totals["decisions"] += 1
			totals["type:" + c.ActionType(d.action_type).name] += 1
	return totals


def audit(totals):
	"""tally's counts as the report: shares matched (of the requests) and approximate (of the matched), by kind."""
	unmatched = {k.split(":", 1)[1]: v for k, v in totals.most_common() if k.startswith("unmatched:")}
	requests = totals["matched"] + sum(unmatched.values())
	return {
		"games": totals["games"], "decisions": totals["decisions"], "errors": totals["errors"],
		"matched_share": totals["matched"] / max(1, requests), "approximate_share_of_matched": totals["approximate"] / max(1, totals["matched"]),
		"matched_by_type": {k.split(":", 1)[1]: v for k, v in totals.most_common() if k.startswith("type:")}, "unmatched": unmatched,
	}


def main():
	config = CampaignConfig()
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--games", type=int, default=4)
	parser.add_argument("--parallel", type=int, default=4)
	parser.add_argument("--days", type=int, default=28, help="each game's last day")
	parser.add_argument("--seed", type=int, default=1, help="the first game's (game i: seed + i)")
	parser.add_argument("--out", default="runtime/teacher/smoke")
	parser.add_argument("--minutes", type=float, default=30.0, help="stop starting games after this long, end those playing")
	parser.add_argument("--max-gb", type=float, default=5.0, help="stop once the saved games take this much")
	parser.add_argument("--game-seconds", type=int, default=config.game_seconds)
	args = parser.parse_args()
	signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # (so the games playing are ended on the way out, not orphaned)
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)
	rng = random.Random(args.seed)
	started, saved_bytes, totals, running, next_game = time.time(), 0, Counter(), {}, 0
	deadline = started + 60 * args.minutes
	try:
		while next_game < args.games or running:
			while next_game < args.games and len(running) < args.parallel and time.time() < deadline and saved_bytes < args.max_gb * 1e9:
				seed = args.seed + next_game
				if (out / f"game_{seed}.pkl").exists():  # (a run started again: its games kept, not played twice)
					next_game += 1
					continue
				players, size = rng.choice(config.players), rng.choice(config.sizes)
				map_name = make_map(seed, players, size, config.levels, config.water)[0]
				run = GameRun(battle_play.SOCKET_DIR / "teacher-none.sock", map_name, args.days, seed, max_seconds=args.game_seconds, battle_ai="BattleAI", game_key=f"t{seed}", teacher=True)
				running[seed] = (run, map_name, players, time.time())
				next_game += 1
			if time.time() >= deadline or saved_bytes >= args.max_gb * 1e9:
				next_game = args.games  # (no more games: those playing end now)
				for run, *_ in running.values():
					run.terminate()
			for seed, (run, map_name, players, begun) in list(running.items()):
				if not run.poll():
					continue
				game, info = run.result()
				del running[seed]
				(battle_play.USER_DIR / (map_name + ".vmap")).unlink(missing_ok=True)
				path = out / f"game_{seed}.pkl"
				with open(path.with_suffix(".part"), "wb") as f:  # (renamed once whole: a stop mid-write leaves no torn game)
					pickle.dump(game, f, protocol=pickle.HIGHEST_PROTOCOL)
				path.with_suffix(".part").rename(path)
				saved_bytes += path.stat().st_size
				tally(game, totals)
				one = audit(tally(game))
				del game
				minutes = (time.time() - started) / 60
				print(f"teacher: game {seed} ({players} players, {info.status}, {time.time() - begun:.0f} s): {one['decisions']} decisions, matched {one['matched_share']:.1%}, {path.stat().st_size / 1e6:.0f} MB; total {saved_bytes / 1e6:.0f} MB, {saved_bytes / 1e6 / max(minutes, 1e-9):.1f} MB/min", flush=True)
			time.sleep(0.5)
	finally:
		for run, map_name, *_ in running.values():
			run.terminate()
			(battle_play.USER_DIR / (map_name + ".vmap")).unlink(missing_ok=True)
	report = audit(totals)
	report.update(mb=saved_bytes / 1e6, minutes=(time.time() - started) / 60, mb_per_minute=saved_bytes / 1e6 / max((time.time() - started) / 60, 1e-9))
	(out / "audit.json").write_text(json.dumps(report, indent=1))
	print(json.dumps(report, indent=1))


if __name__ == "__main__":
	main()
