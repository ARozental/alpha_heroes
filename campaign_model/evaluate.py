"""Evaluation of the campaign model against Nullkiller2 (learner.py runs it every eval_minutes): eval_games games on
fixed random maps (the same in every evaluation of every run: EVAL_SEED; 2, 3 and 4 players in turn, 36 x 36 with
the underground), our campaign model in one seat (it moves round the seats from game to game), Nullkiller2 in the
others. Our battles are played by our battle model, greedy (as the battle evals); Nullkiller2's and the wandering
monsters' by BattleAI. Each game stops when the day after eval_days begins.

	score   our share of all empires' value (empire.py) at the end of eval_days, times the number of players: 1 = as
	        much as the average player of its game, Nullkiller2s included; the mean over the games, and its standard
	        error (score_se)
	vs_previous, vs_first
	        paired: the mean over the games of this eval's score minus the previous eval's (the first eval's: the model
	        the run started with) on the same game (map, seed and seat), and its standard error. A map's and a seat's
	        luck is the same for both, so it cancels: the version-to-version difference is measured, not two noisy
	        means apart
The ledger is our AI's (one per day, of every player), so a game where we were knocked out early still has one; our
share is then 0 (labels.py).
"""
import csv
import json
import random
from pathlib import Path
import shutil
import threading
import time

import numpy as np

from battle_model import play as battle_play

from .empire import EMPIRE_VERSION, STATUS_INGAME, STATUS_LOST, EmpireConfig, empire_value
from .failures import NAMES as FAILURES, failure_counts, failure_modes
from .labels import COLORS, share_at, share_table
from .play import make_map, play

EVAL_SEED = 20261027
STATUS_WON = 2  # EPlayerStatus::WINNER
EVAL_SIZE = 36
COLUMNS = [
	"version", "epoch", "games", "score", "score_se", "vs_previous", "vs_previous_se", "vs_first", "vs_first_se", "score_d28", "score_2p", "score_3p", "score_4p", "share", "phi", "phi_nullkiller",
	"heroes", "heroes_nullkiller", "eliminated", "won", "short", "ai_errors", "failed", "seconds", *(f"failure_{k}" for k in FAILURES),
]


def setups(count):
	"""(seed, players, our color) of each evaluation game: the same for every run."""
	rng = random.Random(EVAL_SEED)
	games = []
	for i in range(count):
		players = (2, 3, 4)[i % 3]
		games.append((rng.randrange(1, 2**31), players, COLORS[(i // 3) % players]))
	return games


# The evaluation's maps, kept with the code: VCMI's generator is not deterministic by seed (2026-10-30: the same binary
# and seed made 3 maps in 3 runs; the PC's eval maps shared no surface tile with the Mac's, so its scores were not the
# Mac's), so every machine plays these, the Mac's set that v1-v36 were scored on
EVAL_MAPS = Path(__file__).resolve().parent / "eval_maps"


def eval_map(seed, players, config):
	"""The game's map: the kept one (EVAL_MAPS, copied into VCMI's maps when missing or different there), else one
	generated and kept on this machine."""
	for attempt in range(5):  # (make_map's next seeds when a player was left without a town)
		name = f"Maps/alpha_random/s{EVAL_SIZE}-p{players}-l{config.levels}-{config.water}-{seed + 1_000_003 * attempt}"
		kept, there = EVAL_MAPS / (name.rsplit("/", 1)[-1] + ".vmap"), battle_play.USER_DIR / (name + ".vmap")
		if kept.exists():
			if not there.exists() or there.read_bytes() != kept.read_bytes():
				there.parent.mkdir(parents=True, exist_ok=True)
				shutil.copyfile(kept, there)
			return name
		if there.exists():
			return name
	return make_map(seed, players, EVAL_SIZE, config.levels, config.water)[0]


def score(game, color, players, days, empire=EmpireConfig()):
	"""One game's results for our player (color), from the snapshot at the end of `days` or the game's last (a
	knock-out ends the game for us: the end-of-game snapshot, after the day's)."""
	table, last = share_table(game, empire)
	share = share_at(table, color, days, last)
	snapshots = sorted([*game.ledgers.values(), *game.final], key=lambda s: (s["day"], bool(s.get("final"))))
	snapshot = game.ledgers.get(days + 1) or (snapshots[-1] if snapshots else {"players": []})
	ours = [p for p in snapshot["players"] if COLORS[p["color"]] == color]
	others = [p for p in snapshot["players"] if COLORS[p["color"]] != color]
	status = ours[0]["status"] if ours else STATUS_LOST  # VCMI's EPlayerStatus: in the game, lost, won
	return {
		"score": share * players,
		"score_d28": share_at(table, color, min(28, days), last) * players,
		"players": players,
		"share": share,
		"phi": empire_value(ours[0], empire)["total"] if ours else 0.0,
		"phi_nullkiller": float(np.mean([empire_value(p, empire)["total"] for p in others])) if others else float("nan"),
		"heroes": len(ours[0]["heroes"]) if ours and status == STATUS_INGAME else 0,
		"heroes_nullkiller": float(np.mean([len(p["heroes"]) for p in others if p["status"] == STATUS_INGAME] or [0])),
		"eliminated": status == STATUS_LOST,
		"won": status == STATUS_WON,
		"short": last < days and status == STATUS_INGAME,  # cut before the end (time limit, crash): its last day stands in
		"ai_errors": sum(s.get("errors", 0) for s in game.done.values()),
	}


def evaluate(config, server, battle, version, epoch, stop=None, model="current"):
	"""Play the evaluation games at once; returns the eval row. server: the campaign model's inference server, model:
	the name it serves the evaluated weights under; battle: the battle learner's actors (actors.ActorPool.eval_settings:
	our battles, greedy, unrecorded)."""
	started = time.time()
	games = setups(config.eval_games)
	results = [None] * len(games)
	slots = threading.Semaphore(max(1, config.eval_parallel))

	def work(i):
		seed, players, color = games[i]
		with slots:
			play_one(i, seed, players, color)

	def play_one(i, seed, players, color):
		try:
			name = eval_map(seed, players, config)
			game, info = play(
				server.socket_for(i), name, config.eval_days, seed, max_seconds=config.eval_seconds, battle_ai="BattleAI", game_key=f"e{i}",
				settings=battle.eval_settings(i) + (("ALPHA_MAP_MODEL", model),), temperature=config.eval_temperature, ours=(color,), stop=stop,
			)
			results[i] = {**score(game, color, players, config.eval_days), "setup": i, "seed": seed, "map": name, "color": color, "temperature": config.eval_temperature, "empire": EMPIRE_VERSION, "seconds_limit": config.eval_seconds, "status": info.status, "seconds": round(info.seconds, 1), "failures": failure_counts(game.decisions)}
			if info.status.startswith("stopped at day") or "exit code 0" in info.status:
				shutil.rmtree(info.run_dir, ignore_errors=True)  # (a failed game's logs are kept)
		except Exception as error:  # left out of the score, counted
			print(f"campaign eval: game {i} failed: {str(error)[:300]}", flush=True)

	threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(len(games))]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()
	played = [r for r in results if r is not None]
	mean = lambda values: float(np.mean(values)) if len(values) else float("nan")  # noqa: E731
	scores = [r["score"] for r in played]
	row = {
		"version": version,
		"epoch": epoch,
		"games": len(played),
		"score": mean(scores),
		"score_se": float(np.std(scores, ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else float("nan"),
		**{f"score_{n}p": mean([r["score"] for r in played if r["players"] == n]) for n in (2, 3, 4)},
		**{key: mean([r[key] for r in played]) for key in ("score_d28", "share", "phi", "phi_nullkiller", "heroes", "heroes_nullkiller")},
		"eliminated": sum(r["eliminated"] for r in played),
		"won": sum(r["won"] for r in played),
		"short": sum(r["short"] for r in played),
		"ai_errors": sum(r["ai_errors"] for r in played),
		"failed": len(games) - len(played),
		"seconds": time.time() - started,
		**{f"failure_{k}": v for k, v in failure_modes(counts={k: tuple(sum(r["failures"][k][j] for r in played) for j in (0, 1)) for k in FAILURES}).items()},
		"per_game": played,  # (eval_games.jsonl)
	}
	return row


def paired(games, earlier):
	"""(mean, standard error) of this eval's score minus an earlier one's, game by game on the same setups (nan when
	fewer than 2 games match)."""
	before = {(g.get("seed"), g["map"], g["color"]): g["score"] for g in earlier}
	diff = [g["score"] - before[key] for g in games if (key := (g.get("seed"), g["map"], g["color"])) in before]
	if len(diff) < 2:
		return float("nan"), float("nan")
	return float(np.mean(diff)), float(np.std(diff, ddof=1) / np.sqrt(len(diff)))


def earlier_evals(path):
	"""{eval: [game results]} of the evals already in eval_games.jsonl, in order: each eval by its number (lines from
	before it had one: by version; two evals at one version, after a resume, are two evals: 2026-10-28 review)."""
	evals = {}
	if path.exists():
		for line in path.read_text().splitlines():
			if line.strip():
				game = json.loads(line)
				evals.setdefault(game.get("eval", f"version {game['version']}"), []).append(game)
	return evals


def log(row, path, writer=None):
	"""Append the row to path (eval.csv) and each game's results to eval_games.jsonl next to it; first its paired
	differences with the previous eval and the first (vs_previous, vs_first)."""
	# (only evals that measure the same thing: another temperature's own sampling, or another version of Φ, whose shares
	# are the score, is another measure, not a change)
	measure = lambda game: (game.get("temperature", 1.0), game.get("empire", 1), game.get("seconds_limit", 240))  # noqa: E731 (a time limit that cuts games short scores them early)
	same = lambda games: measure(games[0]) == measure(row["per_game"][0]) if games and row.get("per_game") else False  # noqa: E731
	everything = list(earlier_evals(path.with_name("eval_games.jsonl")).values())
	evals = [games for games in everything if same(games)]
	row["vs_previous"], row["vs_previous_se"] = paired(row.get("per_game", []), evals[-1]) if evals else (float("nan"), float("nan"))
	row["vs_first"], row["vs_first_se"] = paired(row.get("per_game", []), evals[0]) if evals else (float("nan"), float("nan"))
	with path.with_name("eval_games.jsonl").open("a") as f:
		for game in row.get("per_game", []):
			f.write(json.dumps({"eval": len(everything) + 1, "version": row["version"], **game}) + "\n")
	new = not path.exists()
	with path.open("a", newline="") as f:
		if new:
			csv.writer(f).writerow(COLUMNS)
		csv.writer(f).writerow([round(row[k], 5) if isinstance(row[k], float) else row[k] for k in COLUMNS])
	if writer is not None:
		for key in COLUMNS[2:]:
			if isinstance(row[key], (int, float)) and row[key] == row[key]:
				writer.add_scalar(f"campaign_eval/{key}", row[key], row["version"])
		# TensorBoard's eval tab: the top-level scores only, eval/battle beside this (battle_model/train.py EVAL_TAB);
		# self-play training never shows there
		writer.add_scalar("eval/campaign", row["score"], row["version"])
		writer.flush()
	print(
		f"campaign eval {row['version']:4d}  SCORE {row['score']:.3f} ± {row['score_se']:.3f} (1 = an average player; Nullkiller2 elsewhere; day 28: {row['score_d28']:.3f})"
		f"  paired: vs the previous eval {row['vs_previous']:+.3f} ± {row['vs_previous_se']:.3f}, vs the first {row['vs_first']:+.3f} ± {row['vs_first_se']:.3f}"
		f"  Φ ours {row['phi']:.0f} vs Nullkiller2 {row['phi_nullkiller']:.0f}  heroes {row['heroes']:.2f} vs {row['heroes_nullkiller']:.2f}"
		f"  eliminated {row['eliminated']}  short {row['short']}  failed {row['failed']}  {row['seconds']:.0f}s",
		flush=True,
	)
