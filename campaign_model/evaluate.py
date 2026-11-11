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

Split evaluation (config.eval_own_games, eval_follow_games; the user, 2026-11-11): the OWN arm plays the first
eval_own_games setups with Nullkiller2's own choice hidden (ALPHA_MAP_NK2_CHOICE=0, its other inputs on: our own
policy, the score it is judged by: score_own), the FOLLOW arm the even ones among them with it shown (the run's follow
chance: score_follow, paired with OWN on those: follow_minus_own). The old score columns stay empty in such a row
(they meant every game with the choice shown); own_vs_previous pairs the OWN arm with the last evaluation that had one.
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
from battle_model import supply
from battle_model.supply import memory_allows

from .empire import EVAL_EMPIRE, EVAL_EMPIRE_VERSION, STATUS_INGAME, STATUS_LOST, empire_value
from .failures import NAMES as FAILURES, failure_counts, failure_modes
from .labels import COLORS, share_at, share_table
from .play import make_map, nk2_settings, play, time_scale, window_settings

EVAL_SEED = 20261027
STATUS_WON = 2  # EPlayerStatus::WINNER
EVAL_SIZE = 36
COLUMNS = [
	"version", "epoch", "games", "score", "score_se", "vs_previous", "vs_previous_se", "vs_first", "vs_first_se", "score_d28", "score_2p", "score_3p", "score_4p", "share", "phi", "phi_nullkiller",
	"heroes", "heroes_nullkiller", "eliminated", "won", "short", "ai_errors", "failed", "seconds", *(f"failure_{k}" for k in FAILURES),
	# (the split evaluation's: the OWN arm, its choice hidden; the FOLLOW arm, shown, on a subset of OWN's setups)
	"games_own", "score_own", "score_own_se", "score_own_d28", "eliminated_own", "won_own", "phi_own", "own_vs_previous", "own_vs_previous_se",
	"games_follow", "score_follow", "score_follow_se", "score_follow_d28", "eliminated_follow", "won_follow", "follow_minus_own", "follow_minus_own_se",
	"idle_movement_own", "idle_movement_follow",  # (failures.py's idle_movement in each arm: config.schedule_action_price is judged by OWN's)
]
# (a split evaluation's row leaves these empty: they meant every game with Nullkiller2's choice shown)
SHOWN_ONLY = ("score", "score_se", "vs_previous", "vs_previous_se", "vs_first", "vs_first_se", "score_d28", "score_2p", "score_3p", "score_4p", "share", "phi", "phi_nullkiller", "heroes", "heroes_nullkiller", "eliminated", "won")


def plan(config):
	"""[(setup index, arm)] of an evaluation's games: every setup of eval_games with Nullkiller2's choice shown ("shown");
	or, split (config.eval_own_games > 0), the first eval_own_games with it hidden ("own": 11 / 11 / 10 games of 2, 3
	and 4 players at 32, setups cycling the player counts) and eval_follow_games of those with it shown ("shown":
	within each player count every other one by our seat, 6 / 5 / 5 at 16). The same in every evaluation, so
	evaluations compare from the first split one on."""
	own, follow = int(getattr(config, "eval_own_games", 0) or 0), int(getattr(config, "eval_follow_games", 0) or 0)
	if own <= 0:
		return [(i, "shown") for i in range(config.eval_games)]
	# FOLLOW's: within each player count, OWN's setups by our seat, every other one (so the seats and the counts are
	# spread as OWN's are), the counts cut in turn from the most to `follow`
	games = setups(own)
	by_players = {n: sorted((i for i in range(own) if games[i][1] == n), key=lambda i: (COLORS.index(games[i][2]), i))[::2] for n in (2, 3, 4)}
	while sum(map(len, by_players.values())) > follow:
		longest = max(by_players, key=lambda n: (len(by_players[n]), n))
		by_players[longest].pop()
	return [(i, "own") for i in range(own)] + [(i, "shown") for i in sorted(i for chosen in by_players.values() for i in chosen)]


def eval_score(row):
	"""The evaluation's score of our own policy: the OWN arm's in a split evaluation (score_own), else the score; nan
	without one. Everything driven by the score reads this (learner's teacher weight, TensorBoard's eval/campaign)."""
	for key in ("score_own", "score"):
		try:
			value = float(row.get(key, ""))
		except (TypeError, ValueError):
			continue
		if value == value:
			return value
	return float("nan")


def arm_settings(arm):
	"""A game's own-choice switch: hidden in the OWN arm (its other inputs as nk2_settings gives them)."""
	return (("ALPHA_MAP_NK2_CHOICE", "0"),) if arm == "own" else ()


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
		name = f"Maps/alpha_random/s{config.eval_size}-p{players}-l{config.levels}-{config.water}-{seed + 1_000_003 * attempt}"
		kept, there = EVAL_MAPS / (name.rsplit("/", 1)[-1] + ".vmap"), battle_play.USER_DIR / (name + ".vmap")
		if kept.exists():
			if not there.exists() or there.read_bytes() != kept.read_bytes():
				there.parent.mkdir(parents=True, exist_ok=True)
				shutil.copyfile(kept, there)
			return name
		if there.exists():
			return name
	return make_map(seed, players, config.eval_size, config.levels, config.water)[0]


STALLED = "killed: no log output"  # (scripts/headless.sh's status for a game silent past its stall limit)


def score(game, color, players, days, empire=EVAL_EMPIRE):
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


def count_waiting(shared, change):
	"""The eval games waiting for memory, in the trainer's supply.Shared (none: no supply, nothing to hold back)."""
	if shared is not None:
		with shared.lock:
			shared.eval_waiting.value += change


def evaluate(config, server, battle, version, epoch, stop=None, model="current"):
	"""Play the evaluation games at once; returns the eval row. server: the campaign model's inference server, model:
	the name it serves the evaluated weights under; battle: the battle learner's actors (actors.ActorPool.eval_settings:
	our battles, greedy, unrecorded)."""
	started = time.time()
	arms = plan(config)
	games = setups(max(config.eval_games, max(i for i, _ in arms) + 1))
	results = [None] * len(arms)
	slots = threading.Semaphore(max(1, config.eval_parallel))
	lock, playing = threading.Lock(), [0]  # (eval games playing now)

	def work(j):
		i, arm = arms[j]
		seed, players, color = games[i]
		with slots:
			# Under the memory floor as the training's games are (supply.memory_allows: the trainer's peak kept free), and
			# first to the memory freed meanwhile (supply.Shared.eval_waiting: no training game starts while one waits)
			# (one eval game always plays while supply.LAST_GAME_GB is available: the memory may be the trainer's, which only
			# games bring back)
			shared = supply.SHARED
			if not memory_allows(playing=playing[0]):
				count_waiting(shared, +1)
				try:
					while not memory_allows(playing=playing[0]) and not (stop is not None and stop.is_set()):
						time.sleep(5)
				finally:
					count_waiting(shared, -1)
			with lock:
				playing[0] += 1
			try:
				play_one(j, i, arm, seed, players, color)
			finally:
				with lock:
					playing[0] -= 1

	def play_one(j, i, arm, seed, players, color):  # (j: the game's slot in this evaluation, i: its setup)
		try:
			name = eval_map(seed, players, config)
			# A game the watchdog ended for silence is played again, then left out: it was scored at the last day it reached
			# (as a time limit's is), and at a resume the inference was not ready yet: four games killed in their first
			# minute stood in at day 1-3 (scores 0.28-0.72) and lifted v229's eval from 0.219 to 0.267 (2026-10-31)
			for attempt in range(2):
				game, info = play(
					server.socket_for(j), name, config.eval_days, seed, max_seconds=config.eval_seconds * time_scale(config.eval_size), battle_ai="BattleAI", game_key=f"e{j}",
					settings=battle.eval_settings(j) + (("ALPHA_MAP_MODEL", model),) + window_settings(config) + tuple(pair for pair in nk2_settings(config) if pair[0] != "ALPHA_MAP_NK2_CHOICE" or arm != "own") + arm_settings(arm), temperature=config.eval_temperature, ours=(color,), stop=stop,
					niceness=config.eval_niceness,
				)
				if not info.status.startswith(STALLED):
					break
				print(f"campaign eval: game {j} {info.status[:60]}" + (": played again" if not attempt else ": left out"), flush=True)
			if info.status.startswith(STALLED):
				raise RuntimeError(f"stalled twice ({info.status[:100]})")
			results[j] = {**score(game, color, players, config.eval_days), "setup": i, "arm": arm, "seed": seed, "map": name, "color": color, "temperature": config.eval_temperature, "empire": EVAL_EMPIRE_VERSION, "seconds_limit": config.eval_seconds, "status": info.status, "seconds": round(info.seconds, 1), "failures": failure_counts(game.decisions)}
			if info.status.startswith("stopped at day") or "exit code 0" in info.status:
				shutil.rmtree(info.run_dir, ignore_errors=True)  # (a failed game's logs are kept)
		except Exception as error:  # left out of the score, counted
			print(f"campaign eval: game {j} failed: {str(error)[:300]}", flush=True)

	threads = [threading.Thread(target=work, args=(j,), daemon=True) for j in range(len(arms))]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()
	if stop is not None and stop.is_set():
		# Cut by the run's stop: no row (as the battle evals). Eval 53 of joint7 (2026-11-01 15:37 CST, a restart) scored
		# its 48 games "stopped: training is over" as eliminated: 0.000, and the teacher's weight went 16.4 -> 19 by it
		print(f"campaign eval: stopped with the run after {time.time() - started:.0f}s: no row", flush=True)
		return None
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
		"failed": len(arms) - len(played),
		"seconds": time.time() - started,
		**{f"failure_{k}": v for k, v in failure_modes(counts={k: tuple(sum(r["failures"][k][j] for r in played) for j in (0, 1)) for k in FAILURES}).items()},
		"per_game": played,  # (eval_games.jsonl)
	}
	if arms[0][1] == "own":  # (a split evaluation)
		row.update(split_row(played))
	return row


def split_row(played):
	"""A split evaluation's columns from its games: each arm's score (its standard error, day 28's), eliminations, wins,
	our Φ in OWN; the FOLLOW arm minus OWN, setup by setup; the old score columns empty (SHOWN_ONLY)."""
	mean = lambda values: float(np.mean(values)) if len(values) else float("nan")  # noqa: E731
	se = lambda values: float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")  # noqa: E731
	out = {key: "" for key in SHOWN_ONLY}
	for arm, name in (("own", "own"), ("shown", "follow")):
		games = [r for r in played if r["arm"] == arm]
		scores = [r["score"] for r in games]
		out.update({f"games_{name}": len(games), f"score_{name}": mean(scores), f"score_{name}_se": se(scores), f"score_{name}_d28": mean([r["score_d28"] for r in games]),
			f"eliminated_{name}": sum(r["eliminated"] for r in games), f"won_{name}": sum(r["won"] for r in games)})
		idle = [r["failures"]["idle_movement"] for r in games if "failures" in r]
		out[f"idle_movement_{name}"] = sum(h for h, _ in idle) / sum(t for _, t in idle) if sum(t for _, t in idle) else float("nan")
	out["phi_own"] = mean([r["phi"] for r in played if r["arm"] == "own"])
	own = {r["setup"]: r["score"] for r in played if r["arm"] == "own"}
	diff = [r["score"] - own[r["setup"]] for r in played if r["arm"] == "shown" and r["setup"] in own]
	out["follow_minus_own"], out["follow_minus_own_se"] = mean(diff), se(diff)
	return out


def paired(games, earlier):
	"""(mean, standard error) of this eval's score minus an earlier one's, game by game on the same setups (nan when
	fewer than 2 games match)."""
	before = {(g.get("seed"), g["map"], g["color"], g.get("arm", "shown")): g["score"] for g in earlier}
	diff = [g["score"] - before[key] for g in games if (key := (g.get("seed"), g["map"], g["color"], g.get("arm", "shown"))) in before]
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
	split = any(g.get("arm") == "own" for g in row.get("per_game", []))  # (a split evaluation: an OWN arm among its games)
	if not split:
		row["vs_previous"], row["vs_previous_se"] = paired(row.get("per_game", []), evals[-1]) if evals else (float("nan"), float("nan"))
		row["vs_first"], row["vs_first_se"] = paired(row.get("per_game", []), evals[0]) if evals else (float("nan"), float("nan"))
	else:  # (OWN against the last evaluation that had an OWN arm)
		owned = [games for games in evals if any(g.get("arm") == "own" for g in games)]
		ours = [g for g in row.get("per_game", []) if g.get("arm") == "own"]
		row["own_vs_previous"], row["own_vs_previous_se"] = paired(ours, owned[-1]) if owned else (float("nan"), float("nan"))
	with path.with_name("eval_games.jsonl").open("a") as f:
		for game in row.get("per_game", []):
			f.write(json.dumps({"eval": len(everything) + 1, "version": row["version"], **game}) + "\n")
	new = not path.exists()
	with path.open("a", newline="") as f:
		if new:
			csv.writer(f).writerow(COLUMNS)
		csv.writer(f).writerow([round(row[k], 5) if isinstance(row.get(k, ""), float) else row.get(k, "") for k in COLUMNS])
	if writer is not None:
		for key in COLUMNS[2:]:
			if isinstance(row.get(key), (int, float)) and row[key] == row[key]:
				writer.add_scalar(f"campaign_eval/{key}", row[key], row["version"])
		# TensorBoard's eval tab: the top-level scores only, eval/battle beside this (battle_model/train.py EVAL_TAB);
		# self-play training never shows there. Our own policy's score (eval_score: OWN's in a split evaluation)
		writer.add_scalar("eval/campaign", eval_score(row), row["version"])
		if split:
			writer.add_scalar("eval/campaign_follow", row["score_follow"], row["version"])
		writer.flush()
	if split:
		print(
			f"campaign eval {row['version']:4d}  OWN (its choice hidden) {row['score_own']:.3f} ± {row['score_own_se']:.3f} on {row['games_own']} games (day 28: {row['score_own_d28']:.3f}), eliminated {row['eliminated_own']}, won {row['won_own']}"
			f"  FOLLOW (shown) {row['score_follow']:.3f} ± {row['score_follow_se']:.3f} on {row['games_follow']}, eliminated {row['eliminated_follow']}, won {row['won_follow']}"
			f"  follow - own on the same setups {row['follow_minus_own']:+.3f} ± {row['follow_minus_own_se']:.3f}  OWN vs the previous OWN {row['own_vs_previous']:+.3f} ± {row['own_vs_previous_se']:.3f}"
			f"  short {row['short']}  failed {row['failed']}  {row['seconds']:.0f}s",
			flush=True,
		)
		return
	print(
		f"campaign eval {row['version']:4d}  SCORE {row['score']:.3f} ± {row['score_se']:.3f} (1 = an average player; Nullkiller2 elsewhere; day 28: {row['score_d28']:.3f})"
		f"  paired: vs the previous eval {row['vs_previous']:+.3f} ± {row['vs_previous_se']:.3f}, vs the first {row['vs_first']:+.3f} ± {row['vs_first_se']:.3f}"
		f"  Φ ours {row['phi']:.0f} vs Nullkiller2 {row['phi_nullkiller']:.0f}  heroes {row['heroes']:.2f} vs {row['heroes_nullkiller']:.2f}"
		f"  eliminated {row['eliminated']}  short {row['short']}  failed {row['failed']}  {row['seconds']:.0f}s",
		flush=True,
	)
