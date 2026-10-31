"""Games of the campaign model: a random map from VCMI's generator (water and the underground on), every player
AlphaMapAI asking the inference server (family "campaign"), or (an evaluation) some players Nullkiller2; battles
played by a battle AI (retreats and surrenders are the battle AI's decisions, not the campaign model's). The AIs stop
at a day cap and the game is killed as soon as each of our players still in it has sent its last records;
scripts/headless.sh caps its time and disk.

	.venv/bin/python -m campaign_model.play --days 5 --seed 1
"""
import argparse
import json
import os
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from battle_model import play as battle_play

from .records import collect

ROOT = battle_play.ROOT
USER_DIR = battle_play.USER_DIR
MAP_DIR = "Maps/alpha_random"
COLORS = ("RED", "BLUE", "TAN", "GREEN", "ORANGE", "PURPLE", "TEAL", "PINK")
STATUS_INGAME = 0


@dataclass
class GameInfo:
	map_name: str
	template: str
	seconds: float
	status: str  # how the game ended
	run_dir: str
	run_mb: float  # the run directory (logs) on disk


def players_without_a_town(path):
	"""The players of a generated map (.vmap) with no starting town: the generator sometimes leaves one out (template
	8SM0f gave red none), and such a player is out at once (and red's loss ended the whole game)."""
	import re
	import zipfile

	text = zipfile.ZipFile(path).read("header.json").decode()
	header = json.loads(re.sub(r",(\s*[}\]])", r"\1", re.sub(r"//[^\n]*", "", text)))
	return sorted(color for color, player in header.get("players", {}).items() if not player.get("mainTown"))


def make_map(seed, players=2, size=36, levels=2, water="random", attempts=5):
	"""Write a random map into the VCMI user data folder; returns (map name, template). A map where a player has no
	starting town is generated again from the next seed (seed + 1000003 x attempt)."""
	for attempt in range(attempts):
		this_seed = seed + 1_000_003 * attempt
		name = f"{MAP_DIR}/s{size}-p{players}-l{levels}-{water}-{this_seed}"
		out = USER_DIR / (name + ".vmap")
		out.parent.mkdir(parents=True, exist_ok=True)
		env = {**os.environ, "VCMI_USER_DIR": str(battle_play._map_builder_dir())}
		command = [str(ROOT / "build" / "bin" / "alpha_make_random_map"), str(out), str(this_seed), str(size), str(players), str(levels), water]
		result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
		if result.returncode != 0 or not out.exists():
			raise RuntimeError(f"generating the map failed:\n{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
		template = result.stdout.rsplit("template ", 1)[-1].strip().rstrip(")") if "template " in result.stdout else "?"
		if not players_without_a_town(out):
			return name, template
		out.unlink(missing_ok=True)
	raise RuntimeError(f"{attempts} maps from seed {seed} ({players} players, {size}x{size}) each left a player without a town")


def ledgers(messages):
	"""The ledger snapshots received so far; one that cannot be read (cut off as its game was killed) is left out."""
	snapshots = []
	for key, payloads in list(messages.items()):
		if key.endswith(":ledger"):
			for payload in list(payloads):
				try:
					snapshots.append(json.loads(payload))
				except (json.JSONDecodeError, UnicodeDecodeError):
					pass
	return snapshots


def finished(messages, days, ours=None):
	"""Each of our players (ours: their colors; None: every player) has sent its last records: all of them (the game
	ended for them), or, once a day past the cap has begun, every one still in the game."""
	daily = [s for s in ledgers(messages) if not s.get("final")]
	if not daily:
		return False
	done = {key.split(":")[-2] for key in list(messages) if key.endswith(":done")}
	players = {COLORS[p["color"]] for p in daily[0]["players"]} if ours is None else set(ours)
	if players <= done:
		return True
	latest = max(daily, key=lambda s: s["day"])
	playing = {COLORS[p["color"]] for p in latest["players"] if p["status"] == STATUS_INGAME} & players
	return latest["day"] > days and playing <= done


def latest_day(messages):
	"""The day of the newest daily ledger."""
	return max((s["day"] for s in ledgers(messages) if not s.get("final")), default=0)


def run_dir_of(pid):
	runs = sorted((ROOT / "runtime" / "runs").glob(f"*-{pid}-*"))
	return runs[-1] if runs else None


def directory_mb(path):
	return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 2**20 if path else 0.0


class GameRun:
	"""One game started in the background: what play() does, in steps, so a caller can do other things meanwhile (the
	data workers of battle_model/supply.py hand the game's battles on as they end). poll() says whether it is over (and
	ends its process then); result() gives (records.Game, GameInfo) once it is. The arguments are play()'s."""

	def __init__(self, socket_path, map_name, days, seed, max_seconds=170, max_mb=300, battle_ai="BattleAI", game_key="g0", settings=(), niceness=0, temperature=1.0, ours=None, our_battle_ai="AlphaBattleAI", neutral_ai="BattleAI", difficulty=None, teacher=False):
		self.server = battle_play.RecordServer()
		self.map_name, self.days, self.ours = map_name, days, ours
		env = dict(os.environ)
		ai = {"adventureEnemyAI": "AlphaMapAI", "adventureAlliedAI": "AlphaMapAI", "combatEnemyAI": battle_ai, "combatAlliedAI": battle_ai, "combatNeutralAI": neutral_ai}
		if ours is not None:
			ai.update(adventureEnemyAI="Nullkiller2", adventureAlliedAI="Nullkiller2")
		if teacher:  # every player Nullkiller2, its turn requests recorded as our decisions (agents/AlphaMapAI/Teacher.cpp)
			ai.update(adventureEnemyAI="AlphaTeacherAI", adventureAlliedAI="AlphaTeacherAI")
		env.update(
			ALPHA_MAP_INFERENCE_SOCKET=str(socket_path),
			ALPHA_MAP_RECORD_SOCKET=str(self.server.path),
			ALPHA_MAP_RECORD_FORMAT="2",  # each decision compressed apart, kept as it comes (records.py)
			ALPHA_MAP_GAME_KEY=game_key,
			ALPHA_MAP_REQUIRE_MODEL="1",  # (a game the model cannot play ends: AlphaMapAI.h, never played on with every turn ended unasked)
			ALPHA_MAP_DAYS=str(days),
			ALPHA_MAP_TEMPERATURE=str(temperature),
			SETTINGS=json.dumps({"server": {"seed": seed}, "ai": ai, **({"general": {"lastDifficulty": int(difficulty)}} if difficulty is not None else {})}),
		)
		for i, color in enumerate(COLORS):
			env[f"ALPHA_MAP_SEED_{color}"] = str(seed * 101 + i)
			env.pop(f"VCMI_BATTLE_AI_{color}", None)
			env.pop(f"VCMI_ADVENTURE_AI_{color}", None)
			if ours is not None and color in ours:
				env[f"VCMI_ADVENTURE_AI_{color}"] = "AlphaMapAI"  # (a vcmi patch: one game seats different AIs)
				env[f"VCMI_BATTLE_AI_{color}"] = our_battle_ai
		env.update(dict(settings))
		command = [str(ROOT / "scripts" / "headless.sh"), map_name, str(max_seconds), str(max_mb), "60"]
		if niceness:
			command = ["nice", "-n", str(niceness)] + command  # (nice execs the script: the same process id)
		self.started = time.time()
		self.process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
		self.status = "the game ended"

	def poll(self, stop=None):
		"""Whether the game is over: its process ended, every one of our players has sent its last records (the process
		is then ended), or stop (an Event) is set (training is over)."""
		if self.process.poll() is not None:
			return True
		if finished(self.server.records, self.days, self.ours):
			self.status = f"stopped at day {latest_day(self.server.records)}"
		elif stop is not None and stop.is_set():
			self.status = "stopped: training is over"
		else:
			return False
		self.process.terminate()  # headless.sh kills the client on its way out
		return True

	def terminate(self):
		"""Ends the game now (a worker stopping): its result still reads what came in."""
		if self.process.poll() is None:
			self.status = "stopped: the worker is stopping"
			self.process.terminate()

	def result(self):
		"""(records.Game, GameInfo) of the game poll() said was over; closes its record server."""
		output = self.process.communicate(timeout=60)[0]
		seconds = time.time() - self.started
		if self.status == "the game ended":
			self.status = next((line.split(":", 1)[1].strip() for line in output.splitlines() if line.startswith("status:")), "the game ended")
		time.sleep(0.3)  # what the last connections still carry
		messages = self.server.drain()
		self.server.__exit__(None, None, None)
		run = run_dir_of(self.process.pid)
		game = collect(messages)
		return game, GameInfo(self.map_name, "", seconds, self.status, str(run), directory_mb(run))


def play(socket_path, map_name, days, seed, max_seconds=170, max_mb=300, battle_ai="BattleAI", game_key="g0", settings=(), niceness=0, temperature=1.0, ours=None, our_battle_ai="AlphaBattleAI", stop=None, neutral_ai="BattleAI", difficulty=None, teacher=False):
	"""Play one game against the inference server at socket_path; returns (records.Game, GameInfo).

	battle_ai plays the players' battles, neutral_ai the wandering monsters' (the game's AI, as in a real game; the
	monsters cannot flee, so AlphaBattleAI's stalemate retreat never ends a battle where ours keep away: joint2,
	2026-10-29, battles of 1000+ decisions, mostly DEFEND and MOVE, ~0.4 MB of record each, killed 30-50% of the
	campaign games through the stall watchdog); settings: more environment (e.g. the battle AI's
	inference server and record keys). ours: None (every player is our AI), or the colors that are, the others
	Nullkiller2 with battle_ai while ours fight with our_battle_ai (an evaluation). stop: an Event that ends the game
	early (training is over). difficulty: the game's (VCMI's 0 easy .. 4 impossible; None: the settings' default, 1), the
	AI players' resource bonus grows with it. teacher: every player is Nullkiller2, recorded (campaign_model/teacher.py)."""
	run = GameRun(socket_path, map_name, days, seed, max_seconds, max_mb, battle_ai, game_key, settings, niceness, temperature, ours, our_battle_ai, neutral_ai, difficulty, teacher)
	while not run.poll(stop):
		time.sleep(0.25)
	return run.result()


def main():
	from .train import serve_model

	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--days", type=int, default=5)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--players", type=int, default=0, help="0: 2 to 4 at random")
	parser.add_argument("--size", type=int, default=36)
	parser.add_argument("--checkpoint", default="")
	args = parser.parse_args()
	players = args.players or random.Random(args.seed).randint(2, 4)
	map_name, template = make_map(args.seed, players, args.size)
	with serve_model(args.checkpoint) as (model, server):
		game, info = play(server.socket_path, map_name, args.days, args.seed)
	info.template = template
	print(info)
	print(f"{len(game.decisions)} decisions, ledgers for days {sorted(game.ledgers)}, done {game.done}")


if __name__ == "__main__":
	main()
