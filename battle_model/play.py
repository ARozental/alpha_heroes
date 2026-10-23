"""Play recorded battles in VCMI: our model on one side, another battle AI on the other, or our model on
both sides (opponent SELF_PLAY: each side has its own AlphaBattleAI with its own record and sampling seed).

A scenario (battle_model/scenarios/*.json) becomes a VCMI "battle only" map: the game starts with the
battle between its two heroes and ends right after it. sides[0] attacks from the left (red), sides[1]
defends on the right (blue). AlphaBattleAI writes the record (every decision plus events.jsonl) that
data.py turns into training examples.

Each game runs through scripts/headless.sh, so it has a time limit, a disk limit and stall detection.
"""
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USER_DIR = ROOT / "runtime" / "vcmi-user"
COLORS = ("RED", "BLUE")  # battle side 0 (attacker) and 1 (defender)
SELF_PLAY = "self"  # Game.opponent for our model against itself


@dataclass(frozen=True)
class Game:
	scenario: str  # path to the scenario json
	our_side: int  # 0 = we play sides[0] (attacker), 1 = sides[1] (defender)
	opponent: str = "BattleAI"  # the other side's battle AI, or SELF_PLAY
	game_seed: int = 1  # VCMI's random seed: dice, obstacles, the opponent's choices
	sampler_seed: int = 0  # our model's sampling
	temperature: float = 1.0
	ai: str = "AlphaBattleAI"  # our side's battle AI library; AlphaBattleAIDev to try C++ changes during training runs
	# Map to play, as returned by build_map; empty = the scenario's shared map (map_name). Training runs use
	# their own copy, so rebuilding a map (e.g. after a map tool fix) never changes a running experiment.
	map: str = ""
	# Actions (dicts as in a record's action.json) our side plays before the model chooses: battles are
	# deterministic given game_seed and our actions, so this restores a state of a recorded battle
	prefix: tuple = ()

	@property
	def recorded_sides(self):
		"""The sides our model plays, each with its own record."""
		return (0, 1) if self.opponent == SELF_PLAY else (self.our_side,)


def record_dirs(game, record_dir):
	"""The record folder of each side in game.recorded_sides."""
	record_dir = Path(record_dir).resolve()
	return [record_dir / f"side_{side}" for side in game.recorded_sides] if game.opponent == SELF_PLAY else [record_dir]


def load_scenario(path):
	return json.loads(Path(path).read_text())


def map_name(scenario_path, variant=""):
	return "Maps/alpha/" + Path(scenario_path).stem + (f"-{variant}" if variant else "")


def build_map(scenario_path, variant=""):
	"""Write the scenario's battle-only map into the VCMI user data folder (overwrites); returns its map name.

	variant: a suffix for a private copy (e.g. the training run's name).
	"""
	name = map_name(scenario_path, variant)
	out = USER_DIR / (name + ".vmap")
	out.parent.mkdir(parents=True, exist_ok=True)
	env = {**os.environ, "VCMI_USER_DIR": str(USER_DIR)}
	result = subprocess.run([str(ROOT / "build" / "bin" / "alpha_make_battle_map"), str(scenario_path), str(out)], env=env, capture_output=True, text=True)
	if result.returncode != 0 or not out.exists():
		raise RuntimeError(f"building the map failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
	return name


def start(game, model_path, record_dir, max_seconds=120, max_mb=200):
	"""Start one battle in the background; returns the process. record_dir must not exist yet.

	Each side our model plays gets its own settings (AlphaBattleAI reads ALPHA_BATTLE_<NAME>_<COLOR> before
	ALPHA_BATTLE_<NAME>): its record folder (record_dirs) and its sampling seed.
	"""
	env = dict(os.environ)
	env.update(
		ALPHA_BATTLE_MODEL=str(Path(model_path).resolve()),
		ALPHA_BATTLE_TEMPERATURE=str(game.temperature),
		SETTINGS=json.dumps({"server": {"seed": game.game_seed}}),
	)
	for side, color in enumerate(COLORS):
		env[f"VCMI_BATTLE_AI_{color}"] = game.ai if side in game.recorded_sides else game.opponent
	for i, (side, folder) in enumerate(zip(game.recorded_sides, record_dirs(game, record_dir))):
		env[f"ALPHA_BATTLE_RECORD_DIR_{COLORS[side]}"] = str(folder)
		env[f"ALPHA_BATTLE_SEED_{COLORS[side]}"] = str(game.sampler_seed + 1_000_003 * i)  # the two sides sample independently
	if game.prefix:
		prefix = Path(str(Path(record_dir).resolve()) + ".prefix.jsonl")
		prefix.parent.mkdir(parents=True, exist_ok=True)
		prefix.write_text("".join(json.dumps(action) + "\n" for action in game.prefix))
		env[f"ALPHA_BATTLE_PREFIX_{COLORS[game.our_side]}"] = str(prefix)
	command = [str(ROOT / "scripts" / "headless.sh"), game.map or map_name(game.scenario), str(max_seconds), str(max_mb), "20"]
	return subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def finish(process, keep_run=False):
	"""Wait for a battle started with start(); returns the runner's report (run dir, status, size).

	The runner's folder (VCMI logs and settings, ~1 MB) is deleted after a clean finish unless keep_run;
	it is kept when the battle failed, for debugging.
	"""
	output, _ = process.communicate()
	report = dict(line.split(":", 1) for line in output.splitlines() if ":" in line)
	report = {key.strip(): value.strip() for key, value in report.items()}
	status = report.get("status", "")
	if not status.startswith("finished") or not status.endswith("(exit code 0)"):  # killed by a limit, or crashed
		raise RuntimeError(f"battle did not finish cleanly ({status}; run {report.get('run')}):\n{output[-2000:]}")
	if not keep_run and report.get("run"):
		shutil.rmtree(report["run"], ignore_errors=True)
	return report


def play(games, model_path, record_root):
	"""Play several battles in parallel. Returns every record dir: per game in the order of `games`, one per
	side in game.recorded_sides (each record says which side it played)."""
	record_root = Path(record_root).resolve()  # VCMI runs in build/bin: paths must be absolute
	model_path = Path(model_path).resolve()
	game_dirs = [record_root / f"game_{i}" for i in range(len(games))]
	processes = [start(game, model_path, d) for game, d in zip(games, game_dirs)]
	reports, failures = [], []
	for process in processes:  # wait for every battle before raising, so none is left writing its record
		try:
			reports.append(finish(process, keep_run=True))
		except RuntimeError as error:
			reports.append({})
			failures.append(str(error))
	dirs = [record_dirs(game, d) for game, d in zip(games, game_dirs)]
	recorded = [all((r / "events.jsonl").exists() for r in records) for records in dirs]
	for report, ok in zip(reports, recorded):
		if report.get("run") and ok:
			shutil.rmtree(report["run"], ignore_errors=True)
	# Failed battles keep their runner folders (VCMI logs) for debugging
	if failures:
		raise RuntimeError(f"{len(failures)} of {len(processes)} battles failed; first: {failures[0]}")
	missing = [(d, report) for d, report, ok in zip(game_dirs, reports, recorded) if report and not ok]
	if missing:
		raise RuntimeError("no record written in " + ", ".join(f"{d} (run {report.get('run')})" for d, report in missing))
	return [r for records in dirs for r in records]
