"""Play recorded battles in VCMI: our model on one side, another battle AI on the other.

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


@dataclass(frozen=True)
class Game:
	scenario: str  # path to the scenario json
	our_side: int  # 0 = we play sides[0] (attacker), 1 = sides[1] (defender)
	opponent: str = "BattleAI"
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
	"""Start one battle in the background; returns the process. record_dir must not exist yet."""
	env = dict(os.environ)
	for side, color in enumerate(COLORS):
		env[f"VCMI_BATTLE_AI_{color}"] = game.ai if side == game.our_side else game.opponent
	env.update(
		ALPHA_BATTLE_MODEL=str(model_path),
		ALPHA_BATTLE_RECORD_DIR=str(record_dir),
		ALPHA_BATTLE_SEED=str(game.sampler_seed),
		ALPHA_BATTLE_TEMPERATURE=str(game.temperature),
		SETTINGS=json.dumps({"server": {"seed": game.game_seed}}),
	)
	if game.prefix:
		prefix = Path(str(Path(record_dir).resolve()) + ".prefix.jsonl")
		prefix.parent.mkdir(parents=True, exist_ok=True)
		prefix.write_text("".join(json.dumps(action) + "\n" for action in game.prefix))
		env["ALPHA_BATTLE_PREFIX"] = str(prefix)
	command = [str(ROOT / "scripts" / "headless.sh"), game.map or map_name(game.scenario), str(max_seconds), str(max_mb), "20"]
	env["ALPHA_BATTLE_MODEL"] = str(Path(model_path).resolve())
	env["ALPHA_BATTLE_RECORD_DIR"] = str(Path(record_dir).resolve())
	return subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def finish(process, keep_run=False):
	"""Wait for a battle started with start(); returns the runner's report (run dir, status, size).

	The runner's folder (VCMI logs and settings, ~1 MB) is deleted after a clean finish unless keep_run;
	it is kept when the battle failed, for debugging.
	"""
	output, _ = process.communicate()
	report = dict(line.split(":", 1) for line in output.splitlines() if ":" in line)
	report = {key.strip(): value.strip() for key, value in report.items()}
	if not report.get("status", "").startswith("finished"):
		raise RuntimeError(f"battle did not finish cleanly:\n{output[-2000:]}")
	if not keep_run and report.get("run"):
		shutil.rmtree(report["run"], ignore_errors=True)
	return report


def play(games, model_path, record_root):
	"""Play several battles in parallel. Returns their record dirs, in the order of `games`."""
	record_root = Path(record_root).resolve()  # VCMI runs in build/bin: paths must be absolute
	model_path = Path(model_path).resolve()
	dirs = [record_root / f"game_{i}" for i in range(len(games))]
	processes = [start(game, model_path, d) for game, d in zip(games, dirs)]
	reports = [finish(process, keep_run=True) for process in processes]
	missing = [(d, report) for d, report in zip(dirs, reports) if not (d / "events.jsonl").exists()]
	for d, report in zip(dirs, reports):
		if report.get("run") and (d / "events.jsonl").exists():
			shutil.rmtree(report["run"], ignore_errors=True)
	if missing:  # their runner folders (VCMI logs) are kept for debugging
		raise RuntimeError("no record written in " + ", ".join(f"{d} (run {report.get('run')})" for d, report in missing))
	return dirs
