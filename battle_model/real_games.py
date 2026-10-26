"""Real games as a source of battles: a whole H3 map played by the campaign AI (Nullkiller2) for every player, every
battle of it, the wandering monsters' too, played by our model (AlphaBattleAI as all of VCMI's battle AIs). Each
battle's record is sent to a RecordServer under the key "<game>:<COLOR>" (AlphaBattleAI numbers them), so many
battles of many games arrive interleaved while the games run.

Usage (a test run): .venv/bin/python -m battle_model.real_games --model battle_model/artifacts/battle_model.onnx
       (an eval set):  ... --collect 40 --out battle_model/scenarios/real_eval.json
"""
import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

from . import play

PLAYER_COLORS = ("RED", "BLUE", "TAN", "GREEN", "ORANGE", "PURPLE", "TEAL", "PINK", play.NEUTRAL)
MAX_GAME_SECONDS = 900  # a game is cut (headless.sh kills it) after this; its finished battles are already sent
REAL_GAME_NICENESS = 15


def maps():
	"""The game's own maps (runtime/vcmi-user/Maps/*.h3m), as headless.sh takes them."""
	return sorted("Maps/" + p.stem for p in (play.USER_DIR / "Maps").glob("*.h3m"))


def start_game(map_name, model_path, socket_path, game_key, seed, temperature=1.0, max_seconds=MAX_GAME_SECONDS):
	"""Start one real game in the background; returns the process. Every player's and the monsters' battles are
	played by our model (sampling at temperature), each side with its own sampling seed, and recorded."""
	env = dict(os.environ)
	ai = "AlphaBattleAI"
	env.update(
		ALPHA_BATTLE_MODEL=str(Path(model_path).absolute()),  # not resolved: a link (actors.py) is followed at each battle
		ALPHA_BATTLE_TEMPERATURE=str(temperature),
		ALPHA_BATTLE_RECORD_SOCKET=str(socket_path),
		SETTINGS=json.dumps({"server": {"seed": seed}, "ai": {"combatEnemyAI": ai, "combatAlliedAI": ai, "combatNeutralAI": ai}}),
	)
	for i, color in enumerate(PLAYER_COLORS):
		env[f"VCMI_BATTLE_AI_{color}"] = ai
		env[f"ALPHA_BATTLE_RECORD_KEY_{color}"] = f"{game_key}:{color}"
		env[f"ALPHA_BATTLE_SEED_{color}"] = str(seed * 101 + i)
	# Low priority: Nullkiller2 plans its turns on every core (TBB), and later in a game that starved the battles
	# (generated and eval ones stalled and were killed); battles come first, the real games take what is left. With
	# the nice command, not preexec_fn: that is unsafe in a process with threads (the actors) and hung the trainer.
	command = ["nice", "-n", str(REAL_GAME_NICENESS), str(play.ROOT / "scripts" / "headless.sh"), map_name, str(max_seconds), "300", "300"]
	return subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def scenario_of(record):
	"""The battle as a scenario (AlphaBattleAI's "scenario" event), or None."""
	return next((e["scenario"] for e in record.events if e.get("event") == "scenario"), None)


def kind_of(scenario):
	return "neutrals" if "neutrals" in scenario else "siege" if "town" in scenario else "heroes"


def collect(model, count, seconds, seed, min_weight=0.3):
	"""count scenarios of real battles that mattered (data.example_weight at least min_weight), from short games on
	different maps, a third of each kind (heroes, sieges, neutrals) as far as there are, distinct armies."""
	from .data import example_weight, parse_record, summarize

	rng, pools, seen = random.Random(seed), {"heroes": [], "siege": [], "neutrals": []}, set()
	all_maps = maps()
	while min(len(p) for p in pools.values()) < count // 3 + 1 and sum(len(p) for p in pools.values()) < 20 * count:
		with play.RecordServer() as server:
			start_game(rng.choice(all_maps), model, server.path, "c", rng.randrange(2**31), max_seconds=seconds).communicate()
		for records in server.records.values():
			for data in records:
				record = parse_record(data)
				scenario = scenario_of(record)
				key = json.dumps(scenario.get("sides") if scenario else None, sort_keys=True)
				if scenario and key not in seen and example_weight(summarize(record.events)) >= min_weight:
					seen.add(key)
					pools[kind_of(scenario)].append(scenario)
		print({k: len(v) for k, v in pools.items()}, flush=True)
	chosen = []
	for kind, pool in pools.items():
		chosen += rng.sample(pool, min(len(pool), count // 3))
	rest = [s for p in pools.values() for s in p if s not in chosen]
	chosen += rng.sample(rest, min(len(rest), count - len(chosen)))
	for i, scenario in enumerate(chosen):
		scenario["seed"] = rng.randrange(1, 2**31)
		scenario["_about"] = f"real {kind_of(scenario)} {i}"
	return chosen


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--model", required=True)
	parser.add_argument("--map", default="")
	parser.add_argument("--seconds", type=int, default=180)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--collect", type=int, default=0, help="collect this many real battles as an eval set (--out)")
	parser.add_argument("--out", default="")
	args = parser.parse_args()
	from .data import parse_record, summarize

	if args.collect:
		scenarios = collect(args.model, args.collect, args.seconds, args.seed)
		Path(args.out).write_text(json.dumps(scenarios, indent="\t"))
		print(f"wrote {len(scenarios)} scenarios to {args.out}")
		return

	map_name = args.map or random.Random(args.seed).choice(maps())
	started = time.time()
	with play.RecordServer() as server:
		process = start_game(map_name, args.model, server.path, "g0", args.seed, max_seconds=args.seconds)
		output, _ = process.communicate()
	print(map_name, f"{time.time() - started:.0f}s")
	print("\n".join(line for line in output.splitlines() if ":" in line and line.split(":")[0].strip() in ("run", "status", "elapsed", "data")))
	battles = [(key, parse_record(r)) for key, records in server.records.items() for r in records]
	print(f"{len(battles)} battle records")
	for key, r in battles[:40]:
		s = summarize(r.events)
		scenario = next((e["scenario"] for e in r.events if e["event"] == "scenario"), {})
		print(f"{key:12s} {s.outcome.name:9s} decisions {r.decisions:3d} reward {s.reward:+.3f} {'neutrals ' + scenario['neutrals']['creature'] if 'neutrals' in scenario else 'town' if 'town' in scenario else 'heroes'}")


if __name__ == "__main__":
	main()
