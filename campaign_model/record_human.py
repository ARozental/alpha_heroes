"""A human player's game at the VCMI GUI, recorded (the client's ALPHA_RECORD_HUMAN: agents/AlphaMapAI/Recorder.cpp,
the AlphaRecorderAI library beside your player interface): each of your adventure-map requests as our turn decision
doing the same (the inputs as they were then; Decision.flags HUMAN), your answers to yes/no, selection and level-up
questions, each request raw (raw.jsonl: a move made of several requests, such as buying every creature from the
dearest down, is there as the requests it was), and every battle of yours as the battle model's records (AlphaBattleAI
observing: each state you were asked to act in, with the action you took).

	.venv/bin/python -m campaign_model.record_human --out runtime/human/first
	.venv/bin/python -m campaign_model.record_human --player-only
	.venv/bin/python -m campaign_model.record_human --opponents alpha --checkpoint <campaign checkpoint> --battle-model <onnx>

The GUI opens as usual: choose the map (or a random one) and your colour in its lobby, and play; close the client when
you are done (a game left mid-way keeps what was recorded). Save and load as you like: each start of a game in the
client (a new one, or a save loaded) is a segment of its own, its decisions, battles and daily ledgers apart, so a save
reloaded to play a day again leaves the first try whole beside it; a save continued in a later run is a segment of that
run (the segments' days say which continue which: summary.json). The computer players are --opponents' (nk2: Nullkiller2,
its battles BattleAI's; alpha: our campaign model at --checkpoint, its battles AlphaBattleAI's with --battle-model, else
BattleAI's). Their decisions are recorded too unless --player-only (Nullkiller2's as the teacher's, AlphaTeacherAI's;
ours as our own), each labelled with who made it; of the battles, only yours. A battle you leave to auto-combat is not
yours: nothing of it is recorded from then on. Each run is saved in a folder of its own, <--out>/<its start time>
(--out default runtime/human: nothing a run saves is ever written over), when the client closes:
  segment-<n>.pkl  records.Game of the n-th start of a game: every recorded decision (Decision.flags: HUMAN yours,
                   TEACHER Nullkiller2's, neither ours), that game's daily ledgers (the labels' empire values)
  battles/<n>/     each battle of yours in it, as AlphaBattleAI sent it (battle_model/data.py parse_record)
  players.json     each segment's players as the records show them (human, nk2, alpha) and their decisions
  raw.jsonl        each of your requests as sent, and what of ours it was (or why none); saves too
  summary.json     each segment's days, decisions, battles, and your requests matched, approximate, and those with no
                   equal among our actions
The run's VCMI user directory (vcmi-user: the shared game data and saves linked in, your GUI settings with the
computer players' AIs set, every hero's map moves instant) and its logs are there too."""
import argparse
import contextlib
import json
import os
import pickle
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from battle_model import play as battle_play

from .records import HUMAN, TEACHER, collect, decision_name

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "runtime" / "vcmi-user"  # (scripts/vcmi.sh's: the game data, maps, saves and your GUI settings)
BATTLE_KEY = "-human-"  # (Recorder.cpp: AlphaBattleAI's record key for your battles, <segment key>-human-<COLOUR>)
SEGMENT = re.compile(r"@(\d+)")  # (AlphaMapAI::sessionKey: every recorder's key ends in @<its game's start, unix seconds>)
SEGMENT_SECONDS = 15  # (the recorders of one start of a game, made one after another: seconds apart at most)


def user_dir(out, ai):
	"""The session's VCMI user directory: the shared game data linked in (Data, Maps, Mp3, Mods, Saves: your saves are
	the usual ones), your GUI settings with the computer players' AIs (ai: settings.json's "ai" entries) set."""
	path = out / "vcmi-user"
	(path / "config").mkdir(parents=True, exist_ok=True)
	for name in ("Data", "Maps", "Mp3", "Mods", "Saves"):
		if (SHARED / name).exists() and not (path / name).exists():
			(path / name).symlink_to(SHARED / name)
	if (SHARED / "config" / "modSettings.json").exists():
		shutil.copy(SHARED / "config" / "modSettings.json", path / "config" / "modSettings.json")
	shared = SHARED / "config" / "settings.json"
	settings = json.loads(shared.read_text()) if shared.exists() else {}
	settings["ai"] = {**settings.get("ai", {}), **ai}
	# Heroes on the map move at once, yours and the computer players' (VCMI's heroMoveTime, enemyMoveTime: a tile in 150 ms
	# by default; 0 instant, the moves still shown; the user, 2026-11-07: no time spent watching heroes walk)
	settings["adventure"] = {**settings.get("adventure", {}), "heroMoveTime": 0, "enemyMoveTime": 0}
	(path / "config" / "settings.json").write_text(json.dumps(settings, indent="\t"))
	return path


def actor(d):
	"""Who made a recorded decision: human (the recorder's), nk2 (the teacher's), alpha (our model's own)."""
	return "human" if d.flags & HUMAN else "nk2" if d.flags & TEACHER else "alpha"


def segments(messages):
	"""A run's messages split by the start of a game they belong to (a new game, a save loaded): each recorder's key
	ends in @<its start, unix seconds>, those of one start seconds apart. [(start seconds, {key: payloads})] in order;
	keys without a start (none expected) a segment of their own, last."""
	starts = sorted({int(m.group(1)) for key in messages if (m := SEGMENT.search(key))})
	groups = []
	for t in starts:
		if groups and t - groups[-1][-1] <= SEGMENT_SECONDS:
			groups[-1].append(t)
		else:
			groups.append([t])
	index = {t: i for i, group in enumerate(groups) for t in group}
	out = [(group[0], {}) for group in groups]
	rest = {}
	for key, payloads in messages.items():
		m = SEGMENT.search(key)
		(out[index[int(m.group(1))]][1] if m else rest)[key] = payloads
	return out + ([(0, rest)] if rest else [])


def save(out, messages):
	"""What the record server received, saved in out (the module's list), a segment a start of a game; returns the
	summary."""
	summary, players = {"segments": []}, {}
	for n, (start, part) in enumerate(segments(messages), start=1):
		battles = {key: payloads for key, payloads in part.items() if BATTLE_KEY in key}
		game = collect({key: payloads for key, payloads in part.items() if key not in battles})
		folder = out / "battles" / str(n)
		folder.mkdir(parents=True, exist_ok=True)
		for key, payloads in sorted(battles.items()):
			for i, payload in enumerate(payloads):
				(folder / f"{key.split('@')[0]}-{key.split(BATTLE_KEY)[-1]}-{i:03d}.rec").write_bytes(payload)
		with open(out / f"segment-{n}.pkl", "wb") as f:
			pickle.dump(game, f)
		seen = {}
		for d in game.decisions:
			counts = seen.setdefault(d.color, {"actor": actor(d), "decisions": 0, "by_type": Counter()})
			counts["decisions"] += 1
			counts["by_type"][decision_name(d)] += 1
		players[n] = {color: {**p, "by_type": dict(p["by_type"].most_common())} for color, p in seen.items()}
		days = [d.day for d in game.decisions]
		summary["segments"].append({
			"segment": n, "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start)) if start else None,
			"days": [min(days), max(days)] if days else None, "decisions": len(game.decisions), "players": {color: p["actor"] for color, p in seen.items()},
			"battles": sum(len(v) for v in battles.values()), "ledger_days": sorted(game.ledgers), "damaged_messages": game.damaged,
			# (each human's own requests: matched, approximate, and unmatched by kind)
			"human_requests": {color: game.done.get(color, {}).get("teacher", {}) for color, p in seen.items() if p["actor"] == "human"},
		})
	(out / "players.json").write_text(json.dumps(players, indent="\t"))
	(out / "summary.json").write_text(json.dumps(summary, indent="\t"))
	return summary


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--out", type=Path, default=ROOT / "runtime" / "human", help="the folder each run is saved in, in a folder of its own (<--out>/<start time>)")
	parser.add_argument("--opponents", choices=("nk2", "alpha"), default="nk2", help="the computer players: Nullkiller2, or our campaign model")
	parser.add_argument("--player-only", action="store_true", help="record only the human player's decisions")
	parser.add_argument("--checkpoint", default="", help="--opponents alpha: the campaign model's checkpoint")
	parser.add_argument("--battle-model", default="", help="an ONNX battle model: --opponents alpha's battles, and its outputs beside your battle decisions")
	args = parser.parse_args()
	out = (args.out / time.strftime("%Y%m%d-%H%M%S")).resolve()
	out.mkdir(parents=True, exist_ok=False)  # (a run never writes into another's)
	if args.opponents == "alpha" and not args.checkpoint:
		parser.error("--opponents alpha needs --checkpoint (the campaign model they play)")
	key = "human-" + out.name
	# The battle AIs: your auto-combat's (combatAlliedAI) is never AlphaBattleAI, which observes your colour's battles
	ai = {"combatAlliedAI": "BattleAI", "combatNeutralAI": "BattleAI", "combatEnemyAI": "BattleAI"}
	env = dict(os.environ)
	for name in ("ALPHA_MAP_RECORD_SOCKET", "ALPHA_BATTLE_RECORD_SOCKET", "ALPHA_BATTLE_RECORD_DIR", "ALPHA_MAP_DAYS", "ALPHA_BATTLE_DELEGATE", "ALPHA_BATTLE_OBSERVE"):
		env.pop(name, None)
	env.update(ALPHA_RECORD_HUMAN="1", ALPHA_RECORD_RAW=str(out / "raw.jsonl"), ALPHA_MAP_GAME_KEY=key, ALPHA_MAP_RECORD_FORMAT="2")
	if args.battle_model:
		env["ALPHA_BATTLE_MODEL"] = str(Path(args.battle_model).resolve())
	else:
		env.pop("ALPHA_BATTLE_MODEL", None)  # (your battles' records with zero outputs)
	with battle_play.RecordServer(concurrent=True) as records, contextlib.ExitStack() as stack:
		env["ALPHA_RECORD_HUMAN_SOCKET"] = str(records.path)
		if not args.player_only:
			env["ALPHA_MAP_RECORD_SOCKET"] = str(records.path)  # (the computer players' records: AlphaTeacherAI's, AlphaMapAI's)
		if args.opponents == "nk2":
			name = "Nullkiller2" if args.player_only else "AlphaTeacherAI"
			ai.update(adventureEnemyAI=name, adventureAlliedAI=name)
		else:
			from .train import serve_model

			_, inference = stack.enter_context(serve_model(args.checkpoint))
			env.update(ALPHA_MAP_INFERENCE_SOCKET=str(inference.socket_path), ALPHA_MAP_TEMPERATURE="1.0")
			ai.update(adventureEnemyAI="AlphaMapAI", adventureAlliedAI="AlphaMapAI")
			if args.battle_model:
				ai["combatEnemyAI"] = "AlphaBattleAI"
		env["VCMI_USER_DIR"] = str(user_dir(out, ai))
		print(f"recording into {out}: computer players {ai['adventureEnemyAI']}{' (not recorded)' if args.player_only else ''}; close the client when done", flush=True)
		started = time.time()
		subprocess.run([str(ROOT / "build" / "bin" / "vcmiclient")], cwd=ROOT / "build" / "bin", env=env)
		time.sleep(0.5)  # (what the last connections still carry)
		messages = records.drain()
	summary = save(out, messages)
	summary["minutes"] = round((time.time() - started) / 60, 1)
	(out / "summary.json").write_text(json.dumps(summary, indent="\t"))
	print(json.dumps(summary, indent="\t"))


if __name__ == "__main__":
	main()
