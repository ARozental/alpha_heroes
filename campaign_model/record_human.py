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
you are done (a game left mid-way keeps what was recorded). The computer players are --opponents' (nk2: Nullkiller2,
its battles BattleAI's; alpha: our campaign model at --checkpoint, its battles AlphaBattleAI's with --battle-model, else
BattleAI's). Their decisions are recorded too unless --player-only (Nullkiller2's as the teacher's, AlphaTeacherAI's;
ours as our own), each labelled with who made it; of the battles, only yours. A battle you leave to auto-combat is not
yours: nothing of it is recorded from then on. Saved in --out when the client closes:
  game.pkl      records.Game: every recorded decision (Decision.flags: HUMAN yours, TEACHER Nullkiller2's, neither ours)
  players.json  each colour's player as the records show it (human, nk2, alpha) and its decisions
  battles/      each battle of yours, as AlphaBattleAI sent it (battle_model/data.py parse_record): <key>-<n>.rec
  raw.jsonl     each of your requests as sent, and what of ours it was (or why none)
  summary.json  what was recorded: your requests matched, approximate, and those with no equal among our actions
The session's VCMI user directory (vcmi-user: the shared game data and saves linked in, your GUI settings with the
computer players' AIs set) and its logs are there too."""
import argparse
import contextlib
import json
import os
import pickle
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from battle_model import play as battle_play

from .records import HUMAN, TEACHER, collect, decision_name

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "runtime" / "vcmi-user"  # (scripts/vcmi.sh's: the game data, maps, saves and your GUI settings)
BATTLE_KEY = "-human-"  # (Recorder.cpp: AlphaBattleAI's record key for your battles, <game key>-human-<COLOUR>)


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
	(path / "config" / "settings.json").write_text(json.dumps(settings, indent="\t"))
	return path


def actor(d):
	"""Who made a recorded decision: human (the recorder's), nk2 (the teacher's), alpha (our model's own)."""
	return "human" if d.flags & HUMAN else "nk2" if d.flags & TEACHER else "alpha"


def save(out, messages):
	"""What the record server received, saved in out (the module's list); returns the summary."""
	battles = {key: payloads for key, payloads in messages.items() if BATTLE_KEY in key}
	game = collect({key: payloads for key, payloads in messages.items() if key not in battles})
	(out / "battles").mkdir(exist_ok=True)
	saved = 0
	for key, payloads in sorted(battles.items()):
		for n, payload in enumerate(payloads):
			(out / "battles" / f"{key}-{n:03d}.rec").write_bytes(payload)
			saved += 1
	with open(out / "game.pkl", "wb") as f:
		pickle.dump(game, f)
	players = {}
	for d in game.decisions:
		counts = players.setdefault(d.color, {"actor": actor(d), "decisions": 0, "by_type": Counter()})
		counts["decisions"] += 1
		counts["by_type"][decision_name(d)] += 1
	(out / "players.json").write_text(json.dumps({color: {**p, "by_type": dict(p["by_type"].most_common())} for color, p in players.items()}, indent="\t"))
	humans = [color for color, p in players.items() if p["actor"] == "human"]
	audit = {color: game.done.get(color, {}).get("teacher", {}) for color in humans}
	summary = {
		"decisions": len(game.decisions), "players": {color: p["actor"] for color, p in players.items()}, "battles": saved,
		"battle_decisions": {key: len(payloads) for key, payloads in battles.items()}, "ledger_days": sorted(game.ledgers), "damaged_messages": game.damaged,
		"human_requests": audit,  # (each human's: matched, approximate, and unmatched by kind)
	}
	(out / "summary.json").write_text(json.dumps(summary, indent="\t"))
	return summary


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--out", type=Path, default=None, help="where the session is saved (default runtime/human/<now>)")
	parser.add_argument("--opponents", choices=("nk2", "alpha"), default="nk2", help="the computer players: Nullkiller2, or our campaign model")
	parser.add_argument("--player-only", action="store_true", help="record only the human player's decisions")
	parser.add_argument("--checkpoint", default="", help="--opponents alpha: the campaign model's checkpoint")
	parser.add_argument("--battle-model", default="", help="an ONNX battle model: --opponents alpha's battles, and its outputs beside your battle decisions")
	args = parser.parse_args()
	out = (args.out or ROOT / "runtime" / "human" / time.strftime("%Y%m%d-%H%M%S")).resolve()
	out.mkdir(parents=True, exist_ok=True)
	if args.opponents == "alpha" and not args.checkpoint:
		parser.error("--opponents alpha needs --checkpoint (the campaign model they play)")
	key = "human-" + out.name.replace(":", "_")
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
