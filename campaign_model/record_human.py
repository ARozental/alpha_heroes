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
import math
import os
import pickle
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

from battle_model import play as battle_play

from .labels import COLORS
from .records import HUMAN, TEACHER, Game, collect, decision_name

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "runtime" / "vcmi-user"  # (scripts/vcmi.sh's: the game data, maps, saves and your GUI settings)
STATUS_WON = 2  # (EPlayerStatus::WINNER: empire.py's STATUS_INGAME 0, STATUS_LOST 1)
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


def kept_days(starts):
	"""The line played to the end, of a run's segments (starts: each one's first day, in order): a segment started at day
	s replays from s, so each earlier one keeps only the days before every later start. [(first, end)] a segment, the
	days first <= day < end kept (end inf: the last's; first >= end: none)."""
	ends, low = [], math.inf
	for start in reversed(starts):
		ends.append(low)
		low = min(low, start)
	return list(zip(starts, reversed(ends)))


def raw_segments(lines):
	"""raw.jsonl's lines (dicts) split by the start of a game they were sent in: each start's first request is the
	client's AdvInterfaceReady."""
	out = []
	for line in lines:
		if not out or line.get("request", "").endswith("AdvInterfaceReady"):
			out.append([])
		out[-1].append(line)
	return out


def battle_days(records, lines):
	"""Each battle record's day (records: parse_record's, in order): its start event's (since 2026-11-07), else the
	day of the raw log's battle its decisions are in (lines: the segment's raw lines; the battles in order, each at
	least as many actions as its record has decisions: auto-combat's are not recorded), None for a record with none."""
	runs, last = [], None
	for line in lines:
		if line.get("request") == "MakeAction":
			if last != line["battle"]:
				runs.append([line["day"], 0])
			runs[-1][1] += 1
			last = line["battle"]
		else:
			last = None
	days, i = [], 0
	for record in records:
		start = next((e for e in record.events if e.get("event") == "start"), {})
		if "day" in start:
			days.append(start["day"])
			continue
		while i < len(runs) and runs[i][1] < max(1, len(record.actions)):
			i += 1
		days.append(runs[i][0] if record.actions and i < len(runs) else None)
		i += 1 if record.actions and i < len(runs) else 0
	return days


def keep_final_line(out):
	"""A saved run cut to the line played to the end (the user, 2026-11-07: "we do not need to save the abandoned
	tries... no need to use my errors"): game.pkl the line's decisions, every player's, and its ledgers (a day's from the
	segment that played the day before: the state the day began in), the end's snapshots; battles/ the line's battles
	of yours; raw.jsonl its requests; the segments' files and the abandoned tries removed. Returns the summary."""
	from battle_model.data import parse_record

	games = [pickle.load(open(path, "rb")) for path in sorted(out.glob("segment-*.pkl"), key=lambda p: int(p.stem.split("-")[1]))]
	lines = [json.loads(line) for line in open(out / "raw.jsonl")] if (out / "raw.jsonl").exists() else []
	raws = raw_segments(lines)
	starts = [min((d.day for d in game.decisions), default=math.inf) for game in games]
	kept = kept_days(starts)
	line = Game()
	first = next((i for i, (a, b) in enumerate(kept) if a < b), None)
	battles = []
	for i, (game, (a, b)) in enumerate(zip(games, kept)):
		line.decisions += [d for d in game.decisions if a <= d.day < b]
		line.ledgers.update({day: snap for day, snap in game.ledgers.items() if a < day <= b or (i == first and day == a)})
		folder = out / "battles" / str(i + 1)
		files = sorted(folder.iterdir()) if folder.exists() else []
		records = [parse_record(f.read_bytes()) for f in files]
		for f, day in zip(files, battle_days(records, raws[i] if i < len(raws) else [])):
			if day is not None and a <= day < b:
				battles.append((day, f))
	line.final, line.done = games[-1].final if games else [], games[-1].done if games else {}
	(out / "battles.tmp").mkdir(exist_ok=True)
	for n, (day, f) in enumerate(sorted(battles, key=lambda t: t[0])):
		shutil.move(str(f), out / "battles.tmp" / f"day{day:03d}-{n:03d}.rec")
	shutil.rmtree(out / "battles", ignore_errors=True)
	(out / "battles.tmp").rename(out / "battles")
	kept_lines = [l for i, (rows, (a, b)) in enumerate(zip(raws, kept)) for l in rows if a <= l["day"] < b or (i == first and l["day"] < a)]
	(out / "raw.jsonl").write_text("".join(json.dumps(l) + "\n" for l in kept_lines))
	with open(out / "game.pkl", "wb") as f:
		pickle.dump(line, f)
	for path in out.glob("segment-*.pkl"):
		path.unlink()
	ours = [l for l in kept_lines if l.get("actor") == "human"]
	unmatched = Counter(l["note"] for l in ours if l.get("ours") is None and l.get("note") and not l["note"].startswith(("the game's", "a battle's", "no question of ours")))  # (the client's, a battle's, a dialog we ask nothing in)
	players = {}
	for d in line.decisions:
		players.setdefault(d.color, {"actor": actor(d), "decisions": 0})["decisions"] += 1
	days = [d.day for d in line.decisions]
	summary = {
		"line": [{"segment": i + 1, "days": [a, None if b == math.inf else b - 1]} for i, (a, b) in enumerate(kept) if a < b],
		"days": [min(days), max(days)] if days else None, "decisions": len(line.decisions), "players": {c: p["actor"] for c, p in players.items()},
		"battles": len(battles), "ledger_days": sorted(line.ledgers),
		"won": any(p.get("status") == STATUS_WON and players.get(COLORS[p["color"]], {}).get("actor") == "human" for f in line.final for p in f["players"]),
		"human_requests": {"matched": sum(l.get("ours") is not None for l in ours), "approximate": sum(l.get("ours") is not None and str(l.get("note") or "").startswith("approximate") for l in ours), "unmatched": dict(unmatched.most_common())},
	}
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
	parser.add_argument("--keep-tries", action="store_true", help="keep every segment (a save loaded again: the abandoned tries too), not only the line played to the end")
	parser.add_argument("--cut", type=Path, default=None, help="only cut a saved run (its folder) to the line played to the end, and exit")
	args = parser.parse_args()
	if args.cut:
		print(json.dumps(keep_final_line(args.cut.resolve()), indent="\t"))
		return
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
	if not args.keep_tries:  # (the line played to the end only: the user, 2026-11-07)
		summary = keep_final_line(out)
	summary["minutes"] = round((time.time() - started) / 60, 1)
	(out / "summary.json").write_text(json.dumps(summary, indent="\t"))
	print(json.dumps(summary, indent="\t"))


if __name__ == "__main__":
	main()
