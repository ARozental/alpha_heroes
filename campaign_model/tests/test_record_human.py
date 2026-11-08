"""record_human.py's saving of a session (2026-11-07): each colour's decisions labelled by who made them (HUMAN the
human's recorder, TEACHER Nullkiller2's, neither our model's), the human's battle records kept apart and saved as sent,
and the human's request audit (their "done" summary's "teacher" counts) in the summary."""
import json
import math
import pickle
import struct
import zlib

from campaign_model import constants as c
from campaign_model.record_human import save
from campaign_model.records import HEADER, HUMAN, IN_BYTES, TEACHER, parse_records


def records(flags, n):
	"""A records message as AlphaMapAI::sendRecords writes it (AMZ2), n turn decisions with these flags."""
	raw = zlib.compress(bytes(IN_BYTES), 1)
	out = b"AMZ2" + struct.pack("<I", n)
	for i in range(n):
		out += HEADER.pack(i + 1, c.DecisionKind.TURN, i, 3, c.ActionType.MOVE, 7, -1, 0.0, *([0.5] * c.N_VALUE_OFFSETS), flags, IN_BYTES) + struct.pack("<I", len(raw)) + raw
	return out


def test_a_session_is_saved_with_each_players_actor_and_the_humans_battles(tmp_path):
	"""One start of a game: its recorders' keys seconds apart (@1000, @1003) are one segment."""
	messages = {
		"human-x@1000:RED": [records(HUMAN, 3)],
		"human-x@1003:BLUE": [records(TEACHER, 2)],
		"human-x@1003:TAN": [records(0, 1)],
		"human-x@1000:ledger": [json.dumps({"day": 1, "players": []}).encode(), json.dumps({"day": 2, "players": []}).encode()],
		"human-x@1000:RED:done": [json.dumps({"teacher": {"matched": 3, "approximate": 1, "unmatched": {"BuyArtifact": 2}}}).encode()],
		"human-x@1000-human-RED": [b"ALPHAREC1\nfirst", b"ALPHAREC1\nsecond"],
	}
	summary = save(tmp_path, messages)
	(segment,) = summary["segments"]
	assert segment["players"] == {"RED": "human", "BLUE": "nk2", "TAN": "alpha"} and segment["decisions"] == 6 and segment["days"] == [1, 3]
	assert segment["battles"] == 2 and segment["ledger_days"] == [1, 2] and segment["damaged_messages"] == 0  # (the battles never reached the records' parser)
	assert segment["human_requests"] == {"RED": {"matched": 3, "approximate": 1, "unmatched": {"BuyArtifact": 2}}}
	assert sorted(p.name for p in (tmp_path / "battles" / "1").iterdir()) == ["human-x-RED-000.rec", "human-x-RED-001.rec"]
	assert (tmp_path / "battles" / "1" / "human-x-RED-001.rec").read_bytes() == b"ALPHAREC1\nsecond"
	players = json.loads((tmp_path / "players.json").read_text())
	assert players["1"]["RED"] == {"actor": "human", "decisions": 3, "by_type": {"MOVE": 3}}
	with open(tmp_path / "segment-1.pkl", "rb") as f:
		game = pickle.load(f)
	assert sum(bool(d.flags & HUMAN) for d in game.decisions) == 3 and sum(bool(d.flags & TEACHER) for d in game.decisions) == 2


def test_a_save_loaded_again_is_a_segment_of_its_own():
	"""A save reloaded in the same client (a day played again): the second start's records, ledgers and battles apart
	from the first's, so neither timeline's labels read the other's ledgers."""
	from campaign_model.record_human import segments

	messages = {
		"k@1000:RED": [records(HUMAN, 2)], "k@1000:ledger": [json.dumps({"day": 5, "players": [], "mark": "first"}).encode()],
		"k@1000-human-RED": [b"first battle"],
		"k@1900:RED": [records(HUMAN, 1)], "k@1901:BLUE": [records(TEACHER, 1)], "k@1900:ledger": [json.dumps({"day": 5, "players": [], "mark": "again"}).encode()],
	}
	(first, a), (second, b) = segments(messages)
	assert (first, second) == (1000, 1900)
	assert set(a) == {"k@1000:RED", "k@1000:ledger", "k@1000-human-RED"} and set(b) == {"k@1900:RED", "k@1901:BLUE", "k@1900:ledger"}


def test_the_human_flag_reads_back_from_a_record():
	"""RECORD_HUMAN (32: AlphaMapAI.h) carries no fields of its own: the record parses as any other."""
	(d,) = parse_records("RED", records(HUMAN, 1))
	assert d.flags == HUMAN and d.index == 7 and zlib.decompress(d.inputs) == bytes(IN_BYTES)


def test_the_sessions_settings_set_the_ais_and_instant_hero_moves(tmp_path, monkeypatch):
	"""user_dir: the shared GUI settings kept, the computer players' AIs set, every hero's map moves instant."""
	from campaign_model import record_human

	shared = tmp_path / "shared"
	(shared / "config").mkdir(parents=True)
	(shared / "config" / "settings.json").write_text(json.dumps({"battle": {"speedFactor": 18}, "adventure": {"tileZoom": 64}}))
	monkeypatch.setattr(record_human, "SHARED", shared)
	path = record_human.user_dir(tmp_path / "run", {"adventureEnemyAI": "Nullkiller2", "combatAlliedAI": "BattleAI"})
	settings = json.loads((path / "config" / "settings.json").read_text())
	assert settings["battle"] == {"speedFactor": 18} and settings["ai"] == {"adventureEnemyAI": "Nullkiller2", "combatAlliedAI": "BattleAI"}
	assert settings["adventure"] == {"tileZoom": 64, "heroMoveTime": 0, "enemyMoveTime": 0}


def test_a_run_makes_its_own_folder_settings_and_environment(tmp_path, monkeypatch):
	"""main(), the GUI client stubbed (it would open a window): the run's folder under --out, the client given the
	recorder's environment (the human recorded, the computer players too unless --player-only), the summary saved."""
	import subprocess
	import sys

	from campaign_model import record_human

	shared = tmp_path / "shared"
	(shared / "config").mkdir(parents=True)
	(shared / "config" / "settings.json").write_text("{}")
	monkeypatch.setattr(record_human, "SHARED", shared)
	runs = []
	monkeypatch.setattr(subprocess, "run", lambda command, cwd, env: runs.append((command, env)))
	for flags, recorded in (([], True), (["--player-only"], False)):
		monkeypatch.setattr(sys, "argv", ["record_human", "--out", str(tmp_path / "human"), *flags])
		record_human.main()
		command, env = runs[-1]
		assert command[0].endswith("build/bin/vcmiclient") and env["ALPHA_RECORD_HUMAN"] == "1" and env["ALPHA_RECORD_HUMAN_SOCKET"]
		assert ("ALPHA_MAP_RECORD_SOCKET" in env) == recorded and "ALPHA_BATTLE_RECORD_SOCKET" not in env
		user = env["VCMI_USER_DIR"]
		settings = json.loads((record_human.Path(user) / "config" / "settings.json").read_text())
		assert settings["ai"]["adventureEnemyAI"] == ("AlphaTeacherAI" if recorded else "Nullkiller2") and settings["ai"]["combatAlliedAI"] == "BattleAI"
		run = record_human.Path(user).parent
		assert run.parent == tmp_path / "human" and env["ALPHA_RECORD_RAW"] == str(run / "raw.jsonl")
		assert json.loads((run / "summary.json").read_text())["line"] == []  # (cut to the line played to the end: none here)
		import time

		time.sleep(1.1)  # (the next run's folder: another start second)


def test_the_line_played_to_the_end_keeps_each_segment_up_to_the_next_load():
	"""kept_days: a load at day s replays from s; each earlier segment keeps only the days before every later start
	(the user's first game: a start abandoned on day 2, the game from day 1, loads of days 21 and 22)."""
	from campaign_model.record_human import kept_days

	assert kept_days([1, 1, 21, 22]) == [(1, 1), (1, 21), (21, 22), (22, math.inf)]
	assert kept_days([1]) == [(1, math.inf)] and kept_days([1, 10, 5]) == [(1, 5), (10, 5), (5, math.inf)]  # (a load of day 5 after one of day 10: that one gone)


def test_a_run_is_cut_to_the_line_played_to_the_end(tmp_path, monkeypatch):
	"""keep_final_line: three segments (the game from day 1, a load of day 3, a load of day 4): every player's decisions
	of the line, a day's ledger from the segment that played the day before, the battles of the line by their day
	(the start event's, else the raw log's battle), the raw lines of the line; the segments' files gone."""
	from types import SimpleNamespace

	from battle_model import data as battle_data
	from campaign_model import record_human
	from campaign_model.records import Decision, Game

	def decision(color, day, flags):
		return Decision(color, day, c.DecisionKind.TURN, 0, c.ActionType.MOVE, 1, -1, -1, 0.0, (0.5,) * c.N_VALUE_OFFSETS, flags, b"")

	def ledger(day, mark):
		return {"day": day, "players": [], "mark": mark}

	segments = [  # (days played, the ledgers it sent: day -> mark)
		(range(1, 6), {d: f"a{d}" for d in range(1, 6)}),  # (the game to day 5: days 3-5 abandoned)
		(range(3, 5), {3: "b3-late", 4: "b4", 5: "b5"}),  # (a load of day 3, played to day 4: day 4 abandoned)
		(range(4, 7), {4: "c4-late", 5: "c5", 6: "c6"}),  # (a load of day 4, played to the end)
	]
	for n, (days, ledgers) in enumerate(segments, start=1):
		game = Game(ledgers={d: ledger(d, m) for d, m in ledgers.items()})
		game.decisions = [decision(color, d, flag) for d in days for color, flag in (("RED", HUMAN), ("BLUE", TEACHER))]
		if n == 3:
			game.final = [{"day": 6, "final": True, "players": [{"color": 0, "status": record_human.STATUS_WON}]}]
		with open(tmp_path / f"segment-{n}.pkl", "wb") as f:
			pickle.dump(game, f)
		(tmp_path / "battles" / str(n)).mkdir(parents=True)
	# battles: segment 1's on days 2 and 4 (the raw log's), segment 3's on day 5 (its start event's)
	(tmp_path / "battles" / "1" / "a.rec").write_bytes(b"one")
	(tmp_path / "battles" / "1" / "b.rec").write_bytes(b"two")
	(tmp_path / "battles" / "3" / "a.rec").write_bytes(b"three")
	records = {b"one": SimpleNamespace(events=[], actions=[1, 2]), b"two": SimpleNamespace(events=[], actions=[1]), b"three": SimpleNamespace(events=[{"event": "start", "day": 5}], actions=[1])}
	monkeypatch.setattr(battle_data, "parse_record", lambda data: records[data])
	raw = [{"request": "17AdvInterfaceReady", "day": 0}, {"request": "MakeAction", "battle": 0, "day": 2}, {"request": "MakeAction", "battle": 0, "day": 2},
		{"request": "EndTurn", "day": 2}, {"request": "MakeAction", "battle": 1, "day": 4}, {"request": "EndTurn", "day": 4},
		{"request": "17AdvInterfaceReady", "day": 3}, {"request": "EndTurn", "day": 3}, {"request": "EndTurn", "day": 4},
		{"request": "17AdvInterfaceReady", "day": 4}, {"request": "EndTurn", "day": 6}]
	(tmp_path / "raw.jsonl").write_text("".join(json.dumps({**r, "actor": "human", "ours": None, "note": None}) + "\n" for r in raw))
	summary = record_human.keep_final_line(tmp_path)
	assert summary["line"] == [{"segment": 1, "days": [1, 2]}, {"segment": 2, "days": [3, 3]}, {"segment": 3, "days": [4, None]}]
	with open(tmp_path / "game.pkl", "rb") as f:
		game = pickle.load(f)
	assert sorted({d.day for d in game.decisions}) == [1, 2, 3, 4, 5, 6] and len(game.decisions) == 12
	assert {d: s["mark"] for d, s in sorted(game.ledgers.items())} == {1: "a1", 2: "a2", 3: "a3", 4: "b4", 5: "c5", 6: "c6"}
	assert summary["won"] and summary["players"] == {"RED": "human", "BLUE": "nk2"}
	assert sorted(p.name for p in (tmp_path / "battles").iterdir()) == ["day002-000.rec", "day005-001.rec"]  # (day 4's abandoned)
	assert [json.loads(l)["day"] for l in (tmp_path / "raw.jsonl").read_text().splitlines()] == [0, 2, 2, 2, 3, 3, 4, 6]  # (each load's start line at its day)
	assert not list(tmp_path.glob("segment-*.pkl"))
