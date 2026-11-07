"""record_human.py's saving of a session (2026-11-07): each colour's decisions labelled by who made them (HUMAN the
human's recorder, TEACHER Nullkiller2's, neither our model's), the human's battle records kept apart and saved as sent,
and the human's request audit (their "done" summary's "teacher" counts) in the summary."""
import json
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
