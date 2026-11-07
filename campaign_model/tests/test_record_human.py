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
	key = "human-test"
	messages = {
		f"{key}:RED": [records(HUMAN, 3)],
		f"{key}:BLUE": [records(TEACHER, 2)],
		f"{key}:TAN": [records(0, 1)],
		f"{key}:ledger": [json.dumps({"day": 1, "players": []}).encode(), json.dumps({"day": 2, "players": []}).encode()],
		f"{key}:RED:done": [json.dumps({"teacher": {"matched": 3, "approximate": 1, "unmatched": {"BuyArtifact": 2}}}).encode()],
		f"{key}-human-RED": [b"ALPHAREC1\nfirst", b"ALPHAREC1\nsecond"],
	}
	summary = save(tmp_path, messages)
	assert summary["players"] == {"RED": "human", "BLUE": "nk2", "TAN": "alpha"} and summary["decisions"] == 6
	assert summary["battles"] == 2 and summary["battle_decisions"] == {f"{key}-human-RED": 2} and summary["ledger_days"] == [1, 2]
	assert summary["human_requests"] == {"RED": {"matched": 3, "approximate": 1, "unmatched": {"BuyArtifact": 2}}}
	assert summary["damaged_messages"] == 0  # (the battles never reached the records' parser)
	assert sorted(p.name for p in (tmp_path / "battles").iterdir()) == [f"{key}-human-RED-000.rec", f"{key}-human-RED-001.rec"]
	assert (tmp_path / "battles" / f"{key}-human-RED-001.rec").read_bytes() == b"ALPHAREC1\nsecond"
	players = json.loads((tmp_path / "players.json").read_text())
	assert players["RED"] == {"actor": "human", "decisions": 3, "by_type": {"MOVE": 3}}
	with open(tmp_path / "game.pkl", "rb") as f:
		game = pickle.load(f)
	assert sum(bool(d.flags & HUMAN) for d in game.decisions) == 3 and sum(bool(d.flags & TEACHER) for d in game.decisions) == 2


def test_the_human_flag_reads_back_from_a_record():
	"""RECORD_HUMAN (32: AlphaMapAI.h) carries no fields of its own: the record parses as any other."""
	(d,) = parse_records("RED", records(HUMAN, 1))
	assert d.flags == HUMAN and d.index == 7 and zlib.decompress(d.inputs) == bytes(IN_BYTES)
