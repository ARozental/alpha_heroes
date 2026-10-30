"""Decision records and daily ledgers of a game, as the AIs send them (agents/AlphaMapAI/AlphaMapAI.cpp sendRecords,
Ledger.cpp) to a RecordServer (battle_model/play.py) under the keys "<game>:<COLOR>" (records), "<game>:ledger"
(snapshots) and "<game>:<COLOR>:done" (a player stopped at the day cap or the end of its game).

A records message: "AMZ2", u32 count, count x decision; a decision: u32 day, u32 kind, u32 decisions earlier that day,
u32 model version, i32 action type (-1 for a question), i32 index, i32 cell, f32 log-probability,
f32 value[N_VALUE_OFFSETS], u32 flags (NO_EFFECT), u32 n, u32 m, m bytes: the n bytes of its inputs (the inference
server's wire encoding) zlib-compressed. A decision's inputs are ~340 KB (eight hero windows of 33 x 33 tiles), so each
is kept compressed (1-4 KB) as it came until a minibatch decodes it (play.py asks for this format: ALPHA_MAP_RECORD_FORMAT=2).
Without it (and before 2026-10-29): "AMZ1", u32 size, zlib("AMR1", u32 count, count x decision with u32 n, n bytes of
inputs), the whole message compressed at once, each decision compressed again here.
"""
import json
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

from battle_model.inference import wire_layout

from . import constants as c
from .schema import INPUTS

IN_LAYOUT = wire_layout(INPUTS, True)
IN_BYTES = sum(entry[-1] for entry in IN_LAYOUT)
HEADER = struct.Struct("<IIIIiiif" + "f" * c.N_VALUE_OFFSETS + "II")
NO_EFFECT = 1  # the action was carried out and changed nothing


@dataclass
class Decision:
	color: str
	day: int
	kind: int
	today: int  # turn decisions earlier that day
	version: int
	action_type: int  # -1 for a question
	index: int
	cell: int
	logp: float
	value: tuple
	flags: int
	inputs: bytes  # zlib-compressed wire encoding (decode_inputs)


@dataclass
class Game:
	decisions: list = field(default_factory=list)  # in each player's order
	ledgers: dict = field(default_factory=dict)  # day -> snapshot taken at the start of that day
	final: list = field(default_factory=list)  # snapshots taken at the end of the game
	done: dict = field(default_factory=dict)  # color -> the AI's summary ({"errors", "failed_actions"})
	record_bytes: int = 0
	damaged: int = 0  # records messages and ledger snapshots that could not be read (left out)
	days: int = 0  # the day cap it was played to (0: none known): reaching it is the game's end (labels.end_labels)


def parse_records(color, payload):
	if payload[:4] == b"AMZ2":
		(count,) = struct.unpack_from("<I", payload, 4)
		offset, out = 8, []
		for _ in range(count):
			day, kind, today, version, action_type, index, cell, logp, *rest = HEADER.unpack_from(payload, offset)
			value, flags, n = tuple(rest[: c.N_VALUE_OFFSETS]), rest[-2], rest[-1]
			(m,) = struct.unpack_from("<I", payload, offset + HEADER.size)
			offset += HEADER.size + 4
			if n != IN_BYTES:
				raise ValueError(f"a decision has {n} input bytes, the contract {IN_BYTES}")
			if offset + m > len(payload):
				raise ValueError("a damaged records message")
			out.append(Decision(color, day, kind, today, version, action_type, index, cell, logp, value, flags, bytes(payload[offset : offset + m])))
			offset += m
		return out
	if payload[:4] != b"AMZ1":
		raise ValueError("not a records message")
	(size,) = struct.unpack_from("<I", payload, 4)
	raw = zlib.decompress(payload[8:])
	if len(raw) != size or raw[:4] != b"AMR1":
		raise ValueError("a damaged records message")
	(count,) = struct.unpack_from("<I", raw, 4)
	offset, out = 8, []
	for _ in range(count):
		day, kind, today, version, action_type, index, cell, logp, *rest = HEADER.unpack_from(raw, offset)
		value, flags, n = tuple(rest[: c.N_VALUE_OFFSETS]), rest[-2], rest[-1]
		offset += HEADER.size
		if n != IN_BYTES:
			raise ValueError(f"a decision has {n} input bytes, the contract {IN_BYTES}")
		out.append(Decision(color, day, kind, today, version, action_type, index, cell, logp, value, flags, zlib.compress(raw[offset : offset + n], 1)))
		offset += n
	return out


def collect(messages):
	"""A Game from a RecordServer's {key: [payload, ...]} of one game."""
	game = Game()
	for key, payloads in sorted(messages.items()):
		parts = key.split(":")
		for payload in payloads:
			if parts[-1] == "ledger":
				try:
					snapshot = json.loads(payload)
				except (json.JSONDecodeError, UnicodeDecodeError):  # cut off (the game killed as it sent it): this snapshot only
					game.damaged += 1
					continue
				if snapshot.get("final"):
					game.final.append(snapshot)
				else:
					game.ledgers[snapshot["day"]] = snapshot
			elif parts[-1] == "done":
				game.done[parts[-2]] = json.loads(payload) if payload else {}
			else:
				game.record_bytes += len(payload)
				try:
					game.decisions += parse_records(parts[-1], payload)
				except (ValueError, zlib.error, struct.error):  # cut off (a game killed by its time limit as it sent them): this message only
					game.damaged += 1
	return game


def decode_inputs(decisions):
	"""{input name: array [N, ...]} of the decisions' inputs, in the model's dtypes."""
	data = np.frombuffer(b"".join(zlib.decompress(d.inputs) for d in decisions), dtype=np.uint8).reshape(len(decisions), IN_BYTES)
	inputs, offset = {}, 0
	for name, shape, wire, dtype, nbytes in IN_LAYOUT:
		column = np.ascontiguousarray(data[:, offset : offset + nbytes]).view(wire)
		inputs[name] = column.astype(dtype).reshape((len(decisions),) + shape)
		offset += nbytes
	return inputs
