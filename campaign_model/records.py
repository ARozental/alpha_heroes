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
TEACHER = 2  # Nullkiller2's decision (AlphaTeacherAI: agents/AlphaMapAI/Teacher.cpp): its choice, no model's logp


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
	# color -> its heroes' battles in order (AlphaMapAI::battleStart/battleEnd, "<game>:<COLOR>:battle"): "serial" (the
	# player's record count before it: the decision that led there is its serial-th), "side" (0 attacker), "armies" and
	# "heroes" at the start per side, "result" (EBattleResult), "winner" (0, 1, 2 a draw), "casualties" per side, "exp"
	battles: dict = field(default_factory=dict)
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
			elif parts[-1] == "battle":
				try:
					game.battles.setdefault(parts[-2], []).append(json.loads(payload))
				except (json.JSONDecodeError, UnicodeDecodeError):
					game.damaged += 1
			else:
				game.record_bytes += len(payload)
				try:
					game.decisions += parse_records(parts[-1], payload)
				except (ValueError, zlib.error, struct.error):  # cut off (a game killed by its time limit as it sent them): this message only
					game.damaged += 1
	return game


def decode_fields(decisions, names):
	"""{name: array [N, ...]} of only these inputs of the decisions (each decision decompressed once; the map's big
	inputs, left out, are most of its bytes)."""
	wanted = set(names)
	fields, offset = [], 0
	for name, shape, wire, dtype, nbytes in IN_LAYOUT:
		if name in wanted:
			fields.append((name, shape, wire, dtype, offset, nbytes))
		offset += nbytes
	# One decision decompressed at a time, only its wanted bytes kept: a decision is ~874 KB whole, so a game's at once
	# was ~2.6 GB, and an update's games one after another took the PC's memory guard down (2026-10-30)
	parts = {name: [] for name, *_ in fields}
	for d in decisions:
		row = padded(zlib.decompress(d.inputs))
		for name, _, _, _, start, nbytes in fields:
			parts[name].append(row[start : start + nbytes])
	return {name: np.frombuffer(b"".join(parts[name]), dtype=wire).astype(dtype).reshape((len(decisions),) + shape) for name, shape, wire, dtype, _, _ in fields}


def decode_wire(decisions):
	"""The decisions' inputs as sent, [N, IN_BYTES] uint8 (zlib only): for inputs_on_device, which cuts and converts
	them on the GPU (half the bytes to copy there, and no dtype copies on the CPU: 2026-10-30, the trainer's thread)."""
	return np.frombuffer(b"".join(padded(zlib.decompress(d.inputs)) for d in decisions), dtype=np.uint8).reshape(len(decisions), IN_BYTES)


MEM_BYTES = IN_LAYOUT[-1][-1] if IN_LAYOUT[-1][0] == "mem" else 0


def padded(row):
	"""A decision's decompressed inputs, a record of schema 9 (before the memory, the last input) given zeros there."""
	return row + bytes(MEM_BYTES) if MEM_BYTES and len(row) == IN_BYTES - MEM_BYTES else row


def decode_inputs(decisions):
	"""{input name: array [N, ...]} of the decisions' inputs, in the model's dtypes."""
	data = decode_wire(decisions)
	inputs, offset = {}, 0
	for name, shape, wire, dtype, nbytes in IN_LAYOUT:
		column = np.ascontiguousarray(data[:, offset : offset + nbytes]).view(wire)
		inputs[name] = column.astype(dtype).reshape((len(decisions),) + shape)
		offset += nbytes
	return inputs


_TORCH_DTYPES = None


def inputs_on_device(wire):
	"""{input name: tensor [N, ...]} in the model's dtypes from decode_wire's bytes as a tensor [N, IN_BYTES] on the
	device (battle_model/inference.py's decode_on_device, for training)."""
	import torch

	global _TORCH_DTYPES
	if _TORCH_DTYPES is None:
		_TORCH_DTYPES = {np.dtype(t): getattr(torch, name) for t, name in ((np.uint8, "uint8"), (np.int32, "int32"), (np.float32, "float32"), (np.bool_, "bool"), (np.int64, "int64"))}
	inputs, offset, n = {}, 0, wire.shape[0]
	for name, shape, wire_dtype, dtype, nbytes in IN_LAYOUT:
		# (a copy always: one row's slice counts as contiguous and kept its odd stride, which the view refused)
		column = wire[:, offset : offset + nbytes].clone(memory_format=torch.contiguous_format).view(_TORCH_DTYPES[np.dtype(wire_dtype)])
		inputs[name] = column.to(_TORCH_DTYPES[np.dtype(dtype)]).reshape((n,) + shape)
		offset += nbytes
	return inputs
