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
import ctypes
import ctypes.util
import itertools
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
ADVICE = 4  # our decision with Nullkiller2's choice in the same state beside it (Decision.advice: Advice.cpp)
ADVICE_FIELDS = struct.Struct("<iii")  # (after the header, when flags has ADVICE: its action type, entry, tile)
PLAN = 8  # Nullkiller2's plan in the same state beside it (Decision.plan: Advice.cpp; a label of the foresight's)
# (after the advice, when flags has PLAN: each of our hero slots' target tile x, y, z, its days, its task's priority; x
# -1: no task in the plan, -2: a task without a tile of the hero's)
PLAN_FIELDS = struct.Struct("<" + "hhhhf" * c.OWN_HEROES)
PLAN_ENTRY = 5
ACCEPT = 16  # more of our entries that follow Nullkiller2 in the same state (Decision.accept: Advice.cpp; advice.py)
ACCEPT_COUNT, ACCEPT_ENTRY = struct.Struct("<H"), struct.Struct("<hi")  # (after the plan, when flags has ACCEPT: their count, then each one's type and entry)
# The questions the teacher's recorder records Nullkiller2's answers to (Teacher.cpp): the others are the model's own
TEACHER_QUESTIONS = (c.DecisionKind.YES_NO, c.DecisionKind.SELECT, c.DecisionKind.LEVEL_UP)


def decision_name(d):
	"""A decision's kind for counts: its action type's name (a turn's), else its question's kind."""
	return c.ActionType(d.action_type).name if d.action_type >= 0 else c.DecisionKind(d.kind).name


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
	advice: tuple = None  # (action type, index, cell) of Nullkiller2's choice in this state (flags ADVICE), else None
	plan: tuple = None  # Nullkiller2's plan in this state (flags PLAN): PLAN_ENTRY values a hero slot (PLAN_FIELDS), else None
	accept: tuple = None  # (flags ACCEPT) more of our entries that follow Nullkiller2 here: type, entry, type, entry, ...


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


def read_accept(buffer, offset, flags):
	"""(Decision.accept or None, the offset past it) of a record whose fields reach offset."""
	if not flags & ACCEPT:
		return None, offset
	(k,) = ACCEPT_COUNT.unpack_from(buffer, offset)
	offset += ACCEPT_COUNT.size
	return struct.unpack_from("<" + "hi" * k, buffer, offset), offset + ACCEPT_ENTRY.size * k


def parse_records(color, payload):
	if payload[:4] == b"AMZ2":
		(count,) = struct.unpack_from("<I", payload, 4)
		offset, out = 8, []
		for _ in range(count):
			day, kind, today, version, action_type, index, cell, logp, *rest = HEADER.unpack_from(payload, offset)
			value, flags, n = tuple(rest[: c.N_VALUE_OFFSETS]), rest[-2], rest[-1]
			offset += HEADER.size
			advice = plan = None
			if flags & ADVICE:
				advice = ADVICE_FIELDS.unpack_from(payload, offset)
				offset += ADVICE_FIELDS.size
			if flags & PLAN:
				plan = PLAN_FIELDS.unpack_from(payload, offset)
				offset += PLAN_FIELDS.size
			accept, offset = read_accept(payload, offset, flags)
			(m,) = struct.unpack_from("<I", payload, offset)
			offset += 4
			if n != IN_BYTES:
				raise ValueError(f"a decision has {n} input bytes, the contract {IN_BYTES}")
			if offset + m > len(payload):
				raise ValueError("a damaged records message")
			out.append(Decision(color, day, kind, today, version, action_type, index, cell, logp, value, flags, bytes(payload[offset : offset + m]), advice, plan, accept))
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
		advice = plan = None
		if flags & ADVICE:
			advice = ADVICE_FIELDS.unpack_from(raw, offset)
			offset += ADVICE_FIELDS.size
		if flags & PLAN:
			plan = PLAN_FIELDS.unpack_from(raw, offset)
			offset += PLAN_FIELDS.size
		accept, offset = read_accept(raw, offset, flags)
		if n != IN_BYTES:
			raise ValueError(f"a decision has {n} input bytes, the contract {IN_BYTES}")
		out.append(Decision(color, day, kind, today, version, action_type, index, cell, logp, value, flags, zlib.compress(raw[offset : offset + n], 1), advice, plan, accept))
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


def decode_wire(decisions, out=None):
	"""The decisions' inputs as sent, [N, IN_BYTES] uint8 (zlib only): for inputs_on_device, which cuts and converts
	them on the GPU (half the bytes to copy there, and no dtype copies on the CPU: 2026-10-30, the trainer's thread).
	Each record inflated straight into its row (_inflater), of `out` when given (C-contiguous): the trainer's pinned
	buffers, resident and reused. 2026-11-03 on the PC, 64 decisions a batch over 3 threads: 38.8 ms a batch through
	zlib.decompress and a join (a new 876 KB object a record, its page faults serializing the threads), 8.1 ms into a
	reused buffer."""
	out = np.empty((len(decisions), IN_BYTES), dtype=np.uint8) if out is None else out
	assert out.shape == (len(decisions), IN_BYTES) and out.dtype == np.uint8 and out.flags.c_contiguous
	inflate, address = _inflater(), out.ctypes.data
	for i, d in enumerate(decisions):
		n = inflate(d.inputs, out[i], address + i * IN_BYTES)
		if n != IN_BYTES:
			if n not in SHORT_RECORDS:
				raise ValueError(f"a decision's inputs are {n} bytes, not {IN_BYTES}")
			out[i, n:] = 0  # (padded: a record of an older schema)
	return out


_INFLATE = None


def _inflater():
	"""inflate(compressed, row, its address) -> the bytes written: libz's uncompress straight into the row (no object
	made, no copy, the interpreter lock released by ctypes), or zlib.decompress and a copy where libz does not load."""
	global _INFLATE
	if _INFLATE is None:
		lib = _libz()
		if lib is None:

			def inflate(data, row, address):
				raw = zlib.decompress(data)
				row[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
				return len(raw)
		else:
			uncompress = lib.uncompress
			uncompress.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong), ctypes.c_char_p, ctypes.c_ulong)
			uncompress.restype = ctypes.c_int

			def inflate(data, row, address):
				size = ctypes.c_ulong(row.nbytes)
				status = uncompress(address, ctypes.byref(size), data, len(data))
				if status != 0:  # (Z_BUF_ERROR -5: more than the row; Z_DATA_ERROR -3: damaged)
					raise zlib.error(f"libz uncompress: error {status}")
				return size.value

		_INFLATE = inflate
	return _INFLATE


def _libz():
	for name in ("libz.so.1", None):  # (Linux's by name; elsewhere where find_library says)
		try:
			path = name or ctypes.util.find_library("z")
			if path:
				return ctypes.CDLL(path)
		except OSError:
			pass
	return None


# Records made before inputs were added at the end (schema 10: mem; schema 11: the question's guards and armed objects,
# from o_battle on) end where those start: decoded with zeros there, which each of them reads as none (no memory, no
# guard, no battle, no armed object)
TAIL_STARTS = ("mem", "o_battle")
_STARTS = {entry[0]: sum(e[-1] for e in IN_LAYOUT[:i]) for i, entry in enumerate(IN_LAYOUT)}
SHORT_RECORDS = frozenset(_STARTS[name] for name in TAIL_STARTS if name in _STARTS)  # the byte lengths of older records
MEM_BYTES = IN_BYTES - _STARTS["mem"]  # what a record of schema 9 (before the memory) lacks


def padded(row):
	"""A decision's decompressed inputs, a record of an older schema (SHORT_RECORDS) given zeros where it ends."""
	return row + bytes(IN_BYTES - len(row)) if len(row) in SHORT_RECORDS else row


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


def heroes_in(decoded):
	"""Our heroes in each decision, [N] on the host (h_present's first OWN_HEROES slots), from decode_inputs' inputs or
	decode_wire's bytes: how many windows a batch needs (model_v2.windows_of's budget)."""
	if isinstance(decoded, dict):
		present = decoded["h_present"]
	else:
		offset, (_, shape, wire_dtype, _, nbytes) = _WIRE_AT["h_present"]
		decoded = np.asarray(decoded)  # (a tensor: train.wire_for's pinned one)
		present = np.ascontiguousarray(decoded[:, offset : offset + nbytes]).view(wire_dtype).reshape((len(decoded),) + shape)
	return np.count_nonzero(present[:, : c.OWN_HEROES], axis=1)


_WIRE_AT = {entry[0]: (offset, entry) for entry, offset in zip(IN_LAYOUT, itertools.accumulate((entry[-1] for entry in IN_LAYOUT), initial=0))}


def inputs_on_device(wire):
	"""{input name: tensor [N, ...]} in the model's dtypes from decode_wire's bytes as a tensor [N, IN_BYTES] on the
	device (battle_model/inference.py's decode_on_device, for training)."""
	return {name: _column(wire, offset, entry) for name, (offset, entry) in _WIRE_AT.items()}


def input_on_device(wire, name):
	"""One of inputs_on_device's inputs alone (the update's g_players beside a graph that converts the rest itself)."""
	offset, entry = _WIRE_AT[name]
	return _column(wire, offset, entry)


def _column(wire, offset, entry):
	import torch

	global _TORCH_DTYPES
	if _TORCH_DTYPES is None:
		_TORCH_DTYPES = {np.dtype(t): getattr(torch, name) for t, name in ((np.uint8, "uint8"), (np.int32, "int32"), (np.float32, "float32"), (np.bool_, "bool"), (np.int64, "int64"))}
	_, shape, wire_dtype, dtype, nbytes = entry
	# (a copy always: one row's slice counts as contiguous and kept its odd stride, which the view refused)
	column = wire[:, offset : offset + nbytes].clone(memory_format=torch.contiguous_format).view(_TORCH_DTYPES[np.dtype(wire_dtype)])
	return column.to(_TORCH_DTYPES[np.dtype(dtype)]).reshape((wire.shape[0],) + tuple(shape))
