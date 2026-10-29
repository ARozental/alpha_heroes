import struct
import zlib

import numpy as np

from campaign_model import constants as c
from campaign_model.records import HEADER, IN_LAYOUT, collect, decode_inputs
from campaign_model.schema import random_inputs


def wire(x, b):
	return b"".join(np.ascontiguousarray(x[name][b]).astype(w).tobytes() for name, _, w, _, _ in IN_LAYOUT)


def message(x, n):
	"""A records message as AlphaMapAI::sendRecords writes it."""
	raw = b"AMR1" + struct.pack("<I", n)
	for b in range(n):
		inputs = wire(x, b)
		raw += HEADER.pack(b + 1, int(x["g_kind"][b]), b, 7, 1, 5, -1, -0.5, *([0.25] * c.N_VALUE_OFFSETS), 1, len(inputs)) + inputs
	return b"AMZ1" + struct.pack("<I", len(raw)) + zlib.compress(raw)


def test_records_round_trip():
	x = random_inputs(3, seed=4)
	game = collect({"g0:RED": [message(x, 3)], "g0:RED:done": [b'{"errors": 0}'], "g0:ledger": [b'{"day": 1, "players": []}']})
	assert [d.day for d in game.decisions] == [1, 2, 3] and game.decisions[0].flags == 1 and game.done == {"RED": {"errors": 0}}
	assert 1 in game.ledgers
	decoded = decode_inputs(game.decisions)
	for name, value in x.items():
		assert np.array_equal(decoded[name], value), name


def test_a_damaged_records_message_is_left_out_not_the_game():
	from campaign_model.records import collect

	game = collect({"g:RED": [b"AMZ1\x10\x00\x00\x00" + b"\x78\x9c\x01"], "g:ledger": [b'{"day": 1, "players": []}']})
	assert game.damaged == 1 and game.decisions == [] and 1 in game.ledgers
