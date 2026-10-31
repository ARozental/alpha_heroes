import struct
import zlib

import numpy as np

from campaign_model import constants as c
from campaign_model.records import HEADER, IN_LAYOUT, collect, decode_inputs
from campaign_model.schema import random_inputs


def wire(x, b):
	return b"".join(np.ascontiguousarray(x[name][b]).astype(w).tobytes() for name, _, w, _, _ in IN_LAYOUT)


def message(x, n):
	"""A records message as AlphaMapAI::sendRecords writes it: each decision's inputs compressed on their own."""
	out = b"AMZ2" + struct.pack("<I", n)
	for b in range(n):
		inputs = wire(x, b)
		packed = zlib.compress(inputs, 1)
		out += HEADER.pack(b + 1, int(x["g_kind"][b]), b, 7, 1, 5, -1, -0.5, *([0.25] * c.N_VALUE_OFFSETS), 1, len(inputs)) + struct.pack("<I", len(packed)) + packed
	return out


def old_message(x, n):
	"""A records message as AlphaMapAI::sendRecords wrote it before 2026-10-29: the whole message compressed."""
	raw = b"AMR1" + struct.pack("<I", n)
	for b in range(n):
		inputs = wire(x, b)
		raw += HEADER.pack(b + 1, int(x["g_kind"][b]), b, 7, 1, 5, -1, -0.5, *([0.25] * c.N_VALUE_OFFSETS), 1, len(inputs)) + inputs
	return b"AMZ1" + struct.pack("<I", len(raw)) + zlib.compress(raw)


def test_records_round_trip():
	x = random_inputs(3, seed=4)
	for make in (message, old_message):
		game = collect({"g0:RED": [make(x, 3)], "g0:RED:done": [b'{"errors": 0}'], "g0:ledger": [b'{"day": 1, "players": []}']})
		assert [d.day for d in game.decisions] == [1, 2, 3] and game.decisions[0].flags == 1 and game.done == {"RED": {"errors": 0}}
		assert 1 in game.ledgers
		decoded = decode_inputs(game.decisions)
		for name, value in x.items():
			assert np.array_equal(decoded[name], value), name


def test_a_records_message_cut_short_is_left_out():
	x = random_inputs(2, seed=5)
	whole = message(x, 2)
	game = collect({"g:RED": [whole[: len(whole) - 100]], "g:ledger": [b'{"day": 1, "players": []}']})
	assert game.damaged == 1 and game.decisions == []


def test_a_damaged_records_message_is_left_out_not_the_game():
	from campaign_model.records import collect

	game = collect({"g:RED": [b"AMZ1\x10\x00\x00\x00" + b"\x78\x9c\x01"], "g:ledger": [b'{"day": 1, "players": []}']})
	assert game.damaged == 1 and game.decisions == [] and 1 in game.ledgers


def test_a_damaged_ledger_snapshot_is_left_out_not_the_game():
	from campaign_model.records import collect

	game = collect({"g:ledger": [b'{"day": 1, "players": []}', b'{"day": 2, "players": [{"color": 0,']})
	assert game.damaged == 1 and list(game.ledgers) == [1]


def test_inputs_decoded_on_a_device_are_the_inputs():
	"""(2026-10-30) The training update sends the bytes as recorded to the GPU and cuts and converts them there
	(records.inputs_on_device): the same inputs as decode_inputs gives on the CPU."""
	import numpy as np
	import torch

	from campaign_model.records import decode_inputs, decode_wire, inputs_on_device
	from campaign_model.schema import random_inputs
	from campaign_model.tests.test_model import encoded
	from campaign_model.tests.test_train import decisions_for
	from campaign_model.model import CampaignModel

	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	x = random_inputs(5, seed=3)
	decisions = [d.__class__(**{**d.__dict__, "inputs": encoded(x, b)}) for b, d in enumerate(decisions_for(model, x))]
	want = decode_inputs(decisions)
	device = "mps" if torch.backends.mps.is_available() else "cpu"
	got = inputs_on_device(torch.from_numpy(decode_wire(decisions)).to(device))
	assert set(got) == set(want)
	for name in want:
		assert got[name].shape == want[name].shape and str(got[name].dtype).replace("torch.", "") == {"bool": "bool", "int64": "int64", "float32": "float32"}[str(want[name].dtype)]
		np.testing.assert_array_equal(got[name].cpu().numpy(), want[name])
