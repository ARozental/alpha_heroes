"""Schema 20 (2026-11-12): where today's walk along the engine's path to each window tile stops, as our MOVE entry
(w_stop). An input only: no parameter reads it yet. The encoder's own invariants (a stop only on tiles a day or more
away, every stop a legal MOVE entry of the same hero) are checked on recorded games by the premise check."""
import zlib
from types import SimpleNamespace

import numpy as np

from campaign_model import constants as c
from campaign_model.records import _STARTS, IN_BYTES, SHORT_RECORDS, decode_inputs
from campaign_model.schema import random_inputs
from campaign_model.tests.test_model import encoded

HO, W = c.OWN_HEROES, c.WINDOW


def test_a_record_from_before_reads_with_no_stop():
	"""A record of schema 19 ends where w_stop starts: decoded, every tile reads "none"; the rest as it was."""
	assert _STARTS["w_stop"] in SHORT_RECORDS and _STARTS["w_stop"] < IN_BYTES
	x = random_inputs(4, seed=3)
	b = int(np.flatnonzero(x["w_stop"].reshape(4, -1).any(1))[0])
	x = {k: v[b : b + 1] for k, v in x.items()}
	full = zlib.decompress(encoded(x, 0))
	now = decode_inputs([SimpleNamespace(inputs=zlib.compress(full))])
	before = decode_inputs([SimpleNamespace(inputs=zlib.compress(full[: _STARTS["w_stop"]]))])
	assert np.array_equal(now["w_stop"], x["w_stop"]) and not before["w_stop"].any()
	for name in x:
		if name != "w_stop":
			assert np.array_equal(now[name], before[name]), name


def test_random_stops_are_legal_moves_of_tiles_a_day_or_more_away():
	x = random_inputs(4, seed=5)
	for b in range(4):
		stop = x["w_stop"][b].reshape(HO, -1).astype(np.int64) - 1
		for slot, cell in zip(*np.nonzero(stop >= 0)):
			assert x["w_turns"][b, slot].reshape(-1)[cell] >= 1
			entry = stop[slot, cell]
			assert x["move_mask"][b].reshape(-1)[entry] and entry // (W * W) == slot
