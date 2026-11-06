"""The content match's Python mirror (content/__init__.py; the C++ is agents/AlphaContent): hashes, the match by key,
the append-only merge, and (with ALPHA_CONTENT_EXPORTS: a folder with the exporter's vocab_sod.json and
vocab_hota.json) the real SoD and HotA content."""
import json
import os
import random
from pathlib import Path

import pytest

import content as C


def test_fnv_is_the_cs_formula():
	"""FNV-1a 64's published values (the C++ in Content.cpp computes the same over the same text)."""
	assert C._fnv("") == "cbf29ce484222325"
	assert C._fnv("a") == "af63dc4c8601ec8c"
	assert C._fnv("foobar") == "85944171f73967e8"


def test_the_content_hash_ignores_order_and_the_vocabulary_hash_does_not():
	"""An engine's content hash: the same keys in another load order give the same hash; a vocabulary's hash: a
	reorder is another vocabulary, and so is a dead row."""
	keys = [f"core:c{i}" for i in range(20)]
	shuffled = keys[:]
	random.Random(1).shuffle(shuffled)
	a, b = C.Content({"creatures": keys}), C.Content({"creatures": shuffled})
	assert a.content_hash() == b.content_hash()
	assert a.vocabulary_hash() != b.vocabulary_hash()
	assert C.Content({"creatures": keys + [""]}).vocabulary_hash() != a.vocabulary_hash()
	assert C.Content({"creatures": keys + [""]}).content_hash() == a.content_hash()


def test_the_match_maps_runtime_ids_to_rows_by_key():
	"""Rows by key whatever the runtime ids; what the engine has and the model not is unknown, the reverse absent; a
	key at two runtime ids is a collision; facts are compared."""
	model = C.Content({"creatures": ["core:a", "core:b", "core:c"], "factions": ["core:x"]}, {"army_slots": 7})
	engine = C.Content({"creatures": ["core:c", "", "core:a", "mod:new", "core:a"], "terrains": ["core:dirt"]}, {"army_slots": 8})
	m = C.match(engine, model)
	t = m.tables["creatures"]
	assert t.rows == [2, -1, 0, -1, 0]
	assert t.unknown == ["mod:new"] and t.absent == ["core:b"]
	assert t.collisions == ["core:a (engine ids 2 and 4)"]
	assert m.tables["factions"].absent == ["core:x"] and m.tables["terrains"].unknown == ["core:dirt"]
	assert m.fact_diffs == ["army_slots: the vocabulary's 7, the engine's 8"]
	assert not m.clean(strict=False)
	assert "COLLISIONS 1" in m.report()


def test_a_clean_match_is_strict_only_without_unknowns():
	model = C.Content({"creatures": ["core:a"]})
	assert C.match(C.Content({"creatures": ["core:a"]}), model).clean(strict=True)
	grown = C.match(C.Content({"creatures": ["core:a", "mod:b"]}), model)
	assert grown.clean(strict=False) and not grown.clean(strict=True)


def test_merge_appends_and_never_moves_a_row():
	"""The vocabulary grows append-only: every old row keeps its key (an absent one too: retired, never reused), new
	keys go at the table's end in the engine's order, a new table starts with them; merging again adds nothing."""
	old = C.Content({"creatures": ["core:a", "core:b"], "factions": ["core:x"]}, {"army_slots": 7})
	engine = C.Content({"creatures": ["core:b", "mod:n1", "core:a", "mod:n2"], "terrains": ["mod:t"]}, {"army_slots": 7})
	new = C.merge(old, engine)
	assert new.tables["creatures"] == ["core:a", "core:b", "mod:n1", "mod:n2"]
	assert new.tables["factions"] == ["core:x"] and new.tables["terrains"] == ["mod:t"]
	assert C.is_prefix(old, new) and not C.is_prefix(new, old)
	assert C.merge(new, engine).tables == new.tables
	m = C.match(engine, new)
	assert m.clean(strict=True) and m.tables["creatures"].rows == [1, 2, 0, 3]


EXPORTS = os.environ.get("ALPHA_CONTENT_EXPORTS")


@pytest.mark.skipif(not EXPORTS, reason="needs ALPHA_CONTENT_EXPORTS: the exporter's vocab_sod.json and vocab_hota.json")
def test_sod_and_hota_as_exported():
	"""The exporter's content (agents/AlphaContent): its hashes are the mirror's; SoD against its own vocabulary is
	the identity; SoD grown by HotA matches HotA strictly and keeps every SoD row at its runtime id."""
	folder = Path(EXPORTS)
	for name in ("vocab_sod.json", "vocab_hota.json"):
		raw = json.loads((folder / name).read_text())["content"]
		c = C.Content.from_json(raw)
		assert (c.content_hash(), c.vocabulary_hash()) == (raw["content_hash"], raw["vocabulary_hash"])
	sod, hota = C.load(folder / "vocab_sod.json"), C.load(folder / "vocab_hota.json")
	own = C.match(sod, sod)
	assert own.clean(strict=True) and own.unknown_count() == 0
	assert all(row == i for t in own.tables.values() for i, row in enumerate(t.rows) if row >= 0)
	merged = C.merge(sod, hota)
	assert C.is_prefix(sod, merged)
	assert C.match(hota, merged).clean(strict=True)
	through = C.match(sod, merged)
	assert through.clean(strict=True)
	assert all(row == i for t in through.tables.values() for i, row in enumerate(t.rows) if row >= 0)
