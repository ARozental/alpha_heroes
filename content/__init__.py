"""The content match's Python mirror (docs/hota_transfer_plan.md, Part 1; the C++ is agents/AlphaContent/Content.cpp).

A vocabulary is each table's keys in row order (the model's), engine content each table's keys by runtime id (what
an engine loaded: the exporter's "content" section). They are matched by key, so numeric ids and mod load order never
reach a record or a model. Tables:
- plain: creatures, artifacts, spells, secondary_skills, heroes, hero_classes, factions, object_types, terrains,
  roads, rivers, battlefields, obstacles, bonus_types;
- per owner: buildings/<faction key> by BuildingID, subtypes/<object type key> by subtype id.

The hashes are FNV-1a 64 over the same text as the C++'s:
- the content hash over each table's keys sorted (load order never enters it);
- the vocabulary hash over them in row order (a reorder is another vocabulary)."""
import json
from dataclasses import dataclass, field
from pathlib import Path

FNV_OFFSET, FNV_PRIME, MASK = 14695981039346656037, 1099511628211, (1 << 64) - 1


@dataclass
class Content:
	tables: dict = field(default_factory=dict)  # name -> [key]; "" where no entity (engine) or a dead row (vocabulary)
	facts: dict = field(default_factory=dict)  # engine constants the model assumes: name -> int
	renamed: list = field(default_factory=list)  # the exporter's renames of a key twice in a table ("t: k -> k#2")

	def content_hash(self):
		return _fnv("".join(f"{name}\n" + "".join(f"{key}\n" for key in sorted(k for k in keys if k)) + "\n" for name, keys in sorted(self.tables.items())))

	def vocabulary_hash(self):
		return _fnv("".join(f"{name}\n" + "".join(f"{key}\n" for key in keys) + "\n" for name, keys in sorted(self.tables.items())))

	def to_json(self):
		return {"tables": self.tables, "facts": self.facts, "renamed": self.renamed, "content_hash": self.content_hash(), "vocabulary_hash": self.vocabulary_hash()}

	@classmethod
	def from_json(cls, node):
		node = node.get("content", node)
		return cls({name: list(keys) for name, keys in node.get("tables", {}).items()}, dict(node.get("facts", {})), list(node.get("renamed", [])))


@dataclass
class TableMatch:
	rows: list = field(default_factory=list)  # runtime id -> the model's row, -1: no entity there or unknown to the model
	unknown: list = field(default_factory=list)  # the engine has it, the model does not
	absent: list = field(default_factory=list)  # the model has it, the engine does not (fine)
	collisions: list = field(default_factory=list)  # one key at two runtime ids, or twice in the vocabulary: must be empty


@dataclass
class Match:
	tables: dict
	fact_diffs: list
	content_hash: str
	vocabulary_hash: str

	def unknown_count(self):
		return sum(len(t.unknown) for t in self.tables.values())

	def clean(self, strict):
		"""No collision, no fact difference, and in strict mode (a training game) no unknown either."""
		return not any(t.collisions for t in self.tables.values()) and not self.fact_diffs and (not strict or self.unknown_count() == 0)

	def report(self, shown=12):
		"""The readable diff: each table with something to say, its counts and keys."""
		keys = lambda values: ", ".join(values[:shown]) + (f", ... ({len(values) - shown} more)" if len(values) > shown else "")  # noqa: E731
		lines = [f"content hash {self.content_hash}, vocabulary hash {self.vocabulary_hash}"]
		for name, t in self.tables.items():
			parts = [f"unknown {len(t.unknown)} [{keys(t.unknown)}]" if t.unknown else "", f"absent {len(t.absent)} [{keys(t.absent)}]" if t.absent else "", f"COLLISIONS {len(t.collisions)} [{keys(t.collisions)}]" if t.collisions else ""]
			if any(parts):
				lines.append(f"{name}: " + " ".join(p for p in parts if p))
		lines += [f"FACT DIFFERS: {d}" for d in self.fact_diffs]
		quiet = sum(1 for t in self.tables.values() if not (t.unknown or t.absent or t.collisions))
		matched = sum(sum(1 for r in t.rows if r >= 0) for t in self.tables.values())
		lines.append(f"{len(self.tables)} tables ({quiet} matched in full): {matched} matched, {self.unknown_count()} unknown, {sum(len(t.absent) for t in self.tables.values())} absent, {sum(len(t.collisions) for t in self.tables.values())} collisions, {len(self.fact_diffs)} fact differences")
		return "\n".join(lines)


def match(engine, model):
	"""engine: what an engine loaded (keys by runtime id); model: the vocabulary (keys in row order)."""
	tables = {}
	for name in sorted(set(engine.tables) | set(model.tables)):
		engine_keys, model_keys, t = engine.tables.get(name, []), model.tables.get(name, []), TableMatch()
		row_of = {}
		for row, key in enumerate(model_keys):
			if key and key in row_of:
				t.collisions.append(f"{key} (twice in the vocabulary)")
			elif key:
				row_of[key] = row
		id_of = {}
		t.rows = [-1] * len(engine_keys)
		for i, key in enumerate(engine_keys):
			if not key:
				continue
			fresh = key not in id_of
			if fresh:
				id_of[key] = i
			else:
				t.collisions.append(f"{key} (engine ids {id_of[key]} and {i})")
			if key in row_of:
				t.rows[i] = row_of[key]
			elif fresh:
				t.unknown.append(key)
		t.absent = sorted(key for key in row_of if key not in id_of)
		tables[name] = t
	diffs = []
	for name, value in sorted(model.facts.items()):
		if name not in engine.facts:
			diffs.append(f"{name}: the vocabulary's {value}, the engine has none")
		elif engine.facts[name] != value:
			diffs.append(f"{name}: the vocabulary's {value}, the engine's {engine.facts[name]}")
	return Match(tables, diffs, engine.content_hash(), model.vocabulary_hash())


def merge(model, engine):
	"""The vocabulary grown by what the engine has and it does not, append-only: every existing row keeps its place
	(an absent key keeps its row: retired, never reused), the new keys go at each table's end in the engine's order, a
	new table starts with them. The facts are the engine's (the match compares them; a difference is not merged)."""
	tables = {name: list(keys) for name, keys in model.tables.items()}
	for name, keys in engine.tables.items():
		known = set(tables.setdefault(name, []))
		for key in keys:
			if key and key not in known:
				tables[name].append(key)
				known.add(key)
	return Content(tables, dict(model.facts or engine.facts))


def is_prefix(old, new):
	"""Whether vocabulary `new` only appended to `old` (every old row is the same key at the same row): records
	written under `old` stay valid under `new`."""
	return all(new.tables.get(name, [])[: len(keys)] == keys for name, keys in old.tables.items())


def load(path):
	return Content.from_json(json.loads(Path(path).read_text()))


def save(content, path):
	Path(path).write_text(json.dumps(content.to_json(), indent=1, sort_keys=True))


def _fnv(text):
	h = FNV_OFFSET
	for byte in text.encode("utf-8"):
		h = ((h ^ byte) * FNV_PRIME) & MASK
	return f"{h:016x}"
