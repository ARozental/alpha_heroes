"""What the ids in the model inputs mean: the engine's tables of creatures, spells, heroes, ...

The model only ever sees raw engine ids (u_creature = 13, h_spells[63] = 1, ...). This module maps
them back to names and base stats, for reading inputs and dumps, for the fixed reward values
(creature costs, artifact prices) and, later, for initialising embeddings from creature stats.

The tables come from the engine itself (scripts/export_vocab.sh writes battle_model/artifacts/vocab.json),
so they always match the ids the C++ side encodes. Not in git: names and stats are Heroes 3 game data.
"""
import json
from functools import lru_cache
from pathlib import Path

from .reward import resources_value

VOCAB_PATH = Path(__file__).resolve().parent / "artifacts" / "vocab.json"
TABLES = ("creatures", "spells", "heroes", "secondary_skills", "artifacts", "battlefields", "obstacles")


@lru_cache(maxsize=None)
def tables(path=VOCAB_PATH):
	"""{table name: {id: row}}; each row has at least id, key and name."""
	path = Path(path)
	if not path.exists():
		raise FileNotFoundError(f"{path} is missing: run scripts/export_vocab.sh")
	raw = json.loads(path.read_text())
	return {name: {row["id"]: row for row in raw[name]} for name in TABLES}


def name(table, id_):
	"""Display name for an id, e.g. name("creatures", 13) -> "Archangel"; "<none>" for -1."""
	if id_ < 0:
		return "<none>"
	row = tables()[table].get(id_)
	return row["name"] if row else f"<unknown {table} id {id_}>"


def creature(id_):
	return tables()["creatures"][id_]


def creature_value(id_):
	"""Gold value of one creature (gold + resources at base prices), for the reward; a war machine's is
	game_values.WAR_MACHINE_CREATURES'."""
	from game_values import WAR_MACHINE_CREATURES

	if id_ in WAR_MACHINE_CREATURES:
		return WAR_MACHINE_CREATURES[id_]
	return resources_value(creature(id_).get("cost", {}))


def artifact_price(id_):
	return tables()["artifacts"][id_]["price"]


