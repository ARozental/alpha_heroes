"""Town buildings' prerequisites (e5 for the user, 2026-11-12: the build features' "unlocks"), from VCMI's own configs:
config/buildingsLibrary.json (every faction's buildings: "upgrades" and "requires") and config/factions/<faction>.json
(each faction's own, which add to or replace them). A building needs the one it upgrades and every building of its
"requires" ("allOf" all of them; "anyOf" its first). Written to campaign_model/priors/building_requires.json as
{"requires": {faction: {building id: [required ids]}}, "hordes": {faction: {building id: level index}}} under vocab's
ids (vocab.buildings' keys), read by vocab.building_requires and vocab.horde_levels. A horde building's extra growth is
the engine's creature "horde" number, read from the H3 data files (CRTRAITS), which these configs do not carry: the
features count it as half its level's base growth (most of H3's are 2 to 3 on growths of 4 to 7).
	python -m campaign_model.building_requires [vcmi checkout]   (from the repository: rewrites the table)"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLE = Path(__file__).resolve().parent / "priors" / "building_requires.json"
FACTIONS = ("castle", "rampart", "tower", "inferno", "necropolis", "dungeon", "stronghold", "fortress", "conflux")  # (VCMI's faction ids 0..8)


def _lenient(path):
	"""A VCMI config as JSON: its // comments and trailing commas taken out."""
	text = re.sub(r"(?m)//[^\n]*", "", path.read_text())
	return json.loads(re.sub(r",(\s*[}\]])", r"\1", text))


def _names(requires):
	"""The building names a "requires" list asks for: ["allOf", [a], [b]] all, ["anyOf", [a], [b]] the first, [a, b] all."""
	if not requires:
		return []
	head = requires[0]
	if head == "allOf":
		return [n for part in requires[1:] for n in _names(part)]
	if head == "anyOf":
		return _names(requires[1]) if len(requires) > 1 else []
	if head == "noneOf":
		return []
	return [n for item in requires for n in (_names(item) if isinstance(item, list) else [item])]


def table(vcmi=ROOT / "vcmi"):
	from . import vocab

	library = _lenient(vcmi / "config" / "buildingsLibrary.json")
	out = {}
	for faction, name in enumerate(FACTIONS):
		own = _lenient(vcmi / "config" / "factions" / f"{name}.json")[name]["town"]["buildings"]
		ids = {row["key"].split(":", 1)[-1]: building for (f, building), row in vocab.buildings().items() if f == faction}
		rows = {}
		for key, building in ids.items():
			spec = {**library.get(key, {}), **(own.get(key) or {})}
			needs = [spec["upgrades"]] if spec.get("upgrades") else []
			needs += _names(spec.get("requires", []))
			rows[str(building)] = sorted({ids[n] for n in needs if n in ids and ids[n] != building})
		out[str(faction)] = rows
	return out


def hordes(vcmi=ROOT / "vcmi"):
	"""{faction: {building id: the level index its extra growth goes to}}: horde1 and its upgrade at the town's "horde"[0],
	horde2 and its upgrade at "horde"[1] (-1: none)."""
	from . import vocab

	out = {}
	for faction, name in enumerate(FACTIONS):
		levels = _lenient(vcmi / "config" / "factions" / f"{name}.json")[name]["town"].get("horde", [-1, -1])
		ids = {row["key"].split(":", 1)[-1]: building for (f, building), row in vocab.buildings().items() if f == faction}
		rows = {}
		for key, level in (("horde1", levels[0]), ("horde1Upgr", levels[0]), ("horde2", levels[1] if len(levels) > 1 else -1), ("horde2Upgr", levels[1] if len(levels) > 1 else -1)):
			if key in ids and level is not None and level >= 0:
				rows[str(ids[key])] = int(level)
		out[str(faction)] = rows
	return out


if __name__ == "__main__":
	import sys

	vcmi = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "vcmi"
	TABLE.write_text(json.dumps({"requires": table(vcmi), "hordes": hordes(vcmi)}, indent=1, sort_keys=True) + "\n")
	print(f"wrote {TABLE}")
