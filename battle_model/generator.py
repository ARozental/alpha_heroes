"""Random battles: sample(rng) draws a scenario (the format of battle_model/scenarios/*.json, which the map tool
builds) from the distribution in battle_model/scenarios/generator.toml, in the order of that file.

Everything the engine knows (creatures, heroes and their classes, spells and mage guilds, artifacts and their slots,
the experience per level) comes from the exported vocab (scripts/export_vocab.sh), so the draws are the engine's
own identifiers and numbers. What the map tool cannot build yet is redrawn or left out, and noted where it happens
(UNSUPPORTED): neutrals and creature banks (battle-only mode starts hero-vs-hero or siege battles only), boats,
special battlefields, a hero's mana below full and the player's gold.

Usage: .venv/bin/python -m battle_model.generator [--count N] [--seed S] [--out DIR]  (writes N scenario files)
"""
import argparse
import json
import math
import random
import tomllib
from functools import lru_cache
from pathlib import Path

from . import vocab
from .reward import resources_value

CONFIG_PATH = Path(__file__).resolve().parent / "scenarios" / "generator.toml"
UNSUPPORTED_BATTLE_TYPES = ("hero_vs_neutrals", "creature_bank")  # redrawn (UNSUPPORTED: needs an engine change)
UNSUPPORTED_TERRAINS = ("water",)  # redrawn (UNSUPPORTED: boats)
FORTIFICATIONS = ("none", "fort", "citadel", "castle")
WAR_MACHINES = {"ballista": "ballista", "first_aid_tent": "firstAidTent", "ammo_cart": "ammoCart"}
GUILD_SPELLS = (5, 4, 3, 2, 1)  # spells a mage guild offers at levels 1-5 (CGTownInstance::spellsAtLevel, no Library)
LEARNABLE = (2, 3, 4, 5)  # the highest spell level a hero can learn, by Wisdom mastery (none, basic, advanced, expert)
MAX_SECONDARY = 8
BACKPACK_START = 19  # ArtifactPosition::BACKPACK_START
RINGS, MISC = (7, 8), (9, 10, 11, 12, 18)  # ArtifactPosition: LEFT_RING, RIGHT_RING; MISC1-4 and MISC5


def short(key):
	"""'core:pikeman' -> 'pikeman' (the map tool looks names up in the core scope)."""
	return key.split(":", 1)[-1]


def load_config(path=CONFIG_PATH):
	return tomllib.loads(Path(path).read_text())


@lru_cache(maxsize=None)
def world(path=vocab.VOCAB_PATH):
	"""The engine's tables, grouped for sampling."""
	v = json.loads(Path(path).read_text())
	factions = [short(f["key"]) for f in v["factions"] if f.get("town") and not f.get("special")]
	creatures = [c for c in v["creatures"] if not c.get("special") and 1 <= c["level"] <= 7 and c.get("cost")]
	upgraded = {short(u) for c in creatures for u in c.get("upgrades", [])}
	by_key = {short(c["key"]): c for c in creatures}
	# A faction's creatures by level: (base, upgrade) pairs; neutrals (no faction town) on their own
	faction_creatures, neutral_creatures = {}, {}
	for c in creatures:
		key, faction = short(c["key"]), short(c["faction"])
		if key in upgraded:
			continue
		ups = [short(u) for u in c.get("upgrades", []) if short(u) in by_key]
		target = faction_creatures.setdefault(faction, {}) if faction in factions else neutral_creatures
		target.setdefault(c["level"], []).append((key, ups))
	classes = {short(c["key"]): c for c in v["hero_classes"]}
	heroes = {}  # (faction, magic) -> hero keys
	for h in v["heroes"]:
		if h.get("special") or short(h.get("class", "")) not in classes:
			continue
		hero_class = classes[short(h["class"])]
		heroes.setdefault((short(hero_class["faction"]), hero_class["magic"]), []).append((short(h["key"]), short(h["class"])))
	spells = [s for s in v["spells"] if not s["creature_ability"] and 1 <= s["level"] <= 5 and s.get("guild_chance")]
	artifacts = {}
	for a in v["artifacts"]:
		if a["class"] != "special" and not a.get("combined") and a.get("slots"):
			artifacts.setdefault(a["class"], []).append(a)
	return {
		"factions": factions,
		"faction_creatures": faction_creatures,
		"neutral_creatures": neutral_creatures,
		"creature": by_key,
		"classes": classes,
		"heroes": heroes,
		"spells": spells,
		"guild_levels": {short(f["key"]): f.get("mage_guild", 5) for f in v["factions"]},
		"skills": [short(s["key"]) for s in v["secondary_skills"]],
		"artifacts": artifacts,
		"experience": v["experience"],
	}


def choose(rng, group):
	"""A key of a group of { p = ... } choices, by p."""
	keys = [k for k, c in group.items() if isinstance(c, dict) and "p" in c]
	return rng.choices(keys, weights=[group[k]["p"] for k in keys])[0]


def between(rng, bounds):
	return rng.randint(bounds["min"], bounds["max"])


def creature_value(key, w):
	return resources_value(w["creature"][key]["cost"])


def draw_creature(rng, level, faction, origin, upgraded, w):
	"""A creature of `level` from the faction (or another, or neutrals), upgraded with chance `upgraded`; None if
	there is none of that level."""
	if origin == "neutral":
		pool = w["neutral_creatures"].get(level, [])
	else:
		if origin == "other_faction":
			faction = rng.choice([f for f in w["factions"] if f != faction])
		pool = w["faction_creatures"].get(faction, {}).get(level, [])
	if not pool:
		return None
	base, ups = rng.choice(pool)
	return rng.choice(ups) if ups and rng.random() < upgraded else base


def draw_army(rng, value, faction, phase, cfg, w):
	"""A hero's army worth about `value` gold: stacks of the faction's creatures (section 4.2)."""
	comp = cfg["composition"]
	stacks = int(choose(rng, comp["stacks"])[1:])
	weights = comp["level_weights"]["weights"][: between(rng, phase["top_tier"])]  # up to the army's highest level
	chosen = []
	for _ in range(stacks * 4):  # redraw when a level has no creature or the creature is already in the army
		if len(chosen) == stacks:
			break
		level = rng.choices(range(1, len(weights) + 1), weights=weights)[0]
		key = draw_creature(rng, level, faction, choose(rng, comp["stack_origin"]), phase["upgraded"], w)
		if key and key not in [k for k, _ in chosen]:
			chosen.append((key, weights[level - 1]))
	total = sum(weight for _, weight in chosen)
	return [[key, max(1, round(value * weight / total / creature_value(key, w)))] for key, weight in chosen]


def army_value(army, w):
	return sum(creature_value(key, w) * count for key, count in army)


def level_ups(rng, hero_class, level, free, cfg):
	"""Primary and secondary skills after levelling up from 1 to `level`: by the class's chances, as the engine levels
	a hero (CGHeroInstance::levelUpAutomatically), or (free) drawn uniformly."""
	primary = list(hero_class["primary_initial"])
	excluded = set(cfg["hero"]["excluded_secondary_skills"]["skills"])
	weights = {short(k): v for k, v in hero_class["secondary_weights"].items() if short(k) not in excluded}
	if free:
		weights = {k: 1 for k in weights}
	secondary = {}

	def learn():
		owned = [k for k, m in secondary.items() if m < 3]
		new = [k for k in weights if k not in secondary and weights[k] > 0]
		if new and (len(secondary) < MAX_SECONDARY) and (not owned or rng.random() < 0.5):
			skill = rng.choices(new, weights=[weights[k] for k in new])[0]
			secondary[skill] = 1
		elif owned:
			secondary[rng.choice(owned)] += 1

	learn()  # a new hero starts with a skill or two
	if rng.random() < 0.5:
		learn()
	for current in range(2, level + 1):
		chances = hero_class["primary_low" if current < 10 else "primary_high"]
		primary[rng.choices(range(4), weights=[1] * 4 if free else chances)[0]] += 1
		learn()
	return primary, [[k, m] for k, m in secondary.items()]


def draw_spells(rng, faction, phase, skills, cfg, w):
	"""The spells of the hero's town's mage guild (the phase's guild level, at most the town's), plus a few random
	ones; only those its Wisdom lets it learn."""
	wisdom = dict(skills).get("wisdom", 0)
	top = min(between(rng, phase["mage_guild"]), w["guild_levels"].get(faction, 5), LEARNABLE[wisdom])
	known = set()
	for level in range(1, top + 1):
		pool = [s for s in w["spells"] if s["level"] == level and s["guild_chance"].get("core:" + faction, 0) > 0]
		for _ in range(min(GUILD_SPELLS[level - 1], len(pool))):
			pick = rng.choices(pool, weights=[s["guild_chance"]["core:" + faction] for s in pool])[0]
			known.add(short(pick["key"]))
			pool.remove(pick)
	learnable = [s for s in w["spells"] if s["level"] <= LEARNABLE[wisdom]]
	for _ in range(between(rng, cfg["hero"]["spells"]["extra_random"])):
		known.add(short(rng.choice(learnable)["key"]))
	return sorted(known)


def draw_artifacts(rng, count, cfg, w):
	"""[[ArtifactPosition, key], ...]: `count` artifacts by class, each in a free slot it fits (redrawn if none)."""
	used, result = set(), []
	for _ in range(count * 4):
		if len(result) == count:
			break
		artifact = rng.choice(w["artifacts"][choose(rng, cfg["hero"]["artifact_class"])])
		free = [s for s in artifact["slots"] if s < BACKPACK_START and s not in used]
		if free and short(artifact["key"]) not in [k for _, k in result]:
			used.add(free[0])
			result.append([free[0], short(artifact["key"])])
	return result


def draw_hero(rng, army_faction, phase, cfg, w):
	"""A hero (section 4.3): its faction and class, level, skills, spells, artifacts, war machines."""
	h = cfg["hero"]
	faction = army_faction if choose(rng, h["faction"]) == "army_faction" else rng.choice(w["factions"])
	magic = choose(rng, h["class"]) == "magic"
	hero, class_key = rng.choice(w["heroes"][(faction, magic)])
	hero_class = w["classes"][class_key]
	level = between(rng, phase["hero_level"])
	primary, secondary = level_ups(rng, hero_class, level, choose(rng, h["growth"]) == "random", cfg)
	experience = w["experience"]
	progress = rng.random()  # its way to the next level
	exp = experience[level - 1] + progress * (experience[level] - experience[level - 1])
	spellbook = rng.random() < h["spellbook"]["magic" if magic else "might"]
	machines = cfg["composition"]["war_machines"]
	return {
		"hero": hero,
		"primary_skills": primary,
		"secondary_skills": secondary,
		"experience": int(exp),
		"spellbook": spellbook,
		"spells": draw_spells(rng, faction, phase, secondary, cfg, w) if spellbook else [],
		"artifacts": draw_artifacts(rng, between(rng, phase["artifacts"]), cfg, w),
		"war_machines": [WAR_MACHINES[m] for m, c in machines.items() if rng.random() < c["p"]],
	}


def sample(rng, cfg=None, w=None):
	"""One random battle: a scenario dict, with "_about" saying what was drawn."""
	cfg = cfg or load_config()
	w = w or world()
	terrain = choose(rng, cfg["terrain"])
	while terrain in UNSUPPORTED_TERRAINS:
		terrain = choose(rng, cfg["terrain"])
	battle_type = choose(rng, cfg["battle_type"])
	while battle_type in UNSUPPORTED_BATTLE_TYPES:
		battle_type = choose(rng, cfg["battle_type"])
	phase_name = rng.choices([k for k in cfg["phase"]], weights=[p["p"] for p in cfg["phase"].values()])[0]
	phase = cfg["phase"][phase_name]

	strong = phase["army_value_median"] * math.exp(rng.gauss(0, cfg["army"]["value_sigma"]))
	ratio_name = choose(rng, cfg["ratio"])
	ratio = 10 ** rng.uniform(cfg["ratio"][ratio_name]["log10"]["min"], cfg["ratio"][ratio_name]["log10"]["max"])
	odds = cfg["ratio"]["stronger_side"][battle_type]
	stronger = 0 if rng.random() < odds["attacker"] / (odds["attacker"] + odds["defender"]) else 1
	values = [strong, strong / ratio] if stronger == 0 else [strong / ratio, strong]

	sides, factions = [], []
	for side in (0, 1):
		faction = choose(rng, cfg["composition"]["main_faction"])
		army = draw_army(rng, values[side], faction, phase, cfg, w)
		sides.append({"faction": faction, **draw_hero(rng, faction, phase, cfg, w), "army": army})
		factions.append(faction)
	scenario = {"name": "generated", "terrain": terrain, "seed": rng.randrange(1, 2**31), "sides": sides}
	about = f"{battle_type}, {phase_name}, {terrain}, armies {army_value(sides[0]['army'], w):.0f} vs {army_value(sides[1]['army'], w):.0f} gold"
	if battle_type == "siege":
		s = cfg["siege"]
		town = factions[1] if rng.random() < s["town_is_defender_faction"] else rng.choice([f for f in w["factions"] if f != factions[1]])
		scenario["town"] = {"faction": town, "fortification": choose(rng, s["fortifications"])}
		if choose(rng, s["defender"]) == "no_hero":
			sides[1] = {"faction": sides[1]["faction"], "army": sides[1]["army"]}  # the town's garrison defends
		about += f", {scenario['town']['fortification']} {town}" + ("" if "hero" in sides[1] else ", garrison only")
	scenario["_about"] = about
	return scenario


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--count", type=int, default=1)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--out", default=".")
	args = parser.parse_args()
	rng = random.Random(args.seed)
	out = Path(args.out)
	out.mkdir(parents=True, exist_ok=True)
	for i in range(args.count):
		scenario = sample(rng)
		path = out / f"generated_{args.seed}_{i}.json"
		path.write_text(json.dumps(scenario, indent="\t"))
		print(path, scenario["_about"])


if __name__ == "__main__":
	main()
