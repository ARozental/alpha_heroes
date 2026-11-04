"""Named input and output tensors of the campaign model (the contract with agents/AlphaMapAI).

Inputs are raw values read from the engine (counts, ids, coordinates, movement points); every transform happens
inside the model, as in the battle model. Masks are engine legality, applied to the outputs only: every output is a
masked log-probability table, -inf where the engine says illegal. Each tensor has a leading batch dimension, which the
C++ side always sets to 1 (shapes below exclude it).

A turn decision samples type_logp, then the type's table (HEADS); a CAST of a targeted spell then also samples its
hero's cast_cell_logp. An answer to an engine question samples option_logp.
"""
import hashlib
import json
from dataclasses import asdict

import numpy as np

from battle_model.schema import TensorSpec

from . import constants as c

H, HO, T, TO, A = c.N_HEROES, c.OWN_HEROES, c.N_TOWNS, c.OWN_TOWNS, c.ARMY_SLOTS
W, HD, O, R = c.WINDOW, c.N_HOLDERS, c.N_OPTIONS, c.N_RESOURCES
MO, G, L, TG = c.MAP_OBJECTS, c.MAP_GRID, c.MAP_LEVELS, c.TRAVEL_GRID


def _i(name, shape, doc, fill=0):
	return TensorSpec(name, "int64", tuple(shape), fill, doc)


def _m(name, shape, doc):
	return TensorSpec(name, "bool", tuple(shape), 0, doc)


INPUTS = (
	# Global
	_i("g_kind", (), "DecisionKind"),
	_i("g_day", (), "day of the game, 1 = the first (VCMI's Date::DAY)"),
	# Our own previous turn decision today, as the model chose it (the engine only echoes it back): where a chain of
	# actions stands is the model's to read from it (2026-10-28, the user: no behaviour decided outside the model)
	_i("g_last_type", (), "ActionType of our previous turn decision today, -1 none yet", fill=-1),
	_i("g_last_index", (), "its entry in its type's table, -1 none", fill=-1),
	_i("g_last_changed", (), "1 if it changed the game, 0 if it did nothing, -1 none", fill=-1),
	_i("g_resources", (R,), "our resources"),
	_i("g_income", (R,), "our daily income (the kingdom overview's)"),
	_i("g_object_income", (R,), "the part of it from our mines and other objects (not towns or heroes: the ledger's income)"),
	_i("g_owned_creature", (c.OWNED_STACKS,), "creatures in the garrisons of our mines and other owned objects (not towns or heroes), -1 none", fill=-1),
	_i("g_owned_count", (c.OWNED_STACKS,), "their counts"),
	_i("g_dwelling_creature", (c.OWNED_STACKS,), "creatures for hire in our dwellings outside towns, -1 none", fill=-1),
	_i("g_dwelling_count", (c.OWNED_STACKS,), "how many"),
	_i("g_map", (3,), "map width, height, levels"),
	_i("g_players", (), "players still in the game, us included"),
	_i("g_decisions_today", (), "our turn decisions so far today"),
	_i("g_explored", (), "tiles of the map we ever revealed"),
	_i("g_markets", (), "our marketplaces (trade rates improve with each)"),
	_i("g_tavern", (c.N_TAVERN,), "hero types for hire, -1 none", fill=-1),
	_i("g_tavern_level", (c.N_TAVERN,), "their levels"),
	# What a hire brings (the user, 2026-10-29: the model saw only the type and level, not the army a hero for hire comes
	# with, the week's two heroes a full one, the others a lone creature): the experience, army and worn artifacts
	_i("g_tavern_exp", (c.N_TAVERN,), "their experience"),
	_i("g_tavern_primary", (c.N_TAVERN, c.N_PRIMARY), "their primary skills (without items)"),
	_i("g_tavern_skills", (c.N_TAVERN, c.N_SECONDARY_SKILLS), "their secondary skills' levels (0 none, 1-3)"),
	_i("g_tavern_army_creature", (c.N_TAVERN, A), "the creatures each comes with, -1 none", fill=-1),
	_i("g_tavern_army_count", (c.N_TAVERN, A), "how many"),
	_i("g_tavern_artifacts", (c.N_TAVERN, c.N_WORN_SLOTS), "the artifact worn in each slot (a spellbook, war machines among them), -1 none", fill=-1),
	_m("g_tavern_spells", (c.N_TAVERN, c.N_SPELLS), "the spells in their spellbooks"),
	# The question asked (dialog kinds)
	_i("d_hero", (), "our hero the question is about (index into the hero tokens), -1 none", fill=-1),
	_i("d_object", (), "object type at that hero's tile or target, -1 none", fill=-1),
	_i("d_subtype", (), "its subtype"),
	_i("d_creature", (), "its guards' (or joining creatures') creature, -1 none", fill=-1),
	_i("d_count", (), "how many: VCMI's quantity id (what a human sees) unless d_detailed"),
	_m("d_detailed", (), "d_count is exact (our hero's Visions reach the monsters)"),
	_i("d_number", (), "the first number in the question's text (e.g. the gold asked), 0 none"),
	_i("d_sound", (), "the question's sound id"),
	# Heroes: ours first (OWN_HEROES), then others we see
	_m("h_present", (H,), "a hero in this slot"),
	_i("h_owner", (H,), "OWNER_SELF, or OWNER_SELF + 1 + k for the k-th other player"),
	_i("h_type", (H,), "hero type id", fill=-1),
	_i("h_class", (H,), "hero class id", fill=-1),
	_i("h_pos", (H, 3), "x, y, level of its tile (the visitable position)"),
	_i("h_level", (H,), "level"),
	_i("h_exp", (H,), "experience"),
	_i("h_primary", (H, c.N_PRIMARY), "attack, defense, spell power, knowledge"),
	_i("h_primary_base", (HO, c.N_PRIMARY), "our heroes' primary skills without items or temporary bonuses (empire.py prices them)"),
	_i("h_mana", (H,), "mana"),
	_i("h_mana_max", (H,), "mana limit"),
	_i("h_spell_cost", (H,), "mana cost of the dearest combat spell it can cast (ours), 0 none: what its mana is worth"),
	_i("h_move", (H,), "movement points left"),
	_i("h_move_max", (H,), "movement points a full day gives"),
	_i("h_army_creature", (H, A), "creature id of each army slot, -1 empty", fill=-1),
	_i("h_army_count", (H, A), "its count (others' heroes: VCMI's quantity id unless h_detailed)"),
	_i("h_artifacts", (H, c.N_WORN_SLOTS), "artifact worn in each slot, -1 none", fill=-1),
	_i("h_backpack", (H, c.N_BACKPACK), "first backpack artifacts, -1 none", fill=-1),
	_i("h_skills", (H, c.N_SECONDARY_SKILLS), "secondary skill levels, 0 none .. 3 expert"),
	_m("h_spells", (H, c.N_SPELLS), "spells in its spellbook (ours only)"),
	# Luck and morale (Φ v3, the user 2026-10-30: a Temple's morale must not be skipped): ours the sum of the hero's
	# bonuses, uncapped (the battle caps them at +-3), with the part that lasts until the next battle apart
	_i("h_luck", (H,), "luck: ours the sum of its luck bonuses (uncapped), others' VCMI's detailed value, else 0"),
	_i("h_morale", (H,), "morale: ours the sum of its morale bonuses (uncapped), others' VCMI's detailed value, else 0"),
	_i("h_battle_luck", (HO,), "ours: the part of h_luck that lasts until the next battle (a Mermaid's, a Fountain of Fortune's)"),
	_i("h_battle_morale", (HO,), "ours: the part of h_morale that lasts until the next battle (a Temple's, a Buoy's, a Warrior's Tomb's -3)"),
	_m("h_boat", (H,), "in a boat"),
	_i("h_town", (H,), "town token it visits or garrisons, -1 none", fill=-1),
	_m("h_garrisoned", (H,), "the garrison hero of that town"),
	_m("h_detailed", (H,), "its army counts and skills are exact (ours; others when VCMI says detailed)"),
	# Towns: ours first (OWN_TOWNS), then others we see
	_m("t_present", (T,), "a town in this slot"),
	_i("t_owner", (T,), "OWNER_NONE, OWNER_SELF, or another player"),
	_i("t_faction", (T,), "faction id", fill=-1),
	_i("t_pos", (T, 3), "x, y, level of its entrance"),
	_m("t_built", (T, c.N_BUILDINGS), "buildings built (ours; others: none)"),
	_i("t_fort", (T,), "fort level 0..3"),
	_i("t_built_today", (T,), "buildings built this turn"),
	_i("t_garrison_creature", (T, A), "garrison creature ids, -1 empty", fill=-1),
	_i("t_garrison_count", (T, A), "their counts (quantity ids unless t_detailed)"),
	_m("t_detailed", (T,), "its garrison counts are exact (ours; others when VCMI says detailed)"),
	_i("t_available", (T, c.N_LEVELS), "creatures for hire at each level"),
	_i("t_available_creature", (T, c.N_LEVELS), "the best creature for hire at each level, -1 none", fill=-1),
	_i("t_growth", (T, c.N_LEVELS), "weekly growth of each level"),
	_i("t_visiting", (T,), "hero token visiting it, -1 none", fill=-1),
	_i("t_garrison_hero", (T,), "hero token in its garrison, -1 none", fill=-1),
	# The other army of a garrison dialog (holder OTHER_ARMY)
	_i("x_creature", (A,), "creature ids, -1 empty", fill=-1),
	_i("x_count", (A,), "counts"),
	# The options of an engine question
	_i("o_kind", (O,), "OptionKind"),
	_i("o_type", (O,), "see OptionKind", fill=-1),
	_i("o_subtype", (O,), "see OptionKind"),
	_i("o_amount", (O,), "see OptionKind"),
	_i("o_pos", (O, 3), "x, y, level (OBJECT)"),
	# The map around each of our heroes (WINDOW x WINDOW tiles, the hero in the middle)
	# The whole map, tile by tile (schema 9: the map CNN's input, every hero reading the same tiles): as a window's, but
	# for no hero in particular (the top object other than a hero: heroes and towns come in as their tokens)
	_i("tile_terrain", (L, c.MAP_TILES, c.MAP_TILES), "terrain id, -1 off the map or not revealed", fill=-1),
	_i("tile_road", (L, c.MAP_TILES, c.MAP_TILES), "road id, 0 none"),
	_i("tile_flags", (L, c.MAP_TILES, c.MAP_TILES), "TILE_* bits (not TILE_VISITED_BY_HERO)"),
	_i("tile_object", (L, c.MAP_TILES, c.MAP_TILES), "type of the top object other than a hero (visitable first), -1 none", fill=-1),
	_i("tile_subtype", (L, c.MAP_TILES, c.MAP_TILES), "its subtype"),
	_i("tile_owner", (L, c.MAP_TILES, c.MAP_TILES), "its owner, relative to us"),
	_i("tile_creature", (L, c.MAP_TILES, c.MAP_TILES), "a wandering monster's creature, -1 none", fill=-1),
	_i("tile_count", (L, c.MAP_TILES, c.MAP_TILES), "its count as a human sees it (TILE_EXACT_COUNT: exact, a hero's Visions)"),
	_i("w_terrain", (HO, W, W), "terrain id, -1 off the map or not revealed", fill=-1),
	_i("w_road", (HO, W, W), "road id, 0 none"),
	_i("w_flags", (HO, W, W), "TILE_* bits"),
	_i("w_object", (HO, W, W), "type of the top object (visitable first), -1 none", fill=-1),
	_i("w_subtype", (HO, W, W), "its subtype (resource, artifact, creature, mine ... id)"),
	_i("w_owner", (HO, W, W), "its owner, relative to us"),
	_i("w_creature", (HO, W, W), "a wandering monster's creature, -1 none", fill=-1),
	_i("w_count", (HO, W, W), "its count as a human sees it: VCMI's quantity id (few, several, pack ...), the exact count when TILE_EXACT_COUNT (Visions)"),
	_i("w_move_left", (HO, W, W), "movement points the hero has left on reaching it today, -1 not today", fill=-1),
	_i("w_turns", (HO, W, W), "days the hero needs to reach it (0 today), -1 never", fill=-1),
	# The whole map: the objects we see (on revealed tiles), and per-block tile counts
	_i("m_object", (MO,), "object type, -1 none", fill=-1),
	_i("m_subtype", (MO,), "its subtype"),
	_i("m_owner", (MO,), "its owner, relative to us"),
	_i("m_pos", (MO, 3), "x, y, level of its entrance (the visitable tile)"),
	_i("m_flags", (MO,), "OBJECT_* bits"),
	_i("m_visitors", (MO,), "bit h: our hero token h visited it (CRewardableObject::wasVisited(hero))"),
	_i("m_creature", (MO,), "a wandering monster's creature, -1 none", fill=-1),
	_i("m_count", (MO,), "its count as a human sees it: VCMI's quantity id, the exact count when OBJECT_EXACT_COUNT"),
	_i("g_tiles", (L, G, G, c.N_GRID_COUNTS), "per block of the map (MAP_GRID): how many of its tiles are each GridCount"),
	# Legal actions (engine rules); a type is legal when its table has a legal entry, END by the kind
	_m("move_mask", (HO, W, W), "MOVE: hero, tile"),
	_m("build_mask", (TO, c.N_BUILDINGS), "BUILD: town, building"),
	_m("build_short", (TO, c.N_BUILDINGS), "buildings our towns could build but for resources (VCMI's EBuildingState::NO_RESOURCES)"),
	_m("recruit_mask", (TO, c.N_LEVELS, c.N_RECRUIT_AMOUNTS, c.N_RECRUIT_TARGETS), "RECRUIT: town, level, RECRUIT_FRACTIONS, into (N_RECRUIT_TARGETS)"),
	_m("hire_mask", (TO, c.N_TAVERN), "HIRE: town, tavern slot"),
	_m("army_mask", (HD, A, HD, c.N_ARMY_AMOUNTS), "ARMY: from holder, slot, to holder, ArmyAmount"),
	_m("upgrade_mask", (HD, A), "UPGRADE: holder, slot (to its best upgrade)"),
	_m("trade_mask", (R, R, c.N_TRADE_AMOUNTS), "TRADE: sell, buy, TRADE_FRACTIONS"),
	_m("cast_mask", (HO, c.N_ADVENTURE_SPELLS), "CAST: hero, spell"),
	_m("cast_cell_mask", (HO, W, W), "the tile of a targeted spell (Dimension Door)"),
	_m("buy_mask", (HO, c.N_BUY_ITEMS), "BUY: hero, BUY_ARTIFACTS"),
	_m("dismiss_mask", (HO,), "DISMISS: hero"),
	_i("travel_days", (HO, L, TG, TG), "days x 100 the hero needs to reach each TRAVEL area's tile (the pathfinder's: whole days and the part of the last day's movement), -1 where TRAVEL is not legal", fill=-1),
	_m("travel_mask", (HO, L, TG, TG), "TRAVEL: hero, level, area (TRAVEL_GRID): an area with a tile the hero reaches in a day or more"),
	_m("option_mask", (O,), "the options of a question"),
	# (last: a record of before it, schema 9, is this one without it: records.decode_wire gives it zeros)
	_i("mem", (c.MEMORY_SLOTS, c.MEMORY_WIDTH), "the model's memory as this player's previous decision left it (its memory output) x MEMORY_SCALE; 0 at the start of a game"),
	# Added after mem (2026-11-04: a record from before carries none of them, records.decode_wire gives them zeros: no
	# guards, no battle, no armed object). A question's guards and what each option does with them (the user: a battle
	# decided as a MOVE's fight is; a monster's offer to join easy to take): a guarded bank's, dwelling's or mine's
	# "fight?" (YES starts the battle), a wandering monster's offer to join (YES takes them, for d_number gold if any; NO
	# fights them) or its flight (YES pursues: the battle)
	_m("o_battle", (O,), "this option starts a battle against the question's guards (d_guard_*)"),
	_m("o_join", (O,), "this option takes the question's creatures (d_guard_*) into the hero's army, for its o_amount gold (0 free)"),
	_i("d_guard_creature", (A,), "the question's guards or joining creatures: each stack's creature (0 with no count)"),
	_i("d_guard_count", (A,), "their counts: exact when d_guard_detailed, else VCMI's quantity id; 0 none"),
	_m("d_guard_detailed", (), "d_guard_count is exact (the dialog shows a bank's guards so)"),
	# Armed objects we see that fight a hero the moment it steps on them, their armies as a human sees them (VCMI's
	# right-click; CGGarrison::onHeroVisit starts the battle): garrisons not ours, the nearest first
	_m("a_present", (c.ARMED_OBJECTS,), "an armed object in this slot"),
	_i("a_pos", (c.ARMED_OBJECTS, 3), "x, y, level of its entrance"),
	_i("a_creature", (c.ARMED_OBJECTS, A), "its army's creatures (0 with no count)"),
	_i("a_count", (c.ARMED_OBJECTS, A), "their counts: VCMI's quantity id unless a_detailed, 0 none"),
	_m("a_detailed", (c.ARMED_OBJECTS,), "a_count is exact"),
)


def _o(name, shape, doc):
	return TensorSpec(name, "float32", tuple(shape), 0, doc)


OUTPUTS = (
	_o("type_logp", (c.N_ACTION_TYPES,), "log p(action type)"),
	_o("move_logp", (HO, W, W), "log p(hero, tile | MOVE)"),
	_o("build_logp", (TO, c.N_BUILDINGS), "log p(town, building | BUILD)"),
	_o("recruit_logp", (TO, c.N_LEVELS, c.N_RECRUIT_AMOUNTS, c.N_RECRUIT_TARGETS), "log p(town, level, amount, into | RECRUIT)"),
	_o("hire_logp", (TO, c.N_TAVERN), "log p(town, slot | HIRE)"),
	_o("army_logp", (HD, A, HD, c.N_ARMY_AMOUNTS), "log p(from, slot, to, amount | ARMY)"),
	_o("upgrade_logp", (HD, A), "log p(holder, slot | UPGRADE)"),
	_o("trade_logp", (R, R, c.N_TRADE_AMOUNTS), "log p(sell, buy, amount | TRADE)"),
	_o("cast_logp", (HO, c.N_ADVENTURE_SPELLS), "log p(hero, spell | CAST)"),
	_o("cast_cell_logp", (HO, W, W), "log p(tile | CAST, hero, targeted spell), each hero's own table"),
	_o("buy_logp", (HO, c.N_BUY_ITEMS), "log p(hero, item | BUY)"),
	_o("dismiss_logp", (HO,), "log p(hero | DISMISS)"),
	_o("travel_logp", (HO, L, TG, TG), "log p(hero, level, area | TRAVEL)"),
	_o("option_logp", (O,), "log p(option) of a question"),
	_o("value", (c.N_VALUE_OFFSETS,), "predicted share of all empires' value VALUE_OFFSETS days after this day"),
	_o("memory", (c.MEMORY_SLOTS, c.MEMORY_WIDTH), "the memory after this decision, -1..1: AlphaMapAI passes it back as the next decision's mem"),
)

# Each action type's table: (output, mask); END has none
HEADS = {
	c.ActionType.MOVE: ("move_logp", "move_mask"),
	c.ActionType.BUILD: ("build_logp", "build_mask"),
	c.ActionType.RECRUIT: ("recruit_logp", "recruit_mask"),
	c.ActionType.HIRE: ("hire_logp", "hire_mask"),
	c.ActionType.ARMY: ("army_logp", "army_mask"),
	c.ActionType.UPGRADE: ("upgrade_logp", "upgrade_mask"),
	c.ActionType.TRADE: ("trade_logp", "trade_mask"),
	c.ActionType.CAST: ("cast_logp", "cast_mask"),
	c.ActionType.BUY: ("buy_logp", "buy_mask"),
	c.ActionType.DISMISS: ("dismiss_logp", "dismiss_mask"),
	c.ActionType.TRAVEL: ("travel_logp", "travel_mask"),
}
SPECS = {s.name: s for s in INPUTS + OUTPUTS}


def contract_dict():
	return {
		"version": c.SCHEMA_VERSION,
		"inputs": [asdict(s) for s in INPUTS],
		"outputs": [asdict(s) for s in OUTPUTS],
		"action_types": [a.name for a in c.ActionType],
		"kinds": [k.name for k in c.DecisionKind],
		"options": [o.name for o in c.OptionKind],
		"value_offsets": list(c.VALUE_OFFSETS),
	}


def contract_hash():
	blob = json.dumps(contract_dict(), sort_keys=True).encode()
	return hashlib.sha256(blob).hexdigest()[:16]


def np_dtype(dtype):
	return {"int64": np.int64, "bool": np.bool_, "float32": np.float32}[dtype]


def empty_inputs(batch=1):
	"""All inputs at their fill values: nothing present, nothing legal."""
	return {s.name: np.full((batch,) + s.shape, s.fill, dtype=np_dtype(s.dtype)) for s in INPUTS}


def random_inputs(batch=1, seed=0):
	"""Random but consistent inputs for tests: a few heroes and towns, some legal moves and builds, or a question."""
	rng = np.random.default_rng(seed)
	x = empty_inputs(batch)
	for b in range(batch):
		kind = int(rng.choice([c.DecisionKind.TURN, c.DecisionKind.TURN, c.DecisionKind.EXCHANGE, c.DecisionKind.YES_NO, c.DecisionKind.SELECT]))
		x["g_kind"][b] = kind
		x["g_day"][b] = rng.integers(1, 60)
		x["g_resources"][b] = rng.integers(0, 30, R) * np.array([1, 1, 1, 1, 1, 1, 1000])
		x["g_income"][b] = [0, 0, 0, 0, 0, 0, 1000]
		x["g_map"][b] = [36, 36, 2]
		x["g_players"][b] = 2
		heroes = rng.integers(1, 4)
		for h in range(heroes):
			x["h_present"][b, h] = True
			x["h_owner"][b, h] = c.OWNER_SELF
			x["h_type"][b, h] = rng.integers(0, 150)
			x["h_class"][b, h] = rng.integers(0, 18)
			x["h_pos"][b, h] = [rng.integers(0, 36), rng.integers(0, 36), 0]
			x["h_level"][b, h] = rng.integers(1, 10)
			x["h_primary"][b, h] = rng.integers(0, 6, c.N_PRIMARY)
			x["h_move"][b, h] = 1500
			x["h_move_max"][b, h] = 1500
			x["h_army_creature"][b, h, 0] = rng.integers(0, 140)
			x["h_army_count"][b, h, 0] = rng.integers(1, 50)
			x["h_detailed"][b, h] = True
			x["w_terrain"][b, h] = rng.integers(0, 10, (W, W))
			x["w_flags"][b, h] = c.TILE_ON_MAP | c.TILE_REVEALED
			x["w_turns"][b, h] = rng.integers(0, 3, (W, W))
			if kind == c.DecisionKind.TURN:
				x["move_mask"][b, h] = rng.random((W, W)) < 0.2
			for _ in range(3):  # wandering monsters, counted as a human sees them (one seen exactly: Visions)
				row, column = rng.integers(0, W, 2)
				x["w_object"][b, h, row, column] = 54
				x["w_creature"][b, h, row, column] = rng.integers(0, 140)
				x["w_count"][b, h, row, column] = rng.integers(1, 10)
			x["w_flags"][b, h, row, column] |= c.TILE_EXACT_COUNT
			x["w_count"][b, h, row, column] = rng.integers(1, 300)
		x["t_present"][b, 0] = True
		x["t_owner"][b, 0] = c.OWNER_SELF
		x["t_detailed"][b, 0] = True
		x["t_garrison_creature"][b, 0, 0] = rng.integers(0, 140)
		x["t_garrison_count"][b, 0, 0] = rng.integers(1, 40)
		other = c.OWN_TOWNS  # someone else's town, its garrison in quantity ids
		x["t_present"][b, other] = True
		x["t_owner"][b, other] = c.OWNER_SELF + 1
		x["t_pos"][b, other] = [rng.integers(0, 36), rng.integers(0, 36), 0]
		x["t_garrison_creature"][b, other, 0] = rng.integers(0, 140)
		x["t_garrison_count"][b, other, 0] = rng.integers(1, 10)
		x["t_faction"][b, 0] = rng.integers(0, 9)
		x["t_built"][b, 0, [5, 7, 10, 30]] = True
		x["t_available"][b, 0, 0] = 10
		x["t_available_creature"][b, 0, 0] = 0
		if kind == c.DecisionKind.TURN:
			x["build_mask"][b, 0, rng.integers(0, c.N_BUILDINGS, 3)] = True
			x["recruit_mask"][b, 0, 0] = True
			x["upgrade_mask"][b, 0, 0] = rng.random() < 0.5
			x["trade_mask"][b, 6, 0, :2] = rng.random() < 0.5
		if kind == c.DecisionKind.EXCHANGE:
			x["army_mask"][b, 0, 0, 1] = True
			x["army_mask"][b, 0, 0, 1, c.ArmyAmount.ONE] = True
		if kind in c.DIALOG_KINDS:
			n = 2 if kind == c.DecisionKind.YES_NO else int(rng.integers(1, c.N_OPTIONS + 1))
			x["option_mask"][b, :n] = True
			x["o_kind"][b, :n] = c.OptionKind.YES if kind == c.DecisionKind.YES_NO else c.OptionKind.COMPONENT
			if kind == c.DecisionKind.YES_NO:
				x["o_kind"][b, 1] = c.OptionKind.NO
			x["o_amount"][b, :n] = rng.integers(0, 2000, n)
		# The whole map (36 x 36, two levels: one tile a block): some revealed, objects on it (a gold mine, a chest,
		# monsters, the towns and heroes above)
		x["g_object_income"][b] = [0, 0, 0, 0, 0, 0, 1000]
		x["g_tiles"][b, :, :, :, c.GridCount.ON_MAP] = 1
		revealed = rng.random((c.MAP_LEVELS, c.MAP_GRID, c.MAP_GRID)) < 0.4
		x["g_tiles"][b, :, :, :, c.GridCount.REVEALED] = revealed
		x["g_tiles"][b, :, :, :, c.GridCount.BLOCKED] = revealed & (rng.random(revealed.shape) < 0.3)
		objects = [(98, 0, c.OWNER_SELF), (98, 1, c.OWNER_SELF + 1), (53, 6, c.OWNER_NONE), (101, 0, c.OWNER_NONE), (54, 10, c.OWNER_NONE), (54, 20, c.OWNER_NONE), (112, 0, c.OWNER_NONE)]
		seen = rng.random((L, c.MAP_TILES, c.MAP_TILES)) < 0.5
		x["tile_flags"][b] = c.TILE_ON_MAP | np.where(seen, c.TILE_REVEALED, 0) | np.where(seen & (rng.random(seen.shape) < 0.2), c.TILE_BLOCKED, 0)
		x["tile_terrain"][b] = np.where(seen, rng.integers(0, 10, seen.shape), -1)
		for i, (obj, subtype, owner) in enumerate(objects[: int(rng.integers(2, len(objects) + 1))]):
			x["m_object"][b, i], x["m_subtype"][b, i], x["m_owner"][b, i] = obj, subtype, owner
			x["m_pos"][b, i] = [rng.integers(0, 36), rng.integers(0, 36), rng.integers(0, 2)]
			tx, ty, tz = x["m_pos"][b, i]
			x["tile_object"][b, tz, ty, tx], x["tile_subtype"][b, tz, ty, tx], x["tile_owner"][b, tz, ty, tx] = obj, subtype, owner
			x["tile_flags"][b, tz, ty, tx] |= c.TILE_REVEALED | c.TILE_VISITABLE
			if obj == 54:
				x["m_creature"][b, i], x["m_count"][b, i] = subtype, rng.integers(1, 10)
				x["tile_creature"][b, tz, ty, tx], x["tile_count"][b, tz, ty, tx] = x["m_creature"][b, i], x["m_count"][b, i]
		for h in range(heroes):
			x["h_spell_cost"][b, h] = rng.choice([0, 5, 16, 24])
			if kind == c.DecisionKind.TURN:  # areas a day or more away
				x["travel_mask"][b, h] = rng.random((c.MAP_LEVELS, c.TRAVEL_GRID, c.TRAVEL_GRID)) < 0.3
				x["travel_days"][b, h] = np.where(x["travel_mask"][b, h], rng.integers(100, 600, (c.MAP_LEVELS, c.TRAVEL_GRID, c.TRAVEL_GRID)), -1)
	return x
