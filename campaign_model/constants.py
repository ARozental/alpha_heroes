"""Sizes and enums of the campaign (adventure map) model's contract, shared by Python and C++
(agents/AlphaMapAI/Schema.generated.h is generated from this file and schema.py: python -m campaign_model.codegen).

Every decision is one of two families:
- a turn decision (kind TURN, or EXCHANGE while two of our armies trade troops): an action type, then that type's
  arguments from its own table (MOVE: a hero and a tile, BUILD: a town and a building, ...);
- an answer to a question the engine asks (the other kinds): one of up to N_OPTIONS options.
"""
from enum import IntEnum

SCHEMA_VERSION = 18  # 18: where today's walk toward each TRAVEL area stops (travel_stop, at the end: older records read with it at 0, none known), the night's danger in TRAVEL's prior; 17: a hero's further windows (w_extra, w_extra_pos, at the end: older records read with them at 0, every window its slot's own hero's), in the window slots of absent heroes: a square around tiles it reaches today that its own window does not show (past a gate or a monolith, on the other level, more than WINDOW_RADIUS tiles down a road), their tiles MOVE targets as its own window's are; TILE_UNSEEN_EXIT and OBJECT_UNSEEN_EXIT (a teleporter's entrance no exit of which we have seen); 16: ARMY's target slot (army_slot_mask in, at the end; army_slot_logp out; the slot in a record's cell), ARMY within one army and swaps, MULTI's SPLIT_SINGLES, MERGE_ALL and GARRISON_SWAP (multi_mask 11 kinds wide: a record of 15 read with its 8 widened); 15: MULTI's TAKE_BEST, TAKE_BEST_FAST, DROP_SLOW and DROP_FOREIGN (multi_mask 8 kinds wide: a record of 14 read with its 4 widened), their armies (x_take_*, h_drop*, after multi_mask); 14: MULTI (multi_mask, after x_heroes; older records read with it at 0), multi_logp out; 13: ARTIFACT (artifact_mask, after a_*; older records read with it at 0), artifact_logp out; 12: resource_worth out (a purchase short of resources trades for itself: AlphaMapAI pays with what is worth the least), the inputs as 11; 11: a question's guards and what its options do with them (o_battle, o_join, d_guard_*), the armed objects that fight a hero stepping on them (a_*: garrisons), after mem (older records read with them at 0); 10: the model's memory (mem in, memory out: a player's decisions carry it forward); 9: the whole map tile by tile (tile_*, MAP_TILES a side), for the model's map CNN (docs/campaign_model_v2_draft.md); 8: heroes' luck and morale with their parts until the next battle (h_luck, h_morale, h_battle_*), the spells of heroes for hire (g_tavern_spells); 7: what a hero for hire brings (g_tavern_exp, g_tavern_army_*, g_tavern_artifacts); 6: build_short (buildable but for resources); 5: every object we see (MAP_OBJECTS 8192), no g_objects (torch counts them); 4: RECRUIT's destination, TRAVEL's days, our previous decision (g_last_*); 3: the whole map (m_* objects, g_tiles, g_objects), TRAVEL, g_object_income, h_spell_cost; 2: counts as a human sees them

N_RESOURCES = 7  # wood, mercury, ore, sulfur, crystal, gems, gold (VCMI's order)

# Entity tokens: our own first, then the others we can see
OWN_HEROES = 8  # VCMI's default hero limit on the map
OTHER_HEROES = 8
N_HEROES = OWN_HEROES + OTHER_HEROES
OWN_TOWNS = 8
OTHER_TOWNS = 8
N_TOWNS = OWN_TOWNS + OTHER_TOWNS
ARMY_SLOTS = 7
N_LEVELS = 7  # creature levels of a town
N_WORN_SLOTS = 19  # ArtifactPosition 0..18
N_BACKPACK = 8  # the first backpack artifacts
N_SECONDARY_SKILLS = 28
N_SPELLS = 70  # the game's spells, ids 0..69
N_PRIMARY = 4  # attack, defense, spell power, knowledge
N_BUILDINGS = 44  # BuildingID 0..43 (DWELL_LVL_7_UP); later ids belong to mods
N_TAVERN = 2  # heroes for hire

# The map around each of our heroes: (2 R + 1)^2 tiles on the hero's level, the hero in the middle. A MOVE goes to a
# tile of this window the hero reaches today; longer trips are several moves. Schema 17: the tiles it reaches today
# that this window does not show (through a gate or a monolith whose exit we have seen: the other level, a far part of
# its own; further than R tiles down a road) are shown by a further window of the same hero (config.further_windows:
# one), a square of the level that holds the most of them around the soonest reached there, in a window slot no hero
# of ours holds (w_extra: the PC's games had 2.8 heroes a decision in the 8 slots, never more than 4, 2026-11-03). Its
# tiles are MOVE targets as its own window's are, with the same features: the table's entry is (window slot, tile),
# the slot's hero w_extra's. The squares stay squares so that what is read from a tile's neighbours (a guard's zone,
# the land a step reveals, what a guard blocks) is read the same way in both.
WINDOW_RADIUS = 16
WINDOW = 2 * WINDOW_RADIUS + 1

# Army holders for troop moves: our heroes, our towns' garrisons, and the other army of a garrison dialog
N_HOLDERS = OWN_HEROES + OWN_TOWNS + 1
OTHER_ARMY = N_HOLDERS - 1

N_OPTIONS = 8  # answers to an engine question

# The whole map, whatever its size (the windows see 16 tiles around each hero): the objects we see, and per-block counts
# of the tiles (model.py's map encoder scatters the objects onto the blocks)
MAP_OBJECTS = 8192  # every object on revealed tiles (VCMI's getAllVisitableObjs), nearest first: a 144 x 144 two-level map has 3,000-6,000
ARMED_OBJECTS = 32  # armed objects that fight a hero stepping on them (garrisons not ours), the nearest first (schema a_*)
MAP_GRID = 36  # blocks per side of each level; a block is ceil(max(width, height) / MAP_GRID) tiles on a side
MAP_LEVELS = 2  # surface and underground
MAP_TILES = 36  # the whole map's cells (tile_*), this many a side: a tile each on a map up to 36 x 36, a block of side x side tiles on a bigger one (side = ceil(its larger side / MAP_GRID); docs/map_size_plan.md)
MAP_SIDE_MAX = 8  # the largest block side the model reads (288 x 288 tiles; HotA's largest maps 252: 7): a bigger map is an error
# The model's memory (model_v2.Memory): MEMORY_SLOTS slots of MEMORY_WIDTH numbers in -1..1 a player's decisions carry
# forward, its own output passed back by AlphaMapAI; on the wire as integers x MEMORY_SCALE (the inputs are int32)
MEMORY_SLOTS = 8
MEMORY_WIDTH = 64
MEMORY_SCALE = 1 << 20
OWNED_STACKS = 16  # stacks listed of our other owned armies (mine garrisons, ...) and of our dwellings outside towns
TRAVEL_GRID = MAP_GRID // 2  # TRAVEL's areas: blocks of the map encoder's middle level, 18 x 18 a level (2 tiles a side on 36 x 36 maps, 8 on 144 x 144)

N_ADVENTURE_SPELLS = 10  # spell ids 0..9 are exactly the adventure spells
SUMMON_BOAT, SCUTTLE_BOAT, VISIONS, VIEW_EARTH, DISGUISE, VIEW_AIR, FLY, WATER_WALK, DIMENSION_DOOR, TOWN_PORTAL = range(10)
# Spells the model may not cast (model_v2's castable): none. View Air, View Earth and Visions were masked 2026-10-31 10:05
# as "only showing the map" (V7 rose after them; casts were 11% of the turn decisions), and the next eval against
# Nullkiller2 fell from 0.22 to 0.07 (eliminated 20 of 24, from 13): what they show includes the enemy's heroes and armies,
# worth nothing against self-play's passive players and much against an attacker. Back the next hour
INFO_SPELLS = ()
TARGETED_SPELLS = (DIMENSION_DOOR,)  # the spells whose target is a tile (cast_cell); Scuttle Boat: not yet

BUY_ARTIFACTS = (0, 4, 5, 6)  # spellbook (mage guild), ballista, ammo cart, first aid tent (blacksmith)
N_BUY_ITEMS = len(BUY_ARTIFACTS)

RECRUIT_FRACTIONS = (0.25, 0.5, 0.75, 1.0)  # of what can be recruited (available and affordable), rounded up
N_RECRUIT_AMOUNTS = len(RECRUIT_FRACTIONS)
N_RECRUIT_TARGETS = 2  # RECRUIT into: 0 the town's garrison (or the hero garrisoned there), 1 the hero visiting it
TRADE_FRACTIONS = (0.1, 0.25, 0.5, 1.0)  # of the most that can be bought with the resource sold, rounded up
N_TRADE_AMOUNTS = len(TRADE_FRACTIONS)


class ArmyAmount(IntEnum):
	ALL = 0  # the whole stack
	HALF = 1  # half of it, rounded up
	ONE = 2  # a single creature: it moves, or (schema 16, its target slot's second half) it stays and the rest move


N_ARMY_AMOUNTS = len(ArmyAmount)
# ARMY's target slot (schema 16; a record's cell): 0 .. ARMY_SLOTS-1 the slot the amount goes to; ARMY_SLOTS .. 2*ARMY_SLOTS-1
# (ONE alone) the slot all but one go to, the one staying (the user's way to peel a stack into singles: 20 of the 54 troop
# moves of their first recorded game)
ARMY_SLOT_CHOICES = 2 * ARMY_SLOTS


class DecisionKind(IntEnum):
	TURN = 0  # our turn: any action type
	EXCHANGE = 1  # two of our armies meet (hero and hero, or a garrison dialog): ARMY moves, END closes it
	YES_NO = 2  # options YES, NO (fight the guards? accept the creatures? pursue?)
	SELECT = 3  # options: the dialog's components (a chest's gold or experience, ...)
	LEVEL_UP = 4  # options: the secondary skills offered
	OBJECT_SELECT = 5  # options: map objects (Town Portal's town, ...)
	RECRUIT_DWELLING = 6  # options: how many to recruit at a dwelling a hero visits


N_KINDS = len(DecisionKind)
DIALOG_KINDS = tuple(k for k in DecisionKind if k not in (DecisionKind.TURN, DecisionKind.EXCHANGE))


class ActionType(IntEnum):
	END = 0  # ends the turn (TURN) or closes the exchange (EXCHANGE)
	MOVE = 1
	BUILD = 2
	RECRUIT = 3  # in our towns
	HIRE = 4
	ARMY = 5  # move troops between two armies that may trade
	UPGRADE = 6
	TRADE = 7  # resources at a marketplace of ours
	CAST = 8  # an adventure spell
	BUY = 9  # spellbook, war machines
	DISMISS = 10
	TRAVEL = 11  # a hero heads for an area of the map (TRAVEL_GRID) days away: it walks the path there as far as it can today
	ARTIFACT = 12  # a hero's artifact: given to the hero it meets (EXCHANGE), worn from its backpack, or assembled with its worn parts (ArtifactTarget)
	MULTI = 13  # several of the engine's requests at once, the C++ making them (MultiKind): recruit all in a town, all of a town's or a hero's army to a hero


N_ACTION_TYPES = len(ActionType)


class ArtifactTarget(IntEnum):
	"""What ARTIFACT does with a hero's artifact (the user, 2026-11-04: move them to the strongest hero; wear one from the
	backpack, the worn one replaced; assemble a combination)."""

	GIVE = 0  # to the other hero of an exchange: its own slot there if free, else that hero's backpack
	EQUIP = 1  # from the backpack into its slot, the artifact worn there into the backpack
	ASSEMBLE = 2  # into the combination it completes with the hero's worn parts


N_ARTIFACT_POSITIONS = N_WORN_SLOTS + N_BACKPACK  # worn slots 0..18, then the backpack's first 8 (VCMI's ArtifactPosition: BACKPACK_START 19)
N_ARTIFACT_TARGETS = len(ArtifactTarget)


class MultiKind(IntEnum):
	"""MULTI's actions (the user, 2026-11-05: "multi-actions" the C++ makes as several requests, predetermined chains of
	the engine's single actions, the model choosing among them by what they make). multi_mask's first axis: a town of
	ours for RECRUIT_ALL, RECRUIT_ALL_HERO, GATHER and GARRISON_SWAP; in an exchange the hero receiving (0: x_heroes'
	first) for GIVE_ALL, TAKE_BEST and TAKE_BEST_FAST; a hero of ours (its token) for DROP_SLOW, DROP_FOREIGN,
	SPLIT_SINGLES and MERGE_ALL."""

	RECRUIT_ALL = 0  # every creature the town has for hire that we can pay for, the highest level first, into its garrison
	RECRUIT_ALL_HERO = 1  # the same into the hero of ours visiting it
	GATHER = 2  # the town's garrison into the hero of ours visiting it, as much as fits (VCMI's bulk move)
	GIVE_ALL = 3  # an exchange: the other hero's army (one creature kept) and artifacts to this one
	# Schema 15
	TAKE_BEST = 4  # an exchange: Nullkiller2's best army of the two (ArmyManager::getBestArmy) into this one, the rest to the other (x_take_*)
	TAKE_BEST_FAST = 5  # the same less its slowest creatures (the army's movement), which go to the other
	DROP_SLOW = 6  # a hero's slowest creatures (its movement) into the garrison of the town it visits, else disbanded (h_drop)
	DROP_FOREIGN = 7  # a hero's creatures not of its army's main faction (its morale) the same way (h_drop)
	# Schema 16 (the user, 2026-11-07: "we need to be able to do everything a human player does")
	SPLIT_SINGLES = 8  # a hero's army: one creature into each free slot in order, from its weakest stack of more than one (Nullkiller2's addSingleCreatureStacks; VCMI's bulk split's order)
	MERGE_ALL = 9  # a hero's army: each creature's stacks merged into its first (VCMI's bulk merge, every creature)
	GARRISON_SWAP = 10  # a town of ours: its visiting hero into the garrison, its garrison hero out, or both swapped (VCMI's garrisonSwap)


N_MULTI = len(MultiKind)
RESERVED_ACTION_TYPES = (ActionType.DISMISS,)  # in the contract, never legal yet


class OptionKind(IntEnum):
	NONE = 0
	YES = 1
	NO = 2
	COMPONENT = 3  # o_type: component type, o_subtype: its subtype, o_amount: its value
	SKILL = 4  # o_type: secondary skill, o_subtype: the level it would reach
	OBJECT = 5  # o_type: object type, o_subtype: its subtype, o_x/o_y/o_z: where
	AMOUNT = 6  # o_type: the creature, o_amount: how many



class ComponentType(IntEnum):
	"""VCMI's ComponentType (lib/networkPacks/Component.h): a COMPONENT option's o_type (not part of the contract)."""

	PRIM_SKILL = 0
	SEC_SKILL = 1
	RESOURCE = 2
	RESOURCE_PER_DAY = 3
	CREATURE = 4
	ARTIFACT = 5
	SPELL_SCROLL = 6
	MANA = 7
	EXPERIENCE = 8
	LEVEL = 9
	SPELL = 10
	MORALE = 11
	LUCK = 12
	BUILDING = 13
	HERO_PORTRAIT = 14
	FLAG = 15


# w_flags bits of a window tile
TILE_ON_MAP = 1
TILE_REVEALED = 2
TILE_BLOCKED = 4
TILE_VISITABLE = 8
TILE_WATER = 16
TILE_GUARDED = 32  # in a wandering monster's zone of control
TILE_VISITED_BY_PLAYER = 64  # its object was visited by us (this week for weekly objects)
TILE_VISITED_BY_HERO = 128  # its object was visited by this hero
TILE_EXACT_COUNT = 256  # w_count is the monsters' exact count (the hero's Visions reach them), not VCMI's quantity id
# (beyond the N_TILE_FLAGS bits the encoders read as numbers: read where it is named)
TILE_UNSEEN_EXIT = 512  # a teleporter's entrance (a subterranean gate, a monolith, a whirlpool) no exit of which we have seen and that we have not found to lead nowhere: VCMI's pathfinder goes through one only to an exit we see, so nothing past it is a target; stepping on it shows land we know nothing of
N_TILE_FLAGS = 9

# m_flags bits of a map object
OBJECT_VISITED_BY_PLAYER = 1  # visited by us (CRewardableObject::wasVisited(player): our own visits only, what we know)
OBJECT_EXACT_COUNT = 2  # m_count is the monsters' exact count (one of our heroes' Visions reaches them)
OBJECT_GUARDED = 4  # its entrance is in a wandering monster's zone of control
OBJECT_UNSEEN_EXIT = 8  # a teleporter's entrance no exit of which we have seen (TILE_UNSEEN_EXIT's)


class GridCount(IntEnum):
	"""g_tiles: per block, how many of its tiles are ..."""

	ON_MAP = 0
	REVEALED = 1
	WATER = 2  # revealed and water
	BLOCKED = 3  # revealed and blocked
	ROAD = 4  # revealed and with a road


N_GRID_COUNTS = len(GridCount)


class ObjectClass(IntEnum):
	"""What kind of object each listed one is (features.object_classes: the model counts them per block itself), those
	we visited apart (VISITED)."""

	PICKUP = 0  # resources to take: piles, chests, campfires, windmills and water wheels (weekly), ...
	TREASURE = 1  # artifacts, scrolls, Pandora's boxes, prisons
	BANK = 2  # guarded treasure: creature banks, crypts, Dragon Utopia, ...
	SKILL = 3  # experience, skills, spells, primary skills: shrines, libraries, arenas, ...
	BOOST = 4  # luck, morale, movement, mana: fountains, oases, wells, ...
	MINE = 5  # mines and other structures to flag (lighthouses), not ours
	OUR_MINE = 6
	DWELLING = 7  # not ours
	OUR_DWELLING = 8
	TOWN = 9  # neutral or another player's
	OUR_TOWN = 10
	HERO = 11  # another player's
	OUR_HERO = 12
	MONSTER = 13  # wandering monsters
	TRANSPORT = 14  # teleports, gates, whirlpools, boats, shipyards
	INFO = 15  # what reveals the map or opens the way: observatories, obelisks, keymasters, seers' huts, ...
	OTHER = 16  # every other object we can visit: markets, taverns, signs, ...
	VISITED = 17  # any object we visited (CRewardableObject::wasVisited(player)), counted here instead of in its class


N_OBJECT_CLASSES = len(ObjectClass)
# Obj ids of each class (the rest: OTHER); an owned class's OUR_ form is the next class
OBJECT_CLASSES = {
	ObjectClass.PICKUP: (79, 101, 12, 82, 29, 39, 105, 22, 86, 112, 109, 59),
	ObjectClass.TREASURE: (5, 93, 6, 62),
	ObjectClass.BANK: (16, 84, 24, 85, 25, 63, 108),
	ObjectClass.SKILL: (88, 89, 90, 41, 4, 61, 81, 47, 107, 32, 23, 51, 102, 100, 113, 104),
	ObjectClass.BOOST: (30, 31, 38, 11, 14, 28, 52, 56, 64, 94, 96, 110, 48, 49),
	ObjectClass.MINE: (53, 220, 42),
	ObjectClass.DWELLING: (17, 18, 19, 20, 78),
	ObjectClass.TOWN: (98,),
	ObjectClass.HERO: (34,),
	ObjectClass.MONSTER: (54,),
	ObjectClass.TRANSPORT: (43, 44, 45, 103, 111, 8, 87),
	ObjectClass.INFO: (10, 57, 58, 60, 13, 27, 37, 83, 215),
}
OWNED_CLASSES = (ObjectClass.MINE, ObjectClass.DWELLING, ObjectClass.TOWN, ObjectClass.HERO)


def object_class(object_id):
	"""The ObjectClass of an Obj id (as not ours)."""
	for kind, ids in OBJECT_CLASSES.items():
		if object_id in ids:
			return kind
	return ObjectClass.OTHER

# Owner of heroes, towns and map objects, relative to us
OWNER_NONE = 0  # neutral or unowned
OWNER_SELF = 1
# OWNER_SELF + 1 + k: the k-th other player, in color order after ours
MAX_PLAYERS = 8
OWNER_VOCAB = 2 + MAX_PLAYERS

# Embedding table sizes (ids from the engine, battle_model/artifacts/vocab.json)
CREATURE_VOCAB = 160  # 0..149
HERO_TYPE_VOCAB = 256
HERO_CLASS_VOCAB = 32
ARTIFACT_VOCAB = 160  # 0..143
FACTION_VOCAB = 16
OBJECT_VOCAB = 256  # Obj ids 0..231
# (object, subtype) pairs, hashed into SUBTYPE_HASH and folded to SUBTYPE_BUCKETS rows (features.bucket); one more row,
# BLOCKED_SUBTYPE, for a tile blocked and not visitable (encoders.TileEncoder). 8191 rows each until 2026-10-31: ~800
# pairs reached them, an obstacle's graphic variants among them, and 1.1M of the model's 2.2M weights were rows no
# gradient ever reached (the user: "wasting 1M+ parameters for something we will not see ... is insane")
SUBTYPE_HASH = 8191
SUBTYPE_BUCKETS = 1021
BLOCKED_SUBTYPE = SUBTYPE_BUCKETS
SUBTYPE_ROWS = SUBTYPE_BUCKETS + 1


def _object_pairs():
	"""{(object id, subtype): its dense row}: every map object's subtypes in the engine's catalogue
	(priors/object_subtypes.csv), in order."""
	from pathlib import Path

	pairs = {}
	for line in (Path(__file__).resolve().parent / "priors" / "object_subtypes.csv").read_text().splitlines():
		if line and not line.startswith(("#", "object,")):
			obj, count = line.split(",")[:2]
			for subtype in range(int(count)):
				pairs[(int(obj), subtype)] = len(pairs)
	return pairs


# The map objects' (object, subtype) pairs, a row each (features.object_pair; 2026-10-31: the tiles' and the question's
# object, and the empire value's correction, were hashed into 8191 rows): 733 in the engine's catalogue, then a row for a
# pair outside it (or no object) and one for a tile blocked and not visitable (encoders.TileEncoder)
OBJECT_PAIRS = _object_pairs()
MAX_SUBTYPES = max(subtype for _, subtype in OBJECT_PAIRS) + 1
PAIR_OTHER = len(OBJECT_PAIRS)
PAIR_BLOCKED = PAIR_OTHER + 1
PAIR_ROWS = PAIR_OTHER + 2
TERRAIN_VOCAB = 16  # 0..9
ROAD_VOCAB = 8  # 0..3
COMPONENT_VOCAB = 32  # ComponentType
SOUND_VOCAB = 128

# Labels: the decision's value is our share of all empires' value this many days after its day (labels.py); the value
# heads predict each, the label is their flat average (the user, 2026-10-27: d+2, d+5, d+7)
VALUE_OFFSETS = (2, 5, 7)
N_VALUE_OFFSETS = len(VALUE_OFFSETS)
MAX_DECISIONS_PER_DAY = 256  # then the turn ends
