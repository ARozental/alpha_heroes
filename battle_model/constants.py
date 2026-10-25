"""Sizes, vocabularies and enums of the battle model contract.

Everything the C++ side (agents/AlphaBattleAI) and the model must agree on lives here.
codegen.py turns it into agents/AlphaBattleAI/Schema.generated.h, and export.py stores
SCHEMA_VERSION and a hash of the full contract in the ONNX metadata, so a model built
against a different contract is rejected at load time instead of silently misbehaving.

Bump SCHEMA_VERSION on any change to sizes, enums, input or output definitions.
"""
from enum import IntEnum

SCHEMA_VERSION = 14  # 8: 20 unit slots (10 per side) instead of 40; 9: CREATURE_SPELL (g_creature_spell, its masks and outputs); 10: spells' predicted value change (s_*_value, c_*_value); 11: engine-played attack values (a_*), engine damage estimates (p_*), surrender cost, start assets, kept counts; 12: hero level and artifacts, units hit by each cast and attack; 13: spell mastery from the engine (h_spell_mastery), h_spells with tomes and scrolls; 14: 14 unit slots per side, tactics, catapult and heal actions, our hero's worn artifacts (h_worn_artifacts)

# Battlefield: VCMI BattleHex indices 0..186 are used as-is (17 columns x 11 rows,
# columns 0 and 16 are the unplayable side columns). No remapping on the C++ side.
FIELD_WIDTH = 17
FIELD_HEIGHT = 11
N_HEXES = FIELD_WIDTH * FIELD_HEIGHT

# Unit slots: up to UNITS_PER_SIDE of each battle side (an army has at most 7 stacks, plus war machines, summoned
# units and clones), dead stacks included (resurrection targets them). Over that the encoder keeps the acting unit,
# then living stacks by value, then dead ones; the rest are not seen (wrong decisions possible, never illegal ones).
UNITS_PER_SIDE = 14  # 7 stacks + 4 war machines + 3 arrow towers fit, and summoned units and clones beside a full army
MAX_UNITS = 2 * UNITS_PER_SIDE
# Active spell effects per unit (VCMI itself has no limit; its UI shows 8).
MAX_EFFECTS = 8
# Heroes are indexed by battle side: 0 = attacker, 1 = defender.
N_SIDES = 2

# SpellID 0..69 are the castable spells (the HERO_SPELL output covers these). Higher ids are creature
# ability effects (poison, aging, petrify, ...) and internal obstacle triggers (fire wall, land mine)
# that only appear as active effects on units, never as something a hero casts.
N_SPELLS = 70
N_SECONDARY_SKILLS = 28
# Worn artifact slots of a hero (ArtifactPosition HEAD 0 .. MISC5 18; the backpack is not worn)
N_WORN_SLOTS = 19

# Value heads: the expected reward formula at the end of this round, of the next round, ... for
# N_VALUE_HORIZONS rounds, then at the end of the battle (output "value", N_VALUE_HORIZONS + 1 entries).
N_VALUE_HORIZONS = 3

# Retaliations are capped: "unlimited" (Royal Griffins) is reported as this value.
MAX_RETALIATIONS = 3

# Embedding vocabularies for raw ids. All ids are the engine's own (VCMI identifiers); their names and
# stats are in battle_model/artifacts/vocab.json (scripts/export_vocab.sh), read by vocab.py.
# Sizes are internal to the model, not part of the contract. Any id outside [0, size) maps to an
# extra "none/unknown" row.
CREATURE_VOCAB = 150  # core H3 creatures 0..149
EFFECT_VOCAB = 128  # spells and ability effects: 0..103 in VCMI 1.7.5
HERO_TYPE_VOCAB = 256
BATTLEFIELD_VOCAB = 64
OBSTACLE_ID_VOCAB = 256
ACCESSIBILITY_VOCAB = 7  # EAccessibility
OBSTACLE_TYPE_VOCAB = 4  # CObstacleInstance::EObstacleType
WALL_STATE_VOCAB = 4  # EWallState without NONE (-1)
GATE_STATE_VOCAB = 5  # EGateState


class ActionType(IntEnum):
	WAIT = 0
	DEFEND = 1
	MOVE = 2  # + move hex
	MELEE = 3  # + target unit, + hex to attack from (our unit's head hex)
	SHOOT = 4  # + target unit
	HERO_SPELL = 5  # + spell, + first target (unit / hex / none), + second target for two-target spells
	RETREAT = 6
	SURRENDER = 7
	CREATURE_SPELL = 8  # the acting unit casts its spell (g_creature_spell): + target unit / hex
	CATAPULT = 9  # a catapult under our control: + the wall part's hex (in the move-hex output: a catapult cannot move)
	HEAL = 10  # a First Aid Tent under our control: + our unit to heal (in the shoot-target output: a tent cannot shoot)
	TACTICS_MOVE = 11  # tactics phase: place the acting unit, + hex (in the move-hex output; no unit moves otherwise then)
	TACTICS_END = 12  # tactics phase: leave the rest of the army where it is and start the battle


N_ACTION_TYPES = len(ActionType)
RESERVED_ACTION_TYPES = ()  # types the C++ side always masks out (none since v14)
# The output each type's argument comes from: the hex of MOVE, TACTICS_MOVE and CATAPULT is chosen in logp_move_hex, the
# unit of SHOOT and HEAL in logp_shoot_target (only one of each group is ever legal at a time)
HEX_TYPES = (ActionType.MOVE, ActionType.TACTICS_MOVE, ActionType.CATAPULT)
TARGET_TYPES = (ActionType.SHOOT, ActionType.HEAL)


class TargetKind(IntEnum):
	NONE = 0
	UNIT = 1
	HEX = 2


# Spells that take a second destination, and its kind. The first destination's kind is not
# static (e.g. Expert Haste has no target, Basic Haste targets a unit), so it is read from
# the masks: a spell's first target is whichever of m_spell_unit / m_spell_hex has legal entries.
SPELL_TELEPORT = 63
SPELL_SACRIFICE = 40
SECOND_TARGET_KIND = {
	SPELL_TELEPORT: TargetKind.HEX,  # unit -> destination hex
	SPELL_SACRIFICE: TargetKind.UNIT,  # dead unit to revive -> own unit to sacrifice
}

# Value of a masked-out logit before log_softmax; outputs carry -inf for illegal entries.
MASKED_LOGIT = -1.0e9
