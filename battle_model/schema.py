"""Named input and output tensors of the battle model.

Every input is a raw value read from the engine; all transforms (log scaling, embeddings,
relative sides, ...) happen inside the model. Each tensor has a leading batch dimension B,
which the C++ side always sets to 1. Shapes below exclude it.

Masks are legality computed by the engine. The model applies them to its outputs only:
every output is a masked log-probability table, -inf where the engine says illegal.
"""
import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from .constants import (
	MAX_EFFECTS,
	MAX_UNITS,
	N_ACTION_TYPES,
	EFFECT_VOCAB,
	N_HEXES,
	N_SECONDARY_SKILLS,
	N_SIDES,
	N_SPELLS,
	N_VALUE_HORIZONS,
	RESERVED_ACTION_TYPES,
	SCHEMA_VERSION,
	SECOND_TARGET_KIND,
	ActionType,
	TargetKind,
)

U, H, S, K, T = MAX_UNITS, N_HEXES, N_SPELLS, MAX_EFFECTS, N_ACTION_TYPES


@dataclass(frozen=True)
class TensorSpec:
	name: str
	dtype: str  # "int64", "bool" or "float32"
	shape: tuple  # without the batch dimension
	fill: int  # value of unused entries (padding unit slots, absent heroes, ...)
	doc: str


def _i(name, shape, doc, fill=0):
	return TensorSpec(name, "int64", shape, fill, doc)


def _m(name, shape, doc):
	return TensorSpec(name, "bool", shape, 0, doc)


INPUTS = (
	# Global
	_i("g_our_side", (), "battle side the model plays: 0 attacker, 1 defender"),
	_i("g_round", (), "battle round, VCMI numbering: 1 = first round (0 = before it, e.g. the tactics phase)"),
	_i("g_battlefield", (), "battlefield type id"),
	_i("g_tactic_distance", (), "columns allowed in the tactics phase, 0 outside it"),
	_i("g_active_unit", (), "index into the unit slots of the unit that acts now"),
	_i("g_hero_can_cast", (), "1 if our hero may cast a spell right now"),
	_i("g_gate_state", (), "EGateState of the town gate, 0 (NONE) outside sieges"),
	_i("g_creature_spell", (), "the spell the acting unit can cast now (CREATURE_SPELL), -1 if none, or for a Master Genie, whose spell the engine draws when it casts", fill=-1),
	# Units, one slot each, ordered by VCMI unit id; unused slots have u_present = 0
	_i("u_present", (U,), "1 if the slot holds a unit"),
	_i("u_alive", (U,), "1 if alive (dead stacks are kept: resurrection can target them)"),
	_i("u_creature", (U,), "creature id", fill=-1),
	_i("u_side", (U,), "battle side of the unit: 0 attacker, 1 defender", fill=-1),
	_i("u_owner_side", (U,), "battle side of the player controlling the unit now (the engine's battleGetOwner): the other side while it is hypnotized", fill=-1),
	_i("u_hex", (U,), "position: front hex (BattleHex index)", fill=-1),
	_i("u_rear_hex", (U,), "second hex of a two-hex unit, -1 for one-hex units", fill=-1),
	_i("u_count", (U,), "creatures in the stack"),
	_i("u_first_hp", (U,), "HP left of the top creature"),
	_i("u_max_hp", (U,), "max HP of one creature (current, with bonuses)"),
	_i("u_attack", (U,), "attack (current, with bonuses)"),
	_i("u_defense", (U,), "defense (current, with bonuses)"),
	_i("u_dmg_min", (U,), "min damage per creature (current)"),
	_i("u_dmg_max", (U,), "max damage per creature (current)"),
	_i("u_speed", (U,), "movement range this round"),
	_i("u_shots", (U,), "shots left"),
	_i("u_retaliations", (U,), "retaliations left this round, unlimited reported as MAX_RETALIATIONS"),
	_i("u_casts", (U,), "creature spell casts left"),
	_i("u_morale", (U,), "morale"),
	_i("u_luck", (U,), "luck"),
	_i("u_moved", (U,), "1 if it already acted this round"),
	_i("u_waited", (U,), "1 if it waited this round"),
	_i("u_defending", (U,), "1 if in defensive stance"),
	_i("u_is_clone", (U,), "1 for clones (vanish after battle)"),
	_i("u_is_summoned", (U,), "1 for summoned units (vanish after battle)"),
	_i("u_queue_now", (U,), "position in the remaining turn order of this round (0 = acting now), -1 if not in it", fill=-1),
	_i("u_queue_next", (U,), "position in next round's turn order, -1 if not in it", fill=-1),
	_i("u_effect_id", (U, K), "active effect: SpellID 0..81, -1 = empty slot", fill=-1),
	_i("u_effect_turns", (U, K), "rounds the effect has left"),
	_i("u_effect_value", (U, K), "raw value of the effect's first bonus"),
	# Heroes, indexed by battle side
	_i("h_present", (N_SIDES,), "1 if that side has a hero"),
	_i("h_type", (N_SIDES,), "hero type id", fill=-1),
	_i("h_attack", (N_SIDES,), "primary skill: attack"),
	_i("h_defense", (N_SIDES,), "primary skill: defense"),
	_i("h_power", (N_SIDES,), "primary skill: spell power"),
	_i("h_knowledge", (N_SIDES,), "primary skill: knowledge"),
	_i("h_mana", (N_SIDES,), "mana points left"),
	_i("h_mana_max", (N_SIDES,), "mana limit"),
	_i("h_skills", (N_SIDES, N_SECONDARY_SKILLS), "secondary skill levels 0..3; all 0 for the enemy hero (hidden from a player)"),
	_i("h_spells", (N_SIDES, S), "1 if the spell is in the spellbook; all 0 for the enemy hero (hidden from a player)"),
	# Hexes, BattleHex index 0..186
	_i("x_access", (H,), "EAccessibility"),
	_i("x_obstacle_type", (H,), "EObstacleType of an obstacle on the hex, -1 if none", fill=-1),
	_i("x_obstacle_id", (H,), "obstacle id (spell id for spell-created obstacles), -1 if none", fill=-1),
	_i("x_obstacle_side", (H,), "caster side of a spell-created obstacle, -1 otherwise", fill=-1),
	_i("x_obstacle_turns", (H,), "rounds left of a spell-created obstacle, -1 otherwise", fill=-1),
	_i("x_wall_state", (H,), "EWallState of a wall part on the hex, -1 if none", fill=-1),
	# Legality masks
	_m("m_type", (T,), "legal action types"),
	_m("m_move_hex", (H,), "MOVE: legal destination hexes"),
	_m("m_melee_target", (U,), "MELEE: legal target units"),
	_m("m_melee_from", (U, H), "MELEE: legal hexes to attack each target from"),
	_m("m_shoot_target", (U,), "SHOOT: legal target units"),
	_m("m_spell", (S,), "HERO_SPELL: castable spells"),
	_m("m_spell_unit", (S, U), "HERO_SPELL: legal unit first targets per spell"),
	_m("m_spell_hex", (S, H), "HERO_SPELL: legal hex first targets per spell"),
	_m("m_spell2_hex", (U, H), "HERO_SPELL (Teleport): legal destination hex per first-target unit"),
	_m("m_spell2_unit", (U, U), "HERO_SPELL (Sacrifice): legal second unit per first-target unit"),
	_m("m_creature_spell_unit", (U,), "CREATURE_SPELL: legal unit targets"),
	_m("m_creature_spell_hex", (H,), "CREATURE_SPELL: legal hex targets"),
)

# Every output is float32 masked log-probabilities, conditional on the previous choices:
#   type -> MOVE: hex | MELEE: target -> from hex | SHOOT: target
#        -> HERO_SPELL: spell -> first target (unit | hex | none) -> second target (Teleport, Sacrifice)
#        -> CREATURE_SPELL: target (unit | hex)
OUTPUTS = (
	TensorSpec("logp_type", "float32", (T,), 0, "log p(action type)"),
	TensorSpec("logp_move_hex", "float32", (H,), 0, "log p(hex | MOVE)"),
	TensorSpec("logp_melee_target", "float32", (U,), 0, "log p(target | MELEE)"),
	TensorSpec("logp_melee_from", "float32", (U, H), 0, "log p(from hex | MELEE, target)"),
	TensorSpec("logp_shoot_target", "float32", (U,), 0, "log p(target | SHOOT)"),
	TensorSpec("logp_spell", "float32", (S,), 0, "log p(spell | HERO_SPELL)"),
	TensorSpec("logp_spell_unit", "float32", (S, U), 0, "log p(unit target | spell)"),
	TensorSpec("logp_spell_hex", "float32", (S, H), 0, "log p(hex target | spell)"),
	TensorSpec("logp_spell2_hex", "float32", (U, H), 0, "log p(second hex | first unit)  (Teleport)"),
	TensorSpec("logp_spell2_unit", "float32", (U, U), 0, "log p(second unit | first unit)  (Sacrifice)"),
	TensorSpec("logp_creature_spell_unit", "float32", (U,), 0, "log p(unit target | CREATURE_SPELL)"),
	TensorSpec("logp_creature_spell_hex", "float32", (H,), 0, "log p(hex target | CREATURE_SPELL)"),
	TensorSpec("value", "float32", (N_VALUE_HORIZONS + 1,), 0, "expected reward formula at the end of this round, the next round, ... (N_VALUE_HORIZONS of them), then at the end of the battle; each in -1..1 (reward.to_scale)"),
)

INPUT_NAMES = tuple(spec.name for spec in INPUTS)
OUTPUT_NAMES = tuple(spec.name for spec in OUTPUTS)
INPUTS_BY_NAME = {spec.name: spec for spec in INPUTS}
OUTPUTS_BY_NAME = {spec.name: spec for spec in OUTPUTS}


def contract_dict():
	"""Everything both sides must agree on, as plain JSON-able data."""
	return {
		"schema_version": SCHEMA_VERSION,
		"inputs": [asdict(spec) for spec in INPUTS],
		"outputs": [asdict(spec) for spec in OUTPUTS],
		"action_types": {a.name: int(a) for a in ActionType},
		"reserved_action_types": [int(a) for a in RESERVED_ACTION_TYPES],
		"second_target_kind": {str(k): int(v) for k, v in SECOND_TARGET_KIND.items()},
	}


def contract_hash():
	blob = json.dumps(contract_dict(), sort_keys=True).encode()
	return hashlib.sha256(blob).hexdigest()[:16]


def _np_dtype(dtype):
	return {"int64": np.int64, "bool": np.bool_, "float32": np.float32}[dtype]


def empty_inputs(batch=1):
	"""All inputs at their fill values (an empty battlefield with nothing legal)."""
	return {s.name: np.full((batch,) + s.shape, s.fill, dtype=_np_dtype(s.dtype)) for s in INPUTS}


def random_inputs(batch=1, seed=0, n_units=14):
	"""Random but internally consistent inputs: every legal action type has legal arguments."""
	rng = np.random.default_rng(seed)
	x = empty_inputs(batch)
	for b in range(batch):
		x["g_our_side"][b] = rng.integers(0, 2)
		x["g_round"][b] = rng.integers(0, 30)
		x["g_battlefield"][b] = rng.integers(0, 40)
		x["g_active_unit"][b] = 0
		x["g_hero_can_cast"][b] = rng.integers(0, 2)
		for u in range(n_units):
			side = u % 2
			x["u_present"][b, u] = 1
			x["u_alive"][b, u] = 1 if u < n_units - 2 else 0
			x["u_creature"][b, u] = rng.integers(0, 150)
			x["u_side"][b, u] = side
			x["u_owner_side"][b, u] = side if rng.random() < 0.9 else 1 - side
			x["u_hex"][b, u] = rng.integers(0, H)
			x["u_rear_hex"][b, u] = rng.integers(0, H) if rng.random() < 0.3 else -1
			x["u_count"][b, u] = rng.integers(1, 3000)
			x["u_first_hp"][b, u] = rng.integers(1, 200)
			x["u_max_hp"][b, u] = rng.integers(1, 1000)
			for name in ("u_attack", "u_defense", "u_dmg_min", "u_dmg_max", "u_speed", "u_shots", "u_morale", "u_luck"):
				x[name][b, u] = rng.integers(0, 40)
			x["u_retaliations"][b, u] = rng.integers(0, 4)
			x["u_queue_now"][b, u] = u
			x["u_queue_next"][b, u] = u
			x["u_effect_id"][b, u, 0] = rng.integers(0, EFFECT_VOCAB)
			x["u_effect_turns"][b, u, 0] = rng.integers(0, 5)
		for side in range(N_SIDES):
			x["h_present"][b, side] = 1
			x["h_type"][b, side] = rng.integers(0, 150)
			x["h_mana"][b, side] = rng.integers(0, 100)
			x["h_mana_max"][b, side] = 100
			x["h_skills"][b, side] = rng.integers(0, 4, N_SECONDARY_SKILLS)
		x["h_spells"][b, x["g_our_side"][b]] = rng.integers(0, 2, S)
		x["x_access"][b] = rng.integers(0, 7, H)

		def pick(n, p=0.3):
			return rng.random(n) < p

		x["m_move_hex"][b] = pick(H)
		targets = pick(U, 0.4) & (x["u_present"][b] == 1)
		x["m_melee_target"][b] = targets
		x["m_melee_from"][b] = pick((U, H), 0.05) & targets[:, None]
		x["m_melee_target"][b] &= x["m_melee_from"][b].any(axis=1)
		x["m_shoot_target"][b] = pick(U, 0.4) & (x["u_present"][b] == 1)
		spells = pick(S, 0.2)
		for s in np.flatnonzero(spells):
			kind = rng.integers(0, 3)  # first target: none / unit / hex
			if kind == 1 or s in SECOND_TARGET_KIND:
				x["m_spell_unit"][b, s] = pick(U, 0.3) & (x["u_present"][b] == 1)
			elif kind == 2:
				x["m_spell_hex"][b, s] = pick(H, 0.3)
		for s, second in SECOND_TARGET_KIND.items():
			for u in np.flatnonzero(x["m_spell_unit"][b, s]):
				if second == TargetKind.HEX:
					x["m_spell2_hex"][b, u] = pick(H, 0.3)
				else:
					x["m_spell2_unit"][b, u] = pick(U, 0.3) & (x["u_present"][b] == 1)
		x["m_spell"][b] = spells
		if rng.random() < 0.3:  # the acting unit casts a spell, on units or on hexes
			x["g_creature_spell"][b] = rng.integers(-1, EFFECT_VOCAB)
			if rng.random() < 0.7:
				x["m_creature_spell_unit"][b] = pick(U, 0.3) & (x["u_present"][b] == 1)
			else:
				x["m_creature_spell_hex"][b] = pick(H, 0.3)
		finalize_masks(x, b)
	return x


def finalize_masks(x, b):
	"""Clear any choice whose follow-up choices are all illegal, then derive m_type from the argument masks.

	The C++ side applies the same rules; check_dumps.py verifies that it does.
	"""
	for s in np.flatnonzero(x["m_spell"][b]):
		units = x["m_spell_unit"][b, s]  # view: updated in place
		hexes = x["m_spell_hex"][b, s]
		second = SECOND_TARGET_KIND.get(int(s))
		# A spell with no target at all has both rows empty; decide that before filtering
		needs_target = second is not None or units.any() or hexes.any()
		if second == TargetKind.HEX:
			units &= x["m_spell2_hex"][b].any(axis=1)
		elif second == TargetKind.UNIT:
			units &= x["m_spell2_unit"][b].any(axis=1)
		if needs_target and not (units.any() or hexes.any()):
			x["m_spell"][b, s] = False
	x["m_melee_target"][b] &= x["m_melee_from"][b].any(axis=1)
	t = x["m_type"][b]
	t[:] = False
	t[ActionType.WAIT] = True
	t[ActionType.DEFEND] = True
	t[ActionType.MOVE] = x["m_move_hex"][b].any()
	t[ActionType.MELEE] = x["m_melee_target"][b].any()
	t[ActionType.SHOOT] = x["m_shoot_target"][b].any()
	t[ActionType.HERO_SPELL] = bool(x["g_hero_can_cast"][b]) and x["m_spell"][b].any()
	# Creature spells without a target are not offered (the core game has none)
	t[ActionType.CREATURE_SPELL] = x["m_creature_spell_unit"][b].any() or x["m_creature_spell_hex"][b].any()
