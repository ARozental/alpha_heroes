"""VCMI's battle rules as tensor ops on the raw model inputs: the numbers a hand-written battle AI works with,
computed the way the engine computes them, for the model (model.py) to build on.

Here:
	least_damage, cross_stack_features   what the acting unit's attack does at least, whatever the dice, and
	                                     which stacks it kills for sure (the kill rule in model.py)
	damage_range, after_damage, kills    the engine's damage estimate (min, max), as BattleAI and StupidAI use it
	reachability                         the engine's pathfinding: distances and the path back (StupidAI)
	walk_distances, fly_distances        distances only, from any hexes, for many units at once (features.py)
	unit_value                           a stack's value in gold, as the reward counts it
	geometry                             neighbours, two-hex units, available hexes

The least damage of an attack. The engine (lib/battle/DamageCalculator.cpp, BattleInfo::getActualDamage) deals
the average of min(10, count) rolls in the range

	[max(1, floor(count * min damage * factor)), max(1, floor(count * max damage * factor))]
	factor = (1 + sum of the attack factors) * product of (1 - each defence factor)

so it never deals less than the range's minimum. A lower bound needs only the factors that lower the damage, and
it stays a lower bound when factors that raise it (luck, Bless's extra damage at higher levels, hate, jousting,
death blow, double damage, archery) are left out. The ones that lower it and that our inputs show:
	attack below the target's defence: -2.5% a point, at most -70%   (above it: +5% a point, at most +400%)
	Curse on the attacker: its minimum damage, lowered by the effect's value (Bless: its maximum, raised by it)
	Shield (melee) and Air Shield (shooting) on the target: minus their percentage
	shooting from more than 10 hexes from every hex of the target, or at a target under Air Shield: half
	shooting while under basic Forgetfulness: half (stronger Forgetfulness forbids shooting: the masks)
	a shooter fighting in melee without NO_MELEE_PENALTY: half
Not visible in our inputs, so not here (the bound can be too high there): the enemy hero's Armorer (its skills
are hidden), petrification, Magic and Psychic Elementals' halved damage, walls (sieges), damage caps, obstacles
that hurt a unit on its way, and bonuses that apply only to shooting or only to melee (the inputs carry the melee
attack and defence).

All of it in exact integer arithmetic, so that a stack whose hit points equal the damage is judged right. The
engine works in floating point: when the exact damage is a whole number and a factor is fractional, it can
land just below it, so that case counts one less.
"""
import torch

from . import vocab
from .constants import CREATURE_VOCAB, FIELD_WIDTH, N_HEXES
from .reward import resources_value

# config/gameConfig.json "combat", in exact steps: attack over defence adds 1/20 a point (0.05), at most 80 points
# (+400%); defence over attack removes 1/40 a point (0.025), at most 28 points (-70%)
ATTACK_STEPS, ATTACK_POINTS_CAP = 20, 80
DEFENSE_STEPS, DEFENSE_POINTS_CAP = 40, 28
SHOOTING_PENALTY_DISTANCE = 10  # GameConstants::BATTLE_SHOOTING_PENALTY_DISTANCE

# Spell effects (SpellID, config/spells/timed.json) that change the damage of an attack or how a unit moves
SHIELD, AIR_SHIELD, BLESS, CURSE, FORGETFULNESS = 27, 28, 41, 42, 61
BLIND, PARALYZE, STONE_GAZE, FLY, QUICKSAND = 62, 74, 70, 6, 10
MAGIC_ELEMENTAL, PSYCHIC_ELEMENTAL = 121, 120  # CreatureID: half damage against some targets (DamageCalculator)

# EAccessibility and CObstacleInstance::EObstacleType values in the inputs (x_access, x_obstacle_type)
ACCESSIBLE, GATE = 0, 4
SPELL_CREATED, MOAT = 2, 3
INFINITE_DISTANCE = 1_000_000  # ReachabilityInfo::INFINITE_DIST
ATTACKER, DEFENDER = 0, 1

# config/gameConfig.json "combat": the chance of good and bad morale and luck, by its value (1, 2, 3), out of a die of
# 24 (goodMoraleChance, badMoraleChance, moraleDiceSize, goodLuckChance, badLuckChance (empty: no bad luck), luckDiceSize)
GOOD_MORALE, BAD_MORALE = (0.0, 1 / 24, 2 / 24, 3 / 24), (0.0, 2 / 24, 4 / 24, 6 / 24)
GOOD_LUCK, BAD_LUCK = (0.0, 1 / 24, 2 / 24, 3 / 24), (0.0, 0.0, 0.0, 0.0)


# Per-creature tables (creature_tables): name -> (dtype, from a vocabulary row)
def _ability_value(row, name, subtype=None):
	"""Sum of the values of a creature's bonuses of type `name` (and of that subtype), from the vocabulary."""
	return sum(v for s, v in row.get("ability_values", {}).get(name, ()) if subtype is None or s == subtype)


CREATURE_TABLES = {
	"value": (torch.float32, lambda row: resources_value(row.get("cost", {}))),
	"shooter": (torch.bool, lambda row: "SHOOTER" in row.get("abilities", ())),
	"flying": (torch.bool, lambda row: "FLYING" in row.get("abilities", ())),
	"no_melee_penalty": (torch.bool, lambda row: "NO_MELEE_PENALTY" in row.get("abilities", ())),
	"no_distance_penalty": (torch.bool, lambda row: "NO_DISTANCE_PENALTY" in row.get("abilities", ())),
	"blocks_retaliation": (torch.bool, lambda row: "BLOCKS_RETALIATION" in row.get("abilities", ())),
	"mind_immune": (torch.bool, lambda row: "MIND_IMMUNITY" in row.get("abilities", ())),
	"spell_immunity_level": (torch.int64, lambda row: _ability_value(row, "LEVEL_SPELL_IMMUNITY")),
	"enemy_defence_reduction": (torch.int64, lambda row: _ability_value(row, "ENEMY_DEFENCE_REDUCTION")),  # percent
	"enemy_attack_reduction": (torch.int64, lambda row: _ability_value(row, "ENEMY_ATTACK_REDUCTION")),  # percent
	"melee_reduction": (torch.int64, lambda row: _ability_value(row, "GENERAL_DAMAGE_REDUCTION", 0)),  # percent, damageTypeMelee
	"ranged_reduction": (torch.int64, lambda row: _ability_value(row, "GENERAL_DAMAGE_REDUCTION", 1)),  # damageTypeRanged
	"all_reduction": (torch.int64, lambda row: _ability_value(row, "GENERAL_DAMAGE_REDUCTION", -1)),  # damageTypeAll
	"damage_cap": (torch.int64, lambda row: _ability_value(row, "DAMAGE_RECEIVED_CAP")),  # percent of max health
	"base_attack": (torch.int64, lambda row: row.get("attack", 0)),
	"base_defense": (torch.int64, lambda row: row.get("defense", 0)),
	"extra_attacks": (torch.int64, lambda row: _ability_value(row, "ADDITIONAL_ATTACK")),  # e.g. Marksmen shoot twice
	"war_machine": (torch.bool, lambda row: "SIEGE_WEAPON" in row.get("abilities", ())),
}


def creature_tables(path=vocab.VOCAB_PATH):
	"""Per creature id, from the vocabulary: its value in gold (reward.py prices) and the abilities the rules use
	(CREATURE_TABLES), [CREATURE_VOCAB + 1] each; and "hate": [attacker, target] its HATE percentage.

	The last row (and every row without a vocabulary) is a creature worth 0 with no abilities.
	"""
	tables = {name: torch.zeros(CREATURE_VOCAB + 1, dtype=dtype) for name, (dtype, _) in CREATURE_TABLES.items()}
	tables["hate"] = torch.zeros(CREATURE_VOCAB + 1, CREATURE_VOCAB + 1, dtype=torch.int64)
	if not path.exists():
		return tables
	for row in vocab.tables(path)["creatures"].values():
		if 0 <= row["id"] < CREATURE_VOCAB:
			for name, (_, value) in CREATURE_TABLES.items():
				tables[name][row["id"]] = value(row)
			for target, percent in row.get("ability_values", {}).get("HATE", ()):
				if 0 <= target < CREATURE_VOCAB:
					tables["hate"][row["id"], target] += percent
	return tables


def hex_distance(a, b):
	"""BattleHex::getDistance: steps between two hexes (rows of FIELD_WIDTH hexes, every other row shifted)."""
	a, b = a.clamp(min=0), b.clamp(min=0)
	y1, y2 = a // FIELD_WIDTH, b // FIELD_WIDTH
	x1, x2 = a % FIELD_WIDTH + y1 // 2, b % FIELD_WIDTH + y2 // 2
	dx, dy = x2 - x1, y2 - y1
	same_direction = ((dx >= 0) & (dy >= 0)) | ((dx < 0) & (dy < 0))
	return torch.where(same_direction, torch.maximum(dx.abs(), dy.abs()), dx.abs() + dy.abs())


def effect_value(x, spell, units=None):
	"""Per unit [B, U] (or for the units `units` [B]): the value of the effect of `spell` on it, and whether it has it."""
	ids, values = x["u_effect_id"], x["u_effect_value"]
	if units is not None:
		rows = torch.arange(ids.shape[0], device=ids.device)
		ids, values = ids[rows, units], values[rows, units]
	has = ids == spell
	return (values * has).sum(dim=-1), has.any(dim=-1)


def least_damage(count, single_min, attack, defense, reduction_percent, halvings):
	"""The least damage of an attack (integer tensors, broadcast), as the engine computes its damage range.

	count, single_min: the attacking stack's creatures and their least damage each; attack, defense: the attacker's
	attack and the target's defence; reduction_percent: Shield or Air Shield on the target (0 = none); halvings:
	how many of the halving penalties apply.
	"""
	over = (attack - defense).clamp(min=0, max=ATTACK_POINTS_CAP)
	under = (defense - attack).clamp(min=0, max=DEFENSE_POINTS_CAP)
	numerator = count * single_min * (ATTACK_STEPS + over) * (DEFENSE_STEPS - under) * (100 - reduction_percent)
	denominator = ATTACK_STEPS * DEFENSE_STEPS * 100 * 2**halvings
	fractional = (over > 0) | (under > 0) | (reduction_percent > 0) | (halvings > 0)
	exact = numerator // denominator
	floating_point_slip = fractional & (exact * denominator == numerator)  # a whole number the engine may round down
	return (exact - floating_point_slip.long()).clamp(min=1)


def cross_stack_features(x, tables):
	"""What the acting unit (slot g_active_unit) can do to each unit slot, [B, U] each:

	least_melee, least_shot: the least damage of its melee attack or of its shot on that unit, whatever the dice
	kills_melee, kills_shot: 1.0 where that attack kills the whole stack whatever the dice
	total_hp: the unit's hit points left (0 when dead or empty)
	value: its value in gold, the creature's price times its creatures, the top one counted by its health left
	value_share: its share of the value of the living army of the side that controls it

	Whether the attack is legal at all is up to the masks (m_melee_target, m_shoot_target).
	"""
	batch = torch.arange(x["g_active_unit"].shape[0], device=x["g_active_unit"].device)
	me = x["g_active_unit"].clamp(0, x["u_count"].shape[1] - 1)

	def mine(name):
		return x[name][batch, me][:, None]  # [B, 1], against every slot

	creature = torch.where((x["u_creature"] >= 0) & (x["u_creature"] < CREATURE_VOCAB), x["u_creature"], torch.full_like(x["u_creature"], CREATURE_VOCAB))
	my_creature = creature[batch, me][:, None]
	alive = (x["u_present"] != 0) & (x["u_alive"] != 0)
	enemy = x["u_owner_side"] != x["g_our_side"][:, None]

	# The attacker's least damage per creature: Curse forces its minimum, Bless its maximum (BlessCurse in the engine)
	curse, cursed = effect_value(x, CURSE, me)
	bless, blessed = effect_value(x, BLESS, me)
	shift = bless - curse
	single_min = torch.where(cursed & ~blessed, mine("u_dmg_min")[:, 0] - curse, torch.where(blessed & ~cursed, mine("u_dmg_max")[:, 0] + bless, mine("u_dmg_min")[:, 0] + torch.where(cursed & blessed, shift, torch.zeros_like(shift))))
	single_min = single_min.clamp(min=1)[:, None]

	shield, _ = effect_value(x, SHIELD)
	air_shield, under_air_shield = effect_value(x, AIR_SHIELD)
	_, forgetful = effect_value(x, FORGETFULNESS, me)
	near = (hex_distance(mine("u_hex"), x["u_hex"]) <= SHOOTING_PENALTY_DISTANCE) | ((x["u_rear_hex"] >= 0) & (hex_distance(mine("u_hex"), x["u_rear_hex"]) <= SHOOTING_PENALTY_DISTANCE))
	far = ~near & ~tables["no_distance_penalty"][my_creature]
	melee_halvings = (tables["shooter"][my_creature] & ~tables["no_melee_penalty"][my_creature]).long().expand_as(shield)
	shot_halvings = (far | under_air_shield).long() + forgetful.long()[:, None]

	least_melee = least_damage(mine("u_count"), single_min, mine("u_attack"), x["u_defense"], shield, melee_halvings)
	least_shot = least_damage(mine("u_count"), single_min, mine("u_attack"), x["u_defense"], air_shield, shot_halvings)
	total_hp = torch.where(alive, (x["u_count"] - 1) * x["u_max_hp"] + x["u_first_hp"], torch.zeros_like(x["u_count"]))
	target = alive & enemy

	health = torch.where(alive, (x["u_count"] - 1).float() + x["u_first_hp"].float() / x["u_max_hp"].clamp(min=1).float(), torch.zeros_like(x["u_count"], dtype=torch.float32))
	value = tables["value"][creature] * health
	side_total = torch.stack([(value * (x["u_owner_side"] == side)).sum(dim=1) for side in (0, 1)], dim=1)  # [B, 2]
	own_total = side_total.gather(1, x["u_owner_side"].clamp(0, 1))
	return {
		"least_melee": least_melee,
		"least_shot": least_shot,
		"kills_melee": (target & (least_melee >= total_hp)).float(),
		"kills_shot": (target & (least_shot >= total_hp)).float(),
		"total_hp": total_hp,
		"value": value,
		"value_share": value / own_total.clamp(min=1.0),
	}


# ---- Geometry (lib/battle/BattleHex.h, BattleHexArray.cpp, Unit.cpp) ----

FIELD_HEIGHT = N_HEXES // FIELD_WIDTH


def _step(hex_, direction):
	"""BattleHex::cloneInDirection for one hex (python ints): the neighbour, or -1 off the field.
	Directions in BattleHex::hexagonalDirections order: top left, top right, right, bottom right, bottom left, left."""
	y, x = divmod(hex_, FIELD_WIDTH)
	odd = y % 2
	nx, ny = [(x - 1 if odd else x, y - 1), (x if odd else x + 1, y - 1), (x + 1, y), (x if odd else x + 1, y + 1), (x - 1 if odd else x, y + 1), (x - 1, y)][direction]
	return ny * FIELD_WIDTH + nx if 0 <= nx < FIELD_WIDTH and 0 <= ny < FIELD_HEIGHT else -1


TOP_LEFT, TOP_RIGHT, RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT, LEFT = range(6)
# [N_HEXES, 6]: each hex's neighbours in direction order, -1 off the field (getNeighbouringTiles keeps the others in this order)
NEIGHBOURS = torch.tensor([[_step(h, d) for d in range(6)] for h in range(N_HEXES)])
# [N_HEXES, 8] per side: the hexes around a two-hex unit standing on a hex (getNeighbouringTilesDoubleWide), -1 padded
_wide = {
	ATTACKER: lambda h: [_step(h, d) for d in range(5)] + [_step(h - 1, d) if h - 1 >= 0 else -1 for d in (BOTTOM_LEFT, LEFT, TOP_LEFT)],
	DEFENDER: lambda h: [_step(h, TOP_LEFT)] + [_step(h + 1, d) if h + 1 < N_HEXES else -1 for d in range(5)] + [_step(h, BOTTOM_LEFT), _step(h, LEFT)],
}
WIDE_NEIGHBOURS = torch.stack([torch.tensor([[n for n in _wide[side](h)] for h in range(N_HEXES)]) for side in (ATTACKER, DEFENDER)])


def neighbours(hexes):
	"""[..., 6] the neighbours of hexes (any shape, -1 = none), -1 off the field."""
	return torch.where(hexes[..., None] >= 0, NEIGHBOURS.to(hexes.device)[hexes.clamp(min=0)], -1)


def available(hexes):
	"""BattleHex::isAvailable: on the field and not in the first or last column."""
	column = hexes % FIELD_WIDTH
	return (hexes >= 0) & (hexes < N_HEXES) & (column > 0) & (column < FIELD_WIDTH - 1)


def rear_hex(hexes, side):
	"""Unit::occupiedHex: the second hex of a two-hex unit of `side` standing on hexes (attackers face right, so
	their second hex is on the left)."""
	return torch.where(side == ATTACKER, hexes - 1, hexes + 1)


def lookup(table, hexes, fill=False):
	"""table[b, hexes[b, ...]] for a [B, N_HEXES] table; fill where hexes is off the field."""
	batch = torch.arange(table.shape[0], device=table.device).view(-1, *([1] * (hexes.dim() - 1)))
	on_field = (hexes >= 0) & (hexes < table.shape[1])
	values = table[batch, hexes.clamp(0, table.shape[1] - 1)]
	if values.dtype == torch.bool:  # (onnxruntime has no Where for booleans)
		return (values | ~on_field) if fill else (values & on_field)
	return torch.where(on_field, values, torch.zeros_like(values) + (int(fill) if isinstance(fill, bool) else fill))


# ---- The damage estimate (lib/battle/DamageCalculator.cpp calculateDmgRange) ----

def units_of(x, tables):
	"""The unit fields the damage rules use, [B, U] each, as the engine reads them."""
	creature = torch.where((x["u_creature"] >= 0) & (x["u_creature"] < CREATURE_VOCAB), x["u_creature"], torch.full_like(x["u_creature"], CREATURE_VOCAB))
	return {
		"creature": creature, "side": x["u_side"], "hex": x["u_hex"], "rear": x["u_rear_hex"],
		"alive": (x["u_present"] != 0) & (x["u_alive"] != 0), "retaliations": x["u_retaliations"], "speed": x["u_speed"],
		"summoned": x["u_is_summoned"] != 0, "clone": x["u_is_clone"] != 0,
		"count": x["u_count"], "first_hp": x["u_first_hp"], "max_hp": x["u_max_hp"],
		"attack": x["u_attack"], "defense": x["u_defense"], "dmg_min": x["u_dmg_min"], "dmg_max": x["u_dmg_max"],
		"effect_id": x["u_effect_id"], "effect_value": x["u_effect_value"],
		**{name: tables[name][creature] for name in CREATURE_TABLES},
	}


def select(units, index):
	"""The fields of one unit per battle (index [B]), shaped [B, 1] to broadcast against every unit."""
	batch = torch.arange(index.shape[0], device=index.device)
	return {k: v[batch, index][:, None] for k, v in units.items()}


def effect(unit, spell):
	"""(value of the spell's effect on the unit, whether it has it), for unit fields shaped [..., E]."""
	has = unit["effect_id"] == spell
	return (unit["effect_value"] * has).sum(dim=-1), has.any(dim=-1)


def damage_range(attacker, target, shooting, tables, dtype=torch.float64):
	"""DamageCalculator::calculateDmgRange: the (min, max) damage of `attacker` hitting `target` (unit fields that
	broadcast against each other), in double precision and in the engine's order of operations, so that it matches the
	engine exactly. Charge distance 0 and no luck (as in the engine's estimates, battleEstimateDamage).

	Not computed (0 here): hero skills and artifacts (Offense, Archery, Armorer: hidden for the enemy hero), Slayer,
	hate of traits, back attacks, HotA's Revenge, walls (sieges), arrow towers and war machines.

	dtype: float64 matches the engine exactly (StupidAI); float32 (the learned model, on a GPU without float64) can be
	off by one where the engine's result is a whole number.
	"""
	f64 = dtype
	# Base damage per creature, Bless and Curse (getBaseDamageBlessCurse), times the creatures
	curse, cursed = effect(attacker, CURSE)
	bless, blessed = effect(attacker, BLESS)
	shift = bless - curse
	low, high = (attacker["dmg_min"] + shift).clamp(min=1), (attacker["dmg_max"] + shift).clamp(min=1)
	low, high = torch.where(cursed & ~blessed, low, torch.where(blessed & ~cursed, high, low)), torch.where(cursed & ~blessed, low, torch.where(blessed & ~cursed, high, high))
	low, high = low * attacker["count"], high * attacker["count"]

	# Attack and defence as they count (getActorAttackEffective, getTargetDefenseEffective)
	attack, defense = attacker["attack"], target["defense"]
	attack_cut = (attack * target["enemy_attack_reduction"] + 100 // 2 - 1) // 100  # vstd::divideAndRound
	attack = attack - torch.where(target["enemy_attack_reduction"] > 0, torch.minimum(attack_cut, attack), torch.zeros_like(attack))
	defense_cut = torch.floor(attacker["enemy_defence_reduction"].to(f64) / 100.0 * defense.to(f64)).long() + 1
	defense = defense - torch.where(attacker["enemy_defence_reduction"] > 0, torch.minimum(defense_cut, defense), torch.zeros_like(defense))
	advantage = (attack - defense).to(f64)

	# Attack factors, added (getAttackFactors): attack over defence, hate
	skill_bonus = torch.where(advantage > 0, torch.clamp(0.05 * advantage, max=4.0), torch.zeros_like(advantage))
	hate = tables["hate"][attacker["creature"], target["creature"]].to(f64) / 100.0
	attack_total = 1.0 + skill_bonus + hate

	# Defence factors, multiplied in the engine's order (getDefenseFactors)
	shield, _ = effect(target, SHIELD)
	air_shield, under_air_shield = effect(target, AIR_SHIELD)
	_, forgetful = effect(attacker, FORGETFULNESS)
	blind, _ = effect(attacker, BLIND)
	paralyzed, _ = effect(attacker, PARALYZE)
	petrified, _ = effect(target, STONE_GAZE)
	near = (hex_distance(attacker["hex"], target["hex"]) <= SHOOTING_PENALTY_DISTANCE) | ((target["rear"] >= 0) & (hex_distance(attacker["hex"], target["rear"]) <= SHOOTING_PENALTY_DISTANCE))
	far = ~near & ~attacker["no_distance_penalty"]
	half = torch.full_like(advantage, 0.5)
	zero = torch.zeros_like(advantage)
	factors = [
		torch.where(advantage < 0, torch.clamp(0.025 * -advantage, max=0.7), zero),  # defence over attack
		target["all_reduction"].to(f64) / 100.0,  # armorer-like abilities
		torch.where(shooting, air_shield + target["ranged_reduction"], shield + target["melee_reduction"]).to(f64) / 100.0,  # Shield, Air Shield
		torch.where(shooting, torch.where(far | under_air_shield, half, zero), torch.where(attacker["shooter"] & ~attacker["no_melee_penalty"], half, zero)),  # range, melee penalty
		(blind + paralyzed).to(f64) / 100.0,  # GENERAL_ATTACK_REDUCTION on the attacker (a blinded unit retaliating)
		torch.where(shooting & forgetful, half, zero),
		petrified.to(f64) / 100.0,
		torch.where((attacker["creature"] == MAGIC_ELEMENTAL) & (target["spell_immunity_level"] >= 5), half, zero),
		torch.where((attacker["creature"] == PSYCHIC_ELEMENTAL) & target["mind_immune"], half, zero),
	]
	defense_total = torch.ones_like(advantage)
	for factor in factors:
		defense_total = defense_total * (1 - torch.clamp(factor, max=1.0))
	factor = attack_total * defense_total

	low = torch.floor(low.to(f64) * factor).long().clamp(min=1)
	high = torch.floor(high.to(f64) * factor).long().clamp(min=1)
	cap = target["max_hp"] * target["damage_cap"] // 100
	low = torch.where(target["damage_cap"] > 0, torch.minimum(low, cap), low)
	high = torch.where(target["damage_cap"] > 0, torch.minimum(high, cap), high)
	return low, high


def total_health(unit):
	return torch.where(unit["count"] > 0, (unit["count"] - 1) * unit["max_hp"] + unit["first_hp"], torch.zeros_like(unit["count"]))


def after_damage(unit, damage):
	"""The unit fields after taking `damage` (CHealth::damage): fewer creatures, the top one hurt."""
	left = (total_health(unit) - damage).clamp(min=0)
	max_hp = unit["max_hp"].clamp(min=1)
	count = (left + max_hp - 1) // max_hp
	return {**unit, "count": count, "first_hp": torch.where(count > 0, left - (count - 1) * max_hp, torch.zeros_like(left))}


# ---- Pathfinding (CBattleInfoCallback::getReachability, makeBFS, getFlyingReachability) ----

MAX_PATH = 64  # steps the search follows; no path on a battlefield is longer


def reachability(x, me, flying):
	"""The engine's reachability of unit `me` (unit fields [B, 1]) from where it stands: (distances [B, N_HEXES], with
	INFINITE_DISTANCE where it cannot go; predecessors [B, N_HEXES], the hex before on the path, -1 if none).

	Walking: the engine's breadth-first search, so that the predecessors are the engine's too: a hex's predecessor is
	the first hex the search took from its queue that leads to it (queue order: discovery order, neighbours in
	direction order). Flying: every hex it may stand on, at its hex distance, reached straight from the start.
	A hex is open if the battlefield marks it accessible (x_access; the gate only for the defender) or it is the
	unit's own; a two-hex unit needs its second hex open too. Moat and quicksand stop a walk (getStoppers).
	"""
	batch = x["x_access"].shape[0]
	hexes = torch.arange(N_HEXES, device=x["x_access"].device).expand(batch, N_HEXES)
	start, side = me["hex"][:, 0], me["side"][:, 0]
	wide = me["rear"][:, 0] >= 0
	own = (hexes == start[:, None]) | (wide[:, None] & (hexes == me["rear"]))
	tile_open = (x["x_access"] == ACCESSIBLE) | ((x["x_access"] == GATE) & (side[:, None] == DEFENDER)) | own
	second = rear_hex(hexes, side[:, None])
	second_open = lookup(tile_open, torch.where((second >= 0) & (second < N_HEXES), second, -1))
	can_stand = tile_open & (~wide[:, None] | second_open)
	stopper = (x["x_obstacle_type"] == MOAT) | ((x["x_obstacle_type"] == SPELL_CREATED) & (x["x_obstacle_id"] == QUICKSAND))
	stopper = stopper & ~own
	stuck = stopper | (wide[:, None] & lookup(stopper, torch.where((second >= 0) & (second < N_HEXES), second, -1)))

	# Flying
	fly_distance = torch.where(can_stand, hex_distance(start[:, None], hexes), torch.full_like(hexes, INFINITE_DISTANCE))
	fly_from = torch.where(can_stand, start[:, None].expand_as(hexes), torch.full_like(hexes, -1))

	# Walking: one breadth-first layer at a time
	distance = torch.where(hexes == start[:, None], 0, INFINITE_DISTANCE)
	came_from = torch.full_like(hexes, -1)
	queue_rank = torch.zeros_like(hexes)  # order in the search's queue within the current layer
	around = neighbours(hexes)  # [B, H, 6]: the hexes a hex can be reached from
	# Direction from neighbour k of h back to h: the opposite direction
	back = torch.tensor([BOTTOM_RIGHT, BOTTOM_LEFT, LEFT, TOP_LEFT, TOP_RIGHT, RIGHT], device=hexes.device)
	never = N_HEXES * 6
	for layer in range(MAX_PATH):
		expands = (distance == layer) & ~stuck  # a hex in a moat or quicksand is reached but not left
		from_open = lookup(expands, around)  # [B, H, 6]
		key = torch.where(from_open, lookup(queue_rank, around) * 6 + back, never)
		best_key, best = key.min(dim=-1)
		new = (best_key < never) & can_stand & (distance == INFINITE_DISTANCE)
		distance = torch.where(new, layer + 1, distance)
		came_from = torch.where(new, around.gather(-1, best[..., None])[..., 0], came_from)
		new_key = torch.where(new, best_key, never)
		queue_rank = torch.where(new, (new_key[:, None, :] < new_key[:, :, None]).sum(dim=-1), 0)
	return torch.where(flying[:, None], fly_distance, distance), torch.where(flying[:, None], fly_from, came_from)


def kills(damage, unit):
	"""The creatures `damage` kills (DamageCalculator::getCasualties): the top one if it covers its hit points left,
	then one per full max hit points, at most all of them."""
	max_hp = unit["max_hp"].clamp(min=1)
	more = torch.div((damage - unit["first_hp"]).clamp(min=0), max_hp, rounding_mode="floor")
	return torch.where(damage >= unit["first_hp"], torch.minimum(1 + more, unit["count"].to(more.dtype)), torch.zeros_like(more))


def unit_value(unit, tables):
	"""A stack's value in gold as the reward counts it: the creature's price times its creatures, the top one counted
	by its health left; 0 when dead or empty, and for summoned units and clones (they vanish after the battle)."""
	health = (unit["count"] - 1).float() + unit["first_hp"].float() / unit["max_hp"].clamp(min=1).float()
	counts = unit["alive"] & (unit["count"] > 0) & ~unit["summoned"] & ~unit["clone"]
	return torch.where(counts, tables["value"][unit["creature"]] * health, torch.zeros_like(health))


# The magic schools and the secondary skill of each (vocab secondary_skills: Fire, Air, Water, Earth Magic)
SCHOOLS = ("fire", "air", "water", "earth")
SCHOOL_SKILLS = (14, 15, 16, 17)


def spell_schools(path=vocab.VOCAB_PATH):
	"""[N_SPELLS, len(SCHOOLS)]: the schools each spell belongs to (several for e.g. Magic Arrow), from the vocabulary."""
	from .constants import N_SPELLS

	schools = torch.zeros(N_SPELLS, len(SCHOOLS), dtype=torch.bool)
	if path.exists():
		for row in vocab.tables(path)["spells"].values():
			if 0 <= row["id"] < N_SPELLS:
				for school in row.get("schools", ()):
					if school in SCHOOLS:
						schools[row["id"], SCHOOLS.index(school)] = True
	return schools


def smallest(x, dim=-1):
	"""x.amin(dim) for a short axis, as elementwise minimums: torch's integer amin over a short last axis is ~100x slower
	on the CPU."""
	parts = x.unbind(dim)
	result = parts[0]
	for part in parts[1:]:
		result = torch.minimum(result, part)
	return result


def walk_distances(open_, sources, steps):
	"""Breadth-first distances [..., N_HEXES] from the source hexes (bool [..., N_HEXES]) over the open hexes (the
	sources count as open), INFINITE_DISTANCE beyond `steps` steps; any leading dimensions (e.g. one per unit)."""
	neighbour = NEIGHBOURS.to(open_.device)
	on_field = neighbour >= 0
	distance = torch.where(sources, 0, INFINITE_DISTANCE)
	for _ in range(steps):
		around = distance[..., neighbour.clamp(min=0)]  # [..., H, 6]
		around = torch.where(on_field, around, INFINITE_DISTANCE)
		closer = smallest(around) + 1
		distance = torch.where((open_ | sources) & (closer < distance), closer, distance)
	return distance


def fly_distances(open_, source_hexes):
	"""Straight distances [..., N_HEXES] from the nearest of source_hexes ([..., S], -1 = none) to every open hex
	(a flyer goes over anything), INFINITE_DISTANCE to the others."""
	hexes = torch.arange(N_HEXES, device=open_.device)
	steps = hex_distance(source_hexes[..., :, None], hexes)  # [..., S, H]
	steps = smallest(torch.where(source_hexes[..., :, None] >= 0, steps, INFINITE_DISTANCE), dim=-2)
	return torch.where(open_ | (steps == 0), steps, INFINITE_DISTANCE)
