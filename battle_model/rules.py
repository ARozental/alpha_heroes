"""VCMI's battle rules as tensor ops on the raw model inputs: the numbers a hand-written battle AI works with,
computed the way the engine computes them, for the model (model.py) to build on.

So far: what the acting unit's attack does to every unit slot at least, whatever the dice, and with it which
stacks it kills for sure.

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
from .constants import CREATURE_VOCAB, FIELD_WIDTH
from .reward import resources_value

# config/gameConfig.json "combat", in exact steps: attack over defence adds 1/20 a point (0.05), at most 80 points
# (+400%); defence over attack removes 1/40 a point (0.025), at most 28 points (-70%)
ATTACK_STEPS, ATTACK_POINTS_CAP = 20, 80
DEFENSE_STEPS, DEFENSE_POINTS_CAP = 40, 28
SHOOTING_PENALTY_DISTANCE = 10  # GameConstants::BATTLE_SHOOTING_PENALTY_DISTANCE

# Spell effects (SpellID, config/spells/timed.json) that change the damage of an attack
SHIELD, AIR_SHIELD, BLESS, CURSE, FORGETFULNESS = 27, 28, 41, 42, 61


def creature_tables(path=vocab.VOCAB_PATH):
	"""Per creature id, from the vocabulary: its value in gold (reward.py prices) and the abilities the rules use.

	Rows [CREATURE_VOCAB + 1]; the last row (and every row without a vocabulary) is a creature worth 0 with none.
	"""
	tables = {name: torch.zeros(CREATURE_VOCAB + 1, dtype=dtype) for name, dtype in (("value", torch.float32), ("shooter", torch.bool), ("no_melee_penalty", torch.bool), ("no_distance_penalty", torch.bool))}
	if not path.exists():
		return tables
	for c in vocab.tables(path)["creatures"].values():
		if 0 <= c["id"] < CREATURE_VOCAB:
			abilities = set(c.get("abilities", ()))
			tables["value"][c["id"]] = resources_value(c.get("cost", {}))
			tables["shooter"][c["id"]] = "SHOOTER" in abilities
			tables["no_melee_penalty"][c["id"]] = "NO_MELEE_PENALTY" in abilities
			tables["no_distance_penalty"][c["id"]] = "NO_DISTANCE_PENALTY" in abilities
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
