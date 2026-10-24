"""Bounded, scale-free features of a battle, computed from the raw inputs with the game's own rules (rules.py): the
numbers a hand-written battle AI works with, for the model (model.py). No parameters.

Every feature is bounded (0..1, a few -1..1) and built only from this battle: ratios of quantities of the same kind
(hit points to hit points, value to value, damage to the health it hits), never raw counts or hit points. So the
features stay about the same if every stack is ten times larger (hero spells, whose damage does not grow with the
stacks, are the exception), and nothing needs batch statistics, which a real battle would not match anyway.

	read_battle       the raw inputs as unit fields: the acting unit, who controls what, values, health
	stack_features    per unit: health, value shares, damage, attack and defence, speed, ammunition, morale, turn, ...
	distance_maps     per unit, the steps to every hex; and the acting unit's steps from every hex to each enemy
	attack_features   the acting unit against each unit: what its melee attack, or its shot, would do
	threat_maps       per hex, for the acting unit standing there: who can attack it, whom it blocks, whom it can reach
	unit_summaries    per unit: the most of an enemy it can take in one attack, the most an enemy in reach can take of it
	battle_features   the battle as a whole: strength, round, the acting unit's danger
	action_features   the above assigned to each action: per melee target and hex, shot target, move hex, action type

The feature names (the *_FEATURES tuples) are the model's: a rule weight per name (model.RuleScore).
"""
from dataclasses import dataclass

import torch

from . import rules
from .constants import MAX_RETALIATIONS, N_HEXES

EPS = 1e-6
REACH_STEPS = 24  # how far the distance maps look: more than any unit moves in a turn


def fraction(part, whole):
	"""part / whole, in 0..1 (0 where whole is 0)."""
	return (part / whole.clamp(min=EPS)).clamp(0.0, 1.0)


def relative(a, b):
	"""a / (a + b) for two quantities of the same kind, in 0..1: 0.5 when they are equal."""
	return a / (a + b).clamp(min=EPS)


@dataclass
class Battle:
	x: dict  # the raw inputs
	tables: dict  # per creature (rules.creature_tables)
	units: dict  # [B, U] unit fields (rules.units_of)
	me: dict  # [B, 1] the acting unit's fields
	acting: torch.Tensor  # [B] its slot
	ours: torch.Tensor  # [B, U] living units we control (a hypnotized enemy too)
	enemy: torch.Tensor  # [B, U] living units the enemy controls
	value: torch.Tensor  # [B, U] gold value as the reward counts it (rules.unit_value)
	health: torch.Tensor  # [B, U] hit points left

	@property
	def our_value(self):
		return (self.value * self.ours).sum(dim=1)  # [B]

	@property
	def enemy_value(self):
		return (self.value * self.enemy).sum(dim=1)

	def mine(self, per_unit):
		"""The acting unit's entry of a [B, U, ...] tensor, [B, ...]."""
		return per_unit[torch.arange(per_unit.shape[0], device=per_unit.device), self.acting]


def read_battle(x, tables):
	units = rules.units_of(x, tables)
	units["can_shoot"] = units["alive"] & units["shooter"] & (x["u_shots"] > 0)
	units["flying"] = units["flying"] | rules.effect(units, rules.FLY)[1]
	acting = x["g_active_unit"].clamp(0, x["u_count"].shape[1] - 1)
	controller = x["u_owner_side"]
	ours = units["alive"] & (controller == x["g_our_side"][:, None])
	enemy = units["alive"] & (controller != x["g_our_side"][:, None]) & (controller >= 0)
	return Battle(x, tables, units, rules.select(units, acting), acting, ours, enemy, rules.unit_value(units, tables), rules.total_health(units))


# ---- Per unit ----

STACK = {}


def stack_feature(f):
	STACK[f.__name__] = f
	return f


@stack_feature
def health(b):
	"""Hit points left of the top creature, of its max: 1 unhurt."""
	return fraction(b.units["first_hp"].float(), b.units["max_hp"].float())


@stack_feature
def value_in_battle(b):
	"""Its value against the average living stack's: 0.5 average, toward 1 far bigger."""
	living = b.units["alive"].float()
	mean = (b.value * living).sum(dim=1, keepdim=True) / living.sum(dim=1, keepdim=True).clamp(min=1)
	return relative(b.value, mean)


@stack_feature
def value_in_army(b):
	"""Its share of the value of the living army it fights for."""
	totals = torch.where(b.ours, b.our_value[:, None], b.enemy_value[:, None])
	return fraction(b.value, totals)


@stack_feature
def damage_per_hit_point(b):
	"""Average damage of one creature against its own hit points: how hard it hits for how much it can take."""
	average = (b.units["dmg_min"] + b.units["dmg_max"]).float() / 2
	return relative(average, b.units["max_hp"].float())


@stack_feature
def damage_spread(b):
	"""How much its damage varies: 0 always the same, toward 1 from little to much."""
	low, high = b.units["dmg_min"].float(), b.units["dmg_max"].float()
	return fraction(high - low, high + low)


@stack_feature
def attack(b):
	"""Attack against 20 points, the 5%-a-point scale of the damage formula: 0.5 at 20."""
	return relative(b.units["attack"].float(), torch.full_like(b.health, 20, dtype=torch.float32))


@stack_feature
def defense(b):
	"""Defence against 40 points, the 2.5%-a-point scale: 0.5 at 40."""
	return relative(b.units["defense"].float(), torch.full_like(b.health, 40, dtype=torch.float32))


@stack_feature
def attack_buff(b):
	"""What its attack bonuses (hero, spells) are worth in damage: tanh of 5% a point over the creature's own."""
	return torch.tanh(0.05 * (b.units["attack"] - b.units["base_attack"]).float())


@stack_feature
def defense_buff(b):
	"""The same for defence, 2.5% a point."""
	return torch.tanh(0.025 * (b.units["defense"] - b.units["base_defense"]).float())


@stack_feature
def attack_style(b):
	"""Attack against attack + defence: offensive (toward 1) or defensive (toward 0), now."""
	return relative(b.units["attack"].float(), b.units["defense"].float())


@stack_feature
def creature_style(b):
	"""The same for the creature as it comes, without bonuses."""
	return relative(b.units["base_attack"].float(), b.units["base_defense"].float())


@stack_feature
def speed(b):
	"""Hexes it moves this round, of 15 (the battlefield's playable width): 1 reaches anything."""
	return fraction(b.units["speed"].float(), torch.full_like(b.health, 15, dtype=torch.float32))


@stack_feature
def ammunition(b):
	"""Shots left, of 12 (a full quiver for most shooters)."""
	return fraction(b.x["u_shots"].float(), torch.full_like(b.health, 12, dtype=torch.float32))


@stack_feature
def can_shoot(b):
	return b.units["can_shoot"].float()


@stack_feature
def retaliations(b):
	return fraction(b.x["u_retaliations"].float(), torch.full_like(b.health, MAX_RETALIATIONS, dtype=torch.float32))


@stack_feature
def casts(b):
	"""Creature spell casts left, of 5."""
	return fraction(b.x["u_casts"].float(), torch.full_like(b.health, 5, dtype=torch.float32))


@stack_feature
def morale(b):
	"""log of the expected actions a round from morale: log(1 + chance) of a second action for good morale, log(1 -
	chance) of losing the turn for bad (gameConfig.json); 0 for units morale does not affect (the engine sends 0)."""
	m = b.x["u_morale"].clamp(-3, 3)
	good = torch.tensor(rules.GOOD_MORALE, device=m.device)[m.clamp(min=0)]
	bad = torch.tensor(rules.BAD_MORALE, device=m.device)[(-m).clamp(min=0)]
	return torch.log1p(good) + torch.log1p(-bad)


@stack_feature
def luck(b):
	"""log of the expected damage multiplier from luck: a lucky strike doubles the base damage, an unlucky one halves it."""
	l = b.x["u_luck"].clamp(-3, 3)
	good = torch.tensor(rules.GOOD_LUCK, device=l.device)[l.clamp(min=0)]
	bad = torch.tensor(rules.BAD_LUCK, device=l.device)[(-l).clamp(min=0)]
	return torch.log1p(good) + torch.log1p(-0.5 * bad)


def _turn(order):
	"""How soon in a turn order it acts: 1 first, toward 0 last, 0 not in it."""
	in_queue = order >= 0
	length = in_queue.sum(dim=1, keepdim=True).clamp(min=1).float()
	return torch.where(in_queue, 1 - order.float() / length, torch.zeros_like(order, dtype=torch.float32))


@stack_feature
def turn_now(b):
	return _turn(b.x["u_queue_now"])


@stack_feature
def turn_next(b):
	return _turn(b.x["u_queue_next"])


for _name, _flag in {
	"alive": lambda b: b.units["alive"],
	"acting": lambda b: torch.arange(b.health.shape[1], device=b.health.device)[None, :] == b.acting[:, None],
	"ours": lambda b: b.ours,
	"on_our_side": lambda b: b.units["side"] == b.x["g_our_side"][:, None],  # differs from ours while hypnotized
	"moved": lambda b: b.x["u_moved"] != 0,
	"waited": lambda b: b.x["u_waited"] != 0,
	"defending": lambda b: b.x["u_defending"] != 0,
	"clone": lambda b: b.x["u_is_clone"] != 0,
	"summoned": lambda b: b.x["u_is_summoned"] != 0,
	"two_hex": lambda b: b.units["rear"] >= 0,
	"flying": lambda b: b.units["flying"],
	"war_machine": lambda b: b.units["war_machine"],
	"melee_target": lambda b: b.x["m_melee_target"],  # the acting unit may hit it now
	"shot_target": lambda b: b.x["m_shoot_target"],
}.items():
	STACK[_name] = (lambda flag: lambda b: flag(b).float())(_flag)

STACK_FEATURES = tuple(STACK)


def stack_features(b):
	"""[B, U, len(STACK_FEATURES)], 0 in empty slots."""
	present = (b.x["u_present"] != 0).float()[..., None]
	return torch.stack([f(b) for f in STACK.values()], dim=-1) * present


# ---- Reach ----

# Positions are a unit's front hex (u_hex, the move mask's hexes); a two-hex unit also covers rules.rear_hex of it.

def per_unit(table, hexes, fill=False):
	"""rules.lookup per unit: table [B, U, H] at hexes [B, U, ...]."""
	return rules.lookup(table.flatten(0, 1), hexes.flatten(0, 1), fill).view(hexes.shape)


def standable(b, footprint, wide, side):
	"""[B, U, H]: the positions each unit could stand on (AccessibilityInfo::accessible): its hex free or its own, and
	for a two-hex unit its second hex too. footprint [B, U, 2]: the unit's hexes now; wide, side [B, U]."""
	hexes = torch.arange(N_HEXES, device=footprint.device)
	free = (b.x["x_access"] == rules.ACCESSIBLE)[:, None, :] | (footprint[..., :, None] == hexes).any(dim=-2)
	second = rules.rear_hex(hexes, side[..., None]).expand_as(free)
	return free & (~wide[..., None] | per_unit(free, second))


def attack_positions(target_hexes, wide, side):
	"""[..., 12]: the positions a unit can melee a unit on target_hexes from (CStack::meleeAttackHexes: a target hex
	around its footprint): next to it, and for a two-hex unit also where its second hex is next to it. -1 = none.
	wide, side broadcast against target_hexes."""
	around = rules.neighbours(target_hexes)  # [..., 6]
	shift = torch.where(side == rules.ATTACKER, 1, -1)[..., None]  # the front hex is right of the second for attackers
	shifted = torch.where((around >= 0) & wide[..., None], around + shift, -1)
	shifted = torch.where((shifted >= 0) & (shifted < N_HEXES), shifted, -1)
	return torch.cat([around, shifted], dim=-1)


def covered(per_hex, wide, side):
	"""per_hex [..., H] over a unit's footprint: for each position, true if true at its hex or, for a two-hex unit (wide,
	side [...]), at its second hex: e.g. it can be attacked there through either hex."""
	hexes = torch.arange(N_HEXES, device=per_hex.device)
	second = rules.rear_hex(hexes, side[..., None]).expand_as(per_hex)
	return per_hex | (wide[..., None] & rules.lookup(per_hex.flatten(0, -2), second.flatten(0, -2)).view(per_hex.shape))


def distance_maps(b):
	"""(reach, to_enemy), [B, U, H] each, over positions; INFINITE_DISTANCE past REACH_STEPS or where it cannot go:
	reach     per unit, the steps from where it stands to every position it could stand on: walking around obstacles and
	          units, or flying straight (two-hex units need both hexes free: rules.reachability, makeBFS)
	to_enemy  per unit, the acting unit's steps from every position to one it could attack that unit from (how far it is
	          from attacking it), on its own terms (its footprint, whether it flies)
	Both searches run as one (walk_distances over 2U rows)."""
	hexes = torch.arange(N_HEXES, device=b.health.device)
	own = torch.stack([b.units["hex"], b.units["rear"]], dim=-1)  # [B, U, 2]
	wide, side = b.units["rear"] >= 0, b.units["side"]
	stand = standable(b, own, wide, side)
	start = (b.units["hex"][..., None] == hexes) & (b.units["hex"][..., None] >= 0)
	my_own = torch.stack([b.me["hex"][:, 0], b.me["rear"][:, 0]], dim=-1)[:, None, :].expand_as(own)
	my_wide, my_side = (b.me["rear"] >= 0).expand_as(wide), b.me["side"].expand_as(side)
	my_stand = standable(b, my_own, my_wide, my_side)
	positions = attack_positions(own, my_wide[..., None], my_side[..., None]).flatten(2)  # [B, U, 24]: to attack each unit from
	positions = torch.where(per_unit(my_stand, positions), positions, -1)
	targets = (positions[..., :, None] == hexes).any(dim=-2)
	walk = rules.walk_distances(torch.cat([stand, my_stand], dim=1), torch.cat([start, targets], dim=1), REACH_STEPS)
	walk_reach, walk_to = walk.split(stand.shape[1], dim=1)
	reach = torch.where(b.units["flying"][..., None], rules.fly_distances(stand, b.units["hex"][..., None]), walk_reach)
	fly_to = rules.fly_distances(my_stand, positions)
	return reach, torch.where(b.me["flying"][..., None], fly_to, walk_to)


def melee_reach(b, reach):
	"""[B, U, H]: whether each unit could attack a unit on each hex in melee this coming turn: whether it can get to a
	position it attacks that hex from (attack_positions)."""
	hexes = torch.arange(N_HEXES, device=reach.device)
	wide, side = (b.units["rear"] >= 0)[..., None], b.units["side"][..., None]  # [B, U, 1]
	positions = attack_positions(hexes.expand(*wide.shape[:2], -1), wide, side)  # [B, U, H, 12]
	steps = per_unit(reach, positions.flatten(2), fill=rules.INFINITE_DISTANCE).view(positions.shape)
	return rules.smallest(steps) <= b.units["speed"][..., None]


def can_attack_at(b, reach):
	"""[B, U, H]: whether each unit could attack a unit on each hex this coming turn: a shooter with ammunition from
	anywhere, any unit in melee if it can get to attack it (melee_reach)."""
	return b.units["can_shoot"][..., None] | melee_reach(b, reach)


# ---- The acting unit's attacks ----

ATTACK_FEATURES = ("damage", "kills", "wipe_chance", "value_destroyed", "threat_removed", "retaliation", "value_lost", "net_value")


def _average_damage(attacker, target, shooting, tables):
	"""The engine's average damage of attacker on target, all its attacks (Marksmen shoot twice), [broadcast]."""
	low, high = rules.damage_range(attacker, target, shooting, tables, dtype=torch.float32)
	attacks = (1 + attacker["extra_attacks"]).float()
	return (low + high).float() / 2 * attacks, low.float() * attacks, high.float() * attacks


def threat_share(b):
	"""[B, U]: each enemy's share of the damage the enemies could do to the acting unit (each capped at its health)."""
	shooting = b.units["can_shoot"]
	average, _, _ = _average_damage(b.units, b.me, shooting, b.tables)
	to_me = torch.minimum(average, b.mine(b.health).float()[:, None]) * b.enemy
	return fraction(to_me, to_me.sum(dim=1, keepdim=True))


def attack_features(b, shooting):
	"""What the acting unit's attack, a shot or a melee attack, would do to each unit, [B, U, len(ATTACK_FEATURES)];
	0 for units that are not enemies:

	damage           of the target's health left, the engine's average (capped)
	kills            of its creatures
	wipe_chance      p(the whole stack dies): how much of the damage range reaches its whole health (1: whatever the dice)
	value_destroyed  the share of the enemy army's value this removes
	threat_removed   the share of the enemies' damage against the acting unit this removes (BattleAI's "damage
	                 reduce": half for creatures killed, half for health)
	retaliation      of the acting unit's health, the target's retaliation after the hit (melee)
	value_lost       the share of our army's value that retaliation removes
	net_value        value destroyed minus value lost, of both armies' value: -1..1
	"""
	units, me = b.units, b.me
	shots = (torch.ones_like if shooting else torch.zeros_like)(b.enemy)
	average, low, high = _average_damage(me, units, shots, b.tables)
	hp = b.health.float()
	damage = fraction(average, hp)
	killed = fraction(rules.kills(average.long(), units).float(), units["count"].float())
	wipe = fraction(high - hp + 1, high - low + 1)
	value_destroyed = fraction(b.value * damage, b.enemy_value[:, None])
	threat_removed = threat_share(b) * (0.5 * killed + 0.5 * damage)

	my_health = b.mine(b.health).float()[:, None]
	if shooting:
		retaliation = torch.zeros_like(damage)
	else:
		hurt = rules.after_damage(units, average.long())
		r_low, r_high = rules.damage_range(hurt, me, torch.zeros_like(shots), b.tables, dtype=torch.float32)
		retaliates = (hurt["count"] > 0) & (b.x["u_retaliations"] > 0) & ~me["blocks_retaliation"]
		retaliation = fraction((r_low + r_high).float() / 2 * retaliates, my_health)
	my_value = b.mine(b.value)[:, None]
	value_lost = fraction(my_value * retaliation, b.our_value[:, None])
	net = (b.value * damage - my_value * retaliation) / (b.our_value + b.enemy_value)[:, None].clamp(min=EPS)
	features = torch.stack([damage, killed, wipe, value_destroyed, threat_removed, retaliation, value_lost, net.clamp(-1, 1)], dim=-1)
	return features * b.enemy[..., None]


# ---- Per hex, for the acting unit ----

HEX_FEATURES = ("reachable", "threat_value", "melee_threat_value", "danger", "blocked_shooters", "next_to_enemy", "new_targets", "progress")


def threat_maps(b, reach, to_enemy):
	"""Per hex [B, H, len(HEX_FEATURES)], for the acting unit standing there:

	reachable         it can move there this turn (m_move_hex)
	threat_value        the share of the enemy army's value that could attack it there in its coming turn
	melee_threat_value  the same for attacks in melee only: what a shooter (or anyone) escapes by keeping its distance
	danger              of its health, the damage those enemies could do to it there
	blocked_shooters    the share of the enemy army's value in shooters next to the hex, which it would block
	next_to_enemy       1 if an enemy stands next to the hex (a shooter there cannot shoot)
	new_targets         the share of the enemy army's value it could attack next turn from there and cannot now
	progress            steps it gains there toward the nearest enemy, of its speed: -1..1
	Also returned: can_attack [B, U, H] (can_attack_at, every unit: unit_summaries, action_features).
	"""
	reaches = melee_reach(b, reach)  # [B, U, H], every unit: can attack a unit on that hex
	can_attack = reaches | b.units["can_shoot"][..., None]
	# The acting unit standing on a position can be attacked through either of its hexes
	me_wide, me_side = (b.me["rear"] >= 0).expand_as(b.enemy), b.me["side"].expand_as(b.enemy)
	melee = covered(reaches, me_wide, me_side) & b.enemy[..., None]
	can = covered(can_attack, me_wide, me_side) & b.enemy[..., None]
	value = b.value[..., None]
	enemy_value = b.enemy_value[:, None].clamp(min=EPS)
	threat_value = (value * can).sum(dim=1) / enemy_value
	melee_threat_value = (value * melee).sum(dim=1) / enemy_value
	shooting = b.units["can_shoot"]
	average, _, _ = _average_damage(b.units, b.me, shooting, b.tables)
	danger = fraction((average[..., None] * can).sum(dim=1), b.mine(b.health).float()[:, None])

	hexes = torch.arange(N_HEXES, device=reach.device)
	on = (b.units["hex"][..., None] == hexes) | (b.units["rear"][..., None] == hexes)  # [B, U, H]
	around = rules.neighbours(hexes)
	next_to = rules.lookup(on.flatten(0, 1), around.expand(on.shape[0] * on.shape[1], -1, -1)).any(dim=-1).view(on.shape)
	next_to = covered(next_to, me_wide, me_side)  # next to the acting unit's footprint on that position
	blocked = (value * (next_to & b.enemy[..., None] & shooting[..., None])).sum(dim=1) / enemy_value
	next_to_enemy = (next_to & b.enemy[..., None]).any(dim=1).float()

	my_speed = b.me["speed"]
	attackable_now = b.x["m_melee_target"] | b.x["m_shoot_target"]
	next_turn = (to_enemy <= my_speed[..., None]) & b.enemy[..., None] & ~attackable_now[..., None] & ~b.me["can_shoot"][..., None]
	new_targets = (value * next_turn).sum(dim=1) / enemy_value
	nearest = torch.where(b.enemy[..., None], to_enemy, rules.INFINITE_DISTANCE).amin(dim=1).clamp(max=REACH_STEPS + 1).float()  # [B, H]
	here = nearest.gather(1, b.me["hex"].clamp(min=0))  # [B, 1]: from where it stands
	progress = ((here - nearest) / my_speed.clamp(min=1).float()).clamp(-1, 1)
	maps = torch.stack([b.x["m_move_hex"].float(), threat_value.clamp(0, 1), melee_threat_value.clamp(0, 1), danger, blocked.clamp(0, 1), next_to_enemy, new_targets.clamp(0, 1), progress], dim=-1)
	return maps, can_attack


# ---- Per unit, against everyone ----

SUMMARY_FEATURES = ("best_hit", "worst_hit", "hit_by_me", "hits_me")


def unit_summaries(b, can_attack):
	"""Per unit [B, U, len(SUMMARY_FEATURES)]:

	best_hit   the most of an enemy's health it can take in one attack this coming turn (enemies in its reach)
	worst_hit  the most of its own health one enemy in reach can take in one attack
	hit_by_me  of its health, what the acting unit's best attack (melee or shot) would take
	hits_me    of the acting unit's health, what its attack would take
	"""
	attacker = {k: v[:, :, None] for k, v in b.units.items()}  # [B, U, 1]
	target = {k: v[:, None, :] for k, v in b.units.items()}  # [B, 1, U]
	average, _, _ = _average_damage(attacker, target, b.units["can_shoot"][:, :, None].expand(-1, -1, b.health.shape[1]), b.tables)
	share = fraction(average, b.health.float()[:, None, :])  # [B, attacker, target]
	controller = b.x["u_owner_side"]
	opposed = (controller[:, :, None] != controller[:, None, :]) & b.units["alive"][:, :, None] & b.units["alive"][:, None, :]
	in_reach = can_attack_at_units(b, can_attack)  # [B, attacker, target]
	usable = share * (opposed & in_reach)
	acting_row = b.mine(share)  # [B, U]: the acting unit against everyone
	acting_column = share[torch.arange(share.shape[0], device=share.device), :, b.acting]  # everyone against the acting unit
	return torch.stack([usable.amax(dim=2), usable.amax(dim=1), acting_row * opposed[torch.arange(share.shape[0]), b.acting], acting_column], dim=-1)


def can_attack_at_units(b, can_attack):
	"""[B, attacker, target]: whether each unit can attack each unit where it stands, through either of its hexes."""
	at = lambda hexes: per_unit(can_attack, hexes[:, None, :].expand(-1, can_attack.shape[1], -1))
	return at(b.units["hex"]) | at(b.units["rear"])


# ---- The battle ----

BATTLE_FEATURES = ("strength", "round", "tactics", "hero_can_cast", "danger_here", "threat_here", "my_share")


def battle_features(b, maps):
	"""[B, len(BATTLE_FEATURES)]: our army's value against both (0.5 even), the round of 20, the tactics phase, whether
	our hero may cast, the acting unit's danger and threat where it stands, its share of our army's value."""
	here = b.me["hex"].clamp(min=0)[:, :, None].expand(-1, -1, maps.shape[-1])
	at_my_hex = maps.gather(1, here)[:, 0]  # [B, HEX_FEATURES]
	return torch.stack([
		relative(b.our_value, b.enemy_value),
		fraction(b.x["g_round"].float(), torch.full_like(b.our_value, 20)),
		fraction(b.x["g_tactic_distance"].float(), torch.full_like(b.our_value, 7)),
		(b.x["g_hero_can_cast"] != 0).float(),
		at_my_hex[:, HEX_FEATURES.index("danger")],
		at_my_hex[:, HEX_FEATURES.index("threat_value")],
		b.mine(value_in_army(b)),
	], dim=-1)


# ---- Per action ----

# A move's (and an attack's hex's) features are changes from where the unit stands: staying (wait, defend) is the
# baseline, a hex is better or worse than that, -1..1. Progress toward the enemy is split by whether the unit can shoot
# (a melee unit closes in to hit, a shooter keeps away to shoot), with "next to an enemy" for shooters (cannot shoot).
FROM_FEATURES = ("threat_change", "melee_threat_change", "danger_change", "blocked_change", "new_targets")
MOVE_FEATURES = ("threat_change", "melee_threat_change", "danger_change", "blocked_change", "new_targets", "progress", "shooter_progress", "shooter_next_to_enemy")
SHOT_FEATURES = tuple(f for f in ATTACK_FEATURES if f not in ("retaliation", "value_lost"))
TYPE_FEATURES = ("danger_here", "threat_here", "strength")


def action_features(b, melee, shot, maps, can_attack, battle):
	"""Each action's own features, in the shape of its logits:

	melee_target [B, U, ATTACK_FEATURES]
	melee_from   [B, U, H, FROM_FEATURES]   where it stands after the attack, against where it stands now; the threat
	                                        counts the target as it is after the attack (what it loses no longer threatens)
	shoot_target [B, U, SHOT_FEATURES]
	move         [B, H, MOVE_FEATURES]      the hex against where it stands now
	type         [B, TYPE_FEATURES]
	"""
	pick = lambda names, source, of: torch.stack([source[..., of.index(n)] for n in names], dim=-1)
	column = lambda name: maps[..., HEX_FEATURES.index(name)]  # [B, H]
	here = lambda name: column(name).gather(1, b.me["hex"].clamp(min=0))  # [B, 1]: where it stands
	change = lambda name: column(name) - here(name)  # [B, H], -1..1
	value = b.value[..., None]
	me_wide, me_side = (b.me["rear"] >= 0).expand_as(b.enemy), b.me["side"].expand_as(b.enemy)
	reaches_me = covered(can_attack, me_wide, me_side)  # [B, U, H]: each unit could attack the acting unit there
	losses = value * melee[..., ATTACK_FEATURES.index("kills"), None] * reaches_me  # the threat value the target loses
	threat_after = (column("threat_value")[:, None, :] - losses / b.enemy_value[:, None, None].clamp(min=EPS)).clamp(0, 1)
	shooter = b.me["can_shoot"].float()  # [B, 1]
	hexes = torch.stack([
		change("threat_value"), change("melee_threat_value"), change("danger"), change("blocked_shooters"), column("new_targets"),
		column("progress") * (1 - shooter), column("progress") * shooter, column("next_to_enemy") * shooter,
	], dim=-1)  # MOVE_FEATURES
	per_target = hexes[:, None, :, : len(FROM_FEATURES)].expand(-1, b.health.shape[1], -1, -1)
	melee_from = torch.cat([(threat_after - here("threat_value")[..., None])[..., None], per_target[..., 1:]], dim=-1)
	return {
		"melee_target": melee,
		"melee_from": melee_from,
		"shoot_target": pick(SHOT_FEATURES, shot, ATTACK_FEATURES),
		"move": hexes,
		"type": pick(TYPE_FEATURES, battle, BATTLE_FEATURES),
	}


# ---- Spells: what each cast would do ----

SPELL_FEATURES = ("net_value", "versus_best", "changes_health")


def spell_features(b):
	"""Per spell target, from the engine's prediction of the cast (inputs s_*_value, c_*_value: the value change, in
	gold as the reward counts it, for the attacker minus for the defender; damage to a side's own units counts
	against it):

	net_value       our gain, of both armies' value: -1..1
	versus_best     our gain against the most this spell gains or costs us at any legal target: -1..1 (1: the best one)
	changes_health  1 if the spell changes any health at any legal target (damage, healing, resurrection, summons);
	                for the others (Slow, Bless, ...) versus_best is 0 here and the model puts a learned value instead

	spell_unit [B, S, U, 3], spell_hex [B, S, H, 3], creature_spell_unit [B, U, 3], creature_spell_hex [B, H, 3].
	"""
	x = b.x
	sign = (1 - 2 * x["g_our_side"].clamp(0, 1)).float()  # +1 when we attack
	both = (b.our_value + b.enemy_value).clamp(min=EPS)

	def per_spell(value, legal):  # value, legal [B, ..., targets]
		ours = value.float() * sign.view(-1, *[1] * (value.dim() - 1)) * legal
		best = ours.abs().amax(dim=-1, keepdim=True)
		net = (ours / both.view(-1, *[1] * (value.dim() - 1))).clamp(-1, 1)
		changes = (best > 0).float().expand_as(ours)
		return torch.stack([net, ours / best.clamp(min=EPS), changes], dim=-1)

	return {
		"spell_unit": per_spell(x["s_unit_value"], x["m_spell_unit"]),
		"spell_hex": per_spell(x["s_hex_value"], x["m_spell_hex"]),
		"creature_spell_unit": per_spell(x["c_unit_value"], x["m_creature_spell_unit"]),
		"creature_spell_hex": per_spell(x["c_hex_value"], x["m_creature_spell_hex"]),
	}


def our_frame(hexes, our_side):
	"""Where hexes [B, ...] are, seen from our starting side: (column toward the enemy's side, row), each 0..1. The
	attacker starts on the left, so for the defender the columns count from the right: "forward" is the same number
	for both."""
	column = (hexes % 17).float() / 16
	row = (hexes // 17).float() / 10
	defender = (our_side.view(-1, *[1] * (hexes.dim() - 1)) == 1)
	return torch.stack([torch.where(defender, 1 - column, column), row], dim=-1) * (hexes >= 0)[..., None]


# ---- Each side's fighting strength ----

STRENGTH_VARIANTS = ("all", "acts_this_round", "melee", "shooting")


def side_strengths(b):
	"""Lanchester's square law: in a fight of attrition, the side with the larger health x damage per round wins, and
	our share of the fight is about (H D)_ours / ((H D)_ours + (H D)_theirs). Per variant, [B, variants, 2]: the logs of
	our health over theirs and of our damage per round over theirs (each -10..10), so the model's share
	sigmoid(a log H + b log D) has a learned exponent per term, starting at 1 (the law itself).

	Health: the living units' hit points. Damage per round: each unit's engine-average damage (all its attacks, a shot
	if it can shoot) on each living unit of the other side, capped at that unit's health and weighted by its share of
	its side's health; summed over the variant's units: all, those still to act this round, melee units, shooters.
	"""
	attacker = {k: v[:, :, None] for k, v in b.units.items()}
	target = {k: v[:, None, :] for k, v in b.units.items()}
	average, _, _ = _average_damage(attacker, target, b.units["can_shoot"][:, :, None].expand(-1, -1, b.health.shape[1]), b.tables)
	hp = b.health.float()
	capped = torch.minimum(average, hp[:, None, :])  # [B, attacker, target]
	ours, enemy = b.ours.float(), b.enemy.float()

	def onto(other):  # [B, U]: each unit's damage per round on the `other` side's units, by their health share
		weight = hp * other
		return (capped * (weight / weight.sum(dim=1, keepdim=True).clamp(min=EPS))[:, None, :]).sum(dim=2)

	damage = onto(enemy) * ours + onto(ours) * enemy  # each unit on its opponents
	health = torch.log((hp * ours).sum(dim=1) + 1) - torch.log((hp * enemy).sum(dim=1) + 1)
	shooters = b.units["can_shoot"].float()
	subsets = (torch.ones_like(hp), (b.x["u_queue_now"] >= 0).float(), 1 - shooters, shooters)
	rows = []
	for subset in subsets:
		ours_d, theirs_d = (damage * ours * subset).sum(dim=1), (damage * enemy * subset).sum(dim=1)
		floor = 0.01 * (ours_d + theirs_d) + EPS  # a side without such units: a large, not an infinite, ratio
		rows.append(torch.stack([health, torch.log(ours_d + floor) - torch.log(theirs_d + floor)], dim=-1))
	return torch.stack(rows, dim=1).clamp(-10, 10)
