"""Bounded, scale-free features of a battle, computed from the raw inputs with the game's own rules (rules.py) and
the engine's own predictions (damage estimates p_*, attacks played on a copy of the battle a_*, spells s_*/c_*): the
numbers a hand-written battle AI works with, for the model (model.py). No parameters.

Every feature is bounded (0..1, a few -1..1) and built only from this battle: ratios of quantities of the same kind
(hit points to hit points, value to value, damage to the health it hits), never raw counts or hit points. So the
features stay about the same if every stack is ten times larger (hero spells, whose damage does not grow with the
stacks, are the exception), and nothing needs batch statistics, which a real battle would not match anyway.
A quantity that has no such partner (a hero's spell power, mana, a stack's attack, the round) and can be anything from
1 to 1000 comes at several scales (scales: x / (x + c) for c a few powers of 4 apart), so that 4 and 10 mana differ as
much as 400 and 1000 do, and nothing saturates.

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


def scales(x, cs):
	"""[..., len(cs)]: x / (x + c) for each scale c, each 0..1 (x >= 0): the same quantity seen at each scale."""
	x = x.float().clamp(min=0)[..., None]
	return x / (x + torch.tensor(cs, dtype=torch.float32, device=x.device))


def signed_scales(x, cs):
	"""The same for a quantity of either sign: x / (|x| + c), each -1..1."""
	x = x.float()[..., None]
	return x / (x.abs() + torch.tensor(cs, dtype=torch.float32, device=x.device))


def damage_factor(advantage):
	"""log of the damage formula's factor for attack minus defence (DamageCalculator: +5% a point above, at most +400%;
	-2.5% a point below, at most -70%): -1.2..1.6, 0 even."""
	a = advantage.float()
	return torch.log1p((0.05 * a.clamp(min=0)).clamp(max=4.0)) + torch.log1p(-(0.025 * (-a).clamp(min=0)).clamp(max=0.7))


STAT_SCALES = (5, 20, 80)  # attack and defence: a level-1 creature has ~5, a high-level hero adds 50 and more
COUNT_SCALES = (2, 8, 32)  # shots, rounds, spell power, ...
MANA_SCALES = (5, 25, 125)  # a cheap spell costs ~5, a pool runs to the hundreds
HIT_SCALES = (1, 4)  # units an attack or a cast hits: one, a few, many
HIT_FEATURES = tuple(f"{who}_hit_{c}" for who in ("enemies", "allies") for c in HIT_SCALES)


def hits(enemies, allies):
	"""[..., len(HIT_FEATURES)]: how many enemy units and how many of our units an attack or a cast hits (the engine's
	affected units: area attacks, breath, chains, ...), each at HIT_SCALES."""
	return torch.cat([scales(enemies, HIT_SCALES), scales(allies, HIT_SCALES)], dim=-1)


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


def adjacency(b):
	"""[B, U, V]: whether unit V stands next to unit U (any hex of one next to any hex of the other), living units."""
	hexes = torch.arange(N_HEXES, device=b.health.device)
	on = ((b.units["hex"][..., None] == hexes) | (b.units["rear"][..., None] == hexes)) & b.units["alive"][..., None]  # [B, U, H]
	around = rules.neighbours(hexes)  # [H, 6]
	near = rules.lookup(on.flatten(0, 1), around.expand(on.shape[0] * on.shape[1], -1, -1)).any(dim=-1).view(on.shape)  # hex next to U
	return (near.float() @ on.float().transpose(1, 2)) > 0


def _neighbours(b, same_side):
	"""[B, U]: the value of the units next to each living unit on its own side (same_side) or the other, of that
	side's army value. Sides by who controls the units now."""
	slots = torch.arange(b.health.shape[1], device=b.health.device)
	adjacent = adjacency(b) & (slots[:, None] != slots[None, :])
	ours, enemy = b.ours[:, :, None], b.enemy[:, :, None]  # [B, U, 1]
	if same_side:
		pair, total = (ours & b.ours[:, None, :]) | (enemy & b.enemy[:, None, :]), torch.where(b.ours, b.our_value[:, None], b.enemy_value[:, None])
	else:
		pair, total = (ours & b.enemy[:, None, :]) | (enemy & b.ours[:, None, :]), torch.where(b.ours, b.enemy_value[:, None], b.our_value[:, None])
	return fraction((adjacent & pair).float().mul(b.value[:, None, :]).sum(dim=2), total) * (b.ours | b.enemy)


@stack_feature
def allies_next_to(b):
	"""Of its side's army value, the units of its side next to it: grouped units, which an area attack (a Hydra's, a
	breath, a Fireball) hits together."""
	return _neighbours(b, True)


@stack_feature
def enemies_next_to(b):
	"""Of the other side's army value, its units next to it."""
	return _neighbours(b, False)


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


def multi_stack_feature(name, cs, signed=False):
	"""Registers one stack feature per scale (name_c): f(b) [B, U] at each of the scales cs."""

	def register(f):
		for i, c in enumerate(cs):
			STACK[f"{name}_{c}"] = (lambda i: lambda b: (signed_scales if signed else scales)(f(b), cs)[..., i])(i)
		return f

	return register


@multi_stack_feature("attack", STAT_SCALES)
def attack(b):
	"""Attack, at several scales."""
	return b.units["attack"]


@multi_stack_feature("defense", STAT_SCALES)
def defense(b):
	"""Defence, at several scales."""
	return b.units["defense"]


@multi_stack_feature("attack_buff", STAT_SCALES, signed=True)
def attack_buff(b):
	"""Its attack over the creature's own (hero, spells), at several scales: -1..1."""
	return b.units["attack"] - b.units["base_attack"]


@multi_stack_feature("defense_buff", STAT_SCALES, signed=True)
def defense_buff(b):
	"""The same for defence."""
	return b.units["defense"] - b.units["base_defense"]


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
	"""Hexes it moves this round, of 15: the battlefield's playable width, so 1 reaches any hex and more speed than
	that changes nothing (a fixed scale on purpose)."""
	return fraction(b.units["speed"].float(), torch.full_like(b.health, 15, dtype=torch.float32))


@multi_stack_feature("ammunition", COUNT_SCALES)
def ammunition(b):
	"""Shots left, at several scales."""
	return b.x["u_shots"]


@stack_feature
def can_shoot(b):
	return b.units["can_shoot"].float()


@stack_feature
def retaliations(b):
	return fraction(b.x["u_retaliations"].float(), torch.full_like(b.health, MAX_RETALIATIONS, dtype=torch.float32))


@multi_stack_feature("casts", (1, 4))
def casts(b):
	"""Creature spell casts left, at two scales."""
	return b.x["u_casts"]


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

ATTACK_FEATURES = ("damage", "kills", "wipe_chance", "value_destroyed", "threat_removed", "retaliation", "value_lost", "net_value", "versus_best", *HIT_FEATURES)


def pair_damage(b, shooting):
	"""The engine's damage estimate of every unit hitting every unit as they stand now (inputs p_*: every damage bonus,
	the heroes' skills and artifacts included; no luck, no charge), with all its attacks (Marksmen shoot twice):
	(average, low, high), [B, attacker, target] each; a shot where shooting (broadcast against that), else melee."""
	x = b.x
	low = torch.where(shooting, x["p_shot_min"], x["p_melee_min"]).float()
	high = torch.where(shooting, x["p_shot_max"], x["p_melee_max"]).float()
	attacks = (1 + b.units["extra_attacks"]).float()[:, :, None]
	return (low + high) / 2 * attacks, low * attacks, high * attacks


def best_damage(b):
	"""[B, attacker, target]: the average damage of each unit's attack on each unit, a shot if it can shoot."""
	return pair_damage(b, b.units["can_shoot"][:, :, None])[0]


def threat_share(b):
	"""[B, U]: each enemy's share of the damage the enemies could do to the acting unit (each capped at its health)."""
	to_me = torch.minimum(b.mine(best_damage(b).transpose(1, 2)), b.mine(b.health).float()[:, None]) * b.enemy
	return fraction(to_me, to_me.sum(dim=1, keepdim=True))


def attack_values(b):
	"""The engine's value of each legal attack of the acting unit (inputs a_*: played on a copy of the battle by the
	game's own rules, dice at their expectation), our gain of both armies' value, -1..1: (melee [B, U, H], shot [B, U])."""
	x = b.x
	sign = (1 - 2 * x["g_our_side"].clamp(0, 1)).float()  # +1 when we attack
	both = (b.our_value + b.enemy_value).clamp(min=EPS)
	norm = lambda value, legal: (value.float() * legal * sign.view(-1, *[1] * (value.dim() - 1)) / both.view(-1, *[1] * (value.dim() - 1))).clamp(-1, 1)
	return norm(x["a_melee_value"], x["m_melee_from"]), norm(x["a_shoot_value"], x["m_shoot_target"])


def versus_best(values, legal):
	"""values [B, n] against the largest gain or loss among the legal ones: -1..1, 1 for the best."""
	best = (values.abs() * legal).amax(dim=-1, keepdim=True)
	return values * legal / best.clamp(min=EPS)


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
	net_value        what the engine says the attack gains us (attack_values: everything included, retaliation, area
	                 attacks, Fire Shield, life drain, ...), of both armies' value, -1..1; for melee from its best hex
	versus_best      net_value against the best legal target's: -1..1, 1 for the best
	*_hit_*          how many enemy units and how many of ours the attack hits (hits; for melee from its best hex)
	The others come from the engine's damage estimates of the attack on the target alone.
	"""
	units, me = b.units, b.me
	shots = (torch.ones_like if shooting else torch.zeros_like)(b.enemy)
	average, low, high = (b.mine(t) for t in pair_damage(b, shots[:, None, :]))  # [B, U] each: the acting unit's row
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
		# The target's melee estimate on the acting unit, for the creatures left after the hit
		hurt = rules.after_damage(units, average.long())
		their = (b.x["p_melee_min"] + b.x["p_melee_max"]).float() / 2  # [B, attacker, target]
		on_me = their[torch.arange(their.shape[0], device=their.device), :, b.acting]  # [B, U]: each unit on the acting unit
		retaliates = (hurt["count"] > 0) & (b.x["u_retaliations"] > 0) & ~me["blocks_retaliation"]
		retaliation = fraction(on_me * fraction(hurt["count"].float(), units["count"].float()) * retaliates, my_health)
	my_value = b.mine(b.value)[:, None]
	value_lost = fraction(my_value * retaliation, b.our_value[:, None])
	melee_values, shot_values = attack_values(b)
	legal = b.x["m_shoot_target"] if shooting else b.x["m_melee_target"]
	if shooting:
		net, enemies, allies = shot_values, b.x["a_shoot_enemies"], b.x["a_shoot_allies"]
	else:
		masked = melee_values.masked_fill(~b.x["m_melee_from"], -2.0)
		best = masked.argmax(dim=-1, keepdim=True)  # [B, U, 1]: its best hex
		net = masked.amax(dim=-1).clamp(min=-1) * legal
		enemies, allies = (b.x[name].gather(-1, best)[..., 0] * legal for name in ("a_melee_enemies", "a_melee_allies"))
	features = torch.cat([torch.stack([damage, killed, wipe, value_destroyed, threat_removed, retaliation, value_lost, net, versus_best(net, legal)], dim=-1), hits(enemies, allies)], dim=-1)
	return features * b.enemy[..., None]


# ---- Per hex, for the acting unit ----

HEX_FEATURES = ("reachable", "threat_value", "melee_threat_value", "danger", "blocked_shooters", "next_to_enemy", "new_targets", "progress", "allies_next_to", "enemies_next_to")


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
	allies_next_to      the share of our army's value (other units) next to it there: grouped with it, an area attack
	                    hits them together
	enemies_next_to     the share of the enemy army's value next to it there
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
	average = b.mine(best_damage(b).transpose(1, 2))  # [B, U]: each unit on the acting unit
	danger = fraction((average[..., None] * can).sum(dim=1), b.mine(b.health).float()[:, None])

	hexes = torch.arange(N_HEXES, device=reach.device)
	on = (b.units["hex"][..., None] == hexes) | (b.units["rear"][..., None] == hexes)  # [B, U, H]
	around = rules.neighbours(hexes)
	next_to = rules.lookup(on.flatten(0, 1), around.expand(on.shape[0] * on.shape[1], -1, -1)).any(dim=-1).view(on.shape)
	next_to = covered(next_to, me_wide, me_side)  # next to the acting unit's footprint on that position
	blocked = (value * (next_to & b.enemy[..., None] & shooting[..., None])).sum(dim=1) / enemy_value
	next_to_enemy = (next_to & b.enemy[..., None]).any(dim=1).float()
	others = b.ours & (torch.arange(b.health.shape[1], device=reach.device)[None, :] != b.acting[:, None])
	allies_next = fraction((value * (next_to & others[..., None])).sum(dim=1), b.our_value[:, None])
	enemies_next = fraction((value * (next_to & b.enemy[..., None])).sum(dim=1), enemy_value)

	my_speed = b.me["speed"]
	attackable_now = b.x["m_melee_target"] | b.x["m_shoot_target"]
	next_turn = (to_enemy <= my_speed[..., None]) & b.enemy[..., None] & ~attackable_now[..., None] & ~b.me["can_shoot"][..., None]
	new_targets = (value * next_turn).sum(dim=1) / enemy_value
	nearest = torch.where(b.enemy[..., None], to_enemy, rules.INFINITE_DISTANCE).amin(dim=1).clamp(max=REACH_STEPS + 1).float()  # [B, H]
	here = nearest.gather(1, b.me["hex"].clamp(min=0))  # [B, 1]: from where it stands
	progress = ((here - nearest) / my_speed.clamp(min=1).float()).clamp(-1, 1)
	maps = torch.stack([b.x["m_move_hex"].float(), threat_value.clamp(0, 1), melee_threat_value.clamp(0, 1), danger, blocked.clamp(0, 1), next_to_enemy, new_targets.clamp(0, 1), progress, allies_next, enemies_next], dim=-1)
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
	share = fraction(best_damage(b), b.health.float()[:, None, :])  # [B, attacker, target]
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

BATTLE_FEATURES = ("strength", *(f"round_{c}" for c in COUNT_SCALES), "tactics", "hero_can_cast", "danger_here", "threat_here", "my_share")


def battle_features(b, maps):
	"""[B, len(BATTLE_FEATURES)]: our army's value against both (0.5 even), the round at several scales, the tactics
	phase (of 7 columns, the most the Tactics skill gives: a fixed scale on purpose), whether our hero may cast, the
	acting unit's danger and threat where it stands, its share of our army's value.

	Not the score if the battle ended now (the zero-sum on the assets as they stand): given to the whole model (v11),
	the value heads took it as the answer, and the defence learned to run from every fight, which delays its losses
	in that score but loses the battle all the same (seed 2: it killed 20% of the attacker instead of 60%; without it
	73%, 2026-10-24)."""
	here = b.me["hex"].clamp(min=0)[:, :, None].expand(-1, -1, maps.shape[-1])
	at_my_hex = maps.gather(1, here)[:, 0]  # [B, HEX_FEATURES]
	return torch.cat([
		relative(b.our_value, b.enemy_value)[:, None],
		scales(b.x["g_round"], COUNT_SCALES),
		torch.stack([
			fraction(b.x["g_tactic_distance"].float(), torch.full_like(b.our_value, 7)),
			(b.x["g_hero_can_cast"] != 0).float(),
			at_my_hex[:, HEX_FEATURES.index("danger")],
			at_my_hex[:, HEX_FEATURES.index("threat_value")],
			b.mine(value_in_army(b)),
		], dim=-1),
	], dim=-1)


# ---- What ending the battle now would score ----

OUTCOME_FEATURES = ("retreat_reward", "surrender_reward")


def outcome_features(b, config=None):
	"""[B, len(OUTCOME_FEATURES)], each -1..1, for the RETREAT and SURRENDER logits only: the exact reward (reward.py,
	zero-sum, from each side's assets) of
	retreat_reward    our retreating now: our army is lost, the enemy wins with what it has
	surrender_reward  our surrendering now: we keep the army and pay the surrender cost and the hero's rehire (0 when we
	                  cannot surrender: the mask forbids it anyway)
	Assets as summarize() counts them: the kept creatures (u_kept) at their value, mana at mana_value x spell power,
	heroes as reward.hero_asset; the enemy hero's level and artifacts are hidden (0 in the inputs), so its hero counts 0
	here (the true reward counts it)."""
	from .reward import HERO_HIRE_COST, RewardConfig

	config = config or RewardConfig()
	x = b.x
	price = b.tables["value"][b.units["creature"]]  # [B, U]
	sides = torch.arange(2, device=price.device)
	on_side = (x["u_side"][:, None, :] == sides[None, :, None]).float()  # [B, side, U]
	army = (on_side * (x["u_kept"].float() * price)[:, None, :]).sum(dim=-1)  # [B, side]
	power = x["h_power"].float().clamp(min=0) * (x["h_present"] != 0)
	mana = config.mana_value * power * x["h_mana"].float().clamp(min=0)
	hero = hero_assets(x, config)  # [B, side]
	start = x["g_start_army"].float() + hero + config.mana_value * power * x["g_start_mana"].float().clamp(min=0)
	our = x["g_our_side"].clamp(0, 1)
	batch = torch.arange(our.shape[0], device=our.device)
	pick = lambda t, side: t[batch, side]
	own_start, enemy_start = pick(start, our), pick(start, 1 - our)
	enemy_end = pick(army, 1 - our) + pick(mana, 1 - our)

	def zero_sum(own_end, enemy_end):
		return (((enemy_start - enemy_end) - (own_start - own_end)) / (own_start + enemy_start).clamp(min=EPS)).clamp(-1, 1)

	enemy_end = enemy_end + pick(hero, 1 - our)
	own_hero = pick(hero, our)
	cost = x["g_surrender_cost"].float()
	surrender = zero_sum(pick(army, our) + own_hero + pick(mana, our) - cost.clamp(min=0) - HERO_HIRE_COST, enemy_end) * (cost >= 0)
	return torch.stack([zero_sum(own_hero + pick(mana, our), enemy_end), surrender], dim=-1)


def hero_levels(x):
	"""[B, side]: each hero's level with its way to the next (h_level, h_level_progress); 0 without a hero."""
	return (x["h_level"].float() + x["h_level_progress"].float() / 1000) * (x["h_present"] != 0)


def hero_assets(x, config):
	"""[B, side]: reward.hero_asset of each hero as the inputs show it (the enemy's level and artifacts hidden: 0)."""
	levels = hero_levels(x)
	return config.level_value * (levels - config.free_levels).clamp(min=0) ** config.level_exponent + x["h_artifacts"].float() * (x["h_present"] != 0)


# ---- Per action ----

# A move's (and an attack's hex's) features are changes from where the unit stands: staying (wait, defend) is the
# baseline, a hex is better or worse than that, -1..1. Progress toward the enemy is split by whether the unit can shoot
# (a melee unit closes in to hit, a shooter keeps away to shoot), with "next to an enemy" for shooters (cannot shoot).
MOVE_FEATURES = ("threat_change", "melee_threat_change", "danger_change", "blocked_change", "new_targets", "allies_next_to_change", "enemies_next_to_change", "progress", "shooter_progress", "shooter_next_to_enemy")
FROM_FEATURES = (*MOVE_FEATURES[:7], "hex_value", *HIT_FEATURES)
SHOT_FEATURES = tuple(f for f in ATTACK_FEATURES if f not in ("retaliation", "value_lost"))
TYPE_FEATURES = ("danger_here", "threat_here", "strength")


def action_features(b, melee, shot, maps, can_attack, battle):
	"""Each action's own features, in the shape of its logits:

	melee_target [B, U, ATTACK_FEATURES]
	melee_from   [B, U, H, FROM_FEATURES]   where it stands after the attack, against where it stands now; the threat
	                                        counts the target as it is after the attack (what it loses no longer threatens);
	                                        hex_value: the engine's value of the attack from this hex against its best
	                                        hex (jousting, which units an area attack hits, obstacles): <= 0; and how many
	                                        units the attack from there hits (hits)
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
		change("allies_next_to"), change("enemies_next_to"),
		column("progress") * (1 - shooter), column("progress") * shooter, column("next_to_enemy") * shooter,
	], dim=-1)  # MOVE_FEATURES
	per_target = hexes[:, None, :, : FROM_FEATURES.index("hex_value")].expand(-1, b.health.shape[1], -1, -1)
	melee_values, _ = attack_values(b)
	legal = b.x["m_melee_from"]
	best_hex = melee_values.masked_fill(~legal, -2.0).amax(dim=-1, keepdim=True)
	hex_value = ((melee_values - best_hex) * legal).clamp(-1, 0)
	hit = hits(b.x["a_melee_enemies"], b.x["a_melee_allies"]) * legal[..., None]
	melee_from = torch.cat([(threat_after - here("threat_value")[..., None])[..., None], per_target[..., 1:], hex_value[..., None], hit], dim=-1)
	return {
		"melee_target": melee,
		"melee_from": melee_from,
		"shoot_target": pick(SHOT_FEATURES, shot, ATTACK_FEATURES),
		"move": hexes,
		"type": pick(TYPE_FEATURES, battle, BATTLE_FEATURES),
	}


# ---- Spells: what each cast would do ----

SPELL_FEATURES = ("net_value", "versus_best", "changes_health", "target_value", *HIT_FEATURES)


def spell_features(b):
	"""Per spell target, from the engine's prediction of the cast (inputs s_*_value, c_*_value: the value change, in
	gold as the reward counts it, for the attacker minus for the defender; damage to a side's own units counts
	against it):

	net_value       our gain, of both armies' value: -1..1
	versus_best     our gain against the most this spell gains or costs us at any legal target: -1..1 (1: the best one)
	changes_health  1 if the spell changes any health at any legal target (damage, healing, resurrection, summons);
	                for the others (Slow, Bless, ...) versus_best is 0 here and the model puts a learned value instead
	target_value    the value of the unit it targets (on the hex, for a hex), of both armies' value: what a buff or a
	                curse is cast on (which unit is worth it, and why, is for the model to learn)
	*_hit_*         how many enemy units and how many of ours the cast affects (hits: area spells, chains)

	spell_unit [B, S, U, n], spell_hex [B, S, H, n], creature_spell_unit [B, U, n], creature_spell_hex [B, H, n].
	"""
	x = b.x
	sign = (1 - 2 * x["g_our_side"].clamp(0, 1)).float()  # +1 when we attack
	both = (b.our_value + b.enemy_value).clamp(min=EPS)
	unit_value = b.value * b.units["alive"] / both[:, None]  # [B, U]
	hexes = torch.arange(N_HEXES, device=unit_value.device)
	on = (b.units["hex"][..., None] == hexes) | (b.units["rear"][..., None] == hexes)  # [B, U, H]
	hex_value = (unit_value[..., None] * on).sum(dim=1).clamp(0, 1)  # [B, H]

	def per_spell(name, legal, target):  # inputs <name>_value, _enemies, _allies and legal [B, ..., targets]; target [B, targets]
		value = x[f"{name}_value"]
		ours = value.float() * sign.view(-1, *[1] * (value.dim() - 1)) * legal
		best = ours.abs().amax(dim=-1, keepdim=True)
		net = (ours / both.view(-1, *[1] * (value.dim() - 1))).clamp(-1, 1)
		changes = (best > 0).float().expand_as(ours)
		target = target.view(target.shape[0], *[1] * (value.dim() - 2), target.shape[-1]).expand_as(ours) * legal
		hit = hits(x[f"{name}_enemies"], x[f"{name}_allies"]) * legal[..., None]
		return torch.cat([torch.stack([net, ours / best.clamp(min=EPS), changes, target], dim=-1), hit], dim=-1)

	return {
		"spell_unit": per_spell("s_unit", x["m_spell_unit"], unit_value),
		"spell_hex": per_spell("s_hex", x["m_spell_hex"], hex_value),
		"creature_spell_unit": per_spell("c_unit", x["m_creature_spell_unit"], unit_value),
		"creature_spell_hex": per_spell("c_hex", x["m_creature_spell_hex"], hex_value),
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

	Health: the living units' hit points. Damage per round: each unit's engine-estimated average damage (all its
	attacks, a shot if it can shoot) on each living unit of the other side, capped at that unit's health and weighted by its share of
	its side's health; summed over the variant's units: all, those still to act this round, melee units, shooters.
	"""
	hp = b.health.float()
	capped = torch.minimum(best_damage(b), hp[:, None, :])  # [B, attacker, target]
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
