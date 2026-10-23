"""VCMI's StupidAI (AI/StupidAI/StupidAI.cpp) as a model of the battle contract: the same inputs and outputs as
model.BattleModel, no trainable parameters, the same decisions as StupidAI. The hand-written algorithm as a forward
pass, built from the engine's rules in rules.py.

	.venv/bin/python -m battle_model.stupid_ai OUT.onnx   # export it; it plays like any model (ALPHA_BATTLE_MODEL)

CStupidAI::activeStack, for the acting unit:
  1. If it can shoot an enemy stack: shoot the one with the best profit, the average damage it would deal (the engine's
     estimate, clipped to the stack's health) minus the average retaliation it would take (none for a shot).
  2. Else, if it can reach and hit one this turn: hit the one with the best profit, from the hex (of those it may
     attack from) with the most shooters next to it.
  3. Else walk towards the enemy closest by path: to a hex next to it if it can get there this turn, else as far as it
     can along the path to the nearest such hex (goTowards).
  4. Else defend.
Ties go to the first in the engine's order (std::max_element and std::min_element keep the first): unit slots in
the engine's order, hexes in increasing order.

StupidAI never waits, casts, retreats or surrenders. War machines and the tactics phase are not the model's to play
(the C++ AI scripts them). The chosen action gets all the probability: log-probability 0, every other legal one about
-1000. The value outputs are 0: StupidAI has no notion of value.

Where it can still differ from StupidAI: the damage rules rules.damage_range does not compute (hero skills and
artifacts, Slayer, sieges, ...), stacks the masks exclude but StupidAI would consider (e.g. attacks ending on
quicksand), invincible units, and spell effects whose first bonus is not the one the rules read (u_effect_value).
check_stupid_ai.py measures it against StupidAI itself.
"""
import sys

import torch
from torch import nn

from . import rules
from .constants import MAX_UNITS, N_HEXES, N_VALUE_HORIZONS, ActionType
from .model import masked_log_softmax

UNLIKELY = -1000.0  # logit of every action StupidAI does not take


def first_best(values, allowed):
	"""(index of the first largest allowed entry on the last axis, whether any is allowed)."""
	lowest = torch.iinfo(values.dtype).min if not values.dtype.is_floating_point else -float("inf")
	return torch.where(allowed, values, torch.full_like(values, lowest)).argmax(dim=-1), allowed.any(dim=-1)


def stable_rank(keys, valid):
	"""Each entry's position after a stable sort by keys (std::sort on these short lists is stable), invalid last."""
	big = torch.full_like(keys, rules.INFINITE_DISTANCE * 2)
	keys = torch.where(valid, keys, big)
	index = torch.arange(keys.shape[-1], device=keys.device)
	before = (keys[..., None, :] < keys[..., :, None]) | ((keys[..., None, :] == keys[..., :, None]) & (index[None, :] < index[:, None]))
	return before.sum(dim=-1)


class StupidAI(nn.Module):
	def __init__(self):
		super().__init__()
		for name, table in rules.creature_tables().items():
			self.register_buffer(name, table, persistent=False)

	def tables(self):
		return {name: getattr(self, name) for name in (*rules.CREATURE_TABLES, "hate")}

	def forward(self, x):
		units = rules.units_of(x, self.tables())
		if "u_base_count" in x:
			units["base_count"] = x["u_base_count"]
		me = rules.select(units, x["g_active_unit"].clamp(0, MAX_UNITS - 1))
		batch = torch.arange(units["count"].shape[0], device=units["count"].device)
		enemy = units["alive"] & (units["side"] != x["g_our_side"][:, None])

		# 1. Shoot the most profitable stack it can shoot
		shoot_target, can_shoot = first_best(self.profit(me, units, shooting=True), x["m_shoot_target"])

		# 2. Else hit the most profitable stack it can reach, from the hex with the most shooters around
		melee_target, can_hit = first_best(self.profit(me, units, shooting=False), x["m_melee_target"])
		melee_from, _ = first_best(self.shooters_around(units), x["m_melee_from"][batch, melee_target])

		# 3. Else towards the enemy closest by path (moveStackToClosestEnemy)
		flying = me["flying"][:, 0] | rules.effect(me, rules.FLY)[1][:, 0]
		distances, predecessors = rules.reachability(x, me, flying)
		to_reach = self.distance_to_reach(me, units, distances)
		closest, any_enemy = first_best(-to_reach, enemy & (units["hex"] >= 0))
		closest_distance = to_reach[batch, closest]
		destination = self.go_towards(me, rules.select(units, closest), distances, predecessors, flying)
		walks = any_enemy & (closest_distance < N_HEXES) & (destination >= 0)

		kind = torch.where(can_shoot, ActionType.SHOOT, torch.where(can_hit, ActionType.MELEE, torch.where(walks, ActionType.MOVE, ActionType.DEFEND)))
		return self.outputs(x, kind, destination, torch.where(can_shoot, shoot_target, melee_target), melee_from)

	def profit(self, me, units, shooting):
		"""EnemyInfo::calcDmg and isMoreProfitable, for the acting unit against every unit [B, U]: the average damage dealt
		minus the average retaliation, each clipped to the target's getTotalHealth() (the retaliation too: StupidAI
		clips it with the target's, `ourHealth = s->getTotalHealth()`, as written).

		getTotalHealth() is the stack's full health at its size at the start of the battle (max HP x unitBaseAmount),
		which the inputs do not have: max HP x count is the same until the stack has lost a creature, and lower after.
		Given the starting sizes (x["u_base_count"], not in the contract; check_stupid_ai.py adds them), it is exact."""
		tables = self.tables()
		shots = (torch.ones_like if shooting else torch.zeros_like)(units["count"], dtype=torch.bool)
		low, high = rules.damage_range(me, units, shots, tables)
		health = units["max_hp"] * units.get("base_count", units["count"])
		dealt = (torch.minimum(low, health) + torch.minimum(high, health)) // 2
		if shooting:  # battleEstimateDamage: no retaliation to a shot
			return dealt
		# The target's retaliation after being hit by the least and by the most (estimateRetaliation)
		retaliations = []
		for hit in (low, high):
			hurt = rules.after_damage(units, hit)
			r_low, r_high = rules.damage_range(hurt, me, torch.zeros_like(shots), tables)
			survives = hurt["count"] > 0
			retaliations.append((torch.where(survives, r_low, 0), torch.where(survives, r_high, 0)))
		r_low = torch.minimum(retaliations[0][0], retaliations[1][0])
		r_high = torch.maximum(retaliations[0][1], retaliations[1][1])
		retaliates = units["alive"] & (units["retaliations"] > 0) & ~me["blocks_retaliation"]
		taken = (torch.minimum(r_low, health) + torch.minimum(r_high, health)) // 2
		return dealt - torch.where(retaliates, taken, 0)

	def shooters_around(self, units):
		"""willSecondHexBlockMoreEnemyShooters: per hex [B, N_HEXES], how many living shooters (of either side) stand on
		the hexes next to it."""
		hexes = torch.arange(N_HEXES, device=units["hex"].device)
		shooter = units["alive"] & units["shooter"]
		on = (units["hex"][..., None] == hexes) | (units["rear"][..., None] == hexes)  # [B, U, H]
		shooter_at = (on & shooter[..., None]).any(dim=1)  # [B, H]
		return rules.lookup(shooter_at, rules.neighbours(hexes).expand(shooter_at.shape[0], -1, -1)).sum(dim=-1)

	def distance_to_reach(self, me, units, distances):
		"""ReachabilityInfo::distToNearestNeighbour(me, unit), for every unit [B, U]: the path length to the nearest
		hex next to the unit's hexes in its row (for a two-hex attacker, also the hexes beside them)."""
		wide = units["rear"] >= 0
		second = rules.rear_hex(torch.where(wide, units["rear"], units["hex"]), units["side"])
		targets = torch.stack([units["hex"], torch.where(wide, units["rear"], -1), torch.where(me["rear"] >= 0, torch.where(wide, units["rear"], units["hex"]), -1), torch.where(me["rear"] >= 0, second, -1)], dim=-1)  # [B, U, 4]
		row = units["hex"] // rules.FIELD_WIDTH
		targets = torch.where((targets >= 0) & (targets // rules.FIELD_WIDTH == row[..., None]) & rules.available(targets), targets, -1)
		around = rules.neighbours(targets)  # [B, U, 4, 6]
		return rules.lookup(distances, around.flatten(1), fill=rules.INFINITE_DISTANCE).view(around.shape).flatten(2).amin(dim=-1)

	def attackable_hexes(self, me, target):
		"""Unit::getAttackableHexes(me) of the target [B, 16]: the hexes around it, and for a two-hex attacker the hexes
		its head can stand on to attack from there; engine order, duplicates dropped, -1 padded."""
		side, hex_ = target["side"][:, 0], target["hex"][:, 0]
		wide_target = target["rear"][:, 0] >= 0
		single = torch.cat([rules.NEIGHBOURS.to(hex_.device)[hex_.clamp(min=0)], torch.full((hex_.shape[0], 2), -1, device=hex_.device)], dim=1)
		double = rules.WIDE_NEIGHBOURS.to(hex_.device)[side.clamp(0, 1), hex_.clamp(min=0)]
		around = torch.where(wide_target[:, None], double, single)  # [B, 8]
		around = torch.where(hex_[:, None] >= 0, around, -1)
		covers = lambda h: (h == hex_[:, None]) | (wide_target[:, None] & (h == target["rear"]))
		my_side = me["side"]
		occupied = torch.where(around >= 0, rules.rear_hex(around, my_side), -1)
		head = torch.where(my_side == rules.ATTACKER, rules.neighbours(around)[..., rules.RIGHT], rules.neighbours(around)[..., rules.LEFT])  # headDirection
		origin_ok = (around >= 0) & ~covers(occupied) & rules.available(occupied)
		head_ok = (around >= 0) & ~covers(head) & rules.available(head)
		pairs = torch.stack([torch.where(origin_ok, around, -1), torch.where(head_ok, head, -1)], dim=-1).flatten(1)  # [B, 16]
		hexes = torch.where(me["rear"] >= 0, pairs, torch.cat([around, torch.full_like(around, -1)], dim=1))
		index = torch.arange(hexes.shape[1], device=hexes.device)
		seen_before = ((hexes[:, None, :] == hexes[:, :, None]) & (index[None, :] < index[:, None])).any(dim=-1)
		return torch.where(seen_before, -1, hexes)

	def go_towards(self, me, target, distances, predecessors, flying):
		"""CStupidAI::goTowards(me, the target's attackable hexes): the hex to move to [B], or -1 to defend."""
		hexes = self.attackable_hexes(me, target)
		valid = hexes >= 0
		position, speed = me["hex"], me["speed"]
		reachable = (distances < rules.INFINITE_DISTANCE) & (distances <= speed)  # battleGetAvailableHexes, its own hex too
		rank = stable_rank(rules.lookup(distances, hexes, fill=rules.INFINITE_DISTANCE), valid)
		big = torch.full_like(rank, 10**6)
		# The first, in that order, it can reach (go there) or already stands on (defend)
		there = valid & rules.lookup(reachable, hexes)
		covered = valid & ((hexes == position) | ((me["rear"] >= 0) & (hexes == me["rear"])))
		first, stops = first_best(-torch.where(there | covered, rank, big), there | covered)
		first_hex = hexes.gather(1, first[:, None])
		go_there = torch.where(rules.lookup(reachable, first_hex)[:, 0] & (first_hex[:, 0] != position[:, 0]), first_hex[:, 0], -1)

		# Otherwise towards the nearest of them: flying to the reachable hex nearest it, walking back along the path
		nearest = hexes.gather(1, first_best(-rank, valid)[0][:, None])  # [B, 1]
		too_far = rules.lookup(distances, nearest, fill=rules.INFINITE_DISTANCE)[:, 0] > N_HEXES
		all_hexes = torch.arange(N_HEXES, device=hexes.device).expand_as(distances)
		landing, _ = first_best(-rules.hex_distance(nearest, all_hexes), reachable)
		step = nearest
		for _ in range(rules.MAX_PATH):
			step = torch.where(rules.lookup(reachable, step), step, rules.lookup(predecessors, step, fill=-1))
		walked = torch.where(flying, landing, step[:, 0])
		arrived = rules.lookup(reachable, walked[:, None])[:, 0]  # false if the path back broke off (the engine defends)
		go_nearer = torch.where(too_far | ~arrived | (walked == position[:, 0]), -1, walked)

		nothing = ~reachable.any(dim=1) | ~valid.any(dim=1)
		return torch.where(nothing, -1, torch.where(stops, go_there, go_nearer))

	def outputs(self, x, kind, move_hex, target, melee_from):
		"""The contract's outputs for the chosen action: its entries get log-probability 0, every other legal one ~-1000."""
		def choose(index, size):
			return torch.where(torch.arange(size, device=index.device) == index[:, None], 0.0, UNLIKELY)

		uniform = lambda name: masked_log_softmax(torch.zeros(x[name].shape, device=kind.device), x[name])
		from_row = torch.where((torch.arange(MAX_UNITS, device=kind.device) == target[:, None])[..., None], choose(melee_from, N_HEXES)[:, None, :], 0.0)
		return {
			"logp_type": masked_log_softmax(choose(kind, x["m_type"].shape[1]), x["m_type"]),
			"logp_move_hex": masked_log_softmax(choose(move_hex, N_HEXES), x["m_move_hex"]),
			"logp_melee_target": masked_log_softmax(choose(target, MAX_UNITS), x["m_melee_target"]),
			"logp_melee_from": masked_log_softmax(from_row, x["m_melee_from"]),
			"logp_shoot_target": masked_log_softmax(choose(target, MAX_UNITS), x["m_shoot_target"]),
			"logp_spell": uniform("m_spell"),
			"logp_spell_unit": uniform("m_spell_unit"),
			"logp_spell_hex": uniform("m_spell_hex"),
			"logp_spell2_hex": uniform("m_spell2_hex"),
			"logp_spell2_unit": uniform("m_spell2_unit"),
			"value": torch.zeros(1, N_VALUE_HORIZONS + 1, device=kind.device) * kind[:, None],  # [B, heads], no notion of value
		}


def main():
	from .export import export

	path = export(StupidAI(), sys.argv[1] if len(sys.argv) > 1 else "stupid_ai.onnx")
	print(f"wrote {path}")


if __name__ == "__main__":
	main()
