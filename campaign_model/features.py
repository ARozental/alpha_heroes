"""The campaign model's hand-made features: functions of the raw inputs, the game's tables (prices, creature stats,
experience per level) and, where a function says so, a few learnable prior parameters passed in. model.CustomFeatures
calls them in order in make_custom_features; each is small and named for what it computes, and returns plain units
(gold, tiles, days, log strength) that the model maps to bounded inputs.

Values are in gold, Φ's (empire.py) and the labels' unit, so they add up: what a tile offers is the object we may
take there, plus the expected return of the fight that stepping there starts, plus the worth of the land it reveals.
value_logit maps gold to a logit for the fixed priors. The game's rules used here are VCMI's: a battle's experience
is the hit points of the creatures killed, +500 for beating a hero or taking a town (BattleResultProcessor.cpp).
"""
import math

import torch
from torch.nn import functional as F

from battle_model.device import constant, moved
from battle_model.features import scales, signed_scales
from battle_model.layers import vocab_index
from game_values import ATTACK_POINTS, ATTACK_STEP, DAMAGE_SPELLS, DEFENSE_POINTS, DEFENSE_STEP, MOVEMENT_POINTS_LAND, SPELL_ROUNDS
from game_values import HERO_HIRE_COST as HERO_GOLD_COST

from . import constants as c
from .schema import HEADS, SPECS

HO, W, G, L, TG = c.OWN_HEROES, c.WINDOW, c.MAP_GRID, c.MAP_LEVELS, c.TRAVEL_GRID
# A typical count for each of VCMI's quantity ids (CCreature::CreatureQuantityId: 0 unknown, few 1-4, several 5-9,
# pack 10-19, lots 20-49, horde 50-99, throng 100-249, swarm 250-499, zounds 500-999, legion 1000+)
QUANTITY_COUNTS = (15, 3, 7, 15, 35, 75, 175, 375, 750, 1500)
OBJ_HERO, OBJ_TOWN, OBJ_MONSTER = 34, 98, 54  # Obj ids
DWELLING_OBJECTS = (17, 20, 78, 106)  # CREATURE_GENERATOR1, 4, REFUGEE_CAMP, WAR_MACHINE_FACTORY: "Would you like to recruit?"
VISIT_OBJECTS = (9, 83, 102, 215)  # BORDER_GUARD (pass: it goes), SEER_HUT, QUEST_GUARD (the quest completed), TREE_OF_KNOWLEDGE
DANGER_RANGE = 15.0  # tiles: an enemy hero this far reaches a tile with even odds (about a day's walk)
SIGHT_RADIUS = 5  # tiles a hero reveals around it (VCMI's scouting radius without the Scouting skill)
HERO_XP = TOWN_XP = 500.0  # experience for beating a hero, for taking a town (VCMI)
# value_logit: OBJECT_BASE x tanh(gold / BASE_GOLD) + sign x log(1 + |gold| / OBJECT_SCALE): about +2 for anything worth
# taking (a +-0.5 logit nudge does nothing among ~300 legal tiles), Nullkiller2's spread on a log scale, 4.0 at 10,000
OBJECT_BASE, BASE_GOLD, OBJECT_SCALE = 2.0, 300.0, 1565.0
MOVEMENT_PER_TILE = 100.0  # movement points a step on normal terrain costs (diagonals more, roads less)


# ---- small helpers ----

def bit(flags, value):
	return torch.remainder(torch.div(flags, value, rounding_mode="floor"), 2) == 1


def log_add(a, b):
	"""log(e^a + e^b) (torch.logaddexp, which ONNX opset 17 lacks)."""
	return torch.maximum(a, b) + torch.log1p(torch.exp(-(a - b).abs()))


def one_hot(index, n):
	return (index.unsqueeze(-1) == torch.arange(n, device=index.device)).float()


def pair_table():
	"""[OBJECT_VOCAB + 1, MAX_SUBTYPES + 1] an (object, subtype)'s row (constants.OBJECT_PAIRS), PAIR_OTHER outside it."""
	table = torch.full((c.OBJECT_VOCAB + 1, c.MAX_SUBTYPES + 1), c.PAIR_OTHER, dtype=torch.long)
	for (obj, subtype), row in c.OBJECT_PAIRS.items():
		if obj < c.OBJECT_VOCAB:
			table[obj, subtype] = row
	return table


def object_pair(table, obj, subtype):
	"""The rows (pair_table: a module's buffer) of (object, subtype) id pairs."""
	obj = torch.where((obj >= 0) & (obj < c.OBJECT_VOCAB), obj, torch.full_like(obj, c.OBJECT_VOCAB))
	subtype = torch.where((subtype >= 0) & (subtype < c.MAX_SUBTYPES), subtype, torch.full_like(subtype, c.MAX_SUBTYPES))
	return table[obj, subtype]


def bucket(kind, value, subtype):
	"""A hashed embedding row for (kind, value, subtype) id triples."""
	return torch.remainder(torch.remainder((kind + 1) * 7919 + (value + 1) * 131 + subtype.clamp(min=0) * 7, c.SUBTYPE_HASH), c.SUBTYPE_BUCKETS)


def bounded(raw, low, high, start, step):
	"""A learnable prior parameter kept within [low, high]: start when raw is 0 (where weight decay pulls it back),
	moving `step` times faster than a plain weight (model.CORRECTION_STEP), smooth everywhere."""
	offset = math.log((start - low) / (high - start))
	return low + (high - low) * torch.sigmoid(step * raw + offset)


def value_logit(gold):
	"""A value in gold as a prior logit: ~+2 for anything worth having, +2.2 for 550 gold, +3.3 for 4,000, +4.0 for
	10,000, the same below 0 for losses, and smooth through 0 (no jump between a small gain and a small loss)."""
	return OBJECT_BASE * torch.tanh(gold / BASE_GOLD) + torch.sign(gold) * torch.log1p(gold.abs() / OBJECT_SCALE)


def offer(logits, legal):
	"""What a table offers: its best legal entry's prior logit, 0 when nothing is worth anything. logits, legal
	[B, ...] -> [B]. The best, not a sum: early in a game hundreds of fogged areas are each mildly worth exploring, and
	a soft sum over them (log 1 + sum e^p - 1) came to ~6 logits from their number alone (it held END back for good)."""
	return torch.relu(logits).masked_fill(~legal, 0.0).flatten(1).amax(dim=1)


def signed_offer(logits, legal):
	"""What a table of moves offers against standing still: its best legal entry's prior logit, below 0 when every one
	is worse than staying (its prior counts the night's danger: model.tile_features, travel_prior), 0 when none is
	legal. offer's floor at 0 left that to the learned logits, which prefer spending a hero's movement: e5's games of
	2026-11-10, a hero in a town with 196 movement left took a TRAVEL whose stop was beside it (P 0.50, END 0.40), every
	TRAVEL as bad, and died there."""
	best = logits.masked_fill(~legal, -1e4).flatten(1).amax(dim=1)
	return torch.where(legal.flatten(1).any(dim=1), best, torch.zeros_like(best))


def relative_position(delta):
	"""[..., 8] features of offsets [..., 3] in tiles (x, y, level): direction at 8 and 32 tiles, Chebyshev distance at
	8, 32 and 128 tiles, the same level. Every distance in tiles: the same on any map size."""
	distance = torch.maximum(delta[..., 0].abs(), delta[..., 1].abs())
	return torch.cat(
		[signed_scales(delta[..., 0], (8, 32)), signed_scales(delta[..., 1], (8, 32)), scales(distance, (8, 32, 128)), (delta[..., 2] == 0).float().unsqueeze(-1)],
		dim=-1,
	)


# ---- armies and heroes ----

def typical_counts(counts, detailed, t):
	"""Exact counts, or the typical count of a quantity id where not detailed (what a human sees)."""
	typical = t.quantity_counts[counts.clamp(0, len(QUANTITY_COUNTS) - 1)]
	return torch.where(detailed, counts.clamp(min=0).float(), typical)


def army(creatures, counts, detailed, t):
	"""An army's creatures as seen [..., A]: (counts, its strength (count x fight value, NK2's getArmyStrength), its
	worth in gold (count x (cost + u), empire.py's), the experience for killing it (count x hit points))."""
	present = creatures >= 0
	index = vocab_index(creatures, c.CREATURE_VOCAB)
	count = typical_counts(counts, detailed.unsqueeze(-1).expand_as(counts), t) * present
	power = (count * t.fight_value[index]).sum(dim=-1)
	gold = (count * (t.creature_cost[index] + t.creature_unit[index])).sum(dim=-1)
	health = (count * t.creature_health[index]).sum(dim=-1)
	return count, power, gold, health


def hero_strength(primary, power):
	"""log of a hero's army strength x fighting strength sqrt((1 + 0.05 attack)(1 + 0.05 defense)) (NK2's
	getTotalStrength); for a town's garrison, power alone (primary 0)."""
	p = primary.clamp(min=0).float()
	return torch.log1p(power) + 0.5 * torch.log((1 + 0.05 * p[..., 0]) * (1 + 0.05 * p[..., 1]))


# ---- a fight's strength, several ways (the user, 2026-11-04: estimates that agree are easier to act on) ----



def army_battle(creatures, counts, detailed, t):
	"""An army's fighting numbers as seen [..., A] -> [...] each: (its damage a round, every stack's average damage x
	count; its attack, damage-weighted; its defense, hit-point-weighted), from the creatures' own stats (vocab)."""
	present = creatures >= 0
	index = vocab_index(creatures, c.CREATURE_VOCAB)
	count = typical_counts(counts, detailed.unsqueeze(-1).expand_as(counts), t) * present
	damage = count * t.creature_damage[index]
	health = count * t.creature_health[index]
	total = damage.sum(dim=-1)
	attack = (damage * t.creature_attack[index]).sum(dim=-1) / total.clamp(min=1e-6)
	defense = (health * t.creature_defense[index]).sum(dim=-1) / health.sum(dim=-1).clamp(min=1e-6)
	return total, attack, defense


def damage_modifier(difference):
	"""VCMI's damage factor for an attack of `difference` points over the defender's defense: +5% a point (at most
	+300%), -2.5% a point under it (at most -70%; game_values' constants)."""
	return 1.0 + ATTACK_STEP * difference.clamp(0.0, ATTACK_POINTS) - DEFENSE_STEP * (-difference).clamp(0.0, DEFENSE_POINTS)


def army_multiplier(attack, defense, power=None, knowledge=None, spellbook=None, step=0.0):
	"""game_values.army_multiplier of tensors: what each hero's battle attack and defense make of the army it leads, and
	its spellbook (with power, knowledge: game_values.spell_factor at step)."""
	dealt = 1.0 + ATTACK_STEP * attack.float().clamp(0.0, ATTACK_POINTS)
	spells = 1.0
	if spellbook is not None:
		spells = 1.0 + step * torch.sqrt(power.float().clamp(min=0.0) * knowledge.float().clamp(min=0.0)) * spellbook.float()
	taken = 1.0 - DEFENSE_STEP * defense.float().clamp(0.0, DEFENSE_POINTS)
	return torch.sqrt(dealt / taken) * spells


def spell_damage(x, t):
	"""Each hero's spell damage a round [B, H]: its best damage spell (DAMAGE_SPELLS, at its spell power) cast as often
	as its mana allows, at most SPELL_ROUNDS times, spread over SPELL_ROUNDS rounds. 0 without one (or what we cannot see:
	the other players' heroes' spells)."""
	device = x["h_spells"].device
	ids = constant([s[0] for s in DAMAGE_SPELLS], device, torch.long)  # (made once per device: no copy inside a CUDA graph)
	base, per_power, cost = (constant([s[k] for s in DAMAGE_SPELLS], device, torch.float32) for k in (1, 2, 3))
	known = x["h_spells"].index_select(-1, ids) > 0  # [B, H, S]
	power = x["h_primary"][..., 2].clamp(min=0).float().unsqueeze(-1)
	casts = torch.minimum(torch.floor(x["h_mana"].clamp(min=0).float().unsqueeze(-1) / cost), torch.full_like(cost, SPELL_ROUNDS))
	return ((base + per_power * power) * casts * known).amax(dim=-1) / SPELL_ROUNDS


def hero_multipliers(primary, artifacts, t):
	"""What each hero makes of the army it leads in Φ [..] (primary [.., 4], its items' included; artifacts [.., worn]):
	unled_share x army_multiplier at its attack, defense, spell power and knowledge, its spellbook (artifact SPELLBOOK
	worn), empire.hero_multiplier's."""
	spellbook = (artifacts == SPELLBOOK).any(dim=-1)
	return t.unled_share * army_multiplier(primary[..., 0], primary[..., 1], primary[..., 2], primary[..., 3], spellbook, t.spell_step)


def hopeless(ratio):
	"""How hopeless a fight is by its log strength ratio (ours over theirs) [...]: ~0 while the enemy is at most ~e times as
	strong, then a unit an e-fold more, /3 (a guard 54x our strength ~0.92), softly bounded at 2 (e^7). Smooth (the user,
	2026-11-04: a softplus at a quarter's temperature for the ramp's kink, a tanh for its bound): ~0.06 at e times."""
	ramp = 0.25 * F.softplus((-ratio - 1.0) / 0.25)
	return 6.0 * torch.tanh(ramp / 6.0) / 3.0


def strength_view(ratio):
	"""A log strength ratio as a feature [...]: itself near even, softly bounded at +-2 (e^2 times either way)."""
	return 2.0 * torch.tanh(ratio / 2.0)


def race_strength(health, damage, attack, defense, their_health, their_damage, their_attack, their_defense, spells=0.0):
	"""A fight's log strength ratio (ours over theirs) as a damage race: each side's hit points x the damage it deals a
	round (its stacks' damage at its attack against the other's defense, damage_modifier; ours plus `spells` a round),
	the square root of each (the units of log1p(army strength): an army twice as big is log 2 stronger)."""
	ours = damage * damage_modifier(attack - their_defense) + spells
	theirs = their_damage * damage_modifier(their_attack - defense)
	return 0.5 * (torch.log1p(health * ours) - torch.log1p(their_health * theirs))


def fractional_level(experience, table):
	"""A hero's level with the way to the next (game_values.fractional_level): 1500 experience -> 2.5."""
	top = table.shape[0]
	level = (experience.unsqueeze(-1) >= table).sum(dim=-1).clamp(min=1)  # table[0] = 0: level 1 at least
	low = table[(level - 1).clamp(max=top - 1)]
	high = table[level.clamp(max=top - 1)]
	way = torch.where(level < top, (experience - low) / (high - low).clamp(min=1), torch.zeros_like(experience))
	return level.float() + way.clamp(0, 1)


def next_level_gain(primary, t):
	"""What a hero's next level-up brings in Φ (empire.py skill_value) [...]: a primary point (the next past those it
	has), a secondary skill level, the level's bonus. primary [..., 4] without items."""
	points = primary.clamp(min=0).sum(dim=-1).float()
	return t.primary_point * (points - t.primary_base + 1).clamp(min=0) + t.skill_level + t.level_bonus


def skill_value(primary, skills, experience, spells, t):
	"""empire.skill_value of heroes [...]: primary [..., 4] without items, skills [..., N_SECONDARY_SKILLS] levels,
	experience [...], spells [..., N_SPELLS] in the spellbook."""
	points = primary.clamp(min=0).sum(dim=-1).float()
	extra = (points - t.primary_base).clamp(min=0)
	level = fractional_level(experience.clamp(min=0).float(), t.level_experience)
	whole = level.floor()
	levels = skills.clamp(min=0).sum(dim=-1).float()
	known = (spells.float() * t.spell_value).sum(dim=-1)
	return t.skill_level * (levels - t.fresh_skill_levels) + t.primary_point * extra * (extra + 1) / 2 + t.level_bonus * (whole - 1).clamp(min=0) + (level - whole) * next_level_gain(primary, t) + known


def main_hero(values, present):
	"""[B, H] bool: the strongest present hero (the most skill_value; the first of equals), empire.heroes_value's."""
	best = values.masked_fill(~present, float("-inf")).argmax(dim=-1, keepdim=True)
	return (torch.arange(values.shape[-1], device=values.device) == best) & present


def armed_share(army_gold, t):
	"""empire.hero_base_value's share of HERO_BASE [...] at a hero's army worth (gold at cost + u): armyless_share with
	none, 1 from armed_army on."""
	return t.armyless_share + (1 - t.armyless_share) * (army_gold.clamp(min=0) / t.armed_army).clamp(max=1.0)


def heroes_value(values, present, armed, t):
	"""empire.heroes_value [B]: each hero's HERO_BASE at its armed_share (armed [B, H]), the strongest hero's skill_value
	(values [B, H]) in full, the others' at SECONDARY_SHARE."""
	weight = torch.where(main_hero(values, present), torch.ones_like(values), torch.full_like(values, t.secondary_share))
	return (t.hero_base * armed + weight * values).mul(present.float()).sum(dim=-1)


def own_armed(x, t):
	"""Our heroes' armed_share [B, HO] (their armies at cost + u, exact counts: ours)."""
	creatures = x["h_army_creature"][:, :HO]
	_, _, gold, _ = army(creatures, x["h_army_count"][:, :HO], torch.ones_like(creatures[..., 0], dtype=torch.bool), t)
	return armed_share(gold, t)


def own_skill_values(x, t):
	"""Our heroes' skill_value [B, HO] and which are present [B, HO]."""
	present = x["h_present"][:, :HO] & (x["h_owner"][:, :HO] == c.OWNER_SELF)
	return skill_value(x["h_primary_base"], x["h_skills"][:, :HO], x["h_exp"][:, :HO], x["h_spells"][:, :HO], t), present


def battle_chance(table, value):
	"""empire._chance: the battle chance (24ths) at luck or morale values [...], capped at -3..+3."""
	return constant(table, value.device)[(value.clamp(-3, 3) + 3).long()]


def speed_factors(creatures, counts, t):
	"""empire.speed_factor of armies [..., A] -> [...]: what a hero's movement makes of the army it leads,
	(MOVEMENT_POINTS_LAND at its slowest creature's speed / the fastest's) ^ speed_power; 1 for an empty army."""
	present = (creatures >= 0) & (counts > 0)
	speed = t.creature_speed[vocab_index(creatures, c.CREATURE_VOCAB)]
	slowest = torch.where(present, speed, torch.full_like(speed, len(MOVEMENT_POINTS_LAND))).amin(dim=-1)
	points = constant(MOVEMENT_POINTS_LAND, slowest.device)[slowest.clamp(0, len(MOVEMENT_POINTS_LAND) - 1).long()]
	return (points / MOVEMENT_POINTS_LAND[-1]) ** t.speed_power


def make_up_morales(creatures, counts, t):
	"""game_values.make_up_morale of armies [..., A] -> [...]: one faction +1, n of them 2 - n, none 0; any undead -1."""
	present = (creatures >= 0) & (counts > 0)
	index = vocab_index(creatures, c.CREATURE_VOCAB)
	ids = torch.arange(-1, c.FACTION_VOCAB + 1, device=creatures.device)  # (-1: a creature of no faction)
	factions = ((t.creature_faction[index].unsqueeze(-1) == ids) & present.unsqueeze(-1)).any(dim=-2).sum(dim=-1).float()
	undead = ((t.creature_undead[index] > 0) & present).any(dim=-1).float()
	return torch.where(factions == 1, 1.0, 2.0 - factions) * (factions > 0).float() - undead


def make_up_values(creatures, counts, rest, t):
	"""empire.make_up_value of armies [..., A] -> [...] (gold): the battle chance their make-up's morale (make_up_morales)
	adds over the rest of their hero's morale (rest [...]: its morale without the make-up's), within the caps,
	army_morale a 24th of the living army at cost + u."""
	from .empire import MORALE_CHANCE

	make_up = make_up_morales(creatures, counts, t)
	gained = battle_chance(MORALE_CHANCE, rest + make_up) - battle_chance(MORALE_CHANCE, rest)
	index = vocab_index(creatures, c.CREATURE_VOCAB)
	worth = counts.clamp(min=0).float() * (t.creature_cost[index] + t.creature_unit[index]) * (creatures >= 0)
	return t.army_morale * gained * (worth * t.creature_morale[index]).sum(dim=-1)


def bulk_moves(into, source):
	"""Which of an army's stacks VCMI's bulk move (CGameHandler::bulkMoveArmy: GIVE_ALL's, GATHER's) takes into another
	[..., A] bool (into, source: creature ids [..., A], -1 empty): in the source's slots' order, onto the creature's stack
	there, else into a free slot while one is left (each new creature's stack takes one), else it stays."""
	there_now = into >= 0
	free = (~there_now).sum(dim=-1)
	moved = []
	for slot in range(source.shape[-1]):
		creature = source[..., slot]
		present = creature >= 0
		there = ((into == creature.unsqueeze(-1)) & there_now).any(dim=-1)
		takes = present & ~there & (free > 0)
		moved.append(present & (there | takes))
		free = free - takes.long()
	return torch.stack(moved, dim=-1)


def led_values(creatures, counts, multiplier, rest, t):
	"""What armies [..., A] count in Φ led by a hero (gold, [...]): their worth x the hero's multiplier (hero_multipliers)
	x its movement (speed_factors), and their make-up's morale (make_up_values, over rest: the hero's morale without its
	make-up's). An exchange's or a gather's armies joined: their slots concatenated."""
	index = vocab_index(creatures, c.CREATURE_VOCAB)
	worth = (counts.clamp(min=0).float() * (t.creature_cost[index] + t.creature_unit[index]) * (creatures >= 0)).sum(dim=-1)
	return worth * multiplier * speed_factors(creatures, counts, t) + make_up_values(creatures, counts, rest, t)


def own_make_up_values(x, t):
	"""empire.make_up_value of our heroes [B, HO]: their army's make-up's morale over the rest of their morale (h_morale
	holds the make-up's: VCMI's bonus of source ARMY)."""
	creatures, counts = x["h_army_creature"][:, :HO], x["h_army_count"][:, :HO]
	return make_up_values(creatures, counts, x["h_morale"][:, :HO].float() - make_up_morales(creatures, counts, t), t)


def boost_values(x, t):
	"""empire.boost_value of our heroes [B, HO]: what their luck and morale until the next battle add (the battle
	chance over the rest of their luck and morale, within the caps), BOOST a 24th of the army each lifts at cost + u."""
	from .empire import LUCK_CHANCE, MORALE_CHANCE

	luck, morale = x["h_luck"][:, :HO], x["h_morale"][:, :HO]
	gained_luck = battle_chance(LUCK_CHANCE, luck) - battle_chance(LUCK_CHANCE, luck - x["h_battle_luck"])
	gained_morale = battle_chance(MORALE_CHANCE, morale) - battle_chance(MORALE_CHANCE, morale - x["h_battle_morale"])
	creatures = x["h_army_creature"][:, :HO]
	index = vocab_index(creatures, c.CREATURE_VOCAB)
	worth = x["h_army_count"][:, :HO].clamp(min=0).float() * (t.creature_cost[index] + t.creature_unit[index]) * (creatures >= 0)
	return t.boost * (gained_luck * (worth * t.creature_luck[index]).sum(dim=-1) + gained_morale * (worth * t.creature_morale[index]).sum(dim=-1))


def level_gains(x, t):
	"""What each hero of ours gains in Φ for a level of experience [B, HO]: its next level-up, at its weight (the
	strongest hero's in full, the others' at SECONDARY_SHARE)."""
	values, present = own_skill_values(x, t)
	weight = torch.where(main_hero(values, present), torch.ones_like(values), torch.full_like(values, t.secondary_share))
	return weight * next_level_gain(x["h_primary_base"], t) * present


def experience_value(experience, gained, level_gain, t):
	"""What gaining experience is worth to a hero in Φ: the levels it makes at the hero's level_gains (each a level-up's
	worth: a fight's experience rarely makes more than one or two)."""
	return (fractional_level(experience + gained, t.level_experience) - fractional_level(experience, t.level_experience)) * level_gain


def hero_worth(x, t):
	"""Each hero of ours' own worth in Φ [B, HO]: HERO_BASE, its skill_value at its weight (the strongest hero's in full,
	the others' at SECONDARY_SHARE) and its artifacts."""
	values, present = own_skill_values(x, t)
	weight = torch.where(main_hero(values, present), torch.ones_like(values), torch.full_like(values, t.secondary_share))
	artifacts = torch.cat([x["h_artifacts"][:, :HO], x["h_backpack"][:, :HO]], dim=-1)
	carried = (t.artifact_price[vocab_index(artifacts, c.ARTIFACT_VOCAB)] * (artifacts >= 0)).sum(dim=-1)
	return (t.hero_base * own_armed(x, t) + weight * values + carried) * present


def tavern_worth(x, t):
	"""What hiring each hero for hire would add to Φ [B, N_TAVERN] (gold), before its price and the crowding: the
	heroes part's change (empire.heroes_value: HERO_BASE, its skills in full if they beat our strongest hero's, else at
	SECONDARY_SHARE, and ours re-weighed), the army it comes with as Φ prices a hero's (led_values: at cost + u by what it
	makes of it, army_multiplier at its primary skills, its movement and its make-up's morale) and its worn artifacts (a
	spellbook, war machines), as empire_value counts a hero of ours (its mana left out); 0 for an empty slot."""
	present = (x["g_tavern"] >= 0).float()
	_, _, army_gold, _ = army(x["g_tavern_army_creature"], x["g_tavern_army_count"], torch.ones_like(x["g_tavern_army_creature"][..., 0], dtype=torch.bool), t)
	worn = x["g_tavern_artifacts"]
	carried = (t.artifact_price[vocab_index(worn, c.ARTIFACT_VOCAB)] * (worn >= 0)).sum(dim=-1)
	theirs = skill_value(x["g_tavern_primary"], x["g_tavern_skills"], x["g_tavern_exp"], x["g_tavern_spells"], t)  # [B, N_TAVERN]
	ours, mine = own_skill_values(x, t)
	armed, theirs_armed = own_armed(x, t), armed_share(army_gold, t)
	before = heroes_value(ours, mine, armed, t).unsqueeze(-1)
	after = torch.stack([heroes_value(torch.cat([ours, theirs[:, s : s + 1]], dim=1), torch.cat([mine, torch.ones_like(mine[:, :1])], dim=1), torch.cat([armed, theirs_armed[:, s : s + 1]], dim=1), t) for s in range(theirs.shape[1])], dim=1)
	creatures, counts = x["g_tavern_army_creature"], x["g_tavern_army_count"].float()
	led = led_values(creatures, counts, hero_multipliers(x["g_tavern_primary"], x["g_tavern_artifacts"], t), torch.zeros_like(army_gold), t)  # (its morale: its army's make-up's alone)
	return (after - before + led + carried) * present


def hire_gain(x, t, worth):
	"""What hiring each hero for hire does to our Φ [B, N_TAVERN] (gold): its worth (tavern_worth) less the hire's
	price and the crowding the extra hero adds (empire.py HERO_PENALTY x heroes^2): + for a week's hero with its army,
	- for one with a lone creature (the user, 2026-10-29)."""
	heroes = (x["h_present"][:, :HO] & (x["h_owner"][:, :HO] == c.OWNER_SELF)).sum(dim=1, keepdim=True).float()
	return worth - HERO_GOLD_COST - t.hero_penalty * (2 * heroes + 1)


# ---- our empire: Φ ----

PHI_PARTS = ("resources", "buildings", "income", "army", "for_hire", "heroes", "artifacts", "mana", "hero_penalty", "explored")
# Φ as the value head reads it: x / (x + c) at these rival values (gold), the 2-player share's own form, so its gradient
# is the share's against a rival worth c. joint1's empires: ~40k on day 1, 88k-272k at a game's end, NK2's ~400k
PHI_SCALES = (3e4, 1e5, 3e5, 1e6)


def empire_value(x, t):
	"""Our empire's value Φ in gold by empire.py's rules from the inputs: its parts [B, PHI_PARTS] and total [B], the
	ledger's to the gold (with the armies in our mines' garrisons and the creatures for hire in our dwellings outside
	towns: without them the value audit saw Φ fall when a hero left troops in a mine, which the ledger does not)."""
	own_heroes = (x["h_present"][:, :HO] & (x["h_owner"][:, :HO] == c.OWNER_SELF)).float()
	own_towns = (x["t_present"][:, : c.OWN_TOWNS] & (x["t_owner"][:, : c.OWN_TOWNS] == c.OWNER_SELF)).float()
	exact = lambda creatures: torch.ones_like(creatures[..., 0], dtype=torch.bool)  # noqa: E731 (ours: exact counts)
	_, _, hero_army, _ = army(x["h_army_creature"][:, :HO], x["h_army_count"][:, :HO], exact(x["h_army_creature"][:, :HO]), t)
	_, _, garrison, _ = army(x["t_garrison_creature"][:, : c.OWN_TOWNS], x["t_garrison_count"][:, : c.OWN_TOWNS], exact(x["t_garrison_creature"][:, : c.OWN_TOWNS]), t)
	_, _, owned, _ = army(x["g_owned_creature"], x["g_owned_count"], exact(x["g_owned_creature"]), t)
	dwelling = x["g_dwelling_creature"]
	dwelling_hire = (x["g_dwelling_count"].clamp(min=0).float() * t.creature_unit[vocab_index(dwelling, c.CREATURE_VOCAB)] * (dwelling >= 0)).sum(dim=-1)
	faction = vocab_index(x["t_faction"][:, : c.OWN_TOWNS], c.FACTION_VOCAB)
	available = x["t_available_creature"][:, : c.OWN_TOWNS]
	artifacts = torch.cat([x["h_artifacts"][:, :HO], x["h_backpack"][:, :HO]], dim=-1)
	usable_mana = torch.minimum(x["h_mana"][:, :HO].clamp(min=0), 3 * x["h_spell_cost"][:, :HO].clamp(min=0)).float()
	parts = torch.stack(
		[
			(x["g_resources"].clamp(min=0).float() * t.resource_price).sum(dim=-1),
			((x["t_built"][:, : c.OWN_TOWNS].float() * t.building_value[faction]).sum(dim=-1) * own_towns).sum(dim=-1),
			t.income_days * (x["g_object_income"].clamp(min=0).float() * t.resource_price).sum(dim=-1),
			(hero_army * hero_multipliers(x["h_primary"][:, :HO], x["h_artifacts"][:, :HO], t) * speed_factors(x["h_army_creature"][:, :HO], x["h_army_count"][:, :HO], t) * own_heroes).sum(dim=-1)
			+ t.unled_share * ((garrison * own_towns).sum(dim=-1) + owned),  # (armies no hero leads: game_values.unled_share)
			((x["t_available"][:, : c.OWN_TOWNS].clamp(min=0).float() * t.creature_unit[vocab_index(available, c.CREATURE_VOCAB)] * (available >= 0)).sum(dim=-1) * own_towns).sum(dim=-1) + dwelling_hire,
			heroes_value(*own_skill_values(x, t), armed_share(hero_army, t), t) + ((boost_values(x, t) + own_make_up_values(x, t)) * own_heroes).sum(dim=-1),
			((t.artifact_price[vocab_index(artifacts, c.ARTIFACT_VOCAB)] * (artifacts >= 0)).sum(dim=-1) * own_heroes).sum(dim=-1),
			(2.0 * x["h_primary"][:, :HO, 2].clamp(min=0).float() * usable_mana * own_heroes).sum(dim=-1),  # game_values.capped_mana_value
			-t.hero_penalty * own_heroes.sum(dim=-1) ** 2,
			t.explored_tile * x["g_explored"].clamp(min=0).float(),
		],
		dim=-1,
	)
	return parts, parts.sum(dim=-1)


def phi_features(parts, total):
	"""Φ as the learned layers see it, every feature bounded: x / (x + c) at PHI_SCALES (0..1) and each part's share of
	the parts' absolute sum (-1..1). Φ comes from the inputs alone: no gradient flows through it, and no weight ever
	sees its size."""
	return torch.cat([scales(total, PHI_SCALES), parts / (parts.abs().sum(dim=-1, keepdim=True) + 1000)], dim=-1)


# ---- enemies: danger and fights ----

def enemies(x, hero_strength):
	"""The other players' heroes we see: positions [B, E, 3], log strength [B, E], present [B, E]."""
	return x["h_pos"][:, HO:], hero_strength[:, HO:], x["h_present"][:, HO:] & (x["h_owner"][:, HO:] != c.OWNER_SELF)


def reach(foes, x_pos, y_pos, z_pos):
	"""How near each enemy hero is to positions [N, ...] (x, y, level), foes (enemies()) per N: [N, ..., E] reach (1 at
	its tile, 1/2 at DANGER_RANGE tiles, ~0 beyond twice that or on another level) and its log strength [N, 1.., E]."""
	pos, strength, present = foes
	shape = (x_pos.shape[0],) + (1,) * (x_pos.dim() - 1) + (present.shape[1],)
	ex, ey, ez = (pos[..., i].reshape(shape) for i in range(3))
	distance = torch.maximum((x_pos.unsqueeze(-1) - ex).abs(), (y_pos.unsqueeze(-1) - ey).abs()).float()
	near = torch.sigmoid(4 * (1 - distance / DANGER_RANGE)) * (z_pos.unsqueeze(-1) == ez) * present.reshape(shape)
	return near, strength.reshape(shape)


def town_danger(x, hero_strength, town_strength):
	"""Our towns' danger [B, OWN_TOWNS]: the most an enemy hero we see threatens one (its reach x P(it beats the
	garrison and the heroes there))."""
	def defender(index):  # [B, TO]: the log strength of a hero there, very low if none
		found = torch.gather(hero_strength, 1, index.clamp(min=0))
		return torch.where(index >= 0, found, torch.full_like(found, -30.0))

	to = c.OWN_TOWNS
	defence = log_add(log_add(town_strength[:, :to], defender(x["t_visiting"][:, :to])), defender(x["t_garrison_hero"][:, :to]))
	pos = x["t_pos"][:, :to]
	near, theirs = reach(enemies(x, hero_strength), pos[..., 0], pos[..., 1], pos[..., 2])  # [B, TO, E]
	return (near * torch.sigmoid(2 * (theirs - defence.unsqueeze(-1)))).amax(dim=-1) * (x["t_present"][:, :to] & (x["t_owner"][:, :to] == c.OWNER_SELF))


def sheltered_danger(x, hero_strength, manned, decision, hero, foes):
	"""Our towns' danger to a hero of ours inside one [K, OWN_TOWNS] (K windows; decision [K] their decisions, hero [K]
	their heroes' slots 0..OWN_HEROES, foes enemies() of their decisions): town_danger's with this hero in the town, an
	enemy hero's reach to it x P(it beats the defence: the garrison and the fort, which fights while a hero defends
	(manned: the model's town strength with it), a hero of ours in its garrison and the visitor, this hero once); and
	[K, OWN_TOWNS] whether the hero is inside it or can get in (shelter_open)."""
	to = c.OWN_TOWNS
	strength = hero_strength.index_select(0, decision)  # [K, H]

	def defender(index):  # [K, TO]: the log strength of another hero of ours there, very low if none
		index = index.index_select(0, decision)
		found = torch.gather(strength, 1, index.clamp(min=0))
		return torch.where((index >= 0) & (index != hero[:, None]), found, torch.full_like(found, -30.0))

	ours = strength.gather(1, hero[:, None])  # [K, 1]
	defence = log_add(log_add(log_add(manned[:, :to].index_select(0, decision), defender(x["t_visiting"][:, :to])), defender(x["t_garrison_hero"][:, :to])), ours)
	pos = x["t_pos"][:, :to].index_select(0, decision)
	near, theirs = reach(foes, pos[..., 0], pos[..., 1], pos[..., 2])  # [K, TO, E]
	return (near * torch.sigmoid(2 * (theirs - defence.unsqueeze(-1)))).amax(dim=-1), shelter_open(x, decision, hero)


def shelter_open(x, decision, hero):
	"""[K, OWN_TOWNS] whether a hero of ours (hero [K] its slot, of decision [K]) is behind a town's walls there: its
	garrison hero, or its visitor while no garrison hero is there (VCMI fights the visitor outside the walls first when
	there is: CGTownInstance::isBattleOutsideTown); or can get in, the visiting place free and no garrison hero (with a
	visitor there a step onto the town is a meeting, and the hero stays outside, beside it). e5's games of 2026-11-10: a
	hero at danger 0.94 stepped onto a town twice, a hero of ours inside, ended beside it both times and was killed."""
	garrison = x["t_garrison_hero"][:, : c.OWN_TOWNS].index_select(0, decision)
	visitor = x["t_visiting"][:, : c.OWN_TOWNS].index_select(0, decision)
	me = hero[:, None]
	return (garrison == me) | ((garrison < 0) & ((visitor < 0) | (visitor == me)))


def town_night(x, hero_strength, manned, unmanned, stakes, decision, hero, foes, pos, last=None):
	"""What a hero of ours ending its day at pos [K, N, 3] (K: its decision, decision [K], and slot, hero [K]; foes enemies()
	of the decisions) changes in our towns' expected loss tonight [K, N] (gold; above 0 it lowers it): over our towns,
	(P(taken) as they stand - P(taken) after) x town_stakes, P the reach of the enemy heroes we see x P(it beats the
	defence). The defence: the garrison and the fort (manned while a hero is in: unmanned, the town's own strength, else),
	our other heroes there, and this one where it ends inside (shelter_open: its garrison hero, or the visitor with none)
	or stays; leaving a town takes it out of its defence. e5's v2830 trace, 2026-11-11: under the hero's night term 11 of
	21 eliminations were the last town falling with a garrison of 0-200 gold while our main army stood alive a few tiles
	away, safe where it was: the hero's term prices its own night, not the town's. last [B] or None: the stake of our only
	town at least this (e5: an emptied last town's buildings are a few thousand while losing it ends the game: our Φ)."""
	to = c.OWN_TOWNS
	strength = hero_strength.index_select(0, decision)
	visiting, garrisoned = x["t_visiting"][:, :to].index_select(0, decision), x["t_garrison_hero"][:, :to].index_select(0, decision)
	me = hero[:, None]

	def other(index):  # [K, TO]: another hero of ours there, very low if none
		found = torch.gather(strength, 1, index.clamp(min=0))
		return torch.where((index >= 0) & (index != me), found, torch.full_like(found, -30.0))

	others = log_add(other(visiting), other(garrisoned))
	other_in = ((visiting >= 0) & (visiting != me)) | ((garrisoned >= 0) & (garrisoned != me))
	m, u = manned[:, :to].index_select(0, decision), unmanned[:, :to].index_select(0, decision)
	without = log_add(torch.where(other_in, m, u), others)
	with_me = log_add(log_add(m, others), strength.gather(1, me))
	tpos = x["t_pos"][:, :to].index_select(0, decision)
	near, theirs = reach(foes, tpos[..., 0], tpos[..., 1], tpos[..., 2])  # [K, TO, E]
	taken = lambda defence: (near * torch.sigmoid(2 * (theirs - defence.unsqueeze(-1)))).amax(dim=-1)  # noqa: E731
	p_with, p_without = taken(with_me), taken(without)
	inside = (visiting == me) | (garrisoned == me)
	p_now = torch.where(inside, p_with, p_without)
	after = (pos.unsqueeze(2) == tpos.unsqueeze(1)).all(dim=-1) & shelter_open(x, decision, hero).unsqueeze(1)  # [K, N, TO]
	p_after = torch.where(after, p_with.unsqueeze(1), p_without.unsqueeze(1))
	own = own_towns(x)
	stake = stakes[:, :to]
	if last is not None:
		stake = torch.where(own & (own.sum(dim=1, keepdim=True) == 1), torch.maximum(stake, last[:, None]), stake)
	stake = (stake * own).index_select(0, decision)
	return ((p_now.unsqueeze(1) - p_after) * stake.unsqueeze(1)).sum(dim=-1)


def night_danger(x, hero_strength, manned, pos):
	"""P(a hero of ours is lost tonight) [B, HO, N] if its day ends at pos [B, HO, N, 3] (x, y, level): the enemy heroes
	we see, each one's reach there x P(it beats the hero), at the most (MOVE's danger); in a town of ours an enemy must
	beat the town's defence with the hero inside (sheltered_danger)."""
	b = pos.shape[0]
	foes = enemies(x, hero_strength)
	near, theirs = reach(foes, pos[..., 0], pos[..., 1], pos[..., 2])  # [B, HO, N, E]
	open_ground = (near * torch.sigmoid(2 * (theirs - hero_strength[:, :HO, None, None]))).amax(dim=-1)
	decision = torch.arange(b, device=pos.device).repeat_interleave(HO)
	hero = torch.arange(HO, device=pos.device).repeat(b)
	sheltered, open_ = sheltered_danger(x, hero_strength, manned, decision, hero, tuple(t.index_select(0, decision) for t in foes))
	sheltered, open_ = sheltered.reshape(b, HO, -1), open_.reshape(b, HO, -1)  # [B, HO, TO]
	own = x["t_present"][:, : c.OWN_TOWNS] & (x["t_owner"][:, : c.OWN_TOWNS] == c.OWNER_SELF)
	inside = (pos.unsqueeze(3) == x["t_pos"][:, None, None, : c.OWN_TOWNS]).all(dim=-1) & (own[:, None, :] & open_)[:, :, None, :]  # [B, HO, N, TO]
	return torch.where(inside.any(dim=-1), (inside.float() * sheltered.unsqueeze(2)).amax(dim=-1), open_ground)


STOP_RADIX = 256  # (schema 18's travel_stop: (level x 256 + y) x 256 + x + 1)


def travel_stops(x):
	"""Where today's walk toward each TRAVEL area stops ([B, HO, AREAS, 3] x, y, level) and whether that is known ([B, HO,
	AREAS]: schema 18's travel_stop; none in a record from before, or where TRAVEL is not legal)."""
	code = x["travel_stop"][:, :HO].reshape(x["travel_stop"].shape[0], HO, -1).long() - 1
	known = code >= 0
	code = code.clamp(min=0)
	return torch.stack([code % STOP_RADIX, torch.div(code, STOP_RADIX, rounding_mode="floor") % STOP_RADIX, torch.div(code, STOP_RADIX * STOP_RADIX, rounding_mode="floor")], dim=-1), known


def left_danger(x, hero_strength, town_strength):
	"""Our towns' danger once their visiting hero has left [B, OWN_TOWNS] (bundle 60's town guard): town_danger's with the
	town's own strength and a garrison hero of ours alone defending it."""
	def defender(index):
		found = torch.gather(hero_strength, 1, index.clamp(min=0))
		return torch.where(index >= 0, found, torch.full_like(found, -30.0))

	to = c.OWN_TOWNS
	defence = log_add(town_strength[:, :to], defender(x["t_garrison_hero"][:, :to]))
	pos = x["t_pos"][:, :to]
	near, theirs = reach(enemies(x, hero_strength), pos[..., 0], pos[..., 1], pos[..., 2])  # [B, TO, E]
	return (near * torch.sigmoid(2 * (theirs - defence.unsqueeze(-1)))).amax(dim=-1) * (x["t_present"][:, :to] & (x["t_owner"][:, :to] == c.OWNER_SELF))


def emptied_danger(x, hero_strength):
	"""Our towns' danger once their upper army has gone into the visitor (MULTI's GATHER) and the visitor with it
	[B, OWN_TOWNS]: an empty town's (town_danger's with no defence: an empty town's fort does not fight, town_strength).
	The upper army is the garrison hero's when one is there (VCMI's getUpperArmy), which then keeps one creature (until
	2026-11-05 counted as staying to defend at full strength)."""
	to = c.OWN_TOWNS
	pos = x["t_pos"][:, :to]
	near, theirs = reach(enemies(x, hero_strength), pos[..., 0], pos[..., 1], pos[..., 2])  # [B, TO, E]
	return (near * torch.sigmoid(2 * theirs)).amax(dim=-1) * (x["t_present"][:, :to] & (x["t_owner"][:, :to] == c.OWNER_SELF))  # (an empty town's log1p(0))


def slot_heroes(x):
	"""[B, HO] each window slot's hero token (schema 17): its own, or in a slot no hero of ours holds the hero whose
	further window it carries (w_extra h + 1). A record from before reads w_extra 0: every slot its own."""
	own = torch.arange(HO, device=x["h_present"].device).expand(x["h_present"].shape[0], HO)
	if "w_extra" not in x:
		return own
	extra = x["w_extra"]
	return torch.where(extra > 0, (extra - 1).clamp(max=HO - 1), own)


def window_heroes(x, windows):
	"""The windows' heroes and middles (schema 17), windows [K] flat window slots (decision x OWN_HEROES + slot): ([K]
	each one's hero as a flat index (decision x OWN_HEROES + hero token), [K, 3] its middle tile (x, y, level)). A
	slot's window is its own hero's, around where it stands; a further window (w_extra) is another hero's of the same
	decision, around w_extra_pos: tiles that hero reaches today outside its own window (past a gate or a monolith,
	down a long road), each a MOVE's target with the features its own window's tiles have."""
	stand = x["h_pos"][:, :HO].reshape(-1, 3).index_select(0, windows)
	if "w_extra" not in x:
		return windows, stand
	extra = x["w_extra"].reshape(-1).index_select(0, windows)
	further = extra > 0
	first = torch.div(windows, HO, rounding_mode="floor") * HO
	heroes = torch.where(further, first + (extra - 1).clamp(max=HO - 1), windows)
	middle = torch.where(further.unsqueeze(-1), x["w_extra_pos"].reshape(-1, 3).index_select(0, windows), stand)
	return heroes, middle


def windows_present(x):
	"""[B, HO] the window slots that hold a window: our heroes' own and their further ones (w_extra)."""
	present = x["h_present"][:, :HO]
	return present | (x["w_extra"] > 0) if "w_extra" in x else present


def expected_loss(ratio, has_enemy):
	"""The share of the army a won fight costs: NK2's (their strength / ours)^2, at most all of it."""
	return has_enemy * torch.exp(-2 * torch.relu(ratio))


def win_chance(ratio, sharpness):
	"""P(we win) from the log-ratio of the strengths (ours / theirs): 1/2 when even."""
	return torch.sigmoid(sharpness * ratio)


def fight_return(win, loss, our_army, our_hero, gain):
	"""A fight's expected return in gold: P(win) x (what winning gains - the army it costs) - P(lose) x (the whole
	army and the hero's own worth: a lost battle takes both)."""
	return win * (gain - loss * our_army) - (1 - win) * (our_army + our_hero)


def worst_guard(values, monster, guarded):
	"""For each tile in a wandering monster's zone of control (guarded), the worst value among the monsters next to it
	(stepping there starts their fight); 0 elsewhere. values, monster, guarded [K, W, W]."""
	big = 1e9
	worst = -F.max_pool2d(-torch.where(monster, values, torch.full_like(values, big)).unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
	return torch.where(guarded & (worst < big / 2), worst, torch.zeros_like(worst))


def next_to_monster(monster):
	"""[K, W, W]: a wandering monster we see on the tile or next to it (worst_guard finds a guard there)."""
	return F.max_pool2d(monster.float().unsqueeze(1), 3, stride=1, padding=1).squeeze(1) > 0


def window_worst(values, monster, default):
	"""[K, W, W]: the lowest of values [K, W, W] among the window's wandering monsters we see (each window's own), default
	in a window that shows none: a guard we cannot see, as the worst fight in view."""
	big = 1e9
	low = torch.where(monster, values, torch.full_like(values, big)).flatten(1).amin(dim=1)
	return torch.where(low < big / 2, low, torch.full_like(low, default))[:, None, None].expand_as(values)


# ---- exploration and the map's blocks ----

def unexplored_around(flags, radius=SIGHT_RADIUS):
	"""The share of the map's tiles within `radius` of each window tile that we have not revealed: what stepping
	there would show. flags [K, W, W] -> [K, W, W]."""
	unrevealed = (bit(flags, c.TILE_ON_MAP) & ~bit(flags, c.TILE_REVEALED)).float().unsqueeze(1)
	return F.avg_pool2d(unrevealed, 2 * radius + 1, stride=1, padding=radius, count_include_pad=False).squeeze(1)


def block_side(x):
	"""Tiles a side of the map encoder's blocks [B]: ceil(max(width, height) / MAP_GRID)."""
	size = x["g_map"].clamp(min=1)
	return torch.div(torch.maximum(size[:, 0], size[:, 1]) + G - 1, G, rounding_mode="floor").clamp(min=1)


def centres(x, per_side):
	"""The centres in tiles [B, L * per_side^2, 3] of the map cut into per_side x per_side blocks a level."""
	b = x["g_day"].shape[0]
	span = (block_side(x) * (G // per_side)).float()
	middle = torch.arange(per_side, device=span.device).float() + 0.5
	cy, cx = torch.meshgrid(middle, middle, indexing="ij")
	levels = torch.arange(L, device=span.device).float()
	return torch.stack(
		[
			cx.reshape(1, 1, -1).expand(b, L, -1) * span[:, None, None],
			cy.reshape(1, 1, -1).expand(b, L, -1) * span[:, None, None],
			levels.reshape(1, L, 1).expand(b, L, per_side * per_side),
		],
		dim=-1,
	).reshape(b, L * per_side * per_side, 3)


# Each Obj id's class (the last row: an id past the vocab), and which classes have an OUR_ form (the next class)
CLASS_OF = torch.tensor([int(c.object_class(i)) for i in range(c.OBJECT_VOCAB)] + [int(c.ObjectClass.OTHER)])
OWNED = torch.tensor([kind in c.OWNED_CLASSES for kind in c.ObjectClass])


def object_classes(x):
	"""Each listed object's ObjectClass [B, MO] (-1 none): its type's, the OUR_ form of an owned class when it is ours,
	VISITED once we visited it."""
	obj = x["m_object"]
	kind = moved(CLASS_OF, obj.device)[vocab_index(obj, c.OBJECT_VOCAB)]
	kind = torch.where(moved(OWNED, obj.device)[kind] & (x["m_owner"] == c.OWNER_SELF), kind + 1, kind)
	kind = torch.where(bit(x["m_flags"], c.OBJECT_VISITED_BY_PLAYER), torch.full_like(kind, int(c.ObjectClass.VISITED)), kind)
	return torch.where(obj >= 0, kind, torch.full_like(kind, -1))


def object_counts(x):
	"""Each map block's count of the objects we see, by class [B, L, G, G, N_OBJECT_CLASSES], from the list of all of
	them (the engine counted them until 2026-10-28; what matters is the model's to find)."""
	b = x["m_object"].shape[0]
	kind = object_classes(x)
	side = block_side(x)
	pos = x["m_pos"].clamp(min=0)
	row = torch.div(pos[..., 1], side[:, None], rounding_mode="floor").clamp(max=G - 1)
	column = torch.div(pos[..., 0], side[:, None], rounding_mode="floor").clamp(max=G - 1)
	batch = torch.arange(b, device=pos.device)[:, None]
	cells = b * L * G * G * c.N_OBJECT_CLASSES
	index = ((((batch * L + pos[..., 2].clamp(max=L - 1)) * G + row) * G + column) * c.N_OBJECT_CLASSES + kind.clamp(min=0))
	index = torch.where((kind >= 0) & (pos[..., 2] < L), index, torch.full_like(index, cells)).reshape(-1)
	counts = torch.zeros(cells + 1, device=pos.device).index_add(0, index, torch.ones(index.shape, device=pos.device))
	return counts[:cells].reshape(b, L, G, G, c.N_OBJECT_CLASSES)


def area_contents(x, counts):
	"""Each TRAVEL area's object counts by class [B, L * TG^2, N_OBJECT_CLASSES] (counts: object_counts) and the share
	of its tiles we have not revealed [B, L * TG^2]: the 2 x 2 map blocks it is made of."""
	b = x["g_day"].shape[0]
	merge = G // TG
	objects = counts.reshape(b * L, G, G, c.N_OBJECT_CLASSES).permute(0, 3, 1, 2)
	objects = F.avg_pool2d(objects, merge) * merge * merge
	tiles = x["g_tiles"].float().reshape(b * L, G, G, c.N_GRID_COUNTS).permute(0, 3, 1, 2)
	tiles = F.avg_pool2d(tiles, merge) * merge * merge
	on_map, revealed = tiles[:, c.GridCount.ON_MAP], tiles[:, c.GridCount.REVEALED]
	hidden = ((on_map - revealed) / on_map.clamp(min=1)).reshape(b, L * TG * TG)
	return objects.permute(0, 2, 3, 1).reshape(b, L * TG * TG, c.N_OBJECT_CLASSES), torch.maximum(hidden, unseen_exits(x))


OBJ_WHIRLPOOL = 111  # (Obj::WHIRLPOOL: a teleporter that takes part of the army it carries)


def unseen_exits(x):
	"""[B, L * TG^2], 0/1: the TRAVEL areas that hold a teleporter's entrance no exit of which we have seen
	(OBJECT_UNSEEN_EXIT, schema 17; not a whirlpool, which costs troops): stepping on it shows land we know nothing of,
	a whole sight of it, so the area counts as one we have not revealed (travel_prior's land to reveal, as the entrance's
	tile is MOVE's: model.tile_features). A record from before has no such flag: none."""
	b = x["g_day"].shape[0]
	flagged = bit(x["m_flags"], c.OBJECT_UNSEEN_EXIT) & (x["m_object"] != OBJ_WHIRLPOOL)
	index = area_index(x, x["m_pos"])
	areas = L * TG * TG
	index = torch.where(flagged & (index >= 0), index, torch.full_like(index, areas))
	return torch.zeros(b, areas + 1, device=index.device).scatter_reduce(1, index, flagged.float(), reduce="amax", include_self=True)[:, :areas]


def movement_discount(days, exponent):
	"""NK2's discount of a target's value by the days to reach it (PriorityEvaluator::evaluateMovement): 1 / (0.75 +
	days^exponent), its exponent 1.3."""
	return 1.0 / (0.75 + days.clamp(min=0) ** exponent)


# ---- our previous decision ----

# What each action type's table's first axis is (our heroes and towns are 8 each: the shapes cannot tell), and the shapes
A = c.ActionType
FIRST_AXIS = {A.MOVE: "hero", A.CAST: "hero", A.BUY: "hero", A.DISMISS: "hero", A.TRAVEL: "hero", A.BUILD: "town", A.RECRUIT: "town", A.HIRE: "town", A.ARMY: "holder", A.UPGRADE: "holder"}
TABLES = {int(t): SPECS[out].shape for t, (out, _) in HEADS.items() if t in FIRST_AXIS}


def last_actors(x):
	"""What our previous turn decision today was about (g_last_*, the model's own choice, echoed by the engine): which
	of our heroes [B, HO] and towns [B, TO] (its table's first axis; ARMY's destination holder too), 0/1."""
	kind, index = x["g_last_type"], x["g_last_index"]
	heroes = torch.zeros(kind.shape[0], c.OWN_HEROES, device=kind.device)
	towns = torch.zeros(kind.shape[0], c.OWN_TOWNS, device=kind.device)
	for t, shape in TABLES.items():
		chosen = ((kind == t) & (index >= 0)).float().unsqueeze(-1)
		axes = [torch.div(index.clamp(min=0), math.prod(shape[1:]), rounding_mode="floor")]
		if t == int(A.ARMY):  # (holder, slot, holder, amount): where the troops went too
			axes.append(torch.remainder(torch.div(index.clamp(min=0), shape[3], rounding_mode="floor"), shape[2]))
		for axis in axes:
			one = F.one_hot(axis.clamp(max=shape[0] - 1), shape[0]).float() * chosen
			if FIRST_AXIS[A(t)] == "hero":
				heroes = heroes + one
			elif FIRST_AXIS[A(t)] == "town":
				towns = towns + one
			else:  # holders: our heroes, then our towns, then the other army
				heroes, towns = heroes + one[:, : c.OWN_HEROES], towns + one[:, c.OWN_HEROES : c.OWN_HEROES + c.OWN_TOWNS]
	return heroes.clamp(max=1), towns.clamp(max=1)


# ---- Nullkiller2's brain, as features ----
# (2026-10-28, the user: "consume all the NK2 brain as ops that create features that can be easily used in the graph and
# give sane gradients".) Each is Nullkiller2's own computation (VCMI's AI/Nullkiller2) from the inputs alone, scaled to
# a few units; the model reads them through weights that start at 0, and where Nullkiller2's habit is a plain rule, as a
# prior with a learnable bounded weight (CustomFeatures)

MAX_GOLD_PRESSURE = 0.3  # nk2ai-settings.json maxGoldPressure: over it Nullkiller2 is short of gold
CITADEL, CASTLE, TOWN_HALL, CITY_HALL, CAPITOL = 8, 9, 11, 12, 13  # building ids
MAGE_GUILD_LEVELS = 5  # building ids 0..4: the mage guild's levels 1..5
SPELLBOOK = 0  # artifact id
SKILLS = (  # VCMI's secondary skill ids, in order (vocab secondary_skills)
	"pathfinding", "archery", "logistics", "scouting", "diplomacy", "navigation", "leadership", "wisdom", "mysticism", "luck", "ballistics", "eagleEye",
	"necromancy", "estates", "fireMagic", "airMagic", "waterMagic", "earthMagic", "scholar", "tactics", "artillery", "learning", "offence", "armorer",
	"intelligence", "sorcery", "resistance", "firstAid",
)
# Nullkiller2's score of each skill for a main hero and for a scout (HeroManager's SecondarySkillScoreMap); the rest 0
MAIN_SKILLS = {
	"diplomacy": 2, "logistics": 2, "earthMagic": 2, "armorer": 2, "offence": 2, "airMagic": 1, "wisdom": 1, "leadership": 1, "intelligence": 1,
	"resistance": 1, "mysticism": -1, "sorcery": -1, "estates": -1, "firstAid": -1, "learning": -1, "scholar": -1, "eagleEye": -1, "navigation": -1,
}
SCOUT_SKILLS = {"logistics": 2, "estates": 2, "pathfinding": 1, "scholar": 1}
SCHOOLS = ("fireMagic", "airMagic", "waterMagic", "earthMagic")


def skill_table(scores):
	return torch.tensor([float(scores.get(name, 0)) for name in SKILLS])


MAIN_SKILL_SCORE, SCOUT_SKILL_SCORE = skill_table(MAIN_SKILLS), skill_table(SCOUT_SKILLS)


def own_towns(x):
	return x["t_present"][:, : c.OWN_TOWNS] & (x["t_owner"][:, : c.OWN_TOWNS] == c.OWNER_SELF)


def hero_roles(x, hero_strength):
	"""Nullkiller2's main heroes [B, HO] (HeroManager::update): ours by fighting strength, the first max(towns, 1) of
	them (a bigger map adds some after day 21), at most twice the towns; the rest are scouts."""
	ours = x["h_present"][:, :HO] & (x["h_owner"][:, :HO] == c.OWNER_SELF)
	towns = own_towns(x).sum(dim=1)
	bigger = (x["g_day"] > 21).long() * torch.div(x["g_map"][:, 0], 108, rounding_mode="floor")
	mains = torch.minimum((towns + bigger).clamp(min=1), 2 * towns)
	strength = torch.where(ours, hero_strength[:, :HO], torch.full_like(hero_strength[:, :HO], -1e9))
	index = torch.arange(HO, device=strength.device)
	stronger = (strength[:, None, :] > strength[:, :, None]) | ((strength[:, None, :] == strength[:, :, None]) & (index[None, None, :] < index[None, :, None]))
	rank = (stronger & ours[:, None, :]).sum(dim=-1)
	return ours & (rank < mains[:, None])


def skill_scores(x, main, t):
	"""Nullkiller2's score of each option of a level-up [B, O] (SecondarySkillEvaluator): its table's (a main hero's or a
	scout's), +1.5 for a new skill worth 2 (or 1, when the hero has at most one upgrade left), and for a main hero +1.5
	for Wisdom past level 10 without it, +1 for a first school of magic; 0 for options that are no skill."""
	b = x["g_day"].shape[0]
	rows = torch.arange(b, device=main.device)
	hero = x["d_hero"].clamp(0, HO - 1)
	known = x["h_skills"][rows, hero]  # [B, N_SECONDARY_SKILLS], 0..3
	is_main = (main[rows, hero] & (x["d_hero"] >= 0)).unsqueeze(-1)
	skill = x["o_type"].clamp(0, c.N_SECONDARY_SKILLS - 1)
	is_skill = (x["o_kind"] == c.OptionKind.SKILL) & (x["o_type"] >= 0)
	score = torch.where(is_main, moved(MAIN_SKILL_SCORE, main.device)[skill], moved(SCOUT_SKILL_SCORE, main.device)[skill])
	new = torch.gather(known, 1, skill) == 0
	upgrades_left = ((3 - known) * (known > 0)).sum(dim=-1, keepdim=True)
	score = score + 1.5 * (new & ((score >= 2) | ((score >= 1) & (upgrades_left <= 1)))).float()
	level = torch.floor(fractional_level(x["h_exp"][rows, hero].clamp(min=0).float(), t.level_experience)).unsqueeze(-1)
	wisdom = SKILLS.index("wisdom")
	score = score + 1.5 * (is_main & (skill == wisdom) & (level > 10) & (known[:, wisdom : wisdom + 1] == 0)).float()
	schools = constant([SKILLS.index(name) for name in SCHOOLS], main.device)
	no_magic = ~(known[:, schools] > 0).any(dim=-1, keepdim=True)
	score = score + (is_main & (skill[..., None] == schools).any(dim=-1) & no_magic).float()
	return score * is_skill


def gold_pressure(x, t):
	"""Nullkiller2's gold pressure [B] (BuildAnalyzer::calculateGoldPressure): what its plans cost, the army its towns
	offer and each town's next hall, over 1 + 2 x the gold it has + a week's gold income."""
	own = own_towns(x)
	creature = x["t_available_creature"][:, : c.OWN_TOWNS]
	army = ((x["t_available"][:, : c.OWN_TOWNS].clamp(min=0).float() * t.creature_cost[vocab_index(creature, c.CREATURE_VOCAB)] * (creature >= 0)).sum(dim=-1) * own).sum(dim=-1)
	built, cost = x["t_built"][:, : c.OWN_TOWNS], t.building_cost[vocab_index(x["t_faction"][:, : c.OWN_TOWNS], c.FACTION_VOCAB)]
	hall = torch.zeros_like(army.unsqueeze(-1).expand(-1, c.OWN_TOWNS))
	for building in (CAPITOL, CITY_HALL, TOWN_HALL):  # the first hall not built (the cheapest last: it overrides)
		hall = torch.where(built[..., building], hall, cost[..., building])
	development = (hall * own).sum(dim=-1)
	gold, income = x["g_resources"][:, 6].clamp(min=0).float(), x["g_income"][:, 6].clamp(min=0).float()
	return (army + development) / (1 + 2 * gold + 7 * income)


def recruit_pause(x, pressure, town_danger):
	"""Nullkiller2 buys no army while short of gold if a town of ours has no City Hall yet and nothing threatens it
	(BuyArmyBehavior) [B], 0..1."""
	waiting = own_towns(x) & ~x["t_built"][:, : c.OWN_TOWNS, CITY_HALL] & (town_danger < 0.5)
	return torch.sigmoid(10 * (pressure - MAX_GOLD_PRESSURE)) * waiting.any(dim=1).float()


def build_rewards(x, t):
	"""Nullkiller2's view of each building of our towns [B, TO, N_BUILDINGS, 4] (BuildThis), scaled: half a week of the
	daily income it adds (gold); 1.5 x a week's growth of its dwelling's creature in strength (an upgraded dwelling:
	what the upgrade adds; a citadel or castle: half the town's weekly army); its cost (gold); a mage guild's level."""
	faction = vocab_index(x["t_faction"][:, : c.OWN_TOWNS], c.FACTION_VOCAB)
	strength = lambda creature: t.fight_value[vocab_index(creature, c.CREATURE_VOCAB)] * (creature >= 0)  # noqa: E731
	creature, base = t.dwelling_creature[faction], t.dwelling_base[faction]
	growth = t.creature_growth[vocab_index(creature, c.CREATURE_VOCAB)]
	army = torch.where(base >= 0, growth * (strength(creature) - strength(base)), 1.5 * growth * strength(creature))
	town_army = (x["t_growth"][:, : c.OWN_TOWNS].clamp(min=0).float() * strength(x["t_available_creature"][:, : c.OWN_TOWNS])).sum(dim=-1)
	buildings = torch.arange(c.N_BUILDINGS, device=army.device)
	army = army + ((buildings == CITADEL) | (buildings == CASTLE)).float() * town_army.unsqueeze(-1) / 2
	mage = torch.where(buildings < MAGE_GUILD_LEVELS, (buildings + 1).float() / MAGE_GUILD_LEVELS, torch.zeros_like(buildings, dtype=army.dtype))
	return torch.stack(
		[torch.log1p(3.5 * t.building_income[faction].clamp(min=0) / 1000), torch.log1p(army.clamp(min=0) / 1000), torch.log1p(t.building_cost[faction] / 1000), mage.expand_as(army)], dim=-1
	)


def upgrade_gains(x, t):
	"""What upgrading each stack of ours adds [B, HD, A, 2] (Nullkiller2's makePossibleUpgrades upgrades every stack it
	can afford, to the upgrade worth the most), scaled: its count x the strength the upgrade adds; what it costs."""
	creature = torch.cat([x["h_army_creature"][:, :HO], x["t_garrison_creature"][:, : c.OWN_TOWNS], x["x_creature"].unsqueeze(1)], dim=1)
	count = torch.cat([x["h_army_count"][:, :HO], x["t_garrison_count"][:, : c.OWN_TOWNS], x["x_count"].unsqueeze(1)], dim=1).clamp(min=0).float()
	now = vocab_index(creature, c.CREATURE_VOCAB)
	up = t.creature_upgrade[now]
	better = (creature >= 0) & (up >= 0)
	after = vocab_index(up, c.CREATURE_VOCAB)
	gain = count * (t.fight_value[after] - t.fight_value[now]) * better
	cost = count * (t.creature_cost[after] - t.creature_cost[now]).clamp(min=0) * better
	return torch.stack([torch.log1p(gain.clamp(min=0) / 1000), torch.log1p(cost / 1000)], dim=-1)


def spellbook_wanted(x, main):
	"""[B, HO] Nullkiller2 buys a spellbook for a main hero who has none (in a town with a mage guild, gold permitting:
	the BUY mask's)."""
	return main & ~(x["h_artifacts"][:, :HO] == SPELLBOOK).any(dim=-1)


# ---- Nullkiller2's plans across days, as weak pulls ----
# (2026-10-28, the user: "weak features, also make it easy to ignore them".) Each is a value in gold; the model takes it
# as a feature through weights that start at 0, and as a prior of a bounded weight that starts at half a logit and can
# fall to 0 (CustomFeatures.pull_weights)

PULL_SCALE = 5000.0  # gold: a pull's prior is its weight x tanh(gold / PULL_SCALE)


def reinforcements(x, t, town_gold):
	"""What each town of ours can give a hero that comes [B, TO] (gold): its garrison, and as much of its army for hire
	as our gold buys (Nullkiller2's BuyArmy for a main hero: howManyReinforcementsCanGet)."""
	creature = x["t_available_creature"][:, : c.OWN_TOWNS]
	for_hire = (x["t_available"][:, : c.OWN_TOWNS].clamp(min=0).float() * t.creature_cost[vocab_index(creature, c.CREATURE_VOCAB)] * (creature >= 0)).sum(dim=-1)
	gold = x["g_resources"][:, 6].clamp(min=0).float()
	return (town_gold[:, : c.OWN_TOWNS] + torch.minimum(for_hire, gold.unsqueeze(-1))) * own_towns(x)


def town_stakes(x, t, town_gold):
	"""What each town of ours is worth to us [B, TO] (gold): its buildings at cost and its garrison (DefenceBehavior)."""
	faction = vocab_index(x["t_faction"][:, : c.OWN_TOWNS], c.FACTION_VOCAB)
	buildings = (x["t_built"][:, : c.OWN_TOWNS].float() * t.building_value[faction]).sum(dim=-1)
	return (buildings + town_gold[:, : c.OWN_TOWNS]) * own_towns(x)


def hire_want(x, pressure, town_danger, kinds):
	"""Nullkiller2 hires its tavern's best hero (RecruitHeroBehavior) when we have no hero, see over 5 treasures a hero of
	ours, have a threat near or gold to spare (its hero brings an army worth over half a hire, as most do), or gold piles
	up (10,000 with a Capitol, 30,000); its priority falls with our heroes, 3 / (heroes + 1). [B] 0..1 (kinds: the listed
	objects' classes, object_classes)."""
	heroes = (x["h_present"][:, :HO] & (x["h_owner"][:, :HO] == c.OWNER_SELF)).sum(dim=1).float()
	treasures = ((kinds == c.ObjectClass.PICKUP) | (kinds == c.ObjectClass.TREASURE)).sum(dim=1).float()
	spare = 1 - torch.sigmoid(10 * (pressure - MAX_GOLD_PRESSURE))
	gold = x["g_resources"][:, 6].clamp(min=0).float()
	capitol = (own_towns(x) & x["t_built"][:, : c.OWN_TOWNS, CAPITOL]).any(dim=1).float()
	want = torch.stack(
		[
			(heroes == 0).float(), torch.sigmoid(treasures - 5 * heroes), torch.maximum(town_danger.amax(dim=1), spare),
			torch.sigmoid((gold - 10000) / 1000) * spare * capitol, torch.sigmoid((gold - 30000) / 1000) * spare,
		],
		dim=-1,
	).amax(dim=-1)
	return want * (3 / (heroes + 1)).clamp(max=1)


def trade_needs(x, t):
	"""What trading could make buildable (Nullkiller2's ResourceTrader trades toward its planned building) [B, R, 2]: the
	gold value of each resource the buildings short of resources lack (build_short: the most any lacks), and of what we
	have of it beyond what they need; scaled."""
	short = x["build_short"][:, : c.OWN_TOWNS].float()  # [B, TO, NB]
	faction = vocab_index(x["t_faction"][:, : c.OWN_TOWNS], c.FACTION_VOCAB)
	cost = t.building_resources[faction] * short.unsqueeze(-1)  # [B, TO, NB, R]
	have = x["g_resources"].clamp(min=0).float()[:, None, None, :]
	lack = torch.relu(cost - have).amax(dim=(1, 2))  # [B, R]
	spare = torch.relu(have[:, 0, 0] - cost.amax(dim=(1, 2)))
	price = t.resource_price
	return torch.stack([torch.log1p(lack * price / 1000), torch.log1p(spare * price / 1000)], dim=-1)


def market_rates(x, t):
	"""VCMI's market at our marketplaces' rate (IMarket::getOffer: its efficiency min((markets + 1) / 20, 0.5) of the
	resource prices): (give, get) [B, R (sold), R (bought)], give units sold for get units bought, as the agent trades."""
	price = t.resource_price  # [R] (VCMI's resource prices)
	efficiency = ((x["g_markets"].float() + 1.0) / 20.0).clamp(max=0.5)  # [B]
	given = price[None, :, None]  # [1, R (sold), 1]
	wanted = price[None, None, :] / efficiency[:, None, None]  # [B, 1, R (bought)]
	dearer = given > wanted
	give = torch.where(dearer, torch.ones_like(given * wanted), torch.floor(wanted / given + 0.5))  # [B, R, R]
	get = torch.where(dearer, torch.ceil(given / wanted), torch.ones_like(given * wanted))
	return give, get


def purchase_trades(x, t):
	"""What a purchase short of resources costs Φ in the trades the agent makes for it (AlphaMapAI::payByTrades, since
	2026-11-04: building or recruiting trades for itself, the user's): (each building's [B, TO, N_BUILDINGS] where
	build_short, one creature's of each RECRUIT level [B, TO, N_LEVELS] where we cannot afford one, as the gold of worth
	lost: the resources sold at their prices less those bought, as Φ counts them; whether we cannot afford that one
	creature [B, TO, N_LEVELS]). Each resource short bought from the spare one that loses the least a unit (market_rates;
	the spare's depletion left out: the agent's plan is exact)."""
	price = t.resource_price
	give, get = market_rates(x, t)
	have = x["g_resources"].clamp(min=0).float()  # [B, R]
	unit = price[None, :, None] * give / get  # [B, R (sold), R (bought)]: what a unit bought gives away

	def lost(cost):  # cost [B, TO, K, R] -> [B, TO, K]
		mine = have[:, None, None, :]
		short, spare = (cost - mine).clamp(min=0.0), (mine - cost).clamp(min=0.0)
		usable = spare.unsqueeze(-1) >= give[:, None, None]  # [B, TO, K, R (sold), R (bought)]
		best = torch.where(usable, unit[:, None, None], torch.full_like(usable, float("inf"), dtype=unit.dtype)).amin(dim=-2)
		gold = torch.where(torch.isfinite(best) & (short > 0), short * (best - price), torch.zeros_like(short))
		return gold.sum(dim=-1).clamp(min=0.0)

	faction = vocab_index(x["t_faction"][:, : c.OWN_TOWNS], c.FACTION_VOCAB)
	creature = t.creature_resources[vocab_index(x["t_available_creature"][:, : c.OWN_TOWNS], c.CREATURE_VOCAB)]  # [B, TO, LV, R]
	needs = creature > 0
	cannot = (needs & (have[:, None, None, :] < creature)).any(dim=-1)  # (not one creature affordable alone)
	return lost(t.building_resources[faction]) * x["build_short"][:, : c.OWN_TOWNS], lost(creature) * cannot, cannot


def trade_options(x, t):
	"""Each TRADE entry's (sell, buy, TRADE_FRACTIONS) [B, R, R, F, 2] (the user, 2026-11-04: every trade is dumb unless
	it buys what could not be bought without it): what it costs Φ (the gold of worth lost: the resources sold at their
	prices less those bought; joint7's trades cost a median 900) and whether it unlocks a purchase: a building (one of
	our towns' buildings short of resources, build_short) or a creature for hire in our towns we cannot afford one of,
	either short of the one resource bought alone and affordable after it (Nullkiller2 mostly sells rare resources for
	gold: 6% of its advised trades unlocked a building, joint7 2026-11-04). The amounts as the agent's Encoder makes the
	entries: VCMI's market (IMarket::getOffer) at our marketplaces' rate, each fraction of the trades our stock of the
	resource sold allows, rounded up, at least one."""
	price = t.resource_price  # [R] (VCMI's resource prices)
	give, get = market_rates(x, t)
	have = x["g_resources"].clamp(min=0).float()  # [B, R]
	units = torch.floor(have[:, :, None] / give.clamp(min=1.0))
	trades = torch.ceil(units.unsqueeze(-1) * constant(c.TRADE_FRACTIONS, price.device, torch.float32)).clamp(min=1.0)  # [B, R, R, F]
	sold, bought = trades * give.unsqueeze(-1), trades * get.unsqueeze(-1)
	cost = sold * price[None, :, None, None] - bought * price[None, None, :, None]
	building = t.building_resources[vocab_index(x["t_faction"][:, : c.OWN_TOWNS], c.FACTION_VOCAB)]  # [B, TO, NB, R]
	creature = x["t_available_creature"][:, : c.OWN_TOWNS]
	recruit = t.creature_resources[vocab_index(creature, c.CREATURE_VOCAB)]  # [B, TO, LV, R]
	need = torch.cat([building, recruit], dim=2).permute(0, 3, 1, 2)  # [B, R, TO, NB + LV]
	wanted = torch.cat([x["build_short"][:, : c.OWN_TOWNS], (creature >= 0) & (x["t_available"][:, : c.OWN_TOWNS] > 0)], dim=2)
	short = (need > have[:, :, None, None]) & wanted[:, None]  # [B, R, TO, NB + LV]
	alone = short & (short.sum(dim=1, keepdim=True) == 1)  # short of this resource only
	covers = have[:, None, :, None, None, None] + bought[..., None, None] >= need[:, None, :, None]  # [B, R, R, F, TO, NB]
	keeps = have[:, :, None, None, None, None] - sold[..., None, None] >= need[:, :, None, None]
	unlocks = (alone[:, None, :, None] & covers & keeps).flatten(4).any(dim=-1)  # [B, R, R, F]
	# or a hero for hire in a tavern of ours we cannot pay for now and can after it (2026-11-05: a trade that buys or hires)
	gold = c.N_RESOURCES - 1
	tavern = (x["g_tavern"] >= 0).any(dim=1)  # [B]
	gold_after = have[:, gold, None, None, None] + bought * (torch.arange(c.N_RESOURCES, device=have.device) == gold)[None, None, :, None] - sold * (torch.arange(c.N_RESOURCES, device=have.device) == gold)[None, :, None, None]
	hires = (tavern & (have[:, gold] < HERO_GOLD_COST))[:, None, None, None] & (gold_after >= HERO_GOLD_COST)
	return torch.stack([cost.clamp(min=0.0), (unlocks | hires).float()], dim=-1)


def area_index(x, pos):
	"""The TRAVEL area [..] (level, row, column of TRAVEL_GRID) of map positions pos [B, .., 3], -1 off the levels."""
	side = (2 * block_side(x)).reshape((-1,) + (1,) * (pos.dim() - 2))
	row = torch.div(pos[..., 1].clamp(min=0), side, rounding_mode="floor").clamp(max=TG - 1)
	column = torch.div(pos[..., 0].clamp(min=0), side, rounding_mode="floor").clamp(max=TG - 1)
	return torch.where(pos[..., 2] < L, (pos[..., 2].clamp(min=0) * TG + row) * TG + column, torch.full_like(row, -1))


def blocked_by_guards(value, region, zone, monster, steps=c.WINDOW_RADIUS):
	"""Kill a guard, take what it blocks (Nullkiller2's ClusterBehavior) [K, W, W]: on each wandering monster's tile, the
	best object value in a region we reach only through its zone. value: the object value (gold) of each tile we cannot
	reach (an object's tile is blocked: never one of the region's); region: passable tiles we cannot reach, outside every
	zone; zone: the monsters' zones; monster: their tiles. The best value spreads from the objects through the region
	(steps hops), then onto the zone tiles next to it and the monsters they surround. (Until 2026-11-04 the values were
	taken from the region's tiles only, where no object stands: it was 0 on every tile of joint7's games.)"""
	pool = lambda v: F.max_pool2d(v.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)  # noqa: E731
	best = torch.where(region | (value > 0), value.clamp(min=0), torch.zeros_like(value))
	for _ in range(steps):
		best = torch.where(region, torch.maximum(best, pool(best)), best)
	return pool(pool(best) * zone) * monster



# ---- the economy and what each building adds (e5 for the user, 2026-11-12: production a first-class quantity) ----
# Each quantity is a tiny pure function _feature_<name> returning a bounded tensor; economy_features and build_features
# collect them under ECONOMY and BUILD_ENTRY. Gold values are list prices at the shared resource prices (vocab).

ECONOMY = (
	"gold_over_pool", "pool_unbought", "gold_idle", "flow_balance", "hall_income_used", "production_built",
	"production_week_3k", "production_week_10k", "production_week_30k", "days_to_growth",
)
BUILD_ENTRY = (
	"income_add", "income_rel", "production_add", "production_rel", "payable", "unlocks", "cost_share", "crowds_out",
	"upgrade_quality", "upgrade_held", "upgrade_elsewhere", "end_of_week",
)
DWELLINGS_BASE = tuple(range(30, 37))  # BuildingID DWELL_LVL_1..7
DWELLINGS_UP = tuple(range(37, 44))  # DWELL_UP_LVL_1..7
HALLS = (10, 11, 12, 13)  # village, town, city hall, capitol
CITADEL, CASTLE_BUILDING = 8, 9
ECON_NEAR = 45.0  # tiles: a stack about three days from a town, exp(-d / 45) (F3's nearness)
CAPITOL_INCOME, CITY_HALL_INCOME = 4000.0, 2000.0


def _economy_inputs(x, t):
	"""What the economy's features share: (G gold, I daily gold income, C_pool for hire in our towns, C_week next Monday's
	growth, B_today our towns' dearest legal builds not yet built today, own [B, TO], faction [B, TO]), each at list price."""
	to = c.OWN_TOWNS
	own = own_towns(x)
	faction = vocab_index(x["t_faction"][:, :to], c.FACTION_VOCAB)
	creature = x["t_available_creature"][:, :to]
	cost = t.creature_cost[vocab_index(creature, c.CREATURE_VOCAB)] * (creature >= 0)
	pool = ((x["t_available"][:, :to].clamp(min=0).float() * cost).sum(dim=-1) * own).sum(dim=-1)
	week = ((x["t_growth"][:, :to].clamp(min=0).float() * cost).sum(dim=-1) * own).sum(dim=-1)
	legal = x["build_mask"][:, :to].bool() & own.unsqueeze(-1) & (x["t_built_today"][:, :to] == 0).unsqueeze(-1)
	dearest = torch.where(legal, t.building_cost[faction], torch.zeros_like(t.building_cost[faction])).amax(dim=-1).sum(dim=-1)
	gold, income = x["g_resources"][:, 6].clamp(min=0).float(), x["g_income"][:, 6].clamp(min=0).float()
	return gold, income, pool, week, dearest, own, faction


def _feature_gold_over_pool(gold, pool):
	"""0..1: our gold against it and what our towns have for hire; 1 all gold, 0 none."""
	return gold / (gold + pool + 1000.0)


def _feature_pool_unbought(gold, pool):
	"""0..1: the share of what is for hire our gold cannot pay for; 0 we can buy it all."""
	return torch.relu(pool - gold) / (pool + 1000.0)


def _feature_gold_idle(gold, pool, dearest):
	"""0..1: gold left over after buying everything for hire and each town's dearest build; 0 none idle."""
	return torch.relu(gold - pool - dearest) / (gold + 1000.0)


def _feature_flow_balance(income, week):
	"""0..1: a week's gold income against what a week's growth costs; 1 income far above, 0 no income."""
	return 7.0 * income / (7.0 * income + week + 1000.0)


def _feature_hall_income_used(x, t, own, faction):
	"""0..1: our halls' income against one capitol and city halls everywhere else; 1 all built."""
	built = x["t_built"][:, : c.OWN_TOWNS].bool()
	halls = (t.building_income[faction][..., list(HALLS)] * built[..., list(HALLS)]).sum(dim=-1)
	cap = CAPITOL_INCOME + CITY_HALL_INCOME * (own.sum(dim=-1).float() - 1).clamp(min=0)
	return ((halls * own).sum(dim=-1) / cap).clamp(0.0, 1.0)


def _dwelling_week(t, faction):
	"""[B, TO, 7]: a week's list price of each level's base dwelling's growth (growth x cost of its creature)."""
	creature = t.dwelling_creature[faction][..., list(DWELLINGS_BASE)]
	index = vocab_index(creature, c.CREATURE_VOCAB)
	return t.creature_growth[index] * t.creature_cost[index] * (creature >= 0)


def _feature_production_built(x, t, own, faction):
	"""0..1: the week's list price of the base dwellings built, over all the levels our towns' factions have; 1 all built."""
	built = x["t_built"][:, : c.OWN_TOWNS].bool()[..., list(DWELLINGS_BASE)] | x["t_built"][:, : c.OWN_TOWNS].bool()[..., list(DWELLINGS_UP)]
	week = _dwelling_week(t, faction)
	have, could = ((week * built).sum(dim=-1) * own).sum(dim=-1), (week.sum(dim=-1) * own).sum(dim=-1)
	return have / could.clamp(min=1.0)


def _feature_production_week(week, scale):
	"""0..1: next Monday's growth at list price, half-way at scale."""
	return week / (week + scale)


def _feature_days_to_growth(x):
	"""0..1: days until the next Monday over 7; 1 it is Monday, 1/7 it is Sunday."""
	return (7.0 - torch.remainder(x["g_day"].float() - 1.0, 7.0)) / 7.0


def economy_features(x, t):
	"""[B, len(ECONOMY)]: our economy's bounded views (each _feature_ above, ECONOMY's order); 0 with the economy off
	(t.economy_on: the model as before, bit for bit)."""
	gold, income, pool, week, dearest, own, faction = _economy_inputs(x, t)
	return t.economy_on() * torch.stack(
		[
			_feature_gold_over_pool(gold, pool), _feature_pool_unbought(gold, pool), _feature_gold_idle(gold, pool, dearest),
			_feature_flow_balance(income, week), _feature_hall_income_used(x, t, own, faction), _feature_production_built(x, t, own, faction),
			_feature_production_week(week, 3000.0), _feature_production_week(week, 10000.0), _feature_production_week(week, 30000.0),
			_feature_days_to_growth(x),
		],
		dim=-1,
	)


def _fort_multiplier(built):
	"""[B, TO]: a town's growth multiplier from its fort line (a citadel +50%, a castle +50% more)."""
	return 1.0 + 0.5 * (built[..., CITADEL] | built[..., CASTLE_BUILDING]).float() + 0.5 * built[..., CASTLE_BUILDING].float()


def _production_added(x, t, faction, built):
	"""[B, TO, N_BUILDINGS]: the list price a week each building adds to its town's growth: a base dwelling its growth x
	cost x the fort's multiplier; a citadel and a castle each half the town's built base dwellings' week; a horde its
	extra growth (half its level's base growth: the engine's number is in the H3 data, not the configs; H3's are 43-67% of
	base, griffins +3 of 7, imps +8 of 15, dendroids +2 of 3) x cost where that
	level is built; an upgraded dwelling growth x (its creature's cost less the base's) only."""
	week = _dwelling_week(t, faction)  # [B, TO, 7]
	levels = built[..., list(DWELLINGS_BASE)] | built[..., list(DWELLINGS_UP)]
	added = torch.zeros(built.shape, dtype=week.dtype, device=week.device)
	added[..., list(DWELLINGS_BASE)] = week * _fort_multiplier(built).unsqueeze(-1)
	half = 0.5 * (week * levels).sum(dim=-1)
	added[..., CITADEL], added[..., CASTLE_BUILDING] = half, half
	level = t.horde_level[faction]  # [B, TO, N_BUILDINGS]
	at = level.clamp(min=0)
	horde = 0.5 * torch.gather(week, -1, at) * torch.gather(levels.float(), -1, at) * (level >= 0)
	added = torch.where(level >= 0, horde, added)
	up, base = t.dwelling_creature[faction][..., list(DWELLINGS_UP)], t.dwelling_base[faction][..., list(DWELLINGS_UP)]
	up_i, base_i = vocab_index(up, c.CREATURE_VOCAB), vocab_index(base, c.CREATURE_VOCAB)
	added[..., list(DWELLINGS_UP)] = t.creature_growth[base_i] * (t.creature_cost[up_i] - t.creature_cost[base_i]).clamp(min=0) * ((up >= 0) & (base >= 0))
	return added


def _feature_income_add(income_added):
	"""0..1: the daily income a building adds, 3.5 days of it half-way at 3,500; 0 none."""
	v = 3.5 * income_added
	return v / (v + 3500.0)


def _feature_income_rel(income_added, income):
	"""0..1: the daily income it adds against our income with it; 0 none."""
	return income_added / (income.reshape(-1, 1, 1) + income_added + 500.0)


def _feature_production_add(production_added):
	"""0..1: the week's growth it adds at list price, half-way at 2,000; 0 none."""
	return production_added / (production_added + 2000.0)


def _feature_production_rel(production_added, week):
	"""0..1: the week's growth it adds against next Monday's with it; 0 none."""
	return production_added / (week.reshape(-1, 1, 1) + production_added + 500.0)


def _feature_payable(production_added, gold, income, cost, pool, week):
	"""0..1: how much of what it adds a week our gold can buy after paying for it and for what is already for hire and due
	Monday (a week's income in); 0 none, or it adds no growth."""
	spare = gold.reshape(-1, 1, 1) - cost + 7.0 * income.reshape(-1, 1, 1) - pool.reshape(-1, 1, 1) - week.reshape(-1, 1, 1)
	return torch.where(production_added > 0, (spare / (production_added + 1.0)).clamp(0.0, 1.0), torch.zeros_like(production_added))


def _unlock_value(t, faction, built, own_value):
	"""[B, TO, N_BUILDINGS] gold: for a building not built, the best own value among the unbuilt buildings it is a missing
	prerequisite of, each over its count of missing prerequisites (a building two steps from a capitol inherits half)."""
	requires = t.building_requires[faction]  # [B, TO, N_BUILDINGS (b'), N_BUILDINGS (b)]
	missing = requires & ~built.unsqueeze(-2) & ~built.unsqueeze(-1)  # (b' not built, needs b, b not built)
	count = missing.sum(dim=-1, keepdim=True).clamp(min=1).to(own_value.dtype)
	share = torch.where(missing, (own_value.unsqueeze(-1) / count).expand_as(missing.float()), torch.zeros((), dtype=own_value.dtype, device=own_value.device))
	return share.amax(dim=-2)


def _feature_unlocks(unlock):
	"""0..1: unlock_value's gold, half-way at 3,500; 0 it opens nothing of worth."""
	return unlock / (unlock + 3500.0)


def _feature_cost_share(cost, gold, worth):
	"""0..1: its cost against our gold and resources' worth; 1 it takes everything."""
	return (cost / (gold.reshape(-1, 1, 1) + worth.reshape(-1, 1, 1) + 1000.0)).clamp(0.0, 1.0)


def _feature_crowds_out(gold_cost, gold, pool, unbought):
	"""0..1: the share of what is for hire it leaves us unable to pay for, beyond what we cannot already; 0 none."""
	left = torch.relu(gold.reshape(-1, 1, 1) - gold_cost)
	return (torch.relu(pool.reshape(-1, 1, 1) - left) / (pool.reshape(-1, 1, 1) + 1000.0) - unbought.reshape(-1, 1, 1)).clamp(0.0, 1.0)


def _feature_upgrade_quality(t, faction):
	"""[B, TO, 7] 0..1: an upgraded dwelling's creature over its base in fight value, less 1; 0 no better."""
	up, base = t.dwelling_creature[faction][..., list(DWELLINGS_UP)], t.dwelling_base[faction][..., list(DWELLINGS_UP)]
	up_v, base_v = t.fight_value[vocab_index(up, c.CREATURE_VOCAB)], t.fight_value[vocab_index(base, c.CREATURE_VOCAB)]
	return ((up_v - base_v) / base_v.clamp(min=1.0)).clamp(0.0, 1.0) * ((up >= 0) & (base >= 0))


def _nearness(a, b):
	"""[B, N, M] exp(-tile distance / ECON_NEAR) between positions a [B, N, 3] and b [B, M, 3]; x0.25 across levels."""
	d = torch.maximum((a[:, :, None, 0] - b[:, None, :, 0]).abs(), (a[:, :, None, 1] - b[:, None, :, 1]).abs()).float()
	return torch.exp(-d / ECON_NEAR) * torch.where(a[:, :, None, 2] == b[:, None, :, 2], 1.0, 0.25)


def _feature_upgrade_held(x, t, faction):
	"""[B, TO, 7] 0..1: what upgrading the base creatures we hold would add (count x the cost difference), each stack by
	how near it is to the town (heroes' armies, garrisons, those for hire); half-way at 2,000; 0 none of them."""
	to, ho = c.OWN_TOWNS, HO
	own_h = (x["h_present"][:, :ho] & (x["h_owner"][:, :ho] == c.OWNER_SELF)).unsqueeze(-1)
	own_t = own_towns(x).unsqueeze(-1)
	creature = torch.cat([x["h_army_creature"][:, :ho], x["t_garrison_creature"][:, :to], x["t_available_creature"][:, :to]], dim=1)  # [B, HO + 2 TO, slots]
	count = torch.cat([x["h_army_count"][:, :ho].float() * own_h, x["t_garrison_count"][:, :to].float() * own_t, x["t_available"][:, :to].clamp(min=0).float() * own_t], dim=1)
	if creature.shape[-1] != count.shape[-1]:
		raise ValueError("army slots and levels differ in width")
	pos = torch.cat([x["h_pos"][:, :ho], x["t_pos"][:, :to], x["t_pos"][:, :to]], dim=1)  # [B, S, 3]
	near = _nearness(x["t_pos"][:, :to], pos)  # [B, TO, S]
	up, base = t.dwelling_creature[faction][..., list(DWELLINGS_UP)], t.dwelling_base[faction][..., list(DWELLINGS_UP)]  # [B, TO, 7]
	delta = (t.creature_cost[vocab_index(up, c.CREATURE_VOCAB)] - t.creature_cost[vocab_index(base, c.CREATURE_VOCAB)]).clamp(min=0) * ((up >= 0) & (base >= 0))
	match = (creature[:, None, None, :, :] == base[..., None, None]) & (base >= 0)[..., None, None]  # [B, TO, 7, S, slots]
	held = (match.float() * count[:, None, None]).sum(dim=-1)  # [B, TO, 7, S]
	h = (held * near[:, :, None, :]).sum(dim=-1) * delta
	return h / (h + 2000.0)


def _feature_upgrade_elsewhere(x, t, faction, built):
	"""[B, TO, 7] 0..1: the nearness of our nearest other town with the same upgraded dwelling built; 0 none."""
	to = c.OWN_TOWNS
	up = t.dwelling_creature[faction][..., list(DWELLINGS_UP)]  # [B, TO, 7]
	have = built[..., list(DWELLINGS_UP)] & own_towns(x).unsqueeze(-1)  # [B, TO, 7]
	same = (up[:, :, None, :] == up[:, None, :, :]) & have[:, None, :, :] & (up[:, :, None, :] >= 0)  # [B, TO (here), TO (there), 7]
	other = ~torch.eye(to, dtype=torch.bool, device=up.device)[None, :, :, None]
	near = _nearness(x["t_pos"][:, :to], x["t_pos"][:, :to]).unsqueeze(-1)
	return torch.where(same & other, near.expand_as(same.float()), torch.zeros((), device=up.device)).amax(dim=2)


def build_features(x, t, economy):
	"""(entry [B, TO, N_BUILDINGS, len(BUILD_ENTRY)] each building's bounded views in each town of ours, value [B, TO,
	N_BUILDINGS, 3] the fixed prior's gold parts: 3.5 days of the income it adds x (1 + pool_unbought), 1.5 x the week's
	growth it adds x payable, and the unlock share). 0 for towns not ours."""
	gold, income, pool, week, _, own, faction = _economy_inputs(x, t)
	built = x["t_built"][:, : c.OWN_TOWNS].bool()
	cost = t.building_cost[faction]
	gold_cost = t.building_resources[faction][..., 6]
	income_added = t.building_income[faction].clamp(min=0)
	production = _production_added(x, t, faction, built)
	unbought = economy[:, ECONOMY.index("pool_unbought")]
	payable = _feature_payable(production, gold, income, cost, pool, week)
	income_value = 3.5 * income_added * (1.0 + unbought.reshape(-1, 1, 1))
	production_value = 1.5 * production * payable
	unlock = _unlock_value(t, faction, built, income_value + production_value)
	worth = (x["g_resources"].clamp(min=0).float() * t.resource_price).sum(dim=-1)
	ups = torch.zeros_like(cost).unsqueeze(-1).expand(*cost.shape, 3).clone()
	ups[..., list(DWELLINGS_UP), 0] = _feature_upgrade_quality(t, faction)
	ups[..., list(DWELLINGS_UP), 1] = _feature_upgrade_held(x, t, faction)
	ups[..., list(DWELLINGS_UP), 2] = _feature_upgrade_elsewhere(x, t, faction, built)
	end_of_week = (1.0 - economy[:, ECONOMY.index("days_to_growth")]).reshape(-1, 1, 1) * (production > 0)
	entry = torch.stack(
		[
			_feature_income_add(income_added), _feature_income_rel(income_added, income), _feature_production_add(production),
			_feature_production_rel(production, week), payable, _feature_unlocks(unlock), _feature_cost_share(cost, gold, worth),
			_feature_crowds_out(gold_cost, gold, pool, unbought), ups[..., 0], ups[..., 1], ups[..., 2], end_of_week,
		],
		dim=-1,
	) * own[..., None, None] * t.economy_on()
	value = torch.stack([income_value, production_value, unlock], dim=-1) * own[..., None, None]
	return entry, value
