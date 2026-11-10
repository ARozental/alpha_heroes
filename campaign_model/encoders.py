"""The campaign model v2's input encoders (docs/campaign_model_v2_draft.md, "encode"): every raw input turned into
bounded numbers the same way by its kind, and the entity encoders built from them. Raw inputs only: no domain
knowledge here (it joins after, model_v2.join), and no ownership (model_v2.take_sides transforms by it).

	amount      a count or quantity in a unit of UNITS (scale s): log1p(x/s), x/(x+s/4), x/(x+s), x/(x+4s), x > 0
	signed      a change in a unit: symlog(x/s), tanh(x/s)
	category    an id: an embedding (-1 or past the vocabulary: the "none" row)
	level       a small integer: one-hot over its range
	bits        flags: 0/1 each
A token is its fields' encodings through a gated MLP, then RMSNorm (Tokenizer); a set (a hero's army, its artifacts,
skills, spells) is its elements' embeddings, summed.
"""
import functools
from typing import NamedTuple

import torch
from torch import nn

from battle_model.layers import small_embedding, vocab_index
from battle_model.model import RMSNorm, SwiGLU

from . import constants as c

# Each unit's scale (s above): about a typical amount of it, set by hand. The four saturating scales around it cover
# three orders of magnitude, so a scale only has to be roughly right
UNITS = {
	"gold": 1000.0,
	"resource": 10.0,  # wood, ore, mercury, sulfur, crystal, gems
	"creatures": 20.0,
	"movement": 1500.0,  # a hero's day of walking
	"mana": 50.0,
	"experience": 5000.0,
	"days": 3.0,
	"tiles": 10.0,
	"points": 5.0,  # primary skill points
	"levels": 5.0,  # hero levels
	"decisions": 16.0,
	"few": 2.0,  # small counts: markets, buildings built today
}
AMOUNT_FEATURES = 5
QUANTITY_IDS = 10  # VCMI's quantity ids (few, several, pack, ...): features.QUANTITY_COUNTS


def amount(x, unit):
	"""[...] amounts in a unit -> [..., AMOUNT_FEATURES], each 0..1 but the log (0..~10)."""
	s = UNITS[unit]
	x = x.float().clamp(min=0)
	return torch.stack([torch.log1p(x / s), x / (x + s / 4), x / (x + s), x / (x + 4 * s), (x > 0).float()], dim=-1)


def amounts(x, unit):
	"""[..., n] -> [..., n * AMOUNT_FEATURES]."""
	return amount(x, unit).flatten(-2)


def signed(x, unit):
	"""[...] changes in a unit -> [..., 2]: symlog(x / s), tanh(x / s)."""
	y = x.float() / UNITS[unit]
	return torch.stack([torch.sign(y) * torch.log1p(y.abs()), torch.tanh(y)], dim=-1)


def level(x, low, high):
	"""[...] integers -> one-hot [..., high - low + 1], clamped to low..high."""
	return (x.long().clamp(low, high).unsqueeze(-1) == torch.arange(low, high + 1, device=x.device)).float()


def bits(x, n):
	"""[...] flag words -> [..., n] of their first n bits."""
	return torch.stack([torch.remainder(torch.div(x, 1 << i, rounding_mode="floor"), 2) for i in range(n)], dim=-1).float()


class Tokenizer(nn.Module):
	"""[the embeddings summed, the numbers] -> a token: a gated MLP, then RMSNorm."""

	def __init__(self, numbers, dim, embedded=True):
		super().__init__()
		self.mlp = SwiGLU((dim if embedded else 0) + numbers, dim, dim)
		self.norm = RMSNorm(dim)

	def forward(self, embedded, numbers):
		return self.norm(self.mlp(numbers if embedded is None else torch.cat([embedded, numbers], dim=-1)))


def counts_seen(counts, exact):
	"""A count as seen: (its amount features where exact, else 0; its quantity id where not exact, else -1)."""
	return amount(torch.where(exact, counts, torch.zeros_like(counts)), "creatures"), torch.where(exact, torch.full_like(counts, -1), counts.clamp(0, QUANTITY_IDS - 1))


class ArmyEncoder(nn.Module):
	"""Stacks (creature, count as seen) -> per-stack vectors [..., A, d] and the army's vector [..., d] (its stacks' sum
	and max): the same function for every army, a hero's, a garrison, a monster's, a bank's guards, one for hire."""

	def __init__(self, dim):
		super().__init__()
		self.creature = small_embedding(c.CREATURE_VOCAB + 1, dim)
		self.quantity = small_embedding(QUANTITY_IDS + 1, dim)
		self.stack = Tokenizer(AMOUNT_FEATURES, dim)
		self.pool = Tokenizer(2 * dim, dim, embedded=False)

	def forward(self, creatures, counts, exact):
		"""creatures, counts [..., A]; exact [...] or [..., A] (counts exact, not quantity ids)."""
		present = creatures >= 0
		exact = exact if exact.shape == counts.shape else exact.unsqueeze(-1).expand_as(counts)
		numbers, quantity = counts_seen(counts, exact)
		embedded = self.creature(vocab_index(creatures, c.CREATURE_VOCAB)) + self.quantity(vocab_index(quantity, QUANTITY_IDS))
		stacks = self.stack(embedded, numbers) * present.unsqueeze(-1)
		top = stacks.masked_fill(~present.unsqueeze(-1), -1e4).amax(dim=-2) * present.any(dim=-1, keepdim=True)
		return stacks, self.pool(None, torch.cat([stacks.sum(dim=-2), top], dim=-1))


class HeroEncoder(nn.Module):
	"""A hero -> its vector, the same function for ours, the others' and the tavern's (fields absent: empty)."""

	NUMBERS = AMOUNT_FEATURES * 11 + 7 * 4 + 5  # 11 amounts (level, experience, 4 primary, 3 of mana, 2 of movement), 4 luck and morale levels, 5 flags

	def __init__(self, dim):
		super().__init__()
		self.hero_type = small_embedding(c.HERO_TYPE_VOCAB + 1, dim)
		self.klass = small_embedding(c.HERO_CLASS_VOCAB + 1, dim)
		self.artifact = small_embedding(c.ARTIFACT_VOCAB + 1, dim)
		# What the artifacts carried in the backpack (not worn: no bonus) add apart from their rows, from the sum of those
		# rows, at 0: worn and carried read alike before (2026-11-05, model_cannot_do 2b)
		self.carried = nn.Linear(dim, dim, bias=False)
		nn.init.zeros_(self.carried.weight)
		self.skill = small_embedding(c.N_SECONDARY_SKILLS * 4, dim)  # (skill, level 0..3): level 0's rows unused
		self.spells = nn.Linear(c.N_SPELLS, dim, bias=False)
		self.army = ArmyEncoder(dim)
		self.token = Tokenizer(self.NUMBERS, dim)

	def forward(self, f):
		"""f: the hero's fields (hero_fields / tavern_fields) -> (vectors [..., d], army stacks [..., A, d])."""
		stacks, army = self.army(f["army_creature"], f["army_count"], f["exact"])
		artifacts = f["artifacts"]  # (the worn slots, then the backpack)
		rows = self.artifact(vocab_index(artifacts, c.ARTIFACT_VOCAB)) * (artifacts >= 0).unsqueeze(-1)
		skills = f["skills"].clamp(0, 3)
		skill_rows = torch.arange(c.N_SECONDARY_SKILLS, device=skills.device) * 4 + skills
		embedded = (
			self.hero_type(vocab_index(f["type"], c.HERO_TYPE_VOCAB)) + self.klass(vocab_index(f["class"], c.HERO_CLASS_VOCAB)) + army
			+ rows.sum(dim=-2) + self.carried(rows[..., c.N_WORN_SLOTS :, :].sum(dim=-2))
			+ (self.skill(skill_rows) * (skills > 0).unsqueeze(-1)).sum(dim=-2) + self.spells(f["spells"].float())
		)
		numbers = torch.cat(
			[
				amount(f["level"], "levels"), amount(f["exp"], "experience"), amounts(f["primary"], "points"),
				amount(f["mana"], "mana"), amount(f["mana_max"], "mana"), amount(f["spell_cost"], "mana"),
				amount(f["move"], "movement"), amount(f["move_max"], "movement"),
				level(f["luck"], -3, 3), level(f["morale"], -3, 3), level(f["battle_luck"], -3, 3), level(f["battle_morale"], -3, 3),
				torch.stack([f["boat"], f["garrisoned"], f["in_town"], f["exact"], f["present"]], dim=-1).float(),
			],
			dim=-1,
		)
		return self.token(embedded, numbers) * f["present"].unsqueeze(-1), stacks


def hero_fields(x):
	"""The hero slots' fields [B, H, ...] for HeroEncoder (the battle-only ones ours alone: others' 0)."""
	b, h = x["h_present"].shape
	pad = lambda t: torch.cat([t, torch.zeros((b, h - t.shape[1]) + tuple(t.shape[2:]), dtype=t.dtype, device=t.device)], dim=1)  # noqa: E731
	return {
		"type": x["h_type"], "class": x["h_class"], "level": x["h_level"], "exp": x["h_exp"], "primary": x["h_primary"],
		"mana": x["h_mana"], "mana_max": x["h_mana_max"], "spell_cost": x["h_spell_cost"], "move": x["h_move"], "move_max": x["h_move_max"],
		"army_creature": x["h_army_creature"], "army_count": x["h_army_count"], "exact": x["h_detailed"],
		"artifacts": torch.cat([x["h_artifacts"], x["h_backpack"]], dim=-1), "skills": x["h_skills"], "spells": x["h_spells"],
		"luck": x["h_luck"], "morale": x["h_morale"], "battle_luck": pad(x["h_battle_luck"]), "battle_morale": pad(x["h_battle_morale"]),
		"boat": x["h_boat"], "garrisoned": x["h_garrisoned"], "in_town": x["h_town"] >= 0, "present": x["h_present"],
	}


def tavern_fields(x):
	"""The heroes for hire's fields [B, N_TAVERN, ...] for HeroEncoder: what the tavern window shows."""
	present = x["g_tavern"] >= 0
	zeros = torch.zeros_like(x["g_tavern"])
	worn = x["g_tavern_artifacts"]
	return {
		"type": x["g_tavern"], "class": torch.full_like(zeros, -1), "level": x["g_tavern_level"], "exp": x["g_tavern_exp"], "primary": x["g_tavern_primary"],
		"mana": zeros, "mana_max": zeros, "spell_cost": zeros, "move": zeros, "move_max": zeros,
		"army_creature": x["g_tavern_army_creature"], "army_count": x["g_tavern_army_count"], "exact": present,
		"artifacts": torch.cat([worn, torch.full(worn.shape[:-1] + (c.N_BACKPACK,), -1, dtype=worn.dtype, device=worn.device)], dim=-1),
		"skills": x["g_tavern_skills"], "spells": x["g_tavern_spells"],
		"luck": zeros, "morale": zeros, "battle_luck": zeros, "battle_morale": zeros,
		"boat": zeros.bool(), "garrisoned": zeros.bool(), "in_town": zeros.bool(), "present": present,
	}


class TownEncoder(nn.Module):
	"""A town -> its vector, its garrison's stacks [.., A, d] and its creature levels' vectors [.., N_LEVELS, d] (RECRUIT
	reads them): its faction, buildings, fort, garrison, the creatures for hire at each level and their growth."""

	def __init__(self, dim):
		super().__init__()
		self.faction = small_embedding(c.FACTION_VOCAB + 1, dim)
		self.built = nn.Linear(c.N_BUILDINGS, dim, bias=False)
		self.creature = small_embedding(c.CREATURE_VOCAB + 1, dim)
		self.level = Tokenizer(2 * AMOUNT_FEATURES + c.N_LEVELS, dim)
		self.army = ArmyEncoder(dim)
		self.token = Tokenizer(4 + AMOUNT_FEATURES + 4, dim)

	def forward(self, x):
		present = x["t_present"]
		stacks, garrison = self.army(x["t_garrison_creature"], x["t_garrison_count"], x["t_detailed"])
		creature = x["t_available_creature"]
		levels = self.level(
			self.creature(vocab_index(creature, c.CREATURE_VOCAB)),
			torch.cat([amount(x["t_available"], "creatures"), amount(x["t_growth"], "creatures"), torch.eye(c.N_LEVELS, device=creature.device).expand(creature.shape + (c.N_LEVELS,))], dim=-1),
		) * (creature >= 0).unsqueeze(-1)
		numbers = torch.cat(
			[
				level(x["t_fort"], 0, 3), amount(x["t_built_today"], "few"),
				torch.stack([x["t_visiting"] >= 0, x["t_garrison_hero"] >= 0, x["t_detailed"], present], dim=-1).float(),
			],
			dim=-1,
		)
		embedded = self.faction(vocab_index(x["t_faction"], c.FACTION_VOCAB)) + self.built(x["t_built"].float()) + garrison + levels.sum(dim=-2)
		return self.token(embedded, numbers) * present.unsqueeze(-1), stacks, levels


class TileEncoder(nn.Module):
	"""The map's tiles [B, L, S, S, ...] -> their vectors [B, L, S, S, dc]: terrain, road, flags, the object on the tile
	and its guards (a wandering monster: its creature and count as seen). Its owner is take_sides'."""

	def __init__(self, dim):
		super().__init__()
		self.terrain = small_embedding(c.TERRAIN_VOCAB + 1, dim)
		self.road = small_embedding(c.ROAD_VOCAB + 1, dim)
		self.object = small_embedding(c.OBJECT_VOCAB + 1, dim)
		self.subtype = small_embedding(c.PAIR_ROWS, dim)  # (an (object, subtype) pair's: constants.OBJECT_PAIRS)
		self.creature = small_embedding(c.CREATURE_VOCAB + 1, dim)
		self.quantity = small_embedding(QUANTITY_IDS + 1, dim)
		self.token = Tokenizer(c.N_TILE_FLAGS + AMOUNT_FEATURES, dim)
		# A teleporter's entrance no exit of which we have seen (TILE_UNSEEN_EXIT, schema 17; the user, 2026-11-10: a gate
		# whose end we see and one whose end we do not are different things): its own vector on that tile, 0 at the
		# start (model.ADDED: a checkpoint from before loads it so, the model as it was)
		self.unseen_exit = nn.Parameter(torch.zeros(dim))
		from .features import pair_table

		self.register_buffer("pairs", pair_table(), persistent=False)
		# The object ids of a class (constants.OBJECT_CLASSES): a blocked tile of one is part of it (a town's walls)
		classed = torch.zeros(c.OBJECT_VOCAB + 1, dtype=torch.bool)
		classed[[i for ids in c.OBJECT_CLASSES.values() for i in ids if i < c.OBJECT_VOCAB]] = True
		self.register_buffer("classed", classed, persistent=False)

	def forward(self, x):
		from .features import bit, object_pair

		obj, flags = x["tile_object"], x["tile_flags"]
		# A tile blocked and not visitable says so and nothing more (the user, 2026-10-31): an obstacle is no object, its
		# graphic variant (subtype) no input; the blocked tiles of an object of a class keep the object, as part of it
		blocked = bit(flags, c.TILE_BLOCKED) & ~bit(flags, c.TILE_VISITABLE)
		obj = torch.where(blocked & ~self.classed[vocab_index(obj, c.OBJECT_VOCAB)], torch.full_like(obj, -1), obj)
		subtype = torch.where(blocked, torch.full_like(obj, c.PAIR_BLOCKED), object_pair(self.pairs, obj, x["tile_subtype"]))
		exact = bit(flags, c.TILE_EXACT_COUNT)
		numbers, quantity = counts_seen(x["tile_count"], exact)
		creature = x["tile_creature"]
		embedded = (
			self.terrain(vocab_index(x["tile_terrain"], c.TERRAIN_VOCAB)) + self.road(vocab_index(x["tile_road"], c.ROAD_VOCAB))
			+ self.object(vocab_index(obj, c.OBJECT_VOCAB)) + self.subtype(subtype)
			+ (self.creature(vocab_index(creature, c.CREATURE_VOCAB)) + self.quantity(vocab_index(quantity, QUANTITY_IDS))) * (creature >= 0).unsqueeze(-1)
			+ self.unseen_exit * bit(flags, c.TILE_UNSEEN_EXIT).unsqueeze(-1)
		)
		return self.token(embedded, torch.cat([bits(flags, c.N_TILE_FLAGS), numbers * (creature >= 0).unsqueeze(-1)], dim=-1)) * bit(flags, c.TILE_ON_MAP).unsqueeze(-1)


class GlobalEncoder(nn.Module):
	"""The decision's global inputs -> one vector: the calendar, our resources and income, our other armies, the map's
	size, the players left, the question asked (its object, guards, number, sound) and our previous decision today."""

	NUMBERS = 7 + 3 * AMOUNT_FEATURES + 3 * c.N_RESOURCES * AMOUNT_FEATURES + 3 * AMOUNT_FEATURES + c.MAX_PLAYERS + 3 * AMOUNT_FEATURES + 3 + AMOUNT_FEATURES * 2

	def __init__(self, dim):
		super().__init__()
		self.kind = small_embedding(c.N_KINDS + 1, dim)
		self.last = small_embedding(c.N_ACTION_TYPES + 1, dim)
		self.object = small_embedding(c.OBJECT_VOCAB + 1, dim)
		self.subtype = small_embedding(c.PAIR_ROWS, dim)  # (the question's object, subtype: constants.OBJECT_PAIRS)
		self.creature = small_embedding(c.CREATURE_VOCAB + 1, dim)
		self.quantity = small_embedding(QUANTITY_IDS + 1, dim)
		self.sound = small_embedding(c.SOUND_VOCAB + 1, dim)
		from .features import pair_table

		self.register_buffer("pairs", pair_table(), persistent=False)
		self.owned = ArmyEncoder(dim)
		self.dwellings = ArmyEncoder(dim)
		self.token = Tokenizer(self.NUMBERS, dim)

	def forward(self, x):
		from .features import object_pair

		day = x["g_day"].clamp(min=1)
		week = torch.div(day - 1, 7, rounding_mode="floor") + 1
		resources = lambda t: torch.cat([amounts(t[:, :6], "resource"), amount(t[:, 6], "gold")], dim=-1)  # noqa: E731
		ones = torch.ones_like(x["g_owned_creature"], dtype=torch.bool)
		_, owned = self.owned(x["g_owned_creature"], x["g_owned_count"], ones)
		_, dwellings = self.dwellings(x["g_dwelling_creature"], x["g_dwelling_count"], ones)
		numbers, quantity = counts_seen(x["d_count"], x["d_detailed"])
		creature = x["d_creature"]
		embedded = (
			self.kind(vocab_index(x["g_kind"], c.N_KINDS)) + self.last(vocab_index(x["g_last_type"], c.N_ACTION_TYPES)) + owned + dwellings
			+ self.object(vocab_index(x["d_object"], c.OBJECT_VOCAB)) + self.subtype(object_pair(self.pairs, x["d_object"], x["d_subtype"]))
			+ (self.creature(vocab_index(creature, c.CREATURE_VOCAB)) + self.quantity(vocab_index(quantity, QUANTITY_IDS))) * (creature >= 0).unsqueeze(-1)
			+ self.sound(vocab_index(x["d_sound"], c.SOUND_VOCAB))
		)
		size = x["g_map"].float()
		numbers = torch.cat(
			[
				level(torch.remainder(day - 1, 7), 0, 6), amount(day, "days"), amount(week, "few"), amount(x["g_decisions_today"], "decisions"),
				resources(x["g_resources"]), resources(x["g_income"]), resources(x["g_object_income"]),
				amount(size[:, 0], "tiles"), amount(size[:, 1], "tiles"), amount(x["g_explored"], "tiles"),
				level(x["g_players"], 1, c.MAX_PLAYERS) + 0.0, amount(x["g_markets"], "few"), amount(torch.maximum(x["g_tavern_level"][:, 0], x["g_tavern_level"][:, 1]), "levels"),
				amount(x["d_number"], "gold"), level(x["g_last_changed"], -1, 1), numbers * (creature >= 0).unsqueeze(-1),
				amount((x["d_hero"] >= 0).long(), "few"),
			],
			dim=-1,
		)
		return self.token(embedded, numbers)


class OptionTables(NamedTuple):
	"""The model's own embeddings an option's identity names (OptionEncoder): the hero encoder's (skill, level) rows,
	artifact rows, spells (its layer's column a spell) and army encoder (a creature's row; a question's guards as an
	army), the global encoder's object and (object, subtype) pair rows and its pair table."""

	skill: nn.Embedding
	artifact: nn.Embedding
	spells: nn.Linear
	army: nn.Module
	object: nn.Embedding
	pair: nn.Embedding
	pairs: torch.Tensor


class OptionEncoder(nn.Module):
	"""A question's options -> (their tokens [B, O, d], their identities [B, O, d]: the sum the token is made of, its numbers
	projected in, a choice's key; the token, through its MLP, starts as its bias: every option's alike), each its own
	identity (the user, 2026-11-04: an embedding for every option, never one YES vector for every question; no hashed
	rows either: 546 of 954 options shared one): what it
	names is the model's own vector for that thing (tables), a skill's of the heroes' skill rows, a creature's of their
	armies', an artifact's, a spell's, an object's of the question's table; a component (a chest's gold or experience, a
	school's attack or defence) by its type and what it names (primary skills and resources their own rows); a YES acts
	on the object asked about (not our hero itself: a step into a monster's zone names it) and an option that fights or
	takes the question's creatures (o_battle, o_join) holds them as an army; with its kind, amount and where (from the
	hero asked)."""

	def __init__(self, dim):
		super().__init__()
		self.kind = small_embedding(len(c.OptionKind) + 1, dim)
		self.component = small_embedding(len(c.ComponentType) + 1, dim)
		self.primary = small_embedding(c.N_PRIMARY + 1, dim)
		self.resource = small_embedding(c.N_RESOURCES + 1, dim)
		self.token = Tokenizer(AMOUNT_FEATURES + 8 + 1, dim)
		self.numbers = nn.Linear(AMOUNT_FEATURES + 8 + 1, dim, bias=False)  # (no bias: one vector in every key tells none apart)

	def forward(self, x, tables):
		from .features import OBJ_HERO, object_pair, relative_position

		kind, what, sub = x["o_kind"], x["o_type"], x["o_subtype"]
		hero = x["d_hero"].clamp(min=0)
		at = torch.gather(x["h_pos"], 1, hero.reshape(-1, 1, 1).expand(-1, 1, 3)).float()  # [B, 1, 3]
		placed = (kind == c.OptionKind.OBJECT) & (x["d_hero"] >= 0).unsqueeze(-1)
		where = relative_position(x["o_pos"].float() - at) * placed.unsqueeze(-1)
		of_kind = lambda k: (kind == k).unsqueeze(-1).float()  # noqa: E731
		named = (
			tables.skill(what.clamp(0, c.N_SECONDARY_SKILLS - 1) * 4 + sub.clamp(0, 3)) * of_kind(c.OptionKind.SKILL)
			+ tables.army.creature(vocab_index(what, c.CREATURE_VOCAB)) * of_kind(c.OptionKind.AMOUNT)
			+ (tables.object(vocab_index(what, c.OBJECT_VOCAB)) + tables.pair(object_pair(tables.pairs, what, sub))) * of_kind(c.OptionKind.OBJECT)
			+ self.component_names(what, sub, tables) * of_kind(c.OptionKind.COMPONENT)
		)
		obj = x["d_object"]
		asked = (tables.object(vocab_index(obj, c.OBJECT_VOCAB)) + tables.pair(object_pair(tables.pairs, obj, x["d_subtype"]))) * ((obj >= 0) & (obj != OBJ_HERO)).unsqueeze(-1)
		present = x["d_guard_count"] > 0
		_, guards = tables.army(torch.where(present, x["d_guard_creature"], torch.full_like(x["d_guard_creature"], -1)), x["d_guard_count"], x["d_guard_detailed"])
		holds = (x["o_battle"] | x["o_join"]).unsqueeze(-1).float()
		named = named + asked.unsqueeze(1) * of_kind(c.OptionKind.YES) + (guards * present.any(dim=-1, keepdim=True)).unsqueeze(1) * holds
		embedded = self.kind(vocab_index(kind, len(c.OptionKind))) + named
		numbers = torch.cat([amount(x["o_amount"], "gold"), where, placed.float().unsqueeze(-1)], dim=-1)
		mask = x["option_mask"].unsqueeze(-1)
		return self.token(embedded, numbers) * mask, (embedded + self.numbers(numbers)) * mask

	def component_names(self, kind, sub, tables):
		"""A component's identity [..., d] (kind: VCMI's ComponentType, sub: its subtype): its type's row and what it names
		(a primary skill, a resource, a secondary skill at basic, a creature, an artifact, a spell); the rest (mana,
		experience, a level, morale, luck, ...) only by its type, their amounts in the option's numbers."""
		T = c.ComponentType
		is_ = lambda *types: functools.reduce(torch.logical_or, [kind == t for t in types]).unsqueeze(-1).float()  # noqa: E731
		spell = tables.spells.weight.t()[sub.clamp(0, c.N_SPELLS - 1)] * ((sub >= 0) & (sub < c.N_SPELLS)).unsqueeze(-1)
		return (
			self.component(vocab_index(kind, len(T)))
			+ self.primary(vocab_index(sub, c.N_PRIMARY)) * is_(T.PRIM_SKILL)
			+ self.resource(vocab_index(sub, c.N_RESOURCES)) * is_(T.RESOURCE, T.RESOURCE_PER_DAY)
			+ tables.skill(sub.clamp(0, c.N_SECONDARY_SKILLS - 1) * 4 + 1) * is_(T.SEC_SKILL)
			+ tables.army.creature(vocab_index(sub, c.CREATURE_VOCAB)) * is_(T.CREATURE)
			+ tables.artifact(vocab_index(sub, c.ARTIFACT_VOCAB)) * is_(T.ARTIFACT)
			+ spell * is_(T.SPELL_SCROLL, T.SPELL)
		)
