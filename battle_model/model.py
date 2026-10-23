"""The battle model: the contract's raw inputs to its outputs (the log-probability of every action, the value heads),
built on features.py: the game's own rules, computed exactly, as bounded and scale-free features.

	forward(x, facts=None)
	  facts        = facts(x)                            features.py, no parameters (training computes them once):
	                   distance_maps                     per unit, the steps to every hex; to each enemy, from every hex
	                   attack_features                   what the acting unit's melee attack and shot would do to each unit
	                   threat_maps                       per hex: threat, danger, blocked shooters, new targets, progress
	                   stack_features, unit_summaries, battle_features
	  tokens       = embed(...)                          the battle, the heroes, the units, the hexes
	  context      = trunk(tokens)                       attention over all of them
	  logits       = score_actions(context, tokens, action_features(...))
	  return log-probabilities (masked) and the values

How a choice is scored (ActionHeads): its logit is the sum of
  - a learned pointer score: the context's query against the choice's token (unit or hex), as in model_v1;
  - its rule score (RuleScore): first order, one weight per feature of this choice (features.*_FEATURES), starting at
    hand-chosen values (RULE_START), so a handful of parameters learns how each meaningful number of an action moves
    that action's own logit; and second order, a low-rank term (rank 4) of the choice's features against a summary of
    all legal choices and the context, for what the first order cannot express;
  - the value of its best follow-up: the largest logit among the legal choices it leads to (a type gets its best
    target's or hex's, a melee target its best hex's, a spell its best target's). So an attack the rules like lifts its
    type, not just itself, and many poor options (most hexes are threatened) do not sink a type with one good one, as
    an average would (the first version averaged, log-mean-exp, and the untrained model stopped moving). With all
    follow-ups equal it adds nothing, so the untrained types keep ACTION_TYPE_PRIOR's odds.

Untrained, the pointer scores are 0 (zero queries), so the model plays its rule scores: kill what dies for sure, prefer
valuable targets and threatening ones, avoid retaliation and threatened hexes, block shooters, move toward the enemy.
The type head starts at ACTION_TYPE_PRIOR. The value heads are as in model_v1: V0..V2 the reward formula at the end of
this and the next two rounds (boosted), V the final reward (standing alone), each tanh, trained with cross-entropy.

Embeddings (creatures, effects, positions, ...) start small (std 0.1) next to the features, so the numbers count from
the first update. Hidden information and a battle memory: later (docs/hidden_information.md).
"""
import math

import torch
from torch import nn

from . import features as F
from . import rules
from .constants import (
	ACCESSIBILITY_VOCAB,
	BATTLEFIELD_VOCAB,
	CREATURE_VOCAB,
	EFFECT_VOCAB,
	GATE_STATE_VOCAB,
	HERO_TYPE_VOCAB,
	MASKED_LOGIT,
	N_ACTION_TYPES,
	N_HEXES,
	N_SECONDARY_SKILLS,
	N_SIDES,
	N_SPELLS,
	N_VALUE_HORIZONS,
	OBSTACLE_ID_VOCAB,
	OBSTACLE_TYPE_VOCAB,
	SECOND_TARGET_KIND,
	WALL_STATE_VOCAB,
	ActionType,
	TargetKind,
)
from .layers import ACTION_TYPE_PRIOR, Block, masked_log_softmax, vocab_index

# The rule weights' starting values, per head and feature (nats per unit of the feature); the rest start at 0.
# Hand-chosen: the untrained model's play, and the defaults training moves from.
RULE_START = {
	"melee_target": {"sure_kill": 2.0, "value_destroyed": 4.0, "threat_removed": 2.0, "value_lost": -4.0},
	"shoot_target": {"sure_kill": 2.0, "value_destroyed": 4.0, "threat_removed": 2.0},
	"melee_from": {"threat_change": -1.0, "danger_change": -1.0, "blocked_change": 1.0},
	# An attacker has to walk into threat to win: getting closer outweighs becoming more threatened
	"move": {"threat_change": -1.0, "danger_change": -1.0, "blocked_change": 1.0, "new_targets": 1.0, "progress": 2.0},
}
EMBEDDING_STD = 0.1


def small_embedding(count, dim):
	embedding = nn.Embedding(count, dim)
	nn.init.normal_(embedding.weight, std=EMBEDDING_STD)
	return embedding


def best_of(logits, legal):
	"""The largest legal entry of the last axis; 0 where none is legal."""
	return torch.where(legal.any(dim=-1), logits.masked_fill(~legal, MASKED_LOGIT).amax(dim=-1), torch.zeros_like(logits[..., 0]))


class RuleScore(nn.Module):
	"""A choice's score from its own features [B, *choices, features]: first order, one weight per feature (starting
	at `start`), plus a second-order term of rank `rank`: the choice's features (projected) against a summary of all
	legal choices' and of the context. The second order starts at 0."""

	def __init__(self, names, start, dim, rank):
		super().__init__()
		self.names = tuple(names)
		self.weight = nn.Parameter(torch.tensor([float(start.get(n, 0.0)) for n in self.names]))
		self.project = nn.Linear(len(self.names), rank, bias=False)
		self.summary = nn.Linear(2 * rank + dim, rank)
		nn.init.zeros_(self.summary.weight)
		nn.init.zeros_(self.summary.bias)

	def forward(self, features, legal, context):
		first = (features * self.weight).sum(dim=-1)
		z = self.project(features)  # [B, *choices, rank]
		flat, flat_legal = z.flatten(1, -2), legal.flatten(1)[..., None].float()
		mean = (flat * flat_legal).sum(dim=1) / flat_legal.sum(dim=1).clamp(min=1)
		top = flat.masked_fill(flat_legal == 0, -1e4).amax(dim=1).clamp(min=-1e3) * (flat_legal.sum(dim=1) > 0)
		g = self.summary(torch.cat([mean, top, context], dim=-1))  # [B, rank]
		second = (z * g.view(g.shape[0], *([1] * (z.dim() - 2)), -1)).sum(dim=-1)
		return first + second


class BattleEmbedding(nn.Module):
	"""The global token: the battlefield, the gate, and the battle's features (features.battle_features)."""

	def __init__(self, dim):
		super().__init__()
		self.battlefield = small_embedding(BATTLEFIELD_VOCAB + 1, dim)
		self.gate = small_embedding(GATE_STATE_VOCAB + 1, dim)
		self.numbers = nn.Linear(len(F.BATTLE_FEATURES), dim)

	def forward(self, b, facts):
		x = b.x
		return self.battlefield(vocab_index(x["g_battlefield"], BATTLEFIELD_VOCAB)) + self.gate(vocab_index(x["g_gate_state"], GATE_STATE_VOCAB)) + self.numbers(facts["battle"])


class HeroEmbedding(nn.Module):
	"""One token per side's hero: its type, whether it is ours, primary skills (x / (x + 10)), mana (left of its
	pool, and the pool against 50), secondary skill levels (of 3) and spellbook (both hidden for the enemy's hero)."""

	def __init__(self, dim):
		super().__init__()
		self.hero_type = small_embedding(HERO_TYPE_VOCAB + 1, dim)
		self.ours = small_embedding(2, dim)
		self.numbers = nn.Linear(7 + N_SECONDARY_SKILLS + N_SPELLS, dim)

	def forward(self, b):
		x = b.x
		skill = lambda name: F.relative(x[name].float(), torch.full_like(x[name], 10, dtype=torch.float32))
		mana, pool = x["h_mana"].float(), x["h_mana_max"].float()
		numbers = torch.cat([
			torch.stack([(x["h_present"] != 0).float(), skill("h_attack"), skill("h_defense"), skill("h_power"), skill("h_knowledge"), F.fraction(mana, pool), F.relative(pool, torch.full_like(pool, 50))], dim=-1),
			x["h_skills"].float().clamp(0, 3) / 3,
			(x["h_spells"] != 0).float(),
		], dim=-1)
		sides = torch.arange(N_SIDES, device=mana.device)
		ours = (sides[None, :] == x["g_our_side"][:, None]).long()
		return self.hero_type(vocab_index(x["h_type"], HERO_TYPE_VOCAB)) + self.ours(ours) + self.numbers(numbers), x["h_present"] != 0


class UnitEmbedding(nn.Module):
	"""One token per unit slot: the creature, its spell effects (with their turns left and value, bounded), where it
	stands, and its numbers: stack_features (itself) and unit_summaries (against the others)."""

	def __init__(self, dim):
		super().__init__()
		self.creature = small_embedding(CREATURE_VOCAB + 1, dim)
		self.effect = small_embedding(EFFECT_VOCAB + 1, dim)
		self.effect_numbers = nn.Linear(2, dim)
		self.position = small_embedding(N_HEXES, dim)
		self.numbers = nn.Linear(len(F.STACK_FEATURES) + len(F.SUMMARY_FEATURES), dim)

	def forward(self, b, facts):
		x = b.x
		has_effect = (x["u_effect_id"] >= 0).float()[..., None]
		effect_numbers = torch.stack([x["u_effect_turns"].float().clamp(0, 10) / 10, x["u_effect_value"].float() / (x["u_effect_value"].float().abs() + 10)], dim=-1)
		effects = ((self.effect(vocab_index(x["u_effect_id"], EFFECT_VOCAB)) + self.effect_numbers(effect_numbers)) * has_effect).sum(dim=2)
		rear = b.units["rear"]
		where = self.position(b.units["hex"].clamp(0, N_HEXES - 1)) + self.position(rear.clamp(0, N_HEXES - 1)) * (rear >= 0).float()[..., None]
		numbers = torch.cat([facts["stack"], facts["summaries"]], dim=-1)
		return self.creature(vocab_index(x["u_creature"], CREATURE_VOCAB)) + effects + where + self.numbers(numbers), x["u_present"] != 0


class HexEmbedding(nn.Module):
	"""One token per hex: its position, accessibility, obstacle and wall, the side and turns of a spell obstacle, and
	the acting unit's maps there (features.threat_maps)."""

	def __init__(self, dim):
		super().__init__()
		self.position = small_embedding(N_HEXES, dim)
		self.access = small_embedding(ACCESSIBILITY_VOCAB + 1, dim)
		self.obstacle_type = small_embedding(OBSTACLE_TYPE_VOCAB + 1, dim)
		self.obstacle_id = small_embedding(OBSTACLE_ID_VOCAB + 1, dim)
		self.wall = small_embedding(WALL_STATE_VOCAB + 1, dim)
		self.numbers = nn.Linear(4 + len(F.HEX_FEATURES), dim)

	def forward(self, b, facts):
		x = b.x
		side = x["x_obstacle_side"]
		our = x["g_our_side"][:, None]
		obstacle = torch.stack([(side == our).float(), ((side >= 0) & (side != our)).float(), x["x_obstacle_turns"].float().clamp(0, 10) / 10, (x["x_obstacle_type"] >= 0).float()], dim=-1)
		return (
			self.position.weight[None, :, :]
			+ self.access(vocab_index(x["x_access"], ACCESSIBILITY_VOCAB))
			+ self.obstacle_type(vocab_index(x["x_obstacle_type"], OBSTACLE_TYPE_VOCAB))
			+ self.obstacle_id(vocab_index(x["x_obstacle_id"], OBSTACLE_ID_VOCAB))
			+ self.wall(vocab_index(x["x_wall_state"], WALL_STATE_VOCAB))
			+ self.numbers(torch.cat([obstacle, facts["maps"]], dim=-1))
		)


class Trunk(nn.Module):
	"""Attention over all tokens (global, heroes, units, hexes; a learned kind embedding each), then the context: the
	global token, the acting unit's and our hero's, through a small MLP."""

	def __init__(self, dim, layers, heads):
		super().__init__()
		self.kind = small_embedding(4, dim)  # global, hero, unit, hex
		self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
		self.norm = nn.LayerNorm(dim)
		self.context = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, dim))

	def forward(self, b, glob, heroes, units, hexes, hero_valid, unit_valid):
		k = self.kind.weight
		tokens = torch.cat([(glob + k[0])[:, None, :], heroes + k[1], units + k[2], hexes + k[3]], dim=1)
		always = torch.ones_like(unit_valid[:, :1])
		valid = torch.cat([always, hero_valid, unit_valid, always.expand(-1, N_HEXES)], dim=1)
		for block in self.blocks:
			tokens = block(tokens, valid)
		tokens = self.norm(tokens)
		glob, heroes = tokens[:, 0], tokens[:, 1 : 1 + N_SIDES]
		units, hexes = tokens[:, 1 + N_SIDES : -N_HEXES], tokens[:, -N_HEXES:]
		batch = torch.arange(glob.shape[0], device=glob.device)
		context = self.context(torch.cat([glob, units[batch, b.acting], heroes[batch, b.x["g_our_side"].clamp(0, N_SIDES - 1)]], dim=-1))
		return context, units, hexes


class ActionHeads(nn.Module):
	"""Every choice's logit: pointer score + rule score + the value of its follow-ups (see the top of this file)."""

	POINTERS = ("move", "melee", "melee_from", "shoot", "spell", "spell2")
	KEYS = ("move_hex", "melee_unit", "melee_from_unit", "melee_from_hex", "shoot_unit", "spell_unit", "spell_hex", "spell2_from_unit", "spell2_hex", "spell2_unit")

	def __init__(self, dim, rank):
		super().__init__()
		self.type_head = nn.Linear(dim, N_ACTION_TYPES)
		self.type_rules = nn.Linear(len(F.TYPE_FEATURES), N_ACTION_TYPES, bias=False)
		self.spell_head = nn.Linear(dim, N_SPELLS)
		self.spell_query = small_embedding(N_SPELLS, dim)
		self.queries = nn.ModuleDict({name: nn.Linear(dim, dim) for name in self.POINTERS})
		self.keys = nn.ModuleDict({name: nn.Linear(dim, dim) for name in self.KEYS})
		names = {"melee_target": F.ATTACK_FEATURES, "melee_from": F.FROM_FEATURES, "shoot_target": F.SHOT_FEATURES, "move": F.MOVE_FEATURES}
		self.rules = nn.ModuleDict({head: RuleScore(names[head], RULE_START.get(head, {}), dim, rank) for head in names})
		second = torch.zeros(N_SPELLS, dtype=torch.long)
		for spell, kind in SECOND_TARGET_KIND.items():
			second[spell] = int(kind)
		self.register_buffer("second_target", second, persistent=False)
		self.register_buffer("type_one_hot", torch.eye(N_ACTION_TYPES), persistent=False)
		with torch.no_grad():  # untrained: the type prior, no type rules, zero pointer scores
			self.type_head.weight.zero_()
			self.type_head.bias.copy_(torch.tensor([ACTION_TYPE_PRIOR.get(a, 0.0) for a in ActionType]))
			self.type_rules.weight.zero_()
			# every pointer score starts at 0: the queries, and the per-unit offsets added to them, start at 0
			for layer in (*self.queries.values(), self.keys["melee_from_unit"], self.keys["spell2_from_unit"], self.spell_head):
				layer.weight.zero_()
				layer.bias.zero_()
			self.spell_query.weight.zero_()

	def forward(self, context, units, hexes, actions, x):
		scale = 1.0 / math.sqrt(context.shape[-1])
		q = {name: layer(context) for name, layer in self.queries.items()}
		k = self.keys
		point = lambda query, items, key: torch.einsum("bd,bnd->bn", query, k[key](items)) * scale
		rule = lambda head, legal: self.rules[head](actions[head], legal, context)

		# Leaves first: a choice's logit includes the value of the best choice it leads to
		melee_from = torch.einsum("bud,bhd->buh", k["melee_from_unit"](units) + q["melee_from"][:, None, :], k["melee_from_hex"](hexes)) * scale + rule("melee_from", x["m_melee_from"])
		melee_target = point(q["melee"], units, "melee_unit") + rule("melee_target", x["m_melee_target"]) + best_of(melee_from, x["m_melee_from"])
		shoot_target = point(q["shoot"], units, "shoot_unit") + rule("shoot_target", x["m_shoot_target"])
		move = point(q["move"], hexes, "move_hex") + rule("move", x["m_move_hex"])

		spell2_from = k["spell2_from_unit"](units) + q["spell2"][:, None, :]
		spell2_hex = torch.einsum("bud,bhd->buh", spell2_from, k["spell2_hex"](hexes)) * scale
		spell2_unit = torch.einsum("bud,bvd->buv", spell2_from, k["spell2_unit"](units)) * scale
		spell_q = self.spell_query.weight[None, :, :] + q["spell"][:, None, :]
		second_value = torch.stack([torch.zeros_like(spell2_hex[..., 0]), best_of(spell2_unit, x["m_spell2_unit"]), best_of(spell2_hex, x["m_spell2_hex"])], dim=1)  # [B, kind, U]
		spell_unit = torch.einsum("bsd,bud->bsu", spell_q, k["spell_unit"](units)) * scale + second_value[:, self.second_target, :]
		spell_hex = torch.einsum("bsd,bhd->bsh", spell_q, k["spell_hex"](hexes)) * scale
		first_targets = torch.cat([spell_unit, spell_hex], dim=-1)
		spell = self.spell_head(context) + best_of(first_targets, torch.cat([x["m_spell_unit"], x["m_spell_hex"]], dim=-1))

		follow = torch.zeros_like(self.type_head(context))
		for action, logits, legal in ((ActionType.MOVE, move, x["m_move_hex"]), (ActionType.MELEE, melee_target, x["m_melee_target"]), (ActionType.SHOOT, shoot_target, x["m_shoot_target"]), (ActionType.HERO_SPELL, spell, x["m_spell"])):
			follow = follow + best_of(logits, legal)[:, None] * self.type_one_hot[action]
		kind = self.type_head(context) + self.type_rules(actions["type"]) + follow
		return {
			"logp_type": masked_log_softmax(kind, x["m_type"]),
			"logp_move_hex": masked_log_softmax(move, x["m_move_hex"]),
			"logp_melee_target": masked_log_softmax(melee_target, x["m_melee_target"]),
			"logp_melee_from": masked_log_softmax(melee_from, x["m_melee_from"]),
			"logp_shoot_target": masked_log_softmax(shoot_target, x["m_shoot_target"]),
			"logp_spell": masked_log_softmax(spell, x["m_spell"]),
			"logp_spell_unit": masked_log_softmax(spell_unit, x["m_spell_unit"]),
			"logp_spell_hex": masked_log_softmax(spell_hex, x["m_spell_hex"]),
			"logp_spell2_hex": masked_log_softmax(spell2_hex, x["m_spell2_hex"]),
			"logp_spell2_unit": masked_log_softmax(spell2_unit, x["m_spell2_unit"]),
		}


class ValueHeads(nn.Module):
	"""V0..V2: the reward formula at the end of this round and the next two, boosted (each adds to the previous logit,
	detached); V: the final reward, standing alone. Each tanh(logit), -1..1 (reward.to_scale); all start at 0."""

	def __init__(self, dim, boost_stop_gradient=True):
		super().__init__()
		self.boost_stop_gradient = boost_stop_gradient
		self.heads = nn.ModuleList(nn.Linear(dim, 1) for _ in range(N_VALUE_HORIZONS + 1))
		with torch.no_grad():
			for head in self.heads:
				head.weight.zero_()
				head.bias.zero_()

	def forward(self, context):
		logits, previous = [], None
		for i, head in enumerate(self.heads):
			logit = head(context).squeeze(-1)
			if previous is not None and i < len(self.heads) - 1:
				logit = logit + (previous.detach() if self.boost_stop_gradient else previous)
			logits.append(logit)
			previous = logit
		value_logit = torch.stack(logits, dim=-1)
		return {"value": torch.tanh(value_logit), "value_logit": value_logit}  # value_logit: for the loss only


class BattleModel(nn.Module):
	def __init__(self, dim=64, layers=2, heads=4, rank=4, boost_stop_gradient=True):
		super().__init__()
		for name, table in rules.creature_tables().items():
			self.register_buffer(f"table_{name}", table, persistent=False)
		self.battle = BattleEmbedding(dim)
		self.heroes = HeroEmbedding(dim)
		self.units = UnitEmbedding(dim)
		self.hexes = HexEmbedding(dim)
		self.trunk = Trunk(dim, layers, heads)
		self.actions = ActionHeads(dim, rank)
		self.values = ValueHeads(dim, boost_stop_gradient)

	def tables(self):
		return {name[len("table_"):]: buffer for name, buffer in self.named_buffers() if name.startswith("table_")}

	def facts(self, x):
		"""Everything features.py computes from the inputs: no parameters, no gradients. Training computes them once per
		decision and epoch and passes them to forward; the exported model computes them itself."""
		with torch.no_grad():
			b = F.read_battle(x, self.tables())
			reach, to_enemy = F.distance_maps(b)
			maps, can_attack = F.threat_maps(b, reach, to_enemy)
			return {
				"melee": F.attack_features(b, shooting=False),
				"shot": F.attack_features(b, shooting=True),
				"maps": maps,
				"can_attack": can_attack,
				"stack": F.stack_features(b),
				"summaries": F.unit_summaries(b, can_attack),
				"battle": F.battle_features(b, maps),
			}

	def forward(self, x, facts=None):
		facts = self.facts(x) if facts is None else facts
		b = F.read_battle(x, self.tables())
		heroes, hero_valid = self.heroes(b)
		units, unit_valid = self.units(b, facts)
		context, units, hexes = self.trunk(b, self.battle(b, facts), heroes, units, self.hexes(b, facts), hero_valid, unit_valid)
		actions = F.action_features(b, facts["melee"], facts["shot"], facts["maps"], facts["can_attack"], facts["battle"])
		return {**self.actions(context, units, hexes, actions, x), **self.values(context)}

	def set_rule_weights(self, head, **weights):
		"""Set first-order rule weights by feature name, e.g. set_rule_weights("melee_target", sure_kill=1000.0)."""
		score = self.actions.rules[head]
		with torch.no_grad():
			for name, value in weights.items():
				score.weight[score.names.index(name)] = value


def build_model(kwargs):
	"""The model a checkpoint's model_kwargs describe: this one, or model_v1's for checkpoints from before (no
	"architecture" key)."""
	kwargs = dict(kwargs)
	if kwargs.pop("architecture", "v1") == "v2":
		return BattleModel(**kwargs)
	from .model_v1 import BattleModelV1

	return BattleModelV1(**{"chain_final_value": True, "kill_rule": False, **kwargs})
