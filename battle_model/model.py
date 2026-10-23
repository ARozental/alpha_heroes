"""A small battle model implementing the schema contract.

The architecture is deliberately simple and replaceable; what matters is the contract:
forward(inputs) takes the dict described by schema.INPUTS and returns the dict described
by schema.OUTPUTS. All feature transforms live here, inputs are raw engine values.

Tokens: 1 global + 2 heroes + MAX_UNITS units + N_HEXES hexes, mixed by a few
self-attention blocks. Output heads score units and hexes by dot products against
query vectors built from the global token, the acting unit and our hero.

Value heads, "boosted" over horizons: V0 predicts the reward formula at the end of this round,
V1 at the end of the next round, ..., V at the end of the battle. Each is one linear layer on the
last hidden state, and each head's logit is added to the previous head's logit, so a later head
only learns the difference to the shorter-horizon estimate. Each outputs tanh(logit), a value in
-1..1: the reward rescaled by reward.to_scale (-1 = worst possible, +1 = best possible), trained
with the matching cross-entropy (losses.value_loss).

The model does not know the reward settings today. Do not design around that: a later version will
probably get the price of mana for this battle as an input (free next to a Magic Well, costly far from
one), and possibly other reward weights, so that one model can play by the map's current priorities.
"""
import math

import torch
from torch import nn

from .constants import (
	ACCESSIBILITY_VOCAB,
	BATTLEFIELD_VOCAB,
	CREATURE_VOCAB,
	GATE_STATE_VOCAB,
	HERO_TYPE_VOCAB,
	MASKED_LOGIT,
	MAX_RETALIATIONS,
	MAX_UNITS,
	N_ACTION_TYPES,
	EFFECT_VOCAB,
	N_HEXES,
	N_SECONDARY_SKILLS,
	N_SIDES,
	N_SPELLS,
	N_VALUE_HORIZONS,
	OBSTACLE_ID_VOCAB,
	OBSTACLE_TYPE_VOCAB,
	WALL_STATE_VOCAB,
	ActionType,
)
from .schema import INPUT_NAMES, OUTPUT_NAMES

# Starting preference between action types, as type-head logits. The head's weights start at zero,
# so an untrained model picks among the legal types with exactly these odds. A modelling choice:
# start aggressive and rarely give up, so the first self-play games are real fights instead of
# mostly retreats. Training moves these freely; types not listed start at 0.
# Giving up is offered at every decision, ~40 times a battle, so its odds compound: at -2 (~2.5% a
# decision each) about 2 in 3 untrained battles ended in a retreat or surrender. At -6 (~0.05% a
# decision each) it is ~4% of battles.
ACTION_TYPE_PRIOR = {
	ActionType.MELEE: 1.0,
	ActionType.SHOOT: 1.0,
	ActionType.HERO_SPELL: 0.5,
	ActionType.MOVE: 0.5,
	ActionType.WAIT: 0.0,
	ActionType.DEFEND: 0.0,
	ActionType.RETREAT: -6.0,
	ActionType.SURRENDER: -6.0,
}


def vocab_index(ids, vocab):
	"""Map raw ids to embedding rows; anything outside [0, vocab) uses the extra row `vocab`."""
	return torch.where((ids >= 0) & (ids < vocab), ids, torch.full_like(ids, vocab))


def masked_log_softmax(logits, mask):
	"""log_softmax over the legal entries only; illegal entries are -inf.

	Rows with no legal entry come out all -inf (never sampled, the parent choice is illegal).
	"""
	z = logits.masked_fill(~mask, MASKED_LOGIT)
	return torch.log_softmax(z, dim=-1).masked_fill(~mask, float("-inf"))


def log_scale(x):
	return torch.log1p(x.clamp(min=0).float())


def flag(x):
	return (x != 0).float()


class Block(nn.Module):
	"""Pre-norm self-attention block written with plain ops so it exports cleanly to ONNX."""

	def __init__(self, dim, heads):
		super().__init__()
		self.heads = heads
		self.norm1 = nn.LayerNorm(dim)
		self.qkv = nn.Linear(dim, 3 * dim)
		self.proj = nn.Linear(dim, dim)
		self.norm2 = nn.LayerNorm(dim)
		self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

	def forward(self, x, key_valid):
		b, n, d = x.shape
		q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
		q, k, v = (t.reshape(b, n, self.heads, d // self.heads).transpose(1, 2) for t in (q, k, v))
		att = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.heads)
		att = att.masked_fill(~key_valid[:, None, None, :], MASKED_LOGIT).softmax(dim=-1)
		x = x + self.proj((att @ v).transpose(1, 2).reshape(b, n, d))
		return x + self.mlp(self.norm2(x))


class BattleModel(nn.Module):
	def __init__(self, dim=64, layers=2, heads=4, boost_stop_gradient=True, mask_features=False):
		"""boost_stop_gradient: each value head adds a detached copy of the previous head's logit, so a
		longer-horizon label never pulls on a shorter-horizon head (true boosting). False lets it.

		mask_features: also feed the acting unit's legal choices into the tokens (they are model inputs
		already, used for the output masks): per hex "can move here", "can attack someone from here",
		"a spell can target this hex"; per unit "can be attacked", "can be shot", "a spell can target it".
		Otherwise the network has to work out reachability and adjacency from positions and speeds.
		"""
		super().__init__()
		self.dim = dim
		self.boost_stop_gradient = boost_stop_gradient
		self.mask_features = mask_features
		self.creature = nn.Embedding(CREATURE_VOCAB + 1, dim)
		self.effect = nn.Embedding(EFFECT_VOCAB + 1, dim)
		self.hero_type = nn.Embedding(HERO_TYPE_VOCAB + 1, dim)
		self.battlefield = nn.Embedding(BATTLEFIELD_VOCAB + 1, dim)
		self.access = nn.Embedding(ACCESSIBILITY_VOCAB + 1, dim)
		self.obstacle_type = nn.Embedding(OBSTACLE_TYPE_VOCAB + 1, dim)
		self.obstacle_id = nn.Embedding(OBSTACLE_ID_VOCAB + 1, dim)
		self.wall_state = nn.Embedding(WALL_STATE_VOCAB + 1, dim)
		self.gate_state = nn.Embedding(GATE_STATE_VOCAB + 1, dim)
		self.hex_pos = nn.Embedding(N_HEXES, dim)
		self.ours = nn.Embedding(2, dim)  # 0 = enemy, 1 = ours (heroes; units: controlled by us now)
		self.home = nn.Embedding(2, dim)  # units: 1 = on our battle side (differs from ours while hypnotized)
		self.token_kind = nn.Embedding(4, dim)  # global, hero, unit, hex
		self.unit_num = nn.Linear(23, dim)
		self.effect_num = nn.Linear(2, dim)
		self.hero_num = nn.Linear(7 + N_SECONDARY_SKILLS + N_SPELLS, dim)
		self.hex_num = nn.Linear(4, dim)
		self.global_num = nn.Linear(3, dim)
		if mask_features:
			self.unit_mask_num = nn.Linear(3, dim)
			self.hex_mask_num = nn.Linear(3, dim)
		self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
		self.norm = nn.LayerNorm(dim)

		self.context = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
		self.type_head = nn.Linear(dim, N_ACTION_TYPES)
		self.spell_head = nn.Linear(dim, N_SPELLS)
		self.spell_query = nn.Embedding(N_SPELLS, dim)
		self.value_heads = nn.ModuleList(nn.Linear(dim, 1) for _ in range(N_VALUE_HORIZONS + 1))
		# One query projection from the context, and one key projection, per pointer head
		self.queries = nn.ModuleDict({name: nn.Linear(dim, dim) for name in ("move", "melee", "melee_from", "shoot", "spell", "spell2")})
		self.keys = nn.ModuleDict({name: nn.Linear(dim, dim) for name in ("move_hex", "melee_unit", "melee_from_unit", "melee_from_hex", "shoot_unit", "spell_unit", "spell_hex", "spell2_from_unit", "spell2_hex", "spell2_unit")})

		with torch.no_grad():
			self.type_head.weight.zero_()
			self.type_head.bias.copy_(torch.tensor([ACTION_TYPE_PRIOR.get(a, 0.0) for a in ActionType]))
			# Every value head starts at 0, the middle of the reward range
			for head in self.value_heads:
				head.weight.zero_()
				head.bias.zero_()

	def _embed(self, x):
		batch = x["g_our_side"].shape[0]
		our_side = x["g_our_side"]

		# Units
		active = x["g_active_unit"].clamp(0, MAX_UNITS - 1)
		slots = torch.arange(MAX_UNITS, device=active.device)
		has_rear = x["u_rear_hex"] >= 0
		numeric = torch.stack(
			[
				log_scale(x["u_count"]),
				log_scale(x["u_first_hp"]),
				log_scale(x["u_max_hp"]),
				log_scale(x["u_dmg_min"]),
				log_scale(x["u_dmg_max"]),
				x["u_attack"].float() / 50,
				x["u_defense"].float() / 50,
				x["u_speed"].float() / 20,
				x["u_shots"].float() / 24,
				x["u_retaliations"].clamp(max=MAX_RETALIATIONS).float() / MAX_RETALIATIONS,
				x["u_casts"].float() / 5,
				x["u_morale"].float() / 3,
				x["u_luck"].float() / 3,
				flag(x["u_alive"]),
				flag(x["u_moved"]),
				flag(x["u_waited"]),
				flag(x["u_defending"]),
				flag(x["u_is_clone"]),
				flag(x["u_is_summoned"]),
				has_rear.float(),
				(slots[None, :] == active[:, None]).float(),
				(x["u_queue_now"].float() + 1) / MAX_UNITS,
				(x["u_queue_next"].float() + 1) / MAX_UNITS,
			],
			dim=-1,
		)
		effect_valid = (x["u_effect_id"] >= 0).float().unsqueeze(-1)
		effect_num = torch.stack([x["u_effect_turns"].float() / 10, torch.sign(x["u_effect_value"].float()) * log_scale(x["u_effect_value"].abs())], dim=-1)
		effects = ((self.effect(vocab_index(x["u_effect_id"], EFFECT_VOCAB)) + self.effect_num(effect_num)) * effect_valid).sum(dim=2)
		unit_ours = (x["u_owner_side"] == our_side[:, None]).long()  # controlled by us now (a hypnotized enemy is ours)
		unit_home = (x["u_side"] == our_side[:, None]).long()
		units = (
			self.creature(vocab_index(x["u_creature"], CREATURE_VOCAB))
			+ self.unit_num(numeric)
			+ effects
			+ self.ours(unit_ours)
			+ self.home(unit_home)
			+ self.hex_pos(x["u_hex"].clamp(0, N_HEXES - 1))
			+ self.hex_pos(x["u_rear_hex"].clamp(0, N_HEXES - 1)) * has_rear.float().unsqueeze(-1)
			+ self.token_kind.weight[2]
		)
		unit_valid = x["u_present"] != 0
		if self.mask_features:
			unit_masks = torch.stack([flag(x["m_melee_target"]), flag(x["m_shoot_target"]), flag(x["m_spell_unit"]).amax(dim=1)], dim=-1)
			units = units + self.unit_mask_num(unit_masks)

		# Heroes (index = battle side)
		hero_numeric = torch.cat(
			[
				torch.stack(
					[
						flag(x["h_present"]),
						x["h_attack"].float() / 50,
						x["h_defense"].float() / 50,
						x["h_power"].float() / 50,
						x["h_knowledge"].float() / 50,
						log_scale(x["h_mana"]),
						log_scale(x["h_mana_max"]),
					],
					dim=-1,
				),
				x["h_skills"].float() / 3,
				flag(x["h_spells"]),
			],
			dim=-1,
		)
		sides = torch.arange(N_SIDES, device=active.device)
		hero_ours = (sides[None, :] == our_side[:, None]).long()
		heroes = self.hero_type(vocab_index(x["h_type"], HERO_TYPE_VOCAB)) + self.hero_num(hero_numeric) + self.ours(hero_ours) + self.token_kind.weight[1]
		hero_valid = x["h_present"] != 0

		# Hexes
		obstacle_side = x["x_obstacle_side"]
		hex_numeric = torch.stack(
			[
				(obstacle_side == our_side[:, None]).float(),
				((obstacle_side >= 0) & (obstacle_side != our_side[:, None])).float(),
				x["x_obstacle_turns"].clamp(min=0).float() / 10,
				(x["x_obstacle_type"] >= 0).float(),
			],
			dim=-1,
		)
		hexes = (
			self.hex_pos.weight[None, :, :]
			+ self.access(vocab_index(x["x_access"], ACCESSIBILITY_VOCAB))
			+ self.obstacle_type(vocab_index(x["x_obstacle_type"], OBSTACLE_TYPE_VOCAB))
			+ self.obstacle_id(vocab_index(x["x_obstacle_id"], OBSTACLE_ID_VOCAB))
			+ self.wall_state(vocab_index(x["x_wall_state"], WALL_STATE_VOCAB))
			+ self.hex_num(hex_numeric)
			+ self.token_kind.weight[3]
		)
		if self.mask_features:
			hex_masks = torch.stack([flag(x["m_move_hex"]), flag(x["m_melee_from"]).amax(dim=1), flag(x["m_spell_hex"]).amax(dim=1)], dim=-1)
			hexes = hexes + self.hex_mask_num(hex_masks)

		# Global
		global_numeric = torch.stack([(x["g_round"].float() + 1) / 20, x["g_tactic_distance"].float() / 7, flag(x["g_hero_can_cast"])], dim=-1)
		glob = (
			self.battlefield(vocab_index(x["g_battlefield"], BATTLEFIELD_VOCAB))
			+ self.gate_state(vocab_index(x["g_gate_state"], GATE_STATE_VOCAB))
			+ self.global_num(global_numeric)
			+ self.token_kind.weight[0]
		)

		tokens = torch.cat([glob[:, None, :], heroes, units, hexes], dim=1)
		valid = torch.cat(
			[
				torch.ones(batch, 1, dtype=torch.bool, device=active.device),
				hero_valid,
				unit_valid,
				torch.ones(batch, N_HEXES, dtype=torch.bool, device=active.device),
			],
			dim=1,
		)
		return tokens, valid

	def forward(self, x):
		tokens, valid = self._embed(x)
		for block in self.blocks:
			tokens = block(tokens, valid)
		tokens = self.norm(tokens)
		glob = tokens[:, 0]
		heroes = tokens[:, 1 : 1 + N_SIDES]
		units = tokens[:, 1 + N_SIDES : 1 + N_SIDES + MAX_UNITS]
		hexes = tokens[:, 1 + N_SIDES + MAX_UNITS :]

		batch_index = torch.arange(glob.shape[0], device=glob.device)
		active = units[batch_index, x["g_active_unit"].clamp(0, MAX_UNITS - 1)]
		our_hero = heroes[batch_index, x["g_our_side"].clamp(0, N_SIDES - 1)]
		context = self.context(torch.cat([glob, active, our_hero], dim=-1))
		scale = 1.0 / math.sqrt(self.dim)
		q = {name: layer(context) for name, layer in self.queries.items()}
		k = self.keys

		def point(query, items, key):  # [B, D] x [B, N, D] -> [B, N]
			return torch.einsum("bd,bnd->bn", query, k[key](items)) * scale

		melee_from = torch.einsum("bud,bhd->buh", k["melee_from_unit"](units) + q["melee_from"][:, None, :], k["melee_from_hex"](hexes)) * scale
		spell_q = self.spell_query.weight[None, :, :] + q["spell"][:, None, :]
		spell_unit = torch.einsum("bsd,bud->bsu", spell_q, k["spell_unit"](units)) * scale
		spell_hex = torch.einsum("bsd,bhd->bsh", spell_q, k["spell_hex"](hexes)) * scale
		spell2_from = k["spell2_from_unit"](units) + q["spell2"][:, None, :]
		spell2_hex = torch.einsum("bud,bhd->buh", spell2_from, k["spell2_hex"](hexes)) * scale
		spell2_unit = torch.einsum("bud,bvd->buv", spell2_from, k["spell2_unit"](units)) * scale

		return {
			"logp_type": masked_log_softmax(self.type_head(context), x["m_type"]),
			"logp_move_hex": masked_log_softmax(point(q["move"], hexes, "move_hex"), x["m_move_hex"]),
			"logp_melee_target": masked_log_softmax(point(q["melee"], units, "melee_unit"), x["m_melee_target"]),
			"logp_melee_from": masked_log_softmax(melee_from, x["m_melee_from"]),
			"logp_shoot_target": masked_log_softmax(point(q["shoot"], units, "shoot_unit"), x["m_shoot_target"]),
			"logp_spell": masked_log_softmax(self.spell_head(context), x["m_spell"]),
			"logp_spell_unit": masked_log_softmax(spell_unit, x["m_spell_unit"]),
			"logp_spell_hex": masked_log_softmax(spell_hex, x["m_spell_hex"]),
			"logp_spell2_hex": masked_log_softmax(spell2_hex, x["m_spell2_hex"]),
			"logp_spell2_unit": masked_log_softmax(spell2_unit, x["m_spell2_unit"]),
			**self._values(context),
		}

	def _values(self, context):
		logits, previous = [], None
		for head in self.value_heads:
			logit = head(context).squeeze(-1)
			if previous is not None:
				logit = logit + (previous.detach() if self.boost_stop_gradient else previous)
			logits.append(logit)
			previous = logit
		value_logit = torch.stack(logits, dim=-1)
		# value_logit is for the training loss only; it is not a model output in the contract
		return {"value": torch.tanh(value_logit), "value_logit": value_logit}


class OnnxWrapper(nn.Module):
	"""Positional-argument wrapper for ONNX export: inputs and outputs in schema order."""

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, *tensors):
		outputs = self.model(dict(zip(INPUT_NAMES, tensors)))
		return tuple(outputs[name] for name in OUTPUT_NAMES)
