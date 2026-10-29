"""The battle model (v3): the contract's raw inputs to its outputs (the log-probability of every action, the value
heads), built on features.py: the game's own rules, computed exactly, as bounded and scale-free features.

	forward(x, facts=None)
	  facts    = facts(x)             features.py, no parameters (training computes them once per decision and epoch)
	  tokens   = tokenizers(...)      one token for the battle, each hero, each unit, each hex
	  tokens   = trunk(tokens)        attention over all of them, knowing ally from enemy
	  context  = Context(tokens)      the battle, the acting unit, our hero, both sides summed, each side's strength
	  logits   = ActionHeads(...)     every choice scored (below); values = ValueHeads(context)

What the model must learn first is made easy to learn:
  - Ours is good, theirs is bad. Each unit's and hero's token has a signed half: multiplied by +1 if ours, -1 if the
    enemy's (ours: whoever controls it now, so a hypnotized enemy is ours). The other half is the same on either side
    ("an Archangel is on the field"). So "a strong stack is good if ours, bad if theirs" is one linear readout, and
    summing the units' tokens gives ours minus theirs. RMSNorm (no mean subtraction) keeps the signed half signed.
  - The balance of power. Strength: Lanchester's square law (features.side_strengths), our share of health x damage
    per round, in four variants, each with a learned exponent per term (starting at 1, the law itself): 8 parameters.
    The shares go into the context and, through their own small layers, straight into the value heads (the final one
    starting at V = 2 share - 1, the "all units" share: a sane prior, not 0) and the type head (starting at 0), so a
    few very informative numbers are not drowned among the context's hundreds. Context also sums each side's value
    and health times learned per-unit factors that start at 1 (exp of a zero layer), and gives our share of those.
  - Ally and enemy. Each attention head has a learned bias for pairs of tokens on the same side and on opposite sides.
  - Forward is forward. Positions also come in our frame (features.our_frame): the same numbers for both sides.
  - What a spell does. Each spell target gets the engine's prediction of the cast (features.spell_features); a spell
    that changes no health (Slow, Bless, ...) gets a learned value per spell instead of versus_best, starting at 0.5,
    and every target its value (target_value): whom a buff or a curse is worth casting on is for the model to learn.
  - What an attack does. Each legal attack gets its value as the engine's own rules play it on a copy of the battle
    (features.attack_values: retaliation, area attacks, Fire Shield, life drain, jousting, ... all included), and
    every damage feature comes from the engine's damage estimates (hero skills and artifacts included).
  - What a creature is. Besides its own embedding, each ability it has (rules.ABILITIES) adds an embedding shared by
    all creatures with it, so an ability learned on one holds for the others.
  - Retreat and surrender. Their exact reward now, and the score if the battle ended now, go into the type logits
    (features.outcome_features).
  - Take what gains (2026-10-29, after joint2's shooters drifted into idling: the policy gradient for those shots was
    noise, so nothing held them up). VCMI's BattleAI takes the action whose value, as the engine plays it, is best,
    and only idles when nothing gains. Here every shot, melee attack (from each hex), hero spell and creature spell
    target gets a value prior, VALUE_PRIOR_LOGITS x gain x tanh(its engine value / VALUE_PRIOR_SCALE), which reaches its
    action type through the best follow-up; waiting, defending and moving get none. A gain per kind is learned within
    x1/5..x5 from 1: the priors help, they do not decide (a shooter may still learn to run).

Each token is a gated MLP (SwiGLU) over [its embeddings summed, its numbers], then RMSNorm; the trunk is pre-norm
attention blocks with RMSNorm and SwiGLU. A choice's logit is as in v2 (ActionHeads): a pointer score (the context's
query against the choice's token), its own features' score (ActionScore: linear, a tiny layer, a wide layer), the
priors (priors.py) and the logit of its best follow-up. Untrained, the pointer and feature scores are 0: every choice
is as likely as its priors say. The value heads are as in v2.
"""
import math

import torch
from torch import nn

from . import features as F
from . import priors, rules
from .constants import (
	ACCESSIBILITY_VOCAB,
	BATTLEFIELD_VOCAB,
	CREATURE_VOCAB,
	EFFECT_VOCAB,
	GATE_STATE_VOCAB,
	HEX_TYPES,
	MASKED_LOGIT,
	N_ACTION_TYPES,
	N_EVENTS,
	N_HEXES,
	N_SECONDARY_SKILLS,
	N_SIDES,
	N_SPELLS,
	OBSTACLE_TYPE_VOCAB,
	SECOND_TARGET_KIND,
	TARGET_TYPES,
	WALL_STATE_VOCAB,
	ActionType,
)
from .reward import RewardConfig
from .layers import EMBEDDING_STD, ActionScore, ValueHeads, best_of, masked_log_softmax, small_embedding, vocab_index

__all__ = ["BattleModel", "best_of", "build_model"]


class RMSNorm(nn.Module):
	"""x / rms(x) * weight: no mean subtraction, so a token's signed half stays exactly signed."""

	def __init__(self, dim, eps=1e-6):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(dim))
		self.eps = eps

	def forward(self, x):
		return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
	"""Gated MLP: down(silu(gate(x)) * up(x))."""

	def __init__(self, inputs, hidden, outputs):
		super().__init__()
		self.up = nn.Linear(inputs, 2 * hidden)
		self.down = nn.Linear(hidden, outputs)

	def forward(self, x):
		up, gate = self.up(x).chunk(2, dim=-1)
		return self.down(nn.functional.silu(gate) * up)


class Tokenizer(nn.Module):
	"""[embeddings summed, numbers] -> a token: a small gated MLP (half the token's width inside: the numbers and the
	embeddings combine nonlinearly, v2 only added a linear projection of the numbers), then RMSNorm. embedded=False:
	numbers only (forward's embedded is None)."""

	def __init__(self, numbers, dim, embedded=True):
		super().__init__()
		self.mlp = SwiGLU((dim if embedded else 0) + numbers, dim // 2, dim)
		self.norm = RMSNorm(dim)

	def forward(self, embedded, numbers):
		return self.norm(self.mlp(numbers if embedded is None else torch.cat([embedded, numbers], dim=-1)))


def signed(tokens, team):
	"""The first half of each token times its team: +1 ours, -1 the enemy's (0: no team, unchanged)."""
	half = tokens.shape[-1] // 2
	sign = torch.where(team == 0, torch.ones_like(team), team).float()[..., None]
	return torch.cat([tokens[..., :half] * sign, tokens[..., half:]], dim=-1)


def teams(b):
	"""[B, U]: +1 for units we control, -1 for the enemy's, 0 for empty slots."""
	controller, present = b.x["u_owner_side"], b.x["u_present"] != 0
	ours = torch.where(controller == b.x["g_our_side"][:, None], torch.ones_like(controller), -torch.ones_like(controller))
	return torch.where(present, ours, torch.zeros_like(controller))


class BattleTokens(nn.Module):
	"""The global token: the battlefield, the gate, and the battle's features (features.battle_features)."""

	def __init__(self, dim):
		super().__init__()
		self.battlefield = small_embedding(BATTLEFIELD_VOCAB + 1, dim)
		self.gate = small_embedding(GATE_STATE_VOCAB + 1, dim)
		self.tokens = Tokenizer(len(F.BATTLE_FEATURES), dim)

	def forward(self, b, facts):
		x = b.x
		embedded = self.battlefield(vocab_index(x["g_battlefield"], BATTLEFIELD_VOCAB)) + self.gate(vocab_index(x["g_gate_state"], GATE_STATE_VOCAB))
		return self.tokens(embedded, facts["battle"])


class HeroTokens(nn.Module):
	"""One token per side's hero, signed by team: whether it is ours; its attack against the other hero's defence and
	the other's attack against its defence, as the damage formula's factor (F.damage_factor: no hero is 0 and 0);
	spell power, knowledge and level at several scales (F.COUNT_SCALES); mana left of its pool, mana and the pool at
	several scales (F.MANA_SCALES); what it is worth (reward.hero_asset) of both sides' start assets; secondary skill
	levels (of 3) and spellbook; how often it has cast this battle and how long ago. The enemy hero's level, artifacts,
	skills and spellbook are 0: a player cannot see them; of its spells, those it was seen casting count as known. No embedding of the hero's type: what a hero is shows in these, and its specialty in its units' numbers and
	its spells' predicted values.

	TODO(hero identity): a small hero-type embedding (a few dims) can matter for specialties the numbers miss, but
	keep it out until training data is varied: campaign games give many consecutive battles of the same heroes, and
	an identity input then lets the model memorise "this hero's battles" instead of learning the battle (overfitting)."""

	RULE_SLOTS = 48  # rule artifacts the bag has room for (rules.rule_artifacts finds ~40)
	RULE_DIMS = 4  # what they become before meeting anything else: an edge case, kept small

	NUMBERS = 4 + 3 * len(F.COUNT_SCALES) + 1 + 2 * len(F.MANA_SCALES) + 1 + N_SECONDARY_SKILLS + N_SPELLS + RULE_DIMS + 2 * len(F.COUNT_SCALES) + 1

	def __init__(self, dim):
		super().__init__()
		self.tokens = Tokenizer(self.NUMBERS, dim, embedded=False)
		# Our hero's rule-changing artifacts (rules.RULE_ARTIFACT_BONUSES: Orb of Vulnerability, Recanter's Cloak, ...):
		# a bag mapped to RULE_DIMS numbers first, starting at 0. The enemy's artifacts are hidden.
		table, _ = rules.rule_artifacts()
		self.register_buffer("rule_index", table.clamp(max=self.RULE_SLOTS - 1), persistent=False)
		self.rule_artifacts = nn.Linear(self.RULE_SLOTS, self.RULE_DIMS, bias=False)
		nn.init.zeros_(self.rule_artifacts.weight)

	def rule_bag(self, worn):
		"""[B, side, RULE_SLOTS]: 1 for each rule artifact the hero wears (worn: [B, side, N_WORN_SLOTS] ids, -1 none)."""
		ids = worn.long()
		index = self.rule_index[torch.where((ids >= 0) & (ids < self.rule_index.shape[0] - 1), ids, torch.full_like(ids, self.rule_index.shape[0] - 1))]
		slots = torch.arange(self.RULE_SLOTS, device=worn.device)
		return ((index[..., None] == slots) & (index[..., None] >= 0)).any(dim=-2).float()

	def forward(self, b):
		x = b.x
		present = x["h_present"] != 0
		attack, defense = x["h_attack"] * present, x["h_defense"] * present
		mana, pool = x["h_mana"].float(), x["h_mana_max"].float()
		sides = torch.arange(N_SIDES, device=mana.device)[None, :].expand_as(x["h_present"])
		ours = sides == x["g_our_side"][:, None]
		other = lambda t: t.flip(-1)  # the other side's hero
		numbers = torch.cat([
			torch.stack([present.float(), ours.float(), F.damage_factor(attack - other(defense)), F.damage_factor(other(attack) - defense)], dim=-1),
			F.scales(x["h_power"], F.COUNT_SCALES),
			F.scales(x["h_knowledge"], F.COUNT_SCALES),
			F.scales(F.hero_levels(x), F.COUNT_SCALES),
			F.fraction(mana, pool)[..., None],
			F.scales(mana, F.MANA_SCALES),
			F.scales(pool, F.MANA_SCALES),
			F.fraction(F.hero_assets(x, RewardConfig()), (x["g_start_army"].float().sum(dim=-1, keepdim=True) + F.hero_assets(x, RewardConfig()).sum(dim=-1, keepdim=True)))[..., None],
			x["h_skills"].float().clamp(0, 3) / 3,
			# The spells it has: all of ours; of the enemy's (hidden, h_spells 0) those it was seen casting this battle
			((x["h_spells"] != 0) | (x["h_seen_spells"] > 0)).float(),
			self.rule_artifacts(self.rule_bag(x["h_worn_artifacts"])),
			# How it has cast so far: how often, and how long ago (a hero that has not cast in rounds has little to cast)
			F.scales(x["h_casts"], F.COUNT_SCALES),
			(x["h_last_cast_round"] < 0).float()[..., None],
			F.scales((x["g_round"][:, None] - x["h_last_cast_round"]).clamp(min=0) * (x["h_last_cast_round"] >= 0), F.COUNT_SCALES),
		], dim=-1)
		team = torch.where(present, 2 * ours.long() - 1, torch.zeros_like(sides))
		return signed(self.tokens(None, numbers), team), team


class UnitTokens(nn.Module):
	"""One token per unit slot, signed by team: its creature and its abilities (one embedding per ability it has,
	rules.ABILITIES, summed: what is learned about an ability on one creature holds for all with it), its spell effects
	(each with its turns left and strength at several scales, through a small gated MLP), where it stands (both hexes
	of a two-hex unit), and its numbers: stack_features, unit_summaries, its position in our frame, and what the battle
	showed of it (history)."""

	EFFECT_TURNS = (1, 3, 9)  # rounds an effect has left
	EFFECT_STRENGTH = (2, 10, 50)  # its bonus's value: a few points of a skill, or tens of percent
	HISTORY_RATIOS = (0.25, 1, 4)  # damage dealt or taken this battle, of the stack's own hit points at the start
	HISTORY_NUMBERS = 1 + 2 * len(HISTORY_RATIOS) + len(F.COUNT_SCALES) + 1 + len(F.COUNT_SCALES)

	@classmethod
	def history(cls, x):
		"""[B, U, HISTORY_NUMBERS]: what the battle showed of each unit (u_start_count .. u_last_round): the share of its
		creatures lost, the damage it dealt and took of its start hit points, its kills, and how long since it acted."""
		start = x["u_start_count"].float().clamp(min=1)
		health = start * x["u_max_hp"].float().clamp(min=1)
		acted = x["u_last_round"] >= 0
		return torch.cat([
			F.fraction(start - x["u_count"].float(), start)[..., None],
			F.scales(x["u_dealt"].float() / health, cls.HISTORY_RATIOS),
			F.scales(x["u_taken"].float() / health, cls.HISTORY_RATIOS),
			F.scales(x["u_kills"], F.COUNT_SCALES),
			(~acted).float()[..., None],
			F.scales((x["g_round"][:, None] - x["u_last_round"]).clamp(min=0) * acted, F.COUNT_SCALES),
		], dim=-1)

	def __init__(self, dim):
		super().__init__()
		self.creature = small_embedding(CREATURE_VOCAB + 1, dim)
		self.abilities = nn.Linear(len(rules.ABILITIES), dim, bias=False)
		nn.init.normal_(self.abilities.weight, std=EMBEDDING_STD)
		self.position = small_embedding(N_HEXES + 1, dim)  # the extra row: no second hex
		self.effect = small_embedding(EFFECT_VOCAB + 1, dim)
		self.effect_mlp = SwiGLU(dim + len(self.EFFECT_TURNS) + len(self.EFFECT_STRENGTH), 16, dim)
		self.tokens = Tokenizer(len(F.STACK_FEATURES) + len(F.SUMMARY_FEATURES) + 2 + self.HISTORY_NUMBERS, dim)

	def forward(self, b, facts):
		x = b.x
		has_effect = (x["u_effect_id"] >= 0).float()[..., None]
		effect_numbers = torch.cat([F.scales(x["u_effect_turns"], self.EFFECT_TURNS), F.signed_scales(x["u_effect_value"], self.EFFECT_STRENGTH)], dim=-1)
		effects = (self.effect_mlp(torch.cat([self.effect(vocab_index(x["u_effect_id"], EFFECT_VOCAB)), effect_numbers], dim=-1)) * has_effect).sum(dim=2)
		where = self.position(vocab_index(b.units["hex"], N_HEXES)) + self.position(vocab_index(b.units["rear"], N_HEXES))
		abilities = self.abilities(b.tables["abilities"][b.units["creature"]])
		embedded = self.creature(vocab_index(x["u_creature"], CREATURE_VOCAB)) + abilities + effects + where
		numbers = torch.cat([facts["stack"], facts["summaries"], F.our_frame(b.units["hex"], x["g_our_side"]), self.history(x)], dim=-1)
		team = teams(b)
		return signed(self.tokens(embedded, numbers), team), team


class HexTokens(nn.Module):
	"""One token per hex: its position (also in our frame), accessibility, obstacle type and wall, the side and turns
	(at two scales) of a spell obstacle, and the acting unit's maps there (features.threat_maps). No embedding of the
	obstacle's id: its type and the accessibility say what it does."""

	OBSTACLE_TURNS = (1, 4)

	def __init__(self, dim):
		super().__init__()
		self.position = small_embedding(N_HEXES, dim)
		self.access = small_embedding(ACCESSIBILITY_VOCAB + 1, dim)
		self.obstacle_type = small_embedding(OBSTACLE_TYPE_VOCAB + 1, dim)
		self.wall = small_embedding(WALL_STATE_VOCAB + 1, dim)
		self.tokens = Tokenizer(3 + len(self.OBSTACLE_TURNS) + len(F.HEX_FEATURES) + 2, dim)

	def forward(self, b, facts):
		x = b.x
		side, our = x["x_obstacle_side"], x["g_our_side"][:, None]
		obstacle = torch.cat([torch.stack([(side == our).float(), ((side >= 0) & (side != our)).float(), (x["x_obstacle_type"] >= 0).float()], dim=-1), F.scales(x["x_obstacle_turns"], self.OBSTACLE_TURNS)], dim=-1)
		hexes = torch.arange(N_HEXES, device=side.device)[None, :].expand_as(side)
		embedded = (
			self.position(hexes)
			+ self.access(vocab_index(x["x_access"], ACCESSIBILITY_VOCAB))
			+ self.obstacle_type(vocab_index(x["x_obstacle_type"], OBSTACLE_TYPE_VOCAB))
			+ self.wall(vocab_index(x["x_wall_state"], WALL_STATE_VOCAB))
		)
		return self.tokens(embedded, torch.cat([obstacle, facts["maps"], F.our_frame(hexes, x["g_our_side"])], dim=-1))


class EventTokens(nn.Module):
	"""One token per entry of the event log (e_*: the battle's last actions of both sides), signed by the actor's team:
	the action's type, spell and target hex (embeddings), the acting unit's and the target's own tokens (or the
	acting hero's), each through its own projection, so attention can tie an event to the units in it; how many rounds
	ago, and the damage and creatures each side lost in it at several scales. What the present state cannot say: who
	has been going for whom, what the enemy hero casts, how hard each blow landed (history, as AlphaGo's input has
	the last positions)."""

	AGE_SCALES = (1, 3, 9)  # rounds ago
	HP_SCALES = (30, 300, 3000)  # hit points of damage
	NUMBERS = len(AGE_SCALES) + 2 * len(HP_SCALES) + 2 * len(F.COUNT_SCALES)

	def __init__(self, dim):
		super().__init__()
		self.action = small_embedding(N_ACTION_TYPES + 1, dim)  # the extra row: an other action (not .type: nn.Module's)
		self.spell = small_embedding(N_SPELLS + 1, dim)  # no spell
		self.hex = small_embedding(N_HEXES + 1, dim)  # no hex
		self.actor = nn.Linear(dim, dim, bias=False)
		self.target = nn.Linear(dim, dim, bias=False)
		self.tokens = Tokenizer(self.NUMBERS, dim)

	def forward(self, b, units, heroes):
		"""units, heroes: their tokens before the trunk ([B, U, dim], [B, side, dim])."""
		x = b.x
		present = x["e_present"] != 0

		def pick(tokens, index):  # tokens[index] per event, 0 where index < 0
			rows = torch.gather(tokens, 1, index.clamp(min=0)[..., None].expand(-1, -1, tokens.shape[-1]))
			return rows * (index >= 0)[..., None].float()

		hero = pick(heroes, x["e_side"].clamp(0, N_SIDES - 1)) * (x["e_actor"] == -1)[..., None].float()
		embedded = (
			self.action(vocab_index(x["e_type"], N_ACTION_TYPES))
			+ self.spell(vocab_index(x["e_spell"], N_SPELLS))
			+ self.hex(vocab_index(x["e_hex"], N_HEXES))
			+ self.actor(pick(units, x["e_actor"]) + hero)
			+ self.target(pick(units, x["e_target"]))
		)
		numbers = torch.cat([
			F.scales((x["g_round"][:, None] - x["e_round"]).clamp(min=0), self.AGE_SCALES),
			F.scales(x["e_dealt"], self.HP_SCALES),
			F.scales(x["e_taken"], self.HP_SCALES),
			F.scales(x["e_kills"], F.COUNT_SCALES),
			F.scales(x["e_lost"], F.COUNT_SCALES),
		], dim=-1)
		ours = x["e_side"] == x["g_our_side"][:, None]
		team = torch.where(present, torch.where(ours, 1, -1), 0)
		return signed(self.tokens(embedded, numbers), team), team


class Block(nn.Module):
	"""Pre-norm attention and a SwiGLU MLP (4/3 of the width inside: as many parameters as v2's GELU MLP of twice the
	width), with RMSNorm. Each head adds a learned bias to the attention between two
	tokens on opposite sides, where either has no side, or on the same side (relation 0, 1, 2); all start at 0."""

	def __init__(self, dim, heads):
		super().__init__()
		self.heads = heads
		self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
		self.qkv = nn.Linear(dim, 3 * dim)
		self.proj = nn.Linear(dim, dim)
		self.mlp = SwiGLU(dim, 4 * dim // 3, dim)
		self.relation = nn.Parameter(torch.zeros(heads, 3))

	def forward(self, x, valid, relation):
		b, n, d = x.shape
		q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
		q, k, v = (t.reshape(b, n, self.heads, d // self.heads).transpose(1, 2) for t in (q, k, v))
		# Not self.relation[:, relation]: on the CPU indexing a parameter with an index this big ([B, T, T]) has a
		# backward that was 80% of a training step; a choice among the three there (and in the export), an embedding
		# lookup on a GPU (each the fastest there, fwd+bwd at batch 256: CPU 23 ms vs 2 s, Apple GPU 17 ms vs 28 ms)
		if x.device.type == "cpu":
			r, rel = self.relation[:, :, None, None], relation[:, None]
			bias = torch.where(rel == 0, r[:, 0], torch.where(rel == 1, r[:, 1], r[:, 2]))  # [B, heads, T, T]
		else:
			bias = nn.functional.embedding(relation, self.relation.T).permute(0, 3, 1, 2)
		att = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.heads) + bias
		att = att.masked_fill(~valid[:, None, None, :], MASKED_LOGIT).softmax(dim=-1)
		x = x + self.proj((att @ v).transpose(1, 2).reshape(b, n, d))
		return x + self.mlp(self.norm2(x))


class Trunk(nn.Module):
	"""Attention over all tokens (the battle, heroes, units, events, hexes; a learned kind embedding each)."""

	def __init__(self, dim, layers, heads):
		super().__init__()
		self.kind = small_embedding(5, dim)  # battle, hero, unit, hex, event
		self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
		self.norm = RMSNorm(dim)

	def forward(self, glob, heroes, units, hexes, hero_team, unit_team, hero_valid, unit_valid, events, event_team, event_valid):
		k = self.kind.weight
		tokens = torch.cat([(glob + k[0])[:, None, :], heroes + k[1], units + k[2], events + k[4], hexes + k[3]], dim=1)
		none = torch.zeros_like(unit_team[:, :1])
		team = torch.cat([none, hero_team, unit_team, event_team, none.expand(-1, N_HEXES)], dim=1)
		relation = (team[:, :, None] * team[:, None, :]).sign() + 1  # 0 opposite sides, 1 either has none, 2 same side
		always = torch.ones_like(unit_valid[:, :1])
		valid = torch.cat([always, hero_valid, unit_valid, event_valid, always.expand(-1, N_HEXES)], dim=1)
		for block in self.blocks:
			tokens = block(tokens, valid, relation)
		tokens = self.norm(tokens)
		units_end = 1 + N_SIDES + units.shape[1]
		return tokens[:, 0], tokens[:, 1 : 1 + N_SIDES], tokens[:, 1 + N_SIDES : units_end], tokens[:, -N_HEXES:]


class Context(nn.Module):
	"""The decision's context, through a gated MLP: the battle's token, the acting unit's and our hero's, the mean of
	the living units' (its signed half: ours minus theirs), and our share of each side's strength.

	Strength k of a side: the sum over its living units of their value (k < strengths) or health (the rest) times
	exp(w_k . token). w starts at 0, so untrained the shares are our share of the armies' value and of their health;
	trained, of whatever makes a side strong (a stack about to die, a stack that cannot reach anyone, ...)."""

	def __init__(self, dim, strengths=8):
		super().__init__()
		self.strengths = strengths
		self.factor = nn.Linear(dim, 2 * strengths)
		nn.init.zeros_(self.factor.weight)
		nn.init.zeros_(self.factor.bias)
		self.mlp = SwiGLU(4 * dim + 2 * strengths + len(F.STRENGTH_VARIANTS), dim // 2, dim)
		self.norm = RMSNorm(dim)

	def forward(self, b, glob, heroes, units, strength):
		living = b.units["alive"].float()[..., None]
		mean = (units * living).sum(dim=1) / living.sum(dim=1).clamp(min=1)
		base = torch.cat([b.value[..., None].expand(-1, -1, self.strengths), b.health.float()[..., None].expand(-1, -1, self.strengths)], dim=-1)
		weighted = base * self.factor(units).clamp(-5, 5).exp()
		ours = (weighted * b.ours[..., None]).sum(dim=1)
		theirs = (weighted * b.enemy[..., None]).sum(dim=1)
		batch = torch.arange(glob.shape[0], device=glob.device)
		our_hero = heroes[batch, b.x["g_our_side"].clamp(0, N_SIDES - 1)]
		return self.norm(self.mlp(torch.cat([glob, units[batch, b.acting], our_hero, mean, F.relative(ours, theirs), torch.sigmoid(strength)], dim=-1)))


class Strength(nn.Module):
	"""Our share of each Lanchester variant's strength (features.side_strengths), as log-odds:
	z = a log(H ours / H theirs) + b log(D ours / D theirs), share = sigmoid(z); a and b start at 1."""

	def __init__(self):
		super().__init__()
		self.exponents = nn.Parameter(torch.ones(len(F.STRENGTH_VARIANTS), 2))

	def forward(self, ratios):
		return (ratios * self.exponents).sum(dim=-1)  # [B, variants]


# The value priors (the module's docstring, "Take what gains"): a shot worth the median of joint2's (0.018 of both
# armies' value) gets ~1.4 logits, one worth 0.05 ~2; a melee trade that loses 0.04 ~-1.9. Over waiting, defending and
# moving that is what the type gets too: in joint2 the 5% of shooter states where an idle type beat SHOOT were behind by
# up to ~1 logit (the median shooter state is 7 ahead)
VALUE_PRIOR_KINDS = ("shot", "melee", "spell", "creature_spell")
VALUE_PRIOR_LOGITS = 2.0
VALUE_PRIOR_SCALE = 0.02  # of both armies' value: the median best shot
VALUE_PRIOR_GAIN = 5.0  # each kind's gain within x1/5..x5
VALUE_PRIOR_STEP = 10.0  # its parameter moves this much faster than a plain weight (x5 in ~1,000 consistent steps)


class ActionHeads(nn.Module):
	"""Every choice's logit: pointer score + its features' score + priors + the value of its best follow-up."""

	POINTERS = ("move", "melee", "melee_from", "shoot", "spell", "spell2", "creature_spell")
	KEYS = ("move_hex", "melee_unit", "melee_from_unit", "melee_from_hex", "shoot_unit", "spell_unit", "spell_hex", "spell2_from_unit", "spell2_hex", "spell2_unit", "creature_spell_unit", "creature_spell_hex")
	FEATURES = {
		"melee_target": F.ATTACK_FEATURES,
		"melee_from": F.FROM_FEATURES,
		"shoot_target": F.SHOT_FEATURES,
		"move": F.MOVE_FEATURES,
		"spell_unit": F.SPELL_FEATURES,
		"spell_hex": F.SPELL_FEATURES,
		"creature_spell_unit": F.SPELL_FEATURES,
		"creature_spell_hex": F.SPELL_FEATURES,
	}
	NO_HEALTH_VALUE = 0.5  # versus_best of a spell that changes no health, before training: half as good as the best

	def __init__(self, dim):
		super().__init__()
		self.type_head = nn.Linear(dim, N_ACTION_TYPES)
		self.type_rules = nn.Linear(len(F.TYPE_FEATURES), N_ACTION_TYPES, bias=False)
		self.type_strength = nn.Linear(len(F.STRENGTH_VARIANTS), N_ACTION_TYPES, bias=False)  # the strength shares' log-odds
		# The escapes: their exact reward now (features.outcome_features), one weight each, starting at 0; and their prior
		# bias lifted when we are much weaker (priors.escape_relief)
		self.escape_rules = nn.Parameter(torch.zeros(len(F.OUTCOME_FEATURES)))
		escapes = torch.zeros(len(F.OUTCOME_FEATURES), N_ACTION_TYPES)
		escapes[0, ActionType.RETREAT] = escapes[1, ActionType.SURRENDER] = 1.0
		escapes[2:, ActionType.RETREAT] = escapes[2:, ActionType.SURRENDER] = 1.0  # the exposure features: both escapes
		self.register_buffer("escape_types", escapes, persistent=False)  # [outcome feature, type]
		self.register_buffer("escape_bias", -priors.action_types() * escapes.sum(dim=0), persistent=False)  # what relief lifts
		self.spell_head = nn.Linear(dim, N_SPELLS)
		self.spell_query = small_embedding(N_SPELLS, dim)
		self.creature_spell_query = small_embedding(EFFECT_VOCAB + 1, dim)  # by the acting unit's spell
		self.queries = nn.ModuleDict({name: nn.Linear(dim, dim) for name in self.POINTERS})
		self.keys = nn.ModuleDict({name: nn.Linear(dim, dim) for name in self.KEYS})
		# Spell targets: three features each, at up to 70 spells x 187 hexes, so one weight per feature (the pointer score
		# is their learned nonlinear part)
		self.rules = nn.ModuleDict({head: ActionScore(names, dim, linear_only="spell" in head) for head, names in self.FEATURES.items()})
		self.spell_mastery = nn.Parameter(priors.mastery())  # [spell, our hero's mastery of its school]
		self.spell_value = nn.Parameter(torch.full((N_SPELLS,), self.NO_HEALTH_VALUE))  # spells that change no health
		self.creature_spell_value = nn.Parameter(torch.full((EFFECT_VOCAB + 1,), self.NO_HEALTH_VALUE))
		second = torch.zeros(N_SPELLS, dtype=torch.long)
		for spell, kind in SECOND_TARGET_KIND.items():
			second[spell] = int(kind)
		self.register_buffer("second_target", second, persistent=False)
		self.register_buffer("type_one_hot", torch.eye(N_ACTION_TYPES), persistent=False)
		self.value_gains = nn.Parameter(torch.zeros(len(VALUE_PRIOR_KINDS)))  # each kind's value prior: log-gain, 0 = x1
		with torch.no_grad():  # untrained: the biases (priors.py), no type rules, zero pointer scores
			self.type_head.weight.zero_()
			self.type_head.bias.copy_(priors.action_types())
			self.type_rules.weight.zero_()
			self.type_strength.weight.zero_()
			for layer in (*self.queries.values(), self.keys["melee_from_unit"], self.keys["spell2_from_unit"]):
				layer.weight.zero_()
				layer.bias.zero_()
			self.spell_head.weight.zero_()
			self.spell_head.bias.copy_(priors.spells())
			self.spell_query.weight.zero_()
			self.creature_spell_query.weight.zero_()

	def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
		"""A checkpoint (or a league snapshot) from before the value priors: they start at their prior."""
		if prefix + "value_gains" not in state_dict:
			state_dict[prefix + "value_gains"] = torch.zeros_like(self.value_gains)
		super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

	def value_prior(self, value, kind):
		"""The logit an option's engine value (of both armies' value, -1..1) adds: VALUE_PRIOR_LOGITS x gain x tanh(value /
		VALUE_PRIOR_SCALE), the gain of its kind (VALUE_PRIOR_KINDS) learned within x1/VALUE_PRIOR_GAIN..xVALUE_PRIOR_GAIN."""
		raw = self.value_gains[VALUE_PRIOR_KINDS.index(kind)]
		gain = VALUE_PRIOR_GAIN ** torch.tanh(VALUE_PRIOR_STEP * raw)
		return VALUE_PRIOR_LOGITS * gain * torch.tanh(value / VALUE_PRIOR_SCALE)

	def mastery_bias(self, x):
		"""[B, N_SPELLS]: each spell's bias for our hero's mastery of it (none, basic, advanced, expert: the engine's,
		h_spell_mastery), learned per spell and mastery from priors.MASTERY_BIAS."""
		our = x["g_our_side"].clamp(0, N_SIDES - 1)
		level = x["h_spell_mastery"][torch.arange(our.shape[0], device=our.device), our].long().clamp(0, 3)
		spells = torch.arange(self.spell_mastery.shape[0], device=level.device)
		return self.spell_mastery[spells[None, :], level]

	def spell_actions(self, actions, x):
		"""The spell features, versus_best replaced by the spell's learned value where the spell changes no health."""
		best, changes = F.SPELL_FEATURES.index("versus_best"), F.SPELL_FEATURES.index("changes_health")
		creature = self.creature_spell_value[vocab_index(x["g_creature_spell"], EFFECT_VOCAB)]  # [B]
		learned = {"spell_unit": self.spell_value[None, :, None], "spell_hex": self.spell_value[None, :, None], "creature_spell_unit": creature[:, None], "creature_spell_hex": creature[:, None]}
		out = dict(actions)
		for head, value in learned.items():
			f = actions[head]
			versus = torch.where(f[..., changes] > 0, f[..., best], value.expand_as(f[..., best]))
			out[head] = torch.cat([f[..., :best], versus[..., None], f[..., best + 1 :]], dim=-1)
		return out

	def forward(self, context, units, hexes, actions, x, strength, outcomes, ratios):
		scale = 1.0 / math.sqrt(context.shape[-1])
		q = {name: layer(context) for name, layer in self.queries.items()}
		k = self.keys
		point = lambda query, items, key: torch.einsum("bd,bnd->bn", query, k[key](items)) * scale
		actions = self.spell_actions(actions, x)
		rule = lambda head, legal: self.rules[head](actions[head], legal, context)

		# Leaves first: a choice's logit includes the value of the best choice it leads to. The engine's value of a melee
		# attack from each hex: the target's (from its best hex) plus that hex's against the best (hex_value, <= 0)
		melee_value = actions["melee_target"][..., F.ATTACK_FEATURES.index("net_value"), None] + actions["melee_from"][..., F.FROM_FEATURES.index("hex_value")]
		melee_from = torch.einsum("bud,bhd->buh", k["melee_from_unit"](units) + q["melee_from"][:, None, :], k["melee_from_hex"](hexes)) * scale + rule("melee_from", x["m_melee_from"]) + self.value_prior(melee_value, "melee")
		melee_target = point(q["melee"], units, "melee_unit") + rule("melee_target", x["m_melee_target"]) + best_of(melee_from, x["m_melee_from"])
		shot_value = actions["shoot_target"][..., F.SHOT_FEATURES.index("net_value")]  # (0 at our own units: a tent's HEAL targets)
		shoot_target = point(q["shoot"], units, "shoot_unit") + rule("shoot_target", x["m_shoot_target"]) + self.value_prior(shot_value, "shot")
		move = point(q["move"], hexes, "move_hex") + rule("move", x["m_move_hex"])

		spell2_from = k["spell2_from_unit"](units) + q["spell2"][:, None, :]
		spell2_hex = torch.einsum("bud,bhd->buh", spell2_from, k["spell2_hex"](hexes)) * scale
		spell2_unit = torch.einsum("bud,bvd->buv", spell2_from, k["spell2_unit"](units)) * scale
		spell_q = self.spell_query.weight[None, :, :] + q["spell"][:, None, :]
		second_value = torch.stack([torch.zeros_like(spell2_hex[..., 0]), best_of(spell2_unit, x["m_spell2_unit"]), best_of(spell2_hex, x["m_spell2_hex"])], dim=1)  # [B, kind, U]
		net = F.SPELL_FEATURES.index("net_value")  # (0 for a spell that changes no health: its learned value is in versus_best)
		spell_unit = torch.einsum("bsd,bud->bsu", spell_q, k["spell_unit"](units)) * scale + rule("spell_unit", x["m_spell_unit"]) + second_value[:, self.second_target, :] + self.value_prior(actions["spell_unit"][..., net], "spell")
		spell_hex = torch.einsum("bsd,bhd->bsh", spell_q, k["spell_hex"](hexes)) * scale + rule("spell_hex", x["m_spell_hex"]) + self.value_prior(actions["spell_hex"][..., net], "spell")
		first_targets = torch.cat([spell_unit, spell_hex], dim=-1)
		spell = self.spell_head(context) + self.mastery_bias(x) + best_of(first_targets, torch.cat([x["m_spell_unit"], x["m_spell_hex"]], dim=-1))

		creature_q = q["creature_spell"] + self.creature_spell_query(vocab_index(x["g_creature_spell"], EFFECT_VOCAB))
		creature_unit = point(creature_q, units, "creature_spell_unit") + rule("creature_spell_unit", x["m_creature_spell_unit"]) + self.value_prior(actions["creature_spell_unit"][..., net], "creature_spell")
		creature_hex = point(creature_q, hexes, "creature_spell_hex") + rule("creature_spell_hex", x["m_creature_spell_hex"]) + self.value_prior(actions["creature_spell_hex"][..., net], "creature_spell")
		creature = (torch.cat([creature_unit, creature_hex], dim=-1), torch.cat([x["m_creature_spell_unit"], x["m_creature_spell_hex"]], dim=-1))

		follow = torch.zeros_like(self.type_head(context))
		hex_types = [(a, move, x["m_move_hex"]) for a in HEX_TYPES]  # MOVE, TACTICS_MOVE and CATAPULT choose their hex there
		target_types = [(a, shoot_target, x["m_shoot_target"]) for a in TARGET_TYPES]  # SHOOT and HEAL their unit
		for action, logits, legal in (*hex_types, (ActionType.MELEE, melee_target, x["m_melee_target"]), *target_types, (ActionType.HERO_SPELL, spell, x["m_spell"]), (ActionType.CREATURE_SPELL, *creature)):
			follow = follow + best_of(logits, legal)[:, None] * self.type_one_hot[action]
		# TODO(non-zero-sum): the reward (reward.py) is zero-sum, so the value heads and these logits see every loss of
		# one side as the other's gain. These actions create or destroy value nobody else gets, and their logits should
		# get a per-side correction (a small separate head, from what the action really costs or earns) once they matter:
		#   SURRENDER       the payment disappears (the winner gets nothing) and the enemy's payment is unknown to us
		#   RETREAT         the army is destroyed without the winner owning it; the rehire is paid to nobody
		#   CATAPULT        walls and towers destroyed: a loss to the town's owner, no gain to the attacker
		#   HERO_SPELL      mana spent is destroyed, not won by the enemy (zero-sum prices it as a gain for them)
		#   and outcomes behind any action: experience and Necromancy for the winner, Eagle Eye, more than two players
		relief = priors.escape_relief(ratios[:, F.STRENGTH_VARIANTS.index("all")].sum(dim=-1))  # [B]
		escape = relief[:, None] * self.escape_bias + (outcomes * self.escape_rules) @ self.escape_types
		kind = self.type_head(context) + self.type_rules(actions["type"]) + self.type_strength(strength) + follow + escape
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
			"logp_creature_spell_unit": masked_log_softmax(creature_unit, x["m_creature_spell_unit"]),
			"logp_creature_spell_hex": masked_log_softmax(creature_hex, x["m_creature_spell_hex"]),
		}


class BattleModel(nn.Module):
	def __init__(self, dim=64, layers=2, heads=4, boost_stop_gradient=True):
		super().__init__()
		for name, table in rules.creature_tables().items():
			self.register_buffer(f"table_{name}", table, persistent=False)
		self.battle = BattleTokens(dim)
		self.heroes = HeroTokens(dim)
		self.units = UnitTokens(dim)
		self.events = EventTokens(dim)
		self.hexes = HexTokens(dim)
		self.trunk = Trunk(dim, layers, heads)
		self.context = Context(dim)
		self.strength = Strength()
		self.actions = ActionHeads(dim)
		self.values = ValueHeads(dim, boost_stop_gradient)
		# The strengths' direct path into the value heads: the final one starts at V = tanh(z / 2) = 2 sigmoid(z) - 1,
		# z the "all units" share's log-odds; the round heads at 0
		self.value_strength = nn.Linear(len(F.STRENGTH_VARIANTS), len(self.values.heads), bias=False)
		with torch.no_grad():
			self.value_strength.weight.zero_()
			self.value_strength.weight[-1, F.STRENGTH_VARIANTS.index("all")] = 0.5

	def tables(self):
		return {name[len("table_"):]: buffer for name, buffer in self.named_buffers() if name.startswith("table_")}

	def facts(self, x):
		"""Everything features.py computes from the inputs: no parameters, no gradients. Training computes them once per
		decision and epoch and passes them to forward; the exported model computes them itself."""
		with torch.no_grad():
			b = F.read_battle(x, self.tables())
			reach, to_enemy = F.distance_maps(b)
			maps, can_attack = F.threat_maps(b, reach, to_enemy)
			outcomes = F.outcome_features(b)
			return {
				"melee": F.attack_features(b, shooting=False),
				"shot": F.attack_features(b, shooting=True),
				"maps": maps,
				"can_attack": can_attack,
				"stack": F.stack_features(b),
				"summaries": F.unit_summaries(b, can_attack),
				"battle": F.battle_features(b, maps),
				"outcomes": outcomes,
				"strength": F.side_strengths(b),
				**{f"spell:{k}": v for k, v in F.spell_features(b).items()},
			}

	def forward(self, x, facts=None):
		facts = self.facts(x) if facts is None else facts
		b = F.read_battle(x, self.tables())
		heroes, hero_team = self.heroes(b)
		units, unit_team = self.units(b, facts)
		events, event_team = self.events(b, units, heroes)
		glob, heroes, units, hexes = self.trunk(self.battle(b, facts), heroes, units, self.hexes(b, facts), hero_team, unit_team, x["h_present"] != 0, x["u_present"] != 0, events, event_team, x["e_present"] != 0)
		strength = self.strength(facts["strength"])
		context = self.context(b, glob, heroes, units, strength)
		actions = F.action_features(b, facts["melee"], facts["shot"], facts["maps"], facts["can_attack"], facts["battle"])
		actions.update({k.split(":", 1)[1]: v for k, v in facts.items() if k.startswith("spell:")})
		return {**self.actions(context, units, hexes, actions, x, strength, facts["outcomes"], facts["strength"]), **self.values(context, self.value_strength(strength))}

	def set_rule_weights(self, head, **weights):
		"""Set a head's linear feature weights by name, e.g. set_rule_weights("melee_target", wipe_chance=2.0)."""
		score = self.actions.rules[head]
		with torch.no_grad():
			for name, value in weights.items():
				score.linear[score.names.index(name)] = value


def build_model(kwargs):
	"""The model a checkpoint's model_kwargs describe: this one ("v3"), model_v2's or model_v1's (no "architecture" key)."""
	kwargs = dict(kwargs)
	architecture = kwargs.pop("architecture", "v1")
	if architecture == "v3":
		return BattleModel(**kwargs)
	if architecture == "v2":
		from .model_v2 import BattleModelV2

		return BattleModelV2(**kwargs)
	from .model_v1 import BattleModelV1

	return BattleModelV1(**{"chain_final_value": True, "kill_rule": False, **kwargs})
