"""The campaign model v2 (docs/campaign_model_v2_draft.md): the same contract as model.py (schema.INPUTS, OUTPUTS),
built as the user's blocks, in order (CampaignModelV2.forward):

	known      = domain_knowledge(x)   features.py and model.CustomFeatures as v1: Φ, armies, fights, tile values and
	                                   danger, the priors; plus the references a size is relative to (our Φ, our main
	                                   army, the strongest rival army we see)
	state      = encode(x)             raw inputs only (encoders.py): the global token, every hero (one function for
	                                   ours, the rivals', the tavern's), town, option, the other army, the map's tiles
	state      = take_sides(state)     ownership transforms every vector: ours, nobody's, a rival's (which rival: its
	                                   own embedding; the rivals' signed half flipped at the start); a token per player
	state      = join(state, known)    the known facts concatenated onto what they describe, sizes also relative
	state      = see(state)            heroes and towns stamped on their tiles, a U-Net over the map, every hero and
	                                   town reading the tiles (its own reach to them, for ours), TRAVEL's areas
	state      = think(state)          attention over the tokens, biased by their tile offsets and owners' relation
	foresight  = foresee(state)        what the next action brings (Foresight): predictions for their losses only,
	                                   their small hidden states for the layers after
	state      = think_again(...)      the foresight and the known facts joined again, one more attention block
	outputs    = decide(state, known)  every logit a match of an actor and a target, plus its domain prior; a type's
	                                   score is its best entry's learned part + what its domain prior offers (a type
	                                   is as good as its best option) + a learned bias from priors/action_types.csv

What the AI reads is unchanged (OUTPUTS); the foresight's predictions (fs_*) are training's only.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from battle_model.features import scales
from battle_model.layers import Block, masked_log_softmax, small_embedding, vocab_index, zero_output
from battle_model.model import RMSNorm, SwiGLU

from . import constants as c
from . import encoders as E
from . import features as ft
from . import priors
from .model import MOVE_FEATURES, CustomFeatures, PairBias, dense, flat_log_softmax, with_added
from .schema import HEADS

HO, TO, A, W, HD = c.OWN_HEROES, c.OWN_TOWNS, c.ARMY_SLOTS, c.WINDOW, c.N_HOLDERS
H, T, L, S, TG, R = c.N_HEROES, c.N_TOWNS, c.MAP_LEVELS, c.MAP_TILES, c.TRAVEL_GRID, c.N_RESOURCES
P = c.MAX_PLAYERS  # player tokens: 0 ours, k the rival whose owner code is OWNER_SELF + k
AREAS = L * TG * TG
FORESIGHT_HIDDEN = 16
# The foresight of an action (move, answer): a battle in its consequences (logit), its score (the battle reward's
# formula, -1..1), and the change of each of Φ's parts until our next turn decision (symlog of thousands of gold)
ACTION_TARGETS = 2 + len(ft.PHI_PARTS)
HERO_TARGETS = 2  # alive and ours tomorrow (logit), its army's log ratio tomorrow
TURN_TARGETS = 1 + R  # Φ's log ratio at the end of our turn, each resource's log ratio tomorrow
DISTANCE_BUCKETS = 12  # the tile readers' distance bias: Chebyshev tiles 0, 1, 2, 3-4, 5-7, 8-11, ..., 96+, another level


class State:
	"""The model's state between the blocks: the inputs (x), the tokens and grids (attributes)."""

	def __init__(self, x):
		self.x = x


class Sides(nn.Module):
	"""take_sides' transform of vectors [..., dim] by their owner codes [...]: per relation (nobody's, ours, a rival's) a
	scale and a shift per channel, plus each rival's own embedding. A rival's first half starts flipped (the battle
	model's signed half: a strong stack is bad when it is theirs), the other half as is."""

	def __init__(self, dim):
		super().__init__()
		half = dim // 2
		gamma = torch.ones(3, dim)
		gamma[2, :half] = -1.0
		self.gamma = nn.Parameter(gamma)
		self.beta = nn.Parameter(torch.zeros(3, dim))
		self.rival = nn.Parameter(torch.zeros(P, dim))

	@staticmethod
	def relation(owner):
		"""(0 nobody's, 1 ours, 2 a rival's; the rival's index 0..P-1)."""
		relation = torch.where(owner == c.OWNER_SELF, torch.ones_like(owner), torch.where(owner > c.OWNER_SELF, torch.full_like(owner, 2), torch.zeros_like(owner)))
		return relation, (owner - c.OWNER_SELF).clamp(0, P - 1)

	def forward(self, vectors, owner):
		relation, rival = self.relation(owner)
		# (lookups as embeddings: a table indexed by a big tensor had CUDA's sorting backward, 95% of a training step's
		# GPU time with the tile reader's, 2026-10-30; the same values)
		return F.embedding(relation, self.gamma) * vectors + F.embedding(relation, self.beta) + F.embedding(rival, self.rival) * (relation == 2).unsqueeze(-1).float()


class MapUNet(nn.Module):
	"""The map's tiles [B * L, C, S, S] through a U-Net: each tile sees the route to it and a day or two around it.
	Returns the tiles [B * L, C, S, S] and the half-resolution level [B * L, 2C, S/2, S/2] (TRAVEL's areas). Off-map
	tiles are zeroed after every block."""

	class Res(nn.Module):
		def __init__(self, channels):
			super().__init__()
			self.norm1, self.norm2 = nn.GroupNorm(1, channels), nn.GroupNorm(1, channels)
			self.conv1, self.conv2 = nn.Conv2d(channels, channels, 3, padding=1), nn.Conv2d(channels, channels, 3, padding=1)
			nn.init.zeros_(self.conv2.weight)
			nn.init.zeros_(self.conv2.bias)

		def forward(self, x, mask):
			return (x + self.conv2(F.gelu(self.norm2(self.conv1(F.gelu(self.norm1(x))))))) * mask

	def __init__(self, channels):
		super().__init__()
		c1, c2, c3 = channels, 2 * channels, 4 * channels
		self.enc1 = self.Res(c1)
		self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
		self.enc2 = self.Res(c2)
		self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
		self.mid1, self.mid2 = self.Res(c3), self.Res(c3)
		self.up2 = nn.Conv2d(c3 + c2, c2, 3, padding=1)
		self.dec2 = self.Res(c2)
		self.up1 = nn.Conv2d(c2 + c1, c1, 3, padding=1)
		self.dec1 = self.Res(c1)

	def forward(self, x, mask):
		m2 = F.max_pool2d(mask, 2, ceil_mode=True)
		m3 = F.max_pool2d(m2, 2, ceil_mode=True)
		e1 = self.enc1(x, mask)
		e2 = self.enc2(F.gelu(self.down1(e1)) * m2, m2)
		m = self.mid2(self.mid1(F.gelu(self.down2(e2)) * m3, m3), m3)
		d2 = self.dec2(F.gelu(self.up2(torch.cat([F.interpolate(m, size=e2.shape[-2:], mode="nearest"), e2], dim=1))) * m2, m2)
		d1 = self.dec1(F.gelu(self.up1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], dim=1))) * mask, mask)
		return d1, d2


def distance_buckets(delta):
	"""Chebyshev distance buckets [..] of tile offsets [.., 3]: 0..10 by distance, DISTANCE_BUCKETS - 1 another level."""
	distance = torch.maximum(delta[..., 0].abs(), delta[..., 1].abs()).float()
	bucket = torch.where(distance < 3, distance, 2 + torch.floor(torch.log2(distance.clamp(min=1)) * 1.5)).clamp(max=DISTANCE_BUCKETS - 2).long()
	return torch.where(delta[..., 2] == 0, bucket, torch.full_like(bucket, DISTANCE_BUCKETS - 1))


class TileReader(nn.Module):
	"""Tokens with a place on the map read the map's tiles: attention [B, N, heads, tiles] with a bias by distance and,
	for our heroes, by their own reach to each tile; its output starts at 0."""

	def __init__(self, dim, cell_dim, heads):
		super().__init__()
		self.heads = heads
		self.q = nn.Linear(dim, dim)
		self.k = nn.Linear(cell_dim, dim)
		self.v = nn.Linear(cell_dim, dim)
		self.distance = nn.Parameter(torch.zeros(heads, DISTANCE_BUCKETS))
		self.reach = nn.Linear(cell_dim, heads)
		self.out = zero_output(nn.Linear(dim, dim))

	def forward(self, tokens, positions, placed, tiles, tile_positions, valid, reach=None):
		"""tokens [B, N, d] at positions [B, N, 3] (placed [B, N]); tiles [B, M, dc] at tile_positions [M, 3] (valid
		[B, M]); reach: its bias [B, N, M, heads] (self.reach of each hero's reach to each tile) or None. Returns what each
		token reads [B, N, d]."""
		b, n, d = tokens.shape
		h, dh = self.heads, d // self.heads
		q = self.q(tokens).reshape(b, n, h, dh)
		k = self.k(tiles).reshape(b, -1, h, dh)
		v = self.v(tiles).reshape(b, -1, h, dh)
		logits = torch.einsum("bnhd,bmhd->bhnm", q, k) / math.sqrt(dh)
		bucket = distance_buckets(tile_positions[None, None, :, :] - positions[:, :, None, :])  # [B, N, M]
		logits = logits + F.embedding(bucket, self.distance.t()).permute(0, 3, 1, 2)  # (an embedding: Sides.forward)
		if reach is not None:
			logits = logits + reach.permute(0, 3, 1, 2)
		att = logits.masked_fill(~valid[:, None, None, :], -1e4).softmax(dim=-1)
		read = torch.einsum("bhnm,bmhd->bnhd", att, v).reshape(b, n, d)
		return self.out(read) * placed.unsqueeze(-1).float()


class Foresight(nn.Module):
	"""What the next action brings (model docstring): per (hero, window tile) for MOVE, per question option, per hero, and
	for the turn, a small hidden state and its predictions. The predictions are for their losses (foresight.py) and
	nothing else; the hidden states feed think_again."""

	def __init__(self, dim, pair_dim):
		super().__init__()
		self.move = SwiGLU(pair_dim, pair_dim, FORESIGHT_HIDDEN)
		self.answer = SwiGLU(2 * dim, dim, FORESIGHT_HIDDEN)
		self.hero = SwiGLU(dim, dim // 2, FORESIGHT_HIDDEN)
		self.turn = SwiGLU(dim, dim // 2, FORESIGHT_HIDDEN)
		self.move_out = nn.Linear(FORESIGHT_HIDDEN, ACTION_TARGETS)
		self.answer_out = nn.Linear(FORESIGHT_HIDDEN, ACTION_TARGETS)
		self.hero_out = nn.Linear(FORESIGHT_HIDDEN, HERO_TARGETS)
		self.turn_out = nn.Linear(FORESIGHT_HIDDEN, TURN_TARGETS)
		# Nullkiller2's plan (CampaignModelV2.plan_of): per (hero, TRAVEL area), and each hero's "no task". Its outputs start
		# at 0: uniform, its loss ln(AREAS + 1), no gradient to the shared layers until it predicts (at random weights on
		# a trained model's pairs it started at 14.5 nats, its shared gradient twice the policy's, 2026-11-02)
		self.plan = SwiGLU(pair_dim, pair_dim, FORESIGHT_HIDDEN)
		self.plan_out = zero_output(nn.Linear(FORESIGHT_HIDDEN, 1))
		self.plan_none = zero_output(nn.Linear(dim, 1))
		# What a wandering monster does when a hero steps on it or into its zone, on every window tile (foresight.py
		# "move_tiles"; the user, 2026-11-04: the engine's answer, never ours): no monster there / it attacks / it offers to
		# join or flees; and on a battle, P(we win it) and the share of the hero's stake it took ("move_win", "move_stake").
		# At 0 to begin with; read by MOVE (detached)
		self.tiles = zero_output(nn.Linear(FORESIGHT_HIDDEN, 3))
		self.outcome = zero_output(nn.Linear(FORESIGHT_HIDDEN, 2))


class Memory(nn.Module):
	"""The model's memory (constants.MEMORY_SLOTS slots of MEMORY_WIDTH, -1..1): what one player's decisions carry
	forward, the model's own output passed back by AlphaMapAI (x["mem"]; x["mem_float"] when a trainer unrolls it).
	  read   the trunk's tokens attend to the slots before think's blocks: a residual whose output starts at 0, so the
	         model is at first what it was without memory
	  write  after think_again each slot attends to the trunk's tokens and moves toward what it read by a gated update
	         (a GRU's: gate x tanh candidate), so a slot stays in -1..1 and keeps what is not overwritten
	The slots' own embeddings and the write start small but not at 0: with a zero read and a zero memory both, neither
	would ever get a gradient."""

	def __init__(self, dim, heads, slots=c.MEMORY_SLOTS, width=c.MEMORY_WIDTH):
		super().__init__()
		self.heads = heads
		self.slot = nn.Parameter(torch.randn(slots, width) * 0.1)
		self.query, self.key, self.value = nn.Linear(dim, dim), nn.Linear(width, dim), nn.Linear(width, dim)
		self.read_out = zero_output(nn.Linear(dim, dim))
		self.write_query, self.write_key, self.write_value = nn.Linear(width, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
		self.gate = nn.Linear(dim + width, width)
		nn.init.constant_(self.gate.bias, -2.0)  # (a slot moves ~12% of the way a decision at first)
		self.candidate = nn.Linear(dim + width, width)

	def attend(self, q, k, v, mask=None):
		"""[B, n, d] queries over [B, m, d] keys and values; mask [B, m] (True: may be attended to)."""
		b, n, d = q.shape
		heads = lambda t: t.reshape(b, t.shape[1], self.heads, d // self.heads).transpose(1, 2)  # noqa: E731
		out = F.scaled_dot_product_attention(heads(q), heads(k), heads(v), attn_mask=None if mask is None else mask[:, None, None, :])
		return out.transpose(1, 2).reshape(b, n, d)

	@staticmethod
	def of(x):
		"""The memory a decision starts from [B, slots, width]: the trainer's unrolled one, else the one passed in."""
		return x["mem_float"] if "mem_float" in x else x["mem"].float() / c.MEMORY_SCALE

	def read(self, tokens, memory):
		slots = memory + self.slot
		return tokens + self.read_out(self.attend(self.query(tokens), self.key(slots), self.value(slots)))

	def write(self, tokens, valid, memory):
		seen = self.attend(self.write_query(memory + self.slot), self.write_key(tokens), self.write_value(tokens), valid)
		both = torch.cat([seen, memory], dim=-1)
		gate = torch.sigmoid(self.gate(both))
		return (1 - gate) * memory + gate * torch.tanh(self.candidate(both))


class CampaignModelV2(nn.Module):
	ARCHITECTURE = "v2"
	plan_outputs = False  # fs_plan too (plan_of): the trainer's model sets it (train.ppo_update); a server's does not

	def load_state_dict(self, state_dict, strict=True, assign=False):
		"""A checkpoint from before a parameter was added (model.ADDED) loads with it at 0: the model as it was; one from
		before 2026-11-04's MOVE features with move_pair's first layer moved to its inputs now (migrate_move_inputs)."""
		return super().load_state_dict(with_added(migrate_move_inputs(state_dict), self.state_dict()), strict=strict, assign=assign)

	def __init__(self, dim=64, cell_dim=16, heads=4, layers=2, share_target="share"):
		"""cell_dim: the map tiles' width (the U-Net's at full resolution; twice and four times it at half and a quarter):
		16, as 32 cost 3x v1's inference on the CPU (2026-10-30: 27 ms a decision against 9 ms, the U-Net most of it)."""
		super().__init__()
		d, dc = dim, cell_dim
		self.dim, self.cell_dim, self.heads = d, dc, heads
		self.share_target = share_target  # (config.share_target: what V2, V5, V7 forecast)
		# Every hero slot's window, absent heroes' masked out (windows_of): fixed shapes and no sync with the host, for a
		# CUDA graph (inference.py sets it on CUDA); the same outputs, more windows computed
		self.fixed_windows = False
		# With fixed windows, this many a batch instead of every slot (windows_of): our heroes' first. Set only around a
		# graph's recording and the eager step it is checked against (graphs.window_budget), by a caller that counted the
		# batch's heroes: a batch with more of them would lose some of its heroes' windows
		self.window_budget = None
		dp = d // 2  # (hero, tile) and (hero, area) pairs
		self.pair_dim = dp
		# domain_knowledge
		self.features = CustomFeatures()
		# encode
		self.encode_global = E.GlobalEncoder(d)
		self.encode_hero = E.HeroEncoder(d)
		self.encode_town = E.TownEncoder(d)
		self.encode_tiles = E.TileEncoder(dc)
		self.encode_options = E.OptionEncoder(d)
		self.encode_other = E.ArmyEncoder(d)
		self.reach = E.Tokenizer(5 + E.AMOUNT_FEATURES + 1, dc, embedded=False)  # a hero's reach to its window's tiles
		self.travel_days = E.Tokenizer(E.AMOUNT_FEATURES + 1, dp, embedded=False)  # its days to each TRAVEL area
		# take_sides
		self.sides, self.tile_sides = Sides(d), Sides(dc)
		self.player = small_embedding(P, d)
		self.players = E.Tokenizer(2 * d, d)
		# join
		self.join_global = E.Tokenizer(d + ft_phi_features() + 3, d, embedded=False)
		self.join_hero = E.Tokenizer(d + HERO_KNOWN, d, embedded=False)
		self.join_town = E.Tokenizer(d + TOWN_KNOWN, d, embedded=False)
		self.join_tiles = nn.Linear(TILE_KNOWN, dc)
		# see
		self.stamp_hero, self.stamp_town = nn.Linear(d, dc), nn.Linear(d, dc)
		self.unet = MapUNet(dc)
		self.read_tiles = TileReader(d, dc, heads)
		self.area = nn.Linear(2 * dc, d)
		# think
		self.pair_bias = PairBias(heads)
		self.relation_bias = nn.Parameter(torch.zeros(heads, 6))
		self.memory = Memory(d, heads)  # (read before the blocks, written after think_again)
		self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
		self.norm = RMSNorm(d)
		# foresee: a (hero, tile) pair is its hero's, its tile's and its reach's projections summed, and the fight stepping
		# there starts (fight_features: its log strength ratio, the user's, 2026-11-03), at 0 to begin with
		self.pair_hero, self.pair_tile, self.pair_reach = nn.Linear(d, dp), nn.Linear(dc, dp), nn.Linear(dc, dp)
		self.pair_fight = zero_output(nn.Linear(FIGHT_FEATURES, dp))
		self.pair_strength = zero_output(nn.Linear(MOVE_STRENGTH + 1, dp))  # (2026-11-04: the four estimates x two sides, hopeless)
		self.foresight = Foresight(d, dp)
		# think_again
		self.again_global = E.Tokenizer(d + FORESIGHT_HIDDEN + ft_phi_features() + 3, d, embedded=False)
		self.again_hero = E.Tokenizer(d + FORESIGHT_HIDDEN + HERO_KNOWN, d, embedded=False)
		self.again_option = E.Tokenizer(d + FORESIGHT_HIDDEN, d, embedded=False)
		self.again_block = Block(d, heads)
		self.again_norm = RMSNorm(d)
		self.move_pair = SwiGLU(dp + FORESIGHT_HIDDEN + MOVE_KNOWN + MIX_HIDDEN, dp, dp)
		# The user's mixer (2026-11-04): MOVE's known features' second-order effects, a small hidden layer onto the logit and
		# into move_pair, beside a direct linear path. Its hidden layer's output starts at 0, the direct path as Nullkiller2's
		# rule (it never starts a fight it cannot win: HOPELESS_LOGIT x hopeless), both learning at the run's rate
		self.move_mix = nn.Linear(MOVE_KNOWN, MIX_HIDDEN)
		self.move_mix_out = zero_output(nn.Linear(MIX_HIDDEN, 1))
		self.move_mix_direct = nn.Linear(MOVE_KNOWN, 1)
		with torch.no_grad():
			self.move_mix_direct.weight.zero_()
			self.move_mix_direct.weight[0, HOPELESS] = HOPELESS_LOGIT
			self.move_mix_direct.bias.zero_()
		self.register_buffer("move_kept", torch.tensor(MOVE_KEPT), persistent=False)
		# The fight's features straight onto MOVE's learned logit too, at 0 to begin with: a few weights a fight's outcome
		# teaches directly (joint7's near-empty heroes walked into hopeless fights at ~1% a tile among ~110: the pairs'
		# learned part told those tiles from the rest by less than a logit)
		self.move_fight = zero_output(nn.Linear(FIGHT_FEATURES, 1))
		self.travel_hero, self.travel_area = nn.Linear(d, dp), nn.Linear(d, dp)
		self.travel_pair = SwiGLU(dp + 1, dp, dp)
		# decide: each table's learned part; every type's bias from the priors
		# Every head's last layer starts at 0 (zero_output): the first policy is the domain priors' exactly, the known-sane
		# start, and the learned parts grow from it
		self.move_out, self.cast_cell_out, self.travel_out = zero_output(nn.Linear(dp, 1)), zero_output(nn.Linear(dp, 1)), zero_output(nn.Linear(dp, 1))
		self.act_move, self.act_travel = zero_output(nn.Linear(d, 1)), zero_output(nn.Linear(d, 1))
		self.building = small_embedding(c.N_BUILDINGS, d)
		self.build_q = zero_output(nn.Linear(d, d))
		self.build_bias = nn.Parameter(torch.zeros(c.N_BUILDINGS))
		self.nk2_build, self.nk2_upgrade, self.nk2_trade = zero_output(nn.Linear(4, 1)), zero_output(nn.Linear(2, 1)), zero_output(nn.Linear(2, 1))
		self.recruit_head = nn.Sequential(nn.Linear(2 * d + 9, d), nn.GELU(), zero_output(nn.Linear(d, c.N_RECRUIT_AMOUNTS * c.N_RECRUIT_TARGETS)))
		self.hire_q, self.hire_k = zero_output(nn.Linear(d, d)), nn.Linear(d, d)
		self.hire_worth = zero_output(nn.Linear(4, 1))
		self.army_src, self.army_dst = nn.Linear(2 * d, d), nn.Linear(d, d)
		self.army_out = nn.Sequential(nn.GELU(), zero_output(nn.Linear(d, c.N_ARMY_AMOUNTS)))
		self.upgrade_out = nn.Sequential(nn.GELU(), zero_output(nn.Linear(d, 1)))
		self.trade_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), zero_output(nn.Linear(d, R * R * c.N_TRADE_AMOUNTS)))
		self.spell = small_embedding(c.N_ADVENTURE_SPELLS, d)
		self.cast_q = zero_output(nn.Linear(d, d))
		self.buy_item = small_embedding(c.N_BUY_ITEMS, d)
		self.buy_q = zero_output(nn.Linear(d, d))
		self.dismiss_out = zero_output(nn.Linear(d, 1))
		self.option_q = zero_output(nn.Linear(2 * d, d))
		self.end_out = zero_output(nn.Linear(d, 1))
		self.type_bias = nn.Parameter(priors.action_types().clone())
		self.value_hidden = nn.Sequential(nn.Linear(3 * d + ft_phi_features(), d), nn.GELU())
		self.value_heads = zero_output(nn.Linear(d, c.N_VALUE_OFFSETS))
		self.growth_heads = zero_output(nn.Linear(d, c.N_VALUE_OFFSETS))
		self.end_head = zero_output(nn.Linear(d, 1))
		self.register_buffer("tile_positions", tile_positions(), persistent=False)
		# The adventure spells the model may cast: not the ones that only show the map (constants.INFO_SPELLS)
		self.register_buffer("castable", torch.tensor([s not in c.INFO_SPELLS for s in range(c.N_ADVENTURE_SPELLS)]), persistent=False)

	def prior_corrections(self):
		"""The parameters that correct a fixed prior (model.CustomFeatures): weight decay pulls them back to it."""
		return self.features.corrections()

	# ---- the whole model ----

	def forward(self, x):
		known = self.domain_knowledge(x)
		state = self.encode(x)
		state = self.take_sides(state)
		state = self.join(state, known)
		state = self.see(state, known)
		state = self.think(state)
		foresight = self.foresee(state, known)
		state = self.think_again(state, foresight, known)
		out = {**self.decide(state, known), **foresight["outputs"], "memory": state.memory_out}
		if self.plan_outputs:
			out["fs_plan"] = self.plan_of(state)
		return out

	def plan_of(self, state):
		"""Nullkiller2's plan as the foresight predicts it [B, HO, AREAS + 1] (foresight.py "plan"): each of our heroes'
		logits over the TRAVEL areas it is sent to (from its TRAVEL pairs: the hero, the area, the days there, the area's
		prior) and over none (no task for it). For the loss only: the AI reads none of it, nothing else depends on it."""
		f = self.foresight
		areas = f.plan_out(f.plan(state.travel_pairs)).squeeze(-1)
		return torch.cat([areas, f.plan_none(state.heroes[:, :HO])], dim=-1)

	# ---- domain_knowledge ----

	def domain_knowledge(self, x):
		"""model.CustomFeatures' facts (Φ, armies, fights, each window tile's value and prior, TRAVEL's prior, the action
		priors), for the windows of our heroes there, and the references sizes are relative to."""
		windows, n, real = windows_of(x, fixed=self.fixed_windows, budget=self.window_budget)
		known = self.features(x, windows)
		known.windows, known.n, known.real = windows, n, real
		known.window_tiles = window_tile_index(x, windows)  # [K, W, W] each window tile's map tile (-1 off the grid)
		ours = x["h_present"][:, :HO] & (x["h_owner"][:, :HO] == c.OWNER_SELF)
		armies = known.hero_gold[:, :HO] * ours
		rivals = known.hero_gold[:, HO:] * (x["h_present"][:, HO:] & (x["h_owner"][:, HO:] > c.OWNER_SELF))
		known.reference = {  # [B] each, floored so that early ratios stay bounded
			"phi": known.phi_total.clamp(min=1000.0),
			"main_army": armies.amax(dim=1).clamp(min=500.0),
			"our_army": armies.sum(dim=1).clamp(min=500.0),
			"rival_army": rivals.amax(dim=1).clamp(min=500.0),
		}
		return known

	# ---- encode: raw inputs only ----

	def encode(self, x):
		state = State(x)
		state.glob = self.encode_global(x)
		state.heroes, state.hero_stacks = self.encode_hero(E.hero_fields(x))
		state.towns, state.town_stacks, state.town_levels = self.encode_town(x)
		state.tavern, _ = self.encode_hero(E.tavern_fields(x))
		state.other_stacks, state.other = self.encode_other(x["x_creature"], x["x_count"], torch.ones_like(x["x_creature"][:, 0], dtype=torch.bool))
		state.options = self.encode_options(x)
		state.tiles = self.encode_tiles(x)  # [B, L, S, S, dc]
		return state

	# ---- take_sides: ownership transforms every vector ----

	def take_sides(self, state):
		x = state.x
		state.heroes = self.sides(state.heroes, x["h_owner"]) * x["h_present"].unsqueeze(-1)
		state.towns = self.sides(state.towns, x["t_owner"]) * x["t_present"].unsqueeze(-1)
		on_map = ft.bit(x["tile_flags"], c.TILE_ON_MAP).unsqueeze(-1)
		state.tiles = self.tile_sides(state.tiles, x["tile_owner"].clamp(min=0)) * on_map
		# A token per player: its heroes' and towns' we see, summed and at their max, and its own embedding
		owners = torch.cat([torch.where(x["h_present"], x["h_owner"], torch.zeros_like(x["h_owner"])), torch.where(x["t_present"], x["t_owner"], torch.zeros_like(x["t_owner"]))], dim=1)
		of = (owners.unsqueeze(-1) == (c.OWNER_SELF + torch.arange(P, device=owners.device))).float()  # [B, H + T, P]
		entities = torch.cat([state.heroes, state.towns], dim=1)
		total = torch.einsum("bnp,bnd->bpd", of, entities)
		top = (entities.unsqueeze(2) + (of.unsqueeze(-1) - 1) * 1e4).amax(dim=1) * (of.sum(dim=1, keepdim=True) > 0).transpose(1, 2).float()
		state.player_valid = (of.sum(dim=1) > 0) | (torch.arange(P, device=owners.device) == 0)
		state.players = self.players(self.player.weight.expand(owners.shape[0], P, -1), torch.cat([total, top], dim=-1)) * state.player_valid.unsqueeze(-1)
		return state

	# ---- join: the known facts onto what they describe ----

	def join(self, state, known):
		x = state.x
		state.glob = self.join_global(None, torch.cat([state.glob, global_known(known)], dim=-1))
		state.heroes = self.join_hero(None, torch.cat([state.heroes, hero_known(x, known)], dim=-1)) * x["h_present"].unsqueeze(-1)
		state.towns = self.join_town(None, torch.cat([state.towns, town_known(x, known)], dim=-1)) * x["t_present"].unsqueeze(-1)
		state.tiles = state.tiles + self.join_tiles(tile_known(x, self.features, known))
		return state

	# ---- see: the map ----

	def see(self, state, known):
		x = state.x
		b = x["g_day"].shape[0]
		dc = self.cell_dim
		cells = state.tiles.reshape(b * L * S * S, dc)
		for vectors, positions, present, stamp in ((state.heroes, x["h_pos"], x["h_present"], self.stamp_hero), (state.towns, x["t_pos"], x["t_present"], self.stamp_town)):
			index, inside = map_index(positions, b)
			keep = (present & inside).reshape(-1)
			# (every row, the others' adding zeros: no nonzero, whose size the host must learn, for a CUDA graph)
			stamped = torch.where(keep.unsqueeze(-1), stamp(vectors.reshape(-1, vectors.shape[-1])), 0.0)
			cells = cells.index_add(0, torch.where(keep, index.reshape(-1), 0), stamped.to(cells.dtype))
		on_map = ft.bit(x["tile_flags"], c.TILE_ON_MAP).reshape(b * L, 1, S, S).float()
		grid, half = self.unet(cells.reshape(b, L, S, S, dc).permute(0, 1, 4, 2, 3).reshape(b * L, dc, S, S), on_map)
		state.tiles = grid.reshape(b, L, dc, S, S).permute(0, 1, 3, 4, 2)  # [B, L, S, S, dc]
		flat = state.tiles.reshape(b, L * S * S, dc)
		valid = on_map.reshape(b, L * S * S) > 0
		# Our heroes' reach on the map, as the tile reader's bias per head: each window's tiles scattered to theirs. The
		# windows are computed padded (windows_of: fixed shapes on Apple's GPU, which keeps a graph per shape); the padding
		# rows, copies of the first window, are left out of the scatter (they would count its reach twice)
		heads = self.heads
		reach = torch.zeros(b * HO * L * S * S, heads, device=flat.device, dtype=flat.dtype)
		if known.n > 0:
			k = known.windows.shape[0]
			state.window_reach = self.reach(None, reach_features(x, known.windows))  # [K, W, W, dc]
			tiles = known.window_tiles.reshape(k, W * W)
			index = (known.windows[:, None] * (L * S * S) + tiles.clamp(min=0)).reshape(-1)
			inside = ((tiles >= 0) & known.real[:, None]).reshape(-1)
			values = torch.where(inside.unsqueeze(-1), self.read_tiles.reach(state.window_reach).reshape(-1, heads), 0.0)
			reach = reach.index_add(0, torch.where(inside, index, 0), values.to(reach.dtype))
		reach = torch.cat([reach.reshape(b, HO, L * S * S, heads), torch.zeros(b, H - HO, L * S * S, heads, device=flat.device, dtype=flat.dtype)], dim=1)
		state.heroes = state.heroes + self.read_tiles(state.heroes, x["h_pos"], x["h_present"], flat, self.tile_positions, valid, reach)
		state.towns = state.towns + self.read_tiles(state.towns, x["t_pos"], x["t_present"], flat, self.tile_positions, valid)
		state.areas = self.area(half.reshape(b, L, 2 * dc, TG * TG).permute(0, 1, 3, 2).reshape(b, AREAS, 2 * dc))
		return state

	# ---- think ----

	def tokens(self, state):
		"""(the tokens [B, N, d], which are there [B, N], their tile positions [B, N, 3] and whether placed, their owner
		codes [B, N]) in the trunk's order: global, players, heroes, towns, the other army, options, the tavern."""
		x = state.x
		b = x["g_day"].shape[0]
		one = torch.ones(b, 1, dtype=torch.bool, device=state.glob.device)
		other_present = (x["x_creature"] >= 0).any(dim=-1, keepdim=True)
		tokens = torch.cat([state.glob.unsqueeze(1), state.players, state.heroes, state.towns, state.other.unsqueeze(1), state.options, state.tavern], dim=1)
		valid = torch.cat([one, state.player_valid, x["h_present"], x["t_present"], other_present, x["option_mask"], x["g_tavern"] >= 0], dim=1)
		none_pos = torch.zeros(b, 1, 3, dtype=x["h_pos"].dtype, device=x["h_pos"].device)
		positions = torch.cat([none_pos, none_pos.expand(b, P, 3), x["h_pos"], x["t_pos"], none_pos, x["o_pos"], none_pos.expand(b, c.N_TAVERN, 3)], dim=1).float()
		placed = torch.cat([~one, torch.zeros_like(state.player_valid), x["h_present"], x["t_present"], ~one, x["option_mask"] & (x["o_kind"] == c.OptionKind.OBJECT), torch.zeros_like(x["g_tavern"] >= 0)], dim=1)
		player_codes = (c.OWNER_SELF + torch.arange(P, device=one.device)).expand(b, P)
		zeros = lambda n: torch.zeros(b, n, dtype=torch.long, device=one.device)  # noqa: E731
		owners = torch.cat([zeros(1), player_codes, torch.where(x["h_present"], x["h_owner"], zeros(H)), torch.where(x["t_present"], x["t_owner"], zeros(T)), zeros(1), zeros(c.N_OPTIONS), zeros(c.N_TAVERN)], dim=1)
		kinds = torch.cat([zeros(1), torch.ones(b, P, dtype=torch.long, device=one.device), torch.ones(b, H + T, dtype=torch.long, device=one.device), zeros(1 + c.N_OPTIONS + c.N_TAVERN)], dim=1)
		return tokens, valid, positions, placed, owners, kinds

	def attention_bias(self, positions, placed, owners, kinds):
		"""[B, heads, N, N]: by tile offset (PairBias) and by the owners' relation (relation_bias): 0 one of them owns
		nothing (global, options, ...), 1 both ours, 2 one rival with itself, 3 ours and a rival's, 4 two rivals, 5 one of
		them nobody's (a neutral town)."""
		owned = kinds > 0
		a, b_ = owners[:, :, None], owners[:, None, :]
		ours_a, ours_b = a == c.OWNER_SELF, b_ == c.OWNER_SELF
		rival_a, rival_b = a > c.OWNER_SELF, b_ > c.OWNER_SELF
		relation = torch.where(ours_a & ours_b, 1, torch.where(rival_a & rival_b & (a == b_), 2, torch.where((ours_a & rival_b) | (rival_a & ours_b), 3, torch.where(rival_a & rival_b, 4, 5))))
		relation = torch.where(owned[:, :, None] & owned[:, None, :], relation, torch.zeros_like(relation))
		return self.pair_bias(positions, placed) + F.embedding(relation, self.relation_bias.t()).permute(0, 3, 1, 2)

	def think(self, state):
		tokens, valid, positions, placed, owners, kinds = self.tokens(state)
		state.valid, state.bias = valid, self.attention_bias(positions, placed, owners, kinds)
		state.mem = Memory.of(state.x)
		tokens = self.memory.read(tokens, state.mem)
		for block in self.blocks:
			tokens = block(tokens, valid, state.bias)
		self.split(state, self.norm(tokens))
		return state

	@staticmethod
	def split(state, tokens):
		i = 0
		for name, n in (("glob", 1), ("players", P), ("heroes", H), ("towns", T), ("other", 1), ("options", c.N_OPTIONS), ("tavern", c.N_TAVERN)):
			part = tokens[:, i : i + n]
			setattr(state, name, part.squeeze(1) if name in ("glob", "other") else part)
			i += n

	# ---- foresee ----

	def pairs(self, state, known):
		"""Each window's (hero, tile) pairs [K, W, W, dp] (K: the windows padded, windows_of): the hero's, its tile's and
		its reach's projections summed."""
		b = state.x["g_day"].shape[0]
		k = known.windows.shape[0]
		heroes = state.heroes[:, :HO].reshape(b * HO, -1).index_select(0, known.windows)
		flat = state.tiles.reshape(b * L * S * S, self.cell_dim)
		decision = torch.div(known.windows, HO, rounding_mode="floor")
		index = known.window_tiles
		cells = torch.where((index >= 0).unsqueeze(-1), flat.index_select(0, (decision[:, None, None] * (L * S * S) + index.clamp(min=0)).reshape(-1)).reshape(k, W, W, -1), torch.zeros(k, W, W, self.cell_dim, device=flat.device, dtype=flat.dtype))
		strength = torch.cat([known.move_strength, known.move_relative[..., :1]], dim=-1)  # (the estimates and hopeless)
		return self.pair_hero(heroes)[:, None, None, :] + self.pair_tile(cells) + self.pair_reach(state.window_reach) + self.pair_fight(fight_features(known)) + self.pair_strength(strength)

	def foresee(self, state, known):
		x = state.x
		b = x["g_day"].shape[0]
		f = self.foresight
		hidden = {"turn": f.turn(state.glob), "hero": f.hero(state.heroes)}
		about = torch.gather(state.heroes, 1, x["d_hero"].clamp(min=0).reshape(b, 1, 1).expand(b, 1, self.dim)) * (x["d_hero"] >= 0).reshape(b, 1, 1).float()
		hidden["answer"] = f.answer(torch.cat([state.options, about.expand(-1, c.N_OPTIONS, -1)], dim=-1))
		device = state.glob.device
		outputs = {
			"fs_turn": f.turn_out(hidden["turn"]), "fs_hero": f.hero_out(hidden["hero"]), "fs_answer": f.answer_out(hidden["answer"]),
			"fs_move": torch.zeros(b, HO, W, W, ACTION_TARGETS, device=device),
			# the battle head's (foresight.py "move_tiles"), its labels' tiles: the legal ones, a wandering monster's or its zone's
			"fs_tiles": torch.zeros(b, HO, W, W, 3, device=device), "fs_outcome": torch.zeros(b, HO, W, W, 2, device=device),
			"fs_stack_tile": torch.zeros(b, HO, W, W, device=device), "fs_move_legal": x["move_mask"].float(),
		}
		foreseen = None
		if known.n > 0:
			state.pairs = self.pairs(state, known)
			hidden["move"] = f.move(state.pairs)
			tiles, outcome = f.tiles(hidden["move"]), f.outcome(hidden["move"])
			outputs["fs_move"] = dense(real_only(f.move_out(hidden["move"]), known), known.windows, known.n, b)
			outputs["fs_tiles"] = dense(real_only(tiles, known), known.windows, known.n, b)
			outputs["fs_outcome"] = dense(real_only(outcome, known), known.windows, known.n, b)
			outputs["fs_stack_tile"] = dense(real_only(known.tile_stacks, known), known.windows, known.n, b)
			# What MOVE reads of them (MOVE_FORESEEN: P(a monster here attacks), P(win), the stake lost), detached: its loss shapes
			# the predictions, never the policy's
			foreseen = torch.stack([tiles.softmax(dim=-1)[..., 1], torch.sigmoid(outcome[..., 0]), torch.sigmoid(outcome[..., 1])], dim=-1).detach()
		return {"hidden": hidden, "outputs": outputs, "foreseen": foreseen}

	# ---- think_again ----

	def think_again(self, state, foresight, known):
		x = state.x
		hidden = foresight["hidden"]
		state.glob = state.glob + self.again_global(None, torch.cat([state.glob, hidden["turn"], global_known(known)], dim=-1))
		state.heroes = state.heroes + self.again_hero(None, torch.cat([state.heroes, hidden["hero"], hero_known(x, known)], dim=-1)) * x["h_present"].unsqueeze(-1)
		state.options = state.options + self.again_option(None, torch.cat([state.options, hidden["answer"]], dim=-1)) * x["option_mask"].unsqueeze(-1)
		tokens, valid, *_ = self.tokens(state)
		tokens = self.again_norm(self.again_block(tokens, valid, state.bias))
		self.split(state, tokens)
		state.memory_out = self.memory.write(tokens, valid, state.mem)
		if known.n > 0:
			pairs = self.pairs(state, known)  # (with the heroes as they are now)
			move_known = torch.cat([known.move_features.index_select(-1, self.move_kept), known.move_strength, known.move_relative, foresight["foreseen"]], dim=-1)  # [K, W, W, MOVE_KNOWN]
			mixed = F.silu(self.move_mix(move_known))
			state.move_mix = self.move_mix_out(mixed) + self.move_mix_direct(move_known)
			state.move_pairs = self.move_pair(torch.cat([pairs, hidden["move"], move_known, mixed], dim=-1))
		b = x["g_day"].shape[0]
		days = x["travel_days"][:, :HO].reshape(b, HO, AREAS)
		travel_in = torch.cat([E.amount(days.clamp(min=0).float() / 100, "days"), (days >= 0).float().unsqueeze(-1)], dim=-1)
		travel = self.travel_hero(state.heroes[:, :HO]).unsqueeze(2) + self.travel_area(state.areas).unsqueeze(1) + self.travel_days(None, travel_in)
		state.travel_pairs = self.travel_pair(torch.cat([travel, torch.tanh(known.travel_prior.reshape(b, HO, AREAS) / 4).unsqueeze(-1)], dim=-1))
		return state

	# ---- decide: every logit a match, one softmax ----

	def decide(self, state, known):
		x = state.x
		b, d = x["g_day"].shape[0], self.dim
		own_heroes, own_towns = state.heroes[:, :HO], state.towns[:, :TO]
		out, learned, prior, offers = {}, {}, {}, {}  # each table's learned part and its domain prior apart
		A_ = c.ActionType
		# MOVE and the targeted spell's tile: each window's pairs, + the hero's act-now score, + the tile's prior
		move = cast_cell = move_prior = torch.full((b, HO, W, W), 0.0, device=state.glob.device)
		if known.n > 0:
			move = dense(real_only(self.move_out(state.move_pairs).squeeze(-1) + self.move_fight(fight_features(known)).squeeze(-1) + state.move_mix.squeeze(-1), known), known.windows, known.n, b)
			move_prior = dense(real_only(known.tile_prior, known), known.windows, known.n, b)
			cast_cell = dense(real_only(self.cast_cell_out(state.move_pairs).squeeze(-1), known), known.windows, known.n, b)
		learned[A_.MOVE], prior[A_.MOVE] = move + self.act_move(own_heroes).reshape(b, HO, 1, 1), move_prior
		offers[A_.MOVE] = ft.offer(move_prior, x["move_mask"])
		out["cast_cell_logp"] = masked_log_softmax(cast_cell.reshape(b, HO, W * W), x["cast_cell_mask"].reshape(b, HO, W * W)).reshape(b, HO, W, W)
		# TRAVEL: (hero, area) pairs, + the area's prior (its value discounted by the days to it)
		learned[A_.TRAVEL] = (self.travel_out(state.travel_pairs).squeeze(-1) + self.act_travel(own_heroes)).reshape(b, HO, L, TG, TG)
		prior[A_.TRAVEL] = known.travel_prior.reshape(b, HO, L, TG, TG)
		offers[A_.TRAVEL] = ft.offer(prior[A_.TRAVEL], x["travel_mask"])
		# BUILD: the town's query against each building
		learned[A_.BUILD], prior[A_.BUILD] = self.build_q(own_towns) @ self.building.weight.t() / math.sqrt(d) + self.build_bias, self.nk2_build(known.build_rewards).squeeze(-1)
		# RECRUIT: the town and each creature level's token, the amounts; Nullkiller2's prior
		available = x["t_available"][:, :TO].clamp(min=0).float()
		gold = x["g_resources"][:, 6].clamp(min=0).float()
		recruit_in = torch.cat(
			[
				own_towns.unsqueeze(2).expand(b, TO, c.N_LEVELS, d), state.town_levels[:, :TO], scales(available, (5, 20, 80)),
				scales(gold, (1000, 5000, 20000)).reshape(b, 1, 1, 3).expand(b, TO, c.N_LEVELS, 3),
				scales(self.features.creature_cost[vocab_index(x["t_available_creature"][:, :TO], c.CREATURE_VOCAB)] * available, (1000, 5000, 20000)),
			],
			dim=-1,
		)
		learned[A_.RECRUIT] = self.recruit_head(recruit_in).reshape(b, TO, c.N_LEVELS, c.N_RECRUIT_AMOUNTS, c.N_RECRUIT_TARGETS)
		prior[A_.RECRUIT] = self.features.recruit_prior(known.town_danger).expand_as(learned[A_.RECRUIT])
		skill_weight, upgrade_weight, book_weight, pause_weight = self.features.nk2_weights()
		home_weight, defend_weight, ferry_weight, hire_weight, trade_weight = self.features.pull_weights()
		offers[A_.RECRUIT] = -pause_weight * known.recruit_pause  # (no army bought while saving for a hall: Nullkiller2's habit)
		# HIRE: the town's query against each hero for hire (the same hero space as ours) + what the hire does to Φ
		worth = torch.cat([scales(known.tavern_worth.clamp(min=0), (1000, 3000, 10000)), torch.tanh(known.hire_gain / 2000).unsqueeze(-1)], dim=-1)
		learned[A_.HIRE] = torch.einsum("btd,bsd->bts", self.hire_q(own_towns), self.hire_k(state.tavern)) / math.sqrt(d) + self.hire_worth(worth).squeeze(-1).unsqueeze(1)
		prior[A_.HIRE] = ft.value_logit(known.hire_gain).unsqueeze(1).expand(b, TO, c.N_TAVERN)
		offers[A_.HIRE] = best_legal(prior[A_.HIRE], x["hire_mask"]) + hire_weight * known.hire_want
		# ARMY and UPGRADE: each stack (its holder's token with it) against each holder
		holders = torch.cat([own_heroes, own_towns, state.other.unsqueeze(1)], dim=1)
		holder_stacks = torch.cat([state.hero_stacks[:, :HO], state.town_stacks[:, :TO], state.other_stacks.unsqueeze(1)], dim=1)
		src = self.army_src(torch.cat([holders.unsqueeze(2).expand(b, HD, A, d), holder_stacks], dim=-1))
		dst = self.army_dst(holders)
		learned[A_.ARMY] = self.army_out(src[:, :, :, None, :] + dst[:, None, None, :, :])
		prior[A_.ARMY] = self.features.army_prior(known.hero_strength[:, :HO]).expand_as(learned[A_.ARMY])
		offers[A_.ARMY] = best_legal(prior[A_.ARMY], x["army_mask"])  # (below 0 when every legal move leaves troops in a mine)
		learned[A_.UPGRADE] = self.upgrade_out(src).squeeze(-1) + self.nk2_upgrade(known.upgrade_gains).squeeze(-1)
		prior[A_.UPGRADE] = upgrade_weight * (known.upgrade_gains[..., 0] > 0).float()
		offers[A_.UPGRADE] = best_legal(prior[A_.UPGRADE], x["upgrade_mask"])
		# TRADE (from the global token), CAST, BUY, DISMISS (each hero's)
		needs = torch.stack([known.trade_needs[:, None, :, 0].expand(b, R, R), known.trade_needs[:, :, None, 1].expand(b, R, R)], dim=-1)
		learned[A_.TRADE] = self.trade_head(state.glob).reshape(b, R, R, c.N_TRADE_AMOUNTS) + self.nk2_trade(needs)
		prior[A_.TRADE] = (trade_weight * ((needs[..., 0] > 0) & (needs[..., 1] > 0)).float()).unsqueeze(-1).expand_as(learned[A_.TRADE])
		offers[A_.TRADE] = best_legal(prior[A_.TRADE], x["trade_mask"])
		learned[A_.CAST] = self.cast_q(own_heroes) @ self.spell.weight.t() / math.sqrt(d)
		prior[A_.CAST] = torch.zeros_like(learned[A_.CAST])
		book = (torch.arange(c.N_BUY_ITEMS, device=gold.device) == c.BUY_ARTIFACTS.index(ft.SPELLBOOK)).float()
		learned[A_.BUY] = self.buy_q(own_heroes) @ self.buy_item.weight.t() / math.sqrt(d)
		prior[A_.BUY] = (book_weight * known.spellbook_wanted.float().unsqueeze(-1) * book).expand_as(learned[A_.BUY])
		offers[A_.BUY] = best_legal(prior[A_.BUY], x["buy_mask"])
		learned[A_.DISMISS] = self.dismiss_out(own_heroes).squeeze(-1)
		prior[A_.DISMISS] = torch.zeros_like(learned[A_.DISMISS])
		# A question's options: each against the global token and the hero it is about; Nullkiller2's skill scores
		about = torch.gather(state.heroes, 1, x["d_hero"].clamp(min=0).reshape(b, 1, 1).expand(b, 1, d)).squeeze(1) * (x["d_hero"] >= 0).float().unsqueeze(-1)
		query = self.option_q(torch.cat([state.glob, about], dim=-1))
		out["option_logp"] = masked_log_softmax((state.options * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(d) + skill_weight * known.skill_scores, x["option_mask"])
		# The action type: each table normalised within itself (the contract's: the AI samples a type, then an entry).
		# A type is as good as its best option: its score is its bias (priors/action_types.csv), what its domain prior
		# offers (offers: the best legal prior entry, as v1's; MOVE's and TRAVEL's the best tile's and area's value as a
		# logit, so value in reach holds END back) and its best legal entry's learned part (a max: the log-mean-exp of
		# before let a few good tiles among 300 count for nothing, and a sum let their number alone decide); END by the
		# global token
		scores, legal_types = [], []
		for action in c.ActionType:
			if action == A_.END:
				scores.append(self.end_out(state.glob).squeeze(-1) + self.type_bias[action])
				legal_types.append((x["g_kind"] == c.DecisionKind.TURN) | (x["g_kind"] == c.DecisionKind.EXCHANGE))
				continue
			name, mask_name = HEADS[action]
			mask = x[mask_name]
			if action == A_.CAST:  # (a hero with only those left has no CAST)
				mask = mask & self.castable
			out[name] = flat_log_softmax(learned[action] + prior[action], mask)
			scores.append(self.type_bias[action] + offers.get(action, 0.0) + best_legal(learned[action], mask))
			legal_types.append(mask.reshape(b, -1).any(dim=-1))
		out["type_logp"] = masked_log_softmax(torch.stack(scores, dim=-1), torch.stack(legal_types, dim=-1))
		self.value(state, known, out)
		return out

	def value(self, state, known, out):
		"""The value heads as v1's (model.CampaignModel.value): V2, V5, V7 (our share at the end of those days), the
		growth heads and V_end, from the global token, our heroes' and towns' mean, our player's token and Φ."""
		x = state.x
		mine = torch.cat([x["h_present"][:, :HO], x["t_present"][:, :TO]], dim=1).float().unsqueeze(-1)
		own = torch.cat([state.heroes[:, :HO], state.towns[:, :TO]], dim=1)
		own_mean = (own * mine).sum(dim=1) / mine.sum(dim=1).clamp(min=1)
		hidden = self.value_hidden(torch.cat([state.glob, own_mean, state.players[:, 0], known.phi], dim=-1))
		share_prior = -torch.log(x["g_players"].clamp(min=2).float() - 1)
		# V2, V5, V7: a share from 1 / players, or its logit's change from none (config.share_target)
		value_logit = self.value_heads(hidden) + (share_prior.unsqueeze(-1) if self.share_target == "share" else 0.0)
		out["value_logit"] = value_logit
		out["value"] = torch.sigmoid(value_logit)
		out["growth_logit"] = self.growth_heads(hidden)
		out["end_logit"] = self.end_head(hidden).squeeze(-1) + share_prior
		out["phi"] = known.phi_total


# ---- helpers: the known facts per entity ----

def best_legal(logits, mask):
	"""The best legal entry [B] of a table [B, ...] (logits broadcastable to mask's shape), 0 where none is legal."""
	flat, legal = logits.expand(mask.shape).reshape(mask.shape[0], -1), mask.reshape(mask.shape[0], -1)
	best = flat.masked_fill(~legal, -1e4).amax(dim=1)
	return torch.where(legal.any(dim=1), best, torch.zeros_like(best))


def ft_phi_features():
	return len(ft.PHI_SCALES) + len(ft.PHI_PARTS)


def log_ratio(value, reference):
	"""A size relative to a reference [..]: log(value / reference), clipped to -6..6 (0 for nothing)."""
	return torch.where(value > 0, torch.log(value.clamp(min=1e-3) / reference).clamp(-6, 6), torch.zeros_like(value))


def global_known(known):
	"""[B, PHI_FEATURES + 3]: Φ as the value heads see it, Nullkiller2's gold pressure, its pause, its want of a hero."""
	return torch.cat([known.phi, torch.stack([known.gold_pressure / (1 + known.gold_pressure), known.recruit_pause, known.hire_want], dim=-1)], dim=-1)


HERO_KNOWN = 1 + 3 + 4 + 2 * E.AMOUNT_FEATURES + 1


def hero_known(x, known):
	"""[B, H, HERO_KNOWN]: a hero's strength and army worth (absolute, and relative to our main army, our whole army and
	the strongest rival's), its worth and its next level's gain (ours), whether it is our main hero."""
	pad = lambda t: F.pad(t.float(), (0, H - HO))  # noqa: E731
	r = known.reference
	gold = known.hero_gold
	return torch.cat(
		[
			(known.hero_strength / 10).unsqueeze(-1), scales(known.hero_power, (1e3, 1e4, 1e5)),
			torch.stack([log_ratio(gold, r["main_army"][:, None]), log_ratio(gold, r["our_army"][:, None]), log_ratio(gold, r["rival_army"][:, None]), log_ratio(gold, r["phi"][:, None])], dim=-1) / 3,
			E.amount(pad(known.hero_worth), "gold"), E.amount(pad(known.level_gain), "gold"), pad(known.main).unsqueeze(-1),
		],
		dim=-1,
	)


TOWN_KNOWN = 1 + 3 * E.AMOUNT_FEATURES + 2 + 1


def town_known(x, known):
	"""[B, T, TOWN_KNOWN]: a town's strength, value and garrison's worth (absolute; the garrison relative to our main army
	and to the strongest rival army), its danger (ours)."""
	r = known.reference
	return torch.cat(
		[
			(known.town_strength / 10).unsqueeze(-1), E.amount(known.town_value, "gold"), E.amount(known.town_gold, "gold"), E.amount(known.town_power, "gold"),
			torch.stack([log_ratio(known.town_gold, r["main_army"][:, None]), log_ratio(known.town_gold, r["rival_army"][:, None])], dim=-1) / 3,
			F.pad(known.town_danger, (0, T - TO)).unsqueeze(-1),
		],
		dim=-1,
	)


TILE_KNOWN = 2 + E.AMOUNT_FEATURES + 2


def tile_known(x, features, known):
	"""[B, L, S, S, TILE_KNOWN]: what taking the tile's object is worth (Nullkiller2's or Φ's gold, where we may take it:
	not ours, not visited), its guards' strength (a wandering monster's, as seen), both also relative to our Φ and main
	army."""
	obj, flags = x["tile_object"], x["tile_flags"]
	gate = (obj >= 0) & (obj != ft.OBJ_HERO) & (obj != ft.OBJ_MONSTER) & (x["tile_owner"] != c.OWNER_SELF) & ~ft.bit(flags, c.TILE_VISITED_BY_PLAYER)
	gold = features.object_gold(obj, x["tile_subtype"]) * gate
	creature = x["tile_creature"]
	count = ft.typical_counts(x["tile_count"], ft.bit(flags, c.TILE_EXACT_COUNT), features) * (creature >= 0)
	power = count * features.fight_value[vocab_index(creature, c.CREATURE_VOCAB)]
	r = known.reference
	return torch.cat(
		[
			E.signed(gold, "gold"), E.amount(power, "gold"),
			torch.stack([log_ratio(gold, r["phi"][:, None, None, None]), log_ratio(power, r["main_army"][:, None, None, None])], dim=-1) / 3,
		],
		dim=-1,
	)


# MOVE's known features (2026-11-04, the user's review of them: dead ones out, relative ones in, several strengths):
#   kept      model.MOVE_FEATURES but defend, home, under_threat and to_main, 0 on ~all of joint7's tiles (99-100%; the
#             duplicate prior went too: "value" is the same)
#   strength  model.tile_features' move_strength: four estimates of the fight's log strength ratio, for this hero and
#             for our main army, within -2..2
#   relative  its move_relative: hopeless, the object, the fight's stake and gain, our town's risk (shares of Φ) and what
#             this hero can take into its army here
#   foreseen  the foresight's P(a monster here attacks), P(we win a battle here), the share of the stake it takes (detached)
MOVE_KEPT = tuple(i for i, name in enumerate(MOVE_FEATURES) if name not in ("defend", "home", "under_threat", "to_main"))
MOVE_STRENGTH, MOVE_RELATIVE, MOVE_FORESEEN = 8, 6, 3
MOVE_KNOWN = len(MOVE_KEPT) + MOVE_STRENGTH + MOVE_RELATIVE + MOVE_FORESEEN
OLD_MOVE_KNOWN = len(MOVE_FEATURES) + 1  # move_pair's known inputs before 2026-11-04 (MOVE_FEATURES, the prior again)
MIX_HIDDEN = 16  # the mixer's hidden layer (the user: more than 6)
HOPELESS = len(MOVE_KEPT) + MOVE_STRENGTH  # hopeless's place among MOVE's known features
HOPELESS_LOGIT = -3.0  # the mixer's direct weight on it to begin with: a guard 54x this hero's strength 3 logits down
FIGHT_FEATURES = 3


def move_pair_columns(old, fold=True):
	"""move_pair's first layer's weights [rows, dp + FORESIGHT_HIDDEN + OLD_MOVE_KNOWN] from before 2026-11-04 (or Adam's
	moments of them: fold False) -> [rows, dp + FORESIGHT_HIDDEN + MOVE_KNOWN + MIX_HIDDEN]: the pairs' and the foresight's
	columns as they were, each kept MOVE_FEATURES column at its new place (the prior's second copy added onto the first's,
	the same input; the moments keep the first's), behind's at 0 (it was 0 on every tile until 2026-11-04: its column never
	learned and would add noise now), every new input's at 0: the same outputs."""
	dp = old.shape[0] // 2
	base = dp + FORESIGHT_HIDDEN
	new = old.new_zeros(old.shape[0], base + MOVE_KNOWN + MIX_HIDDEN)
	new[:, :base] = old[:, :base]
	for to, frm in enumerate(MOVE_KEPT):
		new[:, base + to] = old[:, base + frm]
	if fold:
		new[:, base + MOVE_KEPT.index(MOVE_FEATURES.index("value"))] += old[:, base + OLD_MOVE_KNOWN - 1]
	new[:, base + MOVE_KEPT.index(MOVE_FEATURES.index("behind"))] = 0.0
	return new


def migrate_move_inputs(state_dict):
	"""A state dict with move_pair's first layer from before 2026-11-04 moved to its inputs now (move_pair_columns); as
	it is otherwise."""
	key = "move_pair.up.weight"
	old = state_dict.get(key)
	if old is None or old.shape[1] != old.shape[0] // 2 + FORESIGHT_HIDDEN + OLD_MOVE_KNOWN:
		return state_dict
	return {**state_dict, key: move_pair_columns(old)}


def fight_features(known):
	"""[K, W, W, FIGHT_FEATURES] the fight stepping on each window tile starts (model.tile_features: the tile's own enemy,
	or the worst guard next to a tile of a monster's zone): whether one does, its log strength ratio ours over theirs
	(/3, within -2..2) and how hopeless it is (beyond e^-1 of their strength: relu(-ratio - 1), within 0..6)."""
	fights = known.tile_fights
	ratio = known.tile_fight_ratio * fights
	return torch.stack([fights, (ratio / 3).clamp(-2.0, 2.0), torch.relu(-ratio - 1.0).clamp(max=6.0) * fights], dim=-1)


def reach_features(x, windows):
	"""[K, W, W, 11] a hero's reach to each tile of its window: the days (-1 never .. 3+, one-hot), the movement left on
	reaching it today, whether this hero visited its object."""
	pick = lambda name: x[name].reshape((-1, W, W)).index_select(0, windows)  # noqa: E731
	turns = pick("w_turns")
	return torch.cat([E.level(turns, -1, 3), E.amount(pick("w_move_left").clamp(min=0), "movement"), ft.bit(pick("w_flags"), c.TILE_VISITED_BY_HERO).float().unsqueeze(-1)], dim=-1)


def tile_positions():
	"""[L * S * S, 3] each map tile's (x, y, level), in the flattened grid's order."""
	z, y, xx = torch.meshgrid(torch.arange(L), torch.arange(S), torch.arange(S), indexing="ij")
	return torch.stack([xx, y, z], dim=-1).reshape(-1, 3)


def map_index(positions, b):
	"""(flat grid indices [B, N] of positions [B, N, 3] in the batch's flattened tiles, whether each is inside)."""
	px, py, pz = positions[..., 0], positions[..., 1], positions[..., 2]
	inside = (px >= 0) & (px < S) & (py >= 0) & (py < S) & (pz >= 0) & (pz < L)
	batch = torch.arange(b, device=positions.device)[:, None]
	index = ((batch * L + pz.clamp(0, L - 1)) * S + py.clamp(0, S - 1)) * S + px.clamp(0, S - 1)
	return index, inside


def window_tile_index(x, windows):
	"""[K, W, W] each window tile's index in its decision's flattened grid (level, row, column), -1 off the grid."""
	k = windows.shape[0]
	pos = x["h_pos"][:, :HO].reshape(-1, 3).index_select(0, windows)  # [K, 3]
	offsets = torch.arange(W, device=pos.device) - c.WINDOW_RADIUS
	dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
	tx, ty = pos[:, 0, None, None] + dx, pos[:, 1, None, None] + dy
	tz = pos[:, 2, None, None].expand(k, W, W)
	inside = (tx >= 0) & (tx < S) & (ty >= 0) & (ty < S) & (tz >= 0) & (tz < L)
	return torch.where(inside, (tz * S + ty) * S + tx, torch.full_like(tx, -1))


def windows_of(x, fixed=False, budget=None):
	"""(flat indices of the windows computed (decision x OWN_HEROES + hero), how many of them dense() places, [K] which
	are our heroes'). Our heroes' alone, as model.CampaignModel's, padded on Apple's GPU with copies of the first (the
	padding not real); fixed: every slot's, absent heroes' not real (CUDA graphs: one shape a batch size, no sync).
	fixed with a budget: that many slots, ours first (each in its decision's order) and then absent ones, not real:
	distinct slots, so dense() places them all and the absent ones' are zeroed as every slot's are. The same outputs
	when the batch has at most `budget` of our heroes (the caller counts them: more would be left out); the PC's games,
	2026-11-03: 2.8 a decision, never more than 4, of 8 slots."""
	present = x["h_present"][:, :HO].reshape(-1)
	if fixed and budget is not None and budget < present.shape[0]:
		windows = torch.argsort((~present).to(torch.int8), stable=True)[:budget]
		return windows, budget, present.index_select(0, windows)
	if fixed:
		return torch.arange(present.shape[0], device=present.device), present.shape[0], present
	windows = torch.nonzero(present).squeeze(-1)
	n = windows.shape[0]
	if n > 0 and windows.device.type == "mps" and not torch.onnx.is_in_onnx_export():
		windows = torch.cat([windows, windows[:1].expand((1 << (n - 1).bit_length()) - n)])
	return windows, n, torch.arange(windows.shape[0], device=windows.device) < n


def real_only(values, known):
	"""values [K, ...] with the windows that are not real zeroed (fixed windows: absent heroes'), as dense() would leave
	them (where, not a product: an absent hero's window may hold inf)."""
	return torch.where(known.real.reshape((-1,) + (1,) * (values.dim() - 1)), values, torch.zeros((), dtype=values.dtype, device=values.device))
