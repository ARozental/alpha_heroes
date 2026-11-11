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
from . import views as V
from .model import MOVE_FEATURES, QUESTION_FEATURES, CustomFeatures, Features, PairBias, army_slot_outputs, army_slots, dense, flat_log_softmax, stack_power, swap_terms, with_added
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
ARMY_SLOT_KEYS = 32  # (schema 16) ARMY's target slot: the stack's query and each slot's key
TURN_TARGETS = 1 + R  # Φ's log ratio at the end of our turn, each resource's log ratio tomorrow
COSINE_BOUND = 0.995  # a choice's cosine scaled by it under the atanh (cosine_logit): each logit within ±3.0
DISTANCE_BUCKETS = 12  # the tile readers' distance bias: Chebyshev tiles 0, 1, 2, 3-4, 5-7, 8-11, ..., 96+, another level
# The bounded views of a MOVE tile or a TRAVEL area (schema 19, the user's rules of 2026-11-11: views.py), the same
# layout for both (CampaignModelV2.advice): its value in six views, its value a day of this hero's and of a baseline
# hero's (3 + 3), the two times as discounts (2 + 2), its prior's gap to this hero's best and to any hero's best (2),
# Nullkiller2's best candidate through the entry (views.CHANNELS), that candidate's destination: its own value's six
# views and whether a window of ours shows it (7), whether the entry is the one Nullkiller2 itself would play now (1)
VIEWS = 6 + 3 + 3 + 4 + 2 + V.F + 7 + 1  # (the last: this entry is Nullkiller2's own choice now, views.label)
PLAN_VIEWS = 2  # the plan head's own forecast for the entry's area: its chance, and against the hero's likeliest area
VIEW_STEP = 10.0  # FastLinear's (model.CORRECTION_STEP's reason)
LOG_SURVIVAL, RISK = V.CHANNELS.index("log_survival"), V.CHANNELS.index("risk")
assert S == 2 * TG  # (a TRAVEL area: two map cells a side; CampaignModelV2.think_again's area of a window tile)


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


def distance_buckets(delta, side=None):
	"""Chebyshev distance buckets [..] of tile offsets [.., 3]: 0..10 by distance, DISTANCE_BUCKETS - 1 another level.
	side (broadcast to delta's [..]): offsets in map blocks of that many tiles a side, the distance in tiles (a bigger
	map's: docs/map_size_plan.md); None, in tiles."""
	distance = torch.maximum(delta[..., 0].abs(), delta[..., 1].abs()).float()
	if side is not None:
		distance = distance * side.to(distance.dtype)
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

	def forward(self, tokens, positions, placed, tiles, tile_positions, valid, reach=None, side=None):
		"""tokens [B, N, d] at positions [B, N, 3] (placed [B, N]); tiles [B, M, dc] at tile_positions [M, 3] (valid
		[B, M]); reach: its bias [B, N, M, heads] (self.reach of each hero's reach to each tile) or None; side [B]: the
		map's block side (the grid's cells blocks of that many tiles; distances in tiles), None tiles. Returns what each
		token reads [B, N, d]."""
		b, n, d = tokens.shape
		h, dh = self.heads, d // self.heads
		q = self.q(tokens).reshape(b, n, h, dh)
		k = self.k(tiles).reshape(b, -1, h, dh)
		v = self.v(tiles).reshape(b, -1, h, dh)
		logits = torch.einsum("bnhd,bmhd->bhnm", q, k) / math.sqrt(dh)
		if side is None:
			bucket = distance_buckets(tile_positions[None, None, :, :] - positions[:, :, None, :])  # [B, N, M]
		else:
			blocks = torch.cat([torch.div(positions[..., :2], side[:, None, None], rounding_mode="floor"), positions[..., 2:]], dim=-1)
			bucket = distance_buckets(tile_positions[None, None, :, :] - blocks[:, :, None, :], side[:, None, None])
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


# The privileged critic (2026-11-06, the user's call on AlphaStar's lesson: its value functions saw the opponent's
# observations in training, best-response win rate 22% -> 82%): a correction to the share heads' and V_end's logits from
# the other players' own critic states, training only (the policy never reads it; the server has no rival states).
# 2026-11-07, the user's design after its first day: the growth heads have none (their fresh loss was 1% worse with it:
# our value's growth is our own data's), the rivals max-pooled as they are, our true share as an input
PRIVILEGED_RIVALS = P - 1
PRIVILEGED_HEADS = c.N_VALUE_OFFSETS + 1  # V2 V5 V7 (our share's change), V_end
PRIVILEGED_WIDTH = 32


class PrivilegedCritic(nn.Module):
	"""The share heads' and V_end's logit corrections [B, PRIVILEGED_HEADS], added to theirs (the public prediction from
	our own data, its bias theirs): x = [our critic state (value_hidden), the rivals' critic states max-pooled (each of
	its numbers the highest over the rivals alive with a state, each rival's at its last decision before our day: one
	shape whatever their number; 0 with none), the logit of our true share as the day began (the ledger's: the change
	heads' anchor)]; each correction a direct path from x plus one small dense layer from x, neither with a bias, both
	paths' outputs at 0 to start (the public prediction exactly; Adam's first steps move it by ~ the rate x |x|). Its
	inputs are detached (train.micro_loss): it learns on top of the critic and moves nothing else."""

	def __init__(self, dim, width=PRIVILEGED_WIDTH):
		super().__init__()
		self.dense = nn.Linear(2 * dim + 1, width)
		self.direct = nn.Linear(2 * dim + 1, PRIVILEGED_HEADS, bias=False)
		self.read = nn.Linear(width, PRIVILEGED_HEADS, bias=False)
		nn.init.zeros_(self.direct.weight)
		nn.init.zeros_(self.read.weight)

	def forward(self, own, rivals, mask, share_logit):
		"""own [B, d] our value_hidden, rivals [B, R, d] theirs (anything where mask is 0), mask [B, R] (1: a rival alive
		with a state), share_logit [B] our true share's logit now."""
		m = mask.unsqueeze(-1) > 0
		most = torch.where(m, rivals, torch.full_like(rivals, -1e4)).amax(dim=1)
		most = torch.where(m.any(dim=1), most, torch.zeros_like(most))  # (no rival: 0)
		x = torch.cat([own, most, share_logit.clamp(-10.0, 10.0).unsqueeze(-1)], dim=-1)
		return self.direct(x) + self.read(F.gelu(self.dense(x)))


class AdvantageHead(nn.Module):
	"""The advantage of playing (type t, entry e), as the policy's credit measures it (train.micro_loss's adv term,
	config.adv_weight; stage 1, 2026-11-11: it only learns, the policy reads none of it):

	  adv_type[t] + adv_entry[t][e] + adv_named[t] x [(t, e) is Nullkiller2's own choice]

	adv_type from the global token, adv_entry MOVE's and TRAVEL's alone (from the pair vectors move_out and travel_out
	read; 0 for the other types), adv_named one weight a type. Every input detached and every weight 0 to start: its
	loss reaches only these weights, and it predicts 0 until it learns."""

	def __init__(self, dim, pair_dim):
		super().__init__()
		self.types, self.move, self.travel = zero_weight(dim, len(c.ActionType)), zero_weight(pair_dim, 1), zero_weight(pair_dim, 1)
		self.named = nn.Parameter(torch.zeros(len(c.ActionType)))

	def forward(self, state, known):
		"""{adv_type [B, types], adv_move [B, HO, W, W], adv_travel [B, HO, L, TG, TG], adv_named [B, types], adv_label
		[B, 2]: Nullkiller2's own choice's type and index (-1 none), as floats (train.advantage_of)}."""
		glob = state.glob.detach()
		b = glob.shape[0]
		move = glob.new_zeros(b, HO, W, W)
		if known.n > 0:
			move = dense(real_only(self.move(state.move_pairs.detach()).squeeze(-1), known), known.windows, known.n, b)
		action, index, _ = known.advice.label
		return {
			"adv_type": self.types(glob), "adv_move": move, "adv_travel": self.travel(state.travel_pairs.detach()).squeeze(-1).reshape(b, HO, L, TG, TG),
			"adv_named": self.named.to(glob.dtype).unsqueeze(0) + glob.new_zeros(b, 1), "adv_label": torch.stack([action, index], dim=-1).to(glob.dtype),
		}


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


class FastLinear(nn.Module):
	"""A linear readout onto a logit, [.., inputs] -> [..]: its weights start at 0 (the readout adds nothing) and move
	`step` times as far a step as a plain weight. Adam moves a weight about the learning rate a step whatever its
	gradient's size: at 1e-5 and 64 steps an update a plain weight on a channel within 0..1 moves its logit 0.0006 an
	update at most, 0.2 a day, so a feature given straight to a logit through one would not be used for weeks (the MOVE
	fight features of 2026-11-03 were inert at this rate). At 10 times: 2 logits a day under a steady push, and a
	weight no gradient pushes walks ~0.01 a day. No bias: an entry with every channel 0 gets nothing."""

	def __init__(self, inputs, step=VIEW_STEP):
		super().__init__()
		self.weight = nn.Parameter(torch.zeros(inputs))
		self.step = step

	def forward(self, x):
		return self.step * (x @ self.weight.to(x.dtype))


def zero_weight(inputs, outputs):
	"""A linear layer without a bias, its weights 0 to begin with: it adds nothing at the start, and never anything
	where its inputs are all 0 (an entry without a candidate, a decision without the advisor)."""
	layer = nn.Linear(inputs, outputs, bias=False)
	nn.init.zeros_(layer.weight)
	return layer


def packed_cells(dest, decision, inverse):
	"""Window cells named as an input names them (dest [N, ..]: window slot x W x W + cell, -1 none; decision [N]: each
	row's decision) -> (their indices among the computed windows' cells [K x W x W], which have one): inverse [B x HO]
	is each window slot's row among the windows computed, -1 not computed (CampaignModelV2.advice)."""
	cells = W * W
	slot = torch.div(dest.clamp(min=0), cells, rounding_mode="floor").clamp(max=HO - 1)
	lead = decision.reshape((-1,) + (1,) * (dest.dim() - 1))
	row = inverse.index_select(0, (lead * HO + slot).reshape(-1)).reshape(dest.shape)
	valid = (dest >= 0) & (row >= 0)
	return torch.where(valid, row * cells + torch.remainder(dest.clamp(min=0), cells), torch.zeros_like(row)), valid


class CampaignModelV2(nn.Module):
	ARCHITECTURE = "v2"
	plan_outputs = False  # fs_plan too (plan_of): the trainer's model sets it (train.ppo_update); a server's does not
	adv_outputs = False  # the advantage head's too (AdvantageHead): the trainer's model sets it with config.adv_weight (train.ppo_update); a server's does not

	def load_state_dict(self, state_dict, strict=True, assign=False):
		"""A checkpoint from before a parameter was added (model.ADDED) loads with it at 0: the model as it was; one from
		before 2026-11-04's MOVE features with move_pair's first layer moved to its inputs now (migrate_move_inputs); one
		from before 2026-11-04's question head with its fight and join weights in option_weights (migrate_option_head)."""
		own = self.state_dict()
		state = with_added(migrate_action_types(migrate_option_head(migrate_move_inputs(state_dict)), own), own)
		return super().load_state_dict(pad_new_inputs(state, own), strict=strict, assign=assign)

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
		# (the value terms' gradient into the shared layers x this: CampaignLearner's value valve, config.value_valve; 1, as
		# before, the forward the same; a buffer set in place, so a recorded graph reads the current one; not saved)
		self.register_buffer("value_valve", torch.ones(()), persistent=False)
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
		# A building's key is its row and, at 0 to begin with, what it is in that town: the town's faction and the creature a
		# dwelling of that faction makes (the same building id is another creature and cost in each faction: 2026-11-04)
		self.build_identity = nn.Linear(d, d, bias=False)
		nn.init.zeros_(self.build_identity.weight)
		self.build_q = zero_output(nn.Linear(d, d))
		self.build_bias = nn.Parameter(torch.zeros(c.N_BUILDINGS))
		self.nk2_build, self.nk2_upgrade, self.nk2_trade = zero_output(nn.Linear(4, 1)), zero_output(nn.Linear(2, 1)), zero_output(nn.Linear(2, 1))
		self.trade_option = zero_output(nn.Linear(2, 1))  # (2026-11-04: each TRADE option's cost and whether it unlocks a building)
		self.recruit_head = nn.Sequential(nn.Linear(3 * d + 11, d), nn.GELU(), zero_output(nn.Linear(d, c.N_RECRUIT_AMOUNTS * c.N_RECRUIT_TARGETS)))  # (2026-11-04: + the town's visiting hero, a purchase's trades)
		self.hire_q, self.hire_k = zero_output(nn.Linear(d, d)), nn.Linear(d, d)
		self.hire_worth = zero_output(nn.Linear(4, 1))
		self.army_src, self.army_dst = nn.Linear(2 * d, d), nn.Linear(d, d)
		self.army_out = nn.Sequential(nn.GELU(), zero_output(nn.Linear(d, c.N_ARMY_AMOUNTS)))
		# ARMY's target slot (schema 16): the moving stack's query against each slot's key (its holder and its stack), at 0
		# to begin with; its prior: VCMI's own slot first, a swap rare (model.CustomFeatures.army_slot_priors)
		self.army_slot_q = zero_output(nn.Linear(d, ARMY_SLOT_KEYS))
		self.army_stay_q = zero_output(nn.Linear(d, ARMY_SLOT_KEYS))  # (ONE's all-but-one half: its own query, the same keys)
		self.army_slot_k = nn.Linear(2 * d, ARMY_SLOT_KEYS)
		self.upgrade_out = nn.Sequential(nn.GELU(), zero_output(nn.Linear(d, 1)))
		self.trade_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), zero_output(nn.Linear(d, R * R * c.N_TRADE_AMOUNTS)))
		self.spell = small_embedding(c.N_ADVENTURE_SPELLS, d)
		self.cast_q = zero_output(nn.Linear(d, d))
		self.buy_item = small_embedding(c.N_BUY_ITEMS, d)
		self.buy_q = zero_output(nn.Linear(d, d))
		self.dismiss_out = zero_output(nn.Linear(d, 1))
		# ARTIFACT (schema 13): a hero's query against its artifact's own row (the hero encoder's) and what to do with it,
		# at 0 to begin with; its prior: what the move does to Φ (model.CustomFeatures.artifact_prior)
		self.artifact_q = zero_output(nn.Linear(d, d))
		self.artifact_target = small_embedding(c.N_ARTIFACT_TARGETS, d)
		# MULTI (schema 14): a town's query (GIVE_ALL's: the receiving hero's) against each MultiKind's row, at 0 to begin
		# with; its prior: what the multi-action does to Φ (model.CustomFeatures.multi_prior)
		self.multi_q = zero_output(nn.Linear(d, d))
		self.multi_kind = small_embedding(c.N_MULTI, d)
		# A question (the user, 2026-11-04: every option its own embedding, never one YES vector for every question; a
		# choice's logit the atanh of a cosine): the decision's state, from the global token and the hero asked, against
		# each option's key: its identity (encode_options: the model's own vectors of what it names, with what it does with
		# the question's creatures, option_flags: o_battle, o_join, at 0 to begin with) and its token after the blocks (at
		# 0 to begin with); the options' known features (model.CustomFeatures.option_knowledge) onto the logit, at 0
		self.option_flags = zero_output(nn.Linear(2, d))
		self.decision = SwiGLU(2 * d, d, d)
		self.option_context = zero_output(nn.Linear(d, d))
		self.option_weights = nn.Linear(QUESTION_FEATURES, 1, bias=False)
		nn.init.zeros_(self.option_weights.weight)
		self.end_out = zero_output(nn.Linear(d, 1))
		self.type_bias = nn.Parameter(priors.action_types().clone())
		self.value_hidden = nn.Sequential(nn.Linear(3 * d + ft_phi_features(), d), nn.GELU())
		self.value_heads = zero_output(nn.Linear(d, c.N_VALUE_OFFSETS))
		self.growth_heads = zero_output(nn.Linear(d, c.N_VALUE_OFFSETS))
		self.end_head = zero_output(nn.Linear(d, 1))
		self.privileged = PrivilegedCritic(d)  # (training only: train.micro_loss, from the rivals' states the update has)
		self.register_buffer("tile_positions", tile_positions(), persistent=False)
		# The adventure spells the model may cast: not the ones that only show the map (constants.INFO_SPELLS)
		self.register_buffer("castable", torch.tensor([s not in c.INFO_SPELLS for s in range(c.N_ADVENTURE_SPELLS)]), persistent=False)
		# The map's block side (features.block_side: 1 up to 36 x 36, 4 on 144 x 144; docs/map_size_plan.md), for a model
		# trained on bigger maps to read a block apart from a tile: at 0, side 1's row (every small map's) never learned
		# (padding_idx: no gradient), so the small maps' outputs stay as they are whatever the training. Made last, the
		# other parameters' initialisation as before
		self.side_tile, self.side_global = side_embedding(dc), side_embedding(d)
		# Schema 19 (2026-11-11, the user's design: Nullkiller2's knowledge as inputs, never a loss on our choices; advice,
		# advise). Every part adds nothing to begin with (a checkpoint from before plays as it did: model.ADDED_HEADS),
		# and nothing ever where the advisor did not run:
		#   time_reach      the two times to a window tile as discounts, into the hero's reach to it (the pairs, the tile
		#                   reader's bias)
		#   view_pair       a MOVE tile's views into its (hero, tile) pair: the foresight's and move_pair's input
		#   travel_view     a TRAVEL area's views into its (hero, area) pair
		#   move_toward,    the pair of the destination the entry's best candidate heads for (the same hero's pair with
		#   travel_toward   that tile, where a window of ours shows it): the first step reads where it leads
		#   view_mix        the views' second-order effects straight onto MOVE's logit (as move_mix for the known features)
		#   view_move,      the views and the plan head's forecast straight onto the logit, each channel one fast weight
		#   view_travel     (FastLinear: 52 weights a table)
		#   plan_in         each hero's plan: its candidates' destinations' tile vectors and channels, weighted by
		#                   Nullkiller2's preference among them, onto the hero's token before the heroes attend
		#   nk_hero, nk_town, nk_global  its threats at our heroes and towns, its plan's tier and size
		self.time_reach, self.view_pair, self.travel_view = zero_weight(4, dc), zero_weight(VIEWS, dp), zero_weight(VIEWS, dp)
		self.move_toward, self.travel_toward = zero_weight(dp, dp), zero_weight(dp, dp)
		self.view_mix = nn.Linear(VIEWS, MIX_HIDDEN)
		self.view_mix_out = zero_output(nn.Linear(MIX_HIDDEN, 1))
		self.view_move, self.view_travel = FastLinear(VIEWS + PLAN_VIEWS), FastLinear(VIEWS + PLAN_VIEWS)
		self.plan_in = zero_weight(dc + V.F, d)
		self.nk_hero, self.nk_town, self.nk_global = zero_weight(V.THREAT_VIEWS, d), zero_weight(V.THREAT_VIEWS, d), zero_weight(V.GLOBAL_VIEWS, d)
		# The advantage head (2026-11-11, stage 1: it only learns; config.adv_weight), made last: the other parameters'
		# initialisation as before
		self.advantage = AdvantageHead(d, dp)

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
		state = self.advise(state, known)
		state = self.think(state)
		foresight = self.foresee(state, known)
		state = self.think_again(state, foresight, known)
		out = {**self.decide(state, known), **foresight["outputs"], "memory": state.memory_out, "resource_worth": known.resource_worth}
		if self.plan_outputs:
			out["fs_plan"] = state.plan
		if self.adv_outputs:
			out.update(self.advantage(state, known))
		return out

	def forward_steps(self, x, steps):
		"""The model over sequences of one player's consecutive decisions (train.sequence_order's micro-batch: `steps`
		blocks of rows, block s holding step s of every sequence in the same order). The first step starts from its
		recorded memory; each later one from the memory the step before left, a TURN decision's output, else the memory
		that step started from (AlphaMapAI's rule): the losses' gradient reaches the writes before them (truncated
		backpropagation through time). The outputs are one batch in x's order. A window budget (window_budget) is the
		whole batch's: each step computes its share of it."""
		size, budget = next(iter(x.values())).shape[0] // steps, self.window_budget
		outs, memory = [], None
		try:
			self.window_budget = None if budget is None else budget // steps
			for s in range(steps):
				part = {name: value[s * size : (s + 1) * size] for name, value in x.items()}
				if memory is not None:
					part["mem_float"] = memory
				out = self(part)
				wrote = (part["g_kind"] == c.DecisionKind.TURN).reshape(-1, 1, 1)
				memory = torch.where(wrote, out["memory"].float().clamp(-1.0, 1.0), Memory.of(part))
				outs.append(out)
		finally:
			self.window_budget = budget
		return {name: torch.cat([out[name] for out in outs]) for name in outs[0]}

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
		# Each window's hero (its flat index, decision x OWN_HEROES + hero token) and middle tile: a slot's own hero's
		# around it, or a further window of a hero of ours (schema 17, features.window_heroes)
		known.heroes, known.middles = ft.window_heroes(x, windows)
		known.side = ft.block_side(x)  # [B] the map grid's block side in tiles (1 up to 36 x 36: docs/map_size_plan.md)
		known.window_tiles = window_tile_index(known.middles, windows, known.side)  # [K, W, W] each window tile's map cell (-1 off the grid)
		ours = x["h_present"][:, :HO] & (x["h_owner"][:, :HO] == c.OWNER_SELF)
		armies = known.hero_gold[:, :HO] * ours
		rivals = known.hero_gold[:, HO:] * (x["h_present"][:, HO:] & (x["h_owner"][:, HO:] > c.OWNER_SELF))
		known.reference = {  # [B] each, floored so that early ratios stay bounded
			"phi": known.phi_total.clamp(min=1000.0),
			"main_army": armies.amax(dim=1).clamp(min=500.0),
			"our_army": armies.sum(dim=1).clamp(min=500.0),
			"rival_army": rivals.amax(dim=1).clamp(min=500.0),
		}
		known.advice = self.advice(x, known)
		return known

	def advice(self, x, known):
		"""The bounded views of every MOVE tile and TRAVEL area (VIEWS; views.py) and Nullkiller2's candidates as tables
		on them (schema 19), computed once for the layers that read them:

		  tables        views.candidate_tables: the candidates' channels at our MOVE and TRAVEL entries, the entries'
		                destination cells, the priors' preferences, each candidate's weight in its hero's plan
		  move          [K, W, W, VIEWS] by computed window; move_time [K, W, W, 4] the two times as discounts
		  travel        [B, HO, AREAS, VIEWS]
		  move_dest, travel_dest   (the index among the computed windows' cells of the destination the entry's best
		                candidate heads for, whether a computed window shows it)

		Our own estimates (a tile's value, the priors' logits, Φ) are read detached: the views are readings of them for
		the learned layers, never a path for the policy's gradient into the priors' corrections. An input dict from
		before the schema reads as a decision without the advisor (views.absent_rows)."""
		b = x["g_day"].shape[0]
		device = known.phi_total.device
		nk = x if "k_hero" in x else {**x, **V.absent_rows(b, device)}
		a = Features(x=nk, tables=V.candidate_tables(nk, dest=False))
		a.label = V.label(x)  # (Nullkiller2's own choice now: its action type, index and cell, -1 none)
		chosen_move = V.label_table(a.label[0], a.label[1], c.ActionType.MOVE, HO * W * W).reshape(b, HO, W, W)
		chosen_travel = V.label_table(a.label[0], a.label[1], c.ActionType.TRAVEL, HO * AREAS).reshape(b, HO, AREAS)
		phi = known.phi_total.detach()
		pace = V.daily_gain(phi, nk["g_phi_past"], nk["g_phi_past_days"])
		none = torch.zeros((), device=device)
		# each window slot's row among the windows computed (-1: none, or not a real one)
		inverse = torch.full((b * HO,), -1, dtype=torch.long, device=device)
		move_prior = torch.zeros(b, HO, W, W, device=device)
		if known.n > 0:
			k = known.windows.shape[0]
			rows = torch.where(known.real, torch.arange(k, device=device), torch.full((k,), -1, dtype=torch.long, device=device))
			inverse = inverse.index_copy(0, known.windows[: known.n], rows[: known.n])
			tile_prior = real_only(known.tile_prior, known).detach()
			move_prior = dense(tile_prior, known.windows, known.n, b)
		travel_logits = known.travel_logits.detach().reshape(b, HO, AREAS)
		# (a hero's best over its windows' tiles and its areas; by window slot: a hero's own slot holds its own)
		best_slot, best_any = V.hero_bests(move_prior, x["move_mask"], travel_logits, x["travel_mask"].reshape(b, HO, AREAS), ft.slot_heroes(x))
		value = None
		if known.n > 0:
			decision = torch.div(known.windows, HO, rounding_mode="floor")
			pick = lambda t: t.reshape((-1,) + tuple(t.shape[2:])).index_select(0, known.windows)  # noqa: E731
			turns = pick(x["w_turns"]).long()
			# (the movement left on arriving: today's is w_move_left, any day's schema 19's w_arrive_left; before it: none left)
			later = pick(V.arrive_left(x)) if "w_arrive_left" in x else torch.full_like(turns, -1)
			left = torch.where(turns == 0, pick(x["w_move_left"]).long(), later)
			mover = lambda name: x[name][:, :HO].reshape(-1).index_select(0, known.heroes)[:, None, None]  # noqa: E731 (the window's hero's)
			own_days, base_days = V.own_days(turns, left, mover("h_move"), mover("h_move_max")), V.base_days(turns, left, mover("h_move"), mover("h_move_max"))
			value = real_only(known.tile_value, known).detach()
			phi_k, pace_k = phi.index_select(0, decision), pace.index_select(0, decision)
			index, valid = packed_cells(pick(a.tables["move_dest_cell"].reshape(b, HO, W, W)), decision, inverse)
			dest_value = value.reshape(-1).index_select(0, index.reshape(-1)).reshape(index.shape) * valid
			time = torch.cat([V.discounts(own_days), V.discounts(base_days)], dim=-1)
			gaps = torch.stack([V.gap(tile_prior, pick(best_slot)[:, None, None]), V.gap(tile_prior, best_any.index_select(0, decision)[:, None, None])], dim=-1)
			a.move = real_only(
				torch.cat(
					[
						V.value_views(value, phi_k, pace_k), V.rate_views(value, own_days, phi_k, pace_k), V.rate_views(value, base_days, phi_k, pace_k), time, gaps,
						pick(a.tables["move"].reshape(b, HO, W, W, V.F)), V.value_views(dest_value, phi_k, pace_k) * valid.unsqueeze(-1), valid.float().unsqueeze(-1),
						pick(chosen_move).unsqueeze(-1),
					],
					dim=-1,
				),
				known,
			)
			a.move_time, a.move_dest = real_only(time, known), (index, valid)
		# TRAVEL: the pathfinder's days there (travel_days), and those days at this hero's movement in a baseline hero's
		days = x["travel_days"][:, :HO].reshape(b, HO, AREAS).float() / 100
		reached = days >= 0
		pace_of = (x["h_move_max"][:, :HO].float().clamp(min=1.0) / V.BASE_DAY).unsqueeze(-1)
		own_days = torch.where(reached, days, torch.full_like(days, V.FAR_DAYS))
		base_days = torch.where(reached, days * pace_of, torch.full_like(days, V.FAR_DAYS))
		gold = known.travel_gold.detach().unsqueeze(1).expand(b, HO, AREAS)
		index, valid = packed_cells(a.tables["travel_dest_cell"], torch.arange(b, device=device), inverse)
		dest_value = value.reshape(-1).index_select(0, index.reshape(-1)).reshape(index.shape) * valid if value is not None else none.expand(b, HO, AREAS)
		a.travel = torch.cat(
			[
				V.value_views(gold, phi, pace), V.rate_views(gold, own_days, phi, pace), V.rate_views(gold, base_days, phi, pace), V.discounts(own_days), V.discounts(base_days),
				torch.stack([V.gap(travel_logits, best_slot.unsqueeze(-1)), V.gap(travel_logits, best_any[:, None, None])], dim=-1),
				a.tables["travel"], V.value_views(dest_value, phi, pace) * valid.unsqueeze(-1), valid.float().unsqueeze(-1), chosen_travel.unsqueeze(-1),
			],
			dim=-1,
		)
		a.travel_dest = (index, valid)
		return a

	# ---- encode: raw inputs only ----

	def encode(self, x):
		state = State(x)
		side = ft.block_side(x).clamp(max=c.MAP_SIDE_MAX)
		state.glob = self.encode_global(x)
		state.glob = state.glob + self.side_global(side).to(state.glob.dtype)
		state.heroes, state.hero_stacks = self.encode_hero(E.hero_fields(x))
		state.towns, state.town_stacks, state.town_levels = self.encode_town(x)
		state.tavern, _ = self.encode_hero(E.tavern_fields(x))
		state.other_stacks, state.other = self.encode_other(x["x_creature"], x["x_count"], torch.ones_like(x["x_creature"][:, 0], dtype=torch.bool))
		tokens, identities = self.encode_options(x, self.option_tables())
		flags = self.option_flags(torch.stack([x["o_battle"], x["o_join"]], dim=-1).float()) * x["option_mask"].unsqueeze(-1)
		state.options, state.option_keys = tokens + flags, identities + flags  # (the key: a choice's in decide)
		state.tiles = self.encode_tiles(x)  # [B, L, S, S, dc]
		state.tiles = state.tiles + self.side_tile(side).to(state.tiles.dtype)[:, None, None, None, :]
		return state

	def build_scores(self, own_towns, x):
		"""BUILD's learned part [B, TO, N_BUILDINGS]: each town's query against each building's key, its row plus
		build_identity of the town's faction and of the creature that building makes there (a dwelling), as
		(query . row) + (query W . faction) + (query W . creature), the creature's from the creature table's every row."""
		d = self.dim
		query = self.build_q(own_towns)  # [B, TO, d]
		faction = vocab_index(x["t_faction"][:, :TO], c.FACTION_VOCAB)
		turned = query @ self.build_identity.weight  # (query W: W's input side, as the key is W applied to the identity)
		creature = self.features.dwelling_creature[faction]  # [B, TO, N_BUILDINGS], -1 not a dwelling
		creatures = turned @ self.encode_hero.army.creature.weight.t()  # [B, TO, CREATURE_VOCAB + 1]
		made = creatures.gather(2, vocab_index(creature, c.CREATURE_VOCAB)) * (creature >= 0)
		town = (turned * self.encode_town.faction(faction)).sum(dim=-1, keepdim=True)
		return (query @ self.building.weight.t() + town + made) / math.sqrt(d)

	def visitors(self, state):
		"""Each town token's visiting hero of ours [B, T, d] (h_town, not the garrison's), 0 where none: RECRUIT's "into the
		visiting hero" sees whom it arms."""
		x = state.x
		ours = x["h_present"] & (x["h_owner"] == c.OWNER_SELF) & ~x["h_garrisoned"]
		at = (x["h_town"].unsqueeze(-1) == torch.arange(x["t_present"].shape[1], device=x["h_town"].device)) & ours.unsqueeze(-1)  # [B, H, T]
		return torch.einsum("bht,bhd->btd", at.float(), state.heroes)

	def option_tables(self):
		"""The embeddings an option's identity names (encoders.OptionTables): the hero encoder's skill and artifact rows, spells
		and army encoder, the global encoder's object and pair rows: the same vectors the model reads those things as
		elsewhere."""
		hero, glob = self.encode_hero, self.encode_global
		return E.OptionTables(hero.skill, hero.artifact, hero.spells, hero.army, glob.object, glob.subtype, glob.pairs)

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
			index, inside = map_index(positions, b, known.side)
			keep = (present & inside).reshape(-1)
			# (every row, the others' adding zeros: no nonzero, whose size the host must learn, for a CUDA graph)
			stamped = torch.where(keep.unsqueeze(-1), stamp(vectors.reshape(-1, vectors.shape[-1])), 0.0)
			cells = cells.index_add(0, torch.where(keep, index.reshape(-1), 0), stamped.to(cells.dtype))
		on_map = ft.bit(x["tile_flags"], c.TILE_ON_MAP).reshape(b * L, 1, S, S).float()
		grid, half = self.unet(cells.reshape(b, L, S, S, dc).permute(0, 1, 4, 2, 3).reshape(b * L, dc, S, S), on_map)
		state.tiles = grid.reshape(b, L, dc, S, S).permute(0, 1, 3, 4, 2)  # [B, L, S, S, dc]
		flat = state.tiles.reshape(b, L * S * S, dc)
		valid = on_map.reshape(b, L * S * S) > 0
		# Our heroes' reach on the map, as the tile reader's bias per head: each window's tiles scattered to theirs (a
		# hero's further windows, schema 17, to the same hero's: known.heroes). The
		# windows are computed padded (windows_of: fixed shapes on Apple's GPU, which keeps a graph per shape); the padding
		# rows, copies of the first window, are left out of the scatter (they would count its reach twice)
		heads = self.heads
		reach = torch.zeros(b * HO * L * S * S, heads, device=flat.device, dtype=flat.dtype)
		if known.n > 0:
			k = known.windows.shape[0]
			state.window_reach = self.reach(None, reach_features(x, known.windows))  # [K, W, W, dc]
			state.window_reach = state.window_reach + self.time_reach(known.advice.move_time.to(state.window_reach.dtype))  # (schema 19: the two times as discounts)
			tiles = known.window_tiles.reshape(k, W * W)
			index = (known.heroes[:, None] * (L * S * S) + tiles.clamp(min=0)).reshape(-1)
			inside = ((tiles >= 0) & known.real[:, None]).reshape(-1)
			values = torch.where(inside.unsqueeze(-1), self.read_tiles.reach(state.window_reach).reshape(-1, heads), 0.0)
			reach = reach.index_add(0, torch.where(inside, index, 0), values.to(reach.dtype))
			# (a bigger map's block holds several of a window's tiles: their reach averaged; at side 1 at most one, as before)
			count = torch.zeros(reach.shape[0], 1, device=reach.device, dtype=reach.dtype).index_add(0, torch.where(inside, index, 0), inside.to(reach.dtype).unsqueeze(-1))
			reach = reach / count.clamp(min=1)
		reach = torch.cat([reach.reshape(b, HO, L * S * S, heads), torch.zeros(b, H - HO, L * S * S, heads, device=flat.device, dtype=flat.dtype)], dim=1)
		state.heroes = state.heroes + self.read_tiles(state.heroes, x["h_pos"], x["h_present"], flat, self.tile_positions, valid, reach, side=known.side)
		state.towns = state.towns + self.read_tiles(state.towns, x["t_pos"], x["t_present"], flat, self.tile_positions, valid, side=known.side)
		state.areas = self.area(half.reshape(b, L, 2 * dc, TG * TG).permute(0, 1, 3, 2).reshape(b, AREAS, 2 * dc))
		return state

	# ---- advise: Nullkiller2's plan onto what it concerns ----

	def advise(self, state, known):
		"""Nullkiller2's plan onto our tokens before they attend to each other (schema 19; each part adds nothing to begin
		with, and nothing where the advisor did not run): each hero of ours its plan vector (the map's tile vectors at
		its candidates' destinations and the candidates' channels, weighted by its preference among them:
		views.candidate_tables' plan_weight) and its threat there; each town of ours its threat; the global token the
		plan's tier and size."""
		x, a = state.x, known.advice
		nk, tables = a.x, a.tables
		b, dc = x["g_day"].shape[0], self.cell_dim
		index, inside = map_index(nk["k_dest"].long(), b, known.side)  # [B, NK]
		weight = (tables["plan_weight"] * inside).unsqueeze(-1)
		cells = state.tiles.reshape(b * L * S * S, dc).index_select(0, index.reshape(-1)).reshape(b, c.NK, dc)
		rows = torch.cat([cells * weight.to(cells.dtype), (tables["channels"] * weight).to(cells.dtype)], dim=-1)  # [B, NK, dc + F]
		hero = (nk["k_hero"].long() - 1).clamp(0, HO - 1)  # (an empty row's weight is 0)
		plan = torch.zeros(b, HO, dc + V.F, device=rows.device, dtype=rows.dtype).scatter_add(1, hero.unsqueeze(-1).expand(-1, -1, dc + V.F), rows)
		ours = (x["h_present"][:, :HO] & (x["h_owner"][:, :HO] == c.OWNER_SELF)).unsqueeze(-1)
		heroes = (self.plan_in(plan) + self.nk_hero(V.threat_views(nk["h_nk_threat"]).to(rows.dtype))) * ours
		state.heroes = state.heroes + F.pad(heroes, (0, 0, 0, H - HO))
		own_towns = (x["t_present"][:, :TO] & (x["t_owner"][:, :TO] == c.OWNER_SELF)).unsqueeze(-1)
		state.towns = state.towns + F.pad(self.nk_town(V.threat_views(nk["t_nk_threat"]).to(rows.dtype)) * own_towns, (0, 0, 0, T - TO))
		state.glob = state.glob + self.nk_global(V.global_views(nk).to(rows.dtype))
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
		heroes = state.heroes[:, :HO].reshape(b * HO, -1).index_select(0, known.heroes)  # (each window's hero: its own slot's, or a further window's)
		flat = state.tiles.reshape(b * L * S * S, self.cell_dim)
		decision = torch.div(known.windows, HO, rounding_mode="floor")
		index = known.window_tiles
		cells = torch.where((index >= 0).unsqueeze(-1), flat.index_select(0, (decision[:, None, None] * (L * S * S) + index.clamp(min=0)).reshape(-1)).reshape(k, W, W, -1), torch.zeros(k, W, W, self.cell_dim, device=flat.device, dtype=flat.dtype))
		strength = torch.cat([known.move_strength, known.move_relative[..., :1]], dim=-1)  # (the estimates and hopeless)
		views = self.view_pair(known.advice.move.to(cells.dtype))  # (schema 19: the tile's bounded views and Nullkiller2's candidate through it)
		return self.pair_hero(heroes)[:, None, None, :] + self.pair_tile(cells) + self.pair_reach(state.window_reach) + self.pair_fight(fight_features(known)) + self.pair_strength(strength) + views

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
		a, dp = known.advice, self.pair_dim
		b = x["g_day"].shape[0]
		pairs = flat = None
		if known.n > 0:
			pairs = self.pairs(state, known)  # (with the heroes as they are now)
			flat = pairs.reshape(-1, dp)
		# The pair of the destination an entry's best candidate heads for (schema 19: the same hero's pair with that tile),
		# where a computed window shows it, else 0
		toward = lambda index, valid: torch.where(valid.unsqueeze(-1), flat.index_select(0, index.reshape(-1)).reshape(index.shape + (dp,)), torch.zeros((), dtype=flat.dtype, device=flat.device))  # noqa: E731
		days = x["travel_days"][:, :HO].reshape(b, HO, AREAS)
		travel_in = torch.cat([E.amount(days.clamp(min=0).float() / 100, "days"), (days >= 0).float().unsqueeze(-1)], dim=-1)
		travel = self.travel_hero(state.heroes[:, :HO]).unsqueeze(2) + self.travel_area(state.areas).unsqueeze(1) + self.travel_days(None, travel_in)
		travel = travel + self.travel_view(a.travel.to(travel.dtype))
		if known.n > 0:
			travel = travel + self.travel_toward(toward(*a.travel_dest))
		state.travel_pairs = self.travel_pair(torch.cat([travel, torch.tanh(known.travel_prior.reshape(b, HO, AREAS) / 4).unsqueeze(-1)], dim=-1))
		# The plan head's own forecast of where Nullkiller2 would send each hero (plan_of; its label stays that plan),
		# read back detached as two channels of each entry's area: its chance, and against the hero's likeliest area.
		# What the entries keep of the plan in a game played without the advisor
		state.plan = self.plan_of(state)
		chance = state.plan.detach().float().softmax(dim=-1)[..., :AREAS]
		forecast = torch.stack([chance, chance / chance.amax(dim=-1, keepdim=True).clamp(min=1e-6)], dim=-1)  # [B, HO, AREAS, PLAN_VIEWS]
		state.travel_view = self.view_travel(torch.cat([a.travel, forecast], dim=-1))  # [B, HO, AREAS]
		if known.n > 0:
			k = known.windows.shape[0]
			move_known = torch.cat([known.move_features.index_select(-1, self.move_kept), known.move_strength, known.move_relative, foresight["foreseen"]], dim=-1)  # [K, W, W, MOVE_KNOWN]
			mixed = F.silu(self.move_mix(move_known))
			# each window tile's TRAVEL area (two map cells a side), for its hero's forecast there
			tiles = known.window_tiles.clamp(min=0)
			area = (torch.div(tiles, S * S, rounding_mode="floor") * TG + torch.div(torch.remainder(torch.div(tiles, S, rounding_mode="floor"), S), 2, rounding_mode="floor")) * TG + torch.div(torch.remainder(tiles, S), 2, rounding_mode="floor")
			there = forecast.reshape(b * HO, AREAS, PLAN_VIEWS).index_select(0, known.heroes).gather(1, area.reshape(k, W * W, 1).expand(k, W * W, PLAN_VIEWS)).reshape(k, W, W, PLAN_VIEWS)
			there = there * (known.window_tiles >= 0).unsqueeze(-1)
			seen = self.view_mix_out(F.silu(self.view_mix(a.move.to(mixed.dtype)))) + self.view_move(torch.cat([a.move, there], dim=-1)).unsqueeze(-1).to(mixed.dtype)
			state.move_mix = self.move_mix_out(mixed) + self.move_mix_direct(move_known) + seen
			state.move_pairs = self.move_pair(torch.cat([pairs + self.move_toward(toward(*a.move_dest)), hidden["move"], move_known, mixed], dim=-1))
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
		# (the act-now score of each window slot's hero: its own, or the hero a further window belongs to, schema 17)
		# Nullkiller2's own preference among its candidates, on the entries they go through (schema 19; nk_prior_weights:
		# all 0 until a prior is turned on): its best candidate's entry + beta, the others by their priority's ratio to
		# it, and what it expects to lose on the path and to meet on arrival
		tables = known.advice.tables
		best, fallback, survival, risk = self.features.nk_prior_weights()
		advised = lambda name: best * tables[f"prior_{name}"] + fallback * tables[f"prior_{name}_fallback"] + survival * tables[name][..., LOG_SURVIVAL] - risk * tables[name][..., RISK]  # noqa: E731
		move_prior = move_prior + advised("move").reshape(b, HO, W, W)
		learned[A_.MOVE], prior[A_.MOVE] = move + self.act_move(own_heroes).reshape(b, HO).gather(1, ft.slot_heroes(x)).reshape(b, HO, 1, 1), move_prior
		offers[A_.MOVE] = ft.signed_offer(move_prior, x["move_mask"])  # (below 0 when every move is worse than staying)
		# Nullkiller2's own choice now (schema 19: views.label), as a prior of nk_label_weight logits on its entry inside the
		# entry's table and on its type's score, END's included (0 until turned on: config.nk2_label_prior); a targeted
		# spell's tile in its hero's cell table too
		label_action, label_index, label_cell = known.advice.label
		label_weight = self.features.nk_label_weight()
		label_hero = torch.div(label_index.clamp(min=0), c.N_ADVENTURE_SPELLS, rounding_mode="floor").clamp(max=HO - 1)
		label_cells = V.label_table(torch.where(label_cell >= 0, label_action, torch.full_like(label_action, -1)), label_hero * (W * W) + label_cell.clamp(0, W * W - 1), A_.CAST, HO * W * W)
		cast_cell = cast_cell + label_weight * label_cells.reshape(b, HO, W, W).to(cast_cell.dtype)
		out["cast_cell_logp"] = masked_log_softmax(cast_cell.reshape(b, HO, W * W), x["cast_cell_mask"].reshape(b, HO, W * W)).reshape(b, HO, W, W)
		# TRAVEL: (hero, area) pairs, + the area's prior (its value discounted by the days to it)
		learned[A_.TRAVEL] = (self.travel_out(state.travel_pairs).squeeze(-1) + state.travel_view.to(state.travel_pairs.dtype) + self.act_travel(own_heroes)).reshape(b, HO, L, TG, TG)
		prior[A_.TRAVEL] = (known.travel_logits + advised("travel")).reshape(b, HO, L, TG, TG)  # (the network reads travel_prior: without the night)
		offers[A_.TRAVEL] = ft.signed_offer(prior[A_.TRAVEL], x["travel_mask"])
		# BUILD: the town's query against each building. A purchase short of resources pays the trades it makes (known.
		# build_trade, recruit_trade: the gold of worth they lose), priced as every change of Φ is (value_logit), in the
		# entry and in its type's offer: the least any legal entry pays, 0 while one needs no trade (within the table
		# alone a charge every entry pays cancels: a BUILD with nothing affordable traded for whatever it chose)
		trade_cost = self.features.trade_cost_weight()
		build_charge = trade_cost * ft.value_logit(known.build_trade)
		learned[A_.BUILD] = self.build_scores(own_towns, x) + self.build_bias
		prior[A_.BUILD] = self.nk2_build(known.build_rewards).squeeze(-1) - build_charge
		offers[A_.BUILD] = best_legal(-build_charge, x["build_mask"])
		# RECRUIT: the town and each creature level's token, the amounts; Nullkiller2's prior
		available = x["t_available"][:, :TO].clamp(min=0).float()
		gold = x["g_resources"][:, 6].clamp(min=0).float()
		recruit_in = torch.cat(
			[
				own_towns.unsqueeze(2).expand(b, TO, c.N_LEVELS, d), state.town_levels[:, :TO], scales(available, (5, 20, 80)),
				scales(gold, (1000, 5000, 20000)).reshape(b, 1, 1, 3).expand(b, TO, c.N_LEVELS, 3),
				scales(self.features.creature_cost[vocab_index(x["t_available_creature"][:, :TO], c.CREATURE_VOCAB)] * available, (1000, 5000, 20000)),
				self.visitors(state)[:, :TO].unsqueeze(2).expand(b, TO, c.N_LEVELS, d),
				torch.stack([known.recruit_short.float(), torch.log1p(known.recruit_trade / 1000.0)], dim=-1),  # (not one affordable alone: the agent trades for it)
			],
			dim=-1,
		)
		learned[A_.RECRUIT] = self.recruit_head(recruit_in).reshape(b, TO, c.N_LEVELS, c.N_RECRUIT_AMOUNTS, c.N_RECRUIT_TARGETS)
		recruit_charge = (trade_cost * ft.value_logit(known.recruit_trade))[..., None, None].expand_as(learned[A_.RECRUIT])
		prior[A_.RECRUIT] = self.features.recruit_prior(known.town_danger).expand_as(learned[A_.RECRUIT]) - recruit_charge
		_, upgrade_weight, book_weight, pause_weight = self.features.nk2_weights()  # (a skill's: model.CustomFeatures.option_knowledge)
		home_weight, defend_weight, ferry_weight, hire_weight, _ = self.features.pull_weights()  # (TRADE: trade_cost_weight since 2026-11-04)
		offers[A_.RECRUIT] = -pause_weight * known.recruit_pause + best_legal(-recruit_charge, x["recruit_mask"])  # (no army bought while saving for a hall: Nullkiller2's habit)
		# HIRE: the town's query against each hero for hire (the same hero space as ours) + what the hire does to Φ
		worth = torch.cat([scales(known.tavern_worth.clamp(min=0), (1000, 3000, 10000)), torch.tanh(known.hire_gain / 2000).unsqueeze(-1)], dim=-1)
		learned[A_.HIRE] = torch.einsum("btd,bsd->bts", self.hire_q(own_towns), self.hire_k(state.tavern)) / math.sqrt(d) + self.hire_worth(worth).squeeze(-1).unsqueeze(1)
		prior[A_.HIRE] = ft.value_logit(known.hire_gain).unsqueeze(1).expand(b, TO, c.N_TAVERN)
		offers[A_.HIRE] = best_legal(prior[A_.HIRE], x["hire_mask"]) + hire_weight * known.hire_want
		# ARMY and UPGRADE: each stack (its holder's token with it) against each holder
		holders = torch.cat([own_heroes, own_towns, state.other.unsqueeze(1)], dim=1)
		holder_stacks = torch.cat([state.hero_stacks[:, :HO], state.town_stacks[:, :TO], state.other_stacks.unsqueeze(1)], dim=1)
		slot_rows = torch.cat([holders.unsqueeze(2).expand(b, HD, A, d), holder_stacks], dim=-1)
		src = self.army_src(slot_rows)
		dst = self.army_dst(holders)
		within, swap_only, default_slot, swap_slot, stay_slot = self.features.army_slot_priors()
		same_army = torch.eye(HD, device=src.device)[None, :, None, :, None]  # (schema 16: a move within one army)
		learned[A_.ARMY] = self.army_out(src[:, :, :, None, :] + dst[:, None, None, :, :]) + within * same_army
		slots = army_slots(x)
		whole = (torch.arange(c.N_ARMY_AMOUNTS, device=src.device) == c.ArmyAmount.ALL).float()
		gain_weight, slot_weight, last_weight = self.features.swap_priors()
		gain, given, last = swap_terms(stack_power(x, self.features.fight_value))  # (a swap priced by the power it moves)
		swapping = (swap_only + gain_weight * gain + last_weight * last[:, :, None, None]) * slots.swap_only.float()
		prior[A_.ARMY] = self.features.army_prior(known.hero_strength[:, :HO]).expand_as(learned[A_.ARMY]) + swapping[..., None] * whole
		offers[A_.ARMY] = best_legal(prior[A_.ARMY], x["army_mask"])  # (below 0 when every legal move leaves troops in a mine)
		# its target slot (schema 16): each (stack, slot) pair's learned score and the slot's kind
		keys = self.army_slot_k(slot_rows)
		slot_logits = torch.einsum("bfsk,btuk->bfstu", self.army_slot_q(src), keys) / math.sqrt(ARMY_SLOT_KEYS)
		slot_logits = slot_logits + default_slot * slots.default.float() + (swap_slot + slot_weight * given[:, None, None, :, :]) * slots.swap.float()
		stay_logits = slot_logits + stay_slot + torch.einsum("bfsk,btuk->bfstu", self.army_stay_q(src), keys) / math.sqrt(ARMY_SLOT_KEYS)
		army_slot_outputs(out, slots, slot_logits, stay_logits)
		learned[A_.UPGRADE] = self.upgrade_out(src).squeeze(-1) + self.nk2_upgrade(known.upgrade_gains).squeeze(-1)
		prior[A_.UPGRADE] = upgrade_weight * (known.upgrade_gains[..., 0] > 0).float()
		offers[A_.UPGRADE] = best_legal(prior[A_.UPGRADE], x["upgrade_mask"])
		# TRADE (from the global token), CAST, BUY, DISMISS (each hero's)
		needs = torch.stack([known.trade_needs[:, None, :, 0].expand(b, R, R), known.trade_needs[:, :, None, 1].expand(b, R, R)], dim=-1)
		# (the user, 2026-11-04: every trade is dumb unless it buys what could not be bought without it: each option's cost to Φ
		# and whether it unlocks a building, features.trade_options, in its prior and into its learned part)
		options = known.trade_options  # [B, R, R, F, 2]: the gold of worth lost, unlocks
		read = torch.stack([torch.log1p(options[..., 0] / 1000.0), options[..., 1]], dim=-1)  # (the learned part's reading: log(1 + cost / 1000))
		learned[A_.TRADE] = self.trade_head(state.glob).reshape(b, R, R, c.N_TRADE_AMOUNTS) + self.nk2_trade(needs) + self.trade_option(read).squeeze(-1)
		prior[A_.TRADE] = -trade_cost * ft.value_logit(options[..., 0]) - self.features.trade_waste_weight() * (1.0 - options[..., 1])  # (the charge a purchase trading for itself pays: no route cheaper; a trade unlocking nothing more)
		offers[A_.TRADE] = best_legal(prior[A_.TRADE], x["trade_mask"])
		learned[A_.CAST] = self.cast_q(own_heroes) @ self.spell.weight.t() / math.sqrt(d)
		prior[A_.CAST] = torch.zeros_like(learned[A_.CAST])
		book = (torch.arange(c.N_BUY_ITEMS, device=gold.device) == c.BUY_ARTIFACTS.index(ft.SPELLBOOK)).float()
		learned[A_.BUY] = self.buy_q(own_heroes) @ self.buy_item.weight.t() / math.sqrt(d)
		prior[A_.BUY] = (book_weight * known.spellbook_wanted.float().unsqueeze(-1) * book).expand_as(learned[A_.BUY])
		offers[A_.BUY] = best_legal(prior[A_.BUY], x["buy_mask"])
		ids = torch.cat([x["h_artifacts"][:, :HO], x["h_backpack"][:, :HO]], dim=-1)
		query = self.artifact_q(own_heroes)  # [B, HO, d]
		own_artifacts = (query.unsqueeze(2) * self.encode_hero.artifact(vocab_index(ids, c.ARTIFACT_VOCAB))).sum(dim=-1)  # [B, HO, P]
		targets = query @ self.artifact_target.weight.t()  # [B, HO, N_ARTIFACT_TARGETS]
		learned[A_.ARTIFACT] = (own_artifacts.unsqueeze(-1) + targets.unsqueeze(2)) / math.sqrt(d)
		prior[A_.ARTIFACT] = known.artifact_prior
		offers[A_.ARTIFACT] = best_legal(prior[A_.ARTIFACT], x["artifact_mask"])
		pair = x["x_heroes"] - 1  # (GIVE_ALL's rows: the exchange's receiving hero, x_heroes' first and second)
		receivers = torch.gather(state.heroes, 1, pair.clamp(min=0).unsqueeze(-1).expand(-1, -1, d)) * (pair >= 0).unsqueeze(-1).float()  # [B, 2, d]
		receivers = torch.cat([receivers, receivers.new_zeros(b, TO - receivers.shape[1], d)], dim=1)  # [B, TO, d]
		kinds = self.multi_kind.weight  # [N_MULTI, d]
		by_town = (self.multi_q(own_towns) @ kinds.t()) / math.sqrt(d)  # [B, TO, N_MULTI]
		by_receiver = (self.multi_q(receivers) @ kinds.t()) / math.sqrt(d)
		kind = torch.arange(c.N_MULTI, device=kinds.device)
		by_hero = (self.multi_q(own_heroes) @ kinds.t()) / math.sqrt(d)  # [B, HO (= TO), N_MULTI]: DROP_SLOW's, DROP_FOREIGN's rows
		exchange = (kind == c.MultiKind.GIVE_ALL) | (kind == c.MultiKind.TAKE_BEST) | (kind == c.MultiKind.TAKE_BEST_FAST)
		hero = (kind == c.MultiKind.DROP_SLOW) | (kind == c.MultiKind.DROP_FOREIGN) | (kind == c.MultiKind.SPLIT_SINGLES) | (kind == c.MultiKind.MERGE_ALL)
		learned[A_.MULTI] = torch.where(exchange, by_receiver, torch.where(hero, by_hero, by_town))
		prior[A_.MULTI] = known.multi_prior
		offers[A_.MULTI] = best_legal(prior[A_.MULTI], x["multi_mask"])
		learned[A_.DISMISS] = self.dismiss_out(own_heroes).squeeze(-1)
		prior[A_.DISMISS] = torch.zeros_like(learned[A_.DISMISS])
		# A question: the decision's state (the global token, the hero asked) against each option's key (its identity and
		# its token after the blocks); the options' priors and known features (model.CustomFeatures.option_knowledge)
		about = torch.gather(state.heroes, 1, x["d_hero"].clamp(min=0).reshape(b, 1, 1).expand(b, 1, d)).squeeze(1) * (x["d_hero"] >= 0).float().unsqueeze(-1)
		decision = self.decision(torch.cat([state.glob, about], dim=-1))
		keys = state.option_keys + self.option_context(state.options)
		chosen = cosine_logit(decision, keys) + known.option_prior + self.option_weights(known.option_features).squeeze(-1)
		out["option_logp"] = masked_log_softmax(chosen, x["option_mask"])
		# The action type: each table normalised within itself (the contract's: the AI samples a type, then an entry).
		# A type is as good as its best option: its score is its bias (priors/action_types.csv), what its domain prior
		# offers (offers: the best legal prior entry, as v1's; MOVE's and TRAVEL's the best tile's and area's value as a
		# logit, so value in reach holds END back) and its best legal entry's learned part (a max: the log-mean-exp of
		# before let a few good tiles among 300 count for nothing, and a sum let their number alone decide); END by the
		# global token
		scores, legal_types = [], []
		for action in c.ActionType:
			named = label_weight * (label_action == int(action)).to(state.glob.dtype)  # [B]: Nullkiller2's own choice is of this type
			if action == A_.END:
				scores.append(self.end_out(state.glob).squeeze(-1) + self.type_bias[action] + named)
				legal_types.append((x["g_kind"] == c.DecisionKind.TURN) | (x["g_kind"] == c.DecisionKind.EXCHANGE))
				continue
			name, mask_name = HEADS[action]
			mask = x[mask_name]
			if action == A_.CAST:  # (a hero with only those left has no CAST)
				mask = mask & self.castable
			table = learned[action] + prior[action]
			chosen = V.label_table(label_action, label_index, action, table[0].numel()).reshape(table.shape)  # (1 at its entry)
			out[name] = flat_log_softmax(table + label_weight * chosen.to(table.dtype), mask)
			scores.append(self.type_bias[action] + offers.get(action, 0.0) + best_legal(learned[action], mask) + named)
			legal_types.append(mask.reshape(b, -1).any(dim=-1))
		out["type_logp"] = masked_log_softmax(torch.stack(scores, dim=-1), torch.stack(legal_types, dim=-1))
		self.follow(out, label_action, label_index, torch.stack(legal_types, dim=-1))
		self.value(state, known, out)
		return out

	def follow(self, out, action, index, legal):
		"""Nullkiller2's own choice followed at a chance (model.CustomFeatures.nk_follow_logs: rho; off: nothing changes):
		the policy the games sample becomes rho x (its choice) + (1 - rho) x (ours), exactly, at every decision whose
		labelled type is legal and whose labelled entry is legal in that type's table (END: the type alone). In the
		contract's terms (the AI samples a type, then an entry of its table):

		  P(type)              = (1 - rho) P_own(type) + rho [type is its choice's]
		  P(entry | its type)  = ((1 - rho) P_own(its type) P_own(entry | its type) + rho [entry is its choice]) / P(its type)

		the other types' tables as they are. A targeted spell's tile and ARMY's target slot stay our own tables'.
		out: type_logp and the tables' log-probabilities, changed in place; action, index [B]: its choice (views.label);
		legal [B, types]."""
		on, log_rho, log_rest = self.features.nk_follow_logs()
		own_types = out["type_logp"]
		b = own_types.shape[0]
		named = action.clamp(0, own_types.shape[1] - 1).unsqueeze(1)  # [B, 1]
		entry_ok = action == int(c.ActionType.END)
		for kind, (name, _) in HEADS.items():
			table = out[name].reshape(b, -1)
			here = (action == int(kind)) & (index >= 0) & (index < table.shape[1])
			entry_ok = entry_ok | (here & torch.isfinite(table.gather(1, index.clamp(0, table.shape[1] - 1).unsqueeze(1)).squeeze(1)))
		valid = on & (action >= 0) & legal.gather(1, named).squeeze(1) & entry_ok  # [B]
		its_type = (named == torch.arange(own_types.shape[1], device=named.device)) & valid.unsqueeze(1)
		rest = (log_rest + own_types).to(own_types.dtype)
		mixed_types = torch.where(its_type, log_sum(rest, log_rho.to(own_types.dtype)), rest)
		out["type_logp"] = torch.where(valid.unsqueeze(1), mixed_types, own_types)
		type_own, type_mixed = own_types.gather(1, named), mixed_types.gather(1, named)  # [B, 1] its type's, ours and mixed
		for kind, (name, _) in HEADS.items():
			table = out[name].reshape(b, -1)
			here = valid & (action == int(kind))
			joint = rest.new_zeros(()) + log_rest + type_own + table  # log((1 - rho) P_own(type) P_own(entry | type))
			its_entry = (index.clamp(0, table.shape[1] - 1).unsqueeze(1) == torch.arange(table.shape[1], device=index.device)) & here.unsqueeze(1)
			mixed = torch.where(its_entry, log_sum(joint, log_rho.to(table.dtype)), joint) - type_mixed
			out[name] = torch.where(here.unsqueeze(1), mixed.to(table.dtype), table).reshape(out[name].shape)

	def value(self, state, known, out):
		"""The value heads as v1's (model.CampaignModel.value): V2, V5, V7 (our share at the end of those days), the
		growth heads and V_end, from the global token, our heroes' and towns' mean, our player's token and Φ."""
		x = state.x
		mine = torch.cat([x["h_present"][:, :HO], x["t_present"][:, :TO]], dim=1).float().unsqueeze(-1)
		own = torch.cat([state.heroes[:, :HO], state.towns[:, :TO]], dim=1)
		own_mean = (own * mine).sum(dim=1) / mine.sum(dim=1).clamp(min=1)
		shared = torch.cat([state.glob, own_mean, state.players[:, 0]], dim=-1)
		shared = shared * self.value_valve + shared.detach() * (1 - self.value_valve)  # (value_valve: the forward the same, the gradient into the shared layers x it)
		hidden = self.value_hidden(torch.cat([shared, known.phi], dim=-1))
		share_prior = -torch.log(x["g_players"].clamp(min=2).float() - 1)
		# V2, V5, V7: a share from 1 / players, or its logit's change from none (config.share_target)
		value_logit = self.value_heads(hidden) + (share_prior.unsqueeze(-1) if self.share_target == "share" else 0.0)
		out["value_logit"] = value_logit
		out["value"] = torch.sigmoid(value_logit)
		out["growth_logit"] = self.growth_heads(hidden)
		out["end_logit"] = self.end_head(hidden).squeeze(-1) + share_prior
		out["phi"] = known.phi_total
		out["value_hidden"] = hidden  # (the privileged critic's: our state, and a rival's at its decisions)


# ---- helpers: the known facts per entity ----

def log_sum(a, b):
	"""log(exp(a) + exp(b)), b finite (torch.logaddexp in operators the ONNX export has; a may be -inf: then b)."""
	top = torch.maximum(a, b)
	return top + torch.log(torch.exp(a - top) + torch.exp(b - top))


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


def cosine_logit(query, keys, bound=COSINE_BOUND):
	"""Each choice's logit [B, O] (the user, 2026-11-04): the atanh of its key's cosine with the decision's query (query
	[B, d], keys [B, O, d]), scaled by bound to keep it and its gradient finite; about 0 for a fresh model's (random
	vectors are nearly orthogonal), every choice as likely."""
	cosine = bound * F.cosine_similarity(query.unsqueeze(1), keys, dim=-1, eps=1e-6)
	return 0.5 * (torch.log1p(cosine) - torch.log1p(-cosine))  # (atanh: ONNX's opset 17 has none)


def migrate_option_head(state_dict):
	"""A state dict from before 2026-11-04's question head (option_q's dot product, answer_fight and answer_join, the
	options' hashed rows): the fight's and the join's weights become option_weights' columns (model.QUESTION_FEATURES:
	the fight's 6, the join's 3, then their flags, which take the two biases), exactly; option_q and the hashed rows go
	(encoders.OptionEncoder: each option's identity now the model's own vectors of what it names); as it is otherwise."""
	gone = ("option_q.", "answer_fight.", "answer_join.", "encode_options.subtype.")
	if not any(key.startswith(gone) for key in state_dict):
		return state_dict
	state = {key: value for key, value in state_dict.items() if not key.startswith(gone)}
	if "answer_fight.weight" in state_dict:
		fight, join = state_dict["answer_fight.weight"], state_dict["answer_join.weight"]
		biases = torch.stack([state_dict["answer_fight.bias"][0], state_dict["answer_join.bias"][0]]).reshape(1, 2).to(fight)
		state["option_weights.weight"] = torch.cat([fight, join, biases], dim=1)
	return state


GROWN_INPUTS = ("recruit_head.0.weight", "option_weights.weight")  # (2026-11-04: + the visiting hero; + a purchase's fit)


def migrate_action_types(state_dict, own):
	"""A state dict from before an action type was added (schema 13: ARTIFACT, the 13th; 14: MULTI, the 14th) or MULTI's
	kinds were (schema 15: 4 -> 8): type_bias gains their entries, multi_kind its rows, and the global encoder's
	previous-type table its rows before the "none" row (vocab_index's last), all as the fresh model (own) has them; as it
	is otherwise."""
	state = dict(state_dict)
	for key in ("type_bias", "multi_kind.weight"):
		if key in state and key in own and state[key].shape[0] < own[key].shape[0]:
			state[key] = torch.cat([state[key], own[key][state[key].shape[0] :].to(state[key])])
	key = "encode_global.last.weight"
	if key in state and key in own and state[key].shape[0] < own[key].shape[0]:
		old, types = state[key], state[key].shape[0] - 1  # (the old none row: its last)
		state[key] = torch.cat([old[:types], own[key][types : own[key].shape[0] - 1].to(old), old[types:]])
	return state


def pad_new_inputs(state_dict, own):
	"""A state dict whose GROWN_INPUTS layers have fewer input columns than the model's (own: its state_dict): their new
	columns, at the end, at 0 (the model as it was); as it is otherwise."""
	state = dict(state_dict)
	for key in GROWN_INPUTS:
		if key in state and key in own and state[key].shape[1] < own[key].shape[1]:
			old = state[key]
			state[key] = torch.cat([old, old.new_zeros(old.shape[0], own[key].shape[1] - old.shape[1])], dim=1)
	return state


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


def side_embedding(dim):
	"""A block side's embedding [MAP_SIDE_MAX + 1, dim]: every row 0 to begin with, side 1's never learned."""
	table = nn.Embedding(c.MAP_SIDE_MAX + 1, dim, padding_idx=1)
	nn.init.zeros_(table.weight)
	return table


def tile_positions():
	"""[L * S * S, 3] each map tile's (x, y, level), in the flattened grid's order."""
	z, y, xx = torch.meshgrid(torch.arange(L), torch.arange(S), torch.arange(S), indexing="ij")
	return torch.stack([xx, y, z], dim=-1).reshape(-1, 3)


def map_index(positions, b, side=None):
	"""(flat grid indices [B, N] of positions [B, N, 3] in the batch's flattened tiles, whether each is inside); side [B]:
	the grid's cells blocks of that many tiles a side (a bigger map's), None tiles."""
	px, py, pz = positions[..., 0], positions[..., 1], positions[..., 2]
	if side is not None:
		px, py = torch.div(px, side[:, None], rounding_mode="floor"), torch.div(py, side[:, None], rounding_mode="floor")
	inside = (px >= 0) & (px < S) & (py >= 0) & (py < S) & (pz >= 0) & (pz < L)
	batch = torch.arange(b, device=positions.device)[:, None]
	index = ((batch * L + pz.clamp(0, L - 1)) * S + py.clamp(0, S - 1)) * S + px.clamp(0, S - 1)
	return index, inside


def window_tile_index(pos, windows, side=None):
	"""[K, W, W] each window tile's index in its decision's flattened grid (level, row, column), -1 off the grid; pos
	[K, 3] the windows' middle tiles (features.window_heroes); side [B]: the grid's cells blocks of that many tiles a
	side (a bigger map's: several window tiles share one), None tiles."""
	k = windows.shape[0]
	offsets = torch.arange(W, device=pos.device) - c.WINDOW_RADIUS
	dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
	tx, ty = pos[:, 0, None, None] + dx, pos[:, 1, None, None] + dy
	if side is not None:
		per = side.index_select(0, torch.div(windows, HO, rounding_mode="floor"))[:, None, None]
		tx, ty = torch.div(tx, per, rounding_mode="floor"), torch.div(ty, per, rounding_mode="floor")
	tz = pos[:, 2, None, None].expand(k, W, W)
	inside = (tx >= 0) & (tx < S) & (ty >= 0) & (ty < S) & (tz >= 0) & (tz < L)
	return torch.where(inside, (tz * S + ty) * S + tx, torch.full_like(tx, -1))


def windows_of(x, fixed=False, budget=None):
	"""(flat indices of the windows computed (decision x OWN_HEROES + window slot), how many of them dense() places, [K]
	which hold a window). The slots that hold one alone (our heroes' own and their further ones, schema 17:
	features.windows_present), as model.CampaignModel's, padded on Apple's GPU with copies of the first (the
	padding not real); fixed: every slot's, empty slots' not real (CUDA graphs: one shape a batch size, no sync).
	fixed with a budget: that many slots, those with a window first (each in its decision's order) and then empty
	ones, not real: distinct slots, so dense() places them all and the empty ones' are zeroed as every slot's are. The
	same outputs when the batch has at most `budget` windows (the caller counts them, records.heroes_in: more would be
	left out); the PC's games, 2026-11-03, before further windows: 2.8 a decision, never more than 4, of 8 slots."""
	present = ft.windows_present(x).reshape(-1)
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
