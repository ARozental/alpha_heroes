"""Building blocks shared by the battle models (model.py, model_v1.py)."""
import math

import torch
from torch import nn

from .constants import MASKED_LOGIT, ActionType
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


class OnnxWrapper(nn.Module):
	"""Positional-argument wrapper for ONNX export: inputs and outputs in schema order."""

	def __init__(self, model):
		super().__init__()
		self.model = model

	def forward(self, *tensors):
		outputs = self.model(dict(zip(INPUT_NAMES, tensors)))
		return tuple(outputs[name] for name in OUTPUT_NAMES)
