"""Building blocks shared by the battle models (model.py, model_v2.py, model_v1.py)."""
import math

import torch
from torch import nn

from . import priors
from .constants import MASKED_LOGIT, N_VALUE_HORIZONS, ActionType
from .schema import INPUT_NAMES, OUTPUT_NAMES

# Starting preference between action types, as type-head logits: priors/action_types.csv (priors.py). The head's
# weights start at zero, so an untrained model picks among the legal types with exactly these odds.
ACTION_TYPE_PRIOR = {ActionType(i): float(bias) for i, bias in enumerate(priors.action_types().tolist())}


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


EMBEDDING_STD = 0.1


def small_embedding(count, dim):
	embedding = nn.Embedding(count, dim)
	nn.init.normal_(embedding.weight, std=EMBEDDING_STD)
	return embedding


def best_of(logits, legal):
	"""The largest legal entry of the last axis; 0 where none is legal."""
	return torch.where(legal.any(dim=-1), logits.masked_fill(~legal, MASKED_LOGIT).amax(dim=-1), torch.zeros_like(logits[..., 0]))


def zero_output(layer):
	nn.init.zeros_(layer.weight)
	nn.init.zeros_(layer.bias)
	return layer


class ActionScore(nn.Module):
	"""How a choice's own features [B, *choices, features] move its logit: linear (one weight per feature), own (a tiny
	layer from the choice's features to 1) and wide (a wider layer from the choice's features, the context and a
	summary of all legal choices' features: their mean and max). All start at 0. linear_only: only the first part
	(for choices with few features and very many entries, such as every spell at every hex)."""

	def __init__(self, names, dim, tiny=8, wide=32, linear_only=False):
		super().__init__()
		self.names = tuple(names)
		n = len(self.names)
		self.linear = nn.Parameter(torch.zeros(n))
		self.linear_only = linear_only
		if linear_only:
			return
		self.own = nn.Sequential(nn.Linear(n, tiny), nn.GELU(), zero_output(nn.Linear(tiny, 1)))
		self.wide_choice = nn.Linear(n, wide)  # per choice
		self.wide_rest = nn.Linear(dim + 2 * n, wide)  # per decision: the context and the summary
		self.wide_out = zero_output(nn.Linear(wide, 1))

	def forward(self, features, legal, context):
		linear = (features * self.linear).sum(dim=-1)
		if self.linear_only:
			return linear
		own = self.own(features).squeeze(-1)
		flat, flat_legal = features.flatten(1, -2), legal.flatten(1)[..., None].float()
		count = flat_legal.sum(dim=1)
		mean = (flat * flat_legal).sum(dim=1) / count.clamp(min=1)
		top = flat.masked_fill(flat_legal == 0, -2.0).amax(dim=1) * (count > 0)  # features are >= -1
		rest = self.wide_rest(torch.cat([context, mean, top], dim=-1))  # [B, wide]
		hidden = nn.functional.gelu(self.wide_choice(features) + rest.view(rest.shape[0], *([1] * (features.dim() - 2)), -1))
		return linear + own + self.wide_out(hidden).squeeze(-1)


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
