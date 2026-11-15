"""config.foresight_minibatch_counts (2026-11-15, e5): the foresight's means over the whole minibatch, so that the
micro-batches add up to the minibatch's mean whatever micro_batches is (foresight.foresight_loss's counts, row_counts)."""
import copy

import numpy as np
import torch

from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.foresight import COUNT_NAMES, ForesightConfig, foresight_loss, row_counts
from campaign_model.model_v2 import ACTION_TARGETS, AREAS, CampaignModelV2
from campaign_model.tests.test_model_v2 import _KeptGradient, rich_inputs
from campaign_model.tests.test_train import decisions_for
from campaign_model.train import ppo_update


def _setup(n, seed=5):
	torch.manual_seed(seed)
	x = rich_inputs(n, 9)
	x["g_kind"][:] = c.DecisionKind.TURN
	x["mem"][:] = np.random.default_rng(10).integers(-c.MEMORY_SCALE // 2, c.MEMORY_SCALE // 2, x["mem"].shape)
	start = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	with torch.no_grad():  # (a trained model's stand-in: a fresh one's read and heads start at 0)
		for p in start.parameters():
			p.add_(0.05 * torch.randn_like(p))
	decisions = decisions_for(start.eval(), x)
	rng = np.random.default_rng(seed)
	# uneven masks: the sparse parts on a few rows only, as a real update's (army/fsweight.py)
	foresight = {
		"move": rng.normal(size=(n, ACTION_TARGETS)).astype(np.float32), "move_mask": np.array([d.action_type == c.ActionType.MOVE for d in decisions]),
		"move_battle": rng.random(n) < 0.15, "move_at": np.array([(d.index // (c.WINDOW * c.WINDOW), (d.index // c.WINDOW) % c.WINDOW, d.index % c.WINDOW) if d.action_type == c.ActionType.MOVE else (0, 0, 0) for d in decisions], np.int64),
		"move_win": rng.integers(0, 2, n).astype(np.float32), "move_stake": rng.random(n).astype(np.float32),
		"move_offer": rng.integers(0, 2, n).astype(bool),
		"answer": rng.normal(size=(n, ACTION_TARGETS)).astype(np.float32), "answer_mask": rng.random(n) < 0.1, "answer_battle": rng.random(n) < 0.03,
		"option": np.zeros(n, np.int64),
		"hero": rng.random((n, c.OWN_HEROES, 2)).astype(np.float32), "hero_mask": x["h_present"][:, : c.OWN_HEROES].copy(), "hero_alive": x["h_present"][:, : c.OWN_HEROES].copy(),
		"turn": rng.normal(size=(n, 1 + c.N_RESOURCES)).astype(np.float32), "turn_mask": rng.random((n, 1 + c.N_RESOURCES)) < 0.2,
		"plan": rng.integers(0, AREAS + 1, (n, c.OWN_HEROES)), "plan_mask": x["h_present"][:, : c.OWN_HEROES].copy(),
	}
	targets = np.full((n, c.N_VALUE_OFFSETS), 0.6, dtype=np.float32)
	advantages = np.linspace(-1, 1, n).astype(np.float32)
	return x, start, decisions, foresight, targets, advantages


def _update(start, x, decisions, foresight, targets, advantages, streams, **settings):
	m = copy.deepcopy(start)
	optimizer = _KeptGradient(m.parameters())
	np.random.seed(3)  # (sequence_order's draws: the same whatever the split)
	torch.manual_seed(0)
	config = CampaignConfig(passes=1, memory_steps=2, model_architecture="v2", target_kl=0.0, grad_clip=1e9, **settings)
	metrics, _ = ppo_update(m, optimizer, decisions, targets, None, config, decode=lambda rows: {k: v[rows] for k, v in x.items()}, advantages=advantages, foresight=foresight, streams=streams)
	return torch.cat([g.reshape(-1) for g in optimizer.kept[0]]), metrics


def test_with_the_minibatchs_counts_the_foresight_gradient_is_one_whatever_the_split():
	"""(a) the foresight's labels on uneven masks: the minibatch's gradient (the foresight's in it) in 1, 16 or 128
	micro-batches is one gradient to float rounding with config.foresight_minibatch_counts; without it, 128 parts move it
	(test_model_v2's 4/2/1 test leaves the foresight out for that reason)."""
	n = 320  # (sequence_order needs 128 two-step sequences in one minibatch of 256: 8 streams of 40 give 152 to 160)
	x, start, decisions, foresight, targets, advantages = _setup(n)
	streams = (np.repeat(np.arange(8), n // 8), np.tile(np.arange(n // 8), 8))
	got = {parts: _update(start, x, decisions, foresight, targets, advantages, streams, batch_size=256, micro_batches=parts, foresight_minibatch_counts=True) for parts in (1, 16, 128)}
	whole, whole_metrics = got[1]
	assert float(whole.norm()) > 0
	for parts in (16, 128):
		g, metrics = got[parts]
		assert float((g - whole).norm() / whole.norm()) < 1e-5, parts
		assert all(abs(metrics[f"foresight_{name}"] - whole_metrics[f"foresight_{name}"]) < 1e-5 for name in ("move_win", "answer_battle", "turn", "move_tiles", "hero_alive")), parts
	old, _ = _update(start, x, decisions, foresight, targets, advantages, streams, batch_size=256, micro_batches=128, foresight_minibatch_counts=False)
	assert float((old - whole).norm() / whole.norm()) > 1e-3  # (today's per-micro-batch means: the split moves it)


def test_in_one_micro_batch_the_counts_change_nothing():
	"""(b) every factor 1 and one micro-batch: the counts' code is today's, to float rounding."""
	n = 64
	x, start, decisions, foresight, targets, advantages = _setup(n, seed=7)
	streams = (np.repeat(np.arange(4), n // 4), np.tile(np.arange(n // 4), 4))
	new, mn = _update(start, x, decisions, foresight, targets, advantages, streams, batch_size=32, micro_batches=1, foresight_minibatch_counts=True)
	old, mo = _update(start, x, decisions, foresight, targets, advantages, streams, batch_size=32, micro_batches=1, foresight_minibatch_counts=False)
	assert float((new - old).norm() / old.norm()) < 1e-6
	assert all(abs(mn[k] - mo[k]) < 1e-6 for k in mo if k.startswith("foresight_") and isinstance(mo[k], float))


def test_a_part_with_no_label_is_exactly_zero_and_the_factors_scale_their_parts():
	"""(d) a minibatch with no label of a part: that part 0 exactly, no nan, with the counts at 0; the factors scale
	their parts alone."""
	torch.manual_seed(2)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	model.plan_outputs = True
	n = 8
	x = rich_inputs(n, 3)
	_, _, decisions, foresight, _, _ = _setup(n, seed=9)
	foresight["answer_mask"][:] = False
	foresight["answer_battle"][:] = False
	foresight["move_battle"][:] = False
	labels = {k: torch.from_numpy(np.asarray(v)) for k, v in foresight.items()}
	with torch.no_grad():
		out = model({k: torch.from_numpy(v) for k, v in x.items()})
	counts = {k: v.sum() for k, v in row_counts(labels, out["fs_move_legal"] > 0, out["fs_stack_tile"] > 0).items()}
	assert set(counts) == set(COUNT_NAMES)
	assert float(counts["answer_rows"]) == 0 and float(counts["move_battle"]) == 0
	loss, parts = foresight_loss(out, labels, ForesightConfig(), counts)
	assert torch.isfinite(loss)
	for name in ("answer_battle", "answer_score", "answer_parts", "move_score", "move_win", "move_stake"):
		assert float(parts[name]) == 0.0, name
	# the counts of one micro-batch's own masks give today's means; a factor scales its parts and nothing else
	_, plain = foresight_loss(out, labels, ForesightConfig())
	_, counted = foresight_loss(out, labels, ForesightConfig(), counts)
	assert all(abs(float(plain[k]) - float(counted[k])) < 1e-6 for k in plain)
	_, scaled = foresight_loss(out, labels, ForesightConfig(outcome_factor=0.5, answer_factor=0.25, answer_score_factor=0.1), counts)
	assert all(float(scaled[k]) == float(counted[k]) for k in ("move_battle", "move_parts", "hero_alive", "hero_army", "turn", "plan", "move_tiles"))
