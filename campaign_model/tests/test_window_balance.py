"""config.window_balance (b1 for e5, 2026-11-16): each minibatch's sequences re-dealt among its micro-batches by their
windows (train.balance_order), so that a micro-batch over window_budget, stepped eagerly over every hero slot (35% of
v3509's steps), is one only when its minibatch is. The minibatch keeps its rows and its counts, so with
foresight_minibatch_counts the update is the same up to the order of summation: its gradient and every metric it logs or
steers by."""
import numpy as np
import pytest
import torch

from campaign_model.records import heroes_in
from campaign_model.tests.test_foresight_counts import _setup, _update
from campaign_model.train import balance_order


@pytest.mark.parametrize("budget", [2.25, 1.8])
def test_the_update_is_the_same_up_to_summation_order_and_logs_the_same(budget):
	"""(1) memory_steps 2, 16 micro-batches of two two-step sequences, the foresight on uneven masks with the minibatch's
	counts, a budget of 2.25 windows a decision (9 a micro-batch; its rows hold 1-3) that rich_inputs' decisions overrun as dealt: the
	minibatch's gradient and every number the update returns (the KL, the clip fraction, the entropy, the loss shares'
	readings, the value statistics, the foresight's parts) are the same with the re-deal as without, to float rounding."""
	n = 96
	x, start, decisions, foresight, targets, advantages = _setup(n)
	streams = (np.repeat(np.arange(8), n // 8), np.tile(np.arange(n // 8), 8))
	settings = dict(batch_size=64, micro_batches=16, foresight_minibatch_counts=True, window_budget=budget)  # (1.8: the minibatch's own mean over it, the packed deal)
	plain, mp = _update(start, x, decisions, foresight, targets, advantages, streams, window_balance=False, **settings)
	dealt, md = _update(start, x, decisions, foresight, targets, advantages, streams, window_balance=True, **settings)
	assert md["window_over_dealt"] > 0 and md["window_over_balanced"] <= md["window_over_dealt"], (md["window_over_dealt"], md["window_over_balanced"])
	assert float(plain.norm()) > 0
	assert float((dealt - plain).norm() / plain.norm()) < 1e-5
	differ = []
	clocks = ("_seconds", "cpu_share", "gpu_mb", "gpu_anchor_mb")  # (wall-clock readings, not the update's numbers)
	for key, value in mp.items():
		if key.endswith(clocks):
			continue
		other = md.get(key)
		if isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(other, (int, float)) and np.isfinite(value):
			if abs(other - value) > 1e-5 * max(1.0, abs(value)):
				differ.append((key, value, other))
	# The one exception: the gradient check (once an update, on its FIRST micro-batch's own eager forward:
	# train.gradient_check): its readings are that micro-batch's, so a re-deal changes which rows they read. They steer
	# only the loss shares' balancing, frozen on the run's line (loss_shares_freeze 1420); every other number sums over
	# the minibatch's counts
	ONE_MICRO_BATCH = {"grad_shared_main", "grad_shared_foresight", "grad_shared_cosine", "grad_dead_share", "grad_raw_policy", "grad_raw_value", "grad_raw_foresight", "grad_raw_plan"}
	assert {key for key, _, _ in differ} <= ONE_MICRO_BATCH, [d for d in differ if d[0] not in ONE_MICRO_BATCH]
	assert len([k for k in mp if not k.endswith(clocks)]) - len(differ) > 50  # (the rest the same: KL, clip fraction, entropy, the value's, the foresight's parts)
	assert "window_over_dealt" not in mp  # (off: nothing of it runs)


def test_it_refuses_without_the_minibatchs_foresight_counts():
	n = 96
	x, start, decisions, foresight, targets, advantages = _setup(n)
	streams = (np.repeat(np.arange(8), n // 8), np.tile(np.arange(n // 8), 8))
	with pytest.raises(ValueError, match="foresight_minibatch_counts"):
		_update(start, x, decisions, foresight, targets, advantages, streams, batch_size=64, micro_batches=16, foresight_minibatch_counts=False, window_balance=True)


def test_the_re_deal_keeps_each_sequence_whole_and_in_step_order_and_evens_the_loads():
	"""(3) balance_order on a pass's order (sequence_order's layout: a micro-batch's step 0 rows, then its step 1 rows):
	the same rows in each minibatch, each sequence's rows together and in their step order, per sequences a micro-batch,
	a deterministic result, and every micro-batch at or under the budget when the minibatch's mean is."""
	rng = np.random.default_rng(1)
	steps, size, micro = 2, 512, 16
	per = size // micro // steps
	order = rng.permutation(4 * size)[: 3 * size]  # (three whole minibatches of rows 0..)
	windows = np.zeros(4 * size, dtype=np.int64)
	windows[order[: size // 2]] = 8  # (the first minibatch's first half heavy: as dealt, its micro-batches over)
	windows[order[size // 2 :]] = rng.integers(0, 6, len(order) - size // 2)
	budget = int(round(size // micro * 5.5))
	new, dealt, after = balance_order(order, windows, steps, size, micro, budget)
	again = balance_order(order, windows, steps, size, micro, budget)
	assert np.array_equal(new, again[0])  # (no draws)
	assert dealt > 0 and after == 0, (dealt, after)
	for first in range(0, len(order), size):
		a, b = order[first : first + size], new[first : first + size]
		assert sorted(a) == sorted(b)  # (the minibatch keeps its rows)
		seq_a = {tuple(r) for r in a.reshape(micro, steps, per).transpose(0, 2, 1).reshape(-1, steps)}
		seq_b = {tuple(r) for r in b.reshape(micro, steps, per).transpose(0, 2, 1).reshape(-1, steps)}
		assert seq_a == seq_b  # (each sequence whole, its rows in their step order)
		load = steps * windows[b].reshape(micro, steps, per).sum(axis=2).max(axis=1)
		assert load.max() <= budget
	# steps 1: single rows
	new1, dealt1, after1 = balance_order(order, windows, 1, size, micro, budget)
	assert dealt1 > 0 and after1 == 0
	# not whole: left as dealt
	assert balance_order(order[: size - 1], windows, steps, size, micro, budget)[0] is not None


def test_the_rows_windows_are_windows_ofs():
	"""The prepass's count (records.heroes_in) is what windows_of computes: our heroes and their further windows."""
	from campaign_model.tests.test_model_v2 import rich_inputs
	from campaign_model import features as ft

	x = rich_inputs(32, 4)
	t = {k: torch.from_numpy(v) for k, v in x.items()}
	assert np.array_equal(heroes_in(x), ft.windows_present(t).reshape(32, -1).sum(dim=1).numpy())


def test_an_over_full_minibatch_gives_up_the_fewest_micro_batches_never_more_than_as_dealt():
	"""e5's cliff (2026-11-16): a minibatch whose own mean is 10% over the budget. Evening its loads would put every
	micro-batch over; the deal gives up the fewest (k: the heaviest sequences fill k, the rest fit the others), never more
	over than as dealt, rows, sequences and step order kept."""
	rng = np.random.default_rng(4)
	steps, size, micro = 2, 8192, 128
	per = size // micro // steps
	order = rng.permutation(size)
	windows = rng.integers(2, 10, size).astype(np.int64)  # (a late game's: 2-9 windows a decision)
	mean = windows.mean()
	budget = int(round(size // micro * mean / 1.1))  # (the minibatch 10% over)
	new, dealt, after = balance_order(order, windows, steps, size, micro, budget)
	assert after <= dealt and after < micro, (dealt, after)
	# the aggregate lower bound: the fewest micro-batches whose heaviest sequences leave the rest fitting the others
	seq_w = windows[order].reshape(micro, steps, per).transpose(0, 2, 1).reshape(-1, steps)
	heavy = np.cumsum(seq_w[np.argsort(-seq_w.max(axis=1), kind="stable")], axis=0)
	total = seq_w.sum(axis=0)
	k = next(j for j in range(1, micro) if steps * (total - heavy[j * per - 1]).max() <= (micro - j) * budget)
	assert after <= k + micro // 10, (after, k, dealt)  # (the bound assumes the rest packs exactly to the budget, a per-sequence count a micro-batch; 41 against 32 here, 127 as dealt)
	assert sorted(new) == sorted(order)
	a = {tuple(r) for r in order.reshape(micro, steps, per).transpose(0, 2, 1).reshape(-1, steps)}
	b = {tuple(r) for r in new.reshape(micro, steps, per).transpose(0, 2, 1).reshape(-1, steps)}
	assert a == b
