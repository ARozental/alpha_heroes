"""A battle's weight by its length (data.length_weight, TrainConfig.battle_length_scale), the games an update rests on
(train.games_rested_on) and the honest signal reading (optim.SignalShare), 2026-11-10."""
from types import SimpleNamespace

import numpy as np
import torch

from battle_model.data import length_weight
from battle_model.losses import AdvantageConfig
from battle_model.optim import SignalShare
from battle_model.supply import assemble
from battle_model.train import games_rested_on


def test_a_battle_counts_as_at_most_about_scale_decisions():
	assert length_weight(1, 30) == 1.0 and length_weight(500, 0) == 1.0
	assert abs(length_weight(30, 30) - 1 / (1 + 29 / 30)) < 1e-12
	totals = [n * length_weight(n, 30) for n in (1, 10, 30, 150, 621, 10_000)]
	assert totals == sorted(totals) and totals[-1] < 30 and abs(totals[4] - 621 / (1 + 620 / 30)) < 1e-9  # (rises toward 30)


def item(n, game, weight=1.0, side=0):
	data = {"advantage": np.random.default_rng(n).normal(size=n).astype(np.float32), "weight": np.full(n, weight, np.float32)}
	return {"data": data, "decisions": n, "game": game, "summary": SimpleNamespace(our_side=side), "source": "generated"}


def test_assemble_weighs_each_item_by_its_length_and_keeps_the_base():
	"""e5's review: the factor in the trainer (assemble), the items' weight their base: each item's decisions x
	length_weight(its decisions); weight_base the joined base weights; scale 0 leaves the weights alone."""
	items = [item(3, "a", weight=0.5), item(40, "b")]
	batch, _ = assemble(items, AdvantageConfig(), length_scale=30)
	expected = torch.cat([torch.full((3,), 0.5 * length_weight(3, 30)), torch.full((40,), length_weight(40, 30))])
	assert torch.allclose(batch["weight"], expected) and torch.equal(batch["weight_base"], torch.cat([torch.full((3,), 0.5), torch.ones(40)]))
	plain, _ = assemble(items, AdvantageConfig())
	assert torch.equal(plain["weight"], plain["weight_base"]) and torch.equal(plain["weight_base"], batch["weight_base"])
	assert np.array_equal(items[1]["data"]["weight"], np.ones(40, np.float32))  # (the item itself unchanged: the value phase's replay reads it)


def test_the_games_an_update_rests_on():
	"""(sum)^2 / sum of squares over the games: one game holding everything 1; four equal games 4; by weight and count;
	the length factor as assemble applies it."""
	assert games_rested_on([item(10, "a"), item(5, "a")]) == (1.0, 1.0)
	equal = [item(4, g) for g in "abcd"]
	assert games_rested_on(equal) == (4.0, 4.0)
	heavy = [item(100, "a", weight=0.1), item(10, "b"), item(10, "c")]  # (100 x 0.1 = 10 by weight: three equal games)
	by_weight, by_count = games_rested_on(heavy)
	assert abs(by_weight - 3.0) < 1e-9 and abs(by_count - 120**2 / (100**2 + 10**2 + 10**2)) < 1e-9
	long = [item(600, "a"), item(30, "b"), item(30, "c")]
	assert games_rested_on(long, 30)[0] > games_rested_on(long)[0]  # (the stand-off counted down: more games carry it)


def run_probe(share, dim=4000, updates=300, sigma=None, seed=0):
	"""Gradients s + n, n fresh every update: noise per coordinate of std sigma (None: 1), the signal sized so that its
	share of |g|^2 is `share`; the optimizer's v-hat set to sigma^2 exactly (so the applied share, signal over noise
	after dividing by sqrt(v), is `share` too when the signal is shaped like sigma). The mean readings after update 60."""
	g = torch.Generator().manual_seed(seed)
	sigma = torch.ones(dim) if sigma is None else sigma
	direction = torch.randn(dim, generator=g) * sigma  # (shaped like the noise: equal shares raw and applied)
	direction = direction / direction.norm()
	noise_norm = float(sigma.pow(2).sum().sqrt())
	signal = direction * noise_norm * (share / (1 - share)) ** 0.5 if share > 0 else torch.zeros(dim)
	p = torch.nn.Parameter(torch.zeros(dim))
	optimizer = torch.optim.Adam([{"params": [p], "name": "trunk"}])
	optimizer.state[p] = {"step": torch.tensor(10_000.0), "exp_avg": torch.zeros(dim), "exp_avg_sq": sigma.pow(2).clone()}
	probe = SignalShare(decay=0.95)
	raw, applied = [], []
	for update in range(updates):
		p.grad = signal + torch.randn(dim, generator=g) * sigma
		got = probe.read(optimizer)
		if update == 0:
			assert got == {}
		elif update > 60:
			raw.append(got["signal_share_trunk"])
			applied.append(got["signal_share_applied_trunk"])
	return float(np.mean(raw)), float(np.mean(applied)), probe


def test_the_signal_share_separates_a_share_of_a_hundredth_from_none():
	"""e5's review: the quantity is ~0.005, so 0 and 0.01 must read apart (dim 4000, 240 readings: the mean's noise
	~0.0005; tolerance 0.004), raw and as applied with a v that is not uniform (each coordinate's own noise)."""
	for sigma in (None, torch.exp(torch.linspace(-1.5, 1.5, 4000))):
		none_raw, none_applied, _ = run_probe(0.0, sigma=sigma)
		some_raw, some_applied, _ = run_probe(0.01, sigma=sigma)
		assert abs(none_raw) < 0.004 and abs(none_applied) < 0.004, (none_raw, none_applied)
		assert abs(some_raw - 0.01) < 0.004 and abs(some_applied - 0.01) < 0.004, (some_raw, some_applied)


def test_the_signal_share_keeps_its_shape_and_state_and_never_raises():
	"""A parameter without a gradient counts as zeros (S keeps its shape and its meaning); S and its counts go through a
	checkpoint; a broken input turns it off without raising."""
	a, b = torch.nn.Parameter(torch.zeros(5)), torch.nn.Parameter(torch.zeros(3))
	optimizer = torch.optim.Adam([{"params": [a, b], "name": "trunk"}])
	probe = SignalShare()
	a.grad, b.grad = torch.ones(5), torch.ones(3)
	probe.read(optimizer)
	b.grad = None
	probe.read(optimizer)
	assert probe.mean["trunk"].shape == (8,) and probe.count["trunk"] == 2
	other = SignalShare()
	other.load_state_dict(probe.state_dict())
	assert other.count == probe.count and torch.equal(other.mean["trunk"], probe.mean["trunk"])
	broken = SignalShare()
	assert broken.read(SimpleNamespace(param_groups=[{"params": [SimpleNamespace(grad=object())]}], state={})) == {} and not broken.on
