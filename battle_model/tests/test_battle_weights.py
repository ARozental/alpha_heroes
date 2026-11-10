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


def item(n, game, weight=1.0, base=True, side=0):
	data = {"advantage": np.random.default_rng(n).normal(size=n).astype(np.float32), "weight": np.full(n, weight, np.float32)}
	if base:
		data["weight_base"] = np.ones(n, np.float32)
	return {"data": data, "decisions": n, "game": game, "summary": SimpleNamespace(our_side=side), "source": "generated"}


def test_assemble_joins_items_from_before_the_length_weight():
	"""A buffer saved before battle_length_scale (no weight_base) joins with new items: their weight is their base."""
	batch, _ = assemble([item(3, "a", weight=0.5), item(2, "b", base=False)], AdvantageConfig())
	assert torch.equal(batch["weight_base"], torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])) and torch.equal(batch["weight"], torch.tensor([0.5, 0.5, 0.5, 1.0, 1.0]))


def test_the_games_an_update_rests_on():
	"""(sum)^2 / sum of squares over the games: one game holding everything 1; four equal games 4; by weight and count."""
	assert games_rested_on([item(10, "a"), item(5, "a")]) == (1.0, 1.0)
	equal = [item(4, g) for g in "abcd"]
	assert games_rested_on(equal) == (4.0, 4.0)
	heavy = [item(100, "a", weight=0.1), item(10, "b"), item(10, "c")]  # (100 x 0.1 = 10 by weight: three equal games)
	by_weight, by_count = games_rested_on(heavy)
	assert abs(by_weight - 3.0) < 1e-9 and abs(by_count - 120**2 / (100**2 + 10**2 + 10**2)) < 1e-9


def test_the_signal_share_reads_what_other_updates_repeat():
	"""Gradients s + n (n fresh every update) with |s|^2 a known share of |g|^2: the mean reading over many updates is
	near that share; pure noise reads ~0; the first update gives no reading; an error turns it off, never raises."""
	torch.manual_seed(0)
	dim = 4000
	signal = torch.randn(dim)
	for share in (0.0, 0.2):
		p = torch.nn.Parameter(torch.zeros(dim))
		optimizer = torch.optim.Adam([{"params": [p], "name": "trunk"}])
		probe = SignalShare(decay=0.95)
		readings = []
		for update in range(300):
			noise = torch.randn(dim)
			p.grad = (np.sqrt(share) * signal / signal.norm() + np.sqrt(1 - share) * noise / noise.norm()) * 3.0
			got = probe.read(optimizer)
			if update == 0:
				assert got == {}
			elif update > 60:
				readings.append(got["signal_share_trunk"])
			optimizer.step()
		mean = float(np.mean(readings))
		assert abs(mean - share) < 0.05, (share, mean)
	broken = SignalShare()
	assert broken.read(SimpleNamespace(param_groups=[{"params": [SimpleNamespace(grad=object())]}], state={})) == {} and not broken.on
