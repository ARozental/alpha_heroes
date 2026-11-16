"""config.loop_charge_moves (e5 for the user's objection, 2026-11-16: the 0.10 charge pushed END up at turn decisions):
True (the default) charges every flagged row as before, bit for bit; False charges a loop's exchange decisions alone,
the turn MOVE that opened a dialog wholly inside it still flagged and counted in the readings, never charged."""
import dataclasses

import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.labels import LOOP_NONE
from campaign_model.tests.test_loop_charge import BACK, EXCHANGE, GIVE, TURN, A, flags
from campaign_model import train as train_module
from campaign_model.train import charge_loops


def charge_loops_641656c(advantage, credited, loops, charge):
	"""641656c's charge_loops verbatim: the reference for loop_charge_moves True."""
	flagged = credited & (loops > 0)
	n = max(int(credited.sum()), 1)
	stats = {"loop_charge": float(charge), "loop_share": int(flagged.sum()) / n, "loop_adv_before": float(advantage[flagged].mean()) if int(flagged.sum()) else float("nan")}
	stats.update({f"loop_share_{t}": int((credited & (loops == t)).sum()) / n for t in (1, 2, 3)})
	if charge == 0.0:
		return advantage, stats
	return advantage - charge * flagged.to(advantage.dtype), stats


def take(seed=0, copies=40):
	"""A take of loops as labels.loop_flags flags them (give and take back, a null dialog, plain turns between), the
	decisions' kinds, a standardized advantage and a credited mask."""
	null = [(TURN, A.MOVE, "s0"), (EXCHANGE, A.ARMY, "s0"), (EXCHANGE, A.ARMY, "s1"), (EXCHANGE, A.END, "s0"), (TURN, A.END, "s0")]
	plain = [(TURN, A.BUILD, "s0"), (TURN, A.MOVE, "s0"), (TURN, A.END, "s0")]
	rng = np.random.default_rng(seed)
	loops, kinds = [], []
	for k in range(copies):
		events = [GIVE + BACK, null, plain][k % 3]
		loops += flags(events).tolist()
		kinds += [int(e[0]) for e in events]
	n = len(loops)
	advantage = torch.from_numpy(rng.normal(size=n).astype(np.float32))
	credited = torch.from_numpy(rng.random(n) > 0.1)
	return advantage, credited, torch.tensor(loops, dtype=torch.int64), torch.tensor(kinds, dtype=torch.int64)


def test_true_is_641656c_bit_for_bit():
	advantage, credited, loops, kinds = take()
	for charge in (0.0, 0.01, 0.1):
		new, s_new = charge_loops(advantage, credited, loops, charge)  # (chargeable None: what ppo_update passes with the flag True)
		old, s_old = charge_loops_641656c(advantage, credited, loops, charge)
		assert torch.equal(new, old) and s_new.keys() == s_old.keys()
		assert all(s_new[k] == s_old[k] or (np.isnan(s_new[k]) and np.isnan(s_old[k])) for k in s_old)


def test_false_charges_no_turn_row_and_the_exchange_rows_as_before():
	advantage, credited, loops, kinds = take()
	exchange = kinds == int(c.DecisionKind.EXCHANGE)
	flagged = credited & (loops > LOOP_NONE)
	assert int((flagged & ~exchange).sum()) > 0 and int((flagged & exchange).sum()) > 0  # (both kinds flagged in the take)
	new, s_new = charge_loops(advantage, credited, loops, 0.1, exchange)
	old, s_old = charge_loops_641656c(advantage, credited, loops, 0.1)
	assert torch.equal(new[~exchange], advantage[~exchange])  # (no charge on a TURN row)
	assert torch.equal(new[exchange], old[exchange])  # (the exchange rows' charge unchanged)
	assert {k: v for k, v in s_new.items() if k != "loop_charged_share"}.keys() == s_old.keys()
	assert all(s_new[k] == s_old[k] for k in s_old if not np.isnan(s_old[k]))  # (the readings count both kinds)
	assert s_new["loop_charged_share"] == int((flagged & exchange).sum()) / int(credited.sum()) < s_new["loop_share"]


@pytest.mark.parametrize("moves", [True, False])
def test_ppo_update_passes_the_exchange_rows_when_false(moves, monkeypatch):
	"""ppo_update end to end (test_train's way: its decisions, decode and end arrays, with loops and the charge): with
	the flag False charge_loops gets the decisions' EXCHANGE kind as its chargeable rows, with True None; the update's
	readings count both kinds either way."""
	from campaign_model.model import build_model
	from campaign_model.schema import random_inputs
	from campaign_model.tests.test_train import decisions_for
	from campaign_model.train import ppo_update

	torch.manual_seed(3)
	model = build_model({"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"})
	x = random_inputs(8, seed=15)
	decisions = [dataclasses.replace(d, kind=int(c.DecisionKind.EXCHANGE)) if b in (1, 2, 5) else d for b, d in enumerate(decisions_for(model.eval(), x))]
	n = len(decisions)
	seen = []
	real = train_module.charge_loops

	def spy(advantage, credited, loops, charge, chargeable=None):
		seen.append(chargeable)
		return real(advantage, credited, loops, charge, chargeable)

	monkeypatch.setattr(train_module, "charge_loops", spy)
	end = {"targets": np.linspace(0.2, 0.8, n).astype(np.float32), "real": np.full(n, 0.7, dtype=np.float32), "step": np.array([1, 2, 3, 4, 5, 6, 7, -1]), "week": np.full(n, -1), "price": np.zeros(n, dtype=np.float32), "weight": 0.5, "outcome": 0.25,
		"loops": np.array([1, 1, 0, 3, 0, 2, 1, 0], dtype=np.int64), "loop_charge": 0.1}  # (flagged: TURN rows 0, 3, 6; EXCHANGE rows 1, 5)
	targets = np.full((n, c.N_VALUE_OFFSETS), 0.6, dtype=np.float32)
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	config = CampaignConfig(batch_size=8, passes=1, target_kl=0.0, loop_charge=0.1, loop_charge_moves=moves)
	metrics, _ = ppo_update(model, torch.optim.SGD(model.parameters(), lr=0.1), decisions, targets, None, config, decode=decode, advantages=np.linspace(-1, 1, n).astype(np.float32), end=end)
	assert len(seen) == 1
	assert abs(metrics["loop_share"] - 5 / 8) < 1e-9  # (both kinds counted either way)
	if moves:
		assert seen[0] is None and "loop_charged_share" not in metrics
	else:
		assert seen[0].tolist() == [b in (1, 2, 5) for b in range(n)]
		assert abs(metrics["loop_charged_share"] - 2 / 8) < 1e-9
