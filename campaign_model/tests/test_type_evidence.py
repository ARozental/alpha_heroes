"""config.type_evidence (e5, 2026-11-13): a reading of the policy's credit by action type. ppo_update hands the learner
each decision's blended advantage and its 7-day and end parts, each standardized; the learner adds them to faded
evidence by type (hidden games), kept in the checkpoint, and logs one line; training is the same with it on or off; an
error in it is a FAILURE line and the update goes on."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.labels import blend_credit
from campaign_model.learner import TYPE_PARTS, CampaignLearner
from campaign_model.model import CampaignModel
from campaign_model.schema import random_inputs
from campaign_model.tests.test_train import decisions_for
from campaign_model.train import ppo_update, standardized


def update(type_evidence, seed=6):
	torch.manual_seed(seed)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	x = random_inputs(8, seed=11)
	decisions = decisions_for(model.eval(), x)
	targets = np.full((8, c.N_VALUE_OFFSETS), np.nan, dtype=np.float32)
	seven = np.linspace(-1, 1, 8).astype(np.float32)
	end = {"targets": np.full(8, 0.5, np.float32), "real": np.full(8, 0.6, np.float32), "step": np.array([1, 2, 3, 4, 5, 6, 7, -1]), "week": np.full(8, -1), "price": np.zeros(8, np.float32), "weight": 0.2, "outcome": 0.25}
	torch.manual_seed(seed + 1)
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, targets, None, CampaignConfig(batch_size=4, passes=1, type_evidence=type_evidence), decode=lambda rows: {k: v[rows] for k, v in x.items()}, advantages=seven, end=end)
	return model, metrics, seven


def test_the_update_hands_over_the_blend_and_its_parts_and_trains_the_same():
	on_model, on, seven = update(True)
	off_model, off, _ = update(False)
	assert "type_credit" not in off and on["type_credit"].shape == (8, 3)
	blended, z7, zend = on["type_credit"].T
	assert np.allclose(z7, standardized(seven), atol=1e-6)
	assert np.allclose(blended, blend_credit(z7, zend, 0.2), atol=1e-6, equal_nan=True)  # (the policy's advantage as blended)
	for a, b in zip(on_model.parameters(), off_model.parameters()):
		assert torch.equal(a, b)  # (a reading: the training is the same, bit for bit)


def test_the_learner_keeps_faded_evidence_by_type_and_logs_it(capsys):
	n = 64
	rng = np.random.default_rng(0)
	types = [c.ActionType.ARMY if i % 4 == 0 else c.ActionType.MOVE for i in range(n)]
	decisions = [SimpleNamespace(action_type=int(t), kind=c.DecisionKind.TURN) for t in types]
	players = np.arange(n) % 16
	credits = rng.normal(0, 1, (n, 3)).astype(np.float32)
	credits[::4, 1] += 1.0  # (ARMY paid by the 7-day part)
	learner = SimpleNamespace(version=7, writer=None)
	CampaignLearner._note_type_evidence(learner, decisions, credits, players, np.ones(n, dtype=bool))
	assert set(learner.type_evidence) == set(TYPE_PARTS)
	n_army, total, _ = learner.type_evidence["seven"]["ARMY"]
	assert n_army == 16 and total / n_army > 0.5
	assert "campaign credit by type" in capsys.readouterr().out
	before = learner.type_evidence["seven"]["ARMY"][0]
	CampaignLearner._note_type_evidence(learner, decisions, credits, players, np.zeros(n, dtype=bool))  # (no hidden game: nothing added, the old faded)
	assert learner.type_evidence["seven"]["ARMY"][0] < before


def test_an_error_in_the_reading_is_a_failure_line_not_the_updates_end(tmp_path, monkeypatch, capsys):
	"""The schedule credit's learner step (test_learner's) with the reading made to fail: the update completes, a FAILURE
	line says so."""
	from campaign_model.tests import test_learner

	def broken(*args, **kwargs):
		raise RuntimeError("a broken reading")

	monkeypatch.setattr(CampaignLearner, "_note_type_evidence", broken)
	test_learner.test_an_update_runs_with_the_schedule_credits_action_price(tmp_path)
	assert "campaign: FAILURE: the type evidence (RuntimeError: a broken reading)" in capsys.readouterr().out
