"""config.end_week_share (e5 for the user, 2026-11-13, deploy 75): the end credit's week share fixed. Off (-1) is the
trust-driven share and the one-step fallback, bit for bit; at 1 the week alone, and a row whose week is unknown gets no
end credit (its 7-day credit stands); the share used and the rows with no end credit are logged; the grades are not
touched (outcome_share_min still sets them)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.labels import end_advantages
from campaign_model.learner import COLUMNS, CampaignLearner
from campaign_model.model import CampaignModel
from campaign_model.schema import random_inputs
from campaign_model.tests.test_train import decisions_for
from campaign_model.train import ppo_update


def old_end_advantages(values, step, week, real, price, turn, decision_price, outcome):
	"""end_advantages as it was before end_week_share (e0d9b95)."""
	there = np.where(step >= 0, values[np.maximum(step, 0)], real)
	td = there - values - np.where(turn, decision_price, 0.0)
	later = np.where(week >= 0, values[np.maximum(week, 0)], real)
	weekly = later - values - price
	return np.where(np.isfinite(weekly), (1 - outcome) * td + outcome * weekly, td).astype(np.float32)


def arrays(seed, n=400):
	rng = np.random.default_rng(seed)
	values = rng.uniform(0.05, 0.95, n).astype(np.float32)
	step = np.where(rng.random(n) < 0.9, rng.integers(0, n, n), -1)
	week = np.where(rng.random(n) < 0.6, rng.integers(0, n, n), -1)
	real = np.where(rng.random(n) < 0.7, rng.uniform(0, 1, n), np.nan).astype(np.float32)
	price = rng.uniform(0, 0.01, n).astype(np.float32)
	turn = rng.random(n) < 0.8
	return values, step, week, real, price, turn


def test_off_is_the_old_end_credit_bit_for_bit():
	for seed in range(5):
		args = arrays(seed)
		for outcome in (0.25, 0.6, 1.0, np.linspace(0.2, 1.0, 400).astype(np.float32)):
			got, want = end_advantages(*args, 1e-4, outcome), old_end_advantages(*args, 1e-4, outcome)
			assert np.array_equal(got, want, equal_nan=True)


def test_the_week_alone_has_no_one_step_fallback():
	values, step, week, real, price, turn = arrays(7)
	got = end_advantages(values, step, week, real, price, turn, 1e-4, 1.0, week_only=True)
	later = np.where(week >= 0, values[np.maximum(week, 0)], real)
	weekly = later - values - price
	known = np.isfinite(weekly)
	assert known.any() and (~known).any()
	assert np.array_equal(got[known], weekly[known].astype(np.float32)) and np.isnan(got[~known]).all()


def end_credit_of(config, trust, check=1.0):
	learner = SimpleNamespace(config=config, end_check=check, grade_trust=lambda: {"Vend": trust})
	return CampaignLearner.end_credit(learner)


def test_end_credit_off_is_the_trust_driven_share_and_fixed_is_the_setting():
	for trust in (0.0, 0.3, 1.0):
		assert end_credit_of(CampaignConfig(), trust) == (1.0 * trust, 1 - (1 - 0.25) * trust)
		assert end_credit_of(CampaignConfig(end_week_share=1.0), trust) == (1.0 * trust, 1.0)
		assert end_credit_of(CampaignConfig(end_week_share=0.5), trust) == (1.0 * trust, 0.5)
	assert end_credit_of(CampaignConfig(end_credit_check=True), 0.8, check=0.4) == (0.8, 1 - 0.75 * 0.4)  # (off: the check as before)


def test_an_update_at_the_week_alone_credits_only_rows_with_a_week_and_logs_the_rest():
	"""A game stopped by the time limit (no real end): the first four rows have a decision a week on, the rest do not.
	Off, the one-step part credits seven rows (the last has no next decision); at 1, the four with a week."""
	torch.manual_seed(6)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	x = random_inputs(8, seed=11)
	decisions = decisions_for(model.eval(), x)
	targets = np.full((8, c.N_VALUE_OFFSETS), np.nan, dtype=np.float32)
	seven = np.full(8, np.nan, dtype=np.float32)
	end = {
		"targets": np.full(8, 0.5, dtype=np.float32), "real": np.full(8, np.nan, dtype=np.float32), "step": np.array([1, 2, 3, 4, 5, 6, 7, -1]),
		"week": np.array([4, 5, 6, 7, -1, -1, -1, -1]), "price": np.zeros(8, dtype=np.float32), "weight": 0.5, "outcome": 1.0,
	}
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	off, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-5), decisions, targets, None, CampaignConfig(batch_size=4, passes=1), decode=decode, advantages=seven, end=end)
	assert off["credited"] == 7 and off["end_credit_none"] == pytest.approx(1 / 8)
	week, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-5), decisions, targets, None, CampaignConfig(batch_size=4, passes=1, end_week_share=1.0), decode=decode, advantages=seven, end=end)
	assert week["credited"] == 4 and week["end_credit_none"] == pytest.approx(0.5)


def test_the_columns_are_logged():
	assert {"end_week_share", "end_credit_none"} <= set(COLUMNS)
