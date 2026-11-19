"""The credit of four points (config.credit_horizons; e5 for the user, 2026-11-19: "ensuring equal effects is the
minimum"): each horizon's terms against today's 7-day credit, the calibrated baselines (config.credit_calibrated), the
four points' mix and its components, the update's readings (each point's share of the policy term's gradient)."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.model_v2 import CampaignModelV2
from campaign_model.tests.test_credit_segments import labels
from campaign_model.tests.test_model_v2 import decisions_for, rich_inputs
from campaign_model.train import GROWTH_CAP, SHARE_CHANGE_CAP, calibrated_baseline, four_point, horizon_terms, make_optimizer, ppo_update, share_baseline

CONFIG = CampaignConfig(batch_size=4, passes=1, model_architecture="v2", credit="schedule", share_target="change")


def test_the_calibrated_baseline_is_least_squares_with_beta_in_0_1_and_never_adds_variance():
	g = torch.Generator().manual_seed(1)
	f = torch.randn(4000, generator=g)
	real = 0.3 * f + torch.randn(4000, generator=g)
	b, alpha, beta = calibrated_baseline(real, f)
	assert beta == pytest.approx(0.3, abs=0.05) and float((real - b).var()) <= float(real.var())
	wide = 4.0 * f  # (a forecast swinging 4x: the trust's 0.4 would add variance, the fit does not)
	_, _, beta_wide = calibrated_baseline(real, wide)
	assert beta_wide == pytest.approx(0.3 / 4.0, abs=0.02)
	assert float((real - 0.4 * wide).var()) > float(real.var()) > float((real - calibrated_baseline(real, wide)[0]).var())
	b_anti, alpha_anti, beta_anti = calibrated_baseline(real, -f)
	assert beta_anti == 0.0 and torch.allclose(b_anti, torch.full_like(real, float(real.mean())))
	real_nan = real.clone()
	real_nan[:10] = float("nan")
	b_nan, _, _ = calibrated_baseline(real_nan, f)
	assert torch.isfinite(b_nan).all()


def test_uncalibrated_the_7_day_point_is_todays_credit_and_the_readings_are_there():
	n = 40
	targets, growth = labels(n, seed=3)
	targets[5, 1:] = np.nan
	growth[6, 2] = np.nan
	value_k, growth_k = torch.randn(n, c.N_VALUE_OFFSETS), torch.randn(n, c.N_VALUE_OFFSETS)
	log_phi = torch.randn(n) * 0.1
	share, grow, stats = horizon_terms(targets, growth, value_k, growth_k, log_phi, 0.4)
	real7 = torch.from_numpy(targets[:, -1]) * 2 * SHARE_CHANGE_CAP - SHARE_CHANGE_CAP
	predicted7 = torch.sigmoid(value_k[:, -1]) * 2 * SHARE_CHANGE_CAP - SHARE_CHANGE_CAP
	today_share = real7 - share_baseline(real7, predicted7, 0.4)
	today_grow = (torch.from_numpy(growth[:, -1]) - log_phi).clamp(-GROWTH_CAP, GROWTH_CAP) - (torch.sigmoid(growth_k[:, -1]) * 2 * GROWTH_CAP - GROWTH_CAP)
	both = torch.isfinite(today_share)
	assert torch.allclose(share[:, 2][both], today_share[both], atol=1e-5) and torch.equal(torch.isfinite(share[:, 2]), both)
	known = torch.isfinite(today_grow)
	assert torch.allclose(grow[:, 2][known], today_grow[known], atol=1e-5)
	assert not torch.isfinite(share[5, 1:]).any() and torch.isfinite(share[5, 0])
	for name in ("share", "growth"):
		for h in (2, 5, 7):
			for what in ("alpha", "beta", "var_ratio", "rho", "s"):
				assert f"credit_{name}{h}_{what}" in stats
	assert stats["credit_share7_beta"] == 0.4 and stats["credit_growth7_beta"] == 1.0 and stats["credit_growth7_alpha"] == 0.0
	_, _, calibrated = horizon_terms(targets, growth, value_k, growth_k, log_phi, 0.4, calibrated=True)
	assert all(0.0 <= calibrated[f"credit_{t}{h}_beta"] <= 1.0 for t in ("share", "growth") for h in (2, 5, 7))
	assert all(calibrated[f"credit_{t}{h}_var_ratio"] <= 1.0 + 1e-4 for t in ("share", "growth") for h in (2, 5, 7))


def test_four_points_each_standardized_at_their_weights_rows_keep_their_known_points():
	g = torch.Generator().manual_seed(2)
	parts = torch.randn(50, 3, generator=g) * torch.tensor([1.0, 3.0, 10.0])  # (three spreads: the points enter alike)
	parts[0, 1:] = float("nan")  # (near the game's end: the 2-day point alone)
	parts[1] = float("nan")
	end = np.random.default_rng(3).standard_normal(50).astype(np.float32) * 5
	end[1] = np.nan  # (no point at all: no credit)
	advantage, components = four_point(parts, end, (0.25, 0.25, 0.25, 0.25))
	assert not torch.isfinite(advantage[1]) and torch.isfinite(advantage[0])
	z2 = (parts[:, 0][torch.isfinite(parts[:, 0])] - parts[:, 0][torch.isfinite(parts[:, 0])].mean()) / (parts[:, 0][torch.isfinite(parts[:, 0])].std() + 1e-8)
	assert torch.isclose(advantage[0], (0.25 * z2[0] + 0.25 * components[0, 3] / components[0, 3] * float((end[0] - np.nanmean(end)) / np.nanstd(end, ddof=1))) / 0.5, atol=1e-3)
	rows = torch.isfinite(advantage)
	assert torch.allclose(components[rows].sum(dim=1), advantage[rows], atol=1e-6)
	full = rows & torch.isfinite(parts).all(dim=1)
	spreads = components[full].std(dim=0)
	assert float(spreads.max() / spreads.min()) < 1.5  # (equal weights: the points' terms alike in spread whatever their units)
	only_two, _ = four_point(parts, end, (1.0, 0.0, 0.0, 0.0))
	z = torch.full((50,), float("nan"))
	known = torch.isfinite(parts[:, 0])
	z[known] = z2
	assert torch.allclose(only_two[known], z[known], atol=1e-5)


def update(model, decisions, x, config):
	torch.manual_seed(7)
	m = copy.deepcopy(model)
	optimizer = make_optimizer(m, config)
	n = len(decisions)
	targets, growth = labels(n)
	metrics, _ = ppo_update(m, optimizer, decisions, targets, None, config, decode=lambda rows: {k: v[rows] for k, v in x.items()}, advantages=np.zeros(n, np.float32),
		growth=growth, growth_real=growth, labels=targets)
	return m, metrics


@pytest.fixture(scope="module")
def setup():
	torch.manual_seed(5)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	x = rich_inputs(8, seed=6)
	return model, x, decisions_for(model.eval(), x)


def test_off_changes_nothing_and_on_reports_each_points_gradient_share(setup):
	model, x, decisions = setup
	plain, off = update(model, decisions, x, CONFIG)
	assert not any(k.startswith("credit_") for k in off)
	moved, on = update(model, decisions, x, replace(CONFIG, credit_horizons=True))
	shares = [on[f"credit_grad_share_{p}"] for p in ("2", "5", "7", "end")]
	assert all(np.isfinite(s) and s >= 0 for s in shares) and sum(shares) == pytest.approx(1.0)
	assert shares[3] == 0.0  # (no end labels here: the end point is never known)
	assert {"credit_share2_var_ratio", "credit_growth7_beta", "credit_point_2_sd", "credit_point_7_known"} <= set(on)
	assert any(not torch.equal(a, b) for (_, a), (_, b) in zip(plain.named_parameters(), moved.named_parameters()))
	_, calibrated = update(model, decisions, x, replace(CONFIG, credit_horizons=True, credit_calibrated=True))
	assert all(0.0 <= calibrated[f"credit_share{h}_beta"] <= 1.0 for h in (2, 5, 7))


def test_the_credit_gate_prints_every_term_and_stops_an_uncalibrated_baseline_that_adds_variance(setup, capsys):
	"""(e5, 2026-11-19) the policy's first update, before any step: rho, s and the variance ratio of the six terms;
	uncalibrated, a ratio over credit_gate_ratio stops the run; calibrated, it cannot (its ratio is at most 1)."""
	from campaign_model.train import CreditGateError, credit_gate_check

	stats = {f"credit_{t}{h}_{w}": 0.5 for t in ("share", "growth") for h in (2, 5, 7) for w in ("rho", "s", "var_ratio", "beta")}
	credit_gate_check(stats, calibrated=False)
	assert capsys.readouterr().out.count(" rho ") == 6
	stats["credit_growth5_var_ratio"] = 11.6
	with pytest.raises(CreditGateError, match="growth5"):
		credit_gate_check(stats, calibrated=False)
	credit_gate_check(stats, calibrated=True)  # (printed, not stopped)
	model, x, decisions = setup
	torch.manual_seed(7)
	m = copy.deepcopy(model)
	n = len(decisions)
	targets, growth = labels(n)
	before = [p.detach().clone() for p in m.parameters()]
	config = replace(CONFIG, credit_horizons=True, credit_gate_ratio=0.0)  # (any ratio over 0: the gate stops it)
	with pytest.raises(CreditGateError):
		ppo_update(m, make_optimizer(m, config), decisions, targets, None, config, decode=lambda rows: {k: v[rows] for k, v in x.items()}, advantages=np.zeros(n, np.float32),
			growth=growth, growth_real=growth, labels=targets, credit_gate=True)
	assert all(torch.equal(a, b) for a, b in zip(before, m.parameters()))  # (before any step)
