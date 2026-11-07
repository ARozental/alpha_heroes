"""docs/optimizer_plan.md (2026-11-06): the optimizer's named groups (train.parameter_groups, make_optimizer), moments by
name across groupings (load_optimizer), the value heads' own rate, clipping by group (battle_model.optim), the signal
share at any beta1, the heads' weights by residual (train.head_weights), and the learner's trust floor, frozen loss
shares and clip calibration."""
import copy

import numpy as np
import pytest
import torch

from battle_model.optim import calibrated_clips, clip_gradients, clip_settings, signal_fraction, stale_clips
from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.learner import CampaignLearner
from campaign_model.model_v2 import CampaignModelV2
from campaign_model.tests.test_model_v2 import legacy_optimizer_state
from campaign_model.train import GRAD_GROUPS, VALUE_MODULES, group_lr, head_weights, load_optimizer, make_optimizer, parameter_groups, ppo_update, running


@pytest.fixture(scope="module")
def model():
	torch.manual_seed(0)
	return CampaignModelV2(dim=32, cell_dim=8, layers=1)


def test_the_groups_hold_every_parameter_once_at_their_settings(model):
	"""Each parameter in one group; the value heads' modules all in "value" at value_lr_factor x the rate, beta1 0.9; the
	corrections alone decayed; the trunk's, the embeddings' and the type bias' at trunk_beta1; the groups name their
	parameters (torch keeps "param_names" in the checkpoint)."""
	config = CampaignConfig(learning_rate=1e-5, value_lr_factor=10.0, trunk_beta1=0.95, prior_decay=5.0)
	optimizer = make_optimizer(model, config)
	groups = {g["name"]: g for g in optimizer.param_groups}
	assert list(groups) == list(GRAD_GROUPS)
	ids = [id(p) for g in optimizer.param_groups for p in g["params"]]
	assert len(ids) == len(set(ids)) == len(list(model.parameters()))
	value = {id(p) for name in VALUE_MODULES for p in getattr(model, name).parameters()}
	assert {id(p) for p in groups["value"]["params"]} == value
	assert {id(p) for p in groups["corrections"]["params"]} == {id(p) for p in model.prior_corrections()}
	assert [n for n, _ in parameter_groups(model)["type"]] == ["type_bias"]
	assert all(n.endswith("weight") for n in groups["embeddings"]["param_names"])
	assert groups["value"]["lr"] == pytest.approx(1e-4) and groups["trunk"]["lr"] == pytest.approx(1e-5)
	assert {name: g["betas"][0] for name, g in groups.items()} == {"trunk": 0.95, "embeddings": 0.95, "type": 0.95, "value": 0.9, "corrections": 0.9, "privileged": 0.9}
	assert {name: g["weight_decay"] for name, g in groups.items()} == {"trunk": 0.0, "embeddings": 0.0, "type": 0.0, "value": 0.0, "corrections": 5.0, "privileged": 0.0}
	assert groups["privileged"]["lr"] == pytest.approx(1e-4)  # (the critic's timescale)
	names = dict(model.named_parameters())
	for g in optimizer.param_groups:
		assert [names[n] for n in g["param_names"]] == g["params"]


def test_the_value_groups_rate_is_its_own():
	"""The KL control's lr_scale moves the policy's groups; the value heads' rate stays learning_rate x value_lr_factor
	(their parameters never move the policy), unless value_lr_kl."""
	config = CampaignConfig(learning_rate=1e-5, value_lr_factor=10.0)
	assert group_lr({"name": "value"}, config, 0.3) == pytest.approx(1e-4)
	assert group_lr({"name": "trunk"}, config, 0.3) == pytest.approx(3e-6)
	assert group_lr({"name": "value"}, CampaignConfig(learning_rate=1e-5, value_lr_factor=10.0, value_lr_kl=True), 0.3) == pytest.approx(3e-5)


def test_a_checkpoint_of_two_unnamed_groups_resumes_named_at_the_configs_settings(model):
	"""A checkpoint of before 2026-11-06 (two groups, no names, beta1 0.9 everywhere): every moment follows its parameter
	by name into the five groups, the step counts too, and the groups' settings are the config's (load_state_dict alone
	would have kept the checkpoint's beta1 0.9 and its one rate)."""
	old = make_optimizer(model, CampaignConfig(trunk_beta1=0.9, value_lr_factor=1.0))
	torch.manual_seed(1)
	for _ in range(3):
		for p in model.parameters():
			p.grad = torch.randn_like(p)
		old.step()
	saved = legacy_optimizer_state(model, old)
	assert [len(g["params"]) for g in saved["param_groups"]] == [len(list(model.parameters())) - len(model.prior_corrections()), len(model.prior_corrections())]
	config = CampaignConfig()
	optimizer = make_optimizer(model, config)
	load_optimizer(optimizer, model, saved, model.state_dict().keys())
	for p in model.parameters():
		for key in ("exp_avg", "exp_avg_sq", "step"):
			assert torch.equal(optimizer.state[p][key], old.state[p][key])
	assert [(g["name"], g["lr"], g["betas"], g["weight_decay"]) for g in optimizer.param_groups] == [(g["name"], g["lr"], g["betas"], g["weight_decay"]) for g in make_optimizer(model, config).param_groups]
	# and a named checkpoint (today's) round-trips the same, whatever its groups' settings were
	again = make_optimizer(model, config)
	load_optimizer(again, model, optimizer.state_dict(), model.state_dict().keys())
	for p in model.parameters():
		assert torch.equal(again.state[p]["exp_avg"], old.state[p]["exp_avg"])


def grads_like(model, scale=None):
	torch.manual_seed(2)
	for name, p in model.named_parameters():
		p.grad = torch.randn_like(p) * (scale(name) if scale else 1.0)


def test_without_groups_the_clip_is_torchs(model):
	"""clips None: the one clip at grad_clip over every group, as torch's clip_grad_norm_ (before 2026-11-06)."""
	optimizer = make_optimizer(model, CampaignConfig())
	grads_like(model)
	twin = copy.deepcopy(model)
	for p, q in zip(model.parameters(), twin.parameters()):
		q.grad = p.grad.clone()
	expected = torch.nn.utils.clip_grad_norm_(twin.parameters(), 0.5)
	total, norms, bound = clip_gradients(optimizer, 0.5, None, GRAD_GROUPS)
	assert float(total) == pytest.approx(float(expected), rel=1e-5) and float(norms.norm()) == pytest.approx(float(total), rel=1e-5)
	assert bool(bound.all())
	for p, q in zip(model.parameters(), twin.parameters()):
		torch.testing.assert_close(p.grad, q.grad)


def test_a_spike_in_one_group_leaves_the_others_gradient(model):
	"""Clipped by group: each group at its own clip, then all x grad_clip / typical; a group ten times its usual norm is
	cut to its clip, the others' gradients are what they were (the one clip shrank them all on such a step)."""
	optimizer = make_optimizer(model, CampaignConfig())
	groups = {g["name"]: g for g in optimizer.param_groups}
	grads_like(model)
	raw = {name: float(torch.nn.utils.get_total_norm([p.grad for p in g["params"]])) for name, g in groups.items()}
	clips = {"typical": 10.0, **{name: 2 * norm for name, norm in raw.items()}}  # (every group under its clip)
	before = {name: [p.grad.clone() for p in g["params"]] for name, g in groups.items()}
	_, norms, bound = clip_gradients(optimizer, 0.5, clips, GRAD_GROUPS)
	assert not bool(bound.any())
	for name, g in groups.items():
		for a, p in zip(before[name], g["params"]):
			torch.testing.assert_close(p.grad, a * 0.05)  # (0.5 / 10)
	grads_like(model, lambda name: 10.0 if name.startswith(VALUE_MODULES) else 1.0)  # the value heads spike
	_, norms, bound = clip_gradients(optimizer, 0.5, clips, GRAD_GROUPS)
	assert bound.tolist() == [float(name == "value") for name in GRAD_GROUPS]
	assert float(norms[GRAD_GROUPS.index("value")]) == pytest.approx(10 * raw["value"], rel=1e-4)
	assert float(torch.nn.utils.get_total_norm([p.grad for p in groups["value"]["params"]])) == pytest.approx(clips["value"] * 0.05, rel=1e-4)
	for name in ("trunk", "embeddings", "corrections"):
		for a, p in zip(before[name], groups[name]["params"]):
			torch.testing.assert_close(p.grad, a * 0.05)


def test_the_clips_calibrate_to_the_medians_and_parse():
	samples = [[10.0 + i, 0.3, 0.1, 0.0, 8.0 + i, 5.0, 0.2] for i in range(5)]
	clips = calibrated_clips(samples, GRAD_GROUPS)
	assert clips == {"typical": 12.0, "trunk": 0.3, "embeddings": 0.1, "type": float("inf"), "value": 10.0, "corrections": 5.0, "privileged": 0.2}
	assert clip_settings(("trunk:0.4", "value:9", "typical:12")) == {"trunk": 0.4, "value": 9.0, "typical": 12.0}
	assert clip_settings(()) == {}
	with pytest.raises(ValueError):
		clip_settings(("trunk:0.4",))


@pytest.mark.parametrize("beta1", [0.9, 0.95])
def test_the_signal_fraction_reads_noise_as_0_and_a_steady_gradient_as_1_at_any_beta1(beta1):
	"""Adam's moments after 3,000 steps of pure noise and of a steady gradient: the signal fraction ~0 and ~1 at either
	beta1, where the median |m| / sqrt(v) of pure noise moves with beta1 (0.155 at 0.9, 0.108 at 0.95)."""
	torch.manual_seed(3)
	noise, steady = torch.nn.Parameter(torch.zeros(4000)), torch.nn.Parameter(torch.zeros(4000))
	optimizer = torch.optim.Adam([{"params": [noise], "name": "noise", "betas": (beta1, 0.999)}, {"params": [steady], "name": "steady", "betas": (beta1, 0.999)}], lr=0.0)
	for _ in range(3000):
		noise.grad, steady.grad = torch.randn(4000), torch.full((4000,), 0.5) + 0.05 * torch.randn(4000)
		optimizer.step()
	fraction = signal_fraction(optimizer)
	assert abs(fraction["noise"]) < 0.03 and fraction["steady"] > 0.95
	assert fraction["all"] == pytest.approx((fraction["noise"] + fraction["steady"]) / 2, abs=1e-6)
	state = optimizer.state[noise]
	ratio = (state["exp_avg"] / (1 - beta1**3000)).abs() / (state["exp_avg_sq"] / (1 - 0.999**3000)).sqrt()
	assert float(ratio.median()) == pytest.approx(0.674 * ((1 - beta1) / (1 + beta1)) ** 0.5, rel=0.1)


VAR = {"share": [0.0004, 0.0007, 0.0009], "growth": [0.0004, 0.0007, 0.0008], "end": [0.0117]}


def test_the_heads_weights_by_residual_keep_the_total_and_follow_the_fit():
	"""head_weights' residuals: each head's weight goes as its old weight over its residual, scaled so the heads' losses
	at a constant prediction add up as at the old weights; with every head's residual its variance times the same
	share, the variance's weights; a head that fits twice as well, twice its weight against the rest."""
	alike = {name: [0.8 * v for v in values] for name, values in VAR.items()}
	by_variance, by_residual = head_weights(VAR, 1.0, 0.5, eps=0.0), head_weights(VAR, 1.0, 0.5, residuals=alike)
	for name in VAR:
		assert by_residual[name] == pytest.approx(by_variance[name], rel=1e-6)
	better = copy.deepcopy(alike)
	better["share"][0] /= 2
	w = head_weights(VAR, 1.0, 0.5, residuals=better)
	total = lambda weights: sum(x * v for name in VAR for x, v in zip(weights[name], VAR[name]))  # noqa: E731
	assert total(w) == pytest.approx(total({"share": [1.0] * 3, "growth": [0.5] * 3, "end": [1.0]}))
	assert w["share"][0] / w["share"][1] == pytest.approx(2 * by_variance["share"][0] / by_variance["share"][1], rel=1e-6)
	assert head_weights(VAR, 1.0, 0.5, residuals={**alike, "end": [0.0]})["end"][0] < 1e4  # (a head's residual floored)


def test_running_blends_list_by_list():
	assert running(None, {"a": [1.0, 2.0]}, 0.9) == {"a": [1.0, 2.0]}
	assert running({"a": [0.0, 0.0], "b": [5.0]}, {"a": [1.0, 2.0]}, 0.9) == {"a": [pytest.approx(0.1), pytest.approx(0.2)]}
	assert running({"a": [0.0]}, {"a": [1.0, 2.0]}, 0.9) == {"a": [1.0, 2.0]}
	assert running({"a": [3.0, 0.0]}, {"a": [float("nan"), 1.0]}, 0.9) == {"a": [3.0, pytest.approx(0.1)]}  # (not measured: kept)
	partial = head_weights(VAR, 1.0, 0.5, residuals={**VAR, "end": [float("nan")]})
	assert partial["end"] == [1.0] and all(w == w for w in partial["share"])  # (an unknown residual: that head as before scaling)


def test_an_update_moves_the_value_heads_at_their_rate_and_reports_the_plans_readings():
	"""ppo_update with make_optimizer's groups: Adam's first steps move each weight by ~its rate, the value heads' 10x
	the trunk's; the KL target is target_kl (no teacher inflation); the update reports each group's gradient norm and
	clip, the signal fraction, each head's fresh excess, the readouts' rms and (head_scale_by "residual") the running
	residuals and slow variances."""
	from campaign_model.tests.test_model_v2 import decisions_for, rich_inputs

	torch.manual_seed(5)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	x = rich_inputs(8, seed=6)
	decisions = decisions_for(model.eval(), x)
	n = len(decisions)
	rng = np.random.default_rng(0)
	targets = (0.5 + 0.05 * rng.standard_normal((n, c.N_VALUE_OFFSETS))).astype(np.float32)
	growth = (10.0 + rng.standard_normal((n, c.N_VALUE_OFFSETS))).astype(np.float32)
	config = CampaignConfig(batch_size=4, passes=1, model_architecture="v2", learning_rate=1e-5, value_lr_factor=10.0, target_kl=0.015, head_scale=True)
	optimizer = make_optimizer(model, config)
	before = {name: p.detach().clone() for name, p in model.named_parameters()}
	metrics, _ = ppo_update(model, optimizer, decisions, targets, np.zeros(n, np.float32), config, decode=lambda rows: {k: v[rows] for k, v in x.items()}, growth=growth, growth_real=growth,
		head_scale={"var": None, "ramp": 1.0, "residual": None, "var_slow": None, "residual_ramp": 0.5}, loss_weights={"value": 3.0, "foresight": 0.5, "kl_teacher": 3.0}, teacher_weight=1.0)
	moved = {name: float((p.detach() - before[name]).abs().max()) for name, p in model.named_parameters()}
	value = max(moved[name] for name in moved if name.startswith(VALUE_MODULES))
	trunk = max(moved[name] for name in moved if name.startswith(("blocks.", "again_block.")))
	steps = metrics["steps"]
	assert 0.5e-4 < value <= steps * 1e-4 * 1.01 and 0.5e-5 < trunk <= steps * 1e-5 * 1.01  # (a first step under lr where a gradient nears Adam's eps)
	assert metrics["kl_target"] == pytest.approx(0.015)
	for name in GRAD_GROUPS:
		assert metrics[f"grad_norm_{name}"] >= 0 and metrics[f"clip_bound_{name}"] in (0.0, 0.5, 1.0)
	assert len(metrics["grad_norms"]) == steps and len(metrics["grad_norms"][0]) == 1 + len(GRAD_GROUPS)
	assert set(metrics["signal_fraction"]) == set(GRAD_GROUPS) - {"privileged"} | {"all"}  # (the privileged critic learns from rivals only: none here)
	assert len(metrics["head_fresh"]["share"]) == c.N_VALUE_OFFSETS and all(v >= 0 for v in metrics["head_fresh"]["share"])
	assert metrics["head_residual"]["share"] == pytest.approx(metrics["head_fresh"]["share"])  # (its first update: the batch's)
	assert metrics["head_var_slow"]["share"] == pytest.approx(metrics["head_var"]["share"])
	assert metrics["value_readout_rms"] > 0 and metrics["value_learning_rate"] == pytest.approx(1e-4)
	assert metrics["kl_first"] == pytest.approx(0.0, abs=1e-6)  # (no step before the first: the anchor's weights, no memory unrolled here)
	assert steps < 3 or np.isfinite(metrics["kl_growth"])
	import dataclasses

	advised = [dataclasses.replace(d, advice=(d.action_type, d.index, d.cell)) for d in decisions]  # (a teacher's pull: the inflation's case)
	for inflate, target in ((False, 0.015), (True, 0.015 * 4)):
		setting = CampaignConfig(batch_size=4, passes=1, model_architecture="v2", kl_teacher_inflation=inflate)
		twin = copy.deepcopy(model)
		metrics, _ = ppo_update(twin, make_optimizer(twin, setting), advised, targets, np.zeros(n, np.float32), setting, decode=lambda rows: {k: v[rows] for k, v in x.items()}, loss_weights={"value": 3.0, "foresight": 0.5, "kl_teacher": 3.0}, teacher_weight=1.0)
		assert metrics["kl_target"] == pytest.approx(target)


def learner(config, **state):
	"""A CampaignLearner with just the state its trust, loss shares and clips read."""
	one = CampaignLearner.__new__(CampaignLearner)
	one.config, one.version, one.value_ev, one.value_pool, one.trust_ema = config, 100, None, None, None
	one.teacher_weight, one.teacher_share, one.loss_scale = 2.0, 0.6, None
	one.grad_clips, one.clip_samples = None, []
	for name, value in state.items():
		setattr(one, name, value)
	return one


def test_trust_is_a_floored_slow_average():
	"""trust: max(trust_floor, its average at trust_memory a value-fit update); 0 and 0: today's value (as before)."""
	from campaign_model.labels import residual_variances

	config = CampaignConfig(trust_floor=0.4, trust_memory=0.98, trust_full_ev=0.4)
	one = learner(config)
	assert one.trust() == 0.0  # (nothing measured yet: no ground for the floor either)
	rng = np.random.default_rng(0)

	def fit(ev):  # value_parts_d5 and d7 whose explained variance is ev
		actual = rng.standard_normal(20000)
		predicted = ev * actual + np.sqrt(ev * (1 - ev)) * rng.standard_normal(20000)
		return {f"value_parts_d{k}": residual_variances(predicted, actual) for k in c.VALUE_OFFSETS}

	one._note_value_ev(fit(0.3))
	today = one.trust_today()
	assert today == pytest.approx(0.75, abs=0.05) and one.trust() == pytest.approx(today)  # (its first measurement starts the average)
	for _ in range(30):
		one._note_value_ev(fit(0.0))
	ema = today * 0.98**30 + sum(0.02 * 0.98**k * 0.0 for k in range(30))
	assert one.trust_today() < 0.1  # (the pooled fit has collapsed)
	assert one.trust_ema > ema and one.trust() == pytest.approx(max(0.4, one.trust_ema))  # (its average falls slowly, from above today's values)
	assert one.trust() >= 0.4  # (and floored)
	assert learner(CampaignConfig(trust_floor=0.0, trust_memory=0.0), value_ev=[0.1, 0.2]).trust() == pytest.approx(0.25)


def test_the_loss_shares_freeze():
	"""From config.loss_shares_freeze's version the weights stay as they are: measured and logged, not moved."""
	metrics = {"grad_raw_policy": 1.0, "grad_raw_value": 0.001, "grad_raw_teacher": 0.5, "grad_raw_foresight": 0.2}
	moving = learner(CampaignConfig(loss_shares=True), version=100)
	moving._note_gradients(dict(metrics))
	assert moving.loss_scale["weight"] != {"value": 1.0, "teacher": 2.0, "foresight": 1.0}
	frozen = learner(CampaignConfig(loss_shares=True, loss_shares_freeze=100), version=100)
	frozen._note_gradients(dict(metrics))
	assert frozen.loss_scale["weight"] == {"value": 1.0, "teacher": 2.0, "foresight": 1.0}
	assert frozen.loss_scale["log_norm"]["value"] == pytest.approx(np.log(0.001))
	before = learner(CampaignConfig(loss_shares=True, loss_shares_freeze=101), version=100)
	before._note_gradients(dict(metrics))
	assert before.loss_scale["weight"] == moving.loss_scale["weight"]


def test_the_clips_calibrate_once_and_stay():
	one = learner(CampaignConfig(grad_clip_calibration=4))
	one._calibrate_clips({"grad_norms": [[10.0, 0.3, 0.1, 0.0, 8.0, 5.0, 0.2]] * 3})
	assert one.grad_clips is None and len(one.clip_samples) == 3
	one._calibrate_clips({"grad_norms": [[12.0, 0.5, 0.1, 0.0, 9.0, 5.0, 0.2]] * 3})
	assert one.grad_clips["typical"] == 11.0 and one.grad_clips["trunk"] == 0.4 and one.grad_clips["type"] == float("inf")
	one._calibrate_clips({"grad_norms": [[100.0] * 7] * 10})
	assert one.grad_clips["typical"] == 11.0  # (kept: re-measured at a deploy only)
	assert one.grad_clips["version"] == one.version
	assert not stale_clips(one.grad_clips, 0) and not stale_clips(one.grad_clips, one.version) and stale_clips(one.grad_clips, one.version + 1)
	assert stale_clips({"typical": 1.0}, 5) and not stale_clips(None, 5)  # (clips of before the switch: version 0)
	off = learner(CampaignConfig(grad_clip_by_group=False, grad_clip_calibration=1))
	off._calibrate_clips({"grad_norms": [[1.0] * 7]})
	assert off.grad_clips is None and off.clip_samples == []


def test_moments_of_another_shape_start_fresh():
	"""A parameter saved under the same name at another shape (a part rebuilt: model.REPLACED_HEADS) gets fresh moments
	(moments_by_name), never the old ones: Adam's step would fail on them or mix them in."""
	from battle_model.optim import moments_by_name

	old, new = torch.nn.Linear(3, 2), torch.nn.Linear(4, 2)
	saved = torch.optim.AdamW(old.parameters(), lr=1e-3)
	old(torch.randn(5, 3)).sum().backward()
	saved.step()
	kept = saved.state[old.bias]["exp_avg"].clone()  # (the loaded moments may share its storage)
	optimizer = torch.optim.AdamW(new.parameters(), lr=1e-3)
	fresh, gone = moments_by_name(optimizer, new, saved.state_dict(), ["weight", "bias"])
	assert fresh == ["weight"] and gone == []
	new(torch.randn(5, 4)).sum().backward()
	optimizer.step()  # (no shape error)
	torch.testing.assert_close(optimizer.state[new.bias]["exp_avg"], kept * 0.9 + 0.1 * new.bias.grad)  # (the bias's kept)
	assert optimizer.state[new.weight]["exp_avg"].shape == new.weight.shape
