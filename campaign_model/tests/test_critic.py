"""The separate critic network (config.critic_net, campaign_model/critic.py; the user, 2026-11-18): off, the model as
before; on, the policy's choices the same at the same weights, the two networks' losses isolated from each other, no
critic weight in what is served, the NK2 pool's rows reaching the critic alone, and its loss falling on its rows."""
import copy

import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.critic import Critic, critic_kwargs, critic_optimizer, rival_extra
from campaign_model.model import build_model
from campaign_model.model_v2 import PRIVILEGED_RIVALS, CampaignModelV2
from campaign_model.tests.test_model_v2 import rich_inputs
from campaign_model.tests.test_train import decisions_for
from campaign_model.train import GROWTH_CAP, critic_loss, model_kwargs, ppo_update

SMALL = dict(model_architecture="v2", model_dim=32, model_cell_dim=8, model_layers=1, share_target="change")


def tensors(x):
	return {k: torch.as_tensor(v) for k, v in x.items()}


def test_off_has_no_link_map_and_link_map_comes_last():
	assert "link_map" not in model_kwargs(CampaignConfig(**SMALL))
	assert model_kwargs(CampaignConfig(**SMALL, critic_net=True))["link_map"] is True
	torch.manual_seed(1)
	off = CampaignModelV2(dim=32, cell_dim=8, layers=1, share_target="change")
	torch.manual_seed(1)
	on = CampaignModelV2(dim=32, cell_dim=8, layers=1, share_target="change", link_map=True)
	a, b = off.state_dict(), on.state_dict()
	assert set(b) - set(a) == {"link_map.weight", "link_map.bias"}
	assert all(torch.equal(a[k], b[k]) for k in a)  # (created last: every other weight's draw as before)


def test_on_the_policys_choices_are_the_same_at_the_same_weights():
	torch.manual_seed(2)
	off = CampaignModelV2(dim=32, cell_dim=8, layers=1, share_target="change").eval()
	on = CampaignModelV2(dim=32, cell_dim=8, layers=1, share_target="change", link_map=True).eval()
	on.load_state_dict({**on.state_dict(), **off.state_dict()})
	on.link_outputs = True
	x = tensors(rich_inputs(8, seed=2))
	with torch.no_grad():
		a, b = off(x), on(x)
	for k in a:
		if torch.is_tensor(a[k]):
			assert torch.equal(a[k], b[k]), k
	assert "link_pred" in b and b["link_pred"].shape == (8, 3 * 32)


def test_the_two_networks_losses_reach_their_own_weights_alone():
	cfg = CampaignConfig(**SMALL, critic_net=True)
	torch.manual_seed(3)
	policy = build_model(model_kwargs(cfg))
	policy.link_outputs = True
	critic = Critic(**critic_kwargs(cfg, model_kwargs(cfg)))
	x = tensors(rich_inputs(6, seed=3))
	# the critic's loss: no gradient on any policy weight
	link, hidden, phi = critic.trunk(x)
	r = torch.zeros(6, PRIVILEGED_RIVALS, dtype=torch.long)
	mask = torch.ones(6, PRIVILEGED_RIVALS)
	out = critic.outputs(critic.head(hidden, hidden[r], mask, torch.zeros(6), rival_extra(phi, phi[r], mask)), torch.zeros(6))
	targets = torch.full((6, c.N_VALUE_OFFSETS), 0.7)
	loss = critic_loss(out, phi, targets, torch.ones_like(targets), None, None, torch.full((6,), 0.4), torch.ones(6), None, 1.0)
	loss.backward()
	assert all(p.grad is None for p in policy.parameters())
	# (its head's output layers start at 0, as PrivilegedCritic's: the head learns at the first step, the body from the
	# second, when the head passes a gradient back)
	assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in critic.head.parameters())
	with torch.no_grad():
		critic.head.direct.weight.normal_(0.0, 0.01)
	critic.zero_grad(set_to_none=True)
	link, hidden, phi = critic.trunk(x)
	out = critic.outputs(critic.head(hidden, hidden[r], mask, torch.zeros(6), rival_extra(phi, phi[r], mask)), torch.zeros(6))
	critic_loss(out, phi, targets, torch.ones_like(targets), None, None, torch.full((6,), 0.4), torch.ones(6), None, 1.0).backward()
	assert all(p.grad is None for p in policy.parameters())
	assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in critic.body.parameters())
	critic.zero_grad(set_to_none=True)
	# the policy's loss and the link loss: no gradient on any critic weight
	target = critic.trunk(x)[0].detach()
	pout = policy(x)
	link_loss = ((pout["link_pred"] - target) ** 2).mean()
	(link_loss + pout["type_logp"].clamp(min=-1e4).sum() * 0.0 + pout["value_logit"].sum() * 0.0).backward()
	assert all(p.grad is None for p in critic.parameters())
	assert policy.link_map.weight.grad is not None and bool(policy.link_map.weight.grad.abs().sum() > 0)


def test_what_is_served_holds_no_critic_weight():
	cfg = CampaignConfig(**SMALL, critic_net=True)
	kwargs = model_kwargs(cfg)
	policy = build_model(kwargs)
	critic = Critic(**critic_kwargs(cfg, kwargs))
	served = build_model(kwargs)  # (an inference server's model: the run's model_kwargs, the policy's state loaded strict)
	served.load_state_dict(policy.state_dict())
	critic_ids = {id(p) for p in critic.parameters()}
	assert not any(id(p) in critic_ids for p in policy.parameters())
	assert not any(k.startswith(("body.", "head.")) for k in policy.state_dict())


def _update_setup(seed=4, n=8):
	cfg = CampaignConfig(**SMALL, critic_net=True, batch_size=4, passes=1, critic_batch=4, value_weight=0.0, privileged_critic=False)
	torch.manual_seed(seed)
	policy = build_model(model_kwargs(cfg))
	critic = Critic(**critic_kwargs(cfg, model_kwargs(cfg)))
	x = rich_inputs(n, seed=seed)
	decisions = decisions_for(policy.eval(), x)
	targets = np.random.default_rng(seed).uniform(0.2, 0.8, (n, c.N_VALUE_OFFSETS)).astype(np.float32)
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	none = {"rivals": np.full((n, PRIVILEGED_RIVALS), -1, np.int64), "present": np.zeros((n, PRIVILEGED_RIVALS), np.float32), "share_logit": np.zeros(n, np.float32)}
	return cfg, policy, critic, decisions, targets, decode, none


def _run(cfg, policy, critic, decisions, targets, decode, none, pool):
	torch.manual_seed(9)
	np.random.seed(9)
	optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
	crit = {"module": critic, "optimizer": critic_optimizer(critic, cfg), "rows": np.array([0, 3, 5], dtype=np.int64), **none, "pool": pool}
	metrics, _ = ppo_update(policy, optimizer, decisions, targets, None, cfg, decode=decode, advantages=np.linspace(-1, 1, len(decisions)).astype(np.float32), critic=crit)
	return metrics


def test_an_update_with_the_critic_runs_and_the_pool_reaches_the_critic_alone():
	cfg, policy, critic, decisions, targets, decode, none = _update_setup()
	p0, c0 = copy.deepcopy(policy), copy.deepcopy(critic)
	m_plain = _run(cfg, policy, critic, decisions, targets, decode, none, None)
	assert np.isfinite(m_plain["link_loss"]) and m_plain["link_loss"] > 0 and m_plain["critic_rows_own"] == 3
	assert np.isfinite(m_plain["critic_loss_own_first"]) and m_plain["critic_rows_pool"] == 0
	n = len(decisions)
	pool = {"decisions": decisions, "decode": decode, **{k: v.copy() for k, v in none.items()}, "targets": targets, "growth": np.full((n, c.N_VALUE_OFFSETS), np.nan, np.float32), "end": np.full(n, 0.3, np.float32)}
	policy2, critic2 = p0, c0
	m_pool = _run(cfg, policy2, critic2, decisions, targets, decode, none, pool)
	assert m_pool["critic_rows_pool"] == n and np.isfinite(m_pool["critic_loss_pool_first"])
	# the pool's rows: never in the policy's loss or the link loss (the same policy step either way), the critic's alone
	for (name, a), (_, b) in zip(policy.named_parameters(), policy2.named_parameters()):
		assert torch.equal(a, b), name
	assert any(not torch.equal(a, b) for a, b in zip(critic.parameters(), critic2.parameters()))


def test_the_critics_loss_falls_on_its_rows():
	cfg = CampaignConfig(**SMALL, critic_net=True)
	torch.manual_seed(5)
	critic = Critic(**critic_kwargs(cfg, model_kwargs(cfg)))
	opt = critic_optimizer(critic, cfg)
	x = tensors(rich_inputs(8, seed=5))
	targets = torch.from_numpy(np.random.default_rng(5).uniform(0.1, 0.9, (8, c.N_VALUE_OFFSETS)).astype(np.float32))
	r = torch.zeros(8, PRIVILEGED_RIVALS, dtype=torch.long)
	mask = torch.zeros(8, PRIVILEGED_RIVALS)
	losses = []
	for _ in range(30):
		_, h, phi = critic.trunk(x)
		out = critic.outputs(critic.head(h, h[r].detach(), mask, torch.zeros(8), rival_extra(phi, phi[r], mask)), torch.zeros(8))
		loss = critic_loss(out, phi, targets, torch.ones_like(targets), None, None, torch.full((8,), 0.4), torch.ones(8), None, 1.0)
		opt.zero_grad()
		loss.backward()
		opt.step()
		losses.append(float(loss))
	assert losses[-1] < losses[0] and all(np.isfinite(losses))


def test_running_out_of_gpu_memory_halves_the_critics_work_and_keeps_the_update(monkeypatch):
	"""(e5, 2026-11-18: 8 prepass chunks and a 512-row batch ran out of the GPU's memory on every update, each update's
	games dropped) The critic's prepass and its steps, out of memory once each, halve and retry the same update."""
	import campaign_model.train as T

	cfg, policy, critic, decisions, targets, decode, none = _update_setup(seed=7)
	monkeypatch.setattr(T, "CRITIC_PREPASS_CHUNKS", 8)
	monkeypatch.setattr(T, "CRITIC_OOM_FLOOR", 1)
	trunk = critic.trunk
	failed = {"prepass": False, "step": False}

	def flaky(x):
		rows = next(iter(x.values())).shape[0]
		if torch.is_grad_enabled() and not failed["step"] and rows >= 2:
			failed["step"] = True
			raise torch.OutOfMemoryError("forced (a critic step)")
		if not torch.is_grad_enabled() and not failed["prepass"] and rows >= 4:
			failed["prepass"] = True
			raise torch.OutOfMemoryError("forced (the prepass)")
		return trunk(x)

	monkeypatch.setattr(critic, "trunk", flaky)
	metrics = _run(cfg, policy, critic, decisions, targets, decode, none, None)
	assert failed["prepass"] and failed["step"]
	assert metrics["critic_oom_retries"] >= 2 and np.isfinite(metrics["link_loss"]) and np.isfinite(metrics["critic_loss_own_last"])


def test_the_critics_day_starts_are_the_same_in_big_calls_and_after_running_out_of_gpu_memory():
	"""critic_day_starts at 512 rows a call (was 64) and with the GPU's memory running out once: the same forecasts as
	at 2 rows a call."""
	import dataclasses
	import threading

	from campaign_model.empire import EmpireConfig
	from campaign_model.learner import critic_day_starts
	from campaign_model.records import Game
	from campaign_model.tests.test_labels import snapshot
	from campaign_model.tests.test_model import encoded

	cfg = CampaignConfig(**SMALL, critic_net=True)
	torch.manual_seed(11)
	policy = build_model(model_kwargs(cfg)).eval()
	critic = Critic(**critic_kwargs(cfg, model_kwargs(cfg))).eval()
	with torch.no_grad():
		critic.head.direct.weight.normal_(0.0, 0.05)
	games = []
	for g in range(2):
		x = rich_inputs(12, seed=30 + g)
		decisions = [dataclasses.replace(d, inputs=encoded(x, b), kind=int(c.DecisionKind.TURN), color=("RED", "BLUE")[b % 2], day=1 + b // 2) for b, d in enumerate(decisions_for(policy, x))]
		games.append(Game(decisions=decisions, ledgers={d: snapshot(d, 1000 + 300 * d, 1000) for d in range(1, 10)}, days=6))
	small = critic_day_starts(critic, games, threading.Lock(), EmpireConfig(), batch=2)
	big = critic_day_starts(critic, games, threading.Lock(), EmpireConfig())
	trunk, failed = critic.trunk, {"once": False}

	def flaky(x):
		if not failed["once"] and next(iter(x.values())).shape[0] >= 4:
			failed["once"] = True
			raise torch.OutOfMemoryError("forced (the day starts)")
		return trunk(x)

	critic.trunk = flaky
	oom = {"retries": 0}
	import campaign_model.train as T

	floor, T.CRITIC_OOM_FLOOR = T.CRITIC_OOM_FLOOR, 1
	try:
		halved = critic_day_starts(critic, games, threading.Lock(), EmpireConfig(), oom=oom)
	finally:
		T.CRITIC_OOM_FLOOR = floor
	assert failed["once"] and oom["retries"] >= 1
	assert any(small)

	def same(one, other, where):
		if isinstance(one, dict):
			assert one.keys() == other.keys(), where
			for key in one:
				same(one[key], other[key], f"{where}/{key}")
		elif isinstance(one, tuple):  # ((row, shares [3], growth [3], log Phi, V_end, prior) a player-day)
			assert len(one) == len(other), where
			for k, (u, v) in enumerate(zip(one, other)):
				same(u, v, f"{where}[{k}]")
		else:
			np.testing.assert_allclose(np.asarray(one, dtype=np.float64), np.asarray(other, dtype=np.float64), rtol=1e-4, atol=1e-5, err_msg=where)

	for g, (a, b, h) in enumerate(zip(small, big, halved)):
		same(a, b, f"game {g} big")
		same(a, h, f"game {g} halved")


def test_the_free_head_starts_at_the_prior_and_its_loss_reaches_the_body():
	"""(the user, 2026-11-18: the other critics, unprivileged, on the critic's body) FreeHead: every output at 0 (V_end and
	the true-share guess at the share prior); its loss, with the auxiliary, moves its own weights and the body's."""
	from campaign_model.critic import share_prior

	cfg = CampaignConfig(**SMALL, critic_net=True)
	torch.manual_seed(12)
	critic = Critic(**critic_kwargs(cfg, model_kwargs(cfg)))
	x = tensors(rich_inputs(6, seed=12))
	_, hidden, phi = critic.trunk(x)
	out = critic.free(hidden, x["g_players"])
	prior = share_prior(x["g_players"])
	assert torch.equal(out["value_logit"], torch.zeros(6, c.N_VALUE_OFFSETS)) and torch.equal(out["growth_logit"], torch.zeros(6, c.N_VALUE_OFFSETS))
	assert torch.allclose(out["end_logit"], prior) and torch.allclose(out["share_now_logit"], prior)
	targets = torch.full((6, c.N_VALUE_OFFSETS), 0.7)
	share_now = torch.tensor([0.5, -1.0, float("nan"), 0.0, 2.0, -2.0])
	growth = phi.detach().clamp(min=1.0).log().unsqueeze(-1) + torch.tensor([0.05, 0.1, 0.2])
	loss = critic_loss(out, phi, targets, torch.ones_like(targets), growth, torch.ones_like(targets), torch.full((6,), 0.4), torch.ones(6), None, 1.0, share_now=share_now)
	assert torch.isfinite(loss)
	loss.backward()
	assert all(p.grad is not None and bool(p.grad.abs().sum() > 0) for name, p in critic.free.named_parameters() if name.endswith("bias"))
	assert all(p.grad is None for p in critic.head.parameters())  # (the privileged head: not in this loss)
	critic.zero_grad(set_to_none=True)
	with torch.no_grad():
		critic.free.value_heads.weight.normal_(0.0, 0.05)
	_, hidden, phi = critic.trunk(x)
	critic_loss(critic.free(hidden, x["g_players"]), phi, targets, torch.ones_like(targets), None, None, None, None, None, 1.0).backward()
	assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in critic.body.parameters())


def test_a_checkpoint_from_before_the_free_head_resumes_with_its_moments_and_a_free_head_at_zero():
	from campaign_model.critic import load_critic

	cfg = CampaignConfig(**SMALL, critic_net=True)
	torch.manual_seed(13)
	old = Critic(**critic_kwargs(cfg, model_kwargs(cfg)), free_head=False)
	opt = critic_optimizer(old, cfg)
	assert len(opt.param_groups) == 2
	x = tensors(rich_inputs(4, seed=13))
	_, h, p = old.trunk(x)
	(h.square().mean() + old.head(h, h[torch.zeros(4, PRIVILEGED_RIVALS, dtype=torch.long)], torch.ones(4, PRIVILEGED_RIVALS), torch.zeros(4), rival_extra(p, p[torch.zeros(4, PRIVILEGED_RIVALS, dtype=torch.long)], torch.ones(4, PRIVILEGED_RIVALS))).square().mean()).backward()
	opt.step()
	saved, saved_opt = copy.deepcopy(old.state_dict()), copy.deepcopy(opt.state_dict())
	torch.manual_seed(99)
	new = Critic(**critic_kwargs(cfg, model_kwargs(cfg)))
	new, new_opt, added = load_critic(new, cfg, saved, saved_opt)
	assert added and all(k.startswith("free.") for k in added)
	for k, v in saved.items():
		assert torch.equal(new.state_dict()[k], v), k
	assert all(bool((q == 0).all()) for q in new.free.parameters())
	assert [g["name"] for g in new_opt.param_groups] == ["critic_body", "critic_head", "critic_free"]
	moved = [q for q in new.body.parameters() if q in new_opt.state and "exp_avg" in new_opt.state[q]]
	assert moved and sum(len(g["params"]) for g in new_opt.param_groups[:2]) == len(saved_opt["param_groups"][0]["params"]) + len(saved_opt["param_groups"][1]["params"])
	# (the next checkpoint, with three groups: loads as it is)
	again, again_opt, none = load_critic(Critic(**critic_kwargs(cfg, model_kwargs(cfg))), cfg, copy.deepcopy(new.state_dict()), copy.deepcopy(new_opt.state_dict()))
	assert none == [] and len(again_opt.param_groups) == 3
	# (anything else missing: refused)
	broken = {k: v for k, v in saved.items() if not k.startswith("head.")}
	try:
		load_critic(Critic(**critic_kwargs(cfg, model_kwargs(cfg))), cfg, broken, saved_opt)
		raise AssertionError("a checkpoint without the privileged head loaded")
	except RuntimeError:
		pass


def test_an_update_trains_the_free_head_and_reports_its_fit():
	cfg, policy, critic, decisions, targets, decode, none = _update_setup(seed=14)
	with torch.no_grad():
		critic.head.direct.weight.normal_(0.0, 0.01)
	before = copy.deepcopy(critic.free.state_dict())
	metrics = _run(cfg, policy, critic, decisions, targets, decode, none, None)
	assert np.isfinite(metrics["critic_unexplained_visible_own"]) and np.isfinite(metrics["critic_loss_visible_own_last"])
	assert any(not torch.equal(before[k], v) for k, v in critic.free.state_dict().items())


def test_the_critics_phi_is_the_policys_after_the_runs_ramps():
	"""(the audit of 2026-11-18) learner.mirror_empire: the critic's empire_* / phi_* buffers set to the policy's (its
	no-town factor 1.0 against the policy's 0.1 before)."""
	from campaign_model.learner import mirror_empire

	cfg = CampaignConfig(**SMALL, critic_net=True)
	policy = build_model(model_kwargs(cfg))
	critic = Critic(**critic_kwargs(cfg, model_kwargs(cfg)))
	policy.features.empire_no_town.fill_(0.1)
	policy.features.phi_hero_base.fill_(1234.0)
	copied = mirror_empire(critic.body, policy)
	assert "empire_no_town" in copied and "phi_hero_base" in copied
	for name, buffer in critic.body.features.named_buffers():
		if name.startswith(("empire_", "phi_")):
			assert torch.equal(buffer, dict(policy.features.named_buffers())[name]), name


def test_with_the_critic_the_value_readings_are_the_critics(monkeypatch):
	"""(the audit of 2026-11-18: the trust and the update line read the policy's dead heads) value_ev and
	value_unexplained from the critic's forecasts: a critic that forecasts the targets exactly explains them all, while
	the policy's heads (at 0) know nothing."""
	import campaign_model.train as T

	cfg, policy, critic, decisions, targets, decode, none = _update_setup(seed=15)
	real = T.critic_outputs

	def knowing(*args, **kwargs):
		out = real(*args, **kwargs)
		logit = torch.logit(torch.from_numpy(targets).to(out["value_logit"].device).clamp(1e-4, 1 - 1e-4))
		return {**out, "value_logit": logit, "value": torch.sigmoid(logit)}

	monkeypatch.setattr(T, "critic_outputs", knowing)
	metrics = _run(cfg, policy, critic, decisions, targets, decode, none, None)
	assert metrics["value_unexplained"] == pytest.approx(0.0, abs=1e-3)
	assert all(metrics[f"value_ev_d{k}"] == pytest.approx(1.0, abs=1e-3) for k in c.VALUE_OFFSETS)


def test_the_held_out_pool_is_read_and_never_trained_on():
	"""(e5, 2026-11-18) crit["pool_heldout"]: its fit read before the steps (critic_unexplained_pool_heldout), its rows
	never in a step: the critic moves the same with or without it."""
	cfg, policy, critic, decisions, targets, decode, none = _update_setup(seed=16)
	n = len(decisions)
	pool = {"decisions": decisions, "decode": decode, **{k: v.copy() for k, v in none.items()}, "targets": targets, "growth": np.full((n, c.N_VALUE_OFFSETS), np.nan, np.float32), "end": np.full(n, 0.3, np.float32)}
	p0, c0 = copy.deepcopy(policy), copy.deepcopy(critic)

	def run(policy_, critic_, heldout):
		torch.manual_seed(9)
		np.random.seed(9)
		optimizer = torch.optim.Adam(policy_.parameters(), lr=1e-3)
		crit = {"module": critic_, "optimizer": critic_optimizer(critic_, cfg), "rows": np.array([0, 3, 5], dtype=np.int64), **none, "pool": None, "pool_heldout": heldout}
		metrics, _ = ppo_update(policy_, optimizer, decisions, targets, None, cfg, decode=decode, advantages=np.linspace(-1, 1, n).astype(np.float32), critic=crit)
		return metrics

	with_heldout = run(policy, critic, pool)
	without = run(p0, c0, None)
	assert np.isfinite(with_heldout["critic_unexplained_pool_heldout"]) and with_heldout["critic_rows_pool_heldout"] == n
	assert "critic_unexplained_pool_heldout" not in without
	for (name, a), (_, b) in zip(critic.named_parameters(), c0.named_parameters()):
		assert torch.equal(a, b), name


def test_a_smaller_eval_plays_every_fourth_setup_of_the_48():
	"""(the user via e5, 2026-11-18) eval_own_every 4 with eval_own_games 48: the setups 0, 4, ... 44, 4 games each of 2,
	3 and 4 players, the same seeds as the 48's."""
	from campaign_model.evaluate import plan, setups

	cfg = CampaignConfig(eval_own_games=48, eval_follow_games=0, eval_own_every=4)
	arms = plan(cfg)
	assert [i for i, _ in arms] == list(range(0, 48, 4)) and {a for _, a in arms} == {"own"}
	full, small = setups(48), setups(max(i for i, _ in arms) + 1)
	assert all(small[i] == full[i] for i, _ in arms)
	counts = {n: sum(1 for i, _ in arms if full[i][1] == n) for n in (2, 3, 4)}
	assert counts == {2: 4, 3: 4, 4: 4}
	assert [i for i, _ in plan(CampaignConfig(eval_own_games=48, eval_follow_games=0))] == list(range(48))


def test_the_per_head_readings_cover_both_heads_on_our_rows_and_the_held_out_pool():
	"""(e5 for the user, 2026-11-18) critic_heads/<V2..Vend>_<own|heldout>_<privileged|free>, through the tracked set;
	critic_head_fit reads 0 for a head that forecasts its targets exactly."""
	from campaign_model.tb_filter import tracked
	from campaign_model.train import CRITIC_HEADS, critic_head_fit

	cfg, policy, critic, decisions, targets, decode, none = _update_setup(seed=17)
	n = len(decisions)
	pool = {"decisions": decisions, "decode": decode, **{k: v.copy() for k, v in none.items()}, "targets": targets, "growth": np.full((n, c.N_VALUE_OFFSETS), np.nan, np.float32), "end": np.full(n, 0.3, np.float32)}
	torch.manual_seed(9)
	np.random.seed(9)
	crit = {"module": critic, "optimizer": critic_optimizer(critic, cfg), "rows": np.array([0, 3, 5], dtype=np.int64), **none, "pool": None, "pool_heldout": pool}
	metrics, _ = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-3), decisions, targets, None, cfg, decode=decode, advantages=np.linspace(-1, 1, n).astype(np.float32), critic=crit)
	for rows in ("own", "heldout"):
		for which in ("", "visible_"):
			for head in ("V2", "V5", "V7", "total"):
				key = f"critic_heads/unexplained_{which}{head}_{rows}"
				assert key in metrics and np.isfinite(metrics[key]), key
				assert tracked(key) == key
	assert np.isfinite(metrics["critic_loss_pool_heldout"]) and np.isfinite(metrics["critic_loss_own_mean"]) and np.isfinite(metrics["critic_loss_visible_pool_heldout"])
	assert tracked("campaign_actions/dismiss") is None and tracked("campaign_actions/move") == "actions/move"
	t = torch.from_numpy(targets)
	exact = {"value_logit": torch.logit(t.clamp(1e-4, 1 - 1e-4))}
	fits = critic_head_fit(exact, torch.arange(n), torch.ones(n), t, torch.ones_like(t))
	assert set(fits) == set(CRITIC_HEADS[:3]) | {"total"} and all(abs(v) < 1e-3 for v in fits.values())


def test_the_critics_loss_is_its_excess_with_the_raw_gradient_zero_at_the_targets_and_never_below():
	"""(the user via e5, 2026-11-18: "the current loss minus the 'perfect answer' ... inside the graph") critic_loss: each
	head's cross-entropy less its targets' entropy. (a) Every critic parameter's gradient equals the raw cross-entropy's at
	the same weights and batch; (b) 0 at the targets' own logits, never below -1e-6; the pass logged, raw beside it."""
	torch.manual_seed(21)
	n = 16
	t = torch.rand(n, c.N_VALUE_OFFSETS) * 0.6 + 0.2
	phi = torch.full((n,), 1000.0)
	growth = phi.log().unsqueeze(-1) + torch.randn(n, c.N_VALUE_OFFSETS) * 0.05
	end = torch.rand(n) * 0.5 + 0.1
	ones = torch.ones_like(t)
	share_now = torch.randn(n)
	head_w = {"w:share": torch.tensor([2.4, 0.94, 0.69]), "w:growth": torch.tensor([1.48, 0.63, 0.48]), "w:end": torch.tensor([0.93])}
	cfg = CampaignConfig(**SMALL, critic_net=True)
	critic = Critic(**critic_kwargs(cfg, model_kwargs(cfg)))
	with torch.no_grad():
		for q in list(critic.head.parameters()) + list(critic.free.parameters()):
			q.normal_(0.0, 0.05)
	x = tensors(rich_inputs(n, seed=21))
	grads = []
	for raw in (True, False):
		critic.zero_grad(set_to_none=True)
		_, h, p = critic.trunk(x)
		sl = torch.zeros(n)
		r = torch.zeros(n, PRIVILEGED_RIVALS, dtype=torch.long)
		mask = torch.ones(n, PRIVILEGED_RIVALS)
		out = critic.outputs(critic.head(h, h[r], mask, sl, rival_extra(p, p[r], mask)), sl)
		g_now = p.detach().clamp(min=1.0).log().unsqueeze(-1) + torch.randn(n, c.N_VALUE_OFFSETS, generator=torch.Generator().manual_seed(3)) * 0.05
		loss = critic_loss(out, p, t, ones, g_now, ones, end, torch.ones(n), head_w, 0.5, raw=raw)
		loss = loss + critic_loss(critic.free(h, x["g_players"]), p, t, ones, g_now, ones, end, torch.ones(n), head_w, 0.5, share_now=share_now, raw=raw)
		loss.backward()
		grads.append({name: q.grad.clone() for name, q in critic.named_parameters() if q.grad is not None})
	assert grads[0].keys() == grads[1].keys() and grads[0]
	for name in grads[0]:
		torch.testing.assert_close(grads[0][name], grads[1][name], rtol=1e-5, atol=1e-7, msg=name)
	logit = lambda q: torch.logit(q.clamp(1e-6, 1 - 1e-6))  # noqa: E731
	scaled = ((growth - phi.log().unsqueeze(-1)).clamp(-GROWTH_CAP, GROWTH_CAP) + GROWTH_CAP) / (2 * GROWTH_CAP)
	exact = {"value_logit": logit(t), "growth_logit": logit(scaled), "end_logit": logit(end), "share_now_logit": share_now}
	assert abs(float(critic_loss(exact, phi, t, ones, growth, ones, end, torch.ones(n), head_w, 0.5, share_now=share_now))) < 1e-6
	for seed in range(20):
		g = torch.Generator().manual_seed(seed)
		other = {"value_logit": torch.randn(n, c.N_VALUE_OFFSETS, generator=g) * 3, "growth_logit": torch.randn(n, c.N_VALUE_OFFSETS, generator=g) * 3, "end_logit": torch.randn(n, generator=g) * 3, "share_now_logit": torch.randn(n, generator=g) * 3}
		assert float(critic_loss(other, phi, t, ones, growth, ones, end, torch.ones(n), head_w, 0.5, share_now=share_now)) > -1e-6
	cfg, policy, critic, decisions, targets, decode, none = _update_setup(seed=22)
	k = len(decisions)
	pool = {"decisions": decisions, "decode": decode, **{kk: v.copy() for kk, v in none.items()}, "targets": targets, "growth": np.full((k, c.N_VALUE_OFFSETS), np.nan, np.float32), "end": np.full(k, 0.3, np.float32)}
	crit = {"module": critic, "optimizer": critic_optimizer(critic, cfg), "rows": np.array([0, 3, 5], dtype=np.int64), **none, "pool": pool, "pool_heldout": pool}
	metrics, _ = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-3), decisions, targets, None, cfg, decode=decode, advantages=np.linspace(-1, 1, k).astype(np.float32), critic=crit)
	assert len(metrics["critic_pass_pool"]) == metrics["critic_steps_pool"] and len(metrics["critic_pass_own"]) == metrics["critic_steps_own"]
	for name in ("own", "pool"):
		assert metrics[f"critic_loss_raw_{name}_mean"] > metrics[f"critic_loss_{name}_mean"] > -1e-6
	assert np.isfinite(metrics["critic_unexplained_pool_heldout_after"]) and np.isfinite(metrics["critic_loss_pool_heldout"]) and np.isfinite(metrics["critic_loss_visible_pool_heldout"])


def test_a_thin_card_reads_the_held_out_tenth_in_smaller_calls_and_skips_the_after_read(monkeypatch):
	"""(e5, 2026-11-18) card_free_gb under CARD_FREE_THIN_GB: the held-out pass at 128 rows a call, no after-read; the
	free memory on the critic's line."""
	import campaign_model.train as T

	cfg, policy, critic, decisions, targets, decode, none = _update_setup(seed=23)
	k = len(decisions)
	pool = {"decisions": decisions, "decode": decode, **{kk: v.copy() for kk, v in none.items()}, "targets": targets, "growth": np.full((k, c.N_VALUE_OFFSETS), np.nan, np.float32), "end": np.full(k, 0.3, np.float32)}
	monkeypatch.setattr(T, "card_free_gb", lambda device: 1.0)
	chunks = []
	real = T.pool_states

	def watched(*args, **kwargs):
		chunks.append(kwargs.get("chunk", 256))
		return real(*args, **kwargs)

	monkeypatch.setattr(T, "pool_states", watched)
	crit = {"module": critic, "optimizer": critic_optimizer(critic, cfg), "rows": np.array([0, 3, 5], dtype=np.int64), **none, "pool": None, "pool_heldout": pool}
	metrics, _ = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-3), decisions, targets, None, cfg, decode=decode, advantages=np.linspace(-1, 1, k).astype(np.float32), critic=crit)
	assert metrics["card_free_gb"] == 1.0 and chunks == [128] and "critic_unexplained_pool_heldout_after" not in metrics


def test_the_critics_gradient_norms_by_source_are_on_its_line():
	"""(e5, 2026-11-18: the pool's targets are 4x wider than ours) critic_grad_own / _pool: the mean pre-clip norm of
	each source's steps, critic_clipped_*: the share clipped."""
	cfg, policy, critic, decisions, targets, decode, none = _update_setup(seed=33)
	k = len(decisions)
	pool = {"decisions": decisions, "decode": decode, **{kk: v.copy() for kk, v in none.items()}, "targets": targets, "growth": np.full((k, c.N_VALUE_OFFSETS), np.nan, np.float32), "end": np.full(k, 0.3, np.float32)}
	crit = {"module": critic, "optimizer": critic_optimizer(critic, cfg), "rows": np.array([0, 3, 5], dtype=np.int64), **none, "pool": pool}
	metrics, _ = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-3), decisions, targets, None, cfg, decode=decode, advantages=np.linspace(-1, 1, k).astype(np.float32), critic=crit)
	for name in ("own", "pool"):
		assert metrics[f"critic_grad_{name}"] >= 0 and 0 <= metrics[f"critic_clipped_{name}"] <= 1


def test_a_failed_step_graph_recording_is_tried_again_after_eager_steps(monkeypatch):
	"""(joint9 2026-11-18: one update's data made the first capture fail, every later step of the process eager)
	graphs.StepGraph: a recording that failed is tried again every RETRY_EVERY eager steps, at most RETRY_MAX times."""
	import campaign_model.graphs as G

	class Replay:
		def replay(self):
			pass

	graph = G.StepGraph(torch.nn.Linear(2, 2), lambda out, batch: (out.sum(),))
	tries = []

	def record(inputs, batch, budget=None, prepare=None):
		tries.append(len(tries))
		if len(tries) < 3:
			return None
		return {"inputs": {k: v.clone() for k, v in inputs.items()}, "batch": {k: v.clone() for k, v in batch.items()}, "outputs": (), "graph": Replay(), "checked": True}

	monkeypatch.setattr(graph, "_record", record)
	x, b = {"x": torch.zeros(1, 2)}, {"w": torch.zeros(1)}
	eager = 0
	while graph.run(x, b) is None:
		eager += 1
		assert eager < 3 * (G.RETRY_EVERY + 1), "never recorded"
	assert len(tries) == 3 and graph.retries == 2 and eager == 2 * G.RETRY_EVERY
	never = G.StepGraph(torch.nn.Linear(2, 2), lambda out, batch: (out.sum(),))
	monkeypatch.setattr(never, "_record", lambda *a, **k: None)
	for _ in range((G.RETRY_MAX + 2) * (G.RETRY_EVERY + 1)):
		assert never.run(x, b) is None
	assert never.retries == G.RETRY_MAX  # (bounded: it stops trying)


def test_the_own_rows_weight_moves_toward_parity_by_the_step_rule_and_reaches_the_loss():
	"""(e5, 2026-11-18) learner.balance_critic_own: toward pool / own x the weight, at most 1.25x an update, 1..32; the
	weight scales our rows' critic loss (their pre-clip gradient norm with it)."""
	from types import SimpleNamespace

	from campaign_model.learner import CRITIC_OWN_CAP, CRITIC_OWN_STEP, balance_critic_own

	one = SimpleNamespace(config=CampaignConfig(critic_own_balance=True), critic_own_weight=1.0)
	balance_critic_own(one, {"critic_grad_own": 0.05, "critic_grad_pool": 1.2})
	assert one.critic_own_weight == pytest.approx(CRITIC_OWN_STEP)
	for _ in range(40):
		balance_critic_own(one, {"critic_grad_own": 0.05, "critic_grad_pool": 1.2})
	assert one.critic_own_weight == pytest.approx(CRITIC_OWN_CAP)
	balance_critic_own(one, {"critic_grad_own": 5.0, "critic_grad_pool": 1.0})
	assert one.critic_own_weight == pytest.approx(CRITIC_OWN_CAP / CRITIC_OWN_STEP)
	off = SimpleNamespace(config=CampaignConfig(), critic_own_weight=1.0)
	balance_critic_own(off, {"critic_grad_own": 0.05, "critic_grad_pool": 1.2})
	assert off.critic_own_weight == 1.0
	norms = []
	for weight in (1.0, 4.0):
		cfg, policy, critic, decisions, targets, decode, none = _update_setup(seed=34)
		crit = {"module": critic, "optimizer": critic_optimizer(critic, cfg), "rows": np.array([0, 3, 5], dtype=np.int64), **none, "pool": None, "own_weight": weight}
		metrics, _ = ppo_update(policy, torch.optim.Adam(policy.parameters(), lr=1e-3), decisions, targets, None, cfg, decode=decode, advantages=np.linspace(-1, 1, len(decisions)).astype(np.float32), critic=crit)
		norms.append(metrics["critic_grad_own"])
		assert metrics["critic_own_weight"] == weight
	assert norms[1] == pytest.approx(4.0 * norms[0], rel=1e-3)


def test_each_micro_batch_and_critic_step_has_its_loss_for_the_per_step_charts():
	"""(the user, 2026-11-18 16:05: "I expected to see the loss at every step during training") ppo_update's
	policy_step_losses: [policy, link] of each micro-batch at its own rows' means (one micro-batch a minibatch here: the
	minibatch's); critic_step_sequence: each critic step's source and loss in order, the passes' lists' values; the
	learner writes them at x = steps since the run began, the counts carried on."""
	from types import SimpleNamespace

	from campaign_model.learner import write_step_losses

	from dataclasses import replace

	cfg, policy, critic, decisions, targets, decode, none = _update_setup()
	p0, c0 = copy.deepcopy(policy), copy.deepcopy(critic)
	m = _run(cfg, policy, critic, decisions, targets, decode, none, None)  # (several micro-batches a minibatch)
	assert len(m["policy_step_losses"]) == m["steps"] * cfg.micro_batches and cfg.micro_batches > 1
	assert all(np.isfinite(p) for p, _ in m["policy_step_losses"])
	assert any(not np.isfinite(k) for _, k in m["policy_step_losses"]) and any(np.isfinite(k) for _, k in m["policy_step_losses"])  # (the micro-batches without the critic's rows: nan)
	m = _run(replace(cfg, micro_batches=1), p0, c0, decisions, targets, decode, none, None)  # (one: each micro-batch the minibatch)
	seq = m["policy_step_losses"]
	assert len(seq) == m["steps"] and all(np.isfinite(v) for pair in seq for v in pair)
	assert np.mean([p for p, _ in seq]) == pytest.approx(m["policy"], rel=1e-4, abs=1e-6)
	assert np.mean([k for _, k in seq]) == pytest.approx(m["link_loss"], rel=1e-4, abs=1e-6)
	steps = m["critic_step_sequence"]
	assert [v for s, v in steps if s == "own"] == pytest.approx(m["critic_pass_own"], abs=1e-4) and {s for s, _ in steps} == {"own"}

	class Rec:
		def __init__(self):
			self.points = []

		def add_scalar(self, tag, value, step):
			self.points.append((tag, step))

	one = SimpleNamespace(step_counts={"policy": 10, "critic": 3}, writer=Rec(), critic=object())
	write_step_losses(one, {"policy_step_losses": [[0.1, 0.2], [0.3, 0.4]], "critic_step_sequence": [("own", 0.01), ("pool", 0.02)]})
	assert one.writer.points == [("policy_steps/policy_loss", 11), ("policy_steps/link_loss", 11), ("policy_steps/policy_loss", 12), ("policy_steps/link_loss", 12), ("critic_steps/loss_own", 4), ("critic_steps/loss_pool", 5)]
	assert one.step_counts == {"policy": 12, "critic": 5}


def test_the_own_fit_median_of_five_starts_from_the_runs_line_in_progress_csv(tmp_path):
	"""(e5, 2026-11-18) critic_unexplained_own_med5: the median of the last five updates' own readings; after a start its
	memory is progress.csv's rows after the last roll-back."""
	from types import SimpleNamespace

	from campaign_model.learner import note_own_fit
	from campaign_model.tb_filter import tracked

	progress = tmp_path / "progress.csv"
	progress.write_text("version,critic_unexplained_own\n40,9.0\n41,9.0\n39,0.90\n40,0.86\n41,1.50\n42,\n43,1.00\n")  # (a roll-back after v41: its rows not read)
	one = SimpleNamespace(progress=progress)
	m = {"critic_unexplained_own": 0.91}
	note_own_fit(one, m)
	assert m["critic_unexplained_own_med5"] == pytest.approx(np.median([0.90, 0.86, 1.50, 1.00, 0.91]))
	for value in (1.69, 0.84, 0.85):
		m = {"critic_unexplained_own": value}
		note_own_fit(one, m)
	assert m["critic_unexplained_own_med5"] == pytest.approx(np.median([1.00, 0.91, 1.69, 0.84, 0.85])) and len(one.own_fit_history) == 5
	fresh = SimpleNamespace(progress=tmp_path / "missing.csv")
	m = {"critic_unexplained_own": 1.2}
	note_own_fit(fresh, m)
	assert m["critic_unexplained_own_med5"] == 1.2
	assert tracked("critic/unexplained_own_med5") == "critic_fit/unexplained_own_med5"


def test_the_representative_twelve_are_balanced_and_charted_on_one_basis(tmp_path):
	"""(e5 for the user, 2026-11-18: every fourth setup was the hard quarter) config.eval_own_set rep12: one setup of each
	class mod 12 (3 of each residue mod 4, 4 of each player count, the seats spread); its charts campaign_eval_rep12/
	(served eval_rep12/) start with the earlier 48-setup evals recomputed on its setups, the every-fourth ones left out,
	and its points never on the old campaign_eval/ charts."""
	from campaign_model.evaluate import EVAL_SETS, log, plan, setups, split_row
	from campaign_model.tb_filter import tracked

	cfg = CampaignConfig(eval_own_games=48, eval_follow_games=0, eval_own_every=4, eval_own_set="rep12")
	chosen = [i for i, _ in plan(cfg)]
	assert chosen == list(EVAL_SETS["rep12"]) and sorted({i % 12 for i in chosen}) == list(range(12))
	games = setups(48)
	assert {n: sum(games[i][1] == n for i in chosen) for n in (2, 3, 4)} == {2: 4, 3: 4, 4: 4}
	assert all(sum(i % 4 == r for i in chosen) == 3 for r in range(4))
	for n in (2, 3, 4):
		seats = sorted((i // 3) % n for i in chosen if games[i][1] == n)
		assert set(seats) == set(range(n)), (n, seats)

	def game(setup, score, version):
		return {"setup": setup, "arm": "own", "score": score, "score_d28": score, "eliminated": score == 0, "won": False, "players": games[setup][1], "phi": 1.0,
			"seed": games[setup][0], "map": f"m{setup}", "color": games[setup][2], "max_level_ours": 3, "major_won": 0.0, "version": version}

	class Rec:
		def __init__(self):
			self.points = []

		def add_scalar(self, tag, value, step):
			self.points.append((tag, round(float(value), 6), step))

		def flush(self):
			pass

	def row(version, setups_played, score_of):
		played = [game(i, score_of(i), version) for i in setups_played]
		return {"version": version, "epoch": 0, "games": len(played), "per_game": played, "short": 0, "failed": 0, "seconds": 1.0, "ai_errors": 0, **split_row(played)}

	path = tmp_path / "eval.csv"
	log(row(1, range(48), lambda i: 0.5 if i in EVAL_SETS["rep12"] else 0.1), path, Rec())  # (a 48-setup eval)
	log(row(18, range(0, 48, 4), lambda i: 0.2), path, Rec())  # (every fourth)
	rec = Rec()
	log(row(50, EVAL_SETS["rep12"], lambda i: 0.3), path, rec)  # (the first on the set)
	scores = [(step, value) for tag, value, step in rec.points if tag == "campaign_eval_rep12/score_own"]
	assert scores == [(1, 0.5), (50, 0.3)]  # (v1 recomputed on the 12, then this eval; v18 not on this basis)
	assert not any(tag.startswith("campaign_eval/") for tag, _, _ in rec.points)
	rec = Rec()
	log(row(60, EVAL_SETS["rep12"], lambda i: 0.4), path, rec)
	assert [(step, value) for tag, value, step in rec.points if tag == "campaign_eval_rep12/score_own"] == [(60, 0.4)]
	assert tracked("campaign_eval_rep12/score_own") == "eval_rep12/score" and tracked("campaign_eval_rep12/eliminated_own") == "eval_rep12/eliminated"
	assert tracked("campaign_eval_rep12/phi_own") is None


def test_the_held_out_is_read_every_update_and_never_trained_on():
	"""(e5 for the user, 2026-11-18/19) crit["pool_heldout"], the NK2 stream's fixed held-out: its readings under
	*_pool_heldout (unexplained, the loss, each head's, the visible head's), its rows never in a step."""
	cfg, policy, critic, decisions, targets, decode, none = _update_setup(seed=17)
	n = len(decisions)
	held = {"decisions": decisions, "decode": decode, **{k: v.copy() for k, v in none.items()}, "targets": targets, "growth": np.full((n, c.N_VALUE_OFFSETS), np.nan, np.float32), "end": np.full(n, 0.3, np.float32)}
	p0, c0 = copy.deepcopy(policy), copy.deepcopy(critic)

	def run(policy_, critic_, fresh):
		torch.manual_seed(9)
		np.random.seed(9)
		optimizer = torch.optim.Adam(policy_.parameters(), lr=1e-3)
		crit = {"module": critic_, "optimizer": critic_optimizer(critic_, cfg), "rows": np.array([0, 3, 5], dtype=np.int64), **none, "pool": None, "pool_heldout": fresh}
		metrics, _ = ppo_update(policy_, optimizer, decisions, targets, None, cfg, decode=decode, advantages=np.linspace(-1, 1, n).astype(np.float32), critic=crit)
		return metrics

	m = run(policy, critic, held)
	for key in ("critic_unexplained_pool_heldout", "critic_loss_pool_heldout", "critic_unexplained_visible_pool_heldout", "critic_loss_visible_pool_heldout"):
		assert np.isfinite(m[key]), key
	assert m["critic_rows_pool_heldout"] == n
	assert any(k.startswith("critic_heads/unexplained_") and k.endswith("_heldout") for k in m)
	assert any(k.startswith("critic_heads/unexplained_visible_") and k.endswith("_heldout") for k in m)
	without = run(p0, c0, None)
	assert "critic_unexplained_pool_heldout" not in without
	for (name, a), (_, b) in zip(critic.named_parameters(), c0.named_parameters()):
		assert torch.equal(a, b), name


def test_the_own_weights_cap_is_a_setting():
	"""(e5, 2026-11-18) config.critic_own_cap: the balancer never above it (32 by default, as before)."""
	from types import SimpleNamespace

	from campaign_model.learner import balance_critic_own

	for cap, start, want in ((32.0, 18.0, 18.0 * 1.25), (4.0, 18.0, 4.0), (4.0, 3.5, 4.0)):
		one = SimpleNamespace(config=CampaignConfig(critic_own_balance=True, critic_own_cap=cap), critic_own_weight=start)
		balance_critic_own(one, {"critic_grad_own": 0.1, "critic_grad_pool": 1.0})  # (own far under parity: up)
		assert one.critic_own_weight == pytest.approx(want), (cap, start)


def test_one_init_the_critics_body_is_the_policys_tensor_for_tensor():
	"""(the user, 2026-11-19: "on batch 0 I expect the shared layer to be identical") learner.body_from_policy: every one
	of the critic body's tensors the policy's at construction, at the small dims and joint10's (128, 16), copies
	(a critic step leaves the policy as it is)."""
	from campaign_model.learner import body_from_policy

	for cfg in (CampaignConfig(**SMALL, critic_net=True), CampaignConfig(model_architecture="v2", model_dim=128, model_cell_dim=16, share_target="change", critic_net=True)):  # (joint10's dims)
		torch.manual_seed(7)
		policy = build_model(model_kwargs(cfg))
		critic = Critic(**critic_kwargs(cfg, model_kwargs(cfg)))
		theirs = policy.state_dict()
		assert any(not torch.equal(v, theirs[k]) for k, v in critic.body.state_dict().items() if v.is_floating_point())  # (two inits before)
		n = body_from_policy(critic, policy)
		ours = critic.body.state_dict()
		assert n == len(ours) and all(torch.equal(v, theirs[k]) for k, v in ours.items())
		first = next(name for name, p in critic.body.named_parameters() if p.requires_grad)
		before = theirs[first].clone()
		with torch.no_grad():
			dict(critic.body.named_parameters())[first].add_(1.0)
		assert torch.equal(policy.state_dict()[first], before)


def test_stage_b_the_policy_takes_the_critics_body_and_the_link_reads_zero():
	"""(e5 for the user, 2026-11-19, stage B) learner.policy_from_critic: every critic body tensor copied into the policy,
	link_map at the identity: the link's prediction is the critic's own link state, its loss 0."""
	from campaign_model.learner import policy_from_critic

	cfg = CampaignConfig(**SMALL, critic_net=True)
	torch.manual_seed(11)
	policy = build_model(model_kwargs(cfg))
	critic = Critic(**critic_kwargs(cfg, model_kwargs(cfg)))
	with torch.no_grad():  # (a trained critic: its body moved off any init)
		for p in critic.body.parameters():
			p.add_(0.01 * torch.randn_like(p))
	n = policy_from_critic(policy, critic)
	theirs, ours = critic.body.state_dict(), policy.state_dict()
	assert n == len(theirs) and all(torch.equal(ours[k], v) for k, v in theirs.items())
	assert torch.equal(policy.link_map.weight, torch.eye(policy.link_map.weight.shape[0])) and not policy.link_map.bias.any()
	policy.eval().link_outputs = True
	critic.eval()
	with torch.no_grad():
		after_again = policy(tensors(rich_inputs(6, seed=11)))["link_pred"]
	policy.link_before_again = True  # (the link at the critic's point: config.link_before_again)
	x = tensors(rich_inputs(6, seed=11))
	with torch.no_grad():
		pred = policy(x)["link_pred"]
		link, _, _ = critic.trunk(x)
	assert torch.allclose(pred, link.float(), atol=1e-5)
	assert not torch.allclose(after_again, link.float(), atol=1e-3)  # (after the foresight's second pass: not the critic's state)


def test_stage_b_a_frozen_critic_takes_no_step_and_the_policy_still_moves():
	cfg, policy, critic, decisions, targets, decode, none = _update_setup(seed=19)
	p0, c0 = copy.deepcopy(policy), copy.deepcopy(critic)
	torch.manual_seed(9)
	np.random.seed(9)
	optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
	crit = {"module": critic, "optimizer": critic_optimizer(critic, cfg), "rows": np.array([0, 3, 5], dtype=np.int64), **none, "pool": None, "frozen": True}
	metrics, _ = ppo_update(policy, optimizer, decisions, targets, None, cfg, decode=decode, advantages=np.linspace(-1, 1, len(decisions)).astype(np.float32), critic=crit)
	assert metrics.get("critic_frozen") == 1.0 and "critic_loss_own_first" not in metrics
	assert all(torch.equal(a, b) for (_, a), (_, b) in zip(critic.named_parameters(), c0.named_parameters()))
	assert any(not torch.equal(a, b) for (_, a), (_, b) in zip(policy.named_parameters(), p0.named_parameters()))


def test_the_fixed_weight_losses_are_the_start_weights_ruler_whatever_head_scale_says():
	"""(e5, 2026-11-19) train.fixed_losses: "six" and "end" at the start's weights recombine to critic_loss at no head
	weights ((4.5 x six + end) / 5.5 with growth_weight 0.5); head weights never move them."""
	from campaign_model.train import fixed_losses

	g = torch.Generator().manual_seed(3)
	n, k = 64, c.N_VALUE_OFFSETS
	out = {"value_logit": torch.randn(n, k, generator=g), "growth_logit": torch.randn(n, k, generator=g), "end_logit": torch.randn(n, generator=g)}
	phi = torch.rand(n, generator=g) * 5000 + 100
	targets, share_valid = torch.rand(n, k, generator=g), torch.ones(n, k)
	growth, growth_valid = torch.log(phi).unsqueeze(-1) + torch.randn(n, k, generator=g) * 0.3, torch.ones(n, k)
	end, end_valid = torch.rand(n, generator=g), torch.ones(n)
	six, ended = fixed_losses(out, phi, targets, share_valid, growth, growth_valid, end, end_valid, 0.5)
	whole = float(critic_loss(out, phi, targets, share_valid, growth, growth_valid, end, end_valid, None, 0.5))
	assert whole == pytest.approx((4.5 * six + ended) / 5.5, rel=1e-5)
	heavy = {"w:share": torch.tensor([3.0, 1.0, 1.0]), "w:growth": torch.tensor([2.0, 0.5, 0.2]), "w:end": torch.tensor([0.3])}
	assert float(critic_loss(out, phi, targets, share_valid, growth, growth_valid, end, end_valid, heavy, 0.5)) != pytest.approx(whole, rel=1e-3)
