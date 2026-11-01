import pytest
import numpy as np
import torch

from campaign_model import constants as c
from campaign_model.model import CampaignModel
from campaign_model.records import Decision
from campaign_model.schema import HEADS, random_inputs
from campaign_model.config import CampaignConfig
from campaign_model.train import choices, chosen_logp, head_entropy, ppo_update


def sample(out, x, b, rng):
	"""A choice drawn as AlphaMapAI::sample draws it: the type, then its table (or an option)."""
	def draw(logp):
		p = logp.exp().numpy().reshape(-1)
		return int(rng.choice(len(p), p=p / p.sum()))

	kind = int(x["g_kind"][b])
	if kind not in (c.DecisionKind.TURN, c.DecisionKind.EXCHANGE):
		return -1, draw(out["option_logp"][b]), -1
	action = draw(out["type_logp"][b])
	if action == c.ActionType.END:
		return action, -1, -1
	return action, draw(out[HEADS[c.ActionType(action)][0]][b]), -1


def decisions_for(model, x, seed=0):
	rng = np.random.default_rng(seed)
	with torch.no_grad():
		out = model({k: torch.from_numpy(v) for k, v in x.items()})
	result = []
	for b in range(len(x["g_kind"])):
		action, index, cell = sample(out, x, b, rng)
		logp = float(out["type_logp"][b, action] + (out[HEADS[c.ActionType(action)][0]][b].reshape(-1)[index] if action > 0 else 0)) if action >= 0 else float(out["option_logp"][b, index])
		result.append(Decision("RED", 1, int(x["g_kind"][b]), b, 1, action, index, cell, logp, tuple(out["value"][b].tolist()), 0, b""))
	return result


def choice_tensors(decisions):
	return {k: torch.from_numpy(v) for k, v in choices(decisions).items() if k in ("kind", "action", "index", "cell")}


def test_chosen_logp_is_the_log_probability_the_ai_sampled_with():
	torch.manual_seed(0)
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	x = random_inputs(8, seed=3)
	decisions = decisions_for(model, x)
	with torch.no_grad():
		logp, entropy = chosen_logp(model({k: torch.from_numpy(v) for k, v in x.items()}), choice_tensors(decisions))
	assert np.allclose(logp.numpy(), [d.logp for d in decisions], atol=1e-5)
	assert (entropy >= 0).all()


def test_entropy_bonus_does_not_favour_types_with_many_options():
	"""The bonus's gradient on the type logits is the type entropy's alone (the parents' probabilities detached)."""
	torch.manual_seed(1)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	x = random_inputs(6, seed=5)
	decisions = [Decision("RED", 1, int(k), 0, 1, 0, -1, -1, 0.0, (0.5,) * 3, 0, b"") for k in x["g_kind"]]
	out = model({k: torch.from_numpy(v) for k, v in x.items()})
	type_logp = out["type_logp"].detach().clone().requires_grad_()
	_, entropy = chosen_logp({**out, "type_logp": type_logp}, choice_tensors(decisions))
	(full,) = torch.autograd.grad(entropy.sum(), type_logp)
	alone_logp = type_logp.detach().clone().requires_grad_()
	turn = torch.tensor([int(k) in (c.DecisionKind.TURN, c.DecisionKind.EXCHANGE) for k in x["g_kind"]])
	(alone,) = torch.autograd.grad((head_entropy(alone_logp) * turn).sum(), alone_logp)
	legal = torch.isfinite(out["type_logp"])
	assert torch.allclose(full[legal], alone[legal], atol=1e-6)


def test_an_update_moves_the_weights_a_little():
	torch.manual_seed(2)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	x = random_inputs(10, seed=6)
	decisions = decisions_for(model.eval(), x)
	before = [p.detach().clone() for p in model.parameters()]
	targets = np.full((10, c.N_VALUE_OFFSETS), 0.6, dtype=np.float32)
	returns = np.linspace(0.3, 0.7, 10).astype(np.float32)
	config = CampaignConfig(batch_size=4, passes=2)
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	metrics, scale = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, targets, returns, config, decode=decode)
	assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
	assert metrics["steps"] == 4 and metrics["unsound"] == 0 and metrics["decisions"] == 10  # 2 passes of 2 full minibatches
	assert np.isfinite(metrics["kl"]) and metrics["kl"] < 0.02 and 0.1 <= scale <= 1.0


def test_the_kl_control_stops_an_update_that_moves_too_far():
	torch.manual_seed(3)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	x = random_inputs(16, seed=7)
	decisions = decisions_for(model.eval(), x)
	targets = np.full((16, c.N_VALUE_OFFSETS), 0.5, dtype=np.float32)
	returns = np.linspace(-1, 1, 16).astype(np.float32)
	config = CampaignConfig(batch_size=4, passes=20, learning_rate=0.05, target_kl=1e-4)
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	metrics, scale = ppo_update(model, torch.optim.Adam(model.parameters()), decisions, targets, returns, config, decode=decode)
	assert metrics["stopped"] and metrics["steps"] < 80 and scale == 1 / 1.5


def test_a_teachers_weight_widens_the_kl_target_by_one_plus_it():
	"""An update with Nullkiller2's advice at teacher_weight w may move (1 + w) x target_kl: its pull is supervised."""
	import dataclasses

	torch.manual_seed(3)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	x = random_inputs(16, seed=7)
	decisions = [dataclasses.replace(d, advice=(d.action_type, d.index, d.cell)) for d in decisions_for(model.eval(), x)]
	targets = np.full((16, c.N_VALUE_OFFSETS), 0.5, dtype=np.float32)
	returns = np.linspace(-1, 1, 16).astype(np.float32)
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	config = CampaignConfig(batch_size=4, passes=20, learning_rate=0.05, target_kl=1e-4)
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters()), decisions, targets, returns, config, decode=decode, teacher_weight=19.0)
	assert metrics["kl_target"] == pytest.approx(20 * 1e-4)
	assert not metrics["stopped"] or metrics["last_kl"] > 1.5 * 20 * 1e-4  # (stopped only past the wider target)
	plain = [dataclasses.replace(d, advice=None) for d in decisions]
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters()), plain, targets, returns, config, decode=decode, teacher_weight=19.0)
	assert metrics["kl_target"] == pytest.approx(1e-4)  # (no teacher: PPO's own)


def test_head_loss_counts_only_the_decisions_with_a_grade():
	from campaign_model.train import head_loss

	logits = torch.tensor([[0.0, 1.0], [2.0, -1.0], [0.5, 0.5]])
	targets = torch.tensor([[0.5, 0.2], [0.9, 0.0], [0.1, 0.7]])
	valid = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
	every = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
	assert torch.allclose(head_loss(logits, targets, valid), torch.stack([every[:2, 0].mean(), every[[0, 2], 1].mean()]))


def test_an_update_leaves_out_missing_grades_and_credit_and_trains_the_growth_heads():
	"""Grades past the day cap (nan) add nothing to their heads' loss, decisions without credit (nan) nothing to the
	policy's; the growth heads learn our value's growth over the model's Φ at the decision."""
	torch.manual_seed(4)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	x = random_inputs(8, seed=9)
	decisions = decisions_for(model.eval(), x)
	targets = np.full((8, c.N_VALUE_OFFSETS), 0.6, dtype=np.float32)
	targets[4:, 1:] = np.nan
	growth = np.full((8, c.N_VALUE_OFFSETS), np.log(50000.0), dtype=np.float32)
	growth[6:] = np.nan
	advantages = np.linspace(-1, 1, 8).astype(np.float32)
	advantages[[1, 5]] = np.nan
	before = model.growth_heads.weight.detach().clone()
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, targets, None, CampaignConfig(batch_size=4, passes=1), decode=decode, advantages=advantages, growth=growth)
	assert metrics["credited"] == 6 and np.isfinite(metrics["value"]) and np.isfinite(metrics["policy"])
	assert not torch.equal(before, model.growth_heads.weight)


def test_an_update_trains_the_end_head_and_credits_through_it_by_its_weight():
	"""V_end learns its grades; the end credit (from V_end of the model as the update starts) takes its weight's share
	of the policy's advantage, and credits decisions the 7-day credit has none for (within a week of a day cap)."""
	torch.manual_seed(6)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	x = random_inputs(8, seed=11)
	decisions = decisions_for(model.eval(), x)
	targets = np.full((8, c.N_VALUE_OFFSETS), np.nan, dtype=np.float32)
	seven = np.full(8, np.nan, dtype=np.float32)
	seven[:4] = np.linspace(-1, 1, 4)
	end = {
		"targets": np.full(8, 0.8, dtype=np.float32), "real": np.full(8, 0.8, dtype=np.float32), "step": np.array([1, 2, 3, 4, 5, 6, 7, -1]),
		"week": np.full(8, -1), "price": np.zeros(8, dtype=np.float32), "weight": 0.0, "outcome": 0.25,
	}
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	config = CampaignConfig(batch_size=4, passes=1)
	before = model.end_head.weight.detach().clone()
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-3), decisions, targets, None, config, decode=decode, advantages=seven, end=end)
	assert metrics["credited"] == 4 and not torch.equal(before, model.end_head.weight)  # weight 0: the 7-day credit alone
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, targets, None, config, decode=decode, advantages=seven, end={**end, "weight": 0.5})
	assert metrics["credited"] == 8 and np.isfinite(metrics["policy"])


def test_heads_that_know_only_the_player_count_explain_nothing():
	"""The share heads start at 1 / players; their explained variance (the credit's trust in them) is over that prior's,
	so a head that knows only the player count earns no trust, however much of the shares the count explains."""
	torch.manual_seed(7)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	x = random_inputs(8, seed=13)
	x["g_players"] = np.array([2, 2, 3, 3, 4, 4, 2, 4], dtype=x["g_players"].dtype)
	decisions = decisions_for(model.eval(), x)  # (recorded values: the untrained heads', 1 / players)
	prior = 1.0 / x["g_players"].astype(np.float32)
	assert np.allclose([d.value for d in decisions], prior[:, None], atol=1e-6)
	labels = np.repeat((prior + np.random.default_rng(0).normal(0, 0.02, 8)).astype(np.float32)[:, None], c.N_VALUE_OFFSETS, axis=1)
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, labels, np.zeros(8, dtype=np.float32), CampaignConfig(batch_size=4, passes=1), decode=decode)
	assert all(abs(metrics[f"value_ev_d{k}"]) < 1e-5 for k in c.VALUE_OFFSETS)  # (plain EV here: ~0.97)


def test_micro_batches_take_the_same_step():
	"""A minibatch in micro-batches (their gradients summed, each normalized by the whole minibatch's counts) moves the
	weights as the whole minibatch at once."""
	import copy

	torch.manual_seed(8)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	x = random_inputs(8, seed=15)
	decisions = decisions_for(model.eval(), x)
	targets = np.full((8, c.N_VALUE_OFFSETS), 0.6, dtype=np.float32)
	targets[5:, 1:] = np.nan
	growth = np.full((8, c.N_VALUE_OFFSETS), np.log(50000.0), dtype=np.float32)
	advantages = np.linspace(-1, 1, 8).astype(np.float32)
	advantages[2] = np.nan
	end = {"targets": np.linspace(0.2, 0.8, 8).astype(np.float32), "real": np.full(8, 0.7, dtype=np.float32), "step": np.array([1, 2, 3, 4, 5, 6, 7, -1]), "week": np.full(8, -1), "price": np.zeros(8, dtype=np.float32), "weight": 0.5, "outcome": 0.25}
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	results = []
	for parts in (1, 3):
		m = copy.deepcopy(model)
		torch.manual_seed(0)
		metrics, _ = ppo_update(m, torch.optim.SGD(m.parameters(), lr=0.1), decisions, targets, None, CampaignConfig(batch_size=8, passes=1, micro_batches=parts, target_kl=0.0), decode=decode, advantages=advantages, growth=growth, end=end)
		results.append((m, metrics))
	(a, ma), (b, mb) = results
	assert all(torch.allclose(p, q, atol=1e-6) for p, q in zip(a.parameters(), b.parameters()))
	assert all(abs(ma[k] - mb[k]) < 1e-5 for k in ("policy", "value", "entropy", "kl", "grad_norm"))


def test_the_fresh_value_loss_is_zero_only_for_exact_predictions():
	"""excess_loss: a head's cross-entropy above its target's entropy, 0 exactly at the target; ppo_update reports it
	before the update (value_fresh), with growth targets that are partly missing."""
	from campaign_model.train import excess_loss

	targets = torch.tensor([[0.0, 0.3, 1.0]])
	exact = torch.logit(targets.clamp(1e-6, 1 - 1e-6))
	assert float(excess_loss(exact, targets).abs().max()) < 1e-4
	assert (excess_loss(torch.zeros(1, 3), targets) > 0).all()
	torch.manual_seed(4)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	x = random_inputs(8, seed=9)
	decisions = decisions_for(model.eval(), x)
	targets = np.full((8, c.N_VALUE_OFFSETS), 0.6, dtype=np.float32)
	growth = np.full((8, c.N_VALUE_OFFSETS), np.log(50000.0), dtype=np.float32)
	growth[6:] = np.nan
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, targets, np.zeros(8, np.float32), CampaignConfig(batch_size=4, passes=1), decode=decode, growth=growth)
	assert np.isfinite(metrics["value_fresh"]) and metrics["value_fresh"] > 0


def test_the_teachers_kl_is_zero_against_itself_and_positive_against_another_model():
	from campaign_model.train import TEACHER_OUTPUTS, teacher_kl

	torch.manual_seed(4)
	model, other = CampaignModel(dim=32, cell_dim=16, layers=1).eval(), CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	x = {k: torch.from_numpy(v) for k, v in random_inputs(8, seed=9).items()}
	out = model(x)
	batch = {"kind": x["g_kind"].long(), **{"t:" + name: out[name].detach() for name in TEACHER_OUTPUTS}}
	assert teacher_kl(out, batch).abs().max() < 1e-4
	moved = other(x)
	kl = teacher_kl(moved, batch)
	learned = (x["g_kind"] == c.DecisionKind.TURN) | (x["g_kind"] == c.DecisionKind.YES_NO) | (x["g_kind"] == c.DecisionKind.SELECT) | (x["g_kind"] == c.DecisionKind.LEVEL_UP)
	assert learned.any() and (~learned).any()  # (both among the inputs)
	assert torch.isfinite(kl).all() and (kl[learned] > 0).all() and (kl[~learned] == 0).all()  # (only what the teacher learned)
	kl.sum().backward()  # (masked entries: -inf in both, no nan reaches a gradient)
	assert all(torch.isfinite(p.grad).all() for p in other.parameters() if p.grad is not None)


def test_an_update_with_a_teacher_moves_toward_it():
	"""With only the teacher's KL to learn from (no credit), an update brings the model's choices closer to the teacher's."""
	from campaign_model.train import TEACHER_OUTPUTS, teacher_kl

	torch.manual_seed(5)
	model, teacher = CampaignModel(dim=32, cell_dim=16, layers=1), CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	x = random_inputs(16, seed=10)
	tensors = {k: torch.from_numpy(v) for k, v in x.items()}
	decisions = decisions_for(model.eval(), x)
	measure = lambda: float(teacher_kl(model.eval()(tensors), {"kind": tensors["g_kind"].long(), **{"t:" + n: o.detach() for n, o in teacher(tensors).items() if n in TEACHER_OUTPUTS}}).mean())  # noqa: E731
	before = measure()
	targets = np.full((16, c.N_VALUE_OFFSETS), 0.5, dtype=np.float32)
	returns = np.full(16, 0.5, dtype=np.float32)
	config = CampaignConfig(batch_size=8, passes=4, teacher_kl=1.0, target_kl=0.0)
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-3), decisions, targets, returns, config, decode=decode, teacher=teacher)
	assert metrics["teacher_kl"] > 0
	assert measure() < before


def test_an_update_with_the_schedule_credit_weighs_the_share_by_the_day():
	"""credit "schedule": w x the share's surprise + (1 - w) x the growth's, w = 10% + 2% a day (day 1 here: 12%)."""
	from campaign_model.model import build_model

	torch.manual_seed(3)
	model = build_model({"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"})
	x = random_inputs(8, seed=11)
	decisions = decisions_for(model.eval(), x)
	targets = np.full((8, c.N_VALUE_OFFSETS), 0.55, dtype=np.float32)  # (the share's logit up 0.3 over each horizon)
	growth_real = np.full((8, c.N_VALUE_OFFSETS), 10.0, dtype=np.float32)  # (log of our value days on)
	config = CampaignConfig(batch_size=4, passes=1, credit="schedule", share_target="change")
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, targets, np.zeros(8, dtype=np.float32), config, decode=decode, growth_real=growth_real)
	assert metrics["schedule_share_weight"] == pytest.approx(0.12)
	assert metrics["steps"] >= 1 and np.isfinite(metrics["kl"])


def test_an_update_with_nk2s_advice_moves_toward_its_choices():
	"""Decisions advised by Nullkiller2 (Decision.advice): the update's teacher cross-entropy raises their probability."""
	import dataclasses

	torch.manual_seed(6)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	x = random_inputs(16, seed=12)
	sampled = decisions_for(model.eval(), x)
	decisions = [dataclasses.replace(d, advice=(d.action_type, d.index, d.cell)) if i % 2 == 0 else d for i, d in enumerate(sampled)]
	tensors = {k: torch.from_numpy(v) for k, v in x.items()}
	advised = [i for i in range(16) if i % 2 == 0]
	measure = lambda: float(chosen_logp(model.eval()(tensors), choice_tensors([decisions[i] for i in range(16)]))[0][advised].mean().detach())  # noqa: E731
	before = measure()
	config = CampaignConfig(batch_size=8, passes=4, teacher_kl=5.0, target_kl=0.0)
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-3), decisions, np.full((16, c.N_VALUE_OFFSETS), 0.5, dtype=np.float32), np.full(16, 0.5, dtype=np.float32), config, decode=decode)
	assert metrics["teacher_ce"] > 0
	assert measure() > before


def test_a_step_graph_whose_check_finds_no_memory_is_dropped_and_the_step_goes_eager():
	from campaign_model.graphs import StepGraph

	def no_memory(out, batch):
		raise torch.OutOfMemoryError("CUDA out of memory (a test)")

	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	graph = StepGraph(model, no_memory)
	graph.entries["key"] = {"checked": False}
	inputs = {k: torch.from_numpy(v) for k, v in random_inputs(2, seed=1).items()}
	assert graph._first(graph.entries["key"], "key", inputs, {}) is None and graph.entries["key"] is None
	assert all(p.grad is None for p in model.parameters())  # (nothing added: the caller's eager step adds it)
