import pytest
import numpy as np
import torch

from campaign_model import constants as c
from campaign_model.model import CampaignModel
from campaign_model.records import Decision
from campaign_model.schema import HEADS, random_inputs
from campaign_model.config import CampaignConfig
from campaign_model.train import _prefetched, choices, chosen_logp, head_entropy, loss_noise, ppo_update, share_weight, windows_for, wire_for


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
	index = draw(out[HEADS[c.ActionType(action)][0]][b])
	slots = slot_choices(out, b, index) if action == c.ActionType.ARMY else None
	return action, index, -1 if slots is None else draw(slots)


def slot_choices(out, b, index):
	"""ARMY's target slot log-probabilities [ARMY_SLOT_CHOICES] for its entry `index`, renormalized over the choices its
	amount allows (AlphaMapAI's draw: the whole stack's legal slots; a part's fit ones; ONE's both halves), -inf else."""
	holder, slot, to, amount = np.unravel_index(index, (c.N_HOLDERS, c.ARMY_SLOTS, c.N_HOLDERS, c.N_ARMY_AMOUNTS))
	fit, whole = out["army_split_mask"][b, holder, slot, to], out["army_whole_mask"][b, holder, slot, to]
	allowed = torch.cat([whole if amount == c.ArmyAmount.ALL else fit, fit if amount == c.ArmyAmount.ONE else torch.zeros_like(fit)])
	return torch.log_softmax(out["army_slot_logp"][b, holder, slot, to].masked_fill(~allowed, float("-inf")), dim=-1)


def decisions_for(model, x, seed=0):
	rng = np.random.default_rng(seed)
	with torch.no_grad():
		out = model({k: torch.from_numpy(v) for k, v in x.items()})
	result = []
	for b in range(len(x["g_kind"])):
		action, index, cell = sample(out, x, b, rng)
		logp = float(out["type_logp"][b, action] + (out[HEADS[c.ActionType(action)][0]][b].reshape(-1)[index] if action > 0 else 0)) if action >= 0 else float(out["option_logp"][b, index])
		logp += float(slot_choices(out, b, index)[cell]) if cell >= 0 else 0.0
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
	"""With config.kl_teacher_inflation (as until 2026-11-06), an update with Nullkiller2's advice at teacher_weight w may
	move (1 + w) x target_kl: its pull is supervised. Without (the default since): target_kl, the actor on the slow
	timescale (docs/optimizer_plan.md step 2)."""
	import copy
	import dataclasses

	torch.manual_seed(3)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	x = random_inputs(16, seed=7)
	decisions = [dataclasses.replace(d, advice=(d.action_type, d.index, d.cell)) for d in decisions_for(model.eval(), x)]
	targets = np.full((16, c.N_VALUE_OFFSETS), 0.5, dtype=np.float32)
	returns = np.linspace(-1, 1, 16).astype(np.float32)
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	default = CampaignConfig(batch_size=4, passes=1, learning_rate=0.05, target_kl=1e-4)
	metrics, _ = ppo_update(copy.deepcopy(model), torch.optim.Adam(model.parameters()), decisions, targets, returns, default, decode=decode, teacher_weight=19.0)
	assert metrics["kl_target"] == pytest.approx(1e-4)
	config = CampaignConfig(batch_size=4, passes=20, learning_rate=0.05, target_kl=1e-4, kl_teacher_inflation=True)
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
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, labels, np.zeros(8, dtype=np.float32), CampaignConfig(batch_size=4, passes=1, credit="horizon", share_target="share"), decode=decode)  # (share heads: no longer the default)
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
	assert np.isnan(metrics["value_unexplained"])  # (targets all alike: the batch's mean guesses them exactly)
	# Targets that differ: the heads' loss over the batch-mean guess's, finite and positive (an untrained head: ~1 or more)
	spread = np.random.default_rng(3).uniform(0.1, 0.9, (8, c.N_VALUE_OFFSETS)).astype(np.float32)
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, spread, np.zeros(8, np.float32), CampaignConfig(batch_size=4, passes=1), decode=decode, growth=growth)
	assert np.isfinite(metrics["value_unexplained"]) and 0.0 < metrics["value_unexplained"] < 100.0


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


def test_the_window_budget_is_a_share_of_the_slots_unless_the_batch_has_more_heroes():
	"""(2026-11-03) train.windows_for: config.window_budget windows a decision, every slot (None) for a batch with more
	of our heroes, when off, or when the budget is every slot anyway."""
	assert windows_for(64, 200, CampaignConfig(window_budget=4.0)) == 256
	assert windows_for(64, 256, CampaignConfig(window_budget=4.0)) == 256
	assert windows_for(64, 257, CampaignConfig(window_budget=4.0)) is None
	assert windows_for(64, 10, CampaignConfig(window_budget=0.0)) is None
	assert windows_for(64, 10, CampaignConfig(window_budget=c.OWN_HEROES)) is None


def test_prefetched_yields_what_measure_found_on_the_host():
	batches = [np.arange(3), np.arange(3, 5)]
	decode = lambda rows: {"a": np.asarray(rows, dtype=np.float32)}  # noqa: E731
	got = list(_prefetched(batches, decode, torch.device("cpu"), measure=lambda decoded: float(decoded["a"].sum())))
	assert [measured for _, _, measured in got] == [3.0, 7.0]
	assert [inputs["a"].tolist() for _, inputs, _ in got] == [[0, 1, 2], [3, 4]]
	assert [len(item) for item in _prefetched(batches, decode, torch.device("cpu"))] == [2, 2]  # (without: as before)


def test_the_wire_filled_in_a_tensor_reaches_the_update_as_it_is():
	"""(2026-11-03) wire_for: decode_wire straight into a tensor (pinned on CUDA), which _prefetched passes on whole."""
	from types import SimpleNamespace  # (what decode_wire reads of a Decision: its inputs)

	from campaign_model.records import decode_wire
	from campaign_model.tests.test_model import encoded

	x = random_inputs(3, seed=2)
	decisions = [SimpleNamespace(inputs=encoded(x, b)) for b in range(3)]
	wire = wire_for(decisions, torch.device("cpu"))
	assert torch.is_tensor(wire) and wire.dtype == torch.uint8
	np.testing.assert_array_equal(wire.numpy(), decode_wire(decisions))
	(rows, inputs, heroes), = list(_prefetched([np.arange(3)], lambda rows: wire_for([decisions[r] for r in rows], torch.device("cpu")), torch.device("cpu"), measure=lambda decoded: int(np.asarray(decoded).sum() > 0)))
	assert torch.equal(inputs["wire"], wire) and heroes == 1


def test_the_loss_noise_takes_the_wire_as_the_pcs_update_decodes_it():
	"""(2026-11-03) joint7 stopped at 07:25 CST: loss_noise decoded with the update's decode, which gives wire_for's
	tensor on CUDA (decode_wire into pinned memory, then inputs_on_device), and torch.from_numpy refused it. The same
	noise from that wire as from the decoded inputs."""
	from campaign_model.records import decode_inputs, inputs_on_device
	from campaign_model.tests.test_model import encoded

	torch.manual_seed(0)
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	x = random_inputs(8, seed=3)
	decisions = [d.__class__(**{**d.__dict__, "inputs": encoded(x, b)}) for b, d in enumerate(decisions_for(model, x))]
	n, k = len(decisions), c.N_VALUE_OFFSETS
	t = {"choice": {name: torch.from_numpy(value) for name, value in choices(decisions).items()}, "advantage": torch.linspace(-1, 1, n), "credited": torch.ones(n),
		"unsound": torch.zeros(n, dtype=torch.bool), "targets": torch.full((n, k), 0.5), "share_valid": torch.ones(n, k), "labels": torch.full((n, k), 0.4), "growth_real": torch.zeros(n, k)}
	cpu = torch.device("cpu")
	model.train()
	wire = loss_noise(model, decisions, lambda rows: wire_for([decisions[r] for r in rows], cpu), inputs_on_device, cpu, t, CampaignConfig(), rows=2, samples=3)
	arrays = loss_noise(model, decisions, lambda rows: decode_inputs([decisions[r] for r in rows]), None, cpu, t, CampaignConfig(), rows=2, samples=3)
	assert wire and wire.keys() == arrays.keys()
	for name in wire:
		assert wire[name] == arrays[name] or (np.isnan(wire[name]) and np.isnan(arrays[name])) or abs(wire[name] - arrays[name]) <= 1e-4 * abs(arrays[name]), name


class _EagerGraphs:
	"""graphs.Graphs' interface, eager on the CPU (each call its own forward): the update's wiring checked without CUDA."""

	def __init__(self, model):
		self.model = model

	@staticmethod
	def usable(model, device):
		return True

	def eval(self, inputs, budget=None, prepare=None):
		with torch.no_grad():
			return self.model(prepare(inputs) if prepare is not None else inputs)

	def train(self, inputs):
		return self.model(inputs)


class _EagerStep:
	"""graphs.StepGraph's interface, eager: the step's loss and backward, its outputs past the loss."""

	budget_ok = None

	def __init__(self, model, fn, forward=None):
		self.model, self.fn, self.forward = model, fn, forward or model

	def run(self, inputs, batch, budget=None, prepare=None):
		loss, *outputs = self.fn(self.forward(prepare(inputs) if prepare is not None else inputs), batch)
		loss.backward()
		return tuple(None if o is None else o.detach() for o in outputs)


def test_the_update_with_the_conversion_in_its_graphs_learns_what_it_learned_without(monkeypatch):
	"""(2026-11-03) config.convert_in_graphs: the graphs take the bytes as sent and convert them inside (records.
	inputs_on_device); the update's other uses of a batch (the first pass's g_players, the gradient check, the eager
	fallback) convert it themselves. The PC's path (wire_for + the conversion on the device) run on the CPU through
	eager stand-ins of the graphs, with the conversion in them and outside: the same weights after."""
	from campaign_model import train
	from campaign_model.tests.test_model import encoded

	monkeypatch.setattr(train, "converts_on_device", lambda device: True)
	monkeypatch.setattr(train, "Graphs", _EagerGraphs)
	monkeypatch.setattr(train, "StepGraph", _EagerStep)
	x = random_inputs(12, seed=4)
	torch.manual_seed(1)
	start = CampaignModel(dim=32, cell_dim=16, layers=1)
	decisions = [d.__class__(**{**d.__dict__, "inputs": encoded(x, b)}) for b, d in enumerate(decisions_for(start.eval(), x))]
	n = len(decisions)
	weights = {}
	for inside in (True, False):
		model = CampaignModel(dim=32, cell_dim=16, layers=1)
		model.load_state_dict(start.state_dict())
		model.fixed_windows = False  # (graphs.Graphs.usable's mark: the stand-ins take it)
		torch.manual_seed(2)
		config = CampaignConfig(batch_size=4, micro_batches=2, passes=1, cuda_graphs=True, step_graphs=True, convert_in_graphs=inside, decode_threads=1)
		ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-3), decisions, np.full((n, c.N_VALUE_OFFSETS), 0.5, np.float32), np.linspace(-1, 1, n).astype(np.float32), config)
		weights[inside] = {k: v.detach().clone() for k, v in model.state_dict().items()}
	assert any(not torch.equal(weights[True][k], start.state_dict()[k]) for k in weights[True])  # (it learned)
	for k in weights[True]:
		torch.testing.assert_close(weights[True][k], weights[False][k], rtol=0, atol=0, msg=k)


def test_a_step_over_the_window_budget_is_stepped_eagerly_not_recorded(monkeypatch):
	"""(2026-11-03) A micro-batch with more of our heroes than config.window_budget allows, while budgets work: stepped
	eagerly, not given to the step graph (whose graph over every slot would hold its pool for the run). The same
	weights as with no step graph at all."""
	from campaign_model import train
	from campaign_model.tests.test_model import encoded

	runs = []

	class Counting(_EagerStep):
		budget_ok = True

		def run(self, inputs, batch, budget=None, prepare=None):
			runs.append(budget)
			return super().run(inputs, batch, budget, prepare)

	monkeypatch.setattr(train, "converts_on_device", lambda device: True)
	monkeypatch.setattr(train, "Graphs", _EagerGraphs)
	monkeypatch.setattr(train, "StepGraph", Counting)
	x = random_inputs(12, seed=4)
	torch.manual_seed(1)
	start = CampaignModel(dim=32, cell_dim=16, layers=1)
	decisions = [d.__class__(**{**d.__dict__, "inputs": encoded(x, b)}) for b, d in enumerate(decisions_for(start.eval(), x))]
	n = len(decisions)
	assert x["h_present"][:, : c.OWN_HEROES].sum() > 0.1 * n  # (every micro-batch over a budget of 0.1 a decision)
	weights = {}
	for steps in (True, False):
		model = CampaignModel(dim=32, cell_dim=16, layers=1)
		model.load_state_dict(start.state_dict())
		model.fixed_windows = False
		torch.manual_seed(2)
		config = CampaignConfig(batch_size=4, micro_batches=2, passes=1, cuda_graphs=True, step_graphs=steps, window_budget=0.1, decode_threads=1)
		ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-3), decisions, np.full((n, c.N_VALUE_OFFSETS), 0.5, np.float32), np.linspace(-1, 1, n).astype(np.float32), config)
		weights[steps] = {k: v.detach().clone() for k, v in model.state_dict().items()}
	assert runs == []  # (no step went to the step graph)
	for k in weights[True]:
		torch.testing.assert_close(weights[True][k], weights[False][k], rtol=0, atol=0, msg=k)


def test_steps_over_budget_take_the_step_graph_once_budgets_are_refused(monkeypatch):
	"""(2026-11-03) With the budget refused (StepGraph.budget_ok False: every step over every slot), the steps go to
	the step graph as before, never eagerly."""
	from campaign_model import train
	from campaign_model.tests.test_model import encoded

	runs = []

	class Refused(_EagerStep):
		budget_ok = False

		def run(self, inputs, batch, budget=None, prepare=None):
			runs.append(budget)
			return super().run(inputs, batch, budget, prepare)

	monkeypatch.setattr(train, "converts_on_device", lambda device: True)
	monkeypatch.setattr(train, "Graphs", _EagerGraphs)
	monkeypatch.setattr(train, "StepGraph", Refused)
	x = random_inputs(8, seed=4)
	model = CampaignModel(dim=32, cell_dim=16, layers=1)
	decisions = [d.__class__(**{**d.__dict__, "inputs": encoded(x, b)}) for b, d in enumerate(decisions_for(model.eval(), x))]
	model.fixed_windows = False
	config = CampaignConfig(batch_size=4, micro_batches=2, passes=1, cuda_graphs=True, step_graphs=True, window_budget=0.1, decode_threads=1)
	ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-3), decisions, np.full((8, c.N_VALUE_OFFSETS), 0.5, np.float32), np.zeros(8, np.float32), config)
	assert runs and all(budget is None for budget in runs)


def test_the_share_weight_ramps_by_the_day_slower_on_bigger_maps():
	"""(2026-11-03, the user) The schedule credit's share weight: start + per day x day on a map of share_weight_side,
	half the rate on a map twice as wide, within 0..1; the same everywhere with share_weight_side 0."""
	config = CampaignConfig(share_weight_start=0.1, share_weight_per_day=0.03, share_weight_side=36.0)
	day = torch.tensor([0.0, 10.0, 30.0, 60.0])
	torch.testing.assert_close(share_weight(day, torch.full((4,), 36.0), config), torch.tensor([0.1, 0.4, 1.0, 1.0]))
	torch.testing.assert_close(share_weight(day, torch.full((4,), 72.0), config), torch.tensor([0.1, 0.25, 0.55, 1.0]))
	unscaled = CampaignConfig(share_weight_start=0.1, share_weight_per_day=0.03, share_weight_side=0.0)
	torch.testing.assert_close(share_weight(day, torch.full((4,), 144.0), unscaled), torch.tensor([0.1, 0.4, 1.0, 1.0]))


def test_the_step_credit_is_a_terms_step_to_the_next_turn_decision_that_day():
	"""train.step_credit (config.step_credit): (1 - s) x (the forecast at the player's next turn decision that day - the
	forecast here) + s x the term, s = 1 - 0.75 x trust; the step alone where the term is unknown, the term alone where
	the next turn decision is another day's (or none), nan where neither is known."""
	from campaign_model.train import step_credit

	term = torch.tensor([0.10, -0.20, 0.30, float("nan"), float("nan"), 0.05])
	forecast = torch.tensor([1.00, 1.50, 1.20, 2.00, 2.40, 0.70])
	step = torch.tensor([1, 2, 3, 4, -1, -1])  # (row 2's next is on day 2; row 4's, row 5's: none)
	day = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
	got = step_credit(term, forecast, step, day, trust=0.4)
	s = 1 - 0.75 * 0.4
	expected = torch.tensor([(1 - s) * 0.5 + s * 0.10, (1 - s) * -0.3 + s * -0.20, 0.30, 0.40, float("nan"), 0.05])
	assert torch.allclose(got[[0, 1, 2, 3, 5]], expected[[0, 1, 2, 3, 5]]) and torch.isnan(got[4])
	assert torch.allclose(step_credit(term, forecast, step, day, trust=0.0)[:2], term[:2])  # (an untrusted head: the term alone)
	assert torch.allclose(step_credit(term, forecast, step, day, trust=1.0)[:2], 0.75 * torch.tensor([0.5, -0.3]) + 0.25 * term[:2])


def test_an_update_with_the_step_credit_runs_and_reports_both_trusts():
	"""config.step_credit with the schedule credit and the end's step indices: the update runs, its stats carry both
	terms' trusts; without end (no steps) it is the schedule credit as before. (Set aside 2026-11-09: the switch changes
	nothing while ppo_update's call is commented out; the update runs all the same.)"""
	from campaign_model.model import build_model

	torch.manual_seed(3)
	model = build_model({"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"})
	x = random_inputs(8, seed=11)
	decisions = decisions_for(model.eval(), x)
	targets = np.full((8, c.N_VALUE_OFFSETS), 0.55, dtype=np.float32)
	growth_real = np.full((8, c.N_VALUE_OFFSETS), 10.0, dtype=np.float32)
	growth_real[6:] = np.nan  # (the last two: no 7-day outcome, their step alone)
	end = {"targets": np.full(8, 0.5, np.float32), "real": np.full(8, 0.5, np.float32), "step": np.array([1, 2, 3, 4, 5, 6, 7, -1]), "week": np.full(8, -1), "price": np.zeros(8, np.float32), "outcome": 0.5, "weight": 0.0}
	config = CampaignConfig(batch_size=4, passes=1, credit="schedule", share_target="change", step_credit=True)
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, targets, np.zeros(8, dtype=np.float32), config, decode=decode, growth_real=growth_real, end=end, share_trust=0.4, growth_trust=1.0)
	assert metrics["schedule_share_trust"] == pytest.approx(0.4) and metrics["schedule_growth_trust"] == pytest.approx(1.0)
	assert metrics["steps"] >= 1 and np.isfinite(metrics["kl"])


def test_the_step_credit_is_set_aside_its_switches_change_nothing_in_an_update():
	"""The step credit is set aside (the user, 2026-11-09: no credit on a decision's own step, the 7-day outcome only):
	ppo_update's call of train.step_credit is commented out, so config.step_credit, step_price and
	step_class_correction leave an update's stats without any of the step's readings (the function and its unit tests
	are kept for a later measurement)."""
	from campaign_model.model import build_model

	torch.manual_seed(3)
	model = build_model({"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"})
	x = random_inputs(8, seed=11)
	decisions = decisions_for(model.eval(), x)
	targets = np.full((8, c.N_VALUE_OFFSETS), 0.55, dtype=np.float32)
	growth_real = np.full((8, c.N_VALUE_OFFSETS), 10.0, dtype=np.float32)
	growth_real[6:] = np.nan
	end = {"targets": np.full(8, 0.5, np.float32), "real": np.full(8, 0.5, np.float32), "step": np.array([1, 2, 3, 4, 5, 6, 7, -1]), "week": np.full(8, -1), "price": np.zeros(8, np.float32), "outcome": 0.5, "weight": 0.0}
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	for extra in ({}, {"step_credit": True}, {"step_credit": True, "step_price": 1e-3, "step_class_correction": True}):
		config = CampaignConfig(batch_size=4, passes=1, credit="schedule", share_target="change", **extra)
		metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, targets, np.zeros(8, dtype=np.float32), config, decode=decode, growth_real=growth_real, end=end, share_trust=0.4, growth_trust=1.0)
		assert np.isfinite(metrics["kl"]) and metrics["steps"] >= 1
		assert not any(name.startswith("step_") or name.startswith("schedule_step_") for name in metrics)


def test_the_step_credit_keeps_each_class_mean_at_the_outcomes_and_charges_each_action():
	"""train.step_credit's classes and price: a class after whose decisions the forecast is biased (its rest's mean) is
	credited the outcome's mean all the same (less the prior's shrink), whatever the step's share; without the classes
	it loses (1 - s) x the bias. The price is taken from each row with a step, never from one without."""
	from campaign_model.train import STEP_CLASS_PRIOR, step_credit

	n = 4000
	generator = torch.Generator().manual_seed(5)
	forecast = torch.randn(n + 1, generator=generator) * 0.01
	step = torch.arange(1, n + 2)
	step[-1] = -1  # (the last row: no next turn decision)
	day = torch.ones(n + 1)
	classes = torch.arange(n + 1) % 2
	bias = torch.where(classes == 1, 0.05, 0.0)  # (the forecast 0.05 too low right after a class-1 decision)
	moved = forecast[1:] - forecast[:-1]
	rest = bias[:-1] + torch.randn(n, generator=generator) * 0.07
	term = torch.cat([moved + rest, torch.tensor([0.3])])
	s = 1 - 0.75 * 1.0
	got = step_credit(term, forecast, step, day, trust=1.0, classes=classes)
	plain = step_credit(term, forecast, step, day, trust=1.0)
	for k in (0, 1):
		rows = (classes == k)[:-1]
		shrink = STEP_CLASS_PRIOR / (int(rows.sum()) + STEP_CLASS_PRIOR)
		assert float(got[:-1][rows].mean()) == pytest.approx(float(term[:-1][rows].mean()) - (1 - s) * float(rest[rows].mean()) * shrink, abs=1e-6)
		assert float(plain[:-1][rows].mean()) == pytest.approx(float(term[:-1][rows].mean()) - (1 - s) * float(rest[rows].mean()), abs=1e-6)
	assert float(term[:-1][(classes == 1)[:-1]].mean() - plain[:-1][(classes == 1)[:-1]].mean()) == pytest.approx(0.75 * 0.05, abs=0.006)
	assert float(got[-1]) == pytest.approx(0.3) and float(plain[-1]) == pytest.approx(0.3)  # (no step: the term alone)
	# the step alone where the term is unknown: with its class's whole mean rest
	unknown = term.clone()
	unknown[10] = float("nan")
	alone = step_credit(unknown, forecast, step, day, trust=1.0, classes=classes)
	rows = ((classes == 0)[:-1]) & (torch.arange(n) != 10)
	assert float(alone[10]) == pytest.approx(float(moved[10]) + float(rest[rows].sum()) / (int(rows.sum()) + STEP_CLASS_PRIOR), abs=1e-6)
	# the price: from every row with a step
	price = torch.full((n + 1,), 2e-4)
	priced = step_credit(term, forecast, step, day, 1.0, classes, price)
	assert torch.allclose(got[:-1] - priced[:-1], torch.full((n,), 2e-4), atol=1e-7) and float(priced[-1]) == pytest.approx(0.3)


def test_the_step_credits_classes_and_its_share_forecasts_trust():
	"""train.step_classes: an action type's class, an exchange's apart, every question one class. train.forecast_trust:
	a forecast's explained variance over trust_full_ev on the rows where both are known, 0..1."""
	from campaign_model.train import forecast_trust, step_classes

	kind = torch.tensor([c.DecisionKind.TURN, c.DecisionKind.EXCHANGE, c.DecisionKind.TURN, c.DecisionKind.YES_NO, c.DecisionKind.SELECT])
	action = torch.tensor([c.ActionType.MULTI, c.ActionType.MULTI, c.ActionType.END, -1, -1])
	classes = step_classes(kind, action)
	assert len(set(classes[:3].tolist())) == 3 and int(classes[3]) == int(classes[4]) == 0 and int(classes.min()) >= 0
	generator = torch.Generator().manual_seed(7)
	signal, noise = torch.randn(20000, generator=generator), torch.randn(20000, generator=generator)
	real = signal + noise  # (half of it explained by a forecast of the signal)
	assert forecast_trust(real, signal, 1.0) == pytest.approx(0.5, abs=0.03)
	assert forecast_trust(real, signal, 0.4) == 1.0  # (capped)
	assert forecast_trust(real, torch.zeros_like(real), 0.4) == pytest.approx(0.0, abs=0.01)
	assert forecast_trust(real, -signal, 0.4) == 0.0  # (worse than a constant: none)
	holes = real.clone()
	holes[::2] = float("nan")
	assert forecast_trust(holes, signal, 1.0) == pytest.approx(0.5, abs=0.04)
	assert forecast_trust(torch.full((4,), float("nan")), torch.zeros(4), 0.4) == 0.0


def test_the_schedule_credits_action_price_charges_each_decision_its_actions_ahead():
	"""config.schedule_action_price with the learner's end["actions"] (labels.actions_ahead): the update runs and its
	stats carry the charge (the price x the count); at 0 (the default), or without the counts, nothing of it."""
	from campaign_model.model import build_model

	torch.manual_seed(3)
	model = build_model({"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"})
	x = random_inputs(8, seed=11)
	decisions = decisions_for(model.eval(), x)
	targets = np.full((8, c.N_VALUE_OFFSETS), 0.55, dtype=np.float32)
	growth_real = np.full((8, c.N_VALUE_OFFSETS), 10.0, dtype=np.float32)
	growth_real[7:] = np.nan  # (the last: no 7-day outcome, no credit and no charge counted)
	actions = np.array([8, 7, 6, 5, 4, 3, 2, 100], dtype=np.float32)
	end = {"targets": np.full(8, 0.5, np.float32), "real": np.full(8, 0.5, np.float32), "step": np.array([1, 2, 3, 4, 5, 6, 7, -1]), "week": np.full(8, -1), "price": np.zeros(8, np.float32), "outcome": 0.5, "weight": 0.0}
	decode = lambda rows: {k: v[rows] for k, v in x.items()}  # noqa: E731
	run = lambda config, end: ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, targets, np.zeros(8, dtype=np.float32), config, decode=decode, growth_real=growth_real, end=end, share_trust=0.4, growth_trust=1.0)[0]  # noqa: E731
	on = CampaignConfig(batch_size=4, passes=1, credit="schedule", share_target="change", schedule_action_price=2e-6)
	metrics = run(on, {**end, "actions": actions})
	assert metrics["schedule_action_price_mean"] == pytest.approx(2e-6 * actions[:7].mean(), rel=1e-4) and np.isfinite(metrics["kl"]) and metrics["steps"] >= 1
	assert metrics["schedule_action_price_std"] == pytest.approx(2e-6 * actions[:7].std(ddof=1), rel=1e-3)
	assert "schedule_action_price_mean" not in run(on, end)  # (no counts: an older caller)
	off = CampaignConfig(batch_size=4, passes=1, credit="schedule", share_target="change")
	assert off.schedule_action_price == 0.0 and "schedule_action_price_mean" not in run(off, {**end, "actions": actions})
