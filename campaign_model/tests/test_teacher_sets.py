"""The teacher over the turn (advice.py: Nullkiller2's advice and every entry that follows it) and the loss shares
(learner.loss_shares, _note_gradients: each term's weight from its running gradient norm)."""
import math
import struct
import zlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model.advice import _RECRUIT, teacher_sets, town_item
from campaign_model.config import CampaignConfig
from campaign_model.learner import LOSS_SCALE_STEP, TEACHER_SHARE_MAX, CampaignLearner, set_agreement, teacher_share_for
from campaign_model.records import ACCEPT, ADVICE, ADVICE_FIELDS, HEADER, IN_BYTES, parse_records
from campaign_model.train import entries_logp

A = c.ActionType


def turn(action, index, advice=None, accept=None, day=1, color="RED", kind=c.DecisionKind.TURN):
	return SimpleNamespace(color=color, day=day, kind=kind, action_type=int(action), index=index, advice=advice, accept=accept)


def entries(types, indices, valid, row):
	return {(int(t), int(i)) for t, i, v in zip(types[row], indices[row], valid[row]) if v}


def test_a_town_action_advised_stays_the_teachers_until_done_never_before():
	"""Nullkiller2 advises building 7 at the second decision; we move twice, then build it: the build is the teacher's at
	the third and fourth decisions (any order of the turn's actions), not at the first (before it was named) and not
	after we built it; the next day starts empty. The plan's steps recorded at a decision (accept) are its alone."""
	build = (int(A.BUILD), 7, -1)
	decisions = [
		turn(A.MOVE, 10, accept=(int(A.MOVE), 11, int(A.TRAVEL), 5)),
		turn(A.MOVE, 12, advice=build),
		turn(A.MOVE, 13),
		turn(A.BUILD, 7),
		turn(A.END, -1),
		turn(A.MOVE, 14, day=2),
	]
	types, indices, valid = teacher_sets(decisions)
	assert entries(types, indices, valid, 0) == {(int(A.MOVE), 11), (int(A.TRAVEL), 5)}
	assert entries(types, indices, valid, 1) == set()  # (the advice's own entry is the advice, not in its set)
	assert entries(types, indices, valid, 2) == {(int(A.BUILD), 7)} == entries(types, indices, valid, 3)
	assert entries(types, indices, valid, 4) == set() and entries(types, indices, valid, 5) == set()


def test_a_gather_accepted_is_not_the_teachers_the_other_multi_actions_are():
	"""advice.taught: Advice.cpp accepts GATHER wherever it can be made (Nullkiller2 takes every garrison on a visit);
	the teacher drops it (it taught our model to leave its towns empty), keeps RECRUIT_ALL and GIVE_ALL."""
	gather, recruit_all = 2 * c.N_MULTI + c.MultiKind.GATHER, 2 * c.N_MULTI + c.MultiKind.RECRUIT_ALL
	give_all = 1 * c.N_MULTI + c.MultiKind.GIVE_ALL
	decisions = [
		turn(A.MOVE, 10, accept=(int(A.MULTI), gather, int(A.MULTI), recruit_all, int(A.MOVE), 11)),
		turn(A.ARMY, 3, accept=(int(A.MULTI), give_all), kind=c.DecisionKind.EXCHANGE),
	]
	types, indices, valid = teacher_sets(decisions)
	assert entries(types, indices, valid, 0) == {(int(A.MULTI), recruit_all), (int(A.MOVE), 11)}
	assert entries(types, indices, valid, 1) == {(int(A.MULTI), give_all)}


def test_a_recruit_is_the_same_item_at_any_amount_and_another_players_turn_is_apart():
	index = int(np.ravel_multi_index((1, 2, 0, 0), _RECRUIT))
	identity, all_amounts = town_item(A.RECRUIT, index)
	assert len(all_amounts) == _RECRUIT[2] and index in all_amounts and town_item(A.MOVE, 3) is None
	decisions = [
		turn(A.MOVE, 10, advice=(int(A.RECRUIT), index, -1)),
		turn(A.MOVE, 10, color="BLUE"),
		turn(A.RECRUIT, all_amounts[-1]),  # (a different amount: the same recruit, done)
		turn(A.MOVE, 11),
	]
	types, indices, valid = teacher_sets(decisions)
	assert entries(types, indices, valid, 1) == set()  # (BLUE's turn: RED's advice is not its)
	assert entries(types, indices, valid, 2) == {(int(A.RECRUIT), i) for i in all_amounts}
	assert entries(types, indices, valid, 3) == set()


def test_a_record_with_more_entries_that_follow_nk2_reads_back_with_them():
	packed = zlib.compress(bytes(IN_BYTES), 1)
	values = [0.5] * c.N_VALUE_OFFSETS
	accept = [int(A.MOVE), 1234, int(A.TRAVEL), 77, int(A.ARMY), 9]
	record = HEADER.pack(3, c.DecisionKind.TURN, 0, 7, A.MOVE, 12, -1, -0.5, *values, ADVICE | ACCEPT, IN_BYTES) + ADVICE_FIELDS.pack(A.MOVE, 40, -1) + struct.pack("<H", 3) + struct.pack("<" + "hi" * 3, *accept) + struct.pack("<I", len(packed)) + packed
	(d,) = parse_records("RED", b"AMZ2" + struct.pack("<I", 1) + record)
	assert d.advice == (A.MOVE, 40, -1) and d.accept == tuple(accept) and zlib.decompress(d.inputs) == bytes(IN_BYTES)


def test_entries_logp_reads_each_table_and_gives_illegal_entries_no_probability():
	n = 2
	out = {"type_logp": torch.log_softmax(torch.randn(n, c.N_ACTION_TYPES), dim=1)}
	from campaign_model.schema import HEADS, OUTPUTS

	shapes = {spec.name: spec.shape for spec in OUTPUTS}
	for _, (name, _) in HEADS.items():
		out[name] = torch.log_softmax(torch.randn((n,) + shapes[name]).reshape(n, -1), dim=1).reshape((n,) + shapes[name])
	out["move_logp"] = out["move_logp"].reshape(n, -1).index_fill(1, torch.tensor([5]), float("-inf")).reshape((n,) + shapes["move_logp"])
	types = torch.tensor([[A.MOVE, A.BUILD, A.MOVE], [A.TRAVEL, A.MOVE, A.MOVE]])
	indices = torch.tensor([[3, 2, 5], [4, 5, 0]])
	valid = torch.tensor([[True, True, True], [True, True, False]])
	logp = entries_logp(out, types, indices, valid)
	expect = out["type_logp"][0, A.MOVE] + out["move_logp"].reshape(n, -1)[0, 3]
	assert torch.isclose(logp[0, 0], expect) and torch.isclose(logp[0, 1], out["type_logp"][0, A.BUILD] + out["build_logp"].reshape(n, -1)[0, 2])
	assert logp[0, 2] == -1e4 and logp[1, 1] == -1e4 and logp[1, 2] == -1e4  # (illegal; not valid)
	assert torch.isfinite(torch.logsumexp(torch.full((1, 3), -1e4), dim=1)).all()


def learner_with(score=None, weight=10.7, shares=True):
	config = CampaignConfig(loss_shares=shares)
	learner = SimpleNamespace(config=config, teacher_weight=weight, teacher_share=teacher_share_for(score) if score is not None else None, loss_scale=None)
	for name in ("loss_shares", "loss_weights", "_note_gradients"):
		setattr(learner, name, getattr(CampaignLearner, name).__get__(learner))
	return learner


def test_the_teacher_share_follows_the_score_and_policy_and_value_keep_theirs():
	assert teacher_share_for(0.285) == TEACHER_SHARE_MAX and teacher_share_for(1.0) == pytest.approx(0.2)
	shares = learner_with(score=0.285).loss_shares()
	assert sum(shares.values()) == pytest.approx(1.0) and shares["policy"] >= 0.15 and shares["value"] >= 0.15
	assert learner_with(score=1.0).loss_shares()["policy"] == pytest.approx(0.4)


def test_each_weight_steps_toward_its_share_and_no_faster():
	"""Measured norms at weight 1 (grad_raw_*): the value's 1e-4 of the policy's, the teacher's 2x: each weight moves at
	most LOSS_SCALE_STEP an update toward share / share_policy x norm_policy / norm, and gets there."""
	learner = learner_with(score=0.285)
	start = learner.loss_weights()
	assert start["teacher"] == 10.7 and start["value"] == 1.0 and start["kl_teacher"] == pytest.approx(3.0)
	shares = learner.loss_shares()
	for _ in range(80):
		metrics = {"grad_raw_policy": 0.1, "grad_raw_value": 1e-5, "grad_raw_teacher": 0.2, "grad_raw_foresight": 0.01}
		before = dict(learner.loss_scale["weight"]) if learner.loss_scale else {"value": 1.0, "teacher": 10.7, "foresight": 1.0}
		learner._note_gradients(metrics)
		for term, w in learner.loss_scale["weight"].items():
			assert w / before[term] <= LOSS_SCALE_STEP * (1 + 1e-9) and before[term] / w <= LOSS_SCALE_STEP * (1 + 1e-9)
	weights = learner.loss_scale["weight"]
	assert weights["value"] == pytest.approx(shares["value"] / shares["policy"] * 0.1 / 1e-5, rel=1e-6)
	assert weights["teacher"] == pytest.approx(shares["teacher"] / shares["policy"] * 0.1 / 0.2, rel=1e-6)
	assert metrics["grad_shared_value"] == pytest.approx(1e-5 * before["value"])  # (as it entered the loss: at the update's weight)


def test_the_value_share_is_a_floor_not_a_target():
	"""learner.VALUE_SHARE_CAP: the value weight rises while the value's share is under its target, holds between it and
	VALUE_SHARE_CAP times it (heads with a lot to re-fit keep their gradient), and is cut past that."""
	from campaign_model.learner import VALUE_SHARE_CAP

	learner = learner_with(score=0.285)
	shares = learner.loss_shares()
	metrics = {"grad_raw_policy": 0.1, "grad_raw_value": 1e-5, "grad_raw_teacher": 0.2, "grad_raw_foresight": 0.01}
	for _ in range(80):  # (at its share: weight = share / share_policy x norm_policy / norm)
		learner._note_gradients(dict(metrics))
	settled = learner.loss_scale["weight"]["value"]
	assert settled == pytest.approx(shares["value"] / shares["policy"] * 0.1 / 1e-5, rel=1e-6)
	for _ in range(60):  # the value's gradient twice as large: its share twice the target, the weight holds
		learner._note_gradients(dict(metrics, grad_raw_value=2e-5))
	assert learner.loss_scale["weight"]["value"] == pytest.approx(settled, rel=1e-6)
	for _ in range(80):  # five times: past the cap, cut to VALUE_SHARE_CAP times the share
		learner._note_gradients(dict(metrics, grad_raw_value=5e-5))
	assert learner.loss_scale["weight"]["value"] == pytest.approx(VALUE_SHARE_CAP * shares["value"] / shares["policy"] * 0.1 / 5e-5, rel=1e-3)
	for _ in range(80):  # a tenth of the first: under the share, raised back to it
		learner._note_gradients(dict(metrics, grad_raw_value=1e-6))
	assert learner.loss_scale["weight"]["value"] == pytest.approx(shares["value"] / shares["policy"] * 0.1 / 1e-6, rel=1e-3)


def test_without_the_shares_the_teacher_is_at_its_weight_and_nothing_moves():
	learner = learner_with(shares=False, weight=6.4)
	assert learner.loss_weights() == {"value": 1.0, "foresight": 1.0, "teacher": 6.4}
	metrics = {"grad_raw_policy": 0.1, "grad_raw_teacher": 0.2}
	learner._note_gradients(metrics)
	assert learner.loss_scale is None and metrics["grad_shared_teacher"] == pytest.approx(1.28)


def test_set_agreement_counts_the_advice_or_any_entry_of_its_set():
	"""The advice's own entry or one of its set; an end by its type alone (ours has index -1, the advice's end 0)."""
	decisions = [turn(A.MOVE, 11, advice=(int(A.MOVE), 10, -1)), turn(A.MOVE, 10, advice=(int(A.MOVE), 10, -1)), turn(A.BUILD, 3, advice=(int(A.MOVE), 10, -1)), turn(A.END, -1)]
	types = np.array([[A.MOVE], [A.MOVE], [A.MOVE], [A.MOVE]])
	indices = np.array([[11], [12], [12], [12]])
	valid = np.array([[True], [True], [True], [False]])
	assert set_agreement(decisions, types, indices, valid) == 2
	ended = [turn(A.END, -1, advice=(int(A.END), 0, -1)), turn(A.END, -1, advice=(int(A.MOVE), 10, -1))]
	assert set_agreement(ended, np.array([[A.MOVE], [A.MOVE]]), np.array([[12], [12]]), np.array([[False], [False]])) == 1


def test_an_update_with_the_teachers_sets_and_the_loss_shares_runs():
	"""ppo_update with teacher_sets (the advice and more entries) and loss_weights: it trains, the KL target is the
	teacher's share's, and a row whose set is all illegal adds nothing (no NaN)."""
	from campaign_model.model import build_model
	from campaign_model.tests.test_model import random_inputs
	from campaign_model.tests.test_train import decisions_for
	from campaign_model.train import ppo_update

	torch.manual_seed(3)
	model = build_model({"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"})
	x = random_inputs(8, seed=11)
	decisions = decisions_for(model.eval(), x)
	for i, d in enumerate(decisions):
		if d.action_type >= 0:
			d.advice = (d.action_type, d.index, d.cell)
	n = len(decisions)
	types, indices, valid = np.zeros((n, 4), np.int64), np.zeros((n, 4), np.int64), np.zeros((n, 4), bool)
	types[:, 0], indices[:, 0], valid[:, 0] = A.MOVE, 0, True  # (mostly illegal: probability 0)
	before = [p.detach().clone() for p in model.parameters()]
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, np.full((n, c.N_VALUE_OFFSETS), 0.5, np.float32), np.zeros(n, np.float32), CampaignConfig(batch_size=4, passes=1, model_architecture="v2"), decode=lambda rows: {k: v[rows] for k, v in x.items()}, teacher_weight=2.0, teacher_sets=(types, indices, valid), loss_weights={"value": 3.0, "foresight": 0.5, "kl_teacher": 1.5})
	assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
	assert all(math.isfinite(float(p.detach().sum())) for p in model.parameters())
	assert metrics["kl_target"] == pytest.approx(CampaignConfig().target_kl) and math.isfinite(metrics["teacher_ce"])  # (no teacher inflation since 2026-11-06: config.kl_teacher_inflation)


def test_the_teachers_label_is_weighed_against_its_family_and_end_only():
	"""train.TEACHER_SPAN / teacher_span_logp (2026-11-05): a hero's label (MOVE, TRAVEL, ARMY; a CAST) among the hero's
	actions and END, a town action's (BUILD, RECRUIT, HIRE, UPGRADE, TRADE, BUY) and an END among both; an exchange's among
	ARMY and END; an accepted entry of another type with them. What lies outside gets no gradient: the teacher never asks
	for ARTIFACT, DISMISS or a MULTI, and a town action at a hero move's decision is not its to judge."""
	from campaign_model.train import TEACHER_SPAN, teacher_span_logp

	hero, town = {A.MOVE, A.TRAVEL, A.ARMY}, {A.BUILD, A.RECRUIT, A.HIRE, A.UPGRADE, A.TRADE, A.BUY}
	spans = lambda exchange, label: {A(t) for t in range(c.N_ACTION_TYPES) if TEACHER_SPAN[exchange, label, t]}  # noqa: E731
	assert spans(0, A.MOVE) == hero | {A.END, A.TRADE, A.MULTI} and spans(0, A.BUILD) == hero | town | {A.END, A.MULTI}  # (TRADE, MULTI at every turn label)
	assert spans(0, A.END) == hero | town | {A.END, A.MULTI} and spans(0, A.CAST) == hero | {A.CAST, A.END, A.TRADE, A.MULTI}
	assert spans(1, A.ARMY) == {A.ARMY, A.END} and spans(1, A.END) == {A.ARMY, A.END}  # (an exchange's MULTI outside)

	torch.manual_seed(0)
	logits = torch.randn(4, c.N_ACTION_TYPES, requires_grad=True)
	out = {"type_logp": torch.log_softmax(logits, dim=1)}
	kind = torch.tensor([c.DecisionKind.TURN, c.DecisionKind.TURN, c.DecisionKind.TURN, c.DecisionKind.EXCHANGE])
	label = torch.tensor([A.MOVE, A.BUILD, A.END, A.ARMY])
	# row 0 has an accepted entry of the other family (a pending recruit: counted beside the hero's family)
	accepted = torch.full((4, 2), -1e4)
	accepted[0, 0] = out["type_logp"][0, A.RECRUIT] - 1.0
	accept_types = torch.tensor([[A.RECRUIT, 0], [0, 0], [0, 0], [0, 0]])
	span = teacher_span_logp(out, kind, label, accepted, accept_types)
	p = out["type_logp"].exp()
	expected = [
		torch.log(sum(p[0, t] for t in hero | {A.END, A.TRADE, A.MULTI}) + p[0, A.RECRUIT] * math.exp(-1.0)),
		torch.log(sum(p[1, t] for t in hero | town | {A.END, A.MULTI})),
		torch.log(sum(p[2, t] for t in hero | town | {A.END, A.MULTI})),
		torch.log(p[3, A.ARMY] + p[3, A.END]),
	]
	assert torch.allclose(span, torch.stack(expected), atol=1e-5)
	loss = -(out["type_logp"].gather(1, label.unsqueeze(1)).squeeze(1) - span).sum()
	loss.backward()
	never = [A.ARTIFACT, A.DISMISS, A.CAST]
	assert all(float(logits.grad[r, t]) == pytest.approx(0.0, abs=1e-6) for r in range(4) for t in never)
	assert float(logits.grad[3, A.MULTI]) == pytest.approx(0.0, abs=1e-6) and float(logits.grad[0, A.MULTI]) > 0  # (an exchange's free, a turn's pushed down)
	assert all(float(logits.grad[0, t]) == pytest.approx(0.0, abs=1e-6) for t in town - {A.RECRUIT, A.TRADE})  # (a hero move's: the town's untouched)
	assert all(float(logits.grad[1, t]) > 0 for t in hero)  # (a town action's: the heroes' pushed down, its visitor stays to buy)
	assert float(logits.grad[0, A.MOVE]) < 0 and float(logits.grad[0, A.END]) > 0  # (the label up, END down)


def test_with_exchange_span_multi_an_exchange_label_pushes_down_the_multi_actions_it_does_not_accept():
	"""config.exchange_span_multi (train.TEACHER_SPAN_MULTI, 2026-11-09): an exchange's label weighed against ARMY, END
	and MULTI; the TAKE_BEST its advice accepts counted with its label, every other MULTI pushed down. Turn decisions'
	spans as TEACHER_SPAN's."""
	from campaign_model.train import TEACHER_SPAN, TEACHER_SPAN_MULTI, teacher_span_logp

	assert torch.equal(TEACHER_SPAN_MULTI[0], TEACHER_SPAN[0])
	assert {A(t) for t in range(c.N_ACTION_TYPES) if TEACHER_SPAN_MULTI[1, A.ARMY, t]} == {A.ARMY, A.END, A.MULTI}
	assert {A(t) for t in range(c.N_ACTION_TYPES) if TEACHER_SPAN_MULTI[1, A.END, t]} == {A.ARMY, A.END, A.MULTI}
	torch.manual_seed(1)
	logits = torch.randn(2, c.N_ACTION_TYPES, requires_grad=True)
	out = {"type_logp": torch.log_softmax(logits, dim=1)}
	kind = torch.tensor([c.DecisionKind.EXCHANGE, c.DecisionKind.EXCHANGE])
	label = torch.tensor([A.ARMY, A.END])
	p = out["type_logp"].exp()
	span = teacher_span_logp(out, kind, label, exchange_multi=True)
	assert torch.allclose(span, torch.log(p[:, A.ARMY] + p[:, A.END] + p[:, A.MULTI]), atol=1e-5)
	assert torch.allclose(teacher_span_logp(out, kind, label), torch.log(p[:, A.ARMY] + p[:, A.END]), atol=1e-5)  # (the default: as before)
	(-(out["type_logp"].gather(1, label.unsqueeze(1)).squeeze(1) - span).sum()).backward()
	assert float(logits.grad[0, A.MULTI]) > 0 and float(logits.grad[1, A.MULTI]) > 0  # (pushed down)


def test_with_exchange_span_advised_only_the_exchanges_its_advice_opened_weigh_the_multi_actions():
	"""config.exchange_span_advised (train.advised_exchanges): the exchange decisions after a turn decision that was the
	advice's own MOVE are the rows MULTI is in the span for; an exchange our own move opened, a turn decision and a
	second player's rows are not. teacher_span_logp takes the rows' mask."""
	from campaign_model.train import advised_exchanges, teacher_span_logp

	def d(color, kind, action, index, advice=None):
		return SimpleNamespace(color=color, kind=int(kind), action_type=int(action), index=index, advice=advice)

	T, X = c.DecisionKind.TURN, c.DecisionKind.EXCHANGE
	decisions = [
		d("RED", T, A.MOVE, 40, (int(A.MOVE), 40, -1)),  # the advice's own move
		d("RED", X, A.MULTI, 4),  # the exchange it opened
		d("RED", X, A.END, 0),
		d("RED", T, A.MOVE, 41, (int(A.MOVE), 40, -1)),  # another tile than the advice's
		d("RED", X, A.MULTI, 4),
		d("RED", T, A.MOVE, 42),  # no advice
		d("RED", X, A.MULTI, 4),
		d("RED", T, A.TRAVEL, 40, (int(A.TRAVEL), 40, -1)),  # its advice, not a move
		d("BLUE", X, A.MULTI, 4),  # another player's: no turn decision of its own before it
		d("BLUE", T, A.MOVE, 7, (int(A.MOVE), 7, -1)),
		d("BLUE", c.DecisionKind.YES_NO, -1, 1),  # a question in between changes nothing
		d("BLUE", X, A.ARMY, 3),
	]
	rows = advised_exchanges(decisions)
	assert rows.tolist() == [False, True, True, False, False, False, False, False, False, False, False, True]
	stream = np.array([0] * 8 + [1] * 4)
	assert advised_exchanges(decisions, stream).tolist() == rows.tolist()
	torch.manual_seed(2)
	out = {"type_logp": torch.log_softmax(torch.randn(3, c.N_ACTION_TYPES), dim=1)}
	kind = torch.tensor([X, X, T])
	label = torch.tensor([A.ARMY, A.ARMY, A.MOVE])
	p = out["type_logp"].exp()
	span = teacher_span_logp(out, kind, label, exchange_multi=torch.tensor([True, False, True]))
	assert float(span[0]) == pytest.approx(float(torch.log(p[0, A.ARMY] + p[0, A.END] + p[0, A.MULTI])), abs=1e-5)
	assert float(span[1]) == pytest.approx(float(torch.log(p[1, A.ARMY] + p[1, A.END])), abs=1e-5)
	assert float(span[2]) == pytest.approx(float(teacher_span_logp(out, kind, label)[2]), abs=1e-6)  # (a turn's: as ever)


def test_a_gather_label_is_weighed_against_the_heros_moves_and_taught_at_its_own_small_weight():
	"""config.nk2_gather_weight (train.advice_weights, gather_advice; Advice.cpp visitAgain): Nullkiller2's GATHER advice
	counts for nothing at 0 (the default) and at that weight above it, every other advice at 1 as before (its TRADE and
	schema 16's new kinds still at 0); a turn's MULTI label is weighed against the hero's moves and the town's actions,
	so leaving without the troops is pushed down."""
	from campaign_model.config import CampaignConfig
	from campaign_model.train import TEACHER_SPAN, advice_weights, gather_advice, taught, teacher_span_logp

	K = c.MultiKind
	advice = np.array([
		(A.MULTI, 3 * c.N_MULTI + K.GATHER, -1),  # the pickup, at a town's fourth slot
		(A.MOVE, 40, -1),
		(A.TRADE, 2, -1),  # (config.nk2_trade_advice: off)
		(A.MULTI, 1 * c.N_MULTI + K.SPLIT_SINGLES, -1),  # (config.nk2_new_advice: off)
		(A.MULTI, 2 * c.N_MULTI + K.RECRUIT_ALL, -1),
		(A.MULTI, 0 * c.N_MULTI + K.GATHER, -1),  # a pickup at a decision with no advice (has: False)
	], dtype=np.int64)
	has = np.array([True, True, True, True, True, False])
	assert gather_advice(advice).tolist() == [True, False, False, False, False, True]
	off, low = CampaignConfig(), CampaignConfig(nk2_gather_weight=0.1)
	assert taught(advice, has, off).tolist() == [False, True, False, False, True, False]
	assert taught(advice, has, low).tolist() == [True, True, False, False, True, False]
	assert advice_weights(advice, has, off).tolist() == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
	weights = advice_weights(advice, has, low)
	assert weights.dtype == np.float32 and weights.tolist() == pytest.approx([0.1, 1.0, 0.0, 0.0, 1.0, 0.0])

	hero, town = {A.MOVE, A.TRAVEL, A.ARMY}, {A.BUILD, A.RECRUIT, A.HIRE, A.UPGRADE, A.TRADE, A.BUY}
	assert {A(t) for t in range(c.N_ACTION_TYPES) if TEACHER_SPAN[0, A.MULTI, t]} == hero | town | {A.END, A.MULTI}
	assert {A(t) for t in range(c.N_ACTION_TYPES) if TEACHER_SPAN[1, A.MULTI, t]} == {A.ARMY, A.END, A.MULTI}  # (an exchange's: as before)
	torch.manual_seed(3)
	logits = torch.randn(1, c.N_ACTION_TYPES, requires_grad=True)
	out = {"type_logp": torch.log_softmax(logits, dim=1)}
	label = torch.tensor([A.MULTI])
	span = teacher_span_logp(out, torch.tensor([c.DecisionKind.TURN]), label)
	(-(out["type_logp"].gather(1, label.unsqueeze(1)).squeeze(1) - span).sum()).backward()
	assert float(logits.grad[0, A.MULTI]) < 0 and all(float(logits.grad[0, t]) > 0 for t in (A.MOVE, A.TRAVEL, A.END, A.RECRUIT))
	assert all(float(logits.grad[0, t]) == pytest.approx(0.0, abs=1e-6) for t in (A.ARTIFACT, A.DISMISS, A.CAST))


def test_an_update_with_the_gather_weight_and_the_advised_exchanges_span_runs():
	"""ppo_update with config.nk2_gather_weight (a GATHER advice at a fraction of a row) and config.exchange_span_advised
	(the rows' mask cut with every micro-batch): it trains, nothing is NaN; with the weight at 0 the GATHER row adds
	nothing to the teacher's count."""
	from campaign_model.model import build_model
	from campaign_model.tests.test_model import random_inputs
	from campaign_model.tests.test_train import decisions_for
	from campaign_model.train import advice_weights, ppo_update

	torch.manual_seed(3)
	model = build_model({"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"})
	x = random_inputs(8, seed=11)
	decisions = decisions_for(model.eval(), x)
	advised = [i for i, d in enumerate(decisions) if d.action_type >= 0]
	for i in advised:
		decisions[i].advice = (decisions[i].action_type, decisions[i].index, decisions[i].cell)
	decisions[advised[0]].advice = (int(A.MULTI), 0 * c.N_MULTI + int(c.MultiKind.GATHER), -1)
	n = len(decisions)
	types, indices, valid = np.zeros((n, 4), np.int64), np.zeros((n, 4), np.int64), np.zeros((n, 4), bool)
	advice = np.array([d.advice if d.advice is not None else (0, 0, -1) for d in decisions], dtype=np.int64)
	has = np.array([d.advice is not None for d in decisions])
	assert advice_weights(advice, has, CampaignConfig())[advised[0]] == 0.0
	for config in (CampaignConfig(batch_size=4, passes=1, model_architecture="v2", nk2_gather_weight=0.1, exchange_span_advised=True), CampaignConfig(batch_size=4, passes=1, model_architecture="v2", exchange_span_advised=True), CampaignConfig(batch_size=4, passes=1, model_architecture="v2", nk2_gather_weight=0.1)):
		metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, np.full((n, c.N_VALUE_OFFSETS), 0.5, np.float32), np.zeros(n, np.float32), config, decode=lambda rows: {k: v[rows] for k, v in x.items()}, teacher_weight=2.0, teacher_sets=(types, indices, valid))
		assert math.isfinite(metrics["teacher_ce"]) and math.isfinite(metrics["kl"])
		assert all(math.isfinite(float(p.detach().sum())) for p in model.parameters())


def test_the_games_record_the_gather_label_only_when_the_trainer_teaches_it():
	"""supply.Runner.campaign_settings: a training game's ALPHA_MAP_GATHER_ADVICE is set with config.nk2_gather_weight
	above 0 and Nullkiller2's advice on (one switch for the label and its weight); never without the advice."""
	from battle_model.supply import Runner

	def settings(**campaign):
		runner = SimpleNamespace(battle_socket="/tmp/b.sock", link="link.pt", records=SimpleNamespace(path="/tmp/r.sock"), config=SimpleNamespace(campaign=CampaignConfigFor(**campaign)))
		return dict(Runner.campaign_settings(runner, "game", 3))

	from campaign_model.config import CampaignConfig as CampaignConfigFor

	assert "ALPHA_MAP_GATHER_ADVICE" not in settings(nk2_advice=0.5) and settings(nk2_advice=0.5)["ALPHA_MAP_NK2_ADVICE"] == "0.5"
	assert settings(nk2_advice=0.5, nk2_gather_weight=0.1)["ALPHA_MAP_GATHER_ADVICE"] == "1"
	assert "ALPHA_MAP_GATHER_ADVICE" not in settings(nk2_gather_weight=0.1) and "ALPHA_MAP_NK2_ADVICE" not in settings(nk2_gather_weight=0.1)


def test_with_the_teacher_off_its_weight_and_share_are_zero_and_the_others_stand():
	"""config.nk2_teacher False (the user, 2026-11-11): the teacher's weight and its share of the KL target are 0, with
	the loss shares and without; the value's and the foresight's weights stay what they were."""
	learner = learner_with(score=0.285)
	taught = learner.loss_weights()
	learner.loss_scale["weight"].update(value=352.6, foresight=3.1)  # (as frozen in the run)
	learner.config = CampaignConfig(loss_shares=True, nk2_teacher=False)
	weights = learner.loss_weights()
	assert taught["teacher"] == 10.7 and weights["teacher"] == 0.0 and weights["kl_teacher"] == 0.0
	assert weights["value"] == 352.6 and weights["foresight"] == 3.1
	shares = learner.loss_shares()
	assert shares["teacher"] == 0.0 and sum(shares.values()) == pytest.approx(1.0) and shares["policy"] == pytest.approx(0.5)
	plain = learner_with(shares=False, weight=6.4)
	plain.config = CampaignConfig(loss_shares=False, nk2_teacher=False)
	assert plain.loss_weights() == {"value": 1.0, "foresight": 1.0, "teacher": 0.0}


def test_an_update_at_the_teacher_s_weight_0_moves_the_weights_as_one_without_its_advice():
	"""The teacher's loss off is its weight 0 (learner.loss_weights with config.nk2_teacher False): the update's weights
	end where an update of the same decisions without any advice ends, and its cross-entropy is still read."""
	from campaign_model.model import build_model
	from campaign_model.tests.test_model import random_inputs
	from campaign_model.tests.test_train import decisions_for
	from campaign_model.train import ppo_update

	def run(advised):
		torch.manual_seed(3)
		model = build_model({"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"})
		x = random_inputs(8, seed=11)
		decisions = decisions_for(model.eval(), x)
		if advised:
			for d in decisions:
				if d.action_type >= 0:
					d.advice = (d.action_type, d.index, d.cell)
		n = len(decisions)
		torch.manual_seed(5)
		metrics, _ = ppo_update(
			model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, np.full((n, c.N_VALUE_OFFSETS), 0.5, np.float32), np.zeros(n, np.float32),
			CampaignConfig(batch_size=4, passes=1, model_architecture="v2", nk2_teacher=False), decode=lambda rows: {k: v[rows] for k, v in x.items()}, teacher_weight=0.0,
			loss_weights={"value": 1.0, "foresight": 1.0, "teacher": 0.0, "kl_teacher": 0.0},
		)
		return [p.detach().clone() for p in model.parameters()], metrics

	with_advice, metrics = run(True)
	without, _ = run(False)
	assert math.isfinite(metrics["teacher_ce"]) and metrics["teacher_ce"] > 0  # (still a reading)
	assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(with_advice, without))
