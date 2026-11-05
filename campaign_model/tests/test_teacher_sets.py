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


def test_without_the_shares_the_teacher_is_at_its_weight_and_nothing_moves():
	learner = learner_with(shares=False, weight=6.4)
	assert learner.loss_weights() == {"value": 1.0, "foresight": 1.0, "teacher": 6.4}
	metrics = {"grad_raw_policy": 0.1, "grad_raw_teacher": 0.2}
	learner._note_gradients(metrics)
	assert learner.loss_scale is None and metrics["grad_shared_teacher"] == pytest.approx(1.28)


def test_set_agreement_counts_the_advice_or_any_entry_of_its_set():
	decisions = [turn(A.MOVE, 11, advice=(int(A.MOVE), 10, -1)), turn(A.MOVE, 10, advice=(int(A.MOVE), 10, -1)), turn(A.BUILD, 3, advice=(int(A.MOVE), 10, -1)), turn(A.END, -1)]
	types = np.array([[A.MOVE], [A.MOVE], [A.MOVE], [A.MOVE]])
	indices = np.array([[11], [12], [12], [12]])
	valid = np.array([[True], [True], [True], [False]])
	assert set_agreement(decisions, types, indices, valid) == 2


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
	assert metrics["kl_target"] == pytest.approx(CampaignConfig().target_kl * 2.5) and math.isfinite(metrics["teacher_ce"])


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
