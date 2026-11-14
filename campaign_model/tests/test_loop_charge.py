"""The loop charge (config.loop_charge; the user, 2026-11-14: penalties on exchanges that are strictly useless; e5's
design; e5's reset rule of 2026-11-14: anything either hero does between the two points breaks it): labels.loop_flags on
hand-made sequences (a null dialog, give and take back with nothing between; no loop with a step of the receiver, a
third hero met, a CAST, a battle or question, an unattributed decision between; an undone move inside a dialog that
goes on, another day = no loop, END never flagged), train.charge_loops after the
standardization on the flagged credited rows alone, the ramp by version, and the learner's update end to end (at 0:
nothing computed, bit for bit)."""
import collections
import dataclasses
import threading
from types import SimpleNamespace

import numpy as np
import torch

from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.labels import LOOP_ACROSS, LOOP_NONE, LOOP_NULL, LOOP_UNDONE, loop_flags
from campaign_model.learner import CampaignLearner, loop_charge_now
from campaign_model.tests.test_learner import Game, snapshot
from campaign_model.train import charge_loops

A = c.ActionType
TURN, EXCHANGE, QUESTION = int(c.DecisionKind.TURN), int(c.DecisionKind.EXCHANGE), int(c.DecisionKind.YES_NO)
PAIR = (7, 9)


THIRD = 5  # a third hero of ours
MEET = "meet"  # a turn MOVE of PAIR's first onto its second


def flags(events, resets=()):
	"""events: [(kind, action, the pair's state at the decision (or (state, day)), who)] -> loop_flags: who a turn
	decision's heroes (a set; None the record cannot say; MEET a MOVE onto the other), an exchange's pair; by default a
	turn MOVE is MEET, an exchange PAIR's, any other decision acts with no hero. Another pair's states never repeat."""
	steps, states, actors, meets = [], [], [], []
	for event in events:
		kind, action, state = event[:3]
		who = event[3] if len(event) > 3 else (MEET if kind == TURN and action == A.MOVE else PAIR if kind == EXCHANGE else frozenset())
		state, day = state if isinstance(state, tuple) else (state, 1)
		pair = (tuple(sorted(who)) if kind == EXCHANGE else None)
		steps.append((kind, int(action), day, pair))
		states.append(state)
		actors.append(frozenset({PAIR[0]}) if who == MEET else None if who is None else frozenset(who))
		meets.append(PAIR if who == MEET else (who[0], THIRD) if isinstance(who, tuple) and kind == TURN else None)
	state_of = lambda i, pair: states[i] if pair == PAIR else ("other", i)  # noqa: E731
	return loop_flags(steps, state_of, set(resets), actors, meets)


# give the army, take it back the same day: the two dialogs and the MOVEs that open them
GIVE = [(TURN, A.MOVE, "s0"), (EXCHANGE, A.ARMY, "s0"), (EXCHANGE, A.END, "s1")]
BACK = [(TURN, A.MOVE, "s1"), (EXCHANGE, A.ARMY, "s1"), (EXCHANGE, A.END, "s0"), (TURN, A.END, "s0")]


def test_a_null_dialog_flags_its_moves_and_its_meeting_but_never_end():
	out = flags([(TURN, A.MOVE, "s0"), (EXCHANGE, A.ARMY, "s0"), (EXCHANGE, A.ARMY, "s1"), (EXCHANGE, A.END, "s0"), (TURN, A.END, "s0")])
	assert out.tolist() == [LOOP_NULL, LOOP_NULL, LOOP_NULL, LOOP_NONE, LOOP_NONE]


def test_give_and_take_back_with_nothing_between_is_a_loop():
	assert flags(GIVE + BACK).tolist() == [LOOP_ACROSS, LOOP_ACROSS, LOOP_NONE, LOOP_ACROSS, LOOP_ACROSS, LOOP_NONE, LOOP_NONE]
	# (between them: a town's or the player's own decision, another hero's step, the other's MOVE onto the first: still one)
	for between in [(TURN, A.BUILD, "s1"), (TURN, A.MOVE, "s1", {THIRD}), (TURN, A.MOVE, "s1", MEET)]:
		assert flags(GIVE + [between] + BACK)[[0, 1, 4, 5]].tolist() == [LOOP_ACROSS] * 4, between


def test_the_receiver_steps_to_an_empty_tile_and_back_then_returns_the_army_is_no_loop():
	away = [(TURN, A.MOVE, "s1", {PAIR[1]}), (TURN, A.MOVE, "s1", {PAIR[1]})]
	assert not flags(GIVE + away + BACK).any()
	assert not flags(GIVE + away[:1] + BACK).any()  # (one step: the lender's spare movement saved, the army used)


def test_the_receiver_meets_a_third_hero_between_is_no_loop():
	third = [(TURN, A.MOVE, "s1", (PAIR[1],)), (EXCHANGE, A.ARMY, "s1", (PAIR[1], THIRD)), (EXCHANGE, A.END, "s1", (PAIR[1], THIRD))]
	assert not flags(GIVE + third + BACK).any()
	assert not flags(GIVE + third[1:] + BACK).any()  # (the third's dialog alone, opened by the third's MOVE or none)


def test_a_cast_or_any_act_of_the_pair_between_is_no_loop():
	for between in [(TURN, A.CAST, "s1", {PAIR[1]}), (TURN, A.TRAVEL, "s1", {PAIR[0]}), (TURN, A.RECRUIT, "s1", {PAIR[1]}), (TURN, A.ARMY, "s1", {PAIR[0], THIRD})]:
		assert not flags(GIVE + [between] + BACK).any(), between


def test_a_decision_the_record_cannot_attribute_is_no_loop():
	assert not flags(GIVE + [(TURN, A.ARMY, "s1", None)] + BACK).any()


def test_a_battle_or_a_question_between_is_no_loop():
	"""(the surrender case: the receiver's walk asks the player whether to accept the creatures' offer)"""
	assert not flags(GIVE + [(TURN, A.BUILD, "s1")] + BACK, resets={3}).any()  # (a battle recorded at the 4th decision)
	assert not flags(GIVE + [(QUESTION, 0, "s1")] + BACK, resets={3}).any()  # (loop_rows resets at every question)
	assert not flags(GIVE + BACK, resets={3}).any()  # (one at the second dialog's MOVE: the battle came before it)


def test_an_undone_move_inside_a_dialog_that_goes_on():
	out = flags([(TURN, A.MOVE, "s0"), (EXCHANGE, A.ARMY, "s0"), (EXCHANGE, A.ARMY, "s1"), (EXCHANGE, A.ARMY, "s0"), (EXCHANGE, A.END, "s2"), (TURN, A.END, "s2")])
	assert out.tolist() == [LOOP_NONE, LOOP_UNDONE, LOOP_UNDONE, LOOP_NONE, LOOP_NONE, LOOP_NONE]


def test_another_day_is_no_loop_and_a_change_kept_is_none():
	days = [(TURN, A.MOVE, ("s0", 1)), (EXCHANGE, A.ARMY, ("s0", 1)), (EXCHANGE, A.END, ("s1", 1)), (TURN, A.MOVE, ("s1", 2)), (EXCHANGE, A.ARMY, ("s1", 2)), (EXCHANGE, A.END, ("s0", 2)), (TURN, A.END, ("s0", 2))]
	assert not flags(days).any()
	assert not flags(GIVE + [(TURN, A.END, "s1")]).any()


def test_the_charge_lands_after_the_standardization_on_flagged_credited_rows_alone():
	advantage = torch.tensor([0.5, -1.0, 2.0, 0.0, -0.3])
	credited = torch.tensor([True, True, True, False, True])
	loops = torch.tensor([LOOP_NULL, LOOP_NONE, LOOP_ACROSS, LOOP_UNDONE, LOOP_NONE])
	charged, stats = charge_loops(advantage, credited, loops, 0.1)
	assert torch.allclose(charged, torch.tensor([0.4, -1.0, 1.9, 0.0, -0.3]))
	assert stats["loop_share"] == 0.5 and stats["loop_share_1"] == 0.25 and stats["loop_share_2"] == 0.0 and stats["loop_share_3"] == 0.25
	assert abs(stats["loop_adv_before"] - 1.25) < 1e-6 and stats["loop_charge"] == 0.1
	same, _ = charge_loops(advantage, credited, loops, 0.0)
	assert torch.equal(same, advantage)


def test_the_charge_ramps_by_version_alone():
	off = CampaignConfig()
	assert all(loop_charge_now(off, v) == 0.0 for v in (0, 3400, 10**6))
	on = CampaignConfig(loop_charge=0.1, loop_charge_from=3400)
	assert loop_charge_now(on, 3399) == 0.0 and abs(loop_charge_now(on, 3400) - 0.01) < 1e-12 and abs(loop_charge_now(on, 3404) - 0.05) < 1e-12
	assert loop_charge_now(on, 3409) == 0.1 and loop_charge_now(on, 3500) == 0.1
	faster = CampaignConfig(loop_charge=0.2, loop_charge_from=10, loop_charge_step=0.02)
	assert abs(loop_charge_now(faster, 14) - 0.1) < 1e-12 and loop_charge_now(faster, 30) == 0.2


def test_an_update_carries_the_flags_and_the_charge_and_at_zero_none(tmp_path, capsys):
	"""CampaignLearner.step end to end (test_learner's price test's way): with config.loop_charge the end's arrays carry each
	decision's loop type and the charge at the version, the update's metrics its readings, the log its line; at 0 none."""
	from campaign_model.empire import EmpireConfig
	from campaign_model.labels import LabelConfig
	from campaign_model.learner import COLUMNS
	from campaign_model.model import build_model
	from campaign_model.schema import random_inputs
	from campaign_model.tests.test_model import encoded
	from campaign_model.tests.test_train import decisions_for
	from campaign_model.train import make_optimizer
	import campaign_model.learner as learner_module

	seen = []
	real_update = learner_module.ppo_update

	def spy(*args, **kwargs):
		seen.append(kwargs.get("end"))
		return real_update(*args, **kwargs)

	for charge in (0.1, 0.0):
		torch.manual_seed(6)
		kwargs = {"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"}
		model = build_model(kwargs).eval()
		config = CampaignConfig(decisions_per_update=8, min_games_per_update=2, batch_size=4, passes=1, learner="thread", device="cpu", credit="schedule", share_target="change", model_architecture="v2", noise_every=0, adv_weight=1.0, loop_charge=charge, loop_charge_from=1)
		games = []
		for g in range(2):
			x = random_inputs(8, seed=30 + g)
			decisions = [dataclasses.replace(d, inputs=encoded(x, b), day=1 + b // 3) for b, d in enumerate(decisions_for(model, x, seed=g))]
			games.append((Game(decisions=decisions, ledgers={d: snapshot(d, 1000 + 300 * d, 1000) for d in range(1, 14)}, days=12), SimpleNamespace(seconds=1.0, map_name="Maps/alpha_random/x")))

		class Writer:
			def add_scalar(self, *args):
				pass

			def flush(self):
				pass

		out = tmp_path / str(charge)
		out.mkdir()
		learner = SimpleNamespace(
			config=config, labels=LabelConfig(), empire=EmpireConfig(), version=1, finished=list(games), lock=threading.Condition(), stats=collections.Counter(),
			model=model, optimizer=make_optimizer(model, config), device=torch.device("cpu"), worker=None, model_lock=threading.Lock(), lr_scale=1.0,
			server=SimpleNamespace(set_model=lambda *args: None), teacher=None, teacher_weight=0.0, games_used=0, out=out, kwargs=kwargs, value_ev=None, credit_evidence={}, credit_check=1.0, head_ev={"Vend": 0.2}, value_pool=None, head_pool={}, progress=out / "progress.csv", writer=Writer(),
			features_cpu=model.features, trust_ema=None, grad_clips=None, clip_samples=[], head_residual=None, head_var_slow=None, head_residual_updates=0, held=None, held_updates=0, gather_ramp_start=None, valve=None, value_readings=[], value_own=0,
		)
		for name in ("_take", "trust", "trust_today", "_save", "_log", "_note_value_ev", "_note_head_ev", "grade_trust", "end_credit", "_refresh_features", "_relabel_waiting", "loss_shares", "loss_weights", "_note_gradients", "gather_weight_now", "valve_now", "_note_valve", "_valve_reading", "_head_scale", "_calibrate_clips", "_judge_value"):
			setattr(learner, name, getattr(CampaignLearner, name).__get__(learner))
		learner.progress.write_text(",".join(COLUMNS) + "\n")
		learner_module.ppo_update = spy
		try:
			assert CampaignLearner.step(learner, epoch=1)
		finally:
			learner_module.ppo_update = real_update
		end = seen[-1]
		printed = capsys.readouterr().out
		if charge:
			counted = sum(len(game.decisions) for game, _ in games)
			assert len(end["loops"]) == counted and abs(end["loop_charge"] - 0.01) < 1e-12  # (version 1, from 1: one step of 0.01)
			assert "campaign loop charge: c 0.010" in printed
		else:
			assert "loops" not in end and "loop_charge" not in end and "campaign loop charge" not in printed
