"""The critic's NK2 stream (campaign_model/nk2_stream.py; the user, 2026-11-19: "each game should only be used once"):
the slim rule, the history check, every group out of the stream exactly once, a starved buffer's short take, the
held-out fixed from the first games and the same after a restart."""
import dataclasses
import random

import numpy as np

from campaign_model import constants as c
from campaign_model.nk2_stream import NK2Stream, history_problem, slim
from campaign_model.schema import empty_inputs
from campaign_model.tests.test_critic_store import small_game
from campaign_model.tests.test_model import encoded


def test_slim_keeps_two_decisions_a_player_day_drawn_at_random_in_the_games_order_and_the_ledgers():
	game = small_game(days=6, per_day=5)
	ledgers = dict(game.ledgers)
	kept = slim(game, random.Random(3))
	per_day = {}
	for d in kept.decisions:
		per_day[(d.color, d.day)] = per_day.get((d.color, d.day), 0) + 1
	assert set(per_day.values()) == {2} and len(per_day) == 12
	assert [(d.day, d.color) for d in kept.decisions] == sorted(((d.day, d.color) for d in kept.decisions), key=lambda t: (t[0], ["RED", "BLUE"].index(t[1])))
	assert kept.ledgers == ledgers
	again = slim(small_game(days=6, per_day=5), random.Random(3))
	assert [d.today for d in again.decisions] == [d.today for d in kept.decisions]  # (seeded: the same draw)
	one = slim(small_game(days=3, per_day=1), random.Random(3))
	assert len(one.decisions) == 6  # (a day of one decision keeps it)


def game_with_history(n=30, empty=False, constant_index=False):
	"""A game of n decisions whose inputs carry the decision-history columns: varying (a real game's), every one -1
	(empty), or one constant."""
	x = empty_inputs(n)
	for b in range(n):
		x["g_last_changed"][b] = -1 if empty else b % 2
		x["g_last_type"][b] = -1 if empty else b % 5
		x["g_last_index"][b] = 7 if constant_index else b
		x["g_decisions_today"][b] = b % 9
	game = small_game(days=15, per_day=1)
	game.decisions = [dataclasses.replace(d, inputs=encoded(x, i % n)) for i, d in enumerate(game.decisions[:n])]
	return game


def test_the_history_check_refuses_empty_or_constant_history_and_short_games():
	assert history_problem(game_with_history()) is None
	assert "g_last_changed empty" in history_problem(game_with_history(empty=True))
	assert "g_last_index constant" in history_problem(game_with_history(constant_index=True))
	assert "only" in history_problem(game_with_history(n=10))


def groups_of(key, n, trained=3):
	return [{"game": key, "day": d, "trained": trained, "id": f"{key}/{d}"} for d in range(n)]


def test_every_group_comes_out_exactly_once_and_the_held_out_never():
	stream = NK2Stream(heldout_games=5, heldout_rows=30, floor_games=0, seed=1)
	games = {f"g{i}": groups_of(f"g{i}", 10 + i % 7) for i in range(60)}
	for key, groups in games.items():
		stream.put(key, groups)
	held = {g["id"] for g in stream.heldout}
	assert held and all(g["game"] in {"g0", "g1", "g2", "g3", "g4"} for g in stream.heldout) and sum(g["trained"] for g in stream.heldout) >= 30
	served = []
	while True:
		taken = stream.take(40, keep_floor=False)
		if not taken:
			break
		served += [g["id"] for g in taken]
	buffered = {g["id"] for key, groups in games.items() if key not in {"g0", "g1", "g2", "g3", "g4"} for g in groups}
	assert len(served) == len(set(served))  # (none twice)
	assert set(served) == buffered  # (every one of the buffer's, once)
	assert not set(served) & held and stream.buffer_games() == 0


def test_a_starved_buffer_gives_a_short_take_never_a_repeat():
	stream = NK2Stream(heldout_games=0, floor_games=10, seed=2)  # (no held-out: every game to the buffer)
	for i in range(12):
		stream.put(f"g{i}", groups_of(f"g{i}", 10))
	first = stream.take(10**6)
	assert 0 < len(first) <= 20  # (what leaves 10 games' groups: 2 games' worth)
	reading = stream.reading()
	assert reading["nk2_rows_taken"] == 3 * len(first) and reading["nk2_rows_wanted"] == 10**6 and reading["nk2_buffer_before"] == 12
	assert stream.take(10**6) == []  # (under its floor: nothing, not the same groups again)
	stream.put("g12", groups_of("g12", 10))
	again = stream.take(10**6)
	assert {g["id"] for g in again}.isdisjoint({g["id"] for g in first})


def test_the_held_out_is_fixed_from_the_first_games_saved_and_the_same_after_a_restart(tmp_path):
	path = tmp_path / "nk2_heldout.pkl"
	stream = NK2Stream(heldout_games=3, heldout_rows=12, floor_games=0, heldout_path=path, seed=3)
	for i in range(3):
		stream.put(f"g{i}", groups_of(f"g{i}", 8))
	assert path.exists() and stream.buffer_games() == 0
	stream.put("g3", groups_of("g3", 8))
	assert stream.buffer_games() == 1
	again = NK2Stream(heldout_games=3, heldout_rows=12, floor_games=0, heldout_path=path, seed=99)
	assert [g["id"] for g in again.heldout] == [g["id"] for g in stream.heldout]
	again.put("g4", groups_of("g4", 8))  # (a restart's first game goes to the buffer: the held-out is never drawn again)
	assert again.buffer_games() == 1 and [g["id"] for g in again.heldout] == [g["id"] for g in stream.heldout]
	stream.dropped("g_last_index constant (7)")
	assert stream.reading()["nk2_dropped"] == 1


def nk2_runner(tmp_path, monkeypatch, game, status="stopped at day 56"):
	"""A supply.Runner whose GameRun is a stand-in that ends at once with `game` (and records how it was made)."""
	from types import SimpleNamespace

	from battle_model import supply
	from battle_model.train import TrainConfig
	from campaign_model import play as campaign_play

	made = {}

	class FakeRun:
		def __init__(self, *args, **kwargs):
			made.update(kwargs, args=args)

		def poll(self, stop=None):
			return True

		def result(self):
			return game, SimpleNamespace(status=status, run_dir=str(tmp_path / "run"))

		def terminate(self):
			pass

	monkeypatch.setattr(campaign_play, "GameRun", FakeRun)
	monkeypatch.setattr(campaign_play, "make_map", lambda seed, players, size, levels, water: ("Maps/alpha_random/nk2-test", None))
	return supply.Runner(TrainConfig(), tmp_path, None, None, None, "w0"), made


def history_game(per_day=3):
	"""Two players, 15 days, per_day decisions each a day, history columns varying: a game that passes the check."""
	x = empty_inputs(8)
	for b in range(8):
		x["g_last_changed"][b], x["g_last_type"][b], x["g_last_index"][b], x["g_decisions_today"][b] = b % 2, b % 5, b, b % 9
	return small_game(days=15, per_day=per_day, inputs=[encoded(x, b) for b in range(8)])


def test_an_nk2_game_is_played_as_teacher_with_our_settings_slimmed_checked_and_yielded(tmp_path, monkeypatch):
	runner, made = nk2_runner(tmp_path, monkeypatch, history_game())
	spec = runner.sample_nk2(random.Random(1), 0, 1)
	assert spec["kind"] == "nk2" and spec["days"] == 56
	items = list(runner.nk2(spec))
	assert [i["kind"] for i in items] == ["nk2"] and items[0]["key"] == spec["key"]
	import socket

	assert items[0]["host"] == socket.gethostname()  # (where it played: the stream's games by machine)
	per_day = {}
	for d in items[0]["game"].decisions:
		per_day[(d.color, d.day)] = per_day.get((d.color, d.day), 0) + 1
	assert set(per_day.values()) == {2}  # (slimmed where it played)
	assert made["teacher"] is True and made["battle_ai"] == "BattleAI"
	settings = dict(made["settings"])
	assert {"ALPHA_MAP_FURTHER_WINDOWS", "ALPHA_MAP_UNSEEN_EXITS", "ALPHA_MAP_NK2_INPUTS"} <= set(settings)  # (its inputs encoded as ours are)


def test_an_nk2_game_failing_the_history_check_or_unfinished_is_a_counted_failure(tmp_path, monkeypatch):
	empty = small_game(days=15, per_day=3, inputs=[encoded(empty_inputs(1), 0)])  # (every history column at its default)
	runner, _ = nk2_runner(tmp_path, monkeypatch, empty)
	items = list(runner.nk2(runner.sample_nk2(random.Random(2), 0, 1)))
	assert len(items) == 1 and items[0]["kind"] == "failed" and items[0]["what"] == "nk2" and items[0]["check"]
	runner, _ = nk2_runner(tmp_path, monkeypatch, history_game(), status="time limit (3600 s)")
	items = list(runner.nk2(runner.sample_nk2(random.Random(3), 0, 1)))
	assert len(items) == 1 and items[0]["kind"] == "failed" and not items[0].get("check")


def test_the_supply_routes_nk2_games_and_their_failures_to_the_learner():
	from collections import Counter
	from types import SimpleNamespace

	from battle_model import supply

	got = []
	learner = SimpleNamespace(nk2_arrived=lambda item: got.append(("in", item["key"])), nk2_failed=lambda item: got.append(("failed", item["message"])))
	fake = SimpleNamespace(stats=Counter(), campaign=learner, pool=None)
	supply.Supply._route(fake, {"kind": "nk2", "key": "n1", "game": None})
	supply.Supply._route(fake, {"kind": "failed", "what": "nk2", "message": "game n2: the history check: x", "check": True})
	assert got == [("in", "n1"), ("failed", "game n2: the history check: x")] and fake.stats["nk2_games"] == 1


class Recorder:
	def __init__(self):
		self.tags = {}

	def add_scalar(self, tag, value, step=None):
		self.tags.setdefault(tag, []).append((step, value))


def test_a_critic_only_update_trains_the_critic_alone_on_stream_rows_taken_once(tmp_path):
	"""campaign.stage "critic" (e5 for the user, 2026-11-19, stage A): ready when a whole take is above the floor; a step
	trains the critic on the stream's rows (no own rows), reads the fixed held-out, moves no policy weight, saves, writes
	critic_progress.csv and the critic/ and nk2_stream/ charts; the takes never share a group; a starved stream: no step."""
	import copy
	import threading
	from types import SimpleNamespace

	import torch

	from campaign_model.config import CampaignConfig
	from campaign_model.critic import Critic, critic_kwargs, critic_optimizer
	from campaign_model.empire import EmpireConfig
	from campaign_model.learner import CampaignLearner, body_from_policy
	from campaign_model.model import build_model
	from campaign_model.tests.test_critic import SMALL
	from campaign_model.tests.test_model_v2 import rich_inputs
	from campaign_model.train import model_kwargs

	cfg = CampaignConfig(**SMALL, critic_net=True, critic_batch=4, nk2_stream=True, stage="critic", nk2_rows=8, nk2_floor_games=1, nk2_heldout_games=1, nk2_heldout_rows=4, head_scale=True)
	torch.manual_seed(5)
	policy = build_model(model_kwargs(cfg))
	critic = Critic(**critic_kwargs(cfg, model_kwargs(cfg)))
	body_from_policy(critic, policy)
	stream = NK2Stream(1, 4, 1, heldout_path=tmp_path / "nk2_heldout.pkl", seed=1)
	learner = SimpleNamespace(config=cfg, critic=critic, critic_optimizer=critic_optimizer(critic, cfg), model=policy, version=0, device=torch.device("cpu"), model_lock=threading.Lock(),
		out=tmp_path, writer=Recorder(), nk2_stream=stream, nk2_heldout_batch=None, empire=EmpireConfig(), labels=None)
	saved = []
	learner._save = lambda path: saved.append(path.name)
	assert not CampaignLearner.ready(learner)  # (no held-out yet)
	x = rich_inputs(8, seed=5)
	for i in range(4):
		CampaignLearner.nk2_arrived(learner, {"key": f"g{i}", "game": small_game(days=4, inputs=[encoded(x, b) for b in range(8)]), "host": "pc" if i % 2 else "mac"})
	assert dict(learner.nk2_hosts) == {"pc": 2, "mac": 2}
	taken, take = [], stream.take
	stream.take = lambda rows, keep_floor=True: (lambda got: (taken.append({id(g) for g in got}), got)[1])(take(rows, keep_floor))
	p0, c0 = copy.deepcopy(policy.state_dict()), copy.deepcopy(critic.state_dict())
	steps = 0
	while CampaignLearner.ready(learner):
		assert CampaignLearner.step(learner, 0)
		steps += 1
	assert steps == 2 and learner.version == 2 and saved == ["checkpoint_latest.pt"] * 2
	assert not taken[0] & taken[1]  # (no group in two takes)
	assert not CampaignLearner.step(learner, 0)  # (starved: no step, no repeat)
	assert all(torch.equal(v, p0[k]) for k, v in policy.state_dict().items())
	assert any(not torch.equal(v, c0[k]) for k, v in critic.state_dict().items())
	assert learner.head_var is not None
	tags = learner.writer.tags
	assert {"critic/loss_pool", "critic/loss_visible_pool", "critic/loss_pool_heldout", "critic/loss_visible_pool_heldout"} <= set(tags)
	assert {f"critic/fixed_{p}_{h}{s}" for p in ("six", "end") for h in ("", "visible_") for s in ("pool", "pool_heldout")} <= set(tags)  # (the fixed-weight ruler)
	assert not any(t.startswith("critic/") and "own" in t for t in tags)  # (no own rows: no own readings)
	assert "nk2_stream/rows_taken" in tags
	rows = (tmp_path / "critic_progress.csv").read_text().splitlines()
	header = rows[0].split(",")
	assert len(rows) == 3 and header[:3] == ["version", "time", "seconds"]
	last = dict(zip(header, rows[-1].split(",")))
	assert last["version"] == "2" and float(last["critic_rows_pool"]) >= 8 and np.isfinite(float(last["critic_loss_pool_heldout"]))


def test_the_stream_never_parks_while_a_critic_update_waits_and_a_partly_taken_game_counts_as_its_part():
	"""(the smoke of 2026-11-19: at the cap in whole games the rows above the floor were less than a take; the pull parked,
	the learner waited, nothing moved) the buffer in games is its groups over the mean groups a game; in stage "critic" the
	pull is never parked while the learner waits for a whole take."""
	from types import SimpleNamespace

	from campaign_model.config import CampaignConfig
	from campaign_model.learner import CampaignLearner

	cfg = CampaignConfig(stage="critic", critic_net=True, nk2_stream=True, nk2_rows=40, nk2_floor_games=2, nk2_buffer_cap_games=3, nk2_heldout_games=0)
	stream = NK2Stream(heldout_games=0, floor_games=2, seed=3)
	learner = SimpleNamespace(config=cfg, nk2_stream=stream)
	for i in range(3):
		stream.put(f"g{i}", groups_of(f"g{i}", 10))
	assert stream.buffer_size() == 3.0
	assert not CampaignLearner.critic_ready(learner) and CampaignLearner.nk2_room(learner)  # (30 rows above the floor, a take 40: not parked)
	stream.put("g3", groups_of("g3", 10))
	assert CampaignLearner.critic_ready(learner) and not CampaignLearner.nk2_room(learner)  # (a take ready, over the cap: parked)
	taken = stream.take(30)
	assert len(taken) == 10 and stream.buffer_size() == 3.0 and stream.buffer_games() == 4  # (partly taken games: their part)
	joint = SimpleNamespace(config=CampaignConfig(critic_net=True, nk2_stream=True, nk2_buffer_cap_games=3, nk2_heldout_games=0), nk2_stream=stream)
	assert not CampaignLearner.nk2_room(joint)  # (the joint stage takes what is there: parked at the cap)
