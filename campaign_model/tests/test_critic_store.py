"""The critic's game-day groups (campaign_model/critic_store.py; e5 for the user, 2026-11-18): rivals inside the group at
the rival's last earlier-day row, rival-only rows never in a step; the learner's NK2 stream path (2026-11-19: the games made
on the fly and each used once; the store, its intake and the own replay ring gone)."""
import copy

import pytest
import dataclasses

import numpy as np
import torch

from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.critic_store import assemble, game_groups
from campaign_model.learner import privileged_rivals
from campaign_model.records import Game
from campaign_model.tests.test_labels import decision, snapshot

CHANGE = CampaignConfig(share_target="change")


def small_game(days=6, per_day=2, inputs=None):
	"""RED and BLUE, `per_day` turn decisions each a day; their only asset gold."""
	decisions = []
	for day in range(1, days + 1):
		for color in ("RED", "BLUE"):
			for today in range(per_day):
				d = decision(color, day, today)
				if inputs is not None:
					d = dataclasses.replace(d, inputs=inputs[len(decisions) % len(inputs)])
				decisions.append(d)
	return Game(decisions=decisions, ledgers={d: snapshot(d, 1000 + 150 * d, 1000 + 40 * d) for d in range(1, days + 4)}, days=days)


def test_a_groups_rivals_are_inside_it_at_the_rivals_last_earlier_day_row():
	game = small_game()
	full = privileged_rivals(game, np.ones(len(game.decisions), dtype=bool), 0)
	groups = game_groups(game, "g", CHANGE, rng=np.random.default_rng(1))
	assert [g["day"] for g in groups] == list(range(1, 7))
	index = {id(d): i for i, d in enumerate(game.decisions)}
	for g in groups:
		assert g["trained"] == 2 and {d.color for d in g["rows"][: g["trained"]]} == {"RED", "BLUE"}
		for k in range(g["trained"]):
			i = index[id(g["rows"][k])]
			for slot, r in enumerate(g["rivals"][k]):
				want = full["rivals"][i, slot]
				if want < 0:
					assert r == -1
				else:
					assert r >= g["trained"] and g["rows"][r] is game.decisions[want] and g["rows"][r].day < g["day"]


def test_rival_only_rows_never_reach_a_step():
	from campaign_model.critic import Critic, critic_kwargs, critic_optimizer
	from campaign_model.model import build_model
	from campaign_model.tests.test_critic import SMALL, _update_setup
	from campaign_model.tests.test_model import encoded
	from campaign_model.tests.test_model_v2 import rich_inputs
	from campaign_model.train import model_kwargs, ppo_update

	cfg, policy, critic, decisions, targets, decode, none = _update_setup(seed=31)
	x = rich_inputs(8, seed=31)
	game = small_game(inputs=[encoded(x, b) for b in range(8)])
	pool = assemble(game_groups(game, "g", CHANGE, rng=np.random.default_rng(2)))
	assert len(pool["train"]) < len(pool["decisions"])
	rival_only = np.setdiff1d(np.arange(len(pool["decisions"])), pool["train"])
	filled = copy.deepcopy({k: v for k, v in pool.items() if k != "decisions"})
	filled["decisions"] = pool["decisions"]
	filled["targets"][rival_only] = 0.9  # (were a rival-only row stepped on, this would move the critic)
	filled["end"][rival_only] = 0.9
	results = []
	for sample in (pool, filled):
		torch.manual_seed(9)
		np.random.seed(9)
		pol, crt = copy.deepcopy(policy), copy.deepcopy(critic)
		crit = {"module": crt, "optimizer": critic_optimizer(crt, cfg), "rows": np.array([0, 3, 5], dtype=np.int64), **none, "pool": sample}
		metrics, _ = ppo_update(pol, torch.optim.Adam(pol.parameters(), lr=1e-3), decisions, targets, None, cfg, decode=decode, advantages=np.linspace(-1, 1, len(decisions)).astype(np.float32), critic=crit)
		results.append((crt, metrics))
	assert results[0][1]["critic_rows_pool"] == len(pool["train"])
	for (name, a), (_, b) in zip(results[0][0].named_parameters(), results[1][0].named_parameters()):
		assert torch.equal(a, b), name


def test_own_replay_groups_keep_two_choices_and_a_use_draws_one_with_its_rivals():
	from campaign_model.critic_store import choose

	game = small_game(days=5, per_day=3)
	groups = game_groups(game, "own", CHANGE, rng=np.random.default_rng(3), keep=2)
	full = privileged_rivals(game, np.ones(len(game.decisions), dtype=bool), 0)
	index = {id(d): i for i, d in enumerate(game.decisions)}
	seen = set()
	for g in groups:
		assert g["trained"] == 4 and [len(c) for c in g["choices"]] == [2, 2]
		for draw in range(6):
			one = choose(g, np.random.default_rng(draw))
			assert one["trained"] == 2 and {d.color for d in one["rows"][:2]} == {"RED", "BLUE"}
			for k in range(2):
				i = index[id(one["rows"][k])]
				seen.add(i)
				for slot, r in enumerate(one["rivals"][k]):
					want = full["rivals"][i, slot]
					assert (r == -1) if want < 0 else (one["rows"][r] is game.decisions[want])
	assert len(seen) > 10  # (both choices of a player-day met over the draws)


def test_v_end_labels_follow_end_labels_real_at_the_cap_none_at_a_time_limit():
	"""(e5's correction, 2026-11-18) a game that reached its day cap: its rows carry their share at the cap; a game
	stopped by the time limit before its cap: no V_end label."""
	capped = small_game(days=4)
	capped.days = 4
	stopped = small_game(days=4)
	stopped.days = 30
	for game, labelled in ((capped, True), (stopped, False)):
		for g in game_groups(game, "g", CHANGE, rng=np.random.default_rng(5), keep=2):
			assert bool(np.all(np.isfinite(g["end"]))) == labelled


def test_the_learner_labels_nk2_games_into_its_stream_and_its_critic_batch_takes_them_once(tmp_path):
	"""(the user, 2026-11-19) nk2_arrived labels a game at the learner's empire into its NK2Stream; the first games are the
	held-out, assembled once and read every update; _critic_batch takes its pool rows from the stream, each group once, and
	a starved stream gives no pool rather than a repeat; a game refused by the history check is counted; no game starts
	past the buffer's cap."""
	from types import SimpleNamespace

	from campaign_model.empire import EmpireConfig
	from campaign_model.learner import CampaignLearner
	from campaign_model.nk2_stream import NK2Stream

	config = CampaignConfig(share_target="change", nk2_stream=True, nk2_rows=6, nk2_floor_games=1, nk2_heldout_games=2, nk2_heldout_rows=4, nk2_buffer_cap_games=4)
	stream = NK2Stream(config.nk2_heldout_games, config.nk2_heldout_rows, config.nk2_floor_games, heldout_path=tmp_path / "nk2_heldout.pkl", seed=5)
	learner = SimpleNamespace(config=config, critic=object(), critic_optimizer=None, empire=EmpireConfig(), labels=None, nk2_stream=stream, nk2_heldout_batch=None)
	for i in range(2):  # (the held-out's two games)
		CampaignLearner.nk2_arrived(learner, {"key": f"h{i}", "game": small_game(days=3)})
	assert stream.heldout is not None and (tmp_path / "nk2_heldout.pkl").exists() and stream.buffer_games() == 0
	assert CampaignLearner.nk2_room(learner)
	for i in range(4):
		CampaignLearner.nk2_arrived(learner, {"key": f"g{i}", "game": small_game(days=3)})
	assert stream.buffer_games() == 4 and not CampaignLearner.nk2_room(learner)  # (at its cap: no game starts)
	parts = [{"rivals": np.full((10, 3), -1), "present": np.zeros((10, 3), np.float32), "share_logit": np.zeros(10, np.float32)}]
	seen = []
	for _ in range(4):
		batch = CampaignLearner._critic_batch(learner, parts, list(range(10)))
		assert batch["pool_heldout"] is not None and len(batch["pool_heldout"]["train"]) >= 4
		if batch["pool"] is None:
			break
		seen += [(d.color, d.day, id(d)) for d in (batch["pool"]["decisions"][i] for i in batch["pool"]["train"])]
	assert seen and len(seen) == len(set(seen))  # (no row twice)
	assert stream.buffer_games() >= 1  # (the floor left in place: the last take was short, then none)
	assert CampaignLearner._critic_batch(learner, parts, list(range(10)))["pool"] is None
	CampaignLearner.nk2_failed(learner, {"what": "nk2", "message": "game n1: the history check: g_last_index constant (7)", "check": True})
	assert stream.reading()["nk2_dropped"] == 1
