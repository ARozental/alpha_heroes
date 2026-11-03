"""supply.memory_allows: a game starts only while the memory available, less the trainer's way back to its recent peak,
is over the floor (joint7 was stopped at 1.06 GB on 2026-11-01: games let in at the trainer's low point between
updates)."""

import multiprocessing

from battle_model import supply


def shared_at(monkeypatch, available, rss, peak):
	shared = supply.Shared(multiprocessing.get_context("spawn"), {"campaign": 4})
	shared.peak.value = peak
	monkeypatch.setattr(supply, "available_gb", lambda: available)
	monkeypatch.setattr(supply, "process_gb", lambda pid, field: rss)
	return shared


def test_the_trainers_way_back_to_its_recent_peak_is_kept_free(monkeypatch):
	# 12 GB available, the trainer at 9 of its recent 14 GB: 7 GB left for games, over the floor
	assert supply.memory_allows(shared_at(monkeypatch, available=12.0, rss=9.0, peak=14.0))
	# 9 GB available looks free, but the trainer takes 5 back at its next update: 4 GB, under the floor
	assert not supply.memory_allows(shared_at(monkeypatch, available=9.0, rss=9.0, peak=14.0))
	# at its peak nothing is owed
	assert supply.memory_allows(shared_at(monkeypatch, available=supply.MEMORY_FLOOR_GB, rss=14.0, peak=14.0))


def test_a_fresh_trainer_owes_nothing_yet(monkeypatch):
	# after a restart (no peak sampled yet), the floor alone: the games bring back what it held
	assert supply.memory_allows(shared_at(monkeypatch, available=15.0, rss=3.9, peak=0.0))


def test_unknown_memory_never_holds_games_back(monkeypatch):
	assert supply.memory_allows(shared_at(monkeypatch, available=None, rss=9.0, peak=14.0))


def test_one_game_of_a_kind_always_plays_over_the_last_game_level(monkeypatch):
	# the trainer at its peak holding what only games would free: 4 GB is under the floor, but with none playing one
	# starts (13:38-15:00 CST on 2026-11-01 no game could, and the trainer waited for games)
	shared = shared_at(monkeypatch, available=4.0, rss=13.0, peak=13.5)
	assert not supply.memory_allows(shared, playing=1)
	assert supply.memory_allows(shared, playing=0)
	assert not supply.memory_allows(shared_at(monkeypatch, available=3.0, rss=13.0, peak=13.5), playing=0)


def test_an_eval_waiting_for_memory_holds_training_games_back_but_never_the_last(tmp_path, monkeypatch):
	shared = shared_at(monkeypatch, available=12.0, rss=9.0, peak=9.0)
	games = supply.Games("campaign", None, tmp_path, [], [], shared, token=None)
	assert games.place() == ("here", None)  # nothing waits: it starts
	shared.eval_waiting.value = 1
	assert games.place() is None  # an eval game waits: the freed memory is its first
	shared.local["campaign"].value = 0
	assert games.place() == ("here", None)  # but a kind's last game always plays (the trainer waits for games)
	(tmp_path / "drain").touch()
	shared.local["campaign"].value, shared.eval_waiting.value = 0, 0
	assert games.place() is None  # draining: nothing starts


def test_the_permits_are_counted_again_with_no_game_event(monkeypatch):
	# the trap of 2026-11-01 15:58: the update took the waiting games while none played; no game event came again
	import threading
	import time

	s = supply.Supply.__new__(supply.Supply)
	s.stop, calls = threading.Event(), []
	s.allow = lambda: calls.append(1)
	monkeypatch.setattr(supply, "PERMITS_EVERY", 0.01)
	thread = threading.Thread(target=s._keep_allowing, daemon=True)
	thread.start()
	time.sleep(0.2)
	s.stop.set()
	thread.join(timeout=1)
	assert len(calls) >= 5 and not thread.is_alive()


def test_the_memory_census_counts_what_the_trainer_holds():
	import threading
	from types import SimpleNamespace

	import numpy as np
	import torch

	assert supply.held_bytes({"a": np.zeros(1000, np.float32), "b": [torch.zeros(10, dtype=torch.float64), b"xyz"]}) == 4000 + 80 + 3
	s = supply.Supply.__new__(supply.Supply)
	s.campaign = SimpleNamespace(lock=threading.Lock(), finished=[(SimpleNamespace(decisions=[SimpleNamespace(inputs=b"x" * 2**20)] * 3), None)], label_futures={})
	s.pool = SimpleNamespace(lock=threading.Lock(), buffer=[{"decisions": 7, "data": {"x": np.zeros(2**18, np.float32)}}])
	line = s.memory_census(5.0, by_type=True)
	assert "campaign waiting 1 games, 3 decisions, inputs 3 MB" in line and "battle buffer 1 battles, 7 decisions, 1 MB" in line and "objects " in line


def test_a_remote_slot_waits_only_for_the_last_game_level(tmp_path, monkeypatch):
	# the trainer's peak kept free puts the floor out of reach (6 GB available, 4 owed): a game here waits, the Mac's
	# (its worker here exists already) plays; under LAST_GAME_GB it waits too
	shared = shared_at(monkeypatch, available=6.0, rss=9.0, peak=13.0)
	shared.local["campaign"].value = 3
	games = supply.Games("campaign", None, tmp_path, [], [], shared, token=None, listener=object())
	monkeypatch.setattr(games, "_accept_slot", lambda: "conn")
	assert games.place() == ("remote", "conn")
	monkeypatch.setattr(supply, "available_gb", lambda: 3.0)
	assert games.place() is None


def test_a_loader_worker_keeps_large_allocations_apart_where_glibc_runs():
	"""(2026-11-03) keep_large_allocations_apart, each loader worker's: glibc's mmap threshold fixed and 2 arenas on
	Linux (where the workers held ~370 MB each of freed battle-record buffers), nothing elsewhere."""
	import sys

	from battle_model.supply import keep_large_allocations_apart

	assert keep_large_allocations_apart() is sys.platform.startswith("linux")
