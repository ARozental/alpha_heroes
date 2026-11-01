"""supply.memory_allows: a game starts only while the memory available, less the trainer's way back to its peak, is over
the floor (joint7 was stopped at 1.06 GB on 2026-11-01: games let in at the trainer's low point between updates)."""

import multiprocessing

from battle_model import supply


def shared_at(tmp_path, monkeypatch, available, rss, peak, recorded=None):
	if recorded is not None:
		(tmp_path / "memory-peak-gb").write_text(f"{recorded}\n")
	shared = supply.Shared(multiprocessing.get_context("spawn"), {"campaign": 4}, tmp_path / "memory-peak-gb")
	monkeypatch.setattr(supply, "available_gb", lambda: available)
	monkeypatch.setattr(supply, "process_gb", lambda pid, field: {"VmRSS": rss, "VmHWM": peak}[field])
	return shared


def test_the_trainers_way_back_to_its_peak_is_kept_free(tmp_path, monkeypatch):
	# 12 GB available, the trainer at 9 of its 14 GB peak: 7 GB left for games, over the floor
	assert supply.memory_allows(shared_at(tmp_path, monkeypatch, available=12.0, rss=9.0, peak=14.0))
	# 10 GB available looks free, but the trainer takes 5 back at its next update: 5 GB, under the floor
	assert not supply.memory_allows(shared_at(tmp_path, monkeypatch, available=10.0, rss=9.0, peak=14.0))
	# at its peak nothing is owed
	assert supply.memory_allows(shared_at(tmp_path, monkeypatch, available=supply.MEMORY_FLOOR_GB, rss=14.0, peak=14.0))


def test_a_peak_of_an_earlier_process_of_the_run_counts_and_a_new_one_is_recorded(tmp_path, monkeypatch):
	# a resumed trainer (its own peak 10 GB so far) owes what the run's earlier process reached
	assert not supply.memory_allows(shared_at(tmp_path, monkeypatch, available=10.0, rss=9.0, peak=10.0, recorded=14.0))
	shared = shared_at(tmp_path, monkeypatch, available=30.0, rss=15.0, peak=16.0, recorded=14.0)
	assert supply.memory_allows(shared)
	assert shared.peak.value == 16.0 and float((tmp_path / "memory-peak-gb").read_text()) == 16.0


def test_unknown_memory_never_holds_games_back(tmp_path, monkeypatch):
	assert supply.memory_allows(shared_at(tmp_path, monkeypatch, available=None, rss=9.0, peak=14.0))
