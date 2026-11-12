"""The small deploy after deploy 68 (e5, 2026-11-12): the credit segments' F stratified by the kind of game; the value-loss
guard's memory the line the run is on (rows after a rollback's version drop); nk_predict's logged loss at its share."""
import csv
from types import SimpleNamespace

import numpy as np

from campaign_model import constants as c
from campaign_model.labels import CREDIT_MIN_PLAYERS, credit_evidence
from campaign_model.learner import CampaignLearner, fresh_readings
from campaign_model.train import pooled_evidence, segment_shrinkage


def evidence(gaps, prefix=""):
	return {f"{prefix}T{i}": [float(len(g)), float(np.sum(g)), float(np.sum(np.square(g)))] for i, g in enumerate(gaps)}


def test_two_kinds_shifted_between_and_not_within_read_near_one_and_the_pooled_above():
	stratified, pooled = [], []
	for seed in range(300):
		rng = np.random.default_rng(seed)
		ev = {}
		for prefix, shift, sizes in (("hidden/", 0.0, [40, 60, 200, 300, 500]), ("shown/", 0.03, [500, 100, 300, 600, 60])):  # (the kinds' mixes of types differ: a constant shift between kinds shows as spread pooled)
			ev.update(evidence([rng.normal(shift, 0.15, n) for n in sizes], prefix))
		stratified.append(segment_shrinkage(ev, 8)[1])
		pooled.append(segment_shrinkage(pooled_evidence(ev), 8)[1])
	assert abs(np.mean(stratified) - 1.0) < 0.15, np.mean(stratified)
	assert np.mean(pooled) > 2.0, np.mean(pooled)


def test_one_kind_is_the_statistic_before():
	rng = np.random.default_rng(3)
	ev = evidence([rng.normal(0.01 * i, 0.15, 300) for i in range(6)])
	k, f, types = segment_shrinkage(ev, 8)
	k2, f2, types2 = segment_shrinkage({f"hidden/{key}": v for key, v in ev.items()}, 8)
	assert (k, f, types) == (k2, f2, types2)
	assert segment_shrinkage(pooled_evidence(ev), 8)[1] == f


def test_min_players_apply_within_a_kind():
	rng = np.random.default_rng(4)
	ev = evidence([rng.normal(0, 0.15, 300) for _ in range(3)], "hidden/")
	ev.update(evidence([rng.normal(0, 0.15, 5)], "shown/"))  # (one type, 5 game-players: below the minimum and alone in its kind)
	assert segment_shrinkage(ev, 8)[2] == 3


def test_the_learner_keeps_each_kinds_evidence_and_fades_it_once():
	learner = CampaignLearner.__new__(CampaignLearner)
	learner.segment_evidence, learner.segment_k = [{"MOVE": [10.0, 1.0, 1.0]}, {}, {}], (0.0, 0.0)
	rng = np.random.default_rng(1)
	n = 400
	decisions = [SimpleNamespace(kind=c.DecisionKind.TURN, action_type=int(t)) for t in rng.integers(0, 6, n)]
	players = np.repeat(np.arange(20), n // 20)
	hidden = players < 10
	out = learner._note_segments(decisions, rng.normal(0, 0.1, (n, 3)).astype(np.float32), players, hidden)
	keys = set(learner.segment_evidence[0])
	assert any(k.startswith("hidden/") for k in keys) and any(k.startswith("shown/") for k in keys)
	assert abs(learner.segment_evidence[0]["MOVE"][0] - 9.0) < 1e-9  # (the old pooled evidence faded once, by 0.9)
	assert {"segment_f2_pooled", "segment_f5_pooled", "segment_f7_pooled"} <= set(out) and np.isfinite(out["segment_f2"])
	one = CampaignLearner.__new__(CampaignLearner)
	one.segment_evidence, one.segment_k = [{}, {}, {}], (0.0, 0.0)
	one._note_segments(decisions, rng.normal(0, 0.1, (n, 3)).astype(np.float32), players, np.ones(n, dtype=bool))  # (one kind only: still its kind)
	assert one.segment_evidence[0] and all(k.startswith("hidden/") for k in one.segment_evidence[0])
	unknown = CampaignLearner.__new__(CampaignLearner)
	unknown.segment_evidence, unknown.segment_k = [{}, {}, {}], (0.0, 0.0)
	unknown._note_segments(decisions, rng.normal(0, 0.1, (n, 3)).astype(np.float32), players, None)  # (the kind unknown: unprefixed)
	assert unknown.segment_evidence[0] and all("/" not in k for k in unknown.segment_evidence[0])


def test_the_guards_memory_is_the_line_after_a_rollback(tmp_path):
	path = tmp_path / "progress.csv"
	with path.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["version", "value_loss_fresh"])
		for v in range(3250, 3315):  # (the abandoned line, low)
			writer.writerow([v, 0.002])
		for v in range(3255, 3262):  # (the rollback's line, higher)
			writer.writerow([v, 0.007])
	assert fresh_readings(path) == [0.007] * 7
	assert fresh_readings(path, since=3258) == [0.007] * 4
	assert fresh_readings(path, since=3315) == []
