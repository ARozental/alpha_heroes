"""The value-loss guard (learner.value_jump, CampaignLearner._judge_value and step's hold; the user, 2026-11-10: on a
jump of the campaign's value loss the learning stops at once)."""
import collections
import csv
import threading
from types import SimpleNamespace

import numpy as np

from campaign_model.config import CampaignConfig
from campaign_model.learner import VALUE_JUMP_BASELINE, VALUE_JUMP_MEMORY, VALUE_JUMP_WINDOW, CampaignLearner, fresh_readings, value_jump
from campaign_model.tests.test_learner import game_of


def calm(n, seed=0, level=0.0005):
	"""Readings as joint7's in a calm stretch: around the level, every ninth one game's outcome makes 2-5 times it."""
	rng = np.random.default_rng(seed)
	values = level * np.exp(rng.normal(0, 0.15, n))
	values[::9] *= rng.uniform(2, 5, len(values[::9]))
	return [float(v) for v in values]


def test_single_high_readings_are_not_a_jump():
	readings = calm(400)
	ratios = [value_jump(readings[:n])[0] for n in range(VALUE_JUMP_WINDOW + VALUE_JUMP_BASELINE, len(readings) + 1)]
	assert max(ratios) < 2.0 and min(ratios) > 0.5
	# (three high readings in a row, as after joint7's restarts: v2659-2661 0.0012, 0.0009, 0.0018 on a level of 0.0005)
	assert value_jump(calm(200, level=0.0005) + [0.0012, 0.0009, 0.0018, 0.0006, 0.00097, 0.00046, 0.00046])[0] < 2.0


def test_a_lasting_rise_is_a_jump_within_a_few_updates():
	readings = calm(200)
	risen = [0.0020, 0.0008, 0.0011, 0.0017, 0.0033, 0.0038, 0.0016, 0.0010, 0.0011, 0.0019]  # (joint7's v2744-2753)
	fired = [n for n in range(1, len(risen) + 1) if value_jump(readings + risen[:n])[0] >= 3.0]
	assert fired and fired[0] <= 7  # (four of the last seven above three times the level)
	ratio, now, level = value_jump(readings + risen)
	assert now == float(np.median(risen[-VALUE_JUMP_WINDOW:])) and 0.0004 < level < 0.0007 and ratio == now / level


def test_too_few_readings_or_no_level_is_no_judgment():
	assert value_jump(calm(VALUE_JUMP_WINDOW + VALUE_JUMP_BASELINE - 1))[0] != value_jump(calm(VALUE_JUMP_WINDOW + VALUE_JUMP_BASELINE - 1))[0]  # nan
	assert value_jump([0.0] * 100)[0] != value_jump([0.0] * 100)[0]
	assert value_jump(calm(100) + [float("nan")] * 3)[0] == value_jump(calm(100))[0]  # (a reading that is not a number is no reading)


def test_the_level_is_the_calm_stretchs_after_a_rollback():
	"""The abandoned updates' high readings stay in progress.csv: they are not the level the next ones are held to."""
	readings = calm(150) + [0.004] * 90
	ratio, now, level = value_jump(readings + calm(VALUE_JUMP_WINDOW, seed=3))
	assert level < 0.0007 and ratio < 2.0
	assert value_jump(readings + [0.004] * VALUE_JUMP_WINDOW)[0] > 5  # (and a model still at the risen loss is far above it)
	assert VALUE_JUMP_MEMORY >= 150 + 90  # (the calm stretch still in the memory after as long a rise as joint7's)


def learner_of(tmp_path, readings, factor=3.0):
	scalars = []
	writer = SimpleNamespace(add_scalar=lambda name, value, step: scalars.append((name, value, step)))
	return SimpleNamespace(config=CampaignConfig(value_jump_stop=factor, decisions_per_update=100, min_games_per_update=2), held=None, held_updates=0, gather_ramp_start=None, valve=None, value_readings=list(readings), value_own=0, version=7, out=tmp_path, writer=writer, scalars=scalars, lock=threading.Condition(), finished=[], stats=collections.Counter(), label_futures={})


def test_the_guard_holds_the_updates_and_says_so(tmp_path, capsys):
	learner = learner_of(tmp_path, calm(200))
	for reading in (0.0006, float("nan"), 0.0005):
		CampaignLearner._judge_value(learner, reading)
	assert learner.held is None and learner.value_own == 2 and len(learner.value_readings) == 202
	for reading in [0.004] * VALUE_JUMP_WINDOW:
		CampaignLearner._judge_value(learner, reading)
	assert learner.held is not None and "times its calm level" in learner.held
	assert (tmp_path / "held").read_text().startswith("version 7: ")
	assert "VALUE LOSS JUMP at version 7" in capsys.readouterr().out
	assert [name for name, _, _ in learner.scalars] == ["campaign/value_jump"] * learner.value_own
	# held: step takes the update's games and drops them, no update
	learner._take = lambda: CampaignLearner._take(learner)
	learner.finished = [(game_of(60), "a"), (game_of(60), "b"), (game_of(60), "c")]
	learner.label_futures = {id(game): None for game, _ in learner.finished}
	assert CampaignLearner.step(learner, 0) is False
	assert len(learner.finished) == 1 and learner.stats["dropped_games"] == 2 and learner.held_updates == 1 and len(learner.label_futures) == 1
	assert "HELD at version 7" in capsys.readouterr().out
	assert CampaignLearner.step(learner, 0) is False and learner.held_updates == 1  # (too few games: nothing taken)


def test_a_restarts_first_updates_are_not_judged_by_the_readings_before_it(tmp_path):
	"""After a rollback the file ends with the abandoned updates' readings: the guard waits for the window of this
	process's own, and a model still at the risen loss is then held again."""
	before = calm(150) + [0.004] * 90
	learner = learner_of(tmp_path, before)
	for reading in calm(VALUE_JUMP_WINDOW + 3, seed=5):
		CampaignLearner._judge_value(learner, reading)
	assert learner.held is None
	learner = learner_of(tmp_path, before)
	for n, reading in enumerate([0.004] * VALUE_JUMP_WINDOW):
		assert learner.held is None
		CampaignLearner._judge_value(learner, reading)
	assert learner.held is not None


def test_the_guard_is_off_at_zero_and_never_ends_the_run(tmp_path, capsys):
	learner = learner_of(tmp_path, calm(200), factor=0.0)
	for reading in [0.004] * 20:
		CampaignLearner._judge_value(learner, reading)
	assert learner.held is None and learner.value_own == 0
	learner = learner_of(tmp_path, calm(200))
	learner.writer = None  # (an error inside: a FAILURE line, the update goes on)
	CampaignLearner._judge_value(learner, 0.0005)
	assert "the value-loss guard failed" in capsys.readouterr().out and learner.held is None
	assert CampaignConfig().value_jump_stop == 3.0


def test_the_readings_come_back_from_progress_csv(tmp_path, capsys):
	path = tmp_path / "progress.csv"
	with path.open("w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(["version", "value_loss_fresh"])
		for n, value in enumerate(["0.0005", "", "nan", "0.0007", "inf"] + ["0.0006"] * 400):
			writer.writerow([n, value])
	readings = fresh_readings(path)
	assert len(readings) == VALUE_JUMP_MEMORY + VALUE_JUMP_WINDOW and set(readings) == {0.0006}
	assert fresh_readings(path, keep=1000)[:2] == [0.0005, 0.0007]
	assert fresh_readings(tmp_path / "none.csv") == [] and "could not read" in capsys.readouterr().out
