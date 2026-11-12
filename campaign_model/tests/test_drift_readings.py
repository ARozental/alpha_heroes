"""e5's drift monitors (2026-11-12): learner.drift_readings per game-player, and their columns in the progress rows."""
from types import SimpleNamespace

from campaign_model import constants as c
from campaign_model import learner as L

NB = c.N_BUILDINGS


def turn(color, day, action, index=0):
	return SimpleNamespace(color=color, day=day, kind=c.DecisionKind.TURN, action_type=action, index=index)


def test_drift_readings_count_per_player():
	decisions = [
		turn("RED", 3, c.ActionType.BUILD, 0 * NB + 30), turn("RED", 20, c.ActionType.BUILD, 1 * NB + 31), turn("RED", 20, c.ActionType.BUILD, 1 * NB + 11),
		turn("RED", 4, c.ActionType.RECRUIT), turn("RED", 5, c.ActionType.RECRUIT), turn("BLUE", 9, c.ActionType.RECRUIT), turn("BLUE", 9, c.ActionType.END),
	]
	battles = {"RED": [{"side": 1, "heroes": [{}, {}]}, {"side": 0, "heroes": [{}, {}]}, {"side": 1, "heroes": [None, {}]}]}
	out = L.drift_readings([(SimpleNamespace(decisions=decisions, battles=battles), None)])
	# RED: dwellings in 2 towns, 1 by d14, 2 by d28 -> 0.5, 1.0; 2 recruits; 1 attack. BLUE: no build (1 town), 1 recruit, 0
	assert out == {"dwellings_d14": 0.25, "dwellings_d28": 0.5, "recruits": 1.5, "attacked": 0.5}
	assert all(v != v for v in L.drift_readings([]).values())  # (nan with no game)


def test_their_columns_are_in_the_progress_rows():
	for what in ("dwellings_d14", "dwellings_d28", "recruits", "attacked"):
		for name in ("shown", "hidden"):
			assert f"{what}_{name}" in L.COLUMNS
