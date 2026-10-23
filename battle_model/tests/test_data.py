import pytest

from battle_model import vocab
from battle_model.data import RESULT_ESCAPE, RESULT_NORMAL, RESULT_SURRENDER, army_value, summarize
from battle_model.reward import Outcome

pytestmark = pytest.mark.skipif(not vocab.VOCAB_PATH.exists(), reason="run scripts/export_vocab.sh to generate the vocabulary")

OURS = [[0, 10], [12, 1]]  # 10 pikemen, 1 angel
THEIRS = [[42, 20]]  # 20 imps


def events(our_side, end, rounds=()):
	armies = [OURS, THEIRS] if our_side == 0 else [THEIRS, OURS]
	start = {"event": "start", "our_side": our_side, "armies": armies, "our_mana": 50, "our_mana_max": 50}
	return [start, *rounds, {"event": "end", "our_mana": 30, **end}]


def test_win_counts_survivors_and_enemy_losses():
	s = summarize(events(0, {"round": 3, "winner": 0, "result": RESULT_NORMAL, "casualties": [[[0, 4]], THEIRS], "alive": [[], []]}))
	assert s.outcome == Outcome.WIN
	assert s.own_kept == army_value(OURS) - army_value([[0, 4]])
	assert s.enemy_lost == army_value(THEIRS)
	assert s.rounds == 3 and abs(s.mana_used_fraction - 0.4) < 1e-9


def test_our_retreat_and_surrender():
	retreat = summarize(events(1, {"round": 2, "winner": 0, "result": RESULT_ESCAPE, "casualties": [[], [[0, 1]]], "alive": [[], []]}))
	assert retreat.outcome == Outcome.RETREAT and retreat.own_kept == 0  # level-1 hero: worth nothing
	surrender = summarize(events(0, {"round": 2, "winner": 1, "result": RESULT_SURRENDER, "casualties": [[], []], "alive": [[], []]}) + [{"event": "surrender", "cost": 900}])
	assert surrender.outcome == Outcome.SURRENDER and surrender.own_kept < army_value(OURS)


def test_enemy_retreat_loses_its_whole_army():
	s = summarize(events(0, {"round": 2, "winner": 0, "result": RESULT_ESCAPE, "casualties": [[], [[42, 2]]], "alive": [[], []]}))
	assert s.outcome == Outcome.WIN and s.enemy_lost == army_value(THEIRS)


def test_round_scores_from_round_end_events():
	rounds = [
		{"event": "round_end", "round": 0, "alive": [OURS, THEIRS], "our_mana": 50},  # battle start: ignored
		{"event": "round_end", "round": 1, "alive": [[[0, 10]], [[42, 10]]], "our_mana": 40},
	]
	s = summarize(events(0, {"round": 2, "winner": 0, "result": RESULT_NORMAL, "casualties": [[], THEIRS], "alive": [[], []]}, rounds))
	assert list(s.round_scores) == [1]


def test_refused_model_makes_the_record_unusable():
	with pytest.raises(RuntimeError):
		summarize([{"event": "error", "message": "cannot use model: x"}] + events(0, {"round": 1, "winner": 0, "result": 0, "casualties": [[], []], "alive": [[], []]}))
