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


def test_enemy_surrender_counts_as_full_destruction():
	s = summarize(events(0, {"round": 2, "winner": 0, "result": RESULT_SURRENDER, "casualties": [[], [[42, 2]]], "alive": [[], []]}))
	assert s.outcome == Outcome.WIN and s.enemy_lost == army_value(THEIRS)


def test_no_winner_is_a_draw():
	s = summarize(events(1, {"round": 1, "winner": -1, "result": RESULT_NORMAL, "casualties": [THEIRS, OURS], "alive": [[], []]}))
	assert s.outcome == Outcome.DRAW and s.own_kept == 0 and s.enemy_lost == army_value(THEIRS)


def write_record(path, decisions, events_):
	"""A record as AlphaBattleAI writes it: events.jsonl and one decision_*.npz (a .npy per tensor + action.json) per decision."""
	import io
	import json
	import zipfile

	import numpy as np

	path.mkdir(parents=True)
	(path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events_))
	for i, (tensors, action) in enumerate(decisions):
		with zipfile.ZipFile(path / f"decision_{i:05d}.npz", "w", zipfile.ZIP_DEFLATED) as archive:
			for name, array in tensors.items():
				buffer = io.BytesIO()
				np.lib.format.write_array(buffer, np.ascontiguousarray(array))
				archive.writestr(name + ".npy", buffer.getvalue())
			archive.writestr("action.json", json.dumps(action) + "\n")


def test_replayed_decisions_are_kept_in_the_line_but_not_trained_on(tmp_path):
	import numpy as np
	import torch

	from battle_model.data import make_dataloader, read_decisions
	from battle_model.distributions import sample
	from battle_model.model import BattleModel
	from battle_model.schema import random_inputs

	torch.manual_seed(0)
	model = BattleModel(dim=32, layers=1).eval()
	decisions = []
	for i in range(4):
		x = random_inputs(batch=1, seed=i)
		with torch.no_grad():
			out = model({k: torch.from_numpy(v) for k, v in x.items()})
		outputs = {k: v.numpy() for k, v in out.items() if k != "value_logit"}
		action = sample(outputs, 0, np.random.default_rng(i))
		action["forced"] = i < 2  # the first two were replayed from a prefix
		decisions.append(({**x, **outputs}, action))
	write_record(tmp_path / "game_0", decisions, events(0, {"round": 3, "winner": 0, "result": RESULT_NORMAL, "casualties": [[], THEIRS], "alive": [[], []]}))

	assert len(read_decisions(tmp_path / "game_0")) == 4
	loader, summaries, lines = make_dataloader([tmp_path / "game_0"], batch_size=8, shuffle=False)
	assert [e["forced"] for e in lines[0]] == [True, True, False, False]
	assert len(loader.dataset) == 2 and not any(e["forced"] for e in loader.dataset)
	assert summaries[0].outcome == Outcome.WIN and summaries[0].decisions == 4


def test_the_enemy_side_scores_the_same_battle_mirrored():
	end = {"round": 3, "winner": 0, "result": RESULT_NORMAL, "casualties": [[[0, 4]], THEIRS], "alive": [[], []]}
	ours = summarize(events(0, end))
	theirs = summarize(events(0, end), side=1)
	assert ours.outcome == Outcome.WIN and theirs.outcome == Outcome.LOSS
	assert theirs.our_side == 1 and theirs.own_kept == 0 and theirs.enemy_lost == army_value([[0, 4]])
	assert theirs.own_start_value == army_value(THEIRS) and theirs.enemy_start_value == army_value(OURS)
	# We retreat: the enemy wins and counts our whole army as destroyed
	fled = summarize(events(1, {"round": 2, "winner": 0, "result": RESULT_ESCAPE, "casualties": [[], []], "alive": [[], []]}), side=0)
	assert fled.outcome == Outcome.WIN and fled.enemy_lost == army_value(OURS)
