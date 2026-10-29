import pytest

from battle_model import vocab
from battle_model.data import RESULT_ESCAPE, RESULT_NORMAL, RESULT_SURRENDER, army_value, summarize
from battle_model.reward import HERO_HIRE_COST, Outcome

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


def test_demons_summoned_for_good_are_kept():
	"""Pit Lords turn our 4 dead Pikemen into 2 demons (id 55), one of which dies again: the engine's casualties count
	the 4 Pikemen and the dead demon, the end event the 2 summoned; we keep the army but the Pikemen, plus one demon.
	The enemy's own summons offset its losses the same way."""
	end = {"round": 3, "winner": 0, "result": RESULT_NORMAL, "casualties": [[[0, 4], [55, 1]], [[42, 20], [55, 1]]], "alive": [[], []], "summoned": [[[55, 2]], [[55, 1]]]}
	s = summarize(events(0, end))
	assert abs(s.own_kept - (army_value(OURS) - army_value([[0, 4]]) + army_value([[55, 1]]))) < 1e-9
	assert abs(s.enemy_lost - army_value(THEIRS)) < 1e-9


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


def test_an_enemy_that_surrendered_keeps_its_army_and_pays_the_rehire():
	"""Zero-sum assets: a surrender keeps the army, so it costs the enemy only what it lost before, the rehire (and a
	payment unknown to us: 0)."""
	s = summarize(events(0, {"round": 2, "winner": 0, "result": RESULT_SURRENDER, "casualties": [[], [[42, 2]]], "alive": [[], []]}))
	assert s.outcome == Outcome.WIN and s.enemy_lost == army_value([[42, 2]]) + HERO_HIRE_COST


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


def test_a_compacted_record_trains_the_same(tmp_path):
	import numpy as np
	import torch

	from battle_model.data import Sparse, compact_record, expand_record, load_record, make_dataloader
	from battle_model.distributions import sample
	from battle_model.model import BattleModel
	from battle_model.schema import random_inputs

	torch.manual_seed(0)
	model = BattleModel(dim=32, layers=1).eval()
	decisions = []
	for i in range(5):
		x = random_inputs(batch=1, seed=i)
		x["s_hex_value"][:, 3:] = 0  # mostly zero, as real ones are
		with torch.no_grad():
			out = model({k: torch.from_numpy(v) for k, v in x.items()})
		outputs = {k: v.numpy() for k, v in out.items() if k != "value_logit"}
		decisions.append(({**x, **outputs}, sample(outputs, 0, np.random.default_rng(i))))
	write_record(tmp_path / "game_0", decisions, events(0, {"round": 3, "winner": 0, "result": RESULT_NORMAL, "casualties": [[], THEIRS], "alive": [[], []]}))

	full = load_record(tmp_path / "game_0")
	compact = compact_record(load_record(tmp_path / "game_0"))
	assert isinstance(compact.tensors["s_hex_value"], Sparse) and not any(k.startswith("logp_") for k in compact.tensors)
	assert compact.tensors["s_hex_value"].nbytes < full.tensors["s_hex_value"].nbytes / 10
	expanded = expand_record(compact)
	(a, _, _), (b, _, _) = (make_dataloader([r], batch_size=8, shuffle=False) for r in (full, expanded))
	for x, y in zip(a.dataset, b.dataset):
		assert all(np.array_equal(x["inputs"][k], y["inputs"][k]) for k in x["inputs"])
		assert torch.equal(x["old_log_prob"], y["old_log_prob"]) and torch.equal(x["advantage"], y["advantage"])
		assert torch.equal(x["value_labels"], y["value_labels"])


def test_war_machines_are_half_their_price_kept_unless_the_side_loses():
	"""The start event lists each side's war machines; the engine's casualties count the destroyed ones, which no
	army's start holds: a winner whose ballista was shot loses its half price (1250), not 2500 off its army; a loser
	loses the rest with its hero; the catapult is worth nothing."""
	BALLISTA, TENT, CATAPULT = 146, 147, 145
	assert vocab.creature_value(BALLISTA) == 1250 and vocab.creature_value(TENT) == 375 and vocab.creature_value(CATAPULT) == 0
	start = {"event": "start", "our_side": 0, "armies": [OURS, THEIRS], "machines": [[BALLISTA, TENT, CATAPULT], [BALLISTA]], "our_mana": 0, "our_mana_max": 0}
	end = {"event": "end", "round": 3, "winner": 0, "result": RESULT_NORMAL, "casualties": [[[0, 4], [BALLISTA, 1]], THEIRS], "alive": [[], []], "our_mana": 0}
	s = summarize([start, end])
	assert s.own_kept == army_value(OURS) - army_value([[0, 4]]) + 375  # the tent stands, the ballista does not
	assert s.enemy_lost == army_value(THEIRS) + 1250  # the loser's ballista goes with its hero
	old = summarize([{k: v for k, v in start.items() if k != "machines"}, end])  # a record from before: none counted
	assert old.own_kept == army_value(OURS) - army_value([[0, 4]])  # (the destroyed ballista was taken off as -2500)


def test_war_machines_in_the_campaign_are_half_their_price():
	from campaign_model.vocab import artifact_prices
	from game_values import artifact_price

	prices = artifact_prices()
	assert (prices[3], prices[4], prices[5], prices[6], prices[0], prices[2]) == (0, 1250, 500, 375, 0, 0)  # catapult, ballista, cart, tent; spellbook, Grail
	assert all(artifact_price(a) == prices[a] for a in (0, 2, 3, 4, 5, 6, 7))
	assert prices[4] == vocab.artifact_price(4) / 2 and prices[6] == vocab.artifact_price(6) / 2
