"""The update's size cap (config.update_decisions_cap, update_games_floor; e5, 2026-11-14): learner.take_count and _take on
queues of games: at 0 the update as before; 9k-decision games give 13 under a 120k cap with a floor of 12; 4k games give 20;
one 40k game still goes (the floor wins); nothing dropped or reordered; the line each update."""
import collections
import threading
from types import SimpleNamespace

from campaign_model.config import CampaignConfig
from campaign_model.learner import CampaignLearner, take_count


def queue(*sizes):
	return [(SimpleNamespace(decisions=[0] * k, tag=i), SimpleNamespace(seconds=1.0)) for i, k in enumerate(sizes)]


def learner(config, finished):
	return SimpleNamespace(config=config, finished=list(finished), lock=threading.Condition(), stats=collections.Counter(), label_futures={})


RUN = dict(min_games_per_update=20, decisions_per_update=32768)


def test_at_zero_the_update_is_as_before():
	off = CampaignConfig(**RUN)
	for sizes in ([9000] * 30, [4000] * 25, [40000] + [4000] * 25, [2000] * 30, [9000] * 19):
		q = queue(*sizes)
		before = None
		taken, decisions = [], 0  # (the rule as it was: games until 20 of them hold 32768 decisions)
		for game, info in q:
			if len(taken) >= off.min_games_per_update and decisions >= off.decisions_per_update:
				break
			taken.append(game)
			decisions += len(game.decisions)
		if len(taken) >= off.min_games_per_update and decisions >= off.decisions_per_update:
			before = (len(taken), decisions, False)
		assert take_count(q, off) == before, sizes


def test_long_games_stop_at_the_cap_short_ones_still_make_twenty(capsys):
	on = CampaignConfig(update_decisions_cap=120000, update_games_floor=12, **RUN)
	assert take_count(queue(*[9000] * 30), on) == (13, 117000, True)
	assert take_count(queue(*[4000] * 30), on) == (20, 80000, False)
	assert take_count(queue(*([40000] + [4000] * 25)), on) == (20, 116000, False)  # (the 40k game first: under the floor it goes)
	assert take_count(queue(*[40000] * 15), on) == (12, 480000, True)  # (the floor wins over the cap)
	assert take_count(queue(*[9000] * 11), on) is None  # (fewer than the floor: wait)
	assert take_count(queue(*[9000] * 12), on) is None  # (12 hold 108k: the next one may still fit, so wait as before)
	assert take_count(queue(*[9000] * 14), on) == (13, 117000, True)


def test_take_hands_the_oldest_in_order_and_leaves_the_rest_waiting(capsys):
	on = CampaignConfig(update_decisions_cap=120000, update_games_floor=12, **RUN)
	q = queue(*[9000] * 30)
	ln = learner(on, q)
	taken = CampaignLearner._take(ln)
	assert [g.tag for g, _ in taken] == list(range(13)) and [g.tag for g, _ in ln.finished] == list(range(13, 30)) and not ln.stats["dropped_games"]
	assert "campaign update size: 13 games, 117000 decisions, 17 games left waiting; the cap bound (cap 120000, floor 12)" in capsys.readouterr().out
	off = learner(CampaignConfig(**RUN), queue(*[9000] * 30))
	assert len(CampaignLearner._take(off)) == 20 and "campaign update size" not in capsys.readouterr().out
	assert CampaignLearner.ready(learner(on, queue(*[9000] * 14))) and not CampaignLearner.ready(learner(on, queue(*[9000] * 13)))  # (13 fit: the 14th decides the cap)
