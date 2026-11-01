"""No stop loses data: a finished campaign game is on disk until the update that trained on it is checkpointed, and a
resumed learner takes the rest back; the battles not trained on yet are saved at the stop and taken back."""

import threading
from types import SimpleNamespace

from battle_model.actors import ActorPool
from campaign_model.learner import CampaignLearner, unspool


def learner(tmp_path):
	spooled = CampaignLearner.__new__(CampaignLearner)
	spooled.spool, spooled.finished, spooled.version, spooled.labeler, spooled.label_futures = tmp_path, [], 0, None, {}
	return spooled


def test_finished_games_survive_a_stop_until_trained(tmp_path):
	first = learner(tmp_path)
	games = [SimpleNamespace(decisions=[1] * n) for n in (3, 5, 7)]
	for game in games:
		first._spool_game(game, {"key": len(game.decisions)})
	unspool(games[:1])  # trained on (its update checkpointed)
	(tmp_path / "123-cut.tmp").write_bytes(b"cut off")  # a write cut by the stop
	# the run stops; the next process of it:
	resumed = learner(tmp_path)
	resumed._reload_spool()
	assert [len(game.decisions) for game, _ in resumed.finished] == [5, 7]
	assert [info["key"] for _, info in resumed.finished] == [5, 7]
	assert not list(tmp_path.glob("*.tmp"))
	unspool(game for game, _ in resumed.finished)
	assert not list(tmp_path.glob("*.pkl"))


def test_the_battle_buffer_survives_a_stop(tmp_path):
	def pool():
		p = ActorPool.__new__(ActorPool)
		p.lock, p.buffer = threading.Condition(), []
		return p

	before = pool()
	before.buffer = [{"data": [1, 2], "decisions": 2, "source": "real", "version": 4}]
	before.save_buffer(tmp_path / "battle-buffer.pt")
	after = pool()
	after.buffer = [{"data": [3], "decisions": 1, "source": "generated", "version": 5}]
	after.load_buffer(tmp_path / "battle-buffer.pt")
	assert [b["decisions"] for b in after.buffer] == [2, 1]
	assert not (tmp_path / "battle-buffer.pt").exists()
