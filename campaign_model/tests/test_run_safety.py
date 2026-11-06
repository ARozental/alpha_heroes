"""The 2026-11-01 review's fixes: an eval cut by the run's stop writes no row (eval 53 scored 0.000 from 48 stopped
games), and the schedule credit's share baseline is the V7 head only as far as it is trusted."""

import threading

import torch

from campaign_model import evaluate
from campaign_model.config import CampaignConfig
from campaign_model.train import share_baseline


def test_the_share_baseline_follows_the_heads_trust():
	real = torch.tensor([0.1, -0.3, float("nan"), 0.5])
	predicted = torch.tensor([0.4, 0.4, 0.4, 0.4])
	mean = (0.1 - 0.3 + 0.5) / 3
	assert torch.allclose(share_baseline(real, predicted, 0.0), torch.full((4,), mean))  # untrusted: the batch's mean
	assert torch.allclose(share_baseline(real, predicted, 1.0), predicted)
	assert torch.allclose(share_baseline(real, predicted, 0.25), 0.25 * predicted + 0.75 * mean)


def test_an_eval_cut_by_the_runs_stop_writes_no_row(monkeypatch):
	stop = threading.Event()
	stop.set()
	monkeypatch.setattr(evaluate, "eval_map", lambda seed, players, config: (_ for _ in ()).throw(RuntimeError("stopped")))
	config = CampaignConfig(eval_games=4, eval_parallel=2)
	assert evaluate.evaluate(config, server=None, battle=None, version=1, epoch=1, stop=stop) is None


def test_no_eval_begins_while_the_run_drains(tmp_path):
	"""learner.eval_due: an eval due does not begin while runtime/drain_restart.sh drains the run (it would be cut by the
	restart); it begins once the drain file is gone."""
	from types import SimpleNamespace

	from campaign_model.learner import CampaignLearner

	(tmp_path / "campaign").mkdir()
	learner = SimpleNamespace(config=CampaignConfig(eval_minutes=60.0, eval_games=4), evaluating=lambda: False, last_eval=None, out=tmp_path / "campaign")
	assert CampaignLearner.eval_due(learner)
	(tmp_path / "drain").touch()
	assert not CampaignLearner.eval_due(learner)
	(tmp_path / "drain").unlink()
	assert CampaignLearner.eval_due(learner)


def test_an_evals_checkpoint_holds_the_weights_it_scores(tmp_path, monkeypatch):
	"""learner.evaluate saves checkpoint_<version>.pt as it pins the eval's weights: saved once the eval ended, it held
	the weights of the updates that landed meanwhile (2026-11-06: the rollback to "v1288" restored v1293's)."""
	from types import SimpleNamespace

	from campaign_model.learner import CampaignLearner

	saved = []
	learner = SimpleNamespace(model_lock=threading.Lock(), version=1288, model=None, out=tmp_path, config=CampaignConfig(), stop=None)
	learner.server = SimpleNamespace(set_model=lambda name, version, model: None)
	learner.battle = SimpleNamespace(pin_eval=lambda: None)
	learner._save = lambda path: saved.append((path.name, learner.version))

	def play(config, server, battle, version, epoch, stop, model):
		learner.version = 1293  # (updates land while the eval plays)
		return None  # (stopped with the run: no row)

	monkeypatch.setattr(evaluate, "evaluate", play)
	assert CampaignLearner.evaluate(learner, epoch=1) is None
	assert saved == [("checkpoint_01288.pt", 1288)]
