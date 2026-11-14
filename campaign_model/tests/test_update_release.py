"""The update's games released before its memory is trimmed (2026-11-14, the heap census: learner.step's `played` and the
loops' last `game` kept the update's restored inputs alive through trim_memory): at the trim no game the update took is
reachable."""
import collections
import dataclasses
import gc
import threading
import weakref
from types import SimpleNamespace

import torch

from campaign_model.config import CampaignConfig
from campaign_model.learner import CampaignLearner
from campaign_model.tests.test_learner import Game, snapshot


def test_no_game_of_the_update_is_alive_when_its_memory_is_trimmed(tmp_path, monkeypatch):
	from campaign_model.empire import EmpireConfig
	from campaign_model.labels import LabelConfig
	from campaign_model.learner import COLUMNS
	from campaign_model.model import build_model
	from campaign_model.schema import random_inputs
	from campaign_model.tests.test_model import encoded
	from campaign_model.tests.test_train import decisions_for
	from campaign_model.train import make_optimizer
	import campaign_model.learner as learner_module

	torch.manual_seed(6)
	kwargs = {"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"}
	model = build_model(kwargs).eval()
	config = CampaignConfig(decisions_per_update=8, min_games_per_update=2, batch_size=4, passes=1, learner="thread", device="cpu", credit="schedule", share_target="change", model_architecture="v2", noise_every=0, adv_weight=1.0)
	finished = []
	for g in range(2):
		x = random_inputs(8, seed=30 + g)
		decisions = [dataclasses.replace(d, inputs=encoded(x, b), day=1 + b // 3) for b, d in enumerate(decisions_for(model, x, seed=g))]
		finished.append((Game(decisions=decisions, ledgers={d: snapshot(d, 1000 + 300 * d, 1000) for d in range(1, 14)}, days=12), SimpleNamespace(seconds=1.0, map_name="Maps/alpha_random/x")))
	refs = [weakref.ref(game) for game, _ in finished]
	del decisions, x

	class Writer:
		def add_scalar(self, *args):
			pass

		def flush(self):
			pass

	learner = SimpleNamespace(
		config=config, labels=LabelConfig(), empire=EmpireConfig(), version=1, finished=finished, lock=threading.Condition(), stats=collections.Counter(),
		model=model, optimizer=make_optimizer(model, config), device=torch.device("cpu"), worker=None, model_lock=threading.Lock(), lr_scale=1.0,
		server=SimpleNamespace(set_model=lambda *args: None), teacher=None, teacher_weight=0.0, games_used=0, out=tmp_path, kwargs=kwargs, value_ev=None, credit_evidence={}, credit_check=1.0, head_ev={"Vend": 0.2}, value_pool=None, head_pool={}, progress=tmp_path / "progress.csv", writer=Writer(),
		features_cpu=model.features, trust_ema=None, grad_clips=None, clip_samples=[], head_residual=None, head_var_slow=None, head_residual_updates=0, held=None, held_updates=0, gather_ramp_start=None, valve=None, value_readings=[], value_own=0,
	)
	for name in ("_take", "trust", "trust_today", "_save", "_log", "_note_value_ev", "_note_head_ev", "grade_trust", "end_credit", "_refresh_features", "_relabel_waiting", "loss_shares", "loss_weights", "_note_gradients", "gather_weight_now", "valve_now", "_note_valve", "_valve_reading", "_head_scale", "_calibrate_clips", "_judge_value"):
		setattr(learner, name, getattr(CampaignLearner, name).__get__(learner))
	learner.progress.write_text(",".join(COLUMNS) + "\n")
	del finished
	alive = []
	monkeypatch.setattr(learner_module, "trim_memory", lambda: (gc.collect(), alive.append(sum(r() is not None for r in refs))))
	assert CampaignLearner.step(learner, epoch=1)
	assert alive == [0], alive  # (at the trim: none of the update's games reachable)
