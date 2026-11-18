"""The fresh run of 2026-11-18 (the user: a fresh campaign beside a frozen battle model, today's settings from its first
update): config.controls_from takes a checkpoint's training controls, none of its weights; battle_frozen trains no
battle and starts no battle loader."""
import inspect

import torch

from campaign_model.learner import controls_of


def test_the_controls_are_the_weights_of_the_loss_shares_and_the_held_value_heads_only():
	state = {"model": {"w": torch.ones(2)}, "loss_scale": {"log_norm": {"policy": 1.0}, "weight": {"value": 352.6, "teacher": 2.03, "foresight": 0.696}}, "head_weights_frozen": {"share": [1.0, 2.0]}, "optimizer": {}}
	scale, held = controls_of(state)
	assert scale == {"log_norm": {}, "weight": {"value": 352.6, "teacher": 2.03, "foresight": 0.696}} and held == {"share": [1.0, 2.0]}
	scale["weight"]["value"] = 1.0
	assert state["loss_scale"]["weight"]["value"] == 352.6  # (a copy)
	assert controls_of({"model": {}}) == (None, None)


def test_a_frozen_battle_model_drops_its_battles_and_has_no_battle_loader():
	from battle_model import supply, train

	source = inspect.getsource(train.train)
	assert "if config.battle_frozen:" in source and "campaign.start_eval(epoch - 1)" in source
	assert train.TrainConfig().battle_frozen is False
	assert 'getattr(config, "battle_frozen", False) else {"battle": config.generated_actors}' in inspect.getsource(supply.Supply.__init__)
	from types import SimpleNamespace

	s = supply.Supply.__new__(supply.Supply)
	s.shared = SimpleNamespace(ceilings={"campaign": 20})
	s.config, s.listener = SimpleNamespace(remote_slots=15), object()
	assert s.workers("battle") == 0 and s.workers("campaign") == 35


def test_the_battle_weights_md5_reads_every_parameter():
	from battle_model.train import weights_md5

	model = torch.nn.Linear(3, 2)
	before = weights_md5(model)
	assert weights_md5(model) == before
	with torch.no_grad():
		model.bias[0] += 1e-6
	assert weights_md5(model) != before
