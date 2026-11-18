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


def test_a_fresh_models_weights_load_unchanged_once_its_reset_markers_are_set():
	"""A fresh model (no checkpoint) holds the one-time reset markers at 1 (learner.mark_resets_done): its published
	weights then load into a server's model as they are, the spellbook row and the RECRUIT hero rows untouched."""
	from campaign_model import constants as c
	from campaign_model import features as ft
	from campaign_model.config import CampaignConfig
	from campaign_model.learner import mark_resets_done
	from campaign_model.model import build_model
	from campaign_model.model_v2 import recruit_rows
	from campaign_model.train import model_kwargs

	kwargs = model_kwargs(CampaignConfig(model_architecture="v2"))
	torch.manual_seed(0)
	fresh = build_model(kwargs)
	if not fresh.features.main_on:
		return
	with torch.no_grad():
		fresh.buy_item.weight.normal_()
		fresh.recruit_head[2].weight.normal_()
	book = c.BUY_ARTIFACTS.index(ft.SPELLBOOK)
	served = build_model(kwargs)
	served.load_state_dict(fresh.state_dict())  # (markers at 0: the resets fire, as they would on every publish)
	assert float(served.buy_item.weight[book].norm()) == 0.0
	mark_resets_done(fresh)
	served = build_model(kwargs)
	served.load_state_dict(fresh.state_dict())
	assert torch.equal(served.buy_item.weight, fresh.buy_item.weight)
	assert torch.equal(served.recruit_head[2].weight[recruit_rows(1)], fresh.recruit_head[2].weight[recruit_rows(1)])
	assert not served.book_row_reset_now and not served.recruit_rows_reset_now
