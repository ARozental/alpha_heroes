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


def test_the_close_preference_moves_end_at_exchange_rows_only_and_off_is_identical():
	"""config.exchange_close (the user, 2026-11-18): off, the model's outputs bit for bit; on, END's type log-probability
	rises at EXCHANGE rows by its start (0.5), every TURN row's outputs identical."""
	import numpy as np

	from campaign_model import constants as c
	from campaign_model.config import CampaignConfig
	from campaign_model.learner import set_advice_prior
	from campaign_model.model import EXCHANGE_CLOSE_START, build_model
	from campaign_model.tests.test_model_v2 import rich_inputs
	from campaign_model.train import model_kwargs

	torch.manual_seed(1)
	model = build_model(model_kwargs(CampaignConfig(model_architecture="v2"))).eval()
	x = {k: torch.as_tensor(v) for k, v in rich_inputs(32, seed=4).items()}
	exchange = x["g_kind"] == c.DecisionKind.EXCHANGE
	assert exchange.any() and (~exchange).any()
	with torch.no_grad():
		before = model(x)
		set_advice_prior(model, CampaignConfig(model_architecture="v2", exchange_close=False))
		off = model(x)
		set_advice_prior(model, CampaignConfig(model_architecture="v2", exchange_close=True))
		on = model(x)
	assert float(model.features.exchange_close_on) == 1.0 and abs(float(model.features.exchange_close_start()) - EXCHANGE_CLOSE_START) < 1e-6
	for k, v in before.items():
		if torch.is_tensor(v):
			assert torch.equal(v, off[k]), k
	END = int(c.ActionType.END)
	turn = ~exchange
	for k, v in before.items():
		if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == len(exchange):
			assert torch.equal(v[turn], on[k][turn]), k
	lift = on["type_logp"][exchange, END] - before["type_logp"][exchange, END]
	finite = torch.isfinite(lift)
	assert finite.any() and (lift[finite] > 0).all()
	p = before["type_logp"][exchange, END].exp()[finite]
	expected = torch.log(np.exp(EXCHANGE_CLOSE_START) / (1 - p + p * np.exp(EXCHANGE_CLOSE_START)))
	assert torch.allclose(lift[finite], expected.to(lift.dtype), atol=1e-4)


def test_take_best_leads_give_all_where_both_are_on_offer_and_off_is_identical():
	"""config.take_best_leads (the user, 2026-11-18): off, MULTI's prior bit for bit; on, wherever TAKE_BEST is legal for a
	receiver GIVE_ALL's prior is at most TAKE_BEST's - TAKE_LEAD_MARGIN, every other entry unchanged."""
	from campaign_model import constants as c
	from campaign_model.config import CampaignConfig
	from campaign_model.learner import set_advice_prior
	from campaign_model.model import TAKE_LEAD_MARGIN, build_model
	from campaign_model.tests.test_model_v2 import rich_inputs
	from campaign_model.train import model_kwargs

	torch.manual_seed(2)
	model = build_model(model_kwargs(CampaignConfig(model_architecture="v2"))).eval()
	x = {k: torch.as_tensor(v) for k, v in rich_inputs(48, seed=6).items()}
	give, take = c.MultiKind.GIVE_ALL, c.MultiKind.TAKE_BEST
	x["multi_mask"] = x["multi_mask"].clone()
	x["multi_mask"][:16, :2, take] = True  # (rich_inputs offers none: both on offer at the first rows, the prior as computed)
	x["multi_mask"][:24, :2, give] = True
	with torch.no_grad():
		before = model.domain_knowledge(x).multi_prior.clone()
		set_advice_prior(model, CampaignConfig(model_architecture="v2", take_best_leads=False))
		off = model.domain_knowledge(x).multi_prior.clone()
		set_advice_prior(model, CampaignConfig(model_architecture="v2", take_best_leads=True))
		on = model.domain_knowledge(x).multi_prior.clone()
	assert torch.equal(before, off)
	legal = x["multi_mask"][:, :2, take]
	assert legal.any(), "the inputs offer no TAKE_BEST"
	assert (on[:, :2, give][legal] <= on[:, :2, take][legal] - TAKE_LEAD_MARGIN + 1e-6).all()
	assert torch.equal(on[:, :2, give][~legal], before[:, :2, give][~legal])
	rest = torch.ones_like(on, dtype=torch.bool)
	rest[:, :2, give] = False
	assert torch.equal(on[rest], before[rest])
