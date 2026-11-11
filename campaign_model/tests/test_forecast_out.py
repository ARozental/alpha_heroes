"""Forecast-out (bundle 60): each forecast about one option fed straight to that option's logit, detached, 0 to start.
At 0 the model is as before (test_map_size's golden outputs); the policy's loss reaches no forecast head through the new
path; a planted weight on each moves its own option's logits and nothing else."""
import torch

from campaign_model import constants as c
from campaign_model.model_v2 import FORE_FIGHT, CampaignModelV2
from campaign_model.tests.test_model_v2 import rich_inputs


def setup(n=8, seed=4):
	torch.manual_seed(3)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	return model, {k: torch.from_numpy(v) for k, v in rich_inputs(n, seed).items()}


def test_the_policy_reaches_no_forecast_head_through_it():
	model, x = setup()
	with torch.no_grad():
		for layer in (model.fore_fight, model.fore_move, model.fore_answer):
			layer.weight.fill_(0.3)
		model.fore_hero.weight.normal_(0.0, 0.3)
	out = model(x)
	finite = lambda t: torch.where(torch.isfinite(t), t, torch.zeros_like(t)).sum()  # noqa: E731
	(finite(out["move_logp"]) + finite(out["option_logp"]) + finite(out["type_logp"])).backward()
	f = model.foresight
	for name, module in (("move_out", f.move_out), ("answer_out", f.answer_out), ("hero_out", f.hero_out), ("tiles", f.tiles), ("outcome", f.outcome)):
		for p in module.parameters():
			assert p.grad is None or float(p.grad.abs().max()) == 0.0, name
	assert float(model.fore_move.weight.grad.abs().max()) > 0 and float(model.fore_answer.weight.grad.abs().max()) > 0


def test_each_planted_weight_moves_its_own_options_logits_only():
	model, x = setup()
	with torch.no_grad():
		for p in model.foresight.answer_out.parameters():  # (a trained head's options differ; a fresh one's are all alike)
			p.normal_(0.0, 0.5)
		base = model(x)
		model.fore_answer.weight.fill_(0.5)
		answers = model(x)
		model.fore_answer.weight.zero_()
		model.fore_move.weight.fill_(0.5)
		moves = model(x)
		model.fore_move.weight.zero_()
	question = ~((x["g_kind"] == c.DecisionKind.TURN) | (x["g_kind"] == c.DecisionKind.EXCHANGE))
	assert not torch.allclose(answers["option_logp"][question], base["option_logp"][question], equal_nan=True)
	for name in ("move_logp", "travel_logp", "type_logp"):
		assert torch.allclose(answers[name], base[name], atol=1e-6, equal_nan=True), name
	assert not torch.allclose(moves["move_logp"], base["move_logp"], equal_nan=True)
	for name in ("option_logp", "build_logp", "recruit_logp"):
		assert torch.allclose(moves[name], base[name], atol=1e-6, equal_nan=True), name


def test_the_hero_forecast_reaches_the_hero_tokens_decide_reads():
	"""D: a planted fore_hero moves the hero tokens decide reads (a fresh model's output heads start at 0, so its logits
	do not show it yet)."""
	import campaign_model.model_v2 as mv2

	model, x = setup()
	seen = []
	original = mv2.CampaignModelV2.decide

	def decide(self, state, known):
		seen.append(state.heroes.clone())
		return original(self, state, known)

	mv2.CampaignModelV2.decide = decide
	try:
		with torch.no_grad():
			model(x)
			model.fore_hero.weight.normal_(0.0, 0.5)
			model(x)
	finally:
		mv2.CampaignModelV2.decide = original
	present = x["h_present"]
	assert float((seen[1] - seen[0])[present].abs().max()) > 0  # (an absent hero's token moves too: the block's attention mixes them after)


def test_the_fight_channels_are_zero_off_a_fight_tile():
	from campaign_model import model_v2 as mv2

	model, x = setup()
	seen = {}
	original = mv2.fight_forecast

	def spy(x_, known, foreseen):
		out = original(x_, known, foreseen)
		seen["out"], seen["gate"] = out, known.tile_fights
		return out

	mv2.fight_forecast = spy
	try:
		with torch.no_grad():
			model(x)
	finally:
		mv2.fight_forecast = original
	out, gate = seen["out"], seen["gate"] > 0
	assert out.shape[-1] == FORE_FIGHT and float(out[~gate].abs().max()) == 0.0
