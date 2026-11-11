"""Schema 19's games side: Nullkiller2's view as inputs on in every evaluation, off in config.nk2_input_dropout of the
training games, drawn from the game's seed (play.nk2_settings, supply.Runner.campaign_settings)."""
from types import SimpleNamespace

from campaign_model.config import CampaignConfig
from campaign_model.play import nk2_settings


def test_evaluations_have_it_whenever_the_config_does():
	assert dict(nk2_settings(CampaignConfig(nk2_inputs=True, nk2_input_dropout=1.0)))["ALPHA_MAP_NK2_INPUTS"] == "1"
	assert dict(nk2_settings(CampaignConfig(nk2_inputs=False)))["ALPHA_MAP_NK2_INPUTS"] == "0"


def test_training_games_go_without_it_at_the_dropout_drawn_from_their_seed():
	on = lambda dropout, seed: dict(nk2_settings(CampaignConfig(nk2_inputs=True, nk2_input_dropout=dropout), seed))["ALPHA_MAP_NK2_INPUTS"] == "1"  # noqa: E731
	assert all(on(0.0, s) for s in range(200)) and not any(on(1.0, s) for s in range(200))
	share = sum(not on(0.2, s) for s in range(4000)) / 4000
	assert 0.17 < share < 0.23
	assert [on(0.2, s) for s in range(50)] == [on(0.2, s) for s in range(50)]  # (the same game the same way)


def test_the_training_games_get_the_switch():
	from battle_model.supply import Runner

	def settings(**campaign):
		runner = SimpleNamespace(battle_socket="/tmp/b.sock", link="link.pt", records=SimpleNamespace(path="/tmp/r.sock"), config=SimpleNamespace(campaign=CampaignConfig(**campaign)))
		return dict(Runner.campaign_settings(runner, "game", 3))

	assert settings(nk2_inputs=True, nk2_input_dropout=0.0)["ALPHA_MAP_NK2_INPUTS"] == "1"
	assert settings(nk2_inputs=True, nk2_input_dropout=1.0)["ALPHA_MAP_NK2_INPUTS"] == "0"
	assert settings(nk2_inputs=False)["ALPHA_MAP_NK2_INPUTS"] == "0"


def test_with_its_view_the_games_ask_its_gather_advice_too():
	"""Its visit again of a town is our GATHER only with ALPHA_MAP_GATHER_ADVICE: on wherever the inputs are, whatever
	nk2_gather_weight; a game without the inputs keeps the settings it had (supply's own)."""
	from battle_model.supply import Runner

	on = dict(nk2_settings(CampaignConfig(nk2_inputs=True, nk2_gather_weight=0.0)))
	assert on == {"ALPHA_MAP_NK2_INPUTS": "1", "ALPHA_MAP_GATHER_ADVICE": "1", "ALPHA_MAP_NK2_CHOICE": "1"}  # (own games: its choice shown, test_own_games)
	assert "ALPHA_MAP_GATHER_ADVICE" not in dict(nk2_settings(CampaignConfig(nk2_inputs=False)))
	runner = SimpleNamespace(battle_socket="/tmp/b.sock", link="link.pt", records=SimpleNamespace(path="/tmp/r.sock"), config=SimpleNamespace(campaign=CampaignConfig(nk2_inputs=True, nk2_advice=1.0, nk2_gather_weight=0.0)))
	assert dict(Runner.campaign_settings(runner, "game", 3))["ALPHA_MAP_GATHER_ADVICE"] == "1"
