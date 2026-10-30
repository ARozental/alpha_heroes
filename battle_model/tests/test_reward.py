"""reward.py: zero-sum from each side's assets."""
import itertools

import pytest

from battle_model.reward import HERO_HIRE_COST, Outcome, RewardConfig, assets_after, hero_asset, hero_level_value, mana_value, other_side_value, resources_value, round_score, terminal_reward, zero_sum

ARMY, ENEMY = 50_000, 40_000


def test_resource_prices():
	assert resources_value({"gold": 3000, "gems": 1}) == 3500
	assert resources_value({"wood": 2, "ore": 2}) == 1000


def test_a_hero_is_worth_its_levels_and_its_artifacts():
	"""(the user, 2026-10-29) Every level counts: the level-up from level L is worth 2000 x min(L, 3)."""
	assert hero_level_value(1) == 0
	assert [hero_level_value(level) for level in (2, 3, 4, 5, 10, 20)] == [2000, 6000, 12000, 18000, 48000, 108000]
	assert hero_level_value(2) < 2500 < hero_level_value(3)  # a lone level-2 hero is not worth a retreat's rehire; a level-3 one is
	assert hero_level_value(12.5) == pytest.approx(60000 + 0.5 * 6000)  # halfway to 13
	assert hero_asset(4, 7000) == 19000


def test_the_winner_over_a_hero_takes_its_artifacts():
	assert assets_after(Outcome.WIN, 30_000, 10_000, 0, captured=4000) == 44_000
	assert assets_after(Outcome.LOSS, 30_000, 10_000, 0) == 0
	assert assets_after(Outcome.RETREAT, 30_000, 10_000, 600) == 10_600 - HERO_HIRE_COST  # the hero and its artifacts escape, to be rehired
	assert assets_after(Outcome.RETREAT, 30_000, 0, 600) == 0  # a hero worth less than its rehire: as if lost


def test_the_two_sides_rewards_always_sum_to_zero():
	for own_end, enemy_end in itertools.product((0, 10_000, ARMY), (0, 5_000, ENEMY)):
		ours = terminal_reward(ARMY, own_end, ENEMY, enemy_end)
		theirs = terminal_reward(ENEMY, enemy_end, ARMY, own_end)
		assert abs(ours + theirs) < 1e-12 and -1 <= ours <= 1
	assert other_side_value(0.3) == -0.3


def test_every_kill_helps_the_killer_and_hurts_the_victim():
	before = terminal_reward(ARMY, 20_000, ENEMY, 10_000)
	assert terminal_reward(ARMY, 20_000, ENEMY, 9_000) > before  # we killed more
	assert terminal_reward(ARMY, 19_000, ENEMY, 10_000) < before  # we lost more


def test_a_loss_that_traded_well_beats_a_win_that_did_not():
	"""No win term: what counts is what each side lost."""
	costly_win = terminal_reward(ARMY, 1_000, ENEMY, assets_after(Outcome.LOSS, 0, 0, 0))  # we lost 49k, they 40k
	good_loss = terminal_reward(20_000, assets_after(Outcome.LOSS, 0, 0, 0), ENEMY, 5_000)  # we lost 20k, they 35k
	assert good_loss > 0 > costly_win


def test_what_each_outcome_leaves():
	hero, mana = 10_000, 600
	assert assets_after(Outcome.WIN, 30_000, hero, mana) == 30_000 + hero + mana
	assert assets_after(Outcome.LOSS, 30_000, hero, mana) == 0 == assets_after(Outcome.DRAW, 30_000, hero, mana)
	assert assets_after(Outcome.RETREAT, 30_000, hero, mana) == hero + mana - HERO_HIRE_COST  # the army is lost, the hero rehired
	assert assets_after(Outcome.SURRENDER, 30_000, hero, mana, 6_000) == 30_000 + hero + mana - 6_000 - HERO_HIRE_COST


def test_mana_is_worth_its_value_times_spell_power():
	assert mana_value(50, 5, RewardConfig(mana_value=2.0)) == 500
	spent = terminal_reward(ARMY + mana_value(50, 5), ARMY + mana_value(10, 5), ENEMY, ENEMY)
	assert spent < 0  # spending mana and achieving nothing costs its value


def test_round_score_is_the_same_formula_on_the_assets_as_they_stand():
	assert round_score(ARMY, 30_000, ENEMY, 25_000) == zero_sum(ARMY, 30_000, ENEMY, 25_000)
