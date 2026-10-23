from battle_model.reward import HERO_HIRE_COST, Outcome, hero_value, own_kept_value, resources_value, retreat_value, terminal_reward

ARMY = 50_000
ENEMY = 40_000
FRESH_HERO = 3_000


def reward(outcome, army_end, hero, enemy_lost, rounds=8, mana=0.3, surrender_cost=0):
	kept = own_kept_value(outcome, army_end, hero, FRESH_HERO, surrender_cost)
	return terminal_reward(outcome, ARMY + hero, kept, ENEMY, enemy_lost, rounds, mana)


def test_resource_prices():
	assert resources_value({"gold": 3000, "gems": 1}) == 3500
	assert resources_value({"wood": 2, "ore": 2}) == 1000


def test_hero_value_is_artifacts_plus_experience():
	assert hero_value([2000, 5000], 14700) == 21700  # level 10 = 14700 XP


def test_retreat_value_floor():
	assert retreat_value(2_000, FRESH_HERO) == 0
	assert retreat_value(30_000, FRESH_HERO) == 27_000


def test_any_win_beats_any_other_outcome():
	hero = 20_000
	worst_win = reward(Outcome.WIN, army_end=0, hero=hero, enemy_lost=ENEMY, rounds=60, mana=1.0)
	best_other = max(
		reward(Outcome.RETREAT, 0, 500_000, ENEMY, rounds=1, mana=0),
		reward(Outcome.SURRENDER, ARMY, 500_000, ENEMY, rounds=1, mana=0, surrender_cost=1_000),
		reward(Outcome.LOSS, 0, hero, ENEMY, rounds=1, mana=0),
	)
	assert worst_win > best_other


def test_scenario_ordering():
	hero = 20_000
	cheap_win = reward(Outcome.WIN, ARMY * 0.9, hero, ENEMY)
	costly_win = reward(Outcome.WIN, ARMY * 0.2, hero, ENEMY)
	surrender = reward(Outcome.SURRENDER, ARMY * 0.8, hero, ENEMY * 0.3, surrender_cost=20_000)
	retreat = reward(Outcome.RETREAT, 0, hero, ENEMY * 0.3)
	death = reward(Outcome.LOSS, 0, hero, ENEMY * 0.3)
	assert cheap_win > costly_win > surrender > retreat > death


def test_weak_hero_retreat_equals_death():
	weak = 1_000  # below a fresh tavern hero
	assert reward(Outcome.RETREAT, 0, weak, 10_000) == reward(Outcome.LOSS, 0, weak, 10_000)


def test_surrender_pays_cost_and_rehire():
	kept = own_kept_value(Outcome.SURRENDER, army_end_value=30_000, hero_value_=10_000, surrender_cost=6_000)
	assert kept == 30_000 + 10_000 - 6_000 - HERO_HIRE_COST


def test_penalties_reduce_reward():
	base = reward(Outcome.WIN, ARMY, 0, ENEMY, rounds=0, mana=0)
	assert reward(Outcome.WIN, ARMY, 0, ENEMY, rounds=10, mana=0) < base
	assert reward(Outcome.WIN, ARMY, 0, ENEMY, rounds=0, mana=0.5) < base


def test_bounds_cover_every_reward_and_round_score():
	from battle_model.reward import MAX_PENALIZED_ROUNDS, reward_bounds, round_score
	low, high = reward_bounds()
	worst = terminal_reward(Outcome.LOSS, 1, 0, 1, 0, rounds=10 * MAX_PENALIZED_ROUNDS, mana_used_fraction=1.0)
	best = terminal_reward(Outcome.WIN, 1, 1, 1, 1, rounds=0, mana_used_fraction=0.0)
	assert worst == low and best == high
	assert low <= round_score(10, 0, 10, 0, rounds=500, mana_used_fraction=1.0) <= round_score(10, 10, 10, 10, 0, 0) <= high


def test_round_score_is_the_reward_without_the_outcome_term():
	from battle_model.reward import round_score
	args = (ARMY, ARMY * 0.7, ENEMY, ENEMY * 0.4, 3, 0.2)
	assert abs(round_score(*args) - (terminal_reward(Outcome.WIN, *args) - 1.0)) < 1e-9


def test_unit_scale_maps_the_reward_range_to_0_1():
	from battle_model.reward import from_unit, reward_bounds, to_unit
	low, high = reward_bounds()
	assert to_unit(low) == 0.0 and to_unit(high) == 1.0
	assert abs(from_unit(to_unit(1.7)) - 1.7) < 1e-12
