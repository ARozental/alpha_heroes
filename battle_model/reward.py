"""Terminal battle reward and the fixed value functions it uses. No trainable parameters.

	reward = (+1 win | -1 otherwise)
	       + enemy value lost / enemy value at start          (0..1)
	       + own value kept / own value at start              (0..1)
	       - round_penalty * rounds - mana_penalty * fraction of the mana pool used

The same formula scores a battle that is still going on at the end of a round (round_score): the
outcome term is 0 there, since nobody has won yet. The value heads predict these scores.

Values are in gold: gold + resources at VCMI's base prices. A hero's value is its artifacts
plus its experience (1000 for level 1 -> 2, then scaled like the XP needed per level, which
makes it exactly 1 gold per XP point).
"""
from dataclasses import dataclass
from enum import IntEnum

# config/resources.json "price"
RESOURCE_PRICES = {"wood": 250, "mercury": 500, "ore": 250, "sulfur": 500, "crystal": 500, "gems": 500, "gold": 1}
# GameConstants::HERO_GOLD_COST: rehiring a retreated or surrendered hero from the tavern
HERO_HIRE_COST = 2500


class Outcome(IntEnum):
	WIN = 0
	LOSS = 1  # all our units died
	RETREAT = 2  # hero fled: army lost, hero back in the tavern with 1 creature
	SURRENDER = 3  # paid to leave: hero back in the tavern with its army


# The round penalty stops growing after this many rounds, so the reward has fixed bounds
MAX_PENALIZED_ROUNDS = 100


@dataclass(frozen=True)
class RewardConfig:
	round_penalty: float = 0.002
	mana_penalty: float = 0.05


def reward_bounds(config=RewardConfig()):
	"""Lowest and highest possible reward or round score: the output range of the value heads."""
	low = -1.0 - config.round_penalty * MAX_PENALIZED_ROUNDS - config.mana_penalty
	return low, 3.0


def resources_value(resources):
	"""Gold value of a cost dict, e.g. {"gold": 3000, "gems": 1}."""
	return sum(RESOURCE_PRICES[name] * amount for name, amount in resources.items())


def stack_value(unit_cost, count):
	return resources_value(unit_cost) * count


def hero_value(artifact_prices, experience):
	return sum(artifact_prices) + experience


def retreat_value(hero_value_, fresh_hero_value):
	"""A retreated hero is only worth what it adds over hiring a fresh hero (who comes with an army)."""
	return max(0.0, hero_value_ - fresh_hero_value)


def surrender_value(hero_value_, army_value, surrender_cost):
	"""Keep hero and army, minus the surrender payment and the rehire cost.

	Surrender is only legal when gold >= surrender_cost + HERO_HIRE_COST (enforced by the C++ mask),
	so the rehire is always affordable here.
	"""
	return hero_value_ + army_value - surrender_cost - HERO_HIRE_COST


def own_kept_value(outcome, army_end_value, hero_value_, fresh_hero_value=0.0, surrender_cost=0.0):
	"""Value we still own after the battle. army_end_value: our surviving army (at the moment of surrender for SURRENDER)."""
	if outcome == Outcome.WIN:
		return army_end_value + hero_value_
	if outcome == Outcome.RETREAT:
		return retreat_value(hero_value_, fresh_hero_value)
	if outcome == Outcome.SURRENDER:
		return surrender_value(hero_value_, army_end_value, surrender_cost)
	return 0.0


def _score(outcome_term, own_start_value, own_kept, enemy_start_value, enemy_lost, rounds, mana_used_fraction, config):
	def fraction(part, whole):
		return min(1.0, max(0.0, part / whole)) if whole > 0 else 0.0

	score = outcome_term
	score += fraction(enemy_lost, enemy_start_value)
	score += fraction(own_kept, own_start_value)
	score -= config.round_penalty * min(rounds, MAX_PENALIZED_ROUNDS)
	score -= config.mana_penalty * min(1.0, max(0.0, mana_used_fraction))
	return score


def terminal_reward(outcome, own_start_value, own_kept, enemy_start_value, enemy_lost, rounds, mana_used_fraction, config=RewardConfig()):
	"""own_start_value / enemy_start_value: army + hero value at the start of the battle."""
	return _score(1.0 if outcome == Outcome.WIN else -1.0, own_start_value, own_kept, enemy_start_value, enemy_lost, rounds, mana_used_fraction, config)


def round_score(own_start_value, own_kept, enemy_start_value, enemy_lost, rounds, mana_used_fraction, config=RewardConfig()):
	"""The reward formula at the end of a round while the battle goes on: outcome term 0, the rest so far.

	own_kept: our surviving army + our hero's value (the hero is still with us); enemy_lost: enemy army lost so far.
	"""
	return _score(0.0, own_start_value, own_kept, enemy_start_value, enemy_lost, rounds, mana_used_fraction, config)
