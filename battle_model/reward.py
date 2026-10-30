"""The battle reward: zero-sum, from what each side owns. No trainable parameters.

	assets(side) = army      the value of the creatures it keeps after the battle (gold + resources at base prices)
	             + hero      what its hero is worth (hero_asset: its levels and its artifacts; its war machines still
	                         standing, at game_values.WAR_MACHINE_CREATURES), while the side has it
	             + mana      mana_value x the hero's spell power x its mana left, while the side still has its hero
	reward(side) = (assets the enemy lost - assets we lost) / (both sides' assets at the start)      in -1..1

So reward(attacker) = -reward(defender), always: every kill is exactly as good for one side as it is bad for the
other, both sides can learn one value function (V_defender = -V_attacker), and the search knows the other side's
value exactly (other_side_value). There is no separate win term: winning means the enemy loses its army, its hero
and its mana; a costly win can be worth less than a loss that killed more than it lost, which is what a campaign
wants. Losing the hero loses its mana; spending mana in battle costs its value (the mana penalty of before).

What each outcome leaves a side (assets_after):
	WIN        its surviving army, its hero, its mana, and the artifacts it takes from a hero it defeated
	LOSS/DRAW  nothing: army, hero and mana are gone (the winner takes the hero's artifacts)
	RETREAT    its hero (back in the tavern) and its mana, minus the cost of rehiring the hero (at least 0); the army
	           is lost
	SURRENDER  its army, hero and mana, minus the surrender payment and the cost of rehiring the hero

The same formula scores a battle still going on at the end of a round (round_score): both heroes still there, the
armies as they stand. The value heads predict these scores.

TODO(non-zero-sum): the zero-sum core treats every loss of one side as the other's gain. These outcomes and actions
really create or destroy value that nobody else gets, so their logits (and their value) should get a per-side
correction, learned by a small separate head, once they matter:
	- SURRENDER: the payment disappears (the server takes it, the winner gets nothing; BattleActionProcessor), and the
	  enemy's surrender cost is unknown to us
	- RETREAT: the army is destroyed without the winner owning it; the hero's rehire is paid to nobody
	- experience: left out on purpose (2026-10-25): the reward stays battle-local and zero-sum, as the campaign-map gains
	  a battle opens (towns, mines, time) are not counted either
	- Necromancy and similar: raising creatures after the battle from the enemy's dead
	- (war machines are counted: half their price as part of their hero, data.summarize; the catapult and arrow towers 0)
	- Eagle Eye: learning the enemy's spells; Learning, Scholar-like effects
	- time: a long battle costs both sides nothing here (the round penalty of before is gone), but movement and days
	  matter in a campaign
	- sieges (the model plays the catapult: ActionType.CATAPULT): damaged walls and towers are a cost to the town's owner
	- campaigns with more than two players: a loss to our enemy is not all our gain
"""
from dataclasses import dataclass
from enum import IntEnum

import game_values
# The prices are the shared list's (game_values), which the campaign model prices with too.
# HERO_HIRE_COST: rehiring a retreated or surrendered hero from the tavern
from game_values import HERO_HIRE_COST, RESOURCE_PRICES, WAR_MACHINE_CREATURES, resources_value  # noqa: F401 (re-exported)


class Outcome(IntEnum):
	WIN = 0
	LOSS = 1  # all our units died, our hero with them
	RETREAT = 2  # hero fled: army lost, hero back in the tavern
	SURRENDER = 3  # paid to leave: hero back in the tavern with its army
	DRAW = 4  # both armies destroyed (e.g. Armageddon): both heroes dead


@dataclass(frozen=True)
class RewardConfig:
	# Gold per mana point per point of spell power. Fixed for now; later from outside (the campaign: a Magic Well
	# nearby makes mana worth ~0). At 2, a power-5 hero's 50 mana is worth 500 gold: ~3% of a week-1 army.
	mana_value: float = game_values.MANA_VALUE
	# A hero's levels (game_values.hero_level_value): the level-up from level L worth level_value x min(L, level_steps),
	# 2000, 4000, then 6000 each: level 2 2000, 3 6000, 4 12000, 10 48000, 20 108000
	level_value: float = game_values.LEVEL_VALUE
	level_steps: int = game_values.LEVEL_STEPS


def stack_value(unit_cost, count):
	return resources_value(unit_cost) * count


def mana_value(mana, power, config=RewardConfig()):
	return config.mana_value * max(power, 0) * max(mana, 0)


def hero_level_value(level, config=RewardConfig()):
	"""What a hero's levels are worth (RewardConfig): level counts its way to the next level (12.5: halfway to 13)."""
	return game_values.hero_level_value(level, config.level_value, config.level_steps)


def hero_asset(level, artifacts, config=RewardConfig()):
	"""What a hero is worth to its side: its levels, and its artifacts at their market price (artifacts: the price of
	those a winner would take). Primary skills and spells are not counted on their own (the level stands for them)."""
	return hero_level_value(level, config) + artifacts


def assets_after(outcome, army_end, hero, mana, surrender_cost=0.0, captured=0.0):
	"""What a side owns after the battle (see the top). army_end: its surviving army (at the moment of surrender for
	SURRENDER); hero: hero_asset; mana: mana_value of the mana it has left; captured: for a WIN over a hero, the
	price of that hero's artifacts, which the winner takes."""
	if outcome == Outcome.WIN:
		return army_end + hero + mana + captured
	# A hero worth less than its rehire is just not rehired: never below losing it. TODO: an approximation; an unwanted
	# retreated hero can still cost more (it takes the tavern place of a better hero), so the true value can be lower
	if outcome == Outcome.RETREAT:
		return max(0.0, hero + mana - HERO_HIRE_COST)
	if outcome == Outcome.SURRENDER:
		return army_end + hero + mana - surrender_cost - HERO_HIRE_COST
	return 0.0


def zero_sum(own_start, own_end, enemy_start, enemy_end):
	"""(assets the enemy lost - assets we lost) / (both sides' assets at the start), in -1..1."""
	total = own_start + enemy_start
	if total <= 0:
		return 0.0
	return min(1.0, max(-1.0, ((enemy_start - enemy_end) - (own_start - own_end)) / total))


def terminal_reward(own_start, own_end, enemy_start, enemy_end):
	"""The reward at the end of the battle; *_start: assets at the start, *_end: assets_after."""
	return zero_sum(own_start, own_end, enemy_start, enemy_end)


def round_score(own_start, own_now, enemy_start, enemy_now):
	"""The reward if the battle were scored at the end of this round: each side's army as it stands, both heroes still
	there with their mana now."""
	return zero_sum(own_start, own_now, enemy_start, enemy_now)


def to_scale(reward, config=None):
	"""The reward is already on the -1..1 scale of the value heads (kept for the places that convert)."""
	return reward


def from_scale(value, config=None):
	return value


def other_side_value(value):
	"""The other side's value from one side's: zero-sum, exactly its negative."""
	return -value
