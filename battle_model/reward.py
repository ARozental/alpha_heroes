"""The battle reward: zero-sum, from what each side owns. No trainable parameters.

	assets(side) = army      the value of the creatures it keeps after the battle (gold + resources at base prices)
	             + hero      what its hero is worth above a fresh tavern hero, while the side still has its hero
	             + mana      mana_value x the hero's spell power x its mana left, while the side still has its hero
	reward(side) = (assets the enemy lost - assets we lost) / (both sides' assets at the start)      in -1..1

So reward(attacker) = -reward(defender), always: every kill is exactly as good for one side as it is bad for the
other, both sides can learn one value function (V_defender = -V_attacker), and the search knows the other side's
value exactly (other_side_value). There is no separate win term: winning means the enemy loses its army, its hero
and its mana; a costly win can be worth less than a loss that killed more than it lost, which is what a campaign
wants. Losing the hero loses its mana; spending mana in battle costs its value (the mana penalty of before).

What each outcome leaves a side (assets_after):
	WIN        its surviving army, its hero, its mana
	LOSS/DRAW  nothing: army, hero and mana are gone
	RETREAT    its hero (back in the tavern) and its mana; the army is lost
	SURRENDER  its army, hero and mana, minus the surrender payment and the cost of rehiring the hero

The same formula scores a battle still going on at the end of a round (round_score): both heroes still there, the
armies as they stand. The value heads predict these scores.

TODO(non-zero-sum): the zero-sum core treats every loss of one side as the other's gain. These outcomes and actions
really create or destroy value that nobody else gets, so their logits (and their value) should get a per-side
correction, learned by a small separate head, once they matter:
	- SURRENDER: the payment disappears (the server takes it, the winner gets nothing; BattleActionProcessor), and the
	  enemy's surrender cost is unknown to us
	- RETREAT: the army is destroyed without the winner owning it; the hero's rehire is paid to nobody
	- experience: the winner gains experience (levels, skills) for what it kills, which the loser does not pay
	- Necromancy and similar: raising creatures after the battle from the enemy's dead
	- Eagle Eye: learning the enemy's spells; Learning, Scholar-like effects
	- time: a long battle costs both sides nothing here (the round penalty of before is gone), but movement and days
	  matter in a campaign
	- sieges (CATAPULT, not played by the model yet): damaged walls and towers are a cost to the town's owner
	- campaigns with more than two players: a loss to our enemy is not all our gain
"""
from dataclasses import dataclass
from enum import IntEnum

# config/resources.json "price"
RESOURCE_PRICES = {"wood": 250, "mercury": 500, "ore": 250, "sulfur": 500, "crystal": 500, "gems": 500, "gold": 1}
# GameConstants::HERO_GOLD_COST: rehiring a retreated or surrendered hero from the tavern
HERO_HIRE_COST = 2500


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
	mana_value: float = 2.0


def resources_value(resources):
	"""Gold value of a cost dict, e.g. {"gold": 3000, "gems": 1}."""
	return sum(RESOURCE_PRICES[name] * amount for name, amount in resources.items())


def stack_value(unit_cost, count):
	return resources_value(unit_cost) * count


def hero_value(artifact_prices, experience):
	"""A hero's value: its artifacts plus its experience (1000 for level 1 -> 2, then scaled like the XP needed per
	level, which makes it exactly 1 gold per XP point)."""
	return sum(artifact_prices) + experience


def mana_value(mana, power, config=RewardConfig()):
	return config.mana_value * max(power, 0) * max(mana, 0)


def hero_asset(hero_value_, fresh_hero_value=0.0):
	"""What a hero is worth to its side: what it adds over hiring a fresh one (who comes with a small army)."""
	return max(0.0, hero_value_ - fresh_hero_value)


def assets_after(outcome, army_end, hero, mana, surrender_cost=0.0):
	"""What a side owns after the battle (see the top). army_end: its surviving army (at the moment of surrender for
	SURRENDER); hero: hero_asset; mana: mana_value of the mana it has left."""
	if outcome == Outcome.WIN:
		return army_end + hero + mana
	if outcome == Outcome.RETREAT:
		return hero + mana
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
