#pragma once

#include "Schema.generated.h"

#include <algorithm>
#include <array>
#include <cmath>

/// The terminal reward of a battle for each side, as training computes it (battle_model/reward.py: zero-sum, from each
/// side's assets; battle_model/data.py summarize() for which numbers go in), for the search's terminal states.
/// Plain numbers only, so tools/SearchSelfTest.cpp can check it against Python (tests/test_search.py).
namespace AlphaReward
{
using namespace AlphaSchema;

constexpr int RESULT_NORMAL = 0, RESULT_ESCAPE = 1, RESULT_SURRENDER = 2; // EBattleResult

struct BattleEnd
{
	std::array<double, 2> armyStart{}; // army value of each side at the start of the battle (gold, reward.py)
	std::array<double, 2> armyEnd{}; // what each side keeps of its army (remainingAfterBattle, Encoder.h)
	std::array<int, 2> manaStart{}, manaEnd{};
	std::array<int, 2> power{}; // each side's hero's spell power
	int winner = -1; // battle side, anything else: no winner (draw)
	int result = RESULT_NORMAL;
	int rounds = 0; // VCMI numbering, the first round is 1
	std::array<double, 2> surrenderCost{}; // paid by the side that surrendered
	std::array<double, 2> heroLevel{}; // each side's hero's level and its way to the next (0: no hero)
	std::array<double, 2> artifacts{}; // the price of the artifacts a winner would take from each side's hero
};

/// reward.py hero_level_value
inline double heroLevelValue(double level)
{
	return LEVEL_VALUE * std::pow(std::max(0.0, level - FREE_LEVELS), LEVEL_EXPONENT);
}

/// reward.py mana_value
inline double manaValue(int mana, int power)
{
	return MANA_VALUE * std::max(power, 0) * std::max(mana, 0);
}

/// data.py summarize outcome_of
inline Outcome outcomeOf(const BattleEnd & e, int side)
{
	if(e.winner != 0 && e.winner != 1)
		return Outcome::DRAW;
	if(e.winner == side)
		return Outcome::WIN;
	return e.result == RESULT_ESCAPE ? Outcome::RETREAT : e.result == RESULT_SURRENDER ? Outcome::SURRENDER : Outcome::LOSS;
}

/// reward.py assets_after
inline double assetsAfter(Outcome outcome, double army, double hero, double mana, double surrenderCost, double captured)
{
	switch(outcome)
	{
		case Outcome::WIN: return army + hero + mana + captured;
		case Outcome::RETREAT: return hero + mana;
		case Outcome::SURRENDER: return army + hero + mana - surrenderCost - HERO_HIRE_COST;
		default: return 0.0;
	}
}

/// reward.py zero_sum
inline double zeroSum(double ownStart, double ownEnd, double enemyStart, double enemyEnd)
{
	const double total = ownStart + enemyStart;
	if(total <= 0)
		return 0.0;
	return std::min(1.0, std::max(-1.0, ((enemyStart - enemyEnd) - (ownStart - ownEnd)) / total));
}

/// The reward of side `side` (data.py summarize(events, side=side)), -1..1
inline double reward(const BattleEnd & e, int side)
{
	std::array<double, 2> start{}, end{};
	for(int s = 0; s < 2; ++s)
	{
		const double hero = heroLevelValue(e.heroLevel[s]) + e.artifacts[s];
		const bool takes = outcomeOf(e, s) == Outcome::WIN && outcomeOf(e, 1 - s) == Outcome::LOSS; // the loser's artifacts
		start[s] = e.armyStart[s] + hero + manaValue(e.manaStart[s], e.power[s]);
		end[s] = assetsAfter(outcomeOf(e, s), e.armyEnd[s], hero, manaValue(e.manaEnd[s], e.power[s]), e.surrenderCost[s], takes ? e.artifacts[1 - s] : 0.0);
	}
	return zeroSum(start[side], end[side], start[1 - side], end[1 - side]);
}

/// reward.py to_scale and from_scale: the reward is already on the -1..1 scale of the value heads
inline double toScale(double r)
{
	return r;
}

inline double fromScale(double v)
{
	return v;
}

/// reward.py other_side_value: zero-sum, exactly the negative
inline double otherSideValue(double v)
{
	return -v;
}
}
