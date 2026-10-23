#pragma once

#include "Schema.generated.h"

#include <algorithm>
#include <array>

/// The terminal reward of a battle for each side, as training computes it (battle_model/reward.py, and
/// battle_model/data.py summarize() for which numbers go in), for the search's terminal states.
/// Plain numbers only, so tools/SearchSelfTest.cpp can check it against Python (tests/test_search.py).
/// Hero values are 0, as in summarize(): not recorded yet, and the scenarios have level-1 heroes without artifacts.
namespace AlphaReward
{
using namespace AlphaSchema;

constexpr int RESULT_NORMAL = 0, RESULT_ESCAPE = 1, RESULT_SURRENDER = 2; // EBattleResult

struct BattleEnd
{
	std::array<double, 2> armyStart{}; // army value of each side at the start of the battle (gold, reward.py)
	std::array<double, 2> armyEnd{}; // army units alive at the end, summoned units and clones excluded
	std::array<int, 2> manaStart{}, manaMax{}, manaEnd{};
	int winner = -1; // battle side, anything else: no winner (draw)
	int result = RESULT_NORMAL;
	int rounds = 0; // VCMI numbering, the first round is 1
	std::array<double, 2> surrenderCost{}; // paid by the side that surrendered
};

/// reward.py _score
inline double score(double outcomeTerm, double ownStart, double ownKept, double enemyStart, double enemyLost, int rounds, double manaUsedFraction, bool heroAlive)
{
	auto fraction = [](double part, double whole) { return whole > 0 ? std::min(1.0, std::max(0.0, part / whole)) : 0.0; };
	double s = outcomeTerm;
	s += fraction(enemyLost, enemyStart);
	s += fraction(ownKept, ownStart);
	s -= ROUND_PENALTY * std::min(rounds, MAX_PENALIZED_ROUNDS);
	if(heroAlive)
		s -= MANA_PENALTY * std::min(1.0, std::max(0.0, manaUsedFraction));
	return s;
}

/// The reward of side `side` (data.py summarize(events, side=side)), in reward units
inline double reward(const BattleEnd & e, int side)
{
	const int enemy = 1 - side;
	Outcome outcome = Outcome::LOSS;
	if(e.winner == side)
		outcome = Outcome::WIN;
	else if(e.winner != 0 && e.winner != 1)
		outcome = Outcome::DRAW;
	else if(e.result == RESULT_ESCAPE)
		outcome = Outcome::RETREAT;
	else if(e.result == RESULT_SURRENDER)
		outcome = Outcome::SURRENDER;

	// An enemy that retreated or surrendered counts as fully destroyed (see summarize)
	const double enemyLost = outcome == Outcome::WIN && (e.result == RESULT_ESCAPE || e.result == RESULT_SURRENDER) ? e.armyStart[enemy] : e.armyStart[enemy] - e.armyEnd[enemy];
	double kept = 0.0; // reward.py own_kept_value with hero values 0
	if(outcome == Outcome::WIN)
		kept = e.armyEnd[side] + 0.0;
	else if(outcome == Outcome::RETREAT)
		kept = std::max(0.0, 0.0 - 0.0);
	else if(outcome == Outcome::SURRENDER)
		kept = 0.0 + e.armyEnd[side] - e.surrenderCost[side] - HERO_HIRE_COST;
	const double manaUsed = e.manaMax[side] > 0 ? static_cast<double>(e.manaStart[side] - e.manaEnd[side]) / e.manaMax[side] : 0.0;
	const int o = static_cast<int>(outcome);
	return score(OUTCOME_TERM[o], e.armyStart[side] + 0.0, kept, e.armyStart[enemy], enemyLost, e.rounds, manaUsed, HERO_SURVIVES[o]);
}

/// reward.py to_scale: reward units -> -1..1, the scale of the value heads
inline double toScale(double r)
{
	const double low = -1.0 - ROUND_PENALTY * MAX_PENALIZED_ROUNDS - MANA_PENALTY;
	const double high = 3.0;
	return 2 * (r - low) / (high - low) - 1;
}

/// reward.py from_scale: -1..1 -> reward units
inline double fromScale(double v)
{
	const double low = -1.0 - ROUND_PENALTY * MAX_PENALIZED_ROUNDS - MANA_PENALTY;
	const double high = 3.0;
	return low + (high - low) * (v + 1) / 2;
}

/// reward.py other_side_value: the other side's expected final reward from one side's, both -1..1
inline double otherSideValue(double v)
{
	return toScale(2.0 - fromScale(v));
}
}
