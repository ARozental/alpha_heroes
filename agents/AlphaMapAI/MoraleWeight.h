// alpha_heroes patch (the fresh run, the user 2026-11-18: "a 4% chance of getting an extra turn is not 1.04 times
// stronger ... make that half this effect"): a stack's power at its morale as our best army (armies::takeBest, TAKE_BEST,
// TAKE_BEST_FAST and their x_take_* inputs) weighs it: 1 + weight x (its morale's battle chance), both signs; weight 1:
// Nullkiller2's ArmyManager::getBestArmy's weighting exactly (x 1.0 is exact), 0.5 the fresh run's. No engine headers:
// tools/MoraleSelfTest.cpp drives it for campaign_model/tests/test_morale_weight.py.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace armies
{
/// The multiplier of a stack's power at `morale` (VCMI's COMBAT_BAD/GOOD_MORALE_CHANCE tables over COMBAT_MORALE_DICE_SIZE).
inline double moraleMultiplier(int morale, const std::vector<int> & bad, const std::vector<int> & good, int dice, double weight)
{
	double multiplier = 1.0;
	if(morale < 0 && !bad.empty())
		multiplier -= weight * (1.0 / dice * bad.at(std::min<size_t>(bad.size(), -morale) - 1));
	else if(morale > 0 && !good.empty())
		multiplier += weight * (1.0 / dice * good.at(std::min<size_t>(good.size(), morale) - 1));
	return multiplier;
}

/// A stack's weighed power, as moraleWeighted sums it.
inline uint64_t weighedPower(uint64_t power, int morale, const std::vector<int> & bad, const std::vector<int> & good, int dice, double weight)
{
	return static_cast<uint64_t>(moraleMultiplier(morale, bad, good, dice, weight) * static_cast<double>(power));
}
}
