#include "StdInc.h"
#include "Armies.h"

#include "lib/CCreatureHandler.h"
#include "lib/GameConstants.h"
#include "lib/IGameSettings.h"
#include "lib/mapObjects/army/CArmedInstance.h"
#include "lib/mapObjects/army/CStackInstance.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>

namespace armies
{
namespace
{
// An army that is no object, for VCMI's morale of its make-up (Nullkiller2's TemporaryArmy, ArmyManager.cpp)
class ScratchArmy : public CArmedInstance
{
public:
	void armyChanged() override {}
	ScratchArmy()
		: CArmedInstance(nullptr, BonusNodeType::UNKNOWN, true)
	{
	}
};

struct Stack
{
	CreatureID creature;
	int count = 0;
	uint64_t power = 0;
};

// Both armies' creatures, each creature's stacks together, the most power first (ArmyManager::getSortedSlots)
std::vector<Stack> sortedStacks(const CCreatureSet & one, const CCreatureSet & other)
{
	std::map<CreatureID, Stack> merged;
	for(const auto * army : {&one, &other})
		for(const auto & [slot, stack] : army->Slots())
		{
			auto & s = merged[stack->getCreatureID()];
			s.creature = stack->getCreatureID();
			s.count += stack->getCount();
			s.power += stack->getPower();
		}
	std::vector<Stack> sorted;
	for(const auto & [id, s] : merged)
		sorted.push_back(s);
	std::stable_sort(sorted.begin(), sorted.end(), [](const Stack & a, const Stack & b) { return a.power > b.power; });
	return sorted;
}

// An army's power as Nullkiller2 weighs it: each stack's at its morale's battle chance (ArmyManager::getBestArmy)
uint64_t moraleWeighted(const CArmedInstance & army, const IGameSettings & settings)
{
	const auto & bad = settings.getVector(EGameSettings::COMBAT_BAD_MORALE_CHANCE);
	const auto & good = settings.getVector(EGameSettings::COMBAT_GOOD_MORALE_CHANCE);
	const int dice = settings.getInteger(EGameSettings::COMBAT_MORALE_DICE_SIZE);
	uint64_t value = 0;
	for(const auto & [slot, stack] : army.Slots())
	{
		const int morale = stack->moraleVal();
		double multiplier = 1.0;
		if(morale < 0 && !bad.empty())
			multiplier -= 1.0 / dice * bad.at(std::min<size_t>(bad.size(), -morale) - 1);
		else if(morale > 0 && !good.empty())
			multiplier += 1.0 / dice * good.at(std::min<size_t>(good.size(), morale) - 1);
		value += static_cast<uint64_t>(multiplier * static_cast<double>(stack->getPower()));
	}
	return value;
}

std::map<CreatureID, int> counted(const Army & army)
{
	std::map<CreatureID, int> counts;
	for(const auto & [creature, count] : army)
		counts[creature] += count;
	return counts;
}
}

std::optional<Take> takeBest(const CArmedInstance & receiver, const CArmedInstance & giver, const IGameSettings & settings, bool fast)
{
	// Nullkiller2's best army of the two (ArmyManager::getBestArmy): the factions added, the strongest first, while the
	// army's power at its morale grows; of them the most powerful creatures, as many as its slots take
	const auto sorted = sortedStacks(receiver, giver);
	std::map<FactionID, uint64_t> factions;
	for(const auto & s : sorted)
		factions[s.creature.toCreature()->getFactionID()] += s.power;
	std::set<FactionID> allowed;
	Army best;
	uint64_t bestValue = 0;
	ScratchArmy scratch;
	while(allowed.size() < factions.size())
	{
		std::optional<FactionID> strongest;
		uint64_t most = 0;
		for(const auto & [faction, power] : factions)
			if(!allowed.count(faction) && (!strongest || power > most))
				std::tie(strongest, most) = std::tuple(faction, power);
		allowed.insert(*strongest);
		Army candidate;
		scratch.clearSlots();
		for(const auto & s : sorted)
			if(allowed.count(s.creature.toCreature()->getFactionID()))
				if(const auto slot = scratch.getSlotFor(s.creature); slot.validSlot())
				{
					scratch.setCreature(slot, s.creature, s.count);
					candidate.emplace_back(s.creature, s.count);
				}
		scratch.updateMoraleBonusFromArmy();
		const uint64_t value = moraleWeighted(scratch, settings);
		if(bestValue >= value)
			break;
		best = candidate;
		bestValue = value;
	}
	if(fast) // (TAKE_BEST_FAST: less the slowest creatures, when there is a faster one)
	{
		int slowest = std::numeric_limits<int>::max();
		int fastest = std::numeric_limits<int>::min();
		for(const auto & [creature, count] : best)
		{
			slowest = std::min(slowest, creature.toCreature()->getBaseSpeed());
			fastest = std::max(fastest, creature.toCreature()->getBaseSpeed());
		}
		if(best.empty() || slowest == fastest)
			return std::nullopt;
		best.erase(std::remove_if(best.begin(), best.end(), [&](const auto & stack) { return stack.first.toCreature()->getBaseSpeed() == slowest; }), best.end());
	}
	const auto taken = counted(best);
	Army rest;
	for(const auto & s : sorted)
		if(!taken.count(s.creature))
			rest.emplace_back(s.creature, s.count);
	if(rest.empty()) // (a hero keeps a creature: the giver one of its own, the least worth a creature, as Nullkiller2's scout unit is weak)
	{
		std::optional<CreatureID> keep;
		int least = std::numeric_limits<int>::max();
		for(const auto & [slot, stack] : giver.Slots())
			if(const int worth = stack->getCreature()->getAIValue(); worth < least)
				std::tie(keep, least) = std::tuple(stack->getCreatureID(), worth);
		if(!keep)
			return std::nullopt;
		for(auto & [creature, count] : best)
			if(creature == *keep)
				--count;
		best.erase(std::remove_if(best.begin(), best.end(), [](const auto & stack) { return stack.second <= 0; }), best.end());
		rest.emplace_back(*keep, 1);
	}
	Army now;
	for(const auto & [slot, stack] : receiver.Slots())
		now.emplace_back(stack->getCreatureID(), stack->getCount());
	if(best.empty() || rest.size() > GameConstants::ARMY_SIZE || counted(now) == counted(best))
		return std::nullopt;
	return Take{best, rest};
}

std::vector<SlotID> slowestStacks(const CCreatureSet & army)
{
	int slowest = std::numeric_limits<int>::max();
	int fastest = std::numeric_limits<int>::min();
	for(const auto & [slot, stack] : army.Slots())
	{
		slowest = std::min(slowest, stack->getCreature()->getBaseSpeed());
		fastest = std::max(fastest, stack->getCreature()->getBaseSpeed());
	}
	std::vector<SlotID> stacks;
	if(slowest < fastest)
		for(const auto & [slot, stack] : army.Slots())
			if(stack->getCreature()->getBaseSpeed() == slowest)
				stacks.push_back(slot);
	return stacks;
}

std::vector<SlotID> foreignStacks(const CCreatureSet & army)
{
	std::map<FactionID, uint64_t> power;
	for(const auto & [slot, stack] : army.Slots())
		power[stack->getCreature()->getFactionID()] += stack->getPower();
	std::vector<SlotID> stacks;
	if(power.size() < 2)
		return stacks;
	const auto main = std::max_element(power.begin(), power.end(), [](const auto & a, const auto & b) { return a.second < b.second; })->first;
	for(const auto & [slot, stack] : army.Slots())
		if(stack->getCreature()->getFactionID() != main)
			stacks.push_back(slot);
	return stacks;
}

bool takes(const CCreatureSet & into, const CCreatureSet & from, const std::vector<SlotID> & stacks)
{
	std::set<CreatureID> needing; // (creatures with no stack of theirs there: a free slot each)
	for(const auto & slot : stacks)
	{
		const auto * stack = from.getStackPtr(slot);
		if(!stack)
			return false;
		bool there = false;
		for(const auto & [s, other] : into.Slots())
			there = there || other->getCreatureID() == stack->getCreatureID();
		if(!there)
			needing.insert(stack->getCreatureID());
	}
	int free = 0;
	for(int s = 0; s < GameConstants::ARMY_SIZE; ++s)
		free += into.slotEmpty(SlotID(s)) ? 1 : 0;
	return static_cast<int>(needing.size()) <= free;
}

SlotID singleSource(const CCreatureSet & army)
{
	if(!army.getFreeSlot().validSlot())
		return SlotID();
	SlotID best;
	int least = std::numeric_limits<int>::max();
	for(const auto & [slot, stack] : army.Slots()) // (in slot order: the first of the least)
		if(stack->getCount() > 1 && stack->getCreature()->getAIValue() < least)
			std::tie(best, least) = std::tuple(slot, stack->getCreature()->getAIValue());
	return best;
}

std::pair<SlotID, SlotID> duplicate(const CCreatureSet & army)
{
	for(int later = 1; later < GameConstants::ARMY_SIZE; ++later)
		for(int earlier = 0; earlier < later; ++earlier)
		{
			const auto * one = army.getStackPtr(SlotID(earlier));
			const auto * other = army.getStackPtr(SlotID(later));
			if(one && other && one->getId() == other->getId())
				return {SlotID(later), SlotID(earlier)};
		}
	return {SlotID(), SlotID()};
}
}
