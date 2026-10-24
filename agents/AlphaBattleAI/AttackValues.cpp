#include "StdInc.h"
#include "AttackValues.h"

#include "BattleSim.h"
#include "Encoder.h"
#include "Policy.h"

#include "lib/CStack.h"
#include "lib/battle/BattleAction.h"
#include "lib/battle/BattleInfo.h"
#include "lib/battle/CObstacleInstance.h"
#include "lib/battle/ReachabilityInfo.h"
#include "lib/bonuses/BonusEnum.h"
#include "lib/gameState/CGameState.h"

#include <chrono>
#include <map>
#include <queue>
#include <tuple>

using namespace AlphaSchema;

namespace
{
constexpr int MAX_BRANCHES = 8; // dice branches played per attack, the most likely first

/// The armies as the reward counts them, attacker minus defender, in gold (see encodeAttackValues)
double keptValue(const CBattleInfoCallback & battle)
{
	double value = 0;
	for(const auto * stack : battle.battleGetStacksIf([](const CStack *) { return true; }))
	{
		if(!keptAfterBattle(stack))
			continue;
		const int64_t maxHp = std::max<int64_t>(1, stack->getMaxHealth());
		const int64_t health = std::min<int64_t>(stack->getAvailableHealth(), static_cast<int64_t>(remainingAfterBattle(stack)) * maxHp);
		value += (stack->unitSide() == BattleSide::ATTACKER ? 1.0 : -1.0) * static_cast<double>(health) * creatureValue(stack->unitType()) / static_cast<double>(maxHp);
	}
	return value;
}

/// The expected change of keptValue when `action` is played on the battle as it is in `game`, over the dice
/// (BattleSimulation::scriptDice): branch by branch, each roll's other outcome a new branch, the most likely first
double expectedChange(BattleSimulation & sim, const CGameState & game, const BattleID & battleID, PlayerColor player, const BattleAction & action, double before, int & runs)
{
	struct Branch
	{
		double chance;
		std::vector<bool> script;
		bool operator<(const Branch & other) const { return chance < other.chance; }
	};
	std::priority_queue<Branch> open;
	open.push({1.0, {}});
	double total = 0, mass = 0;
	for(int played = 0; played < MAX_BRANCHES && !open.empty(); ++played)
	{
		const auto branch = open.top();
		open.pop();
		sim.restoreBattle(game, battleID);
		sim.scriptDice(branch.script);
		++runs;
		if(!sim.applyActionOnly(player, action))
			throw std::runtime_error("the simulation refused " + action.toString() + ": " + sim.complaint());
		const auto & chances = sim.rollChances();
		const auto & outcomes = sim.rollOutcomes();
		double chance = 1, prefix = 1;
		for(size_t i = 0; i < chances.size(); ++i)
		{
			if(i >= branch.script.size()) // rolled no success here: its success is another branch
			{
				std::vector<bool> script(outcomes.begin(), outcomes.begin() + static_cast<std::ptrdiff_t>(i));
				script.push_back(true);
				open.push({prefix * chances[i], std::move(script)});
			}
			prefix *= outcomes[i] ? chances[i] : 1 - chances[i];
		}
		chance = prefix;
		total += chance * (keptValue(sim.battle()) - before);
		mass += chance;
	}
	sim.scriptDice(std::nullopt);
	return mass > 0 ? total / mass : 0.0;
}

std::vector<uint32_t> ids(const battle::Units & units)
{
	std::vector<uint32_t> result;
	for(const auto * u : units)
		result.push_back(u->unitId());
	std::sort(result.begin(), result.end());
	result.erase(std::unique(result.begin(), result.end()), result.end());
	return result;
}
}

void encodeAttackValues(EncodedBattle & e, const CGameState & game, const BattleID & battleID, PlayerColor player, AttackValueStats & stats)
{
	auto & t = e.tensors;
	if(!t.any(In::m_melee_target) && !t.any(In::m_shoot_target))
		return;
	const auto started = std::chrono::steady_clock::now();
	++stats.calls;
	const CBattleInfoCallback & battle = *game.getBattle(battleID);
	const CStack * active = e.units[e.activeSlot];
	auto unit = [&](int slot) -> const battle::Unit * { return slot >= 0 && slot < MAX_UNITS ? e.units[slot] : nullptr; };
	auto sim = BattleSimulation::copyOf(game, battleID, 0);
	const double before = keptValue(sim->battle());
	auto value = [&](const ChosenAction & a)
	{
		return std::llround(expectedChange(*sim, game, battleID, player, engineAction(active->unitSide(), active, a, unit), before, stats.runs));
	};

	for(int u = 0; u < MAX_UNITS; ++u)
		if(t.get(In::m_shoot_target, u))
		{
			++stats.attacks;
			++stats.played;
			t.set(In::a_shoot_value, u, value(ChosenAction{.type = static_cast<int>(ActionType::SHOOT), .unit = u}));
		}

	// Attacks from different hexes end the same when the same units are hit, both ways, after the same charge (if
	// it counts: jousting), and no obstacle that acts on a unit is on the battlefield (its path could cross it)
	bool triggering = false;
	for(const auto & obstacle : battle.battleGetAllObstacles())
		triggering = triggering || obstacle->triggersEffects();
	const bool jousting = active->hasBonusOfType(BonusType::JOUSTING);
	const auto reach = jousting ? std::optional<ReachabilityInfo>(battle.getReachability(active)) : std::nullopt;
	std::map<std::tuple<int, int, int, std::vector<uint32_t>, std::vector<uint32_t>>, int64_t> outcomes;
	for(int u = 0; u < MAX_UNITS; ++u)
	{
		if(!t.get(In::m_melee_target, u))
			continue;
		const CStack * target = e.units[u];
		for(int h = 0; h < N_HEXES; ++h)
		{
			if(!t.get(In::m_melee_from, u, h))
				continue;
			++stats.attacks;
			const BattleHex from(h);
			battle::Units hit;
			for(const auto & hex : target->getHexes())
				for(const auto * v : battle.getAttackedBattleUnits(active, target, hex, false, from, target->getPosition()))
					hit.push_back(v);
			const auto retaliated = battle.getAttackedBattleUnits(target, active, from, false, target->getPosition(), from);
			const auto key = std::make_tuple(u, jousting ? static_cast<int>(reach->distances[from.toInt()]) : -1, triggering ? h : -1, ids(hit), ids(retaliated));
			auto found = outcomes.find(key);
			if(found == outcomes.end())
			{
				++stats.played;
				found = outcomes.emplace(key, value(ChosenAction{.type = static_cast<int>(ActionType::MELEE), .hex = h, .unit = u})).first;
			}
			t.set(In::a_melee_value, u, h, found->second);
		}
	}
	stats.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}
