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

/// The living units among `units` by who controls them: (enemies of player, player's own but `self`)
std::pair<int, int> sides(const CBattleInfoCallback & battle, const std::vector<uint32_t> & units, PlayerColor player, uint32_t self)
{
	int enemies = 0, allies = 0;
	for(const auto id : units)
		if(const auto * unit = battle.battleGetUnitByID(id); unit && unit->alive() && id != self)
			++(battle.battleGetOwner(unit) == player ? allies : enemies);
	return {enemies, allies};
}

/// The units an attack hits, as the server's attack finds them (BattleActionProcessor::makeAttack:
/// getAttackedCreatures, aimed at the target's hex, from attackerPos), the target too; sorted ids
std::vector<uint32_t> attacked(const CBattleInfoCallback & battle, const CStack * attacker, const CStack * target, bool shooting, BattleHex attackerPos)
{
	std::vector<uint32_t> result = {target->unitId()};
	for(const auto * unit : battle.getAttackedCreatures(attacker, target->getPosition(), shooting, attackerPos).first)
		result.push_back(unit->unitId());
	std::sort(result.begin(), result.end());
	result.erase(std::unique(result.begin(), result.end()), result.end());
	return result;
}
}

AttackValueCache::AttackValueCache() = default;
AttackValueCache::~AttackValueCache() = default;

void AttackValueCache::clear()
{
	sim.reset();
	game = nullptr;
	battle = -1;
}

namespace
{
/// encodeAttackValues' work, the attacks played on sim (a copy of battle battleID of game, any earlier state of it)
void valueAttacks(EncodedBattle & e, BattleSimulation & sim, const CGameState & game, const BattleID & battleID, PlayerColor player, AttackValueStats & stats)
{
	auto & t = e.tensors;
	const CBattleInfoCallback & battle = *game.getBattle(battleID);
	const CStack * active = e.units[e.activeSlot];
	auto unit = [&](int slot) -> const battle::Unit * { return slot >= 0 && slot < MAX_UNITS ? e.units[slot] : nullptr; };
	sim.resetScriptedDice();
	sim.restoreBattle(game, battleID);
	const double before = keptValue(sim.battle());
	auto value = [&](const ChosenAction & a)
	{
		return std::llround(expectedChange(sim, game, battleID, player, engineAction(active->unitSide(), active, a, unit), before, stats.runs));
	};

	for(int u = 0; u < MAX_UNITS; ++u)
		if(t.get(In::m_shoot_target, u))
		{
			++stats.attacks;
			++stats.played;
			t.set(In::a_shoot_value, u, value(ChosenAction{.type = static_cast<int>(ActionType::SHOOT), .unit = u}));
			const auto [enemies, allies] = sides(battle, attacked(battle, active, e.units[u], true, active->getPosition()), player, active->unitId());
			t.set(In::a_shoot_enemies, u, enemies);
			t.set(In::a_shoot_allies, u, allies);
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
			const auto hit = attacked(battle, active, target, false, from);
			const auto [enemies, allies] = sides(battle, hit, player, active->unitId());
			t.set(In::a_melee_enemies, u, h, enemies);
			t.set(In::a_melee_allies, u, h, allies);
			// the retaliation, aimed at the attacker where it attacks from (its area attack can hit others there)
			std::vector<uint32_t> retaliated;
			for(const auto * unit : battle.getAttackedCreatures(target, from, false, target->getPosition()).first)
				retaliated.push_back(unit->unitId());
			std::sort(retaliated.begin(), retaliated.end());
			const auto key = std::make_tuple(u, jousting ? static_cast<int>(reach->distances[from.toInt()]) : -1, triggering ? h : -1, hit, retaliated);
			auto found = outcomes.find(key);
			if(found == outcomes.end())
			{
				++stats.played;
				found = outcomes.emplace(key, value(ChosenAction{.type = static_cast<int>(ActionType::MELEE), .hex = h, .unit = u})).first;
			}
			t.set(In::a_melee_value, u, h, found->second);
		}
	}
}
}

void encodeAttackValues(EncodedBattle & e, const CGameState & game, const BattleID & battleID, PlayerColor player, AttackValueStats & stats, AttackValueCache * cache)
{
	auto & t = e.tensors;
	if(e.mode != EncodedBattle::Mode::NORMAL || (!t.any(In::m_melee_target) && !t.any(In::m_shoot_target)))
		return; // the special decisions (tactics, catapult, heal) have no attacks
	const auto started = std::chrono::steady_clock::now();
	++stats.calls;
	if(!cache)
	{
		auto sim = BattleSimulation::copyOf(game, battleID, 0);
		valueAttacks(e, *sim, game, battleID, player, stats);
	}
	else
	{
		if(!cache->sim || cache->game != &game || cache->battle != battleID.getNum())
		{
			cache->sim = BattleSimulation::copyOf(game, battleID, 0);
			cache->game = &game;
			cache->battle = battleID.getNum();
		}
		std::optional<EncodedBattle> fresh;
		if(cache->check)
			fresh = e;
		if(cache->freshCheck) // (ALPHA_BATTLE_ATTACK_CHECK=2: two new copies compared, the kept one unused)
			valueAttacks(e, *BattleSimulation::copyOf(game, battleID, 0), game, battleID, player, stats);
		else
			valueAttacks(e, *cache->sim, game, battleID, player, stats);
		if(fresh)
		{
			AttackValueStats unused;
			auto sim = BattleSimulation::copyOf(game, battleID, 0);
			valueAttacks(*fresh, *sim, game, battleID, player, unused);
			std::string differences;
			for(int u = 0; u < MAX_UNITS; ++u)
			{
				if(t.get(In::a_shoot_value, u) != fresh->tensors.get(In::a_shoot_value, u))
					differences += " shoot " + std::to_string(u) + ": " + std::to_string(t.get(In::a_shoot_value, u)) + " vs " + std::to_string(fresh->tensors.get(In::a_shoot_value, u)) + ";";
				for(int h = 0; h < N_HEXES; ++h)
					if(t.get(In::a_melee_value, u, h) != fresh->tensors.get(In::a_melee_value, u, h))
						differences += " melee " + std::to_string(u) + " from " + std::to_string(h) + ": " + std::to_string(t.get(In::a_melee_value, u, h)) + " vs " + std::to_string(fresh->tensors.get(In::a_melee_value, u, h)) + ";";
			}
			++cache->checked;
			if(!differences.empty())
			{
				++cache->mismatches;
				const CStack * active = e.units[e.activeSlot];
				logAi->error("AlphaBattleAI: ATTACK CHECK: %s valued attacks unlike a new copy (round %d, %s of side %d):%s", cache->freshCheck ? "a new copy" : "the kept copy", game.getBattle(battleID)->battleGetRound(), active->unitType()->getJsonKey(), static_cast<int>(active->unitSide()), differences);
			}
		}
	}
	stats.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
}
