#include "StdInc.h"
#include "BattleSearch.h"

#include "Encoder.h"
#include "Model.h"
#include "Policy.h"

#include "lib/CCreatureHandler.h"
#include "lib/CStack.h"
#include "lib/battle/BattleAction.h"
#include "lib/battle/CPlayerBattleCallback.h"
#include "lib/bonuses/BonusEnum.h"
#include "lib/mapObjects/CGHeroInstance.h"

#include <chrono>

using namespace AlphaSchema;

namespace
{
double msSince(std::chrono::steady_clock::time_point start)
{
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// An action id: the type in bits 0-3, then three 20-bit fields, each value + 1 (0: not used).
// MOVE: hex. MELEE: target unit id, hex. SHOOT: target unit id. HERO_SPELL: spell, first and second target,
// a target being a unit id or HEX_TARGET + hex.
constexpr int64_t FIELD_BITS = 20;
constexpr int64_t FIELD_MASK = (1 << FIELD_BITS) - 1;
constexpr int HEX_TARGET = 1 << 18;

int64_t pack(int type, int f1 = -1, int f2 = -1, int f3 = -1)
{
	for(int f : {f1, f2, f3})
		if(f + 1 > FIELD_MASK)
			throw std::runtime_error("action field out of range: " + std::to_string(f));
	return type | static_cast<int64_t>(f1 + 1) << 4 | static_cast<int64_t>(f2 + 1) << (4 + FIELD_BITS) | static_cast<int64_t>(f3 + 1) << (4 + 2 * FIELD_BITS);
}

int field(int64_t id, int i)
{
	return static_cast<int>((id >> (4 + i * FIELD_BITS)) & FIELD_MASK) - 1;
}

/// The action of an id, its unit fields holding unit ids
ChosenAction unpack(int64_t id)
{
	ChosenAction a;
	a.type = static_cast<int>(id & 15);
	auto target = [&](int value, int & unit, int & hex)
	{
		if(value >= HEX_TARGET)
			hex = value - HEX_TARGET;
		else if(value >= 0)
			unit = value;
	};
	switch(static_cast<ActionType>(a.type))
	{
		case ActionType::MOVE:
			a.hex = field(id, 0);
			break;
		case ActionType::MELEE:
			a.unit = field(id, 0);
			a.hex = field(id, 1);
			break;
		case ActionType::SHOOT:
			a.unit = field(id, 0);
			break;
		case ActionType::HERO_SPELL:
			a.spell = field(id, 0);
			target(field(id, 1), a.targetUnit, a.targetHex);
			target(field(id, 2), a.target2Unit, a.target2Hex);
			break;
		default:
			break;
	}
	return a;
}

double creatureValue(const CCreature * creature)
{
	double value = 0;
	for(int r = 0; r < static_cast<int>(RESOURCE_PRICES.size()); ++r)
		value += RESOURCE_PRICES[r] * creature->getRecruitCost(GameResID(r));
	return value;
}
}

double armyValue(const CCreatureSet & army)
{
	double value = 0;
	for(const auto & [slot, stack] : army.Slots())
		value += creatureValue(stack->getCreatureID().toCreature()) * stack->getCount();
	return value;
}

double aliveArmyValue(const CBattleInfoCallback & battle, BattleSide side)
{
	double value = 0;
	for(const auto * s : battle.battleGetAllStacks(true))
		if(s->unitSide() == side && s->alive() && !s->isClone() && !s->summoned && s->unitSlot().validSlot())
			value += creatureValue(s->unitType()) * s->getCount();
	return value;
}

BattleSearchEnv::BattleSearchEnv(const BattleSimulation & root, const BattleModel & model, const Environment & env, const AlphaReward::BattleEnd & start, int roundCap)
	: rootSim(root)
	, model(model)
	, env(env)
	, start(start)
	, roundCap(roundCap)
{
}

int64_t BattleSearchEnv::actionId(const EncodedBattle & e, const ChosenAction & a)
{
	auto unit = [&](int slot) { return static_cast<int>(e.units.at(slot)->unitId()); };
	switch(static_cast<ActionType>(a.type))
	{
		case ActionType::MOVE:
			return pack(a.type, a.hex);
		case ActionType::MELEE:
			return pack(a.type, unit(a.unit), a.hex);
		case ActionType::SHOOT:
			return pack(a.type, unit(a.unit));
		case ActionType::HERO_SPELL:
		{
			auto target = [&](int slot, int hex) { return slot >= 0 ? unit(slot) : hex >= 0 ? HEX_TARGET + hex : -1; };
			return pack(a.type, a.spell, target(a.targetUnit, a.targetHex), target(a.target2Unit, a.target2Hex));
		}
		default:
			return pack(a.type);
	}
}

BattleSearchEnv::State BattleSearchEnv::root(uint64_t seed)
{
	const auto started = std::chrono::steady_clock::now();
	State s;
	s.sim = rootSim.clone(static_cast<int>(seed & 0x7fffffff));
	copyMs += msSince(started);
	return s;
}

bool BattleSearchEnv::terminal(const State & s) const
{
	return s.sim->finished();
}

std::array<double, 2> BattleSearchEnv::rewards(State & s)
{
	const auto & battle = s.sim->battle();
	auto end = start;
	for(int side = 0; side < 2; ++side)
	{
		end.armyEnd[side] = aliveArmyValue(battle, static_cast<BattleSide>(side));
		const auto * hero = battle.battleGetFightingHero(static_cast<BattleSide>(side));
		end.manaEnd[side] = hero ? hero->mana : 0;
	}
	end.winner = s.sim->winner() ? static_cast<int>(*s.sim->winner()) : -1;
	end.result = s.sim->result();
	end.rounds = battle.battleGetRound();
	end.surrenderCost = s.surrenderCost;
	return {AlphaReward::toScale(AlphaReward::reward(end, 0)), AlphaReward::toScale(AlphaReward::reward(end, 1))};
}

const CStack * BattleSearchEnv::active(const State & s) const
{
	const auto & battle = s.sim->battle();
	const auto * unit = battle.battleActiveUnit();
	const auto * stack = unit ? battle.battleGetStackByID(unit->unitId(), false) : nullptr;
	if(!stack)
		throw std::runtime_error("search: no active unit in an unfinished battle");
	return stack;
}

int BattleSearchEnv::mover(const State & s) const
{
	const auto & battle = s.sim->battle();
	return static_cast<int>(battle.playerToSide(battle.battleGetOwner(active(s))));
}

int64_t BattleSearchEnv::key(const State & s) const
{
	return static_cast<int64_t>(active(s)->unitId()) * 2 + mover(s);
}

EncodedBattle BattleSearchEnv::encode(const State & s)
{
	const auto started = std::chrono::steady_clock::now();
	++encodings;
	const auto & battle = s.sim->battle();
	const auto * stack = active(s);
	const auto player = battle.battleGetOwner(stack);
	const CPlayerBattleCallback view(battle.getBattle(), player);
	const auto side = battle.playerToSide(player);
	auto encoded = encodeBattle(view, stack, env, player, battle.battleGetFightingHero(side == BattleSide::ATTACKER ? BattleSide::DEFENDER : BattleSide::ATTACKER));
	encodeMs += msSince(started);
	return encoded;
}

GumbelSearch::Evaluation BattleSearchEnv::evaluate(State & s)
{
	const auto encoded = encode(s);
	const auto started = std::chrono::steady_clock::now();
	const auto out = model.run(encoded.tensors);
	modelMs += msSince(started);
	++evaluations;
	GumbelSearch::Evaluation e;
	const int side = mover(s); // both sides play by our network in the tree: vs BattleAI only an approximation, which only matters in eval; never train against AIs the tree can't model
	e.values[side] = out.data(Out::value).back(); // V, the final-reward head
	e.values[1 - side] = AlphaReward::otherSideValue(e.values[side]); // the model predicts only its own side's (reward.py other_side_value)
	if(s.sim->battle().battleGetRound() >= roundCap)
		return e;
	for(const auto & a : legalActions(encoded))
	{
		e.actions.push_back(actionId(encoded, a));
		e.logits.push_back(logProb(out, a));
	}
	return e;
}

std::vector<int64_t> BattleSearchEnv::legal(State & s)
{
	const auto encoded = encode(s);
	std::vector<int64_t> ids;
	for(const auto & a : legalActions(encoded))
		ids.push_back(actionId(encoded, a));
	return ids;
}

void BattleSearchEnv::apply(State & s, int64_t id)
{
	const auto started = std::chrono::steady_clock::now();
	const auto & battle = s.sim->battle();
	const auto * stack = active(s);
	const auto player = battle.battleGetOwner(stack);
	const auto side = battle.playerToSide(player);
	const auto a = unpack(id);
	const auto action = engineAction(side, stack, a, [&](int unitId) { return battle.battleGetUnitByID(unitId); });
	if(a.type == static_cast<int>(ActionType::SURRENDER))
		s.surrenderCost[static_cast<int>(side)] = battle.battleGetSurrenderCost(player);
	if(!s.sim->apply(player, action))
		throw std::runtime_error("search: the simulation refused " + action.toString() + ": " + s.sim->complaint());
	playNonDecisions(s);
	applyMs += msSince(started);
}

void BattleSearchEnv::playNonDecisions(State & s)
{
	for(int guard = 0; guard < 100 && !s.sim->finished(); ++guard)
	{
		const auto * stack = active(s);
		if(!stack->hasBonusOfType(BonusType::SIEGE_WEAPON))
			return;
		const auto & battle = s.sim->battle();
		if(!s.sim->apply(battle.battleGetOwner(stack), BattleAction::makeDefend(stack)))
			throw std::runtime_error("search: the simulation refused a war machine's defend: " + s.sim->complaint());
	}
}
