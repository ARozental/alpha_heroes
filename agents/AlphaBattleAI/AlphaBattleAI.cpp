#include "StdInc.h"
#include "AlphaBattleAI.h"

#include "Dump.h"
#include "Encoder.h"
#include "Model.h"
#include "Policy.h"

#include "lib/CStack.h"
#include "lib/battle/BattleAction.h"
#include "lib/battle/CPlayerBattleCallback.h"
#include "lib/bonuses/BonusEnum.h"
#include "lib/callback/CBattleCallback.h"
#include "lib/callback/CDynLibHandler.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <unistd.h>

using namespace AlphaSchema;

namespace
{
constexpr int ROUND_CAP = 30;

std::string envString(const char * name)
{
	const char * value = std::getenv(name);
	return value ? std::string(value) : std::string();
}

std::atomic<int> dumpedDecisions{0};
}

AlphaBattleAI::~AlphaBattleAI()
{
	if(cb)
		cb->waitTillRealize = wasWaitingForRealize;
}

void AlphaBattleAI::initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB)
{
	env = ENV;
	cb = CB;
	// Save before the fallback touches the callback, both restore the same original value
	wasWaitingForRealize = CB->waitTillRealize;
	fallback = CDynLibHandler::getNewBattleAI("StupidAI");
	fallback->initBattleInterface(ENV, CB);
	CB->waitTillRealize = false;

	const auto seed = envString("ALPHA_BATTLE_SEED");
	rng.seed(seed.empty() ? std::random_device{}() : static_cast<unsigned>(std::stoul(seed)));
	const auto temp = envString("ALPHA_BATTLE_TEMPERATURE");
	temperature = temp.empty() ? 1.0f : std::stof(temp);

	const auto path = envString("ALPHA_BATTLE_MODEL");
	try
	{
		if(path.empty())
			throw std::runtime_error("ALPHA_BATTLE_MODEL is not set");
		model = BattleModel::load(path);
	}
	catch(const std::exception & e)
	{
		static std::atomic<bool> reported{false};
		if(!reported.exchange(true))
			logAi->error("AlphaBattleAI: cannot use model '%s': %s. StupidAI plays instead.", path, e.what());
	}
}

void AlphaBattleAI::initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB, AutocombatPreferences autocombatPreferences)
{
	initBattleInterface(ENV, CB);
}

void AlphaBattleAI::battleStart(const BattleID & battleID, const CCreatureSet * army1, const CCreatureSet * army2, int3 tile, const CGHeroInstance * hero1, const CGHeroInstance * hero2, BattleSide side, bool replayAllowed)
{
	stats = Stats(); // the neutral player's battle AI is reused for every battle
	stats.siege = cb->getBattle(battleID)->battleGetDefendedTown() != nullptr;
	fallback->battleStart(battleID, army1, army2, tile, hero1, hero2, side, replayAllowed);
}

void AlphaBattleAI::battleEnd(const BattleID & battleID, const BattleResult * br, QueryID queryID)
{
	std::string types;
	for(int t = 0; t < N_ACTION_TYPES; ++t)
		types += std::to_string(stats.types[t]) + (t + 1 < N_ACTION_TYPES ? "," : "");
	const auto & st = stats.spellTargets;
	logAi->info("AlphaBattleAI: battle summary: decisions=%d fallbacks=%d errors=%d types=[%s] spell_targets(none,unit,hex,two)=[%d,%d,%d,%d] siege=%d avg_model_ms=%.2f", stats.decisions, stats.fallbacks, stats.errors, types, st[0], st[1], st[2], st[3], stats.siege, stats.decisions ? stats.modelMs / stats.decisions : 0.0);
	fallback->battleEnd(battleID, br, queryID);
}

void AlphaBattleAI::yourTacticPhase(const BattleID & battleID, int distance)
{
	cb->battleMakeTacticAction(battleID, BattleAction::makeEndOFTacticPhase(cb->getBattle(battleID)->battleGetTacticsSide()));
}

void AlphaBattleAI::useFallback(const BattleID & battleID, const CStack * stack)
{
	++stats.fallbacks;
	fallback->activeStack(battleID, stack);
}

void AlphaBattleAI::activeStack(const BattleID & battleID, const CStack * stack)
{
	auto battle = cb->getBattle(battleID);
	if(!model || stack->hasBonusOfType(BonusType::SIEGE_WEAPON) || battle->battleGetTacticDist() || battle->battleGetRound() >= ROUND_CAP)
	{
		useFallback(battleID, stack);
		return;
	}
	try
	{
		playWithModel(battleID, stack);
	}
	catch(const std::exception & e)
	{
		++stats.errors;
		logAi->error("AlphaBattleAI: %s; StupidAI plays this action", e.what());
		useFallback(battleID, stack);
	}
}

void AlphaBattleAI::playWithModel(const BattleID & battleID, const CStack * stack)
{
	auto battle = cb->getBattle(battleID);
	const auto start = std::chrono::steady_clock::now();
	const auto encoded = encodeBattle(*battle, stack, *env, cb->getPlayerID().value_or(PlayerColor::NEUTRAL));
	const auto outputs = model->run(encoded.tensors);
	const auto action = chooseAction(outputs, rng, temperature);
	stats.modelMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	++stats.decisions;
	++stats.types[action.type];
	if(action.type == static_cast<int>(ActionType::HERO_SPELL))
		++stats.spellTargets[action.target2Hex >= 0 || action.target2Unit >= 0 ? 3 : action.targetHex >= 0 ? 2 : action.targetUnit >= 0 ? 1 : 0];

	const auto dumpDir = envString("ALPHA_BATTLE_DUMP_DIR");
	if(!dumpDir.empty())
	{
		const auto limit = envString("ALPHA_BATTLE_DUMP_LIMIT");
		const int n = dumpedDecisions++;
		if(n < (limit.empty() ? 50 : std::stoi(limit)))
			dumpDecision(dumpDir, "decision_" + std::to_string(getpid()) + "_" + std::to_string(n), encoded.tensors, outputs, action);
	}
	execute(battleID, stack, encoded, action);
}

void AlphaBattleAI::execute(const BattleID & battleID, const CStack * stack, const EncodedBattle & encoded, const ChosenAction & a)
{
	const auto side = cb->getBattle(battleID)->battleGetMySide();
	auto unit = [&](int slot) -> const CStack *
	{
		if(slot < 0 || slot >= MAX_UNITS || !encoded.units[slot])
			throw std::runtime_error("action refers to an empty unit slot " + std::to_string(slot));
		return encoded.units[slot];
	};

	switch(static_cast<ActionType>(a.type))
	{
		case ActionType::WAIT:
			cb->battleMakeUnitAction(battleID, BattleAction::makeWait(stack));
			return;
		case ActionType::DEFEND:
			cb->battleMakeUnitAction(battleID, BattleAction::makeDefend(stack));
			return;
		case ActionType::MOVE:
			cb->battleMakeUnitAction(battleID, BattleAction::makeMove(stack, BattleHex(a.hex)));
			return;
		case ActionType::MELEE:
			cb->battleMakeUnitAction(battleID, BattleAction::makeMeleeAttack(stack, unit(a.unit), BattleHex(a.hex)));
			return;
		case ActionType::SHOOT:
			cb->battleMakeUnitAction(battleID, BattleAction::makeShotAttack(stack, unit(a.unit)));
			return;
		case ActionType::RETREAT:
			cb->battleMakeUnitAction(battleID, BattleAction::makeRetreat(side));
			return;
		case ActionType::SURRENDER:
			cb->battleMakeUnitAction(battleID, BattleAction::makeSurrender(side));
			return;
		case ActionType::HERO_SPELL:
		{
			battle::Target target;
			if(a.targetUnit >= 0)
				target.emplace_back(unit(a.targetUnit));
			if(a.targetHex >= 0)
				target.emplace_back(BattleHex(a.targetHex));
			if(a.target2Hex >= 0)
				target.emplace_back(BattleHex(a.target2Hex));
			if(a.target2Unit >= 0)
				target.emplace_back(unit(a.target2Unit));
			BattleAction cast;
			cast.actionType = EActionType::HERO_SPELL;
			cast.spell = SpellID(a.spell);
			cast.side = side;
			cast.stackNumber = -1;
			cast.setTarget(target);
			// The engine asks this unit for its action again after the hero's cast
			cb->battleMakeSpellAction(battleID, cast);
			return;
		}
		default:
			throw std::runtime_error("model chose action type " + std::to_string(a.type) + ", which is not supported");
	}
}
