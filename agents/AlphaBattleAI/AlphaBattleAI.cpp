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
#include "lib/json/JsonNode.h"
#include "lib/networkPacks/PacksForClientBattle.h"
#include "lib/spells/CSpellHandler.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/mapObjects/army/CCreatureSet.h"
#include "lib/mapObjects/army/CStackInstance.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <unistd.h>

using namespace AlphaSchema;

namespace
{
constexpr int ROUND_CAP = 30;
constexpr int MAX_ATTEMPTS = 8; // choices per unit turn before StupidAI takes over

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

	std::stringstream faultList(envString("ALPHA_BATTLE_TEST_FAULT"));
	for(std::string fault; std::getline(faultList, fault, ',');)
		if(!fault.empty())
			faults.push_back(fault);

	const auto prefixPath = envString("ALPHA_BATTLE_PREFIX");
	if(!prefixPath.empty())
	{
		std::ifstream file(prefixPath);
		for(std::string line; std::getline(file, line);)
		{
			if(line.empty())
				continue;
			const JsonNode node(reinterpret_cast<const std::byte *>(line.data()), line.size(), prefixPath);
			ChosenAction a;
			a.type = node["type"].Integer();
			a.hex = node["hex"].Integer();
			a.unit = node["unit"].Integer();
			a.spell = node["spell"].Integer();
			a.targetUnit = node["target_unit"].Integer();
			a.targetHex = node["target_hex"].Integer();
			a.target2Hex = node["target2_hex"].Integer();
			a.target2Unit = node["target2_unit"].Integer();
			a.forced = true;
			prefix.push_back(a);
		}
	}

	recordDir = envString("ALPHA_BATTLE_RECORD_DIR");
	if(!recordDir.empty())
		std::filesystem::create_directories(recordDir);

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
		if(!recordDir.empty()) // a training record must never silently contain StupidAI's battle
		{
			std::string message = e.what();
			std::replace(message.begin(), message.end(), '"', '\'');
			recordEvent("{\"event\": \"error\", \"message\": \"cannot use model: " + message + "\"}");
		}
	}
}

void AlphaBattleAI::initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB, AutocombatPreferences autocombatPreferences)
{
	initBattleInterface(ENV, CB);
}

void AlphaBattleAI::battleStart(const BattleID & battleID, const CCreatureSet * army1, const CCreatureSet * army2, int3 tile, const CGHeroInstance * hero1, const CGHeroInstance * hero2, BattleSide side, bool replayAllowed)
{
	stats = Stats(); // the neutral player's battle AI is reused for every battle
	heroes = {hero1, hero2};
	lastTurn.reset();
	rejected.clear();
	faultsDoneThisBattle.clear();
	stats.siege = cb->getBattle(battleID)->battleGetDefendedTown() != nullptr;
	stats.walls = cb->getBattle(battleID)->battleGetWallState(EWallPart::KEEP) != EWallState::NONE;

	if(!recordDir.empty())
	{
		auto army = [](const CCreatureSet * set)
		{
			std::string json = "[";
			for(const auto & [slot, stack] : set->Slots())
				json += std::string(json.size() > 1 ? ", " : "") + "[" + std::to_string(stack->getCreatureID().getNum()) + ", " + std::to_string(stack->getCount()) + "]";
			return json + "]";
		};
		const auto * hero = cb->getBattle(battleID)->battleGetMyHero();
		recordEvent("{\"event\": \"start\", \"our_side\": " + std::to_string(static_cast<int>(side)) + ", \"armies\": [" + army(army1) + ", " + army(army2) + "], \"our_mana\": " + std::to_string(hero ? hero->mana : 0) + ", \"our_mana_max\": " + std::to_string(hero ? hero->manaLimit() : 0) + "}");
	}
	fallback->battleStart(battleID, army1, army2, tile, hero1, hero2, side, replayAllowed);
}

void AlphaBattleAI::recordEvent(const std::string & json) const
{
	std::ofstream(std::filesystem::path(recordDir) / "events.jsonl", std::ios::app) << json << "\n";
}

std::string AlphaBattleAI::aliveArmies(const BattleID & battleID) const
{
	// Army units only: summoned units and clones vanish after the battle and are worth nothing
	std::array<std::string, 2> sides = {"[", "["};
	for(const auto * s : cb->getBattle(battleID)->battleGetAllStacks(true))
	{
		if(!s->alive() || s->isClone() || s->summoned || !s->unitSlot().validSlot())
			continue;
		auto & json = sides[static_cast<int>(s->unitSide())];
		json += std::string(json.size() > 1 ? ", " : "") + "[" + std::to_string(s->creatureId().getNum()) + ", " + std::to_string(s->getCount()) + "]";
	}
	return "[" + sides[0] + "], " + sides[1] + "]]";
}

int AlphaBattleAI::ourMana(const BattleID & battleID) const
{
	const auto * hero = cb->getBattle(battleID)->battleGetMyHero();
	return hero ? hero->mana : 0;
}

void AlphaBattleAI::battleNewRoundFirst(const BattleID & battleID)
{
	// Called before the new round's changes: the state here is the end of the previous round
	if(!recordDir.empty())
		recordEvent("{\"event\": \"round_end\", \"round\": " + std::to_string(cb->getBattle(battleID)->battleGetRound()) + ", \"alive\": " + aliveArmies(battleID) + ", \"our_mana\": " + std::to_string(ourMana(battleID)) + "}");
	fallback->battleNewRoundFirst(battleID);
}

void AlphaBattleAI::battleEnd(const BattleID & battleID, const BattleResult * br, QueryID queryID)
{
	if(!recordDir.empty())
	{
		auto casualties = [&](BattleSide side)
		{
			std::string json = "[";
			for(const auto & [creature, count] : br->casualties[side])
				json += std::string(json.size() > 1 ? ", " : "") + "[" + std::to_string(creature.getNum()) + ", " + std::to_string(count) + "]";
			return json + "]";
		};
		recordEvent("{\"event\": \"end\", \"round\": " + std::to_string(cb->getBattle(battleID)->battleGetRound()) + ", \"winner\": " + std::to_string(static_cast<int>(br->winner)) + ", \"result\": " + std::to_string(static_cast<int>(br->result)) + ", \"casualties\": [" + casualties(BattleSide::ATTACKER) + ", " + casualties(BattleSide::DEFENDER) + "], \"alive\": " + aliveArmies(battleID) + ", \"our_mana\": " + std::to_string(ourMana(battleID)) + "}");
	}
	std::string types;
	for(int t = 0; t < N_ACTION_TYPES; ++t)
		types += std::to_string(stats.types[t]) + (t + 1 < N_ACTION_TYPES ? "," : "");
	const auto & st = stats.spellTargets;
	logAi->info("AlphaBattleAI: battle summary: decisions=%d fallbacks=%d errors=%d types=[%s] spell_targets(none,unit,hex,two)=[%d,%d,%d,%d] siege=%d walls=%d server_rejections=%d presend_rejections=%d faults=%d avg_model_ms=%.2f", stats.decisions, stats.fallbacks, stats.errors, types, st[0], st[1], st[2], st[3], stats.siege, stats.walls, stats.serverRejections, stats.preSendRejections, stats.faultsInjected, stats.decisions ? stats.modelMs / stats.decisions : 0.0);
	fallback->battleEnd(battleID, br, queryID);
}

void AlphaBattleAI::battleSpellCast(const BattleID & battleID, const BattleSpellCast * sc)
{
	if(sc->side == cb->getBattle(battleID)->battleGetMySide())
		spellCastSeen = true;
	fallback->battleSpellCast(battleID, sc);
}

bool AlphaBattleAI::previousActionRejected(const CPlayerBattleCallback & battle) const
{
	switch(static_cast<ActionType>(lastActionType))
	{
		case ActionType::RETREAT:
		case ActionType::SURRENDER:
			return true; // accepted would have ended the battle
		case ActionType::HERO_SPELL:
			return !spellCastSeen;
		default:
			return false; // unit actions end the unit's turn; being asked again is good morale
	}
}

bool AlphaBattleAI::injectFault(const std::string & fault)
{
	if(std::find(faults.begin(), faults.end(), fault) == faults.end() || std::find(faultsDoneThisBattle.begin(), faultsDoneThisBattle.end(), fault) != faultsDoneThisBattle.end())
		return false;
	faultsDoneThisBattle.push_back(fault);
	++stats.faultsInjected;
	return true;
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
	const auto player = cb->getPlayerID().value_or(PlayerColor::NEUTRAL);

	// Asked again for the same unit in the same round: find out whether our last action was rejected
	const TurnKey turn{battleID.getNum(), battle->battleGetRound(), static_cast<int>(stack->unitId())};
	if(lastTurn != turn)
		rejected.clear();
	else if(previousActionRejected(*battle))
	{
		// Must NEVER happen for the model's own choices: the masks use the engine's legality checks and are
		// stricter than the server's. A single occurrence (outside ALPHA_BATTLE_TEST_FAULT) is a bug in the
		// masks that must be fixed right away. The recovery below only keeps the unit from losing its turn.
		++stats.serverRejections;
		if(lastActionForced)
			logAi->info("AlphaBattleAI: the server rejected test fault action type %d of unit %d, choosing the next best action", lastActionType, stack->unitId());
		else
			logAi->error("AlphaBattleAI: BUG: the server rejected action type %d of unit %d although the masks allowed it; choosing the next best action", lastActionType, stack->unitId());
	}
	else
		rejected.clear(); // e.g. our hero's spell went through and the unit acts now
	lastTurn = turn;

	for(int attempt = 0; attempt < MAX_ATTEMPTS; ++attempt)
	{
		const auto start = std::chrono::steady_clock::now();
		auto encoded = encodeBattle(*battle, stack, *env, player, heroes[1 - static_cast<int>(battle->battleGetMySide())]);
		for(const auto & action : rejected)
			excludeAction(encoded, action);
		if(!encoded.tensors.any(In::m_type))
			throw std::runtime_error("no legal action left for this unit");

		// Test hook: send something the server must reject (masks are bypassed on purpose)
		std::optional<ChosenAction> forced;
		if(battle->battleGetSurrenderCost() < 0 && injectFault("surrender")) // our mask is stricter than the server: only force what it must refuse
			forced = ChosenAction{.type = static_cast<int>(ActionType::SURRENDER)};
		else if(battle->battleGetMyHero() && injectFault("spell"))
			for(int s = 0; s < N_SPELLS && !forced; ++s)
				if(!encoded.tensors.get(In::m_spell, s) && SpellID(s).toSpell() && !SpellID(s).toSpell()->isAdventure())
					forced = ChosenAction{.type = static_cast<int>(ActionType::HERO_SPELL), .spell = s};
		if(forced)
		{
			logAi->info("AlphaBattleAI: test fault: sending %s, which the server must reject", forced->toJson());
			rejected.push_back(*forced);
			lastActionType = forced->type;
			lastActionForced = true;
			spellCastSeen = false;
			execute(battleID, stack, encoded, *forced);
			return;
		}

		const auto outputs = model->run(encoded.tensors);
		auto action = chooseAction(outputs, rng, temperature);
		if(injectFault("decoder"))
		{
			logAi->info("AlphaBattleAI: test fault: corrupting chosen action %s", action.toJson());
			action.type = static_cast<int>(ActionType::MOVE);
			action.hex = 0; // side column: never a legal destination
		}
		stats.modelMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

		if(prefixPlayed < prefix.size())
		{
			// Replaying a recorded battle: its action must be legal here, or the battle did not replay as recorded
			action = prefix[prefixPlayed++];
			if(!isLegal(encoded, action))
			{
				const auto message = "prefix action " + std::to_string(prefixPlayed - 1) + " " + action.toJson() + " is not legal here: the battle did not replay as recorded";
				if(!recordDir.empty())
				{
					std::string quoted = message;
					std::replace(quoted.begin(), quoted.end(), '"', '\'');
					recordEvent("{\"event\": \"error\", \"message\": \"" + quoted + "\"}");
				}
				throw std::runtime_error(message);
			}
		}

		// Check before sending. Must NEVER trigger (except for the "decoder" test fault): the sampler only
		// picks entries the masks allow, so a failure here is a bug in our code that must be fixed right away.
		// It costs a few mask lookups; once full games show it never triggers it may be commented out.
		if(!isLegal(encoded, action))
		{
			++stats.preSendRejections;
			logAi->error("AlphaBattleAI: BUG: chosen action %s is not legal under the masks; choosing again", action.toJson());
			rejected.push_back(action);
			continue;
		}

		++stats.decisions;
		++stats.types[action.type];
		if(action.type == static_cast<int>(ActionType::HERO_SPELL))
			++stats.spellTargets[action.target2Hex >= 0 || action.target2Unit >= 0 ? 3 : action.targetHex >= 0 ? 2 : action.targetUnit >= 0 ? 1 : 0];

		if(!recordDir.empty())
		{
			char name[32];
			std::snprintf(name, sizeof(name), "decision_%05d", recordedDecisions++);
			dumpDecision(recordDir, name, encoded.tensors, outputs, action);
		}
		const auto dumpDir = envString("ALPHA_BATTLE_DUMP_DIR");
		if(!dumpDir.empty())
		{
			const auto limit = envString("ALPHA_BATTLE_DUMP_LIMIT");
			const int n = dumpedDecisions++;
			if(n < (limit.empty() ? 50 : std::stoi(limit)))
				dumpDecision(dumpDir, "decision_" + std::to_string(getpid()) + "_" + std::to_string(n), encoded.tensors, outputs, action);
		}
		rejected.push_back(action); // excluded only if the server asks this unit again because it was rejected
		lastActionType = action.type;
		lastActionForced = false;
		spellCastSeen = false;
		execute(battleID, stack, encoded, action);
		return;
	}
	throw std::runtime_error("no legal action found after " + std::to_string(MAX_ATTEMPTS) + " attempts");
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
			if(!recordDir.empty())
				recordEvent("{\"event\": \"surrender\", \"cost\": " + std::to_string(cb->getBattle(battleID)->battleGetSurrenderCost()) + "}");
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
