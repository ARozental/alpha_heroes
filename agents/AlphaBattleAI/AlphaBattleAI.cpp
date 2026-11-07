#include "StdInc.h"
#include "AlphaBattleAI.h"
#include "AttackValues.h"

#include "BattleSearch.h"
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
#include "lib/gameState/CGameState.h"

#include <vcmi/Environment.h>
#include "lib/networkPacks/PacksForClientBattle.h"
#include "lib/spells/CSpellHandler.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/mapObjects/army/CCreatureSet.h"
#include "lib/mapObjects/army/CStackInstance.h"
#include "lib/mapObjects/CGTownInstance.h"
#include "lib/entities/artifact/CArtifact.h"
#include "lib/entities/hero/CHero.h"
#include "lib/entities/faction/CFaction.h"
#include "lib/CSkillHandler.h"
#include "lib/TerrainHandler.h"
#include "lib/BattleFieldHandler.h"
#include "lib/CCreatureHandler.h"
#include "lib/GameLibrary.h"

#include <atomic>
#include <map>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cstdlib>
#include <sstream>
#include <unistd.h>

using namespace AlphaSchema;

namespace
{
constexpr int SEARCH_ROUND_CAP = 30; // the search's horizon (BattleSearchEnv)
// A battle where nothing is damaged for this many rounds in a row ends with the attacker retreating, when it can (a
// siege whose catapult is gone, no towers, walls standing and no side able to reach the other: VCMI has no round limit)
constexpr int STALEMATE_ROUNDS = 30;
constexpr int DEFENDER_STALEMATE_DELAY = 5; // the defender retreats this many rounds after the attacker would
// Whatever the damage, a battle that reaches this round ends: the side not ahead retreats when it can, the side ahead
// AHEAD_LIMIT_DELAY rounds later (joint2, 2026-10-29: both sides ours, battles of 1700 decisions per side, mostly
// DEFEND and MOVE, that the quiet rule above never ended; each held its campaign game until the stall watchdog killed
// it, ~0.4 MB of record per decision). Ordinary battles end within 15 rounds.
constexpr int ROUND_LIMIT = 50;
constexpr int AHEAD_LIMIT_DELAY = 5;
constexpr int MAX_ATTEMPTS = 8; // choices per unit turn before the unit defends (playDefault)
std::atomic<int> recordsSent{0}; // per process: each battle's record gets its own key (a game has many battles)

std::string envString(const char * name)
{
	const char * value = std::getenv(name);
	return value ? std::string(value) : std::string();
}

/// ALPHA_BATTLE_<NAME>_<COLOR> (e.g. ALPHA_BATTLE_RECORD_DIR_BLUE) overrides ALPHA_BATTLE_<NAME> for one player,
/// so that two AlphaBattleAIs in one game (self-play) can use their own model, seed and record folder
std::string playerSetting(const CBattleCallback & cb, const std::string & name)
{
	if(const auto player = cb.getPlayerID(); player && (player->isValidPlayer() || *player == PlayerColor::NEUTRAL))
	{
		const std::string color = player->isValidPlayer() ? boost::to_upper_copy(player->toString()) : std::string("NEUTRAL"); // wandering monsters
		auto perPlayer = envString((name + "_" + color).c_str());
		if(!perPlayer.empty())
			return perPlayer;
	}
	return envString(name.c_str());
}

std::atomic<int> dumpedDecisions{0};
}

AlphaBattleAI::~AlphaBattleAI()
{
	if(cb && !observing) // (an observer's callback is the player's: never ours to set)
		cb->waitTillRealize = wasWaitingForRealize;
}

void AlphaBattleAI::initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB)
{
	env = ENV;
	cb = CB;
	// Save before the fallback touches the callback, both restore the same original value
	wasWaitingForRealize = CB->waitTillRealize;
	observing = !playerSetting(*CB, "ALPHA_BATTLE_OBSERVE").empty();
	delegated = observing || !playerSetting(*CB, "ALPHA_BATTLE_DELEGATE").empty();
	if(delegated && !observing) // the baseline evals: another AI plays, we record what it does
	{
		fallback = CDynLibHandler::getNewBattleAI(playerSetting(*CB, "ALPHA_BATTLE_DELEGATE"));
		fallback->initBattleInterface(ENV, CB);
	}
	if(!observing)
		CB->waitTillRealize = false;

	const auto seed = playerSetting(*CB, "ALPHA_BATTLE_SEED");
	rng.seed(seed.empty() ? std::random_device{}() : static_cast<unsigned>(std::stoul(seed)));
	const auto temp = playerSetting(*CB, "ALPHA_BATTLE_TEMPERATURE");
	temperature = temp.empty() ? 1.0f : std::stof(temp);

	std::stringstream faultList(playerSetting(*CB, "ALPHA_BATTLE_TEST_FAULT"));
	for(std::string fault; std::getline(faultList, fault, ',');)
		if(!fault.empty())
			faults.push_back(fault);

	simCheckEnabled = playerSetting(*CB, "ALPHA_BATTLE_SIM_CHECK") == "1";
	attackCache.check = !playerSetting(*CB, "ALPHA_BATTLE_ATTACK_CHECK").empty();
	attackCache.freshCheck = playerSetting(*CB, "ALPHA_BATTLE_ATTACK_CHECK") == "2";
	const auto simulations = playerSetting(*CB, "ALPHA_BATTLE_SEARCH_SIMULATIONS");
	searchSimulations = simulations.empty() ? 0 : std::stoi(simulations);
	const auto considered = playerSetting(*CB, "ALPHA_BATTLE_SEARCH_CONSIDERED");
	searchConsidered = considered.empty() ? 16 : std::stoi(considered);
	const auto prefixPath = playerSetting(*CB, "ALPHA_BATTLE_PREFIX");
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

	recordDir = playerSetting(*CB, "ALPHA_BATTLE_RECORD_DIR");
	if(!recordDir.empty())
		std::filesystem::create_directories(recordDir);
	recordSocket = playerSetting(*CB, "ALPHA_BATTLE_RECORD_SOCKET");
	recordKey = playerSetting(*CB, "ALPHA_BATTLE_RECORD_KEY");
	if(!recordDir.empty() || !recordSocket.empty())
		record = std::make_unique<BattleRecord>();

	modelPath = playerSetting(*CB, "ALPHA_BATTLE_MODEL");
	inferenceSocket = playerSetting(*CB, "ALPHA_BATTLE_INFERENCE_SOCKET");
	inferenceName = playerSetting(*CB, "ALPHA_BATTLE_INFERENCE_MODEL");
	if(inferenceName.empty())
		inferenceName = "current";
	if(!observing || !modelPath.empty() || !inferenceSocket.empty()) // (an observer without a model records zero outputs)
		loadModel();
}

void AlphaBattleAI::loadModel()
{
	try
	{
		if(!inferenceSocket.empty()) // one connection for the AI's lifetime; the server always has the newest weights
		{
			if(!model)
				model = std::make_shared<RemoteBattleModel>(inferenceSocket, inferenceName);
			modelFile = "remote:" + inferenceName + "@" + std::to_string(dynamic_cast<const RemoteBattleModel &>(*model).version());
			return;
		}
		if(modelPath.empty())
			throw std::runtime_error("ALPHA_BATTLE_MODEL is not set");
		const auto time = std::filesystem::last_write_time(modelPath); // follows a symlink: training replaces its target
		const auto file = std::filesystem::weakly_canonical(modelPath).string();
		model = BattleModel::load(file);
		modelTime = time;
		modelFile = file;
	}
	catch(const std::exception & e)
	{
		model.reset();
		logAi->error("AlphaBattleAI: cannot use model '%s': %s. Every unit defends.", modelPath, e.what());
		// Training and evals: a side without its model must fail the battle, not play it defending (unrecorded league
		// opponents did, after a resume misnamed their models, and their battles were trained on as normal)
		if(!playerSetting(*cb, "ALPHA_BATTLE_REQUIRE_MODEL").empty())
		{
			logAi->error("AlphaBattleAI: ALPHA_BATTLE_REQUIRE_MODEL is set: the game ends here");
			std::abort();
		}
		if(record) // a training record must never silently contain a battle the model did not play
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
	history.reset();
	attackCache.clear();
	history.seeUnits(*cb->getBattle(battleID));
	if(!delegated && !inferenceSocket.empty())
		loadModel(); // the version this battle starts with, for its record
	else if(!delegated && !modelPath.empty())
	{
		std::error_code error;
		const auto time = std::filesystem::last_write_time(modelPath, error);
		if(!model || (!error && time != modelTime)) // a newer model (training), e.g. for the reused neutral AI
			loadModel();
	}
	heroes = {hero1, hero2};
	tacticsQueue.clear();
	tacticsPending = false;
	lastDamageRound = 0;
	lastTurn.reset();
	rejected.clear();
	faultsDoneThisBattle.clear();
	stats.siege = cb->getBattle(battleID)->battleGetDefendedTown() != nullptr;
	stats.walls = cb->getBattle(battleID)->battleGetWallState(EWallPart::KEEP) != EWallState::NONE;
	battleStartValues = AlphaReward::BattleEnd();
	surrenderCostPaid = 0;
	for(int s = 0; s < 2; ++s)
	{
		battleStartValues.armyStart[s] = armyValue(*(s == 0 ? army1 : army2));
		battleStartValues.manaStart[s] = heroes[s] ? heroes[s]->mana : 0;
		battleStartValues.power[s] = heroes[s] ? heroes[s]->getPrimSkillLevel(PrimarySkill::SPELL_POWER) : 0; // a player sees both
		battleStartValues.heroLevel[s] = heroLevel(heroes[s]); // true values: the reward is no input
		battleStartValues.artifacts[s] = heroArtifacts(heroes[s]);
	}

	if(record)
	{
		auto army = [](const CCreatureSet * set)
		{
			std::string json = "[";
			for(const auto & [slot, stack] : set->Slots())
				json += std::string(json.size() > 1 ? ", " : "") + "[" + std::to_string(stack->getCreatureID().getNum()) + ", " + std::to_string(stack->getCount()) + "]";
			return json + "]";
		};
		auto machines = [&](BattleSide s) // each side's war machines (creature ids): the reward counts them (data.py summarize)
		{
			std::string json = "[";
			for(const auto * stack : cb->getBattle(battleID)->battleGetStacksIf([s](const CStack * st) { return st->unitSide() == s && st->unitSlot() == SlotID::WAR_MACHINES_SLOT; }))
				json += std::string(json.size() > 1 ? ", " : "") + std::to_string(stack->creatureId().getNum());
			return json + "]";
		};
		const auto * hero = cb->getBattle(battleID)->battleGetMyHero();
		const auto * enemyHero = heroes[1 - static_cast<int>(side)];
		// which model version plays (the link may point at a newer one by now); observing, "human": the player played
		// (ALPHA_BATTLE_OBSERVE's actor, ALPHA_BATTLE_ACTOR: Recorder.cpp sets it), the model's outputs beside it if any
		const auto played = observing ? playerSetting(*cb, "ALPHA_BATTLE_ACTOR").empty() ? std::string("observed") : playerSetting(*cb, "ALPHA_BATTLE_ACTOR") : modelFile;
		recordEvent("{\"event\": \"start\", \"model\": \"" + played + "\", \"our_side\": " + std::to_string(static_cast<int>(side)) + ", \"armies\": [" + army(army1) + ", " + army(army2) + "], \"machines\": [" + machines(BattleSide::ATTACKER) + ", " + machines(BattleSide::DEFENDER) + "], \"our_mana\": " + std::to_string(hero ? hero->mana : 0) + ", \"our_mana_max\": " + std::to_string(hero ? hero->manaLimit() : 0) + ", \"enemy_mana\": " + std::to_string(enemyHero ? enemyHero->mana : 0) + ", \"enemy_mana_max\": " + std::to_string(enemyHero ? enemyHero->manaLimit() : 0) + ", \"powers\": [" + std::to_string(battleStartValues.power[0]) + ", " + std::to_string(battleStartValues.power[1]) + "], \"heroes\": [[" + std::to_string(battleStartValues.heroLevel[0]) + ", " + std::to_string(battleStartValues.artifacts[0]) + "], [" + std::to_string(battleStartValues.heroLevel[1]) + ", " + std::to_string(battleStartValues.artifacts[1]) + "]]}");
		try
		{
			recordEvent("{\"event\": \"scenario\", \"scenario\": " + scenarioJson(battleID, army1, army2) + "}");
		}
		catch(const std::exception & e)
		{
			logAi->warn("AlphaBattleAI: cannot export the battle as a scenario: %s", e.what());
		}
	}
	if(fallback)
		fallback->battleStart(battleID, army1, army2, tile, hero1, hero2, side, replayAllowed);
}

void AlphaBattleAI::recordEvent(const std::string & json)
{
	record->addEvent(json);
	if(!recordDir.empty())
		std::ofstream(std::filesystem::path(recordDir) / "events.jsonl", std::ios::app) << json << "\n";
}

std::string AlphaBattleAI::aliveArmies(const BattleID & battleID) const
{
	// What each side would keep if the battle ended now (remainingAfterBattle): summoned units and clones vanish,
	// creatures raised only for the battle too
	std::array<std::string, 2> sides = {"[", "["};
	for(const auto * s : cb->getBattle(battleID)->battleGetAllStacks(true))
	{
		const int remaining = remainingAfterBattle(s);
		if(remaining <= 0)
			continue;
		auto & json = sides[static_cast<int>(s->unitSide())];
		json += std::string(json.size() > 1 ? ", " : "") + "[" + std::to_string(s->creatureId().getNum()) + ", " + std::to_string(remaining) + "]";
	}
	return "[" + sides[0] + "], " + sides[1] + "]]";
}

std::string AlphaBattleAI::summonedArmies(const BattleID & battleID) const
{
	// Units each side summoned for good (Pit Lords' demons; the engine adds the living ones to the army), with the
	// number summoned: the battle's casualties count the ones killed again, so kept = start - casualties + these
	std::array<std::string, 2> sides = {"[", "["};
	for(const auto * s : cb->getBattle(battleID)->battleGetStacksIf([](const CStack * s) { return keptAfterBattle(s) && s->unitSlot() == SlotID::SUMMONED_SLOT_PLACEHOLDER; }))
	{
		auto & json = sides[static_cast<int>(s->unitSide())];
		json += std::string(json.size() > 1 ? ", " : "") + "[" + std::to_string(s->creatureId().getNum()) + ", " + std::to_string(s->unitBaseAmount()) + "]";
	}
	return "[" + sides[0] + "], " + sides[1] + "]]";
}

int AlphaBattleAI::enemyMana(const BattleID & battleID) const
{
	const auto * hero = heroes[1 - static_cast<int>(cb->getBattle(battleID)->battleGetMySide())];
	return hero ? hero->mana : 0;
}

int AlphaBattleAI::ourMana(const BattleID & battleID) const
{
	const auto * hero = cb->getBattle(battleID)->battleGetMyHero();
	return hero ? hero->mana : 0;
}

void AlphaBattleAI::battleNewRoundFirst(const BattleID & battleID)
{
	// Called before the new round's changes: the state here is the end of the previous round
	if(record)
		recordEvent("{\"event\": \"round_end\", \"round\": " + std::to_string(cb->getBattle(battleID)->battleGetRound()) + ", \"alive\": " + aliveArmies(battleID) + ", \"our_mana\": " + std::to_string(ourMana(battleID)) + ", \"enemy_mana\": " + std::to_string(enemyMana(battleID)) + "}");
	if(fallback)
		fallback->battleNewRoundFirst(battleID);
}

void AlphaBattleAI::battleEnd(const BattleID & battleID, const BattleResult * br, QueryID queryID)
{
	if(record)
	{
		auto casualties = [&](BattleSide side)
		{
			std::string json = "[";
			for(const auto & [creature, count] : br->casualties[side])
				json += std::string(json.size() > 1 ? ", " : "") + "[" + std::to_string(creature.getNum()) + ", " + std::to_string(count) + "]";
			return json + "]";
		};
		// The reward of each side by Reward.h (the search's terminal reward), to check it against data.summarize.
		// The enemy's surrender cost is unknown to us, as in summarize.
		auto end = battleStartValues;
		for(int s = 0; s < 2; ++s)
		{
			end.armyEnd[s] = aliveArmyValue(*cb->getBattle(battleID), static_cast<BattleSide>(s));
			end.manaEnd[s] = heroes[s] ? heroes[s]->mana : 0;
		}
		end.winner = static_cast<int>(br->winner);
		end.result = static_cast<int>(br->result);
		end.rounds = cb->getBattle(battleID)->battleGetRound();
		end.surrenderCost[static_cast<int>(cb->getBattle(battleID)->battleGetMySide())] = surrenderCostPaid;
		char rewards[64];
		std::snprintf(rewards, sizeof(rewards), "[%.17g, %.17g]", AlphaReward::reward(end, 0), AlphaReward::reward(end, 1));
		recordEvent("{\"event\": \"end\", \"round\": " + std::to_string(cb->getBattle(battleID)->battleGetRound()) + ", \"winner\": " + std::to_string(static_cast<int>(br->winner)) + ", \"result\": " + std::to_string(static_cast<int>(br->result)) + ", \"casualties\": [" + casualties(BattleSide::ATTACKER) + ", " + casualties(BattleSide::DEFENDER) + "], \"alive\": " + aliveArmies(battleID) + ", \"summoned\": " + summonedArmies(battleID) + ", \"our_mana\": " + std::to_string(ourMana(battleID)) + ", \"enemy_mana\": " + std::to_string(enemyMana(battleID)) + ", \"cpp_rewards\": " + rewards + "}");

		if(!recordSocket.empty())
		{
			try
			{
				record->send(recordSocket, recordKey + "#" + std::to_string(recordsSent++)); // RecordServer keeps every battle of a key
			}
			catch(const std::exception & e)
			{
				logAi->error("AlphaBattleAI: the record of this battle is lost: %s", e.what()); // the trainer reports it missing
			}
		}
		record = std::make_unique<BattleRecord>(); // the next battle (an AI can play several) starts a new record
	}
	std::string types;
	for(int t = 0; t < N_ACTION_TYPES; ++t)
		types += std::to_string(stats.types[t]) + (t + 1 < N_ACTION_TYPES ? "," : "");
	const auto & st = stats.spellTargets;
	logAi->info("AlphaBattleAI: battle summary: decisions=%d fallbacks=%d errors=%d types=[%s] spell_targets(none,unit,hex,two)=[%d,%d,%d,%d] siege=%d walls=%d server_rejections=%d presend_rejections=%d faults=%d avg_model_ms=%.2f", stats.decisions, stats.fallbacks, stats.errors, types, st[0], st[1], st[2], st[3], stats.siege, stats.walls, stats.serverRejections, stats.preSendRejections, stats.faultsInjected, stats.decisions ? stats.modelMs / stats.decisions : 0.0);
	{
		const auto & av = stats.attackValues;
		const double n = std::max(1, av.calls);
		logAi->info("AlphaBattleAI: attack values: decisions=%d avg_attacks=%.1f avg_played=%.1f avg_runs=%.1f avg_ms=%.2f", av.calls, av.attacks / n, av.played / n, av.runs / n, av.ms / n);
	}
	if(searchSimulations > 0)
	{
		const double n = std::max(1, stats.searched);
		logAi->info("AlphaBattleAI: search: simulations=%d searched=%d errors=%d changed=%d avg_ms=%.1f (copy %.1f, encode %.1f, model %.1f, apply %.1f) avg_network_calls=%.1f", searchSimulations, stats.searched, stats.searchErrors, stats.searchChanged, stats.searchMs / n, stats.searchCopyMs / n, stats.searchEncodeMs / n, stats.searchModelMs / n, stats.searchApplyMs / n, stats.searchEvaluations / n);
	}
	if(attackCache.check)
		logAi->info("AlphaBattleAI: attack check: decisions=%d mismatches=%d", attackCache.checked, attackCache.mismatches);
	attackCache.clear(); // (its copy of the game state: freed between battles)
	if(simCheckEnabled)
		logAi->info("AlphaBattleAI: simulation check: checks=%d rejected=%d exact_mismatches=%d dice_mismatches=%d avg_copy_ms=%.3f avg_apply_ms=%.3f", stats.simChecks, stats.simRejected, stats.simExactMismatches, stats.simDiceMismatches, stats.simChecks ? stats.simCopyMs / stats.simChecks : 0.0, stats.simChecks ? stats.simApplyMs / stats.simChecks : 0.0);
	if(fallback)
		fallback->battleEnd(battleID, br, queryID);
}

void AlphaBattleAI::battleSpellCast(const BattleID & battleID, const BattleSpellCast * sc)
{
	history.spellCast(*cb->getBattle(battleID), *sc);
	if(sc->side == cb->getBattle(battleID)->battleGetMySide())
		spellCastSeen = true;
	if(fallback)
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
	if(delegated)
	{
		if(fallback) // (observing: the player places their units)
			fallback->yourTacticPhase(battleID, distance);
		return;
	}
	// Our units, most valuable first, each placed once (or the model ends the phase); each placement is chosen from
	// the board the previous one left (actionFinished continues), as this is called once, before anything moves
	auto battle = cb->getBattle(battleID);
	auto ours = battle->battleGetStacksIf([&](const CStack * s) { return s->alive() && s->unitSide() == battle->battleGetMySide() && !s->hasBonusOfType(BonusType::SIEGE_WEAPON); });
	std::stable_sort(ours.begin(), ours.end(), [](const CStack * a, const CStack * b) { return creatureValue(a->unitType()) * a->getCount() > creatureValue(b->unitType()) * b->getCount(); });
	tacticsQueue.clear();
	for(auto it = ours.rbegin(); it != ours.rend(); ++it) // popped from the back
		tacticsQueue.push_back((*it)->unitId());
	placeNextTactic(battleID);
}

void AlphaBattleAI::placeNextTactic(const BattleID & battleID)
{
	auto battle = cb->getBattle(battleID);
	const auto side = battle->battleGetMySide();
	const auto player = cb->getPlayerID().value_or(PlayerColor::NEUTRAL);
	while(model && !tacticsQueue.empty() && battle->battleGetTacticDist())
	{
		const CStack * stack = battle->battleGetStackByID(tacticsQueue.back());
		tacticsQueue.pop_back();
		if(!stack || !stack->alive())
			continue;
		try
		{
			auto encoded = encodeBattle(*battle, stack, *env, player, heroes[1 - static_cast<int>(side)], battleStartValues, &history);
			if(encoded.mode != EncodedBattle::Mode::TACTICS || !encoded.tensors.get(In::m_type, static_cast<int>(ActionType::TACTICS_MOVE)))
				continue; // a unit with nowhere to go in the tactics zone
			const auto outputs = model->run(encoded.tensors);
			const auto action = chooseAction(outputs, rng, temperature);
			if(!isLegal(encoded, action))
				throw std::runtime_error("chosen tactics action " + action.toJson() + " is not legal under the masks");
			++stats.decisions;
			++stats.types[action.type];
			if(record)
				record->addDecision(encoded.tensors, outputs, action);
			if(action.type == static_cast<int>(ActionType::TACTICS_END))
				break;
			tacticsPending = true; // actionFinished places the next unit
			cb->battleMakeTacticAction(battleID, toBattleAction(battleID, stack, encoded, action));
			return;
		}
		catch(const std::exception & e)
		{
			++stats.errors;
			logAi->error("AlphaBattleAI: tactics: %s; the phase ends", e.what());
			break;
		}
	}
	tacticsQueue.clear();
	tacticsPending = false;
	if(battle->battleGetTacticDist())
		cb->battleMakeTacticAction(battleID, BattleAction::makeEndOFTacticPhase(side));
}

void AlphaBattleAI::battleStacksAttacked(const BattleID & battleID, const std::vector<BattleStackAttacked> & bsa, bool ranged)
{
	history.stacksAttacked(*cb->getBattle(battleID), bsa);
	for(const auto & attacked : bsa)
		if(attacked.damageAmount > 0)
			lastDamageRound = cb->getBattle(battleID)->battleGetRound();
	if(fallback)
		fallback->battleStacksAttacked(battleID, bsa, ranged);
}

void AlphaBattleAI::playDefault(const BattleID & battleID, const CStack * stack)
{
	++stats.fallbacks;
	auto battle = cb->getBattle(battleID);
	if(stack->creatureId() == CreatureID::CATAPULT) // a catapult has no "defend": aim at the first wall part it can hit
	{
		const auto parts = battle->getAttackableWallParts();
		if(!parts.empty())
		{
			ChosenAction a{.type = static_cast<int>(ActionType::CATAPULT), .hex = parts.front().toInt()};
			cb->battleMakeUnitAction(battleID, engineAction(battle->battleGetMySide(), stack, a, [](int) { return nullptr; }));
			return;
		}
	}
	cb->battleMakeUnitAction(battleID, BattleAction::makeDefend(stack));
}

void AlphaBattleAI::activeStack(const BattleID & battleID, const CStack * stack)
{
	auto battle = cb->getBattle(battleID);
	if(delegated)
	{
		if(record)
		{
			const auto player = cb->getPlayerID().value_or(PlayerColor::NEUTRAL);
			delegatePending = std::make_unique<EncodedBattle>(encodeBattle(*battle, stack, *env, player, heroes[1 - static_cast<int>(battle->battleGetMySide())], battleStartValues, &history));
			encodeAttackValues(*delegatePending, env->game()->gameState(), battleID, player, stats.attackValues, &attackCache);
			delegateStack = stack->unitId();
		}
		if(fallback) // (observing: the player acts)
			fallback->activeStack(battleID, stack);
		return;
	}
	// The defender too, a few rounds later (the attacker's rule goes first when both sides are ours): against BattleAI,
	// which never retreats, a defending hero whose last unit kept flying out of reach held a campaign game until its
	// time limit (joint1, 2026-10-28). A defender without a hero (a garrison, wandering monsters) cannot flee: open
	// Only a side not ahead retreats: one far ahead whose enemy's last unit stays out of reach would throw away a won
	// army (2026-10-28 review); its battle runs on until ROUND_LIMIT (the game's time limit came too late: the stall
	// watchdog killed the game first)
	const int quiet = battle->battleGetRound() - lastDamageRound;
	const bool attacker = battle->battleGetMySide() == BattleSide::ATTACKER;
	const auto mySide = battle->battleGetMySide();
	const bool ahead = aliveArmyValue(*battle, mySide) > aliveArmyValue(*battle, mySide == BattleSide::ATTACKER ? BattleSide::DEFENDER : BattleSide::ATTACKER);
	if(quiet >= STALEMATE_ROUNDS + (attacker ? 0 : DEFENDER_STALEMATE_DELAY) && battle->battleCanFlee() && !ahead)
	{
		++stats.fallbacks;
		logAi->info("AlphaBattleAI: no damage for %d rounds: the %s retreats", quiet, attacker ? "attacker" : "defender");
		cb->battleMakeUnitAction(battleID, BattleAction::makeRetreat(battle->battleGetMySide()));
		return;
	}
	const int round = battle->battleGetRound();
	if(round >= ROUND_LIMIT + (ahead ? AHEAD_LIMIT_DELAY : 0) && battle->battleCanFlee())
	{
		++stats.fallbacks;
		logAi->info("AlphaBattleAI: round %d (no damage for %d rounds): the %s, %s, retreats", round, quiet, attacker ? "attacker" : "defender", ahead ? "ahead" : "not ahead");
		cb->battleMakeUnitAction(battleID, BattleAction::makeRetreat(battle->battleGetMySide()));
		return;
	}
	if(round == ROUND_LIMIT + AHEAD_LIMIT_DELAY && !battle->battleCanFlee())
		logAi->info("AlphaBattleAI: round %d (no damage for %d rounds): the %s, %s, cannot retreat", round, quiet, attacker ? "attacker" : "defender", ahead ? "ahead" : "not ahead");
	if(!model)
	{
		playDefault(battleID, stack);
		return;
	}
	try
	{
		playWithModel(battleID, stack);
	}
	catch(const std::exception & e)
	{
		++stats.errors;
		logAi->error("AlphaBattleAI: %s; the unit defends", e.what());
		playDefault(battleID, stack);
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
		auto encoded = encodeBattle(*battle, stack, *env, player, heroes[1 - static_cast<int>(battle->battleGetMySide())], battleStartValues, &history);
		encodeAttackValues(encoded, env->game()->gameState(), battleID, player, stats.attackValues, &attackCache);
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
		if(action.type == static_cast<int>(ActionType::CREATURE_SPELL))
			action.spell = static_cast<int>(encoded.tensors.get(In::g_creature_spell));
		// The record keeps the network's outputs, and a "search" event with the search's policy (searchTarget)
		searchTarget.clear();
		if(searchSimulations > 0 && prefixPlayed >= prefix.size() && faults.empty() && rejected.empty())
			if(auto searched = searchAction(battleID, encoded, outputs))
				action = *searched;
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
				if(record)
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

		if(record)
		{
			if(!searchTarget.empty())
				recordEvent("{\"event\": \"search\", \"decision\": " + std::to_string(record->size()) + ", " + searchTarget + "}");
			record->addDecision(encoded.tensors, outputs, action);
			if(!recordDir.empty())
			{
				char name[32];
				std::snprintf(name, sizeof(name), "decision_%05d", recordedDecisions++);
				dumpDecision(recordDir, name, encoded.tensors, outputs, action);
			}
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

BattleAction AlphaBattleAI::toBattleAction(const BattleID & battleID, const CStack * stack, const EncodedBattle & encoded, const ChosenAction & a) const
{
	return engineAction(cb->getBattle(battleID)->battleGetMySide(), stack, a, [&](int slot) -> const battle::Unit * { return slot >= 0 && slot < MAX_UNITS ? encoded.units[slot] : nullptr; });
}

std::optional<ChosenAction> AlphaBattleAI::searchAction(const BattleID & battleID, const EncodedBattle & encoded, const OutputTensors & outputs)
{
	const auto started = std::chrono::steady_clock::now();
	try
	{
		const auto root = BattleSimulation::copyOf(env->game()->gameState(), battleID, 0); // its dice are replaced in every simulation
		BattleSearchEnv game(*root, *model, *env, battleStartValues, SEARCH_ROUND_CAP);
		GumbelSearch::Config config;
		config.simulations = searchSimulations;
		config.maxConsidered = searchConsidered;
		config.gumbelScale = temperature > 0 ? 1.0 : 0.0; // greedy play: the search's best action, no noise
		GumbelSearch::Search<BattleSearchEnv> search(game, config);
		const auto result = search.run(static_cast<uint64_t>(rng()) << 32 | rng());

		std::optional<ChosenAction> chosen; // the search's action in the slots of this decision's encoding
		std::map<int64_t, ChosenAction> byId;
		for(const auto & a : legalActions(encoded))
			byId[BattleSearchEnv::actionId(encoded, a)] = a;
		if(byId.count(result.action))
			chosen = byId.at(result.action);
		if(!chosen)
			throw std::runtime_error("the searched action is not legal in the real battle");
		if(record) // the improved policy over every root action, in the slots of this decision
		{
			std::string actions = "[", policy = "[";
			char probability[32];
			for(size_t i = 0; i < result.actions.size(); ++i)
			{
				const auto & a = byId.at(result.actions[i]);
				actions += std::string(i ? ", " : "") + "[" + std::to_string(a.type) + ", " + std::to_string(a.hex) + ", " + std::to_string(a.unit) + ", " + std::to_string(a.spell) + ", " + std::to_string(a.targetUnit) + ", " + std::to_string(a.targetHex) + ", " + std::to_string(a.target2Hex) + ", " + std::to_string(a.target2Unit) + "]";
				std::snprintf(probability, sizeof(probability), "%.6g", result.policy[i]);
				policy += std::string(i ? ", " : "") + probability;
			}
			searchTarget = "\"actions\": " + actions + "], \"policy\": " + policy + "]";
		}
		std::mt19937 noRandomness;
		auto networkBest = chooseAction(outputs, noRandomness, 0.0f);
		if(networkBest.type == static_cast<int>(ActionType::CREATURE_SPELL))
			networkBest.spell = static_cast<int>(encoded.tensors.get(In::g_creature_spell));
		++stats.searched;
		stats.searchChanged += BattleSearchEnv::actionId(encoded, networkBest) != result.action;
		stats.searchEvaluations += game.evaluations;
		stats.searchCopyMs += game.copyMs;
		stats.searchEncodeMs += game.encodeMs;
		stats.searchModelMs += game.modelMs;
		stats.searchApplyMs += game.applyMs;
		stats.searchMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
		return chosen;
	}
	catch(const std::exception & e)
	{
		++stats.searchErrors;
		logAi->error("AlphaBattleAI: search failed: %s; the network's action is played", e.what());
		return std::nullopt;
	}
}

void AlphaBattleAI::execute(const BattleID & battleID, const CStack * stack, const EncodedBattle & encoded, const ChosenAction & a)
{
	const auto action = toBattleAction(battleID, stack, encoded, a);
	if(simCheckEnabled)
	{
		const auto player = cb->getPlayerID().value_or(PlayerColor::NEUTRAL);
		const auto started = std::chrono::steady_clock::now();
		simCheck = BattleSimulation::copyOf(env->game()->gameState(), battleID, static_cast<int>(rng()));
		const auto copied = std::chrono::steady_clock::now();
		if(!simCheck->applyActionOnly(player, action))
		{
			++stats.simRejected;
			logAi->error("AlphaBattleAI: SIM: the simulation refused %s: %s", action.toString(), simCheck->complaint());
		}
		stats.simCopyMs += std::chrono::duration<double, std::milli>(copied - started).count();
		stats.simApplyMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - copied).count();
		simCheckAction = action;
	}
	if(a.type == static_cast<int>(ActionType::SURRENDER))
		surrenderCostPaid = cb->getBattle(battleID)->battleGetSurrenderCost();
	if(a.type == static_cast<int>(ActionType::SURRENDER) && record)
		recordEvent("{\"event\": \"surrender\", \"cost\": " + std::to_string(cb->getBattle(battleID)->battleGetSurrenderCost()) + "}");
	if(a.type == static_cast<int>(ActionType::HERO_SPELL))
		cb->battleMakeSpellAction(battleID, action); // the engine asks this unit for its action again after the hero's cast
	else
		cb->battleMakeUnitAction(battleID, action);
}

void AlphaBattleAI::actionStarted(const BattleID & battleID, const BattleAction & action)
{
	history.actionStarted(*cb->getBattle(battleID), action);
	// ALPHA_BATTLE_DELEGATE: the action the other AI took for the state it was asked about (observing: the player's); a
	// hero's own (a spell, a retreat, a surrender) is the side's, not the stack's: taken in the stack's state too
	const bool byHero = action.actionType == EActionType::HERO_SPELL || action.actionType == EActionType::RETREAT || action.actionType == EActionType::SURRENDER;
	if(delegatePending && (action.stackNumber == static_cast<int>(delegateStack) || byHero) && action.side == cb->getBattle(battleID)->battleGetMySide())
	{
		if(const auto chosen = chosenAction(*delegatePending, action))
		{
			auto outputs = model ? model->run(delegatePending->tensors) : OutputTensors();
			if(!model) // a record has every output: zeros
				for(int o = 0; o < static_cast<int>(Out::COUNT); ++o)
					outputs.data(static_cast<Out>(o)).assign(numel(OUTPUTS[o]), 0.0f);
			record->addDecision(delegatePending->tensors, outputs, *chosen);
			++stats.decisions;
		}
		else
			logAi->warn("AlphaBattleAI: the delegate's action %s has no model action; not recorded", action.toString());
		if(action.actionType == EActionType::HERO_SPELL)
			refreshPending = true; // (the stack still acts: its state once the spell is done, actionFinished)
		else
			delegatePending.reset();
	}
	if(fallback)
		fallback->actionStarted(battleID, action);
}

void AlphaBattleAI::actionFinished(const BattleID & battleID, const BattleAction & action)
{
	history.actionFinished();
	if(refreshPending && action.actionType == EActionType::HERO_SPELL) // (delegated: the stack's state after the spell)
	{
		refreshPending = false;
		auto battle = cb->getBattle(battleID);
		const auto * stack = battle->battleGetStackByID(delegateStack);
		if(stack && stack->alive())
		{
			const auto player = cb->getPlayerID().value_or(PlayerColor::NEUTRAL);
			delegatePending = std::make_unique<EncodedBattle>(encodeBattle(*battle, stack, *env, player, heroes[1 - static_cast<int>(battle->battleGetMySide())], battleStartValues, &history));
			encodeAttackValues(*delegatePending, env->game()->gameState(), battleID, player, stats.attackValues, &attackCache);
		}
		else
			delegatePending.reset();
	}
	if(simCheck && simCheckAction && action.actionType == simCheckAction->actionType && action.stackNumber == simCheckAction->stackNumber && action.side == simCheckAction->side)
		compareWithSimulation(battleID);
	if(tacticsPending && action.side == cb->getBattle(battleID)->battleGetMySide() && cb->getBattle(battleID)->battleGetTacticDist())
	{
		tacticsPending = false;
		placeNextTactic(battleID);
	}
	if(fallback)
		fallback->actionFinished(battleID, action);
}

void AlphaBattleAI::compareWithSimulation(const BattleID & battleID)
{
	++stats.simChecks;
	const auto real = cb->getBattle(battleID);
	const auto & sim = simCheck->battle();
	auto report = [&](int & counter, const std::string & what)
	{
		if(counter++ < 20)
			logAi->warn("AlphaBattleAI: SIM: after %s: %s", simCheckAction->toString(), what);
	};
	for(const auto * s : real->battleGetAllStacks(true))
	{
		const auto * t = sim.battleGetStackByID(s->unitId(), false);
		const auto name = "unit " + std::to_string(s->unitId()) + " (" + s->getName() + ")";
		if(!t)
		{
			report(stats.simExactMismatches, name + " missing in the simulation");
			continue;
		}
		auto exact = [&](const char * field, int64_t a, int64_t b)
		{
			if(a != b)
				report(stats.simExactMismatches, name + " " + field + ": real " + std::to_string(a) + ", simulated " + std::to_string(b));
		};
		const bool sameDice = s->alive() == t->alive() && s->getCount() == t->getCount() && s->getFirstHPleft() == t->getFirstHPleft();
		if(!sameDice)
			report(stats.simDiceMismatches, name + " count/health: real " + std::to_string(s->getCount()) + "/" + std::to_string(s->getFirstHPleft()) + ", simulated " + std::to_string(t->getCount()) + "/" + std::to_string(t->getFirstHPleft()));
		if(!sameDice)
			continue; // what follows can depend on whether the unit died
		exact("position", s->getPosition().toInt(), t->getPosition().toInt());
		exact("defending", s->defending, t->defending);
		exact("waited", s->waitedThisTurn, t->waitedThisTurn);
		exact("moved", s->moved(), t->moved());
		exact("shots", s->shots.available(), t->shots.available());
		exact("retaliations", s->counterAttacks.available(), t->counterAttacks.available());
		auto realSpells = s->activeSpells();
		auto simSpells = t->activeSpells();
		std::sort(realSpells.begin(), realSpells.end());
		std::sort(simSpells.begin(), simSpells.end());
		if(realSpells != simSpells)
			report(stats.simExactMismatches, name + " spell effects differ");
	}
	for(const auto * t : sim.battleGetAllStacks(true))
		if(!real->battleGetStackByID(t->unitId(), false))
			report(stats.simExactMismatches, "unit " + std::to_string(t->unitId()) + " only in the simulation");
	for(int side = 0; side < 2; ++side)
	{
		const auto * realHero = heroes[side]; // the player's callback hides the enemy hero
		const auto * simHero = sim.battleGetFightingHero(static_cast<BattleSide>(side));
		if(realHero && simHero && realHero->mana != simHero->mana)
			report(stats.simExactMismatches, "hero mana of side " + std::to_string(side) + ": real " + std::to_string(realHero->mana) + ", simulated " + std::to_string(simHero->mana));
	}
	simCheck.reset();
	simCheckAction.reset();
}

std::string AlphaBattleAI::scenarioJson(const BattleID & battleID, const CCreatureSet * army1, const CCreatureSet * army2) const
{
	// This battle as a battle_model scenario (MakeBattleMap's format), so a battle of a real game can be played again
	// as a battle-only game (eval sets, calibrating the generator, hard cases). What the game state shows the client.
	auto battle = cb->getBattle(battleID);
	auto key = [](const auto & entity) { auto k = entity->getJsonKey(); return k.substr(k.find(':') + 1); };
	auto armyNode = [&](const CCreatureSet * set)
	{
		JsonNode army;
		for(const auto & [slot, stack] : set->Slots())
		{
			JsonNode pair;
			pair.Vector().emplace_back(key(stack->getCreatureID().toCreature()));
			pair.Vector().emplace_back(static_cast<si64>(stack->getCount()));
			army.Vector().push_back(pair);
		}
		return army;
	};
	JsonNode scenario;
	scenario["name"].String() = "exported";
	scenario["terrain"].String() = key(battle->battleTerrainType().toEntity(LIBRARY));
	scenario["battlefield"].String() = battle->battleGetBattlefieldType().getInfo()->getJsonKey();
	scenario["seed"].Integer() = 0;
	for(int s = 0; s < 2; ++s)
	{
		const auto * hero = heroes[s];
		const auto * army = s == 0 ? army1 : army2;
		JsonNode side;
		side["army"] = armyNode(army);
		if(hero)
		{
			side["hero"].String() = key(hero->getHeroTypeID().toHeroType());
			for(int p = 0; p < GameConstants::PRIMARY_SKILLS; ++p)
				side["primary_skills"].Vector().emplace_back(static_cast<si64>(hero->getBasePrimarySkillValue(PrimarySkill(p))));
			for(const auto & [skill, level] : hero->secSkills)
			{
				JsonNode pair;
				pair.Vector().emplace_back(key(skill.toSkill()));
				pair.Vector().emplace_back(static_cast<si64>(level));
				side["secondary_skills"].Vector().push_back(pair);
			}
			side["secondary_skills"].setType(JsonNode::JsonType::DATA_VECTOR);
			side["experience"].Integer() = hero->exp;
			side["spellbook"].Bool() = hero->hasSpellbook();
			for(const auto & spell : hero->getSpellsInSpellbook())
				side["spells"].Vector().emplace_back(key(spell.toSpell()));
			side["spells"].setType(JsonNode::JsonType::DATA_VECTOR);
			side["war_machines"].setType(JsonNode::JsonType::DATA_VECTOR);
			side["artifacts"].setType(JsonNode::JsonType::DATA_VECTOR);
			for(const auto & [slot, info] : hero->artifactsWorn)
			{
				const auto * art = info.getArt();
				if(!art || info.locked || slot == ArtifactPosition::SPELLBOOK || slot == ArtifactPosition::MACH4)
					continue;
				if(art->getType()->getWarMachine() != CreatureID::NONE)
					side["war_machines"].Vector().emplace_back(key(art->getType()));
				else
				{
					JsonNode pair;
					pair.Vector().emplace_back(static_cast<si64>(slot.num));
					pair.Vector().emplace_back(key(art->getType()));
					side["artifacts"].Vector().push_back(pair);
				}
			}
		}
		scenario["sides"].Vector().push_back(side);
	}
	if(const auto * town = battle->battleGetDefendedTown())
	{
		scenario["town"]["faction"].String() = key(town->getFactionID().toFaction());
		scenario["town"]["fortification"].String() = town->hasBuilt(BuildingID::CASTLE) ? "castle" : town->hasBuilt(BuildingID::CITADEL) ? "citadel" : town->hasBuilt(BuildingID::FORT) ? "fort" : "none";
	}
	else if(!heroes[1] && battle->sideToPlayer(BattleSide::DEFENDER) == PlayerColor::NEUTRAL)
	{
		// Wandering monsters: one creature (split into stacks), sometimes with one stack of its upgrade
		std::map<CreatureID, int> counts;
		for(const auto & [slot, stack] : army2->Slots())
			counts[stack->getCreatureID()] += stack->getCount();
		const auto main = std::max_element(counts.begin(), counts.end(), [](const auto & a, const auto & b) { return a.second < b.second; });
		int total = 0;
		for(const auto & [creature, count] : counts)
			total += count;
		scenario["neutrals"]["creature"].String() = key(main->first.toCreature());
		scenario["neutrals"]["count"].Integer() = total;
		scenario["neutrals"]["upgraded_stack"].Bool() = counts.size() > 1;
		scenario["sides"].Vector().pop_back();
	}
	auto json = scenario.toCompactString(); // one line in events.jsonl: VCMI's "compact" output still has line breaks
	json.erase(std::remove_if(json.begin(), json.end(), [](char c) { return c == '\n' || c == '\r' || c == '\t'; }), json.end());
	return json;
}
