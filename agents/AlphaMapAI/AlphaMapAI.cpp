#include "StdInc.h"
#include "AlphaMapAI.h"

#include "InferenceClient.h"
#include <chrono>
#include <optional>
#include <thread>
#include "Ledger.h"

#include "lib/AsyncRunner.h"
#include "lib/CConfigHandler.h"
#include "lib/CCreatureHandler.h"
#include "lib/IGameSettings.h"
#include "lib/UnlockGuard.h"
#include "lib/battle/BattleStateInfoForRetreat.h"
#include "lib/callback/CCallback.h"
#include "lib/entities/artifact/CArtifactInstance.h"
#include "lib/gameState/CGameState.h"
#include "lib/json/JsonNode.h"
#include "lib/networkPacks/PacksForClientBattle.h"
#include "lib/mapObjects/CGDwelling.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/mapObjects/CGTownInstance.h"
#include "lib/mapObjects/MiscObjects.h"
#include "lib/networkPacks/Component.h"
#include "lib/networkPacks/PacksForClient.h"
#include "lib/networkPacks/PacksForServer.h"
#include "lib/serializer/CTypeList.h"

#include <zlib.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace MapSchema;

namespace
{
std::string env(const std::string & name, const std::string & fallback = "")
{
	const char * value = std::getenv(name.c_str());
	return value ? value : fallback;
}

/// Sends one message to the trainer's RecordServer (battle_model/play.py): u32 key length, key, payload
void sendToRecordServer(const std::string & socketPath, const std::string & key, const std::string & payload)
{
	sockaddr_un address{};
	address.sun_family = AF_UNIX;
	if(socketPath.size() >= sizeof(address.sun_path))
		throw std::runtime_error("record socket path too long: " + socketPath);
	std::copy(socketPath.begin(), socketPath.end(), address.sun_path);
	const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if(fd < 0)
		throw std::runtime_error("cannot create a socket for the records");
	if(connect(fd, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) != 0)
	{
		close(fd);
		throw std::runtime_error("cannot connect to the record socket " + socketPath);
	}
	std::string message;
	const auto length = static_cast<uint32_t>(key.size());
	message.append(reinterpret_cast<const char *>(&length), 4);
	message += key;
	message += payload;
	for(size_t sent = 0; sent < message.size();)
	{
		const auto n = ::send(fd, message.data() + sent, message.size() - sent, 0);
		if(n <= 0)
		{
			close(fd);
			throw std::runtime_error("sending records failed");
		}
		sent += static_cast<size_t>(n);
	}
	close(fd);
}

template<typename T>
void put(std::string & out, T value)
{
	out.append(reinterpret_cast<const char *>(&value), sizeof(T));
}

/// The first whole number in a question's text (e.g. the gold asked for creatures), 0 if none
int64_t firstNumber(const std::string & text)
{
	int64_t number = 0;
	bool found = false;
	for(char ch : text)
	{
		if(ch >= '0' && ch <= '9')
		{
			number = std::min<int64_t>(number * 10 + (ch - '0'), 1000000000);
			found = true;
		}
		else if(found)
			break;
	}
	return number;
}

std::mutex ledgerMutex;
int ledgerDay = 0; // the last day a ledger was sent for, by any AlphaMapAI of this process
}

AlphaMapAI::AlphaMapAI()
	: asyncTasks(std::make_unique<AsyncRunner>())
{
}

AlphaMapAI::~AlphaMapAI()
{
	finish();
}

std::string AlphaMapAI::getBattleAIName() const
{
	if(settings["ai"]["combatEnemyAI"].getType() == JsonNode::JsonType::DATA_STRING)
		return settings["ai"]["combatEnemyAI"].String();
	return "BattleAI";
}

void AlphaMapAI::initGameInterface(std::shared_ptr<Environment> environment, std::shared_ptr<CCallback> callback)
{
	cb = callback;
	cbc = callback;
	env = environment;
	playerID = *cb->getPlayerID();
	cb->waitTillRealize = true;
	cb->unlockGsWhenWaiting = true;

	const auto color = boost::to_upper_copy(playerID.toString());
	modelName = ::env("ALPHA_MAP_MODEL", "current");
	recordSocket = ::env("ALPHA_MAP_RECORD_SOCKET");
	recordsPerDecision = ::env("ALPHA_MAP_RECORD_FORMAT", "1") == "2";
	gameKey = ::env("ALPHA_MAP_GAME_KEY", "game");
	dayCap = std::stoi(::env("ALPHA_MAP_DAYS", "0"));
	temperature = std::stod(::env("ALPHA_MAP_TEMPERATURE", "1"));
	rng.seed(std::stoull(::env("ALPHA_MAP_SEED_" + color, std::to_string(std::random_device()()))));
	inferenceSocket = ::env("ALPHA_MAP_INFERENCE_SOCKET");
	requireModel = !::env("ALPHA_MAP_REQUIRE_MODEL", "").empty();
	connectInference(requireModel);
	initAdvice();
	logAi->info("AlphaMapAI %s: model %s, day cap %d", color, modelName, dayCap);
}

// ---- threads and queries ----

void AlphaMapAI::addQuery(QueryID id, const std::string & description)
{
	if(id == QueryID(-1))
		return;
	std::scoped_lock lock(statusMutex);
	queries[id] = description;
	statusChanged.notify_all();
}

void AlphaMapAI::answer(QueryID id, int selection)
{
	if(id == QueryID(-1))
		return;
	cb->selectionMade(selection, id); // returns when the server has applied it (waitTillRealize)
	std::scoped_lock lock(statusMutex);
	queries.erase(id);
	statusChanged.notify_all();
}

void AlphaMapAI::runAsync(const std::string & description, std::function<void()> task)
{
	asyncTasks->run([this, description, task]() noexcept
	{
		std::shared_lock gsLock(CGameState::mutex);
		try
		{
			task();
		}
		catch(const std::exception & e)
		{
			logAi->error("AlphaMapAI %s: %s failed: %s", playerID.toString(), description, e.what());
			++errors;
		}
	});
}

bool AlphaMapAI::isFree()
{
	return queries.empty() && battle == 0 && visits == 0 && !moving;
}

void AlphaMapAI::waitTillFree()
{
	auto unlock = vstd::makeUnlockSharedGuard(CGameState::mutex);
	std::unique_lock lock(statusMutex);
	while(!stopping && !isFree())
		statusChanged.wait_for(lock, std::chrono::milliseconds(10));
}

void AlphaMapAI::yourTurn(QueryID queryID)
{
	addQuery(queryID, "your turn");
	{
		std::scoped_lock lock(statusMutex);
		havingTurn = true;
	}
	runAsync("answering the turn", [this, queryID]() { answer(queryID, 0); });
	asyncTasks->run([this]() noexcept
	{
		{
			std::unique_lock lock(statusMutex);
			while(!stopping && !isFree())
				statusChanged.wait_for(lock, std::chrono::milliseconds(10));
		}
		std::shared_lock gsLock(CGameState::mutex);
		makeTurn();
	});
}

void AlphaMapAI::makeTurn()
{
	try
	{
		const int today = cb->getDate(Date::DAY);
		if(today != day)
		{
			day = today;
			decisionsToday = 0;
			futile.clear();
		}
		activeHero = ObjectInstanceID(); // the dialogs' context: no hero yet this turn (a question before our first
		activeTarget = int3(-1, -1, -1); // move, a start-of-turn event, got yesterday's last hero: 2026-10-28 review)
		lastChoice = Choice();
		lastChanged = -1;
		if(dayCap > 0 && day > dayCap)
			return; // the records are sent (playerStartsTurn); the turn stays open, so the game waits here until
					// the trainer stops it (ending turns would race through the days, a ledger each)
		for(int n = 0; n < MAX_DECISIONS_PER_DAY && !stopping; ++n)
		{
			waitTillFree();
			if(cb->getPlayerStatus(playerID, false) != EPlayerStatus::INGAME)
				return;
			unifyStacks();
			auto decision = encode(DecisionKind::TURN);
			std::optional<Choice> advice; // (Nullkiller2's, at this decision's chance: before ours, in the same state)
			std::optional<PlanTargets> plan; // (and its plan there: labels both, never inputs)
			std::vector<std::pair<int, int>> accept; // (and our other entries that follow it)
			if(adviceDrawn())
				advice = advise(decision, plan, accept);
			const auto choice = decide(decision, advice, plan, accept);
			if(choice.type <= static_cast<int>(ActionType::END))
				break;
			const bool changed = execute(decision, choice);
			if(!changed)
			{
				noEffect(choice);
				futile.insert({choice.type, choice.index, choice.cell});
			}
			else
				futile.clear(); // the state changed: what failed before may work now
			lastChoice = choice;
			lastChanged = changed ? 1 : 0;
		}
		waitTillFree();
	}
	catch(const std::exception & e)
	{
		logAi->error("AlphaMapAI %s: the turn failed: %s", playerID.toString(), e.what());
		++errors;
	}
	sendRecords(false);
	endTurn();
}

void AlphaMapAI::endTurn()
{
	if(cb->getPlayerStatus(playerID, false) != EPlayerStatus::INGAME)
		return;
	for(int attempt = 0; attempt < 10 && !stopping; ++attempt)
	{
		{
			std::scoped_lock lock(statusMutex);
			if(!havingTurn)
				return;
		}
		cb->endTurn();
	}
	logAi->error("AlphaMapAI %s: could not end the turn", playerID.toString());
}

void AlphaMapAI::requestRealized(PackageApplied * pa)
{
	if(pa->packType == CTypeList::getInstance().getTypeID<EndTurn>(nullptr) && pa->result)
	{
		std::scoped_lock lock(statusMutex);
		havingTurn = false;
		statusChanged.notify_all();
	}
}

void AlphaMapAI::playerBlocked(int reason, bool start)
{
	std::scoped_lock lock(statusMutex);
	if(start && reason == PlayerBlocked::UPCOMING_BATTLE)
		battle = 1;
	if(reason == PlayerBlocked::ONGOING_MOVEMENT)
		moving = start;
	statusChanged.notify_all();
}

void AlphaMapAI::heroVisit(const CGHeroInstance * visitor, const CGObjectInstance * visitedObj, bool start)
{
	std::scoped_lock lock(statusMutex);
	visits += start ? 1 : -1;
	visits = std::max(visits, 0);
	statusChanged.notify_all();
}

namespace
{
JsonNode battleNumber(int64_t value)
{
	JsonNode node;
	node.Integer() = value;
	return node;
}

JsonNode battlePair(int64_t first, int64_t second)
{
	JsonNode pair;
	pair.Vector() = {battleNumber(first), battleNumber(second)};
	return pair;
}

JsonNode battleStacks(const CCreatureSet * army)
{
	JsonNode list;
	list.Vector();
	if(army)
		for(const auto & [slot, stack] : army->Slots())
			list.Vector().push_back(battlePair(stack->getId().getNum(), stack->getCount()));
	return list;
}

/// A battle's hero as the battle reward prices it (battle_model/reward.py): its experience, mana, spell power and the
/// artifacts it carries (the war machines among them); null without one
JsonNode battleHero(const CGHeroInstance * hero)
{
	JsonNode node;
	if(!hero)
		return node;
	node["exp"].Integer() = hero->exp;
	node["level"].Integer() = hero->level;
	node["mana"].Integer() = hero->mana;
	node["power"].Integer() = hero->getPrimSkillLevel(PrimarySkill::SPELL_POWER);
	node["artifacts"].Vector();
	for(const auto & [position, slot] : hero->artifactsWorn)
		if(slot.getArt())
			node["artifacts"].Vector().push_back(battleNumber(slot.getArt()->getTypeId().getNum()));
	for(const auto & slot : hero->artifactsInBackpack)
		if(slot.getArt())
			node["artifacts"].Vector().push_back(battleNumber(slot.getArt()->getTypeId().getNum()));
	return node;
}
}

void AlphaMapAI::battleStart(const BattleID & battleID, const CCreatureSet * army1, const CCreatureSet * army2, int3 tile, const CGHeroInstance * hero1, const CGHeroInstance * hero2, BattleSide side, bool replayAllowed)
{
	{
		std::scoped_lock lock(statusMutex);
		battle = 2;
	}
	{
		// The battle as it starts, for the trainer's foresight labels (records.py): the decision that led here (the
		// last record before it: a MOVE, or the answer that agreed to fight), both sides' armies and heroes
		std::scoped_lock lock(recordsMutex);
		battleNode = JsonNode();
		battleNode["serial"].Integer() = recordSerial;
		battleNode["day"].Integer() = cb->getDate(Date::DAY);
		battleNode["side"].Integer() = side == BattleSide::ATTACKER ? 0 : 1;
		battleNode["armies"].Vector() = {battleStacks(army1), battleStacks(army2)};
		battleNode["heroes"].Vector() = {battleHero(hero1), battleHero(hero2)};
		battleOpen = true;
	}
	CAdventureAI::battleStart(battleID, army1, army2, tile, hero1, hero2, side, replayAllowed);
}

void AlphaMapAI::battleEnd(const BattleID & battleID, const BattleResult * br, QueryID queryID)
{
	{
		std::scoped_lock lock(statusMutex);
		battle = 3;
	}
	{
		std::scoped_lock lock(recordsMutex);
		if(battleOpen && br)
		{
			battleNode["result"].Integer() = static_cast<int>(br->result); // EBattleResult: normal, escape, surrender
			battleNode["winner"].Integer() = static_cast<int>(br->winner); // 0 attacker, 1 defender, 2 a draw
			battleNode["casualties"].Vector();
			battleNode["exp"].Vector();
			for(const auto battleSide : {BattleSide::ATTACKER, BattleSide::DEFENDER})
			{
				JsonNode lost;
				lost.Vector();
				for(const auto & [creature, count] : br->casualties[battleSide])
					lost.Vector().push_back(battlePair(creature.getNum(), count));
				battleNode["casualties"].Vector().push_back(lost);
				battleNode["exp"].Vector().push_back(battleNumber(br->exp[battleSide]));
			}
			battles.push_back(battleNode.toString());
		}
		battleOpen = false;
	}
	CAdventureAI::battleEnd(battleID, br, queryID);
}

void AlphaMapAI::battleResultsApplied()
{
	std::scoped_lock lock(statusMutex);
	battle = 0;
	statusChanged.notify_all();
}

std::optional<BattleAction> AlphaMapAI::makeSurrenderRetreatDecision(const BattleID & battleID, const BattleStateInfoForRetreat & battleState)
{
	// Not the campaign model's decision (the user, 2026-10-27): Nullkiller2's rule and thresholds
	// (config/ai/nk2ai/nk2ai-settings.json): retreat a weak army that is close to death while we still have a town
	const double ours = static_cast<double>(battleState.getOurStrength());
	const double ratio = ours / std::max<double>(1.0, static_cast<double>(battleState.getEnemyStrength()));
	if(!cb->getTownsInfo().empty() && ours < 2000 && ratio < 0.1 && battleState.canFlee)
		return BattleAction::makeRetreat(battleState.ourSide);
	return std::nullopt;
}

void AlphaMapAI::finish()
{
	stopping = true;
	statusChanged.notify_all();
	if(asyncTasks)
	{
		asyncTasks->wait();
		asyncTasks.reset();
	}
}

// ---- the daily ledger: every player's assets, for the empire values (campaign_model/empire.py) ----

void AlphaMapAI::playerStartsTurn(PlayerColor player)
{
	if(recordSocket.empty())
		return;
	const int today = cb->getDate(Date::DAY);
	if(dayCap > 0 && today > dayCap)
		sendRecords(true); // every player's last records (and its summary) as the first day past the cap begins
	std::scoped_lock lock(ledgerMutex);
	if(today <= ledgerDay)
		return;
	ledgerDay = today;
	try
	{
		sendToRecordServer(recordSocket, gameKey + ":ledger#" + std::to_string(today), ledgerJson(static_cast<const IGameInfoCallback &>(*cb).gameState(), today, false));
	}
	catch(const std::exception & e)
	{
		logAi->error("AlphaMapAI: the ledger of day %d was not sent: %s", today, e.what());
	}
}

void AlphaMapAI::gameOver(PlayerColor player, const EVictoryLossCheckResult & victoryLossCheckResult)
{
	if(!recordSocket.empty() && player == playerID)
	{
		try
		{
			sendRecords(true);
			// The end as this player saw it; the trainer takes it as the last snapshot of the game
			std::scoped_lock lock(ledgerMutex);
			sendToRecordServer(recordSocket, gameKey + ":ledger#end-" + playerID.toString(), ledgerJson(static_cast<const IGameInfoCallback &>(*cb).gameState(), cb->getDate(Date::DAY), true));
		}
		catch(const std::exception & e)
		{
			logAi->error("AlphaMapAI %s: the end of the game was not sent: %s", playerID.toString(), e.what());
		}
	}
	if(player == playerID)
	{
		stopping = true;
		statusChanged.notify_all();
	}
}

// ---- the model ----

void AlphaMapAI::connectInference(bool waiting)
{
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
	while(true)
	{
		try
		{
			if(inferenceSocket.empty())
				throw std::runtime_error("ALPHA_MAP_INFERENCE_SOCKET is not set");
			inference = std::make_unique<InferenceClient>(inferenceSocket, FAMILY, CONTRACT_HASH);
			return;
		}
		catch(const std::exception & e)
		{
			if(!waiting)
			{
				logAi->error("AlphaMapAI %s: no model: %s", playerID.toString(), e.what());
				++errors;
				return;
			}
			if(std::chrono::steady_clock::now() > deadline)
			{
				logAi->error("AlphaMapAI %s: no model for a minute (%s): ALPHA_MAP_REQUIRE_MODEL is set, the game ends here", playerID.toString(), e.what());
				std::abort();
			}
			logAi->warn("AlphaMapAI %s: no model yet (%s): waiting", playerID.toString(), e.what()); // (a log line: the stall watchdog's sign of life)
			std::this_thread::sleep_for(std::chrono::seconds(2));
		}
	}
}

Choice AlphaMapAI::decide(Decision & decision, const std::optional<Choice> & advice, const std::optional<PlanTargets> & plan, const std::vector<std::pair<int, int>> & accept)
{
	Choice choice;
	try
	{
		decision.inputs.values[static_cast<size_t>(In::mem)] = memory;
		const auto inputs = decision.inputs.wire();
		std::string reply;
		uint32_t version = 0;
		const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
		for(;;)
		{
			try
			{
				std::scoped_lock lock(inferenceMutex);
				if(!inference)
					throw std::runtime_error("no model");
				version = inference->request(modelName, 1, inputs, reply);
				break;
			}
			catch(const std::exception & e)
			{
				if(!requireModel)
					throw;
				// The server went away (a restart of the run or of a worker): connect again, for a minute at most. A
				// request that keeps failing (the server's ERROR for this decision) ends the game as well: it was asked
				// again at once, forever, each failure logged as a sign of life for the stall watchdog (2026-10-31 review)
				if(std::chrono::steady_clock::now() > deadline)
				{
					logAi->error("AlphaMapAI %s: no decision for a minute (%s): ending the game (ALPHA_MAP_REQUIRE_MODEL)", playerID.toString(), e.what());
					std::abort();
				}
				logAi->error("AlphaMapAI %s: no decision (%s): connecting again", playerID.toString(), e.what());
				std::this_thread::sleep_for(std::chrono::seconds(1));
				std::scoped_lock lock(inferenceMutex);
				inference.reset();
				connectInference(true);
			}
		}
		const auto outputs = Outputs::fromWire(reply);
		for(int r = 0; r < N_RESOURCES; ++r) // (what a purchase short of resources trades away the least of: payByTrades)
			decision.worth[r] = outputs[Out::resource_worth][r];
		// The memory this decision leaves, for the next one: turn decisions only (the others read it). The imitation
		// learned it over the teacher's turn decisions alone (Teacher.cpp records no questions or exchanges), so the
		// memory's sequence here is that one
		if(decision.kind == DecisionKind::TURN)
		{
			const auto & next = outputs[Out::memory];
			for(size_t i = 0; i < memory.size() && i < next.size(); ++i)
				memory[i] = std::llround(std::clamp(next[i], -1.0f, 1.0f) * MEMORY_SCALE);
		}
		choice = sample(decision, outputs);
		Record entry;
		entry.day = cb->getDate(Date::DAY);
		entry.kind = static_cast<int>(decision.kind);
		entry.decisionsToday = decisionsToday;
		entry.version = version;
		entry.choice = choice;
		if(advice)
		{
			entry.advice = advice;
			entry.flags |= RECORD_ADVICE;
		}
		if(plan)
		{
			entry.plan = plan;
			entry.flags |= RECORD_PLAN;
		}
		if(!accept.empty())
		{
			entry.accept = accept;
			entry.flags |= RECORD_ACCEPT;
		}
		for(int k = 0; k < N_VALUE_OFFSETS; ++k)
			entry.value[k] = outputs[Out::value][k];
		entry.inputs = inputs;
		std::scoped_lock lock(recordsMutex);
		entry.serial = choice.record = ++recordSerial;
		entry.choice.record = entry.serial;
		records.push_back(std::move(entry));
	}
	catch(const std::exception & e)
	{
		logAi->error("AlphaMapAI %s: no decision: %s", playerID.toString(), e.what());
		++errors;
		choice = Choice();
	}
	if(decision.kind == DecisionKind::TURN)
	{
		++decisionsToday;
		// A sign of life for the games' stall watchdog (headless.sh: no log output for 60 s kills a game): with a few
		// heroes a turn is dozens of decisions, each waiting for the inference server, and logged nothing (2026-10-30)
		if(decisionsToday % 20 == 0)
			logAi->info("AlphaMapAI %s: day %d, %d turn decisions so far today", playerID.toString(), cb->getDate(Date::DAY), decisionsToday);
	}
	return choice;
}

Choice AlphaMapAI::sample(const Decision & decision, const Outputs & outputs)
{
	std::scoped_lock lock(rngMutex);
	// One draw among the legal (finite) entries of [begin, end), with probabilities p^(1/temperature); returns the
	// index and the model's own log-probability of it
	auto draw = [this](const std::vector<float> & logp, size_t begin, size_t end) -> std::pair<int, float>
	{
		int best = -1;
		double total = 0;
		std::vector<double> weights(end - begin, 0.0);
		double top = -std::numeric_limits<double>::infinity();
		for(size_t i = begin; i < end; ++i)
			if(std::isfinite(logp[i]))
				top = std::max<double>(top, logp[i]);
		if(!std::isfinite(top))
			return {-1, 0.0f};
		for(size_t i = begin; i < end; ++i)
		{
			if(!std::isfinite(logp[i]))
				continue;
			if(temperature <= 0)
			{
				if(best < 0 || logp[i] > logp[best])
					best = static_cast<int>(i);
				continue;
			}
			weights[i - begin] = std::exp((logp[i] - top) / temperature);
			total += weights[i - begin];
		}
		if(temperature > 0)
		{
			double target = std::uniform_real_distribution<double>(0.0, total)(rng);
			for(size_t i = begin; i < end; ++i)
			{
				if(weights[i - begin] <= 0)
					continue;
				best = static_cast<int>(i);
				target -= weights[i - begin];
				if(target <= 0)
					break;
			}
		}
		return {best, logp[best]};
	};

	Choice choice;
	if(decision.kind != DecisionKind::TURN && decision.kind != DecisionKind::EXCHANGE)
	{
		const auto [index, logp] = draw(outputs[Out::option_logp], 0, N_OPTIONS);
		choice.index = index;
		choice.logp = logp;
		return choice;
	}
	const auto [type, typeLogp] = draw(outputs[Out::type_logp], 0, N_ACTION_TYPES);
	choice.type = type;
	choice.logp = typeLogp;
	if(type <= static_cast<int>(ActionType::END))
		return choice;
	const auto & table = outputs[head(static_cast<ActionType>(type)).first];
	const auto [index, indexLogp] = draw(table, 0, table.size());
	choice.index = index;
	choice.logp += indexLogp;
	if(type == static_cast<int>(ActionType::CAST) && index >= 0)
	{
		const int hero = index / N_ADVENTURE_SPELLS;
		const int spell = index % N_ADVENTURE_SPELLS;
		if(std::find(TARGETED_SPELLS.begin(), TARGETED_SPELLS.end(), spell) != TARGETED_SPELLS.end())
		{
			const size_t cells = WINDOW * WINDOW;
			const auto [cell, cellLogp] = draw(outputs[Out::cast_cell_logp], hero * cells, (hero + 1) * cells);
			choice.cell = cell < 0 ? -1 : cell - static_cast<int>(hero * cells);
			choice.logp += cellLogp;
		}
	}
	if(index < 0)
		choice.type = -1; // a legal type with no legal entry: the model and the masks disagree
	return choice;
}

std::string AlphaMapAI::teacherJson() const
{
	if(!teacher)
		return "";
	std::string unmatched;
	for(const auto & [kind, count] : teacherUnmatched)
		unmatched += (unmatched.empty() ? "\"" : ", \"") + kind + "\": " + std::to_string(count);
	return ", \"teacher\": {\"matched\": " + std::to_string(teacherMatched) + ", \"approximate\": " + std::to_string(teacherApproximate) + ", \"unmatched\": {" + unmatched + "}}";
}

void AlphaMapAI::noEffect(const Choice & choice)
{
	++failedActions;
	logAi->warn("AlphaMapAI %s: action %d entry %d changed nothing", playerID.toString(), choice.type, choice.index);
	// The decision's own record: a dialog the action raised (a Town Portal's choice, a yes/no) records after it
	std::scoped_lock lock(recordsMutex);
	for(auto r = records.rbegin(); r != records.rend(); ++r)
		if(r->serial == choice.record)
		{
			r->flags |= RECORD_NO_EFFECT;
			break;
		}
}

void AlphaMapAI::sendRecords(bool final)
{
	if(recordSocket.empty())
		return;
	std::vector<Record> pending;
	std::vector<std::string> fought;
	{
		std::scoped_lock lock(recordsMutex);
		if(final && done)
			return; // sent already
		pending.swap(records);
		fought.swap(battles);
		done = done || final;
	}
	const auto color = boost::to_upper_copy(playerID.toString());
	try
	{
		if(!pending.empty())
		{
			// "AMZ2" (ALPHA_MAP_RECORD_FORMAT=2, asked by a trainer that reads it): the count, then each decision
			// (campaign_model/records.py) with its inputs zlib-compressed on their own at level 1, which the trainer keeps as
			// they come. Else "AMZ1": the whole message compressed, at level 1 (it was 6: ~40% of this AI's CPU; the trainer
			// decompresses it and compresses each decision again)
			auto header = [&](std::string & out, const Record & r)
			{
				put<uint32_t>(out, r.day);
				put<uint32_t>(out, r.kind);
				put<uint32_t>(out, r.decisionsToday);
				put<uint32_t>(out, r.version);
				put<int32_t>(out, r.choice.type);
				put<int32_t>(out, r.choice.index);
				put<int32_t>(out, r.choice.cell);
				put<float>(out, r.choice.logp);
				for(float v : r.value)
					put<float>(out, v);
				put<uint32_t>(out, r.flags);
				put<uint32_t>(out, static_cast<uint32_t>(r.inputs.size()));
				if(r.flags & RECORD_ADVICE) // (Nullkiller2's choice there: its type, entry, tile)
				{
					put<int32_t>(out, r.advice->type);
					put<int32_t>(out, r.advice->index);
					put<int32_t>(out, r.advice->cell);
				}
				if(r.flags & RECORD_PLAN) // (Nullkiller2's plan there: each hero slot's target tile, its days, its task's priority)
					for(const auto & target : *r.plan)
					{
						put<int16_t>(out, target.x);
						put<int16_t>(out, target.y);
						put<int16_t>(out, target.z);
						put<int16_t>(out, target.turns);
						put<float>(out, target.priority);
					}
				if(r.flags & RECORD_ACCEPT) // (more entries that follow Nullkiller2: their count, then each one's type and entry)
				{
					put<uint16_t>(out, static_cast<uint16_t>(r.accept.size()));
					for(const auto & [type, index] : r.accept)
					{
						put<int16_t>(out, static_cast<int16_t>(type));
						put<int32_t>(out, index);
					}
				}
			};
			auto deflate = [](const std::string & raw)
			{
				uLongf size = compressBound(raw.size());
				std::string packed(size, '\0');
				if(compress2(reinterpret_cast<Bytef *>(packed.data()), &size, reinterpret_cast<const Bytef *>(raw.data()), raw.size(), 1) != Z_OK)
					throw std::runtime_error("compressing the records failed");
				packed.resize(size);
				return packed;
			};
			std::string payload;
			if(recordsPerDecision)
			{
				payload = "AMZ2";
				put<uint32_t>(payload, static_cast<uint32_t>(pending.size()));
				for(const auto & r : pending)
				{
					header(payload, r);
					const auto packed = deflate(r.inputs);
					put<uint32_t>(payload, static_cast<uint32_t>(packed.size()));
					payload += packed;
				}
			}
			else
			{
				std::string raw = "AMR1";
				put<uint32_t>(raw, static_cast<uint32_t>(pending.size()));
				for(const auto & r : pending)
				{
					header(raw, r);
					raw += r.inputs;
				}
				payload = "AMZ1";
				put<uint32_t>(payload, static_cast<uint32_t>(raw.size()));
				payload += deflate(raw);
			}
			sendToRecordServer(recordSocket, gameKey + ":" + color + "#" + std::to_string(recordsSent++), payload);
		}
		for(const auto & battleJson : fought) // (after the records: the decisions they point to are in)
			sendToRecordServer(recordSocket, gameKey + ":" + color + ":battle#" + std::to_string(battlesSent++), battleJson);
		if(final)
		{
			const auto done = ", \"traded_purchases\": " + std::to_string(tradedPurchases) + ", \"boats\": " + std::to_string(boatsBuilt) + ", \"skills_bought\": " + std::to_string(skillsBought) + ", \"stacks_merged\": " + std::to_string(stacksMerged) + ", \"multi_actions\": " + std::to_string(multiActions);
			const auto summary = "{\"errors\": " + std::to_string(errors) + ", \"failed_actions\": " + std::to_string(failedActions) + done + teacherJson() + adviceJson() + "}";
			sendToRecordServer(recordSocket, gameKey + ":" + color + ":done", summary);
		}
	}
	catch(const std::exception & e)
	{
		logAi->error("AlphaMapAI %s: records not sent: %s", color, e.what());
		++errors;
	}
}

// ---- questions the engine asks ----

void AlphaMapAI::showBlockingDialog(const std::string & text, const std::vector<Component> & components, QueryID askID, const int soundID, bool selection, bool cancel, bool safeToAutoaccept)
{
	addQuery(askID, "blocking dialog");
	if(!selection && !cancel)
	{
		runAsync("closing a dialog", [this, askID]() { answer(askID, 0); });
		return;
	}
	runAsync("a dialog", [this, text, components, askID, soundID, selection, cancel]()
	{
		auto decision = blockingQuestion(text, components, soundID, selection, cancel);
		const auto choice = decide(decision);
		answer(askID, choice.index >= 0 ? decision.answers[choice.index] : decision.answers.front());
	});
}

Decision AlphaMapAI::blockingQuestion(const std::string & text, const std::vector<Component> & components, int soundID, bool selection, bool cancel)
{
	auto decision = encode(selection ? DecisionKind::SELECT : DecisionKind::YES_NO);
	encodeOptionContext(decision, activeHero, activeTarget);
	decision.inputs.at(In::d_number) = firstNumber(text);
	decision.inputs.at(In::d_sound) = soundID;
	auto & in = decision.inputs;
	if(selection)
	{
		for(int i = 0; i < std::min<int>(components.size(), N_OPTIONS); ++i)
		{
			in.at(In::o_kind, i) = static_cast<int>(OptionKind::COMPONENT);
			in.at(In::o_type, i) = static_cast<int>(components[i].type);
			in.at(In::o_subtype, i) = components[i].subType.getNum();
			in.at(In::o_amount, i) = components[i].value.value_or(0);
			in.at(In::option_mask, i) = 1;
			decision.answers.push_back(i + 1); // components are numbered from 1
		}
		if(cancel && decision.answers.size() < static_cast<size_t>(N_OPTIONS)) // declining is an answer too (0)
		{
			const int i = static_cast<int>(decision.answers.size());
			in.at(In::o_kind, i) = static_cast<int>(OptionKind::NO);
			in.at(In::option_mask, i) = 1;
			decision.answers.push_back(0);
		}
	}
	else
	{
		in.at(In::o_kind, 0) = static_cast<int>(OptionKind::YES);
		in.at(In::o_kind, 1) = static_cast<int>(OptionKind::NO);
		in.at(In::option_mask, 0) = in.at(In::option_mask, 1) = 1;
		decision.answers = {1, 0};
		encodeQuestionGuards(decision, text, components);
	}
	return decision;
}

void AlphaMapAI::heroGotLevel(const CGHeroInstance * hero, PrimarySkill pskill, std::vector<SecondarySkill> & skills, QueryID queryID)
{
	addQuery(queryID, "level up");
	const auto heroID = hero->id;
	runAsync("a level up", [this, heroID, skills, queryID]()
	{
		const auto * h = cb->getHero(heroID);
		if(skills.empty() || !h)
		{
			answer(queryID, 0);
			return;
		}
		auto decision = levelUpQuestion(h, skills);
		const auto choice = decide(decision);
		answer(queryID, choice.index >= 0 ? decision.answers[choice.index] : 0);
	});
}

Decision AlphaMapAI::levelUpQuestion(const CGHeroInstance * h, const std::vector<SecondarySkill> & skills)
{
	auto decision = encode(DecisionKind::LEVEL_UP);
	encodeOptionContext(decision, h->id, h->visitablePos());
	for(int i = 0; i < std::min<int>(skills.size(), N_OPTIONS); ++i)
	{
		decision.inputs.at(In::o_kind, i) = static_cast<int>(OptionKind::SKILL);
		decision.inputs.at(In::o_type, i) = skills[i].getNum();
		decision.inputs.at(In::o_subtype, i) = h->getSecSkillLevel(skills[i]) + 1;
		decision.inputs.at(In::option_mask, i) = 1;
		decision.answers.push_back(i);
	}
	return decision;
}

void AlphaMapAI::commanderGotLevel(const CCommanderInstance * commander, std::vector<ui32> skills, QueryID queryID)
{
	addQuery(queryID, "commander level up");
	runAsync("a commander level up", [this, queryID]() { answer(queryID, 0); });
}

void AlphaMapAI::showMapObjectSelectDialog(QueryID askID, const Component & icon, const MetaString & title, const MetaString & description, const std::vector<ObjectInstanceID> & objects)
{
	addQuery(askID, "object selection");
	runAsync("an object selection", [this, askID, objects]()
	{
		auto decision = encode(DecisionKind::OBJECT_SELECT);
		encodeOptionContext(decision, activeHero, activeTarget);
		for(const auto & id : objects)
		{
			const auto * obj = cb->getObj(id, false);
			const int i = static_cast<int>(decision.answers.size());
			if(!obj || i >= N_OPTIONS)
				continue;
			decision.inputs.at(In::o_kind, i) = static_cast<int>(OptionKind::OBJECT);
			decision.inputs.at(In::o_type, i) = obj->ID.getNum();
			decision.inputs.at(In::o_subtype, i) = obj->subID.getNum();
			const auto pos = obj->visitablePos();
			decision.inputs.at(In::o_pos, i, 0) = pos.x;
			decision.inputs.at(In::o_pos, i, 1) = pos.y;
			decision.inputs.at(In::o_pos, i, 2) = pos.z;
			decision.inputs.at(In::option_mask, i) = 1;
			decision.answers.push_back(id.getNum()); // the reply is the object
		}
		if(decision.answers.empty())
		{
			answer(askID, objects.empty() ? 0 : objects.front().getNum());
			return;
		}
		const auto choice = decide(decision);
		answer(askID, decision.answers[std::max(choice.index, 0)]);
	});
}

void AlphaMapAI::showRecruitmentDialog(const CGDwelling * dwelling, const CArmedInstance * dst, int level, QueryID queryID)
{
	addQuery(queryID, "recruitment");
	const auto dwellingID = dwelling->id;
	const auto dstID = dst->id;
	runAsync("a recruitment", [this, dwellingID, dstID, queryID]()
	{
		const auto * d = dynamic_cast<const CGDwelling *>(cb->getObj(dwellingID, false));
		const auto * army = dynamic_cast<const CArmedInstance *>(cb->getObj(dstID, false));
		// The dearest creature on offer that fits the army; the model picks how many (options: none and fractions)
		int bestLevel = -1;
		CreatureID creature;
		int most = 0;
		if(d && army)
		{
			int bestCost = -1;
			for(int i = 0; i < static_cast<int>(d->creatures.size()); ++i)
			{
				const auto & [count, ids] = d->creatures[i];
				if(ids.empty() || count <= 0 || !army->getSlotFor(ids.back()).validSlot())
					continue;
				const int cost = ids.back().toCreature()->getRecruitCost(EGameResID::GOLD);
				const int affordable = std::min<int>(count, cb->getResourceAmount() / ids.back().toCreature()->getFullRecruitCost());
				if(affordable > 0 && cost > bestCost)
				{
					bestCost = cost;
					bestLevel = i;
					creature = ids.back();
					most = affordable;
				}
			}
		}
		if(bestLevel < 0)
		{
			answer(queryID, 0);
			return;
		}
		auto decision = encode(DecisionKind::RECRUIT_DWELLING);
		encodeOptionContext(decision, dstID, d->visitablePos());
		std::vector<int> amounts = {0};
		for(double fraction : RECRUIT_FRACTIONS)
		{
			const int amount = std::max(1, static_cast<int>(std::ceil(most * fraction)));
			if(amount != amounts.back())
				amounts.push_back(amount);
		}
		for(int i = 0; i < std::min<int>(amounts.size(), N_OPTIONS); ++i)
		{
			decision.inputs.at(In::o_kind, i) = static_cast<int>(OptionKind::AMOUNT);
			decision.inputs.at(In::o_type, i) = creature.getNum();
			decision.inputs.at(In::o_amount, i) = amounts[i];
			decision.inputs.at(In::option_mask, i) = 1;
			decision.answers.push_back(amounts[i]);
		}
		const auto choice = decide(decision);
		const int amount = choice.index >= 0 ? decision.answers[choice.index] : 0;
		if(amount > 0)
			cb->recruitCreatures(d, army, creature, amount, bestLevel);
		answer(queryID, 0);
	});
}

void AlphaMapAI::showGarrisonDialog(const CArmedInstance * up, const CGHeroInstance * down, bool removableUnits, QueryID queryID, const MetaString & customTitle)
{
	addQuery(queryID, "garrison");
	const auto upID = up->id;
	const auto downID = down->id;
	if(!removableUnits || up->tempOwner != down->tempOwner && up->tempOwner.isValidPlayer())
	{
		runAsync("closing a garrison", [this, queryID]() { answer(queryID, 0); });
		return;
	}
	runAsync("a garrison", [this, upID, downID, queryID]() { exchange(upID, downID, queryID); });
}

void AlphaMapAI::heroExchangeStarted(ObjectInstanceID hero1, ObjectInstanceID hero2, QueryID queryID)
{
	addQuery(queryID, "hero exchange");
	runAsync("a hero exchange", [this, hero1, hero2, queryID]()
	{
		if(cb->getHero(hero1) && cb->getHero(hero2))
			exchange(hero1, hero2, queryID);
		else
			answer(queryID, 0);
	});
}

void AlphaMapAI::exchange(ObjectInstanceID first, ObjectInstanceID second, QueryID queryID)
{
	// Troop moves between the two armies while the server holds the exchange open, until the model ends it, 8 at
	// most (it was 32: joint7's model, never taught exchanges, used all 32 in nearly every one, 55% of its decisions,
	// 2026-11-01; Nullkiller2 makes a few)
	futile.clear();
	for(int n = 0; n < 8 && !stopping; ++n)
	{
		auto decision = encode(DecisionKind::EXCHANGE, first, second);
		if(!decision.inputs.any(In::army_mask) && !decision.inputs.any(In::artifact_mask) && !decision.inputs.any(In::multi_mask))
			break;
		std::optional<Choice> advice;
		std::vector<std::pair<int, int>> accept;
		if(adviceDrawn())
			advice = adviseExchange(decision, first, second, accept);
		const auto choice = decide(decision, advice, std::nullopt, accept);
		if(choice.type != static_cast<int>(ActionType::ARMY) && choice.type != static_cast<int>(ActionType::ARTIFACT) && choice.type != static_cast<int>(ActionType::MULTI))
			break;
		if(!execute(decision, choice))
		{
			noEffect(choice);
			futile.insert({choice.type, choice.index, choice.cell});
		}
		else
			futile.clear();
		if(choice.type == static_cast<int>(ActionType::MULTI)) // (GIVE_ALL is the whole exchange: the troop moves after it sent
			break;                                                  // its troops back, 567 of 870, 2026-11-05)
	}
	futile.clear();
	answer(queryID, 0);
}

void AlphaMapAI::showTeleportDialog(const CGHeroInstance * hero, TeleportChannelID channel, TTeleportExitsList exits, bool impassable, QueryID askID)
{
	// Not a model decision (rare: monoliths with several exits): the exit the path goes through, else the engine's choice
	addQuery(askID, "teleport");
	int chosen = -1;
	for(size_t i = 0; i < exits.size(); ++i)
	{
		const auto * exit = cb->getObj(exits[i].first, false);
		if(exits[i].second == intendedExit || (exit && exit->visitablePos() == intendedExit))
			chosen = static_cast<int>(i);
	}
	runAsync("a teleport", [this, askID, chosen]() { answer(askID, chosen); });
}

void AlphaMapAI::showTavernWindow(const CGObjectInstance * object, const CGHeroInstance * visitor, QueryID queryID)
{
	addQuery(queryID, "tavern"); // map taverns: not yet a decision (hiring is in our towns)
	runAsync("closing a tavern", [this, queryID]() { answer(queryID, 0); });
}

void AlphaMapAI::showMarketWindow(const IMarket * market, const CGHeroInstance * visitor, QueryID queryID)
{
	addQuery(queryID, "market"); // trading posts, altars, black markets: not yet decisions
	runAsync("closing a market", [this, queryID]() { answer(queryID, 0); });
}

void AlphaMapAI::showUniversityWindow(const IMarket * market, const CGHeroInstance * visitor, QueryID queryID)
{
	// Each skill on offer the hero can learn, a yes/no (the user, 2026-11-04: four yes/no choices, the skill's own
	// embedding for YES): YES buys it (CGameHandler::buySecSkill's rules: not known, learnable, its price ours)
	addQuery(queryID, "university");
	const auto marketID = market->getObjInstanceID();
	const auto heroID = visitor ? visitor->id : ObjectInstanceID();
	runAsync("a university", [this, marketID, heroID, queryID]()
	{
		struct Close // (the window answered however the questions end: an open query would stall the turn)
		{
			AlphaMapAI * ai;
			QueryID id;
			~Close() { ai->answer(id, 0); }
		} close{this, queryID};
		const auto * m = cb->getMarket(marketID);
		const int price = cb->getSettings().getInteger(EGameSettings::MARKETS_UNIVERSITY_GOLD_COST);
		for(const auto & item : m ? m->availableItemsIds(EMarketMode::RESOURCE_SKILL) : std::vector<TradeItemBuy>())
		{
			const auto * hero = cb->getHero(heroID);
			const auto skill = item.as<SecondarySkill>();
			if(!hero || hero->getSecSkillLevel(skill) || !hero->canLearnSkill(skill) || cb->getResourceAmount()[EGameResID::GOLD] < price)
				continue;
			auto decision = universityQuestion(hero, skill, price);
			if(decide(decision).index == 0)
			{
				cb->trade(marketID, EMarketMode::RESOURCE_SKILL, GameResID(EGameResID::GOLD), skill, 1, hero);
				++skillsBought;
			}
		}
	});
}

Decision AlphaMapAI::universityQuestion(const CGHeroInstance * hero, SecondarySkill skill, int price)
{
	auto decision = encode(DecisionKind::YES_NO);
	encodeOptionContext(decision, hero->id, hero->visitablePos());
	auto & in = decision.inputs;
	in.at(In::o_kind, 0) = static_cast<int>(OptionKind::SKILL);
	in.at(In::o_type, 0) = skill.getNum();
	in.at(In::o_subtype, 0) = 1; // (a University teaches the basic level)
	in.at(In::o_amount, 0) = price;
	in.at(In::o_kind, 1) = static_cast<int>(OptionKind::NO);
	in.at(In::option_mask, 0) = in.at(In::option_mask, 1) = 1;
	decision.answers = {1, 0};
	return decision;
}

void AlphaMapAI::showShipyardDialog(const IShipyard * obj)
{
	// A map shipyard a hero of ours visits (a town's boat is BUILD's SHIP): a yes/no on its embedding (the user,
	// 2026-11-04), YES builds the boat, when its water is free and its price ours. No query of the server's: held as
	// one (the turn waits for it), released however the decision ends
	const auto * object = dynamic_cast<const CGObjectInstance *>(obj->getObject());
	TResources price;
	obj->getBoatCost(price);
	if(!object || dynamic_cast<const CGTownInstance *>(object) || obj->shipyardStatus() != IBoatGenerator::GOOD || !cb->getResourceAmount().canAfford(price))
		return;
	const QueryID hold(-2);
	addQuery(hold, "a shipyard");
	const auto id = object->id;
	runAsync("a shipyard", [this, id, hold]()
	{
		struct Release
		{
			AlphaMapAI * ai;
			QueryID id;
			~Release()
			{
				std::scoped_lock lock(ai->statusMutex);
				ai->queries.erase(id);
				ai->statusChanged.notify_all();
			}
		} release{this, hold};
		const auto * yard = dynamic_cast<const IShipyard *>(cb->getObj(id, false));
		if(!yard)
			return;
		auto decision = encode(DecisionKind::YES_NO);
		encodeOptionContext(decision, activeHero, activeTarget);
		auto & in = decision.inputs;
		in.at(In::o_kind, 0) = static_cast<int>(OptionKind::YES);
		in.at(In::o_kind, 1) = static_cast<int>(OptionKind::NO);
		in.at(In::option_mask, 0) = in.at(In::option_mask, 1) = 1;
		decision.answers = {1, 0};
		if(decide(decision).index == 0)
		{
			cb->buildBoat(yard);
			++boatsBuilt;
		}
	});
}
