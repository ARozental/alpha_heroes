#include "StdInc.h"
#include "AlphaMapAI.h"

#ifdef ALPHA_MAP_NK2_ADVICE
#include "AIGateway.h" // Nullkiller2's (AI/Nullkiller2 is on this library's include path: CMakeLists.txt)

#include "lib/UnlockGuard.h"
#include "lib/battle/BattleAction.h"
#include "lib/callback/CCallback.h"
#include "lib/callback/IClient.h"
#include "lib/gameState/CGameState.h"
#include "lib/gameState/UpgradeInfo.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/mapping/TerrainTile.h"
#include "lib/networkPacks/PacksForServer.h"
#ifdef NK2_KEEPS_PLAN // (our fork's Nullkiller2: Nullkiller.h)
#include "Goals/Composition.h"
#include "Goals/ExecuteHeroChain.h"
#endif
#endif

#include <chrono>
#include <set>

using namespace MapSchema;

// Nullkiller2's advice: the teacher's labels on our own states (the user, 2026-10-31: "a real teacher", not a model
// imitating it). This player's AlphaMapAI keeps a Nullkiller2 of its own whose callback sends its requests to a client
// that sends nothing (AdviceClient). Before a sampled turn decision of ours, that Nullkiller2 makes its turn as it
// would from this state; its first request (what it would do now) is matched to our legal entry doing the same
// (matchRequest, as the teacher's recorder does) and stops it there. Its turn starts again each time: Nullkiller2
// plans from the state (and remembers the objects it can see: memorizeVisitableObjs), so it needs none of our events.
// Our decision's record carries it (RECORD_ADVICE); the trainer takes a loss toward it (campaign.nk2_advice).
// An exchange between two of our heroes gets Nullkiller2's as well (adviseExchange): the troop move toward the army it
// would leave the one hero, or the end of the exchange once that army is there.

namespace
{
std::string adviceEnvironment(const char * name, const std::string & fallback = "")
{
	const char * value = std::getenv(name);
	return value ? value : fallback;
}

#ifdef ALPHA_MAP_NK2_ADVICE
struct AdviceTaken
{
}; // (no std::exception: Nullkiller2's planning catches only its own kinds; its makeTurn's catch (...) ends there)

/// The client behind the advising Nullkiller2's callback: each request it would send is shown to `taken`, none is sent
class AdviceClient : public IClient
{
public:
	std::function<void(const CPackForServer &)> taken;

	std::optional<BattleAction> makeSurrenderRetreatDecision(PlayerColor, const BattleID &, const BattleStateInfoForRetreat &) override
	{
		return std::nullopt;
	}

	int sendRequest(const CPackForServer & request, PlayerColor, bool) override
	{
		if(taken)
			taken(request);
		throw AdviceTaken();
	}
};

#ifdef NK2_KEEPS_PLAN
/// Nullkiller2's plan (Nullkiller::lastPlan: its tasks, the best first) as our hero slots' targets (labels of the
/// foresight's, never inputs): in a hero chain each hero's last tile in it (the path's nodes run from the target back
/// to the start: a hero's first node there is the target for the hero that ends the chain, the meeting for one that
/// brings it an army; a Composition's chain alike); a hero whose task has no chain gets x -2, one in no task keeps -1.
/// A hero in two tasks keeps the better one's
PlanTargets planTargets(const Decision & decision, const NK2AI::Goals::TTaskVec & plan)
{
	PlanTargets out{};
	std::array<bool, OWN_HEROES> set{};
	const auto slot = [&decision](const CGHeroInstance * hero)
	{
		for(int i = 0; hero && i < OWN_HEROES; ++i)
			if(decision.heroes[i] == hero->id)
				return i;
		return -1;
	};
	const auto chain = [&](const NK2AI::Goals::ExecuteHeroChain & goal, float priority)
	{
		for(const auto & node : goal.getPath().nodes)
		{
			const int i = slot(node.targetHero);
			if(i < 0 || set[i])
				continue;
			set[i] = true;
			out[i] = {static_cast<int16_t>(node.coord.x), static_cast<int16_t>(node.coord.y), static_cast<int16_t>(node.coord.z), static_cast<int16_t>(node.turns), priority};
		}
	};
	for(const auto & task : plan)
	{
		bool chained = false;
		if(const auto * goal = dynamic_cast<const NK2AI::Goals::ExecuteHeroChain *>(task.get()))
		{
			chain(*goal, task->priority);
			chained = true;
		}
		else if(const auto * composition = dynamic_cast<const NK2AI::Goals::Composition *>(task.get()))
			for(const auto & step : composition->getSubtasks())
				for(const auto & sub : step)
					if(const auto * goal = dynamic_cast<const NK2AI::Goals::ExecuteHeroChain *>(sub.get()))
					{
						chain(*goal, task->priority);
						chained = true;
					}
		if(const int i = slot(task->getHero()); !chained && i >= 0 && !set[i])
		{
			set[i] = true;
			out[i].x = -2;
			out[i].priority = task->priority;
		}
	}
	return out;
}

/// Our MOVE and TRAVEL entries that walk Nullkiller2's plan (more entries that follow it beside its first request:
/// RECORD_ACCEPT, the teacher's targets): each hero's MOVE to any tile of its own part of a planned hero chain (a short
/// step along the path or a long one alike), its TRAVEL to an area the path reaches in a day or more
void planEntries(const Decision & decision, const NK2AI::Goals::TTaskVec & plan, const int3 & size, std::set<std::pair<int, int>> & out)
{
	const int area = 2 * std::max(1, (std::max(size.x, size.y) + MAP_GRID - 1) / MAP_GRID); // (Encoder's TRAVEL areas)
	const auto & travels = decision.entries[static_cast<int>(ActionType::TRAVEL)];
	const auto chain = [&](const NK2AI::Goals::ExecuteHeroChain & goal)
	{
		for(const auto & node : goal.getPath().nodes)
			for(int i = 0; i < OWN_HEROES; ++i)
			{
				if(!node.targetHero || decision.heroes[i] != node.targetHero->id)
					continue;
				const auto & center = decision.centers[i];
				const int row = node.coord.y - center.y + WINDOW_RADIUS;
				const int column = node.coord.x - center.x + WINDOW_RADIUS;
				if(node.coord != center && node.coord.z == center.z && row >= 0 && row < WINDOW && column >= 0 && column < WINDOW && decision.inputs.get(In::move_mask, i, row, column) != 0)
					out.insert({static_cast<int>(ActionType::MOVE), (i * WINDOW + row) * WINDOW + column});
				if(node.turns >= 1 && node.coord.z >= 0 && node.coord.z < MAP_LEVELS)
				{
					const int index = ((i * MAP_LEVELS + node.coord.z) * TRAVEL_GRID + std::min(node.coord.y / area, TRAVEL_GRID - 1)) * TRAVEL_GRID + std::min(node.coord.x / area, TRAVEL_GRID - 1);
					if(travels.count(index))
						out.insert({static_cast<int>(ActionType::TRAVEL), index});
				}
				break;
			}
	};
	for(const auto & task : plan)
	{
		if(const auto * goal = dynamic_cast<const NK2AI::Goals::ExecuteHeroChain *>(task.get()))
			chain(*goal);
		else if(const auto * composition = dynamic_cast<const NK2AI::Goals::Composition *>(task.get()))
			for(const auto & step : composition->getSubtasks())
				for(const auto & sub : step)
					if(const auto * goal = dynamic_cast<const NK2AI::Goals::ExecuteHeroChain *>(sub.get()))
						chain(*goal);
	}
}
#endif
#endif
}

#ifdef ALPHA_MAP_NK2_ADVICE
struct AlphaMapAI::Advisor
{
	AdviceClient client;
	std::shared_ptr<CCallback> callback;
	std::shared_ptr<NK2AI::AIGateway> nk2;
};
#else
struct AlphaMapAI::Advisor
{
};
#endif

void AlphaMapAI::initAdvice()
{
	adviceProbability = std::stod(adviceEnvironment("ALPHA_MAP_NK2_ADVICE", "0"));
#ifdef ALPHA_MAP_NK2_ADVICE
	if(adviceProbability <= 0)
		return;
	auto a = std::make_shared<Advisor>();
	// The game's state, shared, not owned (the client's): Nullkiller2 reads it through its own callback
	const auto & state = static_cast<const IGameInfoCallback &>(*cb).gameState();
	a->callback = std::make_shared<CCallback>(std::shared_ptr<CGameState>(const_cast<CGameState *>(&state), [](CGameState *) {}), playerID, &a->client);
	a->nk2 = std::make_shared<NK2AI::AIGateway>();
	a->nk2->initGameInterface(env, a->callback);
#ifdef NK2_KEEPS_PLAN
	a->nk2->nullkiller->keepPlan = true; // (its plan, made before its turn's first request: advise's labels)
#endif
	advisor = a;
	logAi->info("AlphaMapAI %s: Nullkiller2's advice at a chance of %.3f a turn decision", playerID.toString(), adviceProbability);
#else
	if(adviceProbability > 0)
		logAi->error("AlphaMapAI %s: ALPHA_MAP_NK2_ADVICE asked, but this build has no Nullkiller2: no advice", playerID.toString());
	adviceProbability = 0;
#endif
}

bool AlphaMapAI::adviceDrawn()
{
	if(!advisor)
		return false;
	std::uniform_real_distribution<double> uniform(0.0, 1.0);
	std::scoped_lock lock(rngMutex);
	return uniform(rng) < adviceProbability;
}

std::optional<Choice> AlphaMapAI::advise(const Decision & decision, std::optional<PlanTargets> & plan, std::vector<std::pair<int, int>> & accept)
{
#ifdef ALPHA_MAP_NK2_ADVICE
	if(!advisor)
		return std::nullopt;
	const auto started = std::chrono::steady_clock::now();
	int step = -1;
	if(auto visit = adviseVisit(decision, accept, step)) // (a hero of ours in a town of ours: its visit's habits first)
	{
		adviceSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
		++advised;
		++adviceMatched;
		++visitAdvised[step];
		return visit;
	}
	std::optional<Choice> advice;
	std::string unmatched = "no request";
	advisor->client.taken = [&](const CPackForServer & pack)
	{
		Choice choice;
		bool exact = true;
		unmatched.clear();
		// A move's label is where it is going: its moves go a tile a request (AIGateway::moveHeroToTile), and the first
		// tile, matched as a move there, taught our model to walk a tile a decision (2026-11-01: our move its tile in 5%
		// of its moves, ~60 decisions a player and day). Its plan sets the hero and the tile before moving
		// (Nullkiller::setActive in ExecuteHeroChain: the chain node's tile, possibly days away: a travel then)
		int3 intended(-1, -1, -1);
		if(const auto * move = dynamic_cast<const MoveHero *>(&pack))
			if(const auto * hero = cb->getHero(move->hid); hero && advisor->nk2->nullkiller->isActive(hero))
				intended = advisor->nk2->nullkiller->getTargetTile();
		if(matchRequest(decision, &pack, choice, exact, unmatched, intended))
			advice = choice;
	};
	try
	{
		// Its makeTurn takes the state's lock itself (shared), which this turn already holds: let go of it meanwhile
		// (our turn: nothing else changes the state)
		auto unlock = vstd::makeUnlockSharedGuard(CGameState::mutex);
		// Its paths, from the state as it is now: our moves since its last advice changed it, and the events that tell
		// it so go to us. Nullkiller::invalidatePathfinderData is not exported; AIGateway::newObject calls it (and
		// remembers the object, a hero of ours, as its own turn's start remembers every object it sees)
		const auto heroes = cb->getHeroesInfo();
		if(!heroes.empty())
			advisor->nk2->newObject(heroes.front());
		advisor->nk2->makeTurn();
	}
	catch(const AdviceTaken &)
	{
	}
	catch(const std::exception & e)
	{
		unmatched = std::string("an error: ") + e.what();
	}
	advisor->client.taken = nullptr;
#ifdef NK2_KEEPS_PLAN
	// Its plan, made before any request of this turn of its (keepPlan): our hero slots' targets, and our entries that
	// walk it (each hero's, whichever hero it would move first: the teacher's targets beside the advice's own)
	if(advisor->nk2->nullkiller->lastPlanValid)
	{
		plan = planTargets(decision, advisor->nk2->nullkiller->lastPlan);
		std::set<std::pair<int, int>> entries;
		planEntries(decision, advisor->nk2->nullkiller->lastPlan, cb->getMapSize(), entries);
		if(advice)
			entries.erase({advice->type, advice->index});
		accept.assign(entries.begin(), entries.end());
	}
#endif
	adviceSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
	++advised;
	if(advice)
		++adviceMatched;
	else
		++adviceUnmatched[unmatched.empty() ? std::string("unknown") : unmatched];
	return advice;
#else
	return std::nullopt;
#endif
}

std::optional<Choice> AlphaMapAI::adviseExchange(const Decision & decision, ObjectInstanceID first, ObjectInstanceID second, std::vector<std::pair<int, int>> & accept)
{
#ifdef ALPHA_MAP_NK2_ADVICE
	const auto * one = cb->getHero(first);
	const auto * two = cb->getHero(second);
	if(!advisor || !one || !two || one->tempOwner != two->tempOwner)
		return std::nullopt;
	// Nullkiller2's exchange (AIGateway::heroExchangeStarted, pickBestCreatures): one hero gets the best army of the
	// two (adviseToward). Which hero: the higher level, else the stronger army, else Nullkiller2's own (the one met: its
	// hero chains, a scout bringing troops to its main hero). Its own alone gave the army to whichever hero our model
	// walked into, which it does with any two heroes: the army went back and forth between two heroes, 2026-11-01
	const auto started = std::chrono::steady_clock::now();
	const auto * destination = activeHero == first ? two : one;
	if(one->level != two->level)
		destination = one->level > two->level ? one : two;
	else if(one->getArmyStrength() != two->getArmyStrength())
		destination = one->getArmyStrength() > two->getArmyStrength() ? one : two;
	const auto * source = destination == one ? two : one;
	bool needed = false;
	std::optional<Choice> advice = adviseToward(decision, destination, source, needed, accept);
	if(!advice && !needed)
	{
		// Its artifacts (pickBestArtifacts, after the army): the hero that got the army gets each of the other's artifacts
		// that fits a free slot of its (and swaps in better ones by its score, not exported: left out). Ours: a GIVE from
		// the other hero landing in a worn slot (getArtAnyPosition's: the slot is free), the first one advised, the rest
		// accepted; END only once none is left (the END it advised before taught our model to keep its artifacts apart)
		for(const auto & [index, e] : decision.entries[static_cast<int>(ActionType::ARTIFACT)])
			if(e.first == source->id && e.second == destination->id && e.c >= 0 && e.c < N_WORN_SLOTS)
			{
				if(advice)
					accept.emplace_back(static_cast<int>(ActionType::ARTIFACT), index);
				else
				{
					Choice choice;
					choice.type = static_cast<int>(ActionType::ARTIFACT);
					choice.index = index;
					advice = choice;
					++exchangeAdviceArtifacts;
				}
			}
	}
	if(!advice && !needed)
	{
		Choice choice;
		choice.type = static_cast<int>(ActionType::END);
		choice.index = 0; // (as the end of a turn's advice: matchRequest)
		advice = choice;
		++exchangeAdviceEnds;
	}
	adviceSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
	++advised;
	++exchangeAdvised;
	if(advice)
	{
		++adviceMatched;
		++exchangeAdviceMatched;
	}
	else
		++adviceUnmatched["exchange (no move of ours toward its army)"];
	return advice;
#else
	return std::nullopt;
#endif
}

std::optional<Choice> AlphaMapAI::adviseToward(const Decision & decision, const CGHeroInstance * destination, const CArmedInstance * source, bool & needed, std::vector<std::pair<int, int>> & accept)
{
	needed = false;
#ifdef ALPHA_MAP_NK2_ADVICE
	// The best army of the two for the hero (ArmyManager::getBestArmy, as its pickBestCreatures; the source keeps its
	// fastest creature, one, when it would have none). Its own requests for it are not matched as they come: most are
	// swaps into a slot of the order it wants or moves within one army, none of them ours (a move of ours takes the
	// creature's own slot or the first free one), so the advice is our legal move toward its army: the missing troops
	// of its strongest creature still missing from the source, else troops it would not keep back to the source.
	// (Its dismissal of small unwanted stacks and its artifacts are not ours.)
	const CArmedInstance * hero = destination;
	std::vector<NK2AI::SlotInfo> best;
	if(source->stacksCount() > 0)
		best = advisor->nk2->nullkiller->armyManager->getBestArmy(destination, destination, source, cb->getTile(source->visitablePos())->getTerrainID());
	std::map<CreatureID, int> wanted, has;
	for(const auto & slot : best)
		wanted[slot.creature->getId()] += slot.count;
	for(const auto & [slot, stack] : destination->Slots())
		has[stack->getId()] += stack->getCount();
	const auto & moves = decision.entries[static_cast<int>(ActionType::ARMY)];
	// The move of `from`'s troops of `creature` to `to` whose amount is nearest `amount` (on a log scale); -1 none
	const auto nearestMove = [&moves](const CArmedInstance * from, const CArmedInstance * to, CreatureID creature, int amount)
	{
		int found = -1;
		double gap = 1e18;
		for(const auto & [index, e] : moves)
		{
			if(e.first != from->id || e.second != to->id || from->getCreature(SlotID(e.a)) == nullptr || from->getCreature(SlotID(e.a))->getId() != creature)
				continue;
			const double g = std::abs(std::log(std::max(1, e.c)) - std::log(std::max(1, amount)));
			if(g < gap || (g == gap && index < found))
			{
				gap = g;
				found = index;
			}
		}
		return found;
	};
	int move = -1;
	for(const auto & slot : best) // (strongest first)
	{
		const auto creature = slot.creature->getId();
		const int missing = wanted[creature] - has[creature];
		if(missing <= 0)
			continue;
		needed = true;
		move = nearestMove(source, hero, creature, missing);
		if(move >= 0)
			break;
	}
	if(move < 0)
		for(const auto & [creature, count] : has)
		{
			const int over = count - wanted[creature];
			if(over <= 0)
				continue;
			needed = true;
			move = nearestMove(hero, source, creature, over);
			if(move >= 0)
				break;
		}
	// Every other move of ours toward its army (the teacher's targets beside the advice's own: any order of the moves
	// reaches it): troops of a creature it wants more of from the source, no more than are missing; troops it would
	// not keep back to the source, no more than the excess
	for(const auto & [index, e] : moves)
	{
		if(index == move || (e.first != source->id && e.first != hero->id) || (e.second != source->id && e.second != hero->id) || e.first == e.second)
			continue;
		const auto * from = e.first == source->id ? source : hero;
		if(from->getCreature(SlotID(e.a)) == nullptr)
			continue;
		const auto creature = from->getCreature(SlotID(e.a))->getId();
		const int missing = wanted[creature] - has[creature];
		if((from == source && missing > 0 && e.c <= missing) || (from == hero && missing < 0 && e.c <= -missing))
			accept.emplace_back(static_cast<int>(ActionType::ARMY), index);
	}
	if(move >= 0)
	{
		Choice choice;
		choice.type = static_cast<int>(ActionType::ARMY);
		choice.index = move;
		return choice;
	}
#endif
	return std::nullopt;
}

std::optional<Choice> AlphaMapAI::adviseVisit(const Decision & decision, std::vector<std::pair<int, int>> & accept, int & step)
{
	step = -1;
#ifdef ALPHA_MAP_NK2_ADVICE
	// Nullkiller2's visit to a town of its own (AIGateway::objectVisited; buildArmyIn alike), in its order: its hero's
	// stacks upgraded where its free resources pay for the best upgrade by AI value (makePossibleUpgrades), the town's
	// troops it wants gathered into the hero (moveCreaturesToHero: pickBestCreatures), a spellbook bought for a main
	// hero without one (its free gold, a mage guild). All of it is a part of the visit and none of its turn's requests:
	// the advice of its turn's first request never said so, and at 0.6-0.8 of the loss our model came to do none of it
	// (2026-11-05, 5,509 turn decisions with a hero of ours in a town of ours: ARMY town -> hero P 0.0002, our heroes
	// leading 2% of our army at day 60; BUY legal in 4,219, chosen 0; UPGRADE legal in 1,689, chosen 26). So a hero of
	// ours visiting a town of ours gets those first, each our entry doing the same (step: 0 upgrade, 1 gather, 2 book)
	const auto & nk = *advisor->nk2->nullkiller;
	const auto & upgrades = decision.entries[static_cast<int>(ActionType::UPGRADE)];
	const auto & buys = decision.entries[static_cast<int>(ActionType::BUY)];
	const auto chosen = [](ActionType type, int index)
	{
		Choice choice;
		choice.type = static_cast<int>(type);
		choice.index = index;
		return choice;
	};
	for(const auto * town : cb->getTownsInfo(true))
	{
		const auto * hero = town->getVisitingHero();
		if(!hero || hero->tempOwner != playerID)
			continue;
		auto free = cb->getResourceAmount() - nk.getLockedResources(); // (Nullkiller::getFreeResources, not exported: its body)
		free.positive();
		for(int s = 0; s < ARMY_SLOTS; ++s)
		{
			const auto * stack = hero->getStackPtr(SlotID(s));
			if(!stack)
				continue;
			UpgradeInfo info(stack->getId());
			cb->fillUpgradeInfo(hero, SlotID(s), info);
			if(!info.hasUpgrades())
				continue;
			const CreatureID best = *vstd::maxElementByFun(info.getAvailableUpgrades(), [](const CreatureID & id) { return id.toCreature()->getAIValue(); });
			if(best.toCreature()->getAIValue() <= stack->getCreature()->getAIValue() || !free.canAfford(info.getUpgradeCostsFor(best) * stack->getCount()))
				continue;
			for(const auto & [index, e] : upgrades)
				if(e.first == hero->id && e.a == s)
				{
					step = 0;
					return chosen(ActionType::UPGRADE, index);
				}
		}
		if(town->armedGarrison())
		{
			bool needed = false;
			std::vector<std::pair<int, int>> others;
			if(auto advice = adviseToward(decision, hero, town->getUpperArmy(), needed, others))
			{
				accept = std::move(others);
				step = 1;
				return advice;
			}
		}
		if(nk.heroManager->getHeroRoleOrDefaultInefficient(hero) == NK2AI::HeroRole::MAIN && !hero->hasSpellbook() && free[EGameResID::GOLD] >= GameConstants::SPELLBOOK_GOLD_COST)
			for(const auto & [index, e] : buys)
				if(e.first == hero->id && e.a == ArtifactID(ArtifactID::SPELLBOOK).getNum())
				{
					step = 2;
					return chosen(ActionType::BUY, index);
				}
	}
#endif
	return std::nullopt;
}

std::string AlphaMapAI::adviceJson() const
{
	if(advised == 0)
		return "";
	std::string unmatched;
	for(const auto & [kind, count] : adviceUnmatched)
		unmatched += (unmatched.empty() ? "\"" : ", \"") + kind + "\": " + std::to_string(count);
	return ", \"advice\": {\"asked\": " + std::to_string(advised) + ", \"matched\": " + std::to_string(adviceMatched) + ", \"exchange\": {\"asked\": " + std::to_string(exchangeAdvised) + ", \"matched\": " + std::to_string(exchangeAdviceMatched) + ", \"end\": " + std::to_string(exchangeAdviceEnds) + ", \"artifacts\": " + std::to_string(exchangeAdviceArtifacts) + "}, \"visit\": {\"upgrade\": " + std::to_string(visitAdvised[0]) + ", \"gather\": " + std::to_string(visitAdvised[1]) + ", \"book\": " + std::to_string(visitAdvised[2]) + "}, \"seconds\": " + std::to_string(adviceSeconds) + ", \"unmatched\": {" + unmatched + "}}";
}

// The events of a seated Nullkiller2's memory (AlphaMapAI.h), passed on to the advising one; none of our own uses them
#ifdef ALPHA_MAP_NK2_ADVICE
#define ALPHA_ADVISE(call) \
	if(advisor) \
		advisor->nk2->call;
#else
#define ALPHA_ADVISE(call)
#endif

void AlphaMapAI::objectRemoved(const CGObjectInstance * obj, const PlayerColor & initiator)
{
	ALPHA_ADVISE(objectRemoved(obj, initiator))
}

void AlphaMapAI::newObject(const CGObjectInstance * obj)
{
	ALPHA_ADVISE(newObject(obj))
}

void AlphaMapAI::tileRevealed(const FowTilesType & pos)
{
	ALPHA_ADVISE(tileRevealed(pos))
}

void AlphaMapAI::tileHidden(const FowTilesType & pos)
{
	ALPHA_ADVISE(tileHidden(pos))
}

void AlphaMapAI::objectPropertyChanged(const SetObjectProperty * sop)
{
	ALPHA_ADVISE(objectPropertyChanged(sop))
}

void AlphaMapAI::heroCreated(const CGHeroInstance * hero)
{
	ALPHA_ADVISE(heroCreated(hero))
}
