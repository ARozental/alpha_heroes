#include "StdInc.h"
#include "AlphaMapAI.h"

#ifdef ALPHA_MAP_NK2_ADVICE
#include "AIGateway.h" // Nullkiller2's (AI/Nullkiller2 is on this library's include path: CMakeLists.txt)

#include "lib/UnlockGuard.h"
#include "lib/battle/BattleAction.h"
#include "lib/callback/CCallback.h"
#include "lib/callback/IClient.h"
#include "lib/gameState/CGameState.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/mapping/TerrainTile.h"
#include "lib/networkPacks/PacksForServer.h"
#ifdef NK2_KEEPS_PLAN // (our fork's Nullkiller2: Nullkiller.h)
#include "Analyzers/DangerHitMapAnalyzer.h"
#include "Engine/PriorityEvaluator.h"
#include "Goals/Composition.h"
#include "Goals/ExecuteHeroChain.h"
#include "lib/mapObjects/CGTownInstance.h"
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

// ---- Schema 19: Nullkiller2's candidate tasks as inputs (candidateInputs) ----

/// One candidate task of Nullkiller2's in our terms (a row of k_*)
struct CandidateRow
{
	int slot = -1;
	int3 dest;
	int destCell = -1;
	int destArea = -1;
	int move = -1;
	int travel = -1;
	double score[2] = {0, 0};
	double days = 0;
	double loss = 0;
	double danger = 0;
	int kind = 6;
	int flags = 0;
};

constexpr int AREAS = MAP_LEVELS * TRAVEL_GRID * TRAVEL_GRID;

/// k_kind from what Nullkiller2 scores the task by: 0 visit or capture, 1 attack an enemy hero, 2 attack a town, 3 defend
/// our town, 4 meet our hero, 5 explore, 6 other
int candidateKind(const NK2AI::EvaluationContext & context)
{
	if(context.isDefend)
		return 3;
	if(context.isExchange)
		return 4;
	if(context.isHero && context.isEnemy)
		return 1;
	if(context.conquestValue > 0)
		return 2;
	if(context.explorePriority > 0) // (set by ExplorePointEvaluator alone, which also gives every such point a strategical value)
		return 5;
	const bool reward = context.goldReward > 0 || context.armyReward > 0 || context.skillReward > 0 || context.strategicalValue > 0 || context.armyGrowth > 0;
	return reward ? 0 : 6;
}

/// The cell (window slot x W x W + row x W + column) of `tile` in the first window of our hero token `hero` showing it,
/// legal or not; -1 none
int windowCell(const Decision & decision, int hero, const int3 & tile)
{
	for(int w = 0; w < OWN_HEROES; ++w)
	{
		if(decision.windowHero[w] != hero)
			continue;
		const auto & center = decision.centers[w];
		const int row = tile.y - center.y + WINDOW_RADIUS;
		const int column = tile.x - center.x + WINDOW_RADIUS;
		if(tile.z == center.z && row >= 0 && row < WINDOW && column >= 0 && column < WINDOW)
			return (w * WINDOW + row) * WINDOW + column;
	}
	return -1;
}

/// The TRAVEL area (level x TG x TG + row x TG + column) a tile is in, -1 off the levels; side: an area's tiles a side
int areaOf(const int3 & tile, int side)
{
	if(tile.z < 0 || tile.z >= MAP_LEVELS || tile.x < 0 || tile.y < 0)
		return -1;
	return (tile.z * TRAVEL_GRID + std::min(tile.y / side, TRAVEL_GRID - 1)) * TRAVEL_GRID + std::min(tile.x / side, TRAVEL_GRID - 1);
}

int64_t fixedPoint(double value, double scale, double most = 2147483647.0)
{
	return static_cast<int64_t>(std::llround(std::clamp(value * scale, -most, most)));
}

/// Nullkiller2's planned next step as our entry, for a decision where its first request matches nothing of ours: the first
/// task of its plan (best first) whose first step maps to an entry, as candidateInputs maps one (the destination's own
/// MOVE entry when legal today, else the path's stop today, else the TRAVEL toward it); none when no task's does
std::optional<Choice> planFirstStep(const Decision & decision, const NK2AI::Goals::TTaskVec & plan, const int3 & size)
{
	const int side = 2 * std::max(1, (std::max(size.x, size.y) + MAP_GRID - 1) / MAP_GRID);
	const auto & travels = decision.entries[static_cast<int>(ActionType::TRAVEL)];
	for(const auto & task : plan)
	{
		const NK2AI::Goals::ExecuteHeroChain * chain = dynamic_cast<const NK2AI::Goals::ExecuteHeroChain *>(task.get());
		if(!chain)
			if(const auto * composition = dynamic_cast<const NK2AI::Goals::Composition *>(task.get()))
				for(const auto & step : composition->getSubtasks())
					for(const auto & sub : step)
						if(!chain)
							chain = dynamic_cast<const NK2AI::Goals::ExecuteHeroChain *>(sub.get());
		if(!chain || chain->getPath().nodes.empty())
			continue;
		const auto & nodes = chain->getPath().nodes;
		const CGHeroInstance * hero = task->getHero() ? task->getHero() : nodes.front().targetHero;
		int slot = -1;
		for(int i = 0; hero && i < OWN_HEROES; ++i)
			if(decision.heroes[i] == hero->id)
				slot = i;
		size_t own = 0;
		while(own < nodes.size() && nodes[own].targetHero != hero)
			++own;
		if(slot < 0 || own == nodes.size())
			continue;
		int move = nodes[own].turns == 0 ? decision.moveEntry(slot, nodes[own].coord) : -1;
		for(size_t k = own; k < nodes.size() && move < 0; ++k)
			if(nodes[k].targetHero == hero && nodes[k].turns == 0)
			{
				move = decision.moveEntry(slot, nodes[k].coord);
				break;
			}
		Choice choice;
		if(move >= 0)
		{
			choice.type = static_cast<int>(ActionType::MOVE);
			choice.index = move;
			return choice;
		}
		for(size_t k = own; k < nodes.size(); ++k)
		{
			if(nodes[k].targetHero != hero || nodes[k].turns < 1)
				continue;
			if(const int a = areaOf(nodes[k].coord, side); a >= 0 && travels.count(slot * AREAS + a))
			{
				choice.type = static_cast<int>(ActionType::TRAVEL);
				choice.index = slot * AREAS + a;
				return choice;
			}
		}
	}
	return std::nullopt;
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
				// (any of the hero's windows: a tile of its path past a gate is a MOVE's target since schema 17)
				if(const int entry = decision.moveEntry(i, node.coord); entry >= 0)
					out.insert({static_cast<int>(ActionType::MOVE), entry});
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
	adviceFixes = adviceEnvironment("ALPHA_MAP_ADVICE_FIXES", "1") != "0";
	gatherAdvice = adviceEnvironment("ALPHA_MAP_GATHER_ADVICE", "0") != "0";
	nk2Inputs = adviceEnvironment("ALPHA_MAP_NK2_INPUTS", "0") != "0";
	followAdvice = std::clamp(std::stod(adviceEnvironment("ALPHA_MAP_FOLLOW_ADVICE", "0")), 0.0, 1.0);
#ifdef ALPHA_MAP_NK2_ADVICE
	if(adviceProbability <= 0 && !nk2Inputs)
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
#ifdef NK2_ADVICE_FIXES
	if(adviceFixes)
	{
		a->nk2->nullkiller->adviceFixes = true;
		a->nk2->nullkiller->exchangeSettled = [this](const CGHeroInstance * one, const CGHeroInstance * two) { return exchangeSettled(one, two); };
	}
#else
	adviceFixes = false;
#endif
	advisor = a;
	logAi->info("AlphaMapAI %s: Nullkiller2's advice at a chance of %.3f a turn decision, meeting fixes %s, its view as inputs %s", playerID.toString(), adviceProbability, adviceFixes ? "on" : "off", nk2Inputs ? "on" : "off");
#else
	if(adviceProbability > 0 || nk2Inputs)
		logAi->error("AlphaMapAI %s: ALPHA_MAP_NK2_ADVICE or ALPHA_MAP_NK2_INPUTS asked, but this build has no Nullkiller2: none", playerID.toString());
	adviceProbability = 0;
	nk2Inputs = false;
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
	std::optional<Choice> advice;
	std::string unmatched = "no request";
	const auto started = std::chrono::steady_clock::now();
	bool gathers = false; // (its advice is a town's visit again: our GATHER, nothing accepted beside it)
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
		// Its visit again of the town its hero stands in (gatherAdvice): our GATHER there, see visitAgain
		if(const auto * move = dynamic_cast<const MoveHero *>(&pack); move && gatherAdvice)
			if(const int index = visitAgain(decision, *move, intended); index >= 0)
			{
				choice.type = static_cast<int>(ActionType::MULTI);
				choice.index = index;
				advice = choice;
				gathers = true;
				return;
			}
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
	adviceFallback = false;
	if(advisor->nk2->nullkiller->lastPlanValid)
	{
		if(!advice) // (its first request is nothing of ours: its planned next step that is, else it asks the same again)
			if((advice = planFirstStep(decision, advisor->nk2->nullkiller->lastPlan, cb->getMapSize())))
			{
				adviceFallback = true;
				++adviceFallbacks;
			}
		plan = planTargets(decision, advisor->nk2->nullkiller->lastPlan);
		std::set<std::pair<int, int>> entries;
		planEntries(decision, advisor->nk2->nullkiller->lastPlan, cb->getMapSize(), entries);
		if(advice)
			entries.erase({advice->type, advice->index});
		accept.assign(entries.begin(), entries.end());
	}
#endif
	acceptMulti(decision, advice, accept);
	if(gathers)
	{
		// The plan's later steps are this hero's way out of the town, and a set's loss is satisfied by any of its entries:
		// accepted beside the GATHER they would teach the hero to leave without the troops (the order matters here, as
		// with a town's purchases before its visitor leaves: train.py's TEACHER_SPAN)
		accept.clear();
		++adviceGathers;
	}
	adviceSeconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
	++advised;
	if(advice && !adviceFallback)
		++adviceMatched;
	else // (a fallback's miss counted as one all the same: what its first request was)
		++adviceUnmatched[unmatched.empty() ? std::string("unknown") : unmatched];
	return advice;
#else
	return std::nullopt;
#endif
}

int AlphaMapAI::visitAgain(const Decision & decision, const MoveHero & move, const int3 & intended) const
{
#ifdef ALPHA_MAP_NK2_ADVICE
	// Nullkiller2's request to move a hero onto the tile it stands on, in a town of ours: it visits the town again.
	// Its visit takes the garrison's best troops into the hero (AIGateway::performObjectInteraction: moveCreaturesToHero),
	// which its plan counted on when it joined the hero with the town's garrison (TownGarrisonActor). None of that is a
	// request, so the visit was its last word, matched to nothing ("MoveHero (into a teleport)": 752 of its 10,781 advice
	// calls in 8 joint7 games, 7.0%; a GATHER was legal at 3,621 of their 7,577 turn decisions and our model chose it
	// once). For us a visit takes nothing: the GATHER entry of that town and hero is what its request does (as the
	// recorders match its bulk move, Match.cpp; ours moves what fits, its own the best seven stacks). -1: not such a
	// request, or no GATHER of ours there (nothing to move)
	const auto * hero = cb->getHero(move.hid);
	if(!advisor || !hero || hero->tempOwner != playerID || move.path.empty())
		return -1;
	const int3 here = hero->visitablePos();
	if((intended.isValid() && intended != here) || hero->convertToVisitablePos(move.path.back()) != here)
		return -1;
	const auto * town = hero->getVisitedTown();
	if(!town || town->tempOwner != playerID || town->getVisitingHero() != hero)
		return -1;
	// (a hero in the garrison: the upper army is that hero's, which Nullkiller2 leaves alone when it has locked the hero
	// for the town's defence; its locks are not exported, and stripping a garrisoned hero is no label to guess at)
	if(town->getGarrisonHero())
		return -1;
	for(const auto & [index, e] : decision.entries[static_cast<int>(ActionType::MULTI)])
		if(static_cast<MultiKind>(e.a) == MultiKind::GATHER && e.second == hero->id)
			return index;
#endif
	return -1;
}

void AlphaMapAI::acceptMulti(const Decision & decision, const std::optional<Choice> & advice, std::vector<std::pair<int, int>> & accept) const
{
	// The multi-actions that do what Nullkiller2 does over its next requests, accepted beside its own label (the user,
	// 2026-11-05: a multi-action chosen must not cost the teacher's loss when the teacher would do the same): all a town
	// has for hire when its advice recruits there (its BuyArmy recruits each level it can pay for, a request each, into
	// the town or the hero visiting it), and a town's garrison into the hero of ours visiting it (its moveCreaturesToHero,
	// on every visit, none of them a request)
	ObjectInstanceID recruiting;
	if(advice && advice->type == static_cast<int>(ActionType::RECRUIT))
		if(const auto found = decision.entries[static_cast<int>(ActionType::RECRUIT)].find(advice->index); found != decision.entries[static_cast<int>(ActionType::RECRUIT)].end())
			recruiting = found->second.first;
	std::vector<std::pair<int, int>> multi; // (first in the list: the trainer keeps a decision's first SET_SIZE, the plan's steps after these)
	for(const auto & [index, e] : decision.entries[static_cast<int>(ActionType::MULTI)])
	{
		const auto kind = static_cast<MultiKind>(e.a);
		const bool recruits = (kind == MultiKind::RECRUIT_ALL || kind == MultiKind::RECRUIT_ALL_HERO) && recruiting != ObjectInstanceID() && e.first == recruiting;
		if(recruits || kind == MultiKind::GATHER)
			multi.emplace_back(static_cast<int>(ActionType::MULTI), index);
	}
	accept.insert(accept.begin(), multi.begin(), multi.end());
}

std::optional<Choice> AlphaMapAI::adviseExchange(const Decision & decision, ObjectInstanceID first, ObjectInstanceID second, std::vector<std::pair<int, int>> & accept)
{
#ifdef ALPHA_MAP_NK2_ADVICE
	const auto * one = cb->getHero(first);
	const auto * two = cb->getHero(second);
	if(!advisor || !one || !two || one->tempOwner != two->tempOwner)
		return std::nullopt;
	// Nullkiller2's exchange (AIGateway::heroExchangeStarted, pickBestCreatures): one hero gets the best army of the
	// two (ArmyManager::getBestArmy; the other keeps its fastest creature, one, when it would have none). Its own
	// requests for it are not matched as they come: most are swaps into a slot of the order it wants or moves within
	// one army, none of them ours (a move of ours takes the creature's own slot or the first free one), so the advice is
	// our legal move toward its army: the missing troops of its strongest creature still missing from the source, else
	// troops it would not keep back to the source, else (its army is there) the end. (Its dismissal of small unwanted
	// stacks and its artifacts are not ours.)
	// Which hero. With the meeting fixes (adviceFixes): the hero met when our move there was its advice (its own
	// exchanges' rule, AIGateway::heroExchangeStarted: the hero it moves gives; its deliveries bring the army to the hero
	// met), else the hero its plans gather armies into: its MAIN role (GatherArmyBehavior delivers to those alone), then
	// its evaluateHero, the level, the army, else the hero met. Without them: the higher level, else the stronger army,
	// else the hero met: the visitor in 39% of the meetings it advised, against the delivery it planned, which it then
	// advised again (2026-11-09 audit). The hero met alone gave the army to whichever hero our model walked into, which it
	// does with any two heroes: the army went back and forth between two heroes, 2026-11-01
	const auto started = std::chrono::steady_clock::now();
	const auto * destination = activeHero == first ? two : one;
	if(!adviceFixes)
	{
		if(one->level != two->level)
			destination = one->level > two->level ? one : two;
		else if(one->getArmyStrength() != two->getArmyStrength())
			destination = one->getArmyStrength() > two->getArmyStrength() ? one : two;
	}
	else if(!meetingAdvised || (activeHero != first && activeHero != second))
	{
		const auto & heroes = *advisor->nk2->nullkiller->heroManager;
		const auto rank = [&heroes](const CGHeroInstance * hero)
		{
			return std::make_tuple(heroes.getHeroRoleOrDefaultInefficient(hero) == NK2AI::HeroRole::MAIN, heroes.evaluateHero(hero), hero->level, hero->getArmyStrength());
		};
		if(rank(one) != rank(two))
			destination = rank(one) > rank(two) ? one : two;
	}
	const auto * source = destination == one ? two : one;
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
	const auto nearestMove = [&moves](const CGHeroInstance * from, const CGHeroInstance * to, CreatureID creature, int amount)
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
	bool needed = false;
	int move = -1;
	for(const auto & slot : best) // (strongest first)
	{
		const auto creature = slot.creature->getId();
		const int missing = wanted[creature] - has[creature];
		if(missing <= 0)
			continue;
		needed = true;
		move = nearestMove(source, destination, creature, missing);
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
			move = nearestMove(destination, source, creature, over);
			if(move >= 0)
				break;
		}
	// Every other move of ours toward its army (the teacher's targets beside the advice's own: any order of the moves
	// reaches it): troops of a creature it wants more of from the source, no more than are missing; troops it would
	// not keep back to the source, no more than the excess
	for(const auto & [index, e] : moves)
	{
		if(index == move || (e.first != source->id && e.first != destination->id) || (e.second != source->id && e.second != destination->id) || e.first == e.second)
			continue;
		const auto * from = e.first == source->id ? source : destination;
		if(from->getCreature(SlotID(e.a)) == nullptr)
			continue;
		const auto creature = from->getCreature(SlotID(e.a))->getId();
		const int missing = wanted[creature] - has[creature];
		if((from == source && missing > 0 && e.c <= missing) || (from == destination && missing < 0 && e.c <= -missing))
			accept.emplace_back(static_cast<int>(ActionType::ARMY), index);
	}
	std::optional<Choice> advice;
	if(move >= 0)
	{
		Choice choice;
		choice.type = static_cast<int>(ActionType::ARMY);
		choice.index = move;
		advice = choice;
	}
	// TAKE_BEST to the hero that gets the army, while Nullkiller2 still moves troops to it: its exchange's whole (its
	// pickBestCreatures, ArmyManager::getBestArmy: Armies.h's takeBest) in one decision (the user, 2026-11-05: a
	// multi-action chosen must not cost the teacher's loss when the teacher does the same in its next actions; schema
	// 14's GIVE_ALL, all of the army, was its only approximation)
	if(needed)
		for(const auto & [index, e] : decision.entries[static_cast<int>(ActionType::MULTI)])
			if(static_cast<MultiKind>(e.a) == MultiKind::TAKE_BEST && e.second == destination->id)
				accept.emplace_back(static_cast<int>(ActionType::MULTI), index);
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

#ifdef ALPHA_MAP_NK2_ADVICE
namespace
{
std::map<CreatureID, int> armyCounts(const CGHeroInstance & hero)
{
	std::map<CreatureID, int> counts;
	for(const auto & [slot, stack] : hero.Slots())
		counts[stack->getId()] += stack->getCount();
	return counts;
}
}
#endif

void AlphaMapAI::settleExchange(ObjectInstanceID first, ObjectInstanceID second)
{
#ifdef ALPHA_MAP_NK2_ADVICE
	const auto * one = cb->getHero(std::min(first, second));
	const auto * two = cb->getHero(std::max(first, second));
	if(!adviceFixes || !advisor || !one || !two) // (a garrison's dialog: no pair of heroes)
		return;
	std::unique_lock lock(settledMutex);
	settled[{one->id, two->id}] = {armyCounts(*one), armyCounts(*two)};
#endif
}

bool AlphaMapAI::exchangeSettled(const CGHeroInstance * one, const CGHeroInstance * two) const
{
#ifdef ALPHA_MAP_NK2_ADVICE
	if(two->id < one->id)
		std::swap(one, two);
	std::shared_lock lock(settledMutex);
	const auto found = settled.find({one->id, two->id});
	return found != settled.end() && found->second.first == armyCounts(*one) && found->second.second == armyCounts(*two);
#else
	return false;
#endif
}

std::string AlphaMapAI::adviceJson() const
{
	if(advised == 0)
		return "";
	std::string unmatched;
	for(const auto & [kind, count] : adviceUnmatched)
		unmatched += (unmatched.empty() ? "\"" : ", \"") + kind + "\": " + std::to_string(count);
	return ", \"advice\": {\"asked\": " + std::to_string(advised) + ", \"matched\": " + std::to_string(adviceMatched) + ", \"fallback\": " + std::to_string(adviceFallbacks) + ", \"exchange\": {\"asked\": " + std::to_string(exchangeAdvised) + ", \"matched\": " + std::to_string(exchangeAdviceMatched) + ", \"end\": " + std::to_string(exchangeAdviceEnds) + "}, \"gathers\": " + std::to_string(adviceGathers) + ", \"seconds\": " + std::to_string(adviceSeconds) + ", \"unmatched\": {" + unmatched + "}}";
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

// Schema 19 (the user's go-ahead 2026-11-11: Nullkiller2's insight as inputs, its labels no longer a loss): the
// advisor's view of this decision, from the turn it just planned (advise, before our forward): its candidate tasks
// (Nullkiller::lastCandidates, scored at its plan's tier) mapped onto our entries, NK_PER_HERO a hero of ours in the
// hero's block of rows, its best first; the tier; its threats to our towns and heroes (DangerHitMapAnalyzer). A task's
// first step today is the destination's own MOVE entry when it is legal, else the path's stop today when that is legal:
// never another tile of the path (2026-11-01: a tile a request taught our model to walk a tile a decision)
void AlphaMapAI::candidateInputs(Decision & decision)
{
#if defined(ALPHA_MAP_NK2_ADVICE) && defined(NK2_KEEPS_PLAN)
	if(!advisor || !advisor->nk2->nullkiller->lastPlanValid)
		return;
	const auto & nk = *advisor->nk2->nullkiller;
	auto & in = decision.inputs;
	const int3 size = cb->getMapSize();
	const int side = 2 * std::max(1, (std::max(size.x, size.y) + MAP_GRID - 1) / MAP_GRID);
	const auto & travels = decision.entries[static_cast<int>(ActionType::TRAVEL)];
	const int tier = nk.lastPlanTier;
	const auto slotOf = [&decision](const CGHeroInstance * hero)
	{
		for(int i = 0; hero && i < OWN_HEROES; ++i)
			if(decision.heroes[i] == hero->id)
				return i;
		return -1;
	};
	auto evaluator = nk.priorityEvaluators->acquire();
	std::array<std::vector<CandidateRow>, OWN_HEROES> rows;
	int count = 0;
	for(const auto & goal : nk.lastCandidates)
	{
		auto * task = goal->asTask();
		if(!task)
			continue;
		if(task->priority > 0)
			++count;
		const NK2AI::Goals::ExecuteHeroChain * chain = dynamic_cast<const NK2AI::Goals::ExecuteHeroChain *>(goal.get());
		bool composed = false;
		if(!chain)
			if(const auto * composition = dynamic_cast<const NK2AI::Goals::Composition *>(goal.get()))
				for(const auto & step : composition->getSubtasks())
					for(const auto & sub : step)
						if(!chain)
						{
							chain = dynamic_cast<const NK2AI::Goals::ExecuteHeroChain *>(sub.get());
							composed = true;
						}
		if(!chain || chain->getPath().nodes.empty())
			continue;
		const auto & nodes = chain->getPath().nodes; // (from the destination back to the start)
		const CGHeroInstance * hero = task->getHero() ? task->getHero() : nodes.front().targetHero; // (a composition's: none of its own)
		const int slot = slotOf(hero);
		if(slot < 0)
			continue;
		CandidateRow row;
		row.slot = slot;
		size_t own = 0; // the hero's last node in the chain: its destination (planTargets')
		while(own < nodes.size() && nodes[own].targetHero != hero)
			++own;
		if(own == nodes.size())
			continue;
		row.dest = nodes[own].coord;
		row.destCell = windowCell(decision, slot, row.dest);
		const int area = areaOf(row.dest, side);
		row.destArea = area < 0 ? -1 : slot * AREAS + area;
		bool jump = false, heroes = composed;
		for(size_t k = 0; k + 1 < nodes.size(); ++k)
		{
			const auto & a = nodes[k];
			const auto & b = nodes[k + 1];
			heroes = heroes || a.targetHero != b.targetHero;
			if(a.targetHero == b.targetHero) // (a jump within one hero's walk: between two heroes' parts of a chain the tiles differ anyway)
				jump = jump || a.coord.z != b.coord.z || std::abs(a.coord.x - b.coord.x) > 1 || std::abs(a.coord.y - b.coord.y) > 1;
		}
		// The first step today: the destination's own entry, else the path's stop today (its node of this hero's with
		// turns 0 nearest the destination)
		if(nodes[own].turns == 0)
			row.move = decision.moveEntry(slot, row.dest);
		if(row.move >= 0)
			row.flags |= 1;
		else
			for(size_t k = own; k < nodes.size(); ++k)
				if(nodes[k].targetHero == hero && nodes[k].turns == 0)
				{
					row.move = decision.moveEntry(slot, nodes[k].coord);
					break;
				}
		// TRAVEL toward it: the destination's area, else that of the path's farthest node of this hero's a day or more away
		for(size_t k = own; k < nodes.size() && row.travel < 0; ++k)
		{
			if(nodes[k].targetHero != hero || nodes[k].turns < 1)
				continue;
			const int a = areaOf(nodes[k].coord, side);
			if(a >= 0 && travels.count(slot * AREAS + a))
				row.travel = slot * AREAS + a;
		}
		// Our pathfinder's reach of the destination: its window tile's days, else TRAVEL to its area
		const bool reached = row.destCell >= 0 ? in.get(In::w_turns, row.destCell / (WINDOW * WINDOW), (row.destCell / WINDOW) % WINDOW, row.destCell % WINDOW) >= 0 : (row.destArea >= 0 && travels.count(row.destArea));
		row.flags |= (reached ? 0 : 2) | (jump ? 4 : 0) | (heroes ? 8 : 0);
		const auto context = evaluator->buildEvaluationContext(goal);
		row.score[0] = task->priority;
		row.score[1] = tier == NK2AI::PriorityEvaluator::EXPLORE_AND_GATHER ? task->priority : evaluator->evaluate(goal, NK2AI::PriorityEvaluator::EXPLORE_AND_GATHER);
		row.days = context.movementCost;
		row.loss = context.armyLossRatio;
		row.danger = context.enemyHeroDangerRatio;
		row.kind = candidateKind(context);
		rows[slot].push_back(row);
	}
	in.values[static_cast<size_t>(In::g_nk_present)][0] = 1;
	in.values[static_cast<size_t>(In::g_nk_tier)][0] = tier;
	in.values[static_cast<size_t>(In::g_nk_count)][0] = count;
	std::map<std::tuple<int, int, int, int>, int> settled;
	for(int slot = 0; slot < OWN_HEROES; ++slot)
	{
		auto & mine = rows[slot];
		std::stable_sort(mine.begin(), mine.end(), [](const CandidateRow & a, const CandidateRow & b) { return a.score[0] != b.score[0] ? a.score[0] > b.score[0] : a.score[1] > b.score[1]; });
		std::set<std::tuple<int, int, int, int, int>> seen; // (one row a destination and first steps: a task found twice keeps its best)
		int r = 0;
		for(const auto & row : mine)
		{
			if(r >= NK_PER_HERO || !seen.insert({row.dest.x, row.dest.y, row.dest.z, row.move, row.travel}).second)
				continue;
			const int at = slot * NK_PER_HERO + r;
			int n = 0;
			if(r < 3) // (the hero's top 3: decisions in a row there, this one's 0 when new)
			{
				const auto key = std::make_tuple(decision.heroes[slot].getNum(), row.dest.x, row.dest.y, row.dest.z);
				const auto before = nkSettled.find(key);
				n = before == nkSettled.end() ? 0 : before->second + 1;
				settled[key] = n;
			}
			in.at(In::k_hero, at) = slot + 1;
			for(int k = 0; k < 3; ++k)
				in.at(In::k_dest, at, k) = k == 0 ? row.dest.x : k == 1 ? row.dest.y : row.dest.z;
			in.at(In::k_dest_cell, at) = row.destCell + 1;
			in.at(In::k_dest_area, at) = row.destArea + 1;
			in.at(In::k_move, at) = row.move + 1;
			in.at(In::k_travel, at) = row.travel + 1;
			for(int k = 0; k < 2; ++k)
				in.at(In::k_score, at, k) = fixedPoint(row.score[k], 100.0);
			in.at(In::k_days, at) = fixedPoint(row.days, 1000.0);
			in.at(In::k_loss, at) = fixedPoint(row.loss, 10000.0);
			in.at(In::k_danger, at) = fixedPoint(row.danger, 1000.0, 1e6);
			in.at(In::k_kind, at) = row.kind;
			in.at(In::k_settled, at) = n;
			in.at(In::k_flags, at) = row.flags;
			++r;
		}
	}
	nkSettled = std::move(settled);
	// Its threats (DangerHitMapAnalyzer, as DefenceBehavior reads them): to our towns, against the town's defence (its
	// garrison hero's strength, else its visitor's, else its garrison's); at our heroes' tiles, against the hero's
	for(int t = 0; t < OWN_TOWNS; ++t)
	{
		const auto * town = dynamic_cast<const CGTownInstance *>(cb->getObj(decision.towns[t], false));
		if(!town || town->getOwner() != playerID)
			continue;
		uint64_t danger = 0;
		int turns = -1;
		for(const auto & threat : nk.dangerHitMap->getTownThreats(town))
		{
			danger = std::max(danger, threat.danger);
			if(threat.danger > 0 && (turns < 0 || threat.turn < turns))
				turns = threat.turn;
		}
		if(danger == 0)
			continue;
		const double defence = town->getGarrisonHero() ? town->getGarrisonHero()->getTotalStrength() : town->getVisitingHero() ? town->getVisitingHero()->getTotalStrength() : town->getUpperArmy()->getArmyStrength();
		in.at(In::t_nk_threat, t, 0) = 1;
		in.at(In::t_nk_threat, t, 1) = turns;
		in.at(In::t_nk_threat, t, 2) = fixedPoint(static_cast<double>(danger) / std::max(defence, 1.0), 1000.0, 1e6);
	}
	for(int h = 0; h < OWN_HEROES; ++h)
	{
		const auto * hero = cb->getHero(decision.heroes[h]);
		if(!hero || hero->getOwner() != playerID)
			continue;
		const auto & node = nk.dangerHitMap->getTileThreat(hero->visitablePos());
		if(node.maximumDanger.danger == 0)
			continue;
		in.at(In::h_nk_threat, h, 0) = 1;
		in.at(In::h_nk_threat, h, 1) = node.fastestDanger.turn;
		in.at(In::h_nk_threat, h, 2) = fixedPoint(static_cast<double>(node.maximumDanger.danger) / std::max(static_cast<double>(hero->getTotalStrength()), 1.0), 1000.0, 1e6);
	}
#endif
}

void AlphaMapAI::labelInputs(Decision & decision, const std::optional<Choice> & advice)
{
	if(!advice || advice->type < 0 || advice->index < 0)
		return;
	decision.inputs.at(In::g_nk_action) = advice->type + 1;
	decision.inputs.at(In::g_nk_index) = advice->index + 1;
	decision.inputs.at(In::g_nk_cell) = advice->cell + 1; // (-1 none: 0)
}
