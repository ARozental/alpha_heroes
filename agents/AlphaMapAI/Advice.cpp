#include "StdInc.h"
#include "AlphaMapAI.h"

#ifdef ALPHA_MAP_NK2_ADVICE
#include "AIGateway.h" // Nullkiller2's (AI/Nullkiller2 is on this library's include path: CMakeLists.txt)

#include "lib/UnlockGuard.h"
#include "lib/battle/BattleAction.h"
#include "lib/callback/CCallback.h"
#include "lib/callback/IClient.h"
#include "lib/gameState/CGameState.h"
#include "lib/networkPacks/PacksForServer.h"
#endif

#include <chrono>

using namespace MapSchema;

// Nullkiller2's advice: the teacher's labels on our own states (the user, 2026-10-31: "a real teacher", not a model
// imitating it). This player's AlphaMapAI keeps a Nullkiller2 of its own whose callback sends its requests to a client
// that sends nothing (AdviceClient). Before a sampled turn decision of ours, that Nullkiller2 makes its turn as it
// would from this state; its first request (what it would do now) is matched to our legal entry doing the same
// (matchRequest, as the teacher's recorder does) and stops it there. Its turn starts again each time: Nullkiller2
// plans from the state (and remembers the objects it can see: memorizeVisitableObjs), so it needs none of our events.
// Our decision's record carries it (RECORD_ADVICE); the trainer takes a loss toward it (campaign.nk2_advice).

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
	advisor = a;
	logAi->info("AlphaMapAI %s: Nullkiller2's advice at a chance of %.3f a turn decision", playerID.toString(), adviceProbability);
#else
	if(adviceProbability > 0)
		logAi->error("AlphaMapAI %s: ALPHA_MAP_NK2_ADVICE asked, but this build has no Nullkiller2: no advice", playerID.toString());
	adviceProbability = 0;
#endif
}

std::optional<Choice> AlphaMapAI::advise(const Decision & decision)
{
#ifdef ALPHA_MAP_NK2_ADVICE
	if(!advisor)
		return std::nullopt;
	std::optional<Choice> advice;
	std::string unmatched = "no request";
	const auto started = std::chrono::steady_clock::now();
	advisor->client.taken = [&](const CPackForServer & pack)
	{
		Choice choice;
		bool exact = true;
		unmatched.clear();
		if(matchRequest(decision, &pack, choice, exact, unmatched))
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

std::string AlphaMapAI::adviceJson() const
{
	if(advised == 0)
		return "";
	std::string unmatched;
	for(const auto & [kind, count] : adviceUnmatched)
		unmatched += (unmatched.empty() ? "\"" : ", \"") + kind + "\": " + std::to_string(count);
	return ", \"advice\": {\"asked\": " + std::to_string(advised) + ", \"matched\": " + std::to_string(adviceMatched) + ", \"seconds\": " + std::to_string(adviceSeconds) + ", \"unmatched\": {" + unmatched + "}}";
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
