#include "StdInc.h"
#include "AlphaMapAI.h"

#include "lib/battle/BattleAction.h"
#include "lib/callback/CCallback.h"
#include "lib/callback/CDynLibHandler.h"
#include "lib/gameState/CGameState.h"
#include "lib/networkPacks/PacksForServer.h"

#include <boost/algorithm/string.hpp>
#include <cstdlib>
#include <pthread.h>
#include <shared_mutex>

namespace
{
bool onMainThread() // (the GUI's: the player's own clicks; macOS only, where people play: elsewhere every request the player's)
{
#ifdef __APPLE__
	return pthread_main_np() != 0;
#else
	return true;
#endif
}
}

// A human player's recorder: the AlphaRecorderAI library, which the VCMI client puts beside a human's player interface
// when ALPHA_RECORD_HUMAN is set (alpha_heroes' patch: vcmi/client/CPlayerInterface.cpp). It never acts: the player plays
// at the GUI, and the interface passes this object the events a recorder needs.
//
// The adventure map: each request the player sends is recorded as the turn decision of ours that does the same, our
// inputs as they were then (AlphaMapAI::observe, as Nullkiller2's are by Teacher.cpp; RECORD_HUMAN), their answers to
// yes/no, selection and level-up questions too, and each request raw (ALPHA_RECORD_RAW, JSON lines: a move made of many
// requests, such as buying every creature from the dearest down, is there as the requests it was). The records go to
// ALPHA_RECORD_HUMAN_SOCKET (campaign_model/record_human.py's record server).
//
// The battles: an AlphaBattleAI observing this player's (ALPHA_BATTLE_OBSERVE, set here for this player alone: no other
// battle of the game is recorded) encodes each state the player is asked to act in and records the action they took,
// as the battle model's records (to the same socket, key <the segment's key>-human-<COLOR>). A battle fought by
// auto-combat is not the player's: the interface passes none of its states. Each start of a game (a new one, a save
// loaded) is its own segment: this recorder's records, and the computer players', under keys ending in @<the start>
// (AlphaMapAI::sessionKey), the ledgers sent anew from the day it starts at.
class RecorderAI : public CGlobalAI
{
	std::shared_ptr<AlphaMapAI> map = std::make_shared<AlphaMapAI>();
	std::shared_ptr<CBattleGameInterface> battle; // AlphaBattleAI, observing
	std::shared_ptr<CCallback> cb;

public:
	~RecorderAI() override
	{
		map->finishRecording(); // (the player left the game: what this turn recorded is sent)
	}

	void initGameInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CCallback> CB) override
	{
		env = ENV;
		cb = CB;
		playerID = *CB->getPlayerID();
		map->initHuman(ENV, CB);
		// AlphaBattleAI's per-player settings (ALPHA_BATTLE_<NAME>_<COLOR>) for this player: it observes and records.
		// (Set as the game starts, while the interfaces are made: no AI is playing yet to read the environment)
		const auto color = boost::to_upper_copy(playerID.toString());
		const char * socket = std::getenv("ALPHA_RECORD_HUMAN_SOCKET");
		setenv(("ALPHA_BATTLE_OBSERVE_" + color).c_str(), "1", 1);
		setenv(("ALPHA_BATTLE_ACTOR_" + color).c_str(), "human", 1); // (the record's start event: "model": "human")
		if(socket && *socket)
			setenv(("ALPHA_BATTLE_RECORD_SOCKET_" + color).c_str(), socket, 1);
		setenv(("ALPHA_BATTLE_RECORD_KEY_" + color).c_str(), (map->recordKey() + "-human-" + color).c_str(), 1); // (this segment's)
		battle = CDynLibHandler::getNewBattleAI("AlphaBattleAI");
		battle->initBattleInterface(ENV, CB);
	}

	// ---- the adventure map: AlphaMapAI's recording of another player (Observe.cpp) ----
	void yourTurn(QueryID askID) override
	{
		map->teacherTurnStarts();
	}

	void playerStartsTurn(PlayerColor player) override
	{
		map->playerStartsTurn(player); // the day's ledger
	}

	void gameOver(PlayerColor player, const EVictoryLossCheckResult & victoryLossCheckResult) override
	{
		map->gameOver(player, victoryLossCheckResult);
	}

	void showBlockingDialog(const std::string & text, const std::vector<Component> & components, QueryID askID, const int soundID, bool selection, bool cancel, bool safeToAutoaccept) override
	{
		map->teacherBlockingDialog(text, components, askID, soundID, selection, cancel); // (before the player answers)
	}

	void heroGotLevel(const CGHeroInstance * hero, PrimarySkill pskill, std::vector<SecondarySkill> & skills, QueryID queryID) override
	{
		map->teacherLevelUp(hero, skills, queryID);
	}

	void requestSent(const CPackForServer * pack, int requestID) override
	{
		// (the player's GUI thread, which holds no lock of the game state's: it may not change while it is encoded; a
		// battle action from another thread is auto-combat's, which plays for the player: raw.jsonl's actor "auto")
		std::shared_lock lock(CGameState::mutex);
		map->observe(pack, !dynamic_cast<const MakeAction *>(pack) || onMainThread());
	}

	// The questions no recorder answers (the player's interface does)
	void commanderGotLevel(const CCommanderInstance * commander, std::vector<ui32> skills, QueryID queryID) override {}
	void showGarrisonDialog(const CArmedInstance * up, const CGHeroInstance * down, bool removableUnits, QueryID queryID, const MetaString & customTitle) override {}
	void showTeleportDialog(const CGHeroInstance * hero, TeleportChannelID channel, TTeleportExitsList exits, bool impassable, QueryID askID) override {}
	void showMapObjectSelectDialog(QueryID askID, const Component & icon, const MetaString & title, const MetaString & description, const std::vector<ObjectInstanceID> & objects) override {}
	std::optional<BattleAction> makeSurrenderRetreatDecision(const BattleID & battleID, const BattleStateInfoForRetreat & battleState) override
	{
		return std::nullopt;
	}

	// ---- the battles: AlphaBattleAI observing ----
	void battleStart(const BattleID & battleID, const CCreatureSet * army1, const CCreatureSet * army2, int3 tile, const CGHeroInstance * hero1, const CGHeroInstance * hero2, BattleSide side, bool replayAllowed) override
	{
		battle->battleStart(battleID, army1, army2, tile, hero1, hero2, side, replayAllowed);
	}

	void activeStack(const BattleID & battleID, const CStack * stack) override
	{
		battle->activeStack(battleID, stack); // (the state the player is asked to act in: its action, actionStarted's)
	}

	void yourTacticPhase(const BattleID & battleID, int distance) override
	{
		battle->yourTacticPhase(battleID, distance);
	}

	void actionStarted(const BattleID & battleID, const BattleAction & action) override
	{
		battle->actionStarted(battleID, action);
	}

	void actionFinished(const BattleID & battleID, const BattleAction & action) override
	{
		battle->actionFinished(battleID, action);
	}

	void battleEnd(const BattleID & battleID, const BattleResult * br, QueryID queryID) override
	{
		battle->battleEnd(battleID, br, queryID); // (the battle's record sent)
	}

	void battleNewRoundFirst(const BattleID & battleID) override
	{
		battle->battleNewRoundFirst(battleID);
	}

	void battleSpellCast(const BattleID & battleID, const BattleSpellCast * sc) override
	{
		battle->battleSpellCast(battleID, sc);
	}

	void battleStacksAttacked(const BattleID & battleID, const std::vector<BattleStackAttacked> & bsa, bool ranged) override
	{
		battle->battleStacksAttacked(battleID, bsa, ranged);
	}
};

#define NAME "AlphaRecorderAI 0.1"

extern "C" DLL_EXPORT int GetGlobalAiVersion()
{
	return AI_INTERFACE_VER;
}

extern "C" DLL_EXPORT void GetAiName(char * name)
{
	strcpy(name, NAME);
}

extern "C" DLL_EXPORT void GetNewAI(std::shared_ptr<CGlobalAI> & out)
{
	out = std::make_shared<RecorderAI>();
}
