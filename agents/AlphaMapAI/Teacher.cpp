#include "StdInc.h"
#include "AlphaMapAI.h"

#include "AIGateway.h" // Nullkiller2's (AI/Nullkiller2 is on this library's include path: CMakeLists.txt)

#include "lib/callback/CCallback.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/mapObjects/CGTownInstance.h"
#include "lib/networkPacks/PacksForClientBattle.h"
#include "lib/networkPacks/PacksForServer.h"

#include <cmath>

using namespace MapSchema;

// Nullkiller2 as the campaign model's teacher: the AlphaTeacherAI library (VCMI_ADVENTURE_AI_<COLOR>=AlphaTeacherAI) is
// Nullkiller2, playing as it always does, with an AlphaMapAI beside it that never acts. Each request Nullkiller2 sends
// on its turn (requestSent: the game state is still as it was, its turn holding the state's lock) is encoded as our
// turn decision there and matched to the legal entry of ours that does the same; the pair is recorded as a decision
// (RECORD_TEACHER) for campaign_model/imitate.py. What has no equal among our actions is counted by kind and sent with
// the player's summary ("teacher": records.py), so how much of the teacher the model can follow is measured, not
// guessed. Its answers to yes/no, selection and level-up questions are recorded too: each question encoded as it is
// asked (on the client's thread, before Nullkiller2 has seen it), kept, and recorded with the option its QueryReply
// chose. Battles (MakeAction: the battle model's) and the other questions are not recorded.

/// Nullkiller2, with its turn requests recorded (AlphaMapAI::observe)
class TeacherAI : public NK2AI::AIGateway
{
	std::shared_ptr<AlphaMapAI> recorder = std::make_shared<AlphaMapAI>();

public:
	void initGameInterface(std::shared_ptr<Environment> environmentPtr, std::shared_ptr<CCallback> callback) override
	{
		AIGateway::initGameInterface(environmentPtr, callback);
		recorder->initTeacher(environmentPtr, callback);
	}

	void yourTurn(QueryID queryID) override
	{
		recorder->teacherTurnStarts();
		AIGateway::yourTurn(queryID);
	}

	void showBlockingDialog(const std::string & text, const std::vector<Component> & components, QueryID askID, const int soundID, bool selection, bool cancel, bool safeToAutoaccept) override
	{
		recorder->teacherBlockingDialog(text, components, askID, soundID, selection, cancel); // (before Nullkiller2 answers)
		AIGateway::showBlockingDialog(text, components, askID, soundID, selection, cancel, safeToAutoaccept);
	}

	void heroGotLevel(const CGHeroInstance * hero, PrimarySkill pskill, std::vector<SecondarySkill> & skills, QueryID queryID) override
	{
		recorder->teacherLevelUp(hero, skills, queryID);
		AIGateway::heroGotLevel(hero, pskill, skills, queryID);
	}

	void requestSent(const CPackForServer * pack, int requestID) override
	{
		recorder->observe(pack);
		AIGateway::requestSent(pack, requestID);
	}

	void playerStartsTurn(PlayerColor player) override
	{
		recorder->playerStartsTurn(player); // the day's ledger; past the day cap, the last records
		AIGateway::playerStartsTurn(player);
	}

	void gameOver(PlayerColor player, const EVictoryLossCheckResult & victoryLossCheckResult) override
	{
		recorder->gameOver(player, victoryLossCheckResult);
		AIGateway::gameOver(player, victoryLossCheckResult);
	}
};

#define NAME "AlphaTeacherAI 0.1"

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
	out = std::make_shared<TeacherAI>();
}
