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

namespace
{
std::string environment(const char * name, const std::string & fallback = "")
{
	const char * value = std::getenv(name);
	return value ? value : fallback;
}
}

void AlphaMapAI::initTeacher(std::shared_ptr<Environment> environmentPtr, std::shared_ptr<CCallback> callback)
{
	cb = callback; // (Nullkiller2's own callback: its settings, waitTillRealize and the rest, are left as it set them)
	cbc = callback;
	env = environmentPtr;
	playerID = *cb->getPlayerID();
	teacher = true;
	recordSocket = environment("ALPHA_MAP_RECORD_SOCKET");
	recordsPerDecision = environment("ALPHA_MAP_RECORD_FORMAT", "1") == "2";
	gameKey = environment("ALPHA_MAP_GAME_KEY", "game");
	dayCap = std::stoi(environment("ALPHA_MAP_DAYS", "0"));
	logAi->info("AlphaMapAI %s: recording Nullkiller2 as the teacher, day cap %d", playerID.toString(), dayCap);
}

void AlphaMapAI::teacherTurnStarts()
{
	const int today = cb->getDate(Date::DAY);
	if(today != day)
	{
		day = today;
		decisionsToday = 0;
		futile.clear();
	}
	activeHero = ObjectInstanceID();
	activeTarget = int3(-1, -1, -1);
	lastChoice = Choice();
	lastChanged = -1;
}

void AlphaMapAI::teacherBlockingDialog(const std::string & text, const std::vector<Component> & components, QueryID askID, int soundID, bool selection, bool cancel)
{
	if(!teacher || stopping || (!selection && !cancel)) // (a dialog that only closes: no question)
		return;
	try
	{
		auto decision = blockingQuestion(text, components, soundID, selection, cancel);
		std::scoped_lock lock(recordsMutex);
		teacherQuestions[askID] = std::move(decision);
	}
	catch(const std::exception & e)
	{
		logAi->error("AlphaMapAI %s: the teacher's question was not encoded: %s", playerID.toString(), e.what());
		++errors;
	}
}

void AlphaMapAI::teacherLevelUp(const CGHeroInstance * hero, const std::vector<SecondarySkill> & skills, QueryID queryID)
{
	if(!teacher || stopping || skills.empty() || !hero)
		return;
	try
	{
		auto decision = levelUpQuestion(hero, skills);
		std::scoped_lock lock(recordsMutex);
		teacherQuestions[queryID] = std::move(decision);
	}
	catch(const std::exception & e)
	{
		logAi->error("AlphaMapAI %s: the teacher's level up was not encoded: %s", playerID.toString(), e.what());
		++errors;
	}
}

void AlphaMapAI::observe(const CPackForServer * pack)
{
	if(!teacher || stopping || (dayCap > 0 && cb->getDate(Date::DAY) > dayCap))
		return;
	if(dynamic_cast<const MakeAction *>(pack))
		return; // a battle's action: the battle model's
	if(const auto * reply = dynamic_cast<const QueryReply *>(pack))
	{
		// A question we encoded when it was asked: recorded with the option of ours its answer is; others (the turn's
		// own query, dialogs that only close) are no decision
		std::scoped_lock lock(recordsMutex);
		const auto found = teacherQuestions.find(reply->qid);
		if(found == teacherQuestions.end())
			return;
		auto decision = std::move(found->second);
		teacherQuestions.erase(found);
		const int answer = reply->reply.value_or(0);
		const auto option = std::find(decision.answers.begin(), decision.answers.end(), answer);
		if(option == decision.answers.end())
		{
			++teacherUnmatched["QueryReply (an answer not among our options)"];
			return;
		}
		Record entry;
		entry.day = cb->getDate(Date::DAY);
		entry.kind = static_cast<int>(decision.kind);
		entry.decisionsToday = decisionsToday;
		entry.flags = RECORD_TEACHER;
		entry.choice.index = static_cast<int>(option - decision.answers.begin());
		entry.inputs = decision.inputs.wire();
		entry.serial = entry.choice.record = ++recordSerial;
		records.push_back(std::move(entry));
		++teacherMatched;
		return;
	}
	try
	{
		auto decision = encode(DecisionKind::TURN);
		Choice choice;
		bool exact = true;
		std::string unmatched; // the request's kind, when it has no equal
		matchRequest(decision, pack, choice, exact, unmatched);
		if(choice.type < 0)
		{
			std::scoped_lock lock(recordsMutex); // (the counts are read by sendRecords, on the client's thread)
			++teacherUnmatched[unmatched.empty() ? std::string(typeid(*pack).name()) : unmatched];
			return;
		}
		Record entry;
		entry.day = cb->getDate(Date::DAY);
		entry.kind = static_cast<int>(DecisionKind::TURN);
		entry.decisionsToday = decisionsToday;
		entry.flags = RECORD_TEACHER;
		entry.choice = choice;
		entry.inputs = decision.inputs.wire();
		{
			std::scoped_lock lock(recordsMutex);
			entry.serial = choice.record = ++recordSerial;
			entry.choice.record = entry.serial;
			records.push_back(std::move(entry));
			++teacherMatched;
			teacherApproximate += exact ? 0 : 1;
		}
		++decisionsToday;
		lastChoice = choice;
		lastChanged = 1;
		if(choice.type == static_cast<int>(ActionType::END))
			sendRecords(false);
	}
	catch(const std::exception & e)
	{
		logAi->error("AlphaMapAI %s: the teacher's request was not recorded: %s", playerID.toString(), e.what());
		++errors;
	}
}

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
