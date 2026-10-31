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

const CArmedInstance * armedObject(const CCallback & cb, ObjectInstanceID id)
{
	return dynamic_cast<const CArmedInstance *>(cb.getObj(id, false));
}

// The entry of `entries` for which `key` is true whose amount `amount(e)` is nearest `wanted` (on a log scale); -1 none
template<typename Key, typename Amount>
int nearest(const std::unordered_map<int, Entry> & entries, Key key, Amount amount, double wanted, bool & exact)
{
	int best = -1;
	double gap = 1e18;
	for(const auto & [index, e] : entries)
	{
		if(!key(e))
			continue;
		const double g = std::abs(std::log(std::max(1.0, amount(e))) - std::log(std::max(1.0, wanted)));
		if(g < gap || (g == gap && index < best))
		{
			gap = g;
			best = index;
		}
	}
	exact = gap < 1e-9;
	return best;
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
		const auto & entries = [&decision](ActionType type) -> const std::unordered_map<int, Entry> & { return decision.entries[static_cast<int>(type)]; };
		Choice choice;
		bool exact = true;
		std::string unmatched; // the request's kind, when it has no equal
		const auto pick = [&choice](ActionType type, int index)
		{
			if(index >= 0)
			{
				choice.type = static_cast<int>(type);
				choice.index = index;
			}
		};
		if(dynamic_cast<const EndTurn *>(pack))
			pick(ActionType::END, 0);
		else if(const auto * p = dynamic_cast<const MoveHero *>(pack))
		{
			const auto * h = cb->getHero(p->hid);
			const int3 target = h && !p->path.empty() ? h->convertToVisitablePos(p->path.back()) : int3(-1, -1, -1);
			if(!h || p->path.empty())
				unmatched = "MoveHero (no hero or path)";
			else if(target == h->visitablePos())
				unmatched = "MoveHero (into a teleport)"; // (our moves go through teleports as part of a path)
			else
			{
				for(int i = 0; i < OWN_HEROES; ++i)
				{
					if(decision.heroes[i] != p->hid)
						continue;
					const auto & center = decision.centers[i];
					const int row = target.y - center.y + WINDOW_RADIUS;
					const int column = target.x - center.x + WINDOW_RADIUS;
					if(target.z == center.z && row >= 0 && row < WINDOW && column >= 0 && column < WINDOW && decision.inputs.get(In::move_mask, i, row, column) != 0)
						pick(ActionType::MOVE, (i * WINDOW + row) * WINDOW + column);
					break;
				}
				if(choice.type < 0) // not a tile of its window in reach today: the area it heads for, the nearest
				{
					const auto distance = [&target](const Entry & e) { return 1.0 + e.tile.dist2dSQ(target) + (e.tile.z != target.z ? 1e6 : 0.0); };
					pick(ActionType::TRAVEL, nearest(entries(ActionType::TRAVEL), [p](const Entry & e) { return e.first == p->hid; }, distance, 1.0, exact));
					exact = false;
				}
				if(choice.type < 0)
					unmatched = "MoveHero (no move or travel of ours there)";
			}
		}
		else if(const auto * p = dynamic_cast<const BuildStructure *>(pack))
		{
			for(const auto & [index, e] : entries(ActionType::BUILD))
				if(e.first == p->tid && e.a == p->bid.getNum())
					pick(ActionType::BUILD, index);
			if(choice.type < 0)
				unmatched = "BuildStructure (not legal for us)";
		}
		else if(const auto * p = dynamic_cast<const RecruitCreatures *>(pack))
		{
			const auto key = [p](const Entry & e) { return e.first == p->tid && e.second == p->dst && e.a == p->crid.getNum(); };
			pick(ActionType::RECRUIT, nearest(entries(ActionType::RECRUIT), key, [](const Entry & e) { return double(e.b); }, double(p->amount), exact));
			if(choice.type < 0)
				unmatched = cb->getTown(p->tid) ? "RecruitCreatures (town: not legal for us)" : "RecruitCreatures (a dwelling's: a question of ours)";
		}
		else if(const auto * p = dynamic_cast<const HireHero *>(pack))
		{
			const auto * town = cb->getTown(p->tid);
			const auto heroes = town ? cb->getAvailableHeroes(town) : std::vector<const CGHeroInstance *>();
			for(const auto & [index, e] : entries(ActionType::HIRE))
				if(e.first == p->tid && e.a >= 0 && e.a < static_cast<int>(heroes.size()) && heroes[e.a]->getHeroTypeID() == p->hid)
					pick(ActionType::HIRE, index);
			if(choice.type < 0)
				unmatched = town ? "HireHero (not legal for us)" : "HireHero (a tavern outside our towns)";
		}
		else if(const auto * p = dynamic_cast<const UpgradeCreature *>(pack))
		{
			for(const auto & [index, e] : entries(ActionType::UPGRADE))
				if(e.first == p->id && e.a == p->pos.getNum())
				{
					pick(ActionType::UPGRADE, index);
					exact = e.b == p->cid.getNum(); // (ours upgrades to the best upgrade)
				}
			if(choice.type < 0)
				unmatched = "UpgradeCreature (not legal for us)";
		}
		else if(const auto * p = dynamic_cast<const TradeOnMarketplace *>(pack))
		{
			if(p->mode != EMarketMode::RESOURCE_RESOURCE || p->r1.size() != 1 || p->r2.size() != 1 || p->val.size() != 1)
				unmatched = "TradeOnMarketplace (not resource for resource)";
			else
			{
				const int sell = p->r1[0].getNum();
				const int buy = p->r2[0].getNum();
				const auto key = [p, sell, buy](const Entry & e) { return e.first == p->marketId && e.a == sell && e.b == buy; };
				pick(ActionType::TRADE, nearest(entries(ActionType::TRADE), key, [](const Entry & e) { return double(e.c); }, double(p->val[0]), exact));
				if(choice.type < 0)
					unmatched = "TradeOnMarketplace (not legal for us)";
			}
		}
		else if(const auto * p = dynamic_cast<const BuyArtifact *>(pack))
		{
			for(const auto & [index, e] : entries(ActionType::BUY))
				if(e.first == p->hid && e.a == p->aid.getNum())
					pick(ActionType::BUY, index);
			if(choice.type < 0)
				unmatched = "BuyArtifact (not a spellbook or war machine of ours)";
		}
		else if(const auto * p = dynamic_cast<const CastAdvSpell *>(pack))
		{
			for(const auto & [index, e] : entries(ActionType::CAST))
				if(e.first == p->hid && e.a == p->sid.getNum())
				{
					pick(ActionType::CAST, index);
					const int i = index / N_ADVENTURE_SPELLS;
					const auto & center = decision.centers[i];
					const int row = p->pos.y - center.y + WINDOW_RADIUS;
					const int column = p->pos.x - center.x + WINDOW_RADIUS;
					if(p->pos.isValid() && p->pos.z == center.z && row >= 0 && row < WINDOW && column >= 0 && column < WINDOW && decision.inputs.get(In::cast_cell_mask, i, row, column) != 0)
						choice.cell = row * WINDOW + column;
				}
			if(choice.type < 0)
				unmatched = "CastAdvSpell (not legal for us)";
		}
		else if(const auto * p = dynamic_cast<const ArrangeStacks *>(pack))
		{
			const auto * source = armedObject(*cb, p->id1);
			const auto * target = armedObject(*cb, p->id2);
			if(p->id1 == p->id2)
				unmatched = "ArrangeStacks (within one army)";
			else if(!source || !target)
				unmatched = "ArrangeStacks (no army)";
			else if(p->what == 1 && !target->slotEmpty(p->p2))
				unmatched = "ArrangeStacks (a swap of two stacks)";
			else
			{
				const double moved = p->what == 3 ? double(p->val - target->getStackCount(p->p2)) : double(source->getStackCount(p->p1));
				const auto key = [p](const Entry & e) { return e.first == p->id1 && e.second == p->id2 && e.a == p->p1.getNum() && e.b == p->p2.getNum(); };
				pick(ActionType::ARMY, nearest(entries(ActionType::ARMY), key, [](const Entry & e) { return double(e.c); }, moved, exact));
				if(choice.type < 0)
					unmatched = "ArrangeStacks (not legal for us)";
			}
		}
		else
			unmatched = typeid(*pack).name();
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
