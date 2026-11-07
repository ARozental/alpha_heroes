#include "StdInc.h"
#include "AlphaMapAI.h"

#include "lib/callback/CCallback.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/mapObjects/CGTownInstance.h"
#include "lib/networkPacks/PacksForClientBattle.h"
#include "lib/networkPacks/PacksForServer.h"

#include <cmath>

using namespace MapSchema;

// The recording of another player's decisions as ours (Nullkiller2's: Teacher.cpp, the AlphaTeacherAI library; a human
// player's: Recorder.cpp, the AlphaRecorderAI library): each request the player sends on its turn (requestSent: the
// game state is still as it was) is encoded as our turn decision there and matched to the legal entry of ours that does
// the same, recorded as a decision with the player's flag (RECORD_TEACHER, RECORD_HUMAN); what has no equal among our
// actions is counted by kind and sent with the player's summary ("teacher": records.py). Its answers to yes/no,
// selection and level-up questions are recorded too: each question encoded as it is asked, kept, and recorded with
// the option its QueryReply chose. A human's requests are written raw too (ALPHA_RECORD_RAW: rawRequest).

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
	{
		rawRequest(pack, 0, nullptr, "a battle's action: the battle model's records have it");
		return;
	}
	if(const auto * reply = dynamic_cast<const QueryReply *>(pack))
	{
		// A question we encoded when it was asked: recorded with the option of ours its answer is; others (the turn's
		// own query, dialogs that only close) are no decision
		uint64_t serial = 0;
		Choice chosen;
		std::string why;
		{
			std::scoped_lock lock(recordsMutex);
			const auto found = teacherQuestions.find(reply->qid);
			if(found == teacherQuestions.end())
				why = "no question of ours";
			else
			{
				auto decision = std::move(found->second);
				teacherQuestions.erase(found);
				const int answer = reply->reply.value_or(0);
				const auto option = std::find(decision.answers.begin(), decision.answers.end(), answer);
				if(option == decision.answers.end())
				{
					why = "QueryReply (an answer not among our options)";
					++teacherUnmatched[why];
				}
				else
				{
					Record entry;
					entry.day = cb->getDate(Date::DAY);
					entry.kind = static_cast<int>(decision.kind);
					entry.decisionsToday = decisionsToday;
					entry.flags = observedFlag;
					entry.choice.index = static_cast<int>(option - decision.answers.begin());
					entry.inputs = decision.inputs.wire();
					entry.serial = entry.choice.record = ++recordSerial;
					serial = entry.serial;
					chosen = entry.choice;
					records.push_back(std::move(entry));
					++teacherMatched;
				}
			}
		}
		rawRequest(pack, serial, serial ? &chosen : nullptr, why);
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
			const auto kind = unmatched.empty() ? std::string(typeid(*pack).name()) : unmatched;
			{
				std::scoped_lock lock(recordsMutex); // (the counts are read by sendRecords, on the client's thread)
				++teacherUnmatched[kind];
			}
			rawRequest(pack, 0, nullptr, kind);
			return;
		}
		Record entry;
		entry.day = cb->getDate(Date::DAY);
		entry.kind = static_cast<int>(DecisionKind::TURN);
		entry.decisionsToday = decisionsToday;
		entry.flags = observedFlag;
		entry.choice = choice;
		entry.inputs = decision.inputs.wire();
		uint64_t serial = 0;
		{
			std::scoped_lock lock(recordsMutex);
			entry.serial = choice.record = ++recordSerial;
			entry.choice.record = entry.serial;
			serial = entry.serial;
			records.push_back(std::move(entry));
			++teacherMatched;
			teacherApproximate += exact ? 0 : 1;
		}
		rawRequest(pack, serial, &choice, exact ? "" : "approximate: our nearest amount or area");
		++decisionsToday;
		lastChoice = choice;
		lastChanged = 1;
		if(choice.type == static_cast<int>(ActionType::END))
			sendRecords(false);
	}
	catch(const std::exception & e)
	{
		logAi->error("AlphaMapAI %s: the teacher's request was not recorded: %s", playerID.toString(), e.what());
		rawRequest(pack, 0, nullptr, std::string("not encoded: ") + e.what());
		++errors;
	}
}

void AlphaMapAI::initHuman(std::shared_ptr<Environment> environmentPtr, std::shared_ptr<CCallback> callback)
{
	initTeacher(environmentPtr, callback);
	observedFlag = RECORD_HUMAN;
	recordSocket = environment("ALPHA_RECORD_HUMAN_SOCKET", recordSocket);
	dayCap = 0; // (a human plays on as long as they like)
	const auto rawPath = environment("ALPHA_RECORD_RAW");
	if(!rawPath.empty())
		raw = std::make_unique<std::ofstream>(rawPath, std::ios::app);
	logAi->info("AlphaMapAI %s: recording the human player: records to %s, raw requests to %s", playerID.toString(), recordSocket.empty() ? "nowhere" : recordSocket, rawPath.empty() ? "nowhere" : rawPath);
}

namespace
{
std::string escaped(const std::string & text)
{
	std::string out;
	for(const char c : text)
	{
		if(c == '"' || c == '\\')
			out += '\\';
		if(static_cast<unsigned char>(c) >= 0x20)
			out += c;
	}
	return out;
}

std::string tile(const int3 & t)
{
	return "[" + std::to_string(t.x) + ", " + std::to_string(t.y) + ", " + std::to_string(t.z) + "]";
}

template<typename T>
std::string numbers(const std::vector<T> & values, const std::function<int64_t(const T &)> & number)
{
	std::string out = "[";
	for(size_t i = 0; i < values.size(); ++i)
		out += (i ? ", " : "") + std::to_string(number(values[i]));
	return out + "]";
}

std::string location(const ArtifactLocation & l)
{
	return "{\"holder\": " + std::to_string(l.artHolder.getNum()) + ", \"slot\": " + std::to_string(l.slot.getNum()) + ", \"creature\": " + (l.creature ? std::to_string(l.creature->getNum()) : "null") + "}";
}

/// A request's fields as sent, JSON ("request": its kind, then its own fields): the raw record of what the player did
std::string described(const CPackForServer * pack)
{
	auto id = [](const auto & value) { return std::to_string(value.getNum()); };
	if(const auto * p = dynamic_cast<const MoveHero *>(pack))
		return "\"request\": \"MoveHero\", \"hero\": " + id(p->hid) + ", \"path\": [" + [&] { std::string out; for(size_t i = 0; i < p->path.size(); ++i) out += (i ? ", " : "") + tile(p->path[i]); return out; }() + "], \"transit\": " + (p->transit ? "true" : "false");
	if(const auto * p = dynamic_cast<const RecruitCreatures *>(pack))
		return "\"request\": \"RecruitCreatures\", \"from\": " + id(p->tid) + ", \"to\": " + id(p->dst) + ", \"creature\": " + id(p->crid) + ", \"amount\": " + std::to_string(p->amount) + ", \"level\": " + std::to_string(p->level);
	if(const auto * p = dynamic_cast<const RazeStructure *>(pack))
		return "\"request\": \"RazeStructure\", \"town\": " + id(p->tid) + ", \"building\": " + id(p->bid);
	if(const auto * p = dynamic_cast<const BuildStructure *>(pack))
		return "\"request\": \"BuildStructure\", \"town\": " + id(p->tid) + ", \"building\": " + id(p->bid);
	if(const auto * p = dynamic_cast<const TradeOnMarketplace *>(pack))
		return "\"request\": \"TradeOnMarketplace\", \"market\": " + id(p->marketId) + ", \"hero\": " + id(p->heroId) + ", \"mode\": " + std::to_string(static_cast<int>(p->mode))
			+ ", \"sell\": " + numbers<TradeItemSell>(p->r1, [](const TradeItemSell & v) { return v.getNum(); }) + ", \"buy\": " + numbers<TradeItemBuy>(p->r2, [](const TradeItemBuy & v) { return v.getNum(); })
			+ ", \"amounts\": " + numbers<ui32>(p->val, [](const ui32 & v) { return static_cast<int64_t>(v); });
	if(const auto * p = dynamic_cast<const BuyArtifact *>(pack))
		return "\"request\": \"BuyArtifact\", \"hero\": " + id(p->hid) + ", \"artifact\": " + id(p->aid);
	if(const auto * p = dynamic_cast<const ArrangeStacks *>(pack))
		return "\"request\": \"ArrangeStacks\", \"what\": " + std::to_string(p->what) + ", \"from\": [" + id(p->id1) + ", " + id(p->p1) + "], \"to\": [" + id(p->id2) + ", " + id(p->p2) + "], \"amount\": " + std::to_string(p->val);
	if(const auto * p = dynamic_cast<const BulkMoveArmy *>(pack))
		return "\"request\": \"BulkMoveArmy\", \"slot\": " + id(p->srcSlot) + ", \"from\": " + id(p->srcArmy) + ", \"to\": " + id(p->destArmy);
	if(const auto * p = dynamic_cast<const BulkSplitStack *>(pack))
		return "\"request\": \"BulkSplitStack\", \"slot\": " + id(p->src) + ", \"army\": " + id(p->srcOwner) + ", \"amount\": " + std::to_string(p->amount);
	if(const auto * p = dynamic_cast<const BulkMergeStacks *>(pack))
		return "\"request\": \"BulkMergeStacks\", \"slot\": " + id(p->src) + ", \"army\": " + id(p->srcOwner);
	if(const auto * p = dynamic_cast<const BulkSplitAndRebalanceStack *>(pack))
		return "\"request\": \"BulkSplitAndRebalanceStack\", \"slot\": " + id(p->src) + ", \"army\": " + id(p->srcOwner);
	if(const auto * p = dynamic_cast<const DisbandCreature *>(pack))
		return "\"request\": \"DisbandCreature\", \"slot\": " + id(p->pos) + ", \"army\": " + id(p->id);
	if(const auto * p = dynamic_cast<const UpgradeCreature *>(pack))
		return "\"request\": \"UpgradeCreature\", \"slot\": " + id(p->pos) + ", \"army\": " + id(p->id) + ", \"creature\": " + id(p->cid);
	if(const auto * p = dynamic_cast<const GarrisonHeroSwap *>(pack))
		return "\"request\": \"GarrisonHeroSwap\", \"town\": " + id(p->tid);
	if(const auto * p = dynamic_cast<const ExchangeArtifacts *>(pack))
		return "\"request\": \"ExchangeArtifacts\", \"from\": " + location(p->src) + ", \"to\": " + location(p->dst);
	if(const auto * p = dynamic_cast<const BulkExchangeArtifacts *>(pack))
		return "\"request\": \"BulkExchangeArtifacts\", \"from\": " + id(p->srcHero) + ", \"to\": " + id(p->dstHero) + ", \"swap\": " + (p->swap ? "true" : "false") + ", \"equipped\": " + (p->equipped ? "true" : "false") + ", \"backpack\": " + (p->backpack ? "true" : "false");
	if(const auto * p = dynamic_cast<const HireHero *>(pack))
		return "\"request\": \"HireHero\", \"hero\": " + id(p->hid) + ", \"next\": " + id(p->nhid) + ", \"town\": " + id(p->tid);
	if(const auto * p = dynamic_cast<const DismissHero *>(pack))
		return "\"request\": \"DismissHero\", \"hero\": " + id(p->hid);
	if(const auto * p = dynamic_cast<const CastAdvSpell *>(pack))
		return "\"request\": \"CastAdvSpell\", \"hero\": " + id(p->hid) + ", \"spell\": " + id(p->sid) + ", \"tile\": " + tile(p->pos);
	if(const auto * p = dynamic_cast<const CastleTeleportHero *>(pack))
		return "\"request\": \"CastleTeleportHero\", \"hero\": " + id(p->hid) + ", \"to\": " + id(p->dest) + ", \"source\": " + std::to_string(p->source);
	if(const auto * p = dynamic_cast<const VisitTownBuilding *>(pack))
		return "\"request\": \"VisitTownBuilding\", \"town\": " + id(p->tid) + ", \"building\": " + id(p->bid);
	if(const auto * p = dynamic_cast<const BuildBoat *>(pack))
		return "\"request\": \"BuildBoat\", \"shipyard\": " + id(p->objid);
	if(const auto * p = dynamic_cast<const DigWithHero *>(pack))
		return "\"request\": \"DigWithHero\", \"hero\": " + id(p->id);
	if(const auto * p = dynamic_cast<const SetFormation *>(pack))
		return "\"request\": \"SetFormation\", \"hero\": " + id(p->hid);
	if(dynamic_cast<const EndTurn *>(pack))
		return "\"request\": \"EndTurn\"";
	if(const auto * p = dynamic_cast<const QueryReply *>(pack))
		return "\"request\": \"QueryReply\", \"query\": " + id(p->qid) + ", \"reply\": " + (p->reply ? std::to_string(*p->reply) : "null");
	if(const auto * p = dynamic_cast<const MakeAction *>(pack))
		return "\"request\": \"MakeAction\", \"battle\": " + id(p->battleID) + ", \"action\": \"" + escaped(p->ba.toString()) + "\"";
	return "\"request\": \"" + escaped(typeid(*pack).name()) + "\"";
}
}

void AlphaMapAI::rawRequest(const CPackForServer * pack, uint64_t serial, const Choice * choice, const std::string & unmatched)
{
	if(!raw)
		return;
	std::string line = "{\"serial\": " + std::to_string(serial) + ", \"day\": " + std::to_string(cb->getDate(Date::DAY)) + ", \"player\": \"" + playerID.toString() + "\", \"actor\": \"" + (observedFlag == RECORD_HUMAN ? "human" : "nk2") + "\", " + described(pack)
		+ ", \"ours\": " + (choice ? "{\"type\": " + std::to_string(choice->type) + ", \"index\": " + std::to_string(choice->index) + ", \"cell\": " + std::to_string(choice->cell) + "}" : "null")
		+ ", \"note\": " + (unmatched.empty() ? "null" : "\"" + escaped(unmatched) + "\"") + "}\n";
	std::scoped_lock lock(rawMutex);
	*raw << line;
	raw->flush(); // (a game the player closes keeps every line)
}
