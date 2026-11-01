#pragma once

#include "Schema.generated.h"
#include "Tensors.h"

#include "lib/callback/CAdventureAI.h"
#include "lib/int3.h"
#include "lib/json/JsonNode.h"

#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <random>
#include <unordered_map>

class AsyncRunner;
class InferenceClient;

VCMI_LIB_NAMESPACE_BEGIN
class CArmedInstance;
class CGHeroInstance;
class CGTownInstance;
class PathfinderCache;
VCMI_LIB_NAMESPACE_END

/// What one legal table entry does: the objects and the numbers the engine call needs (by action type, see Actions.cpp)
struct Entry
{
	ObjectInstanceID first;
	ObjectInstanceID second;
	int a = 0;
	int b = 0;
	int c = 0;
	int d = 0;
	int3 tile;
};

/// One decision: the model's inputs and what each legal entry of its tables means
struct Decision
{
	Inputs inputs;
	MapSchema::DecisionKind kind = MapSchema::DecisionKind::TURN;
	std::array<ObjectInstanceID, MapSchema::N_HEROES> heroes;   // hero tokens
	std::array<ObjectInstanceID, MapSchema::N_TOWNS> towns;     // town tokens
	std::array<ObjectInstanceID, MapSchema::N_HOLDERS> holders; // army holders (ARMY, UPGRADE)
	std::array<int3, MapSchema::OWN_HEROES> centers;            // the tile in the middle of each window
	std::array<std::unordered_map<int, Entry>, MapSchema::N_ACTION_TYPES> entries;
	std::vector<int> answers; // the reply to send for each option of a question

	int3 windowTile(int hero, int row, int column) const;
	int holderOf(ObjectInstanceID id) const;
};

/// The model's choice: an action type and its table entry (turn), or an option (question)
struct Choice
{
	int type = -1;  // ActionType; -1 for a question
	int index = -1; // flat index in the type's table, or the option
	int cell = -1;  // a targeted spell's tile (flat index in its hero's window), -1 none
	float logp = 0;
	uint64_t record = 0; // the serial of its Record (0: none), so a no-op flags this decision, not a later dialog's
};

/// A decision as sent to the trainer (campaign_model/records.py)
struct Record
{
	int day = 0;
	int kind = 0;
	int decisionsToday = 0;
	uint32_t version = 0;
	Choice choice;
	std::array<float, MapSchema::N_VALUE_OFFSETS> value{};
	uint32_t flags = 0; // RECORD_NO_EFFECT: the action was carried out and changed nothing
	std::optional<Choice> advice; // Nullkiller2's choice in the same state (RECORD_ADVICE: Advice.cpp)
	std::string inputs; // wire encoded
	uint64_t serial = 0; // this player's count of records so far (not sent)
};

constexpr uint32_t RECORD_NO_EFFECT = 1;
constexpr uint32_t RECORD_TEACHER = 2; // Nullkiller2's decision (Teacher.cpp): its choice is the teacher's, no model's probability
constexpr uint32_t RECORD_ADVICE = 4; // our decision with Nullkiller2's choice in the same state beside it (Advice.cpp)

/// Adventure-map AI: every decision is the campaign model's (campaign_model/), asked of the inference server; battles
/// are played by the configured battle AI. Built on Nullkiller2's patterns for threads and queries (AIGateway).
///
/// Environment: ALPHA_MAP_INFERENCE_SOCKET (required), ALPHA_MAP_MODEL (model name, default "current"),
/// ALPHA_MAP_RECORD_SOCKET and ALPHA_MAP_GAME_KEY (records and daily ledgers for the trainer), ALPHA_MAP_DAYS (stop
/// after this day), ALPHA_MAP_SEED_<COLOR> (sampling), ALPHA_MAP_TEMPERATURE (1 = the model's own probabilities).
class AlphaMapAI : public CAdventureAI
{
	std::shared_ptr<CCallback> cb;
	std::unique_ptr<AsyncRunner> asyncTasks;
	std::unique_ptr<InferenceClient> inference;
	std::string inferenceSocket;
	/// ALPHA_MAP_REQUIRE_MODEL (training and evals): a game the model cannot play is ended, not played on with every turn
	/// ended unasked (2026-10-31: whenever the inference servers were down, thousands of such games reached their day caps
	/// with no decision and were taken as finished). The servers are waited for a minute first, then the process aborts
	bool requireModel = false;
	void connectInference(bool waiting);
	std::mutex inferenceMutex;
	std::string modelName = "current";
	std::string recordSocket;
	bool recordsPerDecision = false; // ALPHA_MAP_RECORD_FORMAT=2: "AMZ2", each decision compressed apart (records.py)
	std::string gameKey = "game";
	int dayCap = 0;
	double temperature = 1.0;
	std::mt19937_64 rng;
	std::mutex rngMutex;

	// What the engine is busy with: the turn waits for these (AIStatus in Nullkiller2)
	std::mutex statusMutex;
	std::condition_variable statusChanged;
	std::map<QueryID, std::string> queries;
	int battle = 0; // 0 none, 1 upcoming, 2 ongoing, 3 ending
	int visits = 0;
	bool moving = false;
	bool havingTurn = false;
	std::atomic<bool> stopping = false;

	// Records for the trainer, sent at the end of each of our turns
	std::mutex recordsMutex;
	std::vector<Record> records;
	uint64_t recordSerial = 0;
	// The battles our heroes fight, for the trainer's foresight labels (campaign_model/records.py): at battleStart both
	// sides' armies and heroes, keyed to the decision that led there (the record serial before it); at battleEnd the
	// outcome and casualties; sent with the records
	JsonNode battleNode;
	bool battleOpen = false;
	std::vector<std::string> battles;
	uint64_t battlesSent = 0;
	int recordsSent = 0;
	int day = 0;
	int decisionsToday = 0;
	int errors = 0;
	int failedActions = 0;
	bool done = false;
	// (action type, table entry, target tile) of the actions that changed nothing since the state last changed: not
	// offered again until something changes (dropFutile), else the same no-op could be chosen MAX_DECISIONS_PER_DAY
	// times. A targeted spell's tile counts: one Dimension Door that failed blocks that tile, not the spell
	std::set<std::tuple<int, int, int>> futile;

	// What a hero is doing, for the questions it raises
	ObjectInstanceID activeHero;
	int3 activeTarget;
	int3 intendedExit; // the next tile of a path through a teleport
	// The model's memory as its last turn decision left it (its memory output x MEMORY_SCALE), passed with every
	// decision (mem): what this player's decisions carry forward (model_v2.Memory); 0 at the start of the game
	std::vector<int64_t> memory = std::vector<int64_t>(MapSchema::MEMORY_SLOTS * MapSchema::MEMORY_WIDTH, 0);
	// Our previous turn decision today and whether it changed anything: echoed to the model (g_last_*), nothing more
	Choice lastChoice;
	int lastChanged = -1;

	// The teacher's recorder (Teacher.cpp): the requests of Nullkiller2's with no equal among our actions, by kind; the
	// matched ones, and those matched to our nearest amount (recruits, trades, splits) or area (a move out of reach)
	bool teacher = false;
	int teacherMatched = 0;
	int teacherApproximate = 0;
	std::map<std::string, int> teacherUnmatched;
	std::map<QueryID, Decision> teacherQuestions; // (asked, not yet answered: under recordsMutex)

	// Nullkiller2's advice (Advice.cpp; ALPHA_MAP_NK2_ADVICE: the chance a turn or exchange decision gets it, 0 off): a
	// Nullkiller2 of this player's, its requests going to a client that sends none; asked before a sampled decision of
	// ours what it would do there, its first request stops it and is matched to our legal entry (matchRequest)
	struct Advisor;
	std::shared_ptr<Advisor> advisor;
	double adviceProbability = 0;
	int advised = 0;
	int adviceMatched = 0;
	int exchangeAdvised = 0;
	int exchangeAdviceMatched = 0;
	int exchangeAdviceEnds = 0;
	double adviceSeconds = 0;
	std::map<std::string, int> adviceUnmatched;
	void initAdvice();
	bool adviceDrawn(); // this decision gets the advice (at adviceProbability)
	std::optional<Choice> advise(const Decision & decision);
	std::optional<Choice> adviseExchange(const Decision & decision, ObjectInstanceID first, ObjectInstanceID second);
	std::string adviceJson() const;
	// A request of Nullkiller2's as our turn decision's legal entry doing the same; false (unmatched: why) when none.
	// intended: a move's destination when the caller knows it (its plan's tile), else the request's own tile is taken
	bool matchRequest(const Decision & decision, const CPackForServer * pack, Choice & choice, bool & exact, std::string & unmatched, int3 intended = int3(-1, -1, -1)) const;
	std::string teacherJson() const;

public:
	AlphaMapAI();
	~AlphaMapAI() override;

	// Nullkiller2 as the campaign model's teacher (Teacher.cpp, the AlphaTeacherAI library): Nullkiller2 plays, and each
	// of its turn requests is recorded as the turn decision of ours that does the same, our inputs as they were then
	// (RECORD_TEACHER; campaign_model/imitate.py learns from them). This object only encodes and records, never acts
	void initTeacher(std::shared_ptr<Environment> environment, std::shared_ptr<CCallback> callback);
	void teacherTurnStarts();
	void observe(const CPackForServer * pack);
	// The teacher's questions (Teacher.cpp): each encoded as it is asked, before Nullkiller2 answers it, and recorded
	// with that answer (observe's QueryReply)
	void teacherBlockingDialog(const std::string & text, const std::vector<Component> & components, QueryID askID, int soundID, bool selection, bool cancel);
	void teacherLevelUp(const CGHeroInstance * hero, const std::vector<SecondarySkill> & skills, QueryID queryID);

	std::string getBattleAIName() const override;
	void initGameInterface(std::shared_ptr<Environment> env, std::shared_ptr<CCallback> callback) override;
	void yourTurn(QueryID queryID) override;
	void heroGotLevel(const CGHeroInstance * hero, PrimarySkill pskill, std::vector<SecondarySkill> & skills, QueryID queryID) override;
	void commanderGotLevel(const CCommanderInstance * commander, std::vector<ui32> skills, QueryID queryID) override;
	void showBlockingDialog(const std::string & text, const std::vector<Component> & components, QueryID askID, const int soundID, bool selection, bool cancel, bool safeToAutoaccept) override;
	void showGarrisonDialog(const CArmedInstance * up, const CGHeroInstance * down, bool removableUnits, QueryID queryID, const MetaString & customTitle) override;
	void showTeleportDialog(const CGHeroInstance * hero, TeleportChannelID channel, TTeleportExitsList exits, bool impassable, QueryID askID) override;
	void showMapObjectSelectDialog(QueryID askID, const Component & icon, const MetaString & title, const MetaString & description, const std::vector<ObjectInstanceID> & objects) override;
	void showRecruitmentDialog(const CGDwelling * dwelling, const CArmedInstance * dst, int level, QueryID queryID) override;
	void showTavernWindow(const CGObjectInstance * object, const CGHeroInstance * visitor, QueryID queryID) override;
	void showMarketWindow(const IMarket * market, const CGHeroInstance * visitor, QueryID queryID) override;
	void showUniversityWindow(const IMarket * market, const CGHeroInstance * visitor, QueryID queryID) override;
	void heroExchangeStarted(ObjectInstanceID hero1, ObjectInstanceID hero2, QueryID queryID) override;
	std::optional<BattleAction> makeSurrenderRetreatDecision(const BattleID & battleID, const BattleStateInfoForRetreat & battleState) override;

	void playerBlocked(int reason, bool start) override;
	void heroVisit(const CGHeroInstance * visitor, const CGObjectInstance * visitedObj, bool start) override;
	// The events Nullkiller2's memory follows, passed on to the advising one (Advice.cpp): it knows what a seated
	// Nullkiller2 would (the user, 2026-10-31: better too much than too little). Only these: its handlers of questions
	// and visits keep the status its actions wait on
	void objectRemoved(const CGObjectInstance * obj, const PlayerColor & initiator) override;
	void newObject(const CGObjectInstance * obj) override;
	void tileRevealed(const FowTilesType & pos) override;
	void tileHidden(const FowTilesType & pos) override;
	void objectPropertyChanged(const SetObjectProperty * sop) override;
	void heroCreated(const CGHeroInstance * hero) override;
	void battleStart(const BattleID & battleID, const CCreatureSet * army1, const CCreatureSet * army2, int3 tile, const CGHeroInstance * hero1, const CGHeroInstance * hero2, BattleSide side, bool replayAllowed) override;
	void battleEnd(const BattleID & battleID, const BattleResult * br, QueryID queryID) override;
	void battleResultsApplied() override;
	void requestRealized(PackageApplied * pa) override;
	void playerStartsTurn(PlayerColor player) override;
	void gameOver(PlayerColor player, const EVictoryLossCheckResult & victoryLossCheckResult) override;
	void finish() override;

private:
	// Threads and queries (AlphaMapAI.cpp)
	void addQuery(QueryID id, const std::string & description);
	void answer(QueryID id, int selection);
	void runAsync(const std::string & description, std::function<void()> task);
	void waitTillFree();
	bool isFree();
	void makeTurn();
	void endTurn();
	void exchange(ObjectInstanceID first, ObjectInstanceID second, QueryID queryID);

	// The model (AlphaMapAI.cpp)
	Choice decide(Decision & decision, const std::optional<Choice> & advice = std::nullopt);
	void noEffect(const Choice & choice);
	Choice sample(const Decision & decision, const Outputs & outputs);
	void record(const Decision & decision, const Choice & choice, const Outputs & outputs, uint32_t version);
	void sendRecords(bool final);

	// Inputs and legality (Encoder.cpp)
	Decision encode(MapSchema::DecisionKind kind, ObjectInstanceID exchangeFirst = {}, ObjectInstanceID exchangeSecond = {});
	void encodeGlobal(Decision & decision);
	void encodeHeroes(Decision & decision, PathfinderCache & paths);
	void encodeTowns(Decision & decision);
	void encodeWindows(Decision & decision, PathfinderCache & paths);
	void encodeMap(Decision & decision, const std::vector<int3> & ours);
	void encodeTiles(Decision & decision);
	void dropFutile(Decision & decision);
	void encodeTurnActions(Decision & decision, PathfinderCache & paths);
	void encodeArmyMoves(Decision & decision, const std::vector<std::pair<int, int>> & pairs);
	void encodeOptionContext(Decision & decision, ObjectInstanceID hero, int3 target);
	// The questions' decisions, as asked (the handlers decide them; the teacher's recorder keeps them for the answer)
	Decision blockingQuestion(const std::string & text, const std::vector<Component> & components, int soundID, bool selection, bool cancel);
	Decision levelUpQuestion(const CGHeroInstance * h, const std::vector<SecondarySkill> & skills);
	int owner(PlayerColor color) const;

	// Carrying out a choice (Actions.cpp); false when nothing happened
	bool execute(const Decision & decision, const Choice & choice);
	bool moveHero(ObjectInstanceID heroID, int3 destination);
};
