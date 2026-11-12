#pragma once

#include "Armies.h"
#include "Schema.generated.h"
#include "Tensors.h"

#include "lib/callback/CAdventureAI.h"
#include "lib/ResourceSet.h"
#include "lib/mapObjects/IMarket.h"
#include "lib/mapObjects/IObjectInterface.h"
#include "lib/int3.h"
#include "lib/json/JsonNode.h"

#include <atomic>
#include <condition_variable>
#include <fstream>
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
struct MoveHero;
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
	// Each window slot's hero token, -1 none (schema 17): the slot's own hero, or in a slot no hero of ours holds one more
	// window of a hero we have (w_extra), around tiles it reaches today that its own window does not show
	std::array<int, MapSchema::OWN_HEROES> windowHero;
	std::array<std::unordered_map<int, Entry>, MapSchema::N_ACTION_TYPES> entries;
	std::vector<int> answers; // the reply to send for each option of a question
	// The model's resource_worth for it (schema 12): what a purchase short of resources trades away the least of
	std::array<float, MapSchema::N_RESOURCES> worth{};

	Decision() { windowHero.fill(-1); }
	int3 windowTile(int window, int row, int column) const;
	/// MOVE's entry that takes our hero token `hero` to `tile` (the first of its windows that shows the tile, where
	/// move_mask allows it), -1 none
	int moveEntry(int hero, const int3 & tile) const;
	/// A choice's entry as the next decision's g_last_index echoes it: a MOVE's window slot as its hero's token (the
	/// model reads the first axis as the hero the decision was about: features.last_actors)
	int echoIndex(int type, int index) const;
	int holderOf(ObjectInstanceID id) const;
};

/// Schema 16: whether ARMY's entry `index` (flat in its table) may take target slot choice `t` (0 .. ARMY_SLOT_CHOICES-1),
/// read from the decision's inputs as the model reads them (campaign_model.model.army_slots): t < ARMY_SLOTS the slot the
/// amount goes to, a whole stack's legal ones (army_slot_mask), a part's fit ones (an empty slot or its own creature's,
/// never its own slot); t >= ARMY_SLOTS (ONE alone) the fit slot t - ARMY_SLOTS all but one go to, the one staying
bool armySlotAllowed(const Decision & decision, int index, int t);

/// A trade of a purchase short of resources: give units of sell (whole lots of the market's offer) for buy
struct Trade
{
	int sell = 0;
	int buy = 0;
	int give = 0;
};

/// What a purchase of cost lacks from have, bought at the market with what it leaves over: for each resource short, the
/// lots of the spare resource that lose the least worth a unit bought, the fewest that cover it (the user, 2026-11-04:
/// building or recruiting trades for itself, no amount for the model to set); none when no such trades cover it
std::optional<std::vector<Trade>> tradesFor(const IMarket * market, const TResources & cost, const TResources & have, const std::array<float, MapSchema::N_RESOURCES> & worth);
/// Each resource's gold at the market (its own sale price; gold 1): the worth that tests whether trades can cover a cost
std::array<float, MapSchema::N_RESOURCES> marketWorth(const IMarket * market);

/// The model's choice: an action type and its table entry (turn), or an option (question)
struct Choice
{
	int type = -1;  // ActionType; -1 for a question
	int index = -1; // flat index in the type's table, or the option
	int cell = -1;  // a targeted spell's tile (flat index in its hero's window), or ARMY's target slot (schema 16), -1 none
	float logp = 0;
	uint64_t record = 0; // the serial of its Record (0: none), so a no-op flags this decision, not a later dialog's
};

/// Nullkiller2's plan for one of our hero slots (RECORD_PLAN: Advice.cpp, a label of the foresight's, never an input):
/// where it sends the hero; x -1 no task of its plan, -2 a task with no tile of its own
struct PlanTarget
{
	int16_t x = -1;
	int16_t y = -1;
	int16_t z = -1;
	int16_t turns = 0; // the days to that tile
	float priority = 0; // its task's
};
using PlanTargets = std::array<PlanTarget, MapSchema::OWN_HEROES>;

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
	std::optional<PlanTargets> plan; // Nullkiller2's plan in the same state (RECORD_PLAN: Advice.cpp)
	std::vector<std::pair<int, int>> accept; // (type, entry) more of ours that follow Nullkiller2 there (RECORD_ACCEPT)
	std::string inputs; // wire encoded
	uint64_t serial = 0; // this player's count of records so far (not sent)
};

constexpr uint32_t RECORD_NO_EFFECT = 1;
constexpr uint32_t RECORD_TEACHER = 2; // Nullkiller2's decision (Teacher.cpp): its choice is the teacher's, no model's probability
constexpr uint32_t RECORD_ADVICE = 4; // our decision with Nullkiller2's choice in the same state beside it (Advice.cpp)
constexpr uint32_t RECORD_PLAN = 8; // Nullkiller2's plan in the same state beside it (Advice.cpp: each hero slot's PlanTarget)
// More of our legal entries that follow Nullkiller2 there, beside the advice's own (Advice.cpp: a step along its plan's
// path for each hero, an exchange's moves toward the army it wants): the teacher's targets, never inputs
constexpr uint32_t RECORD_ACCEPT = 16;
constexpr uint32_t RECORD_FALLBACK = 64; // (with RECORD_ADVICE) the advice is Nullkiller2's planned next step, its first request matching nothing of ours (Advice.cpp)
constexpr uint32_t RECORD_HUMAN = 32; // a human player's decision (Recorder.cpp), recorded as a teacher's: its choice the player's

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
	// What the agent did for the model's choices beyond the action itself (the game's summary): purchases short of
	// resources paid by trades, boats built, skills bought at a University, stacks of one creature merged
	int tradedPurchases = 0;
	int boatsBuilt = 0;
	int multiActions = 0; // (MULTI's made: executeMulti)
	int takeMismatches = 0; // (TAKE_BEST's made whose receiver ended other than the army priced: arrange)
	int skillsBought = 0;
	int stacksMerged = 0;
	// The objects whose guards a question has shown us (encodeQuestionGuards): their armies in a_* from then on, as they
	// are each time (after a battle, a new week), as a scouted bank's or guarded artifact's army shows to a human
	std::set<ObjectInstanceID> revealedGuards;
	// The teleport channels a hero of ours entered and found to lead nowhere (showTeleportDialog's impassable: a gate
	// with no other end): their entrances are no unseen exits from then on (unseenExit)
	std::set<TeleportChannelID> deadChannels;
	bool done = false;
	// (action type, table entry, target tile) of the actions that changed nothing since the state last changed: not
	// offered again until something changes (dropFutile), else the same no-op could be chosen MAX_DECISIONS_PER_DAY
	// times. A targeted spell's tile counts: one Dimension Door that failed blocks that tile, not the spell
	std::set<std::tuple<int, int, int>> futile;
	// alpha_heroes patch (bundle 60, the loop guard; e5: an ARTIFACT pair alternated 106 times in a game): ALPHA_MAP_LOOP_GUARD,
	// 1 unless set. The day's states (stateFingerprint) after each of our actions; an action that brings one back is
	// dropped like a futile one for the rest of the day (loopsToday: futile is cleared by any change, so a cycle of two
	// changing actions went on); counted in the summary's "loops_cut"
	bool loopGuard = true;
	std::set<std::vector<int64_t>> seenToday;
	std::set<std::tuple<int, int, int>> loopsToday;
	int loopsCut = 0;
	std::map<int, int> loopsCutTypes; // (by action type: the summary's "loops_cut_types")
	std::vector<int64_t> stateFingerprint() const;

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
	uint32_t observedFlag = RECORD_TEACHER; // (RECORD_HUMAN: a human player's recorder, Recorder.cpp)
	std::unique_ptr<std::ofstream> raw; // ALPHA_RECORD_RAW (a human's recorder): every request as sent, JSON lines
	std::mutex rawMutex;
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
	// Its meeting fixes (the 2026-11-09 audit; ALPHA_MAP_ADVICE_FIXES, on unless "0"): its Nullkiller2's adviceFixes and
	// exchangeSettled (Nullkiller.h). meetingAdvised: the turn decision being executed is its advice (an exchange that
	// opens gives the hero met the army, as Nullkiller2's own do). settled: each pair's armies as their last exchange
	// ended (the pair's lower id first), no exchange of theirs planned again while both armies are so
	bool adviceFixes = false;
	// Its visit again of a town its hero stands in as our GATHER (ALPHA_MAP_GATHER_ADVICE, off unless set: the trainer's
	// campaign.nk2_gather_weight teaches the label; Advice.cpp visitAgain), and how many of its advices were
	bool gatherAdvice = false;
	// Schema 19: Nullkiller2's view as inputs (ALPHA_MAP_NK2_INPUTS, off unless set; supply.py turns it off in a share of
	// training games): the advisor plans before every turn decision and candidateInputs writes its candidates, tier and
	// threats into the decision's k_*, g_nk_*, t_nk_threat, h_nk_threat (g_nk_present 0 without it). nkSettled: each
	// (hero, destination) in a hero's top 3 at the last such decision, the decisions in a row it has been there
	bool nk2Inputs = false;
	std::map<std::tuple<int, int, int, int>, int> nkSettled;
	void candidateInputs(Decision & decision);
	/// Schema 19's g_nk_action, g_nk_index, g_nk_cell: Nullkiller2's own choice here (its advice, as the record's), 0 none
	void labelInputs(Decision & decision, const std::optional<Choice> & advice);
	// alpha_heroes patch (2026-11-11, the user's own-games change): ALPHA_MAP_NK2_CHOICE, 1 unless set; at 0 labelInputs
	// writes nothing (g_nk_action, g_nk_index, g_nk_cell stay 0: Nullkiller2's own choice hidden, so the model's follow
	// mixture and the advantage head's named term are off and our own policy plays the whole game), while its candidates,
	// views, plan and threats are written as before and its advice is still asked and recorded on the Decision. One
	// setting a game (play.nk2_settings), shown in the summary as "choice_shown"
	bool nk2Choice = true;
	// alpha_heroes patch (2026-11-11, bundle 60, the teleport part): ALPHA_MAP_REVISIT_SKIP, 1 unless set. Nullkiller2's
	// request to re-visit the town of ours its hero stands in, with nothing for our GATHER there, is a no-op in the game
	// (a Defend composition holding the hero in the town: 102 of 132 own-tile misses in 8 games to day 28; e5's "into a
	// teleport" bucket). Its advice is then its next request, the hero held for defence (Nullkiller::adviceHeld), up to
	// three heroes in turn; counted in the summary's advice "revisit_skipped"
	bool revisitSkip = true;
	int revisitsSkipped = 0;
	int skipsMovingHeld = 0; // (a skip's next request moving a hero it held: the summary's "skip_moved_held")
	// The heroes held this day (cleared at our turn's start): a Defend composition's hero stays held at every advice of
	// our turn, as Nullkiller2 itself keeps it in the garrison it puts it in (ExchangeSwapTownHeroes' lock); held for one
	// advice only, its next replanning sent it out (the town guard's pair: 34 of 40 weak-town walk-outs its choice)
	std::set<ObjectInstanceID> heldToday;
	bool holdDay = true; // (ALPHA_MAP_HOLD_DAY, 1 unless set; 0: a hero held for one advice only)
	// alpha_heroes patch (bundle 60, the boat part): ALPHA_MAP_BOAT_ADVICE, 1 unless set. Nullkiller2 builds a boat at any
	// shipyard not an enemy's without going there (BuildBoat: VCMI's server allows it); a human builds one in a town of
	// theirs (our BUILD's SHIP) or by visiting a map shipyard and saying yes (our question, showShipyardDialog). Its
	// BuildBoat is matched to those: a town's SHIP; at a map shipyard its hero's MOVE onto it (TRAVEL toward it beyond
	// today), the shipyard then remembered (boatAdvised) so our question there takes YES as its advice. Counted in the
	// summary's advice "boats" (matched) and "boat_answers" (questions advised YES)
	bool boatAdvice = true;
	// alpha_heroes patch (bundle 60, teleports and boats; the user: what a human can do, the model must be able to do):
	// ALPHA_MAP_REVISIT, 1 unless set. The centre cell of a hero's own MOVE window (its own tile) is a MOVE's target where
	// re-visiting what it stands on does something a human does with Space: a teleporter entrance whose exit we have not
	// seen (going through it: Nullkiller2's Explore probes them, 25 requests in 4 island games), a map shipyard where a
	// boat can be built now (its question again: 9 of its BuildBoat requests in those games). Carried out as the
	// re-visit itself (Actions.cpp moveHero); Nullkiller2's own-tile request there is matched to it (Match.cpp)
	bool revisitEntry = true;
	// (the centre cell carried out: on a map shipyard the purchase itself, cb->buildBoat with no question, so it cannot
	// come to nothing; on an entrance the re-visit, a probe: one that finds no way is counted apart from failed actions,
	// the summary's "probes_no_way")
	bool probing = false;
	int probesNoWay = 0;
	bool revisitUseful(const CGHeroInstance * hero) const;
	mutable std::set<ObjectInstanceID> boatAdvised; // (matchRequest's, a const reading of the request: mutable)
	mutable int boatsMatched = 0;
	int boatAnswers = 0;
	/// the advising Nullkiller2's hero executing its task now (none: an invalid id)
	ObjectInstanceID advisorHero() const;
	// A test's only (ALPHA_MAP_FOLLOW_ADVICE, a chance, 0 unless set; "1" plays its every advice): the advised entry played
	// instead of our sampled one at that chance (paired_eval's follow arms)
	double followAdvice = 0;
	// Its first request matching nothing of ours (a teleport's move, a boat, a disband): the advice is the first task of its
	// plan whose first step is an entry of ours (planFirstStep), flagged RECORD_FALLBACK; counted apart in adviceJson
	bool adviceFallback = false;
	int adviceFallbacks = 0;
	std::mt19937 followRng{20261111};
	// The model's phi (our empire's value in gold) at the player's first turn decision of each day (schema 19's
	// g_phi_past: a week ago, else the oldest day there is)
	std::map<int, double> phiByDay;
	int adviceGathers = 0;
	int visitAgain(const Decision & decision, const MoveHero & move, const int3 & intended) const;
	std::atomic<bool> meetingAdvised = false;
	std::map<std::pair<ObjectInstanceID, ObjectInstanceID>, std::pair<std::map<CreatureID, int>, std::map<CreatureID, int>>> settled;
	mutable std::shared_mutex settledMutex; // (read by its pathfinding's threads)
	void settleExchange(ObjectInstanceID first, ObjectInstanceID second);
	bool exchangeSettled(const CGHeroInstance * one, const CGHeroInstance * two) const;
	void initAdvice();
	bool adviceDrawn(); // this decision gets the advice (at adviceProbability)
	// plan: Nullkiller2's plan there, its targets in our hero slots (with a Nullkiller2 that keeps it: NK2_KEEPS_PLAN)
	std::optional<Choice> advise(const Decision & decision, std::optional<PlanTargets> & plan, std::vector<std::pair<int, int>> & accept);
	// accept: our other moves toward the army Nullkiller2 wants (RECORD_ACCEPT)
	std::optional<Choice> adviseExchange(const Decision & decision, ObjectInstanceID first, ObjectInstanceID second, std::vector<std::pair<int, int>> & accept);
	// MULTI's entries doing what Nullkiller2's next requests would (its advice recruiting in a town, a hero visiting a town), accepted beside its label
	void acceptMulti(const Decision & decision, const std::optional<Choice> & advice, std::vector<std::pair<int, int>> & accept) const;
	std::string adviceJson() const;
	// A request of Nullkiller2's as our turn decision's legal entry doing the same; false (unmatched: why) when none.
	// intended: a move's destination when the caller knows it (its plan's tile), else the request's own tile is taken
	bool matchRequest(const Decision & decision, const CPackForServer * pack, Choice & choice, bool & exact, std::string & unmatched, int3 intended = int3(-1, -1, -1)) const;
	/// alpha_heroes diagnostic (2026-11-11, logging only): an own-tile move request named by what is under the hero (Match.cpp)
	std::string ownTileRequest(const Decision & decision, const CGHeroInstance * hero, const int3 & intended, size_t path) const;
	/// alpha_heroes diagnostic: the advising Nullkiller2's executing task (its description; empty without it)
	std::string advisorTask() const;
	std::string teacherJson() const;
	// Our marketplace (the first town of ours with one; its rates count all of them): trades and purchases short of
	// resources go through it; nullptr with none
	const IMarket * ourMarket() const;
	// A purchase of cost short of resources: its trades made first (tradesFor at decision's worth; the market's own
	// prices when that finds none)
	void payByTrades(const TResources & cost, const Decision & decision);
	// The stacks of one creature in an army of ours merged into one (the user, 2026-11-04: "whenever possible, unify")
	bool mergeDuplicates(const CArmedInstance * army, bool keepSingles = false); // (a creature's stacks in one army into one; whether any merged)
	void unifyStacks(); // (before each turn decision: every army of ours's duplicates merged, a single creature kept)
	// A University's skill as a yes/no: YES buys it (the option a SKILL: the skill, basic, its price)
	Decision universityQuestion(const CGHeroInstance * hero, SecondarySkill skill, int price);

public:
	AlphaMapAI();
	~AlphaMapAI() override;

	// Nullkiller2 as the campaign model's teacher (Teacher.cpp, the AlphaTeacherAI library): Nullkiller2 plays, and each
	// of its turn requests is recorded as the turn decision of ours that does the same, our inputs as they were then
	// (RECORD_TEACHER; campaign_model/imitate.py learns from them). This object only encodes and records, never acts
	void initTeacher(std::shared_ptr<Environment> environment, std::shared_ptr<CCallback> callback);
	void teacherTurnStarts();
	void observe(const CPackForServer * pack, bool byPlayer = true); // (byPlayer false: a human's auto-combat's battle action)
	// The teacher's questions (Teacher.cpp): each encoded as it is asked, before Nullkiller2 answers it, and recorded
	// with that answer (observe's QueryReply)
	void teacherBlockingDialog(const std::string & text, const std::vector<Component> & components, QueryID askID, int soundID, bool selection, bool cancel);
	void teacherLevelUp(const CGHeroInstance * hero, const std::vector<SecondarySkill> & skills, QueryID queryID);
	// A human player's recorder (Recorder.cpp, the AlphaRecorderAI library beside the client's player interface): the
	// teacher's recording of the player's requests (RECORD_HUMAN), to ALPHA_RECORD_HUMAN_SOCKET, each request raw too
	// (ALPHA_RECORD_RAW), no day cap; finishRecording sends what is left when the player leaves the game
	void initHuman(std::shared_ptr<Environment> environment, std::shared_ptr<CCallback> callback);
	void finishRecording();
	const std::string & recordKey() const { return gameKey; } // (the segment's key: Recorder.cpp's battles go under it too)
	static std::string sessionKey(const std::string & key); // ALPHA_MAP_GAME_KEY's, as this start of a game records it

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
	void showShipyardDialog(const IShipyard * obj) override;
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
	Choice decide(Decision & decision, const std::optional<Choice> & advice = std::nullopt, const std::optional<PlanTargets> & plan = std::nullopt, const std::vector<std::pair<int, int>> & accept = {});
	void noEffect(const Choice & choice);
	Choice sample(const Decision & decision, const Outputs & outputs);
	void record(const Decision & decision, const Choice & choice, const Outputs & outputs, uint32_t version);
	void sendRecords(bool final);
	// observe's raw line of a request (ALPHA_RECORD_RAW): its fields as sent, and what of ours it was (serial: its
	// record's, 0 none; choice: our action, or none with unmatched: why)
	void rawRequest(const CPackForServer * pack, uint64_t serial, const Choice * choice, const std::string & unmatched, bool byPlayer = true);

	// Inputs and legality (Encoder.cpp)
	Decision encode(MapSchema::DecisionKind kind, ObjectInstanceID exchangeFirst = {}, ObjectInstanceID exchangeSecond = {});
	void encodeGlobal(Decision & decision);
	void encodeHeroes(Decision & decision, PathfinderCache & paths);
	void encodeTowns(Decision & decision);
	void encodeWindows(Decision & decision, PathfinderCache & paths);
	/// A teleporter's entrance none of whose exits we have seen (TILE_UNSEEN_EXIT, OBJECT_UNSEEN_EXIT)
	bool unseenExit(const CGObjectInstance * obj) const;
	void encodeMap(Decision & decision, const std::vector<int3> & ours);
	void encodeTiles(Decision & decision);
	void encodeTileBlocks(Decision & decision, const std::vector<const CGHeroInstance *> & heroes, int side); // (a map bigger than MAP_TILES a side)
	void dropFutile(Decision & decision);
	void encodeTurnActions(Decision & decision, PathfinderCache & paths);
	void encodeArmyMoves(Decision & decision, const std::vector<std::pair<int, int>> & pairs);
	// ARTIFACT's entries: GIVE between an exchange's two heroes of ours (first, second: their hero tokens), EQUIP and
	// ASSEMBLE in a turn (first < 0)
	void encodeArtifacts(Decision & decision, int first, int second);
	void encodeMulti(Decision & decision, int first, int second); // (MULTI: a turn's towns' with first -1, else an exchange of heroes first and second's)
	void executeMulti(const Entry & e); // (MULTI's requests: Actions.cpp)
	void arrange(const CArmedInstance * giver, const CArmedInstance * receiver, const armies::Take & take); // (TAKE_BEST's moves: the armies made take's)
	void encodeQuestionGuards(Decision & decision, const std::string & text, const std::vector<Component> & components);
	void encodeOptionContext(Decision & decision, ObjectInstanceID hero, int3 target);
	// The questions' decisions, as asked (the handlers decide them; the teacher's recorder keeps them for the answer)
	Decision blockingQuestion(const std::string & text, const std::vector<Component> & components, int soundID, bool selection, bool cancel);
	Decision levelUpQuestion(const CGHeroInstance * h, const std::vector<SecondarySkill> & skills);
	int owner(PlayerColor color) const;

	// Carrying out a choice (Actions.cpp); false when nothing happened
	bool execute(const Decision & decision, const Choice & choice);
	bool moveHero(ObjectInstanceID heroID, int3 destination);
};
