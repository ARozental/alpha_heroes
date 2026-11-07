#pragma once

#include "lib/callback/CBattleGameInterface.h"

#include "AttackValues.h"
#include "BattleSim.h"
#include "Dump.h"
#include "Encoder.h"
#include "History.h"
#include "Policy.h"
#include "Reward.h"

#include "lib/battle/BattleAction.h"

#include <filesystem>
#include <optional>
#include <random>
#include <vector>

VCMI_LIB_NAMESPACE_BEGIN
class CPlayerBattleCallback;
VCMI_LIB_NAMESPACE_END

class IBattleModel;

/// Battle AI that plays with an ONNX model implementing battle_model's contract.
///
/// Configured through environment variables:
///   ALPHA_BATTLE_MODEL        path to the .onnx file (required; without it StupidAI plays)
///   ALPHA_BATTLE_TEMPERATURE  sampling temperature, 0 = always the most likely action (default 1)
///   ALPHA_BATTLE_SEED         random seed (default: random)
///   ALPHA_BATTLE_DUMP_DIR     write inputs/outputs/action of decisions there (battle_model/check_dumps.py)
///   ALPHA_BATTLE_DUMP_LIMIT   decisions to dump per process (default 50)
///   ALPHA_BATTLE_RECORD_DIR   record the battle for training (battle_model/data.py): every decision plus
///                             events.jsonl (start armies, units alive at each round end, end result)
///   ALPHA_BATTLE_SEARCH_SIMULATIONS  > 0: search every decision with that many simulations (Gumbel AlphaZero,
///                             BattleSearch.h); the network's choice when the search fails. Default 0: no search
///   ALPHA_BATTLE_SEARCH_CONSIDERED   root actions the search starts with (default 16)
///   ALPHA_BATTLE_DELEGATE     another battle AI (e.g. BattleAI) plays every decision; the battle is still
///                             recorded: its events, and each decision with the action the other AI took (the
///                             outputs are our model's), e.g. for baselines and for checking battle_model/stupid_ai.py
///   ALPHA_BATTLE_TEST_FAULT   test hook, comma list: "surrender" / "spell" send one action per battle the
///                             server must reject, "decoder" corrupts one chosen action per battle
///
/// Every chosen action is checked against the masks before it is sent; an illegal one is excluded and
/// the model chooses again. If the server still rejects a hero spell, retreat or surrender, it asks the
/// same unit again: that choice is excluded and the next best legal action is taken.
///
/// The model plays every decision: units, our war machines when the hero controls them (CATAPULT, HEAL, the ballista's
/// SHOOT), and the tactics phase (TACTICS_MOVE one unit after another, TACTICS_END). An attacker retreats when it can
/// after STALEMATE_ROUNDS rounds in a row with no damage done (VCMI has no round limit), and any side at ROUND_LIMIT
/// (the side ahead a few rounds later). An error is counted and the
/// unit defends (playDefault); errors must stay at 0.
class AlphaBattleAI : public CBattleGameInterface
{
	std::shared_ptr<CBattleCallback> cb;
	std::shared_ptr<Environment> env;
	std::shared_ptr<CBattleGameInterface> fallback; // only ALPHA_BATTLE_DELEGATE's AI (baseline evals): no StupidAI
	std::shared_ptr<IBattleModel> model;
	BattleHistory history; // what the battle showed so far (History.h), for the encoder
	std::string inferenceSocket; // ALPHA_BATTLE_INFERENCE_SOCKET: the training's inference server plays instead of an ONNX file
	std::string inferenceName; // ALPHA_BATTLE_INFERENCE_MODEL: which of its models ("current", the default, or e.g. "v350")
	std::string modelPath; // ALPHA_BATTLE_MODEL; reloaded at a battle's start when the file changed (training updates it)
	std::filesystem::file_time_type modelTime{};
	std::string modelFile; // the file modelPath pointed at when the model was loaded: the version that plays
	void loadModel();
	// Tactics phase: our units still to place, one at a time (each from the board the last placement left)
	std::vector<uint32_t> tacticsQueue;
	bool tacticsPending = false; // a TACTICS_MOVE was sent: the next placement waits for it to be applied
	void placeNextTactic(const BattleID & battleID);
	int lastDamageRound = 0; // the last round any unit took damage (the stalemate rule)
	void playDefault(const BattleID & battleID, const CStack * stack);
	std::string scenarioJson(const BattleID & battleID, const CCreatureSet * army1, const CCreatureSet * army2) const;
	std::array<const CGHeroInstance *, 2> heroes{}; // per battle side, from battleStart
	std::mt19937 rng;
	float temperature = 1.0f;
	bool delegated = false; // ALPHA_BATTLE_DELEGATE, or observing
	// ALPHA_BATTLE_OBSERVE: another plays (a human's interface: agents/AlphaMapAI/Recorder.cpp passes us its battle
	// events) and we record what it does, as delegated but with no AI of ours to play, its callback left as it is
	bool observing = false;
	bool refreshPending = false; // (delegated: a hero spell recorded, the stack still to act: its state again after it)
	std::unique_ptr<EncodedBattle> delegatePending; // the state the delegate was asked to act in, until its action starts
	uint32_t delegateStack = 0;
	bool wasWaitingForRealize = false;

	// Rejection handling: actions excluded for the current unit's turn
	struct TurnKey
	{
		int battle;
		int round;
		int unit;
		bool operator==(const TurnKey &) const = default;
	};
	std::optional<TurnKey> lastTurn;
	std::vector<ChosenAction> rejected;
	int lastActionType = -1;
	bool lastActionForced = false; // last action was a test fault, not the model's choice
	bool spellCastSeen = false;
	std::vector<std::string> faults; // ALPHA_BATTLE_TEST_FAULT
	std::vector<std::string> faultsDoneThisBattle;

	// Forced opening (ALPHA_BATTLE_PREFIX, a JSON-lines file of actions as in action.json): played instead of
	// the model's choice for the first decisions, to replay a recorded battle up to a point. Battles are
	// deterministic given VCMI's seed and our actions, so this restores any state of a recorded battle.
	std::vector<ChosenAction> prefix;
	size_t prefixPlayed = 0;

	// Training record: kept in memory (record) and sent to the trainer at the end of the battle
	// (ALPHA_BATTLE_RECORD_SOCKET, ALPHA_BATTLE_RECORD_KEY); also written to ALPHA_BATTLE_RECORD_DIR for debugging
	std::unique_ptr<BattleRecord> record; // null when not recording
	std::string recordSocket;
	std::string recordKey;
	std::string recordDir;
	int recordedDecisions = 0;
	void recordEvent(const std::string & json);
	std::string aliveArmies(const BattleID & battleID) const;
	std::string summonedArmies(const BattleID & battleID) const;
	int ourMana(const BattleID & battleID) const;
	int enemyMana(const BattleID & battleID) const; // the enemy hero's spell points, as a player sees them in battle

	struct Stats
	{
		int decisions = 0;
		int fallbacks = 0;
		int errors = 0;
		std::array<int, 16> types{};
		std::array<int, 4> spellTargets{}; // no target, unit, hex, two targets
		double modelMs = 0;
		bool siege = false;
		bool walls = false;
		int serverRejections = 0; // handled: server rejected an action, chose again
		int preSendRejections = 0; // handled: chosen action failed the mask check, chose again
		int faultsInjected = 0;
		// ALPHA_BATTLE_SIM_CHECK: our actions replayed in a BattleSimulation and compared with the real game
		int simChecks = 0;
		int simRejected = 0; // the simulation's rules refused an action the real game accepted
		int simExactMismatches = 0; // a field that does not depend on dice differs (a bug in the simulation)
		int simDiceMismatches = 0; // count or health differs: expected when dice were rolled (damage, luck)
		double simCopyMs = 0;
		double simApplyMs = 0;
		// ALPHA_BATTLE_SEARCH_SIMULATIONS
		int searched = 0;
		int searchErrors = 0;
		int searchChanged = 0; // the search played another action than the network's most likely one
		int searchEvaluations = 0;
		double searchMs = 0, searchCopyMs = 0, searchEncodeMs = 0, searchModelMs = 0, searchApplyMs = 0;
		AttackValueStats attackValues;
	} stats;

	int searchSimulations = 0;
	int searchConsidered = 16;
	AlphaReward::BattleEnd battleStartValues; // armies and mana at the start of the battle, as the reward counts them
	double surrenderCostPaid = 0;
	/// The last search's root actions and improved policy (JSON fields), recorded with its decision for training
	/// toward the search's policy (battle_model/data.py: the "search" event)
	std::string searchTarget;
	/// The searched action in the slots of encoded, or nothing if the search failed
	std::optional<ChosenAction> searchAction(const BattleID & battleID, const EncodedBattle & encoded, const AlphaSchema::OutputTensors & outputs);

	// ALPHA_BATTLE_SIM_CHECK=1: each of our actions is first played in a copy of the battle (BattleSimulation);
	// when the real game has applied it (actionFinished) the two battles are compared
	bool simCheckEnabled = false;
	std::unique_ptr<BattleSimulation> simCheck;
	AttackValueCache attackCache; // encodeAttackValues' copy of the game state, one per battle
	std::optional<BattleAction> simCheckAction;
	void compareWithSimulation(const BattleID & battleID);

	bool previousActionRejected(const CPlayerBattleCallback & battle) const;
	bool injectFault(const std::string & fault);

	void playWithModel(const BattleID & battleID, const CStack * stack);
	BattleAction toBattleAction(const BattleID & battleID, const CStack * stack, const EncodedBattle & encoded, const ChosenAction & action) const;
	void execute(const BattleID & battleID, const CStack * stack, const EncodedBattle & encoded, const ChosenAction & action);

public:
	~AlphaBattleAI() override;

	void initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB) override;
	void initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB, AutocombatPreferences autocombatPreferences) override;

	void battleStart(const BattleID & battleID, const CCreatureSet * army1, const CCreatureSet * army2, int3 tile, const CGHeroInstance * hero1, const CGHeroInstance * hero2, BattleSide side, bool replayAllowed) override;
	void battleEnd(const BattleID & battleID, const BattleResult * br, QueryID queryID) override;
	void actionStarted(const BattleID & battleID, const BattleAction & action) override;
	void actionFinished(const BattleID & battleID, const BattleAction & action) override;
	void battleSpellCast(const BattleID & battleID, const BattleSpellCast * sc) override;
	void battleNewRoundFirst(const BattleID & battleID) override;
	void activeStack(const BattleID & battleID, const CStack * stack) override;
	void battleStacksAttacked(const BattleID & battleID, const std::vector<BattleStackAttacked> & bsa, bool ranged) override;
	void yourTacticPhase(const BattleID & battleID, int distance) override;
};
