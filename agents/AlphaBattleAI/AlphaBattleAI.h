#pragma once

#include "lib/callback/CBattleGameInterface.h"

#include "Policy.h"

#include <optional>
#include <random>
#include <vector>

VCMI_LIB_NAMESPACE_BEGIN
class CPlayerBattleCallback;
VCMI_LIB_NAMESPACE_END

class BattleModel;
struct EncodedBattle;

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
///   ALPHA_BATTLE_TEST_FAULT   test hook, comma list: "surrender" / "spell" send one action per battle the
///                             server must reject, "decoder" corrupts one chosen action per battle
///
/// Every chosen action is checked against the masks before it is sent; an illegal one is excluded and
/// the model chooses again. If the server still rejects a hero spell, retreat or surrender, it asks the
/// same unit again: that choice is excluded and the next best legal action is taken.
///
/// War machines and the tactics phase are scripted (StupidAI), as is everything from round
/// ROUND_CAP on, so a battle cannot go on forever. Any error falls back to StupidAI for that decision.
class AlphaBattleAI : public CBattleGameInterface
{
	std::shared_ptr<CBattleCallback> cb;
	std::shared_ptr<Environment> env;
	std::shared_ptr<CBattleGameInterface> fallback;
	std::shared_ptr<BattleModel> model;
	std::array<const CGHeroInstance *, 2> heroes{}; // per battle side, from battleStart
	std::mt19937 rng;
	float temperature = 1.0f;
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

	// Training record (ALPHA_BATTLE_RECORD_DIR)
	std::string recordDir;
	int recordedDecisions = 0;
	void recordEvent(const std::string & json) const;
	std::string aliveArmies(const BattleID & battleID) const;
	int ourMana(const BattleID & battleID) const;

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
	} stats;

	bool previousActionRejected(const CPlayerBattleCallback & battle) const;
	bool injectFault(const std::string & fault);

	void playWithModel(const BattleID & battleID, const CStack * stack);
	void execute(const BattleID & battleID, const CStack * stack, const EncodedBattle & encoded, const ChosenAction & action);
	void useFallback(const BattleID & battleID, const CStack * stack);

public:
	~AlphaBattleAI() override;

	void initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB) override;
	void initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB, AutocombatPreferences autocombatPreferences) override;

	void battleStart(const BattleID & battleID, const CCreatureSet * army1, const CCreatureSet * army2, int3 tile, const CGHeroInstance * hero1, const CGHeroInstance * hero2, BattleSide side, bool replayAllowed) override;
	void battleEnd(const BattleID & battleID, const BattleResult * br, QueryID queryID) override;
	void battleSpellCast(const BattleID & battleID, const BattleSpellCast * sc) override;
	void battleNewRoundFirst(const BattleID & battleID) override;
	void activeStack(const BattleID & battleID, const CStack * stack) override;
	void yourTacticPhase(const BattleID & battleID, int distance) override;
};
