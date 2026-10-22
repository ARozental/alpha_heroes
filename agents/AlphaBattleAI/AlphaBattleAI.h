#pragma once

#include "lib/callback/CBattleGameInterface.h"

#include <random>

class BattleModel;
struct ChosenAction;
struct EncodedBattle;

/// Battle AI that plays with an ONNX model implementing battle_model's contract.
///
/// Configured through environment variables:
///   ALPHA_BATTLE_MODEL        path to the .onnx file (required; without it StupidAI plays)
///   ALPHA_BATTLE_TEMPERATURE  sampling temperature, 0 = always the most likely action (default 1)
///   ALPHA_BATTLE_SEED         random seed (default: random)
///   ALPHA_BATTLE_DUMP_DIR     write inputs/outputs/action of decisions there (battle_model/check_dumps.py)
///   ALPHA_BATTLE_DUMP_LIMIT   decisions to dump per process (default 50)
///
/// War machines and the tactics phase are scripted (StupidAI), as is everything from round
/// ROUND_CAP on, so a battle cannot go on forever. Any error falls back to StupidAI for that decision.
class AlphaBattleAI : public CBattleGameInterface
{
	std::shared_ptr<CBattleCallback> cb;
	std::shared_ptr<Environment> env;
	std::shared_ptr<CBattleGameInterface> fallback;
	std::shared_ptr<BattleModel> model;
	std::mt19937 rng;
	float temperature = 1.0f;
	bool wasWaitingForRealize = false;

	struct Stats
	{
		int decisions = 0;
		int fallbacks = 0;
		int errors = 0;
		std::array<int, 16> types{};
		std::array<int, 4> spellTargets{}; // no target, unit, hex, two targets
		double modelMs = 0;
		bool siege = false;
	} stats;

	void playWithModel(const BattleID & battleID, const CStack * stack);
	void execute(const BattleID & battleID, const CStack * stack, const EncodedBattle & encoded, const ChosenAction & action);
	void useFallback(const BattleID & battleID, const CStack * stack);

public:
	~AlphaBattleAI() override;

	void initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB) override;
	void initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB, AutocombatPreferences autocombatPreferences) override;

	void battleStart(const BattleID & battleID, const CCreatureSet * army1, const CCreatureSet * army2, int3 tile, const CGHeroInstance * hero1, const CGHeroInstance * hero2, BattleSide side, bool replayAllowed) override;
	void battleEnd(const BattleID & battleID, const BattleResult * br, QueryID queryID) override;
	void activeStack(const BattleID & battleID, const CStack * stack) override;
	void yourTacticPhase(const BattleID & battleID, int distance) override;
};
