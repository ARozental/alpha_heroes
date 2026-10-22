#pragma once

#include "lib/callback/CBattleGameInterface.h"

/// Deliberately dumb battle AI: every unit walks to the reachable hex closest to the
/// top-left corner of the battlefield, attacking any enemy it ends up standing next to.
/// Armies can get stuck out of each other's reach, so from round HANDOVER_ROUND on
/// StupidAI plays instead so the battle cannot run forever.
class UpLeftAI : public CBattleGameInterface
{
	std::shared_ptr<CBattleCallback> cb;
	std::shared_ptr<Environment> env;
	std::shared_ptr<CBattleGameInterface> fallback;
	bool wasWaitingForRealize = false;

public:
	~UpLeftAI() override;

	void initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB) override;
	void initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB, AutocombatPreferences autocombatPreferences) override;

	void battleStart(const BattleID & battleID, const CCreatureSet * army1, const CCreatureSet * army2, int3 tile, const CGHeroInstance * hero1, const CGHeroInstance * hero2, BattleSide side, bool replayAllowed) override;
	void activeStack(const BattleID & battleID, const CStack * stack) override;
	void yourTacticPhase(const BattleID & battleID, int distance) override;
};
