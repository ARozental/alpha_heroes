#pragma once

#include "lib/callback/CBattleGameInterface.h"

/// Deliberately dumb battle AI: every unit walks to the reachable hex closest to the
/// top-left corner of the battlefield, attacking any enemy it ends up standing next to.
class UpLeftAI : public CBattleGameInterface
{
	std::shared_ptr<CBattleCallback> cb;
	std::shared_ptr<Environment> env;
	bool wasWaitingForRealize = false;

public:
	~UpLeftAI() override;

	void initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB) override;
	void initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB, AutocombatPreferences autocombatPreferences) override;

	void activeStack(const BattleID & battleID, const CStack * stack) override;
	void yourTacticPhase(const BattleID & battleID, int distance) override;
};
