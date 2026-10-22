#include "StdInc.h"
#include "UpLeftAI.h"

#include "lib/CStack.h"
#include "lib/battle/BattleAction.h"
#include "lib/battle/CPlayerBattleCallback.h"
#include "lib/bonuses/BonusEnum.h"
#include "lib/callback/CBattleCallback.h"
#include "lib/callback/CDynLibHandler.h"

namespace
{
constexpr int HANDOVER_ROUND = 10;
}

UpLeftAI::~UpLeftAI()
{
	// Restore previous state of the callback, it may be shared with the adventure AI
	if(cb)
		cb->waitTillRealize = wasWaitingForRealize;
}

void UpLeftAI::initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB)
{
	env = ENV;
	cb = CB;

	// Save before the fallback touches the callback, both restore the same original value
	wasWaitingForRealize = CB->waitTillRealize;
	fallback = CDynLibHandler::getNewBattleAI("StupidAI");
	fallback->initBattleInterface(ENV, CB);
	CB->waitTillRealize = false;
}

void UpLeftAI::initBattleInterface(std::shared_ptr<Environment> ENV, std::shared_ptr<CBattleCallback> CB, AutocombatPreferences autocombatPreferences)
{
	initBattleInterface(ENV, CB);
}

void UpLeftAI::yourTacticPhase(const BattleID & battleID, int distance)
{
	cb->battleMakeTacticAction(battleID, BattleAction::makeEndOFTacticPhase(cb->getBattle(battleID)->battleGetTacticsSide()));
}

void UpLeftAI::battleStart(const BattleID & battleID, const CCreatureSet * army1, const CCreatureSet * army2, int3 tile, const CGHeroInstance * hero1, const CGHeroInstance * hero2, BattleSide side, bool replayAllowed)
{
	fallback->battleStart(battleID, army1, army2, tile, hero1, hero2, side, replayAllowed);
}

void UpLeftAI::activeStack(const BattleID & battleID, const CStack * stack)
{
	auto battle = cb->getBattle(battleID);

	if(battle->battleGetRound() >= HANDOVER_ROUND)
	{
		fallback->activeStack(battleID, stack);
		return;
	}

	if(stack->hasBonusOfType(BonusType::SIEGE_WEAPON))
	{
		cb->battleMakeUnitAction(battleID, BattleAction::makeDefend(stack));
		return;
	}

	// Hit an enemy we are already standing next to
	for(const CStack * enemy : battle->battleGetStacks(CBattleInfoEssentials::ONLY_ENEMY))
	{
		if(!enemy->isInvincible() && CStack::isMeleeAttackPossible(stack, enemy, stack->getPosition()))
		{
			logAi->debug("UpLeftAI: %s attacks %s", stack->nodeName(), enemy->nodeName());
			cb->battleMakeUnitAction(battleID, BattleAction::makeMeleeAttack(stack, enemy->getPosition(), stack->getPosition()));
			return;
		}
	}

	// Otherwise walk to the hex closest to the top-left corner
	const auto score = [](const BattleHex & hex) { return hex.getX() + hex.getY(); };
	BattleHex best = stack->getPosition();
	for(const BattleHex & hex : battle->battleGetAvailableHexes(stack, false))
	{
		if(score(hex) < score(best))
			best = hex;
	}

	logAi->debug("UpLeftAI: %s at (%d %d) goes to (%d %d)", stack->nodeName(), stack->getPosition().getX(), stack->getPosition().getY(), best.getX(), best.getY());
	if(best == stack->getPosition())
		cb->battleMakeUnitAction(battleID, BattleAction::makeDefend(stack));
	else
		cb->battleMakeUnitAction(battleID, BattleAction::makeMove(stack, best));
}
