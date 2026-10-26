#include "StdInc.h"
#include "History.h"

#include "lib/CStack.h"
#include "lib/battle/BattleAction.h"
#include "lib/battle/CBattleInfoCallback.h"
#include "lib/networkPacks/PacksForClientBattle.h"

using namespace AlphaSchema;

namespace
{
/// The model's action type of an engine action, -1 for one it has none for (a turn lost to bad morale, ...)
int actionType(const BattleAction & action, bool tactics)
{
	switch(action.actionType)
	{
		case EActionType::WAIT: return static_cast<int>(ActionType::WAIT);
		case EActionType::DEFEND: return static_cast<int>(ActionType::DEFEND);
		case EActionType::WALK: return static_cast<int>(tactics ? ActionType::TACTICS_MOVE : ActionType::MOVE);
		case EActionType::WALK_AND_ATTACK: return static_cast<int>(ActionType::MELEE);
		case EActionType::SHOOT: return static_cast<int>(ActionType::SHOOT);
		case EActionType::HERO_SPELL: return static_cast<int>(ActionType::HERO_SPELL);
		case EActionType::MONSTER_SPELL: return static_cast<int>(ActionType::CREATURE_SPELL);
		case EActionType::RETREAT: return static_cast<int>(ActionType::RETREAT);
		case EActionType::SURRENDER: return static_cast<int>(ActionType::SURRENDER);
		case EActionType::CATAPULT: return static_cast<int>(ActionType::CATAPULT);
		case EActionType::STACK_HEAL: return static_cast<int>(ActionType::HEAL);
		case EActionType::END_TACTIC_PHASE: return static_cast<int>(ActionType::TACTICS_END);
		default: return -1;
	}
}
}

void BattleHistory::reset()
{
	*this = BattleHistory();
}

void BattleHistory::seeUnits(const CBattleInfoCallback & battle)
{
	for(const auto * stack : battle.battleGetStacksIf([](const CStack *) { return true; }))
	{
		auto & unit = units[stack->unitId()];
		if(unit.startCount < 0)
			unit.startCount = stack->getCount();
	}
}

void BattleHistory::actionStarted(const CBattleInfoCallback & battle, const BattleAction & action)
{
	seeUnits(battle);
	current = Event();
	current.round = battle.battleGetRound();
	current.side = static_cast<int>(action.side);
	current.type = actionType(action, battle.battleGetTacticDist() > 0);
	if(action.actionType != EActionType::HERO_SPELL && action.stackNumber >= 0)
		current.actor = static_cast<uint32_t>(action.stackNumber);
	if(action.spell.hasValue())
		current.spell = action.spell.getNum();

	const auto target = action.getTarget(&battle);
	const size_t index = action.actionType == EActionType::WALK_AND_ATTACK ? 1 : 0; // a melee attack: where from, then whom
	if(target.size() > index)
	{
		const auto & destination = target[index];
		current.hex = destination.hexValue.isValid() ? destination.hexValue.toInt() : -1;
		const auto * unit = destination.unitValue ? destination.unitValue : (destination.hexValue.isValid() ? battle.battleGetUnitByPos(destination.hexValue, true) : nullptr);
		if(unit && action.actionType != EActionType::WALK)
			current.target = unit->unitId();
	}
	if(current.actor != HERO)
	{
		auto & unit = units[current.actor];
		unit.lastRound = current.round;
		if(current.target >= 0)
			unit.lastTarget = current.target;
	}
	inAction = true;
}

void BattleHistory::stacksAttacked(const CBattleInfoCallback & battle, const std::vector<BattleStackAttacked> & attacked)
{
	for(const auto & hit : attacked)
	{
		const auto * victim = battle.battleGetUnitByID(hit.stackAttacked);
		if(!victim)
			continue;
		units[hit.stackAttacked].taken += hit.damageAmount;
		if(battle.battleGetUnitByID(hit.attackerID))
		{
			units[hit.attackerID].dealt += hit.damageAmount;
			units[hit.attackerID].kills += hit.killedAmount;
		}
		if(!inAction)
			continue;
		if(static_cast<int>(victim->unitSide()) != current.side)
		{
			current.dealt += hit.damageAmount;
			current.kills += hit.killedAmount;
		}
		else
		{
			current.taken += hit.damageAmount;
			current.lost += hit.killedAmount;
		}
	}
}

void BattleHistory::spellCast(const CBattleInfoCallback & battle, const BattleSpellCast & cast)
{
	const int spell = cast.spellID.getNum();
	if(cast.castByHero && cast.side != BattleSide::NONE)
	{
		const int side = static_cast<int>(cast.side);
		if(spell >= 0 && spell < N_SPELLS)
			++seenSpells[side][spell];
		++casts[side];
		lastCastRound[side] = battle.battleGetRound();
	}
	if(inAction && current.spell < 0)
		current.spell = spell;
}

void BattleHistory::actionFinished()
{
	if(!inAction)
		return;
	events.push_front(current);
	while(events.size() > static_cast<size_t>(N_EVENTS))
		events.pop_back();
	inAction = false;
}
