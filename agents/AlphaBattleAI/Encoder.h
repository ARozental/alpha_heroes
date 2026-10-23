#pragma once

#include "Tensors.h"

#include "lib/battle/BattleSide.h"
#include "lib/constants/EntityIdentifiers.h"

#include <functional>
#include <vector>

VCMI_LIB_NAMESPACE_BEGIN
class BattleAction;
class CGHeroInstance;
class CPlayerBattleCallback;
class CStack;
class Environment;
namespace battle
{
class Unit;
}
VCMI_LIB_NAMESPACE_END

struct ChosenAction;

/// Raw battle state and legality masks for the acting unit, as the model's input tensors
struct EncodedBattle
{
	AlphaSchema::InputTensors tensors;
	std::array<const CStack *, AlphaSchema::MAX_UNITS> units{}; // unit slot -> stack
	std::array<AlphaSchema::TargetKind, AlphaSchema::N_SPELLS> spellTargetKind{}; // first destination per spell
	int activeSlot = -1;
};

/// enemyHero: the enemy side's hero (from battleStart) or nullptr; encoded with what a player sees of it in
/// battle (the client's hero info windows): primary skills, mana and mana limit, not skills or spellbook.
EncodedBattle encodeBattle(const CPlayerBattleCallback & battle, const CStack * active, const Environment & env, PlayerColor player, const CGHeroInstance * enemyHero);

/// Recompute every mask that is derived from more specific ones: a choice is legal only if
/// some complete action through it is legal (melee targets from hexes, spells from targets, types).
void refreshDerivedMasks(EncodedBattle & e);

/// Remove one complete action from the masks (it was rejected), keeping the masks consistent
void excludeAction(EncodedBattle & e, const ChosenAction & action);

/// True if the complete action is legal under the masks (mirrors battle_model/distributions.py is_legal)
bool isLegal(const EncodedBattle & e, const ChosenAction & action);

/// Every legal full action, from the masks alone, in the order of battle_model/distributions.py enumerate_actions
std::vector<ChosenAction> legalActions(const EncodedBattle & e);

/// The engine's action for a model action of unit `active` of side `side`. unit(i) is the unit an action field
/// refers to (a unit slot of an encoding, or a unit id); throws for an unknown unit or action type.
BattleAction engineAction(BattleSide side, const CStack * active, const ChosenAction & a, const std::function<const battle::Unit *(int)> & unit);
