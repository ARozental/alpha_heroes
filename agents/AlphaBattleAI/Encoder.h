#pragma once

#include "Tensors.h"

#include "lib/constants/EntityIdentifiers.h"

VCMI_LIB_NAMESPACE_BEGIN
class CPlayerBattleCallback;
class CStack;
class Environment;
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

EncodedBattle encodeBattle(const CPlayerBattleCallback & battle, const CStack * active, const Environment & env, PlayerColor player);

/// Recompute every mask that is derived from more specific ones: a choice is legal only if
/// some complete action through it is legal (melee targets from hexes, spells from targets, types).
void refreshDerivedMasks(EncodedBattle & e);

/// Remove one complete action from the masks (it was rejected), keeping the masks consistent
void excludeAction(EncodedBattle & e, const ChosenAction & action);

/// True if the complete action is legal under the masks (mirrors battle_model/distributions.py is_legal)
bool isLegal(const EncodedBattle & e, const ChosenAction & action);
