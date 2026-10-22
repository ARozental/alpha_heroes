#pragma once

#include "Tensors.h"

#include "lib/constants/EntityIdentifiers.h"

VCMI_LIB_NAMESPACE_BEGIN
class CPlayerBattleCallback;
class CStack;
class Environment;
VCMI_LIB_NAMESPACE_END

/// Raw battle state and legality masks for the acting unit, as the model's input tensors
struct EncodedBattle
{
	AlphaSchema::InputTensors tensors;
	std::array<const CStack *, AlphaSchema::MAX_UNITS> units{}; // unit slot -> stack
	int activeSlot = -1;
};

EncodedBattle encodeBattle(const CPlayerBattleCallback & battle, const CStack * active, const Environment & env, PlayerColor player);
