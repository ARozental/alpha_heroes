#pragma once

#include "Reward.h"
#include "Tensors.h"

#include "lib/battle/BattleSide.h"
#include "lib/constants/EntityIdentifiers.h"

#include <functional>
#include <optional>
#include <vector>

VCMI_LIB_NAMESPACE_BEGIN
class BattleAction;
class CCreature;
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
/// start: the armies and mana at the start of the battle (g_start_*). The attack values (a_*) need the game state
/// around the battle: encodeAttackValues (AttackValues.h) adds them.
EncodedBattle encodeBattle(const CPlayerBattleCallback & battle, const CStack * active, const Environment & env, PlayerColor player, const CGHeroInstance * enemyHero, const AlphaReward::BattleEnd & start);

/// Recompute every mask that is derived from more specific ones: a choice is legal only if
/// some complete action through it is legal (melee targets from hexes, spells from targets, types).
void refreshDerivedMasks(EncodedBattle & e);

/// Remove one complete action from the masks (it was rejected), keeping the masks consistent
void excludeAction(EncodedBattle & e, const ChosenAction & action);

/// True if the complete action is legal under the masks (mirrors battle_model/distributions.py is_legal)
bool isLegal(const EncodedBattle & e, const ChosenAction & action);

/// A creature's value in gold (reward.py prices: gold and resources), as the reward counts it
double creatureValue(const CCreature * creature);

/// A hero's level and its way to the next from its experience (12.5: halfway to 13), as reward.py's
/// hero_level_value takes it; 0 without a hero
double heroLevel(const CGHeroInstance * hero);

/// The market price in gold of the artifacts a winner takes from a defeated hero (BattleResultProcessor: its removable
/// worn artifacts and its backpack, not the Grail); 0 without a hero
double heroArtifacts(const CGHeroInstance * hero);

/// True for a unit the side keeps after the battle if it is alive then: its army's stacks and permanently summoned
/// units (Pit Lords' demons: the engine adds them to the army, BattleResultProcessor); not summoned units and clones,
/// which vanish, nor war machines and arrow towers
bool keptAfterBattle(const CStack * unit);

/// The creatures of a unit its side would keep if the battle ended now: its count minus the ones only raised for the
/// battle (a temporary Resurrection), as the engine's casualties count them (CUnitState::getKilled); 0 for units
/// not kept after the battle. The reward counts what remains, never what was killed.
int remainingAfterBattle(const CStack * unit);

/// Every legal full action, from the masks alone, in the order of battle_model/distributions.py enumerate_actions
std::vector<ChosenAction> legalActions(const EncodedBattle & e);

/// The model action of an engine action of the acting unit (in the slots of e), or nothing for actions the
/// contract does not describe; for recording what another AI did (ALPHA_BATTLE_DELEGATE)
std::optional<ChosenAction> chosenAction(const EncodedBattle & e, const BattleAction & action);

/// The engine's action for a model action of unit `active` of side `side`. unit(i) is the unit an action field
/// refers to (a unit slot of an encoding, or a unit id); throws for an unknown unit or action type.
BattleAction engineAction(BattleSide side, const CStack * active, const ChosenAction & a, const std::function<const battle::Unit *(int)> & unit);
