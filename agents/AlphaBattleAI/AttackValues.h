#pragma once

#include "lib/constants/EntityIdentifiers.h"

VCMI_LIB_NAMESPACE_BEGIN
class BattleID;
class CGameState;
VCMI_LIB_NAMESPACE_END

struct EncodedBattle;

/// The cost of encodeAttackValues, summed over calls
struct AttackValueStats
{
	int calls = 0;
	int attacks = 0; // legal attacks valued
	int played = 0; // of them played in the simulation (the others share an outcome with one played)
	int runs = 0; // simulation runs: dice branches of the attacks played
	double ms = 0;
};

/// What each legal attack of the acting unit would do, as the engine's own rules play it on a copy of the battle
/// (BattleSim): a_melee_value [U, H] for attacking unit u from hex h, a_shoot_value [U] for shooting it. Everything
/// the rules do counts: retaliation, multi-hex and area attacks, Fire Shield, life drain (also resurrecting),
/// death stare, death blow, jousting, rebirth, spells cast after an attack, obstacles on the way, hero skills and
/// artifacts. Dice at their expectation: damage rolls at their mean, and every luck and ability roll both ways,
/// weighted by its chance (the most likely branches first, at most MAX_BRANCHES of them).
///
/// The value is the change of army value in gold as the reward counts it (each unit's health it would keep after the
/// battle, remainingAfterBattle, at its creature's value per hit point), for the attacker minus for the defender, as
/// s_unit_value. game: the game state that battleID of `e` was encoded from (the AI's own, or a search's copy).
void encodeAttackValues(EncodedBattle & e, const CGameState & game, const BattleID & battleID, PlayerColor player, AttackValueStats & stats);
