#pragma once

#include "lib/constants/EntityIdentifiers.h"

#include <memory>

VCMI_LIB_NAMESPACE_BEGIN
class BattleID;
class CGameState;
VCMI_LIB_NAMESPACE_END

struct EncodedBattle;
class BattleSimulation;

/// The copy of the game state that encodeAttackValues plays the attacks on, kept for a whole battle: a new copy every
/// decision (the whole game state through savegame serialization) was ~40% of a battle client's CPU. Every attack
/// starts from the battle as it is now (BattleSimulation::restoreBattle) with dice as a new copy has them, so only the
/// game state around the battle is the one of the battle's first decision; an attack reads nothing there that changes
/// during a battle (the heroes' skills and artifacts, the armies; their mana changes, and no attack uses it).
/// ALPHA_BATTLE_ATTACK_CHECK=1 values every decision on a new copy too and logs any difference (check, mismatches); =2
/// compares two new copies, as before the kept one. Measured 2026-10-29 (480 generated battles each): 2 decisions of
/// 36,879 differed by 1-15 gold with the kept copy, 3 of 36,530 between two new copies alike: the engine hits the units
/// of an area attack in the order of their addresses (BattleActionProcessor::makeAttack, a std::set of pointers), and the
/// alternate rounding of BattleSimulation's mean dice then depends on that order. Not the kept copy's.
struct AttackValueCache
{
	std::unique_ptr<BattleSimulation> sim;
	const CGameState * game = nullptr; // what sim was copied from
	int battle = -1;
	bool check = false;
	bool freshCheck = false;
	int checked = 0, mismatches = 0;

	AttackValueCache();
	~AttackValueCache();
	void clear(); // at the end of a battle: the next one is copied anew
};

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
/// cache: the AI's copy for the battle (AttackValueCache), or none: a new copy for this decision (a search's states).
void encodeAttackValues(EncodedBattle & e, const CGameState & game, const BattleID & battleID, PlayerColor player, AttackValueStats & stats, AttackValueCache * cache = nullptr);
