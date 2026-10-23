#pragma once

#include "lib/battle/BattleSide.h"
#include "lib/constants/EntityIdentifiers.h"

#include <memory>
#include <optional>
#include <string>

VCMI_LIB_NAMESPACE_BEGIN
class BattleAction;
class BattleID;
class CBattleInfoCallback;
class CGameState;
VCMI_LIB_NAMESPACE_END

class BattleActionProcessor;
class BattleFlowProcessor;

/// A private copy of a battle, and of the game state around it, that VCMI's own battle rules (the server's
/// BattleActionProcessor and BattleFlowProcessor) run on, without clients or network. For search: copy the
/// battle, try an action, look at the result, start again from another copy. See docs/battle_search.md.
///
/// The copy is a deep copy through VCMI's savegame serialization (the game state) and the serialization that
/// sends a battle to the clients (the battle itself). Packs are applied to it with GameStatePackVisitor, as on
/// the server and the clients. Dice (damage, luck, morale, abilities) come from the copy's own random generators.
class BattleSimulation
{
public:
	/// A copy of battle battleID of game (for an AI: its own view of the game); its dice are seeded with seed
	static std::unique_ptr<BattleSimulation> copyOf(const CGameState & game, const BattleID & battleID, int seed);
	/// A copy of this simulation, with new dice
	std::unique_ptr<BattleSimulation> clone(int seed) const;
	~BattleSimulation();

	const CBattleInfoCallback & battle() const;

	/// Plays an action the way the server does when player sends it: the rules apply it (or reject it: false,
	/// see complaint()), then, unless the battle ended, the next unit is activated (morale, new round, ...)
	bool apply(PlayerColor player, const BattleAction & action);
	/// apply() without activating the next unit, to compare with the real game right after an action
	bool applyActionOnly(PlayerColor player, const BattleAction & action);

	bool finished() const;
	std::optional<BattleSide> winner() const;
	const std::string & complaint() const;

private:
	class Server;
	BattleSimulation();

	std::unique_ptr<CGameState> game;
	int battleNumber = -1;
	std::unique_ptr<Server> server;
	std::unique_ptr<BattleActionProcessor> actions;
	std::unique_ptr<BattleFlowProcessor> flow;
};
