#include "StdInc.h"
#include "BattleSim.h"

#include "server/battles/BattleActionProcessor.h"
#include "server/battles/BattleFlowProcessor.h"
#include "server/battles/IBattleServer.h"

#include "lib/CRandomGenerator.h"
#include "lib/IGameSettings.h"
#include "lib/battle/BattleAction.h"
#include "lib/battle/BattleInfo.h"
#include "lib/callback/GameRandomizer.h"
#include "lib/gameState/CGameState.h"
#include "lib/networkPacks/PacksForClientBattle.h"
#include "lib/networkPacks/SetStackEffect.h"
#include "lib/spells/ISpellMechanics.h"

namespace
{
/// Uniform draws at the middle of their range (damage: the engine's average, as in its estimates), alternately rounded
/// down and up so that the sum of several (a stack's ten damage rolls) has the exact mean; binomials at their mean
class MeanRandom final : public vstd::RNG
{
	bool up = false;

	int64_t middle(int64_t lower, int64_t upper)
	{
		up = !up;
		return lower + (upper - lower + (up ? 1 : 0)) / 2;
	}

public:
	int nextInt(int lower, int upper) override { return static_cast<int>(middle(lower, upper)); }
	int64_t nextInt64(int64_t lower, int64_t upper) override { return middle(lower, upper); }
	double nextDouble(double lower, double upper) override { return (lower + upper) / 2; }
	int nextInt(int upper) override { return nextInt(0, upper); }
	int64_t nextInt64(int64_t upper) override { return nextInt64(0, upper); }
	double nextDouble(double upper) override { return upper / 2; }
	int nextInt() override { return 0; }
	int nextBinomialInt(int coinsCount, double coinChance) override { return static_cast<int>(std::lround(coinsCount * coinChance)); }
};

/// Luck, morale and ability rolls as a script says (see BattleSimulation::scriptDice), with the chance of each
/// (the game's settings for luck and morale, as GameRandomizer; without its bias, which only evens out a series)
class ScriptedRolls final : public IBattleRandomizer
{
	const IGameInfoCallback & game;

	double moraleLuckChance(int value, EGameSettings chances, EGameSettings diceSize) const
	{
		const auto vector = game.getSettings().getVector(chances);
		if(value <= 0 || vector.empty())
			return 0.0;
		return vector[std::min<size_t>(vector.size(), value) - 1] / static_cast<double>(game.getSettings().getInteger(diceSize));
	}

	bool roll(double chance)
	{
		chance = std::clamp(chance, 0.0, 1.0);
		if(chance <= 0.0 || chance >= 1.0)
			return chance >= 1.0; // certain: not a branch
		const bool success = chances.size() < script.size() && script[chances.size()];
		chances.push_back(chance);
		outcomes.push_back(success);
		return success;
	}

public:
	std::vector<bool> script;
	std::vector<double> chances;
	std::vector<bool> outcomes;

	explicit ScriptedRolls(const IGameInfoCallback & game)
		: game(game)
	{
	}
	bool rollGoodMorale(ObjectInstanceID, int value) override { return roll(moraleLuckChance(value, EGameSettings::COMBAT_GOOD_MORALE_CHANCE, EGameSettings::COMBAT_MORALE_DICE_SIZE)); }
	bool rollBadMorale(ObjectInstanceID, int value) override { return roll(moraleLuckChance(value, EGameSettings::COMBAT_BAD_MORALE_CHANCE, EGameSettings::COMBAT_MORALE_DICE_SIZE)); }
	bool rollGoodLuck(ObjectInstanceID, int value) override { return roll(moraleLuckChance(value, EGameSettings::COMBAT_GOOD_LUCK_CHANCE, EGameSettings::COMBAT_LUCK_DICE_SIZE)); }
	bool rollBadLuck(ObjectInstanceID, int value) override { return roll(moraleLuckChance(value, EGameSettings::COMBAT_BAD_LUCK_CHANCE, EGameSettings::COMBAT_LUCK_DICE_SIZE)); }
	bool rollCombatAbility(ObjectInstanceID, int percentageChance) override { return roll(percentageChance / 100.0); }
};
}

/// The game, for the battle rules of a simulation: packs go to the copy's game state, dice to its own
/// generators (or scripted ones), nothing to clients. Also the battle the rules call back (end of battle, automatic
/// actions).
class BattleSimulation::Server final : public IBattleServer, public IBattleOwner
{
	/// For spells: the same packs and dice; adventure-map effects do not happen in a battle
	class SpellEnvironment final : public SpellCastEnvironment
	{
		Server & server;

	public:
		explicit SpellEnvironment(Server & server)
			: server(server)
		{
		}
		void complain(const std::string & problem) override { server.complain(problem); }
		bool describeChanges() const override { return false; } // no battle log texts
		vstd::RNG * getRNG() override { return &server.getRandomGenerator(); }
		void apply(CPackForClient & pack) override { server.sendAndApply(pack); }
		void apply(BattleLogMessage & pack) override { server.sendAndApply(pack); }
		void apply(BattleStackMoved & pack) override { server.sendAndApply(pack); }
		void apply(BattleUnitsChanged & pack) override { server.sendAndApply(pack); }
		void apply(SetStackEffect & pack) override { server.sendAndApply(pack); }
		void apply(StacksInjured & pack) override { server.sendAndApply(pack); }
		void apply(BattleObstaclesChanged & pack) override { server.sendAndApply(pack); }
		void apply(CatapultAttack & pack) override { server.sendAndApply(pack); }
		const CMap * getMap() const override { return &server.sim.game->getMap(); }
		const IGameInfoCallback * getCb() const override { return server.sim.game.get(); }
		void createBoat(const int3 &, BoatId, PlayerColor) override {}
		bool moveHero(ObjectInstanceID, int3, EMovementMode) override { return false; }
		void showGarrisonDialog(ObjectInstanceID, ObjectInstanceID, bool, const MetaString &) override {}
		void genericQuery(Query *, PlayerColor, std::function<void(std::optional<int32_t>)>) override {}
	};

public:
	BattleSimulation & sim;
	CRandomGenerator rng;
	GameRandomizer randomizer;
	MeanRandom mean;
	ScriptedRolls scripted;
	bool scripting = false;
	SpellEnvironment spells;
	std::string complaint;
	bool ended = false;
	std::optional<BattleSide> winner;
	EBattleResult result = EBattleResult::NORMAL;

	Server(BattleSimulation & sim, int seed)
		: sim(sim)
		, rng(seed)
		, randomizer(*sim.game)
		, scripted(*sim.game)
		, spells(*this)
	{
		randomizer.setSeed(seed);
	}

	// IBattleServer
	void sendAndApply(CPackForClient & pack) override
	{
		sim.game->apply(pack); // GameStatePackVisitor, as on the server and the clients
	}
	bool complain(const std::string & problem) override
	{
		complaint = problem;
		return true;
	}
	ServerCallback * spellcastEnvironment() const override { return const_cast<SpellEnvironment *>(&spells); }
	SpellCastEnvironment & obstacleSpellEnvironment() override { return spells; }
	vstd::RNG & getRandomGenerator() override { return scripting ? static_cast<vstd::RNG &>(mean) : rng; }
	IBattleRandomizer & battleRandomizer() override { return scripting ? static_cast<IBattleRandomizer &>(scripted) : randomizer; }
	IGameInfoCallback & gameInfo() override { return *sim.game; }
	void giveResource(PlayerColor, GameResID, int) override {}
	void useChargeBasedSpell(const ObjectInstanceID &, const SpellID &) override {}
	void onBattleStart(const BattleID &) override {}
	void onBattleNextStack(const BattleID &, const CStack &) override {}

	// IBattleOwner: as BattleProcessor, except that the end of the battle only marks the simulation finished
	void updateGateState(const CBattleInfoCallback &) override {} // TODO: sieges
	bool checkBattleStateChanges(const CBattleInfoCallback & battle) override
	{
		if(ended)
			return true;
		if(auto result = battle.battleIsFinished())
		{
			setBattleResult(battle, EBattleResult::NORMAL, *result);
			return true;
		}
		return false;
	}
	bool makeAutomaticBattleAction(const CBattleInfoCallback & battle, const BattleAction & ba) override
	{
		return sim.actions->makeAutomaticBattleAction(battle, ba);
	}
	void setBattleResult(const CBattleInfoCallback &, EBattleResult how, BattleSide victor) override
	{
		ended = true;
		winner = victor;
		result = how;
	}
};

BattleSimulation::BattleSimulation() = default;
BattleSimulation::~BattleSimulation() = default;

std::unique_ptr<BattleSimulation> BattleSimulation::copyOf(const CGameState & source, const BattleID & battleID, int seed)
{
	std::unique_ptr<BattleSimulation> sim(new BattleSimulation());
	sim->game = source.cloneWithBattle(battleID);
	sim->battleNumber = battleID.getNum();
	sim->server = std::make_unique<Server>(*sim, seed);
	sim->actions = std::make_unique<BattleActionProcessor>(sim->server.get(), sim->server.get());
	sim->flow = std::make_unique<BattleFlowProcessor>(sim->server.get(), sim->server.get());
	return sim;
}

std::unique_ptr<BattleSimulation> BattleSimulation::clone(int seed) const
{
	auto copy = copyOf(*game, BattleID(battleNumber), seed);
	copy->server->ended = server->ended;
	copy->server->winner = server->winner;
	copy->server->result = server->result;
	return copy;
}

const CBattleInfoCallback & BattleSimulation::battle() const
{
	return *game->getBattle(BattleID(battleNumber));
}

const CGameState & BattleSimulation::gameState() const
{
	return *game;
}

void BattleSimulation::restoreBattle(const CGameState & source, const BattleID & battleID)
{
	const BattleInfo * from = source.getBattle(battleID);
	if(!from)
		throw std::runtime_error("restoreBattle: no battle " + std::to_string(battleID.getNum()));
	game->restoreBattle(*from);
	server->ended = false;
	server->winner.reset();
	server->result = EBattleResult::NORMAL;
}

void BattleSimulation::scriptDice(std::optional<std::vector<bool>> script)
{
	server->scripting = script.has_value();
	server->scripted.script = script.value_or(std::vector<bool>());
	server->scripted.chances.clear();
	server->scripted.outcomes.clear();
}

const std::vector<double> & BattleSimulation::rollChances() const
{
	return server->scripted.chances;
}

const std::vector<bool> & BattleSimulation::rollOutcomes() const
{
	return server->scripted.outcomes;
}

bool BattleSimulation::applyActionOnly(PlayerColor player, const BattleAction & action)
{
	server->complaint.clear();
	if(server->ended)
		return false;
	const bool accepted = actions->makePlayerBattleAction(battle(), player, action);
	return accepted && server->complaint.empty();
}

bool BattleSimulation::apply(PlayerColor player, const BattleAction & action)
{
	const bool accepted = applyActionOnly(player, action);
	// As BattleProcessor::makePlayerBattleAction: the next unit is activated after any action, unless the battle ended
	if(!server->ended)
		flow->onActionMade(battle(), action);
	return accepted;
}

bool BattleSimulation::finished() const
{
	return server->ended;
}

std::optional<BattleSide> BattleSimulation::winner() const
{
	return server->winner;
}

int BattleSimulation::result() const
{
	return static_cast<int>(server->result);
}

const std::string & BattleSimulation::complaint() const
{
	return server->complaint;
}
