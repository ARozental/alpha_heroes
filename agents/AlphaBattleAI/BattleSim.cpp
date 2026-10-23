#include "StdInc.h"
#include "BattleSim.h"

#include "server/battles/BattleActionProcessor.h"
#include "server/battles/BattleFlowProcessor.h"
#include "server/battles/IBattleServer.h"

#include "lib/CRandomGenerator.h"
#include "lib/battle/BattleAction.h"
#include "lib/battle/BattleInfo.h"
#include "lib/callback/GameRandomizer.h"
#include "lib/gameState/CGameState.h"
#include "lib/networkPacks/PacksForClientBattle.h"
#include "lib/networkPacks/SetStackEffect.h"
#include "lib/spells/ISpellMechanics.h"

/// The game, for the battle rules of a simulation: packs go to the copy's game state, dice to its own
/// generators, nothing to clients. Also the battle the rules call back (end of battle, automatic actions).
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
		vstd::RNG * getRNG() override { return &server.rng; }
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
	SpellEnvironment spells;
	std::string complaint;
	bool ended = false;
	std::optional<BattleSide> winner;
	EBattleResult result = EBattleResult::NORMAL;

	Server(BattleSimulation & sim, int seed)
		: sim(sim)
		, rng(seed)
		, randomizer(*sim.game)
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
	vstd::RNG & getRandomGenerator() override { return rng; }
	GameRandomizer & battleRandomizer() override { return randomizer; }
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
