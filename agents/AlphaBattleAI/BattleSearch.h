#pragma once

#include "BattleSim.h"
#include "GumbelSearch.h"
#include "Reward.h"

#include <memory>

VCMI_LIB_NAMESPACE_BEGIN
class CBattleInfoCallback;
class CCreatureSet;
class CStack;
class Environment;
VCMI_LIB_NAMESPACE_END

class BattleModel;
struct ChosenAction;
struct EncodedBattle;

/// Value of an army in gold (reward.py), as data.py army_value
double armyValue(const CCreatureSet & army);
/// Value of a side's army units alive in a battle; summoned units and clones vanish after the battle and are excluded
double aliveArmyValue(const CBattleInfoCallback & battle, BattleSide side);

/// A battle as a game for GumbelSearch: a state is a BattleSimulation (a copy of the battle with its own dice),
/// a player is a battle side, the network gives the priors (its policy for the acting unit) and the acting side's
/// value (V, one network call per leaf; the other side's is estimated from it, Reward.h otherSideValue), the
/// terminal rewards are training's (Reward.h, exact for both sides).
///
/// Not decisions, so not searched: war machines defend (the real AI lets StupidAI play them), and from round
/// roundCap on (StupidAI plays there) a state is a leaf the network evaluates.
class BattleSearchEnv
{
public:
	struct State
	{
		std::unique_ptr<BattleSimulation> sim;
		std::array<double, 2> surrenderCost{};
	};

	/// root: the battle at the decision (its dice are replaced in every simulation); start: the start of the
	/// battle as the reward counts it (army values and mana; armyEnd, manaEnd, winner and result are unused)
	BattleSearchEnv(const BattleSimulation & root, const BattleModel & model, const Environment & env, const AlphaReward::BattleEnd & start, int roundCap);

	State root(uint64_t seed);
	bool terminal(const State & s) const;
	std::array<double, 2> rewards(State & s);
	int mover(const State & s) const;
	int64_t key(const State & s) const;
	GumbelSearch::Evaluation evaluate(State & s);
	std::vector<int64_t> legal(State & s);
	void apply(State & s, int64_t action);

	/// An action as the search's id: unit fields are unit ids, which, unlike the slots of an encoding, stay the
	/// same in every copy of the battle
	static int64_t actionId(const EncodedBattle & e, const ChosenAction & a);

	// Where the time goes
	int evaluations = 0; // network calls
	int encodings = 0;
	double copyMs = 0, encodeMs = 0, modelMs = 0, applyMs = 0;

private:
	const BattleSimulation & rootSim;
	const BattleModel & model;
	const Environment & env;
	AlphaReward::BattleEnd start;
	int roundCap;

	const CStack * active(const State & s) const;
	EncodedBattle encode(const State & s);
	void playNonDecisions(State & s);
};
