// Runs the C++ search (GumbelSearch.h) on a toy game and the C++ reward (Reward.h) on given battle ends, for
// battle_model/tests/test_search.py, which checks that the readable Python copies (battle_model/search.py,
// battle_model/reward.py) give the same results. Reads commands from stdin, one per line, and prints one JSON line each:
//   search SEED SIMULATIONS MAX_CONSIDERED GUMBEL_SCALE
//   reward ARMY_START0 ARMY_START1 ARMY_END0 ARMY_END1 MANA_START0 MANA_START1 POWER0 POWER1 MANA_END0 MANA_END1 WINNER RESULT ROUNDS SURRENDER_COST0 SURRENDER_COST1 LEVEL0 LEVEL1 ARTIFACTS0 ARTIFACTS1
//   other_side_value VALUE
#include "GumbelSearch.h"
#include "Reward.h"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

using GumbelSearch::SplitMix64;

namespace
{
uint64_t mix(uint64_t x)
{
	return SplitMix64(x).next();
}

/// A stochastic general-sum race; the same game as ToyGame in battle_model/tests/test_search.py. Each player moves
/// its counter forward by its action (0..4) plus a die (0..2). A 2 lets the same player move again (like good
/// morale). Which actions are legal depends on the position, so on the dice. The first to 12 wins; after 10 moves
/// nobody does. Some positions are leaves the search does not go beyond. The rewards are not zero-sum. The
/// "network" is a hash of the position.
struct ToyGame
{
	struct State
	{
		std::array<int, 2> position{};
		int turn = 0;
		int mover = 0;
		SplitMix64 dice{0};
	};

	static uint64_t hash(const State & s)
	{
		return mix(static_cast<uint64_t>(s.position[0]) * 1000003 + static_cast<uint64_t>(s.position[1]) * 1009 + static_cast<uint64_t>(s.turn) * 17 + static_cast<uint64_t>(s.mover));
	}

	State root(uint64_t seed)
	{
		State s;
		s.position = {1, 0};
		s.dice = SplitMix64(seed);
		return s;
	}

	bool terminal(const State & s) const { return s.position[0] >= 12 || s.position[1] >= 12 || s.turn >= 10; }

	std::array<double, 2> rewards(State & s)
	{
		if(s.position[0] < 12 && s.position[1] < 12)
			return {0.04 * s.position[0] - 0.3, 0.04 * s.position[1] - 0.3};
		const int winner = s.position[0] >= 12 ? 0 : 1;
		std::array<double, 2> r{};
		r[winner] = 1.0 - 0.05 * s.turn;
		r[1 - winner] = -0.6 + 0.03 * s.position[1 - winner];
		return r;
	}

	int mover(const State & s) const { return s.mover; }
	int64_t key(const State & s) const { return s.mover; } // like the acting unit in battles: the same node sees several states

	std::vector<int64_t> legal(State & s)
	{
		std::vector<int64_t> actions;
		for(int a = 0; a < 5; ++a)
			if(mix(hash(s) + static_cast<uint64_t>(a)) % 3 != 0)
				actions.push_back(a);
		if(actions.empty())
			actions.push_back(0);
		return actions;
	}

	GumbelSearch::Evaluation evaluate(State & s)
	{
		GumbelSearch::Evaluation e;
		const uint64_t h = hash(s);
		for(int p = 0; p < 2; ++p)
			e.values[p] = static_cast<double>(mix(h + 7919 * static_cast<uint64_t>(p + 1)) % 2001) / 1000.0 - 1.0;
		if(s.turn >= 2 && h % 4 == 0) // a leaf: the search does not go on from here (like the round cap in battles)
			return e;
		e.actions = legal(s);
		for(int64_t a : e.actions)
			e.logits.push_back(static_cast<double>(mix(h ^ (static_cast<uint64_t>(a + 1) * 256)) % 4001) / 1000.0 - 2.0);
		return e;
	}

	void apply(State & s, int64_t action)
	{
		const int die = static_cast<int>(s.dice.next() % 3);
		s.position[s.mover] += static_cast<int>(action) + die;
		s.turn += 1;
		if(die != 2)
			s.mover = 1 - s.mover;
	}
};

std::string number(double x)
{
	if(std::isnan(x))
		return "null";
	char text[40];
	std::snprintf(text, sizeof(text), "%.17g", x);
	return text;
}

template<typename T, typename F>
std::string list(const std::vector<T> & values, F format)
{
	std::string json = "[";
	for(size_t i = 0; i < values.size(); ++i)
		json += (i ? ", " : "") + format(values[i]);
	return json + "]";
}
}

int main()
{
	for(std::string line; std::getline(std::cin, line);)
	{
		std::istringstream in(line);
		std::string command;
		in >> command;
		if(command == "search")
		{
			uint64_t seed;
			GumbelSearch::Config config;
			in >> seed >> config.simulations >> config.maxConsidered >> config.gumbelScale;
			ToyGame game;
			GumbelSearch::Search<ToyGame> search(game, config);
			const auto r = search.run(seed);
			auto integer = [](auto x) { return std::to_string(x); };
			std::cout << "{\"action\": " << r.action << ", \"actions\": " << list(r.actions, integer) << ", \"visits\": " << list(r.visits, integer) << ", \"q\": " << list(r.q, number) << ", \"policy\": " << list(r.policy, number) << ", \"value\": " << number(r.value) << ", \"nodes\": " << r.nodes << ", \"simulations\": " << r.simulations << "}" << std::endl;
		}
		else if(command == "reward")
		{
			AlphaReward::BattleEnd e;
			in >> e.armyStart[0] >> e.armyStart[1] >> e.armyEnd[0] >> e.armyEnd[1] >> e.manaStart[0] >> e.manaStart[1] >> e.power[0] >> e.power[1] >> e.manaEnd[0] >> e.manaEnd[1] >> e.winner >> e.result >> e.rounds >> e.surrenderCost[0] >> e.surrenderCost[1] >> e.heroLevel[0] >> e.heroLevel[1] >> e.artifacts[0] >> e.artifacts[1];
			std::cout << "{\"rewards\": [" << number(AlphaReward::reward(e, 0)) << ", " << number(AlphaReward::reward(e, 1)) << "], \"scaled\": [" << number(AlphaReward::toScale(AlphaReward::reward(e, 0))) << ", " << number(AlphaReward::toScale(AlphaReward::reward(e, 1))) << "]}" << std::endl;
		}
		else if(command == "other_side_value")
		{
			double value;
			in >> value;
			std::cout << "{\"value\": " << number(AlphaReward::otherSideValue(value)) << "}" << std::endl;
		}
		else if(!command.empty())
		{
			std::cerr << "unknown command: " << command << std::endl;
			return 1;
		}
	}
	return 0;
}
