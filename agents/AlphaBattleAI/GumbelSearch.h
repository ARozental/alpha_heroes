#pragma once

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

/// Gumbel AlphaZero search ("Policy improvement by planning with Gumbel", Danihelka et al., ICLR 2022; the
/// formulas of DeepMind's mctx gumbel_muzero_policy), for two players who each maximise their own reward
/// (general-sum), in a game with dice.
///
/// - Root: Gumbel-Top-k picks the actions to consider, Sequential Halving spends the simulations on them,
///   and the action played is the best of the last ones left. With few simulations this still improves on
///   the network's policy, where visit-count MCTS needs many more.
/// - Below the root: the action maximising pi'(a) - N(a) / (1 + sum N), pi' = softmax(logits + sigma(completed Q)).
/// - Training target (later): the improved policy softmax(logits + sigma(completed Q)) at the root.
/// - Each player's Q is its own: a node's mover chooses by its own expected reward; a leaf's evaluation gives
///   one value per player.
/// - Open loop: a simulation starts from a fresh copy of the root with new dice and replays the tree's actions.
///   A node's statistics are over the states its action sequence reached. A child is split when the next state
///   has another key (in battles: another acting unit, e.g. after good morale or a death), since its actions and
///   priors are another unit's. At each node only the actions legal in the current state are considered.
///
/// battle_model/search.py is a slow, readable copy of this file, only for review; battle_model/tests/test_search.py
/// checks on a toy game (tools/SearchSelfTest.cpp) that both make the same decisions. Change one, change both.
namespace GumbelSearch
{
/// splitmix64: the search's noise and seeds, and the toy game's dice; the same numbers in search.py
struct SplitMix64
{
	uint64_t state;
	explicit SplitMix64(uint64_t seed)
		: state(seed)
	{
	}
	uint64_t next()
	{
		uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
		z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
		return z ^ (z >> 31);
	}
	/// Uniform in (0, 1), never 0 or 1
	double uniform() { return (static_cast<double>(next() >> 11) + 0.5) * 0x1.0p-53; }
	double gumbel() { return -std::log(-std::log(uniform())); }
};

struct Config
{
	int simulations = 32;
	int maxConsidered = 16; // root actions Sequential Halving starts with
	double gumbelScale = 1.0; // 0: no noise, the best action by the search (greedy play)
	// sigma(q) = (maxvisitInit + max N) * valueScale * q, q rescaled to 0..1 at each node (mctx defaults)
	double valueScale = 0.1;
	double maxvisitInit = 50.0;
};

struct Evaluation
{
	std::vector<int64_t> actions; // legal actions; empty: the search does not go deeper here (a leaf)
	std::vector<double> logits; // the network's log-probabilities of the actions (any offset)
	std::array<double, 2> values{}; // each player's expected final reward, from its own view
};

struct Result
{
	int64_t action = -1; // to play
	std::vector<int64_t> actions; // the root's legal actions
	std::vector<int> visits;
	std::vector<double> q; // the mover's Q per root action, NaN if not visited
	std::vector<double> policy; // improved policy: the training target
	double value = 0; // the network's value of the root for the mover
	int nodes = 0;
	int simulations = 0;
};

/// mctx get_sequence_of_considered_visits: for each simulation, the visit count the root action it explores must have
inline std::vector<int> consideredVisitSequence(int maxConsidered, int simulations)
{
	std::vector<int> sequence;
	if(maxConsidered <= 1)
	{
		for(int i = 0; i < simulations; ++i)
			sequence.push_back(i);
		return sequence;
	}
	int log2max = 0; // ceil(log2(maxConsidered))
	while((1 << log2max) < maxConsidered)
		++log2max;
	std::vector<int> visits(maxConsidered, 0);
	int considered = maxConsidered;
	while(static_cast<int>(sequence.size()) < simulations)
	{
		const int extra = std::max(1, static_cast<int>(static_cast<double>(simulations) / (log2max * considered)));
		for(int e = 0; e < extra; ++e)
		{
			sequence.insert(sequence.end(), visits.begin(), visits.begin() + considered);
			for(int i = 0; i < considered; ++i)
				visits[i] += 1;
		}
		considered = std::max(2, considered / 2);
	}
	sequence.resize(simulations);
	return sequence;
}

inline std::vector<double> softmax(const std::vector<double> & logits)
{
	double top = -std::numeric_limits<double>::infinity();
	for(double l : logits)
		top = std::max(top, l);
	std::vector<double> p(logits.size());
	double sum = 0;
	for(size_t i = 0; i < logits.size(); ++i)
	{
		p[i] = std::exp(logits[i] - top);
		sum += p[i];
	}
	for(double & x : p)
		x /= sum;
	return p;
}

/// Index of the first largest entry
inline int argmax(const std::vector<double> & x)
{
	int best = 0;
	for(int i = 1; i < static_cast<int>(x.size()); ++i)
		if(x[i] > x[best])
			best = i;
	return best;
}

struct Node
{
	int mover = 0;
	std::vector<int64_t> actions;
	std::vector<double> logits;
	std::array<double, 2> value{}; // the network's, when the node was made
	std::vector<int> visits; // per action
	std::vector<std::array<double, 2>> totals; // per action: sum of the values backed up through it, per player
	std::vector<std::vector<std::pair<int64_t, int>>> children; // per action: (key of the next state, node index)

	Node(int mover, const Evaluation & e)
		: mover(mover)
		, actions(e.actions)
		, logits(e.logits)
		, value(e.values)
		, visits(e.actions.size(), 0)
		, totals(e.actions.size(), {0.0, 0.0})
		, children(e.actions.size())
	{
	}

	double q(int i) const { return totals[i][mover] / visits[i]; }

	/// mctx qtransform_completed_by_mix_value over the actions `available` (indices): sigma(completed Q)
	std::vector<double> sigma(const std::vector<int> & available, const Config & config) const
	{
		std::vector<double> availableLogits;
		for(int i : available)
			availableLogits.push_back(logits[i]);
		const auto prior = softmax(availableLogits);
		int sumVisits = 0;
		int maxVisits = 0;
		double sumProbs = 0;
		for(size_t k = 0; k < available.size(); ++k)
		{
			sumVisits += visits[available[k]];
			maxVisits = std::max(maxVisits, visits[available[k]]);
			if(visits[available[k]] > 0)
				sumProbs += std::max(DBL_MIN, prior[k]);
		}
		double weightedQ = 0;
		for(size_t k = 0; k < available.size(); ++k)
			if(visits[available[k]] > 0)
				weightedQ += std::max(DBL_MIN, prior[k]) * q(available[k]) / sumProbs;
		// The value of the unvisited actions: the network's value mixed with the visited actions' Q
		const double mixed = (value[mover] + sumVisits * weightedQ) / (sumVisits + 1);
		std::vector<double> completed;
		for(int i : available)
			completed.push_back(visits[i] > 0 ? q(i) : mixed);
		const double low = *std::min_element(completed.begin(), completed.end());
		const double high = *std::max_element(completed.begin(), completed.end());
		std::vector<double> result;
		for(double c : completed)
			result.push_back((config.maxvisitInit + maxVisits) * config.valueScale * ((c - low) / std::max(high - low, 1e-8)));
		return result;
	}

	/// The child for (action i, key), or -1
	int child(int i, int64_t key) const
	{
		for(const auto & [k, index] : children[i])
			if(k == key)
				return index;
		return -1;
	}
};

/// Env must provide:
///   using State = ...                       a position: movable
///   State root(uint64_t seed)               a fresh copy of the position searched from, its dice seeded with seed
///   bool terminal(const State &)
///   std::array<double, 2> rewards(State &)  each player's final reward (terminal states)
///   int mover(const State &)                the player to act, 0 or 1
///   int64_t key(const State &)              states with different keys get different nodes
///   Evaluation evaluate(State &)            legal actions, priors and values (non-terminal states)
///   std::vector<int64_t> legal(State &)     the legal actions (non-terminal states)
///   void apply(State &, int64_t action)
template<typename Env>
class Search
{
	using State = typename Env::State;

	Env & env;
	Config config;
	std::vector<Node> nodes; // nodes[0] is the root
	std::vector<double> gumbel; // per root action
	std::vector<int> sequence; // considered visits per simulation

	std::vector<int> all(const Node & node) const
	{
		std::vector<int> indices(node.actions.size());
		for(size_t i = 0; i < indices.size(); ++i)
			indices[i] = static_cast<int>(i);
		return indices;
	}

	/// mctx score_considered: Gumbel + logits + sigma(Q), only for actions visited exactly `considered` times
	std::vector<double> rootScores(int considered) const
	{
		const Node & root = nodes[0];
		const auto sigma = root.sigma(all(root), config);
		const double top = *std::max_element(root.logits.begin(), root.logits.end());
		std::vector<double> scores(root.actions.size());
		for(size_t i = 0; i < scores.size(); ++i)
			scores[i] = root.visits[i] == considered ? std::max(-1e9, gumbel[i] + (root.logits[i] - top) + sigma[i]) : -std::numeric_limits<double>::infinity();
		return scores;
	}

	int selectAtRoot() const
	{
		int simulation = 0;
		for(int v : nodes[0].visits)
			simulation += v;
		return argmax(rootScores(sequence[simulation]));
	}

	/// mctx gumbel_muzero_interior_action_selection, over the available actions; returns an action index
	int selectBelowRoot(const Node & node, const std::vector<int> & available) const
	{
		const auto sigma = node.sigma(available, config);
		std::vector<double> improved;
		int sumVisits = 0;
		for(size_t k = 0; k < available.size(); ++k)
		{
			improved.push_back(node.logits[available[k]] + sigma[k]);
			sumVisits += node.visits[available[k]];
		}
		const auto pi = softmax(improved);
		std::vector<double> scores;
		for(size_t k = 0; k < available.size(); ++k)
			scores.push_back(pi[k] - static_cast<double>(node.visits[available[k]]) / (1 + sumVisits));
		return available[argmax(scores)];
	}

	/// The actions of node legal in state (indices into node.actions)
	std::vector<int> available(const Node & node, State & state)
	{
		const auto legal = env.legal(state);
		std::vector<int> indices;
		for(size_t i = 0; i < node.actions.size(); ++i)
			if(std::find(legal.begin(), legal.end(), node.actions[i]) != legal.end())
				indices.push_back(static_cast<int>(i));
		return indices;
	}

	void simulate(uint64_t seed)
	{
		State state = env.root(seed); // new dice in every simulation: stochastic dice may affect the search strategy (revisit, e.g. chance nodes)
		std::vector<std::pair<int, int>> path; // (node, action index)
		int node = 0;
		std::array<double, 2> value{};
		while(true)
		{
			const int i = node == 0 ? selectAtRoot() : selectBelowRoot(nodes[node], available(nodes[node], state));
			path.emplace_back(node, i);
			env.apply(state, nodes[node].actions[i]);
			if(env.terminal(state))
			{
				value = env.rewards(state);
				break;
			}
			const int64_t key = env.key(state);
			const int next = nodes[node].child(i, key);
			if(next < 0)
			{
				const int mover = env.mover(state);
				nodes.emplace_back(mover, env.evaluate(state));
				nodes[node].children[i].emplace_back(key, static_cast<int>(nodes.size()) - 1);
				value = nodes.back().value;
				break;
			}
			if(available(nodes[next], state).empty()) // a leaf, or a state none of whose actions the node knows
			{
				value = env.evaluate(state).values;
				break;
			}
			node = next;
		}
		for(const auto & [n, i] : path)
		{
			nodes[n].visits[i] += 1;
			nodes[n].totals[i][0] += value[0];
			nodes[n].totals[i][1] += value[1];
		}
	}

public:
	Search(Env & env, const Config & config)
		: env(env)
		, config(config)
	{
	}

	Result run(uint64_t seed)
	{
		SplitMix64 rng(seed);
		State rootState = env.root(rng.next());
		nodes.clear();
		nodes.emplace_back(env.mover(rootState), env.evaluate(rootState));
		const int n = static_cast<int>(nodes[0].actions.size());
		if(n == 0)
			throw std::runtime_error("search: no legal action at the root");
		gumbel.assign(n, 0.0);
		for(double & g : gumbel)
			g = config.gumbelScale * rng.gumbel();

		Result result;
		if(n > 1) // one legal action: nothing to search
		{
			sequence = consideredVisitSequence(std::min(config.maxConsidered, n), config.simulations);
			for(int s = 0; s < config.simulations; ++s)
				simulate(rng.next());
			result.simulations = config.simulations;
		}

		const Node & searched = nodes[0];
		const int considered = *std::max_element(searched.visits.begin(), searched.visits.end());
		result.action = searched.actions[argmax(rootScores(considered))];
		result.actions = searched.actions;
		result.visits = searched.visits;
		for(int i = 0; i < n; ++i)
			result.q.push_back(searched.visits[i] > 0 ? searched.q(i) : std::numeric_limits<double>::quiet_NaN());
		const auto sigma = searched.sigma(all(searched), config);
		std::vector<double> improved;
		for(int i = 0; i < n; ++i)
			improved.push_back(searched.logits[i] + sigma[i]);
		result.policy = softmax(improved);
		result.value = searched.value[searched.mover];
		result.nodes = static_cast<int>(nodes.size());
		return result;
	}
};
}
