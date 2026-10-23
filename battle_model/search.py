"""READABLE COPY ONLY: nothing imports this file to play or train, and changing it changes nothing.

The search that plays is the C++ one, agents/AlphaBattleAI/GumbelSearch.h, next to the ONNX model (fast).
This file is the same algorithm written to be read and reviewed. tests/test_search.py runs both on a toy game
and fails if they make different decisions: a change to one must be made to the other too.

Gumbel AlphaZero ("Policy improvement by planning with Gumbel", Danihelka et al., ICLR 2022, with the formulas
of DeepMind's mctx gumbel_muzero_policy), for two players who each maximise their own reward (general-sum),
in a game with dice:

- Root: Gumbel-Top-k picks the actions to consider and Sequential Halving spends the simulations on them; the
  action played is the best of the last ones left. With few simulations this still improves on the network's
  policy, where visit-count MCTS needs many more.
- Below the root: the action maximising pi'(a) - N(a) / (1 + sum N), with pi' = softmax(logits + sigma(completed Q)).
- Training target (later): the improved policy softmax(logits + sigma(completed Q)) at the root.
- Each player's Q is its own: a node's mover chooses by its own expected reward; a leaf's evaluation gives one
  value per player.
- Open loop: every simulation starts from a fresh copy of the root with new dice and replays the tree's actions.
  A node's statistics are over the states its action sequence reached. A child is split when the next state has
  another key (in battles: another acting unit, e.g. after good morale or a death), since its actions and priors
  are another unit's. At each node only the actions legal in the current state are considered.

The environment (env) gives:
	root(seed)       a fresh copy of the position searched from, its dice seeded with seed
	terminal(state)  rewards(state): each player's final reward (terminal states)
	mover(state)     the player to act, 0 or 1
	key(state)       states with different keys get different nodes
	evaluate(state)  (legal actions, their log-prior logits, [value of player 0, value of player 1]); no
	                 actions: the search does not go deeper there (a leaf)
	legal(state)     the legal actions
	apply(state, action)
"""
import math
import sys
from dataclasses import dataclass, field

MASK64 = (1 << 64) - 1


class SplitMix64:
	"""The search's noise and seeds, and the toy game's dice: the same numbers as in C++."""

	def __init__(self, seed):
		self.state = seed & MASK64

	def next(self):
		self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
		z = self.state
		z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
		z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
		return z ^ (z >> 31)

	def uniform(self):
		"""Uniform in (0, 1), never 0 or 1."""
		return ((self.next() >> 11) + 0.5) * 2.0**-53

	def gumbel(self):
		return -math.log(-math.log(self.uniform()))


@dataclass
class Config:
	simulations: int = 32
	max_considered: int = 16  # root actions Sequential Halving starts with
	gumbel_scale: float = 1.0  # 0: no noise, the best action by the search (greedy play)
	# sigma(q) = (maxvisit_init + max N) * value_scale * q, with q rescaled to 0..1 at each node (mctx defaults)
	value_scale: float = 0.1
	maxvisit_init: float = 50.0


def considered_visit_sequence(max_considered, simulations):
	"""For each simulation, the visit count the root action it explores must have (Sequential Halving).

	Each phase gives every action still considered the same number of extra visits, then the better half stays.
	"""
	if max_considered <= 1:
		return list(range(simulations))
	log2max = 0
	while (1 << log2max) < max_considered:
		log2max += 1
	sequence = []
	visits = [0] * max_considered
	considered = max_considered
	while len(sequence) < simulations:
		extra = max(1, int(simulations / (log2max * considered)))
		for _ in range(extra):
			sequence.extend(visits[:considered])
			for i in range(considered):
				visits[i] += 1
		considered = max(2, considered // 2)
	return sequence[:simulations]


def softmax(logits):
	top = max(logits)
	p = [math.exp(x - top) for x in logits]
	total = 0.0
	for x in p:
		total += x
	return [x / total for x in p]


def argmax(values):
	"""Index of the first largest entry."""
	best = 0
	for i in range(1, len(values)):
		if values[i] > values[best]:
			best = i
	return best


@dataclass
class Node:
	mover: int
	actions: list
	logits: list
	value: list  # the network's [player 0, player 1], when the node was made
	visits: list = field(default_factory=list)  # per action
	totals: list = field(default_factory=list)  # per action: sum of the values backed up through it, per player
	children: list = field(default_factory=list)  # per action: {key of the next state: node}

	def __post_init__(self):
		self.visits = [0] * len(self.actions)
		self.totals = [[0.0, 0.0] for _ in self.actions]
		self.children = [{} for _ in self.actions]

	def q(self, i):
		"""The mover's mean reward after action i."""
		return self.totals[i][self.mover] / self.visits[i]

	def sigma(self, available, config):
		"""sigma(completed Q) for the actions `available` (indices), mctx qtransform_completed_by_mix_value.

		Unvisited actions get the network's value mixed with the visited actions' Q ("completed Q"); then Q is
		rescaled to 0..1 at this node and scaled up as the visits grow.
		"""
		prior = softmax([self.logits[i] for i in available])
		visited = [k for k, i in enumerate(available) if self.visits[i] > 0]
		sum_visits = sum(self.visits[i] for i in available)
		max_visits = max(self.visits[i] for i in available)
		sum_probs = 0.0
		for k in visited:
			sum_probs += max(sys.float_info.min, prior[k])
		weighted_q = 0.0
		for k in visited:
			weighted_q += max(sys.float_info.min, prior[k]) * self.q(available[k]) / sum_probs
		mixed = (self.value[self.mover] + sum_visits * weighted_q) / (sum_visits + 1)
		completed = [self.q(i) if self.visits[i] > 0 else mixed for i in available]
		low, high = min(completed), max(completed)
		return [(config.maxvisit_init + max_visits) * config.value_scale * ((c - low) / max(high - low, 1e-8)) for c in completed]


class Search:
	def __init__(self, env, config):
		self.env = env
		self.config = config

	def root_scores(self, considered):
		"""Gumbel + logits + sigma(Q), only for the root actions visited exactly `considered` times (mctx score_considered)."""
		root = self.root
		sigma = root.sigma(list(range(len(root.actions))), self.config)
		top = max(root.logits)
		return [
			max(-1e9, self.gumbel[i] + (root.logits[i] - top) + sigma[i]) if root.visits[i] == considered else -math.inf
			for i in range(len(root.actions))
		]

	def select_at_root(self):
		simulation = sum(self.root.visits)
		return argmax(self.root_scores(self.sequence[simulation]))

	def select_below_root(self, node, available):
		"""The action index maximising pi'(a) - N(a) / (1 + sum N) over the available actions (mctx interior selection)."""
		sigma = node.sigma(available, self.config)
		pi = softmax([node.logits[i] + s for i, s in zip(available, sigma)])
		sum_visits = sum(node.visits[i] for i in available)
		return available[argmax([p - node.visits[i] / (1 + sum_visits) for p, i in zip(pi, available)])]

	def available(self, node, state):
		"""The node's actions legal in this state (indices)."""
		legal = set(self.env.legal(state))
		return [i for i, a in enumerate(node.actions) if a in legal]

	def simulate(self, seed):
		env = self.env
		state = env.root(seed)  # new dice in every simulation: stochastic dice may affect the search strategy (revisit, e.g. chance nodes)
		path = []
		node = self.root
		while True:
			i = self.select_at_root() if node is self.root else self.select_below_root(node, self.available(node, state))
			path.append((node, i))
			env.apply(state, node.actions[i])
			if env.terminal(state):
				value = env.rewards(state)
				break
			key = env.key(state)
			child = node.children[i].get(key)
			if child is None:
				actions, logits, values = env.evaluate(state)
				node.children[i][key] = child = Node(env.mover(state), actions, logits, values)
				value = child.value
				break
			if not self.available(child, state):  # a leaf, or a state none of whose actions the node knows
				value = env.evaluate(state)[2]
				break
			node = child
		for n, i in path:
			n.visits[i] += 1
			n.totals[i][0] += value[0]
			n.totals[i][1] += value[1]

	def run(self, seed):
		"""Search from env.root; returns a dict like the C++ Result."""
		env, config = self.env, self.config
		rng = SplitMix64(seed)
		root_state = env.root(rng.next())
		actions, logits, values = env.evaluate(root_state)
		self.root = root = Node(env.mover(root_state), actions, logits, values)
		self.nodes = 1
		n = len(root.actions)
		if n == 0:
			raise RuntimeError("search: no legal action at the root")
		self.gumbel = [config.gumbel_scale * rng.gumbel() for _ in range(n)]

		simulations = 0
		if n > 1:  # one legal action: nothing to search
			self.sequence = considered_visit_sequence(min(config.max_considered, n), config.simulations)
			for _ in range(config.simulations):
				self.simulate(rng.next())
			simulations = config.simulations

		considered = max(root.visits)
		sigma = root.sigma(list(range(n)), config)
		return {
			"action": root.actions[argmax(self.root_scores(considered))],
			"actions": list(root.actions),
			"visits": list(root.visits),
			"q": [root.q(i) if root.visits[i] > 0 else None for i in range(n)],
			"policy": softmax([root.logits[i] + sigma[i] for i in range(n)]),
			"value": root.value[root.mover],
			"nodes": count_nodes(root),
			"simulations": simulations,
		}


def count_nodes(node):
	return 1 + sum(count_nodes(child) for children in node.children for child in children.values())
