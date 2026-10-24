"""The C++ search and reward (agents/AlphaBattleAI/GumbelSearch.h, Reward.h) against their Python copies
(battle_model/search.py, reward.py + data.summarize), through tools/SearchSelfTest.cpp.

This is the alarm for the two copies of the search: if one is changed without the other, it fails.
It builds the self-test binary first (a few seconds when something changed).
"""
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from battle_model import vocab
from battle_model.data import RESULT_ESCAPE, RESULT_NORMAL, RESULT_SURRENDER, summarize
from battle_model.reward import other_side_value, to_scale
from battle_model.search import Config, Search, SplitMix64

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build"
BINARY = BUILD / "bin" / "alpha_search_selftest"
MASK64 = (1 << 64) - 1


def mix(x):
	return SplitMix64(x).next()


class State:
	def __init__(self, seed):
		self.position = [1, 0]
		self.turn = 0
		self.mover = 0
		self.dice = SplitMix64(seed)


class ToyGame:
	"""The toy game of tools/SearchSelfTest.cpp: a stochastic general-sum race. Each player moves its counter
	forward by its action (0..4) plus a die (0..2); a 2 lets the same player move again (like good morale). Which
	actions are legal depends on the position, so on the dice. The first to 12 wins; after 10 moves nobody does.
	Some positions are leaves the search does not go beyond. The rewards are not zero-sum (the search must not assume it). The "network" is a
	hash of the position."""

	@staticmethod
	def hash(s):
		return mix(s.position[0] * 1000003 + s.position[1] * 1009 + s.turn * 17 + s.mover)

	def root(self, seed):
		return State(seed)

	def terminal(self, s):
		return s.position[0] >= 12 or s.position[1] >= 12 or s.turn >= 10

	def rewards(self, s):
		if s.position[0] < 12 and s.position[1] < 12:
			return [0.04 * s.position[0] - 0.3, 0.04 * s.position[1] - 0.3]
		winner = 0 if s.position[0] >= 12 else 1
		r = [0.0, 0.0]
		r[winner] = 1.0 - 0.05 * s.turn
		r[1 - winner] = -0.6 + 0.03 * s.position[1 - winner]
		return r

	def mover(self, s):
		return s.mover

	def key(self, s):
		return s.mover  # like the acting unit in battles: the same node sees several states

	def legal(self, s):
		return [a for a in range(5) if mix((self.hash(s) + a) & MASK64) % 3 != 0] or [0]

	def evaluate(self, s):
		h = self.hash(s)
		values = [(mix((h + 7919 * (p + 1)) & MASK64) % 2001) / 1000.0 - 1.0 for p in range(2)]
		if s.turn >= 2 and h % 4 == 0:  # a leaf: the search does not go on from here (like the round cap in battles)
			return [], [], values
		actions = self.legal(s)
		return actions, [(mix(h ^ ((a + 1) * 256)) % 4001) / 1000.0 - 2.0 for a in actions], values

	def apply(self, s, action):
		die = s.dice.next() % 3
		s.position[s.mover] += action + die
		s.turn += 1
		if die != 2:
			s.mover = 1 - s.mover


@pytest.fixture(scope="module")
def selftest():
	"""Run commands through the (freshly built) C++ self-test; returns its JSON answers."""
	if not (BUILD / "build.ninja").exists():
		pytest.fail("no VCMI build in build/: configure it (README) so the C++ search can be checked against search.py")
	built = subprocess.run(["cmake", "--build", str(BUILD), "--target", "alpha_search_selftest"], capture_output=True, text=True)
	assert built.returncode == 0, built.stdout[-3000:] + built.stderr[-3000:]

	def run(commands):
		done = subprocess.run([str(BINARY)], input="".join(c + "\n" for c in commands), capture_output=True, text=True, check=True)
		return [json.loads(line) for line in done.stdout.splitlines()]

	return run


SEARCH_CASES = [
	(seed, simulations, considered, scale)
	for seed in range(1, 9)
	for simulations, considered, scale in [(1, 16, 1.0), (2, 2, 1.0), (7, 4, 0.0), (16, 16, 1.0), (32, 16, 1.0), (33, 3, 1.0), (64, 1, 1.0), (100, 16, 0.5)]
]


def test_cpp_search_matches_python(selftest):
	answers = selftest([f"search {seed} {sims} {considered} {scale}" for seed, sims, considered, scale in SEARCH_CASES])
	assert len(answers) == len(SEARCH_CASES)
	for (seed, sims, considered, scale), cpp in zip(SEARCH_CASES, answers):
		python = Search(ToyGame(), Config(simulations=sims, max_considered=considered, gumbel_scale=scale)).run(seed)
		case = f"seed {seed}, {sims} simulations, {considered} considered, gumbel scale {scale}"
		for key in ("action", "actions", "visits", "nodes", "simulations"):
			assert cpp[key] == python[key], f"{case}: {key} differs: C++ {cpp[key]}, Python {python[key]}"
		for key in ("q", "policy"):
			for a, b in zip(cpp[key], python[key]):
				assert (a is None) == (b is None) and (a is None or math.isclose(a, b, rel_tol=0, abs_tol=1e-12)), f"{case}: {key} differs: C++ {cpp[key]}, Python {python[key]}"
		assert math.isclose(cpp["value"], python["value"], abs_tol=1e-12)


def test_search_cases_exercise_the_algorithm(selftest):
	"""The toy cases must reach what the battles will: several nodes per root action, more than one action chosen."""
	answers = selftest([f"search {seed} {sims} {considered} {scale}" for seed, sims, considered, scale in SEARCH_CASES])
	assert len({a["action"] for a in answers}) > 1
	assert max(a["nodes"] for a in answers) > 50
	big = [a for a, (_, sims, _, _) in zip(answers, SEARCH_CASES) if sims == 100]
	assert all(sum(a["visits"]) == 100 for a in big)


def test_search_cases_reach_every_branch(monkeypatch):
	"""The cases must reach every branch of the search, or a change there would go unnoticed: terminal states,
	children split by key, nodes whose actions are not all legal in the current state, leaves, and states none of
	whose actions the node knows."""
	import battle_model.search as search

	seen = dict.fromkeys(["terminal", "split", "restricted", "leaf", "unknown state"], 0)
	game = ToyGame()
	terminal, select, evaluate, available = game.terminal, search.Search.select_below_root, game.evaluate, search.Search.available

	def counting_terminal(s):
		seen["terminal"] += terminal(s)
		return terminal(s)

	def counting_select(self, node, indices):
		seen["restricted"] += len(indices) < len(node.actions)
		return select(self, node, indices)

	def counting_available(self, node, state):
		indices = available(self, node, state)
		seen["unknown state"] += bool(node.actions) and not indices
		return indices

	def counting_evaluate(s):
		result = evaluate(s)
		seen["leaf"] += not result[0]
		return result

	game.terminal, game.evaluate = counting_terminal, counting_evaluate
	monkeypatch.setattr(search.Search, "select_below_root", counting_select)
	monkeypatch.setattr(search.Search, "available", counting_available)
	for seed, sims, considered, scale in SEARCH_CASES:
		s = Search(game, Config(simulations=sims, max_considered=considered, gumbel_scale=scale))
		s.run(seed)
		stack = [s.root]
		while stack:
			node = stack.pop()
			seen["split"] += sum(len(children) > 1 for children in node.children)
			stack += [child for children in node.children for child in children.values()]
	assert all(seen.values()), seen


def test_improved_policy_prefers_better_actions():
	"""Sanity of the algorithm itself: with many simulations the improved policy follows the searched Q."""
	result = Search(ToyGame(), Config(simulations=200, gumbel_scale=0.0)).run(3)
	visited = [i for i, n in enumerate(result["visits"]) if n > 0]
	best_q = max(visited, key=lambda i: result["q"][i])
	assert result["policy"][best_q] == max(result["policy"][i] for i in visited)
	assert abs(sum(result["policy"]) - 1) < 1e-12


CREATURE = 0  # pikeman: armies of one creature, counts -> values


def battle_end_cases(n=300, seed=0):
	rng = np.random.default_rng(seed)
	for _ in range(n):
		start = rng.integers(0, 60, 2)
		end = np.array([rng.integers(0, s + 1) for s in start])
		mana_max = rng.integers(0, 3, 2) * rng.integers(1, 60, 2)
		mana_start = np.array([rng.integers(0, m + 1) for m in mana_max])
		mana_end = np.array([rng.integers(0, s + 1) for s in mana_start])
		power = rng.integers(0, 10, 2)
		winner = int(rng.choice([0, 1, -1]))
		result = int(rng.choice([RESULT_NORMAL, RESULT_ESCAPE, RESULT_SURRENDER])) if winner >= 0 else RESULT_NORMAL
		cost = [0, 0]
		if result == RESULT_SURRENDER:
			cost[1 - winner] = int(rng.integers(0, 4000))
		level = (rng.integers(0, 30, 2) + rng.random(2).round(3)) * (rng.random(2) < 0.8)  # 0: no hero
		artifacts = rng.integers(0, 20000, 2) * (rng.random(2) < 0.5)
		yield dict(start=start.tolist(), end=end.tolist(), mana_start=mana_start.tolist(), mana_max=mana_max.tolist(), mana_end=mana_end.tolist(), power=power.tolist(), winner=winner, result=result, rounds=int(rng.integers(1, 130)), cost=cost, level=level.tolist(), artifacts=artifacts.tolist())


def python_reward(case, side):
	"""Training's reward for side: data.summarize over the events the battle would record from that side."""
	armies = [[[CREATURE, case["start"][s]]] for s in (0, 1)]
	casualties = [[[CREATURE, case["start"][s] - case["end"][s]]] for s in (0, 1)]
	start = {"event": "start", "our_side": side, "armies": armies, "our_mana": case["mana_start"][side], "our_mana_max": case["mana_max"][side], "enemy_mana": case["mana_start"][1 - side], "enemy_mana_max": case["mana_max"][1 - side], "powers": case["power"], "heroes": [[case["level"][s], case["artifacts"][s]] for s in (0, 1)]}
	end = {"event": "end", "round": case["rounds"], "winner": case["winner"], "result": case["result"], "casualties": casualties, "alive": [[], []], "our_mana": case["mana_end"][side], "enemy_mana": case["mana_end"][1 - side]}
	events = [start, end]
	if case["result"] == RESULT_SURRENDER and case["winner"] == 1 - side:
		events.append({"event": "surrender", "cost": case["cost"][side]})
	return summarize(events).reward


@pytest.mark.skipif(not vocab.VOCAB_PATH.exists(), reason="run scripts/export_vocab.sh to generate the vocabulary")
def test_cpp_reward_matches_training_reward(selftest):
	"""Each side as its own AI scores it: it knows what it paid to surrender, not what the enemy paid (0)."""
	value = vocab.creature_value(CREATURE)
	cases = [(c, side) for c in battle_end_cases() for side in (0, 1)]
	commands = []
	for c, side in cases:
		cost = [c["cost"][s] if s == side else 0 for s in (0, 1)]
		numbers = [c["start"][0] * value, c["start"][1] * value, c["end"][0] * value, c["end"][1] * value, *c["mana_start"], *c["power"], *c["mana_end"], c["winner"], c["result"], c["rounds"], *cost, *c["level"], *c["artifacts"]]
		commands.append("reward " + " ".join(str(x) for x in numbers))
	for (c, side), cpp in zip(cases, selftest(commands), strict=True):
		expected = python_reward(c, side)
		assert math.isclose(cpp["rewards"][side], expected, rel_tol=0, abs_tol=1e-12), (c, side, cpp, expected)
		assert math.isclose(cpp["scaled"][side], to_scale(expected), rel_tol=0, abs_tol=1e-12), (c, side, cpp)


def test_cpp_other_side_value_matches_python(selftest):
	values = [-1.0, -0.73, -0.2, 0.0, 0.1, 0.55, 1.0]
	for value, cpp in zip(values, selftest([f"other_side_value {v!r}" for v in values]), strict=True):
		assert math.isclose(cpp["value"], other_side_value(value), rel_tol=0, abs_tol=1e-12)
