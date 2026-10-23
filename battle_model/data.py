"""Battle records -> training examples -> DataLoader.

A record (written by AlphaBattleAI when play.py sets ALPHA_BATTLE_RECORD_DIR) holds one side's view of
one battle: every decision (model inputs, model outputs and the chosen action) and events.jsonl (the
armies at the start, the army units alive at the end of each round, how the battle ended).

From it we compute, for every decision:
	inputs        the model inputs (schema.INPUTS), dense
	action        the action taken (distributions.ACTION_FIELDS)
	old_log_prob  log p(action) under the model that played, from its recorded outputs
	advantage     GAE advantage (losses.battle_returns_and_advantages), with V (the final-reward head)
	value_labels  labels of all value heads (losses.value_targets), on the -1..1 scale
	forced        replayed from a prefix (play.Game.prefix), not chosen by the model: not trained on
and per battle a BattleSummary with the reward (reward.py) and the outcome.

A decision is one compressed decision_*.npz (a .npy per tensor plus action.json, ~15 KB; the tensors are
mostly zeros and -inf); older records were a decision_* directory of .npy files (~0.3 MB on disk). Training
deletes the records after each epoch. Keeping data across epochs will want the compact format (sparse
masks, ~115 bytes a decision).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import vocab
from .distributions import ACTION_FIELDS, log_prob, stack_actions
from .losses import AdvantageConfig, battle_returns_and_advantages, normalize_advantages, value_targets
from .reward import Outcome, RewardConfig, own_kept_value, round_score, terminal_reward, to_scale
from .schema import INPUTS, OUTPUTS

# EBattleResult
RESULT_NORMAL, RESULT_ESCAPE, RESULT_SURRENDER = 0, 1, 2


@dataclass
class BattleSummary:
	our_side: int
	outcome: Outcome
	reward: float  # terminal reward, reward units
	rounds: int
	own_start_value: float
	enemy_start_value: float
	own_kept: float
	enemy_lost: float
	mana_used_fraction: float
	round_scores: dict = field(default_factory=dict)  # {round: round_score at its end} while the battle went on
	decisions: int = 0
	advantage_std: float = 0.0  # std of the raw (not normalised) advantages, -1..1 scale: how much there is to learn
	value_error: float = 0.0  # mean |V - final reward| over the decisions, -1..1 scale, V from the model that played


def army_value(stacks):
	"""Value of [[creature id, count], ...] (gold + resources)."""
	return float(sum(vocab.creature_value(creature) * count for creature, count in stacks))


def read_events(record_dir):
	return [json.loads(line) for line in (Path(record_dir) / "events.jsonl").read_text().splitlines() if line.strip()]


def summarize(events, hero_value=0.0, fresh_hero_value=0.0, config=RewardConfig()):
	"""The reward and outcome of one battle, from our side's events.

	hero_value: our hero's value (reward.hero_value); 0 for level-1 heroes without artifacts, as in the
	current scenarios (not recorded yet). The enemy hero's value is not counted either.
	"""
	errors = [e["message"] for e in events if e["event"] == "error"]
	if errors:
		raise RuntimeError("battle record is unusable: " + "; ".join(errors))
	start = next(e for e in events if e["event"] == "start")
	end = next(e for e in events if e["event"] == "end")
	surrender = next((e for e in events if e["event"] == "surrender"), None)
	our = start["our_side"]
	enemy = 1 - our
	own_army_start = army_value(start["armies"][our])
	enemy_army_start = army_value(start["armies"][enemy])
	own_start = own_army_start + hero_value

	def mana_used(mana):
		return (start["our_mana"] - mana) / start["our_mana_max"] if start["our_mana_max"] > 0 else 0.0

	if end["winner"] == our:
		outcome = Outcome.WIN
	elif end["winner"] not in (0, 1):  # no winner: both armies destroyed
		outcome = Outcome.DRAW
	elif end["result"] == RESULT_ESCAPE:
		outcome = Outcome.RETREAT
	elif end["result"] == RESULT_SURRENDER:
		outcome = Outcome.SURRENDER
	else:
		outcome = Outcome.LOSS

	# An enemy that retreated has lost its whole army. One that surrendered keeps its army but pays us for
	# it, so it also counts as fully destroyed. The real value of an enemy surrender depends on the game:
	# this is right for a 2-player zero-sum game; with more players a surrender might even beat a retreat,
	# which needs information from outside the battle.
	if outcome == Outcome.WIN and end["result"] in (RESULT_ESCAPE, RESULT_SURRENDER):
		enemy_lost = enemy_army_start
	else:
		enemy_lost = army_value(end["casualties"][enemy])
	own_army_end = own_army_start - army_value(end["casualties"][our])
	kept = own_kept_value(outcome, own_army_end, hero_value, fresh_hero_value, surrender["cost"] if surrender else 0.0)
	rounds = end["round"]  # VCMI numbering: the first round is 1
	reward = terminal_reward(outcome, own_start, kept, enemy_army_start, enemy_lost, rounds, mana_used(end["our_mana"]), config)

	scores = {}
	for e in events:
		if e["event"] == "round_end" and e["round"] >= 1:  # round 0 is the start of the battle
			own_alive = army_value(e["alive"][our]) + hero_value
			enemy_lost_so_far = enemy_army_start - army_value(e["alive"][enemy])
			scores[e["round"]] = round_score(own_start, own_alive, enemy_army_start, enemy_lost_so_far, e["round"], mana_used(e["our_mana"]), config)

	return BattleSummary(our, outcome, reward, rounds, own_start, enemy_army_start, kept, enemy_lost, mana_used(end["our_mana"]), scores)


def load_decision(path):
	"""(tensors, action) of one recorded decision: every input and output tensor by name, with the batch axis."""
	path = Path(path)
	names = [s.name for s in INPUTS + OUTPUTS]
	if path.suffix == ".npz":
		with np.load(path) as archive:
			return {name: archive[name] for name in names}, json.loads(archive["action.json"])
	return {name: np.load(path / f"{name}.npy") for name in names}, json.loads((path / "action.json").read_text())


def decision_paths(record_dir):
	"""A record's decisions in the order they were made (decision_00000.npz, ...; or older decision_* directories)."""
	return sorted(Path(record_dir).glob("decision_*"))


def read_decisions(record_dir):
	"""[(inputs, outputs, action)] in the order they were made; each inputs/outputs without the batch axis."""
	decisions = []
	for path in decision_paths(record_dir):
		tensors, action = load_decision(path)
		inputs = {s.name: tensors[s.name][0] for s in INPUTS}
		outputs = {s.name: tensors[s.name][0] for s in OUTPUTS}
		decisions.append((inputs, outputs, action))
	return decisions


def battle_examples(record_dir, reward_config=RewardConfig(), advantage_config=AdvantageConfig()):
	"""(examples, summary) for one recorded battle. Each example is a dict as described at the top."""
	summary = summarize(read_events(record_dir), config=reward_config)
	decisions = read_decisions(record_dir)
	summary.decisions = len(decisions)
	if not decisions:
		return [], summary

	outputs = {s.name: torch.from_numpy(np.stack([d[1][s.name] for d in decisions])) for s in OUTPUTS}
	actions = stack_actions([d[2] for d in decisions])
	old_log_prob = log_prob(outputs, actions)
	values = outputs["value"][:, -1]  # V, the final-reward head (-1..1)
	rewards = torch.zeros(len(decisions))
	rewards[-1] = to_scale(summary.reward, reward_config)
	_, advantages = battle_returns_and_advantages(values, rewards, advantage_config)
	summary.advantage_std = float(advantages.std()) if len(advantages) > 1 else 0.0
	summary.value_error = float((values - rewards[-1]).abs().mean())
	rounds = [int(d[0]["g_round"]) for d in decisions]
	labels = value_targets(rounds, summary.round_scores, summary.reward, reward_config=reward_config)

	examples = [
		{"inputs": d[0], "action": d[2], "old_log_prob": old_log_prob[i], "advantage": advantages[i], "value_labels": labels[i], "forced": bool(d[2].get("forced", False))}
		for i, d in enumerate(decisions)
	]
	return examples, summary


def collate(examples):
	"""List of examples -> one batch: inputs as model-ready tensors, the rest stacked."""
	return {
		"inputs": {s.name: torch.from_numpy(np.stack([e["inputs"][s.name] for e in examples])) for s in INPUTS},
		"action": {f: torch.tensor([e["action"][f] for e in examples], dtype=torch.long) for f in ACTION_FIELDS},
		"old_log_prob": torch.stack([e["old_log_prob"] for e in examples]),
		"advantage": torch.stack([e["advantage"] for e in examples]),
		"value_labels": torch.stack([e["value_labels"] for e in examples]),
	}


def make_dataloader(record_dirs, batch_size=64, shuffle=True, reward_config=RewardConfig(), advantage_config=AdvantageConfig()):
	"""(DataLoader over the model's own decisions in the given battle records, the battles' summaries, every
	battle's examples in order including replayed (forced) ones).

	Advantages are computed over whole battles, then normalised per side we played over the decisions
	trained on, not per batch (losses.normalize_advantages).
	"""
	examples, summaries, lines, sides, battles = [], [], [], [], []
	for i, record_dir in enumerate(record_dirs):
		battle, summary = battle_examples(record_dir, reward_config, advantage_config)
		lines.append(battle)
		summaries.append(summary)
		chosen = [e for e in battle if not e["forced"]]
		examples += chosen
		sides += [summary.our_side] * len(chosen)
		battles += [i] * len(chosen)
	if examples:
		advantages = normalize_advantages(torch.stack([e["advantage"] for e in examples]), torch.tensor(sides), torch.tensor(battles), advantage_config)
		for example, advantage in zip(examples, advantages):
			example["advantage"] = advantage
	loader = torch.utils.data.DataLoader(examples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
	return loader, summaries, lines
