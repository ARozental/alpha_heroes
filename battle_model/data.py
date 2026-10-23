"""Battle records -> training examples -> DataLoader.

A record (Record) holds one side's view of one battle: every decision (model inputs, model outputs and the
chosen action) and its events (the armies at the start, the army units alive at the end of each round, how
the battle ended). AlphaBattleAI keeps it in memory and sends it in one piece at the end of the battle to
play.py's RecordServer (parse_record): training data never touches the disk. For debugging it can also write
it to a folder (ALPHA_BATTLE_RECORD_DIR: events.jsonl and a decision_*.npz per decision, load_record).

From it we compute, for every decision:
	inputs        the model inputs (schema.INPUTS), dense
	action        the action taken (distributions.ACTION_FIELDS)
	old_log_prob  log p(action) under the model that played, from its recorded outputs
	advantage     GAE advantage (losses.battle_returns_and_advantages), with V (the final-reward head)
	value_labels  labels of all value heads (losses.value_targets), on the -1..1 scale
	forced        replayed from a prefix (play.Game.prefix), not chosen by the model: not trained on
and per battle a BattleSummary with the reward (reward.py) and the outcome.

A sent record is ~190 KB a decision, in memory only. Keeping data across epochs will want a compact format
(sparse masks, ~115 bytes a decision).
"""
import io
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import vocab
from .distributions import ACTION_FIELDS, log_prob, stack_actions
from .losses import AdvantageConfig, battle_returns_and_advantages, normalize_advantages, policy_advantages, value_targets
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
	values: list = field(default_factory=list)  # V at every decision (-1..1), for the value head's explained variance
	# Per decision but the last: (our army value lost, enemy army value killed, change of V) until the next decision,
	# values as fractions of the armies at the start: what V's changes, which make the advantages, reward (credit_check)
	credit: list = field(default_factory=list)


def army_value(stacks):
	"""Value of [[creature id, count], ...] (gold + resources)."""
	return float(sum(vocab.creature_value(creature) * count for creature, count in stacks))


def army_values(tensors, side):
	"""[T] value of a battle side's living army units at every decision of a record (summoned units and clones excluded)."""
	alive = (tensors["u_present"] != 0) & (tensors["u_alive"] != 0) & (tensors["u_side"] == side) & (tensors["u_is_summoned"] == 0) & (tensors["u_is_clone"] == 0)
	prices = np.vectorize(lambda c: vocab.creature_value(int(c)) if c >= 0 else 0.0)(tensors["u_creature"]) if tensors["u_creature"].size else np.zeros_like(tensors["u_creature"], dtype=float)
	return (prices * tensors["u_count"] * alive).sum(axis=1)


def credit_check(summaries):
	"""What the value head rewards, from the battles of one side: (explained variance of V, change of V per fraction
	of our army lost, change of V per fraction of the enemy army killed), by least squares over every step between
	two decisions. These changes of V are what GAE turns into the advantages (losses.battle_returns_and_advantages).

	On a lost battle the reward counts only kills (about +0.47 on the -1..1 scale per whole enemy army), so an
	honest V changes by ~0 when we lose units and positively when we kill; a negative kill coefficient or a strongly
	negative loss one means the advantages teach the wrong thing."""
	values = np.array([v for s in summaries for v in s.values])
	rewards = np.array([to_scale(s.reward) for s in summaries for _ in s.values])
	ev = float(1 - np.var(rewards - values) / np.var(rewards)) if len(values) > 1 and np.var(rewards) > 1e-12 else float("nan")
	rows = np.array([c for s in summaries for c in s.credit])
	if len(rows) < 3:
		return ev, float("nan"), float("nan")
	coef, *_ = np.linalg.lstsq(np.c_[rows[:, :2], np.ones(len(rows))], rows[:, 2], rcond=None)
	return ev, float(coef[0]), float(coef[1])


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


@dataclass
class Record:
	"""One side's record of one battle. tensors: every input and output by name, decisions stacked on the first axis."""
	events: list
	actions: list
	tensors: dict

	@property
	def decisions(self):
		return len(self.actions)


RECORD_MAGIC = b"ALPHAREC1\n"


def parse_record(data):
	"""A Record from the bytes AlphaBattleAI sends (agents/AlphaBattleAI/Dump.h BattleRecord): magic, then
	entries of u32 name length, name, u64 size, bytes."""
	if not data.startswith(RECORD_MAGIC):
		raise ValueError("not a battle record")
	view, offset, entries = memoryview(data), len(RECORD_MAGIC), {}
	while offset < len(data):
		(name_length,) = struct.unpack_from("<I", data, offset)
		name = bytes(view[offset + 4 : offset + 4 + name_length]).decode()
		offset += 4 + name_length
		(size,) = struct.unpack_from("<Q", data, offset)
		entries[name] = view[offset + 8 : offset + 8 + size]
		offset += 8 + size
	lines = lambda name: [json.loads(line) for line in bytes(entries[name]).decode().splitlines() if line.strip()]
	tensors = {s.name: np.load(io.BytesIO(entries[s.name + ".npy"])) for s in INPUTS + OUTPUTS}
	return Record(lines("events.jsonl"), lines("actions.jsonl"), tensors)


def load_record(path):
	"""A Record from disk: a record folder written for debugging (ALPHA_BATTLE_RECORD_DIR), or a file holding sent bytes."""
	path = Path(path)
	if path.is_file():
		return parse_record(path.read_bytes())
	decisions = [load_decision(d) for d in decision_paths(path)]
	names = [s.name for s in INPUTS + OUTPUTS]
	if decisions:
		tensors = {name: np.concatenate([t[name] for t, _ in decisions]) for name in names}
	else:
		tensors = {s.name: np.zeros((0,) + s.shape, dtype=np.dtype({"int64": np.int64, "bool": np.bool_, "float32": np.float32}[s.dtype])) for s in INPUTS + OUTPUTS}
	return Record(read_events(path), [a for _, a in decisions], tensors)


def as_record(record):
	return record if isinstance(record, Record) else load_record(record)


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


def read_decisions(record):
	"""[(inputs, outputs, action)] of a Record (or a record folder) in the order they were made, without the batch axis."""
	record = as_record(record)
	return [
		({s.name: record.tensors[s.name][i] for s in INPUTS}, {s.name: record.tensors[s.name][i] for s in OUTPUTS}, action)
		for i, action in enumerate(record.actions)
	]


def battle_examples(record, reward_config=RewardConfig(), advantage_config=AdvantageConfig()):
	"""(examples, summary) for one recorded battle (a Record, or a record folder). Each example is a dict as described at the top."""
	record = as_record(record)
	summary = summarize(record.events, config=reward_config)
	decisions = read_decisions(record)
	summary.decisions = len(decisions)
	if not decisions:
		return [], summary

	outputs = {s.name: torch.from_numpy(record.tensors[s.name]) for s in OUTPUTS}
	actions = stack_actions([d[2] for d in decisions])
	old_log_prob = log_prob(outputs, actions)
	values = outputs["value"][:, -1]  # V, the final-reward head (-1..1)
	rewards = torch.zeros(len(decisions))
	rewards[-1] = to_scale(summary.reward, reward_config)
	_, final_advantages = battle_returns_and_advantages(values, rewards, advantage_config)
	rounds = [int(d[0]["g_round"]) for d in decisions]
	labels = value_targets(rounds, summary.round_scores, summary.reward, reward_config=reward_config)
	advantages = policy_advantages(final_advantages, labels, outputs["value"], advantage_config)
	summary.advantage_std = float(advantages.std()) if len(advantages) > 1 else 0.0
	summary.value_error = float((values - rewards[-1]).abs().mean())
	summary.values = values.tolist()
	own = army_values(record.tensors, summary.our_side) / max(summary.own_start_value, 1.0)
	enemy = army_values(record.tensors, 1 - summary.our_side) / max(summary.enemy_start_value, 1.0)
	summary.credit = list(zip(-np.diff(own), -np.diff(enemy), np.diff(values.numpy()).tolist()))

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


def make_dataloader(records, batch_size=64, shuffle=True, reward_config=RewardConfig(), advantage_config=AdvantageConfig()):
	"""(DataLoader over the model's own decisions in the given battle records (Records or record folders), the
	battles' summaries, every battle's examples in order including replayed (forced) ones).

	Advantages are computed over whole battles, then normalised per side we played over the decisions
	trained on, not per batch (losses.normalize_advantages).
	"""
	examples, summaries, lines, sides, battles = [], [], [], [], []
	for i, record in enumerate(records):
		battle, summary = battle_examples(record, reward_config, advantage_config)
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
