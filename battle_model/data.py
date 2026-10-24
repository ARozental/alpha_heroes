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
	search        for a decision the search played (ALPHA_BATTLE_SEARCH_SIMULATIONS): its improved policy over the
	              root actions, (actions [K, len(ACTION_FIELDS)], probabilities [K]), from the record's "search"
	              event; None otherwise (no search, or it failed and the network's action was played)
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
from .reward import Outcome, RewardConfig, assets_after, hero_asset, mana_value, round_score, terminal_reward, to_scale
from .schema import INPUTS, OUTPUTS

# EBattleResult
RESULT_NORMAL, RESULT_ESCAPE, RESULT_SURRENDER = 0, 1, 2


@dataclass
class BattleSummary:
	our_side: int
	outcome: Outcome
	reward: float  # terminal reward, -1..1 (reward.py: zero-sum)
	rounds: int
	own_start_value: float  # assets at the start: army + hero + mana (reward.py)
	enemy_start_value: float
	own_kept: float  # assets after the battle
	enemy_lost: float  # the enemy's assets lost
	mana_used_fraction: float
	round_scores: dict = field(default_factory=dict)
	# Outcome measures that do not depend on the reward's definition (to compare runs across reward changes): the
	# share of our army we keep after the battle, and of the enemy's army it loses
	army_kept: float = 0.0
	army_killed: float = 0.0  # {round: round_score at its end} while the battle went on
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

	The reward is zero-sum (reward.py): killing a whole enemy army is worth its share of both armies' assets, losing
	ours costs our share. But an honest V only moves on what it did not expect: if every battle ends alike, kills that
	were coming anyway move it by ~0. A least-squares fit over steps also mixes kills with the exchanges they come
	in; an intervention (remove part of a stack in the inputs, see V change) measures the sign without that."""
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


def summarize(events, hero_value=0.0, fresh_hero_value=0.0, config=RewardConfig(), side=None):
	"""The reward and outcome of one battle, from our side's events (reward.py: zero-sum, from each side's assets).

	hero_value: our hero's value (reward.hero_value); 0 for level-1 heroes without artifacts, as in the current
	scenarios (not recorded yet), and the enemy hero's is not known: 0. side: score the battle for this battle side
	instead of the recording one (e.g. to check the C++ reward, Reward.h, of both sides: the end event's cpp_rewards).
	The cost of a surrender the enemy made is unknown to us (taken as 0).
	"""
	errors = [e["message"] for e in events if e["event"] == "error"]
	if errors:
		raise RuntimeError("battle record is unusable: " + "; ".join(errors))
	start = next(e for e in events if e["event"] == "start")
	end = next(e for e in events if e["event"] == "end")
	surrender = next((e for e in events if e["event"] == "surrender"), None)  # always the recording side's
	recorder = start["our_side"]
	our = recorder if side is None else side
	enemy = 1 - our
	powers = start.get("powers", [0, 0])  # each battle side's hero's spell power
	heroes = {our: hero_asset(hero_value, fresh_hero_value), enemy: 0.0}

	def mana(event, s):  # the mana value of side s's hero in an event
		return mana_value(event.get("our_mana" if s == recorder else "enemy_mana", 0), powers[s], config)

	def outcome_of(s):
		if end["winner"] not in (0, 1):  # no winner: both armies destroyed
			return Outcome.DRAW
		if end["winner"] == s:
			return Outcome.WIN
		return {RESULT_ESCAPE: Outcome.RETREAT, RESULT_SURRENDER: Outcome.SURRENDER}.get(end["result"], Outcome.LOSS)

	# What each side keeps of its army: its start - the engine's casualties (creatures raised for the battle only count
	# as lost) + units summoned for good (Pit Lords' demons join the army; the casualties count those killed again)
	summoned = end.get("summoned", [[], []])
	army_start = {s: army_value(start["armies"][s]) for s in (our, enemy)}
	army_end = {s: army_start[s] - army_value(end["casualties"][s]) + army_value(summoned[s]) for s in (our, enemy)}
	assets_start = {s: army_start[s] + heroes[s] + mana(start, s) for s in (our, enemy)}
	cost = {s: surrender["cost"] if surrender and s == recorder else 0.0 for s in (our, enemy)}  # only the recorder's is known
	assets_end = {s: assets_after(outcome_of(s), army_end[s], heroes[s], mana(end, s), cost[s]) for s in (our, enemy)}
	reward = terminal_reward(assets_start[our], assets_end[our], assets_start[enemy], assets_end[enemy])

	scores = {}
	for e in events:
		if e["event"] == "round_end" and e["round"] >= 1:  # round 0 is the start of the battle
			now = {s: army_value(e["alive"][s]) + heroes[s] + mana(e, s) for s in (our, enemy)}
			scores[e["round"]] = round_score(assets_start[our], now[our], assets_start[enemy], now[enemy])

	mana_key = "our_mana" if our == recorder else "enemy_mana"
	pool = start.get(mana_key + "_max", 0)
	mana_used = (start.get(mana_key, 0) - end.get(mana_key, 0)) / pool if pool > 0 else 0.0
	keeps_army = lambda s: outcome_of(s) in (Outcome.WIN, Outcome.SURRENDER)
	share = lambda part, whole: part / whole if whole > 0 else 0.0
	army_kept = share(army_end[our] if keeps_army(our) else 0.0, army_start[our])
	army_killed = share(army_start[enemy] - (army_end[enemy] if keeps_army(enemy) else 0.0), army_start[enemy])
	return BattleSummary(our, outcome_of(our), reward, end["round"], assets_start[our], assets_start[enemy], assets_end[our], assets_start[enemy] - assets_end[enemy], mana_used, scores, army_kept=army_kept, army_killed=army_killed)


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
	labels = value_targets(rounds, summary.round_scores, summary.reward)
	advantages = policy_advantages(final_advantages, labels, outputs["value"], advantage_config)
	summary.advantage_std = float(advantages.std()) if len(advantages) > 1 else 0.0
	summary.value_error = float((values - rewards[-1]).abs().mean())
	summary.values = values.tolist()
	own = army_values(record.tensors, summary.our_side) / max(summary.own_start_value, 1.0)
	enemy = army_values(record.tensors, 1 - summary.our_side) / max(summary.enemy_start_value, 1.0)
	summary.credit = list(zip(-np.diff(own), -np.diff(enemy), np.diff(values.numpy()).tolist()))

	searches = {e["decision"]: (torch.tensor(e["actions"], dtype=torch.long).view(-1, len(ACTION_FIELDS)), torch.tensor(e["policy"], dtype=torch.float32)) for e in record.events if e["event"] == "search"}
	examples = [
		{"inputs": d[0], "action": d[2], "old_log_prob": old_log_prob[i], "advantage": advantages[i], "value_labels": labels[i], "forced": bool(d[2].get("forced", False)), "search": searches.get(i)}
		for i, d in enumerate(decisions)
	]
	return examples, summary


def collate(examples):
	"""List of examples -> one batch: inputs as model-ready tensors, the rest stacked. When any example has a search
	policy: "search" = {"action": {field: [B, K]}, "probs": [B, K]}, padded to the longest with probability 0 (and
	all fields -1); an example without one has all probabilities 0."""
	batch = {
		"inputs": {s.name: torch.from_numpy(np.stack([e["inputs"][s.name] for e in examples])) for s in INPUTS},
		"action": {f: torch.tensor([e["action"][f] for e in examples], dtype=torch.long) for f in ACTION_FIELDS},
		"old_log_prob": torch.stack([e["old_log_prob"] for e in examples]),
		"advantage": torch.stack([e["advantage"] for e in examples]),
		"value_labels": torch.stack([e["value_labels"] for e in examples]),
	}
	searches = [e.get("search") for e in examples]
	if any(s is not None for s in searches):
		k = max(len(s[1]) for s in searches if s is not None)
		actions = torch.full((len(examples), k, len(ACTION_FIELDS)), -1, dtype=torch.long)
		probs = torch.zeros(len(examples), k)
		for i, s in enumerate(searches):
			if s is not None:
				actions[i, : len(s[1])], probs[i, : len(s[1])] = s
		batch["search"] = {"action": {f: actions[..., j] for j, f in enumerate(ACTION_FIELDS)}, "probs": probs}
	return batch


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
