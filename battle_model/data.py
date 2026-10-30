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
	foresight     what happened between it and our next decision, model.Foresight's targets (foresight_labels);
	              None for the battle's last decision
and per battle a BattleSummary with the reward (reward.py) and the outcome.

A sent record is ~190 KB a decision, in memory only. Keeping data across epochs will want a compact format
(sparse masks, ~115 bytes a decision).
"""
import copy
import io
import json
import re
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import vocab
from .distributions import ACTION_FIELDS, is_legal, log_prob, stack_actions
from .losses import AdvantageConfig, battle_returns_and_advantages, normalize_advantages, policy_advantages, value_targets
from .reward import WAR_MACHINE_CREATURES, Outcome, RewardConfig, assets_after, hero_asset, mana_value, round_score, terminal_reward, to_scale
from .constants import MAX_UNITS, N_HEXES
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


def summarize(events, config=RewardConfig(), side=None):
	"""The reward and outcome of one battle, from our side's events (reward.py: zero-sum, from each side's assets).

	Heroes: their level and artifacts from the start event ("heroes": true values, the reward is no input; 0 for records
	without them and sides without a hero); the winner over a hero takes its artifacts. side: score the battle for this
	battle side instead of the recording one (e.g. to check the C++ reward, Reward.h, of both sides: the end event's
	cpp_rewards).
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
	hero_facts = start.get("heroes", [[0.0, 0.0], [0.0, 0.0]])  # per battle side: [level and its way to the next, artifacts' price]
	heroes = {s: hero_asset(hero_facts[s][0], hero_facts[s][1], config) for s in (our, enemy)}

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
	# War machines: part of their hero (half price; the catapult and arrow towers 0: game_values), kept unless the side
	# loses; the engine's casualties count those destroyed, which no army's start counts (records before 2026-10-28 have
	# no "machines": none counted, and their casualties no longer taken off the army: a destroyed ballista was -2500)
	machine = lambda stacks: [[c, n] for c, n in stacks if c in WAR_MACHINE_CREATURES]  # noqa: E731
	troops = lambda stacks: [[c, n] for c, n in stacks if c not in WAR_MACHINE_CREATURES]  # noqa: E731
	machines = start.get("machines", [[], []])
	machines_start = {s: army_value([[c, 1] for c in machines[s]]) for s in (our, enemy)}
	machines_end = {s: max(0.0, machines_start[s] - army_value(machine(end["casualties"][s]))) for s in (our, enemy)}
	army_end = {s: army_start[s] - army_value(troops(end["casualties"][s])) + army_value(summoned[s]) for s in (our, enemy)}
	assets_start = {s: army_start[s] + heroes[s] + machines_start[s] + mana(start, s) for s in (our, enemy)}
	cost = {s: surrender["cost"] if surrender and s == recorder else 0.0 for s in (our, enemy)}  # only the recorder's is known
	captured = {s: hero_facts[1 - s][1] if outcome_of(s) == Outcome.WIN and outcome_of(1 - s) == Outcome.LOSS else 0.0 for s in (our, enemy)}
	assets_end = {s: assets_after(outcome_of(s), army_end[s], heroes[s] + machines_end[s], mana(end, s), cost[s], captured[s]) for s in (our, enemy)}
	reward = terminal_reward(assets_start[our], assets_end[our], assets_start[enemy], assets_end[enemy])

	scores = {}
	for e in events:
		if e["event"] == "round_end" and e["round"] >= 1:  # round 0 is the start of the battle
			now = {s: army_value(e["alive"][s]) + heroes[s] + machines_start[s] + mana(e, s) for s in (our, enemy)}  # (machines lost on the way: not known)
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
	tensors = {s.name: compact(npy_view(entries[s.name + ".npy"])) for s in INPUTS + OUTPUTS}
	return Record(lines("events.jsonl"), lines("actions.jsonl"), tensors)


NPY_HEADER = re.compile(rb"'descr': *'([^']+)', *'fortran_order': *False, *'shape': *\(([^)]*)\)")


def npy_view(entry):
	"""An .npy file's array as a read-only view of its bytes (np.load copied every record's ~0.4 MB a decision twice):
	format 1.0, C order, as AlphaBattleAI writes them (Dump.cpp); anything else through np.load."""
	if bytes(entry[:8]) == b"\x93NUMPY\x01\x00":
		(length,) = struct.unpack_from("<H", entry, 8)
		match = NPY_HEADER.search(bytes(entry[10 : 10 + length]))
		if match:
			shape = tuple(int(d) for d in match.group(2).split(b",") if d.strip())
			return np.frombuffer(entry, dtype=np.dtype(match.group(1).decode()), count=int(np.prod(shape, dtype=np.int64)), offset=10 + length).reshape(shape)
	return np.load(io.BytesIO(entry))


def compact(array):
	"""An int64 array as int32 when its values fit (all inputs do: ~630k at most): records wait in the actors' buffer
	by the thousand, at ~675 KB a decision in int64, and the machine swapped (collate widens them back). AlphaBattleAI
	sends them as int32 since 2026-10-29: those pass as they are."""
	if array.dtype == np.int64 and array.size and np.abs(array).max() < 2**31:
		return array.astype(np.int32)
	return array


class Sparse:
	"""A large, mostly zero array kept as its nonzero entries (compact_record): their flat positions, and their values
	unless it is a mask."""

	__slots__ = ("shape", "dtype", "index", "values")

	def __init__(self, array):
		self.shape, self.dtype = array.shape, array.dtype
		flat = array.reshape(-1)
		self.index = np.flatnonzero(flat).astype(np.int32 if flat.size < 2**31 else np.int64)
		self.values = None if array.dtype == np.bool_ else flat[self.index]

	def dense(self):
		out = np.zeros(self.shape, self.dtype)
		out.reshape(-1)[self.index] = True if self.values is None else self.values
		return out

	@property
	def nbytes(self):
		return self.index.nbytes + (self.values.nbytes if self.values is not None else 0)


# The inputs kept sparse in the actors' buffer: the per-spell and per-unit tables over the hexes and units (rows of
# spells the hero lacks, of absent units, of hexes nothing can be done from: mostly zero)
SPARSE_INPUTS = tuple(s.name for s in INPUTS if int(np.prod(s.shape)) >= 1000)


def compact_record(record):
	"""The record as the actors' buffer keeps it: of the policy's outputs only the chosen actions' log-probabilities
	(tensors["old_log_prob"], [n]: all PPO needs of them) and the value heads, and the large input tables as Sparse
	(expand_record makes them dense again for an update). A record is ~0.4 MB a decision in full, over half of it in
	tables that are mostly zero, and the buffer held 16k decisions (~6.5 GB) on a 26 GB machine that swapped.
	Changes the record in place and returns it; check_record it before (it needs every output)."""
	tensors = dict(record.tensors)
	with warnings.catch_warnings():  # (parse_record's read-only views: only read here)
		warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
		logp = {s.name: torch.from_numpy(np.ascontiguousarray(tensors.pop(s.name))) for s in OUTPUTS if s.name.startswith("logp_")}
	if logp:
		tensors["old_log_prob"] = log_prob(logp, stack_actions(record.actions)).numpy() if record.actions else np.zeros(0, np.float32)
	for name in SPARSE_INPUTS:
		if isinstance(tensors.get(name), np.ndarray):
			tensors[name] = Sparse(tensors[name])
	for name, value in tensors.items():  # (parse_record's arrays are views of the whole message: kept, they would keep it)
		if isinstance(value, np.ndarray) and value.base is not None:
			tensors[name] = value.copy()
	record.tensors = tensors
	return record


def expand_record(record):
	"""A copy of a compacted record with its inputs dense again (its outputs stay compact); a full one as it is."""
	if not any(isinstance(v, Sparse) for v in record.tensors.values()):
		return record
	out = copy.copy(record)  # (with what else it carries: its source)
	out.tensors = {k: v.dense() if isinstance(v, Sparse) else v for k, v in record.tensors.items()}
	return out


def record_bytes(record):
	return sum(v.nbytes for v in record.tensors.values())


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


def check_record(record):
	"""None if every recorded decision is sound, else what is wrong: a decision count that differs between its parts,
	or an action that is out of range or illegal under that decision's own masks. A record that fails is never
	trained on: a corrupt one (an action type of 30570) once stopped a run inside a training step."""
	n = len(record.actions)
	for name, tensor in record.tensors.items():
		if len(tensor) != n:
			return f"{name} has {len(tensor)} decisions, the actions {n}"
	m_type = record.tensors.get("m_type")
	for i, action in enumerate(record.actions):
		kind = action.get("type", -1)
		if not isinstance(kind, int) or not 0 <= kind < m_type.shape[1]:
			return f"decision {i}: action type {kind!r} out of range"
		if not m_type[i, kind]:
			return f"decision {i}: action type {kind} not legal under its masks"
		try:
			legal = is_legal(action, {k: np.asarray(v)[i : i + 1] for k, v in record.tensors.items() if k.startswith("m_")})
		except (IndexError, KeyError, TypeError):  # a field out of its table
			legal = False
		if not legal:
			return f"decision {i}: action {action} not legal under its masks"
	return None


def read_decisions(record):
	"""[(inputs, outputs, action)] of a Record (or a record folder) in the order they were made, without the batch axis."""
	record = as_record(record)
	return [
		({s.name: record.tensors[s.name][i] for s in INPUTS}, {s.name: record.tensors[s.name][i] for s in OUTPUTS if s.name in record.tensors}, action)
		for i, action in enumerate(record.actions)
	]


MIN_WEIGHT, FULL_WEIGHT_SHARE = 0.05, 0.1


def unit_health(inputs):
	"""Each unit slot's hit points [U]: (count - 1) x max HP + the top creature's HP; 0 for an empty or dead slot."""
	count = inputs["u_count"].astype(np.float64)
	alive = (inputs["u_present"] != 0) & (inputs["u_alive"] != 0) & (count > 0)
	return np.where(alive, (count - 1) * inputs["u_max_hp"] + inputs["u_first_hp"], 0.0)


def foresight_labels(decisions):
	"""model.Foresight's targets for each decision of one side's battle: what happened between it and the side's next
	decision (our action, then the enemy's reply). Per unit slot alive then (and the same creature next: the slots are
	VCMI's unit ids in order, kept for the dead, so a slot is one unit throughout): "dies", "lost" (the share of its hit
	points it lost), "dealt" (the hit points it dealt over the other side's then), each 0..1; "unit_mask"; per hex,
	"hex_enemy": an enemy unit, alive, stands on it at the next decision. None for the last decision (no next one)."""
	labels = []
	for i, (now, _, _) in enumerate(decisions):
		if i + 1 >= len(decisions):
			labels.append(None)
			break
		after = decisions[i + 1][0]
		before, later = unit_health(now), unit_health(after)
		alive = before > 0
		mask = alive & (after["u_present"] != 0) & (after["u_creature"] == now["u_creature"])
		side = now["u_owner_side"]
		others = np.array([before[(side != s) & alive].sum() for s in side])  # the other side's hit points, per unit
		dealt = (after["u_dealt"].astype(np.float64) - now["u_dealt"]) / np.maximum(others, 1.0)
		enemy = (after["u_present"] != 0) & (after["u_alive"] != 0) & (after["u_owner_side"] != now["g_our_side"])
		hexes = np.zeros(N_HEXES, dtype=np.float32)
		for hex_input in ("u_hex", "u_rear_hex"):
			cells = after[hex_input][enemy]
			hexes[cells[(cells >= 0) & (cells < N_HEXES)]] = 1.0
		labels.append({
			"dies": (mask & (later <= 0)).astype(np.float32),
			"lost": np.where(mask, np.clip(1.0 - later / np.maximum(before, 1.0), 0.0, 1.0), 0.0).astype(np.float32),
			"dealt": np.where(mask, np.clip(dealt, 0.0, 1.0), 0.0).astype(np.float32),
			"unit_mask": mask.astype(np.float32),
			"hex_enemy": hexes,
		})
	return labels


def example_weight(summary):
	"""How much a battle's decisions count in training: its smaller side's share of both sides' assets over
	FULL_WEIGHT_SHARE, in MIN_WEIGHT..1 (an even fight 1; a 100k hero crushing 2k of monsters 0.2: its outcome was never
	in doubt, so its decisions teach little and would flood the data, as real games' battles mostly are).

	TODO(stake): an initial guess. Weigh by what the battle risks of the player's whole empire (the value of the
	forces fighting over all of the player's assets) once the campaign AI computes the empire's value."""
	own, enemy = max(summary.own_start_value, 0.0), max(summary.enemy_start_value, 0.0)
	share = min(own, enemy) / max(own + enemy, 1.0)
	return min(1.0, max(MIN_WEIGHT, share / FULL_WEIGHT_SHARE))


def battle_examples(record, reward_config=RewardConfig(), advantage_config=AdvantageConfig()):
	"""(examples, summary) for one recorded battle (a Record, or a record folder). Each example is a dict as described at the top."""
	record = as_record(record)
	summary = summarize(record.events, config=reward_config)
	decisions = read_decisions(record)
	summary.decisions = len(decisions)
	if not decisions:
		return [], summary

	with warnings.catch_warnings():  # (a full record's outputs: parse_record's read-only views, only read here)
		warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
		outputs = {s.name: torch.from_numpy(record.tensors[s.name]) for s in OUTPUTS if s.name in record.tensors}
	actions = stack_actions([d[2] for d in decisions])
	old_log_prob = torch.from_numpy(record.tensors["old_log_prob"]) if "old_log_prob" in record.tensors else log_prob(outputs, actions)  # (compact_record's)
	values = outputs["value"][:, -1]  # V, the final-reward head (-1..1)
	rewards = torch.zeros(len(decisions))
	rewards[-1] = to_scale(summary.reward, reward_config)
	rounds = [int(d[0]["g_round"]) for d in decisions]
	_, final_advantages = battle_returns_and_advantages(values, rewards, advantage_config, rounds)
	labels = value_targets(rounds, summary.round_scores, summary.reward)
	advantages = policy_advantages(final_advantages, labels, outputs["value"], advantage_config)
	summary.advantage_std = float(advantages.std()) if len(advantages) > 1 else 0.0
	summary.value_error = float((values - rewards[-1]).abs().mean())
	summary.values = values.tolist()
	own = army_values(record.tensors, summary.our_side) / max(summary.own_start_value, 1.0)
	enemy = army_values(record.tensors, 1 - summary.our_side) / max(summary.enemy_start_value, 1.0)
	summary.credit = list(zip(-np.diff(own), -np.diff(enemy), np.diff(values.numpy()).tolist()))

	searches = {e["decision"]: (torch.tensor(e["actions"], dtype=torch.long).view(-1, len(ACTION_FIELDS)), torch.tensor(e["policy"], dtype=torch.float32)) for e in record.events if e["event"] == "search"}
	weight = example_weight(summary)
	summary.weight = weight
	foresight = foresight_labels(decisions)
	examples = [
		{"inputs": d[0], "action": d[2], "old_log_prob": old_log_prob[i], "advantage": advantages[i], "value_labels": labels[i], "weight": torch.tensor(weight, dtype=torch.float32), "forced": bool(d[2].get("forced", False)), "search": searches.get(i), "foresight": foresight[i]}
		for i, d in enumerate(decisions)
	]
	return examples, summary


def collate(examples, widen=True):
	"""List of examples -> one batch: inputs as model-ready tensors, the rest stacked. When any example has a search
	policy: "search" = {"action": {field: [B, K]}, "probs": [B, K]}, padded to the longest with probability 0 (and
	all fields -1); an example without one has all probabilities 0. widen=False: the int64 inputs as the records keep
	them (int32: half the bytes to stack and to move to the GPU), for widened() on the device, a minibatch at a time."""
	stack = lambda s: np.stack([e["inputs"][s.name] for e in examples])  # noqa: E731
	batch = {
		"inputs": {s.name: torch.from_numpy(stack(s).astype(np.int64) if s.dtype == "int64" and widen else stack(s)) for s in INPUTS},
		"action": {f: torch.tensor([e["action"][f] for e in examples], dtype=torch.long) for f in ACTION_FIELDS},
		"old_log_prob": torch.stack([e["old_log_prob"] for e in examples]),
		"advantage": torch.stack([e["advantage"] for e in examples]),
		"value_labels": torch.stack([e["value_labels"] for e in examples]),
		"weight": torch.stack([e.get("weight", torch.tensor(1.0)) for e in examples]),
	}
	if any(e.get("foresight") is not None for e in examples):  # (the battle's last decision has none: valid 0)
		empty = {"dies": np.zeros(MAX_UNITS, np.float32), "lost": np.zeros(MAX_UNITS, np.float32), "dealt": np.zeros(MAX_UNITS, np.float32), "unit_mask": np.zeros(MAX_UNITS, np.float32), "hex_enemy": np.zeros(N_HEXES, np.float32)}
		rows = [e.get("foresight") or empty for e in examples]
		batch["foresight"] = {key: torch.from_numpy(np.stack([r[key] for r in rows])) for key in empty}
		batch["foresight"]["valid"] = torch.tensor([e.get("foresight") is not None for e in examples], dtype=torch.float32)
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


INT64_INPUTS = frozenset(s.name for s in INPUTS if s.dtype == "int64")


def widened(inputs):
	"""collate(widen=False)'s inputs (or a selection of them) as the model takes them: the int64 ones int64 again."""
	return {name: value.long() if name in INT64_INPUTS and value.dtype != torch.int64 else value for name, value in inputs.items()}


def make_dataloader(records, batch_size=64, shuffle=True, reward_config=RewardConfig(), advantage_config=AdvantageConfig()):
	"""(DataLoader over the model's own decisions in the given battle records (Records or record folders), the
	battles' summaries, every battle's examples in order including replayed (forced) ones).

	Advantages are computed over whole battles, then normalised per side we played and per source over the decisions
	trained on, not per batch (losses.normalize_advantages). The source is a record's `source` attribute (actors.py:
	"generated" or "real"; none counts as generated): normalised together, the campaign games' battles (a joint run's
	real battles) got 50% larger advantages than the generated ones (mean |A| 0.94 against 0.62) while their gradient
	was unrelated to the generated battles' (cosine -0.06): they took a larger share of every step than of the data.
	"""
	examples, summaries, lines, sides, battles = [], [], [], [], []
	for i, record in enumerate(records):
		battle, summary = battle_examples(record, reward_config, advantage_config)
		lines.append(battle)
		summaries.append(summary)
		chosen = [e for e in battle if not e["forced"]]
		examples += chosen
		source = 1 if getattr(record, "source", "generated") == "real" else 0
		sides += [summary.our_side + 2 * source] * len(chosen)  # a normalisation group: side x source
		battles += [i] * len(chosen)
	if examples:
		advantages = normalize_advantages(torch.stack([e["advantage"] for e in examples]), torch.tensor(sides), torch.tensor(battles), advantage_config)
		for example, advantage in zip(examples, advantages):
			example["advantage"] = advantage
	loader = torch.utils.data.DataLoader(examples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)
	return loader, summaries, lines
