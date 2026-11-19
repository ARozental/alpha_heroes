"""The critic's Nullkiller2 games as a stream (the user, 2026-11-19: "there shouldn't be a pool at all! the IterableDataset
should hold code to make them on the fly" ... "each game should only be used once"). supply.Games' kind "nk2" plays an
all-Nullkiller2 game (AlphaTeacherAI records every seat's requests as our decisions), slims it (slim) and checks its
decision-history inputs (history_problem) where it played, here or on a remote slot; the learner labels it at its own
empire (critic_store.game_groups: one game-day a group) and puts its groups into an NK2Stream, which hands each group out
exactly once (take: a group leaves the buffer as it is taken; a buffer under its floor gives a short take, never a repeat).
The first heldout_games games are set aside, never trained on, sampled once to heldout_rows trained rows and saved, so a
restart reads the same held-out (its chart one line on one basis)."""
import pickle
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

SLIM_PER_DAY = 2  # decisions kept a player-day, drawn uniformly (the pool's slim_nk2b rule: heroes with movement spent)
# The history gate (2026-11-18: 102 games from an old AlphaMapAI build had every decision-history input at its default and
# poisoned the critic; b1_work/j9/check_slim.py): these must vary within a game, these must not be -1 on every row, and
# every listed column sits inside the ranges our own rows and the pool's showed
MUST_VARY = ("g_last_index", "g_decisions_today")  # (not g_kind: a game can hold turn decisions only)
NOT_ALL_EMPTY = ("g_last_changed", "g_last_type")  # (never -1 on every row; constant 1 is a real game's)
RANGES = {
	"g_last_changed": (-1, 1), "g_last_type": (-1, 20), "g_last_index": (-1, 20000), "g_decisions_today": (0, 300),
	"g_kind": (0, 10), "t_built_today": (0, 1), "x_heroes": (0, 16), "x_take_creature": (-1, 400),
}
MIN_DECISIONS = 20


def slim(game, rng):
	"""The game with its decisions cut to SLIM_PER_DAY drawn uniformly from each player-day's (one where the day has one),
	in the game's order; its ledgers (the labels) and everything else as they were. rng: a random.Random."""
	days = {}
	for d in game.decisions:
		days.setdefault((d.color, d.day), []).append(d)
	picked = {id(d) for group in days.values() for d in rng.sample(group, min(SLIM_PER_DAY, len(group)))}
	game.decisions = [d for d in game.decisions if id(d) in picked]
	if hasattr(game, "all_decisions"):
		game.all_decisions = game.decisions
	return game


def history_problem(game):
	"""Why a game's decision-history inputs cannot be trusted (the reason), or None. Decodes only the columns it reads."""
	from .records import decode_fields

	if len(game.decisions) < MIN_DECISIONS:
		return f"only {len(game.decisions)} decisions"
	x = decode_fields(game.decisions, tuple(sorted(set(MUST_VARY) | set(NOT_ALL_EMPTY) | set(RANGES))))
	for name in MUST_VARY:
		if float(np.asarray(x[name], dtype=np.float64).std()) == 0.0:
			return f"{name} constant ({float(np.asarray(x[name]).ravel()[0]):g})"
	for name in NOT_ALL_EMPTY:
		if bool((np.asarray(x[name], dtype=np.float64) == -1).all()):
			return f"{name} empty (-1) on every row"
	for name, (low, high) in RANGES.items():
		v = np.asarray(x[name], dtype=np.float64)
		if v.size and (v.min() < low or v.max() > high):
			return f"{name} outside [{low}, {high}] ({v.min():g}..{v.max():g})"
	return None


class NK2Stream:
	"""The learner's NK2 games, in memory: put(key, groups) a labelled game as it arrives (the first heldout_games into
	the held-out, the rest into the buffer), take(rows) whole groups drawn at random until rows trained rows, each removed
	from the buffer as it is taken. Thread-safe (the supply's thread puts, the update takes)."""

	def __init__(self, heldout_games=150, heldout_rows=5000, floor_games=125, heldout_path=None, seed=None):
		self.heldout_games, self.heldout_rows, self.floor_games = int(heldout_games), int(heldout_rows), int(floor_games)
		self.heldout_path = Path(heldout_path) if heldout_path else None
		self.rng = np.random.default_rng(seed)
		self.lock = threading.Lock()
		self.buffer, self.in_buffer = [], Counter()  # groups; groups of each game still in the buffer
		self.buffer_rows = 0  # (the buffer's trained rows: room_rows)
		self.pending, self.heldout = {}, None  # the held-out's games while it fills; its groups once fixed
		self.games_seen, self.group_counts = 0, []  # (games put; groups a game, for the floor in groups)
		self.counts = Counter()  # since the last reading: games in, dropped by the history check, held out
		self.read_at = time.time()
		self.last = {}
		if self.heldout_path is not None and self.heldout_path.exists():  # (a restart: the same held-out, never drawn again)
			with open(self.heldout_path, "rb") as f:
				self.heldout = pickle.load(f)
		elif self.heldout_games <= 0:  # (none wanted: every game to the buffer)
			self.heldout = []

	def put(self, key, groups):
		"""A labelled game's groups (critic_store.game_groups'): the held-out's while it fills, else the buffer's."""
		with self.lock:
			self.games_seen += 1
			self.group_counts = (self.group_counts + [len(groups)])[-200:]
			if self.heldout is None:
				self.pending[key] = groups
				self.counts["held_out"] += 1
				if len(self.pending) >= self.heldout_games:
					self._fix_heldout()
				return
			self.buffer.extend(groups)
			self.buffer_rows += sum(int(g["trained"]) for g in groups)
			self.in_buffer[key] += len(groups)
			self.counts["games_in"] += 1

	def dropped(self, reason):
		"""A game the history check refused (counted, never trained on)."""
		with self.lock:
			self.counts["dropped"] += 1
			self.last["dropped_reason"] = str(reason)[:120]

	def _fix_heldout(self):
		"""The held-out from the pending games: whole groups drawn until heldout_rows trained rows, saved (whole or not at
		all) so a restart reads it as it is."""
		every = [g for groups in self.pending.values() for g in groups]
		chosen, rows = [], 0
		for i in self.rng.permutation(len(every)):
			if rows >= self.heldout_rows:
				break
			chosen.append(every[i])
			rows += every[i]["trained"]
		self.heldout, self.pending = chosen, {}
		if self.heldout_path is not None:
			part = self.heldout_path.with_suffix(".part")
			with open(part, "wb") as f:
				pickle.dump(chosen, f, protocol=pickle.HIGHEST_PROTOCOL)
			part.rename(self.heldout_path)

	def room_rows(self):
		"""About the trained rows a take may have now: the buffer's above its floor (floor_games' groups at the mean
		groups a game, each at the buffer's mean trained rows a group)."""
		with self.lock:
			if not self.buffer:
				return 0
			per_game = float(np.mean(self.group_counts)) if self.group_counts else 0.0
			room = len(self.buffer) - int(round(self.floor_games * per_game))
			return int(max(0, room) * self.buffer_rows / len(self.buffer))

	def buffer_games(self):
		"""Distinct games with a group still in the buffer (a game partly taken counts whole)."""
		with self.lock:
			return len(self.in_buffer)

	def buffer_size(self):
		"""The buffer in games: its groups over the mean groups a game (the floor's and the cap's unit: a game partly taken
		counts as the part left, so the cap can never hold the buffer where the floor leaves no whole take)."""
		with self.lock:
			return self._size()

	def _size(self):
		per_game = float(np.mean(self.group_counts)) if self.group_counts else 0.0
		return len(self.buffer) / per_game if per_game > 0 else 0.0

	def take(self, rows, keep_floor=True):
		"""Whole groups drawn uniformly from the buffer until `rows` trained rows (a group's own drawn rows), each removed
		as it is taken: a group is served once and never again. keep_floor: at most what leaves floor_games' groups in
		the buffer (a starved buffer gives a short take, never a repeat)."""
		with self.lock:
			before = round(self._size(), 1)
			per_game = float(np.mean(self.group_counts)) if self.group_counts else 0.0
			room = len(self.buffer) - (int(round(self.floor_games * per_game)) if keep_floor else 0)
			chosen, taken, keep = [], 0, np.ones(len(self.buffer), dtype=bool)
			if room > 0 and rows > 0:
				for i in self.rng.permutation(len(self.buffer)):
					if taken >= rows or len(chosen) >= room:
						break
					chosen.append(self.buffer[i])
					keep[i] = False
					taken += self.buffer[i]["trained"]
			self.buffer_rows -= taken
			for g in chosen:
				self.in_buffer[g["game"]] -= 1
				if self.in_buffer[g["game"]] <= 0:
					del self.in_buffer[g["game"]]
			self.buffer = [g for g, k in zip(self.buffer, keep) if k]
			self.last.update(buffer_games_before=before, buffer_games_after=round(self._size(), 1), rows_taken=taken, rows_wanted=int(rows), groups_taken=len(chosen))
			return chosen

	def reading(self):
		"""The stream since the last reading (the log line's and the charts'): games in, dropped by the history check,
		held out, games a minute, the buffer before and after the last take, its rows taken and wanted; then reset."""
		with self.lock:
			now = time.time()
			minutes = max((now - self.read_at) / 60.0, 1e-6)
			out = {
				"nk2_games_in": self.counts["games_in"], "nk2_dropped": self.counts["dropped"], "nk2_held_out": self.counts["held_out"],
				"nk2_games_per_minute": (self.counts["games_in"] + self.counts["held_out"] + self.counts["dropped"]) / minutes,
				"nk2_buffer_before": self.last.get("buffer_games_before", round(self._size(), 1)), "nk2_buffer_after": self.last.get("buffer_games_after", round(self._size(), 1)),
				"nk2_rows_taken": self.last.get("rows_taken", 0), "nk2_rows_wanted": self.last.get("rows_wanted", 0),
				"nk2_heldout_ready": self.heldout is not None, "nk2_heldout_games_pending": len(self.pending),
			}
			self.counts, self.read_at = Counter(), now
			return out
