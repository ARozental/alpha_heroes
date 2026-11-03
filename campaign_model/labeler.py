"""The campaign labels computed as each game arrives, in processes of their own, so the trainer's thread (and the GPU
with it) does not stop for them: on the PC they took ~27 s of each campaign update's ~140 s (2026-10-30), the GPU idle.

What a game needs that no model forward gives (learner.step computed it there, in order): over its usable decisions
(train.usable), the real shares, the label mix and the price (labels.label), the growth labels (labels.growth_labels),
the end labels and steps (labels.end_labels, end_steps), and, for the v2 model, the foresight's labels
(foresight.labels_of) with the features module's weights as they were when the game arrived: its priors' corrections
move slowly (CampaignConfig.prior_decay), so a label from one update before is the same label to within noise.
What needs the model as it is at the update (day_starts, the ladder's grades, the credit) stays in learner.step.

	Labeler(features)        the processes (LABEL_WORKERS)
	.submit(game, ...)       a Future of precompute's result (or None: a game with nothing to train on)
	precompute(...)          the work itself, in a labeler process
"""
import concurrent.futures
import copy

import torch

from battle_model import forkserver

from .empire import EmpireConfig
from .labels import LabelConfig, end_labels, end_steps, growth_labels, label, share_change_labels

LABEL_WORKERS = 2  # a game's labels take ~1 s of a core; ~15 games arrive a minute on the PC
_features = None  # a labeler process's features module (model.CustomFeatures), the weights loaded per game


def _features_module(state):
	global _features
	if _features is None:
		from .model import CustomFeatures

		_features = CustomFeatures().eval()
	_features.load_state_dict(state)
	return _features


def precompute(game, features_state, labels=LabelConfig(), empire=EmpireConfig(), foresight=True, share_target="share"):
	"""The labels of game (records.Game, every decision as recorded; or the path of its file in the learner's spool,
	read here: the trainer keeps no inputs of a waiting game), over its usable decisions, as learner.step takes them;
	None when it has none or no ledger."""
	from .foresight import labels_of

	if isinstance(game, str):
		import pickle

		with open(game, "rb") as f:
			game = pickle.load(f)[0]
	from .train import usable

	full = game.decisions
	kept = usable(full)
	if not kept or not game.ledgers:
		return None
	g = copy.copy(game)
	g.decisions, g.all_decisions = kept, full  # (labels_of: the battles' serials count every record; the kept rows come back)
	t, mixed, price = label(g, labels, empire)
	if share_target == "change":
		t = share_change_labels(g, empire)
	step, week = end_steps(g)
	out = {"n": len(kept), "t": t, "mixed": mixed, "price": price, "z": growth_labels(g, empire), "real": end_labels(g, empire), "step": step, "week": week}
	if foresight:
		with torch.no_grad():
			out["foresight"] = labels_of(g, _features_module(features_state))
	return out


class Labeler:
	def __init__(self, labels=LabelConfig(), empire=EmpireConfig(), foresight=True, workers=LABEL_WORKERS, share_target="share"):
		self.labels, self.empire, self.foresight, self.share_target = labels, empire, foresight, share_target
		# (the run's forkserver, its preload set: this pool may be what starts it, at a resume's labels of the spooled games)
		self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=forkserver.context())

	def submit(self, game, features):
		"""A Future of precompute(game) (a records.Game, or its spool file's path) with the features module's weights as
		they are now (copied: the learner refreshes them in place after each update)."""
		state = {key: value.detach().clone() for key, value in features.state_dict().items()} if self.foresight else None
		return self.pool.submit(precompute, game, state, self.labels, self.empire, self.foresight, self.share_target)

	def close(self):
		self.pool.shutdown(wait=False, cancel_futures=True)
