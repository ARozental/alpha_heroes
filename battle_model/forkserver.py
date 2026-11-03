"""The forkserver context every process pool of a run takes (the loaders' workers: supply.Supply; the labels'
processes: campaign_model/labeler.py). A process has one forkserver, started by the first process any of them starts,
with the preload as set at that moment and never after: on 2026-11-03 the PC's labeler pool started it first (the
spooled games' labels, at a resume) before Supply set the preload, so it preloaded nothing, and each of the 25 loader
workers imported torch and the rest itself (torch alone: ~286 MB of a process's own memory; ~380 MB a worker in all,
9.5 GB), which capped the games in flight."""
import multiprocessing
import sys

PRELOAD = ["battle_model.fork_preload"]


def context():
	"""The forkserver context with fork_preload as its preload (spawn on Windows, which has no forkserver)."""
	if sys.platform == "win32":
		return multiprocessing.get_context("spawn")
	ctx = multiprocessing.get_context("forkserver")
	ctx.set_forkserver_preload(PRELOAD)
	return ctx
