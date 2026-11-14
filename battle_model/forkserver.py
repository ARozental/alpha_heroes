"""The forkserver context every process pool of a run takes (the loaders' workers: supply.Supply; the labels'
processes: campaign_model/labeler.py). A process has one forkserver, started by the first process any of them starts,
with the preload as set at that moment and never after: on 2026-11-03 the PC's labeler pool started it first (the
spooled games' labels, at a resume) before Supply set the preload, so it preloaded nothing, and each of the 25 loader
workers imported torch and the rest itself (torch alone: ~286 MB of a process's own memory; ~380 MB a worker in all,
9.5 GB), which capped the games in flight."""
import multiprocessing
import sys
import threading
import time

PRELOAD = ["battle_model.fork_preload"]


def context():
	"""The forkserver context with fork_preload as its preload (spawn on Windows, which has no forkserver)."""
	if sys.platform == "win32":
		return multiprocessing.get_context("spawn")
	ctx = multiprocessing.get_context("forkserver")
	ctx.set_forkserver_preload(PRELOAD)
	return ctx


TRIM_EVERY = 30.0  # s between a pool worker's malloc_trim (start_trimmer)


def start_trimmer(seconds=TRIM_EVERY):
	"""A pool worker's freed heap pages back to the system every `seconds`, in a daemon thread, without a collection
	(Linux; elsewhere nothing). e5, 2026-11-14: a loader worker's game is freed by the loader's feeder thread after it
	pickled it, which is after the generator's own trim_memory, so idle workers sat at their last game's heap; the labels'
	processes had no trim at all (0.55 and 0.97 GB PSS on the PC). The loaders' workers start it in supply._worker_init,
	the labels' in their pool's initializer."""
	if not sys.platform.startswith("linux"):
		return None
	thread = threading.Thread(target=_trim_every, args=(seconds,), daemon=True, name="worker-trim")
	thread.start()
	return thread


def _trim_every(seconds):
	import ctypes

	libc = ctypes.CDLL("libc.so.6")
	while True:
		time.sleep(seconds)
		libc.malloc_trim(0)
