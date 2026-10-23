"""Check stupid_ai.StupidAI against VCMI's StupidAI itself.

	.venv/bin/python -m battle_model.check_stupid_ai [--seeds N] [--opponents StupidAI,BattleAI]

1. Decisions: StupidAI plays (our C++ AI delegates to it, ALPHA_BATTLE_DELEGATE, and records each state with the action
   StupidAI took); the PyTorch model, and the exported ONNX model as the C++ AI ran it (the records' outputs), must
   choose the same action in every recorded state.
2. Battles: the exported model plays the same battles itself (greedy); every battle must end exactly as StupidAI's.

Both scenarios, both sides, against each opponent, on N seeds. The records stay in memory. Exits with 1 on a mismatch.
"""
import argparse
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from . import play, vocab
from .constants import ActionType
from .distributions import sample
from .export import export, verify
from .stupid_ai import StupidAI

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = [ROOT / "battle_model" / "scenarios" / name for name in ("castle_vs_inferno.json", "sure_kills.json")]
FIELDS = {ActionType.MOVE: ("hex",), ActionType.MELEE: ("unit", "hex"), ActionType.SHOOT: ("unit",)}


def same(a, b):
	kind = ActionType(a["type"])
	return a["type"] == b["type"] and all(a[f] == b[f] for f in FIELDS.get(kind, ()))


def starting_counts(record, x):
	"""u_count's shape: each unit's number of creatures at the start of the battle, from the record's start event
	(by side and creature; the scenarios have each creature once per side), else its current number."""
	start = next(e for e in record.events if e["event"] == "start")
	sizes = {(side, creature): count for side in (0, 1) for creature, count in start["armies"][side]}
	base = x["u_count"].clone()
	for index in np.ndindex(*base.shape):
		key = (int(x["u_side"][index]), int(x["u_creature"][index]))
		if x["u_present"][index] and key in sizes:
			base[index] = sizes[key]
	return base


def describe(record, t, action):
	x = {k: v[t] for k, v in record.tensors.items()}
	unit = lambda u: f"{vocab.name('creatures', int(x['u_creature'][u]))}x{int(x['u_count'][u])}@{int(x['u_hex'][u])}"
	kind = ActionType(action["type"])
	return f"{kind.name}" + (f" {unit(action['unit'])}" if action["unit"] >= 0 else "") + (f" hex {action['hex']}" if action["hex"] >= 0 else "")


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--seeds", type=int, default=6)
	parser.add_argument("--opponents", default="StupidAI,BattleAI")
	args = parser.parse_args()

	model = StupidAI().eval()
	work = Path(tempfile.mkdtemp(prefix="check-stupid-ai-", dir=ROOT / "runtime"))
	maps = {}
	try:
		model_path = export(model, work / "stupid_ai.onnx")
		verify(model, model_path)
		games = []
		for scenario in SCENARIOS:
			maps[scenario] = play.build_map(scenario, variant="check-stupid-ai")
			for opponent in args.opponents.split(","):
				games += [play.Game(str(scenario), our_side=side, opponent=opponent, game_seed=2000 + s, temperature=0.0, map=maps[scenario]) for side in (0, 1) for s in range(args.seeds)]
		delegated = [g.__class__(**{**g.__dict__, "settings": (("ALPHA_BATTLE_DELEGATE", "StupidAI"),)}) for g in games]
		records, own = [], []
		for i in range(0, len(games), 8):
			records += play.play(delegated[i : i + 8], model_path)
			own += play.play(games[i : i + 8], model_path)
	finally:
		shutil.rmtree(work, ignore_errors=True)
		for battle_map in maps.values():
			(play.USER_DIR / (battle_map + ".vmap")).unlink(missing_ok=True)

	decisions, torch_agree, onnx_agree, base_agree, kinds, problems = 0, 0, 0, 0, Counter(), []
	for game, record in zip(games, records):
		x = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in record.tensors.items() if not k.startswith("logp_") and k != "value"}
		with torch.no_grad():
			out = model(x)
			based = model({**x, "u_base_count": starting_counts(record, x)})
		for t, played in enumerate(record.actions):
			decisions += 1
			kinds[ActionType(played["type"]).name] += 1
			ours = sample(out, b=t, greedy=True)
			onnx = sample({k: v for k, v in record.tensors.items() if k.startswith("logp_")}, b=t, greedy=True)
			torch_agree += same(ours, played)
			onnx_agree += same(onnx, played)
			base_agree += same(sample(based, b=t, greedy=True), played)
			if not same(ours, played) or not same(onnx, played):
				problems.append(f"{Path(game.scenario).stem} side {game.our_side} vs {game.opponent} seed {game.game_seed} decision {t}: StupidAI {describe(record, t, played)}, model {describe(record, t, ours)}, ONNX {describe(record, t, onnx)}")
	identical = sum(r.events[-1] == o.events[-1] and r.decisions == o.decisions for r, o in zip(records, own))
	for problem in problems[:30]:
		print(problem)
	print(f"decisions: {decisions} recorded from StupidAI ({dict(kinds)}); the PyTorch model chose the same in {torch_agree}, the ONNX model in {onnx_agree}")
	print(f"           given the stacks' starting sizes (not a model input): the same in {base_agree}")
	print(f"battles: {len(games)}; the exported model's own battles ended exactly as StupidAI's in {identical}")
	sys.exit(0 if torch_agree == onnx_agree == decisions and identical == len(games) else 1)


if __name__ == "__main__":
	main()
