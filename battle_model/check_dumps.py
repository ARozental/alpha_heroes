"""Replay decisions dumped by the C++ AlphaBattleAI through the model in Python.

	.venv/bin/python -m battle_model.check_dumps <dump dir> [--model PATH]

For every decision_* directory (written when ALPHA_BATTLE_DUMP_DIR is set) it checks that:
  - every input/output tensor has the contract's dtype and shape,
  - the masks are consistent (each legal choice has legal follow-up choices, reserved types are off),
  - onnxruntime in Python produces the same outputs from the same inputs as the C++ side did,
  - the action the C++ side executed is legal under the masks and has finite log-probability.
Exits non-zero on any failure.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .constants import RESERVED_ACTION_TYPES, SECOND_TARGET_KIND, ActionType, TargetKind
from .distributions import is_legal, log_prob, stack_actions
from .export import DEFAULT_PATH
from .schema import INPUT_NAMES, INPUTS, OUTPUT_NAMES, OUTPUTS

DTYPES = {"int64": np.int64, "bool": np.bool_, "float32": np.float32}


def load_decision(path):
	tensors = {}
	for spec in INPUTS + OUTPUTS:
		array = np.load(path / f"{spec.name}.npy")
		if array.dtype != DTYPES[spec.dtype] or array.shape != (1,) + spec.shape:
			raise AssertionError(f"{spec.name}: dtype {array.dtype} shape {array.shape}, contract says {spec.dtype} {(1,) + spec.shape}")
		tensors[spec.name] = array
	action = json.loads((path / "action.json").read_text())
	return tensors, action


def mask_problems(x):
	m = {k: v[0] for k, v in x.items() if k.startswith("m_")}
	t = m["m_type"]
	problems = []

	def expect(condition, message):
		if not condition:
			problems.append(message)

	expect(not t[[int(a) for a in RESERVED_ACTION_TYPES]].any(), "a reserved action type is legal")
	expect(t[ActionType.MOVE] == m["m_move_hex"].any(), "MOVE legality disagrees with m_move_hex")
	expect(t[ActionType.MELEE] == m["m_melee_target"].any(), "MELEE legality disagrees with m_melee_target")
	expect((m["m_melee_target"] == m["m_melee_from"].any(axis=1)).all(), "melee target without hexes to attack from (or vice versa)")
	expect(t[ActionType.SHOOT] == m["m_shoot_target"].any(), "SHOOT legality disagrees with m_shoot_target")
	expect(t[ActionType.HERO_SPELL] == (bool(x["g_hero_can_cast"][0]) and m["m_spell"].any()), "HERO_SPELL legality disagrees with m_spell")
	for s in range(m["m_spell"].shape[0]):
		units, hexes = m["m_spell_unit"][s], m["m_spell_hex"][s]
		expect(not (units.any() and hexes.any()), f"spell {s} has both unit and hex targets")
		if (units.any() or hexes.any()) and not m["m_spell"][s]:
			problems.append(f"spell {s} has targets but is not castable")
		second = SECOND_TARGET_KIND.get(s)
		for u in np.flatnonzero(units):
			if second == TargetKind.HEX:
				expect(m["m_spell2_hex"][u].any(), f"spell {s} target {u} has no second hex")
			elif second == TargetKind.UNIT:
				expect(m["m_spell2_unit"][u].any(), f"spell {s} target {u} has no second unit")
	active = int(x["g_active_unit"][0])
	expect(x["u_present"][0, active] == 1, "active unit slot is empty")
	return problems


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("dump_dir")
	parser.add_argument("--model", default=str(DEFAULT_PATH))
	args = parser.parse_args()

	session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
	decisions = sorted(Path(args.dump_dir).glob("decision_*"))
	if not decisions:
		sys.exit(f"no decision_* directories in {args.dump_dir}")

	failures, worst, types = [], 0.0, Counter()
	for path in decisions:
		try:
			x, action = load_decision(path)
			for problem in mask_problems(x):
				failures.append(f"{path.name}: {problem}")
			replay = dict(zip(OUTPUT_NAMES, session.run(list(OUTPUT_NAMES), {k: x[k] for k in INPUT_NAMES})))
			for spec in OUTPUTS:
				cpp, py = x[spec.name], replay[spec.name]
				if not np.array_equal(np.isneginf(cpp), np.isneginf(py)):
					failures.append(f"{path.name}: {spec.name}: illegal entries differ between C++ and Python")
					continue
				finite = np.isfinite(py)
				worst = max(worst, float(np.abs(cpp[finite] - py[finite]).max(initial=0.0)))
			if not is_legal(action, {k: x[k] for k in x if k.startswith("m_")}):
				failures.append(f"{path.name}: executed action {action} is illegal under the masks")
			lp = log_prob({k: torch.from_numpy(replay[k]) for k in OUTPUT_NAMES}, stack_actions([action]))
			if not torch.isfinite(lp).all():
				failures.append(f"{path.name}: executed action has log-probability {float(lp[0])}")
			types[ActionType(action["type"]).name] += 1
		except Exception as error:  # report every broken decision, not just the first
			failures.append(f"{path.name}: {error}")

	print(f"decisions checked: {len(decisions)}, max |C++ - Python| output difference: {worst:.2e}")
	print("action types:", dict(types))
	if worst > 1e-4:
		failures.append(f"outputs differ by {worst}")
	for failure in failures:
		print("FAIL", failure)
	sys.exit(1 if failures else 0)


if __name__ == "__main__":
	main()
