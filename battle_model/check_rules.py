"""Play battles in VCMI with the hand-made rules alone, an untrained model.BattleModel whose first-order rule weights
make the rule (everything else as initialised), and check in the records that each rule did what it says:

  kill rule  sure_kill and value_destroyed at 1000 for melee and shot targets: at every decision where the acting unit
             could kill an enemy stack whatever the dice, it attacked the most valuable such stack, and that stack
             was dead at our next decision.

	.venv/bin/python -m battle_model.check_rules [--scenario PATH] [--seeds N] [--opponent BattleAI]

Greedy play (temperature 0, or --temperature) against the opponent, on N seeds per side. The records stay in memory. Exits with 1
if a rule was not followed or a sure kill did not kill.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import play, vocab
from .constants import ActionType
from .export import export
from .model import BattleModel
from .rules import cross_stack_features

ROOT = Path(__file__).resolve().parent.parent
SCENARIO = ROOT / "battle_model" / "scenarios" / "sure_kills.json"


def kill_rule_check(record, tables):
	"""(opportunities, followed, killed, shots, problems) of one record: decisions with a legal sure kill, those where
	the most valuable one was attacked, those where it was dead at our next decision, how many of those attacks were
	shots, and what went wrong."""
	x = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in record.tensors.items()}
	f = cross_stack_features(x, tables)
	score = 1 + f["value_share"]
	legal = {ActionType.MELEE: f["kills_melee"].bool() & x["m_melee_target"], ActionType.SHOOT: f["kills_shot"].bool() & x["m_shoot_target"]}
	opportunities = followed = killed = shots = 0
	problems = []
	for t, action in enumerate(record.actions):
		best = max(float((score[t] * legal[kind][t]).max()) for kind in legal)
		if best <= 0:
			continue
		opportunities += 1
		kind, unit = ActionType(action["type"]), action["unit"]
		units = lambda: " ".join(f"{u}:{vocab.name('creatures', int(x['u_creature'][t, u]))}x{int(x['u_count'][t, u])}{'(sure)' if any(bool(legal[k][t, u]) for k in legal) else ''}" for u in range(x["u_count"].shape[1]) if x["u_present"][t, u] and x["u_alive"][t, u])
		if kind not in legal or not bool(legal[kind][t, unit]) or float(score[t, unit]) < best - 1e-6:
			problems.append(f"decision {t}: played {kind.name} on slot {unit}, best sure kill scores {best:.3f}; units {units()}")
			continue
		followed += 1
		shots += kind == ActionType.SHOOT
		if t + 1 < len(record.actions):
			if x["u_alive"][t + 1, unit]:
				problems.append(f"decision {t}: {kind.name} on slot {unit} ({int(x['u_count'][t, unit])} x {vocab.name('creatures', int(x['u_creature'][t, unit]))}, {int(f['total_hp'][t, unit])} hp; least damage {int(f['least_melee' if kind == ActionType.MELEE else 'least_shot'][t, unit])}) did not kill it")
			else:
				killed += 1
		else:  # our last decision: at the end of the battle, that side must have lost at least that many of the creature
			end = next(e for e in record.events if e["event"] == "end")
			side, creature = int(x["u_side"][t, unit]), int(x["u_creature"][t, unit])
			before = int(sum(x["u_count"][t, u] for u in range(x["u_count"].shape[1]) if x["u_present"][t, u] and x["u_alive"][t, u] and int(x["u_side"][t, u]) == side and int(x["u_creature"][t, u]) == creature))
			after = sum(n for c, n in end["alive"][side] if c == creature)
			if after > before - int(x["u_count"][t, unit]):
				problems.append(f"decision {t} (the last): {kind.name} on slot {unit} did not kill it ({after} of {before} {vocab.name('creatures', creature)} alive at the end)")
			else:
				killed += 1
	return opportunities, followed, killed, shots, problems


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--scenario", default=str(SCENARIO))
	parser.add_argument("--seeds", type=int, default=8, help="battles per side, each on its own VCMI seed")
	parser.add_argument("--opponent", default="BattleAI")
	parser.add_argument("--model-seed", type=int, default=0, help="torch seed of the untrained weights")
	parser.add_argument("--temperature", type=float, default=0.0, help="0: greedy; 1: sampled")
	args = parser.parse_args()

	torch.manual_seed(args.model_seed)
	model = BattleModel().eval()
	for head in ("melee_target", "shoot_target"):
		model.set_rule_weights(head, sure_kill=1000.0, value_destroyed=1000.0)
	work = Path(tempfile.mkdtemp(prefix="check-rules-", dir=ROOT / "runtime"))
	battle_map = play.build_map(args.scenario, variant="check-rules")
	try:
		model_path = export(model, work / "rules.onnx")
		games = [play.Game(args.scenario, our_side=side, opponent=args.opponent, game_seed=1000 + s, sampler_seed=s, temperature=args.temperature, map=battle_map) for side in (0, 1) for s in range(args.seeds)]
		records = []
		for i in range(0, len(games), 8):
			records += play.play(games[i : i + 8], model_path)
	finally:
		shutil.rmtree(work, ignore_errors=True)
		(play.USER_DIR / (battle_map + ".vmap")).unlink(missing_ok=True)

	tables = model.tables()
	totals = np.zeros(4, dtype=int)
	decisions = 0
	failed = False
	for game, record in zip(games, records):
		*counts, problems = kill_rule_check(record, tables)
		totals += counts
		decisions += record.decisions
		for problem in problems:
			failed = True
			print(f"side {game.our_side} seed {game.game_seed}: {problem}")
	opportunities, followed, killed, shots = totals
	print(f"{len(records)} battles, {decisions} decisions: {opportunities} with a sure kill; the most valuable attacked in {followed} ({shots} shots), dead afterwards in {killed}")
	sys.exit(1 if failed or followed < opportunities or killed < followed else 0)


if __name__ == "__main__":
	main()
