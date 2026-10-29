"""A battle record as readable text: what the model did, decision by decision. For debugging what a run learned.

	.venv/bin/python -m battle_model.trace runtime/train/<run>/checkpoint_latest.pt [--side 0|1|self] [--temperature 0]

One line per decision: round, the acting unit, the action (with targets by name), the probability the model gave
it, V (the final-reward head, -1..1), and the enemy army. The end: the outcome and the reward.
"""
import argparse

import numpy as np

from . import vocab
from .constants import ActionType
from .data import summarize
from .reward import to_scale


def describe_action(x, action):
	"""An action of a decision as text, units and spells by name."""
	def unit(slot):
		return f"{vocab.name('creatures', int(x['u_creature'][slot]))} x{int(x['u_count'][slot])} @{int(x['u_hex'][slot])}"

	kind = ActionType(action["type"]).name
	if kind == "MOVE":
		return f"MOVE -> {action['hex']}"
	if kind == "MELEE":
		return f"MELEE {unit(action['unit'])} from {action['hex']}"
	if kind == "SHOOT":
		return f"SHOOT {unit(action['unit'])}"
	if kind == "HERO_SPELL":
		target = unit(action["target_unit"]) if action["target_unit"] >= 0 else (f"hex {action['target_hex']}" if action["target_hex"] >= 0 else "-")
		return f"SPELL {vocab.name('spells', action['spell'])} on {target}"
	if kind == "CREATURE_SPELL":
		target = unit(action["target_unit"]) if action["target_unit"] >= 0 else f"hex {action['target_hex']}"
		spell = vocab.name("spells", action["spell"]) if action["spell"] >= 0 else "random spell"
		return f"CAST {spell} on {target}"
	return kind


def army(x, side):
	"""The living units of one battle side, by name."""
	units = [u for u in range(len(x["u_present"])) if x["u_present"][u] and x["u_alive"][u] and int(x["u_side"][u]) == side]
	return " ".join(f"{vocab.name('creatures', int(x['u_creature'][u]))}x{int(x['u_count'][u])}" for u in units)


def trace(record):
	"""A data.Record as text."""
	summary = summarize(record.events)
	ours = summary.our_side
	lines = [f"we play side {ours} ({'attacker' if ours == 0 else 'defender'}), {record.decisions} decisions"]
	for i, action in enumerate(record.actions):
		x = {k: v[i] for k, v in record.tensors.items()}
		active = int(x["g_active_unit"])
		# (a compacted record, data.compact_record, has only the chosen action's whole probability)
		p = f"p(type)={float(np.exp(x['logp_type'][action['type']])):.2f}" if "logp_type" in x else f"p(action)={float(np.exp(x['old_log_prob'])):.2f}"
		value = float(x["value"][-1])
		lines.append(
			f"r{int(x['g_round']):<2} {vocab.name('creatures', int(x['u_creature'][active])):>14} x{int(x['u_count'][active]):<3} "
			f"{p} V={value:+.2f}  {describe_action(x, action):48s} | enemy: {army(x, 1 - ours)}"
		)
	lines.append(f"outcome {summary.outcome.name} after {summary.rounds} rounds, reward {summary.reward:+.3f} (scaled {to_scale(summary.reward):+.3f}), kept {summary.own_kept:.0f} of {summary.own_start_value:.0f}, killed {summary.enemy_lost:.0f} of {summary.enemy_start_value:.0f}")
	return "\n".join(lines)


def main():
	from pathlib import Path

	from . import play
	from .evaluate import load
	from .export import export
	from .train import TrainConfig

	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("checkpoint")
	parser.add_argument("--side", default="0", help="0 = attack, 1 = defence, self = our model on both sides")
	parser.add_argument("--opponent", default="BattleAI")
	parser.add_argument("--temperature", type=float, default=0.0)
	parser.add_argument("--seed", type=int, default=None, help="VCMI seed; default: the training seed of the side")
	parser.add_argument("--scenario", default=TrainConfig.scenario)
	args = parser.parse_args()
	config = TrainConfig()
	model, epoch = load(args.checkpoint)
	model_path = Path(args.checkpoint).with_name("trace.onnx")
	export(model, model_path)
	battle_map = play.build_map(args.scenario, variant="trace")
	side = 0 if args.side == "self" else int(args.side)
	game = play.Game(args.scenario, our_side=side, opponent=play.SELF_PLAY if args.side == "self" else args.opponent, game_seed=args.seed if args.seed is not None else config.game_seeds[side], temperature=args.temperature, map=battle_map)
	try:
		for record in play.play([game], model_path):
			print(f"== epoch {epoch}\n{trace(record)}\n")
	finally:
		model_path.unlink(missing_ok=True)
		(play.USER_DIR / (battle_map + ".vmap")).unlink(missing_ok=True)


if __name__ == "__main__":
	main()
