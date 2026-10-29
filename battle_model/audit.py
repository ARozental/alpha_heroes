"""The battle side's value audit: the value claims the model sees at decision time, checked against what then happened
in real battles, by spell, with a flag wherever they disagree (campaign_model/audit.py does the same for the map).

	.venv/bin/python -m battle_model.audit [--battles 20] [--checkpoint path]

casts   a hero's cast is priced before it is made: the engine's predicted value change at its target (s_unit_value,
        s_hex_value: attacker - defender, gold; Encoder.cpp castValue). The next decision of the record is the same
        unit acting in the same round, nothing else in between, so the realized change is exact: our army's value
        minus the enemy's, by health (health_values: what each side keeps, summoned units and clones left out), at the
        next decision minus now. A spell whose price and realized change disagree (a factor 2 or the sign) is flagged.
        Temporary health (Resurrection below expert) counts in the realized change but not in the price.
missed  a damaging cast at a unit that left it untouched (2026-10-28, 30 battles: 3 of ~45; each one resisted by magic
        resistance, the Resistance skill of the target's hero, or bounced off Magic Mirror onto our own stacks). Since
        2026-10-28 the engine's price (castValue) is expected over each unit's resistance, and the model counts Magic
        Mirror (features.spell_features); a miss is then the dice, not a mispricing.
The reward's own accounting (what a side keeps: summoned units, clones and creatures raised for the battle vanish) is
the same rule as the round scores' (AlphaBattleAI::aliveArmies, remainingAfterBattle), so it has no claim to check.
The exit code is the number of flags (0: clean).
"""
import argparse
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from .constants import ActionType
from .data import summarize

MIN_COUNT = 5


def health_values(t, side):
	"""[T] the gold value of a side's living army at every decision of a record, by health: the price per hit point x
	(count - 1) x full health + the top creature's hit points left (summoned units and clones left out, as the reward
	does). Counts alone miss a cast's damage that kills no whole creature."""
	from . import vocab

	alive = (t["u_present"] != 0) & (t["u_alive"] != 0) & (t["u_side"] == side) & (t["u_is_summoned"] == 0) & (t["u_is_clone"] == 0)
	price = np.vectorize(lambda c: vocab.creature_value(int(c)) if c >= 0 else 0.0, otypes=[float])(t["u_creature"])
	full = np.vectorize(lambda c: float(vocab.creature(int(c)).get("hit_points", 1)) if c >= 0 else 1.0, otypes=[float])(t["u_creature"])
	health = (np.maximum(t["u_count"], 1) - 1) * full + t["u_first_hp"]
	return (price / np.maximum(full, 1.0) * health * alive).sum(axis=1)
GOLD_SLACK = 100.0  # amounts this small are noise
RATIO = 2.0


def audit(records):
	from . import vocab

	flags = []
	casts = defaultdict(lambda: [[], []])  # spell -> predicted, realized (our net gold)
	missed = defaultdict(lambda: [0, 0])  # spell -> damaging casts at a unit that left it untouched, all of them
	for record in records:
		t, actions = record.tensors, record.actions
		if not actions:
			continue
		our = summarize(record.events).our_side
		sign = 1.0 if our == 0 else -1.0  # the prices are the attacker's gain minus the defender's
		ours, theirs = health_values(t, our), health_values(t, 1 - our)
		for i, a in enumerate(actions[:-1]):
			# A hero's cast only: the same unit then acts in the same round, nothing between (a creature's cast is its
			# turn, and the enemy acts before our next decision); a cast the engine refused passes the turn: left out
			if a["type"] != ActionType.HERO_SPELL or a.get("spell", -1) < 0:
				continue
			if int(t["g_active_unit"][i]) != int(t["g_active_unit"][i + 1]) or int(t["g_round"][i]) != int(t["g_round"][i + 1]):
				continue
			s, unit, hex_ = a["spell"], a.get("target_unit", -1), a.get("target_hex", -1)
			price = t["s_unit_value"][i, s, unit] if unit >= 0 else t["s_hex_value"][i, s, hex_] if hex_ >= 0 else None
			name = vocab.name("spells", s)
			if price is None:
				continue
			if unit >= 0 and price != 0:  # a damaging cast at a unit: did it touch it? (the mask let it; did the engine?)
				untouched = int(t["u_count"][i, unit]) == int(t["u_count"][i + 1, unit]) and int(t["u_first_hp"][i, unit]) == int(t["u_first_hp"][i + 1, unit])
				missed[name][0] += untouched
				missed[name][1] += 1
				if untouched:
					continue
			casts[name][0].append(sign * float(price))
			casts[name][1].append(float((ours[i + 1] - ours[i]) - (theirs[i + 1] - theirs[i])))
	print("\nCASTS: the engine's price of the cast at its target (our side's gain, gold) against the realized change of our army minus theirs")
	print(f"  {'spell':28s} {'n':>4s} {'priced':>9s} {'realized':>9s}")
	for name, (priced, realized) in sorted(casts.items(), key=lambda kv: -len(kv[1][0])):
		n, p, r = len(priced), float(np.mean(priced)), float(np.mean(realized))
		apart = max(abs(p), abs(r)) > GOLD_SLACK and (p * r < 0 or max(abs(p), abs(r)) > RATIO * max(min(abs(p), abs(r)), GOLD_SLACK / RATIO))
		flag = ""
		if n >= MIN_COUNT and apart:
			flag = "  <-- FLAG"
			flags.append(f"cast {name}: priced {p:+.0f}, realized {r:+.0f} (n {n})")
		print(f"  {name:28s} {n:4d} {p:+9.0f} {r:+9.0f}{flag}")
	print("\nMISSED: damaging casts at a unit that left it untouched (resisted or reflected: priced as an expectation since 2026-10-28)")
	for name, (untouched, total) in sorted(missed.items(), key=lambda kv: -kv[1][0]):
		if untouched:
			flag = ""
			if untouched >= 2 or untouched / total >= 0.1:
				flag = "  <-- FLAG"
				flags.append(f"cast {name}: {untouched} of {total} damaging casts left their target untouched")
			print(f"  {name:28s} {untouched} of {total}{flag}")
	print(f"\n{len(flags)} flag(s)" + ("".join(f"\n  - {f}" for f in flags) if flags else ""))
	return flags


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--battles", type=int, default=20)
	parser.add_argument("--seed", type=int, default=5)
	parser.add_argument("--checkpoint", default=str(Path(__file__).resolve().parent.parent / "runtime" / "saved" / "battle_joint1_02840.pt"))
	args = parser.parse_args()
	import torch

	torch.set_num_threads(1)
	from .evaluate import load
	from .export import export
	from .gradient_study import collect_generated

	model, _ = load(args.checkpoint)
	with tempfile.TemporaryDirectory(prefix="audit-") as work:
		export(model, Path(work) / "model.onnx")
		records = collect_generated(Path(work) / "model.onnx", args.battles, 1, args.seed)
	print(f"{len(records)} battle records, {sum(r.decisions for r in records)} decisions", flush=True)
	raise SystemExit(len(audit(records)))


if __name__ == "__main__":
	main()
