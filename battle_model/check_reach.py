"""Check the reach maps (features.distance_maps, attack_positions) against the engine's own masks, in real battles:
in every recorded decision, where the acting unit can move (m_move_hex) and which positions it can attack each target
from (m_melee_from) must be exactly what the maps say, two-hex units included.

	.venv/bin/python -m battle_model.check_reach [--seeds N]

BattleAI plays both sides (our C++ AI delegates to it and records each state). Exits with 1 on any difference.
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from . import features as F
from . import play
from .constants import MAX_UNITS, N_HEXES
from .export import export
from .model import BattleModel

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ("castle_vs_inferno", "sure_kills")


def differences(record, model):
	"""(decisions, by two-hex units, decisions whose move hexes differ, whose attack-from hexes differ)."""
	x = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in record.tensors.items() if not k.startswith("logp_") and k != "value"}
	b = F.read_battle(x, model.tables())
	reach, _ = F.distance_maps(b)
	mine, speed = b.mine(reach), b.me["speed"]
	hexes = torch.arange(N_HEXES)
	moves = (mine <= speed) & (hexes != b.me["hex"])
	wide = (b.me["rear"] >= 0)[..., None].expand(-1, MAX_UNITS, 1)
	positions = F.attack_positions(torch.stack([b.units["hex"], b.units["rear"]], dim=-1), wide, b.me["side"][..., None].expand(-1, MAX_UNITS, 1)).flatten(2)
	attack_from = (positions[..., :, None] == hexes).any(dim=-2) & (mine <= speed)[:, None, :]
	melee_differs = ((attack_from != x["m_melee_from"]) & x["m_melee_target"][..., None]).any(dim=(1, 2))
	return len(record.actions), int((b.me["rear"][:, 0] >= 0).sum()), int((moves != x["m_move_hex"]).any(dim=1).sum()), int(melee_differs.sum())


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--seeds", type=int, default=4)
	args = parser.parse_args()
	model = BattleModel().eval()
	work = Path(tempfile.mkdtemp(prefix="check-reach-", dir=ROOT / "runtime"))
	maps, records = [], []
	try:
		model_path = export(model, work / "model.onnx")
		for name in SCENARIOS:
			scenario = ROOT / "battle_model" / "scenarios" / f"{name}.json"
			maps.append(play.build_map(scenario, variant="check-reach"))
			games = [play.Game(str(scenario), our_side=side, opponent="BattleAI", game_seed=3000 + s, temperature=0.0, map=maps[-1], settings=(("ALPHA_BATTLE_DELEGATE", "BattleAI"),)) for side in (0, 1) for s in range(args.seeds)]
			records += play.play(games, model_path)
	finally:
		shutil.rmtree(work, ignore_errors=True)
		for battle_map in maps:
			(play.USER_DIR / (battle_map + ".vmap")).unlink(missing_ok=True)
	totals = np.sum([differences(r, model) for r in records], axis=0)
	print(f"{totals[0]} decisions ({totals[1]} by two-hex units): move hexes differ from the engine's in {totals[2]}, attack-from hexes in {totals[3]}")
	sys.exit(1 if totals[2] or totals[3] else 0)


if __name__ == "__main__":
	main()
