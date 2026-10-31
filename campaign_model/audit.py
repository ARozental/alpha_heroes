"""The campaign's value audit: every claim the model's hand-made parts make about what something is worth, checked
against what then happened in real games, by category, with a flag wherever they disagree. The bugs it is for score
something that cannot be had (a guarded treasure counted when its fight is lost), price two things in different
units, or make an action pay that changes nothing; any one of them can make training useless.

	.venv/bin/python -m campaign_model.audit [--games 2] [--days 7] [--players 3] [--checkpoint path]

What is checked (Φ in the graph is the ledger's to the gold, so a decision's realized change is exact: Φ at the
player's next turn decision that day minus Φ now, the explored tiles apart):
	moves       each MOVE's predicted value of its tile (the object x P(win) + the fight's return, gold) against the
	            realized change, by what the tile held and the fight it started
	fights      predicted P(win) against how often the hero was still ours at the next decision, by bins
	exploring   the land a move was predicted to reveal against the tiles it did
	actions     the realized change of every other turn action type (buying at list price is Φ-neutral: a type that
	            pays is an exploit), and how often each changed nothing
The exit code is the number of flags (0: clean).
"""
import argparse
import shutil
from collections import defaultdict

import numpy as np
import torch

from . import constants as c
from . import features as ft
from . import priors
from .model import HO, build_model
from .records import NO_EFFECT, decode_inputs

W = c.WINDOW
IMMEDIATE = priors.immediate_objects()
MIN_COUNT = 5  # a category is judged from this many cases on
GOLD_SLACK = 300.0  # amounts this small are noise
RATIO = 2.0  # predicted and realized may differ by this factor before a flag (or by sign)


def features_of(model, x):
	"""(Φ [N], explored tiles [N], per MOVE decision: its window slot's tile values) of decoded inputs, in chunks."""
	out = defaultdict(list)
	n = len(x["g_day"])
	for start in range(0, n, 64):
		t = {k: torch.from_numpy(v[start : start + 64]) for k, v in x.items()}
		with torch.no_grad():
			windows, k = model.windows(t)
			f = model.features(t, windows)
		out["phi"].append(f.phi_total.numpy())
		out["windows"].append(windows[:k].numpy() + start * HO)
		if k:
			for name in ("tile_value", "tile_object_gold", "tile_fight_gold", "move_features"):
				out[name].append(f[name][:k].numpy())
	result = {"phi": np.concatenate(out["phi"]), "windows": np.concatenate(out["windows"])}
	for name in ("tile_value", "tile_object_gold", "tile_fight_gold", "move_features"):
		result[name] = np.concatenate(out[name]) if out[name] else np.zeros((0, W, W))
	return result


def next_turn_decisions(decisions):
	"""For each decision, the index of the same player's next turn decision that day (-1: none)."""
	following, latest = np.full(len(decisions), -1), {}
	for i in reversed(range(len(decisions))):
		d = decisions[i]
		following[i] = latest.get((d.color, d.day), -1)
		if d.kind == c.DecisionKind.TURN:
			latest[(d.color, d.day)] = i
	return following


def tile_category(x, i, hero, row, column, fight_gold):
	"""(what the tile held, the fight stepping there started: none unless one was predicted) of decision i's tile."""
	obj = int(x["w_object"][i, hero, row, column])
	flags = int(x["w_flags"][i, hero, row, column])
	owner = int(x["w_owner"][i, hero, row, column])
	held = "empty" if obj < 0 else ("our " if owner == c.OWNER_SELF else "") + c.object_class(obj).name.lower()
	if obj >= 0 and owner != c.OWNER_SELF and obj not in IMMEDIATE:
		held += " (option)"  # its prior is option value Φ does not price: shown, never judged
	if x["w_creature"][i, hero, row, column] >= 0:
		fight = "monster"
	elif fight_gold == 0:
		fight = "none"
	elif c.object_class(obj) == c.ObjectClass.BANK:
		fight = "bank"
	elif obj == ft.OBJ_TOWN:
		fight = "town"
	elif obj == ft.OBJ_HERO:
		fight = "hero"
	elif flags & c.TILE_GUARDED:
		fight = "guard"
	else:
		fight = "other"
	return held, fight


def audit(games, model):
	flags = []
	moves = defaultdict(lambda: [[], []])  # (held, fight) -> predicted, realized
	wins = defaultdict(lambda: [[], []])  # P(win) bin -> predicted, won
	explore = [[], []]
	actions = defaultdict(lambda: [[], 0, 0])  # type -> realized changes (all but MOVE), changed nothing, decisions
	for game in games:
		decisions = game.decisions
		if not decisions:
			continue
		x = decode_inputs(decisions)
		f = features_of(model, x)
		slot = {int(w): s for s, w in enumerate(f["windows"])}
		following = next_turn_decisions(decisions)
		explored = x["g_explored"].astype(np.float64)
		for i, d in enumerate(decisions):
			if d.kind != c.DecisionKind.TURN or d.action_type < 0:
				continue
			kind = c.ActionType(d.action_type)
			j = following[i]
			change = (f["phi"][j] - f["phi"][i]) - (explored[j] - explored[i]) if j >= 0 else None
			actions[kind.name][1] += bool(d.flags & NO_EFFECT)
			actions[kind.name][2] += 1
			if kind == c.ActionType.MOVE:
				hero, cell = divmod(d.index, W * W)
				row, column = divmod(cell, W)
				s = slot.get(i * HO + hero)
				if s is None or change is None:
					continue
				object_gold, fight_gold = f["tile_object_gold"][s, row, column], f["tile_fight_gold"][s, row, column]
				win = f["move_features"][s, row, column, 4]
				reveal = f["move_features"][s, row, column, 5]
				immediate = object_gold * (win if fight_gold != 0 else 1.0) + fight_gold
				held, fight = tile_category(x, i, hero, row, column, fight_gold)
				moves[(held, fight)][0].append(immediate)
				moves[(held, fight)][1].append(change)
				if fight != "none":
					hero_type = x["h_type"][i, hero]
					still = bool(((x["h_type"][j, :HO] == hero_type) & x["h_present"][j, :HO]).any())
					bin_ = min(int(win * 4), 3)
					wins[bin_][0].append(win)
					wins[bin_][1].append(float(still))
				explore[0].append(reveal)
				explore[1].append(explored[j] - explored[i])
			elif change is not None:
				actions[kind.name][0].append(change)

	print("\nMOVES: what the tile's object and fight were predicted to be worth (gold) against Φ's change (explored tiles apart)")
	print(f"  {'held':16s} {'fight':8s} {'n':>5s} {'predicted':>10s} {'realized':>10s}")
	for (held, fight), (predicted, realized) in sorted(moves.items(), key=lambda kv: -len(kv[1][0])):
		n, p, r = len(predicted), float(np.mean(predicted)), float(np.mean(realized))
		flag = ""
		apart = max(abs(p), abs(r)) > GOLD_SLACK and (p * r < 0 or max(abs(p), abs(r)) > RATIO * max(min(abs(p), abs(r)), GOLD_SLACK / RATIO))
		if n >= MIN_COUNT and apart and "(option)" not in held:
			flag = "  <-- FLAG"
			flags.append(f"moves {held}/{fight}: predicted {p:+.0f}, realized {r:+.0f} (n {n})")
		print(f"  {held:16s} {fight:8s} {n:5d} {p:+10.0f} {r:+10.0f}{flag}")
	print("\nFIGHTS: predicted P(win) against the hero still ours at the next decision")
	for bin_, (predicted, won) in sorted(wins.items()):
		n, p, w = len(predicted), float(np.mean(predicted)), float(np.mean(won))
		flag = ""
		if n >= MIN_COUNT and abs(p - w) > 0.3:
			flag = "  <-- FLAG"
			flags.append(f"fights P(win) {p:.2f}: won {w:.2f} (n {n})")
		print(f"  P(win) {bin_ / 4:.2f}..{(bin_ + 1) / 4:.2f}: n {n:4d}, predicted {p:.2f}, still ours {w:.2f}{flag}")
	if len(explore[0]) >= MIN_COUNT:
		reveal, tiles = np.array(explore[0]), np.array(explore[1])
		correlation = float(np.corrcoef(reveal, tiles)[0, 1]) if reveal.std() > 0 and tiles.std() > 0 else float("nan")
		flag = "  <-- FLAG" if correlation < 0 else ""
		if flag:
			flags.append(f"exploring: the predicted reveal and the tiles revealed correlate {correlation:+.2f}")
		print(f"\nEXPLORING: the predicted share of a sight's new land against the tiles revealed: correlation {correlation:+.2f} over {len(reveal)} moves "
			  f"(tiles revealed: {tiles[reveal > 0.2].mean() if (reveal > 0.2).any() else float('nan'):.0f} where over a fifth was predicted, {tiles[reveal <= 0.2].mean() if (reveal <= 0.2).any() else float('nan'):.0f} elsewhere){flag}")
	print("\nACTIONS: Φ's change after each other turn action type (buying at list price is neutral), and how often each changed nothing")
	for name, (changes, futile, total) in sorted(actions.items()):
		n = len(changes)
		mean = float(np.mean(changes)) if n else float("nan")
		share = futile / max(total, 1)
		flag = ""
		if name != "MOVE" and n >= MIN_COUNT and mean > 200:
			flag = "  <-- FLAG (pays)"
			flags.append(f"action {name}: Φ rises {mean:+.0f} a time (n {n})")
		if total >= MIN_COUNT and share > 0.2:
			flag += "  <-- FLAG (changes nothing)"
			flags.append(f"action {name}: changed nothing {share:.0%} of {total}")
		print(f"  {name:8s} n {total:4d}  mean change {mean:+8.0f}  changed nothing {share:.0%}{flag}")
	print(f"\n{len(flags)} flag(s)" + ("".join(f"\n  - {f}" for f in flags) if flags else ""))
	return flags


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--games", type=int, default=2)
	parser.add_argument("--days", type=int, default=7)
	parser.add_argument("--players", type=int, default=3)
	parser.add_argument("--seed", type=int, default=31)
	parser.add_argument("--checkpoint", default="", help="the model to audit (default: a new one)")
	args = parser.parse_args()
	torch.set_num_threads(1)
	from battle_model import play as battle_play

	from .play import make_map, play
	from .train import serve_model

	torch.manual_seed(0)
	model = build_model().eval()
	if args.checkpoint:
		state = torch.load(args.checkpoint, map_location="cpu")
		model = build_model(state.get("model_kwargs", {})).eval()
		from .train import migrate_subtypes  # (here: train imports this module's neighbours)
		migrate_subtypes(state, model)  # (a checkpoint from before the dense subtype tables, 2026-10-31)
		model.load_state_dict(state["model"])
	games = []
	with serve_model() as (served, server):
		served.load_state_dict(model.state_dict())
		server.set_model("current", 2, served)
		for g in range(args.games):
			name, _ = make_map(args.seed + g, args.players, 36)
			game, info = play(server.socket_path, name, args.days, args.seed + g, max_seconds=300, battle_ai="BattleAI")
			(battle_play.USER_DIR / (name + ".vmap")).unlink(missing_ok=True)
			shutil.rmtree(info.run_dir, ignore_errors=True)
			print(f"game {g}: {info.status}, {len(game.decisions)} decisions, {info.seconds:.0f}s", flush=True)
			games.append(game)
	raise SystemExit(len(audit(games, model)))


if __name__ == "__main__":
	main()
