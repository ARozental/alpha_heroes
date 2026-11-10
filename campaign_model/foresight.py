"""The campaign model v2's foresight (model_v2.Foresight): what each decision's consequences were, as labels, and the
loss of the foresight's predictions against them. The predictions are never used (the AI does not read them); the
loss shapes the layers the policy reads (docs/campaign_model_v2_draft.md).

The consequence window of a decision: from it to the player's next turn decision that day, every question and battle
in between included (a chest's gold or experience, the University's skill, the guarded dwelling agreed to: all in the
window of the MOVE that led there). END has none (what follows is the other players' turns and the night).

	move    at a MOVE decision, its (hero, tile): a battle in the window (BCE), its score by the battle reward's formula
	        (battle_model/reward.py: what the enemy lost - what we lost, over both sides' assets, -1..1; Huber), the
	        change of each of Φ's parts over the window (symlog of thousands of gold; Huber)
	answer  at a question, its chosen option: the same, over the window from the answer
	hero    each of our heroes: still ours at the player's first decision the next day (BCE), its army's worth then
	        against now (log ratio, -3..1; Huber, where it is still ours)
	turn    Φ at the player's last turn decision that day against now (log ratio, -2..2), each resource at its first
	        decision the next day against now (log((r' + k) / (r + k)), -3..3); Huber
	plan    at a decision with Nullkiller2's advice, its plan in the same state (Decision.plan: Advice.cpp, our fork's
	        Nullkiller2): each of our heroes' TRAVEL area it sends the hero to, or none (no task for it); cross-entropy.
	        A label only, never an input: the AI plays without Nullkiller2
"""
from dataclasses import dataclass


import numpy as np
import torch
from torch.nn import functional as F

import game_values
from battle_model import reward

from . import constants as c
from . import features as ft
from .model_v2 import ACTION_TARGETS, AREAS, HERO_TARGETS, TURN_TARGETS
from .records import PLAN_ENTRY

W, HO = c.WINDOW, c.OWN_HEROES
PARTS = len(ft.PHI_PARTS)
RESOURCE_FLOOR = np.array([5, 5, 5, 5, 5, 5, 1000], dtype=np.float64)  # k of each resource's log ratio (gold: 1,000)


@dataclass
class ForesightConfig:
	weight: float = 0.2  # each head's loss weight (its gradient's share of the policy's on the shared layers is logged)
	huber: float = 1.0


def symlog(x):
	return np.sign(x) * np.log1p(np.abs(x))


# ---- the battle score of a recorded battle ----

def side_assets(battle, side):
	"""(its army's value at the start, its hero's asset, its mana's value, the price of its artifacts a winner takes)."""
	army = sum(count * game_values.creature_value(creature) for creature, count in battle["armies"][side])
	hero = battle["heroes"][side]
	if not hero:
		return army, 0.0, 0.0, 0.0
	artifacts = sum(game_values.artifact_price(a) for a in hero.get("artifacts", []))
	return army, reward.hero_asset(hero.get("level", 1), artifacts), reward.mana_value(hero.get("mana", 0), hero.get("power", 0)), artifacts


def outcome_of(battle, side):
	"""reward.Outcome of a side: the winner's WIN; the other's LOSS, RETREAT (it escaped) or SURRENDER; a draw: LOSS."""
	if battle["winner"] == side:
		return reward.Outcome.WIN
	return {1: reward.Outcome.RETREAT, 2: reward.Outcome.SURRENDER}.get(battle["result"], reward.Outcome.LOSS)


def battle_score(battle):
	"""The battle's score for our side (battle["side"]) by the battle reward's formula, -1..1. Mana spent is not recorded:
	each side's mana counts as at the start (while it keeps its hero)."""
	ours = battle["side"]
	start, end = {}, {}
	for side in (0, 1):
		army, hero, mana, _ = side_assets(battle, side)
		lost = sum(count * game_values.creature_value(creature) for creature, count in battle["casualties"][side])
		other_artifacts = side_assets(battle, 1 - side)[3] if battle["heroes"][1 - side] else 0.0
		start[side] = army + hero + mana
		outcome = outcome_of(battle, side)
		end[side] = reward.assets_after(outcome, max(army - lost, 0.0), hero, mana, captured=other_artifacts if outcome == reward.Outcome.WIN else 0.0)
	return reward.terminal_reward(start[ours], end[ours], start[1 - ours], end[1 - ours])


def stake_lost(battle):
	"""The share of our side's stake a battle took, 0..1 (the user, 2026-11-04: battle_score divides by both sides' assets,
	so a hero lost to a stack 50x its army scored ~-0.05): all of it when we did not win (the hero leaves the map with
	everything it had), else its casualties over its army's value and its hero's asset at the start (battle_score's)."""
	ours = battle["side"]
	if battle["winner"] != ours:
		return 1.0
	army, hero, _, _ = side_assets(battle, ours)
	lost = sum(count * game_values.creature_value(creature) for creature, count in battle["casualties"][ours])
	return float(min(1.0, lost / max(army + hero, 1.0)))


# ---- the labels of a game ----

LABEL_NAMES = frozenset((  # game_labels' arrays (learner.step computes a game's again when its precomputed ones lack one)
	"move", "move_mask", "move_battle", "move_at", "move_offer", "move_win", "move_stake", "answer", "answer_mask", "answer_battle", "option",
	"hero", "hero_mask", "hero_alive", "turn", "turn_mask", "plan", "plan_mask",
))


def game_labels(game, x, phi_parts, hero_gold):
	"""The foresight's labels for every decision of a game (records.Game, its decoded inputs x, Φ's parts [N, PARTS] and
	each hero slot's army worth [N, H] from the inputs): a dict of arrays over the decisions, each with its mask."""
	n = len(game.decisions)
	labels = {
		"move": np.zeros((n, ACTION_TARGETS), np.float32), "move_mask": np.zeros(n, bool), "move_battle": np.zeros(n, bool),
		"move_at": np.zeros((n, 3), np.int64), "move_offer": np.zeros(n, bool), "move_win": np.zeros(n, np.float32), "move_stake": np.zeros(n, np.float32),
		"answer": np.zeros((n, ACTION_TARGETS), np.float32), "answer_mask": np.zeros(n, bool), "answer_battle": np.zeros(n, bool),
		"option": np.zeros(n, np.int64),
		"hero": np.zeros((n, HO, HERO_TARGETS), np.float32), "hero_mask": np.zeros((n, HO), bool), "hero_alive": np.zeros((n, HO), bool),
		"turn": np.zeros((n, TURN_TARGETS), np.float32), "turn_mask": np.zeros((n, TURN_TARGETS), bool),
		"plan": np.zeros((n, HO), np.int64), "plan_mask": np.zeros((n, HO), bool),
	}
	plan_labels(game, x, labels)
	by_color = {}
	for i, d in enumerate(game.decisions):
		by_color.setdefault(d.color, []).append(i)
	phi_total = np.maximum(phi_parts.sum(axis=1), 1000.0)
	for color, indices in by_color.items():
		battles = sorted(game.battles.get(color, []), key=lambda b: b["serial"])
		serials = [b["serial"] for b in battles]
		turn = [i for i in indices if game.decisions[i].kind == c.DecisionKind.TURN]
		first_of_day, last_of_day = {}, {}
		for i in turn:
			first_of_day.setdefault(game.decisions[i].day, i)
			last_of_day[game.decisions[i].day] = i
		for position, i in enumerate(indices):
			d = game.decisions[i]
			serial = position + 1
			# The window: to the next turn decision that day
			later = [j for j in indices[position + 1 :] if game.decisions[j].kind == c.DecisionKind.TURN]
			end = later[0] if later and game.decisions[later[0]].day == d.day else None
			is_move = d.kind == c.DecisionKind.TURN and d.action_type == c.ActionType.MOVE
			is_answer = d.kind in c.DIALOG_KINDS and d.index >= 0
			if end is not None and (is_move or is_answer):
				end_serial = indices.index(end) + 1
				fought = [b for b, s in zip(battles, serials) if serial <= s < end_serial]
				target = np.zeros(ACTION_TARGETS, np.float32)
				target[0] = float(bool(fought))
				target[1] = battle_score(fought[0]) if fought else 0.0
				target[2:] = np.clip(symlog((phi_parts[end] - phi_parts[i]) / 1000.0), -6, 6)
				head = "move" if is_move else "answer"
				labels[head][i], labels[f"{head}_mask"][i], labels[f"{head}_battle"][i] = target, True, bool(fought)
				if is_move:
					hero, rest = divmod(d.index, W * W)  # (the window slot, as the heads' tables are laid out: schema 17)
					labels["move_at"][i] = (hero, rest // W, rest % W)
					# The engine's answer to the step, never ours (the user, 2026-11-04): a wandering monster that offers to join or
					# flees asks us first (VCMI's CGCreature: a yes/no question about it), one that fights does not
					window = indices[position + 1 : indices.index(end)]
					labels["move_offer"][i] = any(game.decisions[j].kind == c.DecisionKind.YES_NO and int(x["d_object"][j]) == ft.OBJ_MONSTER for j in window) if "d_object" in x else False
					if fought:  # (the battle's outcome: the battle heads' "move_win" and "move_stake")
						labels["move_win"][i] = float(fought[0]["winner"] == fought[0]["side"])
						labels["move_stake"][i] = stake_lost(fought[0])
				else:
					labels["option"][i] = d.index
			# Our heroes tomorrow, the turn's end, the resources tomorrow
			if d.kind != c.DecisionKind.TURN:
				continue
			tomorrow = first_of_day.get(d.day + 1)
			if tomorrow is not None:
				types_then = {int(x["h_type"][tomorrow, h]): h for h in range(HO) if x["h_present"][tomorrow, h] and x["h_owner"][tomorrow, h] == c.OWNER_SELF}
				for h in range(HO):
					if x["h_present"][i, h] and x["h_owner"][i, h] == c.OWNER_SELF:
						then = types_then.get(int(x["h_type"][i, h]))
						labels["hero_mask"][i, h] = True
						labels["hero_alive"][i, h] = then is not None
						labels["hero"][i, h, 0] = float(then is not None)
						if then is not None:
							labels["hero"][i, h, 1] = np.clip(np.log((hero_gold[tomorrow, then] + 100.0) / (hero_gold[i, h] + 100.0)), -3, 1)
				now, later_resources = x["g_resources"][i].astype(np.float64), x["g_resources"][tomorrow].astype(np.float64)
				labels["turn"][i, 1:] = np.clip(np.log((later_resources + RESOURCE_FLOOR) / (now + RESOURCE_FLOOR)), -3, 3)
				labels["turn_mask"][i, 1:] = True
			last = last_of_day.get(d.day)
			if last is not None:
				labels["turn"][i, 0] = np.clip(np.log(phi_total[last] / phi_total[i]), -2, 2)
				labels["turn_mask"][i, 0] = True
	return labels


def plan_labels(game, x, labels):
	"""Nullkiller2's plan at each decision that has it (Decision.plan), into labels["plan"] (each hero slot's TRAVEL area,
	AREAS: none) and labels["plan_mask"] (one of our heroes, its target an area or none; a task without a tile of the
	hero's, -2, is no label). The areas are features.area_index's (TRAVEL's: blocks of 2 x the map encoder's a side)."""
	rows = np.array([i for i, d in enumerate(game.decisions) if getattr(d, "plan", None) is not None], dtype=np.int64)
	if len(rows) == 0:
		return
	plan = np.asarray([game.decisions[i].plan for i in rows], dtype=np.float64).reshape(len(rows), HO, PLAN_ENTRY)
	tx, ty, tz = (plan[..., k].astype(np.int64) for k in range(3))
	size = np.maximum(x["g_map"][rows].astype(np.int64), 1)
	side = (2 * np.maximum((np.maximum(size[:, 0], size[:, 1]) + c.MAP_GRID - 1) // c.MAP_GRID, 1))[:, None]
	tg = c.TRAVEL_GRID
	area = (np.clip(tz, 0, c.MAP_LEVELS - 1) * tg + np.minimum(np.maximum(ty, 0) // side, tg - 1)) * tg + np.minimum(np.maximum(tx, 0) // side, tg - 1)
	none = tx == -1
	placed = (tx >= 0) & (ty >= 0) & (tz >= 0) & (tz < c.MAP_LEVELS)
	ours = x["h_present"][rows, :HO].astype(bool) & (x["h_owner"][rows, :HO] == c.OWNER_SELF)
	labels["plan"][rows] = np.where(none, AREAS, area)
	labels["plan_mask"][rows] = ours & (none | placed)


# The inputs the labels need: Φ's (features.empire_value), the heroes' armies and types, the resources
LABEL_INPUTS = (
	"g_resources", "g_object_income", "g_owned_creature", "g_owned_count", "g_dwelling_creature", "g_dwelling_count", "g_explored", "g_players", "g_day", "g_map",
	"t_present", "t_owner", "t_faction", "t_built", "t_garrison_creature", "t_garrison_count", "t_available", "t_available_creature",
	"h_present", "h_owner", "h_type", "h_detailed", "h_army_creature", "h_army_count", "h_artifacts", "h_backpack", "h_mana", "h_spell_cost",
	"h_primary", "h_primary_base", "h_skills", "h_exp", "h_spells", "h_luck", "h_morale", "h_battle_luck", "h_battle_morale",
	"d_object",  # (a question's object: a wandering monster's offer, the "move_offer" label)
)


def labels_of(game, features, chunk=512):
	"""game_labels of a game (records.Game), its inputs decoded here (only LABEL_INPUTS), Φ's parts and the heroes' army
	worth computed with the model's tables (features: model.CustomFeatures)."""
	from types import SimpleNamespace

	from .records import decode_fields

	# Every record as sent (game.all_decisions, learner.step: before the unusable ones are dropped): a battle's "serial"
	# counts them; the rows of the decisions kept are returned
	full = getattr(game, "all_decisions", game.decisions)
	x = decode_fields(full, LABEL_INPUTS)
	parts, gold = [], []
	with torch.no_grad():
		for start in range(0, len(full), chunk):
			t = {name: torch.from_numpy(value[start : start + chunk]) for name, value in x.items()}
			parts.append(ft.empire_value(t, features)[0].numpy())
			gold.append(ft.army(t["h_army_creature"], t["h_army_count"], t["h_detailed"], features)[2].numpy())
	labels = game_labels(SimpleNamespace(decisions=full, battles=game.battles), x, np.concatenate(parts).astype(np.float64), np.concatenate(gold).astype(np.float64))
	if full is not game.decisions:
		kept = {id(d) for d in game.decisions}
		rows = np.array([id(d) in kept for d in full], dtype=bool)
		labels = {name: value[rows] for name, value in labels.items()}
	return labels


def gradient_check(model, main, foresight, terms=None, prefix="grad_shared_"):
	"""The shared layers' gradient norm from the policy's and the values' losses (main) and from the foresight's, their
	cosine, and the share of the parameters neither reaches (a path cut off), without touching .grad; terms ({name: a
	term of the loss}, train.micro_loss's split): each one's norm there too (<prefix><name>), on the same scale."""
	shared = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith(("encode_", "blocks.", "unet.", "read_tiles.", "sides.", "tile_sides.", "join_", "stamp_"))]
	params = [p for p in model.parameters() if p.requires_grad]
	g_main = torch.autograd.grad(main, params, retain_graph=True, allow_unused=True)
	g_fore = torch.autograd.grad(foresight, params, retain_graph=True, allow_unused=True)
	index = {id(p): i for i, p in enumerate(params)}
	flat = lambda grads: torch.cat([(grads[index[id(p)]] if grads[index[id(p)]] is not None else torch.zeros_like(p)).flatten() for p in shared])  # noqa: E731
	a, b = flat(g_main), flat(g_fore)
	dead = sum(1 for x, y in zip(g_main, g_fore) if (x is None or not bool(x.any())) and (y is None or not bool(y.any())))
	norms = {}
	for name, term in (terms or {}).items():
		if term.requires_grad:
			norms[f"{prefix}{name}"] = float(flat(torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)).norm())
	return {
		"grad_shared_main": float(a.norm()), "grad_shared_foresight": float(b.norm()),
		"grad_shared_cosine": float(F.cosine_similarity(a, b, dim=0)), "grad_dead_share": dead / max(len(params), 1), **norms,
	}


# ---- the loss ----

def masked_mean(values, mask):
	mask = mask.float()
	return (values * mask).sum() / mask.sum().clamp(min=1.0)


def action_loss(predicted, target, battle, config, rows=None):
	"""[K, ACTION_TARGETS] predictions against their targets, over the rows where rows [K] is set (all by default): BCE
	on the battle, Huber on its score where there was one, Huber on the parts' changes; 0 with no row."""
	rows = torch.ones_like(battle) if rows is None else rows
	fought = masked_mean(F.binary_cross_entropy_with_logits(predicted[:, 0], target[:, 0], reduction="none"), rows)
	score = masked_mean(F.huber_loss(predicted[:, 1], target[:, 1], reduction="none", delta=config.huber), battle & rows)
	parts = masked_mean(F.huber_loss(predicted[:, 2:], target[:, 2:], reduction="none", delta=config.huber), rows.unsqueeze(-1).expand(-1, predicted.shape[1] - 2))
	return fought, score, parts


def battle_heads_loss(out, labels, every, at):
	"""The battle heads' losses (model_v2.Foresight.tiles, .outcome; the user, 2026-11-04: heads that mean something, and
	the engine's answer only, never our own):
	  move_tiles  what a wandering monster does when a hero steps on it or into its zone, on every legal window tile of a
	              MOVE decision: no monster here / it attacks / it offers to join or flees (VCMI asks us then: what we
	              answer is ours, not the engine's). A tile not chosen is known only to have a monster or not
	              (fs_stack_tile, our features'), its loss -log P(none) or -log P(attacks or offers), the two kinds weighted
	              alike whatever their numbers; the chosen one by what the engine did (a step that met neither a question
	              nor a battle, a move cut short, left out)
	  move_win    on a MOVE a battle followed: P(we won it) (BCE)
	  move_stake  and the share of the hero's stake it took (foresight.stake_lost; BCE against it)"""
	logp = F.log_softmax(out["fs_tiles"].float(), dim=-1)  # [B, HO, W, W, 3]
	legal, stack = out["fs_move_legal"] > 0, out["fs_stack_tile"] > 0
	move = labels["move_mask"]
	chosen = torch.zeros_like(legal)
	chosen[every, at[:, 0], at[:, 1], at[:, 2]] = move  # (a row with no MOVE: its at is 0s, nothing set)
	rest = legal & ~chosen
	plain = masked_mean(-logp[..., 0], rest & ~stack)
	met = masked_mean(-torch.logsumexp(logp[..., 1:], dim=-1), rest & stack)
	picked, stack_at = logp[every, at[:, 0], at[:, 1], at[:, 2]], stack[every, at[:, 0], at[:, 1], at[:, 2]]
	zero = torch.zeros_like(at[:, 0])
	offer, battle = labels["move_offer"], labels["move_battle"]
	kind = torch.where(stack_at, torch.where(offer, zero + 2, zero + 1), zero)
	known = move & (~stack_at | offer | battle)
	tiles = 0.5 * (plain + met) + masked_mean(-picked.gather(1, kind.unsqueeze(1)).squeeze(1), known)
	outcome = out["fs_outcome"][every, at[:, 0], at[:, 1], at[:, 2]].float()
	fought = battle & move
	won = masked_mean(F.binary_cross_entropy_with_logits(outcome[:, 0], labels["move_win"], reduction="none"), fought)
	stake = masked_mean(F.binary_cross_entropy_with_logits(outcome[:, 1], labels["move_stake"], reduction="none"), fought)
	return tiles, won, stake


PART_NAMES = ("move_battle", "move_score", "move_parts", "answer_battle", "answer_score", "answer_parts", "hero_alive", "hero_army", "turn", "plan", "move_tiles", "move_win", "move_stake")  # foresight_loss's parts, in this order


def foresight_loss(out, labels, config=ForesightConfig()):
	"""(the loss, {part: its value}) of the foresight's predictions (out["fs_*"]) against a batch's labels (game_labels'
	rows, as tensors on the device)."""
	# Every part over every row, its mask weighting it (0 with none): no host read of the masks (bool(mask.any()) and
	# torch.nonzero each waited for the GPU, five times a micro-batch: the trainer's thread could not run ahead of it,
	# 2026-10-31). The labels of the rows a part does not apply to are zeros, valid indices among them (game_labels)
	parts, zero = {}, out["fs_turn"].sum() * 0.0
	every = torch.arange(out["fs_turn"].shape[0], device=out["fs_turn"].device)
	at = labels["move_at"]
	predicted = out["fs_move"][every, at[:, 0], at[:, 1], at[:, 2]]
	parts["move_battle"], parts["move_score"], parts["move_parts"] = action_loss(predicted, labels["move"], labels["move_battle"], config, labels["move_mask"])
	parts["move_tiles"], parts["move_win"], parts["move_stake"] = battle_heads_loss(out, labels, every, at)
	predicted = out["fs_answer"][every, labels["option"]]
	parts["answer_battle"], parts["answer_score"], parts["answer_parts"] = action_loss(predicted, labels["answer"], labels["answer_battle"], config, labels["answer_mask"])
	predicted = out["fs_hero"][:, :HO]
	parts["hero_alive"] = masked_mean(F.binary_cross_entropy_with_logits(predicted[..., 0], labels["hero"][..., 0], reduction="none"), labels["hero_mask"])
	parts["hero_army"] = masked_mean(F.huber_loss(predicted[..., 1], labels["hero"][..., 1], reduction="none", delta=config.huber), labels["hero_alive"])
	parts["turn"] = masked_mean(F.huber_loss(out["fs_turn"], labels["turn"], reduction="none", delta=config.huber), labels["turn_mask"])
	parts["plan"] = zero
	if "fs_plan" in out and "plan" in labels:  # (model_v2.plan_of: the trainer's model only)
		logits = out["fs_plan"].float()
		crossed = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels["plan"].reshape(-1), reduction="none").reshape(labels["plan"].shape)
		parts["plan"] = masked_mean(crossed, labels["plan_mask"])
	total = sum(parts.values(), zero)
	return config.weight * total, parts
