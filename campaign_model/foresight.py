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
"""
from dataclasses import dataclass


import numpy as np
import torch
from torch.nn import functional as F

import game_values
from battle_model import reward

from . import constants as c
from . import features as ft
from .model_v2 import ACTION_TARGETS, HERO_TARGETS, TURN_TARGETS

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


# ---- the labels of a game ----

def game_labels(game, x, phi_parts, hero_gold):
	"""The foresight's labels for every decision of a game (records.Game, its decoded inputs x, Φ's parts [N, PARTS] and
	each hero slot's army worth [N, H] from the inputs): a dict of arrays over the decisions, each with its mask."""
	n = len(game.decisions)
	labels = {
		"move": np.zeros((n, ACTION_TARGETS), np.float32), "move_mask": np.zeros(n, bool), "move_battle": np.zeros(n, bool),
		"move_at": np.zeros((n, 3), np.int64),
		"answer": np.zeros((n, ACTION_TARGETS), np.float32), "answer_mask": np.zeros(n, bool), "answer_battle": np.zeros(n, bool),
		"option": np.zeros(n, np.int64),
		"hero": np.zeros((n, HO, HERO_TARGETS), np.float32), "hero_mask": np.zeros((n, HO), bool), "hero_alive": np.zeros((n, HO), bool),
		"turn": np.zeros((n, TURN_TARGETS), np.float32), "turn_mask": np.zeros((n, TURN_TARGETS), bool),
	}
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
					hero, rest = divmod(d.index, W * W)
					labels["move_at"][i] = (hero, rest // W, rest % W)
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


# The inputs the labels need: Φ's (features.empire_value), the heroes' armies and types, the resources
LABEL_INPUTS = (
	"g_resources", "g_object_income", "g_owned_creature", "g_owned_count", "g_dwelling_creature", "g_dwelling_count", "g_explored", "g_players", "g_day", "g_map",
	"t_present", "t_owner", "t_faction", "t_built", "t_garrison_creature", "t_garrison_count", "t_available", "t_available_creature",
	"h_present", "h_owner", "h_type", "h_detailed", "h_army_creature", "h_army_count", "h_artifacts", "h_backpack", "h_mana", "h_spell_cost",
	"h_primary", "h_primary_base", "h_skills", "h_exp", "h_spells", "h_luck", "h_morale", "h_battle_luck", "h_battle_morale",
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


def gradient_check(model, main, foresight):
	"""The shared layers' gradient norm from the policy's and the values' losses (main) and from the foresight's, their
	cosine, and the share of the parameters neither reaches (a path cut off), without touching .grad."""
	shared = [p for name, p in model.named_parameters() if p.requires_grad and name.startswith(("encode_", "blocks.", "unet.", "read_tiles.", "sides.", "tile_sides.", "join_", "stamp_"))]
	params = [p for p in model.parameters() if p.requires_grad]
	g_main = torch.autograd.grad(main, params, retain_graph=True, allow_unused=True)
	g_fore = torch.autograd.grad(foresight, params, retain_graph=True, allow_unused=True)
	index = {id(p): i for i, p in enumerate(params)}
	flat = lambda grads: torch.cat([(grads[index[id(p)]] if grads[index[id(p)]] is not None else torch.zeros_like(p)).flatten() for p in shared])  # noqa: E731
	a, b = flat(g_main), flat(g_fore)
	dead = sum(1 for x, y in zip(g_main, g_fore) if (x is None or not bool(x.any())) and (y is None or not bool(y.any())))
	return {
		"grad_shared_main": float(a.norm()), "grad_shared_foresight": float(b.norm()),
		"grad_shared_cosine": float(F.cosine_similarity(a, b, dim=0)), "grad_dead_share": dead / max(len(params), 1),
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


PART_NAMES = ("move_battle", "move_score", "move_parts", "answer_battle", "answer_score", "answer_parts", "hero_alive", "hero_army", "turn")  # foresight_loss's parts, in this order


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
	predicted = out["fs_answer"][every, labels["option"]]
	parts["answer_battle"], parts["answer_score"], parts["answer_parts"] = action_loss(predicted, labels["answer"], labels["answer_battle"], config, labels["answer_mask"])
	predicted = out["fs_hero"][:, :HO]
	parts["hero_alive"] = masked_mean(F.binary_cross_entropy_with_logits(predicted[..., 0], labels["hero"][..., 0], reduction="none"), labels["hero_mask"])
	parts["hero_army"] = masked_mean(F.huber_loss(predicted[..., 1], labels["hero"][..., 1], reduction="none", delta=config.huber), labels["hero_alive"])
	parts["turn"] = masked_mean(F.huber_loss(out["fs_turn"], labels["turn"], reduction="none", delta=config.huber), labels["turn_mask"])
	total = sum(parts.values(), zero)
	return config.weight * total, {k: v.detach() for k, v in parts.items()}
