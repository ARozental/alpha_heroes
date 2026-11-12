"""Predicting Nullkiller2's planner (stage 1, 2026-11-11; the user: "this is an auxiliary loss, we can have it if it
helps"; "I do not want a monstrosity to be served live"): a head that learns what the planner says from a pass of the
trunk that never sees the planner, so that a later stage can read the head instead of the live planner.

  blind_inputs   the planner's fields at their schema fills ("absent", as a game without the advisor encodes them),
                 the memory at its start: the planner's information must not flow in through what earlier decisions
                 wrote either
  targets        from the live inputs of the same rows (views.candidate_tables):
                   a  has       P(the planner has a task through this entry), every legal MOVE cell and TRAVEL area
                   b  forms     where it has one: the task's preference against the decision's best and against its
                                hero's best (views.preference, +1: 0..1), the two day discounts (views.discounts) and its
                                rank among its hero's tasks (1 / (1 + the tasks above it)), each 0..1
                   c  plan      a hero's distribution over its entries and "no task": each of its candidates' plan
                                weight (views' e^(4 x its hero-relative preference), summing to 1) at its first step's
                                entry (its MOVE entry, else its TRAVEL one)
  loss           a's binary cross-entropy + b's squared errors + c's soft cross-entropy, each a mean
  readings       a's AUC, b's explained variance per form, c's cross-entropy against the uniform's over the hero's
                 options, its top-1 agreement with the hero's best candidate and with the recorded own choice

What the planner says of danger (k_loss, k_danger, the threats) is not a target: our own forecasts, learned from what
really happened, stand in for those."""
import torch
import torch.nn.functional as F

from . import constants as c
from . import views as V
from .schema import SPECS

HO, W, CELLS = c.OWN_HEROES, c.WINDOW, c.WINDOW * c.WINDOW
AREAS = c.MAP_LEVELS * c.TRAVEL_GRID * c.TRAVEL_GRID
FORMS = ("preference_all", "preference_hero", "soon", "later", "rank")
OUT = 2 + len(FORMS)  # an entry's outputs: has (logit), the forms (logits: sigmoid), its plan logit
# The planner's fields (schema 19): every k_* row, its presence, tier and count, its own choice, the threats
BLIND = tuple(name for name in SPECS if name.startswith(("k_", "g_nk_"))) + ("t_nk_threat", "h_nk_threat")


def blind_inputs(x):
	"""x with the planner's fields at their fills and the memory at its start (a new dict; x unchanged)."""
	out = dict(x)
	for name in BLIND:
		if name in out:
			out[name] = torch.full_like(out[name], SPECS[name].fill)
	out["mem"] = torch.zeros_like(out["mem"])
	out.pop("mem_float", None)
	return out


def targets(x, computed=None):
	"""The head's targets and masks from the live inputs x (no gradient); computed [B, HO]: the MOVE window slots the
	blind pass computed (its window budget), the others' cells left out of every target (None: all)."""
	with torch.no_grad():
		t = V.candidate_tables(x, dest=False)
		b = x["g_day"].shape[0]
		present = x["g_nk_present"].bool()
		move_legal = x["move_mask"].bool().reshape(b, HO, CELLS) & present.reshape(b, 1, 1)
		if computed is not None:
			move_legal = move_legal & computed.unsqueeze(-1)
		travel_legal = x["travel_mask"].bool().reshape(b, HO, AREAS) & present.reshape(b, 1, 1)
		key, hero = t["key"], t["hero_index"]
		scored = key > 0
		above = ((key.unsqueeze(1) > key.unsqueeze(2)) & (hero.unsqueeze(1) == hero.unsqueeze(2)) & scored.unsqueeze(1)).sum(dim=2)  # [B, NK]: the hero's tasks above each
		rank = scored.float() / (1.0 + above.float())
		out = {"move_legal": move_legal, "travel_legal": travel_legal}
		for name, legal, size in (("move", move_legal, HO * CELLS), ("travel", travel_legal, HO * AREAS)):
			table = t[name].reshape(b, HO, -1, V.F)
			has = (table[..., 0] > 0) & legal
			forms = torch.stack([
				1.0 + table[..., V.CHANNELS.index("preference_all")], 1.0 + table[..., V.CHANNELS.index("preference_hero")],
				table[..., V.CHANNELS.index("soon")], table[..., V.CHANNELS.index("later")],
				V._scatter(t[f"{name}_index"], rank, size).reshape(b, HO, -1),
			], dim=-1).clamp(0.0, 1.0)
			out[f"{name}_has"], out[f"{name}_forms"] = has, forms
		# c: each hero's candidates' plan weights at their entries ([B, HO hero, HO x CELLS + AREAS + 1])
		belongs = window_heroes(x)  # [B, HO hero, HO slot]
		allowed = torch.cat([(belongs.unsqueeze(-1) & move_legal.unsqueeze(1)).reshape(b, HO, HO * CELLS), travel_legal, torch.ones(b, HO, 1, dtype=torch.bool, device=hero.device)], dim=-1)
		rows = V.candidate_rows(x)
		move, travel = rows["move"], rows["travel"]
		weight = t["plan_weight"] * scored
		size = HO * CELLS + AREAS
		in_move = (move >= 0) & (move < HO * CELLS)
		in_travel = (travel >= 0) & (travel < HO * AREAS)
		position = torch.where(in_move, move, torch.where(in_travel, HO * CELLS + travel.remainder(AREAS), torch.full_like(move, size + 1)))
		heroes = hero.clamp(max=HO)  # (HO: no hero)
		flat = torch.zeros(b, (HO + 1) * (size + 2), device=hero.device)
		flat = flat.scatter_add(1, heroes * (size + 2) + position, weight * (heroes < HO))
		plan = flat.reshape(b, HO + 1, size + 2)[:, :HO, : size + 1] * allowed
		ours = x["h_present"][:, :HO].bool() & (x["h_owner"][:, :HO] == c.OWNER_SELF) & present.reshape(b, 1)
		total = plan.sum(dim=-1, keepdim=True)
		none = torch.zeros_like(plan)
		none[..., -1] = 1.0
		out["plan"] = torch.where(total > 0, plan / total.clamp(min=1e-12), none)
		out["plan_allowed"], out["plan_heroes"] = allowed, ours
		out["plan_best"] = torch.where(total.squeeze(-1) > 0, plan.argmax(dim=-1), torch.full_like(total.squeeze(-1), size, dtype=torch.long))
		choice = own_choice(x, belongs)
		out["choice"] = torch.where((choice >= 0) & allowed.gather(2, choice.clamp(min=0).unsqueeze(-1)).squeeze(-1), choice, torch.full_like(choice, -1))  # (in a window the blind pass left out: not counted)
		return out


def window_heroes(x):
	"""[B, HO hero, HO slot]: which of our heroes each MOVE window slot is (its own slot, or a further window of it: w_extra)."""
	b = x["g_day"].shape[0]
	slots = torch.arange(HO, device=x["g_day"].device)
	own = (x["h_present"][:, :HO].bool() & (x["h_owner"][:, :HO] == c.OWNER_SELF)).unsqueeze(-1) & (slots.reshape(1, HO, 1) == slots.reshape(1, 1, HO))
	further = x["w_extra"][:, :HO].long() - 1  # [B, slot]: the hero, -1 none
	return own | (further.reshape(b, 1, HO) == slots.reshape(1, HO, 1))


def own_choice(x, belongs):
	"""[B, HO hero]: Nullkiller2's recorded own choice as a position among the hero's options (targets' plan layout), -1
	where it named none of that hero's MOVE or TRAVEL entries (hidden, another type, another hero)."""
	b = x["g_day"].shape[0]
	action, index = x["g_nk_action"].long() - 1, x["g_nk_index"].long() - 1
	out = torch.full((b, HO), -1, dtype=torch.long, device=action.device)
	slot = (index // CELLS).clamp(0, HO - 1)
	is_move = (action == c.ActionType.MOVE) & (index >= 0) & (index < HO * CELLS)
	move_hero = belongs.float().argmax(dim=1).gather(1, slot.reshape(b, 1)).squeeze(1)  # (the slot's hero)
	out = torch.where(is_move.reshape(b, 1) & (torch.arange(HO, device=out.device).reshape(1, HO) == move_hero.reshape(b, 1)), index.reshape(b, 1), out)
	is_travel = (action == c.ActionType.TRAVEL) & (index >= 0) & (index < HO * AREAS)
	travel_hero = (index // AREAS).clamp(0, HO - 1)
	return torch.where(is_travel.reshape(b, 1) & (torch.arange(HO, device=out.device).reshape(1, HO) == travel_hero.reshape(b, 1)), (HO * CELLS + index.remainder(AREAS)).reshape(b, 1), out)


def plan_logits(pred, target):
	"""The head's per-hero logits over the plan layout [B, HO, HO x CELLS + AREAS + 1], -inf where not allowed."""
	b = pred["nkp_move"].shape[0]
	move = pred["nkp_move"][..., -1].reshape(b, 1, HO * CELLS).expand(b, HO, HO * CELLS)
	logits = torch.cat([move, pred["nkp_travel"][..., -1], pred["nkp_none"].unsqueeze(-1)], dim=-1).float()
	return logits.masked_fill(~target["plan_allowed"], float("-inf"))


def loss(pred, target):
	"""(the head's loss, its parts {a, b, c}): a's BCE over the legal entries, b's squared errors where a task is, c's soft
	cross-entropy over our heroes; each a mean over what it counts."""
	parts = {}
	logits, has, legal = [], [], []
	forms_pred, forms_true, where = [], [], []
	for name in ("move", "travel"):
		p = pred[f"nkp_{name}"].reshape(pred[f"nkp_{name}"].shape[0], HO, -1, OUT).float()
		logits.append(p[..., 0])
		has.append(target[f"{name}_has"].float())
		legal.append(target[f"{name}_legal"].float())
		forms_pred.append(torch.sigmoid(p[..., 1 : 1 + len(FORMS)]))
		forms_true.append(target[f"{name}_forms"])
		where.append(target[f"{name}_has"].float())
	logits, has, legal = torch.cat(logits, dim=-1), torch.cat(has, dim=-1), torch.cat(legal, dim=-1)
	parts["a"] = (F.binary_cross_entropy_with_logits(logits, has, reduction="none") * legal).sum() / legal.sum().clamp(min=1.0)
	forms_pred, forms_true, where = torch.cat(forms_pred, dim=2), torch.cat(forms_true, dim=2), torch.cat(where, dim=-1)
	parts["b"] = (((forms_pred - forms_true) ** 2).sum(dim=-1) * where).sum() / where.sum().clamp(min=1.0)
	scores = plan_logits(pred, target)
	logp = torch.where(target["plan_allowed"], scores.log_softmax(dim=-1), torch.zeros_like(scores))
	heroes = target["plan_heroes"].float()
	parts["c"] = (-(target["plan"] * logp).sum(dim=-1) * heroes).sum() / heroes.sum().clamp(min=1.0)
	return parts["a"] + parts["b"] + parts["c"], parts


def auc(scores, labels):
	"""The area under the ROC curve of scores [N] for binary labels [N] (Mann-Whitney's, ties averaged); nan with one class."""
	positives = labels.sum()
	negatives = labels.numel() - positives
	if positives == 0 or negatives == 0:
		return float("nan")
	order = scores.argsort()
	ranks = torch.empty_like(scores, dtype=torch.float64)
	ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float64, device=scores.device)
	return float((ranks[labels > 0].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def readings(pred, target):
	"""The head's fit on rows it did not learn from (the prepass's): {nkp_auc, nkp_ev_<form>, nkp_ce, nkp_ce_uniform,
	nkp_top1_best, nkp_top1_choice, nkp_rows_*}; nan where nothing counts."""
	with torch.no_grad():
		out = {}
		scores, labels = [], []
		fp, ft = [], []
		for name in ("move", "travel"):
			p = pred[f"nkp_{name}"].reshape(pred[f"nkp_{name}"].shape[0], HO, -1, OUT).float()
			legal = target[f"{name}_legal"]
			scores.append(p[..., 0][legal])
			labels.append(target[f"{name}_has"][legal].float())
			where = target[f"{name}_has"]
			fp.append(torch.sigmoid(p[..., 1 : 1 + len(FORMS)])[where])
			ft.append(target[f"{name}_forms"][where])
		scores, labels = torch.cat(scores), torch.cat(labels)
		out["nkp_auc"] = auc(scores.double(), labels)
		out["nkp_positives"] = float(labels.mean()) if labels.numel() else float("nan")
		fp, ft = torch.cat(fp), torch.cat(ft)
		for i, form in enumerate(FORMS):
			if ft.shape[0] > 1 and float(ft[:, i].var()) > 0:
				out[f"nkp_ev_{form}"] = float(1.0 - (ft[:, i] - fp[:, i]).var() / ft[:, i].var())
			else:
				out[f"nkp_ev_{form}"] = float("nan")
		logits = plan_logits(pred, target)
		heroes = target["plan_heroes"]
		logp = torch.where(target["plan_allowed"], logits.log_softmax(dim=-1), torch.zeros_like(logits))
		ce = -(target["plan"] * logp).sum(dim=-1)
		options = target["plan_allowed"].sum(dim=-1).float()
		out["nkp_ce"] = float(ce[heroes].mean()) if heroes.any() else float("nan")
		out["nkp_ce_uniform"] = float(options[heroes].log().mean()) if heroes.any() else float("nan")
		top = logits.argmax(dim=-1)
		tasked = heroes & (target["plan"][..., -1] < 1.0)
		out["nkp_top1_best"] = float((top == target["plan_best"])[tasked].float().mean()) if tasked.any() else float("nan")
		chosen = target["choice"] >= 0
		out["nkp_top1_choice"] = float((top == target["choice"])[chosen].float().mean()) if chosen.any() else float("nan")
		out["nkp_heroes"], out["nkp_choices"] = float(heroes.sum()), float(chosen.sum())
		return out
