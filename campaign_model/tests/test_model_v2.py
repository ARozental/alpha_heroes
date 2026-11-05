"""The campaign model v2 (model_v2.py) and its foresight (foresight.py): the contract, the export, every part learning,
the labels of a game, an update with them."""
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.foresight import battle_score, game_labels
from campaign_model.model import CORRECTION_STEP, CampaignModel, build_model
from campaign_model.graphs import window_budget
from campaign_model.model_v2 import ACTION_TARGETS, AREAS, CampaignModelV2, Sides, windows_of
from campaign_model.records import Decision, Game
from campaign_model.schema import HEADS, OUTPUTS, random_inputs
from campaign_model.tests.test_train import decisions_for
from campaign_model.train import chosen_logp, choices, load_optimizer, make_optimizer, ppo_update


def tensors(x):
	return {k: torch.from_numpy(v) for k, v in x.items()}


def rich_inputs(batch, seed):
	"""random_inputs with what they lack: artifacts, skills, spells, luck, an exchange's other army, heroes for hire."""
	x = random_inputs(batch, seed)
	rng = np.random.default_rng(seed)
	x["h_artifacts"][:, :4, :3] = rng.integers(7, 100, (batch, 4, 3))
	x["h_skills"][:, :4, :4] = rng.integers(0, 4, (batch, 4, 4))
	x["h_spells"][:, :4, 10:20] = rng.random((batch, 4, 10)) < 0.3
	x["h_luck"][:, :4] = rng.integers(-1, 3, (batch, 4))
	x["x_creature"][:, :2], x["x_count"][:, :2] = rng.integers(0, 140, (batch, 2)), rng.integers(1, 30, (batch, 2))
	x["g_tavern"][:, :2], x["g_tavern_level"][:, :2] = rng.integers(0, 150, (batch, 2)), 1
	x["g_tavern_army_creature"][:, :, 0], x["g_tavern_army_count"][:, :, 0] = rng.integers(0, 140, (batch, 2)), 5
	x["g_owned_creature"][:, 0], x["g_owned_count"][:, 0] = 3, 4
	x["g_dwelling_creature"][:, 0], x["g_dwelling_count"][:, 0] = 5, 6
	x["d_creature"][:], x["d_count"][:] = 7, 3
	for b in range(batch):
		if x["g_kind"][b] == c.DecisionKind.TURN:
			x["hire_mask"][b, 0, 0] = True
			x["cast_mask"][b, 0, 0] = True
			x["buy_mask"][b, 0, 0] = True
			x["army_mask"][b, 0, 0, c.OWN_HEROES, c.ArmyAmount.ONE] = True  # a hero's stack into the town it visits
	return x


@pytest.fixture(scope="module")
def model():
	torch.manual_seed(0)
	return CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()


def test_the_outputs_are_the_contract_and_each_table_a_distribution(model):
	x = rich_inputs(6, seed=1)
	with torch.no_grad():
		out = model(tensors(x))
	for spec in OUTPUTS:
		assert out[spec.name].shape == (6,) + spec.shape, spec.name
	for b in range(6):
		if x["g_kind"][b] == c.DecisionKind.TURN:
			assert abs(float(out["type_logp"][b].exp().sum()) - 1) < 1e-5
			for action, (name, mask) in HEADS.items():
				if x[mask][b].any():
					assert abs(float(out[name][b].exp().sum()) - 1) < 1e-4, name
					assert torch.isneginf(out[name][b][~torch.from_numpy(x[mask][b])]).all(), name


def test_build_model_makes_v2_from_its_kwargs_and_v1_without():
	assert isinstance(build_model({"architecture": "v2", "dim": 32, "cell_dim": 8, "layers": 1}), CampaignModelV2)
	assert not isinstance(build_model({"dim": 32, "cell_dim": 16, "layers": 1}), CampaignModelV2)


def test_a_rivals_signed_half_starts_flipped_and_ours_as_it_is():
	sides = Sides(8)
	v = torch.ones(3, 8)
	out = sides(v, torch.tensor([c.OWNER_SELF, c.OWNER_SELF + 2, c.OWNER_NONE]))
	assert out[0].tolist() == [1.0] * 8 and out[2].tolist() == [1.0] * 8
	assert out[1].tolist() == [-1.0] * 4 + [1.0] * 4


def test_the_export_matches_torch(tmp_path):
	from campaign_model.export import export, verify

	torch.manual_seed(2)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	with torch.no_grad():  # (every zero-initialised layer given weights: every path checked)
		for p in model.parameters():
			if float(p.abs().sum()) == 0:
				p.normal_(0, 0.05)
	path = export(model, tmp_path / "v2.onnx")
	assert verify(model, path, seeds=(1, 2), batch_sizes=(1, 3)) < 1e-4


def test_every_part_learns_after_its_zero_starts_move():
	"""After a few steps (the zero-initialised outputs no longer 0), every block's parameters get a gradient from the
	policy's, the values' and the foresight's outputs: no path cut off."""
	torch.manual_seed(3)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	model.plan_outputs = True  # (as the trainer's: train.ppo_update)
	x = tensors(rich_inputs(8, seed=4))
	x["h_backpack"][:, :2, 0] = 9  # (a carried artifact: encode_hero.carried reads the backpack's)
	optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

	weights = {}

	def loss():  # (each table's log-probabilities weighted at random: the mean of a uniform table's has no gradient)
		out = model(x)
		# (fs_plan too: its outputs start at 0, where a square has no gradient)
		# (and the battle heads', 2026-11-04, alike)
		starts_at_0 = ("fs_plan", "fs_tiles", "fs_outcome")
		tables = sum((v.clamp(min=-30) * weights.setdefault(k, torch.randn(v.shape, generator=torch.Generator().manual_seed(len(k))))).mean() for k, v in out.items() if k.endswith("_logp") or k in starts_at_0)
		return tables + out["value"].mean() + out["growth_logit"].mean() + out["end_logit"].mean() + sum(v.pow(2).mean() for k, v in out.items() if k.startswith("fs_") and k not in starts_at_0 and v.requires_grad)

	for _ in range(3):
		optimizer.zero_grad()
		loss().backward()
		optimizer.step()
	optimizer.zero_grad()
	loss().backward()
	required = (
		"encode_hero.", "encode_town.", "encode_tiles.", "encode_global.", "encode_options.", "encode_other.", "reach.", "travel_days.", "sides.", "tile_sides.",
		"player.", "players.", "join_", "stamp_", "unet.", "read_tiles.", "area.", "blocks.", "norm.", "pair_", "foresight.", "again_", "move_pair.", "travel_",
		"move_out.", "move_mix", "act_", "build_q.", "building.", "recruit_head.", "hire_", "army_src.", "army_dst.", "army_out.", "trade_head.", "cast_q.", "spell.",
		"buy_q.", "buy_item.", "decision.", "option_context.", "end_out.", "type_bias", "value_hidden.", "value_heads.", "growth_heads.", "end_head.",
	)
	dead = [name for name, p in model.named_parameters() if name.startswith(required) and (p.grad is None or float(p.grad.abs().sum()) == 0)]
	# (embedding rows of ids absent from these inputs get none: what must learn is every block's dense part)
	dead = [name for name in dead if not any(e in name for e in ("hero_type", "klass", "artifact", "skill", "creature", "quantity", "faction", "subtype", "terrain", "road", "object", "sound", "kind", "last", "player.weight", "building", "spell.weight", "buy_item.weight", "primary", "resource"))]
	assert not dead, sorted({".".join(name.split(".")[:3]) for name in dead})


def test_an_update_with_foresight_labels_runs_and_reports_them():
	torch.manual_seed(5)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	x = rich_inputs(8, seed=6)
	decisions = decisions_for(model.eval(), x)
	n = len(decisions)
	foresight = {
		"move": np.random.default_rng(0).normal(size=(n, ACTION_TARGETS)).astype(np.float32), "move_mask": np.array([d.action_type == c.ActionType.MOVE for d in decisions]),
		"move_battle": np.ones(n, bool), "move_at": np.array([(d.index // (c.WINDOW * c.WINDOW), (d.index // c.WINDOW) % c.WINDOW, d.index % c.WINDOW) if d.action_type == c.ActionType.MOVE else (0, 0, 0) for d in decisions], np.int64),
		"move_win": np.random.default_rng(2).integers(0, 2, n).astype(np.float32), "move_stake": np.random.default_rng(3).random(n).astype(np.float32),
		"move_offer": np.random.default_rng(4).integers(0, 2, n).astype(bool),
		"answer": np.zeros((n, ACTION_TARGETS), np.float32), "answer_mask": np.array([d.action_type < 0 for d in decisions]), "answer_battle": np.zeros(n, bool),
		"option": np.array([max(d.index, 0) if d.action_type < 0 else 0 for d in decisions], np.int64),
		"hero": np.zeros((n, c.OWN_HEROES, 2), np.float32), "hero_mask": x["h_present"][:, : c.OWN_HEROES].copy(), "hero_alive": x["h_present"][:, : c.OWN_HEROES].copy(),
		"turn": np.zeros((n, 1 + c.N_RESOURCES), np.float32), "turn_mask": np.ones((n, 1 + c.N_RESOURCES), bool),
		"plan": np.random.default_rng(1).integers(0, AREAS + 1, (n, c.OWN_HEROES)), "plan_mask": x["h_present"][:, : c.OWN_HEROES].copy(),
	}
	targets = np.full((n, c.N_VALUE_OFFSETS), 0.5, np.float32)
	before = [p.detach().clone() for p in model.parameters()]
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, targets, np.zeros(n, np.float32), CampaignConfig(batch_size=4, passes=1, model_architecture="v2"), decode=lambda rows: {k: v[rows] for k, v in x.items()}, foresight=foresight)
	assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
	assert metrics["foresight_turn"] >= 0 and metrics["foresight_hero_alive"] > 0
	assert metrics["grad_shared_foresight"] > 0 and -1 <= metrics["grad_shared_cosine"] <= 1
	# The plan's cross-entropy (its outputs start at 0: uniform over the areas and none) and each term's gradient norm on
	# the shared layers: the plan's 0 at the check (an update's first micro-batch) until its outputs move
	assert model.plan_outputs and abs(metrics["foresight_plan"] - math.log(AREAS + 1)) < 0.05
	assert all(metrics[f"grad_raw_{term}"] >= 0 for term in ("policy", "value", "entropy", "foresight")) and metrics["grad_raw_plan"] == 0  # (at weight 1: the learner weighs them, grad_shared_*)


def battle(won=True, side=0, casualties=((0, 5), (1, 20)), result=0):
	"""Our 20 pikemen (side) against 20 of creature 1; we lose casualties[0], they lose casualties[1]."""
	ours, theirs = [[0, 20]], [[1, 20]]
	armies = [ours, theirs] if side == 0 else [theirs, ours]
	lost = [[list(casualties[0])], [list(casualties[1])]] if side == 0 else [[list(casualties[1])], [list(casualties[0])]]
	return {"serial": 1, "day": 1, "side": side, "armies": armies, "heroes": [None, None], "casualties": lost, "winner": side if won else 1 - side, "result": result, "exp": [0, 0]}


def test_the_battle_score_is_the_battle_rewards_formula():
	from game_values import creature_value

	ours, theirs = 20 * creature_value(0), 20 * creature_value(1)
	expected = (theirs - 5 * creature_value(0)) / (ours + theirs)  # they lost everything, we a quarter
	assert battle_score(battle()) == pytest.approx(expected)
	assert battle_score(battle(side=1)) == pytest.approx(expected)  # the same from the defender's seat
	assert battle_score(battle(won=False, casualties=((0, 20), (1, 4)))) == pytest.approx((4 * creature_value(1) - ours) / (ours + theirs))


def test_a_moves_window_takes_the_questions_and_battles_it_led_to():
	"""RED: a MOVE (serial 1) that raised a question (serial 2), answered with a fight; then a BUILD (serial 3); END;
	the next day one turn decision. The MOVE's window ends at the BUILD and holds the battle; so does the answer's."""
	W = c.WINDOW
	d = lambda day, kind, action, index: Decision("RED", day, kind, 0, 1, action, index, -1, 0.0, (0.5,) * 3, 0, b"")  # noqa: E731
	moves = [
		d(1, c.DecisionKind.TURN, c.ActionType.MOVE, 1 * W * W + 3 * W + 4),
		d(1, c.DecisionKind.YES_NO, -1, 0),
		d(1, c.DecisionKind.TURN, c.ActionType.BUILD, 5),
		d(1, c.DecisionKind.TURN, c.ActionType.END, -1),
		d(2, c.DecisionKind.TURN, c.ActionType.END, -1),
	]
	game = Game(decisions=moves)
	game.battles = {"RED": [dict(battle(), serial=2)]}
	n = len(moves)
	x = {"h_present": np.zeros((n, c.N_HEROES), bool), "h_owner": np.zeros((n, c.N_HEROES), np.int64), "h_type": np.full((n, c.N_HEROES), -1), "g_resources": np.full((n, c.N_RESOURCES), 10)}
	x["h_present"][:, 0], x["h_owner"][:, 0], x["h_type"][:, 0] = True, c.OWNER_SELF, 7
	x["h_present"][4, 0] = False  # gone by the next day
	phi_parts = np.zeros((n, 10))
	phi_parts[2, 0] = 2000.0  # the window's resources: +2,000
	labels = game_labels(game, x, phi_parts, np.full((n, c.N_HEROES), 500.0))
	assert labels["move_mask"][0] and labels["move_battle"][0] and tuple(labels["move_at"][0]) == (1, 3, 4)
	assert labels["move"][0, 0] == 1.0 and labels["move"][0, 2] == pytest.approx(np.log1p(2.0))
	assert labels["answer_mask"][1] and labels["answer_battle"][1] and labels["option"][1] == 0
	assert not labels["move_mask"][2] and not labels["answer_mask"][2]  # a BUILD: no head of its own
	assert labels["hero_mask"][0, 0] and not labels["hero_alive"][0, 0]  # our hero was gone the next day
	assert labels["turn_mask"][0].all()


def test_the_memory_carries_what_a_later_decision_needs():
	"""Memory alone, through backpropagation over 4 steps: a bit shown only at the first step is predicted at the last
	(~100% with the memory, a coin toss with it held at 0); and its read starts at 0 (the model as without memory)."""
	from campaign_model.model_v2 import Memory

	def run(memory_on, seed=0):
		torch.manual_seed(seed)
		dim, steps, batch = 16, 4, 64
		memory, head = Memory(dim, heads=2, slots=4, width=8), torch.nn.Linear(dim, 1)
		optimizer = torch.optim.Adam([*memory.parameters(), *head.parameters()], lr=1e-2)
		valid = torch.ones(batch, 2, dtype=torch.bool)
		accuracy = 0.0
		for _ in range(300):
			bit = torch.randint(0, 2, (batch,)).float()
			state = torch.zeros(batch, 4, 8)
			for t in range(steps):
				tokens = torch.randn(batch, 2, dim) * 0.1
				if t == 0:
					tokens[:, 0, 0] += 2 * bit - 1
				tokens = memory.read(tokens, state)
				state = memory.write(tokens, valid, state) if memory_on else state
			logit = head(tokens[:, 1]).squeeze(-1)
			loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, bit)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			accuracy = 0.9 * accuracy + 0.1 * float(((logit > 0).float() == bit).float().mean())
		return accuracy

	assert run(True) > 0.9
	assert run(False) < 0.7
	fresh = Memory(16, heads=2, slots=4, width=8)
	tokens = torch.randn(3, 5, 16)
	assert torch.equal(fresh.read(tokens, torch.rand(3, 4, 8)), tokens)


def test_the_change_heads_start_at_no_change_whatever_the_players():
	from campaign_model.model import build_model

	torch.manual_seed(0)
	model = build_model({"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"}).eval()
	x = {k: torch.from_numpy(v) for k, v in random_inputs(4, seed=2).items()}
	x["g_players"] = torch.tensor([2, 3, 4, 8])
	with torch.no_grad():
		out = model(x)
	assert torch.allclose(out["value"], torch.full_like(out["value"], 0.5))  # (a share's would be 1/2, 1/3, 1/4, 1/8)


def test_the_move_discount_starts_as_the_old_score_and_moves_only_the_move_scores(model):
	"""features.move_discount (CustomFeatures.move_discount_power): a checkpoint without it loads (v2 and v1) with it at
	0, the model as it was; its gradient there is not 0; away from 0 it changes the MOVE options' scores and the action
	type's (it reads them), nothing else."""
	x = tensors(rich_inputs(6, seed=3))
	old = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	old.load_state_dict({k: v for k, v in model.state_dict().items() if k != "features.move_discount"})
	assert float(old.features.move_discount.detach()) == 0 and float(old.features.move_discount_power().detach()) == 0
	v1 = CampaignModel(dim=32, cell_dim=16, layers=1)
	v1.load_state_dict({k: v for k, v in v1.state_dict().items() if k != "features.move_discount"})
	with torch.no_grad():
		before = old(x)
		old.features.move_discount.fill_(1 / CORRECTION_STEP)  # (its power 1.5 tanh 1: ~1.14)
		after = old(x)
	changed = {k for k in before if torch.is_tensor(before[k]) and not torch.equal(before[k], after[k])}
	assert "move_logp" in changed and changed <= {"move_logp", "type_logp"}, changed
	with torch.no_grad():
		old.features.move_discount.zero_()
	logp = old(x)["move_logp"].flatten(1)
	rows = torch.isfinite(logp).any(dim=1)
	assert rows.any()
	logp[rows].max(dim=1).values.sum().backward()
	grad = old.features.move_discount.grad
	assert grad is not None and torch.isfinite(grad).all() and float(grad.abs()) > 0


def test_adams_moments_carry_over_to_a_model_that_gained_a_parameter():
	"""train.load_optimizer: a checkpoint's optimizer from before features.move_discount keeps every other parameter's
	moments, matched by name (the new one sits among them, not at the end), and the new one starts fresh."""
	config = CampaignConfig(learning_rate=1e-3)
	torch.manual_seed(0)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	optimizer = make_optimizer(model, config)
	for p in model.parameters():
		p.grad = torch.randn_like(p)
	optimizer.step()
	names = {id(p): n for n, p in model.named_parameters()}
	order = [names[id(p)] for group in optimizer.param_groups for p in group["params"]]
	gone = order.index("features.move_discount")
	assert gone < len(order) - 1
	saved = optimizer.state_dict()
	place = {i: i - (i > gone) for i in range(len(order)) if i != gone}  # each parameter's place in the old model's
	old = {
		"state": {place[i]: s for i, s in saved["state"].items() if i != gone},
		"param_groups": [{**g, "params": [place[i] for i in g["params"] if i != gone]} for g in saved["param_groups"]],
	}
	fresh = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	fresh_optimizer = make_optimizer(fresh, config)
	load_optimizer(fresh_optimizer, fresh, old, [k for k in model.state_dict() if k != "features.move_discount"])
	trained = dict(model.named_parameters())
	for name, p in fresh.named_parameters():
		state = fresh_optimizer.state.get(p, {})
		if name == "features.move_discount":
			assert not state
		else:
			assert torch.equal(state["exp_avg"], optimizer.state[trained[name]]["exp_avg"]), name
			assert torch.equal(state["exp_avg_sq"], optimizer.state[trained[name]]["exp_avg_sq"]), name


def test_the_plan_head_changes_nothing_the_ai_reads(model):
	"""model_v2.plan_of (the trainer's model: plan_outputs): fs_plan [B, HO, AREAS + 1] beside outputs that are
	bit-identical to a server's (plan_outputs off): the AI reads none of it."""
	x = tensors(rich_inputs(6, seed=7))
	with torch.no_grad():
		served = model(x)
		model.plan_outputs = True
		trained = model(x)
		del model.plan_outputs
	assert "fs_plan" not in served and trained["fs_plan"].shape == (6, c.OWN_HEROES, AREAS + 1)
	assert all(torch.equal(served[k], trained[k]) for k in served if torch.is_tensor(served[k]))


def test_a_checkpoint_without_the_plan_head_loads_it_as_a_new_model_starts_it():
	"""model.ADDED_HEADS: a checkpoint from before foresight.plan* loads (strict) with the loading model's own fresh
	weights there (not 0: a SwiGLU at 0 never gets a gradient), everything else the checkpoint's."""
	torch.manual_seed(0)
	saved = CampaignModelV2(dim=32, cell_dim=8, layers=1).state_dict()
	old = {k: v for k, v in saved.items() if not k.startswith("foresight.plan")}
	assert len(old) < len(saved)
	torch.manual_seed(1)
	fresh = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	own = {k: v.clone() for k, v in fresh.state_dict().items()}
	fresh.load_state_dict(old)
	for k, v in fresh.state_dict().items():
		assert torch.equal(v, own[k] if k.startswith("foresight.plan") else saved[k]), k
	assert float(fresh.foresight.plan.up.weight.abs().sum()) > 0  # (its SwiGLU at its fresh weights; its outputs start at 0)


def test_the_plan_labels_are_its_targets_travel_areas():
	"""foresight.plan_labels: each of our hero slots' target as its TRAVEL area (features.area_index's), none (-1) as
	AREAS; a task without a tile (-2), a slot not ours and a decision without a plan are no label."""
	from types import SimpleNamespace

	from campaign_model import features as ft
	from campaign_model.foresight import plan_labels
	from campaign_model.records import PLAN_ENTRY

	n, ho = 3, c.OWN_HEROES
	x = {
		"g_map": np.array([[36, 36, 2], [72, 72, 2], [36, 36, 2]], np.int64),
		"h_present": np.zeros((n, c.N_HEROES), bool), "h_owner": np.full((n, c.N_HEROES), c.OWNER_SELF, np.int64),
	}
	x["h_present"][:, :4] = True
	x["h_owner"][:, 3] = c.OWNER_SELF + 1  # (a slot that is not ours)
	plan = np.zeros((ho, PLAN_ENTRY))
	plan[:, 0] = -1
	plan[0, :3], plan[1, :3], plan[2, 0], plan[3, :3] = (5, 7, 0), (35, 30, 1), -2, (1, 1, 0)
	decisions = [SimpleNamespace(plan=tuple(plan.ravel())), SimpleNamespace(plan=tuple(plan.ravel())), SimpleNamespace(plan=None)]
	labels = {"plan": np.zeros((n, ho), np.int64), "plan_mask": np.zeros((n, ho), bool)}
	plan_labels(SimpleNamespace(decisions=decisions), x, labels)
	pos = torch.tensor(np.stack([plan[:, :3]] * n), dtype=torch.long)
	areas = ft.area_index({"g_map": torch.from_numpy(x["g_map"])}, pos).numpy()
	for i in (0, 1):
		assert labels["plan"][i, 0] == areas[i, 0] and labels["plan"][i, 1] == areas[i, 1] and labels["plan_mask"][i, :2].all()
		assert not labels["plan_mask"][i, 2] and not labels["plan_mask"][i, 3]  # (no tile; not ours)
		assert labels["plan"][i, 4] == AREAS and not labels["plan_mask"][i, 4]  # (none, but no hero in the slot)
	assert labels["plan"][0, 0] == (0 * c.TRAVEL_GRID + 7 // 2) * c.TRAVEL_GRID + 5 // 2  # (36 x 36: areas 2 tiles a side)
	assert labels["plan"][1, 0] == (0 * c.TRAVEL_GRID + 7 // 4) * c.TRAVEL_GRID + 5 // 4  # (72 x 72: 4)
	assert not labels["plan_mask"][2].any()


def test_a_window_budget_takes_our_heroes_first_then_distinct_absent_slots():
	"""(2026-11-03) windows_of's budget, the CUDA graphs' windows: every slot of ours first, in order, then absent ones
	(not real), all distinct (dense() places them all); every slot without a budget or with one past the slots."""
	present = torch.zeros(2, c.N_HEROES, dtype=torch.bool)
	present[0, [1, 3]] = True
	present[1, 0] = True
	present[1, c.OWN_HEROES :] = True  # (others' heroes: no windows)
	windows, n, real = windows_of({"h_present": present}, fixed=True, budget=5)
	assert windows.tolist()[:3] == [1, 3, c.OWN_HEROES] and n == 5 and real.tolist() == [True] * 3 + [False] * 2
	assert len(set(windows.tolist())) == 5
	for budget in (None, 2 * c.OWN_HEROES):
		windows, n, real = windows_of({"h_present": present}, fixed=True, budget=budget)
		assert windows.tolist() == list(range(2 * c.OWN_HEROES)) and n == 2 * c.OWN_HEROES
	windows, n, real = windows_of({"h_present": present}, budget=5)  # (no budget without fixed windows: ours alone)
	assert windows.tolist() == [1, 3, c.OWN_HEROES] and n == 3


def test_a_window_budget_gives_the_exact_windows_outputs_and_gradients():
	"""(2026-11-03) The budget's windows (ours and absent slots to spare, or exactly ours) against our heroes' alone:
	the same outputs and gradients (the PC's update computes them so, train.windows_for)."""
	torch.manual_seed(0)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).train()
	m.plan_outputs = True
	x = tensors(rich_inputs(6, 5))
	heroes = int(x["h_present"][:, : c.OWN_HEROES].sum())
	assert 0 < heroes <= 6 * c.OWN_HEROES - 4

	def run(fixed, budget):
		m.zero_grad(set_to_none=True)
		with window_budget(m, budget):
			m.fixed_windows = fixed
			out = m(x)
		sum(v.float().clamp(-1e4, 1e4).sum() for v in out.values() if torch.is_tensor(v) and v.requires_grad).backward()
		return {k: v.detach() for k, v in out.items() if torch.is_tensor(v)}, {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}

	exact, exact_grads = run(False, None)
	for budget in (heroes, heroes + 4):
		out, grads = run(True, budget)
		assert m.window_budget is None  # (never left set)
		assert set(out) == set(exact) and set(grads) == set(exact_grads)
		for name in exact:
			torch.testing.assert_close(out[name], exact[name], rtol=1e-5, atol=1e-5, equal_nan=True)
		# (relative to the largest gradient, as graphs.StepGraph's check: the sums' order differs; a softmax's bias has
		# none but that rounding)
		scale = max(float(g.abs().max()) for g in exact_grads.values())
		worst = max(float((grads[name] - exact_grads[name]).abs().max()) for name in exact_grads)
		assert worst <= 1e-6 * scale, (worst, scale)
	m.fixed_windows = False


def test_the_window_budget_is_never_left_set():
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	with pytest.raises(RuntimeError):
		with window_budget(m, 7):
			assert m.window_budget == 7 and m.fixed_windows
			raise RuntimeError
	assert m.window_budget is None


def test_the_update_converting_in_its_graphs_falls_back_and_checks_its_gradient_as_without(monkeypatch):
	"""(2026-11-03) config.convert_in_graphs on the PC's path (the bytes as sent): the gradient check (v2's foresight)
	and the eager step a graph could not take convert the batch themselves. Eager stand-ins of the graphs whose step
	declines (the fallback), with the conversion in them and outside: the same weights after."""
	from campaign_model import train
	from campaign_model.tests.test_model import encoded
	from campaign_model.tests.test_train import _EagerGraphs, _EagerStep

	class Declining(_EagerStep):
		def run(self, inputs, batch, budget=None, prepare=None):
			return None

	monkeypatch.setattr(train, "converts_on_device", lambda device: True)
	monkeypatch.setattr(train, "Graphs", _EagerGraphs)
	monkeypatch.setattr(train, "StepGraph", Declining)
	torch.manual_seed(5)
	start = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	x = rich_inputs(8, seed=6)
	decisions = [d.__class__(**{**d.__dict__, "inputs": encoded(x, b)}) for b, d in enumerate(decisions_for(start.eval(), x))]
	n = len(decisions)
	foresight = {
		"move": np.random.default_rng(0).normal(size=(n, ACTION_TARGETS)).astype(np.float32), "move_mask": np.array([d.action_type == c.ActionType.MOVE for d in decisions]),
		"move_battle": np.ones(n, bool), "move_at": np.array([(d.index // (c.WINDOW * c.WINDOW), (d.index // c.WINDOW) % c.WINDOW, d.index % c.WINDOW) if d.action_type == c.ActionType.MOVE else (0, 0, 0) for d in decisions], np.int64),
		"move_win": np.random.default_rng(2).integers(0, 2, n).astype(np.float32), "move_stake": np.random.default_rng(3).random(n).astype(np.float32),
		"move_offer": np.random.default_rng(4).integers(0, 2, n).astype(bool),
		"answer": np.zeros((n, ACTION_TARGETS), np.float32), "answer_mask": np.array([d.action_type < 0 for d in decisions]), "answer_battle": np.zeros(n, bool),
		"option": np.array([max(d.index, 0) if d.action_type < 0 else 0 for d in decisions], np.int64),
		"hero": np.zeros((n, c.OWN_HEROES, 2), np.float32), "hero_mask": x["h_present"][:, : c.OWN_HEROES].copy(), "hero_alive": x["h_present"][:, : c.OWN_HEROES].copy(),
		"turn": np.zeros((n, 1 + c.N_RESOURCES), np.float32), "turn_mask": np.ones((n, 1 + c.N_RESOURCES), bool),
		"plan": np.random.default_rng(1).integers(0, AREAS + 1, (n, c.OWN_HEROES)), "plan_mask": x["h_present"][:, : c.OWN_HEROES].copy(),
	}
	weights, metrics = {}, {}
	for inside in (True, False):
		model = CampaignModelV2(dim=32, cell_dim=8, layers=1)
		model.load_state_dict(start.state_dict())
		torch.manual_seed(7)
		config = CampaignConfig(batch_size=4, micro_batches=2, passes=1, model_architecture="v2", cuda_graphs=True, step_graphs=True, convert_in_graphs=inside, decode_threads=1)
		metrics[inside], _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, np.full((n, c.N_VALUE_OFFSETS), 0.5, np.float32), np.zeros(n, np.float32), config, foresight=foresight)
		weights[inside] = {k: v.detach().clone() for k, v in model.state_dict().items()}
	assert metrics[True]["grad_shared_foresight"] > 0 and metrics[True]["grad_shared_foresight"] == metrics[False]["grad_shared_foresight"]
	assert any(not torch.equal(weights[True][k], start.state_dict()[k]) for k in weights[True])
	for k in weights[True]:
		torch.testing.assert_close(weights[True][k], weights[False][k], rtol=0, atol=0, msg=k)


def test_a_graphs_outputs_are_compared_relative_to_their_scale_and_masks():
	"""(2026-11-03) graphs._difference, a first-pass graph's check: relative to the output's largest magnitude (at least
	1); a masked (-inf) entry where the other has a number, or another shape, is a difference without bound."""
	from campaign_model.graphs import _difference

	a = torch.tensor([[0.5, -float("inf"), 100.0]])
	assert _difference(a, a.clone()) == 0.0
	assert abs(_difference(a + torch.tensor([[0.0, 0.0, 0.05]]), a) - 0.05 / 100.0) < 1e-6  # (float32)
	assert abs(_difference(torch.tensor([0.2]), torch.tensor([0.1])) - 0.1) < 1e-6  # (scale at least 1)
	assert _difference(torch.tensor([[0.5, 0.0, 100.0]]), a) == float("inf")
	assert _difference(a, a.reshape(3, 1)) == float("inf")


def test_the_fight_features_tell_a_hopeless_fight_from_an_even_one():
	"""(2026-11-03, the user: the strength ratio as a feature of the (hero, tile) pair) model_v2.fight_features: no
	fight 0; an even fight 1, ratio 0, not hopeless; one 54x our strength (joint7's near-empty heroes' median) hopeless."""
	from types import SimpleNamespace

	from campaign_model.model_v2 import fight_features

	ratio = torch.tensor([[[0.0, 0.0, math.log(1 / 54)]]])
	fights = torch.tensor([[[0.0, 1.0, 1.0]]])
	f = fight_features(SimpleNamespace(tile_fights=fights, tile_fight_ratio=ratio))[0, 0]
	assert f[0].tolist() == [0.0, 0.0, 0.0]
	assert f[1].tolist() == [1.0, 0.0, 0.0]
	assert f[2, 0] == 1 and f[2, 1] == pytest.approx(math.log(1 / 54) / 3) and f[2, 2] == pytest.approx(math.log(54) - 1)


def test_the_fight_layer_moves_only_the_fight_tiles_move_logits():
	"""model_v2.move_fight (0 to begin with): once its 'hopeless' weight is negative, MOVE's learned logit falls on the
	tiles where a fight starts, by how hopeless, and nowhere else."""
	from campaign_model.model import dense

	torch.manual_seed(3)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = tensors(rich_inputs(4, 7))
	with torch.no_grad():
		known = m.domain_knowledge(x)
		fights = dense(known.tile_fights, known.windows, known.n, 4)
		hopeless = dense(torch.relu(-known.tile_fight_ratio * known.tile_fights - 1.0).clamp(max=6.0) * known.tile_fights, known.windows, known.n, 4)
		before = m(x)["move_logp"].clone()
		m.move_fight.weight.data[0, 2] = -1.0
		after = m(x)["move_logp"]
	assert float(fights.sum()) > 0 and float(hopeless.sum()) > 0  # (the inputs have fights, some hopeless)
	# (MOVE's table is one log-softmax a decision: each legal tile moves by the decision's constant, the fight tiles by
	# -hopeless besides)
	legal = x["move_mask"] & torch.isfinite(before) & torch.isfinite(after)
	for i in range(4):
		rest = ((after - before) + hopeless)[i][legal[i]]
		if rest.numel():
			assert torch.allclose(rest, rest[0].expand_as(rest), atol=1e-4), i
	assert float((hopeless * legal).sum()) > 0  # (some legal tile was hopeless: the test saw the effect)


def test_a_checkpoint_without_the_fight_layers_loads_them_at_0():
	"""model.ADDED_HEADS: a checkpoint from before pair_fight and move_fight loads (strict) with them as a new model starts
	them, their outputs 0: the same outputs as the model that saved it."""
	torch.manual_seed(0)
	saving = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	old = {k: v for k, v in saving.state_dict().items() if not k.startswith(("pair_fight", "move_fight"))}
	assert len(old) < len(saving.state_dict())
	loading = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	loading.load_state_dict(old)
	x = tensors(rich_inputs(3, 2))
	with torch.no_grad():
		a, b = saving(x), loading(x)
	for name in a:
		if torch.is_tensor(a[name]):
			torch.testing.assert_close(a[name], b[name], equal_nan=True)


# ---- 2026-11-04: the user's review of MOVE's features (strength estimates, relative features, the mixer, battle heads) ----

def test_a_guard_is_credited_with_the_object_it_blocks_on_a_blocked_tile():
	"""features.blocked_by_guards: VCMI marks every object's tile blocked (joint7: 3,302 of 3,302), so the region of passable
	unreachable tiles never held a value and 'behind' was 0 on every tile; now an unreachable object's value enters the
	region from its own tile and reaches the monster through its zone. A reachable object counts for no guard."""
	from campaign_model import features as ft

	W = c.WINDOW
	value, region, zone, monster = (torch.zeros(1, W, W) for _ in range(4))
	monster[0, 16, 16] = 1.0
	zone[0, 15:18, 15:18] = 1.0
	zone[0, 16, 16] = 0.0
	region[0, 10:15, 13:20] = 1.0  # north of the zone, unreachable and passable
	value[0, 11, 16] = 2000.0  # an object (on a blocked tile, not one of the region's) inside it
	behind = ft.blocked_by_guards(value, region.bool(), zone, monster)
	assert float(behind[0, 16, 16]) == pytest.approx(2000.0)
	assert float(behind.sum()) == pytest.approx(2000.0)  # (on the monster's tile only)
	assert float(ft.blocked_by_guards(torch.zeros(1, W, W), region.bool(), zone, monster).sum()) == 0.0  # (nothing behind it)


def test_vcmis_damage_modifier_and_the_damage_race():
	"""features.damage_modifier: +5% an attack point over the defense (at most +300%), -2.5% a point under (at most -70%);
	race_strength in log1p(army strength)'s units: twice the army is ~log 2 stronger, an empty army hopeless."""
	from campaign_model import features as ft

	d = torch.tensor([0.0, 10.0, 60.0, 100.0, -10.0, -28.0, -50.0])
	assert ft.damage_modifier(d).tolist() == pytest.approx([1.0, 1.5, 4.0, 4.0, 0.75, 0.3, 0.3])
	one = lambda v: torch.tensor([float(v)])  # noqa: E731
	even = ft.race_strength(one(1000), one(100), one(5), one(5), one(1000), one(100), one(5), one(5))
	double = ft.race_strength(one(2000), one(200), one(5), one(5), one(1000), one(100), one(5), one(5))
	empty = ft.race_strength(one(0), one(0), one(0), one(0), one(1000), one(100), one(5), one(5))
	assert float(even) == pytest.approx(0.0, abs=1e-6)
	assert float(double) == pytest.approx(math.log(2.0), abs=1e-3)
	assert float(empty) < -5.0
	# our attack 10 over their defense: our damage x1.5, our race up by log(1.5) / 2
	stronger = ft.race_strength(one(1000), one(100), one(15), one(5), one(1000), one(100), one(5), one(5))
	assert float(stronger) == pytest.approx(0.5 * math.log(1.5), abs=1e-3)


def test_a_heros_spell_damage_is_its_best_spell_as_its_mana_allows():
	"""features.spell_damage: the best of the damage spells it knows at its spell power, cast at most SPELL_ROUNDS times as
	its mana allows, a round's share; 0 without a spell or the mana for one."""
	from types import SimpleNamespace

	from campaign_model import features as ft

	x = {"h_spells": torch.zeros(1, 3, c.N_SPELLS), "h_primary": torch.zeros(1, 3, 4), "h_mana": torch.zeros(1, 3)}
	x["h_spells"][0, 0, 17] = 1  # Lightning Bolt: 10 + 25 x power, 10 mana
	x["h_primary"][0, 0, 2], x["h_mana"][0, 0] = 5, 30
	x["h_spells"][0, 1, 15] = x["h_spells"][0, 1, 18] = 1  # Magic Arrow (10 + 10 x power, 5) and Implosion (100 + 75 x power, 30)
	x["h_primary"][0, 1, 2], x["h_mana"][0, 1] = 2, 40
	x["h_spells"][0, 2, 17] = 1  # Lightning Bolt without the mana for it
	x["h_mana"][0, 2] = 9
	damage = ft.spell_damage(x, SimpleNamespace())[0]
	assert float(damage[0]) == pytest.approx(10 + 25 * 5)  # 3 casts over 3 rounds
	assert float(damage[1]) == pytest.approx(max((100 + 75 * 2) * 1, (10 + 10 * 2) * 3) / 3)  # Implosion once beats 3 Arrows
	assert float(damage[2]) == 0.0


def test_the_strength_estimates_are_the_main_armys_for_our_strongest_hero_and_zero_without_a_fight():
	"""model.tile_features' move_strength (four estimates x this hero, our main army, each's soft view): nothing where no
	fight starts; the same for this hero and the main army in our strongest hero's window, the main army's no weaker in
	the others (army estimate); hopeless (features.hopeless, smooth) only where a fight starts, from the kindest of this
	hero's estimates."""
	torch.manual_seed(0)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = tensors(rich_inputs(4, 7))
	with torch.no_grad():
		known = m.domain_knowledge(x)
	s, r = known.move_strength, known.move_relative
	fights = known.tile_fights > 0
	assert bool(fights.any()) and s.shape[-1] == 8 and r.shape[-1] == 6
	assert float(s[~fights].abs().sum()) == 0.0 and float(r[..., 0][~fights].abs().sum()) == 0.0
	strength = known.hero_strength[:, : c.OWN_HEROES]
	own = x["h_present"][:, : c.OWN_HEROES] & (x["h_owner"][:, : c.OWN_HEROES] == c.OWNER_SELF)
	strongest = torch.where(own, strength, torch.full_like(strength, -1e9)).argmax(dim=1)
	for k, w in enumerate(known.windows.tolist()):
		if not bool(known.real[k]) or not bool(fights[k].any()):
			continue
		b, slot = divmod(w, c.OWN_HEROES)
		this, main = s[k][fights[k]][:, :4], s[k][fights[k]][:, 4:]
		if slot == int(strongest[b]):
			torch.testing.assert_close(this, main)
		else:
			assert bool((main[:, 0] >= this[:, 0] - 1e-5).all())
	# (move_strength is the estimates' soft view, features.strength_view: hopeless reads them before it, so compare where
	# the view inverts well)
	from campaign_model import features as ft

	raw = 2.0 * torch.atanh((s[..., :4] / 2.0).clamp(-0.999, 0.999))
	hopeless = ft.hopeless(raw.amax(dim=-1)) * fights
	inside = (s[..., :4].abs() < 1.9).all(dim=-1)
	torch.testing.assert_close(r[..., 0][inside], hopeless[inside], atol=1e-3, rtol=1e-3)


def test_the_mixer_starts_as_nk2s_rule_on_hopeless_tiles_only():
	"""model_v2's mixer: its hidden layer's output starts at 0 and its direct path at HOPELESS_LOGIT x hopeless, so the
	fresh model's MOVE logits differ from the same model without that weight by exactly that, on the hopeless tiles."""
	from campaign_model.model import dense
	from campaign_model.model_v2 import HOPELESS, HOPELESS_LOGIT

	torch.manual_seed(3)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = tensors(rich_inputs(4, 7))
	with torch.no_grad():
		known = m.domain_knowledge(x)
		hopeless = dense(known.move_relative[..., 0], known.windows, known.n, 4)
		with_rule = m(x)["move_logp"].clone()
		m.move_mix_direct.weight[0, HOPELESS] = 0.0
		without = m(x)["move_logp"]
	assert float(hopeless.sum()) > 0
	legal = x["move_mask"] & torch.isfinite(with_rule) & torch.isfinite(without)
	for i in range(4):
		rest = ((with_rule - without) - HOPELESS_LOGIT * hopeless)[i][legal[i]]
		if rest.numel():
			assert torch.allclose(rest, rest[0].expand_as(rest), atol=1e-4), i


def test_an_old_checkpoint_loads_into_the_new_move_inputs_with_the_same_outputs():
	"""model_v2.migrate_move_inputs: a state dict from before 2026-11-04 (move_pair's inputs: the pairs, the foresight,
	MOVE_FEATURES and the prior again) loads (strict) with move_pair's columns moved: from a model whose new columns are 0,
	the dropped features' and behind's weights 0 and the prior's split over its two copies, the same outputs (the mixer's
	rule aside)."""
	from campaign_model import model_v2

	torch.manual_seed(5)
	source = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	base = source.pair_dim + model_v2.FORESIGHT_HIDDEN
	with torch.no_grad():
		source.move_pair.up.weight[:, base + len(model_v2.MOVE_KEPT):] = 0.0  # (new inputs, the mixer's: 0 in a migrated model)
		source.move_pair.up.weight[:, base + model_v2.MOVE_KEPT.index(MOVE_FEATURES_INDEX("behind"))] = 0.0
		source.move_mix_direct.weight.zero_()
	new = source.move_pair.up.weight.detach()
	old = torch.zeros(new.shape[0], base + model_v2.OLD_MOVE_KNOWN)
	old[:, :base] = new[:, :base]
	for to, frm in enumerate(model_v2.MOVE_KEPT):
		old[:, base + frm] = new[:, base + to]
	value = base + MOVE_FEATURES_INDEX("value")
	old[:, value], old[:, base + model_v2.OLD_MOVE_KNOWN - 1] = 0.25 * new[:, base], 0.75 * new[:, base]  # (two copies of one input)
	state = {k: v for k, v in source.state_dict().items() if not k.startswith(("pair_strength", "move_mix", "foresight.tiles", "foresight.outcome"))}
	state["move_pair.up.weight"] = old
	loaded = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	loaded.load_state_dict(state)
	with torch.no_grad():
		loaded.move_mix_direct.weight.zero_()  # (the rule aside)
		x = tensors(rich_inputs(3, 2))
		a, b = source(x), loaded(x)
	for name in ("move_logp", "type_logp", "value", "fs_move"):
		torch.testing.assert_close(a[name], b[name], equal_nan=True)


def MOVE_FEATURES_INDEX(name):
	from campaign_model.model import MOVE_FEATURES

	return MOVE_FEATURES.index(name)


def test_adams_moments_of_move_pair_move_with_its_columns():
	"""train.fit_moments: moments loaded for move_pair's first layer from before 2026-11-04 (another shape: the first step
	would fail) are moved as its weights are (no folding: the first copy's), any other misfit started fresh."""
	from campaign_model import model_v2
	from campaign_model.train import fit_moments

	model = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	optimizer = make_optimizer(model, CampaignConfig())
	p = model.move_pair.up.weight
	base = model.pair_dim + model_v2.FORESIGHT_HIDDEN
	old = torch.arange(p.shape[0] * (base + model_v2.OLD_MOVE_KNOWN), dtype=torch.float32).reshape(p.shape[0], -1)
	optimizer.state[p] = {"step": torch.tensor(5.0), "exp_avg": old.clone(), "exp_avg_sq": old.clone() ** 2}
	q = model.act_move.weight
	optimizer.state[q] = {"step": torch.tensor(5.0), "exp_avg": torch.zeros(3), "exp_avg_sq": torch.zeros(3)}
	fit_moments(optimizer, model)
	state = optimizer.state[p]
	assert state["exp_avg"].shape == p.shape and state["exp_avg_sq"].shape == p.shape
	torch.testing.assert_close(state["exp_avg"][:, :base], old[:, :base])
	for to, frm in enumerate(model_v2.MOVE_KEPT):
		expected = 0.0 * old[:, base + frm] if frm == MOVE_FEATURES_INDEX("behind") else old[:, base + frm]
		torch.testing.assert_close(state["exp_avg"][:, base + to], expected)
	assert float(state["exp_avg"][:, base + len(model_v2.MOVE_KEPT):].abs().sum()) == 0.0
	assert q not in optimizer.state  # (a misfit with no migration: fresh)


def test_the_battle_heads_labels_and_loss():
	"""foresight: a MOVE a battle followed is labelled with its outcome (won, the stake's share lost: a loss takes it
	all) and with the engine's answer, never ours (the user, 2026-11-04): a wandering monster that asked us first (it
	offered to join, or fled: a yes/no question about a monster) offered, whatever we answered; the battle heads' loss
	pushes a plain tile toward 'no monster', a monster's toward 'attacks or offers', a chosen one toward what it did."""
	from campaign_model.foresight import LABEL_NAMES, battle_heads_loss, stake_lost

	assert stake_lost(battle()) == pytest.approx(5 * game_values_creature(0) / (20 * game_values_creature(0)))
	assert stake_lost(battle(won=False, casualties=((0, 20), (1, 4)))) == 1.0
	W = c.WINDOW
	d = lambda day, kind, action, index: Decision("RED", day, kind, 0, 1, action, index, -1, 0.0, (0.5,) * 3, 0, b"")  # noqa: E731
	step = 1 * W * W + 3 * W + 4
	moves = [
		d(1, c.DecisionKind.TURN, c.ActionType.MOVE, step),  # attacked at once (serial 1)
		d(1, c.DecisionKind.TURN, c.ActionType.MOVE, step), d(1, c.DecisionKind.YES_NO, -1, 0),  # asked first: we fought (serial 3)
		d(1, c.DecisionKind.TURN, c.ActionType.MOVE, step), d(1, c.DecisionKind.YES_NO, -1, 1),  # asked first: we let it go
		d(1, c.DecisionKind.TURN, c.ActionType.END, -1), d(2, c.DecisionKind.TURN, c.ActionType.END, -1),
	]
	game = Game(decisions=moves)
	game.battles = {"RED": [dict(battle(won=False, casualties=((0, 20), (1, 4))), serial=1), dict(battle(), serial=3)]}
	n = len(moves)
	x = {"h_present": np.zeros((n, c.N_HEROES), bool), "h_owner": np.zeros((n, c.N_HEROES), np.int64), "h_type": np.full((n, c.N_HEROES), -1), "g_resources": np.full((n, c.N_RESOURCES), 10), "d_object": np.full(n, -1)}
	x["d_object"][[2, 4]] = 54  # (the questions about a wandering monster)
	labels = game_labels(game, x, np.zeros((n, 10)), np.full((n, c.N_HEROES), 500.0))
	assert set(labels) == LABEL_NAMES
	assert labels["move_battle"][0] and not labels["move_offer"][0] and labels["move_win"][0] == 0.0 and labels["move_stake"][0] == 1.0
	assert labels["move_offer"][1] and labels["move_battle"][1] and labels["move_win"][1] == 1.0
	assert labels["move_offer"][3] and not labels["move_battle"][3]  # (the engine's answer: offered, whatever ours)
	# the loss: 3 decisions; a monster at (0, 3, 3) and (0, 1, 1); chosen: (0, 1, 1) attacked, (0, 1, 1) offered, (0, 2, 2) plain
	b = 3
	out = {"fs_tiles": torch.zeros(b, c.OWN_HEROES, W, W, 3, requires_grad=True), "fs_outcome": torch.zeros(b, c.OWN_HEROES, W, W, 2, requires_grad=True),
		"fs_stack_tile": torch.zeros(b, c.OWN_HEROES, W, W), "fs_move_legal": torch.zeros(b, c.OWN_HEROES, W, W)}
	out["fs_move_legal"][:, 0, :5, :5] = 1.0
	out["fs_stack_tile"][:, 0, 3, 3] = out["fs_stack_tile"][:, 0, 1, 1] = 1.0
	t = {"move_mask": torch.tensor([True, True, True]), "move_battle": torch.tensor([True, True, False]), "move_offer": torch.tensor([False, True, False]),
		"move_win": torch.tensor([1.0, 0.0, 0.0]), "move_stake": torch.tensor([0.2, 1.0, 0.0])}
	every, at = torch.arange(b), torch.tensor([[0, 1, 1], [0, 1, 1], [0, 2, 2]])
	tiles, won, stake = battle_heads_loss(out, t, every, at)
	assert all(bool(torch.isfinite(v)) for v in (tiles, won, stake))
	(tiles + won + stake).backward()
	g = out["fs_tiles"].grad
	assert float(g[0, 0, 0, 0, 0]) < 0 and float(g[0, 0, 3, 3, 0]) > 0  # a plain tile toward 'no monster', a monster's away from it
	assert float(g[0, 0, 1, 1, 1]) < 0 and float(g[1, 0, 1, 1, 2]) < 0  # the chosen: 'attacks' where it did, 'offers' where it asked
	assert float(g[2, 0, 2, 2, 0]) < 0  # a chosen plain tile: 'no monster'
	assert float(out["fs_outcome"].grad[0, 0, 1, 1, 0]) < 0 and float(out["fs_outcome"].grad[2].abs().sum()) == 0.0  # (no battle: no outcome)


def game_values_creature(creature):
	from game_values import creature_value

	return creature_value(creature)


def test_nk2s_safety_rule_lowers_a_hopeless_tiles_prior_by_its_weight():
	"""model.CustomFeatures.safety_weight (6 at the start, 0..12): MOVE's prior on a tile whose fight is hopeless by every
	estimate is lower by safety x hopeless, and nowhere else; a checkpoint from before it loads with it at its start."""
	torch.manual_seed(3)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = tensors(rich_inputs(4, 7))
	assert float(m.features.safety_weight()) == pytest.approx(6.0)
	with torch.no_grad():
		known = m.domain_knowledge(x)
		with_rule = known.tile_prior.clone()
		hopeless = known.move_relative[..., 0]
		m.features.safety.fill_(-1e3)  # (its weight at 0)
		without = m.domain_knowledge(x).tile_prior
	assert float(hopeless.sum()) > 0
	torch.testing.assert_close(with_rule - without, -6.0 * hopeless, atol=1e-4, rtol=1e-4)
	state = {k: v for k, v in CampaignModelV2(dim=32, cell_dim=8, layers=1).state_dict().items() if k != "features.safety"}
	loaded = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	loaded.load_state_dict(state)
	assert float(loaded.features.safety_weight()) == pytest.approx(6.0)


def test_a_trade_unlocks_a_building_only_if_it_buys_its_last_missing_resource_and_keeps_the_rest():
	"""features.trade_options (the user, 2026-11-04: every trade is dumb unless it buys what could not be bought without
	it): at one marketplace VCMI sells a crystal for 20 wood (IMarket::getOffer at efficiency (1 + 1) / 20), which costs
	Φ 20 x 250 - 500; with 30 wood, a building short of one crystal alone (and needing 5 wood) is unlocked by it; one that
	needs 26 wood too is not (the trade leaves 10); selling wood for gold (25 a wood) unlocks neither."""
	from types import SimpleNamespace

	from campaign_model import features as ft

	R, F = c.N_RESOURCES, c.N_TRADE_AMOUNTS
	WOOD, CRYSTAL, GOLD = 0, 4, 6
	tables = torch.zeros(c.FACTION_VOCAB + 1, c.N_BUILDINGS, R)
	tables[0, 5, WOOD], tables[0, 5, CRYSTAL] = 5, 1  # building 5: 5 wood, 1 crystal
	tables[0, 6, WOOD], tables[0, 6, CRYSTAL] = 26, 1  # building 6: 26 wood, 1 crystal
	creatures = torch.zeros(c.CREATURE_VOCAB + 1, R)
	creatures[0, GOLD], creatures[0, 5] = 3000, 1  # an Angel: 3,000 gold and a gem
	t = SimpleNamespace(resource_price=torch.tensor([250.0, 500, 250, 500, 500, 500, 1]), building_resources=tables, creature_resources=creatures)
	x = {"g_markets": torch.tensor([1]), "g_resources": torch.zeros(1, R, dtype=torch.long), "t_faction": torch.full((1, c.N_TOWNS), -1), "build_short": torch.zeros(1, c.OWN_TOWNS, c.N_BUILDINGS, dtype=torch.bool),
		"t_available_creature": torch.full((1, c.N_TOWNS, c.N_LEVELS), -1), "t_available": torch.zeros(1, c.N_TOWNS, c.N_LEVELS, dtype=torch.long)}
	x["g_resources"][0, WOOD] = 30
	x["t_faction"][0, 0] = 0
	x["build_short"][0, 0, 5] = True
	options = ft.trade_options(x, t)[0]  # [R, R, F, 2]
	assert options.shape == (R, R, F, 2)
	torch.testing.assert_close(options[WOOD, CRYSTAL, :, 0], torch.full((F,), 4500.0))
	assert options[WOOD, CRYSTAL, :, 1].tolist() == [1.0] * F
	assert float(options[WOOD, GOLD, :, 1].sum()) == 0.0
	torch.testing.assert_close(options[WOOD, GOLD, 0, 0], torch.tensor(3 * 250 - 3 * 25.0))  # (10% of 30 trades: 3)
	x["build_short"][0, 0, 5], x["build_short"][0, 0, 6] = False, True
	assert float(ft.trade_options(x, t)[0, ..., 1].sum()) == 0.0
	# a creature for hire we cannot afford one of (an Angel: no gem), with 3,000 gold: the wood for a gem unlocks it
	x["g_resources"][0, GOLD] = 3000
	x["t_available_creature"][0, 0, 6], x["t_available"][0, 0, 6] = 0, 2
	options = ft.trade_options(x, t)[0]
	assert options[WOOD, 5, :, 1].tolist() == [1.0] * F and float(options[..., 1].sum()) == F


# ---- 2026-11-04: garrisons, a question's battle and join (schema 11) ----

def _strongest_and_cheapest():
	from campaign_model import vocab

	rows = {k: r for k, r in vocab.creatures().items() if 0 <= k < c.CREATURE_VOCAB and r.get("fight_value", 0) > 0}
	return max(rows, key=lambda k: rows[k]["fight_value"]), min(rows, key=lambda k: rows[k]["fight_value"])


def test_a_garrison_on_a_tile_is_a_fight_there_by_its_army():
	"""schema 11's a_*: a garrison not ours fights the hero that steps on it (VCMI's CGGarrison), so its tile is a fight
	tile with its army in the strength estimates: 40 of the strongest creatures against random heroes are hopeless."""
	strong, _ = _strongest_and_cheapest()
	x = rich_inputs(2, 3)
	b, hero = 0, 0
	px, py, pz = (int(v) for v in x["h_pos"][b, hero])
	x["a_present"][b, 0] = True
	x["a_pos"][b, 0] = (px + 1, py, pz)
	x["a_creature"][b, 0], x["a_count"][b, 0], x["a_detailed"][b, 0] = strong, 40, True
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	with torch.no_grad():
		known = m.domain_knowledge(tensors(x))
	k = [i for i, w in enumerate(known.windows.tolist()) if w == b * c.OWN_HEROES + hero and bool(known.real[i])][0]
	R = c.WINDOW_RADIUS
	assert float(known.tile_fights[k, R, R + 1]) == 1.0
	assert float(known.move_strength[k, R, R + 1, :4].max()) < -1.5 and float(known.move_relative[k, R, R + 1, 0]) > 0.3
	without = rich_inputs(2, 3)
	with torch.no_grad():
		plain = m.domain_knowledge(tensors(without))
	assert float(plain.tile_fights[k, R, R + 1]) == float(without["w_creature"].reshape(-1, c.WINDOW, c.WINDOW)[b * c.OWN_HEROES + hero, R, R + 1] >= 0)


def _question(x, b, creature, count, battle=None, join=None, d_object=16):
	x["d_hero"][b] = 0
	x["d_object"][b], x["d_subtype"][b] = d_object, 0
	x["d_guard_creature"][b, 0], x["d_guard_count"][b, 0], x["d_guard_detailed"][b] = creature, count, True
	x["option_mask"][b] = False
	x["option_mask"][b, :2] = True
	x["o_kind"][b, 0], x["o_kind"][b, 1] = int(c.OptionKind.YES), int(c.OptionKind.NO)
	x["o_battle"][b] = False
	x["o_join"][b] = False
	if battle is not None:
		x["o_battle"][b, battle] = True
	if join is not None:
		x["o_join"][b, join] = True


def test_a_banks_strong_guards_make_yes_unlikely_and_a_free_join_likely():
	"""model.CustomFeatures.option_knowledge: a bank's 'fight the guards?' (o_battle on YES) against 200 of the strongest
	creatures puts YES far below NO (the fight's value, NK2's safety rule); a wandering monster's free offer of 30 of
	them to join (o_join on YES, NO the battle) puts YES far above NO. The priors land on those options only."""
	strong, _ = _strongest_and_cheapest()
	torch.manual_seed(1)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = rich_inputs(3, 5)
	_question(x, 0, strong, 200, battle=0)
	_question(x, 1, strong, 30, join=0, battle=1, d_object=54)
	_question(x, 2, strong, 200)  # (the same guards, no option flagged: no prior)
	with torch.no_grad():
		known = m.domain_knowledge(tensors(x))
		out = m(tensors(x))
	prior = known.option_prior
	assert float(prior[0, 0]) < -5.0 and float(prior[0, 1]) == 0.0
	assert float(prior[1, 0]) > 3.0 and float(prior[1, 1]) < 0.0  # (the join; refusing it, the battle)
	assert float(prior[2].abs().max()) == 0.0
	p = out["option_logp"].exp()
	assert float(p[0, 0]) < 0.01 and float(p[1, 0]) > 0.99
	assert abs(float(p[2, 0]) - 0.5) < 0.2  # (unflagged: no prior moves it)


def test_a_monsters_question_counts_no_reward_wherever_it_is_asked():
	"""question_guards: a fleeing monster's offer (o_battle on YES) values the same whether d_object is the monster
	(an attack), our hero itself (a step into its zone: topObject's blocking fallback), or a town; a join counts none
	either; a bank's question still counts its reward (its object_gold, as often as we win)."""
	from battle_model.layers import vocab_index
	from campaign_model import features as ft

	_, weak = _strongest_and_cheapest()
	torch.manual_seed(1)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	with torch.no_grad():
		m.features.object_type_delta.weight[vocab_index(torch.tensor(ft.OBJ_HERO), c.OBJECT_VOCAB)] = -3000.0  # (a hero's object_gold well below 0)
	x = {k: np.repeat(v, 6, axis=0) for k, v in rich_inputs(1, 5).items()}  # (one state: the rows differ in the question only)
	for b, d_object in enumerate((ft.OBJ_MONSTER, ft.OBJ_HERO, ft.OBJ_TOWN)):
		_question(x, b, weak, 2, battle=0, d_object=d_object)
	_question(x, 3, weak, 2, join=0, battle=1, d_object=ft.OBJ_MONSTER)
	_question(x, 4, weak, 2, join=0, battle=1, d_object=ft.OBJ_HERO)
	_question(x, 5, weak, 2, battle=0, d_object=84)  # (a Crypt's guards)
	with torch.no_grad():
		prior = m.domain_knowledge(tensors(x)).option_prior
		gold = float(m.features.object_gold(torch.tensor([84]), torch.tensor([0]))[0])
		hero_gold = float(m.features.object_gold(torch.tensor([ft.OBJ_HERO]), torch.tensor([0]))[0])
	assert hero_gold < -1000
	fights = torch.cat([prior[:3, 0], prior[3:5, 1], prior[5:, 0]])  # (the option that fights: YES, or NO for a join)
	assert torch.allclose(fights[:3], fights[0].expand(3)) and torch.allclose(fights[3], fights[4])
	assert gold > 0 and float(fights[5]) > float(fights[0])  # (the same fight with a bank's reward on top)


def test_the_questions_nullkiller2_answers_yes_lean_yes_and_nothing_else_does():
	"""model_v2's answers: a dwelling's "Would you like to recruit?" (d_object 17, 20, 78 or 106, no option flagged) gets
	recruit_yes_weight on YES, a Quest Guard's "complete the quest?" and a border guard's "pass?" (215, 9) visit_yes_weight,
	4 each at the start, as Nullkiller2 answers them; an unflagged question at another object and a guarded dwelling's
	"fight the guards?" (o_battle on YES) get nothing from them."""
	strong, _ = _strongest_and_cheapest()
	torch.manual_seed(1)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = {k: np.repeat(v, 6, axis=0) for k, v in rich_inputs(1, 5).items()}
	_question(x, 0, 0, 0, d_object=17)  # (a dwelling's: no guards, no flags)
	_question(x, 1, 0, 0, d_object=47)  # (a School of Magic's)
	_question(x, 2, 0, 0, d_object=78)  # (a Refugee Camp's)
	_question(x, 3, strong, 5, battle=0, d_object=17)  # (a guarded dwelling's guards)
	_question(x, 4, 0, 0, d_object=215)  # (a Quest Guard's)
	_question(x, 5, 0, 0, d_object=9)  # (a border guard's)

	def yes_over_no():
		with torch.no_grad():
			logp = m(tensors(x))["option_logp"]
		return logp[:, 0] - logp[:, 1]

	with torch.no_grad():
		assert m.features.nk2_yes(tensors(x)).tolist() == pytest.approx([4.0, 0.0, 4.0, 0.0, 4.0, 4.0], abs=1e-4)
		starts = float(m.features.recruit_yes_weight()), float(m.features.visit_yes_weight())
	with_priors = yes_over_no()
	with torch.no_grad():
		m.features.recruit_yes.fill_(-1e4)  # (the weights at their floor, 0)
		m.features.visit_yes.fill_(-1e4)
	without = yes_over_no()
	assert max(abs(w - 4.0) for w in starts) < 1e-4
	assert torch.allclose(with_priors - without, torch.tensor([4.0, 0.0, 4.0, 0.0, 4.0, 4.0]), atol=1e-3)


def test_a_yes_names_what_it_does_and_a_no_does_not():
	"""encoders.OptionEncoder (the user, 2026-11-04: never one YES vector for every question): a YES's key names the object
	asked about, so YES at a Quest Guard and YES at a dwelling differ; NO, declining, is the same at both; the option that
	fights the question's guards holds them."""
	strong, _ = _strongest_and_cheapest()
	torch.manual_seed(1)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = {k: np.repeat(v, 3, axis=0) for k, v in rich_inputs(1, 5).items()}
	_question(x, 0, 0, 0, d_object=215)
	_question(x, 1, 0, 0, d_object=17)
	_question(x, 2, strong, 5, battle=0, d_object=17)
	with torch.no_grad():
		_, keys = m.encode_options(tensors(x), m.option_tables())
	assert not torch.allclose(keys[0, 0], keys[1, 0], atol=1e-3)
	assert torch.allclose(keys[0, 1], keys[1, 1])
	assert not torch.allclose(keys[1, 0], keys[2, 0], atol=1e-3)


def test_nullkiller2s_selection_and_purchase_priors_lean_where_it_goes():
	"""option_knowledge's Nullkiller2 rules (AIGateway::showBlockingDialog, recruitCreatures): a selection leans to its last
	component, not its NO; a Treasure Chest (gold, experience) to its gold under gold pressure and to its experience for
	the main hero with gold to spare; a dwelling's purchase to its largest amount."""
	T = c.ComponentType
	torch.manual_seed(1)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = {k: np.repeat(v, 4, axis=0) for k, v in rich_inputs(1, 5).items()}
	with torch.no_grad():
		assert bool(m.domain_knowledge(tensors(x)).main[0, 0])  # (our one hero: the main one)
	x["d_hero"][:] = 0
	x["g_resources"][:, 6], x["g_income"][:, 6] = 200_000, 0  # (gold to spare)
	x["g_resources"][1, 6] = 0  # (row 1: none, under pressure)
	x["option_mask"][:], x["o_battle"][:], x["o_join"][:] = False, False, False
	rows = [  # each row's options: kind, type, subtype, amount
		[(c.OptionKind.COMPONENT, T.PRIM_SKILL, 0, 1), (c.OptionKind.COMPONENT, T.PRIM_SKILL, 1, 1), (c.OptionKind.NO, 0, 0, 0)],
		[(c.OptionKind.COMPONENT, T.RESOURCE, 6, 1500), (c.OptionKind.COMPONENT, T.EXPERIENCE, 0, 1000)],
		[(c.OptionKind.COMPONENT, T.RESOURCE, 6, 1500), (c.OptionKind.COMPONENT, T.EXPERIENCE, 0, 1000)],
		[(c.OptionKind.AMOUNT, 12, 0, n) for n in (0, 5, 10, 20)],
	]
	for b, options in enumerate(rows):
		for o, (kind, what, sub, n) in enumerate(options):
			x["option_mask"][b, o] = True
			x["o_kind"][b, o], x["o_type"][b, o], x["o_subtype"][b, o], x["o_amount"][b, o] = int(kind), int(what), sub, n
	with torch.no_grad():
		prior = m.domain_knowledge(tensors(x)).option_prior
	picks = [int(prior[b, : len(options)].argmax()) for b, options in enumerate(rows)]
	assert picks == [1, 0, 1, 3], picks
	assert float(prior[0, 1]) == pytest.approx(3.0, abs=1e-3) and float(prior[3, 3]) == pytest.approx(2.0, abs=1e-3)


def test_a_purchases_fit_with_the_heros_army():
	"""model.CustomFeatures.recruit_fit: into an army of Pikemen (Castle), more Pikemen fit (no new slot, morale kept);
	Swordsmen keep its one faction but take a slot; Centaurs (Rampart) bring a second faction (+1 morale to 0); Skeletons
	(Necropolis, undead) a second faction and the undead among the living (+1 to -1); an amount of 0 buys nothing."""
	from campaign_model import vocab

	by_name = {row["name"]: k for k, row in vocab.creatures().items()}
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = {k: np.repeat(v, 5, axis=0) for k, v in rich_inputs(1, 5).items()}
	x["d_hero"][:] = 0
	x["h_army_creature"][:, 0] = -1
	x["h_army_count"][:, 0] = 0
	x["h_army_creature"][:, 0, 0], x["h_army_count"][:, 0, 0] = by_name["Pikeman"], 20
	x["option_mask"][:], x["o_battle"][:], x["o_join"][:] = False, False, False
	x["option_mask"][:, 0] = True
	x["o_kind"][:, 0] = int(c.OptionKind.AMOUNT)
	x["o_type"][:, 0] = [by_name["Pikeman"], by_name["Swordsman"], by_name["Centaur"], by_name["Skeleton"], by_name["Pikeman"]]
	x["o_amount"][:, 0] = [5, 5, 5, 5, 0]
	with torch.no_grad():
		fit = m.features.recruit_fit(tensors(x))[:, 0]  # (has, faction, morale change / 3, new slot, free after / 7)
	free = (c.ARMY_SLOTS - 1) / c.ARMY_SLOTS
	expected = torch.tensor([
		[1, 1, 0, 0, free], [0, 1, 0, 1, free - 1 / c.ARMY_SLOTS], [0, 0, -1 / 3, 1, free - 1 / c.ARMY_SLOTS],
		[0, 0, -2 / 3, 1, free - 1 / c.ARMY_SLOTS], [0, 0, 0, 0, 0],
	])
	assert torch.allclose(fit, expected, atol=1e-5), fit


def test_a_buildings_key_starts_as_before_and_can_tell_factions_apart():
	"""model_v2.build_scores: with build_identity at 0 (a fresh model's, and a migrated checkpoint's) BUILD's learned part
	is the old query . row exactly; once it moves, the same building id in towns of two factions scores apart."""
	torch.manual_seed(2)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = tensors(rich_inputs(1, 5))
	x["t_faction"][0, :2] = torch.tensor([0, 4])  # (Castle, Necropolis)
	towns = torch.randn(1, c.OWN_TOWNS, 32)
	towns[0, 1] = towns[0, 0]  # (the same query: only the identity tells them apart)
	with torch.no_grad():
		m.build_q.weight.normal_(0, 0.1)
		start = m.build_scores(towns, x)
		old = m.build_q(towns) @ m.building.weight.t() / math.sqrt(32)
		assert torch.allclose(start, old, atol=1e-6)
		m.build_identity.weight.normal_(0, 0.1)
		moved = m.build_scores(towns, x)
	dwelling = 30  # (the level 1 dwelling: Pikemen in Castle, Skeletons in Necropolis)
	assert abs(float(moved[0, 0, dwelling] - moved[0, 1, dwelling])) > 1e-3


def test_a_towns_visitor_is_our_hero_there_not_its_garrison_or_a_rival():
	"""model_v2.visitors: a town's visiting hero of ours (h_town, not h_garrisoned) is its token there; a garrisoned hero
	and a rival's hero in the same town are not."""
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = tensors(rich_inputs(1, 5))
	x["h_present"][0, :3] = True
	x["h_owner"][0, :3] = torch.tensor([c.OWNER_SELF, c.OWNER_SELF, c.OWNER_SELF + 1])
	x["h_town"][0, :3] = torch.tensor([0, 0, 0])
	x["h_garrisoned"][0, :3] = torch.tensor([False, True, False])

	class State:
		pass

	state = State()
	state.x, state.heroes = x, torch.randn(1, x["h_present"].shape[1], 32)
	seen = m.visitors(state)
	assert torch.allclose(seen[0, 0], state.heroes[0, 0]) and float(seen[0, 1:].abs().max()) == 0.0


def test_the_teachers_trade_advice_is_left_out_unless_asked_for():
	"""train.taught: Nullkiller2's TRADE advice is not taught (config.nk2_trade_advice False, the default), its other
	advice is; with the flag every advised decision counts."""
	from campaign_model.train import taught

	types = np.array([int(c.ActionType.TRADE), int(c.ActionType.BUILD), int(c.ActionType.TRADE), int(c.ActionType.MOVE)])
	advised = np.array([True, True, False, True])
	assert taught(types, advised, CampaignConfig()).tolist() == [False, True, False, True]
	assert taught(types, advised, CampaignConfig(nk2_trade_advice=True)).tolist() == advised.tolist()


def test_a_purchase_short_of_resources_is_priced_by_the_trades_it_makes():
	"""features.purchase_trades (the agent trades for a purchase short of resources: AlphaMapAI::payByTrades): a building
	short of 5 wood with gold to spare costs what the cheapest spare source of wood gives away beyond the wood's price
	(market_rates, as the agent trades); one we can afford costs nothing; a creature we cannot afford one of is short
	and priced the same way."""
	from campaign_model import features as ft

	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	t = m.features
	x = tensors(rich_inputs(1, 5))
	x["g_markets"][:] = 1
	x["g_resources"][0] = torch.tensor([5, 0, 0, 0, 0, 0, 10_000])
	faction = int(x["t_faction"][0, 0])
	short_one, enough = 30, 31  # (two buildings of the town's faction: costs set here)
	t.building_resources[faction, short_one] = torch.tensor([10.0, 0, 0, 0, 0, 0, 2000])
	t.building_resources[faction, enough] = torch.tensor([5.0, 0, 0, 0, 0, 0, 2000])
	x["build_short"][0] = False
	x["build_short"][0, 0, short_one] = x["build_short"][0, 0, enough] = True
	x["t_available_creature"][0, 0, 0] = 12  # (a creature priced here: 100 crystal)
	t.creature_resources[12] = torch.tensor([0.0, 0, 0, 0, 100, 0, 0])
	build, recruit, cannot = ft.purchase_trades(x, t)
	give, get = ft.market_rates(x, t)
	price = t.resource_price
	wood = min(float(price[s] * give[0, s, 0] / get[0, s, 0]) for s in range(7) if s != 0 and float(x["g_resources"][0, s]) - float(t.building_resources[faction, short_one, s]) >= float(give[0, s, 0]))
	assert float(build[0, 0, short_one]) == pytest.approx(5 * (wood - float(price[0])), rel=1e-5)
	assert float(build[0, 0, enough]) == 0.0
	assert bool(cannot[0, 0, 0]) and float(recruit[0, 0, 0]) > 0


def test_a_build_with_nothing_affordable_pays_its_least_trade_in_its_types_score():
	"""model_v2 (2026-11-05): a purchase short of resources pays its trades' loss of Φ at value_logit's price, in its entry
	and in its type's offer: with every legal building short, BUILD's score against END's falls by the charge (else, the
	table normalized, every entry paying it cancelled); with an affordable building legal too, it is as it was."""
	from campaign_model import features as ft

	torch.manual_seed(7)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = {k: np.repeat(v, 2, axis=0) for k, v in rich_inputs(1, 5).items()}
	x["g_kind"][:] = c.DecisionKind.TURN
	x["g_markets"][:] = 1
	x["g_resources"][:] = [5, 0, 0, 0, 0, 0, 10_000]
	faction = int(x["t_faction"][0, 0])
	m.features.building_resources[faction, 30] = torch.tensor([10.0, 0, 0, 0, 0, 0, 2000])  # (short of 5 wood)
	x["build_mask"][:], x["build_short"][:] = False, False
	x["build_mask"][:, 0, 30], x["build_short"][:, 0, 30] = True, True
	x["build_mask"][1, 0, 31] = True  # (row 1: an affordable building legal too)
	original = m.domain_knowledge

	def free_trades(x):
		known = original(x)
		known.build_trade = torch.zeros_like(known.build_trade)
		return known

	with torch.no_grad():
		lost = float(original(tensors(x)).build_trade[0, 0, 30])
		charged = m(tensors(x))["type_logp"]
		m.domain_knowledge = free_trades
		free = m(tensors(x))["type_logp"]
	B, END = int(c.ActionType.BUILD), int(c.ActionType.END)
	drop = (free[:, B] - free[:, END]) - (charged[:, B] - charged[:, END])
	weight = m.features.trade_cost_weight()
	assert lost > 1000 and float(drop[0]) == pytest.approx(float(weight * ft.value_logit(torch.tensor(lost))), rel=1e-4)
	assert abs(float(drop[1])) < 1e-5


def test_an_artifacts_move_is_priced_by_what_it_does_to_the_armies_the_heroes_lead():
	"""model.CustomFeatures.artifact_prior (schema 13): a +2 attack sword handed by a hero leading nothing to one leading
	an army gains (the army's multiplier rises); the other way it loses; worn from the backpack into an empty slot it
	gains; a combination's parts worn together assemble at its price over theirs."""
	from campaign_model import features as ft

	T = c.ArtifactTarget
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	t = m.features
	sword, part, combination = 7, 40, 129  # (Centaur's Axe and the tables set here)
	t.artifact_primary[sword] = torch.tensor([2.0, 0, 0, 0])
	t.artifact_slot[sword] = 3
	t.artifact_combination[part] = combination
	t.artifact_price[part], t.artifact_price[combination] = 1000.0, 9000.0
	x = {k: np.repeat(v, 2, axis=0) for k, v in rich_inputs(1, 5).items()}
	x["h_present"][:, :2], x["h_owner"][:, :2] = True, c.OWNER_SELF
	x["h_primary"][:, :2] = 0
	x["h_artifacts"][:, :2], x["h_backpack"][:, :2] = -1, -1
	x["h_army_creature"][:, :2], x["h_army_count"][:, :2] = -1, 0
	x["h_army_creature"][:, 1, 0], x["h_army_count"][:, 1, 0] = 12, 30  # (hero 1 leads an army, hero 0 none)
	x["h_artifacts"][0, 0, 3] = sword  # (row 0: hero 0 wears it; hero 1 wears it in row 1)
	x["h_artifacts"][1, 1, 3] = sword
	x["h_primary"][0, 0, 0] = x["h_primary"][1, 1, 0] = 2  # (h_primary counts what its items add)
	x["h_backpack"][1, 1, 0] = sword  # (and carries another in its backpack, slot 3 taken)
	x["h_artifacts"][0, 1, 5], x["h_artifacts"][0, 1, 6] = part, part  # (two parts worn by hero 1 in row 0)
	x["x_heroes"][:] = [1, 2]  # (an exchange of heroes 0 and 1)
	with torch.no_grad():
		prior = m.domain_knowledge(tensors(x)).artifact_prior  # [B, HO, P, T]
	cost = float(t.artifact_cost_weight())  # (a move must gain more than it: 5 at the start)
	assert float(prior[0, 0, 3, T.GIVE]) > 0.5 - cost  # (to the hero leading the army)
	assert float(prior[1, 1, 3, T.GIVE]) < -0.5 - cost  # (away from it)
	assert float(prior[1, 1, c.N_WORN_SLOTS, T.EQUIP]) == pytest.approx(-cost, abs=1e-4)  # (its slot holds the same sword: nothing gained)
	gained = float(prior[0, 1, 5, T.ASSEMBLE])
	assert gained == pytest.approx(float(ft.value_logit(torch.tensor(9000.0 - 2000.0))) - cost, rel=1e-4)


def test_an_artifact_move_trains_its_query_and_its_cost():
	"""train.chosen_logp of an exchange's ARTIFACT decision reaches the head's learned part (artifact_q at 0: its gradient
	is the hero's state against the artifact's row) and the prior's cost (features.artifact_cost, through ARTIFACT's offer
	to the type head); a decision that ends the exchange reaches both too (the type's score holds its best legal learned
	entry), the cost the other way."""
	torch.manual_seed(2)
	T = c.ArtifactTarget
	x = {k: np.repeat(v, 2, axis=0) for k, v in rich_inputs(1, 6).items()}
	x["h_present"][:, :2], x["h_owner"][:, :2] = True, c.OWNER_SELF
	x["artifact_mask"][:] = False
	x["artifact_mask"][:, 0, 0, T.GIVE] = True
	x["artifact_mask"][:, 1, c.N_WORN_SLOTS, T.EQUIP] = True
	x["x_heroes"][:] = [1, 2]
	ARTIFACT, END = int(c.ActionType.ARTIFACT), int(c.ActionType.END)
	grads = []
	for action in (ARTIFACT, END):
		torch.manual_seed(2)
		m = CampaignModelV2(dim=32, cell_dim=8, layers=1)
		out = m(tensors(x))
		choice = {"kind": torch.full((2,), int(c.DecisionKind.EXCHANGE)), "action": torch.full((2,), action), "index": torch.zeros(2, dtype=torch.long), "cell": torch.full((2,), -1)}
		logp, _ = chosen_logp(out, choice)
		(-logp.sum()).backward()
		grads.append((m.artifact_q.weight.grad.abs().max(), m.features.artifact_cost.grad))
	(q_taken, cost_taken), (q_ended, cost_ended) = grads
	assert float(q_taken) > 0 and float(q_ended) > 0
	assert float(cost_taken) > 0 > float(cost_ended)  # (descending: a move taken lowers the cost, one passed raises it)


def test_options_that_differ_never_share_an_identity():
	"""encoders.OptionEncoder has no hashed rows (features.bucket put 546 of 954 options on a shared row, 2026-11-04: NO with
	a scroll of spell 57, a skill with a creature's amount): those pairs differ, and so do two resources and two primary
	skills a component names."""
	torch.manual_seed(1)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = {k: np.repeat(v, 4, axis=0) for k, v in rich_inputs(1, 5).items()}
	x["option_mask"][:] = False
	x["option_mask"][:, :2] = True
	x["o_battle"][:], x["o_join"][:], x["o_amount"][:] = False, False, 0
	rows = [  # (kind, type, subtype) of each row's two options
		((c.OptionKind.NO, 0, 0), (c.OptionKind.COMPONENT, 6, 57)),
		((c.OptionKind.SKILL, 0, 1), (c.OptionKind.AMOUNT, 12, 0)),
		((c.OptionKind.COMPONENT, 2, 0), (c.OptionKind.COMPONENT, 2, 6)),  # (wood, gold)
		((c.OptionKind.COMPONENT, 0, 0), (c.OptionKind.COMPONENT, 0, 1)),  # (attack, defence)
	]
	for b, options in enumerate(rows):
		for o, (kind, what, sub) in enumerate(options):
			x["o_kind"][b, o], x["o_type"][b, o], x["o_subtype"][b, o] = int(kind), what, sub
	with torch.no_grad():
		_, keys = m.encode_options(tensors(x), m.option_tables())
	for b in range(len(rows)):
		assert float(F.cosine_similarity(keys[b, 0], keys[b, 1], dim=0)) < 0.99, rows[b]


def test_a_choices_logit_is_the_atanh_of_a_bounded_cosine():
	"""model_v2.cosine_logit: about 0 for random vectors (a fresh model's choices nearly even), atanh(COSINE_BOUND) (3.0)
	for a key along the query, its negative against it."""
	from campaign_model.model_v2 import COSINE_BOUND, cosine_logit

	torch.manual_seed(0)
	query, keys = torch.randn(64, 128), torch.randn(64, 4, 128)
	assert float(cosine_logit(query, keys).abs().mean()) < 0.15
	along = torch.stack([query, -query, 2 * query], dim=1)
	top = math.atanh(COSINE_BOUND)
	assert torch.allclose(cosine_logit(query, along), torch.tensor([top, -top, top]).expand(64, 3), atol=1e-3)


def test_a_level_ups_choice_teaches_the_heroes_own_skill_rows():
	"""The options' identities are the model's own vectors: a level-up's log-probability has a gradient on the hero
	encoder's (skill, level) rows it offers (no hero here has those skills: the options are their only way in)."""
	torch.manual_seed(1)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	x = rich_inputs(1, 5)
	x["h_skills"][0, :, 3], x["h_skills"][0, :, 7] = 0, 0
	x["d_hero"][0] = 0
	x["option_mask"][0] = False
	x["option_mask"][0, :2] = True
	x["o_kind"][0, :2], x["o_type"][0, :2], x["o_subtype"][0, :2] = int(c.OptionKind.SKILL), (3, 7), 1
	m(tensors(x))["option_logp"][0, 0].backward()
	grad = m.encode_hero.skill.weight.grad
	assert float(grad[3 * 4 + 1].abs().sum()) > 0 and float(grad[7 * 4 + 1].abs().sum()) > 0
	assert float(grad[5 * 4 + 1].abs().sum()) == 0  # (a skill nobody has and nothing offers)


def test_an_old_question_head_loads_with_its_fight_and_join_weights_exactly():
	"""model_v2.migrate_option_head: a checkpoint from before the question head's keys (option_q, answer_fight,
	answer_join) loads; the fight's and the join's weights and biases land in option_weights, giving each flagged option
	what answer_fight or answer_join gave it."""
	from campaign_model.model import RECRUIT_FIT

	torch.manual_seed(4)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	state = {k: v for k, v in m.state_dict().items() if not k.startswith(("option_weights.", "decision.", "option_context."))}
	fight, join = nn.Linear(6, 1), nn.Linear(3, 1)
	state.update({"option_q.weight": torch.randn(32, 64), "option_q.bias": torch.randn(32)})
	state.update({"answer_fight.weight": fight.weight.detach(), "answer_fight.bias": fight.bias.detach(), "answer_join.weight": join.weight.detach(), "answer_join.bias": join.bias.detach()})
	fresh = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	fresh.load_state_dict(state)
	f, j = torch.randn(5, 6), torch.randn(5, 3)
	ones, zeros = torch.ones(5, 1), torch.zeros(5, 1)
	with torch.no_grad():
		fit = torch.zeros(5, RECRUIT_FIT)  # (a purchase's fit after them: model.recruit_fit, at 0 here)
		fights = fresh.option_weights(torch.cat([f, torch.zeros(5, 3), ones, zeros, fit], dim=-1))
		takes = fresh.option_weights(torch.cat([torch.zeros(5, 6), j, zeros, ones, fit], dim=-1))
		assert torch.allclose(fights, fight(f), atol=1e-6) and torch.allclose(takes, join(j), atol=1e-6)


def test_adams_moments_survive_a_model_that_lost_parameters():
	"""train.load_optimizer: a checkpoint's optimizer from a model with parameters since removed (2026-11-04: option_q,
	in the middle of the order) keeps every remaining parameter's moments by name and drops the gone ones'."""
	config = CampaignConfig(learning_rate=1e-3)
	torch.manual_seed(0)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	optimizer = make_optimizer(model, config)
	for p in model.parameters():
		p.grad = torch.randn_like(p)
	optimizer.step()
	saved = optimizer.state_dict()
	keys = list(model.state_dict())
	rest = optimizer.param_groups[0]["params"]
	names = {id(p): n for n, p in model.named_parameters()}
	at = [names[id(p)] for p in rest].index("option_flags.weight")
	gone = {"option_q.weight": torch.randn(32, 64), "option_q.bias": torch.randn(32)}
	place = lambda i: i + len(gone) * (i >= at)  # noqa: E731  (each parameter's place among the old model's)
	old_state = {place(i): s for i, s in saved["state"].items()}
	old_state.update({at + k: {"step": torch.tensor(1.0), "exp_avg": v, "exp_avg_sq": v.abs()} for k, v in enumerate(gone.values())})
	groups = [{**saved["param_groups"][0], "params": list(range(len(rest) + len(gone)))}, {**saved["param_groups"][1], "params": [place(i) for i in saved["param_groups"][1]["params"]]}]
	where = keys.index("option_flags.weight")
	old_keys = keys[:where] + list(gone) + keys[where:]
	fresh = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	fresh_optimizer = make_optimizer(fresh, config)
	load_optimizer(fresh_optimizer, fresh, {"state": old_state, "param_groups": groups}, old_keys)
	trained = dict(model.named_parameters())
	for name, p in fresh.named_parameters():
		assert torch.equal(fresh_optimizer.state[p]["exp_avg"], optimizer.state[trained[name]]["exp_avg"]), name


def test_records_from_before_schema_11_read_with_no_guards_battle_join_or_armed_object():
	"""records: a record of schema 10 (before o_battle) and one of schema 9 (before mem) decode with zeros from where they
	end: no memory, no question guards, no flags, no armed object."""
	import zlib
	from types import SimpleNamespace

	from campaign_model.records import IN_BYTES, SHORT_RECORDS, decode_inputs
	from campaign_model.tests.test_model import encoded

	x = rich_inputs(2, 4)
	x["o_battle"][:, 0] = True
	x["a_present"][:, 0] = True
	x["a_count"][:, 0, 0] = 5
	x["artifact_mask"][:, 0, 0, 0] = True
	x["x_heroes"][:, :] = [1, 2]
	x["multi_mask"][:, 0, 0] = True
	full = zlib.decompress(encoded(x, 0))
	assert len(full) == IN_BYTES and len(SHORT_RECORDS) == 5
	lengths = sorted(SHORT_RECORDS)  # (records of schemas 9, 10, 11-12, 13, and 14 widened: each ends where the next's inputs start)
	for n in lengths[:4]:
		got = decode_inputs([SimpleNamespace(inputs=zlib.compress(full[:n]))])
		before_11 = ("o_battle", "o_join", "d_guard_count", "a_present", "a_count") if n < lengths[2] else ()
		before_14 = ("artifact_mask", "x_heroes") if n < lengths[3] else ()  # (schema 13's: none in any record before it)
		for name in before_11 + before_14 + ("multi_mask",):  # (schema 14's: none in any older record)
			assert not got[name].any(), (n, name)


def test_a_record_of_schema_14_reads_with_its_multi_mask_widened():
	"""records.widen_schema_14: a record of schema 14 ends with multi_mask at 4 kinds a town; decoded (both ways: the
	trainer's decode_wire, padded) its kinds keep their towns, kinds 4-7 and schema 15's armies read as none."""
	import zlib
	from types import SimpleNamespace

	from campaign_model.records import IN_LAYOUT, SCHEMA_14_BYTES, decode_inputs, padded
	from campaign_model.tests.test_model import encoded

	x = rich_inputs(1, 7)
	x["multi_mask"][:] = False
	x["multi_mask"][0, 0, c.MultiKind.RECRUIT_ALL] = x["multi_mask"][0, 3, c.MultiKind.GATHER] = x["multi_mask"][0, 1, c.MultiKind.GIVE_ALL] = True
	full = zlib.decompress(encoded(x, 0))
	start = sum(e[-1] for e in IN_LAYOUT[: [e[0] for e in IN_LAYOUT].index("multi_mask")])
	mask = np.frombuffer(full[start : start + c.OWN_TOWNS * c.N_MULTI], dtype=np.uint8).reshape(c.OWN_TOWNS, c.N_MULTI)
	old = full[:start] + mask[:, :4].tobytes()  # (schema 14's: its 4 kinds a town, nothing after)
	assert len(old) == SCHEMA_14_BYTES
	for got in (decode_inputs([SimpleNamespace(inputs=zlib.compress(old))]), {"multi_mask": np.frombuffer(padded(old), dtype=np.uint8)[start : start + c.OWN_TOWNS * c.N_MULTI].reshape(1, c.OWN_TOWNS, c.N_MULTI).astype(bool)}):
		assert got["multi_mask"][0].sum() == 3 and got["multi_mask"][0, 0, c.MultiKind.RECRUIT_ALL] and got["multi_mask"][0, 3, c.MultiKind.GATHER] and got["multi_mask"][0, 1, c.MultiKind.GIVE_ALL]
		assert not got["multi_mask"][0, :, 4:].any()
	got = decode_inputs([SimpleNamespace(inputs=zlib.compress(old))])
	assert not got["h_drop"].any() and not got["h_drop_kept"].any() and not got["x_take_count"].any()


def test_a_sequence_carries_the_memory_its_turn_decisions_leave_and_its_gradient_reaches_the_writer():
	"""model_v2.forward_steps: each sequence's step 1 starts from the memory its step 0 left (a TURN decision's output,
	else the memory step 0 started from: AlphaMapAI's rule), its outputs those of a plain forward with that memory; a loss
	on step 1 reaches the memory's writer through step 0."""
	from campaign_model.model_v2 import Memory

	torch.manual_seed(3)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	with torch.no_grad():  # (a trained model's stand-in: a fresh one's read and heads start at 0, nothing depends on its memory)
		for p in m.parameters():
			p.add_(0.05 * torch.randn_like(p))
	x = rich_inputs(4, 7)  # (2 sequences of 2 steps, step-major: rows 0 and 1 step 0 of each, rows 2 and 3 step 1)
	x["g_kind"][:] = [c.DecisionKind.TURN, c.DecisionKind.YES_NO, c.DecisionKind.TURN, c.DecisionKind.TURN]
	x["mem"][:] = np.random.default_rng(8).integers(-c.MEMORY_SCALE // 2, c.MEMORY_SCALE // 2, x["mem"].shape)
	t = tensors(x)
	out = m.forward_steps(t, 2)
	with torch.no_grad():
		first = m({k: v[:2] for k, v in t.items()})
		left = torch.stack([first["memory"][0].clamp(-1, 1), Memory.of(t)[1]])  # (the question writes nothing)
		second = m({**{k: v[2:] for k, v in t.items()}, "mem_float": left})
	for name in ("type_logp", "value_logit", "memory"):
		assert torch.allclose(out[name][:2], first[name], atol=1e-5) and torch.allclose(out[name][2:], second[name], atol=1e-5), name
	assert not torch.allclose(second["value_logit"][0], m({k: v[2:3] for k, v in t.items()})["value_logit"][0])  # (the recorded memory would differ)
	out["value_logit"][2].sum().backward()
	assert float(m.memory.candidate.weight.grad.abs().max()) > 0 and float(m.memory.gate.weight.grad.abs().max()) > 0


def test_a_pass_as_sequences_keeps_each_players_consecutive_records_together():
	"""train.sequence_order: each sequence is one player's consecutive records (another player's or a gap in the
	positions starts a new run), laid out step-major a micro-batch at a time, whole minibatches only, no row twice."""
	from campaign_model.train import sequence_order

	np.random.seed(4)
	stream = np.array([0] * 11 + [1] * 9 + [2] * 6)
	position = np.concatenate([np.delete(np.arange(12), 4), np.arange(9), np.arange(6)])  # (player 0's fifth record missing)
	mixed = np.random.permutation(len(stream))  # (the update's rows: the players' records interleaved)
	stream, position = stream[mixed], position[mixed]
	for steps, size, micro in ((2, 8, 2), (3, 6, 1)):
		order = sequence_order(stream, position, steps, size, micro)
		assert len(order) > 0 and len(order) % size == 0 and len(set(order.tolist())) == len(order)
		for block in order.reshape(-1, steps, size // micro // steps):  # (a micro-batch: its steps x its sequences)
			for s in range(block.shape[1]):
				rows = block[:, s]
				assert len(set(stream[rows].tolist())) == 1 and (np.diff(position[rows]) == 1).all(), (steps, rows)


def test_an_update_over_sequences_trains_the_memorys_writer_and_one_without_leaves_it():
	"""ppo_update with config.memory_steps 2 and the decisions' streams: the memory's writer learns (its write reaches
	the next decision's losses); with 1, as before, no loss reaches it."""
	torch.manual_seed(5)
	x = rich_inputs(16, 9)
	x["g_kind"][:] = c.DecisionKind.TURN
	x["mem"][:] = np.random.default_rng(10).integers(-c.MEMORY_SCALE // 2, c.MEMORY_SCALE // 2, x["mem"].shape)
	n = len(x["g_kind"])
	streams = (np.repeat([0, 1], n // 2), np.tile(np.arange(n // 2), 2))
	moved = {}
	for steps in (1, 2):
		torch.manual_seed(6)
		m = CampaignModelV2(dim=32, cell_dim=8, layers=1)
		with torch.no_grad():  # (a trained model's stand-in: a fresh one's read and heads start at 0)
			for p in m.parameters():
				p.add_(0.05 * torch.randn_like(p))
		decisions = decisions_for(m.eval(), x)
		before = m.memory.candidate.weight.detach().clone()
		config = CampaignConfig(batch_size=8, micro_batches=2, passes=1, memory_steps=steps, model_architecture="v2")
		metrics, _ = ppo_update(m, torch.optim.Adam(m.parameters(), lr=1e-3), decisions, np.full((n, c.N_VALUE_OFFSETS), 0.5, np.float32), np.zeros(n, np.float32), config, decode=lambda rows: {k: v[rows] for k, v in x.items()}, streams=streams)
		assert metrics["memory_steps"] == steps and metrics["steps"] > 0
		moved[steps] = float((m.memory.candidate.weight - before).abs().max())
	assert moved[1] == 0 and moved[2] > 0


def test_a_worn_artifact_and_a_carried_one_read_alike_until_the_carried_map_learns():
	"""encoders.HeroEncoder (2026-11-05): the backpack's artifacts pass through `carried` too (at 0): a sword worn and the
	same sword carried give the same hero vector at first, different ones once it moves; a checkpoint from before loads
	it at 0 (model.ADDED_HEADS)."""
	from campaign_model.encoders import hero_fields

	torch.manual_seed(4)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = {k: np.repeat(v, 2, axis=0) for k, v in rich_inputs(1, 5).items()}
	x["h_artifacts"][:, 0], x["h_backpack"][:, 0] = -1, -1
	x["h_artifacts"][0, 0, 3] = 7  # (row 0: worn)
	x["h_backpack"][1, 0, 0] = 7  # (row 1: carried)
	fields = hero_fields(tensors(x))
	with torch.no_grad():
		worn_first, _ = m.encode_hero(fields)
		assert torch.allclose(worn_first[0, 0], worn_first[1, 0], atol=1e-6)
		nn.init.normal_(m.encode_hero.carried.weight, std=0.1)
		learned, _ = m.encode_hero(fields)
		assert not torch.allclose(learned[0, 0], learned[1, 0], atol=1e-4)
	old = {k: v for k, v in m.state_dict().items() if not k.startswith("encode_hero.carried")}
	fresh = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	fresh.load_state_dict(old)
	assert float(fresh.encode_hero.carried.weight.abs().max()) == 0


def test_stepping_on_a_hero_or_a_town_of_ours_is_worth_what_the_army_gains_where_it_counts_most():
	"""model.CustomFeatures.meeting_value (2026-11-05): a hero of ours on a window tile is worth the best exchange there
	(features.led_values: the two armies joined under the stronger, or joined less their slowest creatures, which the
	other keeps) over the two apart, at least 0; a town of ours, for a main hero, its army joined with the garrison less
	the garrison at unled_share and what taking it adds to the town's expected loss; nothing elsewhere (the window's own
	hero's tile included). A hero holding only creatures that would slow ours is no meeting worth making: schema 15's
	first games met 308 times a game, a pair up to 6 times a day, on Φ v4's factors."""
	from campaign_model import features as ft
	from campaign_model import vocab

	key = {row["key"]: i for i, row in vocab.creatures().items()}
	angel, dwarf, crusader = key["core:angel"], key["core:dwarf"], key["core:crusader"]
	torch.manual_seed(9)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	t = m.features
	x = tensors({k: np.repeat(v, 1, axis=0) for k, v in rich_inputs(1, 5).items()})
	x["h_present"][:] = False
	x["h_present"][0, :2], x["h_owner"][0, :2], x["h_garrisoned"][0, :2] = True, c.OWNER_SELF, False
	x["h_pos"][0, 0] = torch.tensor([10, 10, 0])
	x["h_pos"][0, 1] = torch.tensor([12, 10, 0])  # (two tiles east of hero 0)
	x["h_level"][0, :2] = torch.tensor([6, 2])  # (hero 0 the stronger)
	x["h_primary"][0, 0], x["h_primary"][0, 1], x["h_artifacts"][0, :2] = torch.tensor([5, 5, 1, 1]), torch.tensor([0, 0, 1, 1]), -1  # (hero 0 makes more of an army)
	x["h_army_creature"][0, :2], x["h_army_count"][0, :2], x["h_detailed"][0, :2] = -1, 0, True
	x["h_army_creature"][0, 0, 0], x["h_army_count"][0, 0, 0] = angel, 5
	x["h_morale"][0, :2] = 1  # (one faction each: +1)
	x["t_present"][:] = False
	x["t_present"][0, 0], x["t_owner"][0, 0], x["t_visiting"][0, 0], x["t_garrison_hero"][0, 0] = True, c.OWNER_SELF, -1, -1
	x["t_pos"][0, 0] = torch.tensor([10, 13, 0])  # (three tiles south)
	x["t_garrison_creature"][0, 0], x["t_garrison_count"][0, 0], x["t_detailed"][0, 0] = -1, 0, True
	x["t_garrison_creature"][0, 0, 0], x["t_garrison_count"][0, 0, 0] = angel, 4  # (as fast: gathered, nothing slows)
	windows = torch.tensor([0])  # (decision 0, hero 0)
	offsets = torch.arange(c.WINDOW) - c.WINDOW_RADIUS
	dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
	tile_x, tile_y = 10 + dx, 10 + dy
	r = c.WINDOW_RADIUS

	def on_tile(pos, present):
		return (pos[:, None, None, :, 0] == tile_x[None, ..., None]) & (pos[:, None, None, :, 1] == tile_y[None, ..., None]) & (pos[:, None, None, :, 2] == 0) & present[:, None, None, :]

	def values(partner, count):
		x["h_army_creature"][0, 1, 0], x["h_army_count"][0, 1, 0] = partner, count
		with torch.no_grad():
			f = m.domain_knowledge(x)
			return t.meeting_value(x, f, torch.tensor([0]), windows, on_tile)[0], f

	def led(creatures, counts, h, f):
		rest = x["h_morale"][0, h].float() - ft.make_up_morales(x["h_army_creature"][0, h], x["h_army_count"][0, h].float(), t)
		return float(ft.led_values(torch.tensor(creatures), torch.tensor(counts, dtype=torch.float32), f.hero_multiplier[0, h], rest, t))

	pad = lambda ids, n: (list(ids) + [-1] * (n - len(ids)), None)  # noqa: E731
	value, f = values(angel, 3)  # (fast Angels of the same faction: joined under hero 0)
	alone = [led(pad([angel], 7)[0], [5.0] + [0.0] * 6, 0, f), led(pad([angel], 7)[0], [3.0] + [0.0] * 6, 1, f)]
	joined = led(pad([angel, angel], 14)[0], [5.0, 3.0] + [0.0] * 12, 0, f)
	gain = joined - sum(alone)
	assert gain > 0 and float(value[r, r + 2]) == pytest.approx(gain * float(t.gain(t.meeting_gain)), rel=1e-4)
	assert bool(f.main[0, 0])
	gathered = led(pad([angel, angel], 14)[0], [5.0, 4.0] + [0.0] * 12, 0, f) - alone[0] - t.unled_share * float(f.town_gold[0, 0]) - float(t.gather_loss(f)[0, 0])
	assert gathered > 0 and float(value[r + 3, r]) == pytest.approx(gathered * float(t.gain(t.meeting_gain)), rel=1e-4)
	assert float(value[r, r]) == 0.0 and float((value > 0).sum()) == 2
	slow, f = values(dwarf, 2)  # (Dwarves alone: joined they slow the Angels, kept apart nothing changes)
	assert float(slow[r, r + 2]) == pytest.approx(0.0, abs=1e-3)
	x["t_garrison_creature"][0, 0, 0], x["t_garrison_count"][0, 0, 0] = crusader, 10  # (speed 6: they would slow the Angels more than leading them adds)
	slowed, f = values(angel, 3)
	assert float(slowed[r + 3, r]) == pytest.approx(0.0, abs=1e-3)


def test_a_take_best_and_a_drop_are_priced_on_the_armies_they_make():
	"""Schema 15 (the user, 2026-11-05): DROP_SLOW of a hero leading five Angels and two Dwarves (speed 3: the army at
	1500 a day) into the garrison it visits gains what the Angels' movement and morale win back, the Dwarves at
	unled_share in the garrison; disbanded, less their worth. TAKE_BEST is priced on the armies the C++ makes (x_take_*:
	the receiver's led by it, the giver's by it)."""
	from campaign_model import features as ft
	from campaign_model import vocab

	key = {row["key"]: i for i, row in vocab.creatures().items()}
	angel, dwarf, pikeman = key["core:angel"], key["core:dwarf"], key["core:pikeman"]
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	t = m.features
	x = tensors(rich_inputs(1, 5))
	x["h_present"][:] = False
	x["h_present"][0, :2], x["h_owner"][0, :2], x["h_garrisoned"][0, :2] = True, c.OWNER_SELF, False
	x["h_army_creature"][0, :2], x["h_army_count"][0, :2] = -1, 0
	x["h_army_creature"][0, 0, :2], x["h_army_count"][0, 0, :2] = torch.tensor([angel, dwarf]), torch.tensor([5, 2])
	x["h_army_creature"][0, 1, 0], x["h_army_count"][0, 1, 0] = pikeman, 20
	x["h_morale"][0, :2] = torch.tensor([0, 1])  # (VCMI's: two factions 0; one +1)
	x["h_drop"][:], x["h_drop_kept"][:] = False, False
	x["h_drop"][0, 0, 0, 1] = x["h_drop_kept"][0, 0, 0] = True  # (DROP_SLOW: the Dwarves, into the garrison)
	x["x_heroes"][0] = torch.tensor([1, 2])
	x["x_take_creature"][:], x["x_take_count"][:] = -1, 0
	x["x_take_creature"][0, 0, 0, 0, :2], x["x_take_count"][0, 0, 0, 0, :2] = torch.tensor([angel, dwarf]), torch.tensor([5, 2])  # (TAKE_BEST to hero 0: the Angels and Dwarves;
	x["x_take_creature"][0, 0, 0, 1, 0], x["x_take_count"][0, 0, 0, 1, 0] = pikeman, 20  # the Pikemen stay: nothing changes)
	K = c.MultiKind
	with torch.no_grad():
		known = m.domain_knowledge(x)
		kept = float(t.multi_gain(x, known)[0, 0, K.DROP_SLOW])
		x["h_drop_kept"][0, 0, 0] = False
		disbanded = float(t.multi_gain(x, known)[0, 0, K.DROP_SLOW])
		take = float(t.multi_gain(x, known)[0, 0, K.TAKE_BEST])
	mult = known.hero_multiplier[0, 0]
	army_c, army_n = x["h_army_creature"][0, 0], x["h_army_count"][0, 0].float()
	rest = x["h_morale"][0, 0].float() - ft.make_up_morales(army_c, army_n, t)
	both = float(ft.led_values(army_c, army_n, mult, rest, t))
	angels = float(ft.led_values(torch.tensor([angel] + [-1] * 6), torch.tensor([5.0] + [0.0] * 6), mult, rest, t))
	dwarves = 2 * float(t.creature_cost[dwarf] + t.creature_unit[dwarf])
	assert kept == pytest.approx(angels - both + t.unled_share * dwarves, rel=1e-4) and kept > 0  # (faster, +1 morale)
	assert disbanded == pytest.approx(angels - both, rel=1e-4) and disbanded < kept
	assert take == pytest.approx(0.0, abs=1e-2)  # (the armies as they were)


def test_a_gather_counts_what_it_leaves_its_town_to():
	"""model.CustomFeatures.gather_loss: an enemy hero beside a town of ours that its garrison holds and its empty fort
	would not: GATHER's gain falls by about the town's buildings (features.emptied_danger ~1, town_danger ~0); with no
	enemy in sight it is Φ's alone. And a GATHER into a scout pays gather_cost_weight."""
	from campaign_model import features as ft

	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	t = m.features
	x = tensors(rich_inputs(1, 5))
	x["h_present"][:] = False
	x["h_present"][0, 0], x["h_owner"][0, 0], x["h_garrisoned"][0, 0], x["h_town"][0, 0] = True, c.OWNER_SELF, False, 0  # (our hero in town 0)
	x["h_primary"][0, 0], x["h_artifacts"][0, 0], x["h_artifacts"][0, 0, 0] = torch.tensor([2, 2, 3, 2]), -1, 0  # (a spellbook)
	x["t_present"][0, 0], x["t_owner"][0, 0], x["t_garrison_hero"][0, 0], x["t_visiting"][0, 0] = True, c.OWNER_SELF, -1, 0
	x["t_garrison_creature"][0, 0], x["t_garrison_count"][0, 0], x["t_detailed"][0, 0] = -1, 0, True
	x["t_garrison_creature"][0, 0, 0], x["t_garrison_count"][0, 0, 0] = 12, 500  # (a garrison that holds)
	with torch.no_grad():
		calm = m.domain_knowledge(x)
		calm_gain = float(t.multi_gain(x, calm)[0, 0, c.MultiKind.GATHER])
	assert float(calm.emptied_danger[0, 0]) == 0.0 and float(calm.town_danger[0, 0]) == 0.0
	assert float(calm.hero_multiplier[0, 0]) > t.unled_share

	def joined_gain(known):  # (Φ v5: the visitor leading its army and the garrison, less both apart; features.led_values)
		army_c, army_n = x["h_army_creature"][0, 0], x["h_army_count"][0, 0].float()
		rest = x["h_morale"][0, 0].float() - ft.make_up_morales(army_c, army_n, t)
		alone = ft.led_values(army_c, army_n, known.hero_multiplier[0, 0], rest, t)
		both = ft.led_values(torch.cat([army_c, x["t_garrison_creature"][0, 0]]), torch.cat([army_n, x["t_garrison_count"][0, 0].float()]), known.hero_multiplier[0, 0], rest, t)
		return float(both - alone - t.unled_share * known.town_gold[0, 0])

	assert calm_gain == pytest.approx(joined_gain(calm), rel=1e-4)
	e = c.OWN_HEROES  # (an enemy hero on the town's tile, 20 of the same creature)
	x["h_present"][0, e], x["h_owner"][0, e], x["h_pos"][0, e] = True, c.OWNER_SELF + 1, x["t_pos"][0, 0]
	x["h_army_creature"][0, e], x["h_army_count"][0, e], x["h_detailed"][0, e] = -1, 0, True
	x["h_army_creature"][0, e, 0], x["h_army_count"][0, e, 0] = 12, 20
	with torch.no_grad():
		known = m.domain_knowledge(x)
		gain = float(t.multi_gain(x, known)[0, 0, c.MultiKind.GATHER])
	empty, now = float(known.emptied_danger[0, 0]), float(known.town_danger[0, 0])
	stakes, gold = float(known.town_stakes[0, 0]), float(known.town_gold[0, 0])
	assert empty > 0.9 and now < 0.1 and stakes > gold > 0
	assert float(t.gather_risk_weight()) == pytest.approx(1.0, abs=1e-5)
	assert gain == pytest.approx(joined_gain(known) - (empty * (stakes - gold) - now * stakes), rel=1e-4)
	assert gain < calm_gain - 0.8 * (stakes - gold)
	# Its prior: the gain as value_logit prices it while the visitor is our main hero (the only one: main); a stronger
	# hero of ours elsewhere makes it a scout (one town, one main hero), whose GATHER pays gather_cost_weight (3)
	with torch.no_grad():
		prior = float(t.multi_prior(x, known)[0, 0, c.MultiKind.GATHER])
	assert bool(known.main[0, 0]) and prior == pytest.approx(float(ft.value_logit(torch.tensor(gain))), abs=1e-4)
	x["h_present"][0, 1], x["h_owner"][0, 1], x["h_garrisoned"][0, 1], x["h_town"][0, 1] = True, c.OWNER_SELF, False, -1
	x["h_army_creature"][0, 1], x["h_army_count"][0, 1], x["h_detailed"][0, 1] = -1, 0, True
	x["h_army_creature"][0, 1, 0], x["h_army_count"][0, 1, 0] = 12, 1000
	with torch.no_grad():
		known = m.domain_knowledge(x)
		scout = float(t.multi_prior(x, known)[0, 0, c.MultiKind.GATHER])
		gain = float(t.multi_gain(x, known)[0, 0, c.MultiKind.GATHER])
	assert not bool(known.main[0, 0]) and bool(known.main[0, 1]) and float(t.gather_cost_weight()) == pytest.approx(3.0, abs=1e-5)
	assert scout == pytest.approx(float(ft.value_logit(torch.tensor(gain))) - 3.0, abs=1e-4)


def test_a_multi_actions_prior_is_what_it_does_to_phi():
	"""model.CustomFeatures.multi_gain (schema 14, the user 2026-11-05): RECRUIT_ALL buys what we can pay for, the highest
	level first, at (unled_share - 1) of its worth (a little below 0); RECRUIT_ALL_HERO into a visiting starting hero
	(attack and defense 0, spell power 3, knowledge 2, a spellbook) 0; GATHER the garrison x (the visitor's factor -
	unled_share); GIVE_ALL, each way, the giver's army x the factors' difference."""
	from campaign_model import features as ft
	from game_values import REFERENCE_KNOWLEDGE, REFERENCE_POWER

	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	t = m.features
	x = tensors(rich_inputs(1, 5))
	x["h_present"][:] = False
	x["h_present"][0, :2], x["h_owner"][0, :2], x["h_garrisoned"][0, :2] = True, c.OWNER_SELF, False
	x["h_town"][0, :2] = torch.tensor([0, -1])  # (hero 0 visits town 0)
	x["h_primary"][0, 0] = torch.tensor([0, 0, REFERENCE_POWER, REFERENCE_KNOWLEDGE])
	x["h_artifacts"][0, :2] = -1
	x["h_artifacts"][0, 0, 0] = 0  # (a spellbook)
	x["h_primary"][0, 1] = torch.tensor([10, 10, 0, 0])  # (hero 1: no spellbook, x sqrt(1.5 / 0.75))
	x["t_available_creature"][0, 0] = -1
	x["t_available_creature"][0, 0, 0], x["t_available"][0, 0, 0] = 12, 10
	t.creature_resources[12] = torch.tensor([0.0, 0, 0, 0, 0, 0, 100.0])
	x["g_resources"][0] = torch.tensor([0, 0, 0, 0, 0, 0, 650])  # (6 of the 10 affordable)
	x["x_heroes"][0] = torch.tensor([1, 2])  # (an exchange of heroes 0 and 1)
	x["h_army_creature"][0, :2], x["h_army_count"][0, :2] = -1, 0
	x["h_army_creature"][0, :2, 0], x["h_army_count"][0, :2, 0] = 12, torch.tensor([20, 30])  # (each leads troops)
	x["h_morale"][0, :2] = 1  # (VCMI's for an army of one faction: its make-up's +1)
	with torch.no_grad():
		known = m.domain_knowledge(x)
		gains = t.multi_gain(x, known)[0]
	K, U = c.MultiKind, t.unled_share
	assert float(known.hero_gold[0, 0]) > 0 and float(known.hero_gold[0, 1]) > 0
	worth = 6 * float(t.creature_cost[12] + t.creature_unit[12])
	assert float(gains[0, K.RECRUIT_ALL]) == pytest.approx(worth * (U - 1), rel=1e-4) and float(gains[0, K.RECRUIT_ALL]) < 0
	# Φ v5: each hero's army as features.led_values prices it (its worth x the hero's multiplier x its movement, and its
	# make-up's morale over the rest of the hero's: h_morale holds the make-up's, these all-Angel armies' +1)
	army_c, army_n = x["h_army_creature"][0, :2], x["h_army_count"][0, :2].float()
	rest = x["h_morale"][0, :2].float() - ft.make_up_morales(army_c, army_n, t)
	led = lambda creatures, counts, h: float(ft.led_values(creatures, counts, known.hero_multiplier[0, h], rest[h], t))  # noqa: E731
	alone = [led(army_c[h], army_n[h], h) for h in (0, 1)]
	recruits_c, recruits_n = x["t_available_creature"][0, 0], torch.tensor([6.0] + [0.0] * (c.N_LEVELS - 1))
	assert float(gains[0, K.RECRUIT_ALL_HERO]) == pytest.approx(led(torch.cat([army_c[0], recruits_c]), torch.cat([army_n[0], recruits_n]), 0) - alone[0] - worth, rel=1e-4, abs=1e-2)
	garrison = led(torch.cat([army_c[0], x["t_garrison_creature"][0, 0]]), torch.cat([army_n[0], x["t_garrison_count"][0, 0].float()]), 0)
	assert float(gains[0, K.GATHER]) == pytest.approx(garrison - alone[0] - U * float(known.town_gold[0, 0]), rel=1e-4, abs=1e-2)
	strong, start = float(known.hero_multiplier[0, 1]), float(known.hero_multiplier[0, 0])
	assert strong == pytest.approx(U * math.sqrt(1.5 / 0.75), rel=1e-5) and start == pytest.approx(1.0, rel=1e-5)
	both = (torch.cat([army_c[0], army_c[1]]), torch.cat([army_n[0], army_n[1]]))
	assert float(gains[1, K.GIVE_ALL]) == pytest.approx(led(*both, 1) - sum(alone), rel=1e-4)  # (to hero 1, the stronger)
	assert float(gains[0, K.GIVE_ALL]) == pytest.approx(led(*both, 0) - sum(alone), rel=1e-4)
	assert float(gains[1, K.GIVE_ALL]) > 0 > float(gains[0, K.GIVE_ALL])  # (one kind of creature: the multipliers decide)
	# The prior: the gain as value_logit prices it; a GIVE_ALL to the weaker hero less its cost (3 at the start). The
	# stronger: the higher level, else the bigger factor, else the bigger army
	cost = float(t.give_all_cost_weight())
	assert cost == pytest.approx(3.0, abs=1e-5)
	logit = lambda row: float(ft.value_logit(gains[row, K.GIVE_ALL]))  # noqa: E731
	for levels, stronger in (((4, 4), 1), ((5, 4), 0), ((3, 4), 1)):  # (one level: hero 1's factor; else the level)
		x["h_level"][0, :2] = torch.tensor(levels)
		with torch.no_grad():
			prior = t.multi_prior(x, m.domain_knowledge(x))[0]
		for kind in (K.RECRUIT_ALL, K.RECRUIT_ALL_HERO):
			assert float(prior[0, kind]) == pytest.approx(float(ft.value_logit(gains[0, kind])), abs=1e-5)
		scout = 1.0 - float(t.main_visitor(x, m.domain_knowledge(x))[0, 0])  # (hero 0 visits; hero 1, stronger, is the main one)
		assert scout == 1.0 and float(prior[0, K.GATHER]) == pytest.approx(float(ft.value_logit(gains[0, K.GATHER])) - scout * float(t.gather_cost_weight()), abs=1e-5)
		assert float(prior[stronger, K.GIVE_ALL]) == pytest.approx(logit(stronger), abs=1e-5), levels
		assert float(prior[1 - stronger, K.GIVE_ALL]) == pytest.approx(logit(1 - stronger) - cost, abs=1e-5), levels
	# One level and one factor: the bigger army (hero 1's 30 against hero 0's 20)
	x["h_level"][0, :2], x["h_primary"][0, 1], x["h_artifacts"][0, 1, 0] = 4, x["h_primary"][0, 0], 0
	with torch.no_grad():
		known = m.domain_knowledge(x)
	assert float(known.hero_multiplier[0, 0]) == float(known.hero_multiplier[0, 1])
	assert t.stronger_receiver(x, known)[0].tolist() == [0.0, 1.0]
