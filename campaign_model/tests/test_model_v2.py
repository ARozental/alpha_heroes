"""The campaign model v2 (model_v2.py) and its foresight (foresight.py): the contract, the export, every part learning,
the labels of a game, an update with them."""
import math

import numpy as np
import pytest
import torch

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
	optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

	weights = {}

	def loss():  # (each table's log-probabilities weighted at random: the mean of a uniform table's has no gradient)
		out = model(x)
		# (fs_plan too: its outputs start at 0, where a square has no gradient)
		tables = sum((v.clamp(min=-30) * weights.setdefault(k, torch.randn(v.shape, generator=torch.Generator().manual_seed(len(k))))).mean() for k, v in out.items() if k.endswith("_logp") or k == "fs_plan")
		return tables + out["value"].mean() + out["growth_logit"].mean() + out["end_logit"].mean() + sum(v.pow(2).mean() for k, v in out.items() if k.startswith("fs_") and k != "fs_plan")

	for _ in range(3):
		optimizer.zero_grad()
		loss().backward()
		optimizer.step()
	optimizer.zero_grad()
	loss().backward()
	required = (
		"encode_hero.", "encode_town.", "encode_tiles.", "encode_global.", "encode_options.", "encode_other.", "reach.", "travel_days.", "sides.", "tile_sides.",
		"player.", "players.", "join_", "stamp_", "unet.", "read_tiles.", "area.", "blocks.", "norm.", "pair_", "foresight.", "again_", "move_pair.", "travel_",
		"move_out.", "act_", "build_q.", "building.", "recruit_head.", "hire_", "army_src.", "army_dst.", "army_out.", "trade_head.", "cast_q.", "spell.",
		"buy_q.", "buy_item.", "option_q.", "end_out.", "type_bias", "value_hidden.", "value_heads.", "growth_heads.", "end_head.",
	)
	dead = [name for name, p in model.named_parameters() if name.startswith(required) and (p.grad is None or float(p.grad.abs().sum()) == 0)]
	# (embedding rows of ids absent from these inputs get none: what must learn is every block's dense part)
	dead = [name for name in dead if not any(e in name for e in ("hero_type", "klass", "artifact", "skill", "creature", "quantity", "faction", "subtype", "terrain", "road", "object", "sound", "kind", "last", "player.weight", "building", "spell.weight", "buy_item.weight"))]
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
	tiles where a fight starts, by how hopeless (FIGHT_STEP x the weight), and nowhere else."""
	from campaign_model.model import dense
	from campaign_model.model_v2 import FIGHT_STEP

	torch.manual_seed(3)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	x = tensors(rich_inputs(4, 7))
	with torch.no_grad():
		known = m.domain_knowledge(x)
		fights = dense(known.tile_fights, known.windows, known.n, 4)
		hopeless = dense(torch.relu(-known.tile_fight_ratio * known.tile_fights - 1.0).clamp(max=6.0) * known.tile_fights, known.windows, known.n, 4)
		before = m(x)["move_logp"].clone()
		m.move_fight.weight.data[0, 2] = -1.0 / FIGHT_STEP
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


def test_one_adam_step_moves_a_hopeless_tiles_logit_by_fight_step_times_the_rate():
	"""model_v2.FIGHT_STEP (joint7, 2026-11-04: at the campaign's 1e-5 the 3 fight weights moved ~3e-4 in 26 updates): one
	step of make_optimizer's AdamW at the PC's rate, its gradient pushing the hopeless tiles down, takes such a tile's
	learned MOVE logit (move_fight's part) lower by FIGHT_STEP x lr x its features, not lr x them."""
	from campaign_model.model_v2 import FIGHT_STEP, fight_features

	torch.manual_seed(3)
	m = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	x = tensors(rich_inputs(4, 7))
	lr = 1e-5
	optimizer = make_optimizer(m, CampaignConfig(learning_rate=lr))
	with torch.no_grad():
		features = fight_features(m.domain_knowledge(x))
	hopeless = features[..., 2] > 0
	assert bool(hopeless.any())
	part = lambda: m.move_fight(FIGHT_STEP * features).squeeze(-1)  # noqa: E731
	before = part().detach()
	part()[hopeless].sum().backward()  # (the loss's gradient: the hopeless tiles' logits up, so the step takes them down)
	optimizer.step()
	with torch.no_grad():
		drop = before - part()
	# (Adam's first step moves each weight by lr (its sign's), the bias too: the bias's lr unscaled)
	expected = FIGHT_STEP * lr * features.abs().sum(dim=-1) + lr
	torch.testing.assert_close(drop[hopeless], expected[hopeless], rtol=1e-3, atol=1e-7)


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
