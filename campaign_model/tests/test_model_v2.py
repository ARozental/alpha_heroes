"""The campaign model v2 (model_v2.py) and its foresight (foresight.py): the contract, the export, every part learning,
the labels of a game, an update with them."""
import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.foresight import battle_score, game_labels
from campaign_model.model import CORRECTION_STEP, CampaignModel, build_model
from campaign_model.model_v2 import ACTION_TARGETS, CampaignModelV2, Sides
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
	x = tensors(rich_inputs(8, seed=4))
	optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

	weights = {}

	def loss():  # (each table's log-probabilities weighted at random: the mean of a uniform table's has no gradient)
		out = model(x)
		tables = sum((v.clamp(min=-30) * weights.setdefault(k, torch.randn(v.shape, generator=torch.Generator().manual_seed(len(k))))).mean() for k, v in out.items() if k.endswith("_logp"))
		return tables + out["value"].mean() + out["growth_logit"].mean() + out["end_logit"].mean() + sum(v.pow(2).mean() for k, v in out.items() if k.startswith("fs_"))

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
	}
	targets = np.full((n, c.N_VALUE_OFFSETS), 0.5, np.float32)
	before = [p.detach().clone() for p in model.parameters()]
	metrics, _ = ppo_update(model, torch.optim.Adam(model.parameters(), lr=1e-4), decisions, targets, np.zeros(n, np.float32), CampaignConfig(batch_size=4, passes=1, model_architecture="v2"), decode=lambda rows: {k: v[rows] for k, v in x.items()}, foresight=foresight)
	assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
	assert metrics["foresight_turn"] >= 0 and metrics["foresight_hero_alive"] > 0
	assert metrics["grad_shared_foresight"] > 0 and -1 <= metrics["grad_shared_cosine"] <= 1


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
