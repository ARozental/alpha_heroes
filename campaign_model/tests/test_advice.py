"""Schema 19 in the model (model_v2.CampaignModelV2.advice, advise; model.CustomFeatures.nk_prior_weights): Nullkiller2's
candidates and the bounded views as inputs and a prior. What must hold: nothing changes until it is turned on, an older
checkpoint loads and plays as it did, a decision without the advisor is the model without it, the prior moves the
advised entries, and every new weight gets a gradient (a feature nothing can learn from is not a feature)."""
import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model import views as V
from campaign_model.model import ADDED, ADDED_HEADS, dense
from campaign_model.model_v2 import AREAS, PLAN_VIEWS, VIEWS, CampaignModelV2, FastLinear, real_only
from campaign_model.schema import random_inputs
from campaign_model.tests.test_model_v2 import rich_inputs, tensors

HO, W = c.OWN_HEROES, c.WINDOW
NEW = ("time_reach", "view_pair", "view_mix", "view_move", "view_travel", "travel_view", "move_toward", "travel_toward", "plan_in", "nk_hero", "nk_town", "nk_global")
NK_INPUTS = tuple(V.absent_rows(1, "cpu"))
LOGPS = ("type_logp", "move_logp", "travel_logp", "build_logp", "recruit_logp", "option_logp", "value", "end_logit")


def small(seed=0):
	torch.manual_seed(seed)
	return CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()


def without_advisor(x):
	"""x with schema 19's inputs at their none: the advisor did not run, no history."""
	out = dict(x)
	for name in NK_INPUTS:
		out[name] = torch.zeros_like(x[name])
	out["w_arrive_left"] = torch.zeros_like(x["w_arrive_left"])
	return out


def same(a, b, names=None):
	for name in names or a:
		assert torch.equal(a[name], b[name]), name


def turn_inputs(batch=6, seed=3):
	"""Inputs whose every decision is a turn decision with candidates (random_inputs draws the kinds)."""
	xs = [tensors(rich_inputs(1, seed=s)) for s in range(seed, seed + 40)]
	turns = [x for x in xs if int(x["g_kind"][0]) == c.DecisionKind.TURN and bool(x["g_nk_present"][0]) and int((x["k_move"] > 0).sum()) > 0 and int((x["k_travel"] > 0).sum()) > 0][:batch]
	assert len(turns) == batch
	return {name: torch.cat([x[name] for x in turns]) for name in turns[0]}


def shaped(model, x):
	"""x with what random rows lack and a game's have: every candidate's destination a window tile worth something (a
	real destination is an object's), in the batch's first half one hero's candidates scored at the gather tier alone
	(a fallback: a defend turn's other heroes), a threat to our first town."""
	x = {name: value.clone() for name, value in x.items()}
	b = x["g_kind"].shape[0]
	x["t_nk_threat"][:, 0] = torch.tensor([1, 2, 1500])
	with torch.no_grad():
		known = model.domain_knowledge(x)
	value = dense(real_only(known.tile_value, known), known.windows, known.n, b).reshape(b, -1)
	for row in range(b):
		worth = torch.nonzero(value[row]).reshape(-1)
		rows = torch.nonzero(x["k_hero"][row] > 0).reshape(-1)
		for i, at in enumerate(rows.tolist()):
			x["k_dest_cell"][row, at] = int(worth[i % len(worth)]) + 1
		if row < b // 2:
			hero = int(x["k_hero"][row, rows[-1]])
			mine = x["k_hero"][row] == hero
			x["k_score"][row, mine, 0] = 0
			x["k_score"][row, mine, 1] = x["k_score"][row, mine, 1].clamp(min=100)
	return x


def with_label(x, kinds=("MOVE", "TRAVEL", "END", "BUILD")):
	"""x with Nullkiller2's own choice at each decision (schema 19's g_nk_action, g_nk_index, g_nk_cell): a legal entry of
	the type kinds[row % len(kinds)] (END: no entry), none where that type has no legal entry."""
	x = {name: value.clone() for name, value in x.items()}
	b = x["g_kind"].shape[0]
	masks = {"MOVE": "move_mask", "TRAVEL": "travel_mask", "BUILD": "build_mask", "RECRUIT": "recruit_mask"}
	action, index = torch.zeros(b, dtype=torch.long), torch.zeros(b, dtype=torch.long)
	for row in range(b):
		kind = kinds[row % len(kinds)]
		if kind == "END":
			action[row] = int(c.ActionType.END) + 1
			continue
		legal = torch.nonzero(x[masks[kind]][row].reshape(-1)).reshape(-1)
		if len(legal):
			action[row], index[row] = int(c.ActionType[kind]) + 1, int(legal[len(legal) // 2]) + 1
	x["g_nk_action"], x["g_nk_index"], x["g_nk_cell"] = action, index, torch.zeros(b, dtype=torch.long)
	return x


def test_the_new_parts_add_nothing_until_they_are_turned_on():
	"""A fresh model (and so an older checkpoint loaded into it: the new parts start as a fresh model has them) gives
	the same outputs whatever schema 19's inputs hold: every new layer's output starts at 0 and the prior is off."""
	model = small()
	x = turn_inputs()
	with torch.no_grad():
		with_rows, without = model(x), model(without_advisor(x))
		before = model({name: value for name, value in x.items() if name not in NK_INPUTS and name != "w_arrive_left"})
		labelled = model(with_label(x))
	same(with_rows, without)
	same(with_rows, before)
	same(with_rows, labelled)  # (its own choice among the inputs: nothing until its prior is on and its channel's weights leave 0)
	assert float(model.features.nk_beta) == 0.0 and float(model.features.nk_beta_fallback) == 0.0 and float(model.features.nk_label) == 0.0
	assert all(float(w) == 0.0 for w in model.features.nk_prior_weights()) and float(model.features.nk_label_weight()) == 0.0


def test_a_checkpoint_from_before_loads_with_the_new_parts_as_a_new_model_starts_them():
	saved = small(0).state_dict()
	old = {k: v for k, v in saved.items() if not k.startswith(NEW) and not k.startswith("features.nk_")}
	assert len(saved) - len(old) == 15 + 10  # (the new layers' tensors; the priors' five gains and three sizes, the follow chance and its shift)
	fresh = small(1)
	own = {k: v.clone() for k, v in fresh.state_dict().items()}
	fresh.load_state_dict(old)
	for k, v in fresh.state_dict().items():
		if k.startswith("features.nk_"):
			assert float(v.abs().sum()) == 0.0, k  # (model.ADDED: at 0, the prior off and each gain at its start)
		else:
			assert torch.equal(v, own[k] if k.startswith(NEW) else saved[k]), k
	assert all(k in ADDED for k in saved if k.startswith("features.nk_"))
	assert all(any(k.startswith(prefix) for prefix in ADDED_HEADS) for k in saved if k.startswith(NEW))


def test_the_prior_raises_the_advised_entries_and_only_them():
	"""nk_beta 2: an entry Nullkiller2's best candidate goes through gains up to 2 logits against an entry with none (its
	gain 1 at the start); entries of a decision without the advisor do not move; the direct weights on the path's
	loss and the danger on arrival count only while a prior is on."""
	model = small()
	x = turn_inputs()
	with torch.no_grad():
		off = model(x)
		model.features.nk_beta.fill_(2.0)
		model.features.nk_beta_fallback.fill_(1.0)
		on = model(x)
		absent = model(without_advisor(x))
	best, fallback, survival, risk = (float(w) for w in model.features.nk_prior_weights())
	assert best == pytest.approx(2.0) and fallback == pytest.approx(1.0) and survival == pytest.approx(0.25) and risk == pytest.approx(0.25)
	same(off, absent, LOGPS)  # (no advisor: the prior adds nothing)
	tables = V.candidate_tables(x, dest=False)
	for name, logp, mask in (("move", "move_logp", "move_mask"), ("travel", "travel_logp", "travel_mask")):
		b = x["g_kind"].shape[0]
		legal = x[mask].reshape(b, -1)
		prior = tables[f"prior_{name}"].reshape(b, -1)
		moved = (on[logp] - off[logp]).reshape(b, -1)
		for row in range(b):
			top = int(torch.where(legal[row], prior[row], torch.full_like(prior[row], -1.0)).argmax())
			plain = legal[row] & (tables[name].reshape(b, -1, V.F)[row, :, 0] == 0)  # (legal entries no candidate goes through)
			if float(prior[row, top]) < 0.99 or not bool(plain.any()):
				continue
			# within the table the best candidate's entry gains its 2 logits (and its path's terms) against a plain one
			gain = float(moved[row, top]) - float(moved[row][plain].mean())
			terms = 0.25 * float(tables[name].reshape(b, -1, V.F)[row, top, V.CHANNELS.index("log_survival")]) - 0.25 * float(tables[name].reshape(b, -1, V.F)[row, top, V.CHANNELS.index("risk")])
			assert gain == pytest.approx(2.0 * float(prior[row, top]) + terms, abs=1e-3), (name, row)
			assert float(moved[row][plain].std()) < 1e-4 if int(plain.sum()) > 1 else True  # (plain entries all move alike: the softmax's shift)


def test_the_override_of_a_paired_test_sets_the_prior(monkeypatch):
	monkeypatch.setenv("ALPHA_NK2_PRIOR", "4")
	monkeypatch.setenv("ALPHA_NK2_PRIOR_FALLBACK", "2")
	model = small()
	best, fallback, survival, risk = (float(w) for w in model.features.nk_prior_weights())
	assert (best, fallback) == (pytest.approx(4.0), pytest.approx(2.0)) and survival == pytest.approx(0.25)
	monkeypatch.setenv("ALPHA_NK2_PRIOR", "0")
	monkeypatch.setenv("ALPHA_NK2_PRIOR_FALLBACK", "0")
	model = small()
	model.features.nk_beta.fill_(3.0)  # (a checkpoint's own prior: the override wins, at 0 too)
	assert all(float(w) == 0.0 for w in model.features.nk_prior_weights())


def test_every_new_weight_gets_a_gradient():
	"""On a model whose other weights are not at 0 (a trained one's: a fresh model's heads are, and stop every gradient
	behind them), a loss on the policy's and the value's outputs reaches every new parameter: none is dead."""
	model = small()
	torch.manual_seed(5)
	with torch.no_grad():
		for name, p in model.named_parameters():
			if float(p.abs().sum()) == 0.0 and not name.startswith("features."):
				p.normal_(0.0, 0.05)
		model.features.nk_beta.fill_(2.0)
		model.features.nk_beta_fallback.fill_(1.0)
		model.features.nk_label.fill_(2.0)
		model.features.nk_follow.fill_(0.5)
	model.train()
	x = with_label(shaped(model, turn_inputs(batch=8, seed=11)))
	tables = V.candidate_tables(x, dest=False)
	assert float(tables["prior_move_fallback"].sum() + tables["prior_travel_fallback"].sum()) > 0  # (the fallback's gain has entries to act on)
	out = model(x)
	torch.manual_seed(6)
	loss = sum((out[name] * torch.randn_like(out[name])).masked_fill(~torch.isfinite(out[name]) | (out[name] < -1e3), 0.0).sum() for name in ("type_logp", "move_logp", "travel_logp", "value_logit", "end_logit"))
	loss.backward()
	dead = [name for name, p in model.named_parameters() if (name.startswith(NEW) or name.startswith("features.nk_")) and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
	assert dead == []
	flat = [name for name, p in model.named_parameters() if name.startswith(NEW) and not torch.isfinite(p.grad).all()]
	assert flat == []


def test_the_views_are_bounded_and_none_is_constant():
	"""advice's two tables: VIEWS wide, every channel within -1..1 but the log survival (-4.61..0.01), finite, and none
	the same at every legal entry of a batch of random decisions."""
	model = small()
	x = with_label(shaped(model, turn_inputs(batch=8, seed=21)), kinds=("MOVE", "TRAVEL"))
	with torch.no_grad():
		a = model.domain_knowledge(x).advice
	survival = 6 + 3 + 3 + 4 + 2 + V.CHANNELS.index("log_survival")
	for table in (a.move, a.travel):
		assert table.shape[-1] == VIEWS and bool(torch.isfinite(table).all())
		others = torch.cat([table[..., :survival], table[..., survival + 1 :]], dim=-1)
		assert float(others.min()) >= -1.0 - 1e-6 and float(others.max()) <= 1.0 + 1e-6
		assert float(table[..., survival].min()) >= -4.61 and float(table[..., survival].max()) <= 0.01
	assert a.travel.shape == (8, HO, AREAS, VIEWS)
	spread = torch.cat([a.travel.reshape(-1, VIEWS)[x["travel_mask"].reshape(-1)], model_moves(model, x, a)]).std(dim=0)
	kinds = [6 + 3 + 3 + 4 + 2 + V.CHANNELS.index(f"kind_{i}") for i in range(V.KINDS)]
	rare = set(kinds) | {6 + 3 + 3 + 4 + 2 + V.CHANNELS.index(name) for name in ("unreachable", "teleport", "chain", "fallback", "others", "gather_all", "gather_hero")}
	assert [i for i in range(VIEWS) if float(spread[i]) == 0.0 and i not in rare] == []


def model_moves(model, x, a):
	"""advice's MOVE views at the legal MOVE entries [N, VIEWS]."""
	known = model.domain_knowledge(x)
	b = x["g_kind"].shape[0]
	return dense(a.move, known.windows, known.n, b).reshape(-1, VIEWS)[x["move_mask"].reshape(-1)]


def test_a_fast_weight_moves_its_logit_ten_times_a_plain_one():
	layer = FastLinear(VIEWS + PLAN_VIEWS)
	x = torch.rand(4, VIEWS + PLAN_VIEWS)
	assert float(layer(x).abs().sum()) == 0.0
	with torch.no_grad():
		layer.weight[3] = 0.01
	assert torch.allclose(layer(x), 10.0 * 0.01 * x[:, 3])


def test_the_plan_reaches_the_heroes_tokens_once_its_layer_has_weights():
	"""advise: with plan_in's weights set, a hero's token differs with and without the advisor, and only tokens of heroes
	of ours with a candidate differ."""
	model = small()
	x = turn_inputs()
	with torch.no_grad():
		model.plan_in.weight.normal_(0.0, 0.1)
		states = []
		for inputs in (x, without_advisor(x)):
			known = model.domain_knowledge(inputs)
			state = model.advise(model.see(model.join(model.take_sides(model.encode(inputs)), known), known), known)
			states.append(state.heroes)
	changed = (states[0] - states[1]).abs().sum(dim=-1) > 0  # [B, H]
	planned = torch.zeros(x["g_kind"].shape[0], HO + 1, dtype=torch.bool).scatter(1, torch.where((x["k_hero"] > 0) & ((x["k_score"] > 0).any(dim=-1)), x["k_hero"].long() - 1, torch.full_like(x["k_hero"].long(), HO)), True)[:, :HO]
	ours = x["h_present"][:, :HO] & (x["h_owner"][:, :HO] == c.OWNER_SELF)
	assert bool(changed[:, :HO].any())
	assert not bool((changed[:, :HO] & ~(planned & ours)).any())
	assert not bool(changed[:, HO:].any())


def test_the_inputs_random_inputs_draws_read_as_candidates():
	"""schema.random_inputs' schema-19 rows (b1's) through views.candidate_tables: entries hold candidates, a prior is
	within 0..1, each hero's plan weights sum to 1."""
	x = tensors(random_inputs(12, seed=2))
	tables = V.candidate_tables(x)
	assert float(tables["move"][..., 0].sum()) > 0 and float(tables["travel"][..., 0].sum()) > 0
	for name in ("prior_move", "prior_travel", "prior_move_fallback", "prior_travel_fallback"):
		assert float(tables[name].min()) >= 0.0 and float(tables[name].max()) <= 1.0 + 1e-6
	hero = torch.where(x["k_hero"] > 0, x["k_hero"].long() - 1, torch.full_like(x["k_hero"].long(), HO))
	total = torch.zeros(12, HO + 1).scatter_add(1, hero, tables["plan_weight"])[:, :HO]
	assert bool(((total - 1.0).abs() < 1e-5)[total > 0].all()) and float(total.sum()) > 0
	assert np.isfinite(tables["channels"].numpy()).all()


def test_the_run_s_line_sets_the_prior_at_every_start_and_the_weights_carry_it():
	"""learner.set_advice_prior: config.nk2_prior and nk2_prior_fallback into the model's buffers; a state dict carries
	them to a server's model; a later start with the prior off turns it off whatever the checkpoint held."""
	from campaign_model.config import CampaignConfig
	from campaign_model.learner import set_advice_prior

	model = small()
	set_advice_prior(model, CampaignConfig(nk2_prior=2.0, nk2_prior_fallback=1.0, nk2_label_prior=4.0, nk2_follow=0.75))
	assert float(model.features.nk_beta) == 2.0 and float(model.features.nk_beta_fallback) == 1.0 and float(model.features.nk_label) == 4.0 and float(model.features.nk_follow) == 0.75
	server = small(1)
	server.load_state_dict(model.state_dict())
	assert float(server.features.nk_beta) == 2.0 and float(server.features.nk_beta_fallback) == 1.0 and float(server.features.nk_label) == 4.0
	set_advice_prior(server, CampaignConfig())
	assert float(server.features.nk_beta) == 0.0 and float(server.features.nk_beta_fallback) == 0.0 and float(server.features.nk_label) == 0.0 and float(server.features.nk_follow) == 0.0


def test_a_candidate_in_a_further_window_is_read_at_that_window_s_entry():
	"""Schema 17's further windows: a candidate whose first step (k_move) and destination (k_dest_cell) are cells of a
	hero's further window, in another slot, lands in that window's views; an entry of its own window whose candidate
	heads for a cell of the further one gathers that cell (the destination's value views, the flag)."""
	from campaign_model.tests.test_schema_17 import SPARE, turns, window_row, with_further

	model = small()
	x = turns(4, seed=3)
	x = with_further(x, 0, x["h_pos"][:, 0] + np.array([3, 0, 0]))
	cells = W * W
	has = 6 + 3 + 3 + 4 + 2
	for b in range(4):
		legal = np.flatnonzero(x["move_mask"][b, SPARE].reshape(-1))
		own = np.flatnonzero(x["move_mask"][b, 0].reshape(-1))
		assert len(legal) and len(own)
		for name in NK_INPUTS:
			x[name][b] = 0
		x["g_nk_present"][b], x["g_nk_tier"][b], x["g_nk_count"][b] = True, 4, 2
		# row 0: hero 0 through an entry of its further window, to that same cell; row 1: through an entry of its own
		# window, heading for a cell of the further one
		x["k_hero"][b, :2] = 1
		x["k_score"][b, 0], x["k_score"][b, 1] = [500, 500], [200, 200]
		x["k_move"][b, 0], x["k_dest_cell"][b, 0] = SPARE * cells + legal[0] + 1, SPARE * cells + legal[0] + 1
		x["k_move"][b, 1], x["k_dest_cell"][b, 1] = own[0] + 1, SPARE * cells + legal[-1] + 1
	y = tensors(x)
	with torch.no_grad():
		known = model.domain_knowledge(y)
	a = known.advice
	for b in range(4):
		legal = np.flatnonzero(x["move_mask"][b, SPARE].reshape(-1))
		own = np.flatnonzero(x["move_mask"][b, 0].reshape(-1))
		further, first = window_row(known, b, SPARE), window_row(known, b, 0)
		views = a.move.reshape(-1, cells, VIEWS)
		assert float(views[further, legal[0], has]) == 1.0 and float(views[further, legal[0], has + 1]) == 0.0  # (a candidate, the decision's best)
		assert float(views[first, own[0], has]) == 1.0 and float(views[first, own[0], has + 1]) < 0.0
		assert float(views[first, own[0], VIEWS - 2]) == 1.0  # (its destination: a cell of a computed window)
		index, valid = a.move_dest
		assert bool(valid.reshape(-1, cells)[first, own[0]]) and int(index.reshape(-1, cells)[first, own[0]]) == further * cells + legal[-1]
		assert float(views[:, :, has].sum(dim=1)[first]) == 1.0  # (one entry of the own window holds a candidate)


def test_its_own_choice_as_a_prior_is_followed_as_far_as_the_prior_is_large():
	"""nk_label: Nullkiller2's own choice gains that many logits inside its table and on its type's score, whatever its
	type (MOVE, TRAVEL, END, BUILD); at 20 it is our choice (probability above 0.99 at every decision that has one), at
	0 nothing moves, and a decision without one is as before at any size."""
	from campaign_model.schema import HEADS

	model = small()
	x = with_label(turn_inputs(batch=8, seed=31))
	action, index = x["g_nk_action"] - 1, x["g_nk_index"] - 1
	assert set(int(a) for a in action if a >= 0) >= {int(c.ActionType.MOVE), int(c.ActionType.TRAVEL), int(c.ActionType.END)}

	def followed(out):  # [B] the log-probability of its choice: its type's, and its entry's in the type's table
		logp = torch.zeros(len(action))
		for row in range(len(action)):
			if action[row] < 0:
				logp[row] = float("nan")
				continue
			logp[row] = out["type_logp"][row, action[row]]
			if int(action[row]) != int(c.ActionType.END):
				logp[row] += out[HEADS[c.ActionType(int(action[row]))][0]][row].reshape(-1)[index[row]]
		return logp

	with torch.no_grad():
		off = model(x)
		model.features.nk_label.fill_(3.0)
		some = model(x)
		model.features.nk_label.fill_(20.0)
		full = model(x)
		none = dict(x)
		none["g_nk_action"] = torch.zeros_like(x["g_nk_action"])
		without = model(none)
	has = action >= 0
	assert int(has.sum()) >= 6
	assert bool((followed(some)[has] > followed(off)[has] + 1.0).all())
	assert bool((followed(full)[has].exp() > 0.99).all())
	same(off, without, LOGPS)  # (no choice of its to follow: the prior adds nothing at any size)
	assert float(model.features.nk_label_weight()) == pytest.approx(20.0)


def test_the_override_of_a_paired_test_sets_the_label_s_prior(monkeypatch):
	monkeypatch.setenv("ALPHA_NK2_LABEL", "4")
	model = small()
	assert float(model.features.nk_label_weight()) == pytest.approx(4.0)
	monkeypatch.setenv("ALPHA_NK2_LABEL", "0")
	model = small()
	model.features.nk_label.fill_(3.0)
	assert float(model.features.nk_label_weight()) == 0.0


def test_following_its_choice_at_a_chance_is_the_exact_mixture_of_it_and_our_own_policy(monkeypatch):
	"""nk_follow (model_v2's follow): with rho on, the sampled policy is rho x (its choice) + (1 - rho) x (ours) at every
	decision whose labelled entry is legal: the type's probabilities, the joint of its type and each entry of that
	type's table, every table still summing to 1; the other tables, a decision without a choice of its and one whose
	choice is not legal are ours untouched. The chance is learnable (a gradient reaches its shift and our own weights),
	and a paired test's ALPHA_NK2_FOLLOW overrides the checkpoint's."""
	from campaign_model.schema import HEADS

	rho = 0.7
	model = small()
	torch.manual_seed(3)
	with torch.no_grad():
		for name, p in model.named_parameters():
			if float(p.abs().sum()) == 0.0 and not name.startswith("features."):
				p.normal_(0.0, 0.05)  # (a trained model's heads: its own policy is not flat)
	x = with_label(turn_inputs(batch=8, seed=31))
	illegal = next(row for row in range(8) if int(x["g_nk_action"][row]) - 1 == int(c.ActionType.MOVE))
	x["g_nk_index"][illegal] = int(torch.nonzero(~x["move_mask"][illegal].reshape(-1))[0]) + 1  # (an entry we cannot play)
	action, index = x["g_nk_action"] - 1, x["g_nk_index"] - 1
	with torch.no_grad():
		own = model(x)
		model.features.nk_follow.fill_(rho)
		mixed = model(x)
	followed = 0
	for row in range(8):
		kind = int(action[row])
		if kind < 0 or row == illegal:
			for name in LOGPS:
				assert torch.equal(mixed[name][row], own[name][row]), (row, name)
			continue
		followed += 1
		expected = (1 - rho) * own["type_logp"][row].exp()
		expected[kind] += rho
		assert torch.allclose(mixed["type_logp"][row].exp(), expected, atol=1e-5), row
		for other, (name, _) in HEADS.items():
			table, before = mixed[name][row].reshape(-1), own[name][row].reshape(-1)
			if int(other) != kind:
				assert torch.equal(table, before), (row, name)
				continue
			joint = (1 - rho) * (own["type_logp"][row, kind] + before).exp()
			joint[index[row]] += rho
			assert torch.allclose((mixed["type_logp"][row, kind] + table).exp(), joint, atol=1e-5), (row, name)
			assert float(table.exp().sum()) == pytest.approx(1.0, abs=1e-4)
			assert bool((torch.isfinite(table) == torch.isfinite(before)).all())  # (no illegal entry made playable)
	assert followed >= 5 and {int(a) for a in action if a >= 0} >= {int(c.ActionType.MOVE), int(c.ActionType.TRAVEL), int(c.ActionType.END), int(c.ActionType.BUILD)}
	# learnable: the chance's shift and our own policy's weights both get a finite gradient
	model.train()
	out = model(x)
	taken = sum(out["type_logp"][row, int(action[row])] for row in range(8) if int(action[row]) >= 0 and row != illegal)
	taken.backward()
	assert float(model.features.nk_follow_shift.grad.abs()) > 0
	grads = [p.grad for name, p in model.named_parameters() if name.startswith("move_out") and p.grad is not None]
	assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads) > 0
	assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
	# the bounds of what it can learn: from 0.9, between 0.31 and 0.9 (bundle 60's cap: the run line's chance at most)
	with torch.no_grad():
		model.features.nk_follow.fill_(0.9)
		model.features.nk_follow_shift.fill_(100.0)
		assert float(model.features.nk_follow_logs()[1].exp()) == pytest.approx(0.9, abs=1e-6)
		model.features.nk_follow_shift.fill_(-100.0)
		assert float(model.features.nk_follow_logs()[1].exp()) == pytest.approx(0.309, abs=2e-3)
	monkeypatch.setenv("ALPHA_NK2_FOLLOW", "0")
	paired = small()
	paired.features.nk_follow.fill_(0.9)
	assert not bool(paired.features.nk_follow_logs()[0])  # (the override wins, at 0 too)
	monkeypatch.setenv("ALPHA_NK2_FOLLOW", "0.5")
	assert float(small().features.nk_follow_logs()[1].exp()) == pytest.approx(0.5)
