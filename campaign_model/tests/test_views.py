"""views.py: every view inside its bound, smooth, with a gradient everywhere; Nullkiller2's candidates as tables."""
import math

import torch

from campaign_model import constants as c
from campaign_model import views as v

HO, CELLS, AREAS = c.OWN_HEROES, c.WINDOW * c.WINDOW, c.MAP_LEVELS * c.TRAVEL_GRID * c.TRAVEL_GRID
NK = 16


def rows(batch=1, **fields):
	"""An input dict of empty candidate rows [batch, NK], the advisor present; fields: {name: {(b, k): value}}."""
	zeros = lambda *shape: torch.zeros((batch, NK) + shape, dtype=torch.int64)  # noqa: E731 (the wire: integers, 0 none, indices + 1)
	x = {
		"k_hero": zeros(), "k_dest": zeros(3), "k_dest_cell": zeros(), "k_dest_area": zeros(), "k_move": zeros(), "k_travel": zeros(), "k_score": zeros(2), "k_days": zeros(),
		"k_loss": zeros(), "k_danger": zeros(), "k_kind": zeros(), "k_settled": zeros(), "k_flags": zeros(), "g_nk_present": torch.ones(batch, dtype=torch.bool),
	}
	for name, values in fields.items():
		for where, value in values.items():
			x[name][where] = torch.as_tensor(value, dtype=x[name].dtype)
	return x


WIRE = {"days": v.DAYS_SCALE, "loss": v.LOSS_SCALE, "danger": v.DANGER_SCALE}


def candidate(x, k, hero, dest, score, gather=None, move=-1, travel=-1, dest_cell=-1, b=0, **more):
	"""A candidate row in its own units, written as the wire holds it (Advice.cpp: scaled integers, indices + 1)."""
	x["k_hero"][b, k] = hero + 1
	x["k_dest"][b, k] = torch.tensor(dest)
	x["k_score"][b, k] = torch.tensor([round(score * v.SCORE_SCALE), round((score if gather is None else gather) * v.SCORE_SCALE)])
	x["k_move"][b, k], x["k_travel"][b, k], x["k_dest_cell"][b, k] = move + 1, travel + 1, dest_cell + 1
	for name, value in more.items():
		x[f"k_{name}"][b, k] = round(value * WIRE.get(name, 1))


def test_value_views_are_bounded_signed_and_ordered():
	gold = torch.tensor([-1e9, -5000.0, -1.0, 0.0, 1.0, 300.0, 5000.0, 2e5, 1e9])
	views = v.value_views(gold, torch.tensor(1e5), torch.tensor(2000.0))
	assert views.shape == (9, 6)
	assert float(views.abs().max()) < 1.0
	assert torch.all(views[3] == 0)  # nothing is nothing in every view
	assert torch.all(views[1:] >= views[:-1])  # monotone in the amount
	assert torch.allclose(views[1], -views[6])  # a loss is a gain's mirror
	# half-way where the docstring says: 2,000 gold; 2% of the empire; a day of growth
	at = v.value_views(torch.tensor([2000.0]), torch.tensor(1e5), torch.tensor(2000.0))[0]
	assert math.isclose(float(at[0]), 0.5, abs_tol=1e-6) and math.isclose(float(at[2]), 0.5, abs_tol=1e-6) and math.isclose(float(at[4]), 0.5, abs_tol=1e-6)


def test_value_views_broadcast_a_decision_s_empire_over_its_tiles():
	gold = torch.full((2, 3, 4), 1000.0)
	views = v.value_views(gold, torch.tensor([1e4, 1e6]), torch.tensor([100.0, 1e4]))
	assert views.shape == (2, 3, 4, 6)
	assert float(views[0, 0, 0, 2]) > float(views[1, 0, 0, 2])  # the same gold is more of a small empire
	assert float(views[0, 0, 0, 4]) > float(views[1, 0, 0, 4])  # and more days of a slow one's growth
	assert torch.equal(views[0, 0, 0, :2], views[1, 0, 0, :2])  # the absolute views do not move


def test_every_view_keeps_a_gradient_at_its_extremes():
	gold = torch.tensor([1e7, -1e7, 0.0], requires_grad=True)
	v.value_views(gold, torch.tensor(1e5), torch.tensor(2000.0)).sum().backward()
	assert torch.all(gold.grad > 0)
	loss = torch.tensor([0.001, 0.5, 0.999], requires_grad=True)  # (the feature's own slope: it tells every loss from the next, certain death included)
	v.log_survival(loss).sum().backward()
	assert torch.all(loss.grad < 0)
	ends = v.log_survival(torch.tensor([0.0, 1.0, 1.2]))
	assert float(ends[0]) <= 0.01 and math.isclose(float(ends[1]), math.log(0.01), rel_tol=1e-5) and float(ends[2]) == float(ends[1])  # finite at certain death
	ratio = torch.tensor([0.05, 1.0, 20.0], requires_grad=True)
	v.risk(ratio).sum().backward()
	assert torch.all(ratio.grad > 0)


def test_daily_gain_has_a_floor():
	phi = torch.tensor([1e5, 1e5, 1e5, 500.0])
	gain = v.daily_gain(phi, torch.tensor([9e4, 1.2e5, 0.0, 0.0]), torch.tensor([7, 7, 0, 0]))
	assert math.isclose(float(gain[0]), 1e4 / 7, rel_tol=1e-5)
	assert math.isclose(float(gain[1]), 0.005 * 1e5)  # a week that lost value: the floor
	assert math.isclose(float(gain[2]), 0.005 * 1e5)  # no history yet: the floor
	assert math.isclose(float(gain[3]), 0.005 * 1000.0)  # an empire all but gone counts as PHI_FLOOR


def test_own_days_and_base_days():
	turns = torch.tensor([0, 0, 1, 2, -1])
	left = torch.tensor([1500, 0, 1000, 500, -1])
	own = v.own_days(turns, left, torch.tensor(1500.0), torch.tensor(2000.0))
	base = v.base_days(turns, left, torch.tensor(1500.0), torch.tensor(2000.0))
	# today: the share of a full day spent; later: whole days + the last day's share (the encoder's travel_days)
	assert torch.allclose(own, torch.tensor([0.0, 0.75, 1.5, 2.75, v.FAR_DAYS]))
	# the same paths in a baseline hero's days: 0, 1500, 1500 + 1000, 1500 + 2000 + 1500 movement points over 1,500
	assert torch.allclose(base, torch.tensor([0.0, 1.0, 2500 / 1500, 5000 / 1500, v.FAR_DAYS]))
	fast = v.own_days(torch.tensor([1]), torch.tensor([0]), torch.tensor(3000.0), torch.tensor(3000.0))
	slow = v.own_days(torch.tensor([2]), torch.tensor([0]), torch.tensor(1500.0), torch.tensor(1500.0))
	assert float(fast) < float(slow)  # the same 6,000 points: fewer days of a fast hero's
	assert torch.allclose(v.base_days(torch.tensor([1]), torch.tensor([0]), torch.tensor(3000.0), torch.tensor(3000.0)), v.base_days(torch.tensor([3]), torch.tensor([0]), torch.tensor(1500.0), torch.tensor(1500.0)))


def test_discounts_and_rates():
	d = v.discounts(torch.tensor([0.0, 1.0, 3.0, 30.0]))
	assert torch.allclose(d[0], torch.ones(2)) and torch.allclose(d[1], torch.tensor([0.5, 0.75])) and torch.allclose(d[2], torch.tensor([0.25, 0.5]))
	assert float(d[3].max()) < 0.1
	near = v.rate_views(torch.tensor([5000.0]), torch.tensor([0.0]), torch.tensor(1e5), torch.tensor(2000.0))
	far = v.rate_views(torch.tensor([5000.0]), torch.tensor([4.0]), torch.tensor(1e5), torch.tensor(2000.0))
	assert torch.all(near > far) and float(near.max()) < 1.0
	assert math.isclose(float(near[0, 0]), 0.5, abs_tol=1e-6)  # 5,000 beside the hero: 20,000 a day, the half-way point


def test_gap_preference_risk_settled():
	assert torch.allclose(v.gap(torch.tensor([[2.0, -2.0, 6.0]]), torch.tensor([2.0])), -torch.tanh(torch.tensor([[0.0, 1.0, 0.0]])))
	p = v.preference(torch.tensor([[10.0, 10.0 / math.e**3, 0.0, 1e-9]]), torch.tensor([10.0]))
	assert math.isclose(float(p[0, 0]), 0.0, abs_tol=1e-6) and math.isclose(float(p[0, 1]), -math.tanh(1.0), abs_tol=1e-5)
	assert float(p[0, 2]) == -1.0 and -1.0 <= float(p[0, 3]) < -0.99
	# six orders of magnitude between decisions change nothing: only the ratio counts
	assert torch.allclose(v.preference(torch.tensor([5.0, 0.5]), torch.tensor(5.0)), v.preference(torch.tensor([5e6, 5e5]), torch.tensor(5e6)))
	r = v.risk(torch.tensor([0.0, 0.01, 1.0, 100.0]))
	assert float(r[0]) == 0.0 and float(r[1]) < 1e-3 and math.isclose(float(r[2]), 0.5) and float(r[3]) < 1.0  # pixies: ~0, never more
	assert torch.allclose(v.settled(torch.tensor([0, 1, 3])), torch.tensor([0.0, 0.5, 0.75]))


def test_hero_bests_counts_a_further_window_as_its_hero_s():
	b, s = 1, HO
	move = torch.zeros(b, s, 2, 2)
	legal = torch.zeros(b, s, 2, 2, dtype=torch.bool)
	move[0, 0, 0, 0], legal[0, 0, 0, 0] = 1.0, True  # hero 0's own window
	move[0, 5, 1, 1], legal[0, 5, 1, 1] = 3.0, True  # slot 5: a further window of hero 0
	move[0, 1, 0, 1] = 9.0  # (not legal: never the best)
	travel, travel_legal = torch.zeros(b, HO, 4), torch.zeros(b, HO, 4, dtype=torch.bool)
	travel[0, 1, 2], travel_legal[0, 1, 2] = 2.0, True  # hero 1's best is a TRAVEL
	slots = torch.arange(HO).reshape(1, HO).clone()
	slots[0, 5] = 0
	by_slot, any_hero = v.hero_bests(move, legal, travel, travel_legal, slots)
	assert float(by_slot[0, 0]) == 3.0 and float(by_slot[0, 5]) == 3.0 and float(by_slot[0, 1]) == 2.0 and float(any_hero[0]) == 3.0
	assert float(by_slot[0, 2]) == -1e4  # a hero with nothing legal


def test_candidate_tables_one_candidate_an_entry():
	x = rows()
	candidate(x, 0, hero=0, dest=(5, 5, 0), score=100.0, move=40, dest_cell=70, loss=0.25, danger=1.0, days=1.0, kind=1, settled=3, flags=v.FLAG_TELEPORT)
	candidate(x, 1, hero=0, dest=(9, 9, 0), score=100.0 / math.e**3, move=40, dest_cell=90)  # the same first step, a weaker candidate (4.98: the wire rounds it to 4.98)
	candidate(x, 2, hero=1, dest=(5, 5, 0), score=50.0, travel=AREAS + 7, dest_cell=CELLS + 3)  # hero 1, the same destination
	t = v.candidate_tables(x)
	assert t["move"].shape == (1, HO, CELLS, v.F) and t["travel"].shape == (1, HO, AREAS, v.F) and t["dest"].shape == (1, HO, CELLS, v.F)
	name = {n: i for i, n in enumerate(v.CHANNELS)}
	entry = t["move"][0, 0, 40]
	assert float(entry[name["has"]]) == 1.0 and float(entry[name["preference_all"]]) == 0.0 and float(entry[name["preference_hero"]]) == 0.0
	assert math.isclose(float(entry[name["loss"]]), 0.25) and math.isclose(float(entry[name["log_survival"]]), math.log(0.76), rel_tol=1e-5)
	assert math.isclose(float(entry[name["risk"]]), 0.5) and math.isclose(float(entry[name["soon"]]), 0.5) and math.isclose(float(entry[name["settled"]]), 0.75)
	assert float(entry[name["kind_1"]]) == 1.0 and float(entry[name["kind_0"]]) == 0.0 and float(entry[name["teleport"]]) == 1.0
	# both candidates stand behind the entry: mass (1 + (1 - tanh 1)) / (1 + that)
	behind = 1.0 + (1.0 - math.tanh(1.0))
	assert math.isclose(float(entry[name["mass"]]), behind / (behind + 1.0), rel_tol=1e-3)  # (the wire rounds 4.9787 to 4.98)
	# hero 1 wants the same destination at half the priority: the other hero's preference is its live one
	assert math.isclose(float(entry[name["others"]]), 1.0 - math.tanh(math.log(2.0) / 3.0), rel_tol=1e-5)
	assert int(t["move_dest_cell"][0, 0, 40]) == 70 and int(t["move_dest_cell"][0, 0, 41]) == -1
	assert float(t["prior_move"][0, 0, 40]) == 1.0 and float(t["prior_move"].sum()) == 1.0 and float(t["prior_move_fallback"].sum()) == 0.0
	assert float(t["dest"][0, 0, 70, name["has"]]) == 1.0 and float(t["dest"][0, 0, 90, name["preference_all"]]) < 0.0
	travel = t["travel"][0, 1, 7]
	assert float(travel[name["has"]]) == 1.0 and math.isclose(float(travel[name["preference_all"]]), -math.tanh(math.log(2.0) / 3.0), rel_tol=1e-5) and float(travel[name["preference_hero"]]) == 0.0
	assert int(t["travel_dest_cell"][0, 1, 7]) == CELLS + 3
	assert math.isclose(float(t["prior_travel"][0, 1, 7]), 1.0 - math.tanh(math.log(2.0) / 3.0), rel_tol=1e-5)
	weights = t["plan_weight"][0]
	assert math.isclose(float(weights[0] + weights[1]), 1.0, rel_tol=1e-5) and math.isclose(float(weights[2]), 1.0, rel_tol=1e-5) and float(weights[3:].sum()) == 0.0
	assert math.isclose(float(weights[1] / weights[0]), math.exp(-v.PLAN_SHARPNESS * math.tanh(1.0)), rel_tol=1e-2)
	assert float(t["move"].abs().sum()) == float(entry.abs().sum())  # nothing anywhere else


def test_candidate_tables_fallback_on_a_defend_turn():
	x = rows()
	# The winning tier scored hero 0's defence only; hero 1's candidates scored at the gather tier alone
	candidate(x, 0, hero=0, dest=(1, 1, 0), score=0.8, gather=0.0, move=10, kind=3)
	candidate(x, 1, hero=1, dest=(7, 7, 0), score=0.0, gather=500.0, move=CELLS + 20)
	candidate(x, 2, hero=1, dest=(8, 8, 0), score=0.0, gather=250.0, move=CELLS + 21)
	candidate(x, 3, hero=0, dest=(2, 2, 0), score=0.0, gather=900.0, move=11)  # hero 0 has a winning one: this is a shadow
	candidate(x, 4, hero=2, dest=(3, 3, 0), score=0.0, gather=0.0, move=2 * CELLS + 5)  # scored nothing anywhere
	t = v.candidate_tables(x)
	name = {n: i for i, n in enumerate(v.CHANNELS)}
	assert float(t["prior_move"][0, 0, 10]) == 1.0 and float(t["prior_move"].sum()) == 1.0
	fallback = t["prior_move_fallback"]
	assert float(fallback[0, 0].sum()) == 0.0  # a shadow candidate gets no prior
	assert 0.0 < float(fallback[0, 1, 21]) < float(fallback[0, 1, 20]) < 1.0  # (against the decision's best gather score, the shadow's 900)
	assert float(t["move"][0, 1, 20, name["fallback"]]) == 1.0 and float(t["move"][0, 0, 11, name["fallback"]]) == 0.0
	assert float(t["move"][0, 0, 11, name["has"]]) == 1.0 and float(t["move"][0, 0, 11, name["gather_all"]]) == 0.0 and float(t["move"][0, 0, 11, name["preference_all"]]) == -1.0
	assert float(t["move"][0, 2].abs().sum()) == 0.0 and float(t["plan_weight"][0, 4]) == 0.0  # unscored: nowhere
	weights = t["plan_weight"][0]
	assert math.isclose(float(weights[3] / weights[0]), math.exp(-v.PLAN_SHARPNESS), rel_tol=1e-4)  # hero 0's plan is its defence, the shadow at e^-4
	assert math.isclose(float(weights[1] + weights[2]), 1.0, rel_tol=1e-5) and float(weights[1]) > float(weights[2])  # hero 1's: the gather tier's


def test_candidate_tables_without_the_advisor_are_zero_and_batches_do_not_mix():
	x = rows(batch=2)
	candidate(x, 0, hero=0, dest=(5, 5, 0), score=10.0, move=3, travel=4, dest_cell=5, b=0)
	candidate(x, 0, hero=0, dest=(5, 5, 0), score=10.0, move=6, travel=7, dest_cell=8, b=1)
	x["g_nk_present"][1] = False
	t = v.candidate_tables(x)
	assert float(t["move"][0].abs().sum()) > 0 and float(t["travel"][0, 0, 4, 0]) == 1.0
	for key, table in t.items():
		if key.endswith("dest_cell"):
			assert int(table[1].max()) == -1, key
		elif key == "hero_index":  # (nk_predict's helpers: "none" where nothing is)
			assert int(table[1].min()) == HO, key
		elif key in ("move_index", "travel_index", "dest_index"):
			assert int(table[1].min()) == HO * (AREAS if key == "travel_index" else CELLS), key
		else:
			assert float(table[1].abs().sum()) == 0.0, key


def test_every_channel_stays_in_its_bound_on_random_rows():
	torch.manual_seed(0)
	batch = 8
	x = rows(batch=batch)
	x["k_hero"] = torch.randint(0, HO + 1, (batch, NK))
	x["k_dest"] = torch.randint(0, 36, (batch, NK, 3))
	x["k_dest_cell"] = torch.randint(0, HO * CELLS + 1, (batch, NK))
	x["k_move"] = torch.randint(0, HO * CELLS + 1, (batch, NK))
	x["k_travel"] = torch.randint(0, HO * AREAS + 1, (batch, NK))
	x["k_score"] = (torch.exp(torch.randn(batch, NK, 2) * 6) * (torch.rand(batch, NK, 2) > 0.3) * v.SCORE_SCALE).clamp(max=2**31 - 1).round().long()  # six orders of magnitude, some 0
	x["k_days"], x["k_loss"], x["k_danger"] = (torch.rand(batch, NK) * 20000).long(), (torch.rand(batch, NK) * 12000).long(), (torch.exp(torch.randn(batch, NK) * 3) * 1000).clamp(max=1e6).long()
	x["k_kind"], x["k_settled"], x["k_flags"] = torch.randint(0, 9, (batch, NK)), torch.randint(0, 400, (batch, NK)), torch.randint(0, 16, (batch, NK))
	t = v.candidate_tables(x)
	for key in ("channels", "move", "travel", "dest"):
		assert torch.isfinite(t[key]).all(), key
		for i, channel in enumerate(v.CHANNELS):
			low, high = v.BOUNDS[channel]
			assert float(t[key][..., i].min()) >= low - 1e-6 and float(t[key][..., i].max()) <= high + 1e-6, (key, channel)
	for key in ("prior_move", "prior_travel", "prior_move_fallback", "prior_travel_fallback", "plan_weight"):
		assert float(t[key].min()) >= 0.0 and float(t[key].max()) <= 1.0 + 1e-6, key
	sums = torch.zeros(batch, HO + 1).scatter_add(1, torch.where(x["k_hero"] > 0, x["k_hero"] - 1, torch.full((batch, NK), HO)), t["plan_weight"])[:, :HO]
	assert torch.all((sums - 1.0).abs().lt(1e-4) | (sums == 0))  # a hero's weights sum to 1, or it has no scored candidate
