"""F3, a town's holding (config.town_hold, e5 for the user, 2026-11-12): on the entries that change a town of ours'
holding, what the change takes off its chance of falling to the nearest strong rival we see x its stake. 0 with no rival
near, for a town safe either way and for one lost either way; above 0 for what adds, below 0 for the visitor leaving;
off at size 0 with its views' weights at 0: the model as before, bit for bit."""
import torch

from campaign_model import constants as c
from campaign_model import features as ft
from campaign_model.model_v2 import CampaignModelV2
from campaign_model.tests.test_model_v2 import rich_inputs

TO, HO = c.OWN_TOWNS, c.OWN_HEROES
HOLD = ("hold_recruit", "hold_hire", "hold_leave", "hold_enter_town")


def model_and_inputs(b=6, seed=4):
	torch.manual_seed(3)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	return model, {k: torch.from_numpy(v) for k, v in rich_inputs(b, seed).items()}


def test_protection_is_zero_far_safe_or_lost_and_signed_by_the_change():
	near = torch.tensor([[[1.0]]])  # [1, TO=1, E=1]
	sharpness = torch.tensor(4.0)
	at = lambda theirs, before, after: float(ft.hold_protection(near, torch.tensor([[[theirs]]]), torch.tensor([[before]]), torch.tensor([[after]]), sharpness))  # noqa: E731
	assert at(8.0, 8.0, 9.0) > 0.4 and at(8.0, 9.0, 8.0) < -0.4  # (an even fight: adding holds it, taking away loses it)
	assert abs(at(8.0, 8.0, 9.0) + at(8.0, 9.0, 8.0)) < 1e-6
	assert abs(at(2.0, 8.0, 9.0)) < 1e-6 and abs(at(14.0, 8.0, 9.0)) < 1e-6  # (safe anyway, lost anyway)
	far = ft.hold_protection(torch.zeros(1, 1, 1), torch.tensor([[[8.0]]]), torch.tensor([[8.0]]), torch.tensor([[9.0]]), sharpness)
	assert float(far) == 0.0  # (no rival near)


def test_a_town_lost_to_a_strong_rival_either_way_reads_zero_whatever_a_weaker_one_does():
	"""Two rivals beside the town: the stronger takes it before and after; adding beats only the weaker: protection 0."""
	near = torch.tensor([[[1.0, 1.0]]])
	theirs = torch.tensor([[[14.0, 8.0]]])  # (the strong one far above the holding, the weak one even with it)
	protection = ft.hold_protection(near, theirs, torch.tensor([[8.0]]), torch.tensor([[9.0]]), torch.tensor(4.0))
	assert abs(float(protection)) < 1e-6
	alone = ft.hold_protection(near[..., 1:], theirs[..., 1:], torch.tensor([[8.0]]), torch.tensor([[9.0]]), torch.tensor(4.0))
	assert float(alone) > 0.4  # (the weak rival alone: the same action saves the town)


def test_the_near_discount_is_fuzzy_and_another_level_is_far():
	pos = torch.tensor([[[10, 10, 0], [55, 10, 0], [10, 10, 1]]])
	foes = (pos, torch.tensor([[5.0, 6.0, 7.0]]), torch.tensor([[True, True, True]]))
	near, theirs = ft.hold_reach(foes, torch.tensor([[[10, 25, 0]]]))
	assert torch.allclose(near[0, 0], torch.tensor([float(torch.exp(torch.tensor(-15 / 45))), float(torch.exp(torch.tensor(-1.0))), 0.0]))
	assert theirs.shape == (1, 1, 3)


def test_the_values_are_finite_bounded_and_signed():
	model, x = model_and_inputs()
	values = captured_values(model, x)
	stake = values["hold_stake"]
	for name in HOLD:
		assert torch.isfinite(values[name]).all(), name
	assert (values["hold_leave"] <= 1e-6).all()
	assert (values["hold_recruit"] >= -1e-6).all() and (values["hold_hire"] >= -1e-6).all() and (values["hold_enter_town"] >= -1e-6).all()
	assert (values["hold_recruit"].abs() <= stake + 1e-3).all() and (values["hold_hire"].abs() <= stake[..., None] + 1e-3).all()
	assert (values["hold_enter_town"].abs() <= stake[:, None, :] + 1e-3).all()


def captured_values(model, x):
	"""town_hold_values' output on x (the forward's own call)."""
	got = {}
	real = model.features.town_hold_values

	def keep(x_, f_):
		got.update(real(x_, f_))
		return dict(got)

	model.features.town_hold_values = keep
	try:
		with torch.no_grad():
			model(x)
	finally:
		model.features.town_hold_values = real
	return got


def test_off_at_size_zero_whatever_the_values():
	"""At town_hold 0 and the views' weights 0 the policy does not read the values: planted values change nothing."""
	model, x = model_and_inputs()
	with torch.no_grad():
		base = model(x)
		real = model.features.town_hold_values

		def planted(x_, f_):
			out = real(x_, f_)
			return {k: (torch.rand_like(v) - 0.5) * 1e6 if k.startswith("hold_") and k not in ("hold_stake", "hold_phi") else v for k, v in out.items()}

		model.features.town_hold_values = planted
		out = model(x)
	for name, value in base.items():
		if torch.is_tensor(value) and value.is_floating_point():
			assert torch.equal(torch.nan_to_num(value, nan=0.0, posinf=1e9, neginf=-1e9), torch.nan_to_num(out[name], nan=0.0, posinf=1e9, neginf=-1e9)), name


def test_on_it_raises_what_saves_a_town_and_lowers_leaving_it():
	"""At town_hold 1: a planted saving on town 0's recruits lifts its RECRUIT entries against another town's; a planted
	cost on hero 0's leaving lowers all its MOVE and TRAVEL entries alike."""
	model, x = model_and_inputs()
	real = model.features.town_hold_values

	def planted(x_, f_):
		out = real(x_, f_)
		out["hold_recruit"], out["hold_leave"] = torch.zeros_like(out["hold_recruit"]), torch.zeros_like(out["hold_leave"])
		out["hold_hire"], out["hold_enter_town"] = torch.zeros_like(out["hold_hire"]), torch.zeros_like(out["hold_enter_town"])
		out["hold_recruit"][:, 0] = 4000.0
		out["hold_leave"][:, 0] = -4000.0
		return out

	with torch.no_grad():
		model.features.town_hold_values = planted
		base = model(x)
		model.features.town_hold.fill_(1.0)
		out = model(x)
	expect = float(ft.value_logit(torch.tensor(4000.0)))
	shift = (out["recruit_logp"] - base["recruit_logp"]).flatten(2)  # [B, TO, entries]: town 0's entries up by expect against the others'
	rows = [r for r in range(shift.shape[0]) if torch.isfinite(shift[r, 0]).any() and torch.isfinite(shift[r, 1:]).any()]
	for r in rows:
		up, rest = shift[r, 0][torch.isfinite(shift[r, 0])], shift[r, 1:][torch.isfinite(shift[r, 1:])]
		assert abs(float(up.mean() - rest.mean()) - expect) < 0.05 and float(up.std(unbiased=False)) < 1e-3, (float(up.mean() - rest.mean()), expect)
	for name, sign in (("move_logp", -1), ("travel_logp", -1)):
		b = next(r for r in range(x["g_day"].shape[0]) if torch.isfinite(base[name][r, 0]).sum() >= 1 and torch.isfinite(base[name][r, 1:]).sum() >= 1)
		a0, a1 = base[name][b].reshape(HO, -1), out[name][b].reshape(HO, -1)
		hero0, others = torch.isfinite(a0[0]), torch.isfinite(a0[1:])
		shift0, shift1 = a1[0][hero0] - a0[0][hero0], a1[1:][others] - a0[1:][others]
		assert torch.allclose(shift0, shift0[0].expand_as(shift0), atol=1e-4), name
		assert abs(float(shift0[0] - shift1[0]) - sign * expect) < 0.05, (name, float(shift0[0] - shift1[0]), expect)
