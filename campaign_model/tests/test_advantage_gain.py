"""Our own policy reading the advantage head (stage 2a, config.adv_gain; model_v2's decide): inert at 0 whatever the
head says; at gain g, a planted prediction moves exactly what it should: an entry's odds within its table by g x its
(clipped) term, the type masses only through the type term and best_legal, Nullkiller2's own choice by its named
weight, and the follow mixture on top of it all."""
import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model.model_v2 import ADV_CLIP, CampaignModelV2
from campaign_model.schema import HEADS
from campaign_model.tests.test_model_v2 import rich_inputs

A_ = c.ActionType
TABLES = [name for name, _ in HEADS.values()]


def tensors(x):
	return {k: torch.from_numpy(v) for k, v in x.items()}


def model_and_inputs(n=6, seed=2):
	torch.manual_seed(3)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	return model, tensors(rich_inputs(n, seed))


def planted(model, x, **parts):
	"""The head's forward replaced by fixed predictions (0 but for parts), in its outputs' shapes."""
	with torch.no_grad():
		model.adv_outputs = True
		shapes = {k: v.shape for k, v in model(x).items() if k.startswith("adv_")}
		model.adv_outputs = False
	values = {name: torch.zeros(shapes[name]) for name in ("adv_type", "adv_move", "adv_travel", "adv_named")}
	values.update(parts)
	label = model.advantage.forward

	def forward(state, known):
		return {**values, "adv_label": label(state, known)["adv_label"]}

	model.advantage.forward = forward


def run(model, x, gain):
	with torch.no_grad():
		model.features.adv_gain.fill_(gain)
		return model(x)


def legal_pair(out, x, name, mask_name, row):
	"""Two legal entries of a table at a row (flat indices), or None."""
	legal = np.flatnonzero(x[mask_name][row].reshape(-1).numpy())
	return (int(legal[0]), int(legal[-1])) if len(legal) >= 2 else None


def test_at_gain_0_the_head_changes_nothing():
	model, x = model_and_inputs()
	before = run(model, x, 0.0)
	with torch.no_grad():
		for p in model.advantage.parameters():
			p.normal_(0.0, 1.0)
	after = run(model, x, 0.0)
	for name, value in before.items():
		assert torch.equal(value, after[name]), name


def test_an_entry_term_moves_its_odds_by_the_gain_and_nothing_else():
	model, x = model_and_inputs()
	base = run(model, x, 0.0)
	rows = [b for b in range(x["g_day"].shape[0]) if legal_pair(base, x, "move_logp", "move_mask", b)]
	assert rows
	b = rows[0]
	e, other = legal_pair(base, x, "move_logp", "move_mask", b)
	for size, expect in ((1.0, 1.0), (10.0, ADV_CLIP)):  # (clipped to +-ADV_CLIP before the gain)
		move = torch.zeros_like(base["move_logp"])
		move.view(move.shape[0], -1)[b, e] = size
		planted(model, x, adv_move=move)
		for gain in (0.5, 2.0):
			out = run(model, x, gain)
			flat, flat0 = out["move_logp"].reshape(len(out["move_logp"]), -1), base["move_logp"].reshape(len(base["move_logp"]), -1)
			shift = (flat[b, e] - flat[b, other]) - (flat0[b, e] - flat0[b, other])
			assert float(shift) == pytest.approx(gain * expect, abs=1e-4)
			for name in TABLES:  # (the other tables as they were)
				if name != "move_logp":
					assert torch.allclose(out[name], base[name], atol=1e-6, equal_nan=True), name
			# the type masses: every ratio between two types other than MOVE as it was (MOVE's score moves through
			# best_legal alone)
			legal = torch.isfinite(base["type_logp"][b]) & (base["type_logp"][b] > -1e3)
			others = [t for t in range(len(A_)) if legal[t] and t != A_.MOVE]
			for t in others[1:]:
				assert float(out["type_logp"][b, t] - out["type_logp"][b, others[0]]) == pytest.approx(float(base["type_logp"][b, t] - base["type_logp"][b, others[0]]), abs=1e-5)
		model, x = model_and_inputs()


def test_a_type_term_moves_only_the_type_masses_by_the_gain():
	model, x = model_and_inputs()
	base = run(model, x, 0.0)
	b = next(r for r in range(len(base["type_logp"])) if int((base["type_logp"][r] > -1e3).sum()) >= 2)
	legal = [t for t in range(len(A_)) if float(base["type_logp"][b, t]) > -1e3]
	favoured, other = legal[-1], legal[0]
	adv_type = torch.zeros(base["type_logp"].shape)
	adv_type[:, favoured] = 1.0
	planted(model, x, adv_type=adv_type)
	out = run(model, x, 1.5)
	shift = (out["type_logp"][b, favoured] - out["type_logp"][b, other]) - (base["type_logp"][b, favoured] - base["type_logp"][b, other])
	assert float(shift) == pytest.approx(1.5, abs=1e-4)
	for name in TABLES:
		assert torch.allclose(out[name], base[name], atol=1e-6, equal_nan=True), name


def test_its_own_choice_is_favoured_by_the_named_weight_inside_its_table():
	model, x = model_and_inputs(24, 4)
	base = run(model, x, 0.0)
	action, index = x["g_nk_action"].long() - 1, x["g_nk_index"].long() - 1
	rows = [b for b in range(len(action)) if int(action[b]) > 0 and index[b] >= 0 and int(x[HEADS[A_(int(action[b]))][1]][b].sum()) >= 2]
	assert rows, "no row with its choice in a table of two legal entries"
	b = rows[0]
	name, mask_name = HEADS[A_(int(action[b]))]
	legal = np.flatnonzero(x[mask_name][b].reshape(-1).numpy())
	other = int([e for e in legal if e != int(index[b])][0])
	named = torch.zeros(base["type_logp"].shape)
	named[:, int(action[b])] = 2.0
	planted(model, x, adv_named=named)
	out = run(model, x, 1.0)
	flat, flat0 = out[name].reshape(len(out[name]), -1), base[name].reshape(len(base[name]), -1)
	shift = (flat[b, int(index[b])] - flat[b, other]) - (flat0[b, int(index[b])] - flat0[b, other])
	assert float(shift) == pytest.approx(2.0, abs=1e-4)


def test_the_follow_mixture_is_exact_on_top_of_it():
	"""With rho 0.9 the tables the games sample from are the mixture of our own tables as read with the head: our own
	policy's (gain on, follow off) mixed by hand is what the model gives with both on."""
	model, x = model_and_inputs(12, 4)
	base = run(model, x, 0.0)
	move = torch.zeros_like(base["move_logp"])
	move.view(move.shape[0], -1)[:, 0] = 1.5
	planted(model, x, adv_move=move, adv_type=torch.full(base["type_logp"].shape, 0.3))
	own = run(model, x, 1.0)
	with torch.no_grad():
		model.features.nk_follow.fill_(0.9)
	mixed = run(model, x, 1.0)
	on, log_rho, log_rest = model.features.nk_follow_logs()
	rho = float(log_rho.exp())
	action, index = x["g_nk_action"].long() - 1, x["g_nk_index"].long() - 1
	checked = 0
	for b in range(len(action)):
		t = int(action[b])
		if t < 0 or float(own["type_logp"][b, t]) < -1e3:
			continue
		p_type = (1 - rho) * own["type_logp"][b].exp()
		p_type[t] += rho
		assert torch.allclose(mixed["type_logp"][b].exp(), p_type, atol=1e-5)
		if t != A_.END:
			name = HEADS[A_(t)][0]
			table = own[name][b].reshape(-1).exp() * (1 - rho) * own["type_logp"][b, t].exp()
			if not torch.isfinite(own[name][b].reshape(-1)[int(index[b])]) or float(own[name][b].reshape(-1)[int(index[b])]) < -1e3:
				continue
			table[int(index[b])] += rho
			assert torch.allclose(mixed[name][b].reshape(-1).exp(), table / p_type[t], atol=1e-5)
		checked += 1
	assert checked >= 2
