"""The privileged critic (2026-11-06; model_v2.PrivilegedCritic, learner.privileged_rivals, train.ppo_update): every value
head's logit corrected from the other players' own critic states and true Φ, training only. It starts at 0 and loads
into older checkpoints there; its rivals are each other player's last decision on an earlier day; it learns on top of
the critic without moving anything else; its corrections enter the credit only with config.privileged_credit."""
import copy

import numpy as np
import pytest
import torch

from campaign_model import constants as c
from campaign_model.config import CampaignConfig
from campaign_model.learner import privileged_rivals
from campaign_model.model_v2 import PRIVILEGED_HEADS, PRIVILEGED_RIVALS, CampaignModelV2
from campaign_model.records import Game
from campaign_model.tests.test_labels import decision, snapshot
from campaign_model.tests.test_model_v2 import decisions_for, rich_inputs
from campaign_model.train import GRAD_GROUPS, make_optimizer, ppo_update


def test_it_starts_at_zero_and_pools_the_rivals_in_any_order():
	torch.manual_seed(0)
	critic = CampaignModelV2(dim=32, cell_dim=8, layers=1).privileged
	own, rivals = torch.randn(5, 32), torch.randn(5, PRIVILEGED_RIVALS, 32)
	mask = torch.zeros(5, PRIVILEGED_RIVALS)
	mask[0, :3], mask[1, 0], mask[2, :2] = 1, 1, 1  # (row 3 on: no rival at all)
	has = mask.clone()
	has[2, 1] = 0  # (row 2's second rival in the game without a state: its Φ alone)
	phi, age = torch.randn(5, PRIVILEGED_RIVALS), torch.ones(5, PRIVILEGED_RIVALS)
	out = critic(own, rivals, mask, has, torch.zeros(5), phi, age)
	assert out.shape == (5, PRIVILEGED_HEADS) and float(out.abs().max()) == 0.0
	with torch.no_grad():
		critic.out.weight.normal_()
	out = critic(own, rivals, mask, has, torch.zeros(5), phi, age)
	assert torch.isfinite(out).all()
	order = torch.tensor([2, 0, 1] + list(range(3, PRIVILEGED_RIVALS)))  # (row 0's three rivals reordered)
	again = critic(own, rivals[:, order], mask[:, order], has[:, order], torch.zeros(5), phi[:, order], age[:, order])
	torch.testing.assert_close(again[0], out[0])
	hidden = rivals.clone()
	hidden[0, 3:] = 99.0  # (a rival not in the game changes nothing)
	hidden[2, 1] = 99.0  # (nor the state of one without a state)
	torch.testing.assert_close(critic(own, hidden, mask, has, torch.zeros(5), phi, age)[:3], out[:3])
	richer = phi.clone()
	richer[2, 1] += 3.0  # (but its Φ counts)
	assert not torch.allclose(critic(own, rivals, mask, has, torch.zeros(5), richer, age)[2], out[2])


def test_an_older_checkpoint_loads_it_at_zero_with_the_same_outputs():
	torch.manual_seed(1)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	state = {k: v for k, v in model.state_dict().items() if not k.startswith("privileged.")}
	old = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	old.load_state_dict(state)
	x = {k: torch.from_numpy(v) for k, v in rich_inputs(3, 2).items()}
	with torch.no_grad():
		a, b = model(x), old(x)
	for name in ("type_logp", "value_logit", "end_logit", "value_hidden"):
		torch.testing.assert_close(a[name], b[name])
	assert float(old.privileged.out.weight.abs().max()) == 0.0


def test_the_rivals_are_each_other_players_last_decision_on_an_earlier_day():
	"""Rows count from first_row over the fresh decisions; a rival's state is from a day before (never the same day:
	it could follow the decision); its age in days; Φ at the start of the decision's day (the ledger's)."""
	game = Game(ledgers={d: snapshot(d, 1000 * d, 500) for d in range(1, 8)})
	game.decisions = [decision("RED", 1), decision("BLUE", 1), decision("BLUE", 2), decision("RED", 3), decision("BLUE", 3), decision("RED", 3), decision("BLUE", 5)]
	fresh = np.array([True, True, True, True, False, True, True])
	out = privileged_rivals(game, fresh, first_row=10)
	# fresh rows: 10 RED d1, 11 BLUE d1, 12 BLUE d2, 13 RED d3, 14 RED d3, 15 BLUE d5
	assert out["rivals"].shape == (6, PRIVILEGED_RIVALS)
	assert out["rivals"][:, 0].tolist() == [-1, -1, 10, 12, 12, 14]  # (each one's one rival: the other color)
	assert (out["rivals"][:, 1:] == -1).all()
	assert (out["present"][:, 0] == 1).all() and (out["present"][:, 1:] == 0).all()  # (day 1's rival in the game, without a state yet)
	assert out["age"][:, 0].tolist() == [0, 0, 1, 1, 1, 2]
	assert out["own_phi"][3] == pytest.approx(np.log1p(3000)) and out["rival_phi"][3, 0] == pytest.approx(np.log1p(500))  # (RED on day 3: the snapshot opening it, labels.tables' end of day 2)


def update(model, decisions, x, config, privileged, clips):
	torch.manual_seed(7)
	m = copy.deepcopy(model)
	optimizer = make_optimizer(m, config)
	n = len(decisions)
	rng = np.random.default_rng(0)
	targets = (0.5 + 0.05 * rng.standard_normal((n, c.N_VALUE_OFFSETS))).astype(np.float32)
	growth = (10.0 + rng.standard_normal((n, c.N_VALUE_OFFSETS))).astype(np.float32)
	metrics, _ = ppo_update(m, optimizer, decisions, targets, None, config, decode=lambda rows: {k: v[rows] for k, v in x.items()}, advantages=np.zeros(n, np.float32),
		growth=growth, growth_real=growth, labels=targets, clips=clips, privileged=privileged)
	return m, metrics


def test_it_learns_on_top_of_the_critic_and_moves_nothing_else():
	"""With every group clipped apart (the live run's), an update with rivals moves the privileged critic and leaves every
	other weight bit-identical to one without; its fit starts as the critic's (its corrections 0) and is logged."""
	torch.manual_seed(5)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1)
	x = rich_inputs(8, seed=6)
	decisions = decisions_for(model.eval(), x)
	n = len(decisions)
	rivals = np.full((n, PRIVILEGED_RIVALS), -1, np.int64)
	rivals[:, 0], rivals[:, 1] = (np.arange(n) + 1) % n, (np.arange(n) + 3) % n
	present = np.zeros((n, PRIVILEGED_RIVALS), np.float32)
	present[:, :3] = 1  # (a third rival in the game without a state)
	privileged = {"rivals": rivals, "present": present, "own_phi": np.full(n, 10.0, np.float32), "rival_phi": np.full((n, PRIVILEGED_RIVALS), 9.0, np.float32), "age": np.ones((n, PRIVILEGED_RIVALS), np.float32)}
	clips = {"typical": 10.0, **{name: 1e9 for name in GRAD_GROUPS}}
	config = CampaignConfig(batch_size=4, passes=1, model_architecture="v2", credit="schedule", share_target="change")
	plain, _ = update(model, decisions, x, config, None, clips)
	with_rivals, metrics = update(model, decisions, x, config, privileged, clips)
	for (name, a), (_, b) in zip(plain.named_parameters(), with_rivals.named_parameters()):
		if name.startswith("privileged."):
			continue
		assert torch.equal(a, b), name
	assert any(not torch.equal(a, b) for (name, a), (_, b) in zip(model.named_parameters(), with_rivals.named_parameters()) if name.startswith("privileged."))
	assert metrics["value_unexplained_privileged"] == pytest.approx(metrics["value_unexplained"], rel=1e-5)
	assert metrics["privileged_correction_rms"] == 0.0 and metrics["privileged_rivals"] == pytest.approx(2.0) and metrics["privileged_present"] == pytest.approx(3.0)
	assert metrics["value_privileged"] > 0
	# privileged_credit with the corrections still 0: the same update as without it
	credit = CampaignConfig(batch_size=4, passes=1, model_architecture="v2", credit="schedule", share_target="change", privileged_credit=True)
	same, _ = update(model, decisions, x, credit, privileged, clips)
	for (name, a), (_, b) in zip(with_rivals.named_parameters(), same.named_parameters()):
		assert torch.equal(a, b), name
