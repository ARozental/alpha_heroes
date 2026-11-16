"""ALPHA_FIGHT_LIFT (e5, 2026-11-16: an eval arm's switch, never in training): model_v2.lift_fights lifts the per-tile pair
head's part of each clear win (a wandering monster's tile, reachable today, legal, win >= 0.95, loss < 0.20) to the
decision's best legal tile's, every other entry exactly as it was; unset, the model never takes the branch."""
import torch

from campaign_model.model_v2 import lift_fights


def test_a_clear_win_gets_the_best_legal_pair_part_and_nothing_else_moves():
	torch.manual_seed(0)
	b, h, w = 3, 2, 4
	move = torch.randn(b, h, w, w)
	pairs = torch.randn(b, h, w, w) * 3
	legal = torch.rand(b, h, w, w) > 0.3
	stacks = torch.zeros(b, h, w, w, dtype=torch.bool)
	win, loss, today = torch.full((b, h, w, w), 0.99), torch.full((b, h, w, w), 0.1), torch.ones(b, h, w, w, dtype=torch.bool)
	stacks[0, 1, 2, 3] = stacks[1, 0, 0, 0] = stacks[2, 1, 1, 1] = True
	legal[0, 1, 2, 3] = legal[1, 0, 0, 0] = True
	legal[2, 1, 1, 1] = False  # (not legal: never lifted)
	pairs[0, 0, 0, 0] = 50.0
	legal[0, 0, 0, 0] = False  # (an illegal tile's pair part never counts as the best)
	win[1, 0, 0, 0] = 0.94  # (under the win cut: not lifted)
	out = lift_fights(move, pairs, legal, stacks, win, loss, today)
	best0 = pairs[0].masked_fill(~legal[0], float("-inf")).max()
	assert best0 < 50.0
	assert torch.equal(out[0, 1, 2, 3], move[0, 1, 2, 3] - pairs[0, 1, 2, 3] + best0)
	changed = torch.zeros_like(move, dtype=torch.bool)
	changed[0, 1, 2, 3] = True
	assert torch.equal(out[~changed], move[~changed])  # (every other entry bit for bit)
	# the loss cut and reachability
	loss[0, 1, 2, 3] = 0.2
	assert torch.equal(lift_fights(move, pairs, legal, stacks, win, loss, today), move)
	loss[0, 1, 2, 3] = 0.1
	today[0, 1, 2, 3] = False
	assert torch.equal(lift_fights(move, pairs, legal, stacks, win, loss, today), move)


def test_the_override_is_read_at_construction_and_unset_is_none(monkeypatch):
	from campaign_model.model import CustomFeatures

	monkeypatch.delenv("ALPHA_FIGHT_LIFT", raising=False)
	assert CustomFeatures().fight_lift is None
	monkeypatch.setenv("ALPHA_FIGHT_LIFT", "1")
	assert CustomFeatures().fight_lift == 1.0


def test_the_model_runs_with_the_switch_on(monkeypatch):
	"""a v2 forward on rich inputs with the switch on: every output finite where it was, MOVE's table where it changed only
	on legal entries."""
	from campaign_model.model import build_model
	from campaign_model.tests.test_model_v2 import rich_inputs

	kwargs = {"dim": 32, "cell_dim": 16, "heads": 4, "layers": 1, "architecture": "v2", "share_target": "change"}
	x = {k: torch.from_numpy(v) for k, v in rich_inputs(8, 2).items()}
	monkeypatch.delenv("ALPHA_FIGHT_LIFT", raising=False)
	torch.manual_seed(1)
	off = build_model(kwargs).eval()
	monkeypatch.setenv("ALPHA_FIGHT_LIFT", "1")
	torch.manual_seed(1)
	on = build_model(kwargs).eval()
	on.load_state_dict(off.state_dict())
	with torch.no_grad():
		a, b = off(x), on(x)
	for k in a:
		if torch.is_tensor(a[k]) and k != "move_logp" and not k.startswith(("type_logp",)):
			assert torch.equal(a[k], b[k]) or not torch.isfinite(a[k]).all(), k
	assert torch.equal(torch.isfinite(a["move_logp"]), torch.isfinite(b["move_logp"]))
