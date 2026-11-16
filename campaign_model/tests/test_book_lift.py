"""ALPHA_BOOK_LIFT (e5, 2026-11-16: an eval arm's switch, never in training): model_v2.lift_books gives each legal spellbook
BUY entry the best learned part among the decision's other legal types, every other entry exactly as it was; 2 also lifts
BUY's type bias to the best legal type's; unset, the model never takes the branch."""
import torch

from campaign_model import constants as c
from campaign_model import features as ft
from campaign_model.model_v2 import lift_books

BOOK = c.BUY_ARTIFACTS.index(ft.SPELLBOOK)


def test_a_legal_book_gets_the_best_other_learned_part_and_nothing_else_moves():
	torch.manual_seed(0)
	learned = torch.randn(4, c.OWN_HEROES, c.N_BUY_ITEMS) * 10
	books = torch.rand(4, c.OWN_HEROES) > 0.5
	best = torch.randn(4) * 5
	out = lift_books(learned, books, best)
	for bi in range(4):
		for h in range(c.OWN_HEROES):
			want = best[bi] if books[bi, h] else learned[bi, h, BOOK]
			assert torch.equal(out[bi, h, BOOK], want)
	other = torch.arange(c.N_BUY_ITEMS) != BOOK
	assert torch.equal(out[..., other], learned[..., other])


def test_the_override_is_read_at_construction_and_unset_is_none(monkeypatch):
	from campaign_model.model import CustomFeatures

	monkeypatch.delenv("ALPHA_BOOK_LIFT", raising=False)
	assert CustomFeatures().book_lift is None
	monkeypatch.setenv("ALPHA_BOOK_LIFT", "2")
	assert CustomFeatures().book_lift == 2.0


def test_the_model_moves_buy_only_where_a_book_is_legal(monkeypatch):
	"""A small v2 model on test_model_v2's inputs, one hero's book legal in rows 0-1 and no BUY in rows 2-3: every output
	of rows 2-3 is as without the switch; BUY's type log-probability moves in rows 0-1; mode 2 gives BUY at least mode 1's."""
	from campaign_model.model_v2 import CampaignModelV2
	from campaign_model.tests.test_model_v2 import rich_inputs, tensors

	x = tensors(rich_inputs(4, 11))
	x["g_kind"] = torch.zeros_like(x["g_kind"])  # (turn decisions)
	mask = torch.zeros(4, c.OWN_HEROES, c.N_BUY_ITEMS, dtype=torch.bool)
	mask[:2, 0, BOOK] = True
	x["buy_mask"] = mask.reshape(x["buy_mask"].shape).to(x["buy_mask"].dtype)
	outs = {}
	for mode in (None, "1", "2"):
		if mode is None:
			monkeypatch.delenv("ALPHA_BOOK_LIFT", raising=False)
		else:
			monkeypatch.setenv("ALPHA_BOOK_LIFT", mode)
		torch.manual_seed(0)
		model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
		with torch.no_grad():
			g = torch.Generator().manual_seed(1)  # (the zero-started parts moved, the same in every mode: a fresh model's learned parts are all 0; noise on every parameter makes the MOVE table NaN, switch or not)
			for param in model.parameters():
				if not bool(param.any()):
					param.add_(0.05 * torch.randn(param.shape, generator=g))
			outs[mode] = model(dict(x))
	for mode in ("1", "2"):
		for name, value in outs[None].items():
			if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 4:
				assert torch.equal(outs[mode][name][2:], value[2:]), (mode, name)
	buy = lambda m: outs[m]["type_logp"][:2, c.ActionType.BUY]  # noqa: E731
	assert torch.isfinite(buy(None)).all() and torch.isfinite(buy("1")).all() and torch.isfinite(buy("2")).all()
	assert not torch.equal(buy("1"), buy(None))
	assert (buy("2") >= buy("1") - 1e-5).all()
