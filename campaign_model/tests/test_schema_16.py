"""Schema 16 (2026-11-07; the user: "make sure we can learn from all my actions"): ARMY's target slot, its moves within
one army and swaps, ONE's all-but-one half, MULTI's SPLIT_SINGLES, MERGE_ALL and GARRISON_SWAP."""
import zlib
from types import SimpleNamespace

import numpy as np
import torch

from campaign_model import constants as c
from campaign_model.model import CampaignModel, army_slots
from campaign_model.model_v2 import CampaignModelV2
from campaign_model.records import IN_LAYOUT, SCHEMA_15_BYTES, decode_inputs, padded
from campaign_model.schema import empty_inputs
from campaign_model.tests.test_model import encoded, tensors
from campaign_model.tests.test_model_v2 import rich_inputs
from campaign_model.train import chosen_logp

A, HD, HO = c.ARMY_SLOTS, c.N_HOLDERS, c.OWN_HEROES
IMP, GREMLIN, GOLEM = 42, 28, 32


def armies():
	"""A turn: hero 0 (imps 10 in slot 0, gremlins 4 in slot 1), hero 1 (imps 3 in slot 0) in the town of ours it visits
	(golems 2 in slot 0); army_slot_mask as AlphaMapAI's encodeArmyMoves writes it for hero 0's stacks."""
	x = empty_inputs(1)
	x["g_kind"][0] = c.DecisionKind.TURN
	x["h_present"][0, :2] = True
	x["h_army_creature"][0, 0, :2], x["h_army_count"][0, 0, :2] = [IMP, GREMLIN], [10, 4]
	x["h_army_creature"][0, 1, 0], x["h_army_count"][0, 1, 0] = IMP, 3
	x["t_present"][0, 0], x["t_owner"][0, 0] = True, c.OWNER_SELF
	x["t_garrison_creature"][0, 0, 0], x["t_garrison_count"][0, 0, 0] = GOLEM, 2
	for s in range(2):
		for to in (0, 1, HO):  # (within itself, to hero 1, to the town)
			for t in range(A):
				if not (to == 0 and t == s):
					x["army_slot_mask"][0, 0, s, to, t] = True
			for amount in c.ArmyAmount:
				x["army_mask"][0, 0, s, to, amount] = True
	return x


def test_the_slots_a_whole_stack_and_a_part_may_take():
	"""model.army_slots: a part's slots are the empty ones and its own creature's, never its own slot; a whole stack's
	also the other creatures' (a swap); the default is VCMI's getSlotFor between two armies, none within one."""
	slots = army_slots(tensors(armies()))
	fit, swap, default = slots.fit[0], slots.swap[0], slots.default[0]
	assert fit[0, 0, 1].tolist() == [True] + [True] * 6  # (imps onto hero 1's imps, or an empty slot)
	assert fit[0, 0, 0].tolist() == [False, False] + [True] * 5  # (within: not its own slot, not onto the gremlins)
	assert swap[0, 0, 0].tolist() == [False, True] + [False] * 5 and swap[0, 0, HO].tolist() == [True] + [False] * 6
	assert default[0, 0, 1].tolist() == [True] + [False] * 6  # (onto its own creature)
	assert default[0, 0, HO].tolist() == [False, True] + [False] * 5  # (the first empty slot)
	assert not default[0, 0, 0].any()
	assert not slots.swap_only[0, 0, 0, 1] and not slots.swap_only[0, 0, 0, HO]


def test_a_heros_last_stack_goes_out_whole_only_by_a_swap_and_its_part_still_splits_off():
	"""encodeArmyMoves' rule, as the model reads it: a hero's only stack's whole-stack slots out are the swaps alone
	(army_slot_mask), while a part of it may go into any fit slot (army_split_mask): the part's renormalization never
	sees an empty set."""
	x = armies()
	x["h_army_creature"][0, 0, 1], x["h_army_count"][0, 0, 1] = -1, 0  # (hero 0: its imps alone)
	x["army_slot_mask"][0, 0, :, HO] = False
	x["army_slot_mask"][0, 0, 0, HO, 0] = True  # (a swap with the golems, its only whole-stack slot)
	slots = army_slots(tensors(x))
	assert slots.swap_only[0, 0, 0, HO] and slots.fit[0, 0, 0, HO].tolist() == [False] + [True] * 6
	model = CampaignModel(dim=32, cell_dim=16, layers=1).eval()
	with torch.no_grad():
		out = model(tensors(x))
	row = out["army_slot_logp"][0, 0, 0, HO]
	assert torch.isfinite(row[: A]).all() and torch.isfinite(row[A + 1 :]).all() and not torch.isfinite(row[A])


def slot_terms(model, x, holder, slot, to, amount):
	"""chosen_logp's target slot term for each of ARMY's choices (the entry's logp with a cell less without one)."""
	with torch.no_grad():
		out = model(tensors(x))
	index = ((holder * A + slot) * HD + to) * c.N_ARMY_AMOUNTS + amount
	n = c.ARMY_SLOT_CHOICES + 1
	choice = {
		"kind": torch.full((n,), c.DecisionKind.TURN), "action": torch.full((n,), int(c.ActionType.ARMY)),
		"index": torch.full((n,), index), "cell": torch.arange(-1, c.ARMY_SLOT_CHOICES),
	}
	expanded = {k: v.expand(n, *v.shape[1:]) for k, v in out.items() if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == 1}
	logp, _ = chosen_logp(expanded, choice)
	return logp[1:] - logp[0]


def test_each_amount_draws_its_slot_from_its_own_choices():
	"""train.chosen_logp: ALL's slot among the whole stack's (army_slot_mask), HALF's among the fit ones, ONE's among
	the fit ones of both halves (the one moving, the one staying), each a distribution (its probabilities sum to 1) and
	no slot of another's choices taken; both models, the slot term ~0 at the default slot of a fresh model's ALL."""
	x = armies()
	for model in (CampaignModel(dim=32, cell_dim=16, layers=1).eval(), CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()):
		for to in (0, 1, HO):
			allowed = army_slots(tensors(x))
			legal, fit = allowed.legal[0, 0, 0, to], allowed.fit[0, 0, 0, to]
			none = torch.zeros_like(fit)
			for amount, choices in ((c.ArmyAmount.ALL, torch.cat([legal, none])), (c.ArmyAmount.HALF, torch.cat([fit, none])), (c.ArmyAmount.ONE, torch.cat([fit, fit]))):
				terms = slot_terms(model, x, 0, 0, to, amount)
				assert torch.isclose(terms[choices].exp().sum(), torch.tensor(1.0), atol=1e-4), (type(model).__name__, to, amount)
		terms = slot_terms(model, x, 0, 0, 1, c.ArmyAmount.ALL)
		assert terms[0] > -0.15  # (VCMI's slot, the imps onto their own: +4 against 6 others, ~0.93)
		ones = slot_terms(model, x, 0, 0, 1, c.ArmyAmount.ONE)
		assert ones[A:].exp().sum() < 0.02  # (the one staying: rare at the start, ONE moved one creature until now)


def test_a_record_of_schema_15_reads_with_its_multi_mask_widened_and_no_target_slot():
	"""records.widen_schema_15: a record of schema 15 (multi_mask 8 kinds a town, its armies after it, no army_slot_mask)
	decodes (both ways: decode_inputs, padded) with its kinds and armies as they were, the new kinds and every target
	slot none."""
	x = rich_inputs(1, 5)
	x["multi_mask"][:] = False
	x["multi_mask"][0, 2, c.MultiKind.TAKE_BEST] = x["multi_mask"][0, 5, c.MultiKind.DROP_SLOW] = True
	x["x_take_count"][0, 1, 0, 0, :2] = [7, 9]
	x["h_drop"][0, 3, 1, 4] = True
	x["army_slot_mask"][:] = False
	full = zlib.decompress(encoded(x, 0))
	names = [e[0] for e in IN_LAYOUT]
	start = sum(e[-1] for e in IN_LAYOUT[: names.index("multi_mask")])
	end = start + c.OWN_TOWNS * c.N_MULTI
	slot_start = sum(e[-1] for e in IN_LAYOUT[: names.index("army_slot_mask")])
	mask = np.frombuffer(full[start:end], dtype=np.uint8).reshape(c.OWN_TOWNS, c.N_MULTI)
	old = full[:start] + mask[:, :8].tobytes() + full[end:slot_start]
	assert len(old) == SCHEMA_15_BYTES
	got = decode_inputs([SimpleNamespace(inputs=zlib.compress(old))])
	again = decode_inputs([SimpleNamespace(inputs=zlib.compress(padded(old)))])
	for read in (got, again):
		assert read["multi_mask"][0].sum() == 2 and read["multi_mask"][0, 2, c.MultiKind.TAKE_BEST] and read["multi_mask"][0, 5, c.MultiKind.DROP_SLOW]
		assert read["x_take_count"][0, 1, 0, 0, :2].tolist() == [7, 9] and read["h_drop"][0, 3, 1, 4]
		assert not read["army_slot_mask"].any()
		for name in ("h_army_creature", "t_garrison_creature", "x_creature", "army_mask"):
			assert (read[name][0] == x[name][0]).all(), name


def test_a_checkpoint_from_before_schema_16_loads_as_the_model_was():
	"""model_v2: a state dict without the target slot's heads, the new priors and MULTI's 3 new kinds loads (the
	kinds' rows the fresh model's, the slot's queries at 0); its outputs for the old entries are the full model's."""
	torch.manual_seed(4)
	full = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	state = {k: v for k, v in full.state_dict().items() if not k.startswith(("army_slot", "army_stay_q", "features.slot_stay", "features.army_within", "features.multi_split"))}
	state["multi_kind.weight"] = state["multi_kind.weight"][:8]
	old = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	old.load_state_dict(state)
	assert all((p == 0).all() for p in (*old.army_slot_q.parameters(), *old.army_stay_q.parameters()))
	x = rich_inputs(2, 3)
	x["multi_mask"][:, :, 8:] = False
	with torch.no_grad():
		a, b = full(tensors(x)), old(tensors(x))
	for name in ("type_logp", "army_logp", "multi_logp", "move_logp"):
		assert torch.allclose(a[name], b[name], atol=1e-5), name
