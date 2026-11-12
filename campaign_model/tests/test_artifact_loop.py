"""The ARTIFACT loop (bundle 60; a pair of EQUIPs alternated 106 times in a game): EQUIP wears an artifact in a free slot
it fits first (AlphaMapAI's encoder; model.artifact_prior reads the same slot), so a misc artifact goes beside the one
worn, not in its place."""
import numpy as np
import torch

from campaign_model import constants as c
from campaign_model.model_v2 import CampaignModelV2
from campaign_model.tests.test_model_v2 import rich_inputs


def test_equip_reads_a_free_slot_it_fits_before_an_occupied_one():
	torch.manual_seed(3)
	model = CampaignModelV2(dim=32, cell_dim=8, layers=1).eval()
	features = model.features
	fits = features.artifact_slots
	several = [a for a in range(c.ARTIFACT_VOCAB) if int(fits[a].sum()) >= 2 and float(features.artifact_primary[a].abs().sum()) > 0]
	assert several, "no artifact with a primary bonus fitting two slots"
	art = several[0]
	slots = np.flatnonzero(fits[art].numpy())
	x = rich_inputs(2, 5)
	for b in range(2):
		x["h_artifacts"][b, 0, :] = -1
		x["h_backpack"][b, 0, :] = -1
		x["h_backpack"][b, 0, 0] = art
		x["h_artifacts"][b, 0, slots[0]] = art  # (its first slot holds its twin)
	x["h_artifacts"][1, 0, slots[1:]] = art  # (decision 1: every slot it fits taken)
	t = {k: torch.from_numpy(v) for k, v in x.items()}
	with torch.no_grad():
		f = model.domain_knowledge(t)
	equip = f.artifact_prior[:, 0, c.N_WORN_SLOTS, 1]  # (hero 0, backpack's first, EQUIP)
	assert float(equip[0]) > float(equip[1]), (float(equip[0]), float(equip[1]))  # (a free slot: the bonus gained, not swapped for its twin's)
