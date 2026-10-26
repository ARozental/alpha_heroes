"""Faster implementations of the rules give the same results as the plain ones they replaced."""
import torch

from battle_model import rules
from battle_model.constants import N_HEXES


def walk_distances_reference(open_, sources, steps):
	"""rules.walk_distances as it was: every hex relaxed from its neighbours each step."""
	neighbour = rules.NEIGHBOURS
	on_field = neighbour >= 0
	distance = torch.where(sources, 0, rules.INFINITE_DISTANCE)
	for _ in range(steps):
		around = torch.where(on_field, distance[..., neighbour.clamp(min=0)], rules.INFINITE_DISTANCE)
		closer = around.amin(dim=-1) + 1
		distance = torch.where((open_ | sources) & (closer < distance), closer, distance)
	return distance


def test_walk_distances_match_the_reference():
	g = torch.Generator().manual_seed(0)
	for density in (0.3, 0.6, 0.85, 1.0):
		open_ = torch.rand(64, N_HEXES, generator=g) < density
		sources = torch.zeros(64, N_HEXES, dtype=torch.bool)
		sources[torch.arange(64), torch.randint(0, N_HEXES, (64,), generator=g)] = True
		sources |= torch.rand(64, N_HEXES, generator=g) < 0.01  # several sources, some on closed hexes
		for steps in (1, 5, 24):
			assert torch.equal(rules.walk_distances(open_, sources, steps), walk_distances_reference(open_, sources, steps)), (density, steps)
