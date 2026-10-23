"""stupid_ai.StupidAI: the contract, legality, and the pathfinding on boards whose answer is known.
check_stupid_ai.py compares it with VCMI's StupidAI itself, in real battles."""
import pytest
import torch

from battle_model import rules, vocab
from battle_model.constants import N_HEXES
from battle_model.distributions import is_legal, sample
from battle_model.export import export, verify
from battle_model.schema import empty_inputs, random_inputs
from battle_model.stupid_ai import StupidAI

pytestmark = pytest.mark.skipif(not vocab.VOCAB_PATH.exists(), reason="run scripts/export_vocab.sh to generate the vocabulary")


def test_no_trainable_parameters_and_the_onnx_export_matches(tmp_path):
	model = StupidAI()
	assert sum(p.numel() for p in model.parameters()) == 0
	verify(model, export(model, tmp_path / "stupid_ai.onnx"))


def test_its_actions_are_legal():
	x = random_inputs(batch=16, seed=3)
	with torch.no_grad():
		out = StupidAI()({k: torch.from_numpy(v) for k, v in x.items()})
	masks = {k: v for k, v in x.items() if k.startswith("m_")}
	for b in range(16):
		assert is_legal(sample(out, b=b, greedy=True), masks, b)


def walker(hex_, side=0):
	"""An open battlefield (every hex accessible but the side columns) with one one-hex unit on `hex_`."""
	x = {k: torch.from_numpy(v) for k, v in empty_inputs().items()}
	column = torch.arange(N_HEXES) % rules.FIELD_WIDTH
	x["x_access"][0] = torch.where((column == 0) | (column == rules.FIELD_WIDTH - 1), 6, rules.ACCESSIBLE)
	me = {"hex": torch.tensor([[hex_]]), "rear": torch.tensor([[-1]]), "side": torch.tensor([[side]])}
	return x, me


@pytest.mark.parametrize("start", [18, 86, 103, 168])  # columns 1, 1, 1 and 15
def test_walking_on_an_open_field_follows_the_hex_distance(start):
	x, me = walker(start)
	distances, predecessors = rules.reachability(x, me, torch.tensor([False]))
	hexes = torch.arange(N_HEXES)
	inside = rules.available(hexes)
	assert torch.equal(distances[0][inside], rules.hex_distance(torch.full_like(hexes, start), hexes)[inside])
	assert (distances[0][~inside] == rules.INFINITE_DISTANCE).all()
	# Every hex's predecessor is one step closer and next to it
	for h in hexes[inside & (hexes != start)].tolist():
		p = int(predecessors[0, h])
		assert distances[0, p] == distances[0, h] - 1 and p in rules.NEIGHBOURS[h].tolist()


def test_walking_goes_around_an_obstacle_flying_does_not():
	x, me = walker(86)  # row 5
	wall = [h for h in range(N_HEXES) if h % rules.FIELD_WIDTH == 8 and 2 <= h // rules.FIELD_WIDTH <= 8]
	x["x_access"][0, wall] = 2  # OBSTACLE: a wall across the middle rows
	walk, _ = rules.reachability(x, me, torch.tensor([False]))
	fly, _ = rules.reachability(x, me, torch.tensor([True]))
	target = 5 * rules.FIELD_WIDTH + 12
	assert fly[0, target] == rules.hex_distance(torch.tensor(86), torch.tensor(target))
	assert walk[0, target] > fly[0, target]
	assert (walk[0, wall] == rules.INFINITE_DISTANCE).all() and (fly[0, wall] == rules.INFINITE_DISTANCE).all()
