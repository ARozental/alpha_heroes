"""What model.py promises to make easy to learn: ours against theirs, the balance of power, what a spell does."""
import numpy as np
import pytest
import torch

from battle_model import features as F
from battle_model import vocab
from battle_model.model import BattleModel, signed
from battle_model.schema import finalize_masks, random_inputs
from battle_model.tests.test_rules import battle

pytestmark = pytest.mark.skipif(not vocab.VOCAB_PATH.exists(), reason="run scripts/export_vocab.sh to generate the vocabulary")

LIGHTNING_BOLT, SLOW = 17, 54


def tensors(x):
	return {k: torch.from_numpy(v) for k, v in x.items()}


def test_the_signed_half_flips_with_the_team_and_the_rest_does_not():
	tokens = torch.randn(1, 3, 8)
	out = signed(tokens, torch.tensor([[1, -1, 0]]))
	assert torch.equal(out[0, 0], tokens[0, 0]) and torch.equal(out[0, 2], tokens[0, 2])
	assert torch.equal(out[0, 1, :4], -tokens[0, 1, :4]) and torch.equal(out[0, 1, 4:], tokens[0, 1, 4:])


def test_untrained_strength_shares_are_our_share_of_value_and_health():
	"""Context's learned factors start at 1: the shares it adds are ours / (ours + theirs) of value, then of health."""
	torch.manual_seed(0)
	x = tensors(random_inputs(batch=2, seed=3))
	model = BattleModel()
	captured = {}
	model.context.mlp.register_forward_hook(lambda module, args, out: captured.update(input=args[0]))
	model(x)
	b = F.read_battle(x, model.tables())
	value = F.relative(b.our_value, b.enemy_value)
	health = F.relative((b.health * b.ours).sum(1), (b.health * b.enemy).sum(1))
	shares = captured["input"][:, -2 * model.context.strengths - len(F.STRENGTH_VARIANTS) : -len(F.STRENGTH_VARIANTS)]
	assert torch.allclose(shares[:, 0], value) and torch.allclose(shares[:, -1], health.float())


def lightning_battle(our_side):
	"""Angels (attacker, slot 0) against Devils; the hero can bolt the Devils (+2000 for the attacker) or the Angels
	(-1500), and Slow either (no health changes)."""
	x = battle([("Angel", 1, 0, 86, 20, 20), ("Devil", 1, 1, 100, 10, 10)])
	x["g_our_side"][0] = our_side
	x["g_hero_can_cast"][0] = 1
	x["m_spell"][0, [LIGHTNING_BOLT, SLOW]] = True
	x["m_spell_unit"][0, LIGHTNING_BOLT, :2] = True
	x["m_spell_unit"][0, SLOW, :2] = True
	x["s_unit_value"][0, LIGHTNING_BOLT, :2] = [-1500, 2000]
	finalize_masks(x, 0)
	return tensors(x)


@pytest.mark.parametrize("our_side", (0, 1))
def test_spell_features_are_our_gain_and_against_the_best_target(our_side):
	x = lightning_battle(our_side)
	b = F.read_battle(x, BattleModel().tables())
	spell = F.spell_features(b)["spell_unit"][0]  # [S, U, features]
	net, versus, changes = (spell[..., F.SPELL_FEATURES.index(n)] for n in F.SPELL_FEATURES)
	sign = 1 if our_side == 0 else -1
	assert torch.sign(net[LIGHTNING_BOLT, 1]) == sign and torch.sign(net[LIGHTNING_BOLT, 0]) == -sign
	# against the largest gain or loss at any target: the attacker's best bolt is the best of all (1), the defender's
	# gains 1500 against the 2000 it could lose (0.75)
	assert torch.isclose(versus[LIGHTNING_BOLT, 1 if our_side == 0 else 0], torch.tensor(1.0 if our_side == 0 else 0.75))
	assert changes[LIGHTNING_BOLT, 0] == 1 and changes[SLOW, 0] == 0


def test_a_spell_that_changes_no_health_gets_its_learned_value():
	x = lightning_battle(0)
	model = BattleModel()
	b = F.read_battle(x, model.tables())
	actions = model.actions.spell_actions(F.spell_features(b), x)
	versus = actions["spell_unit"][0, :, :, F.SPELL_FEATURES.index("versus_best")]
	assert torch.allclose(versus[SLOW, :2], torch.tensor(model.actions.NO_HEALTH_VALUE))
	assert versus[LIGHTNING_BOLT, 1] == 1.0


def test_the_spell_value_moves_the_choice_once_it_matters():
	"""With a weight on net_value the bolt goes to the Devils for the attacker and to the Angels for the defender's
	hero (the enemy's units, from its side)."""
	for our_side, best in ((0, 1), (1, 0)):
		x = lightning_battle(our_side)
		model = BattleModel().eval()
		model.set_rule_weights("spell_unit", net_value=20.0)
		with torch.no_grad():
			logp = model(x)["logp_spell_unit"][0, LIGHTNING_BOLT, :2]
		assert int(logp.argmax()) == best


def test_our_frame_counts_columns_from_our_starting_side():
	hexes = torch.tensor([[17 * 5 + 1, 17 * 5 + 15]])
	attacker = F.our_frame(hexes, torch.tensor([0]))[0, :, 0]
	defender = F.our_frame(hexes, torch.tensor([1]))[0, :, 0]
	assert torch.allclose(attacker, torch.tensor([1 / 16, 15 / 16])) and torch.allclose(defender, torch.tensor([15 / 16, 1 / 16]))
