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
	net, versus, changes = (spell[..., F.SPELL_FEATURES.index(n)] for n in ("net_value", "versus_best", "changes_health"))
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


# ---- v11: what the engine predicts, what ending the battle now would score, abilities, scales ----


def ending_battle(our_side, cost=1000):
	"""Angels x2 (attacker) against Devils x3, both heroes with mana; 30 000 and 40 000 gold of armies at the start."""
	x = battle([("Angel", 2, 0, 86, 20, 20), ("Devil", 3, 1, 100, 10, 10)], our_side=our_side)
	x["u_kept"][0, :2] = x["u_count"][0, :2]
	x["g_start_army"][0] = (30_000, 40_000)
	x["g_start_mana"][0] = (20, 10)
	x["h_present"][0], x["h_power"][0], x["h_mana"][0] = (1, 1), (3, 2), (12, 10)
	x["g_surrender_cost"][0] = cost
	return x


@pytest.mark.parametrize("our_side", (0, 1))
def test_ending_now_scores_as_the_reward_would(our_side):
	"""score_now, retreat_reward and surrender_reward are reward.py's zero-sum on the assets summarize() counts."""
	from battle_model.reward import HERO_HIRE_COST, Outcome, assets_after, mana_value, zero_sum

	x = ending_battle(our_side)
	b = F.read_battle(tensors(x), BattleModel().tables())
	got = dict(zip(F.OUTCOME_FEATURES, F.outcome_features(b)[0].tolist()))
	army = [2 * vocab.creature_value(int(x["u_creature"][0, 0])), 3 * vocab.creature_value(int(x["u_creature"][0, 1]))]
	mana = [mana_value(12, 3), mana_value(10, 2)]
	start = [30_000 + mana_value(20, 3), 40_000 + mana_value(10, 2)]
	us, them = our_side, 1 - our_side
	expect = {
		"score_now": zero_sum(start[us], army[us] + mana[us], start[them], army[them] + mana[them]),
		"retreat_reward": zero_sum(start[us], assets_after(Outcome.RETREAT, army[us], 0, mana[us]), start[them], army[them] + mana[them]),
		"surrender_reward": zero_sum(start[us], assets_after(Outcome.SURRENDER, army[us], 0, mana[us], 1000), start[them], army[them] + mana[them]),
	}
	assert got == pytest.approx(expect, abs=1e-5)
	assert expect["surrender_reward"] > expect["retreat_reward"]  # paying 1000 + rehire keeps the Angels
	x = ending_battle(our_side, cost=-1)  # cannot surrender
	assert F.outcome_features(F.read_battle(tensors(x), BattleModel().tables()))[0, 2] == 0


@pytest.mark.parametrize("our_side", (0, 1))
def test_attack_values_are_our_gain_and_the_best_hex_scores_zero(our_side):
	"""a_* are attacker minus defender, like s_*: our gain either way; from-hex values against the target's best hex."""
	them = 1 - our_side
	x = battle([("Champion", 2, our_side, 86, 16, 16), ("Devil", 1, them, 88, 26, 28), ("Imp", 5, them, 70, 3, 3)], our_side=our_side)
	sign = 1 if our_side == 0 else -1
	x["m_melee_target"][0, 1] = True
	x["m_melee_from"][0, 1, [87, 104]] = True
	x["a_melee_value"][0, 1, 87], x["a_melee_value"][0, 1, 104] = 900 * sign, 1200 * sign  # jousting: a longer charge
	x["m_shoot_target"][0, 2] = True
	x["a_shoot_value"][0, 2] = -300 * sign
	finalize_masks(x, 0)
	b = F.read_battle(tensors(x), BattleModel().tables())
	melee, shot = F.attack_features(b, False), F.attack_features(b, True)
	net = F.ATTACK_FEATURES.index("net_value")
	assert melee[0, 1, net] > 0 and shot[0, 2, net] < 0
	assert torch.isclose(melee[0, 1, F.ATTACK_FEATURES.index("versus_best")], torch.tensor(1.0))
	maps, can = F.threat_maps(b, *F.distance_maps(b))
	outcomes = F.outcome_features(b)
	actions = F.action_features(b, melee, shot, maps, can, F.battle_features(b, maps, outcomes), outcomes)
	hex_value = actions["melee_from"][0, 1, :, F.FROM_FEATURES.index("hex_value")]
	assert hex_value[104] == 0 and hex_value[87] < 0


def test_the_damage_factor_is_the_formulas():
	"""+5% a point of attack over defence up to +400%, -2.5% a point under it down to -70%."""
	f = F.damage_factor(torch.tensor([0, 10, 80, 200, -10, -28, -100]))
	assert torch.allclose(f.exp(), torch.tensor([1.0, 1.5, 5.0, 5.0, 0.75, 0.3, 0.3]))


def test_abilities_are_shared_between_creatures():
	from battle_model.rules import ABILITIES, creature_tables

	tables = creature_tables()["abilities"]
	ids = {vocab.name("creatures", i): i for i in range(150)}
	assert tables[ids["Champion"], ABILITIES.index("JOUSTING")] == 1 and tables[ids["Cavalier"], ABILITIES.index("JOUSTING")] == 1
	assert tables[ids["Efreet Sultan"], ABILITIES.index("FIRE_SHIELD")] == 1 and tables[ids["Imp"], ABILITIES.index("FIRE_SHIELD")] == 0
	assert tables[ids["Cerberus"], ABILITIES.index("THREE_HEADED_ATTACK")] == 1


def test_multi_scale_features_tell_small_numbers_apart_and_do_not_saturate():
	"""4 and 10 mana differ as much as 400 and 1000 do, somewhere among the scales."""
	small, large = F.scales(torch.tensor([4.0, 10.0]), F.MANA_SCALES), F.scales(torch.tensor([400.0, 1000.0]), F.MANA_SCALES)
	assert (small[1] - small[0]).max() > 0.2 and (large[1] - large[0]).max() > 0.1
