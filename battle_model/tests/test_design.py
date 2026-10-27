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
	x["h_level"][0, our_side], x["h_level_progress"][0, our_side], x["h_artifacts"][0, our_side] = 12, 500, 6000  # the enemy's: hidden
	x["g_surrender_cost"][0] = cost
	return x


@pytest.mark.parametrize("our_side", (0, 1))
def test_ending_now_scores_as_the_reward_would(our_side):
	"""retreat_reward and surrender_reward are reward.py's zero-sum on the assets summarize() counts."""
	from battle_model.reward import Outcome, assets_after, hero_asset, mana_value, zero_sum

	x = ending_battle(our_side)
	b = F.read_battle(tensors(x), BattleModel().tables())
	got = dict(zip(F.OUTCOME_FEATURES, F.outcome_features(b)[0].tolist()))
	army = [2 * vocab.creature_value(int(x["u_creature"][0, 0])), 3 * vocab.creature_value(int(x["u_creature"][0, 1]))]
	mana = [mana_value(12, 3), mana_value(10, 2)]
	us, them = our_side, 1 - our_side
	hero = [0.0, 0.0]
	hero[us] = hero_asset(12.5, 6000)  # what we see of the enemy's hero: nothing
	start = [30_000 + hero[0] + mana_value(20, 3), 40_000 + hero[1] + mana_value(10, 2)]
	theirs = army[them] + hero[them] + mana[them]
	expect = {
		"retreat_reward": zero_sum(start[us], assets_after(Outcome.RETREAT, army[us], hero[us], mana[us]), start[them], theirs),
		"surrender_reward": zero_sum(start[us], assets_after(Outcome.SURRENDER, army[us], hero[us], mana[us], 1000), start[them], theirs),
	}
	assert {k: got[k] for k in expect} == pytest.approx(expect, abs=1e-5)
	assert expect["surrender_reward"] > expect["retreat_reward"]  # paying 1000 + rehire keeps the Angels
	x = ending_battle(our_side, cost=-1)  # cannot surrender
	assert F.outcome_features(F.read_battle(tensors(x), BattleModel().tables()))[0, 1] == 0


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
	actions = F.action_features(b, melee, shot, maps, can, F.battle_features(b, maps))
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


def test_grouped_units_see_their_neighbours():
	"""The slow path to area attacks: each unit sees how much of its side stands next to it, and each hex how much of
	ours the acting unit would stand next to there (a Hydra or a breath hits them together)."""
	x = battle([("Angel", 1, 0, 86, 20, 20), ("Pikeman", 10, 0, 87, 4, 5), ("Hydra", 1, 1, 120, 16, 18), ("Imp", 5, 0, 30, 2, 3)])
	finalize_masks(x, 0)
	b = F.read_battle(tensors(x), BattleModel().tables())
	stack = F.stack_features(b)[0]
	allies = stack[:, F.STACK_FEATURES.index("allies_next_to")]
	assert allies[0] > 0 and allies[1] > 0 and allies[3] == 0  # the Angel and the Pikemen stand together, the Imps alone
	assert stack[2, F.STACK_FEATURES.index("enemies_next_to")] == 0
	maps, _ = F.threat_maps(b, *F.distance_maps(b))
	near = maps[0, :, F.HEX_FEATURES.index("allies_next_to")]
	assert near[88] > 0 and near[100] == 0  # next to the Pikemen (87), and nowhere near anyone


def test_escapes_are_about_never_unless_we_are_much_weaker():
	"""The escapes' prior bias (-6) holds while the armies are about even and lifts when the enemy is far stronger."""
	from battle_model import priors

	relief = lambda ratio: float(priors.escape_relief(torch.tensor(np.log(ratio))))
	assert relief(1.0) < 0.01 and relief(4.0) < 0.01 and relief(1 / 9) < 0.15  # even, stronger, the enemy 3x our army
	assert relief(1 / 100) > 0.9  # the enemy 10x our army: about as likely as waiting


def test_example_weight_counts_battles_by_what_is_at_stake():
	from battle_model.data import BattleSummary, example_weight
	from battle_model.reward import Outcome
	even = BattleSummary(our_side=0, outcome=Outcome.WIN, reward=0.0, rounds=1, own_start_value=50_000, enemy_start_value=45_000, own_kept=0, enemy_lost=0, mana_used_fraction=0)
	crush = BattleSummary(our_side=0, outcome=Outcome.WIN, reward=0.0, rounds=1, own_start_value=100_000, enemy_start_value=2_000, own_kept=0, enemy_lost=0, mana_used_fraction=0)
	assert example_weight(even) == 1.0
	assert 0.05 <= example_weight(crush) < 0.25


@pytest.mark.skipif(not vocab.VOCAB_PATH.exists(), reason="run scripts/export_vocab.sh to generate the vocabulary")
def test_rule_artifacts_reach_our_hero_and_plain_stats_do_not():
	"""Orb of Vulnerability (it negates immunities) is in our hero's bag; Centaur's Axe (+2 attack) is not."""
	from battle_model.model import HeroTokens
	ids = {r["key"]: i for i, r in vocab.tables()["artifacts"].items()}
	heroes = HeroTokens(16)
	worn = torch.full((1, 2, 19), -1)
	worn[0, 0, 0], worn[0, 0, 1] = ids["core:orbOfVulnerability"], ids["core:centaurAxe"]
	bag = heroes.rule_bag(worn)
	assert bag[0, 0].sum() == 1 and bag[0, 1].sum() == 0


def test_exposure_features_are_shares():
	x = {k: torch.from_numpy(v) for k, v in random_inputs(batch=8, seed=4).items()}
	b = F.read_battle(x, BattleModel().tables())
	e = F.exposure_features(b)
	assert e.shape == (8, len(F.EXPOSURE_FEATURES)) and (e >= 0).all() and (e <= 1).all()


def test_unsound_records_are_caught():
	"""data.check_record: an action out of range or illegal under its own masks is caught (never trained on)."""
	import numpy as np
	from battle_model.data import Record, check_record
	from battle_model.schema import empty_inputs, OUTPUTS

	x = empty_inputs(2)
	x["m_type"][:, 0] = True  # WAIT legal
	tensors = {**x, **{s.name: np.zeros((2,) + s.shape, np.float32) for s in OUTPUTS}}
	wait = {"type": 0, "hex": -1, "unit": -1, "spell": -1, "target_unit": -1, "target_hex": -1, "target2_hex": -1, "target2_unit": -1}
	assert check_record(Record([], [wait, wait], tensors)) is None
	assert "out of range" in check_record(Record([], [wait, dict(wait, type=30570)], tensors))
	assert "not legal" in check_record(Record([], [wait, dict(wait, type=1)], tensors))
	assert "decisions" in check_record(Record([], [wait], tensors))


def test_full_shot_needs_a_target_in_full_range():
	"""train.full_shot: a shooter counts as able to shoot at full damage only with a legal target within 10 hexes
	(waiting for one to come closer is no failure)."""
	import numpy as np
	from battle_model.schema import empty_inputs
	from battle_model.train import full_shot

	x = empty_inputs(1)
	x["g_active_unit"][0] = 0
	x["u_creature"][0, 0] = 1  # a creature with the distance penalty (no NO_DISTANCE_PENALTY in the vocabulary row)
	x["u_hex"][0, 0], x["u_hex"][0, 1] = 1, 15  # 14 hexes apart, the same row
	x["m_shoot_target"][0, 1] = True
	assert not full_shot(x, 0)
	x["u_hex"][0, 1] = 8  # 7 hexes
	assert full_shot(x, 0)
