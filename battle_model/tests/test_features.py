"""features.py and how the model uses them: every feature bounded, the same when every stack is 10x larger, and
each action's features scoring that action's own logit."""
import numpy as np
import pytest
import torch

from battle_model import features as F
from battle_model import vocab
from battle_model.model import BattleModel, best_of
from battle_model.schema import finalize_masks, random_inputs
from battle_model.tests.test_rules import allow_melee, battle, melee_position

pytestmark = pytest.mark.skipif(not vocab.VOCAB_PATH.exists(), reason="run scripts/export_vocab.sh to generate the vocabulary")

BOUNDS = {"net_value": (-1, 1), "progress": (-1, 1), "threat_change": (-1, 1), "danger_change": (-1, 1), "blocked_change": (-1, 1), "attack_buff": (-1, 1), "defense_buff": (-1, 1), "morale": (np.log1p(-6 / 24), np.log1p(3 / 24)), "luck": (np.log1p(-0.5 * 3 / 24), np.log1p(3 / 24))}


def everything(x):
	"""Every feature of the inputs, by name: {name: tensor}."""
	model = BattleModel()
	b = F.read_battle({k: torch.as_tensor(v) for k, v in x.items()}, model.tables())
	reach, to_enemy = F.distance_maps(b)
	maps, can = F.threat_maps(b, reach, to_enemy)
	melee, shot = F.attack_features(b, False), F.attack_features(b, True)
	out = {}
	for names, values in ((F.STACK_FEATURES, F.stack_features(b)), (F.ATTACK_FEATURES, melee), (F.ATTACK_FEATURES, shot), (F.HEX_FEATURES, maps), (F.SUMMARY_FEATURES, F.unit_summaries(b, can)), (F.BATTLE_FEATURES, F.battle_features(b, maps))):
		for i, name in enumerate(names):
			out.setdefault(name, []).append(values[..., i].flatten())
	actions = F.action_features(b, melee, shot, maps, can, F.battle_features(b, maps))
	for name, names in (("melee_from", F.FROM_FEATURES), ("move", F.MOVE_FEATURES), ("type", F.TYPE_FEATURES)):
		for i, n in enumerate(names):
			out.setdefault(n, []).append(actions[name][..., i].flatten())
	return {name: torch.cat(parts) for name, parts in out.items()}


@pytest.mark.parametrize("seed", range(4))
def test_every_feature_is_bounded(seed):
	for name, values in everything(random_inputs(batch=6, seed=seed)).items():
		low, high = BOUNDS.get(name, (0, 1))
		assert torch.isfinite(values).all(), name
		assert values.min() >= low - 1e-6 and values.max() <= high + 1e-6, (name, float(values.min()), float(values.max()))


# Features that count whole creatures, so small stacks are lumpy: two Angels cannot kill one Devil (97 of its 160 hit
# points), twenty kill six of twenty, and the Devils' retaliation and threat shrink with what they lose. That is the
# game, not a scale problem; it fades as stacks grow.
WHOLE_CREATURES = {"kills", "threat_removed", "retaliation", "value_lost", "net_value", "threat_change"}


def scaled(x, factor):
	big = {k: v.copy() for k, v in x.items()}
	big["u_count"] = big["u_count"] * factor
	return big


def test_features_do_not_change_when_every_stack_is_ten_times_larger():
	"""The point of building features from ratios: a battle of ten times larger stacks looks the same."""
	x = melee_position()
	finalize_masks(x, 0)
	small, large = everything(x), everything(scaled(x, 10))
	for name in small:
		if name not in WHOLE_CREATURES:
			# (to half a percent: the engine rounds damage down to whole hit points)
			assert torch.allclose(small[name], large[name], atol=0.005), (name, small[name][small[name] != large[name]][:5], large[name][small[name] != large[name]][:5])


def test_large_battles_look_the_same_at_any_size():
	"""Ten and a hundred times larger: once the stacks are big, even the creature counts are scale-free."""
	x = melee_position()
	finalize_masks(x, 0)
	ten, hundred = everything(scaled(x, 10)), everything(scaled(x, 100))
	for name in ten:
		assert torch.allclose(ten[name], hundred[name], atol=0.05), (name, float((ten[name] - hundred[name]).abs().max()))


def test_each_action_scores_its_own_logit():
	"""Untrained, the pointer scores and the second-order rule terms are 0, so each choice's logit is its own features'
	first-order score, plus (for a choice with follow-ups) its best follow-up's logit."""
	torch.manual_seed(0)
	x = melee_position()
	allow_melee(x, 1, 64)
	x["m_move_hex"][0, [97, 98, 99]] = True
	finalize_masks(x, 0)
	t = {k: torch.from_numpy(v) for k, v in x.items()}
	model = BattleModel().eval()
	with torch.no_grad():
		model.set_rule_weights("melee_from", threat_change=-3.0, blocked_change=2.0, new_targets=0.5)
		out = model(t)
		b = F.read_battle(t, model.tables())
		maps, can = F.threat_maps(b, *F.distance_maps(b))
		actions = F.action_features(b, F.attack_features(b, False), F.attack_features(b, True), maps, can, F.battle_features(b, maps))
		rules = model.actions.rules
		from_logits = (actions["melee_from"] * rules["melee_from"].weight).sum(-1)
		target_logits = (actions["melee_target"] * rules["melee_target"].weight).sum(-1) + best_of(from_logits, t["m_melee_from"])
		move_logits = (actions["move"] * rules["move"].weight).sum(-1)
	# The hexes of each target: the first-order scores, normalised over that target's legal hexes
	for u in range(4):
		legal = t["m_melee_from"][0, u]
		if legal.any():
			expected = torch.log_softmax(from_logits[0, u][legal], dim=0)
			assert torch.allclose(out["logp_melee_from"][0, u][legal], expected, atol=1e-5), u
	legal = t["m_melee_target"][0]
	assert torch.allclose(out["logp_melee_target"][0][legal], torch.log_softmax(target_logits[0][legal], dim=0), atol=1e-5)
	legal = t["m_move_hex"][0]
	assert torch.allclose(out["logp_move_hex"][0][legal], torch.log_softmax(move_logits[0][legal], dim=0), atol=1e-5)


def test_a_better_attack_lifts_its_type():
	"""The best follow-up's value: making one melee target better raises the probability of melee itself."""
	torch.manual_seed(0)
	x = melee_position()
	finalize_masks(x, 0)
	t = {k: torch.from_numpy(v) for k, v in x.items()}
	model = BattleModel().eval()
	with torch.no_grad():
		before = model(t)["logp_type"][0].exp()
		model.set_rule_weights("melee_target", value_destroyed=20.0)
		after = model(t)["logp_type"][0].exp()
	melee = 3
	assert after[melee] > before[melee] + 0.05


def test_threat_maps_see_who_can_reach_a_hex():
	"""A lone Devil (speed 11, flying) can reach the acting Angel anywhere near it; nobody threatens a far corner from
	a slow Pikeman."""
	x = battle([("Angel", 1, 0, 86, 20, 20), ("Pikeman", 1, 1, 101 - 17 * 5, 4, 5)])
	finalize_masks(x, 0)
	t = {k: torch.from_numpy(v) for k, v in x.items()}
	b = F.read_battle(t, BattleModel().tables())
	maps, can = F.threat_maps(b, *F.distance_maps(b))
	threat = maps[0, :, F.HEX_FEATURES.index("threat_value")]
	pikeman_hex = 101 - 17 * 5
	assert threat[pikeman_hex - 1] == 1.0  # next to the Pikeman: it can attack there
	assert threat[17 * 9 + 2] == 0.0  # far away: a speed 4 Pikeman cannot get there
