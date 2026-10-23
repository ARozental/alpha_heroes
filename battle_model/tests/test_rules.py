"""rules.py against the engine's own formulas, and the model's kill rule on hand-made positions."""
import math
from collections import deque

import numpy as np
import pytest
import torch

from battle_model import vocab
from battle_model.constants import FIELD_WIDTH, N_HEXES, ActionType
from battle_model.distributions import sample
from battle_model.model import BattleModel
from battle_model.rules import creature_tables, cross_stack_features, hex_distance, least_damage
from battle_model.schema import empty_inputs, finalize_masks

ROWS = N_HEXES // FIELD_WIDTH


def neighbours(hex_):
	"""BattleHex::moveInDirection for the six directions (lib/battle/BattleHex.h): odd rows are shifted."""
	y, x = divmod(hex_, FIELD_WIDTH)
	odd = y % 2
	steps = [(x - 1 if odd else x, y - 1), (x if odd else x + 1, y - 1), (x + 1, y), (x if odd else x + 1, y + 1), (x - 1 if odd else x, y + 1), (x - 1, y)]
	return [ny * FIELD_WIDTH + nx for nx, ny in steps if 0 <= nx < FIELD_WIDTH and 0 <= ny < ROWS]


def test_hex_distance_is_the_number_of_steps():
	for start in (0, 18, 35, 86, 93, 152, 186):
		steps = {start: 0}
		queue = deque([start])
		while queue:
			h = queue.popleft()
			for n in neighbours(h):
				if n not in steps:
					steps[n] = steps[h] + 1
					queue.append(n)
		hexes = torch.tensor(sorted(steps))
		got = hex_distance(torch.full_like(hexes, start), hexes)
		assert got.tolist() == [steps[h] for h in sorted(steps)], start


def engine_least_damage(count, single_min, attack, defense, reduction=0, halvings=0):
	"""DamageCalculator in double precision, as the engine does it: the lower end of the damage range."""
	advantage = attack - defense
	attack_total = 1.0 + (min(0.05 * advantage, 4.0) if advantage > 0 else 0.0)
	defense_total = 1.0
	for factor in [min(0.025 * -advantage, 0.7) if advantage < 0 else 0.0, reduction / 100.0] + [0.5] * halvings:
		defense_total *= 1 - min(1.0, factor)
	return max(1, math.floor(count * single_min * (attack_total * defense_total)))


def test_least_damage_never_exceeds_the_engine_and_misses_it_only_by_its_rounding():
	rng = np.random.default_rng(0)
	cases = [(int(rng.integers(1, 300)), int(rng.integers(1, 60)), int(rng.integers(0, 60)), int(rng.integers(0, 60)), int(rng.choice([0, 0, 15, 30])), int(rng.integers(0, 3))) for _ in range(20000)]
	cases += [(c, m, a, d, 0, 0) for c in (1, 2, 5, 10) for m in (1, 7, 10, 50) for a in range(0, 41, 3) for d in range(0, 41, 3)]
	t = torch.tensor(cases)
	ours = least_damage(*t.unbind(dim=1)).tolist()
	for case, got in zip(cases, ours):
		engine = engine_least_damage(*case)
		assert got == engine or (got == engine - 1 and float(engine) == count_times_factor(*case)), (case, got, engine)
	# The engine's floating point really slips: 5 damage at 24 defence points over attack is exactly 5 x 0.4 = 2, and
	# 1 in the engine (1 - 0.6000000000000001 = 0.3999999999999999)
	assert engine_least_damage(1, 5, 0, 24) == 1 and least_damage(*torch.tensor([[1, 5, 0, 24, 0, 0]]).unbind(dim=1)).item() == 1


def count_times_factor(count, single_min, attack, defense, reduction, halvings):
	"""The exact damage as a fraction, for the comparison above."""
	from fractions import Fraction

	over, under = min(max(attack - defense, 0), 80), min(max(defense - attack, 0), 28)
	return float(Fraction(count * single_min * (20 + over) * (40 - under) * (100 - reduction), 20 * 40 * 100 * 2**halvings))


pytestmark = pytest.mark.skipif(not vocab.VOCAB_PATH.exists(), reason="run scripts/export_vocab.sh to generate the vocabulary")


def creature_id(name):
	return next(r["id"] for r in vocab.tables()["creatures"].values() if r["name"] == name)


def battle(units, active=0, our_side=0):
	"""Model inputs for one position: units = [(name, count, side, hex, attack, defense), ...] with the
	creature's own hit points and damage; slot 0 acts. Every unit is alive and full health."""
	x = empty_inputs()
	x["g_our_side"][0], x["g_active_unit"][0], x["g_round"][0] = our_side, active, 1
	for slot, (name, count, side, hex_, attack, defense) in enumerate(units):
		row = vocab.creature(creature_id(name))
		values = {"u_present": 1, "u_alive": 1, "u_creature": row["id"], "u_side": side, "u_owner_side": side, "u_hex": hex_, "u_count": count, "u_first_hp": row["hit_points"], "u_max_hp": row["hit_points"],
			"u_attack": attack, "u_defense": defense, "u_dmg_min": row["damage_min"], "u_dmg_max": row["damage_max"], "u_speed": row["speed"], "u_shots": row["shots"], "u_retaliations": 1}
		for key, value in values.items():
			x[key][0, slot] = value
	x["m_move_hex"][0, 100] = True
	return x


def allow_melee(x, slot, hex_):
	x["m_melee_from"][0, slot, hex_] = True
	x["m_melee_target"][0, slot] = True


def greedy(model, x):
	finalize_masks(x, 0)
	with torch.no_grad():
		return sample(model({k: torch.from_numpy(v) for k, v in x.items()}), greedy=True)


def melee_position():
	"""Two Angels (attack 20, 50 damage each) next to three stacks: three Imps and a Pit Fiend they kill whatever
	the dice, and two Devils they cannot (2 x 50 at -2.5% is 97 of their 320 hit points), the most valuable.

	Against the Imps and the Pit Fiend the damage is exactly 185 and 135; whole numbers from fractional factors
	count one less (rules.least_damage), so 184 and 134."""
	x = battle([("Angel", 2, 0, 80, 20, 20), ("Imp", 3, 1, 81, 3, 3), ("Pit Fiend", 1, 1, 97, 13, 13), ("Devil", 2, 1, 64, 19, 21)])
	for slot, hex_ in ((1, 80), (2, 80), (3, 80)):
		allow_melee(x, slot, hex_)
	return x


def test_least_damage_and_kills_for_sure_in_a_position():
	x = melee_position()
	f = cross_stack_features({k: torch.from_numpy(v) for k, v in x.items()}, creature_tables())
	assert f["least_melee"][0, 1:4].tolist() == [184, 134, 97]  # 100 x 1.85, x 1.35 (whole: one less), x 0.975 (floor)
	assert f["total_hp"][0, 1:4].tolist() == [12, 45, 320]
	assert f["kills_melee"][0, :4].tolist() == [0.0, 1.0, 1.0, 0.0]  # never our own Angels
	values = [150.0, 500.0, 2 * 3200.0]
	assert f["value_share"][0, 1:4].tolist() == pytest.approx([v / sum(values) for v in values])


@pytest.mark.parametrize("seed", range(5))
def test_untrained_model_attacks_the_most_valuable_stack_it_kills_for_sure(seed):
	torch.manual_seed(seed)
	model = BattleModel().eval()
	action = greedy(model, melee_position())
	assert (action["type"], action["unit"]) == (ActionType.MELEE, 2)  # the Pit Fiend, not the Imps nor the Devils


@pytest.mark.parametrize("seed", range(5))
def test_shooting_far_halves_the_damage(seed):
	"""Ten Archers (20 least damage, +15%: 23, a whole number, so 22) kill four Imps 3 hexes away for sure, and would
	kill the more valuable five Imps 15 hexes away if not for the halved damage at that range (11 of their 20 hit points)."""
	torch.manual_seed(seed)
	x = battle([("Archer", 10, 0, 35, 6, 3), ("Imp", 4, 1, 38, 3, 3), ("Imp", 5, 1, 50, 3, 3)])
	x["m_shoot_target"][0, 1:3] = True
	f = cross_stack_features({k: torch.from_numpy(v) for k, v in x.items()}, creature_tables())
	assert f["least_shot"][0, 1:3].tolist() == [22, 11] and f["kills_shot"][0, 1:3].tolist() == [1.0, 0.0]
	action = greedy(BattleModel().eval(), x)
	assert (action["type"], action["unit"]) == (ActionType.SHOOT, 1)


def test_the_rule_changes_nothing_without_a_legal_sure_kill():
	"""The Imps would die for sure, but only the Devils may be attacked: the outputs are the untrained network's."""
	x = melee_position()
	x["m_melee_target"][0, 1:3] = False
	x["m_melee_from"][0, 1:3] = False
	finalize_masks(x, 0)
	inputs = {k: torch.from_numpy(v) for k, v in x.items()}
	torch.manual_seed(0)
	model = BattleModel().eval()
	with torch.no_grad():
		with_rule = model(inputs)
		model.kill_rule_weight.zero_()
		without = model(inputs)
	for name in ("logp_type", "logp_melee_target", "logp_shoot_target"):
		assert torch.equal(with_rule[name], without[name]), name
