"""The starting biases (priors.py, battle_model/priors/*.csv) and how the model uses them."""
import pytest
import torch

from battle_model import priors, rules, vocab
from battle_model.constants import N_ACTION_TYPES, N_SPELLS, ActionType
from battle_model.model import BattleModel
from battle_model.schema import empty_inputs


def test_action_type_file_lists_every_type_once_by_its_name():
	rows = priors.read("action_types.csv")
	assert sorted(rows) == list(range(N_ACTION_TYPES))
	assert all(name == ActionType(i).name for i, (name, _) in rows.items())


@pytest.mark.skipif(not vocab.VOCAB_PATH.exists(), reason="run scripts/export_vocab.sh to generate the vocabulary")
def test_spell_file_lists_every_spell_once_by_its_name():
	rows = priors.read("spells.csv")
	assert sorted(rows) == list(range(N_SPELLS))
	assert all(name == vocab.name("spells", i) for i, (name, _) in rows.items())


def test_the_model_starts_at_the_files_biases():
	model = BattleModel()
	assert torch.equal(model.actions.type_head.bias.detach(), priors.action_types())
	assert torch.equal(model.actions.spell_head.bias.detach(), priors.spells())
	assert torch.equal(model.actions.spell_mastery.detach(), priors.mastery())


@pytest.mark.skipif(not vocab.VOCAB_PATH.exists(), reason="run scripts/export_vocab.sh to generate the vocabulary")
def test_mastery_bias_follows_our_heros_skill_in_the_spells_school():
	"""Expert Air Magic: Lightning Bolt (air) gets the expert bias; Slow (earth) none; Magic Arrow (all four) expert."""
	x = {k: torch.from_numpy(v) for k, v in empty_inputs().items()}
	x["g_our_side"][0] = 1
	x["h_skills"][0, 1, rules.SCHOOL_SKILLS[rules.SCHOOLS.index("air")]] = 3
	x["h_skills"][0, 0, rules.SCHOOL_SKILLS[rules.SCHOOLS.index("earth")]] = 3  # the enemy's: does not count
	bias = BattleModel().actions.mastery_bias(x)[0]
	spell = lambda key: next(i for i, r in vocab.tables()["spells"].items() if r["key"] == "core:" + key)
	assert bias[spell("lightningBolt")] == priors.MASTERY_BIAS[3]
	assert bias[spell("slow")] == priors.MASTERY_BIAS[0]
	assert bias[spell("magicArrow")] == priors.MASTERY_BIAS[3]
