import pytest

from battle_model import vocab
from battle_model.constants import CREATURE_VOCAB, EFFECT_VOCAB, N_SECONDARY_SKILLS, N_SPELLS, SECOND_TARGET_KIND, SPELL_SACRIFICE, SPELL_TELEPORT

pytestmark = pytest.mark.skipif(not vocab.VOCAB_PATH.exists(), reason="run scripts/export_vocab.sh to generate the vocabulary")


def test_ids_are_dense_and_fit_the_model_vocabularies():
	t = vocab.tables()
	assert sorted(t["creatures"]) == list(range(CREATURE_VOCAB))
	assert sorted(t["secondary_skills"]) == list(range(N_SECONDARY_SKILLS))
	assert max(t["spells"]) < EFFECT_VOCAB


def test_known_ids():
	assert vocab.name("creatures", 0) == "Pikeman"
	assert vocab.name("creatures", 13) == "Archangel"
	assert vocab.creature_value(13) == 5000 + 3 * 500  # 5000 gold + 3 gems
	assert vocab.tables()["spells"][SPELL_TELEPORT]["key"] == "core:teleport"
	assert vocab.tables()["spells"][SPELL_SACRIFICE]["key"] == "core:sacrifice"
	assert set(SECOND_TARGET_KIND) == {SPELL_TELEPORT, SPELL_SACRIFICE}


def test_castable_spell_range():
	spells = vocab.tables()["spells"]
	assert all(not spells[s]["creature_ability"] for s in range(N_SPELLS))
	# Above the castable range: creature abilities and obstacle triggers only
	assert all(spells[s]["creature_ability"] or spells[s]["key"].endswith("Trigger") or not spells[s]["combat"] for s in spells if s >= N_SPELLS)
