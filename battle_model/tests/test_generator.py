"""The battle generator (battle_model/generator.py): its draws are valid engine data and follow generator.toml."""
import random

import pytest

from battle_model import generator

N = 300


@pytest.fixture(scope="module")
def drawn():
	cfg, w = generator.load_config(), generator.world()
	rng = random.Random(0)
	return cfg, w, [generator.sample(rng, cfg, w) for _ in range(N)]


def test_reproducible():
	assert generator.sample(random.Random(5)) == generator.sample(random.Random(5))


def test_probabilities_sum_to_one():
	cfg = generator.load_config()

	def groups(node, path=""):
		if isinstance(node, dict):
			choices = [v for v in node.values() if isinstance(v, dict) and "p" in v and "desc" in v]
			if choices and path != "composition.war_machines":  # drawn independently
				yield path, sum(c["p"] for c in choices)
			for k, v in node.items():
				yield from groups(v, f"{path}.{k}" if path else k)

	for path, total in groups(cfg):
		assert total == pytest.approx(1.0, abs=1e-6), path
	assert sum(p["p"] for p in cfg["phase"].values()) == pytest.approx(1.0)


def test_identifiers_exist(drawn):
	_, w, scenarios = drawn
	import json
	from battle_model import vocab
	v = json.loads(vocab.VOCAB_PATH.read_text())
	known = {kind: {generator.short(e["key"]) for e in v[kind]} for kind in ("creatures", "heroes", "spells", "artifacts", "secondary_skills")}
	for s in scenarios:
		for side in s["sides"]:
			assert all(c in known["creatures"] and n >= 1 for c, n in side["army"])
			assert 1 <= len(side["army"]) <= 7 and len({c for c, _ in side["army"]}) == len(side["army"])
			if "hero" not in side:
				continue
			assert side["hero"] in known["heroes"]
			assert set(side["spells"]) <= known["spells"]
			assert all(a in known["artifacts"] for _, a in side["artifacts"])
			assert all(k in known["secondary_skills"] and 1 <= m <= 3 for k, m in side["secondary_skills"])


def test_heroes(drawn):
	cfg, w, scenarios = drawn
	levels = []
	for s in scenarios:
		for side in s["sides"]:
			if "hero" not in side:
				continue
			skills = dict(side["secondary_skills"])
			assert not set(skills) & set(cfg["hero"]["excluded_secondary_skills"]["skills"]) and len(skills) <= generator.MAX_SECONDARY
			slots = [slot for slot, _ in side["artifacts"]]
			assert len(slots) == len(set(slots))
			assert side["spellbook"] or not side["spells"]
			top = generator.LEARNABLE[skills.get("wisdom", 0)]
			level_of = {generator.short(e["key"]): e["level"] for e in w["spells"]}
			assert all(level_of[sp] <= top for sp in side["spells"])
			level = sum(1 for e in w["experience"] if e <= side["experience"])
			levels.append(level)
			assert sum(side["primary_skills"]) >= level - 1  # a point per level up, above the class's initial ones
	top = max(p["hero_level"]["max"] for p in cfg["phase"].values()) + cfg["hero"]["level_tail"]["extra"]["max"]
	assert min(levels) >= 1 and max(levels) <= top


def test_force_ratio(drawn):
	cfg, w, scenarios = drawn
	ratios = []
	for s in scenarios:
		values = [generator.army_value(side["army"], w) for side in s["sides"]]
		if "neutrals" in s:
			values.append(generator.creature_value(s["neutrals"]["creature"], w) * s["neutrals"]["count"])
		ratios.append(max(values) / min(values))
	ratios.sort()
	assert ratios[len(ratios) // 2] < 1.6  # mostly even (counts are whole creatures and stacks are redrawn)
	assert ratios[-1] < 30
	assert sum(r > 3 for r in ratios) / len(ratios) > 0.04  # but lopsided at times


def test_battle_types(drawn):
	_, _, scenarios = drawn
	sieges = [s for s in scenarios if "town" in s]
	assert 0.1 < len(sieges) / len(scenarios) < 0.4
	assert all(s["town"]["fortification"] in generator.FORTIFICATIONS for s in sieges)
	assert any("hero" not in s["sides"][1] for s in sieges)  # garrisons
	assert all("hero" in s["sides"][0] for s in scenarios)
	neutrals = [s for s in scenarios if "neutrals" in s]
	assert 0.25 < len(neutrals) / len(scenarios) < 0.55
	assert all(len(s["sides"]) == 1 and "town" not in s and s["neutrals"]["count"] >= 1 for s in neutrals)
	assert all(len(s["sides"]) == 2 for s in scenarios if "neutrals" not in s)
	assert all(s["terrain"] not in generator.UNSUPPORTED_TERRAINS for s in scenarios)


def test_hard_cases(drawn):
	cfg, w, scenarios = drawn
	hard = [s for s in scenarios if s["_about"].startswith("hard one_big_stack")]
	assert 0.03 < len(hard) / len(scenarios) < 0.2  # cfg hard.p of them
	h = cfg["hard"]["one_big_stack"]
	for s in hard:
		side = s["sides"][0]
		assert s["neutrals"]["upgraded_stack"] is False and side["spellbook"] and side["mana_fraction"] == 1.0
		ratio = generator.creature_value(s["neutrals"]["creature"], w) * s["neutrals"]["count"] / generator.army_value(side["army"], w)
		assert h["value_ratio"]["min"] * 0.8 < ratio < h["value_ratio"]["max"] * 1.2  # counts are whole creatures
		assert set(side["spells"]) & set(h["spells"])


def test_enemy_shooters_ratio_counts_our_flyer_and_the_hopeless_hero_is_worth_saving():
	"""enemy_shooters: the enemy's army against ours with its flyer is within value_ratio (it was drawn against the army
	before the flyer: 0.69-1.0 of it, 2026-10-28 review); hopeless: our hero at hero_level or above (a hero worth less
	than its rehire keeps nothing by retreating)."""
	cfg, w = generator.load_config(), generator.world()
	rng = random.Random(3)
	h = cfg["hard"]["enemy_shooters"]
	for _ in range(20):
		ours, theirs = generator.hard_enemy_shooters(rng, cfg, w)["sides"]
		ratio = generator.army_value(theirs["army"], w) / generator.army_value(ours["army"], w)
		assert h["value_ratio"]["min"] * 0.8 < ratio < h["value_ratio"]["max"] * 1.2  # counts are whole creatures
	lowest = cfg["hard"]["hopeless"]["hero_level"]["min"]
	for _ in range(20):
		experience = generator.hard_hopeless(rng, cfg, w)["sides"][0]["experience"]
		assert sum(1 for e in w["experience"] if e <= experience) >= lowest

