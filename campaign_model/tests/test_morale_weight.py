"""The fresh run's morale weight in our best army (agents/AlphaMapAI/MoraleWeight.h, the user 2026-11-18: "a 4% chance
of getting an extra turn is not 1.04 times stronger ... make that half this effect"), through its self-test
(tools/MoraleSelfTest.cpp, compiled here): weight 1 is Nullkiller2's weighting bit for bit; 0.5 halves each chance;
takeBest's rule (a faction admitted while the army's weighed power rises) admits a cheap second faction at 0.5 that 1.0
does not, and still leaves a lone skeleton out."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "agents" / "AlphaMapAI" / "tools" / "MoraleSelfTest.cpp"
BAD, GOOD, DICE = (2, 4, 6), (1, 2, 3), 24


@pytest.fixture(scope="module")
def run(tmp_path_factory):
	compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
	if compiler is None:
		pytest.skip("no C++ compiler")
	exe = tmp_path_factory.mktemp("morale") / "morale_selftest"
	subprocess.run([compiler, "-std=c++17", "-O2", str(TOOL), "-o", str(exe)], check=True)

	def ask(*commands):
		out = subprocess.run([str(exe)], input="\n".join(commands) + "\n", capture_output=True, text=True, check=True).stdout.split()
		return out

	return ask


def old_multiplier(morale):
	"""Armies.cpp's moraleWeighted before the patch (Nullkiller2's ArmyManager::getBestArmy weighting), in doubles."""
	m = 1.0
	if morale < 0:
		m -= 1.0 / DICE * BAD[min(len(BAD), -morale) - 1]
	elif morale > 0:
		m += 1.0 / DICE * GOOD[min(len(GOOD), morale) - 1]
	return m


def test_weight_one_is_the_old_weighting_bit_for_bit(run):
	morales = range(-4, 5)
	got = run(*[f"mult {m} 1" for m in morales])
	assert [float(g) for g in got] == [old_multiplier(m) for m in morales]
	powers = (1, 7, 999, 123457, 9876543211)
	got = run(*[f"value 1 {p} {m}" for p in powers for m in morales])
	assert [int(g) for g in got] == [int(old_multiplier(m) * float(p)) for p in powers for m in morales]


def test_half_weight_halves_each_chance(run):
	got = [float(g) for g in run(*[f"mult {m} 0.5" for m in (1, 2, 3, -1, -2, -3)])]
	assert got == pytest.approx([1 + 1 / 48, 1 + 2 / 48, 1 + 3 / 48, 1 - 2 / 48, 1 - 4 / 48, 1 - 6 / 48], abs=1e-12)


def admitted(run, weight, alone, joined):
	"""takeBest's rule: the next faction joins when the joined army's weighed power is above the army without it."""
	best, value = (int(v) for v in run(f"value {weight} " + " ".join(f"{p} {m}" for p, m in alone), f"value {weight} " + " ".join(f"{p} {m}" for p, m in joined)))
	return value > best


def test_a_cheap_second_faction_joins_at_half_weight_and_not_at_full(run):
	# one faction: morale +1; with a second faction's stack worth 3% of it: both at 0 (two factions)
	alone, joined = [(1000, 1)], [(1000, 0), (30, 0)]
	assert not admitted(run, 1.0, alone, joined)
	assert admitted(run, 0.5, alone, joined)


def test_a_lone_skeleton_is_left_out_at_both_weights(run):
	# a single-faction army at +1; with one skeleton (undead: the living at -1, the skeleton at 0) it would be weaker
	alone, joined = [(800, 1), (200, 1)], [(800, -1), (200, -1), (6, 0)]
	assert not admitted(run, 1.0, alone, joined)
	assert not admitted(run, 0.5, alone, joined)
