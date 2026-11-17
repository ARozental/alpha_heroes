"""Deploy 86 (2026-11-16): M is on by default (ALPHA_MAIN_HERO unset). Tests of the behaviour from before M (Nullkiller2's
priors as they were, an old checkpoint loading bit for bit, goldens written before M) carry @pytest.mark.m_off and run
with ALPHA_MAIN_HERO=0; every other test runs M on, as the run does."""
import pytest


def pytest_configure(config):
	config.addinivalue_line("markers", "m_off: run with M off (ALPHA_MAIN_HERO=0): a test of the behaviour from before M")


@pytest.fixture(autouse=True)
def _m_off(request, monkeypatch):
	if request.node.get_closest_marker("m_off"):
		monkeypatch.setenv("ALPHA_MAIN_HERO", "0")
