"""config.fixed_mix and config.real_keep (2026-11-02, the user: 90% fewer battles, most of them the campaign's, a fixed
share from the generated sources): the generated actors play only their share, an update waits at its mix and fills
from the other source only after actors.FILL_AFTER, and the battle learner gets real_keep of the campaign's battles."""

import random
import threading
from types import SimpleNamespace

from battle_model import actors, supply
from battle_model.actors import ActorPool


def pool(fixed_mix, buffer):
	p = ActorPool.__new__(ActorPool)
	p.config = SimpleNamespace(real_share=0.6, decisions_per_update=1000, max_record_age=3, ahead=1.0, buffer_decisions=100_000, fixed_mix=fixed_mix, game_share=1.0, campaign=SimpleNamespace(games=10))
	p.version, p.buffer, p.lock, p.short_since = 10, buffer, threading.Condition(), None
	p.stats = {"failures": 0, "dropped_stale": 0, "dropped_overflow": 0}
	return p


def battle(source, n, game):
	return {"source": source, "decisions": n, "version": 10, "game": game}


def test_the_generated_actors_play_only_their_share_under_a_fixed_mix():
	assert ActorPool.generated_room(pool(True, [])) == 0.4 * 1000  # no campaign battles waiting: still only their 40%
	assert ActorPool.generated_room(pool(False, [])) == 1000  # (without it: the whole update)


def test_an_update_waits_at_its_mix_then_fills(monkeypatch):
	short = [battle("generated", 100, f"g{i}") for i in range(10)] + [battle("real", 100, "r0")]  # 1,000 generated, 100 real
	assert sum(t for t in ActorPool._select(pool(True, list(short)), 1000, fill=False)[2].values()) == 500  # 400 generated + the 100 real
	assert sum(ActorPool._select(pool(True, list(short)), 1000, fill=True)[2].values()) == 1000
	p = pool(True, list(short))
	monkeypatch.setattr(actors, "FILL_AFTER", 3600)
	assert ActorPool.take(p, 1000, stop=lambda: p.short_since is not None) == (None, {})  # short: it waits (gives up here)
	monkeypatch.setattr(actors, "FILL_AFTER", 0.0)
	chosen, stats = ActorPool.take(p, 1000, stop=lambda: False)  # short past FILL_AFTER: the generated fill it
	assert sum(b["decisions"] for b in chosen) == 1000 and stats["real_share"] == 0.1 and p.short_since is None
	full = [battle("generated", 100, f"g{i}") for i in range(4)] + [battle("real", 100, f"r{i}") for i in range(6)]
	chosen, stats = ActorPool.take(pool(True, full), 1000, stop=lambda: False)
	assert stats["real_share"] == 0.6  # at its mix: no waiting


def test_the_battle_learner_gets_real_keep_of_the_campaigns_battles(monkeypatch):
	monkeypatch.setattr(supply, "parse_record", lambda data: data, raising=False)
	import battle_model.data as data_module

	monkeypatch.setattr(data_module, "parse_record", lambda data: data)
	monkeypatch.setattr(supply, "prepare", lambda record, source, game, config: ({"kind": "battle", "record": record}, None))
	runner = supply.Runner.__new__(supply.Runner)
	runner.records = SimpleNamespace(drain=lambda: {"g1:RED": list(range(4000))})
	random.seed(1)
	for keep, low, high in ((1.0, 4000, 4000), (0.0, 0, 0), (0.12, 380, 580)):
		runner.config = SimpleNamespace(real_keep=keep)
		kept = sum(1 for item in runner._battles("g1") if item.get("kind") == "battle")
		assert low <= kept <= high, (keep, kept)


def test_once_filling_the_generated_actors_play_for_the_shortfall(monkeypatch):
	# a campaign that stopped sending battles: past FILL_AFTER the generated actors may play the whole update, so the
	# update fills (else take() raised at its timeout)
	p = pool(True, [])
	assert ActorPool.generated_room(p) == 400
	monkeypatch.setattr(actors, "FILL_AFTER", 0.0)
	p.short_since = 1.0
	assert ActorPool.generated_room(p) == 1000
