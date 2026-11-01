import math
import torch

from battle_model.train import WeightAverage


def test_the_weight_average_is_one_running_copy():
	model = torch.nn.Linear(3, 2)
	average = WeightAverage(model, decay=0.99)
	start = [p.detach().clone() for p in model.parameters()]
	assert all(not p.requires_grad for p in average.model.parameters())
	with torch.no_grad():
		for p in model.parameters():
			p.add_(1.0)
	average.update(model)  # d = min(0.99, 1/10): mostly the new weights while there are few updates
	for a, s in zip(average.model.parameters(), start):
		assert torch.allclose(a, s + 0.9)
	average.updates = 10_000  # long after the start: d = decay
	before = [p.detach().clone() for p in average.model.parameters()]
	with torch.no_grad():
		for p in model.parameters():
			p.add_(1.0)
	average.update(model)
	for a, b, m in zip(average.model.parameters(), before, model.parameters()):
		assert torch.allclose(a, 0.99 * b + 0.01 * m)
	again = WeightAverage(model, 0.99, average.model.state_dict(), average.updates)  # (a checkpoint's)
	for a, b in zip(again.model.parameters(), average.model.parameters()):
		assert torch.equal(a, b)
	assert again.updates == 10_001


def test_generated_actors_fill_what_the_real_games_leave_short():
	"""Backpressure: the generated actors stop at their share of an update, or at what the real games' waiting battles
	leave of it (the update fills with generated ones), so the learner never waits on slow real games."""
	from types import SimpleNamespace

	from battle_model.actors import ActorPool

	pool = SimpleNamespace(config=SimpleNamespace(real_share=0.4, decisions_per_update=3200, max_record_age=3, ahead=1.0, buffer_decisions=100_000), version=10, buffer=[])
	room = lambda: ActorPool.generated_room(pool)  # noqa: E731
	assert room() == 3200  # no real battles waiting: the whole update
	pool.buffer = [{"source": "real", "decisions": 600, "version": 10}, {"source": "real", "decisions": 900, "version": 2}]  # (the second too old)
	assert room() == 2600
	pool.buffer.append({"source": "real", "decisions": 2000, "version": 9})
	assert room() == 0.6 * 3200  # real battles enough: generated only their share
	pool.config.ahead = 2.0  # two updates' worth ahead (supply.py): twice the room, the real battles waiting counted once
	assert room() == 2 * 0.6 * 3200
	pool.buffer = [{"source": "real", "decisions": 600, "version": 10}]
	assert room() == 2 * 3200 - 600
	pool.config.buffer_decisions = 5000  # never more than the buffer keeps (take() drops the rest): 5,000 less the 600 real
	assert room() == 4400


def test_an_old_checkpoints_optimizer_keeps_its_moments_for_the_parameters_it_had():
	"""load_optimizer: Adam's state saved before the value priors existed maps onto the same parameters (by name) of the
	new model, and the new parameters start fresh; the progress row logs the priors' gains (1 at the start)."""
	import torch

	from battle_model.model import BattleModel
	from battle_model.train import load_optimizer, value_gains

	torch.manual_seed(0)
	model = BattleModel()
	names = [n for n, _ in model.named_parameters()]
	old_params = [p for n, p in model.named_parameters() if not n.endswith("value_gains")]
	old = torch.optim.Adam(old_params, lr=1e-4)
	for p in old_params:
		p.grad = torch.ones_like(p)
	old.step()
	state = {"model": {k: v for k, v in model.state_dict().items() if not k.endswith("value_gains")}, "optimizer": old.state_dict()}
	new = torch.optim.Adam(model.parameters(), lr=1e-4)
	load_optimizer(new, model, state)
	kept = new.state_dict()["state"]
	gains_index = names.index("actions.value_gains")
	assert gains_index not in kept and len(kept) == len(names) - 1
	first = next(i for i, n in enumerate(names) if not n.endswith("value_gains"))
	assert torch.equal(kept[first]["exp_avg"], old.state_dict()["state"][0]["exp_avg"])
	assert value_gains(model) == {"value_gain_shot": 1.0, "value_gain_melee": 1.0, "value_gain_spell": 1.0, "value_gain_creature_spell": 1.0}


def test_campaign_permits_cover_the_games_an_update_needs():
	"""Long games (joint7's ~7,000 decisions): the decisions of `ahead` updates were in from a few games, the permits
	fell to 0 and the learner waited for its min_games_per_update forever. The permits now cover those games too."""
	import threading
	from types import SimpleNamespace

	from battle_model.supply import Supply

	value = lambda v: SimpleNamespace(value=v)  # noqa: E731
	shared = SimpleNamespace(lock=threading.Lock(), started=value(3), permits=value(0))
	learner = SimpleNamespace(waiting_decisions=lambda: 100_000, waiting_games=lambda: 12)
	supply = SimpleNamespace(campaign=learner, shared=shared, received=0, game_decisions=7000.0, config=SimpleNamespace(ahead=3.0, campaign=SimpleNamespace(decisions_per_update=32768, min_games_per_update=20)))
	Supply.allow(supply)
	assert shared.permits.value == math.ceil((3.0 * 20 * 7000 - 100_000) / 7000 - 3)  # (`ahead` updates of 20 games, not of 32,768 decisions)


def test_campaign_permits_keep_games_playing_while_an_update_waits_and_runs():
	"""joint7 (2026-11-01, ~12,000 decisions a game): `ahead` updates of decisions alone were 8 games, so once an
	update's 20 were in or playing no game started, and the slots stayed empty through the update. With the update's 20
	games in and none playing, the permits keep the slots full."""
	import threading
	from types import SimpleNamespace

	from battle_model.supply import Supply

	value = lambda v: SimpleNamespace(value=v)  # noqa: E731
	shared = SimpleNamespace(lock=threading.Lock(), started=value(0), permits=value(0))
	learner = SimpleNamespace(waiting_decisions=lambda: 240_000, waiting_games=lambda: 20)
	supply = SimpleNamespace(campaign=learner, shared=shared, received=0, game_decisions=12_000.0, config=SimpleNamespace(ahead=3.0, campaign=SimpleNamespace(decisions_per_update=32768, min_games_per_update=20)))
	Supply.allow(supply)
	assert shared.permits.value == 40  # (3 updates of 20 games, one of them in)


def test_a_campaign_game_asks_the_fallback_while_its_own_server_is_down(monkeypatch):
	"""The user (2026-11-01): the training goes on when the Mac stops being available. A game asks its own endpoint
	while it answers, else a server here (campaign.fallback_device), else none starts."""
	from battle_model import supply

	up = {"tcp:mac:7951"}
	monkeypatch.setattr(supply, "inference_ready", lambda address, family: address in up)
	assert supply.campaign_socket("tcp:mac:7951", ["/here/a.sock"]) == "tcp:mac:7951"
	up = {"/here/a.sock"}
	assert supply.campaign_socket("tcp:mac:7951", ["/here/a.sock"]) == "/here/a.sock"
	up = set()
	assert supply.campaign_socket("tcp:mac:7951", ["/here/a.sock"]) is None


def test_the_fallback_store_gives_every_model_to_both_servers(monkeypatch):
	from types import SimpleNamespace

	from battle_model import supply
	from battle_model.inference import FallbackStore, ModelStore

	got = []
	fallback = SimpleNamespace(socket_paths=["/here/a.sock"], set_model=lambda name, version, model: got.append((name, version)), drop=lambda name: got.append(("drop", name)))
	closed = []
	store = FallbackStore(ModelStore("campaign", ["tcp:mac:7951", "tcp:mac:7952"], {"dim": 8}), fallback, SimpleNamespace(close=lambda: closed.append(1)))
	store.set_model("current", 3, {"w": torch.zeros(2)})
	assert got == [("current", 3)] and store.serving()["current"][0] == 3 and store.model_kwargs == {"dim": 8}
	monkeypatch.setattr(supply, "inference_ready", lambda address, family: address == "tcp:mac:7952")
	assert store.socket_for(1) == "tcp:mac:7952" and store.socket_for(0) == "/here/a.sock"  # (7951 down: the fallback)
	store.close()
	store.close()
	assert closed == [1]
