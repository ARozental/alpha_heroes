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

	pool = SimpleNamespace(config=SimpleNamespace(real_share=0.4, decisions_per_update=3200, max_record_age=3), version=10, buffer=[])
	room = lambda: ActorPool.generated_room(pool)  # noqa: E731
	assert room() == 3200  # no real battles waiting: the whole update
	pool.buffer = [{"source": "real", "decisions": 600, "version": 10}, {"source": "real", "decisions": 900, "version": 2}]  # (the second too old)
	assert room() == 2600
	pool.buffer.append({"source": "real", "decisions": 2000, "version": 9})
	assert room() == 0.6 * 3200  # real battles enough: generated only their share
