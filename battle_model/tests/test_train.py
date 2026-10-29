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
