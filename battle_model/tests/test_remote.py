import json
import socket
import threading
import time

import pytest

from battle_model import remote
from battle_model.train import TrainConfig
from battle_model.worker import config_from_dict


class FakePool:
	"""What remote.Hub reads of actors.ActorPool, and what it hands it."""

	class Server:
		tcp_ports = [7801]

	def __init__(self):
		self.server, self.version, self.lock, self.running = self.Server(), 12, threading.Condition(), threading.Event()
		self.running.set()
		self.added, self.failures = [], []

	def generated_waiting(self):
		return 0

	def generated_room(self):
		return 1

	def past_versions(self):
		return ["v50"]

	def _add(self, records, source, game, checked=False):
		self.added.append((records, source, game, checked))

	def _failed(self, error):
		self.failures.append(error)


def free_port():
	with socket.socket() as s:
		s.bind(("127.0.0.1", 0))
		return s.getsockname()[1]


@pytest.fixture
def hub():
	config = TrainConfig(run="remote-test", remote_port=free_port(), remote_token="secret")
	pool = FakePool()
	h = remote.Hub(config, pool)
	yield h, pool
	h.close()


def hello(port, **fields):
	conn = socket.create_connection(("127.0.0.1", port))
	message = {"protocol": remote.PROTOCOL, "token": "secret", "name": "test", "contracts": remote.contracts(), **fields}
	remote.send_frame(conn, json.dumps(message).encode())
	return conn, json.loads(remote.read_frame(conn))


def test_a_worker_without_the_token_or_on_other_code_is_refused(hub):
	h, _ = hub
	for fields in ({"token": "guess"}, {"contracts": {"battle": "other", "campaign": "other"}}, {"protocol": 0}):
		conn, reply = hello(h.config.remote_port, **fields)
		assert "error" in reply, fields
		conn.close()


def test_a_worker_gets_the_run_and_its_records_reach_the_buffer(hub):
	h, pool = hub
	conn, welcome = hello(h.config.remote_port)
	assert welcome["ok"] and welcome["battle_ports"] == [7801] and welcome["campaign_ports"] == []
	assert config_from_dict(welcome["config"]) == h.config
	kind, state = remote.read_message(conn)
	assert kind == "state" and state["version"] == 12 and state["league"] == ["v50"] and state["room"] and state["running"]

	from battle_model.data import Record

	remote.send_message(conn, ("battle", [Record([], [{}] * 5, {})], "generated", "g0_1"))
	remote.send_message(conn, ("failed", "battle", "a stall"))
	deadline = time.time() + 10
	while (not pool.added or not pool.failures) and time.time() < deadline:
		time.sleep(0.05)
	(records, source, game, checked), = pool.added
	assert source == "generated" and game.endswith(":g0_1") and game.startswith(welcome["name"]) and checked
	assert pool.failures and "a stall" in pool.failures[0]
	conn.close()


class ServingServer(FakePool.Server):
	"""A master's inference server as remote.Hub mirrors it to a --local worker."""

	model_kwargs = {"architecture": "v3"}

	def __init__(self):
		self.served = {"current": (12, {"w": 1}), "v50": (50, {"w": 2})}

	def serving(self):
		return dict(self.served)


def read_until(conn, kind):
	while True:
		message = remote.read_message(conn)
		if message[0] == kind:
			return message


def test_a_local_worker_gets_the_weights_then_only_their_changes(hub):
	h, pool = hub
	pool.server = ServingServer()
	conn, welcome = hello(h.config.remote_port, models=True)
	assert welcome["model_kwargs"] == {"battle": {"architecture": "v3"}}
	first = remote.read_message(conn)
	assert first == ("models", "battle", {"current": (12, {"w": 1}), "v50": (50, {"w": 2})})  # before any state
	assert remote.read_message(conn)[0] == "state"
	pool.server.served["current"] = (13, {"w": 3})
	del pool.server.served["v50"]
	assert read_until(conn, "models") == ("models", "battle", {"current": (13, {"w": 3}), "v50": None})
	conn.close()


def test_a_worker_on_the_masters_servers_gets_no_weights(hub):
	h, pool = hub
	pool.server = ServingServer()
	conn, _ = hello(h.config.remote_port)
	kinds = {remote.read_message(conn)[0] for _ in range(3)}
	assert kinds == {"state"}
	conn.close()


def test_apply_models_sets_and_drops():
	from battle_model.worker import apply_models

	class Server:
		def __init__(self):
			self.calls = []

		def set_model(self, name, version, state):
			self.calls.append(("set", name, version, state))

		def drop(self, name):
			self.calls.append(("drop", name))

	server = Server()
	apply_models({"battle": server}, "battle", {"current": (5, {"w": 0}), "v50": None})
	apply_models({"battle": server}, "campaign", {"current": (1, {})})  # (no such server here: ignored)
	assert server.calls == [("set", "current", 5, {"w": 0}), ("drop", "v50")]


def test_local_servers_serve_the_weights_sent():
	"""--local end to end on this machine: the servers the welcome describes, given the master's weights as the hub
	sends them, answer as the master's model does."""
	import numpy as np
	import torch

	from battle_model.inference import InferenceClient
	from battle_model.model import BattleModel
	from battle_model.schema import random_inputs
	from battle_model.worker import apply_models, local_servers

	torch.manual_seed(0)
	model = BattleModel().eval()
	state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
	config = TrainConfig(run="remote-test", inference_processes=2, inference_threads=1)
	servers = local_servers(config, {"model_kwargs": {"battle": {"architecture": "v3"}}})
	try:
		apply_models(servers, "battle", {"current": (7, state)})
		x = random_inputs(batch=2, seed=3)
		with torch.no_grad():
			ref = model({k: torch.from_numpy(v) for k, v in x.items()})
		for index in range(2):  # (each process)
			version, out = InferenceClient(servers["battle"].socket_for(index)).run("current", x)
			assert version == 7
			for name, value in out.items():
				expected = ref[name].numpy()
				assert np.allclose(np.where(np.isfinite(expected), value, 0), np.where(np.isfinite(expected), expected, 0), atol=1e-5), name
	finally:
		for server in servers.values():
			server.close()
