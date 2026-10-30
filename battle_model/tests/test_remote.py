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
