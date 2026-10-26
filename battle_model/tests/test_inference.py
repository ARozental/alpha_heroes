"""The inference server (battle_model/inference.py) gives the model's own outputs, by name and version."""
import json

import numpy as np
import pytest
import torch

from battle_model import play
from battle_model.inference import InferenceClient, InferenceServer, read_message, send_message
from battle_model.model import BattleModel
from battle_model.schema import random_inputs


@pytest.fixture(scope="module")
def served():
	torch.manual_seed(0)
	model = BattleModel().eval()
	play.SOCKET_DIR.mkdir(parents=True, exist_ok=True)
	server = InferenceServer(play.SOCKET_DIR / "test-inference.sock", device="cpu", model_kwargs={"architecture": "v3"}, threads=1)
	server.set_model("current", 3, model)
	yield server, model
	server.close()


def test_outputs_are_the_models(served):
	server, model = served
	client = InferenceClient(server.socket_path)
	x = random_inputs(batch=4, seed=1)
	version, out = client.run("current", x)
	with torch.no_grad():
		ref = model({k: torch.from_numpy(v) for k, v in x.items()})
	assert version == 3
	for name, value in out.items():
		expected = ref[name].numpy()
		assert (np.isfinite(value) == np.isfinite(expected)).all(), name
		assert np.allclose(np.where(np.isfinite(expected), value, 0), np.where(np.isfinite(expected), expected, 0), atol=1e-5), name
	assert client.run("current")[0] == 3  # the version alone
	with pytest.raises(RuntimeError, match="no model named"):
		client.run("v1", x)


def test_a_new_version_replaces_the_weights(served):
	server, model = served
	other = BattleModel().eval()
	server.set_model("v50", 50, other)
	client = InferenceClient(server.socket_path)
	x = random_inputs(batch=1, seed=2)
	version, out = client.run("v50", x)
	with torch.no_grad():
		ref = other({k: torch.from_numpy(v) for k, v in x.items()})
	assert version == 50 and np.allclose(np.nan_to_num(out["value"]), np.nan_to_num(ref["value"].numpy()), atol=1e-5)


def test_a_different_contract_is_refused(served):
	import socket

	server, _ = served
	conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
	conn.connect(server.socket_path)
	send_message(conn, json.dumps({"family": "battle", "contract_hash": "not-this-one"}).encode())
	assert "error" in json.loads(read_message(conn))
