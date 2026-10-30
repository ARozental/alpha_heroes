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


GPU = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None


@pytest.mark.skipif(GPU is None, reason="no GPU")
def test_a_gpu_server_decodes_one_decision_and_many():
	"""The GPU path cuts the inputs out of the batch's bytes on the device: a batch of one decision failed there
	("stride(0) must be divisible by 4", 2026-10-30, the PC's first CUDA run), a batch of several did not."""
	torch.manual_seed(0)
	model = BattleModel().eval()
	server = InferenceServer(play.SOCKET_DIR / "test-inference-gpu.sock", device=GPU, model_kwargs={"architecture": "v3"}, threads=1)
	try:
		server.set_model("current", 3, model)
		client = InferenceClient(server.socket_path)
		for batch in (1, 3):
			x = random_inputs(batch=batch, seed=batch)
			_, out = client.run("current", x)
			with torch.no_grad():
				ref = model({k: torch.from_numpy(v) for k, v in x.items()})
			for name, value in out.items():
				expected = ref[name].numpy()
				assert (np.isfinite(value) == np.isfinite(expected)).all(), name
				assert np.allclose(np.where(np.isfinite(expected), value, 0), np.where(np.isfinite(expected), expected, 0), atol=1e-3), (batch, name)
	finally:
		server.close()


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


def free_port():
	import socket

	with socket.socket() as s:
		s.bind(("127.0.0.1", 0))
		return s.getsockname()[1]


def test_a_client_on_another_machine_speaks_tcp_compressed():
	"""Over TCP (worker.py's games) the same outputs as over the Unix socket, each message zlib-compressed."""
	torch.manual_seed(1)
	model = BattleModel().eval()
	port = free_port()
	server = InferenceServer(play.SOCKET_DIR / "test-inference-tcp.sock", device="cpu", model_kwargs={"architecture": "v3"}, threads=1, tcp_host="127.0.0.1", tcp_port=port)
	try:
		server.set_model("current", 7, model)
		x = random_inputs(batch=3, seed=5)
		local, distant = InferenceClient(server.socket_path), InferenceClient(server.endpoints("127.0.0.1")[0])
		(v1, a), (v2, b) = local.run("current", x), distant.run("current", x)
		assert v1 == v2 == 7 and distant.packed and not local.packed
		for name in a:
			assert np.array_equal(np.isfinite(a[name]), np.isfinite(b[name])) and np.allclose(np.nan_to_num(a[name], neginf=0), np.nan_to_num(b[name], neginf=0), atol=1e-6), name
	finally:
		server.close()
