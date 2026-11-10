"""The inference server (battle_model/inference.py) gives the model's own outputs, by name and version."""
import json
import time

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


def test_a_request_that_waits_long_is_said_and_a_command_that_fails_does_not_end_the_worker(capfd, monkeypatch):
	"""A request for a model never set waits MODEL_WAIT (5 s) for it, parked: with SLOW at 1 s the server says in the
	log that it waits and what the worker is at; the reply (an error) still comes. Weights that do not fit the model
	(a command of the trainer's that raises): said in the log, the model's own weights kept, the requests after it
	answered."""
	monkeypatch.setenv("ALPHA_INFERENCE_SLOW_SECONDS", "1")
	torch.manual_seed(0)
	model = BattleModel().eval()
	play.SOCKET_DIR.mkdir(parents=True, exist_ok=True)
	server = InferenceServer(play.SOCKET_DIR / "test-inference-slow.sock", device="cpu", model_kwargs={"architecture": "v3"}, threads=1)
	try:
		server.set_model("current", 3, model)
		client = InferenceClient(server.socket_path)
		x = random_inputs(batch=2, seed=4)
		before = client.run("current", x)
		with pytest.raises(RuntimeError, match="no model named"):
			client.run("v1", x)
		server.controls[0].send(("set", "battle", "current", 9, {"not_a_weight": torch.zeros(3)}))
		time.sleep(0.5)
		version, out = client.run("current", x)
		assert version == before[0] == 3 and all(np.array_equal(out[k], before[1][k], equal_nan=True) for k in out)
	finally:
		server.close()
	said = capfd.readouterr().out
	assert "a request for battle/v1 (2 decisions) has waited" in said and "the worker: idle, 1 requests parked for a model not set" in said, said
	assert "a command of the trainer's failed" in said, said


def test_a_request_for_a_model_not_set_holds_no_other_back_and_is_answered_when_its_weights_come(served):
	"""A request for a model the trainer has not set is parked up to MODEL_WAIT (5 s). Meanwhile another client's
	requests for a served model are answered at once (until 2026-11-10 they waited behind it, the worker itself
	waiting); the parked one gets its reply as soon as its model is set, and an error if it never is."""
	import threading

	server, model = served
	x = random_inputs(batch=2, seed=5)
	results = {}

	def ask(name):
		started = time.monotonic()
		try:
			results[name] = (InferenceClient(server.socket_path).run(name, x)[0], time.monotonic() - started)
		except RuntimeError as error:
			results[name] = (str(error), time.monotonic() - started)

	waiting = [threading.Thread(target=ask, args=(name,)) for name in ("later", "never")]
	for thread in waiting:
		thread.start()
	time.sleep(0.3)  # (both parked by now)
	client = InferenceClient(server.socket_path)
	started = time.monotonic()
	for _ in range(5):
		assert client.run("current", x)[0] == server.serving()["current"][0]
	assert time.monotonic() - started < 2.0, time.monotonic() - started
	server.set_model("later", 11, model)
	for thread in waiting:
		thread.join(timeout=20)
	assert results["later"][0] == 11 and results["later"][1] < 3.0, results
	assert "no model named 'never'" in results["never"][0] and 4.5 < results["never"][1] < 8.0, results
	server.drop("later")


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("family", ["battle", "campaign"])
def test_cuda_graphs_follow_new_weights(family):
	"""On CUDA the server replays recorded graphs: a weight update after the recording must show in the replies (the
	weights are copied in place), at a padded size too, as the model's own outputs."""
	from battle_model.inference import families
	from campaign_model import schema as campaign_schema

	kwargs = {"architecture": "v3"} if family == "battle" else {"architecture": "v2", "dim": 32, "cell_dim": 8, "heads": 2, "layers": 1}
	inputs_of = random_inputs if family == "battle" else campaign_schema.random_inputs
	build = families()[family].build
	torch.manual_seed(0)
	first, second = build(kwargs).eval(), build(kwargs).eval()
	with torch.no_grad():
		for p in second.parameters():
			p.add_(torch.randn_like(p) * 0.05)
	server = InferenceServer(play.SOCKET_DIR / f"test-graphs-{family}.sock", device="cuda", family=family, model_kwargs=kwargs, threads=1)
	try:
		client = InferenceClient(server.socket_path, family=family)
		for model, version in ((first, 1), (second, 2)):
			server.set_model("current", version, model)
			deadline = time.time() + 20  # (set_model returns before the server applies it: the version shows when it has)
			while client.run("current")[0] != version and time.time() < deadline:
				time.sleep(0.05)
			for batch in (1, 3):
				x = inputs_of(batch=batch, seed=batch)
				got_version, out = client.run("current", x)
				with torch.no_grad():
					ref = model({k: torch.from_numpy(v) for k, v in x.items()})
				assert got_version == version
				for name, value in out.items():
					expected = ref[name].numpy()
					assert (np.isfinite(value) == np.isfinite(expected)).all(), (family, version, batch, name)
					assert np.allclose(np.where(np.isfinite(expected), value, 0), np.where(np.isfinite(expected), expected, 0), atol=2e-3), (family, version, batch, name)
	finally:
		server.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cuda_graphs_of_two_models_share_inputs_and_stay_apart():
	"""The league's models share each size's input buffers on CUDA: asked in turn, each answers as itself."""
	torch.manual_seed(0)
	models = {"current": BattleModel().eval(), "v50": BattleModel().eval()}
	server = InferenceServer(play.SOCKET_DIR / "test-graphs-two.sock", device="cuda", model_kwargs={"architecture": "v3"}, threads=1)
	try:
		for version, (name, model) in enumerate(models.items(), start=1):
			server.set_model(name, version, model)
		client = InferenceClient(server.socket_path)
		deadline = time.time() + 20
		while time.time() < deadline:
			try:
				client.run("v50")
				break
			except RuntimeError:
				time.sleep(0.05)
		for round_ in range(3):
			for name, model in models.items():
				x = random_inputs(batch=2, seed=round_)
				_, out = client.run(name, x)
				with torch.no_grad():
					ref = model({k: torch.from_numpy(v) for k, v in x.items()})
				for key, value in out.items():
					expected = ref[key].numpy()
					assert np.allclose(np.where(np.isfinite(expected), value, 0), np.where(np.isfinite(expected), expected, 0), atol=2e-3), (name, round_, key)
	finally:
		server.close()
