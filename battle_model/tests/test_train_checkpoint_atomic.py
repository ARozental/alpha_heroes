"""battle_model.train.save_atomically: a checkpoint is written whole or not at all."""
import pytest
import torch

from battle_model.train import save_atomically


def test_a_checkpoint_is_replaced_whole_and_a_failed_save_leaves_the_old_one(tmp_path, monkeypatch):
	path = tmp_path / "checkpoint_latest.pt"
	save_atomically({"version": 1}, path)
	assert torch.load(path)["version"] == 1 and not (tmp_path / "checkpoint_latest.pt.tmp").exists()
	real = torch.save

	def torn(state, target):  # (a stop in the middle of the write)
		with open(target, "wb") as f:
			f.write(b"PK\x03\x04 half a zip")
		raise KeyboardInterrupt("signal 2")

	monkeypatch.setattr(torch, "save", torn)
	with pytest.raises(KeyboardInterrupt):
		save_atomically({"version": 2}, path)
	monkeypatch.setattr(torch, "save", real)
	assert torch.load(path)["version"] == 1  # (the old one, whole)
	save_atomically({"version": 3}, path)
	assert torch.load(path)["version"] == 3
