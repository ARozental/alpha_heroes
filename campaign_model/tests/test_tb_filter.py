"""TensorBoard's tracked set in critic mode (campaign_model/tb_filter.py; the user via e5, 2026-11-18)."""
from campaign_model.tb_filter import FilteredWriter, NullWriter, tracked


class Recorder:
	def __init__(self):
		self.calls = []

	def add_scalar(self, tag, value, step=None):
		self.calls.append((tag, value, step))

	def flush(self):
		self.calls.append(("flush", None, None))


def test_only_the_tracked_tags_pass_renamed():
	assert tracked("campaign_eval/score_own") == "eval/score" and tracked("campaign/last_kl") == "policy/kl"
	assert tracked("campaign_actions/move") == "actions/move" and tracked("critic/link_loss") == "critic/link_loss"
	for dead in ("eval/campaign_value_loss", "eval/campaign_value_unexplained", "campaign_heads/fresh_V7", "campaign/teacher_kl", "campaign/value_ev_d7"):
		assert tracked(dead) is None, dead
	rec = Recorder()
	w = FilteredWriter(rec)
	w.add_scalar("campaign/policy_loss", 0.5, 3)
	w.add_scalar("eval/campaign_value_unexplained", 1.0, 3)
	w.add_histogram("anything", [1, 2], 3)
	w.flush()
	assert rec.calls == [("policy/policy_loss", 0.5, 3), ("flush", None, None)]


def test_the_null_writer_does_nothing():
	w = NullWriter()
	w.add_scalar("x", 1.0, 1)
	w.flush()
	w.close()


def test_the_gpu_mode_file_reads_80_or_falls_back_to_95(tmp_path):
	from campaign_model.learner import gpu_mode

	f = tmp_path / "gpu_mode"
	assert gpu_mode(f) == 95  # (missing)
	f.write_text("80\n")
	assert gpu_mode(f) == 80
	f.write_text("95")
	assert gpu_mode(f) == 95
	f.write_text("garbage")
	assert gpu_mode(f) == 95


def test_the_critic_tab_holds_its_losses_alone():
	"""(the user via e5, 2026-11-18: "the only things I want there are the losses that should go to 0 and are active")
	critic/ the losses; critic_fit/ the unexplained readings; critic_info/ the counts and the own weight (tb_relay's
	served names, so the relay passes them through unchanged)."""
	for name in ("loss_own", "loss_pool_heldout", "loss_visible_own", "loss_visible_pool_heldout", "link_loss"):
		assert tracked(f"critic/{name}") == f"critic/{name}"
	for name in ("unexplained_own", "unexplained_pool_heldout", "unexplained_pool_heldout_after", "unexplained_visible_own", "unexplained_visible_pool_heldout"):
		assert tracked(f"critic/{name}") == f"critic_fit/{name}"
	for name in ("rows_own", "rows_pool", "own_weight"):
		assert tracked(f"critic/{name}") == f"critic_info/{name}"
	for dead in ("critic/loss_free_own", "critic/loss_free_pool", "critic/unexplained_pool", "campaign_actions/dismiss"):
		assert tracked(dead) is None, dead
