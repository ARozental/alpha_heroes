"""TensorBoard's tracked set in critic mode (the user via e5, 2026-11-18: "only the long runs are relevant to keep in the
TB and only the things we actually track"): the campaign writer passes only WHITELIST's tags, renamed, into
<run>/tensorboard/main; progress.csv keeps every column. NullWriter: no TensorBoard at all (the frozen battle model's
run writer)."""

WHITELIST = {
	# eval/ (evaluate.py's campaign_eval/*, our own policy's)
	"campaign_eval/score_own": "eval/score", "campaign_eval/score_own_se": "eval/score_se", "campaign_eval/score_own_d28": "eval/score_d28",
	"campaign_eval/eliminated_own": "eval/eliminated", "campaign_eval/won_own": "eval/won", "campaign_eval/short": "eval/short",
	"campaign_eval/max_level_ours": "eval/max_level_ours", "campaign_eval/max_level_theirs": "eval/max_level_theirs", "campaign_eval/major_won": "eval/major_won",
	# the critic's (the learner's critic/ writes; the user via e5, 2026-11-18: "the only things I want there are the losses
	# that should go to 0 and are active"): critic/ its losses alone, critic_fit/ the unexplained readings, critic_info/
	# the counts and the own weight (runtime/tb_relay.py's names: the relay a pass-through on them)
	**{f"critic/{name}": f"critic/{name}" for name in ("loss_own", "loss_pool_heldout", "loss_visible_own", "loss_visible_pool_heldout", "link_loss", "loss_pool", "loss_visible_pool", "fixed_six_own", "fixed_six_pool", "fixed_six_pool_heldout", "fixed_six_visible_own", "fixed_six_visible_pool", "fixed_six_visible_pool_heldout", "fixed_end_own", "fixed_end_pool", "fixed_end_pool_heldout", "fixed_end_visible_own", "fixed_end_visible_pool", "fixed_end_visible_pool_heldout")},
	**{f"critic/{name}": f"critic_fit/{name}" for name in (
		"unexplained_own", "unexplained_own_med5", "unexplained_pool_heldout", "unexplained_pool_heldout_after", "unexplained_visible_own", "unexplained_visible_pool_heldout",
	)},
	**{f"critic/{name}": f"critic_info/{name}" for name in ("rows_own", "rows_pool", "own_weight")},
	# (e5, 2026-11-19: unexplained_own reads the V7 forecasts' calibration, 1.5x too wide, and hides their ranking, corr
	# +0.24 on our rows against +0.21 for position and momentum alone: train.value_fit's correlation, the critic's V7 on
	# our rows off the share prior; a fit reading, so critic_fit/, not the losses' critic/)
	"campaign/value_corr": "critic_fit/corr_own",
	# policy/
	"campaign/policy_loss": "policy/policy_loss", "campaign/last_kl": "policy/kl", "campaign/entropy": "policy/entropy",
	"campaign/clip_fraction": "policy/clip_fraction", "campaign/link_share_mean": "policy/link_share", "campaign/grad_norm": "policy/grad_norm", "campaign/learning_rate": "policy/learning_rate",
	# play/
	"campaign/decisions_per_player_day": "play/decisions_per_player_day", "campaign/days": "play/days", "campaign/heroes": "play/heroes",
	"campaign/phi_end": "play/phi_end", "campaign/eliminated": "play/eliminated", "campaign_failures/idle_movement": "play/idle_movement",
	"campaign_failures/unmoved_heroes": "play/unmoved_heroes", "campaign_failures/own_hero_moves": "play/own_hero_moves",
	"campaign_failures/repeat_meetings": "play/repeat_meetings", "campaign_failures/meeting_movement": "play/meeting_movement",
	"campaign/army_on_heroes": "play/army_on_heroes", "campaign/attacked_hidden": "play/attacked", "campaign/recruits_hidden": "play/recruits",
	"campaign/dwellings_d28_hidden": "play/dwellings_d28",
	# speed/
	"campaign/update_seconds": "speed/update_seconds", "campaign/games": "speed/games", "campaign/decisions": "speed/decisions",
	"campaign/wait_seconds": "speed/wait_seconds", "campaign/game_seconds": "speed/game_seconds", "campaign/prepass_seconds": "speed/prepass_seconds",
}
PREFIXES = {"campaign_actions/": "actions/", "critic_heads/": "critic_heads/", "critic_steps/": "critic_steps/", "policy_steps/": "policy_steps/", "nk2_stream/": "nk2/", "credit/": "credit/"}  # (the action-type shares, their names kept; the critic's per-head fits)
# A named eval set's charts (evaluate.write_set_charts' campaign_eval_<set>/<key>, served as eval_<set>/<name>)
SET_CHARTS = {"score_own": "score", "score_own_se": "score_se", "score_own_d28": "score_d28", "eliminated_own": "eliminated", "won_own": "won",
	"max_level_ours": "max_level_ours", "max_level_theirs": "max_level_theirs", "major_won": "major_won"}
DROPPED = {"campaign_actions/dismiss"}  # (constant 0 in joint9: the user via e5, 2026-11-18)


def tracked(tag):
	"""The tag's name in the tracked set, or None."""
	if tag in DROPPED:
		return None
	if tag.startswith("campaign_eval_"):  # (a named eval set's charts, evaluate.write_set_charts: eval_<set>/<chart>)
		group, _, key = tag.partition("/")
		return f"eval_{group[len('campaign_eval_'):]}/{SET_CHARTS[key]}" if key in SET_CHARTS else None
	if tag in WHITELIST:
		return WHITELIST[tag]
	for old, new in PREFIXES.items():
		if tag.startswith(old):
			return new + tag[len(old) :]
	return None


class FilteredWriter:
	"""A SummaryWriter passing only the tracked tags (renamed); every other add_* dropped."""

	def __init__(self, writer):
		self.writer = writer

	def add_scalar(self, tag, value, step=None, *args, **kwargs):
		name = tracked(tag)
		if name is not None:
			self.writer.add_scalar(name, value, step, *args, **kwargs)

	def flush(self):
		self.writer.flush()

	def close(self):
		self.writer.close()

	def __getattr__(self, name):  # (add_histogram, add_text, ...: dropped)
		return lambda *args, **kwargs: None


class NullWriter:
	"""No TensorBoard: every call does nothing."""

	def __getattr__(self, name):
		return lambda *args, **kwargs: None
