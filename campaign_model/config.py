"""The campaign side of a joint run (learner.py; battle_model/train.py runs it when campaign.games > 0): its games, its
updates, its evaluation. Set from the command line as campaign.<name>=<value>, as ppo.<name> is."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CampaignConfig:
	# Self-play games: a new random map each (play.make_map), every player our campaign model, every player's battle
	# our battle model (the wandering monsters are BattleAI, as in a real game: play.play's neutral_ai), whose records go
	# to the battle learner as real games' do
	games: int = 0  # played at once; 0: no campaign training (a battle-only run)
	# Each game's last day, drawn per game (learner.draw_days): days_min plus a geometric number of days with mean
	# days_tail, at most days_max. Past days_min every day is as likely to be the last as the one before (the user,
	# 2026-10-28: the 7-day horizon can't see what pays later, so the end head V_end learns the share at the game's
	# end): the model can't see the end coming, so nothing is worth grabbing just before it, and V_end, from any day,
	# looks ~days_tail days past days_min. The AI never sees the cap (it only stops there); evaluations stay at eval_days
	days_min: int = 28
	days_tail: float = 28.0
	days_max: int = 112
	players: tuple = (2, 3, 4)  # drawn uniformly (the user: more than 2 players)
	sizes: tuple = (36,)  # map width and height, drawn uniformly (the user: small maps only, for now)
	game_map_share: float = 1 / 3  # games on one of the game's own maps (maps.GAME_MAPS), the rest on generated ones
	levels: int = 2  # with the underground (the user: water and underground stay on)
	water: str = "random"
	game_seconds: int = 600  # headless.sh's limit for one game (what it recorded until then is used); a 56-day game takes ~20 s, the slowest of 1,935 took 150 s (games now run to days_max = 112), and an endless battle (no round limit in VCMI) never ends
	niceness: int = 10  # as the generated battles'
	temperature: float = 1.0
	# The campaign family's inference server
	inference_processes: int = 1
	inference_threads: int = 2
	# Updates (train.ppo_update), with their own optimizer
	decisions_per_update: int = 8192
	min_games_per_update: int = 4  # one game's decisions are alike: an update takes several games
	max_record_age: int = 4  # model versions: older decisions are dropped
	learning_rate: float = 1e-4  # never above (the user's rule)
	target_kl: float = 0.015
	lr_scale_max: float = 1.0
	passes: int = 2
	batch_size: int = 256
	micro_batches: int = 4  # each minibatch's forward and backward in this many parts, their gradients summed: the same step at a fraction of the peak memory
	clip: float = 0.2
	value_weight: float = 0.5
	growth_weight: float = 0.5  # each growth head's weight in the value loss, a share head's 1 (labels.py: helpers)
	# Entropy: a floor, not a bonus (battle_model/losses.py, 2026-10-28): decisions whose entropy is under entropy_floor
	# are pushed back up; the bonus (0.01 until then) is off
	entropy_weight: float = 0.0
	entropy_floor: float = 0.3
	entropy_floor_weight: float = 0.1
	grad_clip: float = 0.5
	# AdamW weight decay of the prior corrections (train.make_optimizer; model.CORRECTION_STEP 10): under a steady push a
	# correction settles at ~CORRECTION_STEP / prior_decay = 2 (a log-gain of 2 is the x5 bound; +2 logits), under noise
	# near 0; left alone it halves in ~1,400 steps (lr 1e-4: 5e-4 a step), ~45 minutes of joint1's updates
	prior_decay: float = 5.0
	# The policy's advantage (labels.horizon_advantages): aimed at the share 7 days on, the longest horizon; the step's
	# estimate (V7 at the next turn decision that day, or V5 two days on) mixed with the real outcome by how much the
	# value heads are trusted (their explained variance over the share prior's, 1 / players, smoothed: 0 at EV <= 0, 1 at
	# EV >= trust_full_ev; plain EV credited the player count, which the heads start from): the
	# outcome's share falls from all of it to outcome_share_min, and the 2-day shortcut (myopic_weight) fades to 0.
	# outcome_share_min and trust_full_ev set the value heads' grades too (labels.ladder_targets, learner.grade_trust).
	# "mix": the old advantage, the flat mix of the shares 2, 5 and 7 days on minus the mix of the heads
	credit: str = "horizon"
	outcome_share_min: float = 0.25
	myopic_weight: float = 0.5
	trust_full_ev: float = 0.4
	# The end head's credit (labels.end_advantages): the policy's advantage is (1 - b) x the 7-day one + b x the end
	# one, b = end_credit_max x the trust in V_end (its explained variance at the day starts over the share prior's,
	# 1 / players, smoothed, over trust_full_ev): while V_end predicts the end no better than the player count does,
	# the 7-day credit alone; as it learns, the credit moves to the end (the user: "maybe moving to use it late during
	# training"). 0: V_end is trained and logged only
	end_credit_max: float = 1.0
	# "process": the campaign updates run in a process of their own (learner.UpdateProcess) on `device` (Apple's GPU
	# or CUDA work alongside the battle learner's: two processes, not two threads); "thread": in a thread of the battle
	# trainer's process, on `device` ("cpu": Apple's GPU fails when two threads use it); "turns": between two battle
	# updates, on the battle learner's device and thread (each campaign update, ~40 s, then held the battle learner
	# up: its updates came every ~60 s instead of ~20)
	learner: str = "process"
	device: str = "auto"  # "auto": CUDA or Apple's GPU when there is one (process), the battle learner's (turns)
	torch_threads: int = 8  # the campaign learner's CPU threads (torch's pool is shared by the process: 0 leaves it)
	model_dim: int = 64
	model_cell_dim: int = 32
	model_heads: int = 4
	model_layers: int = 2
	seed: int = 0
	init_from: str = ""  # a campaign checkpoint a new run starts from (its weights; a new optimizer)
	label_weights: tuple = (1 / 3, 1 / 3, 1 / 3)  # of the shares at VALUE_OFFSETS (labels.py)
	decision_price: float = 1e-4  # share points per turn decision
	# Evaluation against Nullkiller2 (evaluate.py): our models in one seat, Nullkiller2 in the others
	eval_minutes: float = 60.0  # 0: none
	# 2, 3 and 4 players in turn (the first 24 are the setups of before, so paired comparisons go on); at 24 sampled games the
	# score's standard error was ~0.07 and the evals' spread below it (2026-10-29): the user asked for a readable eval
	eval_games: int = 48
	eval_parallel: int = 24  # games at once (a VCMI client each, next to the training's)
	eval_days: int = 56  # (the score at day 28 is logged too: score_d28)
	eval_temperature: float = 0.0  # greedy, as the battle evals (2026-10-29: at 1.0 an eval's own sampling hid any change under ~0.1); evals compare only with those played the same way
	eval_seconds: int = 240  # an eval game takes 10-30 s; an endless siege (BattleAI never retreats, VCMI has no round limit) held a whole eval, and with it the training, for 900 s
