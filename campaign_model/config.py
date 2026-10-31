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
	game_seconds: int = 1200  # headless.sh's limit for one game (what it recorded until then is used); since the games have taverns and dwellings (2026-10-29) a 56-day game takes ~200 s (~2,200 decisions) and a 113-day one ~560 s
	niceness: int = 10  # as the generated battles'
	temperature: float = 1.0
	# The campaign family's inference server
	inference_processes: int = 1
	inference_threads: int = 2
	# The campaign server's device: "auto" = CUDA when there is one (one process batches every game; on the CPU a decision
	# costs 5-10 ms of a core, the machine's games' biggest share), else the CPU (Apple's GPU is slower for these small
	# batches and cannot be shared between threads); or e.g. "cpu", "cuda"
	inference_device: str = "auto"
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
	prepass_batch: int = 0  # the update's first pass (no gradients: the KL anchor, the old values) in batches of this many; 0: a micro-batch's
	cuda_graphs: bool = True  # on CUDA: the update's forward and backward replayed from recorded graphs (graphs.py: launches cost under WSL)
	step_graphs: bool = True  # with cuda_graphs: a micro-batch's forward, loss and backward as one recorded graph (graphs.StepGraph)
	# The update's inputs decoded ahead in this many threads (train._prefetched: zlib, ~430 decisions/s a thread on the PC). With
	# the step recorded whole the PC's update waited ~15 s for 3 of them, the CPU half idle (2026-10-31): each decision is
	# decoded three times an update (the first pass, two passes)
	decode_threads: int = 3
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
	# "turns" (the default since 2026-10-30): the campaign updates run between battle updates, on the battle learner's
	# device and thread, whichever has its data first (battle_model/train.py: one GPU process for both models; on a
	# Windows GPU every process is a time slice taken from the trainer). "process": in a process of their own
	# (learner.UpdateProcess) on `device`; "thread": in a thread of the battle trainer's process, on `device` ("cpu":
	# Apple's GPU fails when two threads use it)
	learner: str = "turns"
	device: str = "auto"  # "auto": CUDA or Apple's GPU when there is one (process), the battle learner's (turns)
	torch_threads: int = 8  # the campaign learner's CPU threads (torch's pool is shared by the process: 0 leaves it)
	# "v1": model.CampaignModel; "v2": model_v2.CampaignModelV2 (docs/campaign_model_v2_draft.md: typed encoders, owners,
	# the whole map's tiles, foresight heads, every logit a match; cell_dim 16 for it)
	model_architecture: str = "v1"
	foresight_weight: float = 0.2  # v2: each foresight head's loss weight (foresight.py; its gradient share is logged)
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
	# 2, 3 and 4 players in turn (the setups of before, so paired comparisons go on).
	# 24, not 48: since the games have taverns and dwellings (vcmi 47c9d57eb) and hires pay, a game has ~4.5x the
	# decisions (2-4 minutes, 24 at once), and 48 of them held the machine for most of 20 minutes an hour
	eval_games: int = 24
	eval_parallel: int = 24  # games at once (a VCMI client each, next to the training's)
	eval_days: int = 56  # (the score at day 28 is logged too: score_d28)
	# The campaign model samples, as in training (seeded per game and seat: AlphaMapAI's ALPHA_MAP_SEED_<COLOR>). Greedy
	# (2026-10-29) was a policy never trained or played: its argmax over the move table is often a neighbouring tile and
	# from there the tile it came from, so heroes walked back and forth a tile at a time until their movement ran out
	# (95% one-tile moves, 88% back onto a tile of that day; sampled: 19% and 2%) and the eval measured that
	eval_temperature: float = 1.0
	eval_seconds: int = 600  # 240 cut half of the first greedy eval's games short (2026-10-29: 120-240 s a game, 24 at once); an endless siege once held a whole eval for 900 s (no longer: vcmi's stalemate rule)
