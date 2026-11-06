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
	game_seconds: int = 2400  # headless.sh's limit for one game; 2400, not 1200, since 2026-11-01: joint7's games of long day caps reached days 64-105 of ~112 by 1200 s and were killed, 74 of ~1,170 (their records not trained on). Since the games have taverns and dwellings (2026-10-29) a 56-day game takes ~200 s (~2,200 decisions) and a 113-day one ~560 s
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
	learning_rate: float = 1e-5  # at most 3e-5, 1e-5 recommended (the user's rule, 2026-10-31; was 1e-4)
	# The critic on its own timescale (docs/optimizer_plan.md step 1, 2026-11-06; the user's OK for the value group above
	# the 3e-5 cap, which was the policy's): the value heads' parameters (train.VALUE_MODULES) at learning_rate x this.
	# Two-timescale actor-critic wants the critic the faster: at the policy's rate a label change took ~200 updates to
	# re-fit and the readouts' rms was 0.006-0.012 after 1,300 updates
	value_lr_factor: float = 10.0
	# The KL control's lr_scale scales the value group too (the plan's wording). False: the critic's rate is its own. Its
	# parameters never move the policy, so the policy's KL is no reason to slow them, and a KL target cut 4x (step 2)
	# would have cut the critic's rate with the policy's
	value_lr_kl: bool = False
	# The trunk's momentum (step 5): Adam's beta1 for the groups that are not the value heads' or the corrections'
	# (trunk, embeddings, type). For pure noise Adam steps |m| / sqrt(v) ~ sqrt((1 - beta1) / (1 + beta1)): 0.23 at 0.9,
	# 0.16 at 0.95, at full size whatever the signal; the campaign's median ratio was 0.17-0.26
	trunk_beta1: float = 0.95
	target_kl: float = 0.015
	# The KL target inflated by the teacher's share against the policy's (1 + kl_teacher: 0.015 x 4 = 0.06 at the 60%
	# teacher share). False since 2026-11-06 (step 2): the target is target_kl, the actor on the slow timescale
	kl_teacher_inflation: bool = False
	lr_scale_max: float = 1.0
	passes: int = 2
	batch_size: int = 256
	micro_batches: int = 4  # each minibatch's forward and backward in this many parts, their gradients summed: the same step at a fraction of the peak memory
	# The memory's training (train.sequence_order, model_v2.forward_steps): each pass's rows as sequences of this many of
	# a player's consecutive decisions, each after the first reading the memory the one before left, so the losses'
	# gradient reaches the writer. 1: every decision alone on its recorded memory, which no loss ever reached (the writer
	# frozen as the imitation left it, 2026-11-04). A micro-batch's rows must be a multiple (batch_size / micro_batches)
	memory_steps: int = 1
	prepass_batch: int = 0  # the update's first pass (no gradients: the KL anchor, the old values) in batches of this many; 0: a micro-batch's
	cuda_graphs: bool = True  # on CUDA: the update's forward and backward replayed from recorded graphs (graphs.py: launches cost under WSL)
	step_graphs: bool = True  # with cuda_graphs: a micro-batch's forward, loss and backward as one recorded graph (graphs.StepGraph)
	convert_in_graphs: bool = True  # with cuda_graphs, on CUDA: the wire's conversion (records.inputs_on_device, ~250 small kernels a micro-batch, ~20 ms of the main thread's ~58 under WSL, 2026-11-03) recorded in the update's graphs, which take the bytes as sent; each such graph checked at its first replay, the conversion done outside again if one fails
	window_budget: float = 4.0  # with cuda_graphs: hero windows computed a decision (model_v2.windows_of), of OWN_HEROES slots: the same outputs for a batch with no more of our heroes (it computes every slot otherwise); the PC's games, 2026-11-03: 2.8 a decision, never more than 4. 0: every slot
	# The update's inputs decoded ahead in this many threads (train._prefetched: zlib, ~430 decisions/s a thread on the PC). With
	# the step recorded whole the PC's update waited ~15 s for 3 of them, the CPU half idle (2026-10-31): each decision is
	# decoded three times an update (the first pass, two passes)
	decode_threads: int = 3
	# Every this many updates, each loss term's gradient noise scale by game phase (train.loss_noise; ~40 s on the PC); 0: never
	noise_every: int = 10
	clip: float = 0.2
	value_weight: float = 0.5
	growth_weight: float = 0.5  # each growth head's weight in the value loss, a share head's 1 (labels.py: helpers)
	# Entropy: a floor, not a bonus (battle_model/losses.py, 2026-10-28): decisions whose entropy is under entropy_floor
	# are pushed back up; the bonus (0.01 until then) is off
	entropy_weight: float = 0.0
	entropy_floor: float = 0.3
	entropy_floor_weight: float = 0.1
	grad_clip: float = 0.5
	# Clipping by parameter group (step 6; train.GRAD_GROUPS, train.clip_gradients): each group's gradient clipped at its own
	# median norm, so a spike in one (the corrections' x10 step, the value heads' ~350 weight) no longer shrinks the
	# others' on that step, then every group times grad_clip / the median total norm: the scale Adam's moments were
	# built at (the one clip at grad_clip always bound, the norm ~10; clipped at their own medians the trunk's gradients
	# would be ~25x larger at once, and with a second moment ~1,000 steps slow its steps up to ~6x their size for hours).
	# The medians are the first grad_clip_calibration optimizer steps' (the one clip meanwhile), kept in the checkpoint;
	# grad_clip_groups ("trunk:0.4,value:9,...") sets them by hand instead. False: the one clip at grad_clip
	grad_clip_by_group: bool = True
	grad_clip_calibration: int = 200
	grad_clip_groups: tuple = ()
	grad_clip_since: int = 0  # clips calibrated before this version are measured again (a deploy's recalibration; 0: kept)
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
	# "attributed" (2026-10-31, the user and e5's idea; train.ppo_update): the parts of our log share's change over 7 days
	# that our decisions make, d log share = (1 - share) d log V_ours + taken / V_all - the others' own growth: our value's
	# log growth less the growth head G7's forecast, times (1 - our share), plus what we took from the other players
	# (labels.attributed_labels: battles against their heroes, their towns and income that became ours); the others' own
	# growth, and the luck we cannot see in it, left out. Measured first (train.loss_noise, v331): the gradient noise scale
	# of our growth's credit 1,310 decisions early against the share's 2,979 and the horizon credit's 5,312; mid and late
	# games no signal at all for the horizon credit. The end credit stays in at attributed_end_weight (denial, which no
	# event records: racing them to a mine), both standardized first
	# "schedule" (the user, 2026-10-31; needs share_target "change"): w x the share's surprise (its logit's real change
	# over 7 days less V7's forecast) + (1 - w) x the growth's (our value's real log growth less G7's), w = min(1,
	# share_weight_start + share_weight_per_day x day): early on growth counts most (the share moves with rivals we cannot
	# see yet), late the share (destroying them); the end credit stays in at attributed_end_weight
	credit: str = "schedule"  # (the default since 2026-11-01: what runs; it needs share_target "change")
	share_weight_start: float = 0.1
	share_weight_per_day: float = 0.02
	share_weight_side: float = 36.0  # the map side share_weight_per_day is for: a map of side S ramps at that x share_weight_side / S (the user, 2026-11-03: faster, and slower on bigger maps, where rivals are met later); 0: the same on every map
	attributed_end_weight: float = 0.2
	outcome_share_min: float = 0.25
	myopic_weight: float = 0.5
	trust_full_ev: float = 0.4
	# Trust as a floored slow average (step 4; learner.trust): max(trust_floor, its exponential average at trust_memory a
	# update, ~50 updates), not today's value: a re-fit (a label or a Φ change) cut it 0.40 -> 0.25 and the credit went
	# three quarters Monte Carlo, slowing the re-fit's own signal. 0 and 0: today's value, as before
	trust_floor: float = 0.4
	trust_memory: float = 0.98
	# The end head's credit (labels.end_advantages): the policy's advantage is (1 - b) x the 7-day one + b x the end
	# one, b = end_credit_max x the trust in V_end (its explained variance at the day starts over the share prior's,
	# 1 / players, smoothed, over trust_full_ev): while V_end predicts the end no better than the player count does,
	# the 7-day credit alone; as it learns, the credit moves to the end (the user: "maybe moving to use it late during
	# training"). 0: V_end is trained and logged only
	end_credit_max: float = 1.0
	# The end credit's own check (b1, 2026-10-31): V_end's step estimate of the end is checked by action type as the 7-day
	# one is (labels.credit_trust over learner.end_evidence); as it fails, the end credit's step part gives way to its week
	# part, as the 7-day credit's step gives way to the week's outcome. Its steps were off by type as the 7-day ones (travel
	# t +8, hire -7, end -5, move -5, build +3..8; ~0.003 of a share), and nothing held them back
	# Off by default: on since 10:29 together with an info-spell mask (10:05-10:44), and the eval against Nullkiller2 fell from
	# 0.22 to 0.07, then 0.12 with the mask undone; to be tried alone against the 48-game eval (2026-10-31)
	end_credit_check: bool = False
	# "turns" (the default since 2026-10-30): the campaign updates run between battle updates, on the battle learner's
	# device and thread, whichever has its data first (battle_model/train.py: one GPU process for both models; on a
	# Windows GPU every process is a time slice taken from the trainer). "process": in a process of their own
	# (learner.UpdateProcess) on `device`; "thread": in a thread of the battle trainer's process, on `device` ("cpu":
	# Apple's GPU fails when two threads use it)
	learner: str = "turns"
	device: str = "auto"  # "auto": CUDA or Apple's GPU when there is one (process), the battle learner's (turns)
	torch_threads: int = 8  # the campaign learner's CPU threads (torch's pool is shared by the process: 0 leaves it)
	# "v1": model.CampaignModel; "v2": model_v2.CampaignModelV2 (docs/campaign_model_v2_draft.md: typed encoders, owners,
	# the whole map's tiles, foresight heads, every logit a match; cell_dim 16 for it). v2 the default since 2026-11-01,
	# at its sizes (dim 128, cell_dim 16): what runs
	model_architecture: str = "v2"
	foresight_weight: float = 0.2  # v2: each foresight head's loss weight (foresight.py; its gradient share is logged)
	model_dim: int = 128
	model_cell_dim: int = 16
	model_heads: int = 4
	model_layers: int = 2
	seed: int = 0
	init_from: str = ""  # a campaign checkpoint a new run starts from (its weights; a new optimizer)
	# What the share heads V2, V5, V7 forecast (the model's "value" output; V_end is our share at the end either way):
	#   "share"   our share of all empires' value VALUE_OFFSETS days on (from 1 / players)
	#   "change"  how much the logit of that share changes by then (labels.share_change_labels; the user's design,
	#             2026-10-31: our log growth less the others', so their value hidden from us cancels), from no change.
	#             The ladder grades them on real outcomes only, and the horizon and mix credits refuse it (they read
	#             the heads as shares); the attributed credit reads the growth heads
	share_target: str = "change"  # (the default since 2026-11-01: what runs)
	# Each value head's loss weighted inversely to its targets' running variance (train.head_weights: the heads' total
	# kept, ramped in over HEAD_SCALE_RAMP updates): V_end's loss at a constant prediction was 35-100x a share-change
	# head's (2026-11-02)
	head_scale: bool = False
	# What head_scale weighs each head by (step 7, 2026-11-06): "residual", 1 / the running mean of its fresh excess loss
	# (train.head_weights' residuals; the heads that learn get more of the shared layers, the ones whose residual is large
	# less), ramped in over HEAD_SCALE_RAMP updates from the variance's weights; "variance", 1 / its targets' running
	# variance (swung with the batch's label spread: the share weights 6 to 9.8 update to update)
	head_scale_by: str = "residual"
	# The teacher's KL (AlphaStar's KL to its supervised policy, 2026-10-31: 1,020 -> 1,400 Elo in its ablation): each
	# decision's KL(teacher || model) over the action type and each type's table (train.teacher_kl), at this weight in
	# the loss, the teacher a frozen imitation of Nullkiller2 (imitate.py's bc_latest.pt: teacher_checkpoint). It keeps
	# what the teacher does (attacks among it: self-play never attacks, 2026-10-27) unless the games' credit pays to
	# leave it. 0: off
	teacher_checkpoint: str = ""
	teacher_kl: float = 0.0
	# Nullkiller2's advice (agents/AlphaMapAI/Advice.cpp; the user, 2026-10-31: "a real teacher"): the chance a training
	# game's turn decision is also asked of a Nullkiller2 in the same state; the update adds the cross-entropy of the
	# model's probabilities toward its choice on those decisions, at the teacher's weight (teacher_mix "score": dominant
	# while it is much better, small once we are; teacher_kl before the first evaluation). 0: off (the default). Each
	# advice is ~15-30 ms of a game's CPU
	nk2_advice: float = 0.0
	# Its TRADE advice taught too (False since 2026-11-04: the user's rule, every trade is dumb unless it buys what could not
	# be bought without it; ~94% of its trades bought nothing that day, and 54% of ours came where it advised one)
	nk2_trade_advice: bool = False
	# The KL a decision is kept near (0: teacher_kl stays the weight): after each update the weight is x1.5 when the
	# update's teacher KL was over 1.5 x this, /1.5 when under 1/1.5 of it (the PPO KL control's rule), within 0.01..10.
	# PPO moves the policy ~target_kl an update, in a random direction where the credit is noise: unanchored, the clone
	# would walk nats away from the teacher in a day (~430 updates); a fixed weight is a guess at once too weak there and
	# too strong where the credit is real
	teacher_kl_target: float = 0.0
	# "score" (the user, 2026-10-31: while the teacher is much better than us its loss dominates; when we are better it is
	# small, not 0): after each evaluation against Nullkiller2 the teacher's share of the loss is lambda = clip(1.2 - score,
	# 0.05, 0.95), its weight lambda / (1 - lambda) (PPO's 1); teacher_kl is the weight until the first evaluation
	teacher_mix: str = ""
	# The loss shares (learner.LOSS_TERMS; the user, 2026-11-02: each loss a sane share of the gradient, the teacher's by how
	# good we are, policy and value never under 10% each): each term's weight set from its running gradient norm on the
	# shared layers (GradNorm's balance in closed form) so the terms get their shares (learner.loss_shares); off: the
	# teacher at its weight (teacher_mix), the others at 1
	loss_shares: bool = False
	# The loss shares' weights frozen from this version on (step 3; 0: never): measured and logged still, no longer moved.
	# The balancer gave the policy's mostly-noise gradient (noise scale 300-6,000 decisions, at 4,096 a step) its share by
	# norm, and moved with the trust and the KL control on one ~10-update time constant through 19 restarts in 19.5 hours
	loss_shares_freeze: int = 0
	label_weights: tuple = (1 / 3, 1 / 3, 1 / 3)  # of the shares at VALUE_OFFSETS (labels.py)
	decision_price: float = 1e-4  # share points per turn decision
	# Evaluation against Nullkiller2 (evaluate.py): our models in one seat, Nullkiller2 in the others
	eval_minutes: float = 60.0  # 0: none
	# The evaluation's games at a lower priority than the training's (niceness 10): they take the CPU the training leaves
	# (the user, 2026-10-31: no slowdown from evaluations; at 0 they took it first, ~50 minutes an evaluation)
	eval_niceness: int = 15
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
