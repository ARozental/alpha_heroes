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
	# Bigger maps (docs/map_size_plan.md: the model reads one a block of tiles a cell): every big_game_every-th game a
	# supply draws on a generated map big_game_size a side (0: never; "give it one large map to see what happens"). A
	# game's time limit (game_seconds, eval_seconds, a teacher's) grows with its map's area (play.time_scale); the day
	# caps stay as they are
	big_game_every: int = 0
	big_game_size: int = 144
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
	window_budget: float = 4.0  # with cuda_graphs: hero windows computed a decision (model_v2.windows_of), of OWN_HEROES slots: the same outputs for a batch with no more windows (it computes every slot otherwise); the PC's games, 2026-11-03: 2.8 heroes a decision, never more than 4; further windows (below) add theirs, ~0.4 a hero in the games looked at: 5.0 with them. 0: every slot
	# Schema 17 (2026-11-10), the games' side of it (play.window_settings: AlphaMapAI's environment, every game's alike).
	# further_windows: the windows a hero may have beside its own (ALPHA_MAP_FURTHER_WINDOWS; 0: none, the windows as
	# before): a square of one level around tiles it reaches today that its own window does not show, in a window slot
	# no hero holds, its tiles MOVE targets. Before it nothing past a gate or a monolith was a MOVE's target (the window
	# is the hero's own level's) nor a TRAVEL's (its areas are a day or more away): the way through was a MOVE onto the
	# gate, a tile worth nothing. One, not more: in a Nullkiller2 game of 40 days (36 x 36, two levels) 43% of the
	# hero-turns had such tiles, a median of 31 against 77 inside the hero's window; a second window held a median of 8
	# more, a third 3, each as much work as a hero's own (the forward on the Mac's CPU: 27.2 ms a decision at 3 windows,
	# +1.6 a window; a game client's CPU the same with them and without, 4.4-4.7 against 4.5-4.6 ms a decision).
	# A label change too: Nullkiller2's advised tiles past a gate become MOVE labels (18 of 865 MOVE labels, 32 of 1,192
	# accepted entries in one game of ours with its advice): measure the type drift after the first update
	further_windows: int = 1
	# unseen_exits: TILE_UNSEEN_EXIT and OBJECT_UNSEEN_EXIT set (ALPHA_MAP_UNSEEN_EXITS; False: never, the prior as
	# before): a gate's or monolith's entrance no exit of which we have seen. VCMI's pathfinder goes through one only to
	# an exit the player sees, so nothing past it is a target and the entrance itself, with nothing on it and the land
	# around it long revealed, was worth nothing; it is priced as a whole sight of new land (model.tile_features'
	# explore; its TRAVEL area as unrevealed, features.unseen_exits), as Nullkiller2's ExplorationBehavior sends a hero
	# into each. A prior change of its own: turn it on apart from further_windows
	unseen_exits: bool = True
	# Schema 19 (2026-11-11, the user's go-ahead: Nullkiller2's insight as inputs, its labels no longer a loss), the games'
	# side (play.nk2_settings): nk2_inputs, AlphaMapAI's advisor plans before every turn decision and its candidates,
	# tier and threats are the model's inputs (ALPHA_MAP_NK2_INPUTS), in training games, evaluations and paired evaluations
	# alike; nk2_input_dropout, the share of training games played without them (g_nk_present 0 throughout, as where it
	# would be too slow or in play without Nullkiller2), drawn from the game's seed: the policy learns to play from the
	# plan head's prediction as well. Both off by default (main's games as before): the deploy turns them on
	nk2_inputs: bool = False
	nk2_input_dropout: float = 0.0
	# The share of training games that show Nullkiller2's OWN CHOICE among the inputs (the user, 2026-11-11: "this can
	# work, but it is also super risky in the sense that we will only learn to fix NK2 and not really learn good moves in
	# general"). In the rest (ALPHA_MAP_NK2_CHOICE=0: g_nk_action, g_nk_index, g_nk_cell left 0) the follow mixture and
	# the advantage head's named term are off and our own policy plays the whole game, at full gradient on its own
	# states, its candidates, scores, plan and threats still among the inputs. One setting a game, every player in it
	# alike (never a hidden-choice player against a follower: that would be training against Nullkiller2), drawn from
	# the game's seed apart from nk2_input_dropout's draw (play.nk2_settings); evaluations and paired tests show it.
	# 1.0: every game shows it, as before
	nk2_choice_games: float = 1.0
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
	# The schedule credit's price of an action (the user, 2026-11-09: "reduce num_actions x tiny_const from the day+N
	# rewards"; 0: none, as it runs): taken from a decision's 7-day credit for each action the player makes from that
	# decision on through the week (labels.actions_ahead: turn and exchange decisions other than END; the policy's
	# credit only, the heads' labels stay the real outcomes). The user's size: 2e-6, at most 2e-5 ("as high as /10
	# instead of /100 but no higher" of the step credit's 2e-4). On 8 joint7 games a decision has 107 actions ahead in
	# its week (std 38; 8.7 of them that day): at 2e-6 the charge is 0.0002, 0.30% of the 7-day outcome's spread
	# (0.0725); at 2e-5 3.0%. What one decision moves of it is its own action (0.003% of the spread at 2e-6) or, ending
	# the day, the day's rest (0.024%; ten times those at 2e-5). A nudge by design: nothing to see within a night
	schedule_action_price: float = 0.0
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
	# The privileged critic (2026-11-06, the user's call on AlphaStar's lesson; model_v2.PrivilegedCritic, the user's
	# design of 2026-11-07): the share heads' and V_end's logits corrected from the other players' own critic states
	# (max-pooled) and our true share now, training only, learned on top of the critic (its inputs detached: the
	# critic, the trunk and the policy train as without it). privileged_critic: it learns and its fit is logged
	# (eval/campaign_value_unexplained_privileged); privileged_credit: the schedule credit's baselines V7 and V_end are
	# its, a lower-variance advantage (still unbiased: no baseline depends on the action), switched on once its fit is
	# clearly better
	privileged_critic: bool = True
	privileged_credit: bool = False
	# SET ASIDE (the user, 2026-11-09): step_credit, step_price and step_class_correction change nothing, train.ppo_update's
	# call of train.step_credit is commented out ("I do not want any value on actual steps, only on day+2/5/7"; to be
	# measured apart later). The policy's schedule credit stays the 7-day outcome alone. Kept for that measurement:
	# The schedule credit's step (the 2026-11-09 audit; train.step_credit): each term its forecast's step to the player's
	# next turn decision that day (growth: log Φ + G7's; share: V7's change as the baseline scales it) + s x the rest of
	# the week, s = 1 - (1 - outcome_share_min) x the forecast's trust (G7's grade_trust; the share's own fit this update,
	# train.forecast_trust); the step alone where the 7-day outcome is unknown. Replayed on 8 joint7 games (v2482,
	# scratchpad audit/c5.py): the advantage's std 0.081 -> 0.042, its variance shared a player-day 98.1% -> 90.6%,
	# credited rows 82% -> 99%; each action type's mean credit in the advantage's own std stays where the outcome alone
	# had it (RECRUIT -0.22 -> -0.19, UPGRADE +0.14 -> +0.08, BUILD -0.01 -> +0.07, an exchange's MULTI +0.01 -> -0.00):
	# it does not teach recruits or upgrades away. (The recruits' -733 of Φ is not Φ's price of a recruit: 149 of 591
	# were paid by the market's trades, AlphaMapAI::payByTrades, 5,500 of resources for 655 of troops; one paid in gold
	# loses the unled share's 4.7%. An upgrade loses 36 of Φ; its step's -0.0070 is G7's forecast.)
	step_credit: bool = False
	# What a decision costs in the step credit (log Φ; END and a question nothing): a useless action's step is ~0, which
	# is right and no reason to stop; this is. 2e-4 is 30 gold of a 150k empire, a tenth of a small pickup's step; the
	# mean MOVE's own step is 6.5e-4, a BUILD's 7e-3. On the decision that made it, not in the day's labels (step_credit)
	step_price: float = 2e-4
	# The step credit's class correction (train.step_credit's classes): each action type's mean rest of the week added
	# back, so its mean credit is the outcome's whatever G7's habits right after such a decision. Off: in the replay it
	# doubles every type's mean credit in the advantage's std (the noise halves, the means stay: RECRUIT -0.22 -> -0.44),
	# and a type's mean rest is its days' as much as its own (recruits are made on the week's first day: the outcome less
	# the forecast BEFORE the day's actions is -0.009 on days with a recruit, +0.006 without). For when campaign_step/*
	# shows a type taught by a habit of the forecast's
	step_class_correction: bool = False
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
	# The teacher's loss (the user, 2026-11-11: "get rid of the teacher"). False: Nullkiller2's advice, where it is asked
	# for, reaches no weight (its cross-entropy at weight 0 in every update, its share of the gradient 0); its agreement
	# and cross-entropy are still logged (teacher_agree, teacher_ce), and its plan stays the foresight's label. What it
	# knows comes in as inputs instead (schema 19's candidates: views.candidate_tables). Measured that day at v3045 on
	# the eval's 48 setups, three replicates side by side: playing its matched advice 0.813 against our own choice's
	# 0.520, after 3,000 updates of this loss at 60% of the shared layers' gradient had brought our choice to its
	# advice at 24% of the decisions; a 51-weight readout of our own known features recovers two thirds of what the
	# loss put into the policy, the plan head's prediction most of the rest
	nk2_teacher: bool = True
	# Nullkiller2's preference among its candidate tasks as a prior on MOVE's and TRAVEL's entries (schema 19; needs
	# nk2_inputs in the games): the logits its best candidate at the plan's tier adds to the entry it goes through, and
	# a gather-tier fallback's best (a hero the plan's tier says nothing about). 0: off, the model as before. Set into
	# the model's buffers at every start (learner.set_advice_prior), which the checkpoints and the servers carry; the
	# sizes are chosen by paired tests (campaign_model.paired_eval with ALPHA_NK2_PRIOR, ALPHA_NK2_PRIOR_FALLBACK)
	nk2_prior: float = 0.0
	nk2_prior_fallback: float = 0.0
	# Nullkiller2's own choice at the decision as a prior (schema 19's g_nk_action, g_nk_index; needs nk2_inputs): this
	# many logits on the entry it would play now, inside its table and on its type's score. 0: off. At a large size our
	# choice is its choice (b1's follow test: +0.29 +- 0.06 on 144 paired games); the size is chosen by paired tests
	# (ALPHA_NK2_LABEL), and the model learns where to leave it (the gain on it, 1/4..2, and everything it reads)
	nk2_label_prior: float = 0.0
	# Nullkiller2's own choice followed at this chance (0..1; needs nk2_inputs; 0: off): the games' policy is
	# rho x (its choice) + (1 - rho) x (our own policy), exactly (model_v2.CampaignModelV2.follow), rho learnable
	# around this start. Our policy is trained on what it chose itself; its choice is the model's input everywhere
	nk2_follow: float = 0.0
	# The advantage head (model_v2.AdvantageHead; stage 1, 2026-11-11: it only learns, the policy reads none of it): its
	# squared error against the update's advantage (as the policy term takes it) on the rows the policy credits, at
	# this weight. One update's 14 games give the policy no direction, while a regression averages over every game:
	# on 192 recorded games a day's share of followed choices predicted the 7-day share change (slope +0.0124 +- 0.0023).
	# 0: off, nothing computed (the update as before, bit for bit)
	adv_weight: float = 0.0
	# Predicting Nullkiller2's planner (nk_predict.py, model_v2.NkPredictHead; stage 1, 2026-11-11, the user: "this is an
	# auxiliary loss, we can have it if it helps"; the planner to leave serving later): the head's loss at this weight, from
	# a blind pass of the trunk (the planner's fields absent, the memory at its start) on 1 / NKP_ROWS of each forward's
	# rows, its own weights alone (value group); its fit on the prepass's rows each update (nkp_*). Nothing reads the head.
	# 0: no blind pass, nothing computed: the update as before, bit for bit
	nk_predict_weight: float = 0.0
	# nk_predict's step on every this-many'th micro-batch only (deploy 63's PC check: the blind pass on every micro-batch
	# made the update 1.41 times as long): its own recorded step (the blind pass on the micro-batch's first 1 / NKP_ROWS
	# rows, the head's loss and backward), the main step untouched; its fit read on the prepass's first NKP_PREPASS chunks
	nk_predict_every: int = 4
	# nk_predict stage 2 (model_v2.nk_read; e5's design, 2026-11-12): "live" (as before), "blind", "late" (diagnostics),
	# "predicted" (the trunk without the planner, the late terms reading the head's table: no planner at serving; the
	# head's loss from the main pass, the planner still on in training games for its targets). The learner sets it on its
	# model and in ALPHA_NK_READ for the servers it starts
	nk_read: str = "live"
	# Our own policy reads the advantage head (stage 2a; model_v2's decide): this many logits a unit of its prediction
	# (the update's advantage in sd units: soft policy iteration's 1 / beta), its table terms clipped to +-3 and centered
	# on their mean under our own policy, its type term clipped to +-3, all before the follow mixture; x a learned gain
	# (1/4..2). 0: off, the model as before, bit for bit. Set into the model at every start (learner.set_advice_prior);
	# ALPHA_ADV_GAIN for a paired test. Turned on only once adv_ev on fresh rows has held above 0 (stage 1's reading)
	adv_gain: float = 0.0
	# The update's advantage standardized within each kind of game (own games and the rest: config.nk2_choice_games), for
	# the policy term and the advantage head alike, rather than pooled. Pooled, the own games' advantage came out at a
	# fifth to a half of the others' spread (v3154-3157: 0.20-0.60 against 1.09-1.39), so their rows pushed our own policy
	# with that much less weight. False: pooled, as before (bit for bit); an update with one kind only is pooled either way
	adv_by_kind: bool = False
	# The town guard (bundle 60; model.CustomFeatures.town_guard_cost): this many logits a unit of value_logit(the expected
	# loss a hero adds to a threatened town of ours by leaving it with too little) on that hero's MOVE and TRAVEL entries,
	# x a learned gain (1/4..2); exactly 0 unless Nullkiller2 sees a threat reaching the town by tomorrow. Our own policy
	# only (the follow mixture's part is untouched); no masking. 0: off, the model as before, bit for bit. Set at every
	# start (learner.set_advice_prior); ALPHA_TOWN_GUARD for a paired test
	town_guard: float = 0.0
	# The credit's horizon by segments (bundle 60; e5's design from 6,424 own-policy player-days: the share-change surprise's
	# spread is 0.050 at 2 days, 0.074 at 5, 0.087 at 7, and what happens in 2 days persists, slope 0.95, so the 7-day
	# credit is the 2-day one plus 2.4 times its variance added in days 3-7). On: the schedule credit is surprise(0-2) +
	# k5 x surprise(2-5) + k7 x surprise(5-7), each segment that window's real change (share and growth alike) less the
	# heads' forecast for it (V2; V5 - V2; V7 - V5); k = each long segment's measured signal fraction, an empirical-Bayes
	# shrinkage (learner: credit_evidence over the recent updates, faded, of each segment's surprise after each action type
	# against the player's other decisions; F the between-type variance of those gaps over their sampling variance, over
	# the types with enough players; k toward max(0, 1 - 1/F) by 0.05 an update, from 0, at most 1). It sees only what
	# depends on the action TYPE, so it errs low (the conservative side). With k near 0 the credit is the 2-day one plus
	# the end credit: the user's "noisy but safe" floor. False: the 7-day credit as before, bit for bit
	credit_segments: bool = False
	# Its TRADE advice taught too (False since 2026-11-04: the user's rule, every trade is dumb unless it buys what could not
	# be bought without it; ~94% of its trades bought nothing that day, and 54% of ours came where it advised one)
	nk2_trade_advice: bool = False
	# Its advice that schema 16's matcher newly reads taught too (False until its label mix is measured: a new label class
	# the model gives ~0 is how 2026-11-05's collapse started): ARTIFACT, MULTI's SPLIT_SINGLES, MERGE_ALL and
	# GARRISON_SWAP, ARMY within one army or all but one moving (ONE's second half), a whole stack legal only as a swap
	# (train.swap_only_moves: its labels and accepted entries). Its other ARMY moves between two armies, and their target
	# slots, are taught as before
	nk2_new_advice: bool = False
	# Nullkiller2's GATHER advice (Advice.cpp visitAgain, the games' ALPHA_MAP_GATHER_ADVICE=1): its visit again of the
	# town its hero stands in, which takes the garrison's troops for it and nothing for us, as our GATHER's label, at
	# this weight in the teacher's cross-entropy (0: not taught, nor asked of the games: supply's campaign_settings). It was its advice at 752 of
	# its 10,781 calls in 8 joint7 games (7.0%), matched to nothing; a GATHER was legal at 3,621 of their 7,577 turn
	# decisions and our model chose it once. Not 2026-11-05's GATHER (accepted wherever legal: the emptied towns, eval
	# 0.265, 28 of 48 players out): the label only where its plan takes those troops now, nothing
	# accepted beside it, MULTI pushed down at every other turn label as ever. A small weight first (train.advice_weights;
	# 0.1 to start): on 8 maps x 14 days of v2501 with the label on (313 labels, 7.1% of the taught advice, 139 pickups
	# asked for) our model gave it 0.00003 (the median; never chose it, and left the town without the troops at 92% of
	# them), so teacher_ce is +0.60 at once at 1.0, +0.31 at 0.5, +0.13 at 0.2, +0.06 at 0.1 (2026-11-05's rule: a jump over
	# +0.3 means stop). The label's span is both families' (train._teacher_span: the hero's moves pushed down there)
	nk2_gather_weight: float = 0.0
	# GATHER's return (2026-11-11): nk2_gather_weight reached in a straight line over this many updates from the first update
	# with it (CampaignLearner.gather_weight_now: 1 / ramp of it at that one), the start kept in the checkpoint so a restart
	# goes on; 0: at once (as before). The games ask Nullkiller2 for its GATHER advice from the first (nk2_gather_weight > 0)
	nk2_gather_ramp: int = 0
	# Nullkiller2's labels into the night (2026-11-11, train.night_dropped): a label is left out of the teacher's
	# cross-entropy (weight 0) where the night terms price its entry more than nk2_night_drop_logit below staying (MOVE's
	# tile_shift, TRAVEL's logits less its prior) and the hero's night there reads nk2_night_drop_reading or more
	# (tile_night; at a TRAVEL's stop features.night_danger). In self-play its labels sat on such priced entries 6.7% of
	# the time against our policy's 2.7% (expected, v3003; 4,000 advised turn decisions), and across v2839-v3003 the
	# weights drifted toward them (P(penalized) on a fixed sample 0.07 -> 0.13, the priors' own effect unchanged); at a
	# reading of 0.6 our heroes are lost 37% of nights, 77% at 0.9. The share of the advised labels dropped is logged
	# (night_dropped). False: every label as before
	nk2_night_drop: bool = False
	# The value valves (2026-11-11; the user: a change that improves the games worsens V for a while, and V's noise then
	# moves the policy, which shares its layers): r_v, the fresh value loss's median of its last 7 readings over an EMA of
	# it (50 updates: a jump counts, a new level is absorbed), drives value_valve (the value terms' gradient into the shared
	# layers x s = clip(1 / r_v, 0.1, 1): V re-fits on its own layer at full rate, the trunk's direction shielded) and
	# kl_valve (the KL control's target x clip(1 / r_v, 0.25, 1): the policy's movement capped while V re-fits; the values'
	# group stays outside lr_scale), each moved at most x1.25 an update (learner.valve_step). In the GATHER collapse
	# (v2736-2839) the value's share of the shared layers' gradient was 0.16 (0.05 calm) and the KL sat at its target. If
	# loss_shares were on, its balancing would raise the value's weight against value_valve. False: as before
	value_valve: bool = False
	kl_valve: bool = False
	nk2_night_drop_logit: float = 1.0
	nk2_night_drop_reading: float = 0.6
	# MULTI inside an exchange's teacher span (train.TEACHER_SPAN_MULTI; the 2026-11-09 audit's M3): every exchange label
	# pushes down the multi-actions it does not accept (TAKE_BEST into the hero that keeps nothing, TAKE_BEST_FAST, ...).
	# Outside it (False) none is ever pushed: after a meeting Nullkiller2 advised, our first exchange choice was its
	# delivery in 35%, TAKE_BEST_FAST in 40%, TAKE_BEST into the visitor in 18%
	exchange_span_multi: bool = False
	# The same only at the exchanges a move of Nullkiller2's advice opened (train.advised_exchanges): there its label is
	# the delivery its plan asked for, the hero met getting the army (Advice.cpp's meeting fixes), and the TAKE_BEST it
	# accepts is that delivery in one decision; an exchange our model walked into on its own keeps the old span (its
	# label there is a rule of ours, Nullkiller2's ranking of the two heroes). exchange_span_multi's all at once was
	# 32% of the decisions pushed off the multi-actions they are nearly all of (the 2026-11-05 collapse's shape). On the
	# meeting fixes' 8 maps x 14 days (v2501): teacher ce 1.668 as it is, 2.358 with exchange_span_multi, 1.697 with
	# this (52 of the 1,015 taught exchange rows; their ce 0.007 -> 2.26: our model's MULTI there is not the delivery)
	exchange_span_advised: bool = False
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
	# The value-loss guard (learner.value_jump; the user, 2026-11-10: on a jump of the campaign's value loss the learning
	# stops at once): the updates are held once the fresh value loss's median over the last 7 updates stands this many
	# times above its calm level (0: off). A restart lifts the hold
	value_jump_stop: float = 3.0
	# The guard's memory starts at this version (0: all of progress.csv's). A hold says "stop and look", not that the
	# change is bad (the user): when the look finds the games sound and the loss's new level theirs (armies on heroes
	# decide more in a battle: the shares move further in a day, the same critic's loss with them), the restart names
	# the version it goes on from, and the guard judges again once it has 57 readings from there. Never set at a
	# deploy to keep the guard quiet through it: only after a hold was looked into
	value_jump_from: int = 0
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
	# The split evaluation (the user, 2026-11-11; evaluate.plan): eval_own_games setups played with Nullkiller2's own choice
	# hidden (our own policy: score_own, what it is judged by), eval_follow_games of them with it shown (score_follow,
	# paired with OWN on those). 0: every eval_games setup with it shown, as before. The run: 32 and 16 (48 games, as before)
	eval_own_games: int = 0
	eval_follow_games: int = 0
	eval_days: int = 56  # (the score at day 28 is logged too: score_d28)
	# The campaign model samples, as in training (seeded per game and seat: AlphaMapAI's ALPHA_MAP_SEED_<COLOR>). Greedy
	# (2026-10-29) was a policy never trained or played: its argmax over the move table is often a neighbouring tile and
	# from there the tile it came from, so heroes walked back and forth a tile at a time until their movement ran out
	# (95% one-tile moves, 88% back onto a tile of that day; sampled: 19% and 2%) and the eval measured that
	eval_temperature: float = 1.0
	eval_size: int = 36  # its maps' side: 36 (the kept eval_maps); another size is generated here and scores into eval_<size>.csv, never eval.csv
	eval_seconds: int = 600  # 240 cut half of the first greedy eval's games short (2026-10-29: 120-240 s a game, 24 at once); an endless siege once held a whole eval for 900 s (no longer: vcmi's stalemate rule)
