#!/bin/bash
# joint7 (2026-10-31 night): the campaign model from its imitation of Nullkiller2 (runtime/saved/campaign_imitation.pt),
# self-play RL with Nullkiller2's own advice as the teacher (50% of turn and exchange decisions: Advice.cpp; its cross-entropy weighted
# by our score against it, teacher_mix "score": dominant while it is much better), the credit's share weight 10% + 2% a
# day, the battle model from joint5's best (7295, trained on); no time limit, the memory guard on. (Until 2026-11-01 09:30
# the campaign's decisions were served by the Mac's 11 servers: on the PC's GPU, time-sliced against the learner, games of
# ~16,000 decisions hit their 20-minute limit; games are ~4,000 decisions since the move labels' fix.)
# Advice on 50% of turn and exchange decisions, not 20% (2026-11-01 01:30 CST, after v6): the teacher's labels are 95% of
# the loss and games the bottleneck; at 20% the advice took ~3% of a game's CPU
# The PC alone (the user, 2026-11-01 09:30 CST: "the assumption is that for a lot of time only the PC is available and it
# still needs to work"): every model's inference here (the campaign's on the GPU, the battles' on the CPU); a Mac adds
# game slots when it is up (remote_slots=4: runtime/mac_slots.sh with SLOTS=4, its own servers mirroring these models),
# never anything the run waits on. 10 games here: the campaign's server here and the 4 slots' idle workers take ~2.5 GB.
# Both models' inference on the PC's CPU (one server, 8 processes since 2026-11-02 evening: the games wait on it, each
# ~460 MB PSS; 6 before, the cores the games' spinning TBB threads freed going to it), the GPU the learner's alone: the campaign's on the
# GPU (09:27-09:45) time-sliced the learner, ~2x its time a decision (v34: 72k decisions in 499 s, against ~114k in
# ~400 s), and the permits then held games back (4 of 14 slots playing) while the CPU idled
# Since 2026-11-02 (the user): the value heads' losses weighted by their targets' variance (campaign.head_scale, ramped
# over 20 updates); the battle learner on ~10% of the battles at a fixed mix, campaign 60%, generated self-play 20%, past
# versions 20% (real_keep 0.12 of the campaign's battles, fixed_mix, league 0.5/0.5), the same batch and learning rate,
# its evals every 3 hours (eval_minutes 180): the GPU's and the CPU's time to the campaign, which improves
# Since 2026-11-02 midday (the user): Nullkiller2's advice at every decision (nk2_advice 1.0, ~5% of a game's time more)
# as the teacher over the turn (advice.py: any of its plan's steps or of the town actions it named that turn), and each
# loss a share of the gradient (campaign.loss_shares: the teacher's by our eval score, at most 60%; policy and value
# never under 15%)
# The schedule credit's share weight since 2026-11-03 evening (the user: "increasing slightly faster and with map size"):
# 0.1 + 0.03 a day on a 36-wide map, the rate x 36 / the map's width elsewhere (train.share_weight): all share by day 30
# here (day 45 before), by 60 on a 72-wide map, 120 on a 144-wide one
# 14 games here and 12 inference processes since 2026-11-03 ~12:10 (8 and 8 before): the forkserver's preload now
# applies (78d55e4), and the 25 loader workers hold 2.9 GB in play, not 9.5 (the least available 12.2 GB since, against
# ~4). The learner takes ~500 games an hour; the 17 slots made ~300 (~17-18 an hour a slot): 6 more games here (~0.35
# GB each: a VCMI client and its worker) and 4 more servers (~0.45 GB each) for them, ~3.9 GB of the ~8 freed; back to
# 8 and 8 if the least available falls under 4 GB
# 8 games here, 9 remote slots, 8 inference processes since 2026-11-03 ~10:25 (11, 6, 10 before): with ahead 2 the
# permits were plenty (22) but only 7 of the 11 local games played, the rest held by the memory floor (supply.py
# MEMORY_FLOOR_GB 4.5; the least available 4.02 GB, median 4.85), while each loader worker holds ~0.41 GB played or not.
# A game here costs this machine ~0.65 GB (its worker and VCMI client), one on the Mac ~0.5 (its worker and half a
# battle worker): fewer local workers and servers, more Mac slots (SLOTS=9 runtime/mac_slots.sh alpha-joint7
# campaign.inference_processes=6; ~17-19 games an hour a Mac slot against ~14 here), ~0.5 GB more free here
# ahead 2 since 2026-11-03 ~09:45 (1.5 in the machine config): the permits held the games finished and in flight to
# 30 games' worth (1.5 updates of 20), so during each update the slots emptied (the supply's snapshots: permits 0 in 5
# of 8, 9-15 of the 17 campaign slots playing, the Mac's 1-4 of 6) while the learner waited ~45% of its time for games.
# 1.5 was for memory (2026-11-01: 60 waiting long games, ~140 MB each); the waiting games' inputs are on disk since
# (strip_inputs) and games are ~2,900 decisions. Labels two versions old are redone (learner._relabel_waiting)
# 6 remote slots since 2026-11-03 ~08:15 (4 before; the user: "allocate more cpus from here", the Mac): the Mac's 2
# campaign inference processes were saturated (97-99% each) at 4 slots, so it runs 6 slots with 4 (SLOTS=6
# runtime/mac_slots.sh alpha-joint7 campaign.inference_processes=4). Each remote slot costs this machine a loader worker
# (~0.41 GB private: the campaign's one a slot, the battles' one per 2), +1.2 GB with the 2 servers' +0.9: the least
# available memory 5.2 -> ~3.1 GB expected; back to 4 slots and 8 servers if it falls under 3 GB
# 10 inference processes since 2026-11-03 ~08:15 (8 before): after the trainer's window budget and decoding fixes the
# learner waits ~40% of the time for its 20 games; the PC's CPU was ~65% idle, the 8 servers at ~0.7 cores each, and
# the games (11 campaign, the battles' and the evals', spread round-robin: socket_for) queue on shared servers. ~0.46 GB
# each; back to 8 if the least available memory falls under 3 GB or the games finished an hour do not rise
# 11 games since 2026-11-02 23:30: with 10, after the games' TBB cap and the loaders' preload, at least 5.3 GB stayed
# available through an hour of updates (a game ~0.65 GB: its loader worker and its VCMI client); back to 10 if the
# least available falls under 4 GB
# 10 games, not 13 (13 at 16:43-16:55 CST on 2026-11-01: the memory guard stopped the run at 1.13 GB, its games in flight
# lost). The budget: ~26.6 GB available with nothing running, less the inference servers (4.0), the trainer at its peak
# (9.3 GB before its first update after a start, 11.75 seen at 15:46) and the guard's 3 GB, leaves ~10 GB for slots
# here at ~0.65 GB each (a loader worker ~0.4 GB PSS, its VCMI client 0.15-0.25) and the Mac's (a worker each here):
# 10 campaign + 4 battle slots + the Mac's 4 is what fits (least available 4.2 GB at a start, 15:46)
# 11 games since 2026-11-01 05:50 CST: the permits keep games playing through an update now (supply.allow), and an update
# takes ~6 GB more (3.7 GB was left with no game playing): 2 fewer games for the memory guard's margin
# 13 games, not the machine's 16 (2026-11-01 01:47): the memory guard stopped the run at 1.4 GB available as the 20
# loader workers held their long games (~13,000 decisions each: 12.7 GB), the learner ~15 finished games and a battle
# eval ran in its process; 16.7 GB was available at the start
# Start: systemd-run --user --unit=alpha-joint7 -p KillSignal=SIGINT -p TimeoutStopSec=900 -p KillMode=mixed \
#          -p LimitNOFILE=65536 ~/projects/alpha_heroes/runtime/advice_run.sh
#        and its keeper (starts it again after a memory-guard stop):
#        systemd-run --user --unit=alpha-joint7-keeper ~/projects/alpha_heroes/runtime/keeper.sh alpha-joint7 runtime/advice_run.sh
# docs/optimizer_plan.md since 2026-11-06 ~v1425 (the user: "implement them all"; campaign_model/config.py's defaults
# since, all at once, bisected in reverse order if the run goes bad): the value heads at 10x the rate (1e-4: the user's OK above
# the 3e-5 cap, which is the policy's), apart from the KL control's scale; the KL target 0.015 (no teacher inflation:
# was 0.06); the loss shares' weights frozen from v1420 (campaign.loss_shares_freeze: value ~353, teacher ~1.8,
# foresight ~0.66); trust max(0.4, its 0.98 average); the trunk's beta1 0.95; each parameter group clipped at its own
# median norm (calibrated over the first 200 steps, then all x 0.5 / the total's median); the value heads' weights by
# their running residual (ramped in over 20 updates). The battle learner: its value heads at 10x, clipped by group
# campaign.target_kl=0.03 since 2026-11-06 ~14:20 (~v1468; the plan's step-2 fallback, "the eval falls two evals
# running": 0.441 (v1422, before) -> 0.363 -> 0.335): at 0.015 the policy's rate fell to 0.2-0.4x, because the update's
# KL barely follows the rate (2.5x less rate, 1.6x less KL; it varies ~5x with the batch at one rate), so the control
# ratchets on the batch's noise; at 0.03 the rate goes back to its cap, the critic still at 10x it (step 1)
# The battle learner since 2026-11-06 ~19:30: its trunk's beta1 0.95 (the plan's step 5: its gradient read as pure noise
# a step since accumulate 2), and the escapes pressed down while its value head says we are ahead (the user: "another -3
# for the logit if we are ahead and maybe something that scales nicely with how much we are ahead": priors.escape_ahead)
cd "$HOME/projects/alpha_heroes"
export MALLOC_ARENA_MAX=2
# (no expandable CUDA segments here: the learner's step graph failed to record with them, "device not ready"; the
# joint run's shapes are padded, fixed, so it does not fragment as imitation did)
rm -f runtime/train/*-joint7/drain  # (runtime/drain_restart.sh's: a start always plays)
runtime/memwatch.sh alpha-joint7 > /dev/null 2>&1 &
# campaign.memory_steps=2 (2026-11-05): the passes over pairs of a player's decisions, the memory carried from the first
# into the second, so the losses reach its writer (the step graph replay 117 ms against 108 at 1, the GPU idle)
# campaign.privileged_credit=true (2026-11-07): the schedule credit's V7 and V_end baselines from the privileged critic
# (0fe2869, the user's design: over its first 45 updates unexplained 0.863 -> 0.806, V_end's fresh excess -23%, the share
# heads' -5.5%); still unbiased (its inputs, the rivals' earlier states and our share as the day began, precede the decision)
# trunk_beta1=0.9 (2026-11-07, b1): the battle trunk's momentum back from 0.95 (6ec628f): shooters wasting a full-damage
# shot rose from that deploy on (epoch ~8500: training 1.4% -> 2.1-2.4% of their chances, evals vs BattleAI 1-2% ->
# 2.5%, 3.7%; the DEFEND prior unchanged at -5, so moves or waits), the escape prior only touching escapes
# campaign.exchange_span_advised=true (2026-11-10, e5; the 2026-11-09 audit's M3 in its gentle form, after the meeting
# fixes held over three evals: 0.420, 0.405, 0.431 against 0.434 before): at the exchanges a move of Nullkiller2's
# advice opened, the multi-actions are inside its label's span, so the delivery its plan asked for (the TAKE_BEST it
# accepts) is taught against the other multi-actions there; ~5% of the taught exchange rows, teacher ce +0.03 measured
# (1.668 -> 1.697). An exchange our model walked into on its own keeps the old span
# campaign.nk2_gather_weight=0.1 (2026-11-10, e5; after the span held over three evals: 0.511, 0.491, 0.489 against
# 0.405-0.469 before): Nullkiller2's visit again of the town its hero stands in is our GATHER's label, at a tenth of a
# label's weight in the teacher's cross-entropy, and the games ask its advice for it (ALPHA_MAP_GATHER_ADVICE). Our
# model gives the label ~0 (the 2026-11-05 collapse's rule: a small weight of its own); teacher ce +0.06 expected.
# Watch: emptied towns, eliminations, the type drift
# OFF AGAIN 2026-11-10 18:30 (e5): the six evals after it 0.475, 0.380, 0.314, 0.345, 0.402, 0.271 (eliminated 15, 21,
# 23, 22, 19, 28 of 48) against 0.511, 0.491, 0.489 (9, 10, 6) before; the campaign back to v2728's weights
# (checkpoint_02728.pt, the last eval before it), the games not asked for the advice. The override is gone from the
# line below: campaign.nk2_gather_weight is 0 (config.py)
# campaign.further_windows=0 campaign.unseen_exits=false (2026-11-10, e5): schema 17's code with both of its switches
# off (config.py's defaults are 1 and true): the games build no further window and flag no unseen exit, and the model
# is the one before it bit for bit (42 outputs and features on the same weights and inputs differ by 0). The code
# first, then further_windows=1, then unseen_exits=true, each judged over three evals
# campaign.further_windows=1 campaign.window_budget=5.0 (2026-11-10 night, e5; deploy 52): a hero that reaches tiles
# outside its own window today (past a gate, a monolith, the other level) gets one more window there, so those tiles
# are MOVE targets; Nullkiller2's advised moves there become labels (about 2% of its MOVE labels). The budget: a
# further window is one more window a decision, and at 4.0 a third of the updates already stepped some micro-batch
# over every hero slot (191 s against 164 s an update). Before the PC (campaign_model/paired_eval.py, v2886's
# weights, the eval's 48 games, two replicates an arm): the score 0.484 -> 0.481, paired -0.003 +- 0.024, eliminated
# 9 and 12 -> 7 and 15: no effect at these weights either way, as expected of targets the model has not trained on;
# the gain, if any, is what it learns there. Watch over three evals: a fall of 0.1 or the guard takes it out;
# "every hero slot in" in the update lines (over the budget), seconds an update, MOVE's share (type_drift)
# night_gain no longer learns (2026-11-10 night, b1's 2458a27 + 23d10d1; rides deploy 52, no change of the forward):
# over 23 updates the teacher's cross-entropy pushed it down in 21 (-0.0054 +- 0.0008) and the policy's credit had no
# sign (+0.00002 +- 0.00022), so a learnable gain was noise plus Nullkiller2's push, and Nullkiller2's choice is not
# evidence of our expected loss tonight. It stays at its value (raw -0.004, gain 0.96). A departure from the rule for
# priors (a learnable correction decayed toward the prior), for this one scalar: push_night_gain_* read nan from here
# campaign.unseen_exits=true (2026-11-11 night, e5; deploy 53): a gate, monolith or whirlpool no exit of which we have
# seen is flagged (VCMI's pathfinder goes through a teleporter only to an exit the player sees, so the land behind
# it was never a target): MOVE's explore prior 1 there, TRAVEL's area unrevealed; encode_tiles.unseen_exit starts at 0.
# Before the PC (paired_eval, v2886's weights, the eval's 48 games, three replicates an arm, both with further
# windows): 0.472 -> 0.531, paired +0.059 +- 0.025, eliminated 7, 15, 13 -> 8, 8, 9. It follows deploy 52 after one
# clean eval (v2934: 0.579 +- 0.054, eliminated 7, no step over the window budget, the value loss at its level)
# Our towns' night (2026-11-11, b1's f032d3f + 74480f8; e5 deploys as 54; no switch on this line: on in the code,
# ALPHA_TOWN_NIGHT=0 turns it off for a test): MOVE's and TRAVEL's logit priors also pay the change in P(a town of
# ours falls tonight) x its stake for the hero that enters or leaves it, our only town at our Φ. Why: the collapsed
# GATHER weights (v2830) stay collapsed under the hero's night term alone (eliminated 20.7 of 48 with and without
# it); the eliminations moved from the hero (first to go in 14 of 22 -> 5 of 21) to the town it stands beside or
# walks out of (2 -> 11), a 30-60k army alive within 7 tiles of an empty last town. Paired, three replicates side
# by side: v2830 0.503 -> 0.582 (+0.079 +- 0.052), eliminated 19/18/18 -> 10/18/9; v2912 (today's weights) 0.531 ->
# 0.518 (-0.013 +- 0.040), eliminated 10/10/8 -> 11/8/11, idle_movement 0.172 -> 0.187. Neutral now, the
# prerequisite of GATHER's return. campaign.nk2_gather_ramp (b1's 5d31166) is in the code and off: GATHER's weight
# is not on this line; its readings (gather_weight, army_on_heroes, eliminated, value_target_std) log from here
# battle_length_scale=30 (2026-11-11, b1's plan of 2026-11-10, deploy 2 of 3; the battle model, its own track): a
# battle's decisions weigh 1 / (1 + (n - 1) / 30), so an update's gradient rests on more battles (one update's was
# ~14 games' for the policy, ~4 for the value: a few 150-decision stand-offs held 28% of its decisions). Baseline
# over 47 updates before it: games_by_weight 49 +- 14, signal_share_applied_trunk -0.001 +- 0.017, value_loss_fresh
# 0.0076 +- 0.0022. b1's predictions: games_by_weight about x2, value_loss_fresh's update-to-update spread down a
# third within 30 updates, signal_share about x2 within ~100; else it comes out
# GATHER's return (2026-11-11, e5; deploy 55): campaign.nk2_gather_weight=0.1 campaign.nk2_gather_ramp=200, its
# label's weight in a straight line from 0 over 200 updates (about 10 hours), the start kept in the checkpoint. The
# 2026-11-10 collapse: armies moved onto heroes in ~10 updates, the heroes died in the field, the emptied towns
# fell. Since then: the hero's night (deploy 50), our towns' night (54; on the collapsed weights eliminated 55 -> 37
# of 144). The rollback point: runtime/saved/campaign_joint7_pre_gather_ramp.pt (the checkpoint at this deploy).
# What takes it out again: the value-loss guard; `eliminated` (self-play, progress.csv) held above 0.03 (0-0.016
# before, 0.06-0.08 in the collapse); value_target_std at twice its 0.02; an eval down 0.1 with 14+ eliminated and a
# Mac replay of its checkpoint agreeing. army_on_heroes (0.08-0.18 before) is expected to rise: that is GATHER.
# campaign.nk2_night_drop (b1's 4785d0f) is in the code and off: its trigger is the deep-end reading (b1's
# measure_a2: P of an entry into a night reading >= 0.6, 0.084 at v3003) passing ~0.10
# The value valves (2026-11-11, the user's diagnosis; e5's design corrected by b1, b1's 8605deb; deploy 56):
# campaign.value_valve=true campaign.kl_valve=true. The user: a change that improves the game makes V worse, and the
# noisy V then moves the policy far more than it would with separate networks. The run's numbers: in the 2026-11-10
# collapse the fresh value loss went 0.00055 -> 0.0048, the value terms' share of the shared layers' gradient 0.05
# -> 0.16 (above 0.3 in 21 of 104 updates), the policy's KL an update at its target throughout. So, while the fresh
# value loss's 7-median stands above its own ~50-update average (r_v > 1): the value terms' gradient into the
# shared layers x clip(1 / r_v, 0.1, 1) (value_hidden, the heads and the privileged critic at full gradient and
# their own rate: V re-fits, the trunk is not dragged), and the KL control's target x clip(1 / r_v, 0.25, 1) (the
# policy moves less an update until V has caught up); each at most x1.25 or /1.25 an update. At r_v <= 1 nothing
# changes. Logged: valve_r, valve_s, kl_target. Not a paired test's matter (training, not play): the PC's update
# check replays the recorded step graph with the valve at 0.5 against the eager step. The guard stays the last resort
# campaign.value_jump_stop is 3 (config.py; the user, 2026-11-10): the campaign's updates are held when its fresh value
# loss jumps (the median of the last 7 updates at 3 times its calm level); the line "VALUE LOSS JUMP" in the log, the
# file campaign/held. A restart lifts the hold
# Schema 18, the night's danger (2026-11-10, b1's 7441d1e; e5 deploys; no switch, live with the code): MOVE's and
# TRAVEL's logit priors pay P(the hero is lost tonight where its day ends) x (army + hero) in gold as a pickup pays,
# at the tile where today's walk stops (travel_stop, from the game); a town of ours a shelter for a hero that can be
# inside it; MOVE's and TRAVEL's type offers go below 0 when every move is worse than staying. The network's inputs
# as before. Why: v2728's heroes were killed on Nullkiller2's turn 18 times in 12 games, 16 of them after a TRAVEL
# (with GATHER the same deaths took 20-43k of army each: the collapse of 2026-11-10). On 36 paired maps, v2728's
# weights against Nullkiller2: the eval's score 0.461 -> 0.595 (+0.134 +- 0.041), eliminated 11 -> 4, big losses on
# the enemy's turn 64 -> 49, land explored not less. Watch: eliminated and score over three evals (0.497, 11 of 48
# before), features.night_gain (if it sinks as danger_gain did, something teaches the heroes out), idle movement
# The overhaul's first deploy (2026-11-11, e5 with b1; deploy 57; the user: "Make the feature changes and get rid of
# the teacher"). Schema 19: Nullkiller2's view of each turn decision as the model's INPUTS (campaign.nk2_inputs=true:
# ALPHA_MAP_NK2_INPUTS and its gather advice in every game, the evaluations' too): its candidate tasks on our MOVE and
# TRAVEL entries with its scores, times, losses and dangers (k_*), its plan's tier and its threats, and its OWN CHOICE
# at the decision (g_nk_action, g_nk_index, g_nk_cell; where its first request is none of our entries, its plan's
# first task that is one); each MOVE tile and TRAVEL area also carries bounded views of its value and time
# (campaign_model/views.py: 51 channels an entry; every new layer's output starts at 0).
# campaign.nk2_follow=0.9: the games' policy is 0.9 x (the entry Nullkiller2 itself would play now) + 0.1 x (our own
# policy), exactly, in the type's and its table's probabilities (model_v2's follow); the chance learns within
# 0.31..0.96. Why this and not its teacher's loss or a prior in logits: playing its choice outright was worth
# +0.293 +- 0.059 on 144 paired games (b1's follow test, v3045, with its gather advice); the teacher's loss held 65% of
# the shared gradient for an agreement of 0.24; a prior on its best-scored candidate gave +0.017 +- 0.058 at 2 and
# -0.013 +- 0.058 at 4 (v3090, 48 paired games); a prior of 8 logits on its own choice +0.052 +- 0.064 (it ran without
# the gather advice: its choice known at 76% of turn decisions, its trades taken at 20%). The mixture itself has NO
# paired test (the user, 12:50: no experiment over 10 minutes; use what is known): its evidence is the follow test.
# campaign.nk2_teacher=false: the teacher's weight and its share of the gradient 0 (its advice still asked and
# logged: teacher_agree is now how often we play its choice; the plan head's label stays). The policy learns where to
# leave its choice from the games alone, and the chance of following learns with it.
# Out of the line: campaign.nk2_gather_weight / nk2_gather_ramp (GATHER's label: the teacher's) and
# campaign.value_valve / kl_valve (their premise failed: in the 2026-11-10 collapse the critic's unexplained share went
# 0.82 -> 0.88, the 8x loss was the targets' spread).
# campaign.value_jump_from=3107: the guard held the campaign at v3106 (11:25: fresh value loss 4.0 times its calm level
# during GATHER's ramp). Looked into: v3090 against v3025 on 144 paired games -0.038 +- 0.042, eliminated 33 and 33, so
# the level was the games' (armies on heroes 0.14 -> 0.30), not a collapse. The guard's memory starts again at 3107 and
# it judges again after 57 updates; until then: eliminated, the evals, a Mac replay of the first evaluated checkpoint.
# What takes it out again: the evaluation below its level before this deploy (0.47-0.53) over two evals with a Mac
# replay agreeing (rollback point runtime/saved/campaign_joint7_pre_overhaul.pt). Expected: the evaluation well above
# 0.6 (the follow test's arm was near 0.8), teacher_agree near 0.9 where its choice is known, the fresh value loss at
# a new level (the games change), KL an update about a tenth of before (the mixture moves a tenth as far as our own
# policy does; the rate's scale is capped at 1, so our own policy is not sped up to make up for it).
# Deploy 58 (2026-11-11, b1 with e5): campaign.adv_weight=1, the advantage head, stage 1 (model_v2.AdvantageHead,
# 428baab): it predicts the policy's own advantage of playing (type, entry), from the global token, MOVE's and
# TRAVEL's pair vectors and whether the entry is Nullkiller2's own choice, every input detached; trained on the
# update's standardized advantage (sd 1) on the rows the policy credits, in the value group at the critic's rate.
# THE POLICY READS NONE OF IT: the games play as before (its outputs are the trainer's only, not the contract's).
# Why: one update's ~14 games give the policy no direction, while a regression averages over every game; at
# nk2_follow=0.9 a followed row passes our own policy only ~P_own/9 of its gradient.
# What decides stage 2 (the policy reading it): adv_ev (its explained variance of each update's advantage, predicted
# before the update's steps: fresh rows) above 0 over 50+ updates; adv_ev_deviating apart (the rows that left
# Nullkiller2's choice: where reading it could change a choice). Logged in progress.csv and the update's line.
# Deploy 59 (2026-11-11, b1 with e5): campaign.nk2_choice_games=0.5, own games. The user: "this can work, but it is
# also super risky in the sense that we will only learn to fix NK2 and not really learn good moves in general". Half
# of the self-play training games hide Nullkiller2's OWN CHOICE from the inputs (ALPHA_MAP_NK2_CHOICE=0: g_nk_action,
# g_nk_index, g_nk_cell left 0): there the follow mixture is off and our own policy plays every decision of the game,
# at full gradient on its own states; its candidates, scores, plan and threats stay among the inputs, and its advice
# is still asked and recorded (teacher_agree). The other half stays at follow 0.9. One setting a game, every player
# alike (never a hidden-choice player against a follower: training against Nullkiller2), drawn from the game's seed.
# The advantage head keeps the two kinds apart (AdvantageHead.unnamed on the decisions without its choice). Readings
# apart for the two kinds each update: choice_hidden, teacher_agree_*, adv_ev_*, adv_raw_sd_*, eliminated_*.
# What would show it failing: the follow-off evaluation (the eval's OWN arm) not rising over a day while the hidden
# games' eliminations stay high; the head's adv_ev falling. The headline evaluation at follow 0.9 says nothing about
# our own policy.
# campaign.eval_own_games=32 campaign.eval_follow_games=16 (the user's "ok do it", 2026-11-11): the PC evaluation in two
# arms, the same 48 games and time: OWN, setups 0-31 with Nullkiller2's own choice hidden (its other inputs on): the
# score our own policy is judged by (eval.csv score_own; eval_score, TensorBoard eval/campaign, the teacher weight's
# score read it); FOLLOW, 16 of those (6 / 5 / 5 by players, our seats spread) with it shown at the run's follow
# chance (score_follow; follow_minus_own paired on those 16). The old score columns stay empty from then on. Noise:
# about +-0.15 at 32 games, +-0.2 at 16: read trends over three or more evaluations. Baselines: own ~0.51 (Mac, v3090,
# job A's follow-0 arm), follow 0.75-0.79 (v3120 0.753, v3138 0.765, job A 0.792).
# campaign.adv_gain=1 (stage 2a; the user, 16:30, "do 1+2": on now, without waiting for 50 updates of adv_ev; "we care
# about what we can learn and how fast, not always the total score"): our own policy reads the advantage head,
# 1 logit a unit of its prediction (the update's advantage in sd units), each table's term clipped to +-3 and centered
# on its mean under our own table, each type's term clipped to +-3, before the follow mixture (5a04565). With the head
# at adv_ev ~0.01-0.03 it moves almost nothing today and grows as the head learns (adv_read, adv_kl each update, shown
# and hidden games apart). What shows it failing: the KL an update jumping; the OWN evaluation falling over three
# evaluations; adv_ev turning negative after the gain is on (the head chasing its own effect).
# campaign.schedule_action_price=2e-6 (the user, 2026-11-11: "we want to turn it on"; their size, never above 2e-5;
# designed 2026-11-09): each decision's 7-day credit less 2e-6 for every action the player makes through that week (the
# policy's credit only; the heads' labels stay the real outcomes). Judged by our heroes' idle movement in the OWN arm and
# in the hidden games (idle_movement_hidden). Comes out if idle movement does not fall over a day, or if END becomes
# the first action of a day more often (end_first, end_first_hidden).
resume=""
ls -d runtime/train/*-joint7 > /dev/null 2>&1 && resume="--resume"
exec .venv/bin/python -m campaign_model.joint --run joint7 --hours 100000000 $resume --machine configs/machines/rtx5090/joint-gpu-free.toml \
	--battle-from runtime/saved/battle_joint5_07295.pt \
	--set remote_port=7700 campaign.model_architecture=v2 campaign.model_cell_dim=16 campaign.model_dim=128 \
		campaign.share_target=change campaign.init_from=runtime/saved/campaign_imitation.pt \
		campaign.credit=schedule campaign.share_weight_start=0.1 campaign.share_weight_per_day=0.03 campaign.entropy_floor_weight=0 \
		campaign.nk2_advice=1.0 campaign.loss_shares=true campaign.teacher_mix=score campaign.teacher_kl=19 campaign.games=14 remote_slots=9 campaign.inference_device=cpu inference_processes=12 ahead=2 \
		campaign.head_scale=true campaign.loss_shares_freeze=1420 campaign.target_kl=0.03 campaign.memory_steps=2 campaign.privileged_credit=true campaign.exchange_span_advised=true campaign.further_windows=1 campaign.window_budget=5.0 campaign.unseen_exits=true campaign.nk2_inputs=true campaign.nk2_follow=0.9 campaign.nk2_teacher=false campaign.value_jump_from=3107 campaign.adv_weight=1 campaign.nk2_choice_games=0.5 campaign.eval_own_games=32 campaign.eval_follow_games=16 campaign.adv_gain=1 campaign.schedule_action_price=2e-6 trunk_beta1=0.9 fixed_mix=true real_share=0.6 league=0.5,0.5,0.0 real_keep=0.12 eval_minutes=180 battle_length_scale=30 \
	>> runtime/train/joint7.log 2>&1 < /dev/null
