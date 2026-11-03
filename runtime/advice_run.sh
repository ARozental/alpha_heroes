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
cd "$HOME/projects/alpha_heroes"
export MALLOC_ARENA_MAX=2
# (no expandable CUDA segments here: the learner's step graph failed to record with them, "device not ready"; the
# joint run's shapes are padded, fixed, so it does not fragment as imitation did)
rm -f runtime/train/*-joint7/drain  # (runtime/drain_restart.sh's: a start always plays)
runtime/memwatch.sh alpha-joint7 > /dev/null 2>&1 &
resume=""
ls -d runtime/train/*-joint7 > /dev/null 2>&1 && resume="--resume"
exec .venv/bin/python -m campaign_model.joint --run joint7 --hours 100000000 $resume --machine configs/machines/rtx5090/joint-gpu-free.toml \
	--battle-from runtime/saved/battle_joint5_07295.pt \
	--set remote_port=7700 campaign.model_architecture=v2 campaign.model_cell_dim=16 campaign.model_dim=128 \
		campaign.share_target=change campaign.init_from=runtime/saved/campaign_imitation.pt \
		campaign.credit=schedule campaign.share_weight_start=0.1 campaign.share_weight_per_day=0.02 campaign.entropy_floor_weight=0 \
		campaign.nk2_advice=1.0 campaign.loss_shares=true campaign.teacher_mix=score campaign.teacher_kl=19 campaign.games=14 remote_slots=9 campaign.inference_device=cpu inference_processes=12 ahead=2 \
		campaign.head_scale=true fixed_mix=true real_share=0.6 league=0.5,0.5,0.0 real_keep=0.12 eval_minutes=180 \
	>> runtime/train/joint7.log 2>&1 < /dev/null
