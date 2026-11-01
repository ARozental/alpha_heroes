#!/bin/bash
# joint7 (2026-10-31 night): the campaign model from its imitation of Nullkiller2 (runtime/saved/campaign_imitation.pt),
# self-play RL with Nullkiller2's own advice as the teacher (50% of turn and exchange decisions: Advice.cpp; its cross-entropy weighted
# by our score against it, teacher_mix "score": dominant while it is much better), the credit's share weight 10% + 2% a
# day, the battle model from joint5's best (7295, trained on); 16 games on the PC, their campaign decisions served by the
# Mac's 11 servers (runtime/mac_slots.sh: on the PC's GPU the inference server, time-sliced against the learner, made
# games hit their 20-minute limit), the battle model's on the PC's CPU; no time limit, the memory guard on.
# Advice on 50% of turn and exchange decisions, not 20% (2026-11-01 01:30 CST, after v6): the teacher's labels are 95% of
# the loss and games the bottleneck; at 20% the advice took ~3% of a game's CPU
# 13 games, not the machine's 16 (2026-11-01 01:47): the memory guard stopped the run at 1.4 GB available as the 20
# loader workers held their long games (~13,000 decisions each: 12.7 GB), the learner ~15 finished games and a battle
# eval ran in its process; 16.7 GB was available at the start
# Start: systemd-run --user --unit=alpha-joint7 -p KillSignal=SIGINT -p TimeoutStopSec=900 -p KillMode=mixed \
#          -p LimitNOFILE=65536 ~/projects/alpha_heroes/runtime/advice_run.sh
cd "$HOME/projects/alpha_heroes"
export MALLOC_ARENA_MAX=2
# (no expandable CUDA segments here: the learner's step graph failed to record with them, "device not ready"; the
# joint run's shapes are padded, fixed, so it does not fragment as imitation did)
runtime/memwatch.sh alpha-joint7 > /dev/null 2>&1 &
resume=""
ls -d runtime/train/*-joint7 > /dev/null 2>&1 && resume="--resume"
exec .venv/bin/python -m campaign_model.joint --run joint7 --hours 100000000 $resume --machine configs/machines/rtx5090/joint-gpu-free.toml \
	--battle-from runtime/saved/battle_joint5_07295.pt \
	--set remote_port=7700 campaign.model_architecture=v2 campaign.model_cell_dim=16 campaign.model_dim=128 \
		campaign.share_target=change campaign.init_from=runtime/saved/campaign_imitation.pt \
		campaign.credit=schedule campaign.share_weight_start=0.1 campaign.share_weight_per_day=0.02 campaign.entropy_floor_weight=0 \
		campaign.nk2_advice=0.5 campaign.teacher_mix=score campaign.teacher_kl=19 campaign.games=13 \
		'inference_endpoints=campaign=tcp:192.168.1.131:7951,tcp:192.168.1.131:7952,tcp:192.168.1.131:7953,tcp:192.168.1.131:7954,tcp:192.168.1.131:7955,tcp:192.168.1.131:7956,tcp:192.168.1.131:7957,tcp:192.168.1.131:7958,tcp:192.168.1.131:7959,tcp:192.168.1.131:7960,tcp:192.168.1.131:7961' \
	>> runtime/train/joint7.log 2>&1 < /dev/null
