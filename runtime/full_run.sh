#!/bin/bash
# The campaign model's full run (2026-10-31), on the PC alone, no time limit:
#   1. imitation of Nullkiller2 (campaign_model/imitate.py --stream): its games played by workers as they are needed
#      (nothing written to disk), each decision seen once, the model with memory learning them in order (64 decisions
#      a chunk, 128 players at once: a chunk's inputs, kept for its backward pass as wire bytes, are ~7.4 GB of RAM; at
#      64 the GPU was 27% busy, its time kernel launches), until the held-out loss
#      (each decision's, before the model trained on it) stops
#      improving or 8 hours pass; again from its checkpoint if this script starts again before the end
#   2. joint6: self-play RL from that model (campaign_model/joint.py), the credit's share weight 10% + 2% a day,
#      the teacher's KL (to that model) weighted by our score against Nullkiller2 (dominant while it is much better,
#      5% once we are), the battle model from joint5's best (7295, +0.096); resumed if it has begun
# Start: systemd-run --user --unit=alpha-full -p KillSignal=SIGINT -p TimeoutStopSec=900 -p KillMode=mixed \
#          -p LimitNOFILE=65536 ~/projects/alpha_heroes/runtime/full_run.sh
# Stop:  systemctl --user stop alpha-full (the phase running stops as SIGINT stops it)
cd "$HOME/projects/alpha_heroes"
export MALLOC_ARENA_MAX=2
# (a step's shapes vary with its heroes' windows: without expandable segments the CUDA cache filled all 32 GB at 128
# lanes, a fragmentation OOM waiting to happen)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The memory guard: the unit stopped (as systemctl stop does) once WSL's available memory falls under 1.5 GB
runtime/memwatch.sh alpha-full > /dev/null 2>&1 &
child=
# (TERM, not INT: bash starts a background job with SIGINT ignored; both Python programs stop on TERM as on INT)
trap '[ -n "$child" ] && kill -TERM "$child" && wait "$child"; exit 0' INT TERM
BC=runtime/teacher/bc_final
mkdir -p "$BC" runtime/train
if [ ! -f "$BC/done" ]; then
	init=""
	[ -f "$BC/bc_latest.pt" ] && init="--init $BC/bc_latest.pt"
	.venv/bin/python -m campaign_model.imitate --stream 10 --data none --out "$BC" $init --variants 128x2 --lanes 128 --steps 64 \
		--eval-every 25 --until-plateau 0.005 --plateau-periods 5 --minutes 480 --decode-threads 8 \
		--tb runtime/train/fullrun-imitation/tensorboard >> runtime/teacher/bc_final.log 2>&1 &
	child=$!
	wait "$child"
	status=$?
	child=
	[ "$status" -eq 0 ] || { echo "$(date +%T) imitation ended with $status: stopping" >> runtime/teacher/bc_final.log; exit "$status"; }
	touch "$BC/done"
fi
resume=""
ls -d runtime/train/*-joint6 > /dev/null 2>&1 && resume="--resume"
.venv/bin/python -m campaign_model.joint --run joint6 --hours 100000000 $resume --machine configs/machines/rtx5090/joint-gpu-free.toml \
	--battle-from runtime/saved/battle_joint5_07295.pt \
	--set inference_device=cuda inference_processes=1 campaign.model_architecture=v2 campaign.model_cell_dim=16 campaign.model_dim=128 \
		campaign.share_target=change campaign.init_from="$BC/bc_latest.pt" campaign.teacher_checkpoint="$BC/bc_latest.pt" \
		campaign.credit=schedule campaign.share_weight_start=0.1 campaign.share_weight_per_day=0.02 \
		campaign.teacher_kl=19 campaign.teacher_mix=score campaign.teacher_kl_target=0 campaign.entropy_floor_weight=0 \
	>> runtime/train/joint6.log 2>&1 < /dev/null &
child=$!
wait "$child"
