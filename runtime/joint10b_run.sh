#!/bin/bash
# joint10 STAGE B (e5 for the user, 2026-11-19; a DRAFT until e5 and the user agree its numbers): the critic frozen (no
# step; its baselines and the link's targets only), the policy alone on our own self-play games at 1e-5. At the stage's
# start, before its first game, the policy's shared layers are set to the trained critic's and link_map to the identity
# (campaign.policy_body_from_critic; the link reads the state the critic's trunk reads: campaign.link_before_again, so
# the link starts at 0). The link at 0.02, fixed, unbalanced. The credit of four points (2, 5, 7 days, the end) at equal
# weights (campaign.credit_horizons); the calibrated baselines (campaign.credit_calibrated) only on the user's word: the
# credit gate at the policy's first update stops the run if an uncalibrated baseline's variance ratio is over 1.2.
# The NK2 stream off (no NK2 games; every slot plays our own games). Resumes joint10's run dir (stage A's checkpoint).
cd "$HOME/projects/alpha_heroes"
export MALLOC_ARENA_MAX=2
export ALPHA_EMPIRE_ALPHA=0.5  # (empire.py's u, as joint9: read at import in every process this script starts)
rm -f runtime/train/*-joint10/drain  # (runtime/drain_restart.sh's: a start always plays)
runtime/memwatch.sh alpha-joint10 > /dev/null 2>&1 &
resume=""
ls -d runtime/train/*-joint10 > /dev/null 2>&1 && resume="--resume"
exec .venv/bin/python -m campaign_model.joint --run joint10 --hours 100000000 $resume --machine configs/machines/rtx5090/joint-gpu-free.toml \
	--battle-from runtime/saved/battle_joint7_frozen.pt \
	--set remote_port=7700 campaign.model_architecture=v2 campaign.model_cell_dim=16 campaign.model_dim=128 \
		campaign.share_target=change campaign.init_from= battle_frozen=true generated_actors=0 \
		campaign.stage=policy campaign.critic_net=true campaign.policy_body_from_critic=true campaign.link_before_again=true campaign.credit_horizons=true campaign.credit_calibrated=false campaign.critic_lr=2e-5 campaign.critic_batch=256 campaign.head_scale=true \
		campaign.nk2_stream=false campaign.nk2_games=0 campaign.nk2_remote=0 campaign.nk2_days=56 campaign.nk2_rows=12000 campaign.nk2_floor_games=125 campaign.nk2_buffer_cap_games=300 \
		campaign.nk2_heldout_games=150 campaign.nk2_heldout_rows=5000 campaign.nk2_keep_games=300 \
		campaign.learning_rate=1e-5 campaign.link_weight=0.02 campaign.link_balanced=false \
		eval_minutes=100000000 campaign.eval_minutes=60 campaign.eval_own_set=rep12 campaign.eval_own_games=48 campaign.eval_own_every=4 campaign.eval_follow_games=0 eval_every=0 \
		campaign.credit=schedule campaign.share_weight_start=0.1 campaign.share_weight_per_day=0.03 campaign.entropy_floor_weight=0 \
		campaign.nk2_advice=1.0 campaign.loss_shares=true campaign.teacher_mix=score campaign.teacher_kl=19 campaign.games=20 remote_slots=15 campaign.inference_device=cpu inference_processes=8 ahead=2 \
		campaign.loss_shares_freeze=0 campaign.target_kl=0.03 campaign.memory_steps=2 campaign.privileged_credit=true campaign.exchange_span_advised=true campaign.further_windows=1 campaign.window_budget=6.0 campaign.unseen_exits=true campaign.nk2_inputs=true campaign.nk2_follow=0 campaign.nk2_teacher=false campaign.value_jump_from=0 campaign.value_jump_stop=0 campaign.adv_weight=1 campaign.nk2_choice_games=0 campaign.adv_gain=1 campaign.schedule_action_price=1e-5 campaign.adv_by_kind=true campaign.town_guard=0 campaign.credit_segments=false campaign.value_shared=1.0 campaign.value_shared_from=-100 campaign.end_week_share=1.0 campaign.nk_predict_weight=0 campaign.build_economy=1 campaign.nk_predict_every=8 trunk_beta1=0.9 fixed_mix=true real_share=0.6 league=0.5,0.5,0.0 real_keep=0.12 battle_length_scale=30 campaign.game_seconds=4800 campaign.eval_seconds=1200 campaign.no_town_factor=0.1 campaign.no_town_from=-100 campaign.loop_charge=0 campaign.exchange_close=true campaign.exchange_actions=1 campaign.take_morale=0.5 campaign.take_best_leads=true campaign.value_lr_factor=2 campaign.loop_charge_step=0.03 campaign.loop_charge_from=0 campaign.update_decisions_cap=135000 campaign.update_games_floor=9 campaign.window_balance=true campaign.phi_hero_base=825 campaign.phi_armyless_share=1.0 campaign.phi_lead_weight=0.5 campaign.phi_stock_floor=0.333333 campaign.phi_from=0 campaign.phi_ramp=1 campaign.unit_own_form=true campaign.give_all_exact=true campaign.foresight_minibatch_counts=true campaign.foresight_outcome_factor=0.526 campaign.foresight_answer_factor=0.735 campaign.foresight_answer_score_factor=0.058 \
	>> runtime/train/joint10.log 2>&1 < /dev/null
