# The campaign optimisation: what to change, in what order, and how to know each step worked (2026-11-06)

For the implementer. Every step names the code it touches, the config flag it adds, the metric that must move, by how
much, within how many updates, and what to do when it does not. Do the steps one at a time, in this order, and change
nothing else meanwhile (no Φ, no schema, no priors): the whole point is that each step's effect is read on its own.

## Why (measured on joint7, 2026-11-05/06)

- Adam gives a noise-dominated parameter a random walk of fixed size: the step is lr × m̂/√v̂, and for pure noise that
  ratio is about √((1−β1)/(1+β1)) = 0.23 at β1 0.9 whatever the signal. The campaign's signal ratio (median |m̂|/√v̂)
  is 0.17 to 0.26: most parameters move mostly at random, at full size.
- Loss weights do nothing for a head's own parameters under Adam (scale invariance); they only change the shared
  trunk's gradient mix. The value term's weight of ~350 never made the value heads learn faster.
- The loss shares balance the trunk's terms by gradient norm. The measured gradient noise scales (`campaign_noise/*`,
  the batch at which a term's noise equals its signal) are 5 to 700 decisions for the value terms and 300 to 6,000,
  often unmeasurable, for the policy, at 4,096 decisions a step. Balancing norms gives the policy's mostly-noise
  gradient 20% of the trunk and the teacher 60%, the one clean term 15%.
- The critic is on the slow timescale: the same lr as the policy, 15% of the trunk, a trunk the teacher owns, and an
  actor whose KL budget is 0.015 × (1 + 0.6/0.2) = 0.06. Two-timescale actor-critic needs the opposite. Measured: a
  label change on 11-05 took ~200 updates to re-fit; the readouts' rms is 0.006–0.012 after 1,300 updates; trust fell
  0.40 → 0.25 and the credit went three quarters Monte Carlo.
- Three adaptive controllers share one time constant (the loss shares' EMA 0.9 with ×1.25 steps, trust's ~10-update
  pooling, the KL control), under 19 restarts in 19.5 hours. The run never left its transient.
- The gradient norm of ~10 against a clip of 0.5: with Adam an always-binding clip is a harmless per-step normalisation,
  but the norm is made by the prior corrections (×10 by CORRECTION_STEP) and the value heads' private parameters
  (×350 by the balancer), so their spikes shrink the trunk's gradient on that step.

## Ground rules

- **Freeze everything else** for the whole sequence: no Φ, schema, prior or accept-set changes. Each step gets its
  window before the next starts. If the freeze is broken, the step's reading is void; start its window again.
- **Windows and noise.** Metrics per update are noisy; read means over the window against the mean of the 60 updates
  before the step. The campaign eval has a standard error of ~0.05 per eval: a step is judged by the training metrics
  named, and the eval only has to stay within one standard error over its window (3 evals).
- **The one metric for the value fit** is `eval/campaign_value_unexplained` (the share heads' fresh loss over a
  constant guess; 1 = no fit). `eval/campaign_value_loss` swings with the batch's label spread (correlation 0.93 to
  0.97 with spread²) and must not be used to judge anything.
- **Deploy** as always through `runtime/drain_restart.sh` with DEPLOY; every step keeps checkpoints loadable (the
  `ADDED` / `load_optimizer` paths) and leaves outputs bit-identical at deploy unless the step says otherwise.
- **If everything is deployed at once** and the result is bad, bisect in reverse order: remove step 7, then 6, … each
  for 60 updates. The expected signature of "all at once" is given at the end.

## Step 0: measure, change nothing (1 deploy, 20 updates)

Add to `train.ppo_update`, before `clip_grad_norm_` (train.py:1063), the gradient norm per parameter group, and log
them as `campaign/grad_norm_<group>`:
- `trunk`: everything not in the other groups;
- `value`: `value_hidden`, `value_heads`, `growth_heads`, `end_head` (model_v2.py:377–380);
- `corrections`: `model.prior_corrections()`;
- `embeddings`: every `small_embedding` table (encoders, tables);
- `type`: `type_bias` and the offers' learned parts.
Also log `campaign/value_readout_rms` (the std of `value_heads.weight`, `growth_heads.weight`, `end_head.weight`) and,
per value head, the fresh excess loss (`fresh["share"]`, `fresh["growth"]`, `fresh["end"]` per column, already
computed in the prepass) as `campaign_heads/fresh_<head>`.
Expected: `corrections` and `value` make most of the total; `trunk` is around 0.1 to 0.5. These numbers set step 6
and are the baselines for steps 1 and 7. Battle side: the same per group (`trunk`, `values`, `priors`).

## Step 1: the critic on its own timescale (lr ×10)

**Change.** `train.make_optimizer` (train.py:296–304): a third parameter group, the value heads' parameters
(`value_hidden`, `value_heads`, `growth_heads`, `end_head`), at `lr = config.learning_rate × config.value_lr_factor`,
new config field `value_lr_factor: float = 10.0`, weight decay 0, betas (0.9, 0.999). The KL control's lr scaling
(train.py:1294–1298) multiplies every group's lr by `lr_scale`: keep that, so the factor is relative.
`train.load_optimizer` (train.py:357–390) assumes two groups: extend it so a group absent from the checkpoint gets a
fresh state (an empty `state` for its indices, `param_groups` appended from the live optimizer's group), and a group
present keeps its moments by name as now. Checkpoints before this step load; after it they carry three groups.
The user's lr cap (3e-5) was made for the policy. The critic's group at 1e-4 is this step's premise; it needs their OK.

**Track** (window 60 updates): `eval/campaign_value_unexplained` mean falls from ~0.90 to ≤ 0.80; `campaign/head_ev_g7`
rises from ~0.27 toward 0.35 and `head_ev_vend` from ~0.29 toward 0.34 (their pre-11-05 levels); `campaign/trust`
rises above 0.33; `campaign/value_readout_rms` grows (it should at least double); the eval within one se.
**Not working:** unexplained unchanged after 100 updates → the features are the limit, not the speed: go to step 8
(the critic branch) before anything else. Unexplained swings by more than ±0.1 update to update, or `head_ev_*` falls →
the critic chases batch noise: `value_lr_factor` 3.

## Step 2: the actor's KL budget back to 0.015

**Change.** train.py:826 `kl_target = config.target_kl * (1.0 + kl_teacher)` → `kl_target = config.target_kl`; new
flag `kl_teacher_inflation: bool = False` keeping the old behaviour when true. The teacher's pull moves the policy as
much as the policy loss does; the critic's target moves either way.
**Track** (60 updates): `campaign/kl` per update mean ≤ 0.02 (was 0.018 to 0.03); `campaign/steps` falls (more early
stops are fine, that is the control working); `campaign/teacher_ce` keeps falling, just slower; `teacher_agree`
within 0.03 of before; the eval within one se.
**Not working:** `teacher_agree` falls by more than 0.05 or the eval falls two evals running → `target_kl` 0.03.

## Step 3: fixed trunk weights (remove the loss-share controller)

**Change.** learner.py `_note_gradients` (the loss-share block, learner.py:842–870): keep measuring and logging
`grad_raw_*` and `grad_shared_*`, stop moving `state["weight"]` once `self.version ≥ config.loss_shares_freeze`
(new field, set it to the version of the deploy). The frozen weights are the EMA values at that moment (value ~350,
teacher ~2.3, foresight ~0.6). From then on `teacher_share_for(score)` only logs. `b69bb20` made the value share a
floor; with the freeze that logic is inert.
Later, when the measurements of step 0 are in hand, the fixed weights may be set from signal rather than norm:
w_i ∝ (desired share_i) / (signal norm_i), with the signal norm the EMA over updates of the *mean* gradient (not the
per-update norm, which is mostly noise), re-measured at deploys only. Not before the freeze has had its window.
**Track** (60 updates): `campaign/loss_weight_*` flat; `grad_shared_value / grad_shared_policy` may drift, that is
expected; `eval/campaign_value_unexplained` and `signal_ratio` no worse than step 1's window; the eval within one se.
**Not working:** the trunk's value share drifts below 0.05 or above 0.4 of the shared norm for 30 updates → set the
value weight by hand to bring it to 0.15 and freeze again.

## Step 4: trust as a floored slow average, not a feedback variable

**Change.** `learner.trust()` (learner.py:663): `trust = max(config.trust_floor, ema)` with `ema` an exponential
average of today's value (memory 0.98, ~50 updates), new fields `trust_floor: float = 0.4`, `trust_memory: float =
0.98`. `horizon_advantages` and the share baseline keep their formulas; they just see a trust that cannot collapse
within a re-fit. The credit check (`check`, per action type) stays as is.
**Track** (60 updates): `campaign/trust` ≥ 0.4 always; `campaign/signal_ratio` rises (the advantage baseline is less
MC); `campaign_credit/*` gaps (the credit check's t-statistics per type) do not grow beyond their pre-step range;
the eval within one se.
**Not working:** a `campaign_credit/<type>` gap beyond |4| for 30 updates → the floor is trusting a critic that misprices
that type: `trust_floor` 0.3 and look at that head's fresh excess (step 0's `campaign_heads/fresh_*`).

## Step 5: longer momentum for the trunk

**Change.** In `make_optimizer`, the trunk group's betas (β1 `config.trunk_beta1`, default 0.95; β2 0.999); the value
group and the corrections keep 0.9. Adam's bias correction handles the switch; `fit_moments` needs no change.
`signal_ratio` reads each group's betas already.
**Track** (60 updates): `campaign/signal_ratio` rises from ~0.2 toward 0.3; `campaign/kl` per update does not rise;
`campaign/weight_change` (if logged; else the trunk's update norm) falls; the eval within one se.
**Not working:** `teacher_ce` or `policy_loss` trend worse for 60 updates → the trunk's target moves faster than
20 steps of averaging allow: back to 0.9. If it works, try 0.98 once, same test.

## Step 6: clip per parameter group

**Change.** Replace the single `clip_grad_norm_(model.parameters(), config.grad_clip)` (train.py:1063) with one call
per group of step 0, each at `config.grad_clip_group[name]`, set to the group's median norm from step 0 (so each clip
binds about half the time; a group's spike then shrinks only that group). Log the total pre-clip norm as now and the
share of steps each group's clip bound (`campaign/clip_bound_<group>`). With Adam this keeps the normalising behaviour
of today's always-binding clip and removes the cross-group starvation. No group's clip above 2× its median.
**Track** (40 updates): `clip_bound_trunk` between 0.3 and 0.7; `signal_ratio` not lower; the eval within one se.
**Not working:** `signal_ratio` falls → the trunk's steps got noisier without the global rescale's damping: set
`grad_clip_group["trunk"]` to its 25th percentile.

## Step 7: head weights by residual, not by label spread

**Change.** `train.head_weights` (train.py:457–480): the weight of each value head ∝ 1 / (EMA of its fresh excess
loss, step 0's `campaign_heads/fresh_<head>`, memory 0.9) instead of 1 / target variance, rescaled as now so the
heads' total loss at a constant prediction is unchanged. A head whose residual is large (long horizons, V_end) gets
less of the shared gradient; a head that is learning well gets more. This replaces `head_scale`'s per-update swing
with the label spread (share weights moved 6 to 9.8 update to update) by a slow, fit-driven weight.
**Track** (60 updates): the per-update log-std of each head's weight under 0.05; `eval/campaign_value_unexplained`
not worse; `head_ev_v2` (the cleanest head) rises.
**Not working:** nothing moves → leave it; the gain is mostly stability. If `head_ev_vend` falls below 0.25 → V_end's
weight was cut too far: floor every head's weight at a quarter of the old one.

## Step 8: only if step 1's "not working" fired: the critic branch

A value branch that reads the trunk's output **detached** plus `known.phi` and the raw Φ features, through one block of
its own (the same width as `value_hidden`, 128, GELU, one more linear), in the value group at its lr. The trunk then
gets no value gradient at all; the policy and the teacher own it, the critic owns its branch. This is the Phasic Policy
Gradient separation without the auxiliary phase. Expected: unexplained falls within 50 updates where step 1 could not
move it. Cost: one small extra block per decision.

## Battle side (same mechanism, fewer parts)

`battle_model/train.py`: Adam, one group, lr 1e-5, β2 0.99 (the PC's config), clip 0.5. Steps 0, 1 and 6 apply:
the value heads (`ValueHeads`, layers.py) as their own group at ×10 (there is no teacher or balancer, so the critic's
only handicap is the shared lr), per-group norms logged, per-group clips at their medians. The credit is a fixed 50:50
since 8797125, so there is no trust loop. Track `credit_attack/value_ev` and `credit_defence/value_ev` (now 0.55/0.41):
they should rise within 60 updates; `eval/battle_value_loss` is the unnormalised kind and swings with the batch, so
use the two EVs.

## If everything is deployed at once

Expected signature when it works: `trust` ≥ 0.4 from the first update, `kl` ≤ 0.02, `signal_ratio` 0.25 to 0.35,
`unexplained` falling to ≤ 0.8 within 60 updates, `head_ev_*` back to their pre-11-05 levels within 100, the eval mean
back to 0.40 ± 0.05 over 5 evals. If the eval falls two evals running, or `unexplained` rises, bisect: revert step 7,
then 6, 5, 4, 3, 2, each for 60 updates, until the signature returns; the last reverted step is the fault. Do not
revert step 1 first: it is the one with the theory behind it and the strongest measured deficit.

## Not changing

The optimizer family (Adam is not the problem). The KL control itself (it stays as the one controller). The teacher's
60% share for now (it is the user's choice; the critic branch of step 8 makes it irrelevant to the critic). The
clip's always-binding behaviour (with Adam it is a normalisation, and a useful one).

## Implementation (2026-11-06, deployed to joint7 at once)

The user: "implement them all (unless you see a bug or a contradiction or something bad)", tested by training on. Every
step is in, each behind its own setting (`campaign_model/config.py`, the defaults are the plan; the battle side's in
`TrainConfig`), so a bisect is a `--set` away:

| step | setting (default) | off |
|---|---|---|
| 0 | always on: `campaign/grad_norm_<group>`, `campaign/clip_bound_<group>`, `campaign/value_readout_rms`, `campaign_heads/fresh_<head>`, `campaign/signal_fraction`, `campaign_signal/<group>`, `campaign/trust_today`; battle `stability/grad_norm_<group>` etc. | n/a |
| 1 | `value_lr_factor` 10, `value_lr_kl` False | `value_lr_factor=1` |
| 2 | `kl_teacher_inflation` False | `kl_teacher_inflation=true` |
| 3 | `loss_shares_freeze` (the run script: 1420) | `0` |
| 4 | `trust_floor` 0.4, `trust_memory` 0.98 | `trust_floor=0 trust_memory=0` |
| 5 | `trunk_beta1` 0.95 | `0.9` |
| 6 | `grad_clip_by_group` True, `grad_clip_calibration` 200, `grad_clip_groups` () | `grad_clip_by_group=false` |
| 7 | `head_scale_by` "residual" | `"variance"` |

**The groups** (`train.GRAD_GROUPS`, each an Adam group of its own, named, its parameters' names in the checkpoint;
`load_optimizer` moves the moments by name, so the 2-group checkpoints of before load whole: v1418's 498 of 498
moments identical):

| campaign group | tensors | weights | rate | beta1 | decay |
|---|---|---|---|---|---|
| trunk | 404 | 2,856,849 | 1e-5 x lr_scale | 0.95 | 0 |
| embeddings | 39 | 407,136 | 1e-5 x lr_scale | 0.95 | 0 |
| type (type_bias) | 1 | 14 | 1e-5 x lr_scale | 0.95 | 0 |
| value (value_hidden, value_heads, growth_heads, end_head) | 8 | 51,975 | 1e-4 | 0.9 | 0 |
| corrections | 46 | 1,309 | 1e-5 x lr_scale | 0.9 | 5.0 |

Battle (`battle_model/train.py BATTLE_GROUPS`): trunk 199 tensors / 429,424 weights; priors (the rule weights, the
value and spell priors, the type head's bias) 17 / 767; values (ValueHeads and the strengths' direct path into them)
9 / 276 at 1e-4. ("The offers' learned parts" of step 0's "type" group are prior corrections already: they stay there.)

**Where the code differs from the plan, and why**

1. **Steps 1 and 2 contradict each other as written.** The KL control's `lr_scale` scaled every group, and its early
   stop ends the update for the critic too. Step 2 cuts the KL budget 4x (0.06 to 0.015): the control will cut
   `lr_scale`, and "keep that, so the factor is relative" would cut the critic's rate with it, the opposite of step 1.
   The value group's parameters never move the policy, so the policy's KL is no reason to slow them: its rate is
   `learning_rate x value_lr_factor`, apart from `lr_scale` (`value_lr_kl=true` restores the plan's wording). The early
   stop still ends the critic's steps with the policy's (steps of its own would cost a full forward and backward each).
2. **`signal_ratio` cannot judge steps 4 to 6.** It is the median of |m̂|/√v̂, whose value for pure noise depends on
   beta1: 0.674 x √((1−β1)/(1+β1)) = 0.155 at 0.9, 0.108 at 0.95. Step 5's "signal_ratio rises from ~0.2 toward 0.3"
   would read as a failure even if the trunk's signal doubled, and steps 4 and 6's "signal_ratio not lower" would
   misfire. Added `signal_fraction` (`battle_model/optim.py`): (mean of min(1, m̂²/v̂) − k)/(1 − k), k = (1−β1)/(1+β1),
   which is μ²/(μ²+σ²) for a gradient of mean μ and noise σ, so 0 for noise and 1 for a steady gradient at any beta1.
   Read it where the plan says signal_ratio. Baselines (v1418, β1 0.9 everywhere): campaign all 0.099, trunk 0.101,
   embeddings 0.033, type 0.064, value 0.246, corrections 0.104; battle trunk −0.008 (pure noise), priors −0.019,
   values 0.158.
3. **Step 6 as written would jolt Adam.** Clipped at their own medians, the groups' gradients would reach Adam
   ~10-50x larger than the one clip at 0.5 left them (the total norm ~10-18). Adam is blind to a constant scale, but
   not to a change of it: its second moment needs ~1,000 steps (β2 0.999) to catch up, and meanwhile the steps run up
   to ~6x their size (a 25x jump: ~6x after 20 steps, ~3x after 100). So each group is clipped at its median and then
   every group is multiplied by grad_clip / the total norm's median: the same training as the plan's, at the scale the
   moments were built at. The medians are not hand-set from a step-0 window and a second deploy: the first 200
   optimizer steps after the deploy (the one clip meanwhile) calibrate them, and the checkpoint keeps them
   (`grad_clip_groups` sets them by hand instead). From then on a group's spike shrinks only that group.
4. **Step 7's "rescaled as now" keeps the swing it is meant to remove.** The heads' total at a constant prediction is
   mostly V_end's (its targets' spread 0.106 against the share heads' 0.019-0.030), and the running variances
   (memory 0.8) move with each batch's spread: every head's weight moved with it. The total is taken from slower
   variances (memory 0.98). The residual weights are ramped in from the variance's over 20 updates, so the heads'
   weights change smoothly at the deploy (a head's weight jump is a gradient-scale jump for Adam, as in 3).

Smaller choices: trust is 0 until a first measurement of the fit (the floor needs one); the `load_optimizer` of before
let the checkpoint's group settings win (a changed `prior_decay` or beta1 never reached a resumed run), now the
config's do; a head with no grade in a batch keeps its running residual.

**Step 8 is not built.** Its trigger (unexplained unchanged after 100 updates of step 1) has not fired. One caution
for then: if the features are the limit, detaching the trunk from the value loss makes the critic's features less
value-shaped, not more; what it buys for sure is a trunk free of the value's noise (the policy's gain), so judge it by
the policy's metrics as much as by unexplained.

**What to read** (the plan's signature, with signal_fraction for signal_ratio): `trust` ≥ 0.4 from the first update;
`kl` ≤ 0.0225 (the stop line, 1.5 x 0.015) and `steps` lower; `campaign/signal_fraction` and `campaign_signal/trunk`
up from 0.10; `campaign/value_readout_rms` up from 0.014; `eval/campaign_value_unexplained` falling to ≤ 0.8 within 60
updates; `head_ev_*` back to their pre-11-05 levels within 100; the clips calibrated within ~10 updates
("the gradient clipped by group" in the log), then `clip_bound_trunk` 0.3-0.7; the eval 0.40 ± 0.05 over 5 evals.
Battle: `credit_attack/value_ev` (0.55) and `credit_defence/value_ev` (0.41) rising within 60 updates.

**First measurements (deployed 10:35, v1423-1426), and what they change**

- Step 0's expectation was wrong: the trunk makes ~88% of the gradient norm (per-update medians, v1423-1426: trunk
  13.3, corrections 1.22, value 1.03, embeddings 0.50, type 0.02; total ~14). The always-binding clip was normalising
  the trunk by its own norm, not starving it for the corrections' or the value heads' spikes. Step 6's cross-group
  starvation is small, so per-group clipping acts almost as a clip on the trunk alone at its median: expect little from
  it either way. Watch `clip_bound_<group>` after the calibration: ~0.5 while a group's clip matches its median; a
  group outside 0.2-0.8 over 10 updates has moved ~2x from it, recalibrate at the next deploy (the other session's
  point: the calibration overlaps the critic's re-fit and the head weights' ramp).
- The battle trunk's pure-noise reading has a first suspect the plan did not list: 8797125 cut `accumulate` 24 -> 2
  (each Adam step 12x fewer battles) in the same commit as the lopsided scale, and the battle `signal_ratio` moved
  exactly there: 0.25 over epochs 8086-8385 (4 steps an update, update KL 5e-5) -> 0.135 over 8386-8454 (48 steps,
  1.3e-3). 12x fewer samples a step cuts a noise-dominated gradient's signal share ~12x by itself. The ratio now sits
  under the pure-noise level (0.155 at beta1 0.9), a sign of heavy tails too, so the x12.8 lopsided scale may add to it.
  The fix to test is more averaging per direction (a larger `accumulate`, or the battle trunk's beta1 0.95 as step 5),
  after the battle's step 1/6 window (battle updates ~3 an hour).
- Step 1's "not working" rule ("unexplained swings by more than +-0.1 update to update") is inside the noise:
  unexplained's per-update std was 0.096 before the deploy. The trigger used: its std over 20 updates past 0.13, or the
  heads' fresh excess trending up, -> `value_lr_factor` 3 (not step 8).
- The readouts are not the bottleneck: at 10x the rate their rms moved 0.0141 -> 0.0144 in 4 updates (+0.5% an update,
  not the doubling a lagging readout would show).

**Step 2 reverted (2026-11-06 ~14:20, v~1468; 53c1df7, `campaign.target_kl=0.03`).** Its fallback fired: the eval fell
two evals running (0.441 at v1422, before the deploy -> 0.363 -> 0.335; each fall ~1 se, together −0.03 against the six
evals before). What step 2 did: the policy's rate fell to 0.2-0.4x of its cap within 20 updates, and the KL fell only
1.6x for that 2.5x. The update's KL varies ~5x from batch to batch at one rate and barely follows the rate (the same in
the history before, where low rates came right after high-KL batches), so with the stop line at 1.5 x target and the
raise line at 0.5 x target the control ratchets on the batch's noise: at 0.015 ~22% of updates crossed the stop line
and ~5% the raise line, the rate falling ~7% an update toward its floor; at 0.03 the odds flip and the rate sits at its
cap. This controller cannot hold an actor "somewhat slower": it gives the floor or the cap. A slower actor, if wanted,
is a lower `learning_rate` with the KL target as a safety stop. The two timescales stay through step 1 alone (the
critic at 1e-4, the actor at 1e-5). The deploy also logs `campaign/kl_first` (the first step's KL, before any step) and
`campaign/kl_growth` (a step), to tell a floor from growth.

State of the rest at v1444-1466 (against v1360-1422): unexplained 0.88 (0.90), its update-to-update std ~0.05-0.07
(0.095); head_ev_g7 0.31 (0.29), vend 0.26-0.32 (0.27), v7 0.15 (0.15); the readouts' rms 0.0141 -> 0.0159 (growing
since ~v1450); teacher_agree 0.319 (0.332), teacher_ce 0.98 (0.97); the dumb-decision rates unchanged or lower
(pickups passed at turn end 1.0%, on TRAVEL 10.9%; battle shooters not shooting 1.6% in training).

**After the revert (eval v1484, 11 updates in): 0.430 ± 0.050** (paired +0.095 ± 0.063 against v1466; Φ ours 120k ->
176k, eliminated 21 -> 12). The three regimes so far (v1360-1422 before / v1423-1473 step 2 on / v1474+ reverted):
evals 0.409 (4) / 0.349 (2) / 0.430 (1); the policy's rate 1e-5 / 3.8e-6 / 6.1e-6 rising; unexplained 0.904 (std
0.095) / 0.886 (0.069) / 0.918 (0.049); head_ev_g7 0.29 / 0.29 / 0.27, vend 0.27 / 0.26 / 0.31. Readings by step: step
2 hurt (the actor at ~0.35x on the KL's batch noise, the eval dipping for exactly that period); step 1 made the fit
smoother, not better (the critic was not speed-limited; with the actor back at full speed the fit is slightly worse,
as two timescales predict); step 7 doubled V_end's weight and its EV rose (0.27 -> 0.31) while V7's dipped; step 6
steadied the per-update gradient norm (std 3.7 -> 1.3) and little else (the trunk is ~88% of it); steps 3-5 show
nothing beyond their mechanics. The first step's KL is 0.00001 (no memory floor): the KL is the policy's own movement
over the passes, and it barely follows the rate (likely the unclipped teacher term saturating the batch's fit over 3
passes; the KL at the end of pass 1 would tell).

**The 60-update read after the revert (v1474-1533, against v1360-1422 before the deploy):** evals 0.430, 0.360, 0.394,
mean 0.395 (0.409); unexplained 0.905 (0.904), its swing 0.065 (0.095); the fresh value loss 0.00188 ± 0.00096 (0.00227
± 0.0011); head_ev_g7 0.31 (0.29), vend 0.29 (0.27), v7 0.15 (0.15); teacher_ce 0.98 (0.97), teacher_agree 0.32 (0.33);
the heads' weights' per-update log-std 0.02-0.035 (under step 7's 0.05); clip_bound_trunk 0.49. What the plan bought:
smoother training (the value loss and the fit swing ~30% less, steady head weights and gradient norms, no jumps) and
G7/V_end part of the way back; not a better fit (unexplained unchanged after 111 updates, the readouts +10%, not the
doubling a lagging critic would show). By step 1's rule that sends us to step 8; not taken: if the critic's inputs are the
limit, a trunk detached from the value loss makes them less value-shaped, and much of the share-change heads' remaining
variance is likely what the opponents do, which no input predicts. Next for the fit: measure how much of it is
predictable at all before adding capacity. Battle (6ec628f, deployed after this read): the escapes pressed down while
its value head says we are ahead, and its trunk's beta1 0.95 (step 5: its gradient read as pure noise a step).
