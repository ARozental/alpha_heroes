# Battle model: why it is flat, and the plan (2026-11-05)

## Decided and built (2026-11-05, after review)

| Item | Decision | Why |
|---|---|---|
| 1. Recipe | accumulate 24 → 2 (~48 steps an update); **LR stays 1e-5**; beta2 stays 0.99 | The user's rule: never raise the rate to speed learning. With Adam each step moves ~lr, so smaller steps give ~3.5x the movement at the same per-update signal-to-noise. The predicted update KL is ~6e-4, inside the config's own 1e-4..1e-3 aim. beta2 0.999 would keep the 6,144-step second moments for ~20 updates (2-3x oversized steps). The GPU cost is unchanged: the same minibatch passes, only more optimizer steps (not +10 s). |
| 2. Credit 50:50 | outcome_share = outcome_share_max = 0.5 (the defaults now) | The user's decision. |
| 3. Lopsided battles | Advantages × total / (2·harmonic mean) = 1/(4p(1−p)): 1 for an even fight, ×3 at 10:1, capped at ×12.8 (~50:1). The share weight (example_weight) is kept. **No absolute stake weight; no real_keep by stake** | min(own, enemy)/10k would cut every early-game battle (both sides < 10k) to 0.1–0.5, and those are often decisive for a small empire. The cap limits how much the value head's errors are multiplied where the reward moves by ~0.01. |
| 4. Shooters | Type features full_shot, half_shot_only. DEFEND under full_shot starts at −2 (BattleAI's rule as a prior); the rest start at 0. Old checkpoints and snapshots migrate on load. | The user's learnable penalty. |
| Entropy floor | **Not turned on** | losses.py, 2026-10-29: the floor was "the one steady push against shooting" (gradient study on 3078: confident shots are the decisions below the floor). It would fight item 4. |
| Shooter foresight head | Deferred | Item 4 targets the same behaviour directly; revisit if idle shots stay above ~1%. |
| 5. Stalls | Stalled clients' folders kept (newest 4 a process). On Linux, headless.sh writes each thread's state, wchan and fds (no `sample` on the PC: no stall sample was ever written there). | 205 generated battles and eval 8347's 17 stalled with nothing kept. |
| Bug found | The adaptive outcome share never reached the pool's workers: they used the config's floor while the trainer logged the adapted share. Fixed: Supply.publish sends the update's credit (Shared.credit). | — |

Caveat on the diagnosis below: the eval was already flat at +0.09 under the 256-decision-step recipe (2026-10-30/31), with an update KL of ~3e-3 and a signal ratio of 0.11–0.12. The recipe froze the policy, but it was not the only reason the eval is flat. Items 2–3 (the signal) are the real bets; item 1 only lets any change show up. With 512-decision steps the signal ratio will read ~0.15–0.2 (each step is noisier); that is not a regression.

The battle eval has read +0.088 to +0.097 for 53 evals (spread ±0.004) over joint7's 1,087 battle updates (4.5 days).
Measured on the PC (TensorBoard `tensorboard/run`, joint7.log, config.txt):

| | v15-window (2026-10-26, learned +0.025 → +0.058 in 11 h) | joint7 (now) |
|---|---|---|
| learning rate | 1e-4 | 1e-5 |
| optimizer steps an update | ~25 (256-decision steps, 2 passes) | 4 (accumulate 24: 6,144-decision steps, 4 passes) |
| updates an hour | ~180 | 4.6 (waiting for real battles 777 s of a 791 s update) |
| update KL | ~3e-3 (the 256-step runs) | 5e-5 median, 3.3e-4 max; target 0.015 |
| clip fraction | | 0.05% |
| signal ratio | | 0.25 (noise floor ~0.155; the config's aim was 0.3+) |
| GPU time | | 7–10 s an update: ~1.3% |

The rtx5090 config (2026-10-31) aimed at an update KL of 1e-4 to 1e-3 with bigger steps at a lower rate; it reached
5e-5. Per hour the policy moves about a thousand times less than in the run that learned. The value heads are fine
(EV 0.55 attacking, 0.41 defending; value excess over label entropy 0.005). The eval is not too coarse (~600 battles
an opponent, total standard error ~0.013). There is headroom: neutrals +0.02, shooter_guard +0.01 to +0.05,
one_big_stack +0.04 to +0.08; shooters idle at 1.4–3.1% of their full-damage chances in greedy evals, 2–3% in
training; entropy drifted 1.02 → 0.93 and a third of the decisions are under the 0.3 floor, whose weight is 0.

Three signal problems sit behind the recipe:
- **60% of each update is lopsided campaign battles**: attack wins 84%, the defender retreats in 70–85% of its
  battles. Their outcome is set by the armies; their reward is divided by both sides' assets (100k + 1k), so what
  the actions could change (about the smaller side's value) is 1/100 of an even fight's unit; the per-source
  normalisation then rescales the real group to unit variance, and the example weight (0.05 at ≤ 0.5% share) damps it.
  Net: the 60% is near-silent noise.
- **Credit is 85% a two-round window** (outcome_share at its 0.15 floor the whole run, EV ≥ 0.4).
- **No exploration pressure**: entropy weight and floor weight 0.

The user (2026-11-05): no more compute for battles now; only the important changes. All of the below fit in the
present ~13 minutes between updates; none adds game CPU.

## The plan, in order

### 1. The recipe (zero compute, zero risk): many small steps again
`accumulate=2` (512-decision steps: ~12 a pass, ~48 an update), `learning_rate=3e-5` (the user's cap), `adam_beta2=0.999`
(the 0.99 was for 4 steps an update). Cost: steps 7–10 s → ~15–20 s an update at the same update rate (≤ 3% GPU).
Check after 20 updates: update KL in 1e-3 to 1e-2, signal ratio ≥ 0.3, clip fraction 1–5%; else the change did not
take. v15-window ran 1e-4 under the same KL control without instability (2e-4 was the unstable one).

### 2. Credit 50:50 (the user's decision)
`advantage.outcome_share=0.5 advantage.outcome_share_max=0.5` (fixed, no longer following EV). Watch signal ratio:
half the credit is now the final outcome, which is noisier than the window; the value EV at 0.4–0.55 carries it.

### 3. Lopsided battles: scale the advantage by what the actions could change, weight the battle by its stake
Two things are conflated in one denominator today. Separate them; the reward, the value heads and the eval stay in
the total-assets unit (their history stays comparable).
- **Advantage scale** (data.battle_examples, before supply.assemble's normalisation): multiply each battle's advantages
  by `(own + enemy) / H`, `H = 2·own·enemy / (own + enemy)`, the harmonic mean of the two sides' start assets
  (summary.own_start_value, enemy_start_value: army + hero levels and artifacts + mana, so a hero's spell damage
  potential and its own worth count). H ≈ 2·min when lopsided, = total/2 when even: an even fight is unchanged (×2), a
  100k vs 1k fight is measured in its 1k unit, where losing 500 gold of the big army is −0.5, as it should be. Not the
  geometric mean (10k for 100k vs 1k: still 10× too small), not min alone (the harmonic mean is min smoothed, no kink).
  No clipping: a 100k side that loses 6k to 1k of monsters is −6 in that unit, and should be.
- **Weight** (example_weight, real battles only): the stake, `clip(min(own, enemy) / STAKE_FULL, 0.05, 1)`, STAKE_FULL
  ≈ 10k gold (a week-2 army); generated battles keep weight 1 (they are the curriculum). This replaces the share-based
  weight for real battles: a 50k hero against 20k of monsters is lopsided (29% share, weight 1 today too) but it is also
  high stakes; 100k vs 1k is weight 0.05 either way, now with a clean signal instead of noise.
- **real_keep by stake** (supply._battles): keep every real battle with min assets ≥ STAKE_FULL/2, sample the rest to
  the 12% budget. Same number of records, the informative ones.
- Measure: advantage_std of the real group, signal ratio, the real set (+0.045) and neutrals (+0.02) eval scores.

### 4. Shooters and exploration
- **Type features** (features.TYPE_FEATURES, model.type_rules: a zero-init linear over them, so this is the user's
  learnable penalty in the model's own pattern): `best_shot_full` (a legal target at full damage: within
  SHOOTING_PENALTY_DISTANCE or no distance penalty), `best_shot_half` (only penalised shots), `shots_left`. Start the
  rule weight of DEFEND under best_shot_full at −2 (BattleAI's rule: it shoots when it can; a known AI's rule is a
  prior, learnable from there), WAIT and MOVE at 0 (a wait that still shoots this round is fine; a move that escapes
  melee is fine: train.failure_counts).
- **Entropy floor on**: `ppo.entropy_floor_weight=0.01` (the campaign's pattern: pushes only decisions under the floor,
  33% of them now), not a global bonus.
- **The user's head** (a foresight part, no effect on the value): for the acting shooter with shots left, what it finds
  at its next activation: {full shot, half shot, no shot (blocked or no target), dead or battle over}. Cross-entropy
  beside dies/lost/dealt/hex_enemy at foresight_weight; the label from the record's later decision of the same unit
  (full_shot's logic on its inputs). Teaches the trunk range and blocking; cheap (one label per shooter decision).

### 5. The eval's MMAI stalls
Eval 8347 dropped 17 MMAI battles (maps 235–267, "no log output for 60 s", killed). Find stall_sample.txt, fix, and
treat an eval with dropped battles as invalid until then.

## Scanned and not now
- A teacher for battles (BattleAI's advice, as NK2's for the campaign): the model is already +0.1 above it; later, if
  ever, adaptive.
- Loss shares / head variance weights (the campaign's): two loss terms here, the value heads fit; no.
- Campaign context as inputs (the battle's stake to the empire, the mana price near a Magic Well, the hero's role for
  retreat decisions): the reward is battle-local by design; revisit when hero deaths are traced to battle decisions
  rather than map ones.
- More generated actors or scenario variety: CPU, which the campaign games own.
- Model width/depth: not the limit while the policy does not move.
