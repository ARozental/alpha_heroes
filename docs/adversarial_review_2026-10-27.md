# Adversarial review, 2026-10-27

Scope: everything in `battle_model/`, `agents/AlphaBattleAI/`, `scripts/`, the VCMI patches list, the two commits
that landed during the review (fc65a2e adaptive outcome share, 6c97ef5 balanced eval score) and the uncommitted
`noise_ceiling.py` / `value_benchmark.py`. The running `v15-window` process was not touched. Numbers come from its
logs (`runtime/train/20261026-1544-v15-window`, `runtime/runs/*-queue/logs`).

Ranked by what fixing it buys. **Verified** means I confirmed it in code and in run data. **Code** means
confirmed in code only. **Estimate** means an arithmetic estimate, not profiled.

| # | Severity | What | Status |
|---|---|---|---|
| 1 | Critical | After any resume, league past-version opponents load no model and only DEFEND; ~13% of generated battles | Verified |
| 2 | High | KL early stop / LR control measures against the stale behaviour policy: 40 one-step updates, LR sawtooth | Verified |
| 3 | High | Actors overproduce: CPU wasted while the learner is CPU-starved, drops unlogged, newest-first skews to short battles | Verified (size of drops: estimate) |
| 4 | Medium | `shooter_not_shooting` counts WAIT, but wait-then-shoot is a legit H3 tactic | Code |
| 5 | Medium | `wipe_chance` assumes uniform damage; VCMI averages up to 10 rolls (bell curve) | Code |
| 6 | Medium | Dense spell×hex tensors are ~55% of every record and GBs per update, almost all zeros | Estimate |
| 7 | Medium | `noise_ceiling.py`: tactics-phase prefixes break replay (crash); continuations share dice; V taken after the cut | Code |
| 8 | Medium | `value_benchmark.py`: the "separate" value net starts from the shared, policy-trained trunk | Code |
| 9 | Medium | `score_balanced` / `score_worst` are dominated by the 8-battle castle set's noise | Verified |
| 10 | Low-Med | Siege features: walk distances ignore moat/quicksand stops and the gate | Code |
| 11 | Low-Med | Summoned units and clones are worth 0 in features and engine values | Code |
| 12 | Low | Evals: ~11% of wall time, fresh VCMI + ONNX session per battle, 8 at a time; eval ONNX never checked against torch | Verified (timing) |
| 13 | Low | `real_eval.json` battles start with full mana and default gold, not the real battle's | Code |
| 14 | Low | Adaptive outcome share (fc65a2e) gauges V by pooled EV, inflated by lopsided battles | Code |
| 15 | Low | Small ones: V0 label in the tactics phase, surrender asymmetry in self-play, v0 in the league, uniform league sampling, `full_shot` edge cases | Code |

---

## 1. Critical: league opponents play "every unit defends" after any resume

- **Where:** `actors.py:61-62` registers saved snapshots on the inference server as
  `path.stem.replace("snapshot_", "v")`, which gives `v00050`. `publish()` (`actors.py:85`) and `past_versions()`
  (`actors.py:105`) use `v50`.
- **What happens:** after a resume, every pre-resume snapshot is requested under a name the server doesn't have. The
  opponent's `AlphaBattleAI::loadModel` fails (`AlphaBattleAI.cpp:153-183`), logs "cannot use model … Every unit
  defends.", and `activeStack` plays `playDefault` (DEFEND) for every unit (`AlphaBattleAI.cpp:510`). That side
  isn't recorded, so nothing reaches training. Our side's record of a battle against a do-nothing army is trained on
  as normal "generated" data.
- **Evidence:** the VCMI logs say `no model named 'v250' (the server has ['current', 'v00000', 'v00050', …`. The
  opponents' battle summaries read `decisions=0 fallbacks=22..26`. Since the r6 resume, 23 of 182 battles (12.6%)
  in the surviving queue logs were like this, which matches 15% league share × 13 of 15 snapshots misnamed.
  `v15-window` has been resumed 5+ times, so most of its life is affected, and so is `v14-11h-r2`. It also inflates
  training win rates.
- **Fix:**
  - Register snapshots as `f"v{int(path.stem.split('_')[1])}"`.
  - A model that fails to load on *any* AlphaBattleAI side should fail the battle (the actor drops it) instead of
    silently defending. `fallbacks>0` with `decisions=0` is the only trace today.
- **Also:** `snapshot_00000` (the untrained model) is a league opponent in every run. Drop v0.

## 2. High: the KL control measures the stale behaviour policy, not the update

- **Where:** `train.py` early-stop check (`float(parts["approx_kl"]) > 1.5 * config.target_kl`, now line ~858) and the
  `lr_scale` update after it. `approx_kl` is `mean(old_log_prob - new_log_prob)`, where `old_log_prob` comes from the
  policy that *played*. Records are about one version old (`record_age` median 1.05), so the previous update's
  movement is already spent before this update takes a step.
- **Evidence (v15-window, 768 updates):**
  - 24% of updates early-stopped.
  - 40 did exactly one PPO step. Their very first minibatch, before any step of their own, was over 0.0225.
  - 77 did 3 steps or fewer. Their median KL was 0.023, right at the threshold, and 54 of the 77 came right after
    a full 24-step update.
  - Every such stop also divides the LR by 1.5. That is the sawtooth in `learning_rate` (0.44× → 1× of the
    schedule).
  - A one-step update spends 3,200 decisions on a single gradient step.
- **Also:** the k1 estimator over 256 samples is noisy at this threshold and can go negative (min −0.0024 logged).
- **Fix:**
  - At update start, one no-grad forward over the update's decisions gives the anchor log-probs. It can share the
    `facts` pass.
  - Early stopping and the LR controller should use KL(anchor ‖ new), with the k3 estimator
    (`(r-1) - log r`), over a fixed subsample or the whole update.
  - Optionally, decoupled PPO (Hilton et al. 2021): clip `π/π_anchor` and weight by `π_anchor/π_behaviour`.

## 3. High: actors overproduce, drops are invisible, and selection skews to short battles

- **Evidence:**
  - `wait_seconds` median 0: the learner never waits.
  - After every `take()` the buffer still holds about one whole update (median 3,217 decisions, max 15,984 of the
    16,000 cap).
  - Chosen records are almost all age 1 (`record_age` 1.05), so the leftover is almost never used and ages out.
  - `stats["dropped_stale"]` is counted (`actors.py:253`) but never logged. Overflow drops (`actors.py:263-265`)
    aren't counted at all.
  - Meanwhile the learner is CPU-starved: the memory notes say 9.4 s an update alone vs 13-19 s in a run, and
    `train_seconds` median here is 17.5 s. CPU spent on battles that get thrown away comes out of the learner.
- **Bias:** `_select` takes the newest version first (`actors.py:286`). At take time the newest usable records are
  battles that both started and finished since the last publish, which are the short ones. In the client logs, about
  2.5% of decisions come from battles over 10 s of wall time: big late-game battles and sieges. Those are the ones
  that get skipped and then dropped. It's a small share but the most valuable part of the data.
- **Fix:**
  - Backpressure: generated actors wait while the buffer already holds one update of fresh decisions.
  - FIFO within `max_record_age`.
  - Log both drop counters.
  - Then tune `generated_actors` by learner update time; fewer actors probably means more decisions trained per
    hour.

## 4. Medium (game logic): `shooter_not_shooting` counts WAIT

- **Where:** `train.py:465` counts MOVE, WAIT or DEFEND as a failure at every decision where a full-damage shot was
  legal.
- **Why it's wrong:** a unit that WAITs is asked again at the end of the same round, so WAIT followed by SHOOT
  counts as one failure out of two decisions. Wait-then-shoot is a real H3 tactic:
  - shoot after the enemy commits (pick the stack that just came forward, or the one about to hit you);
  - shoot last in round r and first in round r+1, before the enemy can block you.
- **Fix:** score per (unit, round). It's a failure only if the unit ended the round without shooting while a full
  shot was available at one of its decisions that round. Equivalently, drop WAIT from the hit list and let the later
  decision decide.
- **Same metric, smaller:** `full_shot` (`train.py:428`) reads no-distance-penalty only from the creature table.
  - The Golden Bow and Bow of the Sharpshooter (hero `NO_DISTANCE_PENALTY`, already in
    `rules.RULE_ARTIFACT_BONUSES`) make every shot full damage, so far targets go uncounted.
  - The siege wall penalty (half damage) is ignored.

## 5. Medium (game mechanics): `wipe_chance` uses a uniform damage distribution

- **Where:** `features.py:477`, `wipe = fraction(high - hp + 1, high - low + 1)`. This is P(damage ≥ hp) for
  damage uniform on [low, high].
- **What the engine does:** VCMI deals the *average of min(10, count) uniform rolls* (`BattleInfo::getActualDamage`,
  `vcmi/lib/battle/BattleInfo.cpp:622`). `rules.py`'s own docstring says so. With 10 rolls the spread is
  sd ≈ 0.09 × range.
- **Example:** with the stack's health at 70% of the range, the feature says 0.30 but the real chance is about
  1.4%. The feature is badly miscalibrated exactly where it decides between targets.
- **Fix:** a normal approximation with sd = range/√(12n), n = min(10, count), or the exact Irwin–Hall CDF for small
  n.

## 6. Medium (engineering): dense spell×hex tensors dominate memory and transfer (estimate)

- **Per decision**, int32 after `compact`, it's about 400 KB. Of that, about 220 KB (~55%) is
  `s_hex_value/enemies/allies` (3 × 70 × 187), `m_spell_hex` and `logp_spell_hex`. These are non-zero only when our
  hero can cast a hex-target spell, which is at most once a round.
- **Per update:**
  - `collate` widens every int input back to int64 (`data.py:325`), about 2 GB for 3,200 decisions moved to the GPU.
  - The once-per-update `facts` add `spell_hex` features [3200, 70, 187, 8] float32 ≈ 1.3 GB.
  - The buffer cap of 16,000 decisions is about 6 GB of RAM.
  - The two corrupt-minibatch incidents happened in a swapping run.
- **Fix:**
  - Keep spell×hex rows only for spells with a legal hex, or only when `m_spell_hex.any()`.
  - Use uint8 for the hit counts (`*_enemies/*_allies`).
  - Keep int32 on the GPU and cast only where indexing needs int64.
  - Compute `spell_hex` facts only for rows with a legal spell.
- **First:** profile one update (collate / transfer / facts / steps) to size the gain. I didn't profile, to stay off
  the live run.

## 7. Medium: `noise_ceiling.py` will crash, and overstates the ceiling when it runs

- **Crash:**
  - `prefix_before` (`noise_ceiling.py:28-30`) includes the round-0 TACTICS_MOVE/END actions.
  - `placeNextTactic` (`AlphaBattleAI.cpp:418`) never reads the prefix, so in a battle with a tactics phase the
    placements are re-sampled.
  - The first round-1 decision is then handed a TACTICS_* prefix action, which is "not legal here". That is an error
    event, so `summarize` raises and the script dies.
  - The generator gives heroes Tactics, so 30 battles will almost surely include one. Same limit for
    `play.Game.prefix` in general.
- **Bias, outcomes:** the continuations share VCMI's dice stream after the cut (same server seed). Within-position
  variance is therefore underestimated and the ceiling overestimated. It needs reseeding at the cut; the
  `IBattleRandomizer` hook from 0d0cb2725 is the natural place.
- **Bias, V:** `value` is V at side 0's first decision of the cut round. In the original battle side 1 may already
  have moved by then, so V sees moves the continuations don't share. Use the first decision of the round, of
  either side.

## 8. Medium: `value_benchmark.py` can't show what a separate value net gains

- **Problem:** `"separate": copy.deepcopy(base)` (`value_benchmark.py:67`) starts the "separate" net from the shared
  trunk that was trained with the policy loss. Eight value-only passes from there measure "stop adding policy
  gradients", not "a value net of its own".
- **Fix:** train the separate net from initialisation (more passes, more battles), or stop policy gradients into the
  trunk in a copy.
- **Also:** its default LR is 2e-4, above the user's 1e-4 rule (it's offline, but say so if it stays).

## 9. Medium: the new balanced and worst-set eval scores (6c97ef5) chase noise

- **Set sizes differ a lot:** castle is 4 seeds × 2 sides = 8 battles; hard 16; neutrals 24; generated and real 48
  each.
- **Noise between consecutive evals:**

  | Set | Score sd vs BattleAI | Score sd vs MMAI |
  |---|---|---|
  | castle | 0.18 | 0.11 |
  | hard | 0.03 | 0.07 |
  | generated, neutrals, real | 0.008-0.036 | 0.008-0.036 |

- **Effect:** weighting each set alike puts 8 battles on par with 48, so `score_balanced` is mostly castle noise.
  `score_worst` is a min over noisy means: biased low, and mostly reports castle/hard noise.
- **Fix:**
  - Per-set standard errors, from per-battle (ours − baseline) differences.
  - Grow castle (4 → 16+ seeds) and hard.
  - Judge "worst" by a lower confidence bound.
- **Side note:** `score_real` sits at +0.01..+0.035 with sd 0.008 in every eval. The real set barely discriminates;
  most of its battles are lopsided.

## 10. Low-Medium: siege features ignore the moat and the gate

- **Where:** `features.standable` (`features.py:345`) treats only `x_access == ACCESSIBLE` as open, and
  `rules.walk_distances` has no stopping hexes. `rules.reachability` (StupidAI) handles both.
- **Effect:**
  - In sieges (17% of generated battles) walks pass through the moat and quicksand instead of stopping.
  - Defenders can't path through their own gate.
  - So `reach`, threat maps, `progress` and `new_targets` are wrong near the walls.

## 11. Low-Medium: summoned units and clones are worth 0 to the features

- `rules.unit_value`, `keptValue` (AttackValues.cpp) and `castValue` count only units kept after the battle. That
  matches the reward.
- But as "what is worth attacking" signals, a clone of 50 Archangels has `value_destroyed` 0, is left out of
  `threat_value`, and gets `net_value` ≤ 0 (its retaliation still costs). Only `threat_removed` and `danger`, which
  are damage-based, see it.
- Worth a feature that prices a unit's threat (damage per round × value per HP of what it can hit) separately from
  its keep value.

## 12. Low: evals are slow, and the eval model is never checked against the trained one

- **Cost:** an eval round is ~3.5 min (2 opponents × 87-108 s) every 30 min, about 11% of wall time, with the
  learner idle and generated actors paused.
- **Why it's slow:** `play_in_groups` starts a fresh VCMI client and ONNX session per battle, `games_per_side=8` at a
  time, on 14 cores.
- **Fix:** use `BattleProcess` and 12-way parallelism.
- **Verification gap:** training plays the torch model through the inference server, but evals play a fresh ONNX
  export that is never passed through `export.verify` on real inputs. The exporter warns about advanced indexing
  with negative indices. I checked the model's index sites and found none negative, so this is a guard, not a known
  bug: verify on a few recorded decisions per eval (milliseconds).

## 13. Low: the real eval set isn't the real battles

`scenarioJson` (`AlphaBattleAI.cpp:843`) exports neither the hero's mana nor the player's gold. Every `real_eval.json`
battle therefore starts at full mana (initHero refills) and the map's default gold. That changes spell and surrender
decisions against what really happened. Export `mana_fraction` and `gold`.

## 14. Low: what the adaptive outcome share (fc65a2e) gauges

- EV from `credit_check` pools all decisions of all battles. Lopsided battles, which are easy to predict, inflate it.
- The window credit needs something else: the quality of V's 2-round *changes* inside even battles.
- A better gauge: EV restricted to battles with `example_weight` ≥ 0.5, or EV of V's 2-round change against the
  realised one.
- Not wrong, just weakly tied to what it controls. (The running process predates this commit.)

## 15. Low, small items

- **V0 label in the tactics phase:** decisions in round 0 get the final reward as their V0 label, because
  `round_end` round 0 is skipped (`data.py:148`). It's an auxiliary head and few decisions.
- **Surrender asymmetry in self-play:** the surrendering side's record counts its payment, the other side's counts
  it as 0 (`data.py:141`). The two records of one battle aren't exact negatives.
- **League sampling:** past snapshots are drawn uniformly, so late in a run most past-opponent battles are against
  very early, weak versions. Recent-weighted or win-rate-weighted (PFSP) sampling teaches more.

## Checked and fine (don't re-review)

- **Round numbering:** `round_end` is the round that just ended (`battleNewRoundFirst` runs before `nextRound`),
  so V0..V2 labels line up.
- **Casualties:** they exclude summoned units and clones (`BattleResultProcessor.cpp:676-692`). Pit Lord demons:
  start − casualties + summoned is correct.
- **Combat constants:** damage factors, caps, morale and luck chances in `rules.py` match `config/gameConfig.json`.
- **Retaliation:** paralyzed and petrified units get `NO_RETALIATION`, so `u_retaliations` is 0 for them.
- **Eval baselines:** eval scenario seeds are unique across all sets, so no baseline keys collide.
- **Sampler and log-prob:** C++ sampler, `log_prob` and the PPO ratio are consistent. Live weight swaps mid-battle
  keep per-decision ratios right.
- **Draws:** DRAW handling (`BattleSide::NONE = -1`) and RETREAT/SURRENDER outcomes are correct.
- **Mid-battle model errors:** none in 31,634 battle summaries on disk.
