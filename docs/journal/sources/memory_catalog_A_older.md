# Memory catalog A (older half)

Length note: the digest is deliberately dense, because the catalog is meant to be queried. Sections 2–4 repeat digest facts in consolidated form, with the source file named for each.

Scope: the 55 older project-memory files (alpha-heroes-goal.md … map-size-blocks.md) in `~/.claude_personal/projects/-Users-alon-projects-alpha-heroes/memory/`, read in full on 2026-11-12, cross-checked against the hooks in `MEMORY.md`. Session codes come from the frontmatter `originSessionId`: a62ff69f = **cd** (founding), 68ab2bd0 = **b1** (main science), 31529395 = **e5** (perf/infra/deploys), df787a6f = **fork** (of b1). Files without `originSessionId` are marked "no id", with a guess from date and content marked *(inferred)*. "User" means the developer. Ids (commits, schema N, deploy N, v-numbers, runs) are copied as written. *[superseded]* / *[wrong]* marks come from later files or MEMORY.md hooks. *[unclear]* marks gaps in the source.

## 1. Per-file digest

### 1.1 alpha-heroes-goal.md
- **Session** cd · **modified** 2026-10-22 · **dates** 2026-10-21 · **type** plan/reference.
- Goal: a superhuman Heroes 3 AI. It was set in an earlier claude.ai chat that the user pasted on 2026-10-21. Stack: VCMI engine plus GOG Complete data on an Apple Silicon Mac (arm64, macOS 27).
- The original plan from that chat: imitation start from Nullkiller (VCMI's adventure AI), a horizon curriculum, a bootstrapped p(win) value, entity-token state encoding (heroes/towns/objects as tokens, creature embeddings initialized from stat tables), battles auto-resolved by a learned predictor, and reuse of vcmi-gym/MMAI for battles. Secondary angle: an alignment / instrumental-convergence study, with Freeciv/CivRealm or Unciv as the fallback.
- Why: the first decision depended on simulation throughput. The guess was 1–5 min per 100-day small-map 1v1 game per core (unmeasured).
- First milestone: run VCMI and time one headless game (`--headless --testmap`) with Nullkiller's timing logs. Speed levers to try: maxPass in `config/ai/nk2ai/nk2ai-settings.json`, one thread per game, many games in parallel.
- *Later superseded:* several parts of this plan were reversed. The NK2 imitation start was rejected on 2026-10-27 ("No NK2 teacher", campaign-value-design), then brought back as a teacher on 2026-10-31 (teacher-and-memory), then turned off again on 2026-11-11 ("NK2's choices OFF", MEMORY.md overhaul hook). Battles are played by a trained battle model, not auto-resolved by a predictor.

### 1.2 data-generation-limits.md
- **Session** cd · **modified** 2026-10-22 · **dates** 2026-10-21/22 · **type** developer rule (from an incident).
- Rule: every data-generating run has hard limits, and the AI reports MB and MB/min. VCMI games go through `scripts/headless.sh`, which gives each run its own directory and kills on a time limit, disk limit, stalled log or script exit. Never run `vcmiclient --headless` directly. After each test, check `pgrep -fl vcmiclient` and kill by PID, not by pattern.
- **Incident 2026-10-21:** four headless clients ran for ~13 h, filled the 500 GB disk and crashed Docker. Causes:
  - `pkill -f build/bin/vcmi` never matched, because the processes run as `./vcmiclient`;
  - the games could never end;
  - VCMI's default trace logging writes ~300 MB/min;
  - all runs shared one log file, which the AI kept deleting while old clients still wrote to it.
- User quote: "In the future you will not fill the disk completely like that, you will tell me how much data you create when creating data over how much time".
- Reference numbers: info-level logging ~7 MB/min per game; debug-level AI logging ~40 MB/min.

### 1.3 vcmi-setup.md
- **Session** cd · **modified** 2026-10-22 · **dates** 2026-10-21 · **type** reference + developer rules.
- User rules (2026-10-21):
  - every project file lives under `/Users/alon/projects/alpha_heroes`, with no /Applications install and no ~/Library data;
  - GitHub repos are private under ARozental.
- `vcmi/` is a submodule on branch `alpha_heroes` from tag 1.7.5. It is a private copy, not a fork, because a fork of a public repo cannot be private. Remotes: origin = ARozental/vcmi, upstream = vcmi/vcmi.
- Stock 1.7.5 crashes under `--headless`/`--testmap`/`--spectate` because `ENGINE` is null. Fixed by cherry-picking upstream fdb1aa216 and 3dfa6763f, plus a missing `#include "GameEngine.h"`.
- Own patch: the `VCMI_USER_DIR` env var (lib/VCMIDirs.cpp). `scripts/vcmi.sh` sets it to `runtime/vcmi-user` (gitignored; holds the GOG Data/Maps/Mp3).
- MMAI (the vcmi-gym battle net) ships inside upstream 1.7.5 via `ENABLE_MMAI`, so vcmi-gym's engine needs no fork.
- First data point: 4 Nullkiller2 AIs played 70 days on "Arrogance" headless in ~20 s, with trace logging on.
- AI plumbing:
  - Our AIs live in `agents/<Name>/` and build through `EXTRA_AI_DIRS` (our patch to vcmi/AI/CMakeLists.txt).
  - A new AI must be added to the enums in `vcmi/config/schemas/settings.json` and to `launcher/settingsView/csettingsview_moc.cpp`.
  - `--ai` is a no-op in 1.7.5; the AI comes from the settings.json keys `ai.adventureEnemyAI` / `combatEnemyAI` / `combatNeutralAI` / `combatAlliedAI`.
- The venv runs onnxruntime 1.31, matching the brew onnxruntime used by C++. `agents/UpLeftAI` is a test battle AI.
- AI mistake: the user wanted the map AI left as Nullkiller2, but the AI misread this once and built a map AI first.

### 1.4 no-github-actions.md
- **Session** cd · **modified** 2026-10-22 · **type** developer rule (from an incident).
- Rule: never consume GitHub Actions. User quote: "I do not want to consume github actions whatever they are". Actions is disabled on ARozental/vcmi and ARozental/alpha_heroes; Dependabot alerts and security updates are off on vcmi.
- **Incident 2026-10-22:** the pushed VCMI copy still held `.github/workflows` and `dependabot.yml`. Dependabot opened 5 PRs, each running the full Linux/Windows/macOS CI matrix. Cost: ~26k billed minutes (upper bound) against a 2,000/month quota, i.e. 100% of the included minutes.
- How to apply: strip workflow and Dependabot files from third-party code before pushing, and re-check after every upstream merge.

### 1.5 battle-model-design.md
- **Session** cd · **modified** 2026-10-22 · **type** design (user decisions) + implementation log.
- User decisions (2026-10-22):
  - **Inputs:** a dict of raw engine values only. Every transform (log, normalization, expected damage, P(wipe), retaliation) is a torch op compiled into ONNX, never C++.
  - **Entities, not hexes:** unit tokens; static abilities come from the creature-ID embedding; active spells are slots of (effect id, rounds left); Royal Griffin retaliations capped at ~3; turn order required.
  - **Heroes:** a multi-hot/concat of spells, school levels and skills feeds a dense layer, with "can cast now" kept as a separate input.
  - **Outputs:** conditional (autoregressive) heads, so every action, including spells, gets a probability. Rare actions such as catapult may be left out of v1.
- Reward (user's proposal): terminal only. ±1 for win/loss, plus enemy value destroyed / enemy value at start, plus own value remaining / own value at start, minus small penalties per round and per % of mana used.
- More decisions (2026-10-22):
  - legality masks come from the engine and are enforced by a masked softmax; whether to also feed them as input is open (ablate);
  - no siege flag (walls imply a siege); clone/summoned flags yes;
  - **board flip:** the naive x→16−x flip breaks 280 of 888 adjacencies. The parity-aware mirror keeps all 888 but maps 6 hexes off the field. Never use the naive flip.
- Values:
  - unit value = gold + resources (wood/ore 250, others 500);
  - retreat keeps the hero only if it is worth more than a fresh tavern hero; rehire costs 2500;
  - surrender cost = army gold × (1 − Diplomacy 20/40/60%);
  - all of these are fixed functions with no trainable parameters.
- Reference: MMAI v13 is a graph network over 165 hexes with a 28,114-float input and a factorized output (4 + 165 + 165), trained by PPO with 85% self-play at the end.
- **Implemented 905b16b:** `battle_model/` (contract, model, distributions, losses, reward, export, codegen, check_dumps) and `agents/AlphaBattleAI` (C++ ONNX). A contract change needs codegen plus a SCHEMA_VERSION bump. Verification:
  - 0 errors and 0 rejections over thousands of decisions; C++ and Python outputs match exactly; wrong-hash and garbage files are refused;
  - 166 walled sieges with 0 errors.
- Engine fact: for unit actions, the server marks the unit as acted before validation, so a rejection loses the turn. User rule: "We need to take the first viable action". AlphaBattleAI therefore checks masks before sending and re-chooses after a rejection. Verified with ALPHA_BATTLE_TEST_FAULT: 369/369 refusals recovered, 310/310 corruptions caught. Our surrender mask (gold ≥ cost + 2500) is stricter than the server's.
- Data plan:
  - compact records, densified by the loader;
  - sparse masks (~57 legal of 32,800 per decision, ~115 B);
  - 3,341 raw ints (~1–2 KB compressed);
  - store the action, old log-prob and value, not model outputs.
- Battle source: VCMI has no "start one battle" mode, so the plan is generated battle maps (.vmap is zipped JSON) plus a seeded eval set.
- **Value heads (contract v2, user):**
  - V0..V2 = the reward formula at the end of this / next / following round; V = end of battle;
  - each head adds the previous head's detached logit ("boosted");
  - since v3 every head is a sigmoid, mapped to reward units −1.25..3.0;
  - BCE everywhere. User rule: every probability or 0..1 quantity is a sigmoid/softmax trained with CE;
  - only V is used for advantages, with the final reward (not GAE returns) as its label;
  - the model is reward-agnostic "for now only": a per-battle mana price input is expected.
- AlphaZero-style search was planned as a later upgrade; it needs HypotheticBattle state copies and stochastic MuZero ideas.
- First experiment, `castle_vs_inferno.json`: our model vs BattleAI, Castle attacks, the seed fixed per side, ~1.7 s/epoch. Goal: "lose in epoch 1, overfit and win by epoch 1000". Fork patches: `VCMI_BATTLE_AI_<COLOR>`, and battle-only games end once the battle is decided.
- *Later superseded:*
  - the chained V was unchained 2026-10-23 (castle-vs-inferno-findings);
  - GAE was replaced by the round window 2026-10-26 (v15-history);
  - the user later rejected search ("User: no search/simulation", plan-2026-10-28, 2026-10-29).

### 1.6 autonomous-experiment-priorities.md
- **Session** cd · **modified** 2026-10-23 · **dates** night 2026-10-22 · **type** developer rule.
- When unattended, results rank: (1) bug fixes (best; "they expect there are many"); (2) a dumb-in-hindsight design mistake; (3) tuning until it wins (last, but OK).
- Look for bugs in new areas before and after each experiment. No experiment runs over 1000 epochs. Have a best solution ready when the user returns. Report bugs, design changes and tuning separately.
- Why: a correct, understood setup matters more than a tuned result that hides problems.

### 1.7 castle-vs-inferno-findings.md
- **Session** cd · **modified** 2026-10-23 · **dates** 2026-10-22/23 · **type** measurement log.
- Scenario switch 2026-10-23 ~19:00: `castle_vs_inferno.json` became the upgraded armies (Archangels resurrect, Pit Lords summon); the old one is `_basic`. Creature spells became castable (contract v9) at the same time. Results do not compare across the switch.
- Pending at the time: `up-ppo` vs `up-mcts16` (Gumbel search, 16 sims, CE to the search policy), 400 epochs. Untrained self-play: the attacker wins 8/8.
- **Failures (basic scenario):**
  - Adam lr 3e-4 or 1e-4 with 1 battle per side: attack wins peaked at ~55–70%, then were forgotten; the defence never won;
  - self-imitation alone: the defence stayed at 0% (it locks onto the best losing line, entropy collapses).
- **Fixes that mattered:** per-side advantage normalisation (joint normalisation coupled the two sides' luck); 8 battles per side; give-up prior −6. Pure PPO then reached attack 98%, defence 83% in ~500–800 epochs.
- Go-Explore (`explore_from_best=3`) plus self-imitation (weight 1.0) solved both sides in ~100 epochs. *[superseded]* Both were removed 2026-10-23: they only work with fixed seeds, and the user rejects fixes that can't work in real training.
- Battles are fully deterministic given the VCMI seed and our actions. *[later partly wrong:* BattleAI's RNG is time-seeded (review-2026-10-27), and VCMI area-attack target order was nondeterministic until vcmi e1be37e4a (perf-2026-10-29)*]*.
- Generalisation: trained on one seed, the attack won 16/18 unseen seeds and the defence 0/18 (it memorised one line).
- Scenario bug: runs before 2026-10-23 01:40 had heroes with default skills (Caitlin 62 mana, Calid knew Haste).
- **Self-play plateau:**
  - cause: V chained onto V0..V2, so it inherited their swings with our surviving army;
  - effect: the defence's V read −0.57 per fraction of army lost and −0.21 for killing (an honest estimate is ~0 and ~+0.47), and GAE taught passivity (42% wait/defend);
  - fix: `chain_final_value=False`, with V0..V2 auxiliary only; `credit_<side>/*` metrics added.
- After the A/B (run B, epoch 1000, random seeds):
  - BattleAI vs BattleAI: the attacker wins 24/24, so the scenario favours the attacker;
  - the paired metric was adopted: greedy on 16 seeds, minus BattleAI's own result on the same seed and side;
  - our defence +0.11 vs BattleAI, our attack −0.08 (it keeps only 10–30% of its army);
  - search hurt vs BattleAI (attack +0.62 → +0.28; 128 sims worse), so the user accepted greedy eval;
  - the two sides' gradients: policy cosine +0.22, value cosine −0.60.
- 400-epoch A/B/C/D (attack/defence, epochs 300–400): base −0.24/+0.01; randomseeds −0.18/+0.05; **v2adv (policy_heads 0,0,1,1) −0.09/+0.10** (best); lowentropy 0.003 −0.51/+0.09 (a dead end). Per-side V bias ≤ 0.03, so per-side value heads were rejected.
- Pending: sp-base-s1, sp-v2adv-s1/s2, sp-allheads, stopped by the user at epochs 46–50.
- *Later superseded:* policy_heads became (0,0,0,1) on 2026-10-28 (plan-2026-10-28 round 6).

### 1.8 model-iteration-loop.md
- **Session** cd · **modified** 2026-10-24 · **dates** 2026-10-24/25 · **type** measurement log.
- User goal: 400-epoch rounds to beat `runtime/train/20261023-1902-up-ppo` (v2). Measure: greedy vs BattleAI on 16 seeds, ours minus BattleAI's. Target over evals 200–400: attack −0.031, defence +0.170, mean +0.069.
- User constraints: no growth in code or parameters without a reason (v3 was cut from +80% to +12% params); bug fixes don't count as complexity.
- Rounds (mean vs baseline, evals 200–400):

  | Round | Change | Result |
  |---|---|---|
  | R1 | v3 without strength | fixed +0.042 / random +0.020 |
  | R2 | v3 + Lanchester strength | +0.036 / +0.081 (defence +0.186) |
  | R4 | zero-sum reward (v3s), metric = total_score; up-ppo backfilled +0.074 | `up-zs` +0.112, `up-zs-rs` +0.094 (ends +0.138); all gain in defence (killed 0.72 vs BattleAI 0.28) |
  | R5 | linear LR | `up-zs-rs-lin` +0.118 (best); fixed-seed `up-zs-lin` +0.026 (shooters moved; lost 16/16 at evals 160–180) → random seeds only |
  | R6 | seed-2 replicate | +0.104 vs +0.118 → run noise ±0.01 |

- R6 user quote, rejecting an epoch-size test: "this is still a sanity check for a model architecture, we are looking for bugs, issues, inefficiencies, missing/needed features".
- **Bug (2026-10-24):** MPS kept a compiled graph per batch size, so trainers grew to 9+ GB and swapped (the earlier kills). Fixed with fixed GPU shapes (d6df406): ~5 GB, flat.
- **Contract v11** (5e40c78, engine 0d0cb2725, user-requested):
  - engine-played attack values (AttackValues.cpp, expected dice);
  - engine p_* damage estimates;
  - retreat/surrender/score-now rewards into the type logits;
  - a 68-ability bag; spell target value; multi-scale features;
  - hero_type and obstacle embeddings removed; 286.6k params.
- **R7 regressed** to +0.052/+0.027, all in defence. Ablation: score_now (the current material zero-sum) in the battle token taught the defence to run. **Lesson: never feed the model the current score as a feature.**
- **v12** (6cba37e, a75640d):
  - hero value 2500×(L−3)^1.3 plus artifacts; the winner takes the loser's artifacts;
  - hit counts; neighbour features;
  - escape rewards only on the RETREAT/SURRENDER logits;
  - escape prior −6 lifted by sigmoid((log(1/25) − z)/0.5). User quote: "batch 0: do not surrender, but if much weaker consider, do not force" (a 3× threshold had a third of lost battles escaping).
- R8 v12: +0.099/+0.102 (regression fixed, no gain); 18/32 eval battles score differently (hero value).
- R9: `up-v12` resumed 400→1000. Windows: +0.122, +0.116, +0.135 (the best then; defence +0.26).
- OOD eval (40 generated, seed 200): epoch 400 −0.091, 1000 −0.081 (noise ~0.04); neutrals worst (−0.14). OOD baseline up-v12 on 32 generated: −0.126.
- **Bugs found 2026-10-24 evening:**
  - arrow towers (off-field hex −2..−4) threw in encodePairDamage, so StupidAI played all sieges (fd41244);
  - VCMI's battle-only mode applied the random starting bonus (1/3 of games got a treasure artifact), which explains the 18/32 (vcmi 43fe4c2d2);
  - h_spells ignored tomes/scrolls and mastery came from school skills, fixed by contract v13 (f84d9a0);
  - the battle-only end patch ended neutral games early (vcmi e07707607).
- **R10 `gen-v13`:** trained on generated battles, 4000 epochs, ~7 s/epoch. Evals ran 0 −0.163 → 1000 +0.088 → 4000 +0.089. Fresh 40 battles (seed 300): epoch 1000 +0.064, 2000 +0.081, 4000 +0.098, against the castle-trained up-v12's −0.048.
  - Three hangs, each fixed in VCMI: the winner levels up and nothing ends the game (end at BattleResultAccepted); arrow towers ending a siege inside onHeroVisit (flag set in startBattle, b1e4844); a StupidAI siege stalemate after ROUND_CAP (AlphaBattleAI retreats from round 50, 66b012f).
  - ~40% of self-play battles were retreats (hero assets worth about as much as armies); retreats from winnable fights were the main eval loss.
- Findings:
  - every defence plateaus at +0.1–0.2; a loss's ceiling is −0.41 (ours −0.58, BattleAI −0.75);
  - credit metrics near 0 late in training are not a bug;
  - v2 learned Hypnotize only after epoch 400.

### 1.9 v14-design.md
- **Session** no id (cd *(inferred: dates 10-25/26, battle work)*) · **dates** 2026-10-25 → 26 · **type** design + incident log.
- User approval 2026-10-25: "make all the changes ... smoke test". Built 99dcb5a, 08aa1ad.
- **Contract v14:**
  - 14 units per side;
  - tactics, catapult and First Aid Tent played by the model;
  - no StupidAI fallback (an error is counted and the unit defends);
  - a 48-slot rule-artifact bag mapped to 4 dims, zero-init (the user's "tiny edge-case path");
  - exposure features (loss/kill bounds before the next turn, best stack share, enemy spell threat).
- Reward:
  - retreat pays the 2,500 rehire, floored at 0 (TODO: negative values are real);
  - experience stays out. User's reason: it would make the reward asymmetric, and map gains aren't counted either;
  - example weights = the smaller side's asset share / 0.1, clamped 0.05..1.
- Data:
  - `actors.py` feeds a buffer: real_share real data, a cap of game_share = 1/8 per real game, records at most 3 versions old, each used once;
  - **league (user, 2026-10-25):** train only against our own model, 85% current self-play + 15% past snapshots (every 50 versions), no fixed AIs;
  - `real_games.py`: H3 maps with Nullkiller2 on the map, every battle (neutrals too) by our model, ~2 battle records/s per game.
- Generator: tactics, terrain patches, gold/mana via `VCMI_BATTLE_GOLD/MANA_<COLOR>` (vcmi fa11e3899), neutral stacks at 2%, 3% tails.
- Evals: generated, neutrals, real_eval.json (40), castle; opponents BattleAI (total_score) and MMAI. User's 2026-10-25 change of mind: eval both, train only vs our own model. Failure metrics: escape while V>0, shooter idle, worthless cast, mana left on a loss.
- **R11 `v14-11h`** (user: 11 h self-play, an eval every 30 min at ~10% of time, "must beat the other AIs with no cheating"):
  - recipe: 6 generated + 5 real-game actors, real_share 0.5, 800 decisions/update, 2 passes, LR 1e-4 linear by clock;
  - smoke fixes: a fixed model version in real games; the per-game cap blocked self-play pairs; a NK2 HeroPtr crash on retreat; NK2 CPU starvation (nice); preexec_fn hung the trainer.
- **Incident, MMAI silently off (found 2026-10-25 23:20; fix 26b150c/4e22867):**
  - parallel map builds raced on `config/modSettings.json` (VCMI truncates and rewrites it at every start), read it empty and saved vcmi+core;
  - MMAI then silently fell back to BattleAI. Eval 545's MMAI half was really BattleAI;
  - fix: per-process map-builder dirs, and evals refuse MMAI when the mod is inactive; the live run's stopgap was chmod 444.
- **Real-game crashes (fixed 2026-10-26):**
  - exit 139 = a NK2 data race (async heroGotLevel vs the unlocked parallel planner), fixed in vcmi 400b76421. Lesson: read `~/Library/Logs/DiagnosticReports/vcmiclient-*.ips` first;
  - exit 134 = "Movement terminated abnormally", fixed in d8c48ba91;
  - headless.sh: `sample` could hang for hours (now a 20 s deadline); silent-but-busy clients are no longer "stalled" (66e61a7).
- **Design mistake, entropy bonus (fixed 0fc65e0, 2026-10-26 02:00):**
  - the exact joint entropy weighted each sub-choice entropy by p(type) with gradient, paying MOVE ~0.01·ln(100) per decision;
  - shooter idle rose 0.03 (update 877) → 0.53 (1242). A replay of model_00850 vs model_01200 proved it was learned;
  - fix: detached parent probabilities, with the test `test_entropy_bonus_does_not_favour_types_with_many_options`. Resumed at update 1346.
- Known weakness (2026-10-26 03:45): the neutrals set is negative vs both opponents. A strong hero attacking 635 skeletons kites (60–130 MOVEs, 2 melee) and retreats (−0.34) where BattleAI wins (+0.11). The probable gap is the "one stack eats the single retaliation" trick.

### 1.10 review-2026-10-27.md
- **Session** no id (cd *(inferred)*) · **dates** 2026-10-27 · **type** review / incident.
- An adversarial review (docs/adversarial_review_2026-10-27.md, untracked), verified against logs and code.
- **Fixed in 3263898**, live in v15-window r7 (resumed at epoch 860):
  - #1 league names (v50, not v00050), plus an `ALPHA_BATTLE_REQUIRE_MODEL` abort;
  - #2 KL measured against the update's start (k3);
  - #3 actor backpressure (oldest-first, drop counters);
  - #4 idle shooters counted per (unit, round);
  - #5 wipe_chance as a normal approximation of VCMI's 10-roll averaging;
  - #9/#12 evals on 12 persistent BattleProcess workers plus ONNX verify and `score_<set>_se`; castle 4→16 seeds, hard 8→16.
- Pushed back on: the claim that v14-11h was hit by #1 (it used ONNX files), and judging the worst set by a lower confidence bound.
- Found: evals are not deterministic even with fresh processes, because BattleAI's RNG is time-seeded (CRandomGenerator::resetSeed). *[superseded 2026-10-29: BattleAI seeded by battle state, 198/200 deterministic (plan-2026-10-28)]*.
- Deferred: #6 sparse spells, #7/#8 bugs in the untracked noise_ceiling.py/value_benchmark.py, #10 siege moat/gate, #11 summoned value, #13 mana/gold in real_eval, #14 EV gauge, league recency.
- **Endless battles (2026-10-27 03:40):** the run stopped at 1241. We defended as 7 Sprites flying between two hexes while BattleAI's last 2 Gorgons chased forever; VCMI has no round limit and BattleAI never retreats. b3de8c8: evals skip such battles. Open: a defender stalemate rule. Training drops these battles, so there is no signal against stalling.

### 1.11 generator-status.md
- **Session** no id (cd *(inferred)*) · **dates** 2026-10-24, 2026-10-27 · **type** design/status.
- User ask (2026-10-24 01:22): a `generator.toml` distribution of random battles, for OOD variety (to prevent collapse on campaign battles) and generalization.
- Built: `battle_model/generator.py`, a richer vocab export, and MakeBattleMap extensions (experience, artifacts, war machines, towns with fort level, neutrals). Neutrals: one hero plus one CGCreature (vcmi e07707607).
- Decision: army counts follow weekly growth (the same weeks per stack, the first stack top tier). The earlier level-weight split gave armies of 10k peasants.
- Gaps: creature banks, boats, battlefields, partial mana, player gold, war machines/tactics by StupidAI, and UNITS_PER_SIDE=10 dropping towers. *(Several were closed by v14.)*
- User decision 2026-10-27: later, more hard-case kinds (only one_big_stack at p=0.1 then) and maybe more hero_vs_neutrals (0.40). Eval neutrals/hard vs MMAI ~0 ± 0.03–0.04.

### 1.12 campaign-value-design.md
- **Session** b1 · **modified** 2026-10-28 · **dates** 2026-10-27/28 · **type** design (user decisions) + run log.
- **User decisions (2026-10-27, two reviews; docs/campaign_ai.md §1):**
  - labels = the share of all empires' Φ at d+2, d+6, d+8; a loser is 0 after elimination; games to the end or day 28; multiplayer random maps with water and underground;
  - Φ: a creature = cost+u, no accrued growth ("the AI must learn the calendar"); hero = 2500·(L−3)^1.3 + artifacts + capped mana − 100·n² hero-count penalty. Free levels 1–3 are intended (the AI had misread this as "early XP is worthless");
  - tiles worth **1** gold. User quote: "10 is too much";
  - **counts as a human sees them** (VCMI quantity ids, exact only under Visions, no resource-pile amounts). User: NK2's exact counts are "cheating";
  - **no NK2 teacher and no starting P(win) bias on moves.** User: "Day+2 can learn quickly". NK2's calculations exist only as in-graph features;
  - campaign battles are played by **our best battle model** (v15-window checkpoint_02106, eval +0.067), which trains alongside with its own optimizer plus generated and hard examples.
- **Built (uncommitted then):**
  - `campaign_model/joint.py`: campaign battles replace NK2 real games at real_share 0.4; 8 generated actors;
  - `learner.py`: 4 self-play games, an update per ≥8192 decisions, its own Adam with KL control;
  - `evaluate.py`: 12 fixed maps vs NK2 every 60 min, score = our share at day 28 × players;
  - vcmi patch `VCMI_ADVENTURE_AI_<COLOR>`; windows only for present heroes (~1.5 ms/decision).
- Smoke measures: 28-day games ~17 s; the first update 8.6k decisions in 30 s on MPS; memory 23/24 GB (the trainer 12 GB). Untrained model 0.24–0.29 ± 0.08 vs NK2, knocked out in 6/12.
- Bug: Φ was 0 for a winner too, so a decisive game's final labels were 1/n for everyone.
- **joint1** (2026-10-27 22:40, 9 h):
  - user settings: V at d+2/d+5/d+7, a 56-day cap, small 2–4 player maps, game maps a third of the time, TYPE_PRIOR as a CSV bias;
  - user rule: fix things during the run (resume), don't wait;
  - 10 findings fixed by 23:50 (docs/campaign_ai.md §9): the engine ended games on red's loss, townless maps, the learner moved to a CPU thread, a backlog cap, eval clocks, war machines/catapult no longer valued (a hire exploit);
  - later: AlphaBattleAI retreats as the defender after 35 quiet rounds; game time caps 360 s / 240 s; 24 eval games;
  - **self-play never attacks:** 2/1223 knocked out in self-play vs ~75% against NK2; the d28 eval went 0.29 → ~0.43, flat since v80;
  - battle evals 0.067 → 0.064–0.082.
- *Later superseded:*
  - "no NK2 teacher" was reversed 2026-10-31 (teacher-and-memory) and the teacher removed again 2026-11-11;
  - labels changed to V2/V5/V7 plus V_end (plan-2026-10-28);
  - tile value became exploration 5/tile on 2026-11-03 (changes-2026-11-03).

### 1.13 v15-history.md
- **Session** cd · **modified** 2026-10-29 · **dates** 2026-10-26/27 · **type** design + measurements.
- User ask (2026-10-26): history "both efficient and improving performance". User quote: "we forget even what spells the enemy has".
- Built 1edee2c:
  - seen enemy spells, per-unit dealt/taken/kills/last target, and an event log of the 16 newest actions as EventTokens; C++ BattleHistory;
  - +4% compute; v14 checkpoints don't load.
- Not built: per-event value change (feedback-loop risk) and cross-battle memory. The AI advised against feeding past value estimates (copy risk); the user did not insist.
- **Credit:**
  - GAE λ 0.8 was the default for only 40 minutes (da86ba9 15:03 → ca3ee7b 15:43; earlier runs used 0.95);
  - then the **user's linear round window replaced GAE**: no exponential decay anywhere, and λ is read only by credit="gae". v15-window, joint1 and every later run use the window.
- **KL control:** target_kl 0.015 (early stop plus LR scale on the last minibatch). Fixed 2e-4 was unstable (KL 0.03–0.09); 1e-4 was too slow. User: "if 2e-4 is too much keep 1e-4".
- Overnight plan (user, 2026-10-26 ~23:55): continue v15-window with constant LR and an outcome share that follows V (fc65a2e: 50% at EV≤0 → 15% at EV≥0.4). The user dislikes many short TB lines (no mid-run restarts except after a crash). The user prefers a round window over per-decision decay ("a wait's return must not be decayed").
- v15-window after the review fixes (r7 from 860, r8 from 1240): TOTAL 853 +0.025 → 955 +0.034 → 1147 +0.029 → 1426 +0.043 → 1621 +0.058 (every set positive, worst +0.005) → 1718 +0.052. The 750–860 flat stretch ended once league opponents loaded their models and KL control measured the update.

### 1.14 python-owns-the-science.md
- **Session** cd · **modified** 2026-10-22 (restated 2026-10-28) · **type** developer rule.
- User quote (2026-10-22): "the stuff written in python is what I would like to care about, so all the science should be there as well as the training loop dataloader etc later". The user reads Python, not C++.
- Restated 2026-10-28 for both AIs: "I do not want c++ code to determine any behaviour, just pass on raw features, the C++ code needs to be small and clean, the pytorch graph should carry the business logic; the model should just get raw engine input and (maybe) its previous outputs." Action chains are the model's to learn (g_last_* echo, torch features, memory). NK2 habits enter as learnable, bounded torch priors.
- Still in C++ (2026-10-28 night): the map AI's no-op/futile mask (a safety net), the battle stalemate retreat (user OK'd it, "like chess's 50-move rule"), castValue's gold pricing (deferred), and the tactics unit order.
- Safety checks in C++ must never fire in normal play; any trigger is a bug.

### 1.15 war-machine-value.md
- **Session** cd · **modified** 2026-10-28 · **type** design (delegated decision).
- The user delegated ("decide what is best") and suspected war machines break zero-sum. Decided: ballista / ammo cart / first aid tent at **half price** (1250/500/375), catapult 0, arrow towers 0, as one constant for Φ and battle.
- Why half: full price brought back joint1's hero spam (a hire measured ~3600 with the catapult at 1000). At 0, a destroyed ballista is free and a bought one a pure Φ loss.
- Zero-sum holds: a defeated side loses its machines whether or not they were shot.
- VCMI facts: destroyed machines are removed (the catapult never); the winner never takes them; retreat/surrender keeps the survivors; battles end without them.
- Built that night: the WAR_MACHINE tables in Φ, summarize, rules and outcome features. Bug found: summarize took destroyed machines off the army at 2500. C++ Reward.h and attack/cast values were still 0.

### 1.16 training-infra.md
- **Session** no id (cd *(inferred)*) · **dates** 2026-10-26, 2026-10-29 · **type** reference/measurements.
- User ask: "make all the engineering improvements", with the future RTX 5090 PC (32 GB) in mind.
- **Central inference server** (`battle_model/inference.py`, Unix sockets, models by name, league v<n>) replaced an ONNX session per battle (1.1 s CPU) and an export per update (1.3 s). On CPU, several processes (2 best here); on CUDA, one. MPS is slower than CPU here and crashes with two threads.
- Speedups: walk_distances as a boolean frontier (features 2×); attention bias without parameter indexing (CPU train step 5×).
- Bugs found:
  - a stale model per real game (half the training data ~80 updates old);
  - the test-map client quit only when red won or lost (the "freezes");
  - MMAI eval randomness (mmai-settings.json seed 1, a local file).
- Persistent battle processes (user: "do it"; vcmi TestQueue): ~0.28 s client CPU per battle (was 1.1).
- **Throughput 73 → 238 decisions/s** trained on (the MEMORY.md hook says 194; the file says 238 *[unclear which is final; perf-2026-10-29 later measured 163→194 for a different setup]*). The learner is CPU-starved, not GIL-bound.
- **Restart lesson (2026-10-29):** shutdown takes 5–10 min, so loop on `kill -0 <pid>` and use no fixed timeout. A fixed 120 s wait once started a duplicate trainer on the same run dir.

### 1.17 plan-2026-10-28.md
- **Session** b1 · **modified** 2026-10-29 · **dates** 2026-10-28/29 · **type** measurement log + design rounds 3–13.
- **Battle drift cause (gradient_study.py, model 2840):**
  - policy-gradient coherence over 20×256 minibatches ~0.2, the same as noise; its effect on shooting |t|<1;
  - the entropy bonus (0.01) is coherent (0.96) and lowers p(SHOOT) (t −9 to −28); the value loss pushes the same way (t ~ −5);
  - p(SHOOT) median 0.99 at 2840 → 0.32 at 3139;
  - campaign battles diluted the signal (cosine −0.06);
  - LR, batch, Adam betas and Muon don't change the bias/signal ratio. The user had bet on them; the AI pushed back with data.
- Campaign credit: a window/TD credit measured worse than MC (SNR −0.20 vs +0.23), because day-level labels blind the critic within a day. Keep MC. Only 13% of moves go to sure gains.
- Battle records were 0.4 MB/decision (a 16k buffer ≈ 6 GB). Fixed by B5 compact records (13× smaller).
- **User review → built (B1–B5, rounds 3–12):**
  - battle: B1 entropy floor, B2 per-(side, source) normalisation, B3 a weight average; the user: greedy shooting should be 0.99+ and shooter-idle > 0.01 prints FAILURE;
  - campaign contract v3: NK2 object-value priors (bias 2 + log1p(score/1565), max 4) with AdamW-decayed corrections; C2/C3 features only through zero-init ActionScore (user: no fixed default logit effects); a whole-map encoder; in-graph Φ into the value head; credit "mc"; UpdateProcess; paired evals. **No Nullkiller2 seats** (user: "we would learn its weaknesses");
  - map sizes (user): one model for every size; per-object travel times rejected (too costly on XL); a CNN plus TRAVEL (action 11) over 18×18 areas;
  - round 3: hand-made features as small named functions (user); fight return in gold; END/TRAVEL couplings;
  - **round 4** (user: "we have to have 0 killers"): value audits `campaign_model.audit`/`battle_model.audit` plus test_value_invariants. They found NK2 score units vs Φ gold (pickups 3.8× underpriced), Φ missing owned garrisons/dwellings, and TRAVEL no-op loops;
  - round 5: failures.py dumb-decision metrics (user: leaving troops in a mine is dumber than passing a pile); garrison prior −4; chained campaign heads;
  - round 6: battle policy = final V only (0,0,0,1); V stays unchained (the user had forgotten the 10-23 lesson); campaign heads chained;
  - shared-luck toy: a one-day ladder vs raw d+7 labels 6.1×/3.5×/1.8×/1.4×/1.1× by edge half-life. The AI's 7× guess was wrong;
  - round 7: share heads V2/V5/V7 with composed targets mixed by trust; growth helper heads G2/G5/G7 (weight 0.5, never in the credit); user: CE learns far better than MSE;
  - round 8: another AI's review, ~12 bugs fixed; round 9: contract v4 (RECRUIT destination, travel_days, g_last_* echo); round 10: contract v5 (every object, NK2 brain as torch ops), battle contract v16. The user wants ALL of NK2's brain as feature ops;
  - round 11: NK2 cross-day plans as weak pulls (user: "weak features, easy to ignore"), contract v6;
  - heads rule: add a head only if the credit never reads it or it is the objective; no d+1;
  - round 12: **V_end** (end_head) plus random game caps (28 + geometric mean 28, max 112; the AI never sees the cap). Credit blends adv7 and adv_end by trust(V_end).
- **joint2** (2026-10-28 22:23, 5604d8e pushed 03:20):
  - the battle drifted after ~3100 (greedy idle 0.03 → 0.25–0.49), so it was frozen at 3078 at 03:30;
  - campaign self-play never attacks (2/1253 knocked out vs 20/24 vs NK2);
  - the weight average was worse than live in all 5 paired evals (−0.0075 ± 0.0025);
  - the entropy floor is the one consistent push against SHOOT; idle costs ~−0.003 aggregate;
  - replays were not deterministic (a VCMI target-order fix followed on 2026-10-30).
- **2026-10-29:**
  - idle shooters = indifference in won battles (73/77) plus ~1/3 genuine kiting;
  - battle value priors **added** to logits (the user insisted additive): 2×gain×tanh(v/0.02);
  - entropy floor and weight average off; BattleAI seeded by battle state, so evals are deterministic (198/200);
  - **User: no search/simulation.** The AI argued bigger batches don't help at the lr cap;
  - idle metric redefined (full-damage chances only); battle lr 3e-5;
  - stall-kills were endless battles with our model playing wandering monsters. Fix: neutrals always played by BattleAI in training, plus AlphaBattleAI ROUND_LIMIT 50 (0b4d641).
- Corrections of the user: "There is no exponent in either credit (the user believed there was)". The TD credit's wrong sign was a dialog confound.

### 1.18 phi-v3.md
- **Session** b1 · **modified** 2026-10-30 · **type** design.
- Φ v3 = d7a92a2. User ask: no map building may give anything Φ ignores, plus a readable list. That list is `docs/empire_value.md`.
- Adds:
  - spells at 500 × level (main hero full, others half; tavern spells as `g_tavern_spells`);
  - one-battle luck/morale = 0.02 × army worth per 24th, capped ±3; bad morale 2/24 per point; bad luck 0; living creatures only.
- Left out: movement, obelisks (no DIG), access, information. Schema 8; EMPIRE_VERSION 3.
- The numbers (500, 2%) were the AI's picks, not reviewed by the user.
- Open:
  - the MOVE prior prices objects per type, not per hero;
  - a Gog ledger mismatch (seed 43, 156 gold);
  - the campaign learner drops ~half its games as too old (25/48 in 20 min).

### 1.19 campaign-model-v2-draft.md
- **Session** b1 · **modified** 2026-10-30 · **type** design (user principles).
- Redesign drafted with the user: docs/campaign_model_v2_draft.md (626419d), built as model_v2.py (05914d0). User: "good for now but I'm not sure about them".
- **Firm user principles:**
  - the forward is named blocks (domain_knowledge, encode, take_sides, join, see, think, foresee, think_again, decide);
  - `encode` takes raw inputs only; known facts are concatenated after encoding and again late, because a single number like P(win) must not be swallowed by lower layers;
  - ownership must strongly transform vectors (several rivals, so not a sign flip);
  - "large" is relative to the week and the situation; logs alone don't solve scaling;
  - one tile-resolution map shared by everyone; any map size with hidden tiles;
  - **a logit is a match**, never a forward pass per action (actor query · target key + a small net over legal pairs + prior); one softmax over all actions if compute allows;
  - foresight heads are auxiliary: the policy never uses their outputs, only their hidden states;
  - auxiliary targets must be bounded and scale-free (Φ per block rejected); hidden-map targets risk memorising a map.

### 1.20 perf-2026-10-29.md
- **Session** e5 · **modified** 2026-10-30 · **dates** 2026-10-29/30 · **type** measurement log.
- User ask: faster training, then a server so home machines play for the RTX 5090 PC. Merged d9cb662 (+cdf5e50), pushed at d7a92a2 on the user's "push".
- **Round 2** (user: "another AI works on the business logic, don't interfere"), 0239df9 + vcmi 3102490f2:
  - canBeCast once per spell instead of per hex (57% of encoding);
  - 41.9 → 34.4 M instructions/decision; client peak 794 → 501 MB; 0 of 261M values differ;
  - `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2` (Mac learners kept 14 GB of MPS cache).
- **Round 3:** battle evals in a background thread (f0b5c75), pushed 4036691. Legal-only heads not done (~10% saving; CPU inference 7.1 ms at batch 1).
- Quiet-machine numbers:
  - actor CPU per battle decision 18.0 → 12.65 ms (163 → 194 decisions/s); without CPU inference 9.2 → 4.0 ms;
  - the earlier "27.6 → 13.4" was measured while joint2 ran;
  - joint A/B: battle updates/h 126 → 156 (+24%). The Mac is GPU-bound; the campaign learner trains on ~1/3 of its games' decisions.
- **Bug:** VCMI area-attack targets were in pointer order, so ~1 in 10k attack values differed. Fixed vcmi e1be37e4a: 0 of 39,140 differ, replay 0 of 107.8M.
- Other fixes: an AttackValueCache, a persistent map builder, zlib 6→1, int64→int32 records. torch.compile was rejected (wrong-looking grads, no gain).
- Workers: `battle_model/remote.py` Hub plus worker.py.
- Method rules:
  - on a shared machine, compare instructions retired, not CPU seconds;
  - otherwise use a quiet machine, alternating old and new (Apple efficiency cores inflate CPU time);
  - real-run A/B: `cp -c -R` clone, resume with evals off.

### 1.21 no-training-vs-eval-ais.md
- **Session** no id (b1 *(inferred: the 2026-10-30 review context)*) · **dates** 2026-10-30 · **type** developer rule.
- Self-play only. Never put Nullkiller2, BattleAI or MMAI in training games, not even as a share.
- User quote: "this is wrong/cheating. I do not want a model that learn to exploit Nullkiller2 bugs, I want a good model that I will be able to play against." Trigger: a review change that made a third of campaign games vs NK2.
- The eval score is a measurement, not a target.
- *Note:* the NK2 **teacher** (imitation labels from NK2 inside self-play, teacher-and-memory 2026-10-31) was treated as compatible with this rule. The user later asked to remove NK2 as a teacher (2026-11-11 hooks).

### 1.22 models-v2-progress.md
- **Session** b1 · **modified** 2026-10-30 · **type** measurement log.
- User ask: "make all needed improvements to the campaign and battle model; if the battle model needs retraining, retrain it to the existing level before doing the campaign; monitor it and verify you do not make ops that will kill gradients or our ability to learn".
- **joint5** (2026-10-30 15:23–17:35): campaign eval (day 56 vs NK2) 0.086 (v1) → 0.171 ± 0.056 (v17) → 0.134 ± 0.046 (v36). Battle +0.096 (5829) → +0.091 → +0.082.
- Saved: campaign_joint5_v37.pt, battle_joint5_06170.pt, battle_joint5_06000.pt.
- "Value worsening" was a misread: per-update EV on 4–7 games goes negative when outcomes are close (label spread 0.030–0.045); pooled EV rose 0 → 0.21.
- PC plan: keep batch 256 (LR capped at 1e-4; KL ~0.001, 15× under target); passes 2, games 8, micro_batches 1 on CUDA.
- **battle3** (from 5518, LR 3e-5, f3e5b1d: foresight heads, context 32→128, zero-init new parts): 5829 = **+0.096 vs BattleAI and MMAI**; shooter idle 3.3% → 0.8%. Foresight beat the base rate: dies 14%, enemy-on-hex 68%. Gradient share 3–16%.
- Campaign v2 review (Fable): type score = bias + offer + best entry; the NK2-games third was rejected (bae4696).
- **AI prediction (to check):** 0.2–0.4 after 12 h, 0.4–0.8 after 48 h; below 0.2 at 12 h means the credit is the limit. *[Result: not verified in this file; the eval fell to 0.100/0.063 on the PC, mostly from the map-set change (rmg-nondeterministic)]*.
- Lessons: battle_model.train ignored SIGINT in the background (16e50ce); zsh doesn't split unquoted variables. joint2's model fought only 1–3 battles in 21 days.

### 1.23 rmg-nondeterministic.md
- **Session** e5 · **modified** 2026-10-31 · **dates** 2026-10-30 · **type** incident/reference.
- VCMI's RMG gives different maps for the same seed (3 runs on 2026-10-30: 3 maps). evaluate.py reuses cached .vmap files, so each machine's eval set is whatever it cached first. The Mac's was cached 2026-10-29 23:11; joint5 v37+ was evaluated on the PC's maps (0 of 1296 surface tiles match).
- Impact: the joint5 drop 0.134 → 0.100 → 0.063 after moving to the PC came mostly from the map change.
- Rule: one canonical eval .vmap set copied to every machine; never trust "seed = same map". *(Echoed later: "Mac A/Bs need pinned maps (make_map regenerates)", MEMORY.md gather-collapse hook.)*

### 1.24 fast-training-design.md
- **Session** e5 · **modified** 2026-10-31 · **dates** 2026-10-30 → 31 morning · **type** design + measurement log.
- User directives: "make it train fast", "high GPU util focused on training, or every core busy on both machines", "all map creation triggered by workers from an IterableDataLoader". Built on branch `fast-train` (worktree alpha_heroes-e5).
- **Design (`battle_model/supply.py`):**
  - `Games` is an IterableDataset per kind; DataLoader workers (forkserver) play one game each and `prepare` arrays;
  - permits/`generated_room` throttle supply to demand, so no game is played only to be dropped (joint5 had dropped ~90%);
  - local first, then remote slots; one trainer process for both models; one inference server for both families.
- **PC measurements (RTX 5090, WSL):**
  - campaign step eager 146 ms per 128 (~875 decisions/s); b1's 2.3 s came from GPU time-slicing;
  - battle step 36 ms eager, 25 ms graphed;
  - **a second GPU process (inference) costs the trainer 54% of its steps** (~40% with a 10–15 ms batch wait), because WDDM splits the GPU by demand;
  - inference as a thread made the trainer 12× slower (GIL);
  - CPU campaign inference ~38 decisions/s per PC core (M4 ~90);
  - so inference stays in its own process; the real fix is Linux, or the Mac serving inference over Ethernet.
- Live tuning, 2026-10-30 23:00 → 10-31 02:00:
  - Mac-served inference over the LAN: PPO 100–113 s → 79–97 s per 35k decisions;
  - `DECODE_THREADS=3`; `labeler.py` processes (labels had taken ~27 s of the trainer thread);
  - CUDA training graphs captured with `allow_unused_input=True` (3.5× an update, gradients within 5e-4); inputs decoded on the GPU;
  - a "stale reference / AccumulateGrad" capture bug was fixed (the gradient check now uses autograd.grad on its own forward);
  - memwatch killed the run at 23:27 (44 loader workers + a ~9 GB trainer). Fixes: `gc.freeze`, `malloc_trim`, MEMORY_FLOOR_GB 4.5;
  - battle inference moved to 3 PC CPU processes, campaign inference to 9–11 Mac servers;
  - `ahead` 2 vs 3, passes 2, micro_batches 4 (at 2 the card filled to 31.99/32.6 GB and was 1.5× slower per decision while nvidia-smi read 94%). Lesson: **"utilisation can be memory-thrash, judge by decisions/s"**.
- **Final state 02:00 (2026-10-31):**
  - passes 2, micro_batches 4, ahead 3, 24 campaign games, 8 battle actors;
  - GPU mean 70%; campaign updates 100–153 s; 500–600 campaign games/h (25–29 versions/h, 1.1–1.2M decisions/h), against the old loop's 17–83 games/h;
  - but battle epochs fell to 28–46/h (90–158 before).
- Handoff: the run stopped at 07:04 on its 12 h limit. The user gave joint5 to b1, and e5 went hands-off. b1 fixes: an idle client replaced after 40 s, slot accept fixes, turns-mode fairness. Campaign eval v174 0.212, v203 0.204 vs v47–v59 0.18/0.249: flat.
- Incidents:
  - scp of two same-named train.py files into one dir gave the PC battle trainer the campaign file;
  - serial slot closes took up to 40 s each (now parallel);
  - a stale `ssh -N` tunnel held ports;
  - a `Future` on the game object made it unpicklable.

### 1.25 no-action-masking.md
- **Session** no id (e5 or b1 *(inferred; it cites b1's action)*) · **dates** 2026-10-31 · **type** developer rule (from an incident).
- User quote (2026-10-31): "the approach of making some actions unlearnable is bad always. If we do something silly too much it means we are not learning well, making the model unable to use a specific spell is just bad."
- Incident: b1 masked the info spells (View Air / View Earth / Visions) on self-play credit evidence. The eval vs NK2 fell **0.22 → 0.07** (the spells reveal enemy heroes).
- Rule: overuse is a credit/value/price learning problem; keep every legal action learnable.

### 1.26 nk2-action-audit.md
- **Session** no id · **dates** 2026-10-31 · **type** measurement/reference.
- Two NK2-vs-NK2 games (36-size, 3 and 4 players, 397 player-turns, 26–33 s each), counted from trace "Sending a request" lines (~50 MB/game).
- Request counts: MoveHero 12981, MakeAction 5357, ArrangeStacks 1532, QueryReply 868, RecruitCreatures 711, TradeOnMarketplace 686, EndTurn 395, GarrisonHeroSwap 228, BuildStructure 156, HireHero 86 (~11 heroes per player in ~59 days), ExchangeArtifacts 67, DisbandCreature 28, BuyArtifact 15, BuildBoat 6, UpgradeCreature 6.
- NK2 casts **no** adventure spells; our model spent ~16% of turn decisions on them.
- Not expressible in our action space then: artifacts, garrison swap, disband, BuildBoat, BuyArtifact beyond spellbook/machines, and arranging within one army.
- NK2's memory (AIMemory.h): seen and visited objects, teleport channels and gates; invisible objects dropped each turn; the danger map uses only visible enemy heroes.

### 1.27 teacher-and-memory.md
- **Session** b1 · **modified** 2026-10-31 · **dates** 2026-10-31 → 11-01 07:55 CST · **type** design + run log.
- **User decisions (2026-10-31)**, after AlphaStar's ablations (RL from scratch 149 Elo, supervised 1,020, +KL 1,400):
  - "I want the memory... do not waste time training a model we will not end up using... no long runs with no memory model";
  - the model must be able to follow a teacher and improve from its own predictions;
  - the campaign restarts from scratch with memory;
  - share heads learn the **change in logit share** (logit(s_{t+k}) − logit(s_t)), clipped. The user's arithmetic slip was corrected and accepted.
  - *This reverses the 2026-10-27 "No NK2 teacher".*
- joint5 stopped 15:13 (eval v357 0.099, −0.002 vs its start: no progress). Saved battle_joint5_final.pt (+0.091) and campaign_joint5_final.pt.
- Built:
  - `Teacher.cpp` (AlphaTeacherAI = NK2 + recorder), matching 91% of turn requests;
  - `teacher.py` (~22 games/min on the PC at parallel 12);
  - `imitate.py` (in-order imitation, truncated BPTT);
  - **Memory, schema 10:** 8 slots × 64, fixed-point ×2^20, read via a zero-init residual, written by a GRU-style gate, TURN decisions only. PPO uses the recorded mem, so RL does not train the writer;
  - teacher KL (adaptive ×1.5/÷1.5); `share_target="change"` (cap ±3).
- Data: t1 313 games (Mac), t2 400 (PC), t3 112 (stopped).
- **Lesson:** rebuilding the C++ schema while a recorder ran old Python gave 82 games with 0 decisions. Stop runs before schema changes.
- User (~16:00): stream, don't record; no nomem/width A/B ("no memory means we lose"); a full run with no time limit, PC only. Unit `alpha-full`: phase 1 streamed imitation (first period held-out loss 3.36, top-1 19%; ~200 decisions/s, GPU 27%), phase 2 joint6.
- **joint7 (2026-10-31 21:57), NK2 advice as teacher** (main 32531c4, branch nk2-advice):
  - the user asked for real NK2 labels on our own states;
  - `Advice.cpp`: per-player NK2 on a dry-run IClient, ~92% matched, 5–30 ms each. The user prefers it knowing too much over too little;
  - teacher_mix "score": share = clip(1.2 − score, 0.05, 0.95). Started from campaign_imitation.pt (held-out ~1.93, top-1 ~37%); nk2_advice 0.2; credit "schedule"; LR 1e-5 (now the default); battle from 7295;
  - campaign inference on the Mac (on the PC GPU it time-sliced against the learner); batch 8192 in 64×128 micro-batches (256 OOM'd in graph recording); no expandable_segments (graph failure).
- **First numbers:** the imitation start eval was 0.113 (eliminated 39/48). **The AI's prediction of 0.3–0.6 was wrong.** Update v2: teacher_ce 5.86 on our states (vs 1.9 on NK2's), agreement 7%.
- **Exchanges (325c0bc):** 55% of decisions, untaught; games ran 5k–25k decisions. Fixed with max 8 troop moves per exchange plus adviseExchange (getBestArmy; the receiver is the higher level). NK2's own rule bounced armies between heroes.
- **KL target (5dea773):** target_kl stopped every update after 2–6 of ~56 steps and cut LR 1e-5 → 3e-6; ~90% of games went unused. Now kl_target = (1 + teacher_weight) × target_kl.
  - Table: v4 6 steps / teacher_ce 3.62 / agreement 14%; v5 18 / 3.20 / 14%; v6 58 / 2.55 / 16%.
  - Advice raised to 0.5 (2322237).
- **NK2 advisor crash (vcmi 823c80710):** 4/~25 games exited 139 after "ExecuteHeroChain". getTile returned nullptr for an unseen tile; reproduced 1/7 under `VCMI_LLDB=1`. The PC has no gdb/lldb.
- v5–v9: value_ev_d7 −1.55 → +0.09; evals 0.101/0.121/0.159. Passes → 3.
- **Move labels were one-step (5e72edc):** moveHeroToTile sends one MoveHero per tile, so the label taught "walk one tile a decision" (~60 decisions per player-day; tile match 5%). Now labelled with getTargetTile() (MOVE if reachable today, else TRAVEL). **Teacher.cpp still labels the first tile**, so the imitation checkpoint learned one-step moves.
- Eval series (day 56 / day 28): 1 0.113/0.357; 2 0.130/0.331; 4 0.109/0.220; 5 0.101/0.386; 7 0.121/0.457; 9 0.159/0.501; 11 0.087/0.477; 15 0.154/0.527; **24 0.221 ± 0.042 / 0.570** (+0.108 ± 0.050; eliminated 27/48; Φ 70k).
- v17: agreement 28.9%; decisions per player-day 72 (v5) → 25–38 (v13–17).

### 1.28 cleanup-plan.md
- **Session** b1 · **modified** 2026-11-02 · **dates** 2026-11-01 · **type** review / reference.
- A second review's "dead code" list, verified at 4336918. **False positives, never delete:**
  - `@stack_feature` functions (registered by the decorator, so they are the battle inputs);
  - `check_reach.py` (a by-hand engine check);
  - `campaign_model/model.py` (v2 imports from it).
- Done: dead functions/fields/imports removed; CampaignConfig defaults = what runs (v2 dim 128 / cell 16, share_target change, credit schedule).
- Deferred to before joint8: v1 classes, the "attributed"/"mix" credits, process mode, the battle onnx path. Keep "horizon" and the frozen teacher.
- Rule: verify "dead" by how code is reached (registries, decorators, CLI), not name grep; smoke-test start/stop/resume on the Mac before startup changes.

### 1.29 minimal-engine-changes.md
- **Session** cd · **modified** 2026-11-02 · **dates** 2026-10-22, 2026-11-02 · **type** developer rule.
- User quote: "minimal unwanted code changes to the engine ... Changes that fix actual bugs in VCMI are good." Prefer map, scenario or own-code fixes; label each patch as a bug fix or a behaviour change.
- Example 2026-10-22: the user's sea-hero workaround can't work, because battle-only mode needs exactly 2 heroes and no towns.
- **2026-11-02 clarification:** "obviously we are patching VCMI fork if we need to, that's why we even have a fork." Not an approval gate; "minimal" means no unneeded behaviour changes.

### 1.30 model-cannot-do-list.md
- **Session** e5 · **modified** 2026-11-03 · **type** reference/process.
- User (2026-11-03): a living list in `docs/model_cannot_do.md` of what the model can **never** learn. User order: "Arranging one army" first, artifacts second. User: "This list will expand later". It is separate from the problem watch list (trainable weak spots).
- Contents then:
  - ARMY needs two holders;
  - no artifact action;
  - the memory writer gets no PPO gradient;
  - RECRUIT only the upgraded creature;
  - yes/no dialogs drop their components;
  - no town spells input; no garrison swap; no boats.
- *Partly fixed later:* schema 16 added in-army arranging and garrison swap; schema 12/13 added dialog components (fix-batches).

### 1.31 hero-value-design.md
- **Session** no id (b1 *(inferred)*) · **dates** 2026-10-29, 2026-11-03 · **type** design (user decisions).
- 552625f:
  - battle reward: hero by level only, level-up from L worth 2000 × min(L,3) (L2 2000, L3 6000, L10 48000). User: "good enough for battles". Levels 1–3 had been free, a bug: no campaign hero ever levelled;
  - campaign Φ: HERO_BASE 1500; +2000 per secondary level (University price); k-th primary point beyond 5 worth 400·k (user: "worth more the more a hero has"); +200/level; the strongest hero full, others 0.5 (user unsure: "seems like it"); primaries without items; spellbook 500 in Φ, 0 in battle;
  - hiring with the week's army is Φ-positive, a lone creature negative; contract 7 shows tavern heroes.
- User on credit: a day+1 head is allowed; same-day/end-of-day values only if all else fails (flagged suspicious); post-action values are a no-go.
- 2026-11-03: HERO_BASE scales with army, 55% with none, full from 1,000 gold (an empty hire ≈ −1k).

### 1.32 changes-2026-11-03.md
- **Session** b1 · **modified** 2026-11-04 · **dates** 2026-11-03/04 · **type** design + measurement.
- **Share weight ramp (0ecb447):** w = 0.1 + 0.03 × day × 36/width (all share by day 30 on 36-wide maps). The user rejected contact detection ("hard to figure out correctly") and a revealed-% floor. Our AI reveals 16% of the map by day 7 and 26% by day 42.
- **Armyless hero Φ (a920331):** HERO_BASE = 1500 × (0.55 + 0.45·min(1, army/1000)). User: "a hero with nothing slightly negative, maybe -1k on average … heroes with creatures should stay positive." Over 471 hires: +1,943 median; −366 without army, now −1,040.
- Exploration 1 → **5 gold/tile** (a full 36×36×2 map = 13k).
- **Fight features on MOVE** (user's idea: the ratio "should affect the logit … via the v"): pair_fight, move_fight, zero-init. Φ changed, so evals from a920331 don't compare with earlier ones.
- **Evidence (joint7, 30 games):**
  - self-play nearly peaceful: 599/600 games hit the day cap; 5 eliminations in ~1,870 player-games; vs NK2 eliminated ~31%, no wins; ~1 hero vs NK2's 3–5;
  - heroes per player-game 5.8 distinct, 3.9 lost; 51% of attacks on neutrals lost;
  - **suicides:** heroes with < 1,000 gold of army attacked 224 times, lost 97%, vs stacks a median 54× stronger; NK2 never advised these moves. The prior was −3.35, the learned part only ~0.6 below the best tile.
  - The user rejected a win-probability payoff (a 4× lead already pays 16/17).
- **move_fight inert:** at lr 1e-5 there are ~300 optimizer steps/h, so a 3-weight layer moves ≤ 0.003/h. Suicide PPO credit is weak (standardized mean −0.1 to −0.2, only 38–50% negative; ~1.5k Φ drowned by the 7-day noise). FIGHT_STEP=100 was prepared (fight-step 4cd9e16) and asked of the user, since it conflicts with the training-stability rule. The AI later corrected its own first estimate (1.6/h ceiling → ~0.06/h).
- Re-measured: 275 attacks, 99% lost (no change, as expected).

### 1.33 move-feature-audit.md
- **Session** b1 · **modified** 2026-11-04 · **type** audit + design + deploy log.
- Audit of v795 (1,200 MOVE decisions):
  - 5 dead features (defend, under_threat, home, to_main, behind); the prior was input twice; win/loss/fight saturate (3× vs 54× indistinguishable); absolute scales;
  - the foresight predicted P(battle) 0.15–0.18 on hopeless tiles (a battle followed almost always).
- **Deployed ab52217** (user: "Go with everything… continue training after you are done"):
  - **bug:** blocked_by_guards ("behind") was always 0, because VCMI marks every object tile blocked (3,302/3,302);
  - foresight labels were correct (94%);
  - 4 strength estimates × (this hero, strongest hero) clamped ±2; relative features log(1+100·x/Φ)/3; 27 known features; mixer 27→16→1 plus a direct path initialized at hopeless × −3; battle heads;
  - measured before deploy: KL(types) 0.005; hopeless mass 0.78% → 0.087%; CPU inference 56.8 → 64.5 ms.
- **Pitfall:** a git-archive test copy lacks `battle_model/artifacts/vocab.json`, so game_tables silently uses defaults.
- **User rule (2026-11-04):** the battle head predicts **the engine's answer only**. User: "predicting our own action is silly" (0b899a4). Classes: no monster / it attacks / it offers or flees. RMG monsters almost never offer (59 attacks, 1 offer).
- Still missing as fight tiles: neutral garrisons (8% of battles), guarded dwellings, pyramids.
- Suicides after ab52217: 193 attacks (from 275), 98% lost, median 27×. **Safety prior (5805895):** prior −= safety_weight (6, bounded 0..12) × hopeless; hopeless mass 0.19% → 0.031%. After it: ~112 per 30 games (0.30 per new hero).
- Eval v854 0.444 (+0.057 vs v832). Between consecutive evals ~12/48 games flip eliminated. campaign_value_loss 0.00055 → 0.00038; V7 EV 0.17 → 0.24.

### 1.34 changes-2026-11-02.md
- **Session** b1 · **modified** 2026-11-02 · **type** design + deploy mechanics.
- User: "make these changes as you see fit and continue training after you do and see that they work".
- 4c2212a:
  - head weights 1/(target variance + 1e-4), rescaled to keep the loss scale, ramped over 20 updates (share target std 0.010–0.017 vs V_end 0.086, so V_end had dominated);
  - battle mix 60:20:20 (real_share 0.6, league 0.5/0.5, real_keep 0.12).
- fe5cad8: MOVE movement discount, NK2's movement_discount(days)^(1.5·tanh(10w)), bit-identical at w=0. `model.ADDED` fills missing keys with 0; Adam moments kept by name.
- 8f78371: no eval during a drain.
- **Compute split:** GPU campaign 99% / battle 1%. User: "99% ok for now, for the future 97% to 3% would be better".
- **Speed:** e5's TBB cap VCMI_MAX_WORKERS=4 took games from 27 → 7 threads (0.185 → 0.065 CPU-s/turn); median 72.5 → 61.8 ms per decision. bf16 CPU inference: **no** (0.95× on Zen 5, 0.33× on M4). fork_preload (f398aec) cut workers 77 → 5 MB.
- **Deploy mechanics:**
  - `git archive | ssh … tar xf -` (the PC HEAD stays at 316d8cb); `runtime/drain_restart.sh`;
  - stop the Mac worker around state-dict changes;
  - `alpha_heroes-e5/build` was a symlink to the main build, so a main build changed the live worker's binaries (~8 games silently damaged) → build-own copy;
  - PC builds `nice -n 19 ninja -j4` (a -j24 build tripped memwatch 11:22:10);
  - never `pkill -f` a pattern in its own command line.
- **NK2 plan head (71ea846, vcmi dcedb18b1).** User rule: "the patch may only affect the loss", so the plan is a label, never an input (the model must run on original VCMI). Plans at 51% of turn decisions.
- **Teacher over the turn (a8b587e).** User requirements: order inside a turn must not matter; a target must have been visible when used; the exact-strength leak is "not important".
  - teacher loss = −log Σ p over accept sets; nk2_advice 1.0;
  - **loss_shares** (closed-form GradNorm): teacher min(0.6, clip(1.2 − score)), the rest 4:3:1; value weight ×1.25/update. User: fine "as long as it is rising with each update".
- Gradient norms: teacher 77–195, policy 1.5–21, value ~0.005.
- The NK2 goal head: the `intended` tile is NK2's next waypoint (median 3 tiles), not the task target; design given to the user.

### 1.35 trades-and-guards.md
- **Session** b1 · **modified** 2026-11-04 · **type** measurement + design + deploy log.
- **User rule (2026-11-04):** "every trade is dumb unless we bought something that could not have been bought without making this trade".
- Measured on joint7's 10 games:
  - 135 trades a game, 77% with no purchase that day;
  - a median 900 Φ lost per trade (~213k a game); one marketplace trades rare-for-wood at 1:20;
  - NK2: 6% of its advised trades unlock a building; our trades match NK2's 15%.
- Trade prior (50678d1): +2 for an unlock, −0.5·log(1+cost/1000), plus a zero-init learned path. Pointless-trade probability −13% at once (−43% where nothing unlocks).
- **AI correction:** the AI had told the user something wrong about VCMI's pathfinder. It never passes through a guarded tile, so "non-fight" suicides were the destinations' own guards (garrisons 9/49, towns, a mine). The user agreed armed objects "should totally be treated as a battle".
- **Schema 11** (778b92f + 6f54540 + c01038f; contract 3653f6c54ad2233b):
  - option o_battle/o_join; question d_guard_*; up to 32 neutral garrisons' armies as quantity ids;
  - monster questions recognized by advobtxt 86/90/91; the zone guard via guardingCreaturePosition;
  - verified on Mac and PC (step graphs vs eager 3e-8);
  - **deployed 2026-11-04 17:09.** First 35 games: 0 AI errors; joins 10/12 accepted; flee pursued 1/36; banks/mines 20 asks, all declined (a Shipwreck asked 12× by one hero).
- **Bug (3b1424d):** object_gold(HERO) = −760 was charged as a fake loss on flee/join fights (−2.5 logits instead of +0.4).
- **Dwellings:** we said NO to 467/481. **AI error:** it first misremembered NK2's behaviour as NO; NK2 answers YES. Rule: check the source before claiming NK2 behaviour. recruit_yes prior (0..8, start 4): YES 3% → 42%.
- **Global NO bias:** the learned part of every yes/no leans NO by 4–6 logits. It dates from before schema 11, when every YES looked alike; with YES never sampled, PPO can't fix it. (This led to the question-head redesign.)
- Loops: a Quest Guard asked 53× in 15 days; a guarded artifact 40× in 11 days. visit_yes prior (08c8106).
- Tooling: macOS `grep` misses strings in dylibs (use `strings | grep`; `grep -a` on the PC); Python puts a script's own folder first on sys.path (keep analysis scripts outside test copies).
- **Atomic checkpoint saves since 50678d1:** a copy of checkpoint_latest.pt taken mid-write was unreadable.

### 1.36 question-head.md
- **Session** b1 · **modified** 2026-11-05 · **dates** 2026-11-04 · **type** design (user) + measurement.
- User quote: "The YES token is the same vector for every yes/no question => this is retarded. Fix it the right way and let us continue the training." User design: an embedding per option, probability = atanh of a cosine; expert rules in their own functions.
- Built d60df0b + 909067b:
  - OptionEncoder: each option is the model's own vector for what it names;
  - **no hashed rows** (features.bucket had put 546 of 954 options on shared rows, found by the e5 peer);
  - the key is a raw identity sum (a fresh Tokenizer gave cosine 0.997 for every option);
  - logit = atanh(0.995·cos) via log1p (ONNX opset 17 has no atanh);
  - `option_knowledge` (question_guards, nk2_yes, nk2_select, nk2_amount, skill scores); migrate_option_head.
- Shift on the real checkpoint: dwelling YES 0.24–0.28 → 0.98; Tree of Knowledge 0.05 → 0.98; flee/join 0.04–0.09 → 0.40–0.69; Arena 0.83 → 0.05.
- In the run (19:44, 29 games): dwellings 466/473 YES; the 53-ask loop gone; joins 9/10; guarded artifact 37 asks (110 before); 0 AI errors.
- Trades still 147 a game (76% pointless): "the next big leak".
- Open: University closed unasked by AlphaMapAI (map taverns and markets too); 39% of dwelling purchases mixed in a new faction (the user wants learnable fit features later).

### 1.37 fix-batches-2026-11-04.md
- **Session** b1 · **modified** 2026-11-05 · **dates** 2026-11-04 evening → 11-05 01:46 · **type** design + deploy log + incident.
- **User answers:**
  - BUILD becomes build+trade; RECRUIT trades only when not even one creature is affordable; no network-set trade amounts; C++ calculation OK with no magic numbers;
  - army: "whenever possible make a unify decision";
  - artifacts: backpack → slot, assembling; "More serious is how we move stuff between heroes"; rework hero value multiplicatively;
  - memory stays a learned vector ("if we see a hero and decide it is important to write it, we will");
  - revealed guards are needed; RECRUIT must see the visiting hero; boats.
- A (4af606c): BUILD identity, RECRUIT reads the visitor, recruit_fit, `nk2_trade_advice False`.
- **B, schema 12** (542fdbe, contract 6ee1ae97fd6670c0): exact trade purchases (tradesFor + resource_worth). **AI overrode the user's 7-vector:** "a deterministic output can't be trained by PPO, and the cheapest plan is computable exactly". Also boats, University yes/no, revealedGuards, unifyStacks before each turn decision.
- C1 (08627ff): army × √(dealt/taken). Finding: at day 59 our heroes led 5.7k and 8.9k of armies worth 72k/138k; the rest sat in garrisons.
- **C2, schema 13** (de82f54cc368f0b3; deployed 23:37): ActionType.ARTIFACT (GIVE/EQUIP/ASSEMBLE) with artifact_cost_weight (without the cost, 460–1,417 zero-gain moves a game). First 29 PC games: 42 GIVEs a game, 0 EQUIP/ASSEMBLE.
- **D (8458ede, 00:38):** memory_steps 2 (BPTT in PPO); step replay 117 vs 108 ms.
- **E:** trades priced as Φ (trade-purchases had lost ~5,200 each, ~150k a game); P(build+trade) 0.049 → 0.024.
- **Incident:** gather/visit/exchange-artifact advice (teaching what NK2 does outside its requests) deployed 00:58–01:23 collapsed the policy, so it was **REVERTED 01:46 (26bc2a1)**. Rollback: v1025 weights renumbered 1044.
- Facts: heroes led 2% of our army at day 60; P(ARMY town→hero) 0.0002; BUY 0 of 4,219; UPGRADE 26/1,689.
- Evals: 967 0.506, 984 0.449, 1008 0.433, 1024 0.378, **1037 0.248 (collapse)**, 1051 0.391 (Φ 152k vs NK2 539k; heroes 0.81 vs 3.83), 1071 0.487, 1097 0.390.
- **Deploy pitfall:** an all-script's `git merge --ff-only main` in e5 takes whatever is committed on main (it shipped C1 to the Mac before the PC). Commit only what is meant to ship.

### 1.38 training-stability.md
- **Session** no id (b1 *(inferred)*) · **dates** 2026-10-26, 10-31, 11-04, 11-05 · **type** developer rule.
- **Rule (2026-10-31):** LR at most 3e-5, 1e-5 recommended (the cap had been 1e-4); stability over speed; long runs fine; flat evals are bad; lower the rate as a run converges; never raise it; KL control may only lower it.
- Evidence: fixed 2e-4 went unstable after ~1.5 h (2026-10-26); streamed imitation was flat at 1e-4 (~1.97 held-out loss), and 3e-5 gave 1.985 → 1.907 at once. User: "If dropping to 3e-5 helped, we should be at e-5 not 3e-5, the LR would need to be lower when we are closer to converging."
- 2026-11-04: on a 30-hour eval plateau the user declined a lower LR: "I'm happy with the drop in the V loss" (0.00055 → 0.00038).
- 2026-11-05: value_loss_fresh jumped 0.001 → 0.008 at v1135 (schema 14). The user proposed an LR ramp-up after every change; **the AI said no** (a data shift, not instability; a ramp only keeps the heads wrong longer). Cost: 7-day growth EV 0.45–0.53 → ~0.24, V_end 0.33–0.37 → 0.20–0.25. The watcher alerts if the growth fit is < 0.15 for 3 updates.
- *Note:* this rule later collided with inert zero-init layers (move_fight, changes-2026-11-03) and with the battle noise walk (MEMORY.md battle-flat-diagnosis / battle-noise-walk hooks).

### 1.39 multi-actions-list.md
- **Session** e5 · **modified** 2026-11-05 · **type** reference / review.
- User ask (2026-11-05): review schema 14's MULTI and list every NK2 multi-request goal so ours can make them too ("that will also make the learning easier"). List in `docs/multi_actions.md`.
- Build order: TAKE_BEST (pickBestCreatures; ArrangeStacks 1,532 per 397 turns) > GARRISON swap / GARRISON_IN/OUT / HIRE-with-swap > UPGRADE_ALL + VISIT_TOWN > EQUIP_BEST > SPLIT_SINGLES > recruit-all at a dwelling.
- **Review finding:** the advice loss is −log(p(label) + Σ p(accepted)), so an entry accepted wherever legal (GATHER) turns the teacher off on those decisions. GATHER's prior priced the town's stacks while the action moved the garrisoned hero's army.

### 1.40 multi-actions-and-phi-unit.md
- **Session** b1 · **modified** 2026-11-05 · **type** design (user) + deploy log + incident.
- **User decisions (2026-11-05):**
  - Φ unit = a starting hero (SP 3, K 2, spellbook): army × U × m(hero), m = √((1+0.05A)/(1−0.025D)) × (1 + 0.02·√(SP·K)); U = 0.953. The AI's first "garrisons ×0.8" broke "recruiting at list price is worth nothing";
  - multi-actions: recruit all (town / into the hero), town army into the visitor, give all at a meeting;
  - **no "gather-once"**; the teacher's loss must stay sane when a multi-action is chosen.
- Built (813e540 Φ + meeting value; **schema 14** e3f5503, contract 379a041445961a27; ~08:10):
  - eval Φ frozen at the 2026-11-04 version (EVAL_EMPIRE, eval version 3; training 4);
  - ActionType.MULTI (13) with RECRUIT_ALL / RECRUIT_ALL_HERO / GATHER / GIVE_ALL via VCMI bulkMoveArmy; teacher acceptMulti;
  - fingerprint now holds artifacts; type KL 0.019.
- **GIVE_ALL direction (2218ada):** Φ alone chose the receiver, so 300 give-alls a game, a third sent back. Fix: the higher level, then the factor, then the bigger army (93% agreement with NK2 vs 71%), plus give_all_cost_weight. A cost on both directions was rejected (troop moves took over).
- **GATHER emptied towns:** eval v1137 0.265 (28/48 eliminated); garrisons held 80% → 21% of our armies; empty towns 0 → 31%; towns lost 0 → 0.13 per player-game. NK2 also gathers unconditionally but defends separately. Fix 0eab9f0 (gather_loss danger + gather_cost).
- **Rollback 10:39:** v1152 still 0.256, so weights went back to v1125 (renumbered 1160) and the teacher drops GATHER (69d0332).
- User (~11:40): "incorporate all insights you think are right ... just do it".
  - **Φ v5** (5f9c560): army × (slowest speed/2000)^0.25 × make-up morale;
  - "composites are options; the price decides" (no masking);
  - **schema 15** (5eaafd1): TAKE_BEST / TAKE_BEST_FAST (NK2 getBestArmy ported to Armies.cpp), DROP_SLOW/DROP_FOREIGN, MERGE_ALL; SPLIT_SINGLES deferred (unifyStacks would undo it). The user flagged weak heroes needing to split as open (cannot-do row 1).
- Bug found by a counter: arrange() stranded one creature (33/119 TAKE_BESTs).
- Eval v1169 0.349 (+0.093 paired), 23/48 knocked out.

### 1.41 review-2026-11-05.md
- **Session** no id (b1 *(inferred)*) · **dates** 2026-11-05 · **type** review.
- User: "review to find more bugs", then "any real bugs need fixing". Fixed:
  - TAKE_BEST duplicates (40c2b89);
  - repeat meetings (Φ v4 factors; 308 exchanges a game) (40c2b89);
  - GIVE_ALL/GATHER priced as both armies whole → bulk_moves (3d712b2);
  - RECRUIT_ALL levels that don't fit (34bac74);
  - town SHIP building 20 legal without wood, 569 BUILD no-effects (f75ea35);
  - logs kept (3cc4e50).
- **User rules:** Summon Boat must not cheat (no legality from hidden information); neutral shipyards matter.
- Open: 2,110 CAST no-effects (likely Summon Boat with no boat; acceptable).

### 1.42 training-speed-2026-11-02.md
- **Session** e5 · **modified** 2026-11-03 (+2026-11-05 note) · **type** measurement log.
- PC, joint7 v265–272:
  - a campaign update every ~380 s; GPU ~60%; ~190 games/h (~580k decisions/h); a game ~216 s, ~3,000 decisions;
  - CPU: inference 5.3 cores, VCMI games 5.2 (60% TBB spin); RAM: loader workers 10.9 GB PSS.
- **Games wait on inference, not the engine** (NK2-only ~60 ms/turn; ours ~70 ms/decision round trip).
- Model on one M4 thread: 20.4 ms/decision; see-block 46%. Wire 876 KB/decision. *(Corrected 2026-11-03: v2 reads 124 of 127 inputs; the earlier "380 KB unread" was wrong.)*
- **TBB spin bug:** oneTBB spawns one worker per hardware thread, and NK2's parallel_for wakes them all. `VCMI_MAX_WORKERS=4` gave 0.185 → 0.065 CPU-s/turn (2.9×).
- Not levers: CPU batching, decode threads, bf16, torch.compile (a Dynamo bug with `tensor == IntEnum`, 67 sites), ONNX Runtime (17.0 vs 17.9 ms). A forward is ~20k aten ops: ~6.7 ms of kernels and ~10 ms of per-op overhead.
- **Window budget (489deaa):** step 61 → 36 ms, GPU peak 10.3 → 6.8 GB.
- **Decode fix (e79960a):** libz into a pinned buffer, 38.8 → 8.1 ms per batch.
- Production tuning (2026-11-03): 237 → 260 → 301 games/h. **Lesson:** "on the PC, memory is the currency; count every change in GB against the 4.5 GB floor".
- convert_in_graphs (ce524f4): only ~7% faster against a 25–35% estimate. **Lesson:** "estimate learner gains from GPU busy time … not from the CPU-side clocks".
- **Forkserver preload never applied (78d55e4):** the labeler's pool started the forkserver first, so each worker imported torch (~380 MB, 9.5 GB for 25 workers). After the fix: 594 MB total; OPENBLAS_NUM_THREADS=1. Then 357 games/h (+51% on 237).
- Rare overflow graph pool ~6 GB (0321bec, now eager).
- 2026-11-05: campaign 3.32M weights, battle 430k (not bigger). Steady state 226 decisions/s → 193–226 after schema 14. 10 restarts after 08:13 cost ~10–15 min each: **"Batch deploys."** Game time 207–219 → 275 s.

### 1.43 checkpoint-compatibility.md
- **Session** cd · **modified** 2026-10-28 (+2026-11-05 note) · **type** reference/rule.
- A SCHEMA_VERSION bump alone only needs a re-export. New or reshaped parameters need a migration (copy old tensors, zero new columns, fresh Adam moments, applied to snapshots too). Only width/depth needs from scratch (324k params then; dim 64 × 2 layers). The AI advised against growing: "the signal, not capacity, is the bottleneck".
- Any reward change makes cached `eval_baseline*.csv` stale (vs_baseline would mix definitions).
- 2026-11-05 dead-gradient audit (Adam exp_avg_sq): campaign 5.7% never updated (unseen ids, reserved DISMISS, v1-only couplings), battle 4.8%. Answer: no restart.

### 1.44 neutral-towns.md
- **Session** b1 · **modified** 2026-11-06 · **dates** 2026-11-05 22:30 → 11-06 · **type** incident (encoder bug) + measurement.
- **Bug:** Encoder.cpp read "other towns" from VCMI's getTownsInfo(false), which walks only players' own towns. So **a neutral town was never in t_***, even though the Python schema expected OWNER_NONE.
  - A neutral garrison was no fight to the MOVE features, and its tile was worth ~0.
  - Evidence: in 40 games, all 17 hopeless attacks the features saw as "no fight" were neutral towns (25–2,120 gold vs 720–12,020), played at p up to 0.73.
- **Fix 296b06a:** other towns from getAllVisitableObjs, nearest 8, plus a test.
  - Shift: 33% of decisions list a neutral town; TURN KL mean 0.037;
  - P(step onto a neutral town): hopeless 0.27 → 0.002, stronger 0.07–0.15 → 0.75.
  - Deployed 2026-11-05 22:59:58 with 25af47e (battle stall fixes).
- **Second bug, unseen guards:** guardingCreaturePosition only checks tile visibility. A tile next to a monster in fog was TILE_GUARDED but valued 0, so exploration's +2.8..+4.2 came on top (5 of 46 hopeless attacks).
  - Fix 635912f: take the window's worst visible monster's estimates, or UNSEEN_GUARD_RATIO −1.
  - Deployed 2026-11-06 00:33:52 with `eval/campaign_value_unexplained`.
- **Eval:** v1288 0.476 ± 0.062 (paired +0.085 vs v1272 0.391; the best of joint7; eliminated 15; heroes 1.27 vs 4.01). Then v1304 0.402 (paired −0.074).
- **Lesson stated:** "Don't claim +0.085 as the fix's effect. Treat it as a gain of ~0.01-0.08 plus fewer eliminations."

### 1.45 teacher-label-changes.md
- **Session** b1 · **modified** 2026-11-05 (content through 2026-11-06) · **type** incident log + developer-relevant rules (frontmatter type: feedback).
- **Rollback 1 (2026-11-05 01:46):** adviseGather / adviseVisit / exchange-artifact advice (de04cf8, 93e39d2, 5d7be3f) replaced NK2's own label at ~40% of turn decisions with labels the model gave p ~0.0002.
  - teacher_ce 1.1 → 2.4; KL control cut LR 1e-5 → 2e-6, but drift accumulated: type KL 2.3 (MOVE 0.53 → 0.14, BUILD 0.02 → 0.25); games 51% ARMY;
  - eval v1037 0.248 (paired −0.129 ± 0.041); rolled back to v1025 (renumbered 1044); quarantine dir.
  - **Why:** shared type parameters move globally under a huge sustained CE. The AI read the CE jump and KL stops as "expected" and did not measure drift.
- Rules:
  - never replace NK2's label wholesale for a class the model assigns ~0;
  - **accept sets are NOT safe** (corrected the same day): −log Σp raises each accepted entry in proportion to its own p, rich-get-richer; GATHER went 0.06 → 0.33 in 7 updates;
  - run `type_drift.py` after any label/prior change; a teacher_ce jump > ~0.3, or two KL-stopped updates in a row, means stop and measure;
  - **a new action shifts NK2's labels** by changing the states it labels: GIVE_ALL turned NK2's exchange labels ARMY 52% → 85%.
- **Rollback 2 (10:39):** GATHER drift. Mini-eval on the same code: v1157 left 59% of towns empty (0.274) vs v1125 24% (0.390). Lesson: compare weights on the same code (mini_eval.py) before blaming code.
- **0d54e0f** (another AI's review; user: "if you agree implement"): teacher CE taken within the label's span. CE 1.231 → 1.111. It drifted in 3 updates (heroes left towns before buying). Fixed by 1463891. **Lesson:** "order doesn't matter" is false for a town's visitor.
- 444ebfc: the v1224–1225 "explosion" was a coherent one-off correction. MULTI out of the spans doubled DROP (destroying ~12k gold of creatures a game) and GATHER, so MULTI went back in the spans plus disband_cost.
- b69bb20, user: "add a big minus [to disbanding] ... we do not want to remove our creatures" (disband_cost start 6). VALUE_SHARE_CAP 3 (the balancer had cut the value weight 356 → 146 when the heads were most wrong). Declined: ×10 readout gain; raising grad_clip mid-run. Agreed: "no more Phi changes for a while".
- **Rollback 3 (~20:00):** evals 0.382 (v1184) → 0.286 (v1248); paired mini-eval −0.159 ± 0.117, so back to v1184. daac154, user: "they can be very negative towards usually useless actions": trade_waste −5, DEFEND with a full shot −5. v1258 0.365.
- **Rollback 4 (2026-11-06 03:33):** evals 0.476 → 0.402 → 0.297.
  - Cause: the score-driven teacher weight λ/(1−λ), λ = 1.2 − score, fell 4.22 → 2.62 after the lucky 0.476, loosening the anchor; restored v1288 as v1331;
  - **bug c4241a1:** checkpoint_<v>.pt was saved when the eval ended, so "01288" held v1293's weights;
  - recommendation (not implemented): smooth the score. v1342 0.342, v1361 0.423. Rule: "judge a change by 2-3 evals, never one".

### 1.46 hota-transfer.md
- **Session** e5 · **modified** 2026-11-06 · **dates** 2026-11-05/06 · **type** plan + decisions.
- User (2026-11-05): "I want my model to be transferable to HotA even if now it is not". Also: one model for both; ids matched and verified at engine start; HotA objects worth 0 as OTHER "needs to be fixed before we even think on training with it"; "embeddings for creatures/artifacts can start at random, their values must be true though".
- Plan: `docs/hota_transfer_plan.md`. A shared `alpha_content` ContentMatch (strict/tolerant), rows by json key, record headers with vocab and content hashes, vocabulary-sized arrays, values derived from engine config, append-only vocab, a coverage gate. Phases A → D.
- 2026-11-06 ~08:00, branch `hota` (worktree alpha_heroes-hota): HotA port 1.7 zip (442 MB, own assets); our VCMI 1.7.5 loads all 61 submods; SoD ids keep positions. HotA's gameBalance changes 31 creatures, 8 artifacts, 3 heroes, so **value tables per ruleset** are needed.
- Part 1 (92a6ba9): AlphaContent (FNV-1a 64 hashes), the Python `content/` mirror, 6 tests.
- **Parked 09:40** (user: "save this work on a branch ... come back to main for a new task"): `hota` at 6dcbe31, local only. Open decisions: per-ruleset tables; battles first.

### 1.47 pc-training.md
- **Session** b1 (frontmatter; includes notes about e5's work) · **modified** 2026-10-30 (content to 2026-11-06) · **type** reference + incident log + developer rules.
- PC: DESKTOP-84NG4T3, RTX 5090 32 GB, Ryzen 9900X 24 threads, WSL2. joint5 moved there ~18:10 on 2026-10-30. The user approved WSL memory of 28 GB.
- Access: ssh + `wsl bash -s`; no GitHub key, so bundles. Runs as systemd user units (KillSignal SIGINT, TimeoutStopSec 900). WSL 2.7 stops ~15 s after the last client, so e5's scheduled keep-alive task (`wsl.exe --exec sleep infinity`) must stay.
- PC-only bugs: select() on fds ≥ 1024 (e505aba); one-decision batch stride (7f9bda6); whole-game decompress (e7014c1); whole-minibatch decode in pinned memory, learner 9.5 GB (7551903); league graph buffers (435fa2b).
- Inference: an eager CUDA forward under WSL costs ~35 ms of launches (~200 ms shared); CUDA graphs serve ~800 campaign / ~1300 battle decisions/s per process; CPU 13–19 ms per decision on PC threads, 9–12 on M4.
- **RAM is the binding limit.** Never run tests on the PC while live. CUDA `table[big index]` backward sorting took 95% of GPU time (c7fce84).
- **AI error acknowledged:** "I blamed launches; WRONG: e5 measured 146 ms … the 2.3 s was WDDM time-slicing."
- Evals: PC 0.100 (v37), 0.063 (v39) vs Mac 0.134/0.171 (*later explained by the map set*, rmg-nondeterministic).
- TensorBoard: `--reload_multifile=true` (without it TB showed nothing for 4.5 h); `--samples_per_plugin scalars=5000` (the default 1000 dropped ~30% of recent points); never `scalars=0` (keeps none).
- 2026-10-31 user: fix bugs, do NOT relaunch; then (~09:20) commit+push, "keep training nonstop", the campaign is the model to train most; battle flat → user's ~10× batch + LR 1e-5 (accumulate=12, 3ff87b5); "NK2 may be a teacher, never an opponent in training".
- Credit: credit_trust 0 since ~v50 (V7 step estimate mispriced by type: move −10.8, travel +10.4). Casting rose 2% → 10%.
- **Info-spell mask incident (10:05 → 10:44):** eval 0.22 → 0.07, eliminated 13 → 20/24; reverted 567c7d5. **"Judge behaviour changes by the eval vs NK2, never by self-play metrics."** v229's 0.267 was inflated by 4 stalled eval games (true 0.219).
- 11:46: end credit check turned OFF; eval games 48.
- 13:09: dense (object, subtype) vocab, model 2.25M → 1.18M weights; battle accumulate 24, lr 3e-5, beta2 0.99, 4 passes.
- **Incident 13:09–13:20:** a refused Mac worker restart left no campaign inference, so 4,312 games ran to their caps with ~8 decisions, and update 320 trained on them. Fix: the learner drops games with AI errors ≥ max(3, days/4); watcher alerts.
- **Incident 2026-11-01 00:26:** a parallel rebuild dropped WSL memory to 1.49 GB, memwatch stopped joint7, ~35 min of games lost. Rule: `nice -n 19 ninja -j1` while a run is live.
- `git archive` of an uncommitted change silently sends nothing.
- **Incident 02:11–02:36:** the WDDM GPU spilled past 32 GB into shared RAM and crawled at "100%". Fix: cuda_memory_fraction 0.85. Then OOMs (graph pools 16.7 GB): micro-batch 64 rows; StepGraph falls back to eager on OOM; an OOM drops the update's games (93bc8f7).
- **Permit starvation (~05:50):** ahead × decisions_per_update = 8 long games < the 20 per update. Fix da4bd80/837facf.
- **User rule (2026-11-01 09:30):** "the PC should do its best training with what it has, if it needs more it can pull it from this mac or from another Mac … for a lot of time only the PC is available and it still needs to work". All inference on the PC CPU; the Mac only adds slots; the AI's "fallback store" design was wrong and reverted (fc9c277).
- **User rule (2026-11-01):** "just make sure that we can't die and lose data, if we do you failed. under this constraint, be as fast as possible". Hence: finished games spooled to disk (~21 MB each), `drain_restart.sh`, memory_allows, memwatch sheds the youngest game, a keeper unit.
- Throughput: joint7 games ~5,900 decisions (joint5 395); decisions/h ~65–75% of joint5; GPU-bound (90–93%).

### 1.48 optimizer-plan.md
- **Session** e5 · **modified** 2026-11-06 · **type** design + deploy + measurement.
- The plan (docs/optimizer_plan.md) was written by another session. User (~10:00): "implement them all (unless you see a bug or a contradiction or something bad)… I want to have smooth training with no fast and huge loss increases". That counts as approval for the value group at 1e-4, above the 3e-5 cap.
- **8c6cea9**, deployed ~v1425, all at once: value_lr_factor 10, KL target, loss_shares_freeze, trust_floor 0.4, trunk_beta1 0.95, grad_clip_by_group (calibrated over 200 steps), head_scale_by residual; named Adam groups.
- **e5's 4 fixes to the plan:**
  - the critic rate kept apart from KL's lr_scale;
  - signal_ratio's noise floor depends on beta1 (0.155 at 0.9, 0.108 at 0.95), so signal_fraction was used;
  - a per-group clip at raw medians = a ~25× jump, so rescaled;
  - slow variances for head_scale.
- Baselines (v1360–1425): unexplained 0.90, trust 0.29, teacher_ce 0.97, agreement 0.33, head EV v7/g7/vend 0.15/0.29/0.27, signal_fraction 0.10 (trunk 0.10); battle trunk signal fraction ~0 (pure noise). Evals 0.423/0.416/0.357.
- Step 0: the trunk makes ~88% of the gradient norm. **Step 2 reverted at v1473** (evals 0.441 → 0.363 → 0.335): the per-update KL varies ~5× by batch, so KL control ratchets on noise and cannot hold a "somewhat slower" actor.
- Rule: judge the fit by unexplained only; bisect in reverse (7 … 2).

### 1.49 alphastar-openai5-lessons.md
- **Session** e5 · **modified** 2026-11-07 · **type** reference + decisions.
- Paper numbers recorded:
  - AlphaStar: privileged value 22% → 82%; V-trace 49 → +TD(λ) 73 → +UPGO 82; Elo no-human 149 / supervised 936 / human init 1020 / +KL 1400 / +z 1540; league exploitability;
  - OpenAI Five: 80/20 past versions; team spirit; GAE horizon; entropy 0.01 → 0.001; lr 5e-5 → 5e-6; staleness ~8 versions = a significant slowdown; reuse 8 may stop learning; surgery; Rerun ~20% compute, > 98% win vs the surgered Five.
- Where we stood: the critic has no hidden info; no campaign past-version opponents; staleness ~1; reuse 2; clip fraction 0.19–0.25; campaign entropy 1.3.
- **User decisions (2026-11-07):** (1) privileged critic, the user's design; (2) past-version opponents only "after we are much better"; (3) **Rerun for joint8: yes**; (4) credit tricks: no; (5) entropy: much later. Past selves are allowed as opponents; NK2 is not.

### 1.50 privileged-critic.md
- **Session** e5 · **modified** 2026-11-07 · **type** design + deploy + measurement.
- User: "the critic can see the last hidden state of all players" through a layer that can learn an OR; later max pooling ("max pooling is as good as any"; the AI had argued for a per-rival layer, overruled).
- First build e9f88da: rivals' value_hidden from an earlier day plus log Φ gap, pooled max+mean, zero-init logit corrections, training only.
- **User's design 0fe2869 (~09:50 2026-11-07):** only V2/V5/V7/V_end corrected (growth heads were ~1% worse with rivals: "the % increase is mostly based off of our own data"); x = [value_hidden, max over rivals, logit of our true share]. First-design results: V_end −24%, unexplained 0.881 → 0.840.
- **privileged_credit ON 2026-11-07 14:38 CST (e620710):** gate met at 45 updates (unexplained 0.863 → 0.806, V_end −23%). Baseline evals v1753 0.444, v1776 0.462, v1796 0.398. The first attempt (14:17) rolled back on a stale check script assertion.
- **e5's prediction wrong:** "V2 falls first". For change-labels, V_end gains, not V2.
- Battle: shooter idle rose with trunk beta1 0.95 (see problem-watch-list).

### 1.51 human-recorder.md
- **Session** no id (b1 or e5 *[unclear]*) · **dates** 2026-11-07 · **type** design/reference.
- User: "save all battles and all campaign moves; we have to know who was the player and who was NK2 (or my AI)... for battles we only want the player battles... record the raw [for multi-actions]... make an option to record with player only".
- `campaign_model.record_human` (1790e5b + vcmi 3daf27f02): ALPHA_RECORD_HUMAN loads libAlphaRecorderAI beside the human's interface; segments per game start; raw.jsonl; summary audit.
- Gaps: multi-request moves raw only; garrison/teleport/select dialogs not recorded; combatAlliedAI must never be AlphaBattleAI. **Not tested with a real human session.** (schema-16 notes "the user's first recording (schema 15)" existed by 2026-11-07.)

### 1.52 problem-watch-list.md
- **Session** b1 · **modified** 2026-11-06 (content to 2026-11-07 23:34) · **type** measurement log / watch list.
- User (2026-10-29): "keep track on the things that are problematic. We'll see if they improve or if we need to make them easier to learn".
- **Baselines (joint2, campaign update 219, 24 games vs NK2):** heroes at day 56 ours 0.17 vs NK2 3.67; eliminated 17/24, won 0; score 0.227 vs d28 0.40 ("we collapse in month 2"); Φ 62k vs 318k; failures pickup_passed_travel 0.011, troops_left 0.014; value EV dips negative.
- Endless eval battle → vcmi d1a5f29bb: at round 60 the attacker retreats (user: "do it now").
- **Diagnosis 2026-10-29 21:55** (user: "did we collapse? too much noise?"): eval spread sd 0.050 < per-eval se 0.068, so noise; vs_first −0.006 ± 0.032 at 260, so **no measurable gain in 270 updates**. Eval heroes fell while self-play heroes rose ("self-play diverges from the NK2 test"). Battle signal_ratio 0.11–0.12, at or below the noise level.
- **Bug (vcmi 47c9d57eb):** generated maps lost towns' default buildings in vmap saving, so **every random-map game (2/3 of training, all evals) started with fort + village hall only: no tavern, no dwelling**. "Likely a big cause of the passive play."
- Open then: 72% of advantage variance per player-day ("credit is shared luck"); type-head drift (p(BUILD|legal) 0.93); heroes never level.
- After the hiring/level fixes: greedy eval 0.103 / d28 0.434, heroes 0.75 vs 3.51, eliminated 19/24. **move_out** layer learned +2.5 logits on monster tiles under the old Φ, so it was zeroed at v313. **The AI decided without the user's answer** (asked twice, past midnight).
- **Greedy evals measured a degenerate mode:** heroes oscillate one tile (95% one-tile moves), so evals went back to temperature 1 (6008500).
- Economy (2026-10-30): gold stuck 400–2000; 10 lone-creature hires; heroes never go home (19/3611), ARMY 0/18; the HIRE offer was relu-clipped (11e8254).
- Joint7 evals 2026-11-05: v1071 11 elim / 20 zero-hero / 0.487; v1097 13/21/0.390; v1120 19/26/0.332; v1137 28/32/0.265.
- Battle shooters idle: in hard one_big_stack endgames the whole army wanders; there is no time cost in the reward ("the user's choice"); a reward change is the user's call.
- **Dumb-decision watch (user 2026-11-06):** "are you also monitoring stupid things like no fire decisions and passing by a resource?" (the AI had not been). Baselines: pickup_passed_travel ~0.11, shooter idle vs BattleAI ~0.016.
- 2026-11-07: battle trunk beta1 0.95 raised shooter idle to 5.15% (8561); back to 0.9 (e78414f) gave 1.57% (8573), score +0.099. User's escape_strong rule (a3b6e96): w × min(r³, 8) ("uncapped = a mask").
- Troops left in mines: 0.03 → 0.33 (policy drift, low harm, no fix).

### 1.53 schema-16.md
- **Session** b1 · **modified** 2026-11-08 · **type** design + deploy + incident.
- User (2026-11-07): "make sure we can learn from all my actions. Make the changes to the model." Built 1608288:
  - ARMY target slot (a second draw), all-but-one (the user's peel-into-singles: 20 of their 54 troop moves), within-army moves;
  - MULTI SPLIT_SINGLES / MERGE_ALL / GARRISON_SWAP; unifyStacks removed; bounded priors;
  - matcher handles every ArrangeStacks; nk2_new_advice=False; teacher games 96.7% matched; 469 tests.
- **Deployed 2026-11-07 23:51 (v1929).** First games: random whole-stack swaps (58% gave away more value) and last-stack ping-pong (p 0.5). Root cause: Advice.cpp's nearestMove chose the swap-only ALL entry. Fix 60a033e: p 0.50 → 0.02.
- value_loss_fresh +40% (0.00086 → 0.00121), from duplicate stacks (6% of decisions). The user approved a partial auto-merge (7192bbd, 08:27).

### 1.54 map-size-design.md
- **Session** e5 · **modified** 2026-11-08 · **type** design (decided).
- **Decided 2026-11-08 by the user:** train on 36×36 with today's caps, small-map cost and outputs untouched, the model able to play big maps later; Design B (a fixed 36×36 block grid, side = ceil(max(w,h)/36)); plan docs/map_size_plan.md for b1. MOVE radius 16 stays (TRAVEL covers far targets).
- Measured:
  - schema 16 carries 259k scalars a decision; only tile_* grows with the map (20.7k, 8%); windows 34%; m_* 32%;
  - Mac forward 12.5 ms (CustomFeatures 30%, MapUNet 19%, TileReader 16%, so ~37% tile-bound); at 144×144 that part alone ≈ 6× today's forward;
  - live update v2047: 56.6k decisions in 230 s, GPU step-bound.
- Design A (true-size classes) kept as the fallback.

### 1.55 map-size-blocks.md
- **Session** no id (b1 *(inferred: implements e5's plan "for b1")*) · **dates** 2026-11-08 · **type** design/deploy.
- User requirements: train and run on larger maps; no slower small-map training; far things the model can use.
- Built 1527c22: encodeTileBlocks from revealed tiles (side 1 bit-identical); side embeddings with **padding_idx=1**. The AI's fix to the plan: a zero-init row would train into a bias big maps lack.
- Knobs: eval_size, big_game_every (off; the user's call), time_scale.
- Checks: golden-file test; same 36 forward time; a 144×144 game with 0 errors. Deploy via wait40 / a40_all / deploy_40.

## 2. Standing rules

How to read this section:
- **Date** is when the rule was set.
- **Trigger** is the incident or request that caused it. "—" means none is recorded.
- **Src** is the digest file.
- A rule later changed or reversed is marked → with the change.
- 2.1–2.3 hold rules the developer set. 2.4 holds operating rules the AI sessions derived from incidents and recorded as rules for themselves.

### 2.1 Process and safety rules (developer)

| # | Date | Rule (verbatim where quoted) | Trigger | Src |
|---|---|---|---|---|
| P1 | 10-21 | Every data-generating run is capped (time, disk, stall); report MB and MB/min; games only via `scripts/headless.sh`; check `pgrep` after tests; kill by PID. "you will tell me how much data you create when creating data over how much time" | 4 clients ran ~13 h, filled the 500 GB disk, crashed Docker | data-generation-limits |
| P2 | 10-21 | All project files under the repo (no /Applications, no ~/Library); GitHub repos private under ARozental | — | vcmi-setup |
| P3 | 10-22 | No GitHub Actions: "I do not want to consume github actions whatever they are". Strip `.github` workflows and dependabot from third-party code before every push and upstream merge | Dependabot PRs ran VCMI CI, ~26k minutes vs a 2k quota | no-github-actions |
| P4 | 10-22 | Minimal unwanted engine changes; VCMI bug fixes welcome; label each patch bug-fix vs behaviour. → 11-02: "obviously we are patching VCMI fork if we need to"; not an approval gate | — | minimal-engine-changes |
| P5 | 10-22 | Python owns the science. C++ only serializes raw features and masks and executes. Restated 10-28: "I do not want c++ code to determine any behaviour…". Explain the Python in depth, the C++ in a sentence. C++ safety nets must never fire in normal play | User reviews only Python | python-owns-the-science |
| P6 | 10-22 | Unattended work ranks bug fixes > dumb design mistakes > tuning; audit new areas around every experiment; ≤ 1000 epochs; have a best solution ready; report the three kinds separately | Overnight autonomy | autonomous-experiment-priorities |
| P7 | 10-23 | Reject fixes that cannot work in real training (Go-Explore and self-imitation removed; they need fixed seeds) | Fixed-seed tricks inflated results | castle-vs-inferno-findings |
| P8 | 10-24 | No growth in code or free parameters without a reason (v3 cut from +80% to +12% params); bug fixes don't count as complexity. Experiments are "a sanity check for a model architecture, we are looking for bugs, issues, inefficiencies, missing/needed features" | — | model-iteration-loop |
| P9 | 10-26 | No mid-run restarts except after a crash (user dislikes many short TB lines). → 10-27 (joint1): "fix things during the run (resume), don't wait" | — | v15-history, campaign-value-design |
| P10 | 10-28 | "we have to have 0 killers": run the value audits (`campaign_model.audit`, `battle_model.audit`) after any change to priors, Φ, rewards or features | Pre-mortem | plan-2026-10-28 |
| P11 | 10-29 | Keep a watch list of problematic things and report each trend; if one stays flat, make it easier to learn | ~220 updates is very little | problem-watch-list |
| P12 | 10-31 | **LR ≤ 3e-5, 1e-5 recommended** (cap was 1e-4); never raise it to speed up; KL control may only lower it; flat evals are bad; lower the rate near convergence. → 11-04: do not lower while the V loss falls. → 11-06: value group allowed at 1e-4 (optimizer plan) | 2e-4 unstable (10-26); imitation improved at once when lowered to 3e-5 | training-stability, optimizer-plan |
| P13 | 10-31 | Keep training nonstop, GPU never idle; the campaign is the model to train most; battle may get less GPU | — | pc-training |
| P14 | 11-01 | "the PC should do its best training with what it has…" A Mac only adds game slots; never make the run wait on a Mac | AI's Mac-first "fallback store" design | pc-training |
| P15 | 11-01 | "just make sure that we can't die and lose data, if we do you failed. under this constraint, be as fast as possible". Hence spooled finished games, `drain_restart.sh`, memory floor, keeper unit | Repeated memwatch/OOM stops lost in-flight games | pc-training |
| P16 | 11-03 | Keep `docs/model_cannot_do.md` (never-learnable gaps) in the user's order: arranging one army first, artifacts second. "This list will expand later" | — | model-cannot-do-list |
| P17 | 11-06 | Monitor dumb decisions (no-fire, passing a resource) in every experiment: "are you also monitoring stupid things…?" | AI had not been, during the optimizer deploy | problem-watch-list |
| P18 | 11-06 | "smooth training with no fast and huge loss increases" | Optimizer plan | optimizer-plan |
| P19 | 11-07 | "make sure we can learn from all my actions" (schema 16) | Human recorder | schema-16 |

### 2.2 Training-signal and fairness rules (developer)

| # | Date | Rule | Trigger | Src |
|---|---|---|---|---|
| T1 | 10-25 | Battle league: train only against our own model (85% current, 15% past snapshots), no fixed AIs; evaluate vs both BattleAI and MMAI | — | v14-design |
| T2 | 10-27 | Inputs show counts as a human sees them (VCMI quantity ids, exact only under Visions, no pile amounts); NK2's exact counts are "cheating" | — | campaign-value-design |
| T3 | 10-28 / 10-30 | **No eval AI in training games.** 10-28: no NK2 seats ("we would learn its weaknesses"). 10-30: "this is wrong/cheating. I do not want a model that learn to exploit Nullkiller2 bugs, I want a good model that I will be able to play against." 10-31: "NK2 may be a teacher, never an opponent". 11-07: our own past versions are allowed as opponents, but only "after we are much better" | A review made a third of campaign games vs NK2 | no-training-vs-eval-ais, plan-2026-10-28, pc-training, alphastar-openai5-lessons |
| T4 | 10-27 → 10-31 | Teacher: 10-27 "No NK2 teacher… Day+2 can learn quickly". **Reversed 10-31** after AlphaStar's ablations: the model must be able to follow a teacher. *(MEMORY.md: NK2's choices turned off again 2026-11-11.)* | — | campaign-value-design, teacher-and-memory |
| T5 | 10-31 | **Never mask an action to stop overuse:** "the approach of making some actions unlearnable is bad always…" | Info-spell mask: eval 0.22 → 0.07 | no-action-masking |
| T6 | 10-31 | A memory model is required: "no long runs with no memory model"; no no-memory or width A/B ("no memory means we lose"); stream teacher games instead of recording | — | teacher-and-memory |
| T7 | 11-02 | NK2 internals may only affect the loss. The model must run on original VCMI, so NK2's plan is a label, never an input | NK2 plan head | changes-2026-11-02 |
| T8 | 11-02 | Teacher over a turn: order inside a turn must not matter; a target must have been visible when used; the exact-strength leak is "not important". *(The order assumption was proved false for a town's visitor on 11-05, teacher-label-changes.)* | — | changes-2026-11-02 |
| T9 | 11-04 | A prediction head predicts **the engine's answer only**: "predicting our own action is silly" | Battle head design | move-feature-audit |
| T10 | 11-05 | **Summon Boat: no cheating.** No legality from hidden information; failures the model can learn from raw data are fine | CAST no-effects | review-2026-11-05 |
| T11 | 11-05 | HotA: ids matched and verified at engine start; HotA content worth 0 "needs to be fixed before we even think on training with it"; "embeddings … can start at random, their values must be true though" | Transfer goal | hota-transfer |
| T12 | 11-07 | AlphaStar/Five decisions: privileged critic yes (user's design); Rerun from scratch for joint8 yes; credit tricks no; entropy much later | Paper review | alphastar-openai5-lessons, privileged-critic |

### 2.3 Design rules (developer)

| # | Date | Rule | Src |
|---|---|---|---|
| D1 | 10-22 | Raw engine values only as inputs; every transform is an in-graph op | battle-model-design |
| D2 | 10-22 | Every probability or 0..1 quantity is a sigmoid/softmax trained with CE (10-28: "CE learns far better than MSE") | battle-model-design, plan-2026-10-28 |
| D3 | 10-22 | An illegal action must never waste a turn: "We need to take the first viable action" | battle-model-design |
| D4 | 10-22 | Never use the naive board flip (it breaks 280/888 adjacencies) | battle-model-design |
| D5 | 10-24 | Escape prior: "batch 0: do not surrender, but if much weaker consider, do not force". 11-07: escapes get w × min(r³, 8), capped because "uncapped = a mask" | model-iteration-loop, problem-watch-list |
| D6 | 10-25 | Experience stays out of the battle reward (symmetry; map gains aren't counted either) | v14-design |
| D7 | 10-26 | Credit uses a linear round window, no exponential decay: "a wait's return must not be decayed" | v15-history |
| D8 | 10-27 | Φ labels are empire shares; a creature = cost+u, no accrued growth ("the AI must learn the calendar"); tiles 1 gold ("10 is too much"; → 5 on 11-03) | campaign-value-design, changes-2026-11-03 |
| D9 | 10-28 | One model for every map size (→ 11-08: Design B; small-map training unchanged) | plan-2026-10-28, map-size-design |
| D10 | 10-28 | Hand-made features as small named functions with a readable forward; no fixed default logit effects (zero-init learned paths); NK2 plans as "weak features, easy to ignore"; all of NK2's brain as feature ops | plan-2026-10-28 |
| D11 | 10-28 | Greedy-eval shooting must be 0.99+; shooter idle > 0.01 prints FAILURE | plan-2026-10-28 |
| D12 | 10-29 | Battle value priors are **added** to logits, never multiplied; no search or simulation | plan-2026-10-28 |
| D13 | 10-29 | Credit: a day+1 head is allowed; same-day / end-of-day values only if all else fails (flagged suspicious); post-action values are a no-go | hero-value-design |
| D14 | 10-29 / 11-03 / 11-05 | Hero value: battle by level 2000·min(L,3) per level-up; campaign by skills (main hero full, others half); an empty hire ~−1k; Φ unit = a starting hero (U = 0.953); recruiting at list price is worth nothing | hero-value-design, changes-2026-11-03, multi-actions-and-phi-unit |
| D15 | 10-30 | v2 architecture: named blocks; `encode` takes raw inputs; known facts joined late ("a single number like P(win) must not be swallowed"); ownership strongly transforms vectors; scale is relative; a logit is a match (query·key + small net + prior); foresight is auxiliary only; auxiliary targets bounded and scale-free | campaign-model-v2-draft |
| D16 | 10-31 | Share heads learn the change in logit share | teacher-and-memory |
| D17 | 11-04 | "every trade is dumb unless we bought something that could not have been bought without making this trade"; no network-set trade amounts; C++ calculation with no magic numbers | trades-and-guards, fix-batches-2026-11-04 |
| D18 | 11-04 | Every option has its own identity (the shared YES vector was "retarded"); expert rules in their own functions | question-head |
| D19 | 11-05 | Composites are options and the price decides (no masking); no "gather-once"; "add a big minus [to disbanding]"; usually-useless actions "can be very negative" at the start; no more Φ changes for a while | multi-actions-and-phi-unit, teacher-label-changes |
| D20 | 11-07 | Privileged critic: rivals' hidden states, max-pooled ("max pooling is as good as any"); only share and V_end heads corrected | privileged-critic |

### 2.4 Operating rules the AI sessions derived for themselves (from incidents)

- **Restarts:** wait on `kill -0 <pid>`, never a fixed timeout (a 120 s wait started a duplicate trainer, 10-29). Wait on a pid, not on `pgrep -f`. Never `pkill -f`/`pgrep -f` a pattern that appears in your own command line (11-02; *MEMORY.md: recurred 11-12, a test pair lost 04:30–07:04*).
- **Builds:** on the PC while live, `nice -n 19 ninja -j1` (-j24 tripped memwatch 11-02; parallel rebuild stop 11-01). Stop recorders and runs before schema changes (10-31: 82 empty games). Stop or fast-forward the Mac worker before a Mac C++ build (11-02).
- **Deploys:**
  - deploy only via `drain_restart.sh`, and batch them (10 restarts cost ~10–15 min each, 11-05);
  - never `git archive` an uncommitted change (it silently sends nothing);
  - never scp same-named files to one directory;
  - the e5 ff-only merge ships whatever main holds;
  - test copies need `vocab.json` and game_values;
  - smoke-test start/stop/resume on the Mac before startup changes.
- **Measuring:**
  - compare instructions retired on a shared machine; time on a quiet machine, alternating old and new (10-30);
  - judge GPU gains by decisions/s and GPU busy time, not nvidia-smi % or CPU-side clocks (10-31, 11-03);
  - count PC changes in GB against the 4.5 GB floor (11-03).
- **Evals:**
  - one canonical eval map set on every machine (10-30);
  - judge a change by 2–3 evals, never one (11-06; *MEMORY.md 11-12: one PC eval is ±0.11, one Mac arm ±0.08*);
  - compare weights on the same code with a paired mini-eval before blaming code (11-05);
  - judge behaviour by the eval vs NK2, never by self-play metrics (10-31).
- **Teacher changes:** run `type_drift.py` after any teacher-label or prior change; stop on a teacher_ce jump > ~0.3 or two KL-stopped updates in a row; after adding an action, measure NK2's label mix per decision kind; accept sets are taught (rich-get-richer) (11-05).
- **Code hygiene:** verify "dead code" by how code is reached (registries, decorators), not by name grep (11-01). Save checkpoints atomically (11-04). Read macOS crash reports first for any segfault; use `VCMI_LLDB=1` to reproduce (10-26, 11-01). Check the VCMI/NK2 source before claiming NK2 behaviour (11-04).
- **Seeds:** never trust "seed = same map" (RMG, 10-30), nor the determinism of BattleAI (time-seeded until 10-29).

## 3. Incidents and failures

Each row gives cause → detection → impact → fix → lesson. Dates are 2026; times are local (CST/CDT). The "Src" column names the digest file.

### 3.1 Operations, infrastructure and data loss

| Date | Cause | Detection | Impact | Fix | Lesson | Src |
|---|---|---|---|---|---|---|
| 10-21 | Headless clients never exit; `pkill -f build/bin/vcmi` mismatched `./vcmiclient`; trace logs at 300 MB/min into one shared log | Disk full | 500 GB filled over ~13 h; Docker crashed | `headless.sh` caps; info logging (~7 MB/min) | P1 | data-generation-limits |
| 10-22 | `.github` workflows and dependabot left in the pushed VCMI copy | Quota exhausted | ~26k billed minutes (upper bound), 100% of the quota | Actions disabled; files stripped | P3 | no-github-actions |
| 10-24 | MPS compiled one graph per batch size | Trainers killed, swap full | Runs died at 9+ GB | Fixed GPU shapes (d6df406), ~5 GB flat | Fixed shapes on MPS | model-iteration-loop |
| 10-25 | Parallel map builds raced on `modSettings.json` (VCMI truncates and rewrites it) | Identical eval rows | MMAI silently replaced by BattleAI (eval 545's MMAI half) | Per-process builder dirs; evals refuse an inactive MMAI | Silent fallbacks need asserts | v14-design |
| 10-26 | Real-game model cache keyed by link | Infra audit | Half of training data ~80 updates old | Fixed in inference rebuild | Check data staleness | training-infra |
| 10-27 | League names v50 vs v00050 | Adversarial review | League opponents didn't load (v15-window flat 750–860) | 3263898 + `ALPHA_BATTLE_REQUIRE_MODEL` | Fail loudly on a missing model | review-2026-10-27, v15-history |
| 10-29 | Fixed 120 s wait after a TERM | Duplicate trainer seen | Two trainers on one run dir (stopped before training) | Loop on `kill -0` | 2.4 | training-infra |
| 10-30 | PC-only bugs: select() on fd ≥ 1024; one-row decode stride; whole-game decompress; minibatch decode in pinned memory (learner 9.5 GB) | Run failures, memory guard | Stops | e505aba, 7f9bda6, e7014c1, 7551903 | New machine = new bug class | pc-training |
| 10-30 23:27 | 44 loader workers + trainer ~9 GB | memwatch stop | Run stopped | gc.freeze, malloc_trim, floor 4.5 GB | RAM is the PC's limit | fast-training-design |
| 10-31 | scp of two `train.py` files into one dir | Battle trainer failed to start | Restart lost | Separate copies | 2.4 | fast-training-design |
| 10-31 | Schema rebuilt while a recorder ran old Python | 0-decision games | 82 games wasted | Stop runs first | 2.4 | teacher-and-memory |
| 10-31 13:09 | Mac worker restart refused (old one still exiting), so no campaign inference | Games with ~8 decisions | 4,312 empty games; update 320 trained on them | Learner drops games with many AI errors (3c7990e); watcher alerts | Guard training against degenerate data | pc-training |
| 11-01 00:26 | Parallel C++ rebuild beside the run | memwatch (1.49 GB free) | ~35 min of games lost | `nice -n 19 ninja -j1` | 2.4 | pc-training |
| 11-01 01:47 | Loaders holding long games (12.7 GB) + learner backlog + in-process battle eval | memwatch (1.4 GB) | Stop | 13 games | Peaks coincide | pc-training |
| 11-01 02:11 | WDDM spills past 32 GB VRAM into shared RAM instead of failing | Update not done in 25 min | v8 lost 3× across restarts | `cuda_memory_fraction 0.85`; 64-row micro-batches; eager fallback on OOM; an OOM drops the update's games | Windows GPU fails slow, not loud | pc-training |
| 11-01 05:50 | Permits = ahead × decisions_per_update = 8 long games < 20 per update | 0 games playing for 15 min | GPU idle | Update worth = max(decisions, min_games × mean) | Throttles must count games and decisions | pc-training |
| 11-02 | `alpha_heroes-e5/build` symlinked the main build | Damaged games (4–40 decisions) | ~8 Mac games silently damaged | `build-own` clone + `mac_worker_build.sh` | Isolate live binaries | changes-2026-11-02 |
| 11-02 11:22 | -j24 NK2 rebuild during a drain | memwatch | Graceful stop, nothing lost | -j4 / -j1 | 2.4 | changes-2026-11-02 |
| 11-02/03 | Forkserver preload never applied (the labeler's pool started the forkserver first) | Per-worker memory audit | Every loader imported torch: 9.5 GB for 25 workers; capped games/h | 78d55e4 forkserver.context(); OPENBLAS threads 1 | A "fix" (f398aec) can silently not apply; verify in the live process | training-speed-2026-11-02 |
| 11-03 | A rare over-budget micro-batch recorded an all-slot graph pool (~6 GB) | GPU memory reserved 22.4 GB; loss-noise OOM | Near-OOM | Such steps run eagerly (0321bec) | Graph pools persist | training-speed-2026-11-02 |
| 11-04 | Copy of checkpoint_latest taken mid-write | Unreadable file | — | Atomic saves (50678d1) | — | trades-and-guards |
| 11-04 | e5's ff-only merge took whatever main held | C1 on the Mac before the PC | Code skew | Commit only what ships | 2.4 | fix-batches-2026-11-04 |
| 11-05 | 10 restarts (one per deploy) | Throughput audit | Hourly 155–212 vs steady 226 decisions/s | Batch deploys | — | training-speed-2026-11-02 |
| 10-30, 11-06 | TensorBoard: no reload_multifile; random 1000-point sampling | Empty or holey charts | 4.5 h blind; ~30% of points dropped | `--reload_multifile=true`, `scalars=5000` (never 0) | — | pc-training |

### 3.2 Engine, map and environment bugs

| Date | Cause | Impact | Fix | Src |
|---|---|---|---|---|
| 10-21 | Stock 1.7.5 headless crashes (ENGINE null) | No headless play | Cherry-picks fdb1aa216, 3dfa6763f | vcmi-setup |
| ≤10-23 01:40 | Map tool gave heroes default skills (Caitlin 62 mana; Calid knew Haste) | Early castle runs invalid | Fixed | castle-vs-inferno-findings |
| 10-24 | Arrow towers (off-field hexes) threw in encodePairDamage | StupidAI played every siege | fd41244 | model-iteration-loop |
| 10-24 | Battle-only mode applied the random starting bonus | 1/3 of battles had a free artifact or gold | vcmi 43fe4c2d2 | model-iteration-loop |
| 10-24 | h_spells ignored tomes/scrolls; mastery from school skills | Wrong spell inputs | Contract v13 (f84d9a0) | model-iteration-loop |
| 10-24/25 | Three battle-only hangs (winner levels up; towers end a siege in onHeroVisit; StupidAI stalemate) | gen-v13 stopped 3× | VCMI fixes; retreat from round 50 (66b012f) | model-iteration-loop |
| 10-26 | NK2 data race (exit 139); unreachable walk (exit 134) | Real-game crashes | vcmi 400b76421, d8c48ba91 | v14-design |
| 10-27, 10-29 | No round limit in VCMI; BattleAI never retreats; our model kited as monsters | Endless eval battles; stall-kills of 30–50% of campaign games; 700 MB record sends | Skip in evals (b3de8c8); neutrals by BattleAI in training; ROUND_LIMIT 50; defender quiet-round retreat; at round 60 the attacker retreats (vcmi d1a5f29bb) | review-2026-10-27, plan-2026-10-28, problem-watch-list |
| 10-27 | BattleAI RNG time-seeded | Evals not repeatable | 10-29: seeded by battle state (198/200 deterministic) | review-2026-10-27, plan-2026-10-28 |
| joint1 | The engine ended games on red's loss; townless generated maps | Corrupted campaign games | Fixed (docs/campaign_ai.md §9) | campaign-value-design |
| 10-29 | vmap saving dropped towns' default buildings | **Every random-map game (2/3 of training, all evals) started without tavern or dwellings**; likely a big cause of passive play | vcmi 47c9d57eb | problem-watch-list |
| 10-30 | Area-attack targets in pointer order (`std::set<const CStack*>`) | ~1 in 10k attack values differed; replays nondeterministic | vcmi e1be37e4a | perf-2026-10-29 |
| 10-30 | RMG gives a different map for the same seed | Mac and PC eval sets differed (0/1296 tiles match); joint5 "drop" 0.134 → 0.063 | One canonical .vmap set | rmg-nondeterministic |
| 11-01 | NK2 advisor read getTile() == nullptr for an unseen tile | 4/25 games segfaulted | vcmi 823c80710 | teacher-and-memory |
| 11-02 | oneTBB spawns one worker per hardware thread; NK2's parallel_for wakes all | 60% of game CPU was spin | VCMI_MAX_WORKERS=4 (2.9× less CPU/turn) | training-speed-2026-11-02 |
| 11-05 | Hidden building 20 "ship" passed canBuildStructure | 569 BUILD no-effects | f75ea35 | review-2026-11-05 |

### 3.3 Encoder, input and labelling bugs (silent learning killers)

| Date | Bug | How found | Effect | Fix | Src |
|---|---|---|---|---|---|
| 10-27 | Φ = 0 for the winner as well as losers | — | Decisive games' last labels 1/n for all | Fixed | campaign-value-design |
| 10-28 | NK2 score units mixed with Φ gold | Value audit | Pickups 3.8× underpriced | Φ gold | plan-2026-10-28 |
| 10-28 | summarize removed destroyed war machines from the army at 2500 | War-machine work | Wrong battle outcome values | Fixed | war-machine-value |
| 10-29 | Hero levels 1–3 worth 0 in battle value | Analysis | No campaign hero ever levelled | 552625f | hero-value-design |
| 10-30 | HIRE type offer relu-clipped | Eval-style games | Bad hires never held back | 11e8254 | problem-watch-list |
| 11-01 | NK2 sends one MoveHero per tile, so the label = a one-tile step | Agreement analysis (tile match 5%) | Taught "walk one tile per decision" (~60 decisions per player-day) | Label the plan target (5e72edc). **Teacher.cpp (imitation recorder) still one-step**, and the imitation checkpoint learned it | teacher-and-memory |
| 11-04 | `blocked_by_guards` ("behind") always 0 (VCMI marks every object tile blocked) | Feature audit | Feature dead | ab52217 | move-feature-audit |
| 11-04 | `features.bucket` hashed 546 of 954 options onto shared rows; the YES token was the same for every question | e5 peer; user | Global NO bias of 4–6 logits; 53-ask loops; dwellings refused 467/481 | Question-head redesign (d60df0b, 909067b) | trades-and-guards, question-head |
| 11-04 | `object_gold(HERO) = −760` charged on flee/join fights | Post-deploy analysis | −2.5 logits instead of +0.4 | 3b1424d | trades-and-guards |
| 11-05 | arrange() stranded one creature; duplicate slots; GIVE_ALL/GATHER priced as both armies whole; RECRUIT_ALL priced non-fitting levels; state fingerprint lacked artifacts | Counters, review | Wrong prices; futile EQUIPs | 40c2b89, 3d712b2, 34bac74, fingerprint fix | multi-actions-and-phi-unit, review-2026-11-05 |
| 11-05 | **Neutral towns never encoded** (getTownsInfo walks only players' towns) | Hopeless-attack census | Heroes walked into neutral garrisons at p up to 0.73 | 296b06a | neutral-towns |
| 11-05 | Fog-adjacent guards priced 0, plus exploration value on top | Same census | 5 of 46 hopeless attacks | 635912f | neutral-towns |
| 11-06 | checkpoint_<v>.pt saved at eval end | Rollback | "01288" held v1293's weights | c4241a1 | teacher-label-changes |
| 11-07/08 | Advice.cpp nearestMove used the swap-only ALL entry for a last stack | First schema-16 games | Last-stack ping-pong at p 0.5; random swaps | 60a033e (p → 0.02) | schema-16 |

### 3.4 Learning-dynamics and design failures

| Date | Failure | Cause | Detection | Fix / status | Lesson | Src |
|---|---|---|---|---|---|---|
| 10-23 | Defence passivity (42% wait/defend) | Final V chained onto V0..V2 | credit metrics | `chain_final_value=False` | Don't let the main value inherit auxiliary swings | castle-vs-inferno-findings |
| 10-23 | Self-imitation alone: defence 0% | Locks onto the best losing line; entropy collapse | — | Dropped | — | castle-vs-inferno-findings |
| 10-24 | R7 regression | `score_now` fed into the battle token | Ablation | Removed | **Never feed the current score as a feature** | model-iteration-loop |
| 10-24 | Fixed-seed overfit (shooters moved; lost 16/16) | Two fixed battles | Eval | Random seeds only | — | model-iteration-loop |
| 10-24/25 | ~40% retreats in self-play | Hero assets ≈ army value | Logs | Escape prior | — | model-iteration-loop |
| 10-26 | Shooter idle 0.03 → 0.53 | Joint entropy weighted with p(type) gradient favours many-option types | Replay of two checkpoints | 0fc65e0 (detach) | Regularizers can carry hidden action preferences | v14-design |
| 10-28/29 | Battle drift from shooting (p(SHOOT) 0.99 → 0.32; greedy idle to 0.49) | Entropy bonus/floor coherent (0.96) vs a policy gradient at noise level (~0.2) | gradient_study.py | Entropy floor and weight average off; additive value priors; battle frozen at 3078 | With a noise-level signal, small consistent biases win | plan-2026-10-28 |
| 10-27 → 11-03 | Campaign self-play never attacks (2/1223, 2/1253 eliminations; 599/600 games to the cap) | No pressure in self-play; Vᵏ/ΣVᵏ pays a lead | Logs vs NK2 evals (~75% eliminated) | Open; NK2 seats rejected (T3) | Self-play ≠ the test | campaign-value-design, plan-2026-10-28, changes-2026-11-03 |
| 10-29 | No measurable gain in 270 updates (vs_first −0.006 ± 0.032) | Shared luck (72% of advantage variance per player-day); trust gate failed → MC | Diagnosis | Many fixes after | — | problem-watch-list |
| 10-30 | move_out learned +2.5 logits on monster tiles | Learned under the old Φ (a weak hero's death was free) | Hero-loss census | Zeroed at v313 (AI decided without the user) | Learned layers keep old-reward habits | problem-watch-list |
| 10-30 | Greedy evals measured a degenerate mode (95% one-tile oscillation) | Argmax of a flat policy | Eval-style games | Evals at temperature 1 (6008500) | Greedy ≠ trained policy | problem-watch-list |
| 10-31 | Info-spell mask | Self-play credit said they were worthless | Eval 0.22 → 0.07 | Reverted (567c7d5) | T5 | pc-training, no-action-masking |
| 11-01 | KL early stop after 2–6 of ~56 steps; LR 1e-5 → 3e-6 | The teacher's supervised pull drove the KL | Update logs | kl_target × (1 + teacher_weight) | PPO's KL target assumes a pure-RL loss | teacher-and-memory |
| 11-01 | EXCHANGE = 55% of decisions; 5k–25k decisions a game | Untaught action, up to 32 moves | Stats | 8-move cap + adviseExchange | — | teacher-and-memory |
| 11-03/04 | Suicide attacks (224/30 games, 97% lost, 54× stronger) | Weak PPO credit (the death drowned in 7-day noise); zero-init move_fight inert at lr 1e-5 (≤ 0.003/h) | Battle census | Strength features (ab52217) + bounded safety prior (5805895): ~112/30 games | **Zero-init layers can't learn at a capped LR; use bounded priors** | changes-2026-11-03, move-feature-audit |
| 11-04 | Trade leak: 135 trades a game, 77% pointless, ~213k Φ a game | No price | Audit | Trade prior; purchase trades priced (E) | — | trades-and-guards, fix-batches-2026-11-04 |
| 11-05 | GIVE_ALL ping-pong (300 a game) | Φ gain alone picked the receiver | Drift alert | Level-first receiver + cost (2218ada) | — | multi-actions-and-phi-unit |
| 11-05 | **GATHER emptied towns** (garrisons 80% → 21% of army; empty towns 31%; eval 0.265) | Always-accepted teacher entry + immediate Φ gain; no defence | Eval + ledgers | Danger cost; rollback v1125; teacher drops GATHER | Immediate shaping + accept sets compound. *(MEMORY.md: a GATHER collapse recurred 2026-11-10.)* | multi-actions-and-phi-unit, teacher-label-changes |
| 11-05 | DROP disbands doubled (~12k gold destroyed a game) | MULTI moved out of the teacher spans | Counters | disband_cost 6 | — | teacher-label-changes |
| 11-05 | value_loss_fresh 0.001 → 0.008 (schema 14); growth EV 0.5 → 0.24 | Data shift | Metric | No LR ramp (AI declined the user's idea) | — | training-stability |
| 11-06 | KL target step reverted (evals 0.441 → 0.335) | Per-update KL varies ~5× by batch | Evals | 53c1df7 | KL control ratchets on noise | optimizer-plan |
| 11-07 | Battle shooter idle 1–2% → 5.15% | Battle trunk beta1 0.95 | Eval split | beta1 0.9 (1.57%) | — | problem-watch-list |
| 11-07 | Troops left in mines 0.03 → 0.33 | Policy drift (no deploy) | Alert | None (low harm) | — | problem-watch-list |

### 3.5 Teacher-label collapses and rollbacks (2026-11-05/06)

| # | Time | Change | Signal ignored or seen | Eval | Action |
|---|---|---|---|---|---|
| 1 | 11-05 00:58 → 01:46 | adviseGather/Visit/artifacts replaced NK2's label at ~40% of turn decisions (labels at p ~0.0002) | teacher_ce 1.1 → 2.4 and KL stops read as "expected"; type KL 2.3 | v1037 0.248 (−0.129) | Revert 26bc2a1; v1025 → 1044; quarantine |
| 2 | 11-05 10:39 | Schema 14 GATHER, always accepted, plus Φ gain | Mini-eval: v1157 59% of towns empty (0.274) vs v1125 24% (0.390) | 0.265 / 0.256 | v1125 → 1160; teacher drops GATHER |
| — | 11-05 16:40 → 17:15 | 0d54e0f teacher span split | type_drift 0.068 vs 0.029 baseline in 3 updates | — | 1463891 |
| 3 | 11-05 ~20:00 | Slide since v1184 (DROP disbands, GATHER rise, span drift) | Paired mini-eval −0.159 ± 0.117 | v1248 0.286 | "checkpoint_01184" restored (it held v1186's weights, the c4241a1 bug); daac154 strong negative priors |
| 4 | 11-06 03:33 | Score-driven teacher weight loosened 4.22 → 2.62 after the lucky 0.476 | Agreement 0.41 → 0.33 | 0.476 → 0.402 → 0.297 | v1288 → 1331; weight 9.33 |

Common causes: shared type parameters; rich-get-richer accept sets; the teacher's labels shift with new actions; eval noise (SE ~0.05, *later shown to be ±0.11 per PC eval*) feeding back into a training hyperparameter.

### 3.6 Measurement and reasoning errors by the dev AIs (recorded in the files)

- Imitation start predicted at 0.3–0.6, measured 0.113 (10-31).
- joint5 predicted at 0.2–0.4 after 12 h. Measured v174 0.212 and v203 0.204 near the 12 h mark: the bottom edge of the range, flat, and on a different map set. joint5 ended at v357 0.099, so the 48 h target (0.4–0.8) was never reached.
- "7× edge" guess for the one-day ladder: wrong (6.1× only for edges gone within a day).
- b1 blamed CUDA launches for a 2.3 s step; it was WDDM time-slicing.
- convert_in_graphs estimated at 25–35%, measured ~7%.
- "380 KB of inputs unread", corrected to 102 KB.
- move_fight ceiling first quoted as 1.6/h, corrected to ~0.06/h.
- The VCMI pathfinder was claimed to pass guarded tiles; it doesn't.
- NK2 was claimed to answer dwellings NO; it answers YES.
- "war machines are scripted": wrong (they were model actions).
- e5 predicted "V2 falls first" for the privileged critic; V_end fell.
- e5's KL-target step (11-06) reverted.
- The value "worsening" (10-30) was a misread of per-update EV on close outcomes.
- "Garrisons ×0.8" broke the zero-price invariant (the user meant hero value).
- The AI's "accept sets are safe" claim was corrected the same day.
- The fixed-120 s restart wait.
- The Mac-first fallback store design.

## 4. Measured results

Comparability warnings:
- **Battle metric definitions:**
  - "vs baseline" (≤ R3) = our reward minus BattleAI's on the same seed and side;
  - "total_score" (from R4) = the zero-sum eval reward vs BattleAI;
  - the v12 hero value changed 18/32 rewards;
  - the eval sets grew 2026-10-27 (castle 4 → 16, hard 8 → 16) and again 2026-10-31 (×3).
- **Campaign eval definitions:**
  - joint1: share × players at day 28, 12 then 24 maps;
  - joint2 onward: day 56, 24 games, then 48 games from 10-31 11:46;
  - temperature 1 since 6008500 (greedy evals 299/310/321 are unrepresentative);
  - Φ changed at d7a92a2 (v3, 10-30) and a920331 (11-03); from 11-05 evals use a frozen EVAL_EMPIRE (eval version 3);
  - the eval map sets differ between the Mac (cached 10-29 23:11) and the PC (joint5 v37+).
- **Noise:** in the files' own estimate, per-eval SE is ~0.04–0.07 (campaign) and battle run-to-run is ±0.01. *MEMORY.md (11-12) later found a single PC eval varies ±0.11.*

### 4.1 Battle model

| Date | Run / checkpoint | Metric | Value | Src |
|---|---|---|---|---|
| 10-23 | basic castle_vs_inferno, PPO with fixes | win rate attack / defence | 98% / 83% (500–800 epochs) | castle-vs-inferno-findings |
| 10-23 | Go-Explore + self-imitation | both sides win | ~100 epochs (fixed seeds; removed) | same |
| 10-23 | single-seed generalisation | unseen seeds won | attack 16/18, defence 0/18 | same |
| 10-23 | BattleAI vs BattleAI (upgraded) | attacker wins | 24/24 | same |
| 10-23 | run B ep. 1000 | vs BattleAI defence / attack | +0.11 / −0.08 | same |
| 10-23 | search in eval (16 vs 128 sims) | attack vs BattleAI | +0.62 → +0.28 (128 worse) | same |
| 10-23 | 400-ep A/B/C/D (att/def, ep. 300–400) | vs baseline | base −0.24/+0.01; randomseeds −0.18/+0.05; v2adv −0.09/+0.10; lowentropy −0.51/+0.09 | same |
| 10-24 | up-ppo (v2) | total_score, evals 200–400 | +0.074 (target +0.069 mean) | model-iteration-loop |
| 10-24 | R1 / R2 (fixed / random seeds) | mean | +0.042/+0.020; +0.036/+0.081 | same |
| 10-24 | R4 up-zs / up-zs-rs | total_score | +0.112 / +0.094 (ends +0.138) | same |
| 10-24 | R5 up-zs-rs-lin / R6 seed 2 | total_score | +0.118 / +0.104 | same |
| 10-24 | R7 v11 / R8 v12 | total_score | +0.052/+0.027; +0.099/+0.102 | same |
| 10-24 | R9 up-v12 400–1000 | windows | +0.122, +0.116, +0.135 (defence +0.26) | same |
| 10-24 | OOD 40 generated (seed 200) | ep. 400 / 1000 | −0.091 / −0.081 (neutrals −0.14) | same |
| 10-24 | up-v12 on 32 generated | total_score | −0.126 | same |
| 10-25 | gen-v13 (generated, 4000 ep.) | eval | 0 −0.163 → 1000 +0.088 → 4000 +0.089 | same |
| 10-25 | fresh 40 (seed 300) | ep. 1000 / 2000 / 4000 | +0.064 / +0.081 / +0.098 (up-v12: −0.048) | same |
| 10-27 | v15-window | TOTAL | 853 +0.025, 955 +0.034, 1147 +0.029, 1426 +0.043, 1621 +0.058, 1718 +0.052 | v15-history |
| 10-27 | checkpoint_02106 | eval | +0.067; joint1 later 0.064–0.082 | campaign-value-design |
| 10-28 | gradient_study (2840) | policy-gradient coherence / entropy coherence | ~0.2 (= noise) / 0.96 | plan-2026-10-28 |
| 10-28 | p(SHOOT) median, 2840 → 3139 | — | 0.99 → 0.32 | same |
| 10-29 | weight average vs live | paired | −0.0075 ± 0.0025 | same |
| 10-29 | 3319 vs 3078 | paired | +0.015 ± 0.008 | same |
| 10-29 | joint2 4167 (BattleAI / MMAI) | score_neutrals; shooter idle | +0.039/+0.013; 0.029/0.042 (was 0.23/0.45 at 3912) | problem-watch-list |
| 10-29 | battle signal_ratio | median \|m\|/√v | 0.11–0.12 (noise ~0.155) | problem-watch-list |
| 10-30 | battle3 5518 → 5809 → 5829 | vs BattleAI / MMAI | +0.090/+0.092 → +0.089/+0.079 → **+0.096/+0.096** | models-v2-progress |
| 10-30 | foresight vs base rate | improvement | dies 14%, health lost 6%, dealt 8%, enemy on hex 68% | same |
| 10-30 | joint5 battle | eval | 0.096 → 0.091 (6001) → 0.082 (6165) → 0.078 | models-v2-progress, pc-training |
| 10-31 | 7068 / 7186 | eval | +0.095 / +0.099 | pc-training |
| 11-06 | battle trunk | signal fraction | ~0 (pure noise); values 0.16 | optimizer-plan |
| 11-07 | 8561 → 8573 | shooter idle vs BattleAI; score | 5.15% → 1.57%; +0.099 | problem-watch-list |

### 4.2 Campaign model vs Nullkiller2

| Date | Run / version | Score (± SE) | Other | Src |
|---|---|---|---|---|
| 10-27 | untrained (d28 metric) | 0.24–0.29 ± 0.08 | eliminated 6/12 | campaign-value-design |
| 10-27/28 | joint1 | d28 0.29 → ~0.43, flat from v80 | self-play eliminations 2/1223 | same |
| 10-29 | joint2 v219 (d56) | 0.227 (d28 0.40) | heroes 0.17 vs 3.67; Φ 62k vs 318k; eliminated 17/24 | problem-watch-list |
| 10-29 | joint2 v1–260 | sd 0.050 < SE 0.068; v260 0.118; vs_first −0.006 ± 0.032 | no measurable gain | same |
| 10-30 | greedy v310 | 0.103 (d28 0.434) | eliminated 19/24 (greedy, not representative) | same |
| 10-30 | joint5 Mac maps | v1 0.086 → v17 0.171 ± 0.056 → v36 0.134 ± 0.046 | d28 0.33 → 0.49 | models-v2-progress |
| 10-30 | joint5 PC maps | v37 0.100, v39 0.063 | map set changed | rmg-nondeterministic |
| 10-31 | joint5 fast loop | v47–59 0.18/0.249; v59 0.249 ± 0.08; v174 0.212; v203 0.204; v229 0.267 (true 0.219) | eliminated 13/24 at v229 | fast-training-design, pc-training |
| 10-31 | after info-spell mask | v243 0.072 (≈0.07) | eliminated 20/24 | pc-training |
| 10-31 | after revert | v278 0.116; v357 0.099 (−0.002 vs start) | — | pc-training, teacher-and-memory |
| 10-31/11-01 | joint7 evals 1–24 (d56 / d28) | 0.113/0.357, 0.130/0.331, 0.109/0.220, 0.101/0.386, 0.121/0.457, 0.159/0.501, 0.087/0.477, 0.154/0.527, **0.221 ± 0.042 / 0.570** | eval 24: eliminated 27/48, Φ 70k | teacher-and-memory |
| 11-04 | v832 / v854 | 0.387 / 0.444 | 12 eliminations at v854 | move-feature-audit |
| 11-04/05 | v967 → v1037 | 0.506, 0.449, 0.433, 0.378, **0.248** (collapse) | — | fix-batches-2026-11-04 |
| 11-05 | v1051, 1071, 1097, 1120 | 0.391, 0.487, 0.390, 0.332 | eliminated 11/13/19 at 1071/1097/1120 | fix-batches, problem-watch-list |
| 11-05 | v1137, v1152 (schema 14 GATHER) | 0.265, 0.256 | 28/48 eliminated | multi-actions-and-phi-unit |
| 11-05 | v1169, 1184, 1218, 1234, 1248, 1258 | 0.349, 0.382, 0.312, 0.340, 0.286, 0.365 | — | teacher-label-changes |
| 11-05/06 | v1272, **1288**, 1304, 1325 | 0.391, **0.476 ± 0.062**, 0.402, 0.297 | eliminated 18/15/15/21; heroes 1.27 vs NK2 4.01 at 1288 | neutral-towns, teacher-label-changes |
| 11-06 | v1342, 1361 | 0.342 ± 0.049, 0.423 ± 0.059 | — | teacher-label-changes |
| 11-06 | v1360–1425 baseline | 0.423 / 0.416 / 0.357 | — | optimizer-plan |
| 11-06 | after KL-target step | 0.441 → 0.363 → 0.335 | reverted v1473 | optimizer-plan |
| 11-07 | before privileged_credit | v1753 0.444, v1776 0.462, v1796 0.398 | — | privileged-critic |

### 4.3 Teacher and value-fit metrics

| Date | Metric | Values | Src |
|---|---|---|---|
| 10-30 | joint5 pooled value EV | 0 → 0.21 (per-update EV negative on close outcomes) | models-v2-progress |
| 10-31 | imitation held-out loss / top-1 | first period 3.36 / 19% → ~1.93 / ~37%; 1e-4 → 3e-5 gave 1.985 → 1.907 | teacher-and-memory, training-stability |
| 11-01 | joint7 teacher_ce on our states / agreement | 5.86 / 7% (v2) → 3.62/14% (v4) → 2.55/16% (v6) → 2.09/28.9% (v17) → 1.8/33% (eval 24) | teacher-and-memory |
| 11-01 | value_ev_d7 | −1.55 → +0.09 (v5–v9) | same |
| 11-01 | decisions per player-day | 72 (v5) → 39 (v12) → 25–38 (v13–17) | same |
| 11-04 | campaign_value_loss (fresh) | 0.00055 → 0.00038; V7 EV 0.17 → 0.24 | move-feature-audit |
| 11-05 | value_loss_fresh at schema 14 | 0.001 → 0.008; growth EV 0.45–0.53 → ~0.24; V_end 0.33–0.37 → 0.20–0.25 | training-stability |
| 11-06 | unexplained / teacher_ce / agreement | 0.90 / 0.97 / 0.33 (v1360–1425) | optimizer-plan |
| 11-07 | privileged critic | first design: unexplained 0.881 → 0.840, V_end −24%; user design: 0.863 → 0.806, V_end −23% | privileged-critic |
| 11-08 | value_loss_fresh at schema 16 | 0.00086 → 0.00121 (+40%) | schema-16 |

### 4.4 Behaviour counts (campaign)

| Behaviour | Values over time | Src |
|---|---|---|
| Suicide attacks by heroes with < 1k gold army (per 30 games) | 224 (97% lost, median 54×) → 275 (99%, 44×) after a920331 → 193 (98%, 27×) after ab52217 → ~112 (0.30 per new hero) after 5805895 | changes-2026-11-03, move-feature-audit |
| Hopeless attacks | ~3 a game (117 in 40) on 11-05; 0 hopeless in 20 games after 296b06a | problem-watch-list, neutral-towns |
| Trades per game (% with no purchase) | 135 (77%) → 127 (74%) → 147 (76%) | trades-and-guards, question-head |
| Dwelling YES | 3% (14/481) → 42% → 466/473 | trades-and-guards, question-head |
| Heroes at end, ours vs NK2 | 0.17 vs 3.67 (10-29) → 0.75 vs 3.51 (10-30) → 0.81 vs 3.83 (11-05) → 1.27 vs 4.01 (v1288) → 0.88 (v1325) → 1.19 (v1361) | problem-watch-list, fix-batches, teacher-label-changes |
| Self-play eliminations | 2/1223 (joint1), 2/1253 (joint2), 5 in ~1,870 player-games (joint7, 11-03) | campaign-value-design, plan-2026-10-28, changes-2026-11-03 |
| Army led by heroes at day 59–60 | 5.7k and 8.9k of 72k/138k; "2% of our army" | fix-batches-2026-11-04 |
| Map revealed | 16% by day 7, 26% by day 42 | changes-2026-11-03 |
| NK2 request mix (397 turns) | MoveHero 12981, ArrangeStacks 1532, Trade 686, Hire 86, no adventure spells (ours ~16%) | nk2-action-audit |

### 4.5 Throughput, latency and memory

| Date | Measure | Value | Src |
|---|---|---|---|
| 10-21 | 4 NK2 AIs, 70 days, Arrogance, headless | ~20 s | vcmi-setup |
| 10-22 / 10-24 | castle epoch / generated epoch | ~1.7 s / ~7 s | battle-model-design, model-iteration-loop |
| 10-26 | decisions trained/s (Mac, 14 cores) | 73 → 238 (*MEMORY.md hook says 194*) | training-infra |
| 10-26 | ONNX session per battle / persistent client per battle | 1.1 s → 0.28 s CPU | training-infra |
| 10-27 | Mac joint1 memory | 23/24 GB (trainer 12 GB) | campaign-value-design |
| 10-28 | battle record | 0.4 MB/decision (16k buffer ≈ 6 GB) → 13× smaller (B5) | plan-2026-10-28 |
| 10-30 | battle client instructions / peak RSS | 41.9 → 34.4 M/decision; 794 → 501 MB | perf-2026-10-29 |
| 10-30 | actor CPU per battle decision (quiet) | 18.0 → 12.65 ms (163 → 194/s); without CPU inference 9.2 → 4.0 ms | perf-2026-10-29 |
| 10-30 | battle updates/h (joint A/B) | 126 → 156 | perf-2026-10-29 |
| 10-30 | battle CPU inference, batch 1 | 7.1 ms (facts 3.5, heads 1.4, trunk 0.9) | perf-2026-10-29 |
| 10-30 | PC: campaign step eager / battle step | 146 ms per 128 / 36 → 25 ms graphed | fast-training-design |
| 10-30 | PC: second GPU process cost to trainer | −54% of steps (−40% with batch wait); a thread: 12× slower | fast-training-design |
| 10-30 | PC: CUDA-graph inference per process | ~800 campaign / ~1300 battle decisions/s; eager launch ~35 ms | pc-training |
| 10-30 | CPU campaign decision | 13–19 ms per PC thread, 9–12 ms per M4 thread; ONNX 0.6–1× | pc-training |
| 10-31 | campaign update (~35k decisions) | ~140 s → 60–75 s (graphs) → 100–153 s at 2 passes | fast-training-design |
| 10-31 | campaign games/h | old loop 17–83 → 500–600 (joint5, ~395 decisions a game); battle epochs/h 90–158 → 28–46 | fast-training-design |
| 11-01 | joint7 game length | ~5,900 decisions; decisions/h 65–75% of joint5; GPU 90–93% busy | pc-training |
| 11-02 | joint7 cycle | ~190 games/h, ~580k decisions/h, ~3,000 decisions/game, ~216 s/game | training-speed-2026-11-02 |
| 11-02 | TBB cap | 0.185 → 0.065 CPU-s per turn; decision wall 72.5 → 61.8 ms | training-speed, changes-2026-11-02 |
| 11-02 | M4 1-thread forward | 20.4 ms (batch 1), 15.2 (batch 4+); ~20k aten ops; bf16 0.95× (Zen 5) / 0.33× (M4) | training-speed-2026-11-02, changes-2026-11-02 |
| 11-03 | window budget | step 61 → 36 ms; GPU peak 10.3 → 6.8 GB | training-speed-2026-11-02 |
| 11-03 | decode | 38.8 → 8.1 ms per 64 decisions | same |
| 11-03 | games/h | 237 → 260 → 301 → 357; learner ~400 decisions/s | same |
| 11-03 | loader workers | 9.5 GB → 594 MB total (25 workers) | same |
| 11-04 | CPU inference after the MOVE rebuild | 56.8 → 64.5 ms per decision | move-feature-audit |
| 11-05 | model sizes | campaign 3.32M (2.25M → 1.18M on 10-31 with dense vocab *[unclear how this relates to joint7's 3.32M]*); battle 286.6k (v11) → 324k (10-28) → 430k | training-speed, pc-training, model-iteration-loop, checkpoint-compatibility |
| 11-05 | steady decisions/s | 226 → 193–226 after schema 14; updates/h 14.2 → 12.2; game time 207–219 → 275 s | training-speed-2026-11-02 |
| 11-08 | contract size | 259k scalars a decision; tile_* 8%, windows 34%, objects 32% | map-size-design |
| 11-08 | Mac forward (batch 8, 2 threads) | 12.5 ms; ~37% tile-bound | map-size-design |
| 11-08 | live update v2047 | 56.6k decisions in 230 s (graphed steps 193 s) | map-size-design |
| — | records | campaign ~853–876 KB raw per decision; spool ~21 MB a game (~4 KB/decision compressed) | perf-2026-10-29, pc-training |

## 5. Cross-cutting insights for the paper

Each insight is grounded in the files named in brackets. Interpretation beyond the files is marked *(interpretation)*.

1. **Bugs dominated tuning, and the biggest ones were silent.** The developer's ranking of bugs above design above tuning (autonomous-experiment-priorities) held up throughout. Several of the largest effects were encoder or environment bugs that broke nothing visibly:
   - every random-map game began without a tavern or dwellings (vcmi 47c9d57eb);
   - neutral towns were never encoded (296b06a);
   - one MOVE feature was always 0 (ab52217);
   - 546 of 954 options shared hashed rows, and the YES vector was identical for every question (d60df0b);
   - early castle runs had wrong hero skills;
   - battle-only games handed out random starting bonuses;
   - the forkserver preload never applied, although it was recorded as fixed (78d55e4).
   
   For low-budget builders: input audits ("can the model even see X?", which became `docs/model_cannot_do.md`) and per-behaviour censuses paid more than hyperparameter work [problem-watch-list, neutral-towns, move-feature-audit, question-head, training-speed-2026-11-02].

2. **Signal-to-noise, not capacity or optimizer, is the binding constraint.**
   - The battle policy gradient's minibatch coherence equals pure noise (~0.2), and battle trunk signal fraction is ~0. LR, batch size, Adam betas and Muon don't change the bias-to-signal ratio (plan-2026-10-28, optimizer-plan). *MEMORY.md (battle-noise-walk) later quantified one battle update as ~0.5% signal.*
   - Consequence 1: any small *consistent* term wins. The entropy bonus/floor (coherence 0.96) and value-loss leakage through the shared trunk drove shooters to stop shooting, twice (v14-design 0fc65e0; plan-2026-10-28).
   - Consequence 2: the AI advised against growing the network ("the signal, not capacity, is the bottleneck") and against restarting (checkpoint-compatibility).
   - Campaign credit is mostly shared luck: 72% of advantage variance is per player-day, and a suicide's −1.5k Φ is drowned by 7-day noise (problem-watch-list, changes-2026-11-03).

3. **Bounded priors with learned corrections beat zero-init learned paths under a low LR.**
   - With the developer's LR cap (≤ 3e-5, 1e-5 used) and ~300 optimizer steps an hour, a new zero-init 3-weight layer moved ≤ 0.003 logits an hour, i.e. inert (changes-2026-11-03).
   - What worked fast was a domain prior (NK2 values, engine values, Φ gains) with a bounded learned correction: safety prior, trade prior, recruit_yes, visit_yes, disband_cost, give_all_cost, battle value priors added to logits. These cut suicides ~2.5× and fixed dwelling answers at once (move-feature-audit, trades-and-guards, teacher-label-changes).
   - The tension between "never raise the rate" and "new features must be able to act" is a recurring design theme (training-stability vs fight-step 4cd9e16).

4. **Reward/Φ shaping creates exploits wherever immediate value and real value diverge.** Instances:
   - war machines at full price caused hire spam (war-machine-value);
   - free hero levels 1–3 meant no hero ever levelled (hero-value-design);
   - Φ = 0 for the winner flattened labels (campaign-value-design);
   - the current score fed as a feature taught the defence to flee (model-iteration-loop R7);
   - the chained V taught passivity (castle-vs-inferno-findings);
   - Φ-paid GATHER plus an always-accepted teacher entry emptied towns (eval 0.487 → 0.265) (multi-actions-and-phi-unit);
   - the meeting value priced on old factors gave 308 exchanges a game (review-2026-11-05);
   - trades without a price leaked ~213k Φ a game (trades-and-guards);
   - no time cost in the battle reward leaves won endgames idling (problem-watch-list).
   
   Remedy pattern: value audits after every change ("0 killers", plan-2026-10-28); price every action (trade_waste, disband_cost); freeze the eval's Φ (EVAL_EMPIRE) so scores stay comparable.

5. **Imitation teachers are powerful and fragile.**
   - The imitation start measured 0.113 (the prediction was 0.3–0.6). Under NK2 advice the eval climbed to 0.221 at eval 24 and to ~0.4–0.5 later (teacher-and-memory, section 4.2).
   - Failure modes documented:
     - the teacher's labels encode the teacher's mechanics: one MoveHero request per tile taught one-tile moves;
     - a label class the model gives ~0 probability, imposed on 40% of decisions, moves the shared type parameters globally (collapse 1);
     - accept sets are not neutral (−log Σp is rich-get-richer);
     - adding an action shifts the teacher's own label mix (GIVE_ALL turned NK2's exchange labels from 52% to 85% ARMY);
     - PPO's KL target must account for the supervised term;
     - tying the teacher weight to a noisy eval score loosened the anchor after a lucky eval (rollback 4).
   - Four rollbacks in ~26 hours (2026-11-05/06) all trace to these (teacher-label-changes). *MEMORY.md: the developer turned NK2's choices off on 2026-11-11 and asked to keep its insights as inputs.*

6. **Self-play alone did not produce a competitive campaign player in this budget.** Campaign self-play was nearly peaceful: 2/1223 eliminations (joint1), 599/600 games to the cap (joint7). Meanwhile ~31–75% of eval games vs NK2 ended in our elimination. Eval heroes fell while self-play heroes rose ("self-play diverges from the NK2 test"). Self-play credit said info spells were worthless; against NK2 they were vital (masking them: 0.22 → 0.07) [campaign-value-design, problem-watch-list, changes-2026-11-03, pc-training].
   - The developer's fairness rule (T3) closed the obvious fix (NK2 opponents) and accepted the cost. AlphaStar-style past-version opponents were deferred "until we are much better".
   - *(Interpretation)* A paper should report this as a real tension: the eval opponent is the only source of pressure, and training against it is ruled out.

7. **Evaluation noise and its feedback into decisions.**
   - The files' working assumption was SE ~0.05 per 48-game campaign eval, with ~12/48 games flipping between consecutive evals. Single evals were repeatedly over-read: the 0.476 "best", and the fix effect "+0.085" was revised to "~0.01–0.08" (neutral-towns, teacher-label-changes). *MEMORY.md 11-12: one PC eval varies ±0.11 run to run.*
   - Practices that emerged:
     - paired evals against a same-code checkpoint (mini_eval.py);
     - 2–3 evals per decision;
     - one canonical map set (the RMG isn't seeded-deterministic);
     - BattleAI seeded by battle state;
     - temperature-1 evals (greedy measured a degenerate oscillation);
     - checkpoints pinned at eval time (c4241a1).

8. **The low-budget hardware story.** Setup: one RTX 5090 (WSL2/WDDM) plus a MacBook M4.
   - **Games, not the GPU, were usually the wall.** Games wait on CPU inference (~33 ms a campaign decision, a ~20k-op forward dominated by per-op overhead), not on the engine.
   - **WDDM time-slices the GPU** between processes (a GPU inference server costs the trainer ~54% of its steps), so inference lives on the CPU.
   - **RAM is the currency** (the 4.5 GB floor; a forkserver preload bug cost 9 GB).
   - The Windows GPU spills to shared RAM instead of failing.
   - nvidia-smi utilisation can be memory thrash.
   - Apple efficiency cores inflate CPU-time benchmarks.
   - Engine-side wins: TBB worker cap 2.9×; persistent battle processes ~4×; canBeCast once per spell (−18% instructions).
   - Pipeline wins: demand-driven game supply (no game played to be dropped; joint5 had dropped ~90%), spooled games and drained deploys (zero data loss), and batching deploys (each restart cost 10–15 min).
   - Throughput went 17–83 → 500–600 campaign games/h (short games), and for long joint7 games 190 → 357 games/h [fast-training-design, training-speed-2026-11-02, pc-training, perf-2026-10-29].

9. **Architecture choices that held.**
   - Raw-only inputs with in-graph transforms (one source of truth, small C++).
   - Entity tokens.
   - Logits as query·key matches plus priors.
   - Option identities taken from the model's own embedding tables (the question head: dwelling YES 0.24 → 0.98).
   - Auxiliary foresight heads whose outputs the policy never reads.
   - Memory slots trained by truncated BPTT (memory_steps 2 in PPO).
   - A privileged critic seeing rivals' hidden states (V_end loss −23%).
   - A fixed 36×36 block grid for any map size (side 1 bit-identical).
   - Migration over retraining: shape changes via zero columns and Adam moments by name. Only width or depth forces from-scratch.
   
   [battle-model-design, campaign-model-v2-draft, question-head, privileged-critic, map-size-blocks, checkpoint-compatibility]

10. **Developer–AI collaboration dynamics (for the "what the dev AIs missed" query).**
    - **AIs corrected the developer with data:**
      - LR/batch/optimizer would not fix drift;
      - there is no exponent in the credit;
      - the battle V is unchained (the developer had forgotten);
      - a deterministic 7-vector cannot be trained by PPO;
      - no LR ramp after data shifts.
    - **The developer corrected the AIs on principle:**
      - masking actions is "bad always";
      - training vs NK2 is cheating;
      - predicting our own action is "silly";
      - the YES token was "retarded";
      - Summon Boat legality must not use hidden information;
      - the PC must work alone;
      - data must never be lost.
    - **AI misses recorded in these files:**
      - read a teacher_ce jump and KL stops as "expected" instead of measuring drift (collapse 1);
      - did not monitor dumb decisions during the optimizer deploy;
      - left Teacher.cpp with one-step labels;
      - a recorded memory fix that never applied;
      - the PC map-set swap misread as model decline;
      - several wrong predictions (section 3.6);
      - decided the move_out reset without the developer (past midnight).
    - The AI re-proposed NK2 seats in campaign training on 10-27 (joint1) and 10-29 (diagnosis) after the developer's 10-28 "no". The developer's firm 10-30 rejection followed (campaign-value-design, problem-watch-list, no-training-vs-eval-ais).
    - Many items ended as "the user's call". *MEMORY.md (decide-dont-defer, 2026-11-09) records the developer objecting a third time to such deferrals.*
    - Parallel sessions interfered through shared state: a symlinked build dir, ff-only merges, peer reviews finding bugs (the e5 peer found the hashed rows).

11. **Rules for safe change in a live run** (generalizable process lessons):
    - one change at a time, judged by 2–3 evals;
    - type_drift after teacher changes;
    - quarantine and renumber on rollback;
    - atomic checkpoints;
    - schema-hash gates, plus a CUDA-graph capture check on every forward change;
    - never benchmark beside the run;
    - test every consumer, including rare paths (MEMORY.md deploy-safety);
    - hard caps on every data-producing process.
    
    The files show these rules being written after each failure, not before [pc-training, teacher-label-changes, changes-2026-11-02, data-generation-limits].
