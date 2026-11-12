# Docs, plans and decks

Scope: `README.md`, the 17 files in `docs/`, `configs/machines/README.md` (no markdown exists under `battle_model/`,
`campaign_model/` or `agents/`), and the two decks in `~/Downloads`. Read 2026-11-12. Source tags: **[doc]** = the
document itself; **[git]** = git log/commit message; **[code]** = a cheap grep of the current tree; **[mem]** = the
project memory notes (`~/.claude_personal/.../memory/*.md`), used only to date or settle a status the docs leave open.
"Untracked" = not in git on `main` (git status at session start). Nothing here is invented; unclear points are marked
**(unclear)**.

---

## 1. Catalog

### README.md
- **Date:** committed, 21 commits 2026-10-23 → 2026-11-01 (last: 4336918, 11-01 18:26, the 11-01 cleanup). Author not
  stated; 24ab45d's message credits e5 for the DataLoader rebuild.
- **Purpose:** setup (macOS, VCMI 1.7.5 submodule on branch `alpha_heroes`, GOG data via innoextract), how to run joint
  training, multi-machine setup, the PC's systemd service, the battle model's file map and C++ contract.
- **Key content:** eval definitions — `eval/battle` = mean reward (−1..1) vs BattleAI and MMAI, greedy, fixed battles;
  `eval/campaign` = our share of all empires' Φ at day 56 × players vs Nullkiller2 (1 = average player). "Training
  itself is self-play, the wandering monsters BattleAI as in a real game." Master/worker game slots (~4 KB per battle
  decision over the network); PC service with `memwatch.sh`, `keeper.sh` (≤3 restarts / 6 h) and `drain_restart.sh`
  (≤45 min drain, "nothing is lost"). Contract (`constants.py`/`schema.py`, `SCHEMA_VERSION`, codegen, ONNX with a
  contract hash, mismatches refused); "illegal actions never cost a turn"; VCMI patch list (`--testmap` race ~1/1000).
- **Status:** partly stale. It says `losses.py` computes GAE, but the battle credit has been a linear round window since
  v15 (plan_2026-10-28 Part 2: "lambda only belongs to the unused 'gae' option"). It says AlphaBattleAI falls back to
  StupidAI for war machines; plan_2026-10-28 Round 10 says that claim "came from a stale docstring". It still describes
  `battle_model.train` with its early defaults (8 battles per side per epoch).
- **Paper value:** reproduction recipe, engine-patch list, multi-machine design for a low budget.

### configs/machines/README.md
- **Date:** 8d96040 (10-29 19:36), d9cb662 (10-30 08:07).
- **Content:** a machine's sizes (actors, games, servers) live in TOML apart from the recipe, so runs compare across
  machines. Specs: `rtx5090` = RTX 5090 32 GB, Ryzen 9 9900X (12c/24t), 31 GB RAM, Windows 11 + WSL2 (20 GB); `mac-m4` =
  MacBook Pro M4, 24 GB. How a case is measured (`wait_seconds` high = battles bottleneck; `train_seconds` high = learner).
- **Status:** in use. **Value:** the hardware table for a "one GPU" paper.

### docs/battle_search.md
- **Date:** 5 commits 2026-10-23 08:56 → 19:02. Author not stated.
- **Purpose:** how to give MCTS an "undo" in VCMI, then results of building it.
- **Key content:** options A–E (replay from start ~1–2 s; restart in process; snapshot live game; `fork()` — does not
  work, VCMI is multithreaded; **E: a local copy running the server's own rules**, ~10–100 µs/action, the only option
  usable in a real game). Built as `CGameState::cloneWithBattle` (the `HypotheticBattle` overlay idea failed: the server
  rules read `CStack`). Stage 1 (`IBattleServer` refactor): 24 seeded battles byte-identical before/after, 882
  decisions. Stage 2 validation: 392 / 372 / 480 actions compared in three scenario groups, 0 refused, 0 non-dice
  mismatches; count/health differ only after attacks (own dice). Copy 3.4 ms, one action 0.05 ms. Stage 3: Gumbel
  AlphaZero (Danihelka et al. 2022) chosen because it improves the policy with 8–32 simulations; C++ with a Python twin
  checked on 64 cases; 16 simulations ≈130 ms/decision (copies ~70 ms). Stage 4: search-policy cross-entropy loss.
  Not covered: sieges, turn-flow check, battle-only copies.
- **Status:** built (stages 1–4), **not used in training**. plan_2026-10-28: "Search lost to PPO in October (16 sims, an
  early model with a weak V), so it waits for a better value head" (runs `20261023-1902-up-mcts16` vs `up-ppo` exist
  in `runtime/train`). The deck (slide 60) lists battle search as an unused trick.
- **Value:** high — engine-level search infrastructure for a game without undo, plus a negative result.

### docs/hidden_information.md
- **Date:** 8aa97f3, 2026-10-23. **Content:** deferral — battle-level hidden information and an LSTM-like battle memory
  "not now", only once battles are good; notes the search copy leaks the enemy's spellbook. **Status:** not built for
  battles (a campaign memory was built instead on 10-31 [mem]). **Value:** low.

### docs/adversarial_review_2026-10-27.md
- **Date:** mtime 10-27 01:09; committed 10-29 (5604d8e). Author: an unnamed reviewing AI session; claims verified by
  the main session [mem]. 15 ranked findings with evidence tags (Verified / Code / Estimate). See §4.

### docs/campaign_ai.md
- **Date:** status line "2026-10-27 (evening)"; mtime 10-28 23:00; committed once (5604d8e, 10-29 03:06). Written by
  the campaign session to the user ("Your decisions").
- **Purpose:** the campaign AI as decided after two user reviews of the same morning's proposal, the joint-training
  system, and joint1's findings.
- **Key content:** decision table (labels = share of all Φ at d+2, d+5, d+7; V_end; random game caps 28 + geometric,
  mean 28, max 112; 2–4 players; buying at list price neutral; hero 2500·(L−3)^1.3; −100·n² hero penalty; 1 gold per
  revealed tile; income × 5 days; MOVE in a 33×33 window K=16; "Don't imitate NK2" → no teacher; 1e-4 share price per
  remaining decision, cap 256 decisions/day; monster counts as quantity ids, "the range a human sees"). Φ table (α=0.25,
  u = α × unupgraded cost). Action space: END, MOVE, BUILD, RECRUIT, HIRE, ARMY, UPGRADE, TRADE, CAST, BUY, DISMISS
  (reserved); 6 question kinds. Model v1: 377k params, 2 attention blocks d=64, per-hero window convs + FiLM,
  `TYPE_PRIOR` (MOVE +1.5 … DISMISS −5). PPO: LR 1e-4, KL stop 1.5 × 0.015, entropy 0.01. Campaign battles 40% of each
  battle update. Untrained model 0.24–0.29 ± 0.08 vs NK2 (knocked out 6/12; NK2 hires 3.3 heroes to our 0.4). joint1
  (10-27 22:40, 9 h; start 0.17 at day 56): 16 findings, among them hero worth ~3,600 for 2,500 gold (heroes per player
  1.5 → 3.2 in an hour), **#13 self-play never attacks** (2 of 1,223 players knocked out; 1,158 kept only their
  starting town; vs NK2 ~75% knocked out), **#14 heroes retreat in ~60% of their battles**, **#16 battle drift**
  (shooter-idle 0.05 → 0.39/0.51), #15 memory binds on the Mac.
- **Status:** superseded in large parts (v2 model, Φ v2–v5, teacher adopted 10-31, labels as logit changes 10-31).
  Still the best record of the founding decisions and the first run.
- **Value:** high.

### docs/plan_2026-10-28.md (51k chars)
- **Date:** 3 commits 10-29 (5604d8e 03:06, 3239517 05:22, dd8c6b8 13:21). Author: the campaign/main session.
- **Purpose:** post-joint1 plan, then a build log of rounds 3–14 with the user's decisions.
- **Key content:** every claim marked measured or predicted. Part 1 (battle drift) measured with
  `battle_model/gradient_study.py`: policy-gradient coherence 0.18–0.33 (noise = 0.22), effect on log p(SHOOT) |t| < 1;
  **entropy bonus coherence 0.96–0.98, t = −27.5 / −9.3 / −17.9**; value loss through the shared trunk t ≈ −5. p(SHOOT)
  where legal, median: 1.00 (v2106), 0.99 (2840), 0.32 (3139). Diagnosis: a consistent push beats a noise-level signal;
  lower LR / bigger batch / Adam betas / Muon would not fix it. Part 2: MC credit SNR +0.23 vs window/TD −0.20 (window
  measured worse, contrary to the predicted ~7× variance cut). Part 3: features as ops (object value, expected loss
  (their/our strength)², danger, representation fixes, type head sees best move). Part 5: C6 NK2 seats — **rejected by
  the user**. Rounds 3–14 (see §2–§4): pre-mortem, value audits, horizon credit, chained → standalone graded heads,
  shared-luck toy, V_end + random ends, NK2's "brain" as torch features, "C++ decides no behaviour", war machines half
  price, battle value priors, BattleAI made deterministic (198/200). Ends with "Things in today's suggestions that were
  wrong" (the AI's own earlier errors: "the cause is the campaign battles", "lower LR + bigger batch", window credit).
- **Status:** mostly implemented. Reversed within the doc: entropy floor (built round 1, off in round 14), weight
  average (built, off in 14). Part 4's "remembered observations, not a learned recurrent state" was overtaken by a
  learned GRU memory (schema 10, 10-31 [mem]).
- **Value:** very high — the richest measured diagnosis in the repo.

### docs/performance_2026-10-29.md
- **Date:** 4 commits 10-30 (d9cb662 08:07 → f0b5c75 15:11). **Purpose:** same computation, cheaper.
- **Key numbers:** CPU per battle decision 18.0 → 12.65 ms (163 → 194 decisions/s); 4.0 ms with the model on a GPU.
  Joint run battle updates/h 126 → 156 (+24%). A game-state copy per decision was 37% of a battle client's CPU → one
  per battle; it exposed a VCMI nondeterminism (area-attack targets in pointer order, fixed e1be37e4a). `torch.compile`
  rejected (CPU gradients differed from eager by up to 110%). CPU inference remains 68% of a CPU actor's cost.
- **Status:** implemented. **Value:** medium-high (low-budget engineering).

### docs/training_loop_2026-10-30.md
- **Date:** 24ab45d, 10-31 09:29 (e5). **Purpose:** the training loop rebuilt around `IterableDataset` + `DataLoader`.
- **Key numbers (PC):** a second GPU process cuts trainer steps 54% (WDDM time-slicing); inference as a thread 12×
  slower (GIL); campaign update of 35k decisions ~80–110 s; GPU mean 70%; 24 games is the RAM limit. Settled
  `passes = 2`, `ahead = 3`, `micro_batches = 4`. "Judge the GPU by decisions/s, not by its utilisation figure." RMG not
  deterministic by seed → eval maps kept with the code.
- **Status:** implemented; the base of joint7. **Value:** medium-high.

### docs/campaign_model_v2_draft.md
- **Date:** 3 commits 10-30 14:52 → 15:23. Header: "draft, not decided … the user: 'I'm not sure about them'", yet
  "Built, 2026-10-30".
- **Key content:** four problems with v1 (little signal to learned parts; ad-hoc input transforms; tiles see 5×5;
  hero choice implicit). Agreed principles: `encode` raw only, known facts joined late; shared hero/army encoders;
  ownership transforms every vector (rival identity embeddings); "large is relative"; **a logit is a match** (query ·
  key + pair net + prior); one softmax over all actions; foresight heads auxiliary only; any map size. Forward:
  `domain_knowledge → encode → take_sides → join → see (U-Net) → think → foresee → think_again → decide`. Typed
  inputs with ~10 hand-set unit scales (gold 1,000, creatures 20, movement 1,500 …). U-Net width 16 (32 made inference
  27 ms vs v1's 9 ms; 16 → 18.5 ms). First version's log-mean-exp type score let "a few good tiles among 300 count for
  nothing" → "a type is as good as its best option". Ideas set aside (Φ per block, hidden-content targets, "where will
  our hero be tomorrow").
- **Status:** implemented and the default (`model_architecture: str = "v2"` in `campaign_model/config.py` [code]); 3.3M
  weights per the deck. Deviation: the draft's single softmax with "the separate type head goes away" was not built as
  such — the built model keeps a type stage scored by its best option, and one softmax over hero × tile within MOVE
  (deck slides 57–58).
- **Value:** high (architecture rationale).

### docs/empire_value.md
- **Date:** d7a92a2 (10-30, "Φ v3"), 75cfe7d (11-10, schema-17 access note). **Purpose:** every Φ rule and every map
  object/building with what Φ adds.
- **Key content:** the rule (user, 10-27): buying at list price leaves Φ unchanged. Hero = 1,500 + skills (k-th primary
  point beyond 5 = 400k; secondary level 2,000; level 200; strongest hero full, others half; −100·heroes²); spells 500 ×
  level; luck/morale until the next battle 2% of the lifted army per 24th; war machines half; mana 2 × power × mana for
  3 casts; 1 per revealed tile. Full object tables. Omits movement, information, access (except MOVE's prior since
  schema 17), threats.
- **Status:** **stale in places** [code]: `EMPIRE_VERSION = 5` (v4 11-05: unled armies at 0.953; v5: army speed and
  make-up morale), evaluation still uses v3 (`EVAL_EMPIRE_VERSION = 3`); `explored_tile = 5.0` since 11-03 (doc says 1);
  armyless heroes at 55% of base (11-03). The deck (slide 51) already says "explored tiles at 5".
- **Value:** high — the reward definition.

### docs/optimizer_plan.md (untracked)
- **Date:** mtime 11-06 18:40; implemented in 8c6cea9 (11-06 10:24, all steps at once); step 2 reverted in 53c1df7
  (11-06 14:11). Author of the plan not stated; the implementation section is by another session (**unclear** which).
- **Key content:** Adam noise argument (pure noise gives |m̂|/√v̂ ≈ 0.23 at β1 0.9; campaign 0.17–0.26: "most parameters
  move mostly at random, at full size"); loss weights don't change a head's own parameters under Adam; gradient noise
  scales 5–700 decisions (value) vs 300–6,000 (policy) at 4,096 per step; the critic on the slow timescale (same LR,
  15% of the trunk; actor KL budget 0.06); three controllers sharing one time constant under 19 restarts in 19.5 h.
  Steps 0–8 (critic LR ×10, KL budget back to 0.015, freeze loss shares, trust floor 0.4, trunk β1 0.95, per-group clip,
  head weights by residual, critic branch). Implementation notes: steps 1 and 2 contradict as written; `signal_ratio`
  cannot judge β1 changes → new `signal_fraction` = (mean min(1, m̂²/v̂) − k)/(1 − k); per-group clips rescaled to avoid
  jolting Adam. First measurement refuted step 0's premise: **the trunk is ~88% of the gradient norm** (13.3 of ~14).
  Step 2's fallback fired (eval 0.441 → 0.363 → 0.335): this KL controller "gives the floor or the cap". After the
  revert: 0.430 ± 0.050; 60-update read: "smoother training … not a better fit" (unexplained 0.905 vs 0.904); step 8 not
  taken.
- **Status:** implemented except step 8; step 2 reverted. **Value:** very high — an honest optimisation study.

### docs/battle_model_plan.md (untracked)
- **Date:** mtime 11-05 13:53, the minute of 8797125 (the deploy). Author not stated.
- **Key content:** battle eval +0.088 to +0.097 for 53 evals over joint7's 1,087 battle updates (4.5 days). v15-window
  (learned +0.025 → +0.058 in 11 h) vs joint7: LR 1e-4 vs 1e-5; ~25 vs 4 optimizer steps/update; ~180 vs 4.6
  updates/h; update KL ~3e-3 vs 5e-5. Three signal problems: 60% of each update is lopsided campaign battles (attack
  wins 84%; reward divided by both sides' assets); credit 85% a two-round window; no exploration pressure. Decided:
  accumulate 24 → 2 **at LR 1e-5** (the plan proposed 3e-5 and β2 0.999; overruled by "never raise the rate"); credit
  50:50; lopsided battles' advantages × 1/(4p(1−p)), cap ×12.8 (the plan's stake weight and stake-based keep
  rejected); shooter type features, DEFEND under a full shot starts at −2; entropy floor not turned on. Bug: the
  adaptive outcome share never reached the workers.
- **Status:** built (8797125). Later: optimizer_plan (11-06) names accumulate 2 as first suspect for the battle trunk's
  pure-noise reading; [mem] 11-10: e5 found accumulate 2 wrong (signal 0.13 < noise 0.155).
- **Value:** high — diagnosis of a flat curve.

### docs/hota_transfer_plan.md (untracked on main)
- **Date:** mtime 11-05 21:02; committed on branch `hota` as 920284a (11-06 08:07) plus three commits to 09:46 (inventory,
  content match library `agents/AlphaContent` + Python mirror, `docs/hota_status.md`).
- **Content:** one model for SoD and HotA: content keys matched at every engine start (strict in training), an
  append-only vocabulary, rows not ids in records, values derived from engine config with SoD regression diffs, a
  coverage gate before any HotA training; phases A–D.
- **Status:** part of Phase A on `hota`, idle since 11-06; its `hota_status.md`: 775 unknown keys, 0 collisions; HotA
  rebalances 31 SoD creatures, 8 artifacts, 3 heroes; "Training with HotA does not work yet." **Value:** medium.

### docs/map_size_plan.md
- **Date:** 1a9fa63 / 1527c22, 11-08 09:22 (e5, "for the implementer"). **Content:** 36×36 grid of `side × side` blocks,
  side = ceil(max(w,h)/36); measured: `tile_*` are 8% of 259k scalars/decision, ~37% of the forward is tile-bound; a
  tile-exact 144×144 grid ~6× the whole forward. Checks: small maps bit-identical, 36×36 forward time unchanged, a
  144×144 self-play game ran.
- **Status:** implemented; no big-map training yet. **Value:** medium-high ("surgery, not restarts").

### docs/model_cannot_do.md (untracked)
- **Date:** mtime 11-10 14:07; audited 11-03 against schema 10; living list (user-ordered).
- **Content:** never-learnable gaps: disband/unequip (NK2 `DisbandCreature` 3,555 and `ExchangeArtifacts` 450 unmatched
  over 941 player-games); worn vs carried artifacts indistinguishable; taking a town's troops into the visiting hero
  (heroes led 2% of our army at day 60; P(ARMY town→hero) 0.0002; BUY 0 of 4,219 — NK2 does these inside its visit,
  never as requests, so the teacher taught against them); memory gradient reaches back one write; basic vs upgraded
  recruit; quests; town spells; map markets; contract sizes. "Hard rather than never": position/threat/scouting have no
  Φ term. Fixed list with impacts: **neutral towns never encoded** (296b06a; eval 0.391 → 0.476; P(step onto hopeless
  neutral town) 0.27 → 0.002), memory trained over pairs (8458ede), schemas 11–13, 16.
- **Value:** high (what the architecture cannot learn; bug impacts).

### docs/multi_actions.md (untracked)
- **Date:** mtime 11-07 22:12. **Content:** where NK2 issues several requests per goal vs our MULTI kinds; counts per
  397 NK2 player-turns (ArrangeStacks 1,532, RecruitCreatures 711, TradeOnMarketplace 686, GarrisonHeroSwap 228,
  HireHero 86, ExchangeArtifacts 67); why it matters ("a sequence … reaches our model as one label plus an accept set");
  review of schema 14 (an accept set turns the teacher off where it accepts → GATHER emptied every garrison).
- **Status:** TAKE_BEST (schema 15), GARRISON_SWAP / SPLIT_SINGLES / MERGE_ALL (16) built; UPGRADE_ALL, VISIT_TOWN,
  EQUIP_BEST, GARRISON_IN/OUT, recruit-all at a dwelling not built (as of the doc). **Value:** medium-high.

### docs/points to note in a future paper.md
- Committed once (5604d8e, 10-29) with 8 items; working copy modified (12 items; item 1 rewritten). The developer's own
  notes. Quoted in §6.

### docs/session_rules_draft.md (untracked)
- mtime 11-12 08:00; "Written 2026-11-11, rules 20, 21 and 24 added 2026-11-12"; "DRAFT for review, not in force". 24
  rules, each from an incident. Quoted in §6.

### The two decks
- `~/Downloads/From AlphaGo to OpenAI Five.pptx` (mtime 11-10 13:10) and `From AlphaGo to Heroes III.pptx` (13:30): two
  revisions of one 62-slide deck. See §5.

---

## 2. Design rationale digest (as the docs state it)

**Division of labour.** "Python owns the science": C++ passes raw engine features, legality masks and the model's own
previous outputs, and decides nothing (user's rule, plan Round 9; README). Engine patches are labelled; examples kept:
per-colour adventure AI (`VCMI_ADVENTURE_AI_<COLOR>`, "not a bug fix"), BattleAI's random choices seeded from battle
state (evals reproducible), red's loss no longer ending multiplayer games, area-attack target order.

**Two models.** A battle model (≈430k weights) plays every battle; a campaign model plays the map. One currency joins
them: the battle is paid its change in value (deck slide 49). The battle model cannot see campaign context (mana for
tomorrow, replaceable creatures, retreat vs surrender economics); "campaign prices for battles" is future work.
battle_model_plan rejects campaign context as battle inputs for now ("the reward is battle-local by design").

**Architecture.**
- Battle: attention over unit, hero, hex, global tokens; conditional heads; the engine's damage formula, kill rule and
  wipe chance as in-graph ops; value heads V0–V2 (round ends) + V (end) (README; deck 46: "a log is not enough … the
  game's own damage formula runs inside the network").
- Campaign v1 (10-27): 377k params, 2 attention blocks, per-hero 33×33 window convs + FiLM, `TYPE_PRIOR`.
- Plan 10-28 Part 3: "features as ops" in the battle model's `ActionScore` pattern, but defaults that steer from game
  one (the battle model starts feature weights at 0). Then Round 10: "every object we see goes in" (8,192 objects),
  NK2's scores as torch features, and the "prior-anchored" rule (fixed prior + decayed learned correction; 1,563
  prior scalars; `CORRECTION_STEP` ×10 because a plain weight at 1e-4 moves "at most ~1.9 in a whole run").
- Campaign v2 (10-30): see catalog; map U-Net (width 16), tile reader with reach as attention bias, foresight heads
  (auxiliary, two-hot symlog targets, DreamerV3 recipe), match-style logits. Memory: 8 slots × 64, GRU-style write
  (schema 10, 10-31 [mem]); trained over decision pairs since 11-05.
- Map size: block grid instead of size classes (cost-neutral on 36×36, "a block is 1/36 of the map on every map").
- **Change over time:** the plan's hand-made couplings (rounds 3, 5: END held back by the best MOVE/TRAVEL offer) are
  removed in the v2 draft, which instead scores a type by its best option.

**Inputs.** Only what a human player sees: monster counts as VCMI quantity ids, exact only with Visions; resource-pile
amounts not an input; per-player and per-hero visited flags; enemy hero levels unknown (campaign_ai). Representation:
bounded and scale-free (fractions, x/(x+c) at several scales, log-ratios through a sigmoid, "never x / 50", plan C4);
v2 adds typed encoders and amounts relative to situation references (our Φ, strongest rival army, income, week). The
11-11 rule (session rules 14) requires each feature to state which vector it attaches to, unit, combined form, and how
fast its weight can move ("a weight that starts at zero moves about 0.2 logits a day here").

**Action space.** Structured, autoregressive: type then arguments (deck 57: ARMY has five arguments). 10-27: 11 turn
types + 6 question kinds. Added: TRAVEL (plan 10-28 status table, action 11: one of 18×18 areas a level, a day or
more away), ARTIFACT (schema 13), MULTI macros (schemas 14–16; 11 kinds per the deck), ARMY target slots (16), MOVE's
further window past gates (schema 17). MOVE = one softmax over (hero, tile) pairs, score = learned match + engine "price" (deck 58). Rules: no
masking to stop overuse; a new action starts at a learnable −4 logit; the cost of macros ("a choice the model can never
make is a choice it can never learn to avoid", deck 57).

**Reward Φ.** Gold value of everything owned; buying at list price neutral; value appears through production, pickups,
fights, captures. Rationale (deck 45, 50): every reward converges on conquest on day 999; the problem is a reward that
teaches sane decisions on day 3 from few games; balanced game design makes neutral pricing a fair start.
Evolution: v1 (10-27, hero 2500·(L−3)^1.3, 1 gold/tile) → v2 (10-29, heroes by skills, 1,500 base) → v3 (10-30, spells,
luck/morale) → 11-03 (5/tile, armyless hero ~−1k) → v4 (11-05, unled armies ×0.953) → v5 (11-05, speed, make-up);
evaluation pinned at v3 [code]. Battle reward: terminal, hero by level only (2000·min(L,3) per level-up, 10-29 [mem]),
war machines half price (10-28). **Labels:** share s = Φ_me/ΣΦ at d+2/5/7, BCE on the raw share (10-27) → change of
logit(share) (user, 10-31, commit 0e59440 [git]; deck 51: "in the logit, rivals' hidden values cancel"). The developer's
note 10 ("log(value_share_difference…)") is in the same direction.

**Credit assignment (campaign).** 10-28: Monte Carlo, flat mean of d+2/5/7 shares (window credit measured worse: the
value head is "blind within the day" since a day's decisions share one label). Round 4: horizon credit (V7 bootstrap at
the next decision, or V5 two days on, mixed with the outcome by trust = explained variance). Round 5: chained heads;
Round 7: standalone heads graded by the next shorter head's later forecast ("the telescope"); round 12: V_end blended in
by its trust over a share prior. Shared luck: decisions' labels share (H−k)/H of their luck; "it biases no fit; it
breaks statistics that count samples"; toy: composed V2/V5/V7 1.1–3.4× data efficiency (deck 52 repeats this). Credit
check per action type (gaps > 3 SE cut the bootstrap). Optimizer plan: trust floored at 0.4. Privileged critic (11-07,
deck 53: max-pool over rivals' hidden states; end-value error −23%, share error −5.5%). Deck 52/60: an 11-09 audit
found a day's decisions share 96.5% of their credit variance; a per-decision term was built and set aside.
**Battle credit:** GAE (README) → linear round window, γ=1 (v15) → policy reads the final V alone (Round 6) →
outcome_share adaptive at its 0.15 floor → fixed 50:50 (11-05).

**Training algorithm.** PPO (clip 0.2), KL control against the update's own start (k3, after the review), LR 1e-4
(10-27; the review cites "the user's 1e-4 rule") → the user's cap 3e-5 (stated by 11-05; date of the change
**unclear**), 1e-5 used for the campaign policy and the battle model (11-05/11-06); the critic at 1e-4 via its own Adam group (11-06). Entropy:
bonus 0.01 → floor (10-28, measured cause of drift) → off (10-29: "the one steady push against confident shots") →
still off (11-05). Battle: weight EMA (on 10-28, off 10-29). Campaign updates 8,192 decisions from ≥4 games (10-27) →
32,768 of 20+ games ("fewer games an update is what the user fears most", 10-31). **Teacher:** "No teacher" (10-27) →
NK2 imitation, KL and per-decision advice from 10-31 [mem; deck 54], ~60% of the trunk gradient (optimizer plan);
11-11: NK2's own choice played as a mixture (ρ 0.9), the teacher loss off; later "NK2's choices OFF (all own games)"
[mem]. **Opponents:** self-play only; NK2 seats rejected (C6, user: "learning its weaknesses instead of being capable");
v2 draft and bae4696 restate it. Battle league of past snapshots (uniform sampling).

**Evaluation.** Battle: fixed battle sets (castle, hard, neutrals, generated, real; castle 4 → 16 seeds after the
review) vs BattleAI and MMAI, greedy. Campaign: vs NK2, share × players at day 56; 24 games every 60 min (10-27) → 48
fixed games (deck 59, hota plan); paired evals (C8; `campaign_model.paired_eval`, 48 setups × 2 replicates [mem]).
Noise estimates **disagree across docs**: ±0.05 per eval (optimizer plan, deck 59) vs "about 0.1 either way … v3229
scored 0.861 one night and 0.697 the next morning" (session rules 9). Dumb-decision metrics (`campaign_model/failures.py`,
battle `failure_counts`) as diagnostics, never forced to zero.

**Contradictions noted** (beyond the stale items in §1): wandering monsters in training games are played by the current
battle model per campaign_ai §6 (10-27) but by BattleAI per README (11-01) **(unclear when it changed)**; campaign_ai's
"no teacher" and deck 54's "never play it" vs the 11-11 deploys that played NK2's choice at ρ 0.9 [mem]; the entropy
floor is "the fix" in plan B1 and off from Round 14 on.

**Infrastructure.** Mac M4 + one RTX 5090 PC under WSL2, driven over SSH; one trainer process, DataLoader workers
playing games, inference servers placed by measurement (campaign inference on the Mac's cores over Wi-Fi at one point,
14–61 ms round trips); records zlib-compressed (campaign ~2 KB/decision; battle 407 → 31 KB after `compact_record`);
CUDA graphs; `drain_restart.sh` deploys. Deck 59: ≈470 self-play games/hour, ≈1,500 decisions each; each training
decision waits ≈70 ms for inference; "the bottleneck is producing games, not the GPU."

---

## 3. Plans vs reality

| Plan (date) | Proposed | Done | Dropped / changed, and why (if stated) |
|---|---|---|---|
| battle_search (10-23) | MCTS via local engine copy; AlphaZero targets | Stages 1–4 built and validated | Not used: lost to PPO at 16 sims with a weak V; waits for a better value head (plan 10-28) |
| hidden_information (10-23) | battle memory for revealed info | — | Deferred by design; campaign memory built instead |
| campaign_ai open points (10-27) | judge guesses (hero penalty, α, price); whole-map grid; longer labels; drop TYPE_PRIOR?; threat level input | whole-map grid (plan 10-28), V_end (round 12) | price later reworked (step credit set aside, action price [mem]) |
| plan 10-28 B1–B6 | entropy floor, per-(side, source) normalisation, weight EMA, compact records, restart from 2840 | all built | B1 and B3 off in Round 14 (floor fought confident shots; evals on trained weights) |
| plan 10-28 C1–C5 | object values, expected loss, danger, representation, type head | built with priors (rounds 3–11) | v2 draft removes hand couplings |
| plan 10-28 Part 4 | memory as remembered observations (map beyond 16 tiles, event log, enemy memory, intent) | whole-map grid; learned GRU memory (10-31) | "not a learned recurrent state" reversed (user 10-31 [mem]: "no memory means we lose") |
| plan 10-28 C6 | NK2 seats in 1/3 of games as opponents | — | Rejected by the user (exploiting its weaknesses ≠ capability) |
| plan 10-28 C7, C8 | learner process on GPU; paired evals | built | — |
| campaign_model_v2_draft (10-30) | new forward, one softmax | built, default | type stage kept |
| performance (10-29/30) | — | +24% battle updates | legal-only scoring, campaign facts once, malloc trim: not done (~1–2% each or shape issues) |
| training_loop (10-31) | levers: smaller updates, Ethernet, Linux | passes 2, ahead 3 | hardware levers not done (**unclear**) |
| battle_model_plan (11-05) | LR 3e-5, β2 0.999, stake weights, entropy floor 0.01, shooter foresight head | accumulate 2 at 1e-5, 50:50, ×1/(4p(1−p)), DEFEND −2 | LR raise rejected (user rule); stake weight min(own, enemy)/10k would cut every early-game battle to 0.1–0.5; floor fights the shooter prior; accumulate 2 later judged wrong [mem] |
| optimizer_plan (11-06) | steps 0–8 one at a time | all 0–7 deployed at once by the user's choice | step 2 reverted (eval fell); step 8 not taken (detached trunk would make critic features less value-shaped) |
| multi_actions (11-05/07) | TAKE_BEST, GARRISON, UPGRADE_ALL, VISIT_TOWN, EQUIP_BEST, recruit-all | TAKE_BEST, GARRISON_SWAP, SPLIT_SINGLES, MERGE_ALL | rest pending; GATHER rolled back once (emptied towns) |
| hota_transfer_plan (11-05) | phases A–D | part of A on `hota` | idle since 11-06 |
| map_size_plan (11-08) | block grid | built same day, bit-identical | big-map training not run |
| model_cannot_do (11-03 →) | fixes per row | rows for neutral towns, memory, artifacts, guards, army arrangement fixed | rows 1, 2b, 1b, 3b, 4, 5, 6, 8b, 9 open at 11-10 |

---

## 4. Reviews and audits

### docs/adversarial_review_2026-10-27.md (and its fixes)
Scope: `battle_model/`, `agents/AlphaBattleAI/`, scripts, VCMI patches, two fresh commits, two untracked scripts; numbers
from the running `v15-window` logs.

| # | Finding | Fixed? |
|---|---|---|
| 1 Critical | After any resume, league snapshots registered as `v00050` but requested as `v50` → opponents loaded no model and **every unit defended**; 23 of 182 battles (12.6%) in surviving logs; trained on as normal data | **Fixed** in 3263898 (10-27 01:10): names, v0 dropped, `ALPHA_BATTLE_REQUIRE_MODEL` aborts the battle |
| 2 High | KL early stop measured vs the stale behaviour policy: 24% of 768 updates stopped, 40 did one step, LR sawtooth | **Fixed**: k3 KL vs an anchor pass at update start |
| 3 High | Actors overproduce, drops unlogged, newest-first selection skews to short battles (big battles ~2.5% of decisions dropped) | **Fixed**: backpressure, oldest-first, drop counters |
| 4 | `shooter_not_shooting` counts WAIT | **Fixed**: per (unit, round) |
| 5 | `wipe_chance` assumes uniform damage; VCMI averages ≤10 rolls (e.g. 0.30 claimed vs ~1.4% true) | **Fixed**: normal approximation |
| 6 | Spell×hex tensors ~55% of ~400 KB records | Deferred; later B5 compact records (407 → 31 KB) |
| 7, 8 | `noise_ceiling.py` crash/bias; `value_benchmark.py` separate net starts from shared trunk | Deferred |
| 9 | balanced/worst eval scores dominated by an 8-battle set (sd 0.18) | **Fixed**: per-set SE, castle 4 → 16 seeds, hard 8 → 16; "worst by lower bound" pushed back |
| 10, 11 | Siege features ignore moat/gate; summoned units worth 0 | Deferred |
| 12 | Evals ~11% of wall time; eval ONNX never checked vs torch | **Fixed**: 12 persistent processes, ONNX verified each eval |
| 13, 14, 15 | real-eval mana/gold; adaptive outcome share gauge; small items | Deferred |

Also from the verification [mem]: evals were not deterministic even with fresh processes (BattleAI time-seeded RNG) —
later fixed by an engine patch (plan Round 14: 198/200 identical). An endless-battle eval stall (defender flying between
two hexes) was found the same night.

### Other reviews inside the docs
- **plan 10-28 Round 3 "pre-mortem"** on untrained play: END with a pickup in reach 14/42 → 3/42; TRAVEL with one in
  reach 11/13 → 4/11; creature-bank prior +3.07 with no danger → −3.3.
- **Round 4 value audits** (`campaign_model.audit`, `battle_model.audit`, invariant tests): pickups underpriced 3.8×
  (485 predicted vs 1,826 realised); in-graph Φ missed mine garrisons and dwellings; futile TRAVELs repeated to the daily
  cap (→ `dropFutile`, no-op rate 0%); battle cast prices matched (Ice Bolt 2,661/2,665) but ignored magic resistance.
- **Round 8 "another AI's review" (10-28)**: 12 bugs fixed (game-end detection, no-effect flag on the wrong record,
  stalemate retreat throwing away won armies, …); three claims rejected with reasons. **Round 10 battle reviewer**:
  resistance moved into `castValue` per unit.
- **performance_2026-10-29** exactness audits (39,140 decisions; 107,819,712 values replayed identically) found the VCMI
  area-attack order bug. **multi_actions' schema-14 review**: the accept set disables the teacher; GATHER's prior prices
  the wrong army under a garrisoned hero; no tests of `executeMulti`. **optimizer_plan's implementation notes** review
  the plan itself (steps 1/2 contradiction, invalid metric under a β1 change, Adam jolt).
- An 11-05 bug review and the 11-09 audit (meetings 52% → 9.5%; 96.5% shared credit) exist only in memory notes and the
  deck, not as docs.

---

## 5. The decks

**Identification.** Both files are the same deck, "From AlphaGo to Heroes III" (62 slides, speaker notes on 24 / 21
slides). The file named "…OpenAI Five" has docProps title "From AlphaGo to OpenAI Five" but its slide 1 reads "From
AlphaGo to Heroes III"; it is the earlier revision (13:10). The "Heroes III" file (13:30) differs only by: the references
slide moved from position 42 to 62; three added speaker notes (slides 7, 16, 23); three added bullets on slide 16
("No joint softmax over type × units × target …"). Slide numbers below follow the Heroes III file. Footers cite Silver
et al. 2016/2017, Vinyals et al. 2019, OpenAI et al. 2019, De Asis et al. 2020, Tromp 2016; Heroes numbers are cited to
"AlphaHeroes run joint7" and project notes.

### Parts I–III (slides 1–41): the history, one trick at a time
Background from the papers, condensed (the Heroes-relevant framing is what matters for a journal):
- 1–2: title ("one trick at a time, and why Heroes III is harder still"); road map Oct 2015 → Oct 2019.
- 3–12 **Go**: brute force infeasible (10³⁶⁰ tree); value net; two heads; supervised policy (57.0%, 29.4M labels; note:
  one label per move vs one per game, but human mistakes); MCTS; ablation 1,457 → 2,890 Elo; "one game, one label"
  (Zero: 4.9M games; AlphaGo 2016's value net memorised results, fixed by 30M games one position each; "a betting
  market would price the position after every move"); **search adds 2,130 Elo** (3,055 → 5,185).
- 13–24 **StarCraft II**: why harder; Tricks 1–8 with ablations (structured action head; imitation; KL to the supervised
  policy 1,020 → 1,400 Elo; V-trace + UPGO; critic sees the opponent 22% → 82%; league with exploiters; PFSP; APM
  limits). **Developer's own note (slide 23):** "MORE APM HURTS: this is an indication of suboptimal training
  procedure … the numbers are suspiciously clean 100% … this is still 'cheating' (able to learn a much better micro)
  due to delay and being accurate." Slide 16 note compares the action head with Heroes ("ARMY has five steps, BUILD
  two"; no delay head since turn-based).
- 25–40 **Dota 2**: what it adds; compromises (17 heroes, scripted builds); Tricks 1–9 (shaped reward, team spirit,
  zero-sum + time decay, horizon 60 s → 840 s, PPO + GAE at scale, fresh data, 80/20 past pool, surgery and Rerun at
  ~20% of compute, randomised worlds).
- 41: four-system summary table.

### Part IV (slides 42–61): Heroes III / AlphaHeroes
42. "A game built for people, not machines … AlphaHeroes (2026, in progress)."
43. The game in three layers (adventure map, towns, battles).
44. **Why harder than all three** (table): fog; 2–8 players, fixed-sum ("rivals can gang up, so two-player self-play
   theory does not carry over"); near-infinite state (stacks and gold to 2³¹); ≈1,500 decisions per small-map game plus
   battles; sub-games; human data "none at scale".
45. Challenge 1: many meaningful decision kinds; learn, don't script (contra OpenAI Five); a neutral reward works
   because good design balances options.
46. Challenge 2: unbounded numbers; "a log is not enough"; damage formula as an in-graph op.
47. Challenge 3: the map is a graph; the engine's pathfinder gives reach and days; MOVE window, TRAVEL areas, U-Net.
48. Challenge 4: a battle in every fight. Battle model 430k weights, "+0.08 to +0.09 over BattleAI and MMAI in the same
   seats" (joint7, 9 Nov); shooters waste 2–3% of full-damage shots.
49. Challenge 5: two models, one currency (Φ); what the battle model cannot see; "a battle is not zero-sum, yet we train
   it as one … a modelling choice, not a fact."
50. Challenge 6: every reward leads to conquest (Yudkowsky quote; Bostrom; Omohundro); "the real problem: which reward
   teaches sane decisions on day 3, from few games."
51. **Trick 1, value-neutral decisions**: Φ (creatures at cost, war machines half, buildings at cost, heroes by level and
   skills, spells 500/level, explored tiles at 5); label Δk = logit s(d+k) − logit s(d), k = 2, 5, 7 and the end. Note:
   unled creatures count at 95.3%; market spreads are real costs.
52. **Trick 2, the telescope** (V7 graded by V5 two days later, V5 by V2, V2 by the real share; ≥25% real outcome once
   trusted): "2–3 days of luck in each label instead of 7 … worth 1.1–3.4× the data" in a linear toy; "Only V7 and V_end
   steer the policy." Note: "the trick we are proudest of … ours, not AlphaStar's or OpenAI Five's"; 96.5% shared credit
   variance (9 Nov).
53. **Trick 3, auxiliary heads**: V2/V5, growth heads, battle V0–V2, teacher, privileged critic (end-value error −23%,
   share error −5.5%, joint7, 7 Nov).
54. **Trick 4, learn from the scripted AI, never play it**: imitation 37% top-1; advice from a private NK2 copy (~92%
   expressible, 5–30 ms each); weight meant to fall as the exam score rises, "held fixed for now, at ≈60% of the
   gradient". Lessons: label the plan not the request (decisions/day 72 → ≈30); a label change is a policy change (eval
   0.38 → 0.25, rolled back); the advisor's own bug (52% of advised moves were heroes meeting each other vs 1.7% in NK2's
   own games; "a quarter of all our movement"; 10% after the 9 Nov fix).
55. **Trick 5, priors not masks; surgery not restarts**: masking "overused" spells dropped the eval 0.22 → 0.07 (they
   revealed enemy heroes); 144×144 support left 36×36 outputs bit-identical; 6 contract changes (schemas 11 → 16) in
   one run; "a clean rerun with the final code is still due."
56. **Trick 6, count the dumb decisions**: pickup_passed_end, troops_left, repeat_meetings, idle_movement;
   shooter_not_shooting, escape_while_ahead, worthless_cast, cast_skipped_on_loss. Shooter idling 1–2% → 3.7% after an
   optimiser change with a flat battle score; meetings 26% → 7% of movement, unused movement 10% → 27% ("the waste had
   moved"), now ~18%.
57. **Trick 7, structured actions and what they hide**: 3 decision kinds; 14 types incl. MULTI (11 macros); "the cost
   of shortcuts".
58. **Trick 8, learn the match, not the price**: (hero, tile) score = learned match (starts at 0) + engine price (Φ gold
   × P(win the guard) − danger; 4,000 gold = +3.3 logits); one softmax over up to 8,712 entries (illustrative numbers).
59. **Status, "One GPU, not 384 TPUs"** (10 Nov): eval vs NK2 (share × players, day 56): imitation 0.11 → moves labelled
   by target 0.22 → before neutral-town fix 0.39 → after 0.48 → last 12 evals mean 0.44; best single 0.54; ±0.05 per
   eval; parity 1.0. "Below the scripted AI, and roughly flat since the 6th." ≈470 games/hour, ≈1,500 decisions each;
   3.3M / 430k weights; ≈70 ms inference wait per decision; the teacher fix cut a game from ~2,500 to 1,500 decisions
   and raised games/hour ~300 → 470.
60. Tricks not used yet: battle search; league; rerun from scratch; strategy statistic z (hiring vanishes in greedy
   play); human games (3 recorded: ≈660 decisions, 802 battle actions; "a test set before a teacher"); campaign prices
   for battles; per-decision credit; open the shortcuts.
61. **Lessons**: neutral by default; teacher not opponent (audit its advice); never mask; labels are the policy (measure
   drift on fixed decisions); count the mistakes; every human action fits one model action.
62. References.

### Framing and claims for an audience
- Heroes III is presented as harder than Go, StarCraft II and Dota 2 on most axes, with a low-budget angle (one GPU vs
  384 TPUs / 770 PFLOPs/s·days) and a "games are the wall" thesis.
- Claimed originality: the telescope (nearest published idea: fixed-horizon TD), value-neutral Φ rewards, match +
  engine-price logits, dumb-decision counters; borrowed: structured actions, KL/teacher, privileged critic, surgery.
- Honest status: about half of parity, flat within noise.
- **Consistency with docs:** battle +0.08–0.09 matches battle_model_plan (+0.088–0.097); 3.3M / 430k match the
  optimizer plan's group sizes; 1.1–3.4× matches plan Round 6; 0.38 → 0.25 matches the teacher-label rollback
  (0.378 → 0.248) in model_cannot_do; neutral-town numbers match (0.391 → 0.476). "Explored tiles at 5" matches the
  code, not empire_value.md. Later events the deck predates [mem]: on 11-11 the teacher loss was removed, NK2's choice
  was played as a ρ 0.9 mixture (contrary to "never play it"), then all-own games; Mac evals of own play then read
  ~0.70–0.80. The 0.22 → 0.07 spell-masking number and the 37% / 92% teacher numbers have no source among the docs
  **(unclear source)**.

---

## 6. Paper-relevant material

### `docs/points to note in a future paper.md` (working copy, verbatim)

```
  1. examples are dependent, luck is shared in the telescope, a good thing in 3 days effect the N+5 day results for multiple things, N+2 head have less correlated stuff. Similar to point 9 but between losses and not between games.
  2. heads learn different horizon and diff from last head, correlated luck results...
  3. all utility functions work, paperclips, but some work better. Try for 0 effect on unclear decisions.
  4. ops over features from hardcoded models, aim for batch0 to be similar to the hardcoded model
  5. dataset generation: hard examples... if they are unlearnable we have an architecture issue
  6. some intro about that we aim only for games that can not be solved by brute forcing the final label like alpha go
  7. metric: count dumb decisions.
  8. AI error section (what mistakes were made by the AI that can happen in future games)
  9. Alpha Go only kept 1 position per game; I can't afford to do it and end up getting a correlated dataset.
  10. Replace value share goal with log(value_share_difference_at +2/5/7 days)
  11. hire actions are rare, so if in eval we do not follow the probabilities and take the likliest action each time we hire much less. (not a real issue, learn about alpha star strategy vector)
  12. find out how NK2 cheats exactly, full numbers but what else
```

The committed version (5604d8e, 10-29) had items 2–8 as above and this item 1, since replaced (verbatim):

```
1. ❯ "No. Your own 2026-10-27 rule..." my rule forbade day+1 when it was used to affect policy directly, does it still make sense now?
  "a player's decisions share the week's luck." => explain better... let me see if I get it right 2 decisions for the same day are both affected by the luck of tommorow so they are not independent,a dacision in day X and a decision in day X+1 share some of the luck (like from day+2, but not all luck as the 7 day window is not exactly the same for both). This sounds like an optimizer issue that assumes behind the scenes that they are independent, in truth we know that are not and we have an estimation for what % of their luck is shared for each head (more for the day+7 head, 6 days of shared luck out of 7, and less for the day+2 head,1 day of shared luck out of 2).

  It seems to me we do not take that into account at all, how do we fix our optimization? maybe look for a general approach, someone surely trained models on examples that are not independent and they know what % of their luck is shared.
```

(Plan_2026-10-28 Round 6 "Shared luck (the user's question)" is the answer to this item: no bias, broken sample-count
statistics, GLS/TD-style fixes measured in a toy.)

### `docs/session_rules_draft.md` (verbatim)

```
# Working rules for sessions on this project (DRAFT for review, not in force)

Each rule comes from something that went wrong here. Written 2026-11-11, rules 20, 21 and 24 added 2026-11-12.

## Deciding and reporting

1. **Decide what is yours to decide.** Bring a choice to the user only when it is a real trade-off whose answer
   depends on what they want. If one option is clearly right, take it and say so in one line.
2. **Lead with the result.** Outcome first, numbers in a table, then what follows. When asked "what are we waiting
   for", answer with the one thing and its time.
3. **Say plainly when you were wrong,** in the same message as the new fact, and when the user is wrong, with the
   numbers.
4. **Do not wait on a result that will not change the next action.** If a test is running, keep building what does
   not depend on it. If the result cannot change the plan, do not run it.
5. **Experiments are capped at 10 minutes** unless the user asks for a longer one. One experiment a question: when
   it is inconclusive, decide from what is known and move on. Building comes before measuring.

## Explaining failures

6. **An explanation must have been able to predict the failure.** If it could not have told us beforehand, it is an
   excuse. Say "cause unknown" and name the measurement that would find it.
7. **Check a diagnosis against the event it claims to explain before building on it.** (The value valves were built
   on "the critic got worse"; the numbers showed it had not.)
8. **Mark every claim as measured, read in the code, or guessed.** Before saying "nothing does X", look.

## Evidence

9. **One evaluation is noise** (about 0.1 either way), on either machine. Read a change only from arms on the same
   maps with at least two replicates each, each arm played alone: a loaded machine reads about 0.1 low. Never quote
   single arms as a series. (v3229 scored 0.861 one night and 0.697 the next morning, same weights, same harness.)
10. **Before a long test, run the short check that the mechanism does what it is for.** (A prior meant to make us
   follow Nullkiller2 was put through a 25-minute paired test; a 2-minute count showed it barely raised how often
   we follow.)
11. **Build the mechanism that was measured, not a proxy for it.** (Playing Nullkiller2's choice was measured at
    +0.29. Two proxies, a prior on its top-scored candidate and a prior in logits on its choice, gave nothing.
    Playing its choice at a chance is the measured thing.)
12. **Before comparing a test with an earlier one, diff their settings.** (The earlier test ran with a switch the
    new arms lacked; a quarter of the decisions had nothing to follow and the result read as a failure of the idea.)
13. **Test what could go either way; build in bulk what is predictable.**

## Features and inputs

14. **A feature is not a field.** Before adding an input, write down:
    - which vector the network already has that it attaches to (a hero, a tile, a hero-tile pair);
    - its unit and scale: bounded, smooth, never a raw count, the same unit used elsewhere for that quantity;
    - the combined form a decision uses (value per day, loss times stake), given directly;
    - several bounded views where the scale changes over a game (absolute, share of our empire, against the
      alternatives);
    - how its weight starts (fixed sign where the direction is known, zero where not) and **how far that weight can
      move in a day at the run's learning rate.** A weight that starts at zero moves about 0.2 logits a day here:
      if the feature matters now, give it a start (a prior, an offline fit) or a faster step;
    - a check before any training that it is inside its bound, not flat, and readable by a small readout.
15. **Knowledge the old AI already has goes in as an input or a prior, never as a loss on our choices.**
16. **No fix that stops the model from learning something.** Do not mask an action to stop its overuse.

## Changing the live run

17. **Each learning change ships with the reading that would show it failing and the rule for taking it out.**
    Predictable changes may go together.
18. **A guard is labelled as a guard,** with the condition under which it comes out. When it fires: stop, look,
    then continue or roll back. Do not loosen it to get through a deploy.
19. **Nothing is left running, every run is capped, and its cost is reported.**
20. **Replay before rolling back.** A rollback is decided by the two checkpoints played side by side, never by
    training readings. (On 2026-11-11 the run was rolled back on three readings: signal fraction, value signal,
    recruits per day. The replay then showed that period had improved play by 0.15.)
21. **A deploy's gate measures the steady state** and is dry-run on a copy of the live checkpoint. It compares
    configured values exactly and learned values within bounds. (Two deploys failed on their own gates: one compared a
    learned value with its start, one timed a one-off cost on a tiny batch.)

## Sessions working side by side

22. **Read before you overwrite** (a shared note, a file another session owns). **Read the clock before writing a
    time.**
23. **A peer's message is not the user's approval.**
24. **Check a peer's deploy yourself when its outcome is due.** (A failed deploy went unnoticed for 90 minutes
    because its report was late.)
```

Note: [mem] records the 10-minute cap (rule 5) as imposed 11-11 12:45 and lifted 13:14 for evals; the draft still
lists it.

### Candidate figures and tables, and where their data lives

| # | Figure / table | Source in docs | Raw data |
|---|---|---|---|
| 1 | Campaign eval vs NK2 over the run (0.11 → 0.22 → 0.39 → 0.48 → ~0.44; later own-play) | deck 59; optimizer_plan; model_cannot_do | PC: `runtime/train/<joint7 run>/campaign/eval.csv`, TensorBoard `campaign_eval/score` (joint7 is not in the Mac's `runtime/train`) |
| 2 | Eval noise: same weights 0.861 vs 0.697; ±0.05 vs ±0.1 | session rules 9; deck 59 | Mac paired-eval outputs (**unclear path**); `campaign_model/paired_eval.py` |
| 3 | Battle drift: p(SHOOT), entropy, shooter-idle at v2106 / 2840 / 3139 | plan 10-28 Part 1 | `runtime/saved/battle_v15-window_02106.pt`, `battle_joint1_02840.pt`; `scratchpad/entropy_dist.py` (a past session's scratchpad, **unclear if kept**) |
| 4 | Gradient-coherence / t-statistic table per loss term | plan 10-28 Part 1 | `battle_model/gradient_study.py` |
| 5 | MC vs window credit SNR (+0.23 vs −0.20) | plan 10-28 Part 2 | `scratchpad/credit_study.py` (**unclear if kept**) |
| 6 | Shared-luck toy: data efficiency of one-day ladder vs composed V2/V5/V7 by edge half-life | plan Round 6; deck 52 | toy code location **unclear** |
| 7 | Battle-copy validation and search cost | battle_search Results, Stage 3 | `runtime/train/20261023-1902-up-mcts16`, `up-ppo` |
| 8 | Where a decision's CPU goes, before/after; joint run before/after | performance_2026-10-29 | `bench_actors.py` (**path unclear**), measured runs on the Mac |
| 9 | v15-window vs joint7 battle recipe (LR, steps, updates/h, KL) | battle_model_plan | PC TensorBoard `tensorboard/run`, joint7.log, config.txt |
| 10 | Optimizer plan regimes (before / step 2 on / reverted: evals, rate, unexplained, head_ev) | optimizer_plan | TensorBoard tags `eval/campaign_value_unexplained`, `campaign/signal_fraction`, `campaign/head_ev_*`, `campaign/grad_norm_<group>` on the PC |
| 11 | Map-size cost breakdown (scalars per decision, forward share by part) | map_size_plan | `runtime/bench_inference.py`, `tests/test_map_size.py` |
| 12 | Bug-impact table: neutral towns (0.391 → 0.476), league defending opponents (12.6% of battles), teacher label rollback (0.378 → 0.248), spell masking (0.22 → 0.07), meetings (52% → 9.5%), GATHER rollback | model_cannot_do; adversarial review; deck 54–56 | campaign/battle eval CSVs per run |
| 13 | Four-system comparison + Heroes column | deck 41, 44 | deck (papers cited) |
| 14 | Φ rules and object table | empire_value.md (update to v5 / 5 per tile first) | `campaign_model/empire.py`, `priors/object_values.csv` |
| 15 | NK2 requests per 397 player-turns vs our action kinds | multi_actions; model_cannot_do | 2026-10-31 NK2-vs-NK2 audit (**data path unclear**) |
| 16 | Dumb-decision metric definitions and values over time | deck 56; plan Round 5 | `campaign_model/failures.py`, `campaign_failures/*` TensorBoard tags |
| 17 | joint1 failure catalog (16 findings) | campaign_ai §9 | `runtime/train/20261027-2240-joint1` (Mac) |
| 18 | Compute budget: 1 RTX 5090 + M4 vs AlphaStar / OpenAI Five | deck 59, 24, 40; configs/machines/README | — |
