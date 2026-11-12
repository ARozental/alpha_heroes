# Memory catalog B (newer half)

Source: `~/.claude_personal/projects/-Users-alon-projects-alpha-heroes/memory/`, the 23 newer files listed in the brief (≈290k chars), read in full, plus the `MEMORY.md` index.

**Conventions used below.**
- Times are US Central (CST, UTC-6), as the files write them. A frontmatter `modified` value ends in Z and is UTC.
- Sessions: **e5** = 31529395 (perf, infra, reviews, deploys; its monitor name is "alpha-heroes-88"); **b1** = 68ab2bd0 (science and builds); **cd** = a62ff69f; **fork** = df787a6f. None of these files comes from cd or the fork.
- "User" means the developer. "AI" means a Claude session.
- vNNNN is a campaign model version (one per learner update). A bare 4-digit number such as 8385 is a battle epoch.
- "Score" is the campaign eval score against Nullkiller2 (NK2) over 48 setups (higher is better). **Unclear:** no file in this half defines the unit. "Day 28" is the same reading taken at day 28. "Eliminated" counts our eliminations out of 48 games.
- "Paired" means the same maps in both arms. ± is one standard error unless a file says otherwise.

**Frontmatter caveats.**
- Three files have no session or `modified` field: prior-anchored-features, stale-v-valves, eval-slide. Their authors are inferred from the text below.
- Several `modified` stamps are older than the newest text in the file: pc-remote-access, battle-flat-diagnosis, deploy-safety, feature-representation-rule, overhaul.
- overhaul notes that e5's notes written before 16:55 on 2026-11-11 were "partly written without reading the clock (10-20 minutes ahead)". The mtime of stale-v-valves (09:27 local) is earlier than the 09:35 event the file describes, which fits that caveat. Treat minute-level times from that day as approximate.

---

## 1. Per-file digest

### 1.1 pc-remote-access.md
e5. Modified 2026-10-30T23:14Z; the text adds a 2026-11-09 detail. Type: reference.
- **Access:** `ssh Owner@192.168.1.178` (DESKTOP-84NG4T3; Windows shell is cmd), then `wsl --cd ~ bash -s`; strip NULs with `tr -d '\0'`. The repo is at `~/projects/alpha_heroes`.
  - The user's other project, Zonkey, trains on the same PC and uses ~18 GB of VRAM. **Never `wsl --shutdown` without the user's go-ahead** (it kills Zonkey).
- **Code sync without GitHub:** WSL has no GitHub key.
  - Make `git bundle`s for main and for vcmi, scp them to C:\Users\Owner.
  - In WSL: fetch with `--no-recurse-submodules`, `merge --ff-only`, then `git submodule update --no-fetch vcmi`.
- **Build:** `nice -n 15 ninja -C build -k 0`. Only vcmilauncher fails (no msgfmt), which is harmless.
- **Remote workers:** an SSH tunnel through Windows (ports 7700/7701, and 7751+ for campaign games); `battle_model.worker --master 127.0.0.1:7700 --token …`.
  - Tested 2026-10-30 (PC at 4036691) with the PC as master and the Mac as a worker with 1 actor: the Mac played ~49 of ~215 battles.
  - Ping is ~16 ms over Wi-Fi, so a worker's actors are latency-bound.
- **Long runs:** start them as `systemd-run --user` units or in tmux. A `setsid nohup` process died within a minute of the SSH session ending.
  - Joint unit settings: TimeoutStopSec=900, KillMode=mixed, KillSignal=SIGINT, LimitNOFILE=65536.
- **WSL keep-alive:** WSL 2.6+ (the PC has 2.7.3) stops 15–20 s after the last wsl.exe client leaves, even with units running.
  - Fix: scheduled task `\alpha_heroes WSL keep-alive` (`wsl.exe -d Ubuntu --exec /bin/sleep infinity`, S4U, session 0, at boot, restarted within 1 min). Verified across 96 s with no client.
  - b1's WMI keep-alive was removed; linger is on; Windows sleep and hibernate on AC are set to never.
  - `.wslconfig` memory=28GB (27 usable).
- b1's 7f9bda6 fixed two PC-only bugs: select() on fds ≥1024, and GPU decode of a 1-decision batch.
- **TensorBoard:** port 6008 (Zonkey holds 6006), tunnelled to WSL's own IP (172.21.77.108 on 2026-11-09). 127.0.0.1 stopped reaching WSL on 2026-11-09.

### 1.2 decide-dont-defer.md
e5. Modified 2026-11-10T00:47Z. Dates in text: 2026-10-28, 11-01, 11-05, 11-09. Type: **developer rule**.
- **Rule, third time (2026-11-09 evening), verbatim:** "Do not make a 'your decision' type replies anymore unless there is a non trivial trade off which is unclear to you, for decisions that obviously have a right answer and a wrong answer just do the right thing."
  - Trigger: an e5 report ended with "Your decision:" on teaching GATHER, copying the other AI's audit label "the user's call" without checking it.
- **Repeated 2026-11-09, stronger:** "STOP saying 'your call' to let me make decisions unless it really is my call, i.e if there is an actual tradeoff, some of these options are stupid."
  - A list of fix "directions" is not an answer. Think each through to its consequence; the user's test was "great, will NK2 just tell them to meet again tomorrow?"
  - Find the root cause with data first: the cause (an NK2 bug) was one log read away.
  - Answers are handed to a weaker AI to code, so name the file, function, rule, switch and check.
  - Another AI calling something "the user's call" settles nothing.
- **Older wording (2026-10-28):** "do not waste my time". Clear calls: decide and report. Balanced calls: recommend with a one-line reason. Major calls (direction, cost, what we train on): ping, still with a recommendation.
- **During a running training (2026-11-01):** "if you see anything good (like switching to the NK2 target you did) do it, but only if it is somewhat clear we are doing something wrong. For now, keep training and watch."
  - Fix clear mistakes mid-run (examples: the one-step move labels, permit starvation, crashes, OOM). Leave tuning alone.
- **2026-11-05:** "'Found, not fixed' => then fix, any real bugs need fixing!"
  - When a cause is unknown, capture it: a background copy of the spool; keeping logs of games with failed actions (3cc4e50).

### 1.3 step-credit-set-aside.md
e5. Modified 2026-11-10T04:11Z (22:11 CST on 11-09). Type: **developer decision** plus measurements.
- **Decision (user, 2026-11-09 night):** "for now I do not want any value on actual steps, only on day+2/5/7 ... note this function that adds a reward per step so we can use it later, for now, comment out the line that calls it."
  - The user's fears: it could teach "never force-buy with trades" (a purchase is an immediate loss of value); `step_price` 2e-4 "should probably be /100".
- **State:** commit 6021234 on branch `e5-review` (on 55f857b) comments out ppo_update's call of `train.step_credit`. The function, its tests, `step_price`, `step_class_correction` and the `campaign_step/*` TensorBoard readings are kept and marked SET ASIDE.
- **Measurements** (8 joint7 games; the unit is log empire value):

  | Decision | Step credit | 7-day credit |
  |---|---|---|
  | MOVE collecting 250+ gold | +0.0209 (97% positive) | — |
  | MOVE that changes nothing (47% of moves) | −0.0007 | — |
  | Trade-paid BUILD | +0.0007 ± 0.0043 | −0.006 |
  | Trade-paid RECRUIT | −0.0079 ± 0.0020 | −0.035 |

  - Price 2e-4 is ≈21 gold a decision at the median value of 107k; that is 0.27% of value a day and 0.095 a game.
  - A full loss is capped at 2.30 per 7-day term, so the user's "more than a loss for the entire game" was not right.
- **The user overruled the AI's "no" on an action-count price** (22:00): "0.2% sounds about right to me... I'll be willing to go as high as /10 instead of /100 but no higher."
  - Built as `config.schedule_action_price` (2466290): the price × `labels.actions_ahead` (non-END decisions from this one through d+7), subtracted from the 7-day credit.
  - The user's size: **2e-6, NEVER above 2e-5**. On the 8 games: 107 actions ahead a decision; 2e-6 is 0.30% of the outcome spread (0.0725) and 2e-5 is 3.0%.
  - Its own deploy, last in the order. *Later (overhaul):* turned on at 2e-6 in deploy 59 ("we want to turn it on") and raised to 1e-5 in deploy 61 by the user.
- **Critic check the user asked for ("is the critic that good?"):**
  - 7-day outcome minus G7's post-purchase forecast: trade-paid BUILD −0.9% ± 0.8% (right on average; a blind forecast would read −8.7%); trade-paid RECRUIT −2.6% ± 0.7% (too optimistic).
  - G7 explains 54% of the 7-day growth variance.
  - User on recruits: "this should be negative unless and until we use these troops to gain something."
- **Already live:** V_end's step term (end_advantages td, decision_price 1e-4) at attributed_end_weight 0.2 × ~0.66, ≈13% of the advantage.

### 1.4 bug-audit-2026-11-09.md
e5. Modified 2026-11-10T12:28Z. Covers 2026-11-09 to 2026-11-10 ~09:00. Type: measurement log, incidents and deploy log.

**Method.** 8 finished joint7 games (v2479–2481) plus checkpoint v2482. Labels, prepass and credit were replayed on the Mac CPU (10,879 decisions) beside 2 NK2-only games.

**Finding 1: the teacher's advice loops on visits to our own heroes and towns.**
- Advice.cpp `advise()` reruns `nk2->makeTurn()` at every decision and takes the first request, so a hero-chain visit stays "first".
- Teacher MOVE labels onto our own hero 54.8%, onto a hero or town 71.0%. Real NK2 does this 1.7% / 2.5% of the time (2,678 moves).
- Our own MOVEs onto our own hero: 54.7% (2,511 of 4,589).
- Movement points spent: 25.7% meeting our own heroes, plus 18.7% stepping into our own towns.
- 70.5% of meetings repeat the same pair's meeting that day; 52% of those repeats return both armies to a state they already had.
- 314 meetings a game. Exchange decisions are 30% of all decisions. No failure metric counted any of it.
- The "fix directions (the user's call)" in this section are **SUPERSEDED by 1b**.

**Finding 1b: the root cause (same evening), read from NK2's "Performing task" lines.**
- 98% of advised meetings come from `GatherArmyBehavior::deliverArmyToHero`.
- 51–63% are **PHANTOM deliveries, an NK2 bug.** The receiver R stands still, so AIPath::containsHero misses it, and the address-order swap in HeroChainCalculationTask makes the carrier's actor "C joined with R". `howManyReinforcementsCanGet` then counts R's army again: priority 10–30k, never satisfied.
- Real NK2 has the same bug: game 9101 shows 30 of 42 deliveries, stopped only by movement points.
- In 40% of advised meetings the visitor holds under 1/10 of the other's army.
- adviseExchange's destination is the visitor in 39%. After a followed meeting the first exchange choice is the plan's delivery in only 35% (394 of 1,120).
- 28–56% of deliveries carry town or dwelling troops (the town-entry loop; GATHER untaught since 2026-11-05).
- **Design M1–M5**, given to the user and built by b1 the same evening:
  - M1: skip a delivery whose chainMask holds the receiver's own actor.
  - M2: the exchange's receiver is the hero met when the meeting was advised.
  - M3: MULTI inside the exchange span.
  - M4: remember each pair's settled armies (the answer to "tomorrow").
  - M5: metrics.

**Finding 2: the credit is one number per player-day.**
- 96.5% of the final advantage's variance lies between player-days (schedule part 98.1%, share term 99.7%). The schedule credit (80% of the advantage) is pure 7-day Monte Carlo minus a baseline.
- A decision's own step has 13–18% of the MC advantage's std, correlated +0.11 (growth) / +0.01 (share).
- V_end EV 0.82 is **CORRECTED** in the same file to 0.35 over the 1/players prior. G7 0.54; privileged V7 0.30 (public 0.15).
- The outcome credit telescopes, so the step is "buried, not missing".
- Step-credit replay: advantage std 0.081 → 0.038; between-player-day share 98.1% → 85%; credited rows 82% → 99%. *(Set aside by the user; see 1.3.)*

**Minor findings.**
- share_trust 0.40 (public EV) scales the privileged baseline.
- loss_shares frozen since v1420, so the teacher weight sits at 2.03 (~60–64% of the gradient) and the "score" teacher_mix is dead.
- The schedule credit has no decision price.

**Verified clean.**
- Model Φ equals the ledger Φ (97% within 0.5%).
- Recorded vs trainer logp (no −inf).
- Day cap alignment (2,998 of 3,000 games end at their cap; 10–26% of decisions lack the 7-day label by design).
- Battle credit path. Battle learner: 48 steps, KL 0.001–0.002, waits 5–19 min an update.

**Observed, by design.**
- BUILDs funded by trades lose ~82k Φ a game (24% of builds).
- RECRUIT's immediate ΔΦ averages −733 (the trade-funded quarter: 149 of 591 recruits); a gold-paid recruit loses only the unled share's designed 4.7%.

**e5's review of b1's build (55f857b).**
- M1/M2/M4 correct. M5 counted a garrisoned hero as a meeting (fixed: h_garrisoned).
- M3 off is right: teacher ce 1.668 → 2.358. e5's earlier note saying b1 overstated this was **WRONG**, caught by a re-run.
- Built the gentler `exchange_span_advised` (1.668 → 1.697).
- b1's reason for keeping the step credit off ("blocked by Φ") was **wrong**.
- **GATHER label:** NK2's re-visit of its own town was dropped as unmatched at 752 of 10,781 advice calls (7.0%), and our model chose GATHER once.
  - Built as Advice.cpp visitAgain (env `ALPHA_MAP_GATHER_ADVICE`) plus `nk2_gather_weight`.
  - A/B: 313 labels, model median p 0.00003. Teacher ce +0.06 at weight 0.1, +0.13 at 0.2, +0.31 at 0.5, +0.60 at 1.0.

**Deploys and verdicts in this file.**
- **Deploy 42**, 2026-11-09 22:56:37, e5. 2466290, every switch off: meeting fixes M1/M2/M4 (env ALPHA_MAP_ADVICE_FIXES, on unless "0") and M5 metrics.
  - Baselines before: evals 0.429 (v2504), 0.440 (v2524), 0.432 (v2549); battle 0.0912 / 0.0887.
  - v2557: teacher_ce 0.842 → 1.295 (as predicted); own_hero_moves 0.77 → 0.33 by v2560; share_multi 0.29 → 0.135.
- **User, ~00:00:** "do not roll back on the next update either way, see at least 3 updates." Then: "errors and even clear bugs should be fixed immediately."
- Evals after: v2577 0.420, v2602 0.405, v2627 0.431 (mean 0.419 against 0.434 before). Verdict 03:06: no change, the fixes **stay**. v2652 0.469.
- **Deploy 43**, 03:11:33. affa756 / vcmi a9b7b0a94: the stalemate battle fix (below).
- **Deploy 44**, 04:34:54. 592fa6f: `exchange_span_advised=true`, b1's idle-movement metrics (414c622) and the inference-server wait diagnostic.
  - Span evals: v2679 0.511 ± 0.058, v2704 0.491 ± 0.052, v2728 0.489 ± 0.047 (eliminated 9, 10, 6). Verdict 08:38: the span **stays**.
  - Exchange behaviour: decisions per meeting 1.28 → 2.56; chosen ARMY 0.03 → 0.37; meetings a game 315 → 60.
  - The exchange cap of 8 now binds in 14% of dialogs; raising it to 16 was proposed.
- **Deploy 45**, 06:35:25. 897db08:
  - Fix for an **NK2 endTurn race** (vcmi d7868e680). 13 eval games were stall-killed in 151 evals: AIGateway::endTurn's loop could end the next turn on fast 2-player maps. Fix: a static count of started turns per colour.
  - The inference server parks requests that wait for weights (8501601) instead of blocking its one worker thread.
- **Deploy 46**, 08:48:04. 36a38de: `nk2_gather_weight=0.1`, plus 1d31d6b (the teacher_set_agree END fix) and 0c8ebfd (the Linux stall sampler). *This caused the GATHER collapse (1.10).*
- **Stalemate bug** (found 00:30, pre-existing):
  - A battle where no side can flee and nothing is damaged for 50 rounds went on forever.
  - joint7.log shows 298 such kills in 9.5 days (~1.3 an hour); one Mac game made 1,723 decisions in 62 s with a 3.8 GB client.
  - Fix, decided by e5 and built by b1: the side whose living army is worth less loses (a tie goes against the attacker).
- **Idle movement after the meeting fixes** (b1):
  - Freed in-town ENDs 10% → 26%. With a reachable unvisited object, p(END) 0.20 → 0.38.
  - The meetings' probability went to END. `schedule_action_price` was HELD until idle movement falls.
- **Stall triage** (165 kept PC run dirs):
  - 127 were idle battle queue clients (harmless).
  - ~10 were stalemates.
  - **25 were real:** AlphaBattleAI waiting in recv for the battle inference server (no receive timeout). The cause is unknown; diagnostic lines were added.
  - 4 had every thread in futex (unknown).
  - A later stall at 06:51 had no server line, so the request never reached a worker.
- **User 07:35:** "why is the battle_value_loss creeping up?"
  - Mix: real battles went from 4% to 60% of an update over Nov 2–4; real ~0.008 vs generated ~0.005, so 0.0047 → 0.0070.
  - The Nov 5 deploy brought it to 0.0062; it crept back to 0.0073 and has been flat since Nov 9.
  - Entropy 1.01 → 0.76, defend 0.059 → 0.106. Continued in 1.7.

### 1.5 meeting-fixes.md
b1. Modified 2026-11-10T01:26Z; the text runs to ~10:15 on 11-10. Type: design and deploy log (b1's view of 1.4).
- **User:** "do these fixes, if you think the other AI is wrong tell me instead."
- **Commits:** M5 10bcddf + 6d54195 (failures.py own_hero_moves, repeat_meetings, meeting_movement, town_movement); M3 2891017 (`exchange_span_multi`, default False); M1+M2+M4 b49632e (vcmi 04b7246e3: `Nullkiller::adviceFixes`, `exchangeSettled`; unset for real NK2, so the eval opponent is unchanged).
- **A/B** on 8 maps × 14 days, v2502 (labels never change play):

  | Measure | Before | After |
  |---|---|---|
  | Advised MOVEs onto our own hero | 52.2% | 9.5% |
  | Repeat meetings the same day | 75.5% | 25.5% |
  | Pair re-advised right after meeting | 58.9% | 0% |
  | TRAVEL share of labels | 10.9% | 16.3% |
  | Teacher ce | 1.10 | 1.67 (MOVE 1.29 → 2.52) |

  - **M3 left OFF:** ce 2.36, exchange END labels ce 5.06. This is the 2026-11-05 collapse shape.
- **20:45:** e5's review merged (55f857b). **22:30, user:** "the other AI will deploy for you, no need to deploy." From then b1 did not deploy or move main unless asked. *(Later reversed in practice: b1 ran deploys 58–65 on e5's go.)*
- Deploy 42 PC game: advised onto hero 41.7% without the fixes, 0.8% with.
- **Idle metric** 414c622: idle_movement / _town / unmoved_heroes / _town. Before 0.095 / 0.41 / 0.048 / 0.20; after (~v2625) 0.269 / 0.646 / 0.195 / 0.439.
- **Exchange drift under the span (v2652 → v2704):** MULTI 0.947 → 0.507, ARMY 0.038 → 0.460. At ~08:20, 14% of dialogs hit the 8-decision cap (1% before).
- **In-set check:** ARMY choices at advised exchanges 3% → 34%, 89% of them inside the teacher's label plus accept set. All choices inside the set 35% → 55%.
- **Watcher incident:** `grep -oE "[0-9]{5}"` on an ABSOLUTE checkpoint path matched the run folder's "20261031", so the watchers never fired from ~00:50.
- **/tmp cleanup** at 01:00 on 11-10 emptied b1's old scratchpad folders (type_drift.py lost; rewritten).
- **Stalemate fix** (b1, 00:45): vcmi a9b7b0a94 / main affa756. The weaker side's units are killed via StacksInjured, then it is a NORMAL defeat. Tested with VCMI_BATTLE_STALEMATE_ROUNDS=2.
- **~09:10, user offline:** "make sure that the PC continues to learn." b1 started a Mac keepalive.sh that restarts the run after 6 min down.
- **~10:15, user:** "we are no longer on the local network of the PC." The PC was unreachable from the Mac until the evening (this caused the 9.5 h GATHER-collapse blind spot).
- **b1-aux** (PPG value phase) safety nets after e5's review: non-finite steps skipped, undo on KL > aux_kl_max 3e-4, its own Adam. Real data: held-out −1.6 to −3.7%.

### 1.6 battle-flat-diagnosis.md
e5. Modified 2026-11-05T18:39Z; the text runs to 2026-11-10. Type: measurement, decision, and a later correction.

**Measured 2026-11-05.**
- accumulate 24 at lr 1e-5 gave 4 optimizer steps an update. Update KL 5e-5 (aim 0.015), clip fraction 0.05%.
- ~4.6 updates an hour, with 777 s of each update waiting for real battles. GPU 1.3%.
- Battle eval **+0.088 to +0.097 for 53 evals** (frozen).
- Real battles are lopsided: attack wins 84%, defence retreats 70–85%. The assets denominator shrinks their reward swing to 1/100.
- outcome_share pinned at 0.15. 33% of decisions have entropy under 0.3.

**User's decisions (2026-11-05):** no more compute for battles; credit window:outcome 50:50; wants a learnable shooter penalty or a "next shot" head.

**e5's decisions, deployed as 8797125.**
- accumulate 24 → 2 at lr 1e-5 (not the plan's 3e-5, by the user's LR rule); beta2 kept at 0.99.
- outcome_share 0.5.
- Lopsided advantages × 1/(4p(1−p)), capped ×12.8. The stake weight min/10k was REJECTED (it would cut early-game battles).
- full_shot / half_shot_only features; DEFEND × full_shot starts at −2.
- The entropy floor was REJECTED (it had been the only push toward shooting); the foresight head was deferred.
- Bug fixed: the adaptive outcome share never reached the pool workers.
- The PC had no `sample` tool, so headless.sh now dumps Linux thread states. 205 generated battles had stalled.
- First update: KL 9.2e-4, signal 0.12. e5 had called that fall "not a regression".

**2026-11-10, found WRONG by e5 itself.**
- After 24 → 2: steps an update 4 → 48, signal_ratio 0.25 → 0.13 (below the pure-noise median of 0.155).
- e5 at first read this as a trunk noise walk causing the entropy and defend slide. Lesson: check signal ABOVE the noise median, not "as predicted".
- **The same day e5 also corrected that:** at accumulate 24, an update's 4 steps reuse the same ~6,144 decisions, so pure noise reads median 0.31 (rms 0.456). The pre-Nov-5 reading of 0.25 was never evidence of signal. The true signal is ~0.5% applied (0.2% raw).
- A persistent 0.5% signal would outgrow the noise after ~200 updates, and over 1,000 had passed. So the flat eval means the steady direction is unhelpful or not stationary.
- e5's concentration finding: an update's gradient rests on ~4 games (value) and ~14 (policy). `battle_length_scale` was built on b1-aux 85ae0cd.
- Rollback point: runtime/saved/battle_joint7_8385_pre_8797125.pt.

### 1.7 battle-noise-walk.md
e5. Modified 2026-11-10T16:23Z. Type: measurement log plus a **developer rule**.
- **User 08:00:** settle it, then fix the battle training. More compute is "a trade off", not a solution; "chasing against a moving target is solvable."
  - **Rule:** "an explanation counts only if it would have let me predict the failure in advance."
- **STANDS: there is no moving target.**
  - Every checkpoint scored on the same buffer (168 battles, 6,187 decisions): 8385 0.00386, 8396 0.00390, 8491 0.00386, 8584 0.00483, 8633 0.00402, 8693 0.00427, 8746 0.00413, 8797 0.00394. V is no better than on Nov 5 (±10% jitter).
  - 8797125 gave ~10× the noise variance for ~1.5× the steady movement.
  - **Deploy 47** (e577a76, accumulate back to 24, live 08:57) is right but **not a cure**.
- **WRONG** (retracted ~10:25): "accumulate 24 restores 24% signal"; "predictable from signal_ratio 0.25"; "the entropy and defend slide is the walk's damage" (they moved as much before Nov 5).
  - VOID: deploy 47's check "signal_ratio > 0.155".
- **Measured.**
  - True signal from the cosine of Adam exp_avg across checkpoints: raw +0.009 ± 0.010, applied +0.023 ± 0.004, i.e. ~0.5%.
  - Weight moves 8090..8386: cosine +0.048; end-to-end 0.57 against 0.37 for a pure walk; a steady part of ~1.6e-3 an update on a trunk of norm 55.8.
  - Participation number of the gradient energy (98 games): policy 14.1, value 3.9, foresight 19. The heaviest game holds 21% (policy) / 43–46% (value).
  - 6 of 168 battles have 150+ decisions (28% of decisions; WAIT/MOVE/DEFEND stand-offs).
  - Weight 1/(1+(n−1)/30): policy 14 → 30 games, value 4 → 28.
- **b1-aux plan (70bdd61), three deploys:**
  1. The readings alone (optim.SignalShare, games_by_weight) for 50+ updates.
  2. `battle_length_scale=30`. Prediction: games_by_weight 33 → ~65, value_loss_fresh spread down a third, signal_share ~2×.
  3. `aux_value` (PPG). Splits by game: held-out −4.0% ± 1.5 / −6.1 ± 3.0 / −3.5 ± 3.2 against a shuffled control of +25 / +100 / +88%. Gated on re-running after step 2.
- **Lessons:**
  - A signal reading must compare DIFFERENT battles (by game).
  - Held-out data must be untouched by the phase.
  - Compute a diagnostic's pure-noise value under the recipe.
  - 8797125 changed many things at once, so its symptoms cannot be attributed.
- *Later status (other files):* step 1 went live with deploy 48 (18:38); step 2 went live in **deploy 54** (05:08 on 11-11), and 8 updates later games_by_weight read 66 against 48 and value_loss_fresh sd 0.0009 against 0.0021. **No file in this half records step 3 (aux_value) being deployed.** The battle eval is still "+0.095, shooters idle FAILURE line at every eval".

### 1.8 prior-anchored-features.md
No session field. File mtime 2026-11-11 00:45. Text dates 2026-10-28, 2026-11-10 and 11-11 00:45. Author of the later parts: e5 (it says "asked b1"); the original author is unclear. Type: **developer rule** (design).
- **Rule (2026-10-28), two cases:**
  - A value NK2 already assigns starts the learned table at that value, as a fixed bias with decayed corrections. Keep it in a sane logit range: ">4 is crazy".
  - Any other feature is "just a number close-ish to higher layers that affects the final decision only after interacting with the trainable params". No fixed weight on the logit.
  - A small prior is OK "IF we have something that is easily learnable that can amplify it". Under Adam a weight moves ~lr a step, so amplifiers are reparametrised (log-gains × CORRECTION_STEP 10, tanh-bounded).
  - Nothing big (Φ in gold) enters a learned layer raw; the user flags "it will explode".
- **2026-11-10 night:** "you mean not actually gold but % of the value of our empire scale by something right? gold is unbound. Also, why not just the %". On the network outvoting a prior: "this is ok, it is a good thing to be able to do so."
  - Fact: the logit priors sum gold through value_logit = 2 tanh(g/300) + log(1 + g/1565), which is unbounded (10k → 4.0, 1M → 8.5).
- **2026-11-11 00:10:** "'night_gain' can be a good auxilliary loss, it is ok to learn it as such but it should not be a part of the V." "…it will not work on very large maps at all, maybe dividing by our total networth makes the number too small, keep monitoring and finding ways to imporve as you push our plan step by step."
- **Share-of-Φ prior measured (b1's 25866df, ALPHA_PRIOR_SHARES):** v2912 0.485 → 0.424 (−0.061 ± 0.046). Eliminated 12/11/7 → 16/13/14; heroes at the end 1.31 → 1.03. **NOT adopted:** the user's "too small" worry held.
  - Parked idea for very large maps: a size-aware REF fixed per game.

### 1.9 move-further-windows.md
e5. Modified 2026-11-10T20:11Z; the text runs to 2026-11-11 01:30. Type: design, measurement and deploy log.
- **User's ask:** MOVE targets covered only the 33×33 tiles on the hero's level, so nothing past a gate. They suggested "the closest 8,712 points regardless of a gate". "Make the right fix (better modeling and no significant impact on cpu usage needed) then fix it here, do not push to PC yet."
- **Built: schema 17** on branch `move-windows` (75cfe7d). The main worktree is checked out on this branch from then on.
  - `further_windows`: one extra square window per moving hero, in a spare slot.
  - `unseen_exits`: TILE_UNSEEN_EXIT 512 / OBJECT_UNSEEN_EXIT 8 flags, priced as exploration.
  - Squares were chosen over a reach-ordered list because the window features need neighbours and unreachable tiles.
- **Measured.**
  - 43% of hero-turns reach tiles outside their own window (median 31 vs 77 inside). A 2nd extra window would hold a median of 8, hence a cap of 1.
  - Old vs new code: 42 outputs identical.
  - Client CPU 4.4–4.7 ms a decision. CPU forward 27.2 ms at 3 windows, +1.6 ms a window.
  - The PC's campaign inference runs on the CPU (`inference_device=cpu`, 12 processes). e5 first said GPU: wrong.
  - window_budget 4.0 overflowed: at 4.0, 37 of 120 updates went over budget (191 s against 164 s). Set to 5.0.
  - The "+~1 window a decision" estimate was too high: 0.0–0.2.
- **The campaign update is ~95% of the cycle** (124–164 s of ~175 s): the learner sets the pace.
- **Deploys:** 49 (19:18, schema 17 inert); **52** (23:44:58, 80f7398: further_windows=1, budget 5.0, night_gain frozen, paired_eval); **53** (01:27:50 on 11-11, 85e6f2e: unseen_exits=true).
- **Paired tests (Mac, pinned 48):**

  | Test | Result |
  |---|---|
  | Further windows at v2886 | +0.014 ± 0.037 and −0.020 ± 0.040; pooled −0.003 ± 0.024 (no immediate effect) |
  | unseen_exits at v2886, 3 replicates | 0.472 → 0.531, **+0.059 ± 0.025** (the immediate gain) |
  | Further windows alone vs both off | −0.012 ± 0.027 |
  | Share-of-Φ prior | −0.061 (not adopted) |
  | Night power 3.18 | −0.001 ± 0.030 (dropped) |
  | Fade test v2839 → v2886 → v2912 | 0.504 → 0.484 → 0.462 |

  - Null pairs: −0.022 ± 0.060 and −0.070 ± 0.042. Identical arms 35 min apart: +0.050 ± 0.033. So: **only side-by-side arms, 2–3 replicates.**
- **e5's cadence rule (not the user's):** one learning change a deploy, each paired-tested first; the next change after one clean PC eval.
- PC evals: deploy 52's first, v2934, 0.579 ± 0.054 (eliminated 7). Before it: v2912 0.493.
- **Open:** the U-Net reads each level apart, with no link between a gate's two ends. Noted in docs/model_cannot_do.md (the user's untracked file).
- Never `git stash` in the main worktree: it holds the user's uncommitted docs.

### 1.10 gather-collapse-2026-11-10.md
e5. Modified 2026-11-11T04:21Z. Type: **incident** analysis, developer rules and deploy log.

**Incident.**
- Deploy 46 (GATHER label 0.1, ~v2735). Evals before: 0.511 / 0.491 / 0.489. After: v2748 0.475, v2761 0.380, v2774 0.314, v2788 0.345, v2801 0.402, v2815 0.271, v2830 0.482, with eliminated 15–28 of 48.
- **Nobody looked for 9.5 h** (the Mac was off the PC's LAN).
- The score-driven teacher weight swung 2.47 → 13.1 → 2.54 on single evals.

**Explanation** (12-game A/B vs NK2; claimed to "have predicted it").
- v2728's heroes were already being killed in the field: 18 kills on the enemy's turn, ~5k of army each, while 80–90% of the army sat in garrisons (a turtle).
- GATHER moved half the army onto heroes. The death rate was the same (16 kills) but each death cost 20–43k (218k lost against 92k).
- **"NOT 'GATHER strips towns': the army dies first."**
- Self-play eliminations 0.2% → 6–8%; value target std doubled; value_loss_fresh 0.0005 → 0.002–0.008; head_ev_g7 0.52 → 0.28; the value's share of the trunk gradient rose to 0.59.
- GATHER is not bad in itself: the games we survive score 0.60–0.80.
- The model's own danger reading predicted the deaths (END at 0.8–1.0 outside a town: killed 68–71%); danger_gain had trained to 0.37.
- **e5's errors here:** told the user the rise began "at the first update" (wrong: restart spikes); counted one "out before day 56" game that was a WIN; ran **unpaired** Mac comparisons (`play.make_map` regenerates maps; b1's catch); told the user evals come "3 hours apart" (wrong: ~77 min).

**User's rules (2026-11-10 evening).**
- A huge jump in campaign_value_loss is the first sign of trouble: stop immediately. "no way such an explosion didn't mess up everything later".
- Rolling back alone is not enough: understand what failed and why. Only explanations that would have foreseen it count. Don't dig past the cause (per-town traces were "overkill").
- **Refined 2026-11-11 00:10:** "The campaign_value_loss had a huge jump but it is ok now as it goes down fast, as long as it keeps going down I'm happy, if it goes up after a fix and back down quickly it is also ok, jumping up and staying up for long or not decreasing over a long time is bad. this is a prerequisite to learning well and pushing it down is needed (but not sufficient) to have good evals later."

**Done.**
- **Deploy 48**, 18:38, second attempt (the first refused itself on a grep that counted a comment line). checkpoint_02728 restored as v2839; the abandoned weights and eval rows quarantined. Eval of the restored weights: 0.497 ± 0.052, eliminated 11.
- **Value-loss guard** (`value_jump_stop` 3.0, ff7acd2 / fab5eeb): it would have fired only at v1137 (the 11-05 collapse, ×29) and at v2749.
- **Deploy 49**, 19:18, second launch (the first refused on a log line update_check never prints): guard plus schema 17 inert.

**User 19:00:** "we do need this fix (no revert) AND the battle and underground tiles AND something that makes it easy for heroes to learn they should run away ... an auxiliary loss for will we lose a hero on the enemy turn? a feature?" A restart from scratch is OK if the explosions cannot be stopped.
- The foresight hero head already ranks tonight's deaths (AUC 0.86 / 0.93) but is under-confident. Acting on it is the gap: TRAVEL and END priors had no danger term, and our own town was priced as open ground.

**Run-away.**
- v1 (63daa24) failed its own prediction: big kills 18 → 17 against a prediction of ≤12. Unpaired, though.
- v2 (7441d1e, shelter_open + signed_offer) went live as **deploy 50** at 20:14.
  - b1's paired test, 36 maps: 0.461 → 0.595 (+0.134 ± 0.041), eliminated 11 → 4.
  - PC: v2862 0.565 (eliminated 7), v2886 0.500, v2912 0.493. Pooled over five estimates: **+0.077 ± 0.023**.
- **Deploy 51**, 21:40: push_* gradient diagnostics. night_gain was pushed down by the teacher (−0.0072 ± 0.0016, 8 of 8 updates).
- **Procedure agreed by e5 and b1 (22:15):** every behaviour change gets `campaign_model.paired_eval` on the 48 setups BEFORE the PC; PC evals only watch for harm.

**GATHER return.**
- Pre-test on v2830: eliminations 20.7 → 20.7 under run-away code, so the ramp WAITS.
- The night term moved deaths from the hero (14 → 5) to the town beside it (together 2 → 11).
- b1 then built the towns' night term (f032d3f).
- Ramp code: 5d31166 / 27c0994. Mac trend v2912 → v2956: +0.045 ± 0.047.

### 1.11 night-danger.md
b1. Modified 2026-11-11T01:28Z; the text runs to ~09:15 on 11-11. Type: design and measurement log (b1's view of 1.10 and 1.12).
- **Run-away design (schema 18):** night term = [P(lost tonight where the day ends) − P(lost here)] × (army + hero + replacement), at a MOVE's target and a TRAVEL's `travel_stop`, with `night_gain`. In our own town P uses `sheltered_danger`.
- **Design rule, measured:** put new knowledge in the LOGIT prior only and never change the meaning of inputs that trained weights read. With the night term also in move_features, in-town heroes went to more dangerous tiles (0.098 → 0.110); logits only: 0.098 → 0.015.
- **v2 paired (36 maps):** enemy-turn big losses 64 → 49; out before day 56 9 → 4. What still dies: unseen enemies (~4 of 15), trapped since morning (~6), avoidable (~5). 14 of 15 ended the day with 0 movement left.
- **Lesson:** one 48-game eval resolves about ±0.1 at 2 se, so pool estimates and never compare one against one.
- **night_gain FROZEN** (2458a27 + 23d10d1; rides deploy 52).
  - Over 23 updates: teacher push −0.0054 ± 0.0008 (21 of 23 below 0); policy +0.00002 ± 0.00022 (no sign). b1's "Adam carries a steady small push" reason was wrong.
  - **A departure from the user's priors rule; reported to the user as theirs to reverse.**
  - danger_gain reset to 1.0: −0.002 ± 0.031, so it is "NOT evidence of harmful erosion".
- **Calibration** (3,753 hero-ENDs): reading 0.29 → 6% lost, 0.50 → 10%, 0.70 → 37%, 0.90 → 77%; in town 1 of 450. γ 3.18 tested neutral, so dropped.
- **Unseen kills:** 34% of kills happen at a reading under 0.2. The signal is a stronger enemy SEEN within 5 tiles in the last 6 days. These are small heroes (median 2,100 gold), so NOT built (it would need C++ sighting memory).
- **Towns' night** (f032d3f, ON by default in 74480f8): v2830 +0.079 ± 0.052 (eliminated 55 → 37); v2912 −0.013 ± 0.040. Went live in **deploy 54**.
- **Eval-slide investigation** (see 1.20):
  - measure_a: the prior's effect is constant (+0.215 to +0.235), but P(penalized entry) rose 0.07 → 0.13.
  - measure_b: NK2 labels penalized entries twice as often as we choose them.
  - The detach idea was REJECTED (NK2 has its own danger model, so it would double count).
  - Built OFF: `nk2_night_drop` 4785d0f.
- **Deploy 55** = the GATHER ramp, with stop rules.
- **User's value-noise hypothesis and the valves:** see 1.12. b1's critique points:
  - The guard's ratio is sticky.
  - Adam undoes loss-weight valves.
  - Throttle the policy through the KL target, not the credit weight.
  - Check corr(kl, grad_shared_value) first.
- **Trunk experiment (b1, 07:54–08:35):** 67 games split by game. Held-out mean EV: heads only (C) 0.350 → 0.369; free trunk (B1) → 0.321; 10× rate (B10) → 0.226. **A separate value trunk fits V WORSE.** A 300-game best case was started ~09:15; **its result is not in this half**.
- The campaign eval is started by the battle trainer's loop once a battle epoch, so the cycle is ~80–90 min.

### 1.12 stale-v-valves.md
No session field. File mtime 2026-11-11 09:27 (local). Author by content: e5. Type: design, measurement and outcome.
- **User (~07:00):** "I think the increases then decreases in gains come from changes that improve game quality, then V estimations become worse … then the policy changes because of the noisy V much more than it would if we had separate networks … These are not ideal solutions, and you need to come up with better one(s) … If there is no better solution than just accept it or restart a run, this is ok, but if there is we should use it."
- **progress.csv check:**
  - KL sat at its target (0.020–0.027) throughout and did not rise with the value's share (corr ≤ +0.15).
  - The value's share of the shared gradient: calm 0.05, collapse 0.11–0.18.
- **Design:** r_v = the 7-median of the fresh value loss over its EMA(50); s = clip(1/r_v, 0.1, 1) scales V's gradient into the trunk; the KL target × clip(1/r_v, 0.25, 1); rate-limited ×1.25 an update.
- **Deploy 56**, 07:22:59, 45b8692, valves ON. GPU check: the replayed graph reads the buffer (shared grad 71.83 → 36.24 at s = 0.5).
- **User's two-network idea (~07:40):** separate P and V nets plus a KL term between their hidden states; "'valve' sounds kinda hacky so I'm worried".
  - e5's answer: this is the Phasic Policy Gradient family. Distill V's OUTPUTS into an auxiliary head on P instead. Cost ~2× learner compute.
  - b1's cheap test showed a separate trunk fits V worse (1.11).
- **e5's correction to the user:** the logged value EV 0.17–0.20 is ladder-based. Against real labels, held-out EV is V2 0.13, V5 0.05, V7 0.03, growth 0.57 / 0.52 / 0.49, V_end 0.66.
- **09:25:** corr(value_loss_fresh, value_target_std²) = 0.98–0.99, so high-loss updates are the ones with eliminated games. The valve closed on ordinary noise.
- **09:35, THE PREMISE FAILED** (b1, eval/campaign_value_unexplained):
  - Unexplained share: calm 0.823 at loss 0.00056; collapse 0.857–0.878 at loss 0.0016–0.0048.
  - The 8× loss rise was target spread. V's explained part went 17.7% → 12.2%.
  - Decision (e5, b1 agreeing): no band, both valves come OFF with the next deploy (done in deploy 57). "The critic explains 12–18% always: that, not staleness, is the learning limit." The two-network idea loses its main argument.

### 1.13 teacher-question-2026-11-11.md
e5. Modified 2026-11-11T16:23Z. Restored from e5's context after **b1 overwrote the first version by mistake**. Type: plan and developer asks.
- **User asks, in order:**
  1. ~08:45: "I fear that the teacher does more harm than good given that it has different actions from us and that we have difficulties distinguishing between actions that should be similar (like moving Hero A to point B then Hero C to point D VS doing it in the other order. Should we remove this teacher completely?" The labels may "harm our ability to learn past the very earliest points". A score per action from NK2 is better than its recommendation; "some vector that we are able to attribute to specific actions cheaply and easily is best".
  2. ~09:05: "my plan is to remove it as a teacher completely but still having its insights". Surpass NK2 without cheating (cheating = things that do not scale to larger maps, or exploits vs NK2 such as training against it).
  3. ~09:35: a major overhaul. Hidden-state MSE between P and V: "my intuition says good". Done = 0.6 from here in a day, or 0.5 from clean in a day. Then interrupted: "instead of doing everything now give me plans".
  4. ~10:05: plan v1 rejected (see 1.21).
- **e5's corrections owed to the user:**
  - "nothing accepts another hero's move" was false (a8b587e planEntries does).
  - Its own computed candidates add +0.000–0.002.
  - "do not feed NK2's numbers in" was wrong: the follow test gave +0.293.
  - The teleport gap is 9–15% of the priority mass, not ~8%.
- **Plan v2 (10:10):**
  - Every new number sits as a channel on an existing (hero, tile) or (hero, area) pair. Four units: logit or share of Φ; log strength ratio + loss share; time as a discount 1/(1+t), 1/(1+t/3); preference as a gap to the best.
  - NK2 priority as ln(priority / best), clamped.
  - First step carries the destination; a plan vector.
  - β by paired test (0, 2, 4).
  - NK2 inputs dropped at random in some games.
  - Gate: a readout beats 1.51 and ≥70% top-1.
- **Plan v3 (10:35), after the user's comments.** Their words: time "should be 2 features"; "Value on arrival and Value per movement points"; "multiple bound ways to count value everywhere"; on the log strength ratio, "can explode! a bunch of pixies VS a strong hero will make this feature override everything"; "I want learning to be smooth and fast".
  - Six bounded views of any gold amount: g/(|g|+2k), g/(|g|+20k), share-of-empire forms, days-of-own-growth forms.
  - Risk as r²/(1+r²).
  - Sign-safe starts.
  - Gates: no explosion, nothing dead.
- **Costs told to the user:**
  - NK2 runs beside every decision: 4 ms at 36×36, 16–23 ms at 72/144.
  - NK2 reads exact monster counts (patch to human-visible counts later).
  - An exception to "C++ decides nothing". Build estimate 1–1.5 days.

### 1.14 teacher-measurements-2026-11-11.md
b1. Modified 2026-11-11T16:22Z. Type: measurement log.
- **Follow test** (4d3f8b9, ALPHA_MAP_FOLLOW_ADVICE, test-only): playing NK2's matched advice at v3045 gives **+0.293 ± 0.059** (144 pairs; 0.520 → 0.813; day 28 +0.213), but eliminations rise 32 → 42. Treated as an upper bound for an NK2 input, not a target.
- **Order worry:**
  - Decisions about other heroes are 31% of teacher CE.
  - At most ⅓ of unfollowed MOVE labels are reached later the same day (12% for TRAVEL). Mostly real disagreement about targets.
  - The prior alone ranks NK2's label 11th–18th.
- **Readouts:**
  - On the policy base, a ≤64-weight readout gains 0.8%.
  - On the prior base, known features recover 67% of the prior → policy gap, and the plan head brings it to 85%.
  - Plan head top-1 0.40, top-3 0.58.
- **NK2 candidates:**
  - 16 a hero hold 95–100% of the priority mass (8 a hero holds only 78–85% on big maps).
  - Teleport paths are 9–17% of the mass and cannot be expressed.
  - The list is unstable (top-1 the same 63–71% of the time).
  - Advisor time: 4.2 ms at 36, 16–17 ms at 72/144 (p99 42–60).
  - "Small beside our inference" was never measured and was retracted.
- **Valves:** premise disproved (1.12).
- **Outcome:** deploy 57 shipped schema 19. The candidate prior (β 2/4) and the label logit prior moved neither score nor following (advised entry taken 29–36%). The follow chance works.

### 1.15 no-long-experiments.md
b1. Modified 2026-11-11T18:59Z. Type: **developer rule** (temporary).
- **User ~12:45 (via e5):** "no experiments that take over 10 minutes anymore; use the results you have." Read as no run over 10 min and no paired tests unless asked.
  - Why: that morning's multi-hour measurements kept being overtaken by design changes; the β and dose tests were stopped part-way.
- **LIFTED at 13:14:** "keep building alone, evals are allowed again, fixes one at a time." The cap held only from 12:45 to 13:14.

### 1.16 advantage-head.md
b1. Modified 2026-11-11T20:10Z. Type: design and deploy record.
- **Stage 1** (e5's spec, b1's build; adv-head 428baab / 59c83ac / 1876d0f):
  - `model_v2.AdvantageHead` predicts the standardized advantage of (type, entry) = adv_type + adv_entry + adv_named × [the entry is NK2's choice].
  - Inputs detached, value group, `adv_weight` 0 = bit for bit, the policy reads none of it.
  - Why: at follow 0.9 a followed row passes ~P_own/9 of its gradient to our own policy, and a regression learns from every row.
- **Deploy 58** ran 15:31 on 11-11.
- **Stage 2a** (adv-gain 5a04565): `adv_gain` adds gain × clip(entry + named, ±3) to the logits. Recorded vs recomputed logp within 6e-5; forward +1–2%. **Deployed with deploy 59** (adv_gain=1).
- Stage 2b (per-state follow chance) not started.
- e5's update_check is unseeded, so compare only seeded runs.

### 1.17 deploy-safety.md
b1 per frontmatter (modified 2026-11-03T13:30Z). The text adds 2026-11-05, 11-10 (written from e5's deploys 48/49, author unclear) and 11-11. Type: **incident lessons and rules**.
- **2026-11-03 07:25 crash:** deploy e79960a made `decode` return a pinned tensor; `loss_noise` (runs every noise_every updates) called torch.from_numpy on it. Down ~2 min, ~15 games lost.
  - Standing user rule: **"make sure that we can't die and lose data, if we do you failed."**
  - Lessons: grep every consumer and every rare path; diagnostics catch Exception and continue; watch a full noise_every cycle.
- **2026-11-05:** a CUDA benchmark beside joint7 (WDDM time-slicing) starved the learner (48 games waiting; 31.9 of 32.6 GB). Rule: **no GPU work beside the run**.
- **2026-11-05 ~20:50:** daac154 indexed with a Python list, which breaks CUDA graph capture. Eager updates took 880–1106 s against 224–296.
  - Rules: no list, numpy or per-call tensors in the forward; run capture_check on the PC; grep the log for "not recorded" / "eager".
- **2026-11-10:** deploys 48 and 49 refused themselves on wrong gates (a comment line counted; a line that is never printed).
  - Rules: copy gate greps from real PC output; count non-comment lines; previews are not gates; `ninja -n` never says "no work".
- **2026-11-11, deploy 58:** an assert expected the learned follow chance to be exactly 0.9; it had learned to 0.90086. Rule: **dry-run gates against a copy of the live checkpoint_latest.pt**.
- **2026-11-11, deploy 60:** a gradient hook calling float() synced inside CUDA capture. Rule: **nothing inside forward/backward may read the device**; grep for float() / item() / tolist().

### 1.18 deploy-60.md
b1. Modified 2026-11-12T04:03Z. Type: deploy log and lessons (deploys 60–65).
- **Deploy 60**, live 21:27 on 11-11 (3719556 / vcmi 826909100, resumed v3205); the first try at 19:32 restored itself.
  - Line: follow 0, choice hidden everywhere, eval 48 own, town_guard 0, adv_by_kind false, adv gain 1, price 2e-6, `credit_segments=true` (k = max(0, 1 − 1/F), EMA 0.05 from 0; credit sd ~0.55 of the 7-day credit's).
- Agreement with NK2 in hidden games is falling (0.205 → 0.15).
- **Meeting cycles doubled v3176 → v3205 through our own weights** (the cut MOVE's probability 0.25 → 0.35). Not the advantage head, not the prior gains.
- **Deploy 61** (23:16, 8519668): price 1e-5 (the user), MOVE/ARMY diagnostics, loops_per_player_day.
- **Deploy 62** (23:47, c9a396e): rollback to v3205 with mixed games, **on proxies** (signal fraction ×¼, RECRUIT −21%). The user: "ok, mixed games".
- Mac replays then showed the all-own line had been improving: v3205 0.672 → v3226 0.820 → v3229 0.861; pooled +0.169 ± 0.062. *(v3229's 0.861 was later shown to be luck; see 1.23.)* PC v3205 0.629 against v3216 0.768.
- **Deploy 63** (00:19): all own from v3229 plus nk-predict.
- **Lessons:**
  - Judge by measured play, never by optimizer proxies alone. Signal fractions after a checkpoint change carry the old Adam history.
  - A rollback makes version numbers repeat: quarantine the spool and eval checkpoints.
- **Split check:** the segments raise trunk signal on all-own batches (0.041 against 0.019), so they stay on.
- **No-advisor pair:** −0.132 ± 0.078 (out of distribution).
- **nk-predict:** output layers zero-init, weight 0.1 (at 1 it tripled the value group's gradient norm); time gate ≤ 1.15.
- **Deploy 64** failed its time gate (1.38): the gate measured the first update's one-off graph recording, and the head's StepGraph was a local re-recorded every update (e5's find).
- **Deploy 65** (a332c25, live 01:36): the graph is kept on the model and the gate runs on the second update (1.027). The head's AUC reached 0.70–0.75 by its 4th update. Eval v3237 own 0.832.
- **Stage 2 diagnostics:**
  - "toward" is a candidate MARKER (0.906 of legal MOVE entries with a destination are the entry's own cell), not routing.
  - w_stop (schema 20, 705c48b) premise failed: it agrees with the planner 0.09 of the time.
  - Predicted gate: AUC 0.96 but precision 0.75 at recall 0.18; MOVE R ≈ 0. No games.
- **Lessons:** never edit a worktree that a running eval imports from; monitor pipelines must not end in `cut` (it buffers); report a deploy's outcome to e5 within 5 minutes.

### 1.19 pgrep-self-match.md
b1. Modified 2026-11-12T13:05Z (07:05 CST). Type: incident and rule.
- The v3255/v3229 replay pair waited from 04:30 to 07:04 and **never ran**: `while pgrep -f "gate/pairG.sh"` matched the launching zsh, whose argv holds the whole heredoc. The load sampler had the same bug, so b1 reported "load during the pair" while no pair was running.
- **Rule:** wait on a pid (`kill -0`) or on a file the job writes; check within one interval that the job started.

### 1.20 eval-slide-2026-11-11.md
No session field. File mtime 2026-11-12 07:59. Author by content: e5, with the final "2026-11-12 morning" section possibly b1 (unclear). Type: incident (false alarm) and measurement.
- **Observed:** run-away v2862 0.565 → v2912 0.493; schema 17 v2934 0.579 → v3003 0.487 (−0.092 ± 0.057). Treated as real.
- **e5's leading explanation:** the teacher (0.60 of the gradient) trains prior gains away. **WRONG in form.**
- **Deploy 54** (05:08, 80793cf): towns' night, battle_length_scale=30, GATHER ramp code off with readings.
- **RESOLVED 06:10: not in the weights.** Mac replay under one code, 3 replicates: v2934 0.506, v2956 0.566, v2979 0.528, v3003 0.534. **"One PC eval is ±0.11; never read a trend from single PC evals."**
  - Deep-end P(penalized) 0.040 → 0.084 (a step between v2862 and v2912).
- **Deploy 55** (06:39:35, 90d153c): GATHER ramp 0.1 over 200 updates from v3032.
  - Stop rules: the guard; 10-update mean of eliminated > 0.04; value_target_std > 0.048; an eval down 0.1 with 14+ eliminated confirmed on the Mac.
  - Readings: PC v3045 0.497, v3069 0.466, v3090 0.498. Mac v3025 → v3045 −0.014 ± 0.033.
  - The deep-end creep began before the ramp.
- **2026-11-12 morning:**
  - v3229 scored 0.861 at 00:00 and 0.697 at 07:58 (−0.164 ± 0.080, same weights). v3255 three runs: 0.632, 0.737, 0.805.
  - **One 48-game arm moves ±0.08 between runs.** A Mac at load 11 reads ~0.12 low.
  - The PC's "fall" (v3254 0.836 → v3289 0.629) is not confirmed on the Mac (v3229 0.697, v3289 0.702).
  - Judge only from arms run alone, ≥2 replicates each.

### 1.21 feature-representation-rule.md
e5. Modified 2026-11-11T16:11Z; the text adds 2026-11-12. Type: **developer rule**.
- **User ~10:10, rejecting "Task list with Hero, target tile, days, priority tier, priority":** "this is not a 'plan' ... you did not say how we can make it something that the network can intuitively learn, our heroes have embeddings, tiles have embeddings AND expected value change upon reaching there, (maybe we should also have some relative value change compared to some other options?). Days are an int, how is that going to be represented in the network? a raw int is not good, a different 'day embeddings' is also not good … the lack of connection between the tile/hero embeddings to these raw features are somewhat worrisome."
- **How e5 applies it:** each feature states its pair, its unit, the combined form, how the first step carries the destination, relative forms, and an offline readability gate. The current model breaks this: window days are a one-hot; MOVE's prior has no within-day discount.
- **2026-11-12, user on three proposed inputs:**
  - A feature must be "easily connected to the relevant actions and not to other things".
  - Where a quantity is unknown (a rival's speed), use a fuzzy form in a known one (distance).
  - "we do not need a feature for the complex decisions, just to make the simple ones simpler without messing things up."
  - An odd feature only "if it can't explode or break anything".

### 1.22 overhaul-2026-11-11.md
e5. Modified 2026-11-11T17:58Z (stale); stamps run to 2026-11-12 08:38. Type: the master log of 11-11 and the night to 11-12. Chronological:

**~11:00–13:30: build and deploy 57.**
- **Go-ahead (~11:00):** "ok. Make the feature changes and get rid of the teacher." At ~11:30 b1 is at e5's disposal.
- **Code:** dad4fcd views.py; d3ee26e `nk2_teacher` (False zeroes the teacher); b1's schema 19 (6be714c, C++ ab18404, vcmi a58753d66); 50a8e40 model (50 channels a MOVE tile / TRAVEL area, plan vector, FastLinear view_move/view_travel ×10 step); 43ce74a `value_jump_from`. Inert proof: bit for bit; Mac forward +~10%.
- **11:25:** the guard HELD the campaign at v3106 (loss 4.0× calm; army_on_heroes 0.29–0.43). Mac v3025 vs v3090: −0.038 ± 0.042, so the ramp neither hurt nor helped.
- **~12:00, Adam-state finding (v3090):** elementwise |m|/(√v+eps).
  - Trunk median 0.20, mean 0.28, against 0.23 for pure noise: **policy weights walk like noise.**
  - Heads: move_out 0.046, travel_out 0.11. Value group median 0.50: the critic has direction.
  - move_mix_direct still at its initial −3 after 3,090 updates.
  - Consequence: a zero-start weight is unused for weeks at lr 1e-5, so starts must come from an offline fit.
- **12:10–12:25:** the candidate prior FAILED (β 0/2/4: 0.562 / 0.579 / 0.549; advised entry taken 29–36% against 92% in the follow test). e5 ran a 25-minute paired test before the 2-minute check.
  - Replaced by NK2's OWN CHOICE as inputs (`g_nk_action/index/cell`, b375bea / ee42a90) plus a label prior. Offline dose: P(ours = its choice) 0.378 at 0 → 0.900 at 8.
- **12:25–12:50:** the label prior was weak too (8: +0.052 ± 0.064). Root cause: the follow test had run with `ALPHA_MAP_GATHER_ADVICE=1` and the arms without; logit priors cannot outvote our negative trade priors.
- **User 12:50:** "you invested too much time in a single experiment ... no experiments that take over 10 minutes anymore".
- **Deploy 57** (13:26, second try; the first self-refused because update_check kept an autograd tensor alive before capture). e005793:
  - **Follow mixture:** P = ρ × (NK2's choice) + (1−ρ) × (ours), exact in logp; learnable `nk_follow_shift`.
  - Fallback advice at a miss (FALLBACK 64).
  - Line: `nk2_inputs=true nk2_follow=0.9 nk2_teacher=false value_jump_from=3107`. GATHER weight/ramp and valves OUT. ρ 0.9 had no paired test of its own.

**13:40–17:30: costs of the mixture, deploys 58–59, design discussion.**
- **13:40, b1:** at a followed row the gradient into our own logits is ≈ P_own/9; schema-19 layers learn 10–10⁴× slower (travel_toward 1e-10 against 1e-6). The answer is the advantage head.
  - Signal check: a day's share of followed choices → 7-day share change, slope +0.0124 ± 0.0023.
- **13:58, mixture paired eval (v3090):** 0.792 vs 0.507 (**+0.285 ± 0.112**). The inputs alone add nothing visible.
- **14:51, first eval of deploy 57 (v3120):** 0.753 ± 0.128, **eliminated 17 of 48** (before 8–13). **15:31 deploy 58** (advantage head).
- **User ~16:00:** following at 0.9 "can work, but it is also super risky in the sense that we will only learn to fix NK2 and not really learn good moves in general."
  - Approved: half the games with the choice hidden; split eval 32 own / 16 follow; score_own is the number.
  - The head's label stays the windowed credit (per-pickup value: "it will learn to be too greedy").
  - Report 0.75 as "with NK2 driving".
- **~16:20:** "we care about what we can learn and how fast, not always the total score"; "lets just build more stuff". Judge by the SLOPE of score_own.
- v3138 0.765, eliminated 17 again. **16:30:** "do 1+2".
- **17:03 deploy 59** (2943d45): `nk2_choice_games=0.5`, split eval, `adv_gain=1`, `schedule_action_price=2e-6` ("we want to turn it on").
- **User 17:00:**
  - No model change for aggression monitoring.
  - e5's teleport story was WRONG (the bucket was town re-visits / NK2's Defend composition).
  - "what a human can do the model must be able to do".
  - Forecast-out externalization discussed, not approved.
- **b1's elimination count (v3090):** following attacks more (14.4 vs 5.9 battles a game) but loses 2.0% vs 14.4%. Eliminations come from being attacked (10 of 13 eliminated games lost their strongest army that way).
- **17:15–17:30, user decisions:**
  - Bundle 60 = teleports, boats, forecast-out ("without breaking the model"), follow cap ("we can limit it to 90%"), artifacts, town guard, battle records in PC evals.
  - **Army arranging POSTPONED.** Battle model later. Critic "a big one".
- **Defence:** towns lost 23 (follow 0) vs 100 (follow 0.9). 71 of 80 weak losses happened because our hero WALKED OUT; in 50 of them the move was OUR OWN 10% → town_guard prior.
- Revisit-skip fix (2b2cf74): unmatched 2.5% → 0.8%.
- First mixed updates: hidden rows get a third of the advantage spread → `adv_by_kind` (e5's decision).

**18:00–19:00: the fresh-run question and bug fixes.**
- **Town guard pair:** +0.010 ± 0.116; towns lost 83 → 64. e5: `town_guard=0.5`.
- **User ~18:00:** wants a FRESH RUN with no NK2 playing as us, because V's loss is back at start-of-training levels.
- **18:22 (verbatim parts):**
  - "we'll do a fresh run without NK2 making choices".
  - Failed actions "need to be fixed now before any fresh run, bugs are fixed first before any improvements".
  - "Every experiment should have at least a chance of yielding actionable insights ... don't do experiments if there is no action".
  - "no extra losses without asking me".
  - "answer for design decisions that are important (like having a schedule or not)".
- **Horizon measurement:** surprise sd 0.0496 / 0.0742 / 0.0868 at 2 / 5 / 7 days; the 7-day credit = 2-day + 2.4× variance.
  - **e5's decision: no fixed schedule; credit segments** s(0–2) + k5·s(2–5) + k7·s(5–7), with k = max(0, 1 − 1/F) by empirical Bayes.
- **18:27, untrained model on priors alone: 0.123 ± 0.029, eliminated 33/48.** RECRUIT 0.028 vs 0.146 trained: a broken start.
  - ARTIFACT loop root cause: EQUIP into an occupied slot; fingerprint by type. Fixed.
- **18:33, user:** the LR is e5's to set ("the 1e-5 was made when we were noisier ... do as you will to train efficiently"); "maybe we can continue this training as of now... I do not know if NK2 at 90% is helping us actually learn or not."
- **e5's decisions:**
  1. CONTINUE joint7; the fresh run is PARKED (0.12 vs 0.51).
  2. NK2 makes NO choices from deploy 60 on (followed rows give ~P_own/9 of the gradient).
  3. LR stays 1e-5 (trunk at noise level).
  4. Deploy 60 after the first split eval.
- **18:35:** bugs done (2c56c92): a 12-game smoke with 0 failed actions; loop guard.
- **First split eval v3169:** own 0.602 ± 0.071; follow 0.688. Deploy 60's pair v3176: +0.007 ± 0.058.

**19:00–23:30: deploy 60, the cycling, deploy 61, the rollback decision.**
- **Deploy 60:** the first try at 19:32 failed (float() in a hook). **b1 reported it 90 minutes late**; e5 learned of it only when the user asked "is anything training now?" Up at 21:27.
  - Rule: outcomes reach e5 within 5 min. Lesson: check the PC yourself after 30 min.
- The own-arm series under deploy 59 (0.602 → 0.739 → 0.802) is mostly one lucky eval (Mac v3205 0.672).
- **Cycling:** cuts per 48 games TAKE_BEST 230 → 455, MOVE 175 → 389 (313 onto our own hero). Plain drift since the teacher went off.
  - e5's proposal: a fixed negative credit on cut decisions. **REJECTED by the user:** "maybe 1e-5 [price], this is still more than we had before by a lot"; "worthless stuff should/can be slowly removed by the small action cost and decisions that should never be made ever might not need to be a part of the model"; "if you find bugs fix them as you find it, if some future decisions are clear and should obviously be taken now, do that".
- **Deploy 61** (23:16). Eval v3216 0.768 ± 0.071.
- **User ~23:10 on live NK2 inputs:** "this seems ugly ... I do not want a monstrosity to be served live, the more elegant the better". They suggested an auxiliary loss → the nk-predict head, **APPROVED**: "do it, do not wait ... we can have it if it helps". Generalized targets: "our many ways to count values vs a 1 way ... a boolean that can/should be probability".
- **b1's lookups:** all-own games → signal fraction trunk 0.066 → 0.018, value 0.083 → −0.007; RECRUIT −21%.
  - **e5's decision: deploy 62, rollback to v3205 with mixed games.** The user: "ok, we can continue with Mixed games".
- **23:27, the night mandate:** "Make the changes and do it, I want to have it all in the model that trains for the night ... remember you can fix bugs as you find them. Go."

**23:55–01:40: the rollback was a mistake; deploys 63–65.**
- **v3226 − v3205 = +0.148 ± 0.067**, so deploy 62 was "a mistake made on proxies". PC v3205 0.629 vs v3216 0.768.
- Deploy 62's signal "collapse" was partly the fading of Adam's history.
- **The day's headline (later CORRECTED):** v3090 0.507 → v3176 0.60 → v3205 0.672 → v3226 0.820 → v3229 0.861.
- Split check: segments ON.
- Deploy 63 in fallback (time gate 1.41). Deploy 64 failed (1.379; StepGraph bug). **Deploy 65 live 01:36.**
- Meeting issue in training: exchange decisions a player-day 2.0 → 5.2–7.4, flat since v3220. The credit is in effect the 0–2 day segment.

**02:00–05:30: stage 2 measurements.**
- Head on PC: AUC 0.70 → 0.86–0.94; plan ce −21% to −50% vs uniform. Cost: none measurable. nkp_loss is logged as a sum (logging slip).
- Stage 2 KL check (v3229, 300 decisions):
  - Late terms R ≈ 0.02; {pairs, toward, late} R 0.75 / 0.81 / 0.85; threats 0.00.
  - C++ facts: MOVE is legal only for cells reached today; TRAVEL walks toward a multi-day area.
- **e5's routing reading was WRONG** (b1's split): 91% of MOVE destinations are the entry's own cell, so the planner gives a candidate marker.
- PC v3254 0.836. w_stop premise failed (0.09). Predicted gate fails (R MOVE ≈ 0).
- **Measurement A** (v3255, ×2, under load 14): live 0.685, blind 0.653, move_live 0.730. gate_pred 0.690. All within noise.
  - Blind − live, pooled with v3226: ~−0.08 ± 0.05.
  - b1's diff: identical except LOAD.
- k7 climbed 0.07 → 0.29 then fell back to 0.10–0.14. PC v3272 0.787.

**07:05–08:38: the "decline", the correction, and the user's direction.**
- **07:05:** PC v3289 **0.629, eliminated 15**.
  - Training side: heroes a player 2.77 → 3.42, recruits 0.728 → 0.589 a player-day. e5's unmeasured guess: the 2-day credit pays hiring and not troops.
  - The pgrep bug lost 2h20m.
- Mac alone: v3255 0.805, v3289 0.702.
- Credit evidence at v3289: 0–2 days HIRE +0.021 (t +5.1), BUILD −0.009 (t −3.0), END −0.010. Deploy 66 (rollback to v3254) PREPARED.
- **07:58 CORRECTION:** v3229 replayed scores 0.697 (vs 0.861). NO fall on the Mac; the level since ~v3226 is ~0.70–0.80. **NO ROLLBACK.**
- **~07:55, user:** what is the architecture missing? (Predictable places of interest = a design flaw; several planners.)
  - e5's gaps, ranked: (1) concentrating force; (2) defence (10–15 towns lost empty per 48 games); (3) relations between our heroes; (4) what an action opens; (5) multi-day goals; (6) credit horizon. Several planners: yes, by question.
- **08:05, user direction:**
  - Inputs, as in 1.21.
  - Training: "they think the MIXED games … improved us a lot, that the later all-own gain may be optimizer memory, that we may want to keep the mixed games and that the rate may be too slow; 'very unsure about either. Please discover what is right.'"
  - e5 read progress.csv: the RATE binds (lr_scale at cap 1, KL 0.003–0.018 against target 0.03); the optimizer-memory kick is real and ~6 updates long. e5's reversal to all-own rested on single arms.
  - Plan: Mac replicates → deploy 67 (mixed) → LR as its own step (2e-5 if mostly signal; user limit 3e-5) → check a weekday confound.
  - F3 town-hold and F1 others' time specified; F2 held.
- **08:38:** "ok, continue doing everything you think is needed."
  - Mac arms: v3301 0.634, v3254 0.772.
  - The calendar idea is WRONG (k7 rode on ARMY's thin evidence). Bug: plain var(means)/mean(var) shrinkage; fix: precision-weighted F.
  - Deploy 67 dry run PASSED. F3 on town-hold 9637796; F1 on f1-others.

### 1.23 morning-2026-11-12.md
b1. Modified 2026-11-12T14:42Z (08:42 CST). Type: state snapshot.
- **Mac replays** (one arm each, alone): v3229 0.697 (9 eliminated), v3255 0.805 (10), v3289 0.702 (13), v3301 0.634 (14), v3254 0.772 (14). "Last night's 0.861 for v3229 was luck."
- **Running at 08:40:** v3301b, v3254b, v3205 ×2 (v3205 = runtime/saved/campaign_joint7_pre_d60.pt, because checkpoint_03205.pt was rewritten at deploy 62).
  - e5's rollback rule, set in advance: only if v3301 is below v3254 by more than 2 se pooled.
- **1b:** it is not the calendar. F7 came from ARMY's thin evidence (56–92 faded game-players against ~600); RECRUIT has 86% of its decisions on a week's first day.
  - Precision-weighted F (8623b77 on d67-mixed). Under the new F, v3254's F7 is 2.33, mostly BUILD (z +2.9). "ARMY drove it" was an artifact; k7's target only 0.67 → 0.57.
- **Prepared:** deploy 66 (d66/, dry run cancelled); deploy 67 (3f4a3a4, dry run passed; refuses while a replay plays).
- **Branches:** F3 9637796; F1 9b7be05 (schema 20 w_other_turns; **conflicts with w-stop's schema 20**).
- **Rules:** nothing runs on the Mac beside an arm except one niced process; wait on a pid.

---

## 2. Standing rules set by the developer

| # | Rule (developer's words where available) | Date | Cause / context | Status |
|---|---|---|---|---|
| R1 | Never hand over a "your call" without a recommendation; "do not waste my time" | 2026-10-28 | a "your call" on a minor fight-prior detail | in force; sharpened by R2 |
| R2 | "Do not make a 'your decision' type replies anymore unless there is a non trivial trade off…"; "STOP saying 'your call'… some of these options are stupid"; give one worked-out solution a weaker AI can code; root cause first | 2026-11-09 (3rd time) | GATHER "your decision" copied from another AI's audit; a list of symptom fixes when the cause was an NK2 bug | in force |
| R3 | During a run: fix clear mistakes yourself, leave tuning; "keep training and watch" | 2026-11-01 | joint run supervision | in force |
| R4 | "'Found, not fixed' => then fix"; "errors and even clear bugs should be fixed immediately"; "bugs are fixed first before any improvements"; "if you find bugs fix them as you find it" | 11-05; 11-10 00:00; 11-11 18:22; 11-11 ~22:40 | reviews listing found-not-fixed items; failed actions before a fresh run | in force |
| R5 | "make sure that we can't die and lose data, if we do you failed." | standing (cited 2026-11-03) | the loss_noise crash | in force |
| R6 | Never `wsl --shutdown` without the go-ahead (it kills Zonkey) | 2026-10-30 | shared PC | in force |
| R7 | After a label-change deploy, no rollback before ≥3 campaign evals ("see at least 3 updates") | 2026-11-10 ~00:00 | e5's pre-announced rollback threshold after deploy 42 | in force; AI practice later moved to paired Mac tests |
| R8 | An explanation counts only if it would have predicted the failure | 2026-11-10 08:00; restated that evening | the battle value-loss question; the GATHER collapse | in force |
| R9 | A campaign_value_loss explosion = stop at once; a rollback is not enough, find why; don't dig past the cause. Refined: a spike that returns is OK; staying up or long-flat is bad; pushing it down is necessary, not sufficient | 2026-11-10 evening; 2026-11-11 00:10 | GATHER collapse #2 | in force (value-loss guard implements it) |
| R10 | No value on a decision's own step; credit only from day+2/5/7; step_credit call commented out | 2026-11-09 night | e5/b1 step-credit build | in force |
| R11 | Action-count price allowed, 2e-6, never above 2e-5 (later set 2e-6 in deploy 59, 1e-5 in deploy 61) | 2026-11-09 22:00; 11-11 17:00 / ~22:40 | user's own idea, overruling the AI's "no" | in force at 1e-5 |
| R12 | No fixed negative credit on cut/no-effect decisions; worthless actions removed slowly by the action cost | 2026-11-11 ~22:40 | the meeting/TAKE_BEST cycling | in force |
| R13 | Prior-anchored features: known NK2 values as a fixed prior plus decayed corrections, logits ≤ ~4; other features only through zero-init trainable weights; nothing raw and big ("it will explode") | 2026-10-28 | campaign object values design | in force; night_gain freeze is a reported exception |
| R14 | Stakes as share of the empire, not gold ("gold is unbound"); the network outvoting a prior is "a good thing" | 2026-11-10 night | night-term design | share-of-Φ variant measured worse; parked for big maps |
| R15 | night_gain may be learned as an auxiliary loss but "should not be a part of the V" | 2026-11-11 00:10 | night prior | in force |
| R16 | Feature plans must say how each number is represented and connected to the hero/tile embeddings; no raw ints, no per-day embeddings; multiple bounded value views; nothing that can explode (pixies vs hero); small non-zero sign-safe starts | 2026-11-11 10:05–10:35 | plan v1 rejected | in force |
| R17 | A feature connects only to its relevant actions; fuzzy in known quantities, never falsely exact; features make simple decisions simpler, none for complex ones; odd features only if they cannot explode or break anything | 2026-11-12 08:05 | e5's three proposed inputs | in force |
| R18 | No experiment over 10 minutes; use the results you have | 2026-11-11 12:45–12:50 | multi-hour paired tests overtaken by design changes | **lifted 13:14** ("evals are allowed again, fixes one at a time") |
| R19 | "Every experiment should have at least a chance of yielding actionable insights" | 2026-11-11 18:22 | same series | in force |
| R20 | "no extra losses without asking me" (priors are allowed; the advantage-head and nk-predict losses were approved) | 2026-11-11 18:22 | overhaul design | in force |
| R21 | No training against eval AIs (self-play only); cheating = anything that does not scale to larger maps, or exploits vs NK2 | older; restated 2026-11-11 09:05 | teacher removal | in force; follow mixture used NK2 as a policy component, later turned off |
| R22 | Report scores obtained under NK2 following as "with NK2 driving"; judge by the slope of score_own; "we care about what we can learn and how fast" | 2026-11-11 16:00–16:20 | deploy 57's 0.75 | in force |
| R23 | Advantage-head label stays the windowed credit (no per-pickup values: "it will learn to be too greedy") | 2026-11-11 16:00 | head design | in force |
| R24 | "what a human can do the model must be able to do" (NK2's remote boat build is not copied, because a human cannot) | 2026-11-11 17:00 | teleports/boats | in force |
| R25 | No model change for aggression monitoring; army arranging postponed (MERGE_ALL + SPLIT_SINGLES suffice); battle model later | 2026-11-11 17:00–17:30 | bundle-60 scope | in force |
| R26 | Live serving must be elegant: "I do not want a monstrosity to be served live" | 2026-11-11 ~23:10 | NK2 planner inputs at inference | drives nk-predict |
| R27 | Learning rate is the AI's to set ("the 1e-5 was made when we were noisier…"); limit 3e-5 cited 2026-11-12 | 2026-11-11 18:33 | fresh-run question | open: LR still 1e-5 |
| R28 | Deploy ownership: "the other AI will deploy for you" (e5 deploys; b1 builds) | 2026-11-09 22:30 | deploy 42 | in practice b1 ran deploys 58–65 on e5's go |

**AI-adopted procedural rules (not set by the developer; agreed between sessions):**
- One learning change per deploy.
- A paired test on the 48 eval setups BEFORE the PC; PC evals only watch for harm (e5 and b1, 2026-11-10 22:15).
- The next change only after one clean eval.
- Only compare side-by-side arms with ≥2–3 replicates, run alone on the Mac.
- Deploy gates are copied from real PC output and dry-run on a copy of the live checkpoint.
- No host reads inside captured graphs. No GPU work beside the run.
- Deploy outcomes reported within 5 minutes; check the PC after 30 minutes yourself.
- Wait on pids, not `pgrep -f`.
- Never edit a worktree a running eval imports from. Never `git stash` in the main worktree.
- Rollbacks quarantine the spool and eval checkpoints.
- A rules draft (docs/session_rules_draft.md, 21 rules, untracked) awaits the developer's review and is **not in force**.

---

## 3. Incidents and failures (consolidated)

| Incident (date) | Cause | Detection | Impact | Fix | Lesson |
|---|---|---|---|---|---|
| Teacher meeting loops (found 2026-11-09) | Stateless advice reruns makeTurn; NK2 phantom deliverArmyToHero bug (receiver counted twice); exchange receiver vs plan mismatch | e5 audit on 8 games; NK2's "Performing task" logs | 71% of teacher MOVE labels onto own hero/town; 26% of movement on meetings; 314 meetings/game | M1–M5 (deploy 42); meetings 315 → 60/game; evals unchanged | Imitating a heuristic AI imports its bugs; read the teacher's own logs first |
| Credit shared per player-day (2026-11-09) | 7-day MC credit with no step term | Variance decomposition | 96.5–98% of advantage variance between player-days | Step credit built, then set aside by the user; segments later (deploy 60) | Outcome credit at day scale gives each decision ~no own signal |
| Idle movement 2–3× after meeting fixes (11-10) | Meeting mass moved to END; credit cannot tell a visit from END | b1 endtown analysis | idle 0.095 → 0.269 | Metric (414c622); price held; GATHER tried | Removing a label moves its probability to the cheapest alternative |
| Infinite stalemate battles (pre-existing; found 11-10 00:30) | No side can flee and VCMI only retreats a side that can | Stall kill + 3.8 GB client | 298 kills in 9.5 days | Weaker side loses (deploy 43) | Engine edge cases need explicit termination |
| NK2 endTurn race (eval stalls) | Shared haveTurn flag; old turn thread ends the new turn on fast 2-player maps | Kept run dirs (2 AIGateway threads in futex) | 13 eval games killed in 151 evals | Started-turn count by colour (deploy 45) | Bug-fixing the opponent is allowed when labelled |
| Inference server model wait (11-10) | Version probe holds the only worker 5 s per request | New diagnostic fired at start | Possible starvation | Parked requests 8501601 (deploy 45) | Instrument before guessing |
| Battle inference stalls (25 real) | Unknown: client waits in recv with no receive timeout; server had no line | Linux thread dumps | ~1 kill/h class | Socket sampler (0c8ebfd); unresolved in this half | — |
| Watchers never fired (11-10 ~00:50) | `grep -oE "[0-9]{5}"` matched "20261031" in an absolute path | Noticed later | Hours of unwatched evals | cd/basename | Test watcher regexes on real paths |
| Scratchpad wiped (11-10 01:00) | /tmp cleanup | Missing type_drift.py | Rewrite | — | Keep tools out of auto-cleaned dirs |
| Battle accumulate 24→2 misjudged (Nov 5 → 11-10) | Signal fell below the noise median, called "expected"; later the noise median itself was mis-computed for 4-pass updates | e5 re-analysis when the user asked about value loss | 5 days of a noisier recipe; no eval change either way | Deploy 47 (back to 24), honest SignalShare, length weight (deploy 54) | Compute diagnostics' pure-noise values under the recipe; signal = between different battles |
| Battle model flat at +0.09 (whole period) | ~0.5% signal per update; gradient rests on 4–14 games; steady direction unhelpful or non-stationary | 53+ flat evals | No battle progress | Length weight 30; aux phase prepared (deploy not recorded here) | More averaging does not fix a non-stationary or wrong direction |
| GATHER collapse #2 (deploy 46, 11-10 08:48) | GATHER label put the army on heroes who were already dying in the field (danger ignored; own town priced as open ground) | Eval drop 0.49 → 0.27–0.40; seen 9.5 h late (Mac off LAN) | Eliminations 6 → 28/48; value loss ×4–10 | Rollback deploy 48 (v2728 → v2839); value-loss guard; run-away prior; towns' night | Check where moved value goes under today's policy; value-loss spikes are an early alarm |
| Self-refused deploys 48, 49 (11-10) | Gate grep matched a comment; gate expected a never-printed line | Deploy scripts | Two wasted drains | Gate rules | Gates only from real output |
| Unpaired Mac A/Bs (11-10) | `play.make_map` regenerates maps (RMG non-deterministic) | b1's catch | Run-away v1 judgement unreliable | Pinned maps (paired_eval) | Pin maps |
| night_gain eroded by teacher (11-10) | Teacher CE pushes a prior's gain; reward gives no sign at that horizon | push_* diagnostics (deploy 51) | Prior weakening | Frozen (deploy 52); reported as a rule exception | Teacher gradients act on hand priors too |
| Eval-slide false alarm (11-11 05:00) | Single PC evals (±0.11) read as a trend | Mac replays, 3 replicates | GATHER ramp held ~1 h; wrong theory sent to b1 | None needed | Trend only from replays |
| Stale-V valves (deployed 07:23, premise failed 09:35) | Hypothesis "V got worse" was target spread from eliminations | value_unexplained readings | Valves closed on ordinary noise | Removed at deploy 57 | Check the premise metric before building |
| Guard hold at v3106 (11-11 11:25) | GATHER ramp raised army_on_heroes 0.29–0.43; loss 4× calm | Value-loss guard | Campaign paused until deploy 57 | Restart with the overhaul | Guard worked as designed |
| Candidate / label priors failed (11-11 12:10–12:50) | Top candidate a loose proxy; env mismatch (GATHER advice off in arms); logit prior cannot beat negative trade priors | Paired tests; env diff | ~1 h of tests | NK2's own choice + follow mixture | Run the 2-minute mechanism check before the 25-minute test |
| Deploy 57 first try (before 13:26, 11-11) | update_check kept an autograd tensor alive before capture | Gate | Retry | no_grad | Known 2026-10-30 error |
| Deploy 58 first try (before 15:31, 11-11) | Assert expected learned follow chance exactly 0.9 (was 0.90086) | Gate | Retry | Bounds asserts; dry-run on the live checkpoint | Learned values drift from config |
| Eliminations 17/48 under follow 0.9 (11-11) | Our own 10% deviations walk heroes out of threatened towns; NK2 parks them there | towns_lost analysis (71 of 80 walk-outs, 50 by our policy) | Towns lost 100 vs 23 | town_guard prior; revisit skip; day hold | A mixture policy can be worse than either part in specific states |
| Mixture starves own-policy gradient (11-11 13:40) | At ρ 0.9 a followed row passes ≈ P_own/9 | b1 gradient measurement | Own policy learns 10–10⁴× slower on followed rows | Advantage head; NK2 choices off (deploy 60) | Exact policy-gradient through a mixture still kills learning of the minor component |
| Untrained model broken (11-11 18:27) | Type priors make it shuffle troops, not recruit | Priors-only eval 0.123 | Fresh run parked | — | Priors calibration matters for a clean start |
| ARTIFACT loop / false failed actions (11-11) | EQUIP into an occupied slot; fingerprint by type; GARRISON_SWAP and exchange's other army missing from the fingerprint | 12-game smoke | Failed-action noise; 106-step loops | 2c56c92, e78372e, 3719556, loop guard | — |
| Deploy 60 first try failed, reported 90 min late (19:32) | float() inside a backward hook during CUDA capture | User asked "is anything training now?" | 90 min unnoticed (ran on deploy 59 safely) | Hook removed; 5-minute reporting rule | Verify a deploy yourself |
| Meeting/TAKE_BEST cycling (after teacher off) | Policy drift; cut cycles cost only the tiny action price | Replay loop counts | Loops 0.16 → 0.6–1.0 a player-day; exchanges 2 → 5–7 a player-day (plateau) | Price 1e-5 (user); fixed negative credit rejected | Without a teacher, nothing penalizes futile actions |
| Deploy 62 rollback on proxies (23:47) | Signal fraction ×¼ and RECRUIT −21% after all-own games, read as a collapse; partly the fading of Adam's history | Replays v3226 (+0.148) and PC v3205 | ~30 min of discarded mixed updates | Deploy 63 back to all-own | Judge by play; but see the next row |
| Deploy 63/64 time gates (00:29, 01:08) | Gate on a tiny batch measured one-off recording; head StepGraph re-recorded each update (real bug) | Gate ratios 1.41 / 1.38 | Two failed attempts | Graph kept on the model; gate on the 2nd update (deploy 65) | Gate the steady state |
| "0.861" headline (00:26 → 07:58) | Single Mac arm read as a level; one arm is ±0.08 | Same weights replayed: 0.697 | e5's all-own reversal rested on it; told the user | Corrected; ≥2 replicates | Noise also cuts the other way: the deploy-62 "mistake" was itself judged on lucky arms |
| pgrep self-match (04:30–07:04, 11-12) | `pgrep -f` matched the launcher's own argv | Job never started | 2h20m of replays lost; false load reports | Wait on a pid | Check the job started |
| Measurement A load bias (11-12 03:32) | 9 games at once at load ~14 | b1 settings diff | Arms' levels ~0.1–0.12 low | Run arms alone | Load is a confounder |
| v3289 PC drop (0.629, 15 eliminated; 07:05) | Unknown; Mac shows no fall (v3229 0.697, v3289 0.702) | PC eval | Deploy 66 prepared, not run | Pending replicates | — |
| Calendar-confound idea (08:14 → 08:38) | k7 rise actually rode on ARMY's thin evidence; the shrinkage statistic was unstable | b1 check | — | Precision-weighted F (8623b77) | Plain var(means)/mean(var) is dominated by the noisiest group |
| Memory file overwritten (11-11) | b1 wrote over e5's teacher-question.md | e5 | Restored from context | — | Shared memory needs per-session ownership |
| Misc. AI mistakes recorded | "inference on GPU" (it is CPU); "+1 window a decision"; "evals 3 h apart"; teleport story; "the user's call" passthrough; first-update timing of the collapse; a win counted as an elimination | Self or peer checks | Minor | Corrected in files | Peer review between sessions caught many |

---

## 4. Measured results

### 4.1 Campaign eval vs NK2 on the PC (48 cached paired setups; one eval ≈ ±0.055 se)

| Date / time | Version | Score ± se | Eliminated /48 | Context | Source |
|---|---|---|---|---|---|
| 11-09 before 22:56 | v2504 / v2524 / v2549 | 0.429 / 0.440 / 0.432 | — | before meeting fixes | bug-audit |
| 11-10 00:23–04:27 | v2577 / v2602 / v2627 / v2652 | 0.420 / 0.405 / 0.431 / 0.469 | 14 / 16 / 12 / 10 | deploy 42 meeting fixes | bug-audit, meeting-fixes |
| 11-10 05:55–08:38 | v2679 / v2704 / v2728 | 0.511 / 0.491 / 0.489 | 9 / 10 / 6 | deploy 44 span | bug-audit |
| 11-10 ~10:00–18:00 | v2748, v2761, v2774, v2788, v2801, v2815, v2830 | 0.475, 0.380, 0.314, 0.345, 0.402, 0.271, 0.482 | 15, 21, 23, 22, 19, 28, 19 | deploy 46 GATHER collapse | gather-collapse |
| 11-10 ~19:00 | v2839 (= v2728 restored) | 0.497 ± 0.052 | 11 | deploy 48 | gather-collapse |
| 11-10 20:33 / 21:5x / 23:1x | v2862 / v2886 / v2912 | 0.565 / 0.500 / 0.493 | 7 / 9 / 8 | deploy 50 run-away | gather-collapse, eval-slide |
| 11-11 00:50–04:xx | v2934 / v2956 / v2979 / v3003 | 0.579 / 0.562 / 0.538 / 0.487 | 7 / 7 / 8 / 11 | deploys 52/53 (schema 17) | eval-slide |
| 11-11 ~06:30 | v3025 | 0.533 ± 0.055 | 10 | deploy 54 | eval-slide |
| 11-11 08–10:37 | v3045 / v3069 / v3090 | 0.497 / 0.466 / 0.498 | 12 / 12 / 11 | deploy 55 GATHER ramp | eval-slide |
| 11-11 14:51 / 16:21 | v3120 / v3138 | 0.753 ± 0.128 / 0.765 ± 0.122 | 17 / 17 | deploy 57, follow 0.9 ("with NK2 driving") | overhaul |
| 11-11 ~18:40 | v3169 | own 0.602 ± 0.071 (32); follow 0.688 ± 0.176 (16) | 6 / 5 | deploy 59 split eval | overhaul |
| 11-11 ~21:0x / 21:3x | v3188 / v3202 | own 0.739 ± 0.090 / 0.802 ± 0.090; follow 0.855 at v3188 | 4 / 3 | deploy 59 | overhaul |
| 11-11 ~23:00 | v3216 | 0.768 ± 0.071 (48 own) | 6 | deploy 60 | overhaul |
| 11-12 ~00:20 | v3205 (replayed on PC) | 0.629 ± 0.073 | 11 | deploy 62 start | overhaul |
| 11-12 ~02:10 | v3237 | 0.832 ± 0.081 | 7 | deploy 63/65 | overhaul |
| 11-12 ~03:45 | v3254 | 0.836 ± 0.083 | 8 | deploy 65 | overhaul |
| 11-12 ~05:25 | v3272 | 0.787 ± 0.093 | 11 | deploy 65 | overhaul |
| 11-12 ~07:05 | v3289 | 0.629 ± 0.080 | 15 | deploy 65 (drop not confirmed on Mac) | overhaul |

### 4.2 Mac paired tests and replays (pinned maps; one 48-game arm ≈ ±0.08 run to run)

| Test (date) | Result | Source |
|---|---|---|
| Run-away v2 vs control, v2728, 36 maps (11-10) | 0.461 → 0.595, +0.134 ± 0.041; eliminated 11 → 4 | gather-collapse |
| Run-away, eval setups, pooled 5 estimates | +0.077 ± 0.023 | gather-collapse, night-danger |
| Further windows (v2886) | pooled −0.003 ± 0.024 | move-further-windows |
| Unseen exits (v2886, 3 replicates) | 0.472 → 0.531, +0.059 ± 0.025 | move-further-windows |
| Share-of-Φ prior (v2912) | 0.485 → 0.424, −0.061 ± 0.046 | prior-anchored, night-danger |
| Night power γ 3.18 | −0.001 ± 0.030 | night-danger |
| danger_gain reset to 1.0 (v2886) | 0.484 → 0.482, −0.002 ± 0.031 | night-danger |
| Towns' night (v2830 / v2912) | +0.079 ± 0.052 (eliminated 55 → 37 over 3×48) / −0.013 ± 0.040 | night-danger |
| GATHER pre-test v2830 | 0.464 → 0.429; eliminated 20.7 both | gather-collapse |
| Fade test v2839 → v2886 → v2912 | 0.504 → 0.484 → 0.462 | move-further-windows |
| v2912 → v2956 | 0.473 → 0.519, +0.045 ± 0.047 | gather-collapse |
| Slide replay v2934 / v2956 / v2979 / v3003 | 0.506 / 0.566 / 0.528 / 0.534 | eval-slide |
| v3025 → v3045; v3025 → v3090 | −0.014 ± 0.033; 0.553 → 0.515, −0.038 ± 0.042 | eval-slide, overhaul |
| Follow test (C++), v3045, 144 pairs | 0.520 → 0.813, +0.293 ± 0.059; eliminated 32 → 42 | teacher-measurements |
| Candidate prior β 0/2/4 (v3090) | 0.562 / 0.579 / 0.549 | overhaul |
| Label prior 2/4/8 | −0.007 ± 0.080 / +0.046 ± 0.053 / +0.052 ± 0.064 | overhaul |
| Mixture follow 0.9 vs 0 (v3090) | 0.792 vs 0.507, +0.285 ± 0.112 | overhaul |
| Town guard 0 vs 1 (v3150, follow 0.9) | 0.849 vs 0.859, +0.010 ± 0.116; towns lost 83 → 64 | overhaul |
| Priors-only untrained model | 0.123 ± 0.029; eliminated 33 | overhaul |
| Deploy-60 bundle pair (v3176) | 0.592 vs 0.599, +0.007 ± 0.058; rerun 0.605 | overhaul |
| v3205 / v3226 / v3229 (night of 11-11) | 0.672 ± 0.073 / 0.820 ± 0.082 / 0.861 ± 0.067 (eliminated 10 / 9 / 2) | overhaul, deploy-60 |
| v3226 − v3205 | +0.148 ± 0.067 | overhaul |
| No-advisor arm (v3226) | 0.688 (−0.132 ± 0.078 vs inputs on) | overhaul |
| Measurement A, v3255 ×2 under load | live 0.685, blind 0.653, move_live 0.730, gate_pred 0.690 | overhaul |
| Morning replays alone (11-12) | v3229 0.697, v3255 0.805, v3289 0.702, v3301 0.634, v3254 0.772 (eliminated 9 / 10 / 13 / 14 / 14) | morning, overhaul |
| Null / control pairs | −0.022 ± 0.060, −0.070 ± 0.042; arms 35 min apart +0.050 ± 0.033; same weights −0.164 ± 0.080 | move-further-windows, eval-slide |

### 4.3 Battle model

| Metric | Values | Date | Source |
|---|---|---|---|
| Battle eval | +0.088 to +0.097 (53 evals); 0.0912 (8734), 0.0887 (8746); "+0.095", shooters-idle FAILURE line at every eval | Nov 5 – Nov 11 | battle-flat, bug-audit, gather-collapse |
| value_loss_fresh | 0.0047 → 0.0070 (mix) → 0.0062 → 0.0073, flat since Nov 9; same-buffer checkpoints 0.00386–0.00483 | Nov 2–10 | bug-audit, battle-noise-walk |
| Update KL / clip / steps | acc 24: KL 5e-5, clip 0.05%, 4 steps; acc 2: KL 9.2e-4, clip 1.3%, 48 steps | Nov 5 | battle-flat |
| True signal share | ~0.5% applied (raw 0.2%) | 11-10 | battle-noise-walk |
| Participation (games carrying the gradient) | policy 14.1, value 3.9, foresight 19; the length weight predicts 30 / 28 | 11-10 | battle-noise-walk |
| games_by_weight | baseline 49 ± 14 (49 updates); after scale 30: 66 vs 48; value_loss_fresh sd 0.0021 → 0.0009 | 11-11 | eval-slide |
| Policy drift | entropy 1.01 → 0.76; defend 0.059 → 0.106; wait 0.15 → 0.21 | Nov 5–10 | bug-audit |
| Throughput | ~4.6 updates/h; 777 s an update waiting for real battles; GPU 1.3% | Nov 5 | battle-flat |

### 4.4 Credit, critic and learning signal (campaign)

| Metric | Value | Date | Source |
|---|---|---|---|
| Advantage variance between player-days | 96.5% (schedule 98.1%, share 99.7%) | 11-09 | bug-audit |
| Critic EV (audit) | V_end 0.35 over prior (0.82 raw), G7 0.54, privileged V7 0.30, public 0.15 | 11-09 | bug-audit |
| Held-out EV vs real labels | V2 0.13, V5 0.05, V7 0.03; growth 0.57 / 0.52 / 0.49; V_end 0.66 | 11-11 | stale-v-valves |
| Value unexplained share | calm 0.823; collapse 0.857–0.878; after 0.812–0.834 | 11-11 | stale-v-valves |
| Separate value trunk | held-out EV C 0.369 > B1 0.321 > B10 0.226 | 11-11 | night-danger, stale-v-valves |
| Adam elementwise \|m\|/√v | trunk median 0.20 (noise 0.23); value group 0.50; move_out 0.046 | 11-11 12:00 | overhaul |
| Horizon surprise sd | 0.0496 / 0.0742 / 0.0868 at 2 / 5 / 7 days | 11-11 | overhaul |
| Credit segments | credit sd 0.54–0.61 of the 7-day credit's; k5 ~0.01–0.04; k7 0.07 → 0.29 → 0.10–0.15 | 11-11/12 | overhaul |
| Signal fraction, mixed vs all-own | trunk 0.066 → 0.018; value 0.083 → −0.007 | 11-11 | overhaul |
| Segments split check | trunk signal 0.041 on / 0.019 off (all-own) | 11-12 | overhaul |
| Credit evidence at v3289 (0–2 d) | HIRE +0.021 (t 5.1), RECRUIT +0.009, MOVE +0.006, BUILD −0.009, END −0.010 | 11-12 | overhaul |
| NK2-prediction head | AUC 0.70 → 0.82–0.96; plan ce ~1–2.7 vs uniform 2.6–4.6 | 11-12 | overhaul |

### 4.5 Throughput, cost, timing

| Item | Value | Source |
|---|---|---|
| Campaign update | 124–164 s of a ~175 s cycle (11-10); 230–322 s (deploy 63); 226–423 s with the head | move-further-windows, overhaul |
| Eager fallback (no CUDA graph) | 880–1106 s vs 224–296 s | deploy-safety |
| Campaign eval | ~17 min of play; cycle ~77–90 min | gather-collapse, night-danger |
| Client CPU per decision | 4.4–4.7 ms; CPU forward 27.2 ms (3 windows) + 1.6 ms per window | move-further-windows |
| NK2 advisor per decision | 4.2 ms (36×36), 16–17 ms (72/144), p99 42–60 ms | teacher-measurements |
| nk-predict cost | 6.1–6.8 ms vs 6.4 ms per decision; GPU 12.0 vs 11.65 GB; step ratio 1.027, prepass +0.67 s | overhaul |
| Mac game | 20–45 s; 48 games × 2 arms ≈ 15 min | gather-collapse |
| GPU memory | 20.8 of 32.6 GB (11-10) | move-further-windows |
| Decisions per game | 1,700 → 2,500 after all-own cycling | overhaul |

---

## 5. Project state as of 2026-11-12 morning (~08:40)

**Running on the PC (joint7, deploy 65, a332c25):**
- Campaign on the all-own line from v3229: NK2's choice hidden in every game and in the eval (48 own); nk2_follow 0.
- NK2's planner outputs stay live inputs (schema 19 candidates, threats, plan).
- `credit_segments=true`; `schedule_action_price=1e-5`; advantage head adv_weight 1 with adv_gain 1; town_guard 0; adv_by_kind false; lr 1e-5.
- NK2-prediction head trains at nk_predict_weight 0.1 every 8th micro-batch (stage 1).
- Teacher loss off since deploy 57; GATHER label gone with it; valves off; value-loss guard active.
- Battle side: accumulate 24 (deploy 47), battle_length_scale 30 (deploy 54), b1-aux readings on; aux_value phase not recorded as deployed.
- Latest PC evals: v3272 0.787, v3289 0.629 (15 eliminated).

**Running on the Mac:**
- Replicate arms v3301b, v3254b, and v3205 ×2 (v3205 from runtime/saved/campaign_joint7_pre_d60.pt).
- Rollback rule set in advance: only if v3301 is below v3254 by more than 2 se pooled.

**Prepared, not run:**
- Deploy 66: rollback to checkpoint_03254 (scratchpad d66/; dry run cancelled).
- Deploy 67: mixed games (follow 0.9, choice games 0.5, adv_by_kind on, town_guard 0) plus the precision-weighted F (8623b77). Branch d67-mixed 3f4a3a4; dry run passed. The go is e5's once the six arms are in (the user: "continue doing everything you think is needed").

**Built on branches, unmerged:**
- F3 town-hold (9637796).
- F1 f1-others (9b7be05, schema 20 w_other_turns; conflicts with w-stop's schema 20).
- w-stop (705c48b; premise failed, no consumer).
- nk-stage2 (nk_read live / blind / late / predicted / gate modes; diagnostics only).
- adv-gain stage 2b not started.

**Decided:**
- NK2 teacher removed.
- Continue joint7; fresh run parked. A fresh run would be reconsidered only if score_own stays flat for two days or the schema-20 migration proves costly.
- Credit by segments, no fixed schedule.
- No step credit and no fixed negative credit; worthless actions left to the price.
- LR stays 1e-5 until a signal reading supports 2e-5 (limit 3e-5).
- Army arranging postponed; teleports need nothing new.
- NK2's planner must eventually leave live serving ("no monstrosity"), via the prediction head.

**Undecided / open:**
- Mixed vs all-own games. The user leans mixed; e5's evidence each way rested on noisy single arms.
- The LR step.
- Whether the live planner inputs are worth keeping: their play value is small and unproven (−0.03 ± 0.075 and −0.13 ± 0.078; pooled ~−0.08 ± 0.05). With a blind model, choose between keeping the planner live and training without it (blind + aux head, or imitation of our own planner-aided policy, which would be a new loss needing approval).
- The schema-20 bundle (artifacts per the user's design, boat value, "hero is the town's defence" input, NK2 evaluator exports).
- Forecast-out externalization (not approved).
- Town-attack forecast (a new loss, needs approval).
- F2, the army after a meeting (on hold).
- Exchange cap 8 → 16 (proposed 11-10).
- The cycling plateau (5–7 exchange decisions a player-day).
- The PC's v3289 drop.
- The rules draft (docs/session_rules_draft.md, 21 rules) awaiting review.
- 320 copied games (pc/trunk, 8 GB) to delete after the 300-game trunk run, whose result is not recorded here.

**Gaps list given to the user (07:59), ranked by where score is lost:**
1. Concentrating force: garrisons 21–28k against 15–16k on all heroes at day 49; the main hero at or below the strongest rival from day 35.
2. Defence: 10–15 towns lost empty per 48 games; self-play never attacks like NK2.
3. Relations between our own heroes.
4. What an action opens (blockers, boats).
5. Multi-day goals (TRAVEL is a pooled area).
6. Credit horizon (the 2-day credit pays HIRE and charges BUILD).

Score trajectory for context: own play rose from ~0.51 (v3090, morning of 11-11) to a Mac level of ~0.70–0.80, flat within noise since ~v3226.

---

## 6. Cross-cutting insights for the paper

1. **Evaluation noise was the most frequent source of wrong decisions.**
   - A 48-game eval is ±0.11 at 2 se; one Mac arm moves ±0.08 between runs; a loaded machine reads ~0.1 low.
   - In three days it produced: an "eval slide" theory (11-11 05:00), the deploy-62 rollback, the "0.861" headline, the reversal to all-own games (deploy 63), and a feared decline (v3289).
   - Working protocol that emerged: pinned maps (the RMG is non-deterministic), side-by-side paired arms, ≥2–3 replicates run alone, pooled estimates, decision thresholds fixed in advance.
   - For a low-budget project this sets a floor: an effect under ~0.05 cannot be seen without ~300 games.
2. **Optimizer and diagnostic proxies misled repeatedly.**
   - Signal ratios under 4-pass updates have a pure-noise value of 0.31, not 0.155.
   - Signal fractions carry Adam's history across checkpoint and recipe changes (~6 updates).
   - Value loss tracks target spread, not critic quality (corr 0.98 with target variance).
   - Always compute a diagnostic's null value under the exact recipe, and confirm with play.
3. **Imitating a heuristic AI imports its bugs and its blind spots.**
   - NK2's phantom-delivery bug produced 71% of the teacher's MOVE labels onto our own heroes and towns.
   - The GATHER label collapsed play twice, because NK2's follow-through (danger avoidance) was not ours.
   - Teacher gradients also eroded hand-made priors (night_gain).
   - Removing the teacher (deploy 57) and later NK2's choices (deploy 60) coincided with the largest measured gains (0.51 → ~0.70–0.80 on the Mac), though attribution between mixed and all-own phases remains unresolved.
4. **Mixture-policy pitfall.** Following NK2's choice with probability ρ = 0.9 scored +0.29, but the exact policy gradient into the learner's own logits shrinks to ≈ P_own/9 on followed rows, so the minor component barely learns. The mixed policy also lost more towns: own deviations walked heroes out of guarded towns.
5. **Credit assignment is the binding limit.**
   - 96.5–98% of advantage variance is shared within a player-day; the critic explains 12–18%.
   - At the weight level the trunk's policy gradient is indistinguishable from noise (|m|/√v 0.20 vs 0.23).
   - Zero-initialized new channels effectively never move at lr 1e-5; their starts must come from offline fits.
   - Battle gradients rest on 4–14 games per update with ~0.5% signal.
   - Measured horizon noise (2-day sd 0.050 vs 7-day 0.087) led to empirical-Bayes credit segments rather than a fixed horizon schedule.
6. **Self-play blind spots.**
   - Own-vs-own games have almost no eliminations (~0.02), so defence, run-away behaviour and army concentration are under-trained.
   - Heroes died in the field to NK2 while self-play never punished it; armies turtle in garrisons.
   - The eval opponent exposes what self-play does not; eliminations, not score margins, carried most losses.
7. **Priors as the safe channel for knowledge.**
   - New knowledge placed in logit priors (night danger, towns' night, unseen exits) gave measurable gains (+0.06 to +0.13 paired).
   - The same knowledge fed as inputs to trained weights backfired (in-town heroes moved to riskier tiles).
   - Bounded value views, sign-safe small starts and share units are the developer's recurring constraints.
   - Unbounded gold units fail on large maps; empire-share units under-price late losses. Unresolved: a per-game size-aware reference.
8. **A planner's value can be a marker, not reasoning.**
   - NK2's planner inputs reached the policy mainly as a candidate marker on cells reachable today (R 0.74–0.81 via "toward").
   - The candidate set is locally predictable (AUC 0.96–0.98); the pick is not (top-1 ~0.30); neither engine-path routing nor a predicted marker reproduced it.
   - Its play value is small and unproven. The developer's elegance constraint (no live "monstrosity") pushes toward predicted or auxiliary-target designs.
9. **Engineering failure modes on a single-GPU, CUDA-graph training loop.**
   - Host reads, Python-list indexing and live autograd tensors break graph capture (3–4× slowdowns or failed deploys).
   - Tiny-batch time gates measure one-off costs.
   - Gate greps must come from real output; learned values drift from configured values.
   - `pgrep -f` matches its own launcher; regexes on absolute paths match dates; /tmp cleanup wipes tooling; editing a worktree under a running eval corrupts it.
   - Diagnostics must never end the run.
   - The learner, not game generation, became the bottleneck (~95% of the cycle). Campaign inference runs on 12 CPU processes; the Mac serves as the paired-eval farm.
10. **Multi-agent development dynamics.**
    - Two Claude sessions split build/measure (b1) from review/deploy (e5) and caught many of each other's errors: unpaired A/Bs, wrong routing reading, wrong calendar idea, wrong step-credit reason.
    - Failures came from passing labels through unchecked ("the user's call"), late reporting (a failed deploy unnoticed for 90 minutes), overwriting shared memory, and acting on proxies or single arms when a 25-minute replay was available.
    - Most standing rules were written in response to these: decide-don't-defer, explanations must predict, no long experiments, the 5-minute outcome report.
11. **The developer's interventions were mostly about method, not content:** demanding root causes, predictive explanations, representation-aware features, elegance, no extra losses without approval, and judging learning by its slope. Several AI proposals were overruled and later borne out:
    - The action price.
    - Not using a fixed negative credit.
    - No mid-run value-gradient valves: their premise failed within hours, with the developer already uneasy ("'valve' sounds kinda hacky").
