# Git ledger (alpha_heroes + vcmi patches)

Source: `git log` of `/Users/alon/projects/alpha_heroes` at HEAD `a332c25` (branch `move-windows`, 2026-11-12 01:15), all 466 commits on that line read in full (~316k chars of messages); `git branch -a` and `git log move-windows..<branch>` for the 30 local branches (510 commits reachable from all refs, so 44 exist only on side branches); the vcmi submodule's 45 commits since 2026-10-20 on its checked-out commit `826909100`. Times are the author-local wall clock git prints (mostly US Central; offsets −0500 until 2026-11-01 02:00, −0600 after, plus an odd −0600 stretch 2026-10-25 16:37 → 10-29 05:22). Items marked **[mem]** come from the project's memory notes (`~/.claude_personal/.../memory/*.md`), not from commit text. "vN" = campaign checkpoint version; four-digit battle numbers (5829, 8797…) are battle epochs.

---

## 1. Overview

### 1.1 Commits per day (HEAD line, 466 commits)

| Date | Commits | Opus 5.5 | Fable 5.1 | Fable 5.2 | No trailer | Main activity |
|---|---|---|---|---|---|---|
| 10-21 | 6 | 6 | | | | VCMI submodule, launcher, UpLeftAI |
| 10-22 | 16 | 16 | | | | battle contract v1–v6, PPO, castle_vs_inferno |
| 10-23 | 33 | 33 | | | | self-play, search (Gumbel), model v2, contract v7–v9 |
| 10-24 | 13 | 13 | | | | model v3, zero-sum reward, contract v10–v13, generator |
| 10-25 | 15 | 15 | | | | contract v14, actors/league, MMAI evals |
| 10-26 | 30 | 30 | | | | inference server, persistent battles, KL control, v15 |
| 10-27 | 3 | 3 | | | | review fixes (3263898) |
| 10-28 | 0 | | | | | campaign AI built uncommitted (lands in 5604d8e) |
| 10-29 | 14 | 12 | | | 2 | campaign AI + joint training; Φ v2; PC setup |
| 10-30 | 32 | 30 | 1 | | 1 | perf pass, campaign model v2, PC/CUDA bring-up |
| 10-31 | 57 | 57 | | | | DataLoader rebuild, joint5, teacher/imitation/memory, NK2 advice, joint7 |
| 11-01 | 36 | 35 | | | 1 | joint7 memory/permit/drain operations |
| 11-02 | 11 | 10 | 1 | | | loss shares, teacher over the turn, TBB cap |
| 11-03 | 16 | 16 | | | | throughput: window budget, decode, forkserver; Φ hero base |
| 11-04 | 19 | 19 | | | | MOVE audit, trades, schemas 11–13 |
| 11-05 | 34 | 34 | | | | schemas 14–15, MULTI, two collapses/rollbacks, neutral towns |
| 11-06 | 6 | 6 | | | | optimizer plan, escape prior |
| 11-07 | 14 | 14 | | | | privileged critic, human recorder, schema 16 |
| 11-08 | 3 | 3 | | | | duplicate-stack merge, any map size |
| 11-09 | 8 | 5 | | 3 | | meeting audit fixes, step credit set aside |
| 11-10 | 31 | 15 | | 15 | 1 | stall fixes, GATHER collapse #2, guard, schemas 17–18, paired_eval |
| 11-11 | 64 | 50 | | 14 | | schema 19, teacher off, follow, advantage head, bundle 60, segments |
| 11-12 | 5 | 5 | | | | nk_predict deploys 63–65 |

Totals: Opus 5.5 427, Fable 5.2 32, Fable 5.1 2 (`1f7db1c` 10-30 15:16; `f2165ba` 11-02 21:08), none 5 (merges `4036691`, `8ef4e49`; git's auto-revert `fc9c277`; `5243c56`, `8d96040` on 10-29 19:36). Fable 5.2 appears only 11-09 20:32 → 11-11 12:58; nearly all are the e5 session's run-line/deploy commits ("joint7: …"), reviews, and "a diagnostic cannot end the update" guards. Commit hours: every hour of the day has commits; 00:00–05:59 holds 71 (15%), reflecting unattended overnight work. Diffstat over the line: 2,059 file-changes, +74,854 / −8,987 lines. Median message length: 359 chars (10-21..27), 412 (10-28..11-03), 676 (11-04..12); total message text 54.8k → 106.6k → 153.4k chars.

### 1.2 Branches (`git log main..<b>` / `move-windows..<b>`)

`main` (`45b8692`, 11-11 07:15, worktree `alpha_heroes-deploy`) stopped at deploy 56; `move-windows` is 56 commits ahead and is the de-facto main line since deploy 52 (11-10 ~23:45). Only 2 merge commits exist; everything else moved by rebase/cherry-pick, so the same change can carry two hashes (e.g. `0239df9` ≡ `84aa3cd`, see `4036691`). `origin/main` was last pushed at `25af47e` (11-05 22:15): 134 commits of the line are local only.

| Branch | Tip | Purpose / status |
|---|---|---|
| move-windows | a332c25 11-12 01:15 | Started as schema 17 (`75cfe7d` 11-10 14:10, merge `8ef4e49` 18:06); main was fast-forwarded along it up to deploy 56 (`45b8692`); alone it carries deploys 57–65 (56 commits). |
| nk2-advice | a332c25 | joint7's launch branch (10-31; `origin/nk2-advice` = `2398670` 11-01 09:42); now the e5 worktree, same tip as move-windows. |
| fast-train | 0d70ff1 10-31 21:45 | e5's DataLoader training-loop rebuild (`24ab45d`); merged. |
| e5-review | 8501601 11-10 04:40 | e5's review of the 11-09 meeting fixes / step credit, stall fixes; merged. |
| b1-aux | 70bdd61 11-10 10:58 | Battle value auxiliary phase (PPG, `b20e171`…`70bdd61`); merged, `aux_value` off. |
| fight-step | 4cd9e16 11-04 01:14 | 1 unmerged commit: MOVE fight layer ×100 step (conflicted with the "never raise the rate" rule; [mem] user asked, never merged). |
| gather-once | 8d7424b 11-05 01:57 | 1 unmerged commit "(branch, not deployed)": town-visit advice once a visit, after the visit-advice collapse. Pushed. |
| hota | 6dcbe31 11-06 09:46 | 4 unmerged commits: HotA transfer plan, port inventory (61 submods; HotA rebalances 31 creatures, 8 artifacts, 3 heroes), AlphaContent key matching, setup script. |
| run-away | 4d3f8b9 11-11 08:48 | b1's night-danger/run-away work; 12 of 13 commits landed on main under new hashes (a9ad07e→3f9e1fe, 2ef7677→7aa44b9, 2458a27→6a73dc1, 23d10d1→b721fe2, 25866df→115dd13, 090d525→7f058b7, 5d31166→27c0994, f032d3f→884b754, 74480f8→e8649fb, 994c5ec→6372cfc, 4785d0f→c4232eb, 8605deb→2bdcdcb); `4d3f8b9` (ALPHA_MAP_FOLLOW_ADVICE test switch) not, its idea re-done in `ee42a90`. |
| overhaul-e5 | 43ce74a 11-11 11:48 | e5's copies of the schema-19 commits (7, all re-landed: dad4fcd→96b6f57 … 43ce74a→535d042). |
| overhaul | e005793 11-11 12:58 | b1's overhaul = deploy 57's head. |
| overhaul-final, adv-head | 1876d0f 11-11 14:05 | Deploy 58 (advantage head stage 1). |
| adv-gain | 5a04565 11-11 14:22 | Advantage head stage 2a (policy reads the head); enabled in deploy 59. |
| own-games | 2943d45 11-11 16:41 | Deploy 59 (own games 0.5, split eval, adv_gain 1, action price 2e-6). |
| teleport-diag | 52f0d37 11-11 16:55 | Teleport diagnostic (≡ `7901c62`). |
| bundle-60-work | c9a396e 11-11 23:28 | Deploy 62's head. |
| bundle-60 | 072db5a 11-12 02:00 | b1 worktree; +1 commit (nkp_* progress columns), "next deploy". |
| nk-predict | 1839a0f 11-12 00:36 | nk_predict stage 1 source; cherry-picked as `254b9b2`, `af98963`. |
| nk63-own / nk63-mixed | 254b9b2 / 3220496 | The two prepared variants of deploy 63; "own" ran, "mixed" (`0e1df92`) did not. |
| nk-stage2, w-stop, nk-gate | 7265d83 / 705c48b / bf4c5e0 (11-12 02:22–04:02) | nk_predict stage 2: `nk_read` modes (live/blind/parts/gate) for paired-eval harnesses, "never served"; w-stop adds schema 20's `w_stop` input (unread). Not deployed. |
| d67-mixed | 8623b77 11-12 08:40 | Deploy 67 line (mixed games on deploy 65's code, `3f4a3a4`) + precision-weighted segment F (`8623b77`); [mem] dry run passed; not deployed in this history. |
| f1-others | 9b7be05 11-12 08:29 | F1 draft: schema 20 `w_other_turns` (other heroes' time to a tile), "unbuilt and untested". |
| town-hold | 9637796 11-12 08:27 | F3 draft: a town's holding value, untested. |

15 git worktrees exist (one per session/task: adv, b1, b60, d67, deploy, e5, e5r, f1, f3, gain, gate, hota, nkp, wstop + main).

---

## 2. Phase timeline

**P0 — Setup (10-21 22:20 → 10-22 13:45).** Goal: VCMI building and playing headless. `e8641c8` 10-21 22:20 submodule + local build; `95c49de`/`948bed9` app launcher; `1a42504` 22:57 UpLeftAI test battle AI; `2cd251f` 10-22 12:31 guarded headless runner with hard limits; `13e484d` hand-over to StupidAI from round 10. Outcome: scripted battles run reproducibly.

**P1 — Battle AI (10-22 14:53 → 10-27 03:37).** Goal: a PPO battle model beating BattleAI/MMAI. `905b16b` 10-22 14:53 contract v1 + ONNX + AlphaBattleAI; `10cf6dc` 22:28 castle_vs_inferno training loop; `27e14e2` 10-23 00:35 per-side advantage normalisation; `f3d2fd9` 07:52 self-play; `ecbf9d1` 09:07 BattleSimulation on VCMI's rules; `8aa97f3` 12:05 Gumbel AlphaZero search; `09f1a9b` 10-24 09:19 model v3; `0f246c8` 12:09 zero-sum asset reward; `5e40c78` 16:00 engine-played attack values; `fd41244` 20:35 generator; `99dcb5a` 10-25 16:37 contract v14, actors, league, real games with NK2; `ab08f16` 10-26 10:17 central inference server; `02e57cf` 11:50 persistent battle processes; `27a8d41`/`7882ace` KL control; `1edee2c` 15:13 contract v15 battle history; `ca3ee7b` 15:43 window credit (replaces GAE); `3263898` 10-27 01:10 review fixes. Outcome: throughput 73 → 238 decisions/s trained; search built but training stayed plain PPO; battle eval later sat at ≈ +0.09 vs BattleAI/MMAI for ~1,100 updates (`3ff87b5`).

**P2 — Campaign AI + joint training (10-27 → 10-30 13:57).** Goal: an adventure-map model trained jointly with the battle model, evaluated vs Nullkiller2 (NK2). joint1 [mem] 10-27 22:40; `5604d8e` 10-29 03:06 (one commit: campaign contract v6, AlphaMapAI, NK2's brain as torch priors, share heads V2/V5/V7 + V_end, battle v16, war machines at half price; run joint2 from 10-28 22:23); `dd8c6b8` 13:21 battle value priors; `bf6bf9e` 22:27 greedy 48-game eval + RMG town-buildings bug; `552625f` 23:05 hiring sees the hire's army, Φ heroes by skills (contract 7); `d9cb662` 10-30 08:07 + `0239df9` 13:57 perf rounds; `d7a92a2` 09:53 Φ v3 (schema 8); `05914d0` 13:26 campaign model v2 (2.2M params, behind a switch). Outcome: campaign learning signal weak; evals ~0.1–0.25.

**P3 — PC/CUDA and the fast loop (10-30 15:16 → 10-31 13:57).** Goal: train on the RTX 5090 via WSL. `1f7db1c` (Fable 5.1) v2 review + 1/3 games vs NK2, reversed 7 min later by `bae4696` (self-play only); `e505aba`/`7f9bda6`/`e7014c1`/`c7fce84` PC-only bugs; `ffa3f91`/`316d8cb` CUDA graphs for inference/update; `24ab45d` 10-31 09:29 DataLoader rebuild (e5); `9e2bdb4` joint bugs; `f7e6e55` 10:13 whole micro-batch step as one CUDA graph; `c76cfbe` 13:07 subtype rows 8,191 hashed → 733 (2.25M → 1.18M weights); `dd0bcd4` junk games; `96266d5` 13:57 "attributed" credit with written predictions. Outcome: joint5 (campaign v2, self-play) flat ([mem] eval v357 0.099, stopped 15:13).

**P4 — The teacher era begins; joint7 launched (10-31 14:43 → 11-01).** Goal: learn from NK2 (AlphaStar's supervised-init lesson). `aeaf885` 14:43 NK2 recorder + imitation + teacher KL; `448292f` 15:19 memory (schema 10); `0e59440` share-change heads; `a2349e7` 16:24 full run (imitation → joint6); `87e0043` schedule credit + score-weighted teacher; `2c884d9` 21:37 LR cap 3e-5; `995fdfa` 21:52 NK2 advice on our own states; `0aa6500` 21:53 joint7 launcher; then 36 commits on 11-01 of run operations: exchange cap (`949545b`), KL target with teacher (`5dea773`), CUDA memory fraction (`4306bc9`), NK2 crash (`01743f0`), memory admission/guard/keeper (`04d7058`…`b1ba575`), no data loss on stop + drains (`6f76d48`), census and spooling (`496055f`, `cd52bc5`). Outcome: joint7 runs continuously from here to the end of the history.

**P5 — Throughput and loss balance (11-02 → 11-03).** `4c2212a` 11-02 09:28 value heads by target variance, battle 60:20:20 mix; `71ea846` NK2 plan as foresight label; `a8b587e` 12:52 teacher over the turn (accept sets, loss shares); `f2165ba` TBB cap; `f398aec`/`78d55e4` forkserver preload; `489deaa` window budget; `e79960a` decode via libz; `ce524f4` conversion in graphs; `a920331` 23:05 Φ hero base by army + MOVE fight features. Outcome: the learner waits on games, not the GPU.

**P6 — Action-space and prior expansion, two collapses (11-04 → 11-05).** `ab52217` 11-04 09:04 MOVE feature audit; `9ebc90d` trade prior; `778b92f` schema 11; `d60df0b` question head; `542fdbe` schema 12; `08627ff` Φ army × hero; `87e79b5` schema 13 ARTIFACT; `8458ede` 11-05 00:27 memory BPTT; visit advice → `26bc2a1` revert; `e3f5503` schema 14 MULTI; `69d0332` GATHER rollback; `5f9c560` Φ v5; `5eaafd1` schema 15; `8797125` battle 512-decision steps; teacher-span flip-flops (`0d54e0f`…`444ebfc`); `296b06a` neutral towns bug (see §5–6). Outcome: many genuine bugs fixed; evals fell from 0.506 (v967) to 0.248 (v1037) and recovered to ~0.39–0.48 [mem].

**P7 — Optimizer, critic, recorder, map size (11-06 → 11-08).** `8c6cea9` optimizer plan (per-group clip, critic ×10 rate, frozen shares); `53c1df7` KL target 0.03 after 0.441 → 0.363 → 0.335; `6ec628f`/`a3b6e96` battle escape priors; `e9f88da` → `0fe2869` privileged critic; `e620710` privileged credit on; `1790e5b`…`8fcc054` human-game recorder; `1608288` schema 16 (every human army move); `7192bbd` duplicate merge back; `1a9fa63` any map size (36×36 block grid). Outcome: slower commit rate (6, 14, 3/day).

**P8 — Audit and numbered deploys (11-09 19:11 → 11-10 23:04).** `10bcddf`…`55f857b` meeting audit M1–M5 (advised own-hero meetings 52.2% → 9.5%); `6021234` step credit set aside; `2466290` action price (off); `affa756`, `592fa6f`, `8501601`, `897db08`, `0c8ebfd` stall/deadlock fixes; `36a38de` GATHER label → collapse #2 → `e011809` off + rollback to v2728; `e577a76` battle back to 6,144-decision steps; `b20e171`…`70bdd61` aux value phase; `fab5eeb` value-loss guard; `75cfe7d` schema 17 further windows/unseen exits; `63daa24`/`7441d1e` schema 18 night danger; `7aa44b9` paired_eval; `6a73dc1` night_gain frozen. Outcome: deploys 42–52 in ~25 h; paired testing before the PC becomes the rule.

**P9 — Overhaul: NK2 as inputs, own play (11-11 00:58 → 11-12 01:15).** `85e6f2e` unseen exits; `884b754`/`e8649fb` towns' night; `90d153c` GATHER ramp; `2bdcdcb`/`45b8692` value valves; schema 19 (`6be714c`, `ab18404`, `2c53445`, `b375bea`); `d75e3d3` teacher loss switch → off in deploy 57; `73d01a5` follow mixture; `428baab`/`5a04565` advantage head; `f812562`/`0edd460` own games + split eval; bundle 60 (`2b2cf74`…`3719556`); `c916412` credit by segments; deploys 57–65; `c5977bf` nk_predict auxiliary head. Outcome [mem]: own-play eval rose from ~0.51 to ~0.70–0.80 (Mac, flat since v3226; a 0.861 arm was luck); NK2 makes no choices from deploy 60 except deploy 62's short mixed spell.

---

## 3. Deploy ledger

### 3.1 Run starts (battle-only and joint)

| Run | Start | Commit / evidence | Notes, outcome |
|---|---|---|---|
| 400-epoch self-play rounds | 10-23 | `693ba88` 13:42 | V+V2 policy heads beat V alone vs BattleAI: attack −0.09 vs −0.24, defence +0.10 vs +0.01 (one seed each). |
| v14-11h | ~10-25 evening (unclear) | cited in `0fc65e0`, `2b7658d` | Entropy bonus push: shooters shot in 90% → 55% of chances. |
| v15-base-lr2 | 10-26 | `27a8d41` | KL grew 0.008 → 0.03–0.09 at fixed LR; motivated KL control. |
| v15-window | 10-26 ~15:43+ | `ca3ee7b`; `64c3f6b`, `9071945` | Closed lid ate 2.3 of 11 h; stopped on garbage action types (`de4418b`, `c64e327`); stopped at update 1241 on an endless eval battle (`b3de8c8`). |
| joint1 | 10-27 22:40 [mem] | `5604d8e` "goes on from joint1's checkpoint 2840" | 9 h [mem]. |
| joint2 | 10-28 22:23 | `5604d8e` (runtime/train/20261028-2223-joint2) | Battle drifted from shooting; froze at 3078 (`3239517`, `dd8c6b8`); stopped at v128 [mem]. |
| battle-priors, battle3 | 10-29 12:50 / 10-30 13:42 [mem] | `16e50ce` (battle3 ignored SIGINT) | Battle-only from 3319 / 5518 at LR 3e-5. |
| joint4 | unclear | [mem] "joint4's start eval" at 5829: +0.096 vs both | No commit names it. |
| joint5 | 10-30 (Mac, then PC 19:46 [mem]) | `3ff87b5`, `96266d5`, `fadf8ae` | Campaign v2, self-play. Stopped 10-31 07:04 (12 h cap), KeyError stop 14:00 (`fadf8ae`), final stop 15:13 at eval v357 0.099 [mem]. |
| fullrun-imitation (+ joint6) | 10-31 16:24 | `a2349e7`, `8e3f457`, `2c884d9` | Imitation held-out loss 2.47 → 1.97 in 2.5 h at 1e-4, then flat; continued at 3e-5 then 1e-5 "with no end (the backup run)". Whether joint6 itself ever ran: **unclear**. |
| joint7 | 10-31 night (after 22:14; exact start not in commits) | `0aa6500` 21:53 launcher; [mem] main `32531c4` | NK2 advice on 20% of turn decisions as teacher, from the imitation; runs to the end of the history (v3301+ on 11-12 [mem]). |

### 3.2 joint7 deploys before the numbers appear in commits (10-31 → 11-08)

Deploy numbering exists in memory notes from deploy 25; commit text names deploys only from 49. Main changes reaching the run (commit = the change; deploy time from [mem] when given):

| When | Commit(s) | Change | Outcome / rollback |
|---|---|---|---|
| 10-31 22:39–23:50 | dcec6c0, 89c3c1c, 675976b, d9e58fe | eval niceness 15, minibatch 8192/64 micro-batches, campaign inference on the Mac, permits by games | permits had fallen to 0, run stopped playing |
| 11-01 00:12–05:44 | 949545b, 5dea773, 2322237, 4306bc9, 01743f0, 93bc8f7, 5e72edc, da4bd80 (+5) | exchange cap 32→8, KL target ×(1+teacher), advice 50%, GPU cap 85%, NK2 crash fix, OOM survivable, move label = plan tile | memwatch stops 00:26, 01:47 [mem] |
| 11-01 09:20–09:42 | 973fbdc → fc9c277 revert → c3da392, 2398670 | PC alone; both models' inference on the PC CPU | fallback store reverted after 4 min |
| 11-01 13:22–17:30 | 04d7058, 0034c12, c4e161d, b1ba575, 6f76d48, da6c46c, 3020025, 48765b4 → f4b2e45, 496055f, cd52bc5 | keeper, memory admission (3 fixes), finished games spooled, drains, census | guard stop 12:21; deadlock 13:55–15:00; eval 53 scored 0.000; 13 games → back to 10 |
| 11-02 09:28–12:52 | 4c2212a, fe5cad8, 71ea846, a8b587e | head_scale, 60:20:20 battle mix, move discount, NK2 plan label, teacher sets at every decision | — |
| 11-02 21:30 [mem] | 58d86d0 + f2165ba (vcmi 30658acd5) | TBB cap 4, 8 inference processes | 0.185 → 0.065 CPU s/turn |
| 11-03 06:10 / 07:00 / 07:25 [mem] | 489deaa / e79960a / ee20e37+fb9c4b2 | window budget; libz decode; loss-noise TypeError stopped joint7 at v391 | fixed in 2 min |
| 11-03 09:3x → 09:35 | db68c00 → c01b4cc | glibc arena tuning | reverted (no effect) |
| 11-03 11:42, 16:57 [mem] | 78d55e4, 0321bec | forkserver preload really applied; no over-budget graph | loaders 9.5 → 2.9 GB |
| 11-03 evening [mem] | 0ecb447, a920331 | share-weight ramp by map width; Φ hero base by army; MOVE fight features | Φ change: evals not comparable across it |
| 11-04 09:24 [mem] | ab52217, 0b899a4, 5805895 | MOVE features rebuilt, battle head = engine's answer, safety prior | suicides 275 → 193, still 98% lost |
| 11-04 17:09 [mem] | 50678d1, 778b92f | trade prior, schema 11 | — |
| 11-04 19:44 [mem] | d60df0b (+909067b) | question head as identity match | — |
| 11-04 23:37 [mem] | 542fdbe, 08627ff, 87e79b5, 9fe35c9, d1950b5 | schemas 12–13 | — |
| 11-05 00:38 [mem] | 8458ede, 7e3ee7b | trades priced as Φ; memory_steps=2 (117 vs 108 ms replay) | — |
| 11-05 00:58–01:23 [mem] | de04cf8, 93e39d2, 5d7be3f | NK2 visit/gather/artifact advice replacing its label | **collapse**: eval v1037 0.248 vs v1024 0.378; reverted `26bc2a1`; **rollback 01:46** to v1025 (renumbered 1044) [mem] |
| 11-05 ~08:10, 08:50, 09:45 [mem] | 813e540+e3f5503, 2218ada, 0eab9f0 | Φ unit, schema 14 MULTI, GIVE_ALL cost, GATHER danger | towns emptied; v1137 28/48 knocked out |
| 11-05 10:39 [mem] | 69d0332 | teacher drops GATHER | **rollback to v1125**; v1169 0.349 [mem] |
| 11-05 11:59 [mem] | 5f9c560, then 5eaafd1… | Φ v5, schema 15 | — |
| 11-05 afternoon (drain) | 8797125 | battle accumulate 24→2, outcome 50%, lopsided scale | a noise walk for 4.8 days (see §6) |
| 11-05 20:00 [mem] | daac154 | very negative starts for useless actions | **rollback to v1184** [mem]; broke CUDA capture → `51c08ce` 20:51 |
| deploy 27, 11-05 22:59:58 [mem] | 296b06a + 25af47e | neutral towns listed; battle stall fixes | eval 0.391 → 0.476 [mem] |
| deploy 28, 11-06 00:33:52 [mem] | 635912f + 7223764 | unseen guard priced; value_unexplained | — |
| 11-06 03:33 [mem] | c4241a1 | eval checkpoint saved as weights are pinned | **rollback to v1288's file**, which held v1293's weights |
| deploy 29, 11-06 ~v1425 [mem] | 8c6cea9 | optimizer plan all at once | evals 0.441 → 0.363 → 0.335 → `53c1df7` KL target 0.03 (14:11) |
| 11-06/07 | 6ec628f → e78414f (live 17:20 [mem]) | battle trunk β1 0.95 → back to 0.9 | idle shooters vs BattleAI 5.15% → 1.57% after revert [mem] |
| 11-07 ~09:50 / 13:27 | 0fe2869 / e620710 | privileged critic redesign; privileged_credit on | unexplained 0.863 → 0.806 |
| deploy 36 [mem] | a3b6e96 | escape_strong prior (via wait36 after battle eval > 8561) | — |
| deploy 37, 11-07 23:51 (v1929) [mem] | 1608288 | schema 16 | value loss +~40% from duplicate stacks |
| deploy 38 [mem] | 60a033e | swap pricing | shuffle p 0.50 → 0.02 |
| deploy 39, 11-08 08:27 [mem] | 7192bbd | duplicates merged again | — |
| deploy 40 [mem] | 1a9fa63 | any map size | small maps bit-identical |

### 3.3 Numbered deploys 42–67

| # | Time | Commit | What changed | Outcome / measurement | Rollback |
|---|---|---|---|---|---|
| 42 | 11-09 22:56:37 [mem] | 2466290 (meeting fixes b49632e, 55f857b, 6021234) | advisor meeting fixes, every switch off | evals 0.420, 0.405, 0.431 vs 0.434 (`b0b4330`) | rollback script prepared, unrun [mem] |
| 43 | 11-10 03:11:33 [mem] | affa756 (vcmi a9b7b0a94) | stalemate rule when neither side can retreat | — | — |
| 44 | 11-10 04:34:54 [mem] | 592fa6f + b0b4330 + 414c622 | exchange_span_advised on; inference slow-request logging; idle-movement metric | evals 0.511, 0.491, 0.489 (`36a38de`) | — |
| 45 | 11-10 ~06:26–06:35 [mem] | 897db08 (vcmi d7868e680) + 8501601 | NK2 endTurn race fix; parked inference requests | eval stalls gone | — |
| 46 | 11-10 08:48:04 [mem] | 36a38de (+1d31d6b, 0c8ebfd) | nk2_gather_weight 0.1 (at v2735) | **collapse #2**: 0.475, 0.380, 0.314, 0.345, 0.402, 0.271; eliminated 15–28/48 (`e011809`) | → 48 |
| 47 | 11-10 08:57 [mem] | e577a76 | battle steps back to 6,144 (accumulate 24) | "right and harmless, not a cure" [mem] | undoes 8797125 |
| 48 | 11-10 18:38 [mem] (2nd try; 1st self-refused on a grep of a comment) | e011809 | GATHER label off | restored weights eval 0.497 ±0.052 [mem] | **rollback to v2728** |
| 49 | 11-10 19:18 [mem] (2nd launch) | b3d805f (fab5eeb guard + schema 17 inert) | value-loss guard at 3; schema 17 switches off | — | — |
| 50 | 11-10 20:14 [mem] | e854c6d (7441d1e) | schema 18 night term live | paired before: big losses 49 vs 64, out 4 vs 9 of 36 maps | — |
| 51 | 11-10 21:40 [mem] | e2e794f (3f9e1fe) | push_* diagnostics on night/danger gains | teacher pushed night_gain 13× policy | — |
| 52 | 11-10 ~23:45 [mem] | 0830504 + 80f7398 (6a73dc1, 7aa44b9) | further_windows=1, window_budget 5.0, night_gain frozen | paired −0.003 ±0.024; first clean eval v2934 0.579 (`85e6f2e`) | — |
| 53 | 11-11 01:27:50 [mem] (v2947) | 85e6f2e | unseen_exits on | paired +0.059 ±0.025 (3 replicates) | — |
| 54 | 11-11 05:08 [mem] | 80793cf (884b754, e8649fb, 27c0994) | towns' night (mode 2), GATHER ramp code (off), battle_length_scale=30 | town night paired +0.079 ±0.052 on v2830, −0.013 ±0.040 on v2912 | — |
| 55 | 11-11 06:39:35 [mem] | 90d153c (c4232eb) | GATHER label back, ramped 0.1 over 200 updates | — | rollback point v3032 saved [mem] |
| 56 | 11-11 07:23 [mem] | 45b8692 (2bdcdcb) | value valves (value_valve, kl_valve) | premise failed 09:35 [mem] (target spread, not worse V) | valves out in 57 |
| 57 | 11-11 13:26 [mem] (2nd try; 1st self-refused: graph alive before capture), v3107 | e005793 | schema 19 NK2 inputs, follow 0.9, teacher loss off, GATHER label + valves out, guard memory from v3107 | first eval v3120 0.753 ±0.128, eliminated 17/48 (8–13 before) [mem] | point v3106 saved |
| 58 | 11-11 15:31 [mem] (v3134) | 59c83ac/1876d0f | advantage head stage 1 (adv_weight 1) | v3138 0.765 ±0.122 [mem] | — |
| 59 | 11-11 17:03 [mem] (v3150) | 2943d45 | own games 0.5, split eval 32/16, adv_gain 1, action price 2e-6 | own arm 0.602 (v3169) → 0.739 → 0.802 ±0.090 [mem] | — |
| 60 | 11-11 21:27 [mem] (2nd try; 1st 19:32 failed capture check), v3205 | 4a71252 + 41d12bf + 3719556 | bundle 60 (revisit skip, boats, town guard code, loop guard, fingerprints), NK2 choices off (follow 0, choice games 0), credit segments on | pair +0.007 ±0.058 vs 2943d45 [mem]; loops cut 0.16 → 0.44 /player-day in 29 updates (`8519668`) | — |
| 61 | 11-11 23:16 [mem] | 8519668 | schedule_action_price 1e-5; MOVE/ARMY diagnostics; loops_per_player_day | — | — |
| 62 | 11-11 23:47 [mem] | c9a396e | mixed games again (follow 0.9, choice 0.5, adv_by_kind, town_guard 0.5) | v3205 PC eval 0.629 vs v3216 all-own 0.768 (−0.138 ±0.083) [mem] | **rollback to v3205** |
| 63 | 11-12 00:19 start; 00:35 up in fallback [mem] | 254b9b2 → e11ab5f | all own games from v3229 + nk_predict 0.1 | time gate 7.31/5.18 = 1.41 > 1.15 → weight 0 (= deploy 61's update) | **rollback to v3229** (mixed line quarantined [mem]) |
| 64 | 11-12 00:57 start, failed 01:08 [mem] | 75f509a → f5b5072 | nk_predict as own step every 8th micro-batch | gate 6.95/5.04 = 1.379 (graph re-recorded each update) | restored to 63 from v3234 [mem] |
| 65 | 11-12 01:36 live [mem] | a332c25 | head's StepGraph kept on the model; gate on 2nd update | steps 1.027 [mem] | — |
| 66 | prepared [mem] | — | rollback to v3254 | dry run cancelled; "NO ROLLBACK" [mem] | not run |
| 67 | prepared [mem] | d67-mixed 3f4a3a4 + 8623b77 | mixed games on deploy 65's code; precision-weighted F | Mac dry run passed [mem] | not in this history |

Deploys 1–24, 26, 30–35, 41: not identifiable from commits or memory.

---

## 4. Schema / contract / model-version ledger

**Battle contract** (input/output wire between AlphaBattleAI and the model):

| v | Commit, date | Added |
|---|---|---|
| 1 | 905b16b 10-22 14:53 | contract, model, PPO losses, ONNX export, AlphaBattleAI |
| 2 | a2d1900 10-22 21:30 | boosted multi-horizon value heads V0, V1, V2, V |
| 3 | c80198b 10-22 21:48 | value heads on 0..1, cross-entropy |
| 4 | — | not mentioned in any commit (**unclear**) |
| 5 | 10cf6dc 10-22 22:28 | enemy hero's secondary skills hidden; round numbering |
| 6 | 6aa0724 10-22 23:45 | tanh value heads, −1..1 reward; lr 1e-4 |
| 7 | ba0469c 10-23 08:03 | engine-checked attack masks; u_owner_side (hypnosis) |
| 8 | d2d2674 10-23 18:18 | 20 unit slots, ordering; priors in CSV files |
| 9 | e4e13e6 10-23 19:02 | CREATURE_SPELL action |
| 10 | 09f1a9b 10-24 09:19 | engine-predicted spell value inputs (s_/c_unit_value…) |
| 11 | 5e40c78 10-24 16:00 | engine-played attack values, engine damage estimates, escape rewards, 68-ability bag |
| 12 | 6cba37e 10-24 17:06 | hero value in reward, units hit per attack/cast, neighbour features |
| 13 | f84d9a0 10-24 21:32 | engine spell mastery, castable spells incl. scrolls; neutral battles |
| 14 | 99dcb5a 10-25 16:37 | 14 slots/side, tactics/catapult/tent actions, rule artifacts, exposure features |
| 15 | 1edee2c 10-26 15:13 | battle history (seen spells, per-unit tallies, last 16 events); v14 checkpoints do not load |
| 16 | 5604d8e 10-29 03:06 | magic resistance/mirror inputs |

Battle **model** versions: v1 `905b16b`; v2 `a5808b3` 10-23 17:17 (bounded rule features, attention trunk; scoring from zero `d2d2674`); v3 `09f1a9b` 10-24 09:19 (signed ownership; Lanchester shares `389a879`); later additions without a version number: foresight + wider context `f3e5b1d` 10-30, value priors `dd8c6b8` 10-29, idle-shot prior `8797125`, escape_ahead `6ec628f`, escape_strong `a3b6e96`.

**Campaign contract / schema** (AlphaMapAI records):

| v | Commit, date | Added |
|---|---|---|
| 3–6 | 5604d8e 10-29 03:06 (v3..v6 per [mem] plan 2026-10-28) | every map object, whole-map encoder, TRAVEL, RECRUIT targets, decision echo |
| 7 | 552625f 10-29 23:05 | hero-for-hire experience, skills, army, artifacts |
| 8 | d7a92a2 10-30 09:53 | luck/morale inputs, tavern spells (Φ v3) |
| 9 | f3e5b1d 10-30 13:07 | tile_* whole map, battle records per hero |
| 10 | 448292f 10-31 15:19 | 8×64 memory in/out |
| 11 | 778b92f 11-04 16:33 | question guards and option effects, armed garrisons a_* |
| 12 | 542fdbe 11-04 22:31 | purchases paid by trades, boats, University, remembered guards, stack merge |
| 13 | 87e79b5 11-04 23:05 | ARTIFACT action (give/equip/assemble), x_heroes |
| 14 | e3f5503 11-05 07:56 | MULTI action: RECRUIT_ALL(_HERO), GATHER, GIVE_ALL |
| 15 | 5eaafd1 11-05 12:09 | MULTI TAKE_BEST(_FAST), DROP_SLOW, DROP_FOREIGN; 8 kinds |
| 16 | 1608288 11-07 22:13 | ARMY target slot, within-army moves, all-but-one; SPLIT_SINGLES, MERGE_ALL, GARRISON_SWAP; 11 kinds |
| 17 | 75cfe7d 11-10 14:10 | further windows (MOVE past gates), unseen-exit flags |
| 18 | 63daa24 11-10 19:27 | travel_stop (today's stop on a TRAVEL path) for the night term |
| 19 | 6be714c/ab18404 11-11 11:24; b375bea 12:17 | NK2 candidates k_* (16/hero), tier/count, threats, w_arrive_left, g_phi_past; NK2's own choice g_nk_* |
| 20 | 705c48b (w-stop), 316dd13 (f1-others), 11-12 | drafts on side branches: w_stop; w_other_turns |

Campaign **model**: v1 `5604d8e` (1.5M); v2 `05914d0` 10-30 13:26 (2.2M, 18.5 ms/decision vs 9.0), default from `4336918` 11-01; subtype rows `c76cfbe` (→1.18M); memory `448292f`; question head `d60df0b`; privileged critic `e9f88da`/`0fe2869`; schema-19 advice block `2c53445`; advantage head `428baab`; NkPredictHead `c5977bf`. Width 128 / 2 layers from `a2349e7`.

**Φ (empire value) versions**: EMPIRE_VERSION 2 `552625f` 10-29; 3 `d7a92a2` 10-30; Φ change without a stated version `a920331` 11-03 ("evals … not comparable"); training 4 / eval 3 `813e540` 11-05 07:36 (eval keeps 2026-11-04's Φ); v5 `5f9c560` 11-05 11:47 (movement, morale; eval keeps v3).

---

## 5. Reverts, rollbacks, fixes-of-fixes

| Item | What it reveals |
|---|---|
| `47f2d64` 10-29 20:27 round-60 limit → `1207a12` 20:44 stalemate rule | First idea replaced within 17 min by a chess-style rule. |
| `1f7db1c` (Fable 5.1) 1/3 games vs NK2 → `bae4696` 7 min later | A model's suggestion overruled by the user's "no training against the eval AI" rule. |
| `bf6bf9e` greedy eval → `6008500` (~4 h) sampling again | Greedy play walked heroes back and forth (95% one-tile moves): a policy never trained greedily. |
| `757f998` info-spell mask (09:59) → `567c7d5` (10:42) undone | Eval 0.22 → 0.07; the spells show enemies, which self-play never taught. Confounded: `ad2b667` notes the end-credit check (`6730a59`) ran over the same window and turned it off too. |
| `973fbdc` fallback inference → `fc9c277` revert (4 min) | Design reversed by the user's "PC first, Mac adds slots" call (`c3da392`). |
| `0034c12` → `c4e161d` → `b1ba575` (11-01 13:37–14:56) | Memory admission tuned three times; one version deadlocked the run for an hour. |
| `48765b4` 13 games → `f4b2e45` back to 10 (14 min) | Memory budget measured by crashing into it. |
| `db68c00` glibc tuning → `c01b4cc` revert (37 min) | A wrong memory theory; the real cause was the forkserver preload never applied (`78d55e4`). |
| `9ebc90d` → `50678d1` → `8458ede` → `4f7ed78` → `daac154` (trade prior chain, 11-04 15:43 → 11-05 19:56) | Each pricing fix opened another route (standalone trades 4 → 13–23/game after `8458ede`). |
| `87e79b5` ARTIFACT → `9fe35c9` (16 min) cost → `2c56c92` (11-11) EQUIP loop | Smoke games handed artifacts back and forth 460–1,417 times; the root loop bug survived a week. |
| `de04cf8`, `93e39d2`, `5d7be3f` → `26bc2a1` revert (11-05 01:43) + rollback | Replacing the teacher's label taught ARMY/BUY/UPGRADE everywhere: MOVE 0.53 → 0.14, eval −0.129 ±0.041. |
| `e3f5503` GATHER accepted → `0eab9f0` danger cost → `69d0332` teacher drops it + rollback | Accept-set entries are taught; towns emptied (59% vs 24%). |
| `0d54e0f` span → `1463891` (35 min) → `444ebfc` (2 h) → `b69bb20` | Teacher span narrowed, widened back in two steps; disbands 6 → 14/game in between. |
| `daac154` → `51c08ce` (55 min) | A CPU list index broke CUDA graph capture; updates 880–1106 s vs 224–296 s. |
| `1608288` removed unifyStacks → `7192bbd` restored with a singles exception (~10 h) | 6% of turn decisions had duplicate stacks; value loss +~40%. |
| `e9f88da` privileged critic → `0fe2869` (2.5 h) user's redesign | Growth heads 1% worse with the first design. |
| `6ec628f` battle trunk β1 0.95 → `e78414f` (22 h) back to 0.9 | Idle shooters rose 1–2% → 5.15% in evals. |
| `8797125` accumulate 2 → `e577a76` (4.8 days) back to 24 | Battle value learned nothing for days; signal ratio 0.13 under pure noise's 0.155. |
| `36a38de` GATHER 0.1 → `e011809` off (9.5 h) + rollback v2728 → `90d153c` back ramped (11-11) | Second GATHER collapse went unseen for 87 updates and five evals (`fab5eeb`): led to the value-loss guard. |
| `115dd13`, `7f058b7` prior switches → `6372cfc` removed | Paired tests rejected both (−0.061 ±0.046; −0.001 ±0.030); measurements kept in commits. |
| `7441d1e` learnable night_gain → `6a73dc1` frozen (2.75 h after live) | The teacher, not reward, drove the scalar (21 of 23 updates). |
| `2bdcdcb`/`45b8692` valves → out in `e005793` (5.5 h) | Premise disproved 2 h after deploy [mem]. |
| teacher on → `d75e3d3` switch → off in deploy 57; follow 0.9 (57) → 0 (60) → 0.9 (62) → 0 (63) | Four reversals of the NK2 role in ~11 h on 11-11/12. |
| `af98963` → `a332c25` | nk_predict graph kept as a local; deploy 64 failed its gate. |
| Rollbacks of joint7 weights [mem]: 11-05 01:46 (v1025), 10:39 (v1125), 20:00 (v1184), 11-06 03:33 (v1288 file = v1293), 11-10 18:38 (v2728), 11-11 23:47 (v3205), 11-12 00:19 (v3229) | Seven rollbacks in eight days; `c4241a1` shows one restored the wrong weights; [mem] rollbacks make versions repeat, so spool/eval files get quarantined. |

Count: explicit reverts 3 (`fc9c277`, `c01b4cc`, `26bc2a1`); ~20 further commits switch off or undo an earlier change; ~15 "fix of a fix" pairs within hours.

---

## 6. Bugs named in commits

Durations are derived from commit times (introduction → fix commit); "≥" where the introduction is an inference.

| Bug | Fixed in | Live for | Impact recorded |
|---|---|---|---|
| Dead unit under a living one legal as spell target | 27e14e2 10-23 00:35 | ~10 h (since 905b16b) | rejected casts |
| Advantages normalised over both sides | 27e14e2 | ~1 day | defence stopped improving |
| Battle maps gave heroes their type's starting skills/spells | 5a415d6 10-23 01:05 | ~2.6 h (10cf6dc) | scenario not as specified (62 mana vs 50) |
| Enemy hero's battle stats all 0 | e044191 10-23 01:08 | ~10 h | model blind to enemy hero |
| VCMI test-map start-up race (vcmi d58daeaa1) | 2d4ad84 10-23 03:17 | upstream | ~1 in 1000 headless games crashed |
| Attack masks by side, wrong under hypnosis | ba0469c 10-23 08:03 | ~17 h | own units offered as targets |
| Chained final value head | 7e067b3 10-23 10:55 | ~13 h (a2d1900) | defence avoided exchanges (42% wait/defend) |
| Reward: summoned units worth 0, temporary resurrection kept, destroyed war machines at 2,500 | e4e13e6 10-23; 09f1a9b 10-24; 5604d8e 10-29 | ~1–6 days | reward wrong |
| MPS graph per shape | d6df406 10-24 13:28 | ~29 h (1940651) | 9+ GB in 30 epochs, swap |
| score_now in battle token (v11 regression) | a75640d 10-24 17:19 | ~1 h | defence killed ~20% vs 73% |
| MMAI modSettings race | 26b150c 10-25 23:25 | ~5.4 h (1235238) | MMAI silently played as BattleAI |
| Entropy bonus weighting | 0fc65e0 10-26 01:59 | — | shooters 90% → 55% |
| Models cached by link name | d5f7468 10-26 09:05 | ~16.5 h (99dcb5a) | ~half of each update far off-policy |
| League snapshot names v50 vs v00050 | 3263898 10-27 01:10 | ≥15 h | league opponents defended every unit |
| Endless battles killed campaign games | 0b4d641 10-29 16:27 | since joint2 v128 | 30–60% of campaign games killed |
| RMG towns lost default buildings (vcmi 47c9d57eb) | bf6bf9e 10-29 22:27 | ≥2 days (joint1 10-27) | no tavern/dwelling in 2/3 of training and all eval games |
| Hire blind to the hire's army | 552625f 10-29 23:05 | since campaign start | 0 hires/game → 12 after |
| Area-attack targets in pointer order (vcmi e1be37e4a) | d4f95bb 10-30 | upstream | ~1% of evals not replayable |
| select() fd > 1024; stride on 1-row batch; decode 2.6 GB/game; failure_counts 10 GB | e505aba, 7f9bda6, e7014c1, 682a14b 10-30 | first PC runs | PC run stops |
| Idle battle clients killed under a battle | 9e2bdb4 10-31 09:29 | exposed by 24ab45d | 115 of 143 joint5 battle failures in 5 h |
| Watchdog-killed eval games scored at early days | 6c5d2eb 10-31 10:31 | — | eval inflated 0.219 → 0.267; "flat since v174" |
| Junk games (servers down, games run with no decision) | 3c7990e, dd0bcd4 10-31 13:26/13:33 | ≥1 day | 6,321 of joint5's 10,206 games counted as finished |
| Attributed credit KeyError on an eliminated player | fadf8ae 10-31 14:02 | 5 min | stopped joint5 |
| Exchanges used all 32 troop moves | 949545b 11-01 00:12 | ~2 h | 55% of joint7 decisions |
| KL target vs teacher pull | 5dea773 11-01 01:00 | ~3 h | updates stopped after 2–6 of ~56 steps; LR 1e-5 → 3e-6 |
| WDDM GPU spill | 4306bc9 11-01 02:36 | — | v8 update >25 min vs ~7 |
| NK2 moveHeroToTile unseen tile (vcmi 823c80710) | 01743f0 11-01 03:40 | upstream | 4 of ~25 joint7 games crashed |
| Advice MOVE label = first tile | 5e72edc 11-01 05:33 | ~7.7 h (995fdfa) | taught walking a tile per decision |
| Permits deadlocks | d9e58fe, da4bd80, 61549d1, 3020025 (10-31 23:50 → 11-01 16:20) | hours each | no games played 15 min of each ~29 min cycle; 15:58–16:16 none |
| Eval cut by stop scored 0.000; 1200 s game limit | da6c46c 11-01 16:07 | — | teacher weight 16.4 → 19; 74 of ~1,170 games killed at 1200 s |
| TBB spinning workers | f2165ba 11-02 21:08 | upstream | ~60% of game CPU |
| Forkserver preload never applied | 78d55e4 11-03 11:35 | ~13 h (f398aec) | loaders 9.5 GB vs 2.9 GB; capped games in flight |
| Loss-noise TypeError on wire path | ee20e37 11-03 07:27 | ~25 min live | stopped joint7 at v391, games lost |
| Over-budget step graph pool | 0321bec 11-03 16:47 | ~10 h (489deaa) | +6 GB GPU reserved |
| 'behind' (blocked_by_guards) always 0; 4 MOVE features dead | ab52217 11-04 09:04 | ≥6 days (5604d8e) | 54× hopeless fights, 97% lost |
| Question reward: fake −760 when d_object is our hero | 3b1424d 11-04 17:38 | ~1 h | flee pursuit at −2.5 logits |
| Hashed option rows | d60df0b 11-04 19:21 | ~5 days (05914d0) | 546 of 954 options shared a row |
| Checkpoint written in place | 6b2719e 11-04 15:46 | since start | torn resume file possible |
| BUILD SHIP via hidden building 20 | f75ea35 11-05 13:15 | ~15 h (542fdbe) | refused boats (12 in a row) |
| DROP prior CPU tensor broke CUDA capture | 51c08ce 11-05 20:51 | 55 min | 3–4× slower updates, games held |
| Generated-battle seeds repeat after restarts | 25af47e 11-05 22:15 | ~5.5 days (24ab45d) | same dead-end battle 5× a day |
| Neutral towns never listed (getTownsInfo(false)) | 296b06a 11-05 22:45 | ≥7.8 days (5604d8e) | 17/17 hopeless attacks unseen; [mem] eval 0.391 → 0.476 |
| Fogged guard priced as free | 635912f 11-05 22:59 | — | 5 of 46 hopeless attacks |
| Eval checkpoint saved after the eval | c4241a1 11-06 03:31 | since per-eval checkpoints | a rollback restored v1293, not v1288 |
| Duplicate stacks after unifyStacks removal | 7192bbd 11-08 08:26 | ~8.6 h live | value loss +~40% |
| Battle neither side can retreat | affa756 11-10 00:41 | since 1207a12 (10-29) | game killed by watchdog |
| NK2 endTurn race (vcmi d7868e680) | 897db08 11-10 06:23 | ~12 days of NK2 evals | 13 eval games killed in 151 evals |
| teacher_set_agree never matched an end | 1d31d6b 11-10 06:40 | ~7.7 days (a8b587e) | reading low (32 of 145 end labels) |
| Battle 512-decision steps = noise walk | e577a76 11-10 08:50 | ~4.8 days (8797125) | value loss off its best by ~10%, entropy 1.01 → 0.76 |
| EQUIP swapped into an occupied slot; fingerprint by type | 2c56c92 11-11 18:34 | ~6.9 days (87e79b5) | 53 loops in ten days in one game |
| GARRISON_SWAP / dialog army missing from fingerprint | e78372e, 3719556 11-11 | ~3.8 days (1608288) | futile actions, 1 failed action per 48 games |
| nk_predict loss 314 at start; AUC ties; graph re-recorded | 74bc858, 254b9b2, a332c25 | minutes–1 h | deploy 63/64 gates failed |

---

## 7. Measurements recorded in commits

**Battle quality.** `693ba88`: V+V2 heads attack −0.09 vs −0.24, defence +0.10 vs +0.01. `bc341e6`: 238 sure kills in 80 battles all taken. `6b6228c`: StupidAI-as-model 838/865 decisions (865/865 given start sizes). `ecbf9d1`: 1,244 simulated actions, 0 mismatches outside dice; copy 3.4 ms, action 0.05 ms. `389a879`: untrained Lanchester prior explains 72% of reward variance (corr 0.86). `a75640d`: ablation, defence killed 73% vs ~20%. `dd8c6b8`: idling checkpoint 3319 beats 3078 by +0.015 ±0.008; weight average −0.0075 ±0.0025 over 5 paired evals. `846930c`: battle value CE ~0.685, learnable part ~0.006. `3ff87b5`: eval ~+0.09 for ~1,100 updates (joint5 5829–6963), set se ~0.03, signal ratio ~0.12. `b7e82e8`: at 3,072-decision steps KL 2e-5/update. `8797125`: median update KL 5e-5 over 1,090 updates. `e577a76`: value loss on one buffer 8385 0.00386 vs 8584/8693/8746 0.00483/0.00427/0.00413; policy gradient 3% signal at 512 decisions, ~24% at 6,144. `12256b3`: aux phase held-out value excess −1.6% to −3.7%. `e78414f`: wasted shots 1.4% → 2.1–2.4% after β1 0.95.

**Campaign evals (vs NK2, 48 games unless noted).** `bf6bf9e`: 24-game temperature-1 eval se ~0.07. `acd6999`: time-cut games scored 0.89 vs full 0.06. `6c5d2eb`: v229 0.267 (true 0.219); v174 0.242, v203 0.228. `567c7d5`: v203 0.228 vs v243 0.043 (19 setups). `26bc2a1`: v1037 0.248 vs v1024 0.378 (−0.129 ±0.041), 22/48 eliminated. `0eab9f0`: v1137 28/48 knocked out (19 at v1120). `69d0332`: v1157 0.274 vs v1125 0.390 (24 games). `53c1df7`: 0.441 → 0.363 → 0.335. `b0b4330`: 0.420, 0.405, 0.431 vs 0.434. `36a38de`: 0.511, 0.491, 0.489. `e011809`: 0.475, 0.380, 0.314, 0.345, 0.402, 0.271. `7aa44b9`: one 48-game eval resolves ~±0.1 at 2 se; halves of the same arms +0.15 and 0.00. `0830504`: 0.484 → 0.481 (−0.003 ±0.024). `85e6f2e`: 0.472 → 0.531 (+0.059 ±0.025); v2934 0.579. `e8649fb`: v2830 0.503 → 0.582 (+0.079 ±0.052), eliminated 55 → 37; v2912 −0.013 ±0.040. `6372cfc`: shares −0.061 ±0.046; night power −0.001 ±0.030. `07a4a98`: candidate prior β2 +0.017 ±0.058, β4 −0.013 ±0.058; playing NK2's choice outright +0.29 ±0.06. `73d01a5`: nk_label prior +0.05 ±0.06. [mem] deploy 57 v3120 0.753 ±0.128; own arm 0.602 → 0.739 → 0.802; one PC eval ±0.11, one Mac arm ±0.08 (v3229 0.861 then 0.697).

**Behaviour audits.** `a920331`: 224 suicide fights by near-armyless heroes, 97% lost, 54× median. `5805895`: 193 (from 275), 98% lost, 27× (from 44×). `9ebc90d`: 135 trades/game, 77% buy nothing, ~213k Φ lost/game. `3b1424d`: NO to 467/481 dwelling offers. `08c8106`: NO to first 150 quests/border guards; one hero asked 53×. `08627ff`: heroes led 5.7k and 8.9k of 72k/138k army at day 59. `8458ede`: 230/1,192 purchases traded away ~5,200 each. `de04cf8`: heroes led 2% of army at day 60 (39% day 14); P(ARMY) 0.0002. `93e39d2`: BUY legal in 4,219 town visits, chosen 0. `10bcddf`: 54.7% of MOVEs onto own heroes, 70.5% repeats, 25.7% of movement. `b49632e`: advised own-hero meetings 52.2% → 9.5%. `414c622`: unused movement 0.095 → 0.269; never-moving heroes 0.048 → 0.195. `75cfe7d`: 43% of hero-turns had reachable tiles outside the window. `63daa24`: 18 heroes of 3k+ gold killed on enemy turn, 16 after a TRAVEL. `7441d1e`: big night losses 49 vs 64 over 36 maps (−0.42 ±0.20/map). `7f058b7`: night reading vs loss — 0.0–0.2 lost 4.0%, 0.4–0.6 10.1%, 0.6–0.8 37.4%, 0.8–1.0 76.7%; in town 1/450; fit γ 3.18. `525d2b1`: towns lost 100 at follow 0.9 vs 23 at follow 0. `2b2cf74`: advice unmatched 2.5% → 0.8%. `2c56c92`: 0 failed actions, 168 loops cut over 12 games. `8519668`: loops 0.16 → 0.44 per player-day. `428baab`: day's followed share predicts 7-day share change, slope +0.0124 ±0.0023. `779bcff`: own-game advantage 0.20–0.60 sd vs 1.09–1.39 pooled. `41d12bf`: surprise sd 0.050 / 0.074 / 0.087 at 2/5/7 days.

**Value/credit diagnostics.** `96266d5`: gradient noise scales (decisions for noise = signal), days 1–14 / 15–35 / 36+: horizon credit 5,312 / none / none; 7-day share MC 2,979 / none / 11,031; growth 1,310 / 6,505 / none. `55f857b`: 96.5–98.1% of advantage variance shared within a player-day. `e620710`: privileged unexplained 0.863 → 0.806. `fab5eeb`: guard fires at v1137 (29× level) and v2749. `2bdcdcb`: value share of shared gradient 0.16 in collapse vs 0.05 calm.

**Throughput / systems.** `1940651` epoch 2.2–2.7 s vs ~6. `ab08f16` 73 → 134 decisions/s; `6bfea88` 194; `02e57cf` 205; `3075d90` 238. `7abefe0` battle step 526 → 405 ms; fused Adam 21 → 1.7 ms. `d9cb662` actor CPU 18.0 → 12.7 ms/decision (9.2 → 4.0 with GPU model); `cdf5e50` battle updates/h 126 → 156. `0239df9` 41.9 → 34.4 M instr/decision, client peak 794 → 501 MB; MPS swap-in 12.4 → 0.6 MB/s. `c7fce84` sort backward 95% of PC GPU step. `316d8cb` 2.3 s launches vs 60 ms GPU per micro-batch. `f7e6e55` ~1,000 launches = 45 s of a 76 s update. `9e2bdb4` battle updates ~40/h vs 90–158. `4306bc9` 31.2 GB + 7.9 GB spilled. `48765b4` trainer peak 7.4 GB (was 11–14). `f2165ba` 0.185 → 0.065 CPU s/turn. `489deaa` 2.8 heroes/decision, ≤4. `e79960a` decode 38.8 → 8.1 ms/batch. `2fb28e7` learner ~500 games/h vs slots ~300. `1a9fa63` 36×36 forward 312–318 ms both. `e11ab5f` update 7.31 vs 5.18 s (1.41). Test suite: 274 (`f0b5c75` 10-30) → 458 (`1608288` 11-07) → 469 (`1a9fa63`) → campaign 304 + battle 191 (`75cfe7d`) → campaign suite 398 / 407 (`c916412`, `41d12bf`); [mem] bundle-60 suite 603. Test files: 3 (`905b16b`) → 26 (`5604d8e`) → 43 (`1608288`) → 61 (`a332c25`).

---

## 8. VCMI engine patches

Fork branch `alpha_heroes` (tip `a58753d66`) plus local `overhaul` (`826909100`, checked out detached). 45 commits since 10-20: 2 upstream-authored headless fixes carried in (`6cc8314bf`, `e81e9e9db`, authored 2026-06-24/25, committed 10-21 22:19) and 43 project commits (37 Opus 5.5, 2 Fable trailers). Net project patch excluding CI removal: 51 files, +915 / −132 lines. 38 gitlink bumps in the main repo map one-to-one to these commits. Messages label each change "(bug fix)" or "(not a bug fix / no behaviour change)", per the minimal-engine-changes policy.

- **Harness/infra (no gameplay change):** `4506cd5a8`/`b16f7f41f` VCMI_USER_DIR (macOS, Linux); `a752e26f8`, `209c9a91d`, `a0d211c18`, `c88b51b85` external AI dirs and AI settings entries; `c0b606d3c` VCMI_ADVENTURE_AI_<COLOR>; `946412f20` battle-only games (per-player battle AI, clean end); `e07707607`, `fa11e3899` battle-only monsters, gold/mana from env; `9b8635c67` test queue (one client plays game after game); `be61b722a` CI/Dependabot removed.
- **Search support:** `b5a284df6` battle rules behind IBattleServer/IBattleOwner (24 battles byte-identical); `695638ba0` cloneWithBattle; `0d0cb2725` IBattleRandomizer + restoreBattle; `4784e11b0` export UnitEffect.
- **Bug fixes (13):** `d58daeaa1` test-map start-up race; `43fe4c2d2` random starting bonus in battle-only; `b54a3b979` level-up hang; `de4f152f8` battle ended before start flag; `1384db446` NK2 HeroPtr dangling (SIGSEGV); `d8c48ba91` unreachable hex aborted the game; `400b76421` NK2 planning without state lock (SIGSEGV); `8bec03755` test map without red never ended; `65cbea04f` red's elimination ended 3–4 player games; `47c9d57eb` RMG towns lost default buildings in saving; `e1be37e4a` area-attack targets in memory-address order; `823c80710` NK2 moveHeroToTile on unseen tile; `d7868e680` NK2 endTurn ended its next turn (deadlock).
- **Determinism:** `8f060394f` BattleAI's random choices seeded by battle state (198/200 eval battles identical on rerun); `e1be37e4a` above.
- **Rule additions (labelled not bug fixes):** `d1a5f29bb` round-60 limit → `b4d563fbe` 50 quiet rounds → attacker retreats → `a9b7b0a94` when neither can retreat the poorer army loses. Reason: VCMI has no round limit and battles ran until the watchdog.
- **Performance:** `3102490f2` canBeCastAtIfCastable (−18% battle-client instructions); `30658acd5` VCMI_MAX_WORKERS caps TBB (~60% of game CPU spinning).
- **NK2 hooks for the teacher/advisor (no behaviour change unless set):** `dcedb18b1` keepPlan/lastPlan; `04b7246e3` adviceFixes, exchangeSettled; `a58753d66` lastCandidates, exports for schema 19; `826909100` executingTask, adviceHeld.
- **Human recorder:** `3daf27f02` recorder beside a human's interface (ALPHA_RECORD_HUMAN); `49b9b25d8`, `e72370e77` auto-combat handling.

Most AI-side logic (Advice.cpp, Match.cpp, Teacher.cpp, Recorder.cpp) lives in `agents/AlphaMapAI` outside the engine; `995fdfa` explicitly notes "No engine change" for the advisor, which works through NK2's own AIGateway.

---

## 9. Observations for the paper

1. **Commit messages were the lab notebook.** Late messages routinely hold: the measured problem (counts from N games), the user's decision quoted with date, the design, before/after numbers on recorded decisions, a "what takes it out" rule, and the deploy line. `96266d5` wrote falsifiable predictions plus a revert threshold; `6372cfc` deletes rejected code but points to the commits holding its measurements. Median message length nearly doubled over three weeks. Risk: side-branch hashes are cited from main commits (`80793cf` cites `f032d3f`, `74480f8`; `6372cfc` cites `25866df`, `090d525`), and memory notes cite pre-rebase hashes (`1527c22` for `1a9fa63`); a reader must map them by subject.
2. **Cadence is bursty and around the clock.** 466 commits in 23 calendar days; 0 on 10-28 while the campaign AI was built off-history (one 10-29 03:06 commit carries all of it); peaks of 57 (10-31) and 64 (11-11). 15% of commits land 00:00–06:00.
3. **Changes were reversed often.** 3 explicit reverts, ~20 switch-offs/undos, ~15 same-day fix-of-fix pairs, 7 weight rollbacks of joint7 in eight days, and 4 reversals of NK2's role in ~11 hours on 11-11/12. Several early decisions rested on single noisy evals; the project only measured, on 11-10, that one 48-game eval resolves about ±0.1 (`7aa44b9`), after which paired tests (2–3 replicates) preceded behaviour deploys.
4. **Teacher signals were the main source of collapses.** Both GATHER collapses (11-05 `e3f5503`/`69d0332`; 11-10 `36a38de`/`e011809`) and the visit-advice collapse (`26bc2a1`) came from changing what the NK2 teacher's label or accept set taught. Lesson recorded in commits: an accept-set entry is taught (`69d0332`); a new action shifts the teacher's labels too.
5. **Deploy safety was learned by failure.** Repeated rule "a diagnostic never ends the run" (`fb9c4b2`, `6d54195`, `e2e794f`, `3e43632`, `5ee4d81`, `fadf8ae`); CUDA-graph capture broken by innocuous code (`51c08ce`; deploys 57, 60, 64 first tries [mem]); self-refusing gates that grepped the wrong thing (deploys 48, 49 [mem]); no data lost on stop (`6f76d48`); atomic checkpoints only on 11-04 (`6b2719e`); the value-loss guard (`fab5eeb`) built after a collapse ran unseen for 87 updates. From 11-09 nearly every change ships behind a switch defaulting to "bit for bit as before" (18 messages say "bit for bit" or "bit-identical"), verified against golden outputs.
6. **Infrastructure dominated early effort.** Of the 10-30 → 11-03 commits, a large share concern memory, permits, CUDA launches, WSL/WDDM behaviour and worker plumbing; the PC's update was launch-bound (2.3 s launches vs 60 ms GPU work) until whole-step CUDA graphs (`f7e6e55`).
7. **Bugs with long silent lives were representation bugs.** Neutral towns invisible (~7.8 days), 'behind' always 0 (~6 days), hashed option collisions (~5 days), EQUIP loop (~6.9 days), battle noise walk (~4.8 days). They were found by audits that counted behaviour on recorded games, not by evals.
8. **Engine changes stayed small** (43 commits, net +915 lines), each labelled bug fix or not; game-rule additions limited to battle-ending rules; NK2 instrumented through hooks rather than modified play.
9. **Test growth tracked the shift to switches:** test files 3 → 61; suite 274 → 469 (11-08), campaign suite 407 (11-11).
10. **Model division of labour:** Opus 5.5 wrote the science and most code; Fable 5.2's 32 commits (11-09 → 11-11) are the e5 ops/review session's run lines and safety guards; Fable 5.1's two include one suggestion reversed within 7 minutes (`1f7db1c`).
11. **Unpushed history:** `origin/main` stops at 11-05 22:15; 134 later commits, including all numbered deploys, live only on local branches/worktrees.
