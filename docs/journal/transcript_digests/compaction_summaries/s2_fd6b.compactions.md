

######## COMPACTION SUMMARY at 2026-10-27 02:04
This session is being continued from a previous conversation that ran out of context. The summary below covers the earlier portion of the conversation.

<system-reminder>
<tools-used-before-summary>Bash, Edit, Read, Write</tools-used-before-summary>
</system-reminder>
Summary:
1. Primary Request and Intent:
   The user (effort set to max for this session) asked, verbatim: "you are to do an adversarial review for all the code changes we have done in this project, note that another process is running. You are to find actual bugs, issues that are a result of not understanding the game (last last one we fixed is the idle shooter metric the measured waiting for enemies to get in range as a bad thing), as well as places where we are working suboptimally in terms of either science or engineering, not best practices but actual inefficiencies where fixing them would let us make better models faster. You may also find issues where our scenario generator is not good enough, but only if they are critical. List them all in a file/report/whatever so we can fix them later."
   - Constraint: another process is running (the live training run `v15-window`, pid 70338, `.venv/bin/python -m battle_model.train --run v15-window --resume ...`), so I must not interfere with it (no killing processes, no heavy CPU/GPU work, no edits to code it uses).
   - The task was to find and list findings, not to fix them.
   - Standing rules from CLAUDE.md / memory: push back plainly, be concise, and lead with the answer. AWS: every new resource in us-west-2 / us-west-2a only, never us-east-1. Data-generation runs must be capped and reported, and nothing left running. LR never above 1e-4. The user reviews the Python, not the C++. Minimal engine changes. No GitHub Actions. Commit only when asked (the user did not ask).

2. Key Technical Concepts:
   - Heroes 3 battle AI on VCMI 1.7.5 (fork with patches), C++ `AlphaBattleAI` (encoder, masks, sampler, history, records sent over a Unix socket) and a Python `battle_model/` (PyTorch model v3, features, PPO).
   - Actor/learner training (`actors.py`): 10 generated actors running persistent VCMI processes (`BattleProcess` / TestQueue), 5 real games (Nullkiller2), a central inference server (`inference.py`, CPU processes, live weights), a buffer with `max_record_age` 3, newest-first selection, and a league of 85% self-play and 15% past snapshots.
   - PPO with per-side advantage normalisation, example weights, KL control (`target_kl` 0.015, early stop at 1.5× on the last minibatch's k1 KL, `lr_scale` ÷1.5 / ×1.5, `lr_scale_max` 1.0), and "window" credit (85% [V two rounds later − V now] + 15% [result − V now]; new commit fc65a2e makes the outcome share adaptive by value EV).
   - Zero-sum reward over assets (army + hero value + mana). Value heads V0..V2 (round scores) and V (final), tanh with BCE.
   - Eval: greedy ONNX export vs BattleAI and MMAI on generated, neutrals, hard, real (`real_eval.json`) and castle sets, with per-set vs-baseline scores. New commit 6c97ef5 adds `score_balanced` / `score_worst`.
   - VCMI mechanics checked:
     - damage = average of min(10, count) uniform rolls (`BattleInfo::getActualDamage`);
     - casualties exclude `summoned` units (clones are summoned);
     - paralyze and stoneGaze give `NO_RETALIATION`;
     - `battleNewRoundFirst` fires before the round increments;
     - `BattleSide::NONE = -1`;
     - gameConfig damage, morale and luck constants match `rules.py`.

3. Files and Code Sections:
   - Read for the review: `battle_model/{train.py, data.py, losses.py, reward.py, distributions.py, schema.py, constants.py, model.py, features.py, rules.py, actors.py, inference.py, play.py, generator.py, scenarios/generator.toml, real_games.py, layers.py, priors.py, export.py, noise_ceiling.py, value_benchmark.py}`, `agents/AlphaBattleAI/{AlphaBattleAI.cpp/.h, Encoder.cpp, History.h/.cpp, Reward.h, Policy.cpp, AttackValues.cpp, InferenceClient.cpp}`, VCMI sources (`BattleResultProcessor.cpp`, `CUnitState.cpp`, `BattleInfo.cpp`, `config/gameConfig.json`, `config/spells/ability.json`, `BattleSide.h`) and the engine patch list (`git log 1.7.5..HEAD` in `vcmi/`).
   - Key code behind the top findings:
     - `actors.py:61-62`: `self.server.set_model(path.stem.replace("snapshot_", "v"), ...)` gives "v00050". But `actors.py:85` uses `set_model(f"v{self.version}", ...)` and `actors.py:105` `past_versions()` returns `f"v{int(...)}"`, i.e. "v50".
     - `AlphaBattleAI.cpp:175`: "cannot use model ... Every unit defends."; `:510` `if(!model) playDefault(...)`.
     - `train.py` (~line 858): `if config.target_kl and float(parts["approx_kl"]) > 1.5 * config.target_kl: stopped = True`; then `lr_scale = max(0.1, lr_scale / 1.5)` or `min(lr_scale_max, lr_scale * 1.5)`.
     - `actors.py:286`: `sorted(..., key=lambda b: -b["version"])` (newest first); `:253` `dropped_stale` is counted but never logged; `:263-265` overflow drops are not counted.
     - `train.py:465`: `counts["shooter_not_shooting"][0] += kind in (ActionType.MOVE, ActionType.WAIT, ActionType.DEFEND)`.
     - `features.py:477`: `wipe = fraction(high - hp + 1, high - low + 1)`.
     - `data.py:325`: collate `.astype(np.int64)`.
     - `noise_ceiling.py:28-30`: `prefix_before` includes round-0 tactics actions, while `placeNextTactic` (`AlphaBattleAI.cpp:418`) ignores the prefix.
     - `value_benchmark.py:67`: `"separate": copy.deepcopy(base)`.
     - `features.py:345`: `standable` uses only `x_access == ACCESSIBLE`.
     - `data.py:148`: `e["round"] >= 1`.
     - `AlphaBattleAI.cpp:843+`: `scenarioJson` exports no mana or gold.
   - CREATED `/Users/alon/projects/alpha_heroes/docs/adversarial_review_2026-10-27.md` (untracked, not committed). It contains a summary table and 15 detailed findings, plus a "Checked and fine" section.
   - CREATED memory `/Users/alon/.claude_personal/projects/-Users-alon-projects-alpha-heroes/memory/review-2026-10-27.md` (project type: top findings, pointer to the report) and added to MEMORY.md: `- [Review 2026-10-27](review-2026-10-27.md) — adversarial review: league-name bug (critical), KL anchor, actor overproduction; full list in docs/`.

4. Errors and fixes:
   - **Shell glob:** a zsh `grep --include=*.cpp` failed with "no matches found". I switched to explicit paths.
   - **Code moved during the review:** train.py line numbers shifted because two new commits landed from another session (fc65a2e, 6c97ef5). I reviewed both and used approximate line numbers ("now line ~858").
   - **Dropped suspicion:** I suspected the retaliation feature ignored paralyzed and petrified units. VCMI gives them `NO_RETALIATION`, so `u_retaliations` = 0 and the feature is fine. I dropped it from the report and listed it under "checked and fine".
   - **Dropped suspicion:** I suspected eval baseline key collisions from real_eval seeds. The seeds are unique, so I dropped it.
   - There was no user feedback on errors during this session.

5. Problem Solving (findings with evidence, as written in the report):
   1. **CRITICAL: league snapshot name mismatch after any resume.** The past-version opponent loads no model and every unit DEFENDs. Evidence from the logs: `no model named 'v250' (the server has ['current', 'v00000', 'v00050', ...`. 23 of 182 battles (12.6%) since the r6 resume were affected, and opponent summaries read `decisions=0 fallbacks=22..26`. Our side is trained on these battles. v14-11h-r2 was also affected. v0 (the untrained model) is in the league. Fix: name snapshots `v{int}`, make a model-load failure on any side fail the battle, drop v0.
   2. **HIGH: KL control is anchored to the stale behaviour policy.**
      - 24% of updates early-stopped; 40 did exactly 1 step; 77 did ≤3 steps, with median KL 0.023.
      - 54 of those 77 followed a full 24-step update.
      - The result is an LR sawtooth.
      - Fix: compute anchor log-probs at update start, use the k3 estimator, optionally decoupled PPO.
   3. **HIGH: actors overproduce.**
      - `wait_seconds` median 0; the buffer holds a median ~3,217 decisions after each take (max 15,984 of the 16,000 cap); `record_age` 1.05.
      - Drops are not logged. The learner is CPU-starved (memory notes: 9.4 s an update alone vs 13-19 s in a run; `train_seconds` median 17.5 s).
      - Newest-first selection skews to short battles: about 2.5% of decisions come from battles over 10 s of wall time.
      - Fix: backpressure, FIFO, log drops, retune the actor count.
   4. **Medium: `shooter_not_shooting` counts WAIT.** Wait-then-shoot is a legit tactic. `full_shot` also ignores hero no-distance-penalty artifacts and the wall penalty.
   5. **Medium: `wipe_chance` assumes uniform damage.** Example: with health at 70% of the range the feature says 0.30, the true chance is ~1.4%.
   6. **Medium: dense spell×hex tensors.** They are ~55% of ~400 KB per decision; an update moves ~2 GB as int64, plus ~1.3 GB of `spell_hex` facts (estimate, not profiled).
   7. **Medium: `noise_ceiling.py`.** It crashes on battles with a tactics phase, continuations share the dice stream, and V is taken after the cut.
   8. **Medium: `value_benchmark.py`.** The "separate" value net starts from the shared trunk, and the LR is 2e-4.
   9. **Medium: balanced and worst-set scores chase noise.** Castle is only 8 battles: its score sd between evals is 0.18 (vs BattleAI) and 0.11 (vs MMAI), against 0.008-0.036 for the large sets. `score_real` barely discriminates.
   10. **Low-Med: siege features ignore moat and quicksand stops and the gate.**
   11. **Low-Med: summoned units and clones are worth 0 in features and engine values.**
   12. **Low: evals are slow and unverified.** About 11% of wall time; fresh VCMI and ONNX session per battle, 8 at a time; the eval ONNX is never checked against torch.
   13. **Low: `real_eval` scenarios start with full mana and default gold.**
   14. **Low: the adaptive outcome-share EV gauge** is inflated by lopsided battles.
   15. **Low, small items:** V0 label in the tactics phase, surrender asymmetry in self-play, uniform league sampling.
   - **Checked and fine:** round numbering, casualty accounting (including Pit Lord demons), combat constants, retaliation, eval seeds, sampler and log-prob consistency, draws, and zero mid-battle model errors in 31,634 battle summaries on disk.
   - **Generator:** no critical issues found.

6. All user messages:
   - "you are to do an adversarial review for all the code changes we have done in this project, note that another process is running. You are to find actual bugs, issues that are a result of not understanding the game (last last one we fixed is the idle shooter metric the measured waiting for enemies to get in range as a bad thing), as well as places where we are working suboptimally in terms of either science or engineering, not best practices but actual inefficiencies where fixing them would let us make better models faster. You may also find issues where our scenario generator is not good enough, but only if they are critical. List them all in a file/report/whatever so we can fix them later."
   - (Before that, the user ran a local `/effort` command, setting effort to max for this session.)
   - (System reminders periodically asked for brief status updates, which I gave. They were not user messages.)

7. Pending Tasks:
   - The review and report are complete. The only remaining step is a final response to the user: the most important findings, the absolute path of the report, a note that nothing was fixed or committed and that the live run was untouched, a recommendation about the currently affected v15-window run (the user's call), and a one-line offer to publish the report as a page.
   - No fixes were requested. They should be done only after the user says so.

8. Current Work:
   I had just finished writing `/Users/alon/projects/alpha_heroes/docs/adversarial_review_2026-10-27.md` and the memory entry (`review-2026-10-27.md` plus its MEMORY.md line). The last verified item was the league-name bug in the live run ("23 of 182 battles (12.6%)"). The final summary response to the user had not been sent yet.

9. Optional Next Step:
   Send the user the final concise response (text only), leading with the answer:
   - The report is at `/Users/alon/projects/alpha_heroes/docs/adversarial_review_2026-10-27.md`.
   - Top finding: the critical league bug. Since the last resume, about 12.6% of generated battles in the running v15-window are against an opponent whose units only DEFEND, and those battles are trained on. The fix is to register snapshots as `f"v{int(...)}"` in `actors.py:61-62`. The user should decide whether to restart v15-window after fixing.
   - Then the two high items: the KL anchor and actor overproduction.
   - Then the game-logic items: the WAIT metric and `wipe_chance`.
   - State that nothing was changed in code or committed, the running process was not touched, and offer the report as a shareable page in one line.
   - Do not start fixing anything unless the user asks ("List them all in a file/report/whatever so we can fix them later").

If you need specific details from before compaction (like exact code snippets, error messages, or content you generated), read the full transcript at: /Users/alon/.claude_personal/projects/-Users-alon-projects-alpha-heroes/fd6b4705-c93e-4733-95c8-5df67a89337c.jsonl
Continue the conversation from where it left off without asking the user any further questions. Resume directly — do not acknowledge the summary, do not recap what was happening, do not preface with "I'll continue" or similar. Pick up the last task as if the break never happened.

The messages after this summary are the most recent messages from before compaction, kept verbatim. The summary was written without seeing them, so something it says has not happened yet may already have happened in them.
