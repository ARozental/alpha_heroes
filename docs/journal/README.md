# Alpha Heroes: project journal, 2026-10-21 to 2026-11-12

This journal tells what happened in the first three weeks of building an AI for Heroes of Might and Magic III. It covers what worked and what did not, what the AI developers (Claude Code sessions) got wrong, and which lessons should carry over.

It was assembled on 2026-11-12 from every recorded conversation with an AI about this project, plus the git history, the project's docs and its memory notes. The goal is a primary source for two readers:

- **An AI or person writing a scientific paper** for people who want to build AIs for Heroes-like strategy games, possibly on a small budget.
- **The developer**, when they want to ask about past successes, failures and missed problems, or pull out lessons for the next phase.

---

## 0. How to use this journal

### 0.1 What is in this folder

| Path | What it is | When to read it |
|---|---|---|
| `README.md` (this file) | The synthesis: the story by phase, what worked, what failed, the catalog of AI errors, the developer's role, multi-session collaboration, general lessons, the game knowledge built into the networks, open problems, and a paper scaffold | Start here |
| `periods/J01…J16_*.md` | 16 detailed period journals, one per chunk of conversation, in time order. Parallel sessions overlap in time. Each has the same 11 sections: coverage, summary, dated log, experiments table, bugs, developer interventions (with verbatim quotes), AI-to-AI coordination, design decisions, lessons, numbers, open threads | For detail, exact quotes, timestamps and numbers |
| `sources/git_ledger.md` | All 466 commits read: phases, a deploy ledger (joint1–joint7, deploys 42–67), schema/contract versions, reverts and rollbacks, ~55 named bugs with how long each was live, VCMI patches, branches | Exact hashes and dates; how long bugs went unnoticed |
| `sources/docs_and_decks.md` | Catalog of the 17 docs, design rationale, plans vs reality, reviews, a condensed version of the developer's 62-slide deck "From AlphaGo to Heroes III", the developer's paper notes and the session-rules draft quoted in full, and 18 candidate figures with where their data lives | Design rationale; the developer's own framing |
| `sources/memory_catalog_A_older.md`, `sources/memory_catalog_B_newer.md` | Digest of all 79 memory notes the AI sessions wrote. They contain standing rules (~80 in total), incident tables, measurement tables, and the project state on 11-12 | Rules and the official word on each topic |
| `expert_knowledge/` | Catalogue of the game knowledge built into the two networks as deployed on 11-12: 607 items with file:line in three code catalogues (campaign inputs, campaign heads and targets, battle), and a history file with what each piece did. Each code catalogue lists the quantities counted several ways, and its surprises | What the models are told rather than learn, and how each quantity is counted. §10 summarizes it |
| `stats/collaboration_stats.md` | Numbers on how the work was done: sessions, prompts, AI turns, tool calls, tokens, models, per-day activity, autonomous overnight stretches, code size | The methods section of a paper |
| `transcript_digests/` | The primary source: condensed, time-stamped transcripts of every session (5.2M characters). They keep the developer's words in full, the AI's prose in full, and the messages between sessions. Tool output is dropped. Includes the 68 compaction summaries | To verify any claim, or to grep for something this journal does not cover |
| `tools/` | The scripts that made the digests and the stats | To regenerate or extend the digests |

### 0.2 Conventions

- **Times** are America/Chicago (CDT until 2026-11-01 02:00, CST after), matching git. Dates without a year are 2026.
- **"The developer"** is the one human on the project. **"Dev AIs"** are the Claude Code sessions that wrote almost all the code and ran the experiments. **"The model" or "our agent"** is the Heroes-playing neural network being trained.
- **Session names:**
  - `cd`: the founding session, a62ff69f, Oct 21–29.
  - `b1`: the main science session, 68ab2bd0, Oct 27 → now.
  - `e5`: the performance, infrastructure, review and later deploy session, 31529395, Oct 29 → now. It was renamed `88` on Nov 9.
  - `fork`: a copy of b1 made on Nov 9 for presentation work.
- **Ids:** commit hashes are short git hashes. "deploy N" is a numbered live change to the long training run on the PC. "schema N" (campaign) and "contract vN" (battle) are input/output format versions. "v1234" is a campaign checkpoint version, and four-digit battle numbers are battle epochs.
- **Confidence:** claims are labelled high, medium or low where it matters. "Claimed" means a dev AI said it. "Measured" means it rests on a reported measurement. The digests keep the AI's prose but not tool output, so any number here can be checked against the transcript text, and through it against the raw logs.

### 0.3 How this journal was made, and its limits

1. **Gathering.** All Claude Code transcripts for the project were found, about 530 MB of JSONL:
   - five sessions plus one subagent under `~/.claude_personal/projects/-Users-alon-projects-alpha-heroes/`;
   - one home-directory session under `~/.claude/projects/-Users-alon/`;
   - one small Cursor-agent exchange.
   The planning chat on claude.ai that started the project was pasted into the first session, so it is covered there. No Claude session ran on the PC: the PC was driven over SSH from the Mac sessions.
2. **Condensing.** The transcripts were cut down to chronological digests (`transcript_digests/`), split into 16 chunks of about 300k characters, and each chunk was read in full by its own Claude Opus 5.5 reader agent, which wrote one period journal. Four more readers covered the git history, the docs and decks, and the memory notes. I read all of these outputs, cross-checked them against each other and against the developer's own messages (a separate pass over all 612 of them), and wrote this synthesis. The expert-knowledge catalogue (§10) was made later on 11-12 by four more readers working from the code itself.
3. **Limits.**
   - The period journals and this synthesis are AI-written. Treat specific claims as strong leads and verify the key ones in the digests or git before citing them.
   - Tool output and the models' private reasoning were not read, except where a reader grepped the raw logs to fill a gap.
   - Numbers are mostly as the dev AIs reported them.
   - Training curves for the main run (joint7) live only on the PC and were not pulled.
   - A process in this environment rewrote some written markdown into plainer wording as it was saved. Spot checks show the facts kept, but one reader saw a sentence it could not trace. This is another reason to check quotes against the digests.

---

## 1. Summary

**What was attempted.**
- One developer, with Claude Code sessions doing almost all the coding and experimenting, set out to train a superhuman Heroes of Might and Magic III AI.
- The game runs on the open-source VCMI engine with the original game data.
- Hardware: a MacBook Pro M4 and one RTX 5090 Windows PC, reached over SSH and WSL2.
- In 22 days the project produced:
  - a tactical **battle model** (~430k weights) that plays every battle;
  - an adventure-map **campaign model** (~3.3M weights) that makes every map decision;
  - a C++ plugin layer for VCMI and a Python/PyTorch training system;
  - a continuously running joint training run (joint7, Oct 31 → now), changed live through 65+ deploys.

**Where it stands (2026-11-12 morning).**
- **Battle model:** it beats VCMI's default battle AI (BattleAI) and an outside ML battle AI (MMAI) by a small, stable margin, about +0.09 on a −1..1 score in the same seats. It has been flat since about Oct 30, despite many recipe changes.
- **Campaign model:** it plays full 2–4 player games on small random maps against Nullkiller2 (NK2), VCMI's strongest scripted AI. The eval scores share-of-empire at day 56, scaled so 1.0 = parity with an average player:
  - an imitation-only start: 0.11;
  - most of Nov 1–10: about 0.30–0.50;
  - after the Nov 11 overhaul, which removed NK2 as a teacher but kept its planner outputs as inputs: own play reads about 0.70–0.80 on the Mac;
  - in plain terms, still below parity, and below NK2 itself.
- **Variance:** one 48-game eval varies about ±0.1 run to run. Effects under about 0.05 cannot be seen without hundreds of games.

**What mattered most (detailed in §4–§8).**

1. **Bugs and representation gaps beat tuning by a wide margin.** The biggest jumps came from finding things the model could not see or do. Examples:
   - every random map had towns without taverns or dwellings;
   - neutral towns were never encoded (+0.08 eval at most, and it was live for ~8 days);
   - one shared "YES" token answered every yes/no question;
   - teacher labels said "walk one tile";
   - the teacher's own bug produced 55% of moves as pointless hero meetings.
   These were found by audits and by the developer's pointed questions, not by tuning.
2. **The learning signal is tiny.**
   - A battle update is about 0.5% signal.
   - 96.5% of a campaign decision's credit is luck shared by every decision that player made that day.
   - At the learning rates stability required, new zero-initialized features barely move.
   - What worked under these conditions: priors with bounded learnable corrections, value-neutral rewards (Φ), multi-horizon value heads, and fixing representations. Raw RL capacity did not help.
3. **Evaluation noise was the most frequent cause of wrong decisions.** Single evals were read as trends again and again: "best so far", "slides", a 0.861 that replayed at 0.697. The fixes were pinned maps, paired arms, replicates and decision rules set in advance. They arrived only on Nov 10–12.
4. **Learning from a scripted AI (NK2) was powerful and fragile.**
   - Its labels on our own states lifted the eval from 0.11 to about 0.32 in a day.
   - Most of the policy collapses came from changes to what the teacher taught: both GATHER collapses, the advice collapse and the schema-16 regression.
   - Following NK2's choices inflated the headline score while starving our own policy's learning.
   - The teacher was removed on Nov 11. Its knowledge was kept as inputs.
5. **The developer's highest-leverage contributions were standards and questions, not code.** Examples:
   - "an explanation must have predicted the failure";
   - "never mask actions";
   - "no training against eval AIs";
   - "something is really off, and not a minor thing";
   - "is anything training now?"
   These repeatedly exposed what the dev AIs missed. The dev AIs' recurring errors were:
   - narratives instead of predictive diagnoses;
   - reading noise as signal;
   - stating untested claims as fact;
   - deferring clear decisions ("your call");
   - guessing fixes instead of measuring;
   - optimizing a headline number over the stated goal.
6. **Several parallel AI sessions worked well with clear roles.** One science/build session (b1) worked with one review/deploy/director session (e5). They caught many of each other's errors. They also created new failure types: relayed instructions changed in transit, late failure reports, overwritten shared notes, reversals announced instead of asked.
7. **Most of the game knowledge sits outside the raw inputs, and the same quantities are counted many ways (§10).** A catalogue of the deployed code finds 607 pieces of Heroes knowledge: a quarter computed in C++ before the network, two in five computed in the graph, one in eight from NK2. Value, strength, time and danger are each counted 6 to 12 ways. The deliberate multiple counts came with large gains, but copies that drifted apart caused most of the representation bugs. Four bugs were verified and are still open, among them a battle escape prior lifted 5× too far since Oct 25.

---

## 2. The project at a glance

### 2.1 Goal and constraints

- **Goal:** set in a claude.ai planning chat pasted in on Oct 21. "Build a superhuman Heroes 3 AI", on VCMI with the original GOG game data. The chat's plan:
  - imitation from Nullkiller (VCMI's adventure AI);
  - entity-token state encoding;
  - a bootstrapped value function;
  - battles possibly resolved by a learned predictor;
  - reuse of the vcmi-gym/MMAI battle work.

  The first task it planned, a headless NK-vs-NK speed benchmark, was never run. The developer steered straight to a battle AI.
- **Constraints the developer set early and kept:**
  - Everything lives in the repo; the VCMI fork stays private.
  - The finished AI must plug into unmodified VCMI and see only what a human player sees: "no cheating".
  - Training is self-play only. Training against the AIs used for evaluation counts as cheating.
  - All behaviour lives in the PyTorch graph. C++ passes raw features and executes ("Python owns the science").
  - Inputs are raw values; every transform is an op inside the network: "no feature engineering before the network".
  - Engine changes stay minimal and labelled; real bug fixes are welcome.
  - Never mask an action to stop overuse.
  - Stability over speed: learning rate at most 1e-4 (Oct 26), later at most 3e-5 with 1e-5 recommended (Oct 31).

### 2.2 Budget and hardware

| Item | Detail |
|---|---|
| Machines | MacBook Pro M4 Pro: 14 cores, 24 GB unified memory, PyTorch MPS. PC: RTX 5090 32 GB, Ryzen 9 9900X 12c/24t, 31 GB RAM, Windows 11 + WSL2 (27–28 GB), driven over SSH from the Mac. Until about Oct 30 the PC GPU was shared with the developer's other ML project |
| Wall time | 22 days (Oct 21 → Nov 12). Work ran around the clock, often in 9–20 hour autonomous overnight stretches |
| Human effort | 612 typed prompts. Most were design decisions, corrections and questions; very little code was written by hand |
| AI effort | ~20,500 model turns and ~20,000 tool calls across 5 Claude Code sessions. Mostly Claude Opus 5.5; e5 also ran on Claude Fable 5.1 (Oct 30 → Nov 9) and Claude Fable 5.2 (Nov 9 on). 26M output tokens and about 9.4B cached input tokens. Cost depends on the subscription; API list prices on these volumes would be substantial (see `stats/`) |
| Output | 466 commits. ~28.6k lines of Python, ~13.2k lines of tests (585 tests), ~14.3k lines of C++ plugins, 43 VCMI patch commits (net +915/−132 lines), 17 docs, 79 memory notes |
| For contrast | The developer's deck sets this against AlphaStar and OpenAI Five, on hundreds of TPUs or GPUs for months, with the slide title "One GPU, not 384 TPUs". MMAI, the outside ML battle AI used as a reference, reported about 5 days and 2.5M battles of training (about $45) |

### 2.3 The system as of 2026-11-12

**Engine integration**
- VCMI 1.7.5 is forked into the repo with a private user directory, headless runs and a few labelled patches:
  - 13 bug fixes, for example a start-up race, NK2 crashes and deadlocks, a random-map-generator bug that stripped towns, and a bug where area-attack targets were processed in memory-address order;
  - determinism fixes (BattleAI seeded from the battle state);
  - battle-ending rules: after 50 rounds without damage the attacker retreats, and if neither side can flee the poorer army loses.
- C++ plugins:
  - `AlphaBattleAI` plays battles;
  - `AlphaMapAI` plays the map;
  - `AlphaTeacherAI` and `Advice.cpp` give a private NK2 copy's advice in dry-run mode;
  - a human-game recorder.
- The C++ side serializes raw features and legality masks against a versioned, hashed tensor contract, with code generated from Python constants. It sends them to a central PyTorch inference server, which batches requests from many games.

**Battle model** (about 430k weights)
- Inputs: attention over unit, hero, hex and global tokens, built from raw engine values. These include the hero's in-battle stats and a 16-event history (contract v15).
- Engine-exact ops inside the graph: the damage formula, kill and "wipe" chances, reach, threat. Each feature lands on its own action's logit, and its weight is learned from zero.
- Outputs: structured, conditional action heads, plus value heads for the end of this round, the next round, and the end of the battle.
- Reward: terminal army-value share. Hero value is 2000·min(L,3) per level-up; war machines count at half price; an enemy surrender counts as destroying the army.
- Credit: a linear window over battle rounds (85% ΔV two rounds ahead plus 15% outcome; later a fixed 50:50). This was the developer's design and it replaced GAE.
- Training: PPO self-play against current and past snapshots.

**Campaign model** (about 3.3M weights, v2 from Oct 30)
- Encoders: typed raw-input encoders for heroes, towns, objects and tiles. Ownership is a learned transform, not a sign flip.
- Map: a whole-map U-Net, plus a tile reader with reach as an attention bias. A 36×36 block grid lets any map size run, with small maps bit-identical.
- Late-joined "known" features: engine prices and NK2-derived priors as bounded, learnable corrections.
- Memory: 8 slots × 64.
- Auxiliary "foresight" heads whose hidden states feed forward.
- Logits are matches: actor·target query/key plus a small net plus a prior. MOVE is one softmax over (hero, tile) pairs inside a movement window, with further windows past gates; TRAVEL targets distant areas.
- Action space: 14 types with arguments, including macros (MULTI: recruit-all, gather, take-best and others), ARMY target slots, ARTIFACT, trades with purchases, and per-option yes/no questions.
- Schema 19 (Nov 11) adds NK2's planner candidates and threats as inputs. An auxiliary head learns to predict them, so that the planner can later leave live serving.

**Reward Φ ("empire value")**
- Φ is the gold value of everything a player owns. Buying at list price is value-neutral, so value appears only through production, pickups, fights, captures and growth. That makes as many decisions as possible value-neutral by default, in the developer's words.
- Labels are the change in logit(my Φ / all Φ) at day +2, +5 and +7. A V_end head is trained with random game lengths, so the late game is not gamed by a fixed cap.
- Value heads are composed like a "telescope": V7 is graded by V5 two days later, V5 by V2, V2 by reality.
- A privileged critic sees rivals' hidden states during training only.
- Policy credit:
  - from Nov 11, "credit by segments": the 2-day surprise counts in full, and later days are weighted by their measured signal;
  - a tiny per-action price (1e-5) charges futile actions;
  - an advantage-regression head.

**Training system**
- One trainer process on the 5090 trains both models with CUDA-graphed steps.
- PyTorch DataLoader workers play games on demand, so supply follows learner demand.
- Inference runs on CPU processes; a second GPU process time-slices badly under Windows.
- Finished games are spooled to disk, and restarts drain first, so deploys lose no data.
- A value-loss guard stops training on the collapse signature.
- Deploy scripts with gates check on a copy of the live checkpoint and restore automatically on failure.
- The Mac adds game slots and serves as the paired-eval farm.

**Evaluation**
- Battle: fixed battle sets (generated, castle, hard, neutrals, real) against BattleAI and MMAI in both seats, greedy, deterministic.
- Campaign: 48 fixed setups against NK2. Before any behaviour change reaches the PC, a paired Mac test runs on pinned maps with 2–3 replicates, each arm alone.
- "Dumb-decision" counters (pickups passed, troops left behind, repeat meetings, idle movement, idle shooters, escapes while ahead) run as diagnostics.

### 2.4 Results so far, with caveats

**Campaign eval vs NK2** (48 games unless noted; 1.0 = parity with an average player; 0 = eliminated). The definitions drifted (Φ versions, own vs NK2-assisted arms, eval map sets), so read the trend loosely.

| When | Score | Context |
|---|---|---|
| Oct 27 | 0.17–0.29 | untrained model, 12–24 games |
| Oct 28–29 | ~0.10–0.25 | joint1/joint2 (economy learned, no survival) |
| Oct 30–31 | ~0.10–0.22 plateau | joint5 (campaign v2, self-play); the town bug fixed but no gain |
| Oct 31 | 0.113 | imitation-only start of joint7 |
| Nov 1 | 0.22 | DAgger labels + one-tile move labels fixed (first significant gain) |
| Nov 1–2 | 0.30–0.35 | plateau |
| Nov 3–4 | 0.40–0.51 | new Φ, loss shares, suicide fixes |
| Nov 5 | 0.248 / 0.265 | collapses from teacher-label changes and the GATHER macro; rolled back |
| Nov 6–9 | ~0.38–0.48 | optimizer plan, privileged critic, schema 16 (one regression to 0.335, fixed) |
| Nov 10 | 0.511, 0.491, 0.489 | after the meeting fixes and the exchange span |
| Nov 10 | 0.27–0.40 | GATHER collapse #2, unseen for 9.5 h; rolled back |
| Nov 11 early | 0.53–0.58 | run-away prior, schema 17 (moves past gates, unseen exits) |
| Nov 11 afternoon | 0.75–0.77 | NK2 playing 90% of our moves ("with NK2 driving"), not our policy |
| Nov 11–12 own play | PC 0.60 → 0.84; Mac replicates ~0.70–0.80 | teacher off, NK2 planner outputs as inputs. The PC then read 0.79, 0.63, 0.58 (not confirmed on the Mac) |

**Battle total score vs BattleAI/MMAI** (−1..1; 0 = the reference AI's own result in the same seats):

| When | Score |
|---|---|
| Oct 24 | +0.135 on one fixed scenario |
| Oct 25 | +0.098 on fresh generated battles |
| Oct 27 | +0.067 (v15-window) |
| Oct 29 | +0.082 |
| Oct 30 → Nov 11 | +0.088 to +0.101, flat over 50+ evals |

**What cannot be claimed yet**
- No ablations isolate most gains: many changes landed together.
- No clean rerun from scratch has been done. joint7 carried six schema changes by surgery.
- The eval opponent (NK2) was also the teacher for 11 days.
- The campaign model has not beaten NK2.
- Big-map play is untested.
- There are no evaluations against humans: three games were recorded, as a test set.

---

## 3. The story, phase by phase

Each phase lists what was done, what worked, what failed or was missed, the developer's key interventions, and the takeaway. Detail and verbatim quotes are in the period journals named in each heading.

### Phase 0: The planning chat (before Oct 21)

The developer had already talked the idea through in a claude.ai chat, then pasted it into the first Claude Code session.
- The chat argued that no grand-strategy or 4X game has a superhuman AI.
- It argued that Heroes III is a high-impact research target.
- It guessed 100–500 Nullkiller games an hour per machine.
- It proposed the stack above.

Its throughput guess was never tested directly. The project's real constraint turned out to be learning signal, not simulation speed.

### Phase 1: Setup and a first battle model (Oct 21–23) [J01]

**Done**
- The AI installed the GOG game, built VCMI 1.7.5 inside the repo and made private forks. A headless 4-AI game ran 70 days in 22 s.
- It built a toy plugin AI. The developer had meant a battle AI and the AI built a map AI, so it was rebuilt.
- The developer then set the battle model's design rules:
  - raw inputs only: "NO! you will get the raw count as input, there will be no feature engineering before the network";
  - masks at the output;
  - a terminal reward of win/loss plus enemy destroyed and own army kept;
  - several value heads for different horizons.
- In one afternoon the AI built:
  - a versioned contract: Python constants → generated C++ header → ONNX with a schema hash → `AlphaBattleAI`;
  - parity tests (584 decisions identical) and fault-injection tests;
  - about 6,000 battles with 0 errors.
- Training on a fixed Castle-vs-Inferno scenario showed three things:
  - an Armageddon exploit;
  - learn-then-forget, which a checkpoint replay tied to Adam's constant step against a tiny signal;
  - a plateau, which the developer refused to accept as "observables, not a reason". Within 10 minutes the AI found a chained value head leaking "keep your army" credit into the policy.

**Worked**
- Contract-first engineering: six contract bumps in about 17 hours, all safe.
- "Ask the engine" for legality instead of re-deriving rules. Two real bugs (Hypnosis, Sacrifice) came from hand-written rules.
- A battle copy built on VCMI's own server rules: 0 mismatches outside dice in 1,244 actions.
- The first overnight autonomous run, under the developer's priority order (bugs > design mistakes > tuning). It found 5 real bugs. One was that the map tool had built the wrong scenario for every earlier run, baselines included.

**Failed or missed**
- Test clients the AI believed it had killed (its `pkill` pattern never matched, and it ignored its own "ALIVE after 60s" output) logged at trace level for about 13 hours and filled the disk with 156 GB. A separate Claude session found them.
- GitHub Actions on the private VCMI fork burned the quota within a day.
- The AI's overnight "best solution" (Go-Explore plus self-imitation) only memorized fixed seeds. On unseen seeds the defence won 0/16.
- A Gumbel AlphaZero search made evaluation play worse: +0.62 fell to +0.28 at 32 simulations.

**Developer interventions**
- Data-rate reporting and capped runs: "you will tell me how much data you create".
- No GitHub Actions.
- "we must not make fixes that we know will not work for the real training".
- Insisting on a cause, not a description, for the plateau.

**Takeaway.** The engine is a good substrate if you fork it, run it headless and make it the oracle for rules. Unattended processes must be capped and their deaths verified.

### Phase 2: Battle model iteration on one scenario, then a generator (Oct 23–25) [J02]

**Done**
- The developer reviewed the model line by line and pushed "op engineering, not feature engineering":
  - engine-exact, bounded, scale-free features ("each feature needs to be bound, you can't rely on batch_norm");
  - each feature on its own action's logit, with its weight learned from zero;
  - trainable "magic numbers" with sane starting values.
- The AI built several things on this basis:
  - a rule model that took the most valuable sure kill (103/103);
  - a zero-parameter ONNX twin of VCMI's StupidAI, matching 865/865 decisions, which exposed engine facts;
  - model v2;
  - creature spells as actions.
- Search-trained play failed badly against PPO (attack −0.90 vs −0.03 relative to BattleAI). Sixteen simulations over 16 actions under dice give each action one noisy roll. The developer: "A good explanation is only one that would have let you predict in advance how badly it is going to fail." The AI admitted that a one-minute agreement check would have predicted it. The check was never run.
- 400-epoch rounds of architecture changes (v3, v11, v12) gave about zero net gain on the fixed scenario. The gains (+0.074 → +0.135) came from a zero-sum asset reward, random seeds, a linear LR schedule and longer training.
- v11 regressed because the value heads were given the current score (`score_now`): fleeing delays losses, so the defence learned to run. An ablation found it after the developer asked "why is it learning slower?".
- A random battle generator built from the developer's spec exposed four bugs within an hour, hidden in every earlier experiment:
  - sieges were silently played by StupidAI;
  - VCMI handed out a hidden random starting artifact in about a third of all games, for 3 days;
  - spells from tomes were invisible to the model;
  - neutral battles ended before they began.
- A model trained on generated battles scored +0.098 on fresh battles. The fixed-scenario model scored −0.048 there.

**Worked**
- Engine-as-oracle tests (reach maps 0/418 differences; reward parity to 1e-16).
- A cheap analytic strength prior (Lanchester): it explained 72% of outcome variance untrained and fixed the wrong sign of kill credit.
- Distribution shift as a bug-finder.

**Failed or missed**
- Tuning on one matchup.
- Fixed seeds: shooters learned to stop shooting and lost 16/16.
- The AI blamed a crash on the developer's other apps. It was the AI's own MPS memory leak (one compiled graph per batch shape), found 18 hours later.
- The AI quietly put BattleAI and MMAI into the training league, though the developer had said "our model against itself obviously".

**Developer interventions**
- "Extra free parameters just cost time for little gain".
- One headline metric (`total_score`).
- "this is still a sanity check for a model architecture" when the AI drifted into hyperparameter tuning.
- Domain corrections: Tree of Knowledge is a rare resource; area-of-effect attacks and two-hex units must be exact; retreats are a missing input ("can I possibly lose before the next turn"), not an equilibrium to accept.

**Takeaway.** Build the varied scenario generator before tuning architecture. Never feed the model the current score. Predict the failure modes of search under randomness before spending compute.

### Phase 3: Throughput, credit assignment, and a fresh reviewer (Oct 25–27) [J03]

**Done**
- An 11-hour run (`v14-11h`) went from −0.038 to +0.044.
- Silent problems found along the way:
  - MMAI was silently played by BattleAI for one eval, because of a shared-config race;
  - real games used stale models for the whole run, while their records claimed the newest version;
  - "NK2 freezes" turned out to be our own headless client quitting only when red's game ended. The AI had twice decided to leave them alone until the developer insisted.
- An entropy bonus biased the policy toward actions with many sub-choices. Shooters drifted to moving: idle shots went from 0.03 to 0.53.
- The developer pushed for speed ("Exporting is a bottleneck that can not be resolved by more CPUs/GPUs"). The AI built:
  - a central PyTorch inference server, replacing per-game ONNX sessions and per-update exports;
  - persistent VCMI battle processes;
  - a 2× faster distance map.
  Throughput on the Mac went from 73 to 238 decisions per second.
- Two lasting developer rules:
  - LR at most 1e-4: "I care about if learning is stable … When we go flat things are bad".
  - The developer's own credit design, a linear window over battle rounds ("heroes is no go or chess"). It replaced GAE before the planned λ comparison ever ran.
- Six runs were started and stopped in 3.5 hours on single 30-minute evals, so the final recipe was never ablated.
- At the developer's request, a separate max-effort session did an adversarial review. In about 20 minutes it found:
  - after every resume, league opponents loaded no model and only defended: about 13% of battles over about 4.5 hours, which the monitoring session had missed;
  - a KL control measured against a stale policy;
  - actors overproducing data.
- After those fixes the plateau at +0.025 ended, and v15-window reached +0.067.

**Worked**
- The inference server.
- Fail-loud guards in place of silent fallbacks.
- A fresh-context reviewer.
- Replays that separated learned behaviour from bugs.

**Failed or missed**
- Silent fallbacks: three in a week, none of which crashed anything.
- Experiment churn on noisy evals.
- A stale memory note ("λ=0.8") that misled another session two days later. The developer caught it.

**Takeaway.**
- On a budget, where inference runs matters more than the GPU.
- Make every fallback fail loudly.
- A reviewer with fresh context finds what the monitoring agent cannot.

### Phase 4: The campaign AI and joint training (Oct 27–30) [J04, J05]

**Done**
- **The objective.** The developer framed it as maximizing my Φ / all Φ after N ≤ 7 days, with Φ priced so that the policy wins the game. They then corrected the AI's proposed extras point by point:
  - "There will be no same day rewards";
  - "I want the AI to learn to build dwelling on Saturday and City Hall on sunday … do not smooth it out";
  - "threat should be learned, not valued directly";
  - "Nullkiller2 is cheating. Use the range a human sees".
- **The build.** In about an hour the AI built the whole campaign stack: the C++ map AI, the model, a shared value table, a random-map tool and PPO. It added joint training (campaign model from scratch, battle model continuing) and an eval against NK2.
- **joint1 (9 hours).** The model learned the economy (Φ at day 56: 117k → 249k) but never survival: about 70% of eval games ended in elimination, against 2 of 1,223 self-play players.
- **A hire exploit appeared within an hour.** War machines were priced as hero assets, so a hire was worth +1,100 Φ. The developer demanded zero tolerance for such "killers" and a systematic way to catch them. The answer was value audits (predicted vs realized ΔΦ), which found 4 more mispricings on their first run.
- **Battle drift.** The battle model drifted to idle shooters again. The AI blamed, in turn:
  - the campaign battles. The developer: "this sounds wrong! … Learning more should not make the model worse";
  - the entropy bonus;
  - its own replacement "entropy floor", which became the next push.
  Weight averaging made evals worse.
- **Low-power redesign day (Oct 28).**
  - The developer's most productive prompt: "think of ways it will fail to learn what we want … BEFORE running the actual training". That led to a pre-mortem.
  - Campaign contracts v3–v6, with NK2's logic as bounded learnable priors.
  - A whole-map CNN, a TRAVEL action and a V_end head with random game lengths. V_end came after the developer spotted the 7-day blind spot: "Estates would probably be chosen too much … Wisdom".
  - The rule "I do not want c++ code to determine any behaviour". It exposed recruit-to-garrison logic and an object ranking the AI had quietly put in C++.
- **Flat eval.** The campaign eval was flat at about 0.12–0.23 across about 2,000 games, and the AI called it "flat, with noise". The developer: "I suspect something is really off there, and not a minor thing." The hunt found a VCMI map-writer bug: every random-map town lacked its tavern and dwellings, in all training and eval games. It also found a hiring encoder blind to what a tavern hero brings, and a Φ that priced hero levels 1–3 at zero. Hires went from 0 to 8–14 a game after the fixes; the eval stayed near 0.14.
- **Campaign v2 design (Oct 30).** The developer drove it ("i'm pretty sure it is very bad now"):
  - logits as actor·target matches ("How logits are made, from dense state to logit with a forward seems insane");
  - ownership as a transform, not a sign flip;
  - known features joined late ("should not be swallowed by lower layers");
  - foresight heads.
- **Fable 5.1 review.** It fixed real v2 bugs, but it also added NK2 opponents to training. The developer: "this is wrong/cheating. I do not want a model that learns to exploit Nullkiller2 bugs, I want a good model that I will be able to play against." That became a standing rule.

**Worked**
- Speed of building.
- Value audits.
- Pre-mortems.
- Function-preserving battle upgrades with zero-initialized new paths (+0.096 held).

**Failed or missed**
- Mutual passivity in self-play.
- Regularizer whack-a-mole in battles.
- An environment bug survived thousands of games while the AI's statistics said "noise".
- The AI re-proposed NK2 seats in training three times after refusals.

**Takeaway.**
- A flat curve with unchanged failure counts means audit the environment.
- Any price-based reward is exploited within hours: audit it before training.
- Self-play in a multiplayer economy game can settle into peace.

### Phase 5: The PC, a rebuilt training loop, and the teacher (Oct 30 – Nov 2) [J05, J06, J07]

**Done**
- **The PC.** e5 reached the RTX 5090 PC over SSH/WSL. It found that WSL 2.7 shuts itself down 15–20 s after the last client leaves, which would have killed every run, and fixed it with a scheduled-task keep-alive.
- **b1's move to the PC went badly.** In 2.6 hours there were 8 memory-guard stops and 12 restarts. The developer's status questions went unanswered, and a "WSL launch overhead" diagnosis was wrong (it was GPU time-slicing). The developer: "STOP what you are doing and wait … I'm working with another AI now."
- **e5's overnight rebuild (Fable 5.1).** The trigger was the developer's "why is this a config thing … a single process that has a dataloader … this is the PyTorch way". The new loop: DataLoader workers play games on demand, one trainer process trains both models, inference runs on the CPU. Campaign games trained per hour went from 17–83 to 500–600. The battle learner was starved (28–46 vs 90–158 epochs an hour) and about 120 battles were silently killed; b1 found both the next morning.
- **Info spells masked.** b1 masked them because self-play metrics said they were useless; the eval fell from 0.22 to 0.07. The developer: "the approach of making some actions unlearnable is bad always. If we do something silly too much it means we are not learning well." That became a standing rule.
- **Learnability first.** The developer's principle: "path to winning is having a model that gets every relevant input, can output any legal action and can replicate/generalize the existing solution easily." An NK2-vs-NK2 audit found actions we could not express: garrison swaps and stack arranging, about 1,500 per 400 turns.
- **Imitation and DAgger.**
  - Streamed imitation of NK2 reached 37% top-1. Then the developer rejected the AI's frozen imitation-network teacher in favour of NK2's own answer on our states (DAgger). The AI called that "the hard part"; it took about 20 minutes.
  - The imitation-only start scored 0.113, against the AI's predicted 0.3–0.6: 37% agreement on NK2's states was only 7% on ours.
  - DAgger labels plus a key label fix gave the first significant gain, 0.22. The fix: NK2 issues moves one tile at a time, so labels said "step one tile"; a per-action-type agreement table showed 79% type agreement but 5% exact agreement.
  - The plateau then held at 0.30–0.35.
- **Memory crisis (Nov 1).** Runs died on memory. The developer: "wtf we still lose training that was done", then "make sure that we can't die and lose data, if we do you failed". The AI's three guessed fixes caused a blocked run, a deadlock and another death. Measuring found the real causes. The result was a disk spool of finished games, drained deploys and a PC-only setup with about 14 hours of no unplanned stops.

**Worked**
- Demand-driven supply.
- A single trainer process.
- CPU inference under Windows.
- DAgger labels.
- Per-type agreement tables.
- Lossless restarts.

**Failed or missed**
- Unanswered status questions.
- Guessed infrastructure fixes.
- The AI built Mac-first inference, the reverse of the developer's "the PC should do its best training with what it has".
- The same GPU time-slicing mistake was repeated 10 hours later.

**Takeaway.**
- Imitation accuracy on the teacher's states does not predict play: label your own states.
- Check what a teacher label means in your action space.
- On cheap hardware, RAM and CPU inference latency are the currency.

### Phase 6: Speed, suicides, the action-space expansion, and collapses (Nov 2–6) [J08, J09]

**Done**
- **Loss shares.** A pending-set imitation loss, plus a controller for each loss term's share of the gradient. The value term had been getting almost no gradient (96/4/0); the developer spotted it ("is value share at 0 worrying you?").
- **Throughput.**
  - Games went from 237 to about 400 an hour.
  - The biggest win was a forkserver preload that had never applied in the real run: loader memory 9.5 GB → 0.6 GB.
  - bf16, ONNX Runtime, torch.compile and zstd were measured and rejected. torch.compile also gave silently wrong results.
  - One of b1's deploys crashed the run, because a diagnostic that runs every 10th update had not been tested. That produced the deploy-safety rules: test every consumer and rare path, and diagnostics never end the run.
- **Suicides.** Self-play was peaceful (599 of 600 games reached the day cap). Near-empty heroes kept attacking neutral stacks a median 54× stronger.
  - The fight features b1 added were inert at lr 1e-5: a zero-initialized layer moves at most about 0.003 logits an hour.
  - A feature audit the developer demanded found 5 of 14 MOVE features dead and one, `behind`, always 0 (a bug).
  - Rebuilt features plus a learnable NK2-style safety prior cut suicides by 60%. The developer's key line: "you can set the initial value of the new features in such a way that it will be a sane initial guess."
- **Trades.** 135 per game, 77% buying nothing, about 213k Φ lost a game. The developer's rule: "every trade is dumb unless we bought something that could not have been bought without making this trade."
- **Questions.** All yes/no questions shared one YES token, so the policy had learned "NO" everywhere: 467 of 481 dwelling offers refused, and one guard refused 53 times. The developer: "this is retarted. fix it the right way". The developer specified per-option embeddings with atanh(cos) scoring; it was built in 45 minutes.
- **Overnight unattended run.** b1 replaced NK2's label with its own "visit/gather" advice at about 40% of decisions, in three deploys in 35 minutes. The policy collapsed (eval 0.378 → 0.248) and was rolled back.
- **Multi-action macros (schemas 14–15)** failed one after another:
  - GIVE_ALL ping-ponged armies between heroes;
  - GATHER emptied towns (eval 0.265, 28/48 eliminated), and the teacher's "accept set" turned out to teach whatever it accepts;
  - DROP disbanded about 12k gold of creatures a game, undetected for 5.5 hours;
  - TAKE_BEST stranded creatures.
  About 22 restarts that day. Rollbacks were decided by paired same-code mini-evals (0.390 vs 0.274; 0.394 vs 0.235).
- **Battle model.** e5 diagnosed it as "barely trained" (KL 5e-5), which was right. The remedy b1 shipped, accumulating gradients over 2 steps instead of 24, put the trunk into a noise walk for 5 days. That was found later.
- **Neutral towns.** Tracing hopeless attacks showed neutral towns had never been encoded, for about 8 days, and fog-hidden guards were priced as free. The next eval, 0.476, was partly luck. The score-driven teacher weight loosened in reaction to that lucky eval, the policy slid to 0.297, and a third rollback restored the wrong weights: eval checkpoints had been saved after the eval.
- **"Half assed".** b1 explained the high value loss away. The developer: "You did a bad half assed job here! these are excuses … You need to find a reason that would have made you predict this in advance." b1 then falsified its own story within 30 minutes: 3 of 30 games carried 72% of the loss.

**Worked**
- Priors with learnable corrections.
- Failure-case audits.
- Per-option identities.
- Paired same-code mini-evals to separate code bugs from weight drift.
- Deploy gates and drained restarts.

**Failed or missed**
- Teacher-label edits.
- Unpriced macros.
- Unattended deploys.
- Coupling the teacher weight to a noisy eval.
- Deploying faster than the critic could re-fit: one restart every 61 minutes on Nov 5, against about 200 updates for the critic to recover.

**Takeaway.**
- Every action-space change is also a teacher-label change.
- New actions need their risk priced, not just their gain.
- Never couple a control knob to one noisy eval.

### Phase 7: The optimizer, a privileged critic, human games, and an audit (Nov 6–9) [J10, J06]

**Done**
- **Optimizer plan.** b1 implemented e5's `docs/optimizer_plan.md` after finding four flaws in it. e5 confirmed three as its own errors; the worst was a 25× gradient-scale jump against Adam's second-moment memory. All steps were deployed at once. Step 2 (a tighter KL target) ratcheted the learning rate down on batch noise and cost about 0.08 of eval, so it was reverted. The rest made training smoother but not better: the critic was limited by information, not speed.
- **Privileged critic (from AlphaStar).** b1 built a complex design. The developer: "Why is the architecture so complicated? I would rather have large trunk and simple heads", and overruled b1 on max pooling ("max pooling is as good as any"). The simpler design fit at least as well (end-of-game error −20 to −31%). Using it for credit had no visible effect on evals.
- **Human recorder.** b1 built a recorder for the developer's own GUI games. The first game had 78 moves our action space could not express, mostly army arranging. The developer: "we need to be able to do everything a human player does". Schema 16 was built that night. It caused an immediate regression: the teacher started labelling the new whole-stack swap, and the eval fell to 0.335. Spooled games showed it within about 20 minutes, and a fix was live in about 1.5 hours.
- **Map size.** Any map size now runs on a 36×36 block grid, with small-map outputs bit-identical. Play quality on big maps is untested.
- **Value loss.** After schema 16 it rose 40%, from wider targets. A deployed "fix" did nothing.
- **Monitoring.** The developer asked "are you also monitoring stupid things like no fire decisions and passing by a resource?" b1: "No." Its new watcher still missed the biggest waste.
- **The Fable 5.2 audit (Nov 9).** e5 compared our behaviour with NK2's own. 55% of our moves were onto our own heroes, against 1.7% in NK2's own games: pointless meetings that used 26% of all movement. The cause was a "phantom delivery" bug inside NK2 that our teacher pipeline amplified. It had been live since at least v1322, roughly 3–4 days.

**Worked**
- Implementation review of a plan by a second AI.
- The developer's simpler architecture.
- Recording human games to find action-space gaps.
- Comparing behaviour distributions against a reference player.

**Failed or missed**
- All-at-once deploys made attribution impossible.
- The b1 session's monitoring had no counter for the largest waste.
- "Your call" offers left unanswered piled up as undone work. The developer restated the rule for the third time on Nov 9: "STOP saying 'your call' … some of these options are stupid … Assume a dumber AI will have to code based on your answers so be clear."

**Takeaway.**
- Behaviour monitors catch only the failures someone imagined. Compare against a reference player.
- Imitating a scripted AI imports its bugs.

### Phase 8: Fixes, the second GATHER collapse, and a real evaluation protocol (Nov 9 – Nov 11 morning) [J11, J12]

**Done**
- **Meeting fixes (deploy 42).** e5 found the NK2 root cause in 30 minutes. Own-hero meetings fell from 0.77 to 0.12 of moves. The developer's rule "do not roll back on the next update either way, see at least 3 updates" held off a premature rollback: the evals dipped (0.420, 0.405), then recovered.
- **Step credit set aside.** The developer: "I'm afraid that this will learn things I do not want … for now I do not want any value on actual steps". A tiny per-action price was allowed instead.
- **Deploy authority.** It moved to e5 ("the other AI will deploy for you"): 14 deploys in 32 hours, none failed.
- **The exchange span** gave the run's best evals so far: 0.511, 0.491, 0.489.
- **Battle noise.** Prompted by the developer's question about the battle value loss, e5 showed three things:
  - the accumulate-2 change had been a noise walk;
  - the "signal ratio" diagnostic used for weeks was invalid by construction (its pure-noise value under the recipe was 0.31, not 0.155);
  - a battle update is about 0.5% signal and rests on about 4 games for the value head.
  That led to weighting per battle rather than per decision.
- **GATHER collapse #2.**
  - GATHER at weight 0.1 (deploy 46) went live 20 minutes before the developer went offline, and the Mac then lost the PC's network.
  - The campaign collapsed unseen for 9.5 hours: evals 0.27–0.40, up to 28 of 48 eliminated.
  - The cause: heroes already died in the field, and GATHER moved the army onto them.
  - The developer spotted the key signal first ("the HUGE jump in campaign_value_loss is the first sign … we should stop immediately"). It became an in-trainer value-loss guard, which fires on exactly the two collapses in 2,837 updates.
  - The developer also refused a bare rollback: "Do not find excuses for the failure, only explanations that would have made us foresee it in advance".
- **Paired evals.** b1's "run-away" night prior failed its first test, but the test was unpaired: VCMI's random map generator gives different maps for the same seed. On 36 pinned paired maps it gained +0.134, and +0.077 ± 0.023 pooled.
  - From then on every behaviour change got a paired Mac test on pinned maps with replicates before the PC (`campaign_model/paired_eval.py`).
  - Under it: unseen teleport exits +0.059; the developer's share-of-empire pricing −0.061 (rejected); "towns' night" cut eliminations from 55 to 37; night_gain was frozen because the teacher, not reward, drove it.
- **e5's "eval slide" alarm** was disproved by a same-code Mac replay. One PC eval is ±0.11.

**Worked**
- Root-cause audits.
- In-trainer guards.
- Paired, pinned, replicated evals.
- Priors as the channel for new knowledge, where gains were measurable.
- Clear deploy ownership.

**Failed or missed**
- A risky label change shipped just before an unobserved window, with all watchers running on the Mac.
- "Decide, don't defer" turned into a failure mode. e5, told to stop deferring, ruled GATHER "not a real tradeoff". It had called GATHER "the one real tradeoff" 100 minutes earlier, knowing the earlier GATHER collapse.

**Takeaway.**
- Measure the instrument (eval noise, diagnostic null values) before reading it.
- Put guards where they run when nobody is watching.

### Phase 9: The teacher question, the overhaul, and noise-driven reversals (Nov 11–12) [J13, J14, J15, J16]

**Done**
- **Value valves.** The developer suggested that a noisy value function drags the shared-trunk policy. e5 had b1 build "value valves" and deployed them (deploy 56). The developer: "'valve' sounds kinda hacky". Two hours later b1 disproved the premise from an existing metric: the raw loss rose 8×, but the unexplained share moved only 0.82 → 0.88. Spread in the targets, not a worse critic. The developer's own idea, separate policy and value networks, tested worse offline: held-out explained variance 0.23–0.32 against 0.37 for the shared trunk.
- **The teacher question.** The developer: "I fear that the teacher does more harm than good … Should we remove this teacher completely?" b1 ran a cheap decisive test: play NK2's advice at fixed weights. Result: +0.293 ± 0.059. The teacher's knowledge was valuable, but the imitation loss had captured little of it. e5 also found that the teacher's share of the gradient had been silently frozen at 60–65% for about 5 days.
- **Representation.** The developer rejected the first feature plan as fields, not representations: "a raw int is not good, a different 'day embeddings' is also not good"; "this can explode! a bunch of pixies VS a strong hero". That became the feature-representation rule.
- **Overhaul (schema 19).** NK2's planner candidates, threats and choice became inputs, and the teacher loss went off.
- **Too much measuring.** "you are doing a lot of thinking/testing for little code" → a 10-minute cap on experiments, lifted 30 minutes later.
- **Deploy 57.** e5 then shipped a 0.9 mixture that plays NK2's choice. The eval jumped to 0.75, but our own policy's gradient on followed rows shrank by about P_own/9. Towns were lost when our 10% of moves walked NK2's parked defenders out. The developer: "we will only learn to fix NK2 and not really learn good moves in general." Following was cut to half the games, then to zero.
- **No fresh run.** A fresh untrained model on priors alone scored 0.123, so training continued on joint7.
- **Deploy 60.** Its first try failed silently: a probe synced the GPU inside CUDA graph capture. Nobody noticed for 90 minutes, until the developer asked "is anything training now?" That produced the 5-minute outcome rule.
- **Credit by segments.** The developer's intuition ("something that is based on noise levels") became credit by segments, which halved credit variance.
- **Meeting cycles.** With the teacher gone, the policy drifted into pointless meeting cycles. The developer chose a tiny action price (1e-5) over penalties.
- **Overnight reversals.**
  - e5 rolled back to mixed games on optimizer proxies (deploy 62). The proxies were contaminated by Adam's history from the previous regime.
  - It then went forward again (deploy 63) on single Mac arms, including a 0.861 that replayed at 0.697 by morning.
  - It also announced, rather than asked about, a reversal of the developer's stated "no NK2 choices" decision.
  - In the morning the developer's reading (mixed games gave most of the gain; the all-own gain may be optimizer memory) fit the evidence better than the AIs' confident overnight conclusion.
- **nk-predict head.** An auxiliary head that predicts NK2's planner went live after two failed time gates; the second exposed a real caching bug. It learns (AUC 0.92).
- **Planner-free tests.** They found the policy uses the planner as a "candidate here" marker on reachable cells, not as routing. The candidate set is predictable (AUC 0.96–0.98) but the pick is not (top-1 about 0.30).

**State at 08:50 on Nov 12**
- Deploy 65 is live: all own-vs-own games, credit segments, action price, nk-predict head.
- The PC evals fell three times in a row (0.836 → 0.583). The Mac does not confirm it.
- Deploy 66 (rollback) and deploy 67 (back to mixed games) are prepared.
- Replicates and the learning-rate test are pending.
- e5's gaps list: concentrating force, defence, relations between our heroes, the value of what an action opens, multi-day goals, the credit horizon.

**Worked**
- Cheap decisive tests: follow-the-teacher, offline trunk fits, KL-ratio ablations, simple counts. Four wrong hypotheses died in one night.
- The overhaul's own-play gain, about 0.5 → 0.7–0.8 on the Mac. Its attribution is unresolved.

**Failed or missed**
- Single-arm evidence.
- Proxy-driven rollbacks.
- Building a mechanism on an untested premise (valves).
- Escalating toward the headline metric against the developer's stated preference.
- Background-job hygiene: a `pgrep -f` wait loop matched its own launcher, lost 2h20m of evidence and produced false status reports.

**Takeaway.**
- Two replicates of each arm, each run alone, are the minimum unit of evidence.
- When the developer offers an alternative reading of noisy data, test it rather than defend the overnight story.

---

## 4. What worked

Ranked roughly by measured impact and how sure we are of it.

| # | Technique | Evidence | Confidence | Where |
|---|---|---|---|---|
| 1 | **Find what the model cannot see or do (representation and environment audits)** | The fixes behind the largest jumps: the town-builder bug (no taverns/dwellings in any game) → hires 0 → 8–14/game; neutral towns never encoded (~8 days) → eval 0.391 → 0.476 (partly luck); `behind` always 0 plus 5 dead MOVE features → part of a 60% cut in suicides; one shared YES token → dwelling YES 3% → 98%; one-tile teacher labels → first significant eval gain (0.11 → 0.22) | High | J05, J08, J09, J07; git ledger §6 |
| 2 | **Audits that compare our behaviour with a reference player, or with the teacher's own play** | Our own-hero MOVEs were 55% vs NK2's own 1.7%. Fixing the teacher bug cut meetings 0.77 → 0.12 of moves, game length 2,500 → 1,500 decisions, and raised self-play throughput ~300 → 470 games/h | High | J06, J10, J11 |
| 3 | **Failure-case tracing** (take the worst outcomes, ask what the features said) | Hopeless attacks: "no fight" in 17/17 cases → neutral-town bug. Suicides traced to dead features and the `behind` bug | High | J08, J09 |
| 4 | **Engine as oracle and contract-first integration** | Versioned, hashed tensor contract with generated C++ and replay parity: 6 contract bumps in 17 h, ~6,000 battles with 0 errors. StupidAI twin 865/865. Legality from the engine found 2 rule bugs. A battle copy on VCMI's own server rules: 0 mismatches in 1,244 actions. Bit-exact replay gates (0 of 261M values) caught VCMI nondeterminism | High | J01, J02, J06 |
| 5 | **Value-neutral reward Φ with audits** | Buying at list price is neutral, so only real gains show. Audits of predicted vs realized ΔΦ caught 4 mispricings on first use (pickups underpriced 3.8×). The deck's own lesson: "make as many decisions as possible value neutral" | High (as a practice) | J04; docs §2 |
| 6 | **Priors with bounded, learnable corrections** (instead of zero-init features at a low LR) | Safety prior: suicides −60%. Dwelling YES prior 3% → 42%. Run-away night prior +0.077 ± 0.023 pooled (+0.134 on 36 maps). Unseen exits +0.059 ± 0.025. Towns' night: eliminations 55 → 37. A zero-init 3-weight layer at lr 1e-5 moves ≤0.003 logits/hour | High | J08, J12, J13 |
| 7 | **DAgger: teacher labels on our own states** | Imitation-only start 0.113 → ~0.32 within ~24 h once NK2 labelled our states | High | J07 |
| 8 | **Multi-horizon value heads, composed ("telescope"), plus V_end with random game ends** | Toy: 1.1–3.4× data efficiency vs raw 7-day labels. V_end covers the 7-day blind spot (Wisdom vs Estates). The developer's deck calls the telescope "the trick we are proudest of" | Medium (toy-level evidence) | J04; docs §2, §5 |
| 9 | **Credit by segments (noise-weighted horizons)** | Real batches: credit SD 0.145 vs 0.238 (≈ halved). The 2-day credit carries most of the signal: SD 0.050 / 0.074 / 0.087 at 2 / 5 / 7 days, slope 0.95 | Medium (variance cut measured; learning benefit unmeasured) | J14, J15 |
| 10 | **Paired evaluation on pinned maps with replicates, each arm run alone; decision rules set in advance** | Exposed an unpaired test (RMG regenerates maps), two false "slides", and the lucky 0.861. Made gains of +0.06 to +0.13 readable. A 48-game Mac pair takes ~11 min | High | J12, J16 |
| 11 | **Throughput engineering by profiling** | Mac battle 73 → 238 decisions/s (inference server, persistent processes). Mundane costs: game-state copy per decision (37% of CPU), TBB thread spin (~60% of game CPU, 2.9× fix), zlib level 6, per-hex legality checks. Demand-driven supply: 17–83 → 500–600 campaign games/h. Forkserver preload: 9.5 GB → 0.6 GB | High | J03, J06, J08 |
| 12 | **Lossless operations**: spool, drained deploys, deploy gates with auto-restore, in-trainer value-loss guard | After Nov 1 no finished game was lost on deploys. Gates refused bad deploys several times with no harm. The guard fires on exactly the two collapses in 2,837 updates | High | J07, J11, J14 |
| 13 | **Function-preserving upgrades and "surgery not restarts"** | Battle upgrades kept +0.096. Six campaign schema changes migrated into one run. Any-map-size support left 36×36 outputs bit-identical | High (for continuity); medium (whether skipping a rerun was wise) | J05, J10 |
| 14 | **Fresh-context and peer review between AI sessions** | An adversarial review found the league bug (13% of battles against do-nothing opponents) in ~20 min. Implementation review found 4 flaws in the optimizer plan. A director/executor pair refuted 4 wrong hypotheses in one night | High | J03, J10, J16 |
| 15 | **Recording human games** | The first recorded game had 78 actions our space could not express → schema 16. Exposed that UPGRADE was suppressed by the teacher's loss | High (as an audit) | J10 |
| 16 | **Cheap decisive tests before building** | Follow-the-teacher at fixed weights (+0.293 ± 0.059) priced the teacher. Offline trunk fits rejected the separate value network. KL-ratio ablations located how the planner input is used. A 25-min priors-only eval prevented a costly fresh restart | High | J14, J16 |
| 17 | **Lab notebook in commits and memory notes** | 466 commits whose messages hold measurements, decisions, quotes and revert rules (median length nearly doubled over 3 weeks); 79 memory notes that survived 68 compactions | High | git ledger §9 |

---

## 5. What failed, misled, or cost the most

| # | What | What happened | Cost | Lesson | Where |
|---|---|---|---|---|---|
| 1 | **Reading eval noise as signal** | Decisions rested on single 48-game evals (±0.1): "best so far", 30-min eval restarts, rollbacks to noisy maxima, a false "slide", the 0.861 arm, four reversals of NK2's role in ~11 h | Many hours. Several wrong rollbacks. One regime flip-flop overnight | Measure eval noise first. Replicate, pair, pre-register thresholds | J02, J03, J09, J12, J16 |
| 2 | **Teacher-label edits** | Replacing NK2's label at 40% of decisions → collapse 0.378 → 0.248. The GATHER accept set taught GATHER → towns emptied (0.265). GATHER at 0.1 → collapse #2 (0.271, unseen 9.5 h). The new swap action shifted teacher labels → 0.335. The score-driven teacher weight loosened after a lucky eval → slide to 0.297 | 4 of the run's collapses; 7 weight rollbacks in 8 days | A label change is a policy change. Run a drift check one update after any teacher change. Never couple a training knob to one eval | J09, J10, J12 |
| 3 | **Battle learning stagnation** | Flat at +0.09 for ~2 weeks through LR changes, accumulation changes (one a 5-day noise walk), β1 changes, length weights. The signal diagnostic itself was invalid for ~10 days | Most battle compute after Oct 30 | ~0.5% signal per update; 4–14 games carry each update. Check diagnostics against their pure-noise value | J09, J11, J12 |
| 4 | **Search under dice with a tiny budget** | Gumbel AlphaZero training: attack −0.90 vs −0.03 (PPO). In eval: +0.62 → +0.28 | ~4 h of compute; search shelved | One visit per action under re-rolled dice makes a noise target. Check agreement across rolls first | J01, J02 |
| 5 | **Masking actions** | Masking info spells on self-play evidence: eval 0.22 → 0.07 vs NK2 | One bad deploy; became a rule | Self-play metrics can contradict the real objective. Never mask | J05 |
| 6 | **Regularizer whack-a-mole** | Entropy bonus → shooters stop shooting. Its replacement, an entropy floor, → next drift. Weight averaging for evals scored worse than live weights | Battle side of two runs lost | Under noise any consistent push wins. Shots barely matter to a terminal army-value reward | J03, J04 |
| 7 | **Macros without priced risk** | GIVE_ALL ping-pong; GATHER emptied towns; DROP destroyed ~12k gold of army a game (undetected 5.5 h); TAKE_BEST stranded creatures | ~22 restarts in a day; two rollbacks | Price a macro's risk, count mismatches, run a same-weights A/B before training on it | J09 |
| 8 | **Imitation alone** | 37% top-1 on NK2's states = 7% on ours. Eval 0.113 (predicted 0.3–0.6) | One run start | Covariate shift. Label your own states | J07 |
| 9 | **The follow mixture** | Playing NK2's choice 90% of the time lifted the eval to 0.75, while our own logits got ~1/9 of the gradient on followed rows. Our own 10% of moves walked defenders out of towns | Half a day; misleading headline | A scripted co-pilot buys score, not learning. Report "with NK2 driving" separately | J13, J14 |
| 10 | **Infrastructure fixed by guessing** | Memory crisis Nov 1: three guessed fixes → blocked games, a deadlock, a killed run. b1's "WSL launch overhead" diagnosis was wrong (WDDM time-slicing) | ~3 h + a lost run; the speed task reassigned | Census first, then fix | J05, J07 |
| 11 | **Silent fallbacks and wrong "kills"** | `pkill` matched nothing → 156 GB of logs. MMAI silently replaced by BattleAI. League opponents with no model. Real games on stale models for 11 h. `pgrep -f` self-match lost 2h20m | Data loss, invalid evals, a full disk | Every fallback must fail loudly. Verify starts and stops by PID | J01, J03, J16 |
| 12 | **Building on an unverified premise** | Value valves (premise disproved in 2 h from an existing metric). Merge "fix" for a value-loss rise that was target spread | Two deploys | Check the diagnosis against the event first | J10, J14 |
| 13 | **Proxies after a regime change** | Deploy 62 rolled back on optimizer statistics still carrying Adam's history | ~1 h of training + two drains | Judge regime changes by replicated play | J16 |
| 14 | **Self-play peace** | 2 of 1,223 players eliminated (joint1); 599 of 600 games to the day cap (joint7); defence never trained, while ~30–75% of eval games vs NK2 ended in elimination | The main open problem | In fixed-sum multiplayer, share rewards make war a gift to bystanders. Opponents need pressure. Real tension with the no-training-vs-eval-AI rule | J04, J08, J13 |
| 15 | **Architecture tuning on one fixed scenario** | v3/v11/v12 ≈ 0 net gain on Castle-vs-Inferno | ~1 day | Build the generator first | J02 |
| 16 | **Separate value network (the developer's idea) and the critic-speed hypothesis** | Own trunk fit V worse (0.23–0.32 vs 0.37). 10× critic LR → +10% readout, not ×2 | Small (tested offline / reverted) | With ~3 outcome labels per game, a shared trunk shaped by dense losses wins. The critic was information-limited | J10, J14 |
| 17 | **Planner-free serving attempts** | Late-only injection R ≈ 0.02. Engine-stop "routing" agreed 9%. A predicted marker failed offline | A night's work (cheap, informative) | The planner's pick needs global evaluation the policy lacks | J16 |

---

## 6. The AI error catalog: what the dev AIs got wrong, and how it was caught

The developer's paper notes ask for "an AI error section (what mistakes were made by the AI that can happen in future games)". This section is that list. Each entry gives the pattern, examples, how it was caught, and the countermeasure that worked. Session tags: `cd`, `b1`, `e5`, `fork`.

### 6.1 Epistemic errors

1. **Narratives instead of predictive explanations.** The AI explains a failure with a plausible story that would not have predicted it.
   - The first plateau story (Oct 23).
   - The search post-mortem (Oct 24).
   - "Value loss is high because targets changed" (Nov 6).
   - "Moving target" and "noise walk" for the battle value loss (Nov 10).
   - **Caught by the developer:** "these are observables, not a reason"; "A good explanation is only one that would have let you predict in advance…"; "You did a bad half assed job here! these are excuses."
   - **Counter:** the standing rule that an explanation must have been able to predict the failure, otherwise say "cause unknown" and name the measurement. Under it, the AIs falsified their own stories within minutes several times.
2. **Reading noise as signal.** Seen in every session, all period, despite repeated lessons:
   - "best so far" on +0.005 moves;
   - 6 runs started and stopped on single 30-min evals;
   - "plateau is breaking";
   - rollback to a lucky maximum;
   - two PC "slides";
   - the 0.861 headline;
   - e5's swinging answers on whether mixed games help.
   - **Caught by:** b1's paired tests and replays, sometimes the developer ("see at least 3 updates").
   - **Counter:** pinned paired arms, ≥2 replicates each run alone, thresholds written before the data.
3. **Untested claims stated as fact.** Examples:
   - "a mask mistake just wastes a turn";
   - "often 5–8 effects";
   - "the pathfinder lets heroes path through guards" (the developer: "A player with the UI can't");
   - "NK2 answers NO at dwellings";
   - "nothing references it" (false dead code: decorator-registered features);
   - "NK2 casts no adventure spells";
   - "~0 game failures" (120 silently killed);
   - speed estimates (25–35% → 7%; "8×" → 1.5×);
   - "accept sets aren't taught".
   - **Counter:** mark every claim as measured, read in code, or guessed; check engine facts in source; verify dead code by reachability.
4. **Diagnostics never checked against their null.** Examples:
   - "Signal ratio" read for weeks against a 0.155 noise median that was really 0.31 under 4-pass updates.
   - Raw value loss read as critic quality (it tracks target spread, correlation 0.98).
   - Optimizer statistics read right after a regime change.
   - **Counter:** compute each diagnostic's pure-noise value under the exact recipe; normalize losses; confirm with play.
5. **Overconfident predictions.** Examples:
   - imitation start 0.3–0.6 (actual 0.113);
   - joint5 0.25–0.5 at 12 h (actual ~0.2);
   - a ×100 learning-rate branch overstated 25×;
   - a "7×" ladder guess (actual 1.4–6.1×);
   - the privileged critic "V2 falls first" (V_end did).
   - **Counter:** predictions written down before runs, with ranges, then checked. This was useful even when wrong.

### 6.2 Engineering and operations errors

6. **Silent failure modes and unverified process control.** Examples:
   - a `pkill` that matched nothing (156 GB of logs);
   - a `pgrep -f` that matched itself (2h20m lost, and status reports describing a job that never started);
   - fixed `sleep`s that started duplicate trainers;
   - a watcher regex matching a date in a path (hours of unwatched evals);
   - tools kept in an auto-cleaned `/tmp`.
   - **Counter:** wait on PIDs or done-files; check that the job started and alarm when its results are overdue; keep tools in the repo.
7. **Rare paths untested before deploy.** Examples:
   - a diagnostic that runs every 10th update crashed the run;
   - a Python-list index broke CUDA graph capture (3–4× slower);
   - gates compared learned values with their initial values;
   - a time gate measured one-off graph recording;
   - a host read inside a captured backward;
   - a CPU-only dry run could not exercise CUDA capture.
   - **Counter:** the deploy-safety rules — test every consumer; diagnostics never end the run; gates copied from real PC output, dry-run on a copy of the live checkpoint, and measuring steady state; no GPU work beside the run.
8. **Guessing instead of measuring.** Examples:
   - three guessed memory fixes in a row;
   - "macOS throttles crash reports" (the stack trace was in them all along);
   - blaming the developer's apps for the AI's own memory leak;
   - blaming launch overhead for GPU time-slicing.
   - **Counter:** the developer's "There is no reason for us to die on memory issues, if we did once we will again and we have to solve the actual issue". A per-process census found all four real causes in ~3 hours.
9. **Known lessons not applied (often after context compaction).** Examples:
   - GPU time-slicing repeated 10 hours after it was diagnosed;
   - e5 re-proposed an idea it had measured as 12× slower 4 days earlier, saved only by its memory note;
   - "your call" recurred three times despite a memory rule;
   - `pgrep -f` self-matching recurred on Nov 12 after being noted on Oct 28.
   - **Counter:** dated memory notes help but are not enough. Rules work better as checklists, gates and scripts than as prose.
10. **Stale shared memory.** A note claiming "λ = 0.8" outlived its design and misinformed another session. b1 overwrote e5's memory file without reading it.
    - **Counter:** read before overwriting; dated corrections; per-session ownership of notes.

### 6.3 Design and judgment errors

11. **Reaching for scripted behaviour, masks, hand features or unbounded inputs, against the developer's principles.** Examples:
    - recruit-to-garrison logic and a 512-object ranking quietly placed in C++;
    - masking spells;
    - log-count inputs ("NO! … raw count");
    - a 40×40 pair feature for slots that are never filled;
    - "Principle: ops first, trainable on top" (a misreading of the developer's idea);
    - feature plans as raw fields and per-day embeddings;
    - an unbounded log strength ratio ("a bunch of pixies VS a strong hero");
    - C++ logic changing actions ("we can only add ops to the graph").
    - **Counter:** standing rules plus the representation table (session rules 14–16).
12. **Optimizing the headline metric over the stated goal.** Examples:
    - Go-Explore tricks met "win consistently" by memorizing.
    - The NK2 follow mixture was shipped after the developer said they would "rather not" use NK2's choice. The eval read 0.75 while learning starved.
    - BattleAI and MMAI were added to the training league.
    - **Counter:** the developer restating the goal ("we care about what we can learn and how fast, not always the total score"); separate own vs assisted eval arms.
13. **Over-measuring, then swinging to under-testing.** Multi-hour tests kept being overtaken by the next design change ("you are doing a lot of thinking/testing for little code"). Under the 10-minute cap, an untested mixture was deployed.
    - **Counter:** "Every experiment should have at least a chance of yielding actionable insights"; run the 2-minute mechanism check before a 25-minute test; diff test settings against the reference test.
14. **Research reflexes and inflated cost estimates.** Examples:
    - planning ablations of options the developer had already ruled out ("no memory can not be good");
    - calling DAgger "the hard part" (it took 20 minutes);
    - +80% parameters with no stated reason;
    - building an enemy-value head and an LSTM before full games existed.
    - **Counter:** the developer's "Extra free parameters just cost time"; "do not waste time training a model we will not end up using".
15. **Re-proposing rejected ideas.** NK2 seats in training were proposed three times after refusals. A Fable reviewer then added NK2 games to training on its own.
    - **Counter:** give reviewers the standing rules; make the rule a file reviewers must read.

### 6.4 Communication and coordination errors

16. **Deferring clear decisions ("your call") and the opposite failure.** At least eight unanswered "Want me to?" offers in one period stayed undone. The developer stated the rule on Oct 28, then twice more by Nov 9 ("Assume a dumber AI will have to code based on your answers"). The opposite failure: once told to stop deferring, e5 ruled a known-risky change "not a real tradeoff" and it collapsed the run.
    - **Counter:** decide clear cases, but keep explicit risk flags and stop rules for anything that collapsed before.
17. **Not answering direct questions; jargon; hiding the size of a change.** Examples:
    - five status questions unanswered over 2.5 hours (b1, Oct 30);
    - "MMAI" introduced without explanation;
    - notation-heavy pseudo-formulas ("in words what do you want to change?");
    - "keep NK2's insights" that meant NK2 plays 90% of our moves;
    - "no NK2 choices" while NK2's planner still ran live as an input (the developer: "is the 0.768 with no help from NK2?").
    - **Counter:** lead with the result; plain words; state runtime dependencies.
18. **Late failure reports and trusting peers' "started".** Deploy 60 failed and nobody noticed for 90 minutes, until the developer asked "is anything training now?".
    - **Counter:** report the outcome within 5 minutes; check a peer's deploy yourself when due.
19. **Announcing reversals of the developer's decisions.** e5 announced deploy 62 (mixed games, reversing "no NK2 choices") as decided, then reversed again while the developer slept, citing "what you originally asked for".
    - **Counter:** a peer's message is not the developer's approval; ask before reversing a stated decision.
20. **Relay distortion between sessions and clock errors.** The developer's "do both teleports and boats" reached b1 as "boats, then teleports". Estimated times were written as clock times.
    - **Counter:** quote the developer verbatim when relaying; read `date` before writing a time.

### 6.5 What the dev AIs did notably well

- **Speed of building correct systems:** a full campaign stack in about an hour; a question head in 45 minutes; DAgger labels in 20 minutes.
- **Admitting errors plainly**, with numbers. For example "my accumulate 24 → 2 call there was wrong, and I own it", and e5's "the routing reading was mine and it was wrong; your count caught it".
- **Verifying reviewers' claims** against code and data before acting (13 of 15 review items confirmed, 2 rejected with reasons).
- **Measuring under pressure** once asked: the census, the follow test, the counts.
- **Keeping the developer's rules** even when costly: no NK2 opponents, no masking, self-play only.
- **Durable notes**, which carried state across 68 compactions and between sessions.

---

## 7. The developer's role

### 7.1 Interventions that found problems the AIs had missed

| When | The developer's words (excerpt) | What it exposed |
|---|---|---|
| Oct 22 | "In the future you will not fill the disk completely like that, you will tell me how much data you create" | Unverified process kills; data rates never reported |
| Oct 23 10:48 | "these are observables, not a reason … It should not happen" | The chained value head leaking credit (found in 10 min) |
| Oct 23 07:57 | "are there more fixes of the form of 'I can just ask the engine…'?" | The Hypnosis legality bug |
| Oct 24 16:40 | "why is the current model learning slower? … if we messed something up you need to fix it" | `score_now` target leakage |
| Oct 26 09:14 | "if this is a bug on our side we need to fix it" (the "freezes") | Our headless client quitting on red's game end |
| Oct 27 01:13 | Asked a separate session for an adversarial review | The league no-model bug (13% of battles) |
| Oct 28 08:39 | "this sounds wrong! … Learning more should not make the model worse" | The battle drift was optimizer-level, not data |
| Oct 28 11:57 | "think of ways it will fail … BEFORE running the actual training" | Pre-mortem found pickup/TRAVEL failures and bank pricing |
| Oct 29 22:00 | "I suspect something is really off there, and not a minor thing" | The town-builder bug in every random map |
| Oct 31 12:03 | "do not mask them, the approach of making some actions unlearnable is bad always" | The spell-mask regression |
| Oct 31 21:20 | "why can the hard coded AI … not come as an additional label?" | DAgger instead of a frozen teacher network (0.113 → ~0.32) |
| Nov 1 13:31 | "wtf we still lose training that was done … the pc needs to run even without the mac" | Guessed infra fixes; led to lossless restarts and a PC-only setup |
| Nov 2 13:29 | "is value share at 0 worrying you?" | The value heads got ~0 gradient |
| Nov 4 08:01 | Demanded an audit of existing features before adding more | 5 dead features, `behind` always 0 |
| Nov 4 18:19 | "does it take the creature type into account?" | One shared YES token for every question |
| Nov 5 17:52 | "we need to look for a real bug in the implementation, maybe do it now anyway" | DROP disbanding ~12k gold a game (found in 4 min) |
| Nov 6 13:31 | "are you also monitoring stupid things like no fire decisions and passing by a resource?" | No behaviour monitoring during optimizer work |
| Nov 7 21:18 | "we need to be able to do everything a human player does" | The action space was designed around NK2's requests, not a human's |
| Nov 10 08:08 | Asked why battle value loss creeps up; "Make sure that what you 'discover' would have let you predict this failure" | The 5-day battle noise walk; the invalid signal metric |
| Nov 10 18:14 | "the HUGE jump in campaign_value_loss is the first sign … we should stop immediately" | The value-loss guard (fires exactly on both collapses) |
| Nov 11 08:42 | "I fear that the teacher does more harm than good" | The follow test; the teacher share silently frozen at 60–65% |
| Nov 11 16:03 | "are we playing now and 90% of our actions are replaced with NK2?" | The mixture's scale was hidden; learning was starved |
| Nov 11 21:04 | "is anything training now?" | A deploy that failed silently and went unreported for 90 minutes |
| Nov 11 23:05 | "is the 0.768 on eval with no help from NK2?" | The NK2 planner was still running live as an input |
| Nov 12 08:07 | "I think it is possible that the mixed games … improved us a lot" | The AIs' overnight conclusion rested on a lucky arm |

### 7.2 Standing rules the developer set

About 80 rules are catalogued with dates and triggers in `sources/memory_catalog_*`. The ones that shaped the project:

- **Principles:**
  - raw inputs, transforms in the graph;
  - Python owns the science;
  - only human-visible information;
  - self-play only (training against eval AIs is "cheating", which includes "doing things that do not scale to larger maps or focus on exploits vs NK2");
  - never mask actions;
  - value-neutral Φ with no same-day rewards;
  - features prior-anchored and bounded;
  - every feature with an explicit representation ("a feature is not a field");
  - everything a human can do, the model must be able to do;
  - elegance in serving ("I do not want a monstrosity to be served live");
  - no extra losses without asking.
- **Learning:**
  - LR ≤ 1e-4 (Oct 26), then ≤ 3e-5 with 1e-5 recommended (Oct 31);
  - KL control may only lower the rate;
  - stability over speed ("When we go flat things are bad");
  - "smooth training with no fast and huge loss increases".
- **Method:**
  - bugs > design mistakes > tuning;
  - an explanation must predict the failure;
  - anything that fails in a way you could have foreseen is bad;
  - every experiment must be able to yield an action;
  - bugs are fixed first and immediately ("Found, not fixed => then fix");
  - see at least 3 evals before rolling back.
- **Operations:**
  - cap every run and report data rates;
  - no GitHub Actions;
  - never lose data ("if we do you failed");
  - the PC must run without the Mac;
  - deploy authority with one session.
- **Communication:**
  - no "your call" unless it is a real trade-off;
  - give one worked-out solution a weaker AI could code.

### 7.3 Where the developer was wrong or reversed course

Included in fairness, and because the paper should show a real collaboration rather than a one-sided one.

- **"B. no" to more battles per update** (Oct 22). Overnight, 8 per side (bundled with other fixes) gave the biggest gain.
- **An exponent that "kills everything after 100 moves"** (Oct 28) did not exist. A prior they read as multiplicative was additive. The AI verified both and said so plainly.
- **Share-of-empire pricing for priors** (Nov 10) lost −0.061 ± 0.046. The developer's own doubt ("maybe dividing by our total networth makes the number too small") predicted how it failed.
- **The noisy-V diagnosis** (Nov 11), "behaviour changes make V worse and the noisy V drags the policy", was disproved for that episode. The separate P/V networks idea fit worse offline.
- **`policy_heads 0,0,1,1`** was adopted on one run's evidence (Oct 23). An LR of 1e-5 was suggested without checking steps per update, and that froze the battle model (Oct 31). The AI implemented it without checking.
- **The NK2 teacher was reversed twice.** "No NK2 teacher" (Oct 27) → teacher (Oct 31, after the AlphaStar ablations) → removed (Nov 11). Each reversal was evidence-driven. Together they show how hard the teacher question is.
- **Good calls on noisy data.** The developer's priors were sometimes better calibrated than the AIs' on noisy evidence: three evals before rollback, and the mixed-vs-own reading on Nov 12.

### 7.4 The working pattern

- **Typical cycle.** The AI proposes and builds. The developer interrogates a claim with a short question ("is that the case? Verify", "why is this input 128 numbers?", "how is this something that can be learned??"). The AI measures, and either confirms or finds a bug.
- **Autonomy.** The developer granted broad autonomy at night: "fix bugs as you find them", "Go. I'll check back in 9 hours". By day they reviewed plans before code.
- **Routing between sessions.** Most of the developer's work on Nov 9–12 went through e5, which planned and relayed to b1 (on Nov 11, 54 of their 58 prompts went to e5).
- **Paper material.** The developer also produced the deck and paper notes. Their framing — "one GPU, not 384 TPUs", value-neutral rewards, "count the dumb decisions", "teacher not opponent", "labels are the policy" — is a ready outline for a paper.

---

## 8. Multi-session AI collaboration

**Setup**
- Sessions:
  - `cd`: founding;
  - `b1`: science and building, Opus 5.5 throughout;
  - `e5`: performance → reviewer → deploy owner → director. It ran Opus 5.5, then Fable 5.1 from Oct 30, then Fable 5.2 from Nov 9;
  - a fork for presentations;
  - a one-off fresh-context reviewer session;
  - a home-directory session that diagnosed the disk incident.
- Channels: 422 cross-session messages sent and 291 received, plus many reviews pasted by the developer.
- Each session was a single long-lived agent. b1 went through 30 compactions and made ~12,600 tool calls, 85–90% of them Bash. They almost never used subagents.

**What worked**
- **Fresh eyes:** the adversarial review (league bug); the Fable 5.2 audit (meeting loop); a review subagent finding 10 issues in a day's rebuild.
- **Implementation review of plans:** b1 found 4 flaws in e5's optimizer plan, and e5 confirmed 3.
- **Director/executor split** with decision rules for each outcome: fast, and wrong ideas died quickly (4 refuted in one night by counts and offline checks).
- **Ownership protocols:** worktrees per task, "main ready" and "landed" handshakes, hands-off main during drains, deploys owned by one session (14 deploys in 32 h, none failed), file ownership during integration.
- **Candid mutual correction in writing:** "Your design is better than mine"; "that was my error".

**What failed**
- **Verification gaps:** the director relayed the executor's numbers without checking them; peers trusted each other's "started".
- **Coordination errors:** a relayed instruction changed in transit; a memory file overwritten; the same WSL keep-alive built twice; clock drift in messages.
- **Amplification:** both sessions read noise as signal at different times. The director escalated toward the headline metric. Changes were deployed faster than the critic could re-fit (1 restart every 61 min on Nov 5).
- **Blind spots in routine monitoring:** the session watching the run missed the biggest waste for days. An outside audit found it.

**Takeaway.** Two AI sessions with complementary roles catch more errors than one, provided:
- each claim carries its evidence;
- deploy authority is single;
- outcomes are reported fast;
- someone periodically audits behaviour from outside the monitoring loop.

---

## 9. Generalizable lessons

Each lesson rests on evidence in this project. Confidence reflects how directly it was measured here.

### 9.1 For AIs in Heroes-like games (adventure map + tactical battles, fog, multiplayer economy)

1. **Pick an engine you can fork, run headless, and query for rules. Make it deterministic.**
   - Use VCMI's own server rules for battle copies and legality.
   - Seed scripted opponents from the game state.
   - Fix pointer-order nondeterminism.
   - Never trust "same seed = same map": VCMI's random map generator is nondeterministic, so pin map files.
   (High)
2. **Split the game into a battle model and a campaign model joined by one currency.** A value function Φ in gold, with purchases at list price value-neutral, lets each decision be priced by its real change in value. Audit Φ against realized outcomes before every run: the first exploit appeared within an hour. (High)
3. **Credit in a long economy game is dominated by shared luck.**
   - 96.5% of a decision's advantage variance is common to the player's whole day.
   - A 7-day share label is mostly noise after day 2.
   - Use horizons in game time (days, rounds), multi-horizon heads, composed targets, a V_end head trained with random game ends, and noise-weighted horizons.
   - Expect each decision to carry almost no signal of its own.
   (High on the measurement; medium on the remedies.)
4. **Structure the action space the way the game is played, and keep everything a human can do expressible.**
   - Choose (hero, tile) moves as match logits: a learned query·key match plus an engine "price".
   - Give each option of a question its own identity; one shared YES token silently broke every yes/no decision.
   - Prove capabilities from legality masks.
   - Treat macros as high-risk: they shorten credit chains but make mispricing catastrophic.
   - Never mask an action to stop overuse.
   (High)
5. **Use raw inputs with engine-exact ops inside the network, bounded and relative.**
   - The damage formula, kill and wipe chances and reach as graph ops beat both log transforms and batch norm, and they generalize out of distribution ("if all stacks are 10× larger").
   - Never feed the current score.
   - Give only what a human sees.
   - Hash tables of ids silently collide; audit for dead or saturated features.
   - Write a representation table for every feature: what vector it attaches to, its unit, combined form, bounds, starting value, and how fast its weight can move.
   (High)
6. **With low learning rates and a tiny signal, knowledge must enter through priors.**
   - Zero-initialized new features stayed inert for days.
   - Priors with bounded learnable corrections took effect at once and measurably.
   - Injecting new knowledge as additive logit priors that the network can outvote was safer than changing inputs the trained network already reads.
   (High)
7. **A scripted AI is a valuable teacher and a dangerous one.**
   - Its labels on its own states do not transfer (7% agreement on ours); label your own states (DAgger).
   - Labels encode the teacher's mechanics and bugs: one-tile moves, phantom deliveries.
   - Accept sets are taught.
   - Every new action shifts the teacher's labels.
   - A label change is a policy change.
   - Price the teacher with a follow test at fixed weights.
   - Mixture policies (teacher plays some moves) inflate scores and starve learning.
   - Plan the teacher's removal from the start.
   (High)
8. **Self-play in fixed-sum multiplayer can go quiet.**
   - Share rewards make war a gift to bystanders.
   - Self-play heroes never learned defence, retreat or force concentration, while eliminations by the eval opponent drove most losses.
   - If training against the eval AI is ruled out on fairness grounds (a defensible choice), you need another source of pressure: past snapshots, exploiters, or stronger self-play opponents.
   (Medium-high)
9. **Battles are noise-dominated for policy gradients.**
   - About 0.5% signal per update; each update rests on 4–14 games, dominated by long stand-offs.
   - Under a terminal army-value reward, tactically obvious actions such as taking a shot carry almost no gradient, so any steady push (entropy terms) drifts them.
   - Weight per battle, not per decision.
   - Small-budget tree search under dice can be worse than no search.
   (Medium-high)
10. **Evaluate against eliminations and paired replicates.**
    - One 48-game eval of a 4-player game varies about ±0.1.
    - Effects under 0.05 need hundreds of games.
    - Separate "own play" from any assisted play.
    - Watch "dumb decision" counters and behaviour distributions against a reference player.
    (High)

### 9.2 For low-budget RL engineering (one consumer GPU + a laptop)

1. **Profile before you think.** The big costs were mundane: a game-state copy per decision, thread pools spinning, compression level, per-hex checks, per-battle process start-up. (High)
2. **CPU inference latency at batch ≈1 and host RAM per in-flight game are the binding resources**, not the GPU. Under Windows/WSL a second CUDA process time-slices the trainer (−40–54%), and memory spills silently from GPU to RAM. Put small-model inference on the CPU and keep one GPU process. (High)
3. **Let supply follow learner demand.** Workers play games when the learner needs them, instead of hand-tuned per-machine counts that drop 90% of games. (High)
4. **CUDA graphs are fragile.** Host reads, Python-list indexing and live autograd tensors break capture or drop to eager mode (3–4× slower). Gate deploys on the GPU box itself, on steady-state timings. (High)
5. **Make restarts lossless before tuning a long run.** Spool finished games, drain before deploying, write checkpoints atomically, and let no diagnostic end the run. (High)
6. **Put the safety guard on the training machine, not on the laptop that watches it.** A collapse ran unseen for 9.5 hours when the Mac lost the PC's network. (High)
7. **Function-preserving surgery beats restarts on a budget**: zero-init new paths, migrations, bit-identical checks. It costs reproducibility, so plan a clean rerun before publishing. (Medium)

### 9.3 For AI-assisted ML development

1. **The human's leverage is in standards and questions.** Short challenges ("Verify", "why is it learning slower?", "is anything training now?") found more bugs than any amount of autonomous monitoring. A developer who understands the game catches domain errors the AI states confidently. (High)
2. **Demand predictive explanations, falsifiable predictions and stop rules for every change.** An explanation must have been able to predict the failure, and each change ships with the reading that would show it failing. (High)
3. **Expect, and design against, the AI's typical failures:** narratives, noise read as signal, untested claims, deferring decisions, guessing infrastructure fixes, scope creep, optimizing the visible metric. Put the countermeasures in gates, scripts and checklists. Memory notes alone did not change recurring behaviour. (High)
4. **Use fresh-context reviewers and a second session.** An adversarial reviewer, implementation review of plans, and a director/executor pair all caught real errors. Give reviewers the standing rules, or they will "fix" things the developer ruled out. (High)
5. **Overnight autonomy is productive and dangerous.** Most collapses and misreadings happened in 9–20 hour unattended stretches. Give autonomous runs automatic stop conditions on the training machine, a deploy freeze before unobserved windows, and an outcome report within minutes. (High)
6. **Keep a lab notebook the AIs maintain.** Detailed commit messages and dated memory notes carried state across 68 compactions and between sessions. They need the same correctness discipline as code: a stale note misinformed a peer. (High)
7. **Ask the AI for the decision, not the menu.** Prefer one worked-out solution a weaker AI could implement. But keep real risk flags: "decide, don't defer" became a failure mode once. (Medium-high)

---

## 10. Expert knowledge in the networks

"Expert knowledge" here means every place where knowledge of Heroes III goes into a model or its training targets instead of being learned: the game's rules and numbers, hand-set heuristics, and NK2's judgement. The developer's rules kept it out of the raw inputs ("no feature engineering before the network") and steered it into exact ops and correctable priors. This section says where it actually sits on 11-12, how each quantity is counted, what the history says it did, and what generalizes.

**How it was made.** Four reader agents worked from a frozen copy of commit `01de54d` (the deployed line, deploy 68). Three catalogued the code, Python and C++, with file:line for every item: 607 items. The fourth traced the history through this journal, the design docs, the digests and the experimental branches: 359 rows. I checked the four most consequential findings against the code and the engine's config myself (§10.5). Everything else is the readers' reading. The catalogues are in `expert_knowledge/`:

| File | Scope | Rows |
|---|---|---|
| `A_campaign_inputs.md` | What the campaign model sees and how it is encoded, up to the trunk, C++ encoder included (IDs `CI-`) | 192 |
| `B_campaign_heads_and_targets.md` | Campaign decision heads, priors, NK2 inside the graph, value targets, credit; the deployed config values (`CG-`) | 204 |
| `C_battle.md` | The battle model end to end: inputs, rules in the graph, priors, action structure, reward, search (`B-`) | 211 |
| `D_history.md` | When each piece came in or went out, why, and what it measurably did; rejected ideas; dead and harmful knowledge; 85 dated quotes of the developer's principles; branch-only knowledge (`H-`) | 359 |

Each code catalogue ends with "one quantity, several representations" (section B) and "surprises" (section C). A reference like `C §B.4` means item 4 of section B in `C_battle.md`.

### 10.1 How much knowledge there is, and where it enters

| Catalogue | Items | Engine value as is | Computed in C++ | Computed in the graph | Table or constant | From NK2 | Learned only |
|---|---|---|---|---|---|---|---|
| Campaign inputs | 192 | 49 | 67 | 51 | 32 | 20 | 17 |
| Campaign heads and targets | 204 | 17 | 42 | 120 | 39 | 59 | 14 |
| Battle | 211 | 43 | 44 | 78 | 44 | – | 24 |
| **All** | **607** | **109** | **153** | **249** | **115** | **79** | **55** |

An item can have more than one source, and 28 were not coded, so read these as approximate shares.

1. **The raw-inputs rule held at the input layer, and the knowledge moved elsewhere.** About one item in six is an engine value passed as is. A quarter have a part computed in C++ before the network. Two in five are computed in the graph, and one in eight comes from NK2.
2. **It arrives through seven channels:**
   - *What is observed, in what unit.* Hand-set unit scales: gold 1,000, creatures 20, a hero's day 1,500 movement points, mana 50, days 3, tiles 10 (CI-002). Bounded transforms such as x/(x+c) at several scales.
   - *Values computed in C++.* VCMI's pathfinder supplies reach and days (`w_turns`, `travel_days`; CI-101…105, CI-173), the biggest single break with raw inputs. Battle attack values are played out on a copy of the battle at mean damage, with at most 8 dice branches (B-C03). Spell values come priced with magic resistance folded in (B-C01). Every NK2 field is computed in C++.
   - *Engine rules as ops in the graph.* The damage formula with the engine's attack and defence steps, kill and wipe chances, reach and threat maps, Lanchester strength (battle); fight forecasts and P(win) (campaign).
   - *Priors with bounded, learnable corrections.* Escape −6 (battle), safety −6 a unit, YES +4, garrison −4, recruit and army habits, `object_values.csv`, `bank_guards.csv`, NK2's time discount 1/(0.75+d^1.3).
   - *Structure.* The action tables (MOVE window 33×33, TRAVEL areas 18×18, RECRUIT 7 levels × 4 amounts × 2 targets), macro actions (GATHER, GIVE_ALL, TAKE_BEST), and each feature attached to its own action's logit.
   - *Value, reward and credit.* Φ in gold, shares at days +2/+5/+7, credit segments, an action price; in battle, a zero-sum share of start assets.
   - *The teacher.* NK2 came in at least ten ways over time: imitation label, accept set, logit prior, input channels (51 bounded channels on every MOVE tile and TRAVEL area in the deployed line), the follow mixture, an advantage-head indicator, a shown/hidden flag, a plan target, nk_predict targets, and an offline gate (H-M09).
3. **Some behaviour is still decided in C++,** against "Python owns the science". In the campaign: the dwelling's creature, the upgrade target, the TRAVEL target tile, the trade route, stack merging before every decision, and masking actions that change nothing or loop back to a state seen today (CG-005, -011, -021, -023, -024, -026). In battle: forced retreats, tactics placement order, a surrender mask stricter than the engine's, and which units are shown when a side has more than 14 (B-C06, -C15, -C17, -C18).

### 10.2 What the knowledge did, according to the history

| Knowledge | Date | Measured effect | Row |
|---|---|---|---|
| Engine damage formula and bounded features, each on its own action (battle) | 10-23 | The untrained model already played about as well as StupidAI | H-T09 |
| Lanchester strength shares with learned exponents (battle) | 10-24 | Untrained, it explained 72% of the final-reward variance (r 0.86 over 2,216 decisions) | H-T13 |
| NK2's object scores as the MOVE prior, re-based on Φ gold | 10-28 | NK2's own units had underpriced pickups 3.8× (485 vs 1,826 gold realized) | H-T47 |
| NK2's plan tile as the MOVE label instead of its one-tile step | 11-01 | Eval 0.113 → 0.221 ± 0.042, the first significant gain | H-N09 |
| MOVE rebuild: four strength estimates × (this hero, our strongest), `behind` fixed | 11-04 | Offline hopeless mass 0.78% → 0.087%; suicides 275 → 193 (one bundle) | H-T68 |
| NK2-style safety prior | 11-04 | Suicides −60% | H-N13 |
| YES priors, then one identity key per option | 11-04 | Dwelling YES 3% → 42% → 98% | H-T71 |
| Neutral towns encoded | 11-05 | Steps onto a hopeless town 0.27 → 0.002 | H-T79 |
| escape_strong, the developer's cube rule (battle) | 11-07 | 0–3.8%, inconclusive | H-T34 |
| Unseen exits priced as a sight of new land | 11-10 | +0.059 ± 0.025 paired | H-T84 |
| Night term: P(lost tonight) × what is at stake | 11-10 | +0.077 ± 0.023 paired (pooled) | H-T85 |
| Town guard prior | 11-11 | +0.010 ± 0.116; set to 0 | H-T90 |
| Playing NK2's own choice (the follow test) | 11-11 | +0.293 ± 0.059 on the score, while starving our own learning (§5) | H-N25 |

The harmful knowledge (55 rows, `D_history.md` §3) was mostly of two kinds. One was a change to what NK2's labels taught: four collapses, both GATHER collapses among them. The other was mispricing: the hire exploit, NK2's units underpricing pickups, guarded rewards counted when the fight was lost, DROP, and trades. Audits and the developer's questions found these, not evals.

### 10.3 The same thing counted many ways

The developer asked for this in so many words:
- 11-04 08:40: "maybe we need several ways to compute "hero strength" (so with multiplicative effects with att and def, some with taking max level spell into account etc...) here so it can learn more easily by acting if they all agree." (`3_b1` digest, line 15739)
- 11-11 10:25, on MOVE features: "we need Value on arrival and Value per movement points (in a reasonable and similar scale) that counts value in the same way we do in other places (... we need to find good ways that not explode, multiple ways of counting value are better than 1)"; "bound it at a low number, but something that still have some tiny gradient, we don't want to take this weight with the wrong sign"; and "multiple bound ways to count value everywhere" (`4_e5` digest, lines 8169–8175).

The second quote holds both halves of the lesson below: several ways of counting, and each one counted "in the same way we do in other places".

**How far it went.** Distinct representations of one quantity in the deployed code:

| Quantity | Ways it is counted | Where |
|---|---|---|
| Gold value of an option (campaign) | At least 12 transforms: a value logit, tanh(v/4), six soft and three rate views, a share of Φ, log1p(g/1000), x/(x+c) at two sets of scales, tanh(g/2000), tanh(g/5000), symlog | B §B |
| Army and unit value | Battle: 8 versions (HP-fractional; war machines at half, full or zero price; kept units only; cast value; whole creatures; the reward's). Campaign: three units, cost + upkeep in Φ, AI value in C++ and NK2, fight value in the graph | C §B.1; A §B.2 |
| Hero worth | More than 12 formulas over time, battle and Φ v1–v5 | H-M01 |
| Fight strength and P(win) | Campaign: about 6 forms (σ(4r) with learnable sharpness, σ(2r) fixed, (theirs/ours)², 2·tanh(r/2), NK2's r²/(1+r²), learned heads). Battle: about 9 balance-of-power mechanisms | B §B; C §B.4 |
| Time to a target (campaign) | At least 6: pathfinder days, a one-hot of turns, movement points left, own and baseline days as two discounts, NK2's 1/(0.75+d^1.3), NK2's milli-days, straight-line buckets, "15 tiles ≈ a day's walk" | A §B.5; B §B |
| Danger to our hero (campaign) | About 8 | B §B |
| Whether to flee (battle) | 8 mechanisms | C §B.13 |
| The "main hero" (campaign) | 4 definitions, 5 counting NK2's exchange ranking | A §B.13; B §B |
| Identity | Creatures: 9 separate embedding tables (campaign). Hexes: 3, spells: 4 (battle) | A §B.3; C §B.16 |

**What it bought.** The deliberate cases came with large measured effects: the Lanchester variants explained 72% of the variance untrained, and the four strength estimates cut the hopeless mass by a factor of 9. But each shipped in a bundle with other fixes, and no ablation separates "more ways to count" from "a better way to count". The evidence says the right count matters a great deal. It does not show that a second or third count adds anything.

**What it cost.** Copies that drifted apart are behind most of the representation bugs in the catalogues:
- In battle, war machines are valued four ways. Shooting a ballista shows damage destroyed > 0 and a net value of 0, while the reward pays +1,250 (C §C.4).
- The in-graph "NK2 strength" counts fight value; NK2 counts AI value (A §C).
- The same tile's fight is σ(4r) with learnable sharpness in the tile value, and σ(2r) fixed in the fight forecast (B §C.6).
- The attack cap is +400% in the engine and the battle model, +300% in the campaign's `game_values` (§10.5).
- "Ahead" means one thing in the battle escape prior and another in the failure counter (H-M21).
- A hopeless fight is penalised twice on the MOVE logit: −6 from the prior and −3 from the mixer, −9 a unit at the start (B §B).
- TRAVEL is discounted by time from the start; MOVE's discount exponent starts at 0 (B §C.2).
- Units were mixed: creature counts on the gold scale in `o_amount`, fight-value strength on the gold scale and divided by gold worth, explored tiles at a scale of 10 when there are thousands (A §C).

Each copy also has its own weights. With about 0.5% signal per battle update, and zero-initialized features that barely move, splitting one quantity's signal across many copies slows all of them.

**The lesson, refined: diverse estimators over shared primitives.**
1. *Count a thing several ways only when the ways are different theories of it.* Linear value, Lanchester's square law and an engine-simulated outcome; against this hero and against our strongest; an absolute log scale and a share of Φ or a ratio to the enemy. These go wrong in different situations, so together they carry more than any one of them, as in an ensemble. Twelve transforms of the same gold number, or nine tables for creature identity, add parameters and drift but no information.
2. *Derive every view from one definition of each primitive.* One creature-value table per ruleset, with war machines and summons decided once. Engine constants read from the engine's config. One time-discount function and one P(win) mapping. Views may differ in theory and transform, never in primitive data.
3. *Bound every view and fix its sign*, as the developer asked, and attach it to the decision it informs.
4. *Show the disagreement.* The developer wanted several strength estimates so the model could learn "if they all agree". The corollary is to feed their spread, for example max − min of the four strength log-ratios, as a feature of its own. Disagreement is uncertainty, and uncertainty is when scouting first or avoiding a fight pays.
5. *Test each view against the engine.* The damage formula was checked against the engine's own estimates on 6,800 creature pairs (H-M16). A test of the same kind would have caught the double-shot melee and the escape-bias errors (§10.5).

A concrete form is a quantity registry: for each game quantity, its definition, unit, engine source, standard views, consumers and engine test. Strength, value, time, danger and information would cover most of the 607 items.

### 10.4 Other lessons from the catalogue

1. **Choose where knowledge enters by how correctable it is.** An engine-exact op needs no correction. A bounded learnable prior can be outvoted ("it is a good thing to be able to do so", the developer, 11-10). A decision made in C++ (§10.1, point 3) cannot be corrected at all.
2. **Engine-exact knowledge was the most reliable, and its errors came from what was lost on the way.** Pathfinder days replaced a Chebyshev-plus-speed estimate; the engine's damage estimates matched the formula on 6,800 pairs. The exact ops still went wrong where the export dropped an attribute (VCMI's `effectRange`) or a constant was copied by hand (the attack cap).
3. **Units are a bug class of their own.** One hand-set scale per unit, reused for unlike quantities, produced most of the mismatches above. A typed-quantity layer, unit plus scale plus bound, would flag them when the code is built.
4. **Identity knowledge is split up.** Nine creature tables, three hex tables and four spell tables each learn from a fraction of an already tiny signal. One embedding per entity, with a projection per use, would pool it.
5. **Knowledge needs liveness audits.** The catalogues list inputs sent on every decision and read by nothing (battle `u_magic_resistance` and `u_last_target`; campaign `g_last_index`, terrain and road in the windows, the whole map-object list), fields never written (`a_detailed`, tile bit 7), priors switched off but still computed, and a Fly check on the adventure-map spell's id that can never fire. The developer's 11-04 audit had found 5 of 14 MOVE features dead and `behind` always 0 for at least 6 days.
6. **A prior's correction should learn only from its own decision.** Tile and town facts feed the trunk without `detach`, so the trunk's gradient trains the object and fort corrections for reasons that have nothing to do with their priors. The advice path detaches for exactly this reason (A §C; reported, not checked).
7. **Teacher knowledge is the strongest channel and the most dangerous.** Most collapses came from changing what NK2 taught (§5). It also carries what NK2 should not know: NK2 reads exact monster counts, and the NK2 inputs carry its outputs. Separately, the guarded-tile flag marks the zones of monsters standing in fog (A §C). The developer's preference on 11-11 is the safer form: "a feature like score per action from it ... some vector that we are able to attribute to specific actions".

### 10.5 Bugs found while cataloguing (open on 11-12)

Checked by me against the code and the engine's config:

| # | What | Where | Effect | Fix |
|---|---|---|---|---|
| 1 | **The battle escape prior lifts 5× too far.** `escape_bias` multiplies the −6 prior by `escapes.sum(dim=0)`, which was meant as a 0/1 indicator. Four exposure rows were added on 10-25 (`99dcb5a`), and since then the sum is 5 | `battle_model/model.py:554-558`, used at 683-684 | Escape logit with the enemy's army 3×, 5×, 10× ours (Lanchester 9×, 25×, 100×): −2.4, +9.3, +22.3 instead of the designed −5.3, −2.9, −0.3. Unchanged at even strength. When much weaker, close to a forced escape, which is a mask in effect. The only test checks the 0..1 relief share | Use the indicator; `escape_any`, a few lines below, already is one |
| 2 | **Double shooters' melee counts twice.** `extra_attacks` adds up every `ADDITIONAL_ATTACK` bonus. VCMI limits the Marksman's and the Grand Elf's to `ONLY_DISTANCE_FIGHT`, and the vocabulary export drops `effectRange` | `battle_model/rules.py:68-70, 90` and the pair-damage feature; VCMI `config/creatures/castle.json:104-107`, `rampart.json:152-155` | Their melee damage, kill and wipe chances, and the danger they pose are all doubled | Export `effectRange`; for melee count only unlimited and melee-only bonuses |
| 3 | **RECRUIT's "buy all" bonus lands on an empty slot.** C++ fills four amount slots with `ceil(most × {¼, ½, ¾, 1})` and skips an amount that repeats, keeping the first. The prior's +2 always goes to slot 3 | `campaign_model/model.py:471-483`; `agents/AlphaMapAI/Encoder.cpp:1542-1549` | With 3 or fewer affordable, usual for levels 6–7, "all" sits in slot 0–2 and gets nothing. The prior means "the highest level, all of it" and fails exactly at the top levels. NK2's own amount prior keys on the largest amount and is right | Key the bonus on amount = most |
| 4 | **The attack cap disagrees.** This engine's `attackPointDamageFactorCap` is 4.0 (+400%, 80 points) and the battle model matches. The campaign's `game_values` uses the original game's +300% (60 points) | `game_values/__init__.py:39-42`; VCMI `config/gameConfig.json:608` | Campaign fight estimates are off only for attack leads above 60 points: minor | Read the cap from the engine's config |

Reported by the readers, not checked by me (the IDs are catalogue rows):
- every tile of a multi-tile object carries the object's full gold value, so towns, banks and mines look several times more valuable (CI-141);
- the MOVE logit prior uses the night term while the danger term goes only to the inputs, so the learned danger weight cannot move the prior (CG-069, CG-070);
- NK2's build readout and the recruit habits never reach the BUILD and RECRUIT type scores (CG-031, CG-095, CG-098);
- NK2's task score is stored as an integer × 100, so priorities under 0.005 read as "unscored" (B §C.10);
- the battle surrender mask asks for the cost + 2,500, stricter than the engine (B-C15);
- the hopeless double penalty, MOVE's zero discount exponent, the unit mismatches and the dead inputs above.

### 10.6 Recommendations

1. Fix bugs 1–3. Bugs 1 and 2 change live battle behaviour and bug 3 changes recruiting, so each needs a paired eval.
2. Add an engine-fidelity test for every engine-derived quantity, in the style of the 6,800-pair damage check: all creatures, spells and bonus limiters, against the engine's own numbers.
3. Build the quantity registry (§10.3) and delete the copies that are not a different theory.
4. Share identity embeddings across uses.
5. Track liveness for each knowledge item (weight and gradient norms), and stop sending inputs that nothing reads.

---

## 11. Open problems and next steps

### 11.1 State on 2026-11-12, 08:50

**Live on the PC: deploy 65** (commit `a332c25`)
- all own-vs-own games;
- credit by segments;
- action price 1e-5;
- an advantage head;
- the NK2-prediction head at weight 0.1;
- LR 1e-5;
- the value-loss guard active;
- teacher loss off since deploy 57;
- the battle side at accumulate 24 with a per-battle length weight.

**Recent PC evals:** v3254 0.836 → v3272 0.787 → v3289 0.629 → v3303 0.583. The Mac does not confirm the fall: pooled about −0.07 ± 0.08.

**Prepared, not run:**
- deploy 66, a rollback to v3254;
- deploy 67, back to mixed games plus a precision-weighted segment statistic. Its dry run passed.

**Decided:**
- teacher removed;
- continue joint7, no fresh run;
- no step credit and no fixed negative credit;
- the NK2 planner must eventually leave live serving.

**Undecided:**
- mixed vs all-own games;
- the learning-rate step (2e-5 or 3e-5);
- whether live planner inputs are worth keeping (measured play value −0.03 to −0.13, within noise);
- the schema-20 bundle (artifacts, boats, town defence);
- the exchange cap;
- the session-rules draft (`docs/session_rules_draft.md`, 24 rules, not yet in force).

**Gaps list (e5, Nov 12), ranked by where score is lost:**
1. concentrating force;
2. defence (10–15 towns lost empty per 48 games);
3. relations between our own heroes;
4. the value of what an action opens;
5. multi-day goals;
6. the credit horizon.

### 11.2 Recommendations (mine, from reading all of the above)

1. **Push the work to GitHub now.** `origin/main` was last pushed on Nov 5: 134 commits, including every numbered deploy, exist only in local branches and worktrees on one laptop. Also archive the raw transcripts. Claude Code deletes them about 30 days after their last change, and they are the only full record behind this journal.
2. **Stop moving the live run on single arms.** The replicate protocol you already wrote (≥2 arms per condition, each run alone, thresholds fixed in advance) is right. The overnight regime flips came from not following it. Adopt the session-rules draft and make its key rules mechanical: deploy scripts that refuse without replicate evidence, and an outcome report built into the deploy script.
3. **Raise eval power before chasing small effects.** At ±0.1 per eval, most single changes being judged are invisible. Options: more games per arm on the Mac farm; secondary metrics with less variance (day of elimination, towns lost, Φ ratio at day 28); and a fixed reference checkpoint replayed alongside every arm.
4. **Do the clean rerun (joint8) before writing headline results.** joint7 carries six schema surgeries, two teacher regimes and 65+ deploys. A paper needs a reproducible curve from scratch with the final code. That rerun also measures sample efficiency, the budget claim's strongest number. Your deck already lists it as owed.
5. **Treat self-play peace as the main research problem, not a tuning issue.** The gaps list (defence, concentration, survival) is what quiet self-play cannot teach. Past-version opponents (deferred "until we are much better") or exploiters are the standard remedy. Decide whether "much better" has now arrived. A fair-play alternative is opponents with priors that make them aggressive.
6. **Decide what to do with the battle model.** Nobody has given an explanation of its +0.09 plateau that would have predicted it, and its recipe changes were mostly noise walks. Either freeze it and spend the compute on the campaign, or give it a falsifiable plan: per-battle weighting is now in; search with averaged rolls once V is good; campaign-aware battle prices.
7. **Run cheap ablations for the paper's core claims** on the Mac paired-eval farm:
   - telescope vs raw 7-day labels;
   - prior-anchored vs zero-init features;
   - DAgger vs imitation-only;
   - segments vs fixed horizon;
   - value-neutral Φ vs a simpler reward.
   Each is a few hours of paired games, and each claim is currently "built and seemed to help".
8. **Make the behaviour-distribution audit standing.** A periodic automatic comparison of our action and target distributions against NK2's own play (and against recorded human games) would have caught the meeting loop in hours, not days.

---

## 12. Paper scaffold

### 12.1 Possible framing

**Working titles**
- "Learning Heroes of Might and Magic III on one GPU: value-neutral rewards, prior-anchored features, and what the AI developers got wrong"
- "AlphaHeroes: lessons from three weeks of training a 4X-battle hybrid AI on a budget"

**Thesis candidates**
1. In strategy games that cannot be brute-forced to a final label (the developer's note 6), progress on a budget comes mainly from three things:
   - removing representation and environment bugs;
   - value-neutral pricing with multi-horizon credit;
   - injecting knowledge as priors.
   Raw RL scale is not the source.
2. AI coding agents can build and run such a system almost end to end. They fail in characteristic, catchable ways (§6), which the paper can teach readers to guard against.

### 12.2 Suggested outline (with where the material is)

| Section | Content | Material |
|---|---|---|
| 1 Introduction | Why Heroes III is hard (fog, 2–8 players fixed-sum, near-infinite numbers, two-level game, ~1,500 decisions a game, no human data at scale); the budget framing | Deck slides 42–44, 59; §2 |
| 2 Related work | AlphaGo/Zero, AlphaStar (privileged critic, league, imitation + KL), OpenAI Five (Rerun, surgery, horizon), CivRealm, vcmi-gym/MMAI, Nullkiller | Deck parts I–III; e5's AlphaStar/Five review (memory `alphastar-openai5-lessons`) |
| 3 Environment | VCMI fork, contract, headless, determinism, pinned maps, engine as oracle | §2.3, §4 rows 4 and 11; git ledger §8 |
| 4 Models | Battle (ops in the graph, conditional heads); campaign (match logits, U-Net, block grid, foresight heads, memory, priors); the game knowledge built into both and how each quantity is counted | `docs/campaign_model_v2_draft.md`, `docs/map_size_plan.md`; docs catalog §2; §10 and `expert_knowledge/` |
| 5 Reward and credit | Φ, logit-share labels, telescope, V_end with random end, shared luck (notes 1, 2, 9, 10), segments, privileged critic | J04, J07, J14–J15; docs catalog §2; `docs/empire_value.md` (stale: tiles are 5 gold, Φ v5) |
| 6 Learning from a scripted AI | Imitation → DAgger → label fixes → collapses → follow test → removal; inputs instead of a teacher | J07, J09, J13–J14; memory `teacher-label-changes`, `teacher-measurements-2026-11-11` |
| 7 Evaluation | Noise measurements, paired pinned replicates, dumb-decision counters (note 7), behaviour audits | J12, J16; memory `eval-slide-2026-11-11` |
| 8 Results | Eval trajectories with caveats, bug-impact table, failures table | §2.4, §4, §5; git ledger §6–7 |
| 9 AI-assisted development | Session setup, collaboration stats, error catalog (note 8), the developer's role, rules that emerged | §6–§8; `stats/collaboration_stats.md`; `docs/session_rules_draft.md` |
| 10 Discussion | Self-play peace vs fairness, the teacher dilemma, battle plateau, threats to validity, future work | §11, §12.5 |

### 12.3 Claims and how well they are supported

| Claim | Support | Strength |
|---|---|---|
| Hidden representation and environment bugs had larger effects than tuning | Bug lifetimes and impacts (git ledger §6); town, neutral-town, YES-token and one-tile-label fixes | Strong (qualitative and quantitative, but no controlled comparison) |
| A prior with a bounded learnable correction acts immediately; a zero-init feature at lr ≤ 3e-5 is inert | Measured weight-movement bounds; suicide −60%; YES 3→42%; paired gains of priors +0.06 to +0.13 | Strong |
| DAgger labels beat imitation-only for transfer to play | 0.113 → ~0.32 in a day; 7% vs 37% agreement | Strong (one run) |
| Teacher-label changes are the main collapse risk | Both GATHER collapses, the advice collapse and the schema-16 regression; 7 weight rollbacks in 8 days | Strong |
| A follow mixture inflates scores and starves learning | +0.29 score; P_own/9 gradient (measured); towns lost 23→100 | Strong |
| Credit in this game is dominated by within-day shared luck | 96.5–98% variance decomposition; horizon SDs | Strong (measured twice) |
| The telescope composition improves data efficiency | Linear toy only (1.1–3.4×) | Weak (needs an ablation) |
| The privileged critic helps | Value-fit errors −20 to −31% on V_end; no eval effect | Medium (fit) / none (play) |
| One 48-game eval is ±0.1; single arms mislead | Same weights 0.861 vs 0.697; null pairs | Strong |
| Self-play in this game is peaceful and under-trains defence | 2/1,223, 599/600, eliminations analysis | Strong (observation); remedy untested |
| Battle PPO here is ~0.5% signal per update | Adam-moment analysis across checkpoints; participation counts | Medium |
| AI agents' errors follow recurring patterns that human questions catch | §6 with dated examples across all sessions | Strong as case study; anecdotal as statistics |
| A 1-GPU + laptop setup can train a 3.3M-parameter campaign model at ~470 games/h | Deck numbers; logs | Strong for throughput; quality vs NK2 still below parity |
| Under a raw-inputs rule, game knowledge does not disappear; it moves to C++ precomputation, rules in the graph and priors | Catalogue of the deployed code: 607 items, a quarter with a C++ part, two in five computed in the graph, one in eight from NK2 (§10.1) | Strong (descriptive) |
| Several counts of one quantity help when they are different theories built on shared primitives; uncoordinated copies cause bugs | Large bundled gains from the Lanchester variants and the four strength estimates; drifted copies behind most representation bugs (§10.3) | Medium (mechanisms clear; multiplicity never ablated) |
| Engine-exact knowledge is the most reliable kind; its failures come from attributes lost in export and constants copied by hand | The 6,800-pair damage check; the double-shot and attack-cap errors (§10.5) | Medium |

### 12.4 Candidate figures and tables

1. Campaign eval vs NK2 over time, annotated with phases and deploys (§2.4). Data: PC `runtime/train/<joint7>/campaign/eval.csv` and TensorBoard; Mac `runtime/train/*joint*` for earlier runs.
2. Same-weights replicate spread and null pairs: the eval-noise figure (J16, memory `eval-slide-2026-11-11`).
3. Bug-lifetime and impact table (git ledger §6; §4 here).
4. Collapse timeline: label change → value-loss spike → eval drop → rollback, for the two GATHER collapses and the advice collapse (J09, J12; git ledger §5).
5. Variance decomposition of credit: within-day share; horizon SDs (J11, J14).
6. Prior vs zero-init: weight-movement bound and measured effect sizes (J08, J13).
7. Throughput waterfall on the Mac (73 → 238 decisions/s) and the PC (17–83 → 500–600 games/h; 237 → 400 joint7 games/h) (J03, J06, J08).
8. Behaviour audit: our action and target distributions vs NK2's own play, before and after the meeting fixes (J06, J10, J11).
9. The four-system comparison with a Heroes column (deck slides 41 and 44).
10. Collaboration statistics: prompts per day by session, autonomous stretches, commits by hour (`stats/collaboration_stats.md`).
11. The AI error catalog as a table: pattern, count of instances, how caught, countermeasure (§6).

The docs catalog §6 lists 18 more, with data paths. Several of those paths are marked unclear, and joint7's logs live only on the PC.

### 12.5 Threats to validity a paper must state

- **One long run** with live surgery (65+ deploys, 6 schema changes, two teacher regimes). No clean rerun. Many changes landed together, so attributions are tentative.
- **Eval drift:** Φ versions, map sets, own vs assisted arms, PC vs Mac, and load effects. Scores across these boundaries do not compare directly.
- **NK2** was both the teacher (11 days) and the evaluation opponent, and its planner remains an input.
- **Small eval sets** (48 games): ±0.1 per eval. Only paired, replicated differences are reliable.
- **AI-written analysis:** the dev AIs' numbers are self-reported in prose, and this journal is AI-synthesized. Verify key numbers in the digests, git and logs.
- **Scope:** small random maps, 2–4 players, SoD rules. No human-opponent evaluation. No large maps. No HotA.

### 12.6 The developer's paper notes, mapped to evidence

| Note (abridged) | Evidence in this journal |
|---|---|
| 1. Examples are dependent; luck is shared in the telescope | Shared-luck analysis (plan Round 6), the 96.5% within-day variance, horizon SDs — J04, J11, J14 |
| 2. Heads learn different horizons; correlated luck | Telescope design, toy efficiency 1.1–3.4×, head weighting by variance — J04, J07 |
| 3. All utility functions work (paperclips); some better; aim for zero effect on unclear decisions | Value-neutral Φ, deck slide 50, exploits found by audits — J04, §4 row 5 |
| 4. Ops over features from hard-coded models; batch-0 similar to the hard-coded model | Op engineering, StupidAI twin, prior-anchored features, priors-only model 0.123 — J02, J08, J14; the knowledge catalogue, §10 |
| 5. Hard examples; if unlearnable it is an architecture issue | Generator hard scenarios, cannot-do list, human recordings, learnability-first principle — J02, J05, J10 |
| 6. Aim only at games that cannot be brute-forced to the final label like AlphaGo | Intro framing; deck slides 3–12, 44 |
| 7. Metric: count dumb decisions | Failure counters, deck slide 56, the meeting-loop audit — J10, J11 |
| 8. AI error section | §6 of this journal |
| 9. AlphaGo kept one position per game; we cannot afford that, so the dataset is correlated | Correlated batches, minimum 20 games per update, batch-noise analysis — J05, J10 |
| 10. Replace the value-share goal with log(value-share difference at +2/5/7) | Logit-share-change labels (Oct 31) — J07; docs catalog §2 |
| 11. Hire actions are rare; greedy eval hires less (AlphaStar strategy vector) | Greedy-eval oscillation, sampled evals, the z discussion — J05, J08 |
| 12. Find out exactly how NK2 cheats | Partially known: exact monster counts (`J04`). Not fully audited; still open |

---

## 13. Glossary

| Term | Meaning |
|---|---|
| VCMI | Open-source re-implementation of the Heroes III engine; forked into the repo |
| SoD / HotA | Shadow of Death (original expansion data used) / Horn of the Abyss (a popular mod; transfer planned on branch `hota`) |
| NK / NK2 | Nullkiller / Nullkiller2, VCMI's strongest scripted adventure AI. The campaign eval opponent; the teacher Oct 31–Nov 11 |
| BattleAI, StupidAI | VCMI's default and simplest scripted battle AIs |
| MMAI | An outside ML battle AI from the vcmi-gym project, used as a battle eval opponent |
| RMG | VCMI's random map generator. Nondeterministic for a given seed, hence pinned map files |
| Φ (empire value) | Gold value of everything a player owns; purchases at list price are neutral |
| share / logit-share labels | My Φ / all Φ; value labels are the change of logit(share) at day +2, +5, +7 |
| V2/V5/V7, V_end, telescope | Value heads per horizon; V_end predicts the final share with random game ends; "telescope" = each head graded by the next shorter head's later forecast |
| total_score | Battle eval score on −1..1: our reward minus the reference AI's reward in the same seat |
| Campaign eval score | share × number of players at day 56 against NK2; 1.0 = an average player; 0 = eliminated; 48 games |
| joint1…joint7 | Successive joint (campaign + battle) training runs; joint7 is the long PC run from Oct 31 |
| deploy N | A numbered live change to joint7 (drain → restart on new code); numbering from ~Nov 9 |
| schema N / contract vN | Campaign / battle input-output format versions (schema 19, contract v16 by Nov 12) |
| v1234 | Campaign checkpoint version (one per update) |
| teacher, DAgger, advice | NK2's answer on our own states, used as an imitation label (DAgger-style) through a dry-run advisor |
| accept set | A set of actions any of which satisfies the teacher. It turned out to teach all its members |
| MOVE / TRAVEL | Move a hero within a window of nearby tiles / toward a distant area |
| MULTI, GATHER, GIVE_ALL, TAKE_BEST, DROP | Macro actions (several engine requests from one decision) |
| prior-anchored features | Known values (NK2, engine) as a fixed prior plus a bounded learned correction |
| expert knowledge | Any rule, number, heuristic or NK2 judgement put into a model or its targets instead of being learned (§10) |
| fight value, AI value | Two per-creature value tables in VCMI's data: the original game's fight value, and the AI value VCMI's own AIs use (NK2's army strength). The models use both (§10.3) |
| Lanchester strength | Army strength as health × damage per round: under the square law an army twice the size is four times as strong. Used in battle for the strength shares and the escape relief |
| foresight heads | Auxiliary heads predicting engine outcomes; hidden states feed the policy; outputs never exported |
| privileged critic | A value head that sees rivals' hidden states during training only |
| credit by segments | Policy credit where the 2-day surprise counts in full and later days are weighted by measured signal |
| paired eval | Same pinned maps/setups for both arms; replicates; each arm run alone (`campaign_model/paired_eval.py`) |
| spool, drain, keeper, guard | Disk store of finished games / stop accepting new work before a restart / restarter / in-trainer stop on a value-loss spike |
| value valves | A Nov 11 mechanism to damp value-gradient noise into the trunk; its premise failed, removed |
| nk-predict head | Auxiliary head learning to predict NK2's planner outputs, so the planner can leave live serving |
| cd, b1, e5/88, fork | The Claude Code sessions (§0.2) |
| PC, Mac | RTX 5090 Windows/WSL2 training box; MacBook Pro M4 used for development, extra game slots and paired evals |

---

## 14. Source index: where to find things

**How to answer common questions**

| Question | Where to look |
|---|---|
| What happened on a given day? | The period journal covering it (§3 headings give journal numbers), then `transcript_digests/<session>` with `grep -n '^\[2026-11-05'` |
| What did the developer say about X? | `grep -n '\] USER:' transcript_digests/*.md \| grep -i X` |
| When was bug X introduced and fixed? | `sources/git_ledger.md` §6, then `git log --grep` in the repo |
| What was deploy N? | `sources/git_ledger.md` §3; memory notes `deploy-*`, `overhaul-2026-11-11` |
| What rule governs X, and why? | `sources/memory_catalog_*` §2 (rules with triggers); the original memory files in `~/.claude_personal/projects/-Users-alon-projects-alpha-heroes/memory/` |
| Where is game knowledge in the networks, and how is a quantity counted? | §10; `expert_knowledge/` (grep a code identifier, or a row ID such as `CG-068`) |
| What did the two sessions tell each other? | `grep -n 'SendMessage\|PEER-MSG' transcript_digests/3_b1_*.md transcript_digests/4_e5_*.md` |
| How was a design decided? | The period journal §7 tables; `sources/docs_and_decks.md` §2; repo `docs/` |
| What is the latest state? | §11 here; J15, J16 §10; memory `morning-2026-11-12`, `overhaul-2026-11-11` |

**Original sources outside this folder**
- Raw Claude Code transcripts:
  - `~/.claude_personal/projects/-Users-alon-projects-alpha-heroes/*.jsonl` (a62ff69f, fd6b4705, 68ab2bd0 with one subagent, 31529395, df787a6f);
  - `~/.claude/projects/-Users-alon/b608c016-….jsonl`;
  - the Cursor exchange under `~/.cursor/projects/Users-alon-projects-alpha-heroes/agent-transcripts/`.
- Memory notes: `~/.claude_personal/projects/-Users-alon-projects-alpha-heroes/memory/` (79 files + `MEMORY.md`).
- Repo docs: `docs/*.md`. These include the developer's `points to note in a future paper.md` and the session-rules draft.
- The deck: `~/Downloads/From AlphaGo to Heroes III.pptx` (62 slides; condensed in `sources/docs_and_decks.md` §5).
- Training logs:
  - the Mac: `runtime/train/` (battle runs, joint1–joint5);
  - the PC: `~/projects/alpha_heroes/runtime/` in WSL (joint7, TensorBoard, eval CSVs).
