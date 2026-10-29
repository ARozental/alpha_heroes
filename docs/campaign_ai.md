# Campaign AI: what an empire is worth, what the AI decides, how it trains

Status, 2026-10-27 (evening):
- **Built and working:** joint training (§6). The campaign model learns by self-play on random maps (2–4 players, water and underground on), alongside the battle model. Each has its own optimizer. Our battle model plays every battle of the campaign games, and those battles are part of its training data. The campaign model is evaluated against Nullkiller2 every hour.
- **Not built yet:** the remaining decision kinds (§3.3).
- **This replaces the proposal of the same morning.** Your two reviews changed the reward, the hero value, the action space and the plan (no imitation of NK2). Everything below is the design as decided and built.

Run:

    .venv/bin/python -m campaign_model.joint --run joint --hours 12      # the joint run (§6); --resume goes on after a stop
    .venv/bin/python -m campaign_model.train --days 5                   # smoke test: one self-play game, one update (~10 s)

---

## 1. Your decisions and where they live

| Your decision | As built |
|---|---|
| Reward: no same-day or next-day reward; the end of day 3 is labelled from day 5, or from a trio such as days 5, 9 and 11 | Each decision gets the player's share of all empires' value at the end of days d+2, d+5 and d+7 (`VALUE_OFFSETS`); past a day cap, no label. Three value heads predict the three shares, each graded against the next shorter head's later forecast of the same day mixed with the real share (`labels.ladder_targets`); three growth heads (helpers) predict our own value's log growth to the same days (`labels.py`). The end head V_end predicts the share at the game's end, graded against itself the next day mixed with the real end (`labels.end_targets`); the policy's credit moves from the 7-day horizon to V_end's as V_end beats the player-count prior (`labels.end_advantages`, `learner.end_credit`). |
| After elimination, the label is the share on the elimination day | An eliminated player's empire is worth 0 from that day on. Snapshots stop at the end of the game; past the last snapshot, the last share stands (`labels.share_at`). |
| Play until the game ends, maybe with a day cap | Full games to a cap drawn per game (`learner.draw_days`): day 28 plus a geometric number of days, mean 28, at most 112, so past day 28 every day is as likely to be the last and the model cannot plan around the end. At the cap the AIs send their last records and the runner stops the game (`play.py`); the cap is the game's end for V_end. Evaluations stay at 56 days. |
| Multiplayer from the start | 2–4 players per random map. The reward is my Φ / Σ over all players' Φ. |
| No command factor; buying creatures should be neutral | A hired creature is worth cost + u, a creature for hire u, with u = α × the cost of its unupgraded form. Recruiting and upgrading are exactly neutral (tests). |
| No accrued growth: a town with 0 for hire on Monday is worth the same on Tuesday; the AI should learn to build dwellings at the end of the week | Growth counts the day it arrives. A new dwelling is worth its cost minus u × its first batch, so building it is neutral and the next Monday is a gain. |
| Heroes: the battle model's formula; hiring not always neutral; a small quadratic penalty on the hero count; one number to watch | Level value 2500·(L−3)^1.3 (fractional level), artifacts at the game's prices, capped mana. Hiring a fresh hero changes Φ by its army − 2500 − the penalty step; the penalty is 100·n². The number to watch is `heroes_mean`: heroes per player, averaged over every player-day still in the game. |
| One shared list of values for the battle and campaign models | `game_values/` holds the resource prices, hero level value, artifact prices and mana prices. `battle_model/reward.py` now imports them (same values, so battle rewards are unchanged). |
| Threat: not valued, learned | No threat term. Labels start at d+2. |
| Mana: a tiny value: spell power × what can be spent in a 3-round battle | `capped_mana_value` = 2 × power × min(mana, 3 × the hero's dearest castable combat spell). Without a combat spell it is 0. |
| Explored tiles: maybe 10 gold a tile; don't reward camping next to Cover of Darkness. Then: "10 gold is too much, maybe 1" | 1 gold per tile ever revealed (a 72×72 two-level map holds 10k). The ledger keeps every player's union of revealed tiles, so re-revealed tiles never pay twice. |
| Income multiple 5 | `income_days = 5` |
| Move: tiles in movement range and at most K tiles away | MOVE picks a hero and a tile of its 33×33 window (K=16) that the hero reaches today. Longer trips take several moves. |
| "Put a hero to sleep" is not a real action | Gone |
| Army moves happen within the hero-to-hero interaction mode | Exchange mode (`EXCHANGE`): ARMY moves between the two armies, then END. **Verified:** the server only allows hero↔hero troop moves while that exchange is open (`CGameHandler::isAllowedExchange`); town↔visiting hero is allowed any time, as a turn action. |
| The engine's questions are answered from campaign data | Yes/no, the choices of a dialog, level-up skills, Town Portal's town, and recruiting at a visited dwelling are all model decisions (option pointer plus the question's context). |
| Retreat or surrender is not ours | In the joint run our battle model plays every battle, and retreat and surrender are in its own action space. Only when BattleAI plays (smoke tests) does it ask the adventure AI, which answers with Nullkiller2's rule. |
| Battles in campaign training: our battle model, the best version; it keeps training on the campaign's battles, plus hard and random examples so it does not regress; its own optimizer | The joint run (§6) starts the battle model from joint1's update 2840 (eval +0.076, every set positive: `joint.BATTLE_FROM`; joint1 itself started from v15-window's 2106). Campaign battles are 40% of each battle update (`real_share`), the wandering monsters' side included. The rest is generated battles, 10% of them hard cases. |
| Monster counts: "the range a human sees", not NK2's exact counts | Wandering monsters' counts are VCMI's quantity ids (few, several, pack, ...): `CGCreature::getHoverText`. Exact only where the engine shows a human the exact number: a hero's Visions (`TILE_EXACT_COUNT`, `d_detailed`). Other players' towns and heroes are quantity ids unless VCMI says detailed (`t_detailed`, `h_detailed`). A resource pile's amount is not shown to a human, so it is not an input. |
| Hero levels 1–3 are free because a level-2 hero with one imp is worth less than the 2500 gold that buys a level-1 hero with a starting army | Kept as is: retreating such a hero saves nothing. (I had read it as "early XP is worthless"; you corrected that.) |
| Teleports: Dimension Door and Town Portal matter, monolith exits don't | CAST covers all ten adventure spells, Dimension Door with a target tile and Town Portal with the town question. A monolith exit follows the planned path, or the engine picks it. |
| Price per decision, cap 256 | 1e-4 share points for every turn decision still to come that day; at most 256 turn decisions per day. |
| Random maps, not only 2 players, water and underground on | `alpha_make_random_map`, VCMI's own generator: any template that fits the size and player count |
| Don't imitate NK2; turn what it calculates into in-graph features | No teacher. In-graph: army strength (count × fight value), hero fighting strength, and per window tile log(our strength / theirs) and its sigmoid (P(win), NK2's danger ratio), for monsters and for enemy heroes or towns standing there. |

## 2. Empire value Φ (`campaign_model/empire.py`)

Gold, at list prices. Buying at list price leaves Φ unchanged; value appears through production, pickups, fights and captures. Python computes Φ from the ledger, which holds each player's assets as raw engine values (`agents/AlphaMapAI/Ledger.cpp`, the reward's view of the whole game state, never the model's).

| Asset | Value |
|---|---|
| Resources | at the resource prices (wood and ore 250, the other four 500) |
| Building | its cost |
| Building with an income (halls, resource silo, grail) | max(cost, 5 × the daily income it adds over the building it upgrades). Town/City Hall and Capitol cost exactly that; the pre-built village hall gets 2,500 |
| Dwelling | cost − u × its creature's growth (the free first batch) |
| Unpriced income (mines, other owned objects) | 5 × daily value |
| Hired creature (heroes, garrisons, other owned armies) | cost + u |
| Creature for hire (own towns, own map dwellings) | u |
| Hero | 2500 × (L − 3)^1.3 for levels above 3 (fractional), + artifacts (not the spellbook, the Grail or war machines, as in the battle reward), + capped mana |
| Hero count | −100 × n² |
| Tiles ever revealed | 1 each |
| A player out of the game | 0 |

α = 0.25. The `EmpireConfig` values (α, income days, hero penalty, tile value) are hyperparameters. They are not fitted yet: my earlier "fit them to remove drift" step is not built.

## 3. Decisions (the action space as built)

### 3.1 Turn actions: an action type, then that type's table

| Type | Table | Legality (engine rules) |
|---|---|---|
| END | — | always in a turn; in exchange mode it closes the exchange |
| MOVE | own hero (8) × window tile (33×33) | reachable today (VCMI's pathfinder), not a garrisoned hero |
| BUILD | own town (8) × building (44) | `canBuildStructure`, plus the server's rules that function skips: automatic and special buildings are never built by hand, and the Grail's needs a visiting hero carrying the Grail |
| RECRUIT | town × level × amount (25/50/75/100% of what is available and affordable) | into the garrison, else the visiting hero |
| HIRE | town × tavern slot | tavern, no visiting hero, 2500 gold, under the hero caps |
| ARMY | from holder × slot × to holder × amount (all, half, one) | town garrison or garrison hero ↔ visiting hero; in exchange mode, the two armies of the exchange; a hero keeps one creature |
| UPGRADE | holder × slot, to its dearest affordable upgrade | `fillUpgradeInfo` (in a town of ours) |
| TRADE | sell × buy × amount (10/25/50/100% of the most possible) | an own marketplace; the engine's rates |
| CAST | hero × adventure spell (ids 0–9); a Dimension Door also picks a window tile | `canBeCast` / `canBeCastAt`, mana; Scuttle Boat not yet |
| BUY | hero × {spellbook, ballista, ammo cart, first aid tent} | hero in an own town with a mage guild / blacksmith, affordable, not owned |
| DISMISS | hero | reserved: in the contract, never legal yet |

### 3.2 Questions the engine asks: a pointer over up to 8 options, with context

The context: the hero involved, the object on its target tile (type, subtype, guard creature and count), the first number in the question's text (e.g. the gold asked for creatures), and the sound id.

| Kind | Options | Examples |
|---|---|---|
| YES_NO | yes, no | fight the guards, accept joining creatures (free or paid), pursue fleeing ones, open a Pandora's Box |
| SELECT | the dialog's components | a chest's gold or experience, an Arena's attack or defense |
| LEVEL_UP | the secondary skills offered | |
| OBJECT_SELECT | map objects | Town Portal's town |
| RECRUIT_DWELLING | none / 25 / 50 / 100% of the dearest creature on offer | visiting a dwelling |
| EXCHANGE | ARMY moves and END (a turn-like mode) | two own heroes meet; a garrison dialog |

### 3.3 Not the model's yet (rules), and why

| Decision | Now | Next |
|---|---|---|
| Artifact moves and equipping | VCMI's auto-placement | a model table (heroes × slots) |
| Map taverns, trading posts, altars, black markets, universities | their windows are closed | TRADE modes, a skill table |
| Hill Fort upgrades | none (no query) | UPGRADE at the fort |
| Monolith exits | the planned path's exit, else the engine's choice | your call: not worth it |
| Scuttle Boat, Dig | not offered | |
| Dismiss | reserved | turn on once training is stable |
| Retreat, surrender | our battle model's own actions in the joint run; Nullkiller2's rule only when BattleAI plays | done |

## 4. Model (`campaign_model/model.py`, 377k parameters)

- **Tokens** go through 2 attention blocks (d = 64):
  - the global state, including the question asked;
  - heroes: 8 own, then 8 others seen, with army, artifacts, skills and spells;
  - towns: 8 own, then 8 others;
  - the other army of a garrison dialog;
  - up to 8 options.
- **Other players' heroes and towns:** only what VCMI shows the player (`getHeroInfo` / `getTownInfo`). Counts are quantity ids unless detailed; the graph maps a quantity id to a typical count (`QUANTITY_COUNTS`). Enemy hero levels are unknown (0).
- **Windows:** one 33×33 grid around each own hero. Per tile:
  - terrain, road and flags (revealed, blocked, visitable, water, guarded, visited, exact count);
  - the top object and its owner;
  - a monster's creature and count as a human sees it;
  - the pathfinder's turns and movement left.

  Two 3×3 convolutions follow, then FiLM by the hero's token; this scores MOVE and Dimension Door targets. A pooled summary joins the hero's token. Only the windows of heroes that exist are computed (a player has 1–3 of its 8 slots), so a decision costs ~1.5 ms on one CPU thread.
- **In-graph functions** (§4.1).
- **Heads:** every table above, each a masked log-softmax (−inf where illegal), plus `value` (3 sigmoids: the share at d+2, d+5, d+7). Training only: `growth_logit` (our own value's log growth to those days, within ×10 and ÷10).
- **Type head starting preferences** (`TYPE_PRIOR`, a nudge, not a rule): MOVE +1.5, BUILD and RECRUIT +1, UPGRADE +0.5, TRADE −2, DISMISS −5, the rest 0. It is the one hand-set bias in the graph; it can be zeroed.
- **ONNX export** matches torch to 1e-6, the in-graph functions included. Games use the shared inference server (`battle_model/inference.py`, family `campaign`).

### 4.1 Functions in the graph

These are torch ops in `CampaignModel.forward`, so ONNX carries them. C++ sends only raw engine values (ids, counts, positions, primary skills). The tables are registered buffers, built from the vocab exported from the engine:
- `fight_value[creature]`: the creature's `fightValue` from VCMI's creature config, the number Nullkiller2 and VCMI's AIs use for strength;
- `creature_cost[creature]`: its recruit cost at the shared resource prices (`game_values`);
- `quantity_counts[quantity id]`: a typical count of each range.

| Function | Formula | From |
|---|---|---|
| typical count | quantity id → 3, 7, 15, 35, 75, 175, 375, 750, 1500 (unknown: 15), unless exact | VCMI's quantity ranges |
| army strength | Σ count × fight value, log-scaled per stack and in total | NK2 `getArmyStrength` |
| hero fighting strength | √((1 + 0.05·attack)(1 + 0.05·defense)) | `CGHeroInstance::getFightingStrength` |
| hero total strength | log army strength + log fighting strength | NK2 `getTotalStrength` |
| per window tile: who is there | an enemy hero or town on the tile, matched by position (its strength), or a wandering monster (count × fight value) | |
| per window tile: danger ratio | log(our hero's strength / theirs), and σ(2 × that) as P(win) (0.5 at equal strength) | NK2's danger ratio |
| calendar | day of the week (one-hot 7) and week of the month (one-hot 4) | for "dwellings on Saturday, City Hall on Sunday" |
| recruit head | gold value of what is for hire (cost × available) and our gold, log-scaled | |

**How they relate to the battle model:** they don't call it. They share its building blocks (`battle_model/layers.py`: attention block, masked log-softmax, log scaling, embeddings), its engine vocab, and `game_values` (which the battle reward and Φ both use). The battle model's own in-graph rules (`battle_model/rules.py`: the damage formula, the kill rule, P(stack wiped), exposure bounds) need a battle's exact state, which the map never shows. So the campaign's P(win) is NK2's coarse strength ratio. The one link between the two models is through the labels: a fight's outcome, played by our battle model, moves Φ at d+2.

## 5. Updates (`campaign_model/train.py: ppo_update`)

PPO as in the battle model:
- advantage: the horizon credit (`labels.horizon_advantages`: aimed at the share 7 days on, V7's step mixed with the real outcome by trust and the credit check), normalized over the update; a decision without one (past the day cap) is left out;
- clipped surrogate (0.2), 2 passes over minibatches of 256 (fixed shapes: the last partial one is left out);
- LR 1e-4 under KL control: the update stops once the KL from its own start passes 1.5 × 0.015. The rate's scale then drops by 1.5, and rises by 1.5 after an update that ended under half the target, never above 1;
- entropy bonus 0.01 with the parents' probabilities detached (the battle lesson; test included);
- value heads trained with BCE against their grades (`labels.ladder_targets`), the growth heads at half weight against our log growth on a 0..1 scale; missing grades left out; gradient clip 0.5.

Each decision's inputs (~340 KB raw, mostly the windows) stay zlib-compressed in memory (~2 KB) until a minibatch decodes them, one minibatch ahead, in a thread.

## 6. Joint training (`campaign_model/joint.py`, `learner.py`, `battle_model/train.py`)

One process, two learners. Each has its own model, optimizer and KL control: the battle learner on the GPU in the main thread, the campaign learner in a process of its own (`campaign.learner` "process", on Apple's GPU or CUDA; "thread": a thread on the CPU, as in joint1, since Apple's GPU fails when two threads use it):

| Piece | What it does |
|---|---|
| battle side | v15-window's recipe, going on from joint1's update 2840: 8 generated-battle actors (10% hard cases), the self-play league, 3,200-decision updates, evals vs BattleAI and MMAI every 30 minutes. Campaign battles replace Nullkiller2's games as the "real" battles: 40% of each update. |
| campaign games | 5 at once, a third on the game's own small maps (`maps.py`), the rest a new random map (2–4 players; 36×36; two levels; water at random), 56-day cap. Every player is the current campaign model; every battle, the wandering monsters' side too, is the current battle model (sampling). Battle records go to the battle buffer. Campaign records and ledgers go to the campaign learner. |
| campaign update | in a process of its own (`campaign.learner` "process"), once ≥ 8,192 decisions from ≥ 4 finished games are in (~2 min an update); decisions more than 4 versions old are dropped, and a backlog over 2 updates' worth loses its oldest games |
| campaign eval | every 60 minutes, and once at the start: 24 games on fixed maps (2, 3, 4 players in turn, 36×36), our models in one seat (rotating), Nullkiller2 elsewhere. Our battles are greedy; theirs and the monsters' are BattleAI's. Score = our share of Φ at day 56 × players (1 = an average player of its game; day 28's is logged too, `score_d28`). |

Logs, under `runtime/train/<time>-<run>/`:
- battle: as before;
- campaign: `campaign/progress.csv` and `campaign/eval.csv`; TensorBoard `tensorboard/campaign` (`campaign/…`, `campaign_actions/…`, `campaign_eval/score`);
- checkpoints: `campaign/checkpoint_latest.pt`, plus one per eval.

What to watch:
- `campaign_eval/score`;
- `campaign/heroes` (the one number for heroes);
- `campaign/phi_end` (mean Φ at the cap: in self-play the shares average 1/n, the absolute Φ does not);
- `campaign/value_ev_d2`;
- `campaign/no_effect`, `ai_errors`, `game_failures`.

**Engine change (labelled, not a bug fix):** `VCMI_ADVENTURE_AI_<COLOR>` picks one player's adventure AI (`client/Client.cpp`), so an eval game can seat AlphaMapAI next to Nullkiller2. It works like the existing `VCMI_BATTLE_AI_<COLOR>`.

## 7. Runtime (`agents/AlphaMapAI`)

- **Threads and queries:** Nullkiller2's patterns: one turn thread; dialog answers on async tasks; wait while queries, battles, visits or movement are pending; the game-state lock released while waiting.
- **Movement:** the human client's batching (stop at teleports, guards, visitable objects, layer changes), repeated until the hero arrives, stops, or its target object is gone.
- **Records:** compressed records go to the trainer's RecordServer at the end of each turn: inputs, the choice, log-probability, values, and a flag when an action changed nothing.
- **Ledger:** one per day, taken as the day begins, plus one at the end of a game.
- **Config:** `AlphaMapAI` is in the settings schema's adventure-AI list. The vocab exporter also writes buildings (cost, income, upgrades), town creatures, object types, terrains and roads.

## 8. Measured (smoke runs on this Mac, each under 3 minutes; no science)

| Run | Result |
|---|---|
| tiny joint run (2 actors, 2 campaign games of 5 days, small updates), 2¼ min | 27 battle updates, 8 campaign updates. Campaign battles were 15–50% of battle updates. 0 actor failures, 0 game failures, 0 AI errors, 0 actions that changed nothing. |
| full recipe (8 actors, 4 campaign games of 28 days, evals off), 2½ min | battle updates of 3,200 decisions every ~20 s, campaign battles 40% of them. 27 campaign games of 28 days (17 s each on average, ~320 decisions each). The first campaign update: 8,609 decisions, 66 steps, 30 s on the GPU, KL 0.002. |
| campaign eval vs Nullkiller2 at version 1 (untrained), 12 games of 28 days, 10–11 s | score 0.24–0.29 ± 0.08 (two runs of the same model: the eval's noise). Knocked out in 6 of 12 games, as early as day 4; Nullkiller2 hires 3.3 heroes to our 0.4 |
| records | ~2 KB a decision compressed; ~17 MB per 27 games, in memory only |
| load (full recipe) | CPU ~70% busy (load 8–9 of 14 cores). Memory: 23 of 24 GB in use, 7 GB compressed. The trainer process held 12 GB, most of it the GPU's (the battle learner's as before; the campaign update adds ~1–2 GB at its peak and is freed after) |

- **Bug found and fixed:** Φ was 0 for the winner as well as for losers (`empire_value` tested "not in the game"). A decisive game's last snapshot then gave every player 1/n, so the loser's last labels were 0.5 instead of 0. Now only losers are worth 0 (test added).
- **Monster counts in real records:** quantity ids only (4 and 5 seen early on), none exact without Visions; no resource amounts.
- **Every decision kind ran in games, except two:** OBJECT_SELECT (Town Portal's town) and RECRUIT_DWELLING.
- **Tests:** `campaign_model/tests` 31 (contract, Φ neutrality, labels, records, log-probabilities, entropy bonus, the update and its KL stop, the learner's batching, the eval score and stop rule, the winner's labels, the `campaign.*` settings); `battle_model/tests` 132, all pass.

## 9. The first run (joint1, 2026-10-27 22:40, 9 h) and what it found

Settings at launch: V from d+2, d+5, d+7 (flat mix); 56-day games on 36×36 maps with 2–4 players; `TYPE_PRIOR` as the type head's starting bias (`priors/action_types.csv`, learned from there); 8 battle actors, 6 campaign games; the battle model from v15-window 2106 (kept read-only in `runtime/saved/battle_v15-window_02106.pt`). Start: battle +0.067 (BattleAI +0.071, MMAI +0.062); campaign 0.17 vs Nullkiller2 at day 56 (0.29 at day 28), knocked out in 8 of 12.

Found and fixed while it ran (each a resume of the same run):

| # | Kind | Finding | Fix |
|---|---|---|---|
| 1 | bug (engine) | The headless test client ended the whole game when red lost, with other players still in it: multiplayer self-play games and evals cut short | `client/NetPacksClient.cpp`: red's loss ends it only when no one else is in the game. My first version waited for a timeout in battle-only games (they end silently, red first) and read freed state; both fixed and checked on neutral, two-sided and persistent battles |
| 2 | bug | The map generator sometimes leaves a player without a town (template 8SM0f): it loses on day 1 | `play.make_map` checks the header and regenerates from the next seed |
| 3 | bug | A run started in the background ignored SIGINT; SIGTERM left its children | `joint.py`: both signals stop it cleanly |
| 4 | design | The campaign update (~40 s) ran in the battle learner's thread: battle updates every ~60 s instead of ~20 | the campaign learner has its own thread on the CPU (`campaign.learner`) |
| 5 | design | A learner slower than its games trained on an ever older backlog | the backlog is capped at 2 updates' worth; the oldest are dropped (counted) |
| 6 | bug | `campaign/progress.csv` rows misaligned after a resume that added columns | the header is upgraded on resume; the file was rewritten |
| 7 | bug | Every resume reset the eval clocks: after 4 restarts no eval had run for 40 minutes | the next eval is due by the last one's time on disk |
| 8 | design (reward) | A fresh hero was worth ~3,600 in Φ for 2,500 gold: the catapult every hero has (1,000) and starting war machines counted as artifacts. Heroes per player rose 1.5 → 3.2 in an hour | `game_values.UNVALUED_ARTIFACTS` (spellbook, Grail, war machines), as the battle reward already does: a hire is now ~neutral for the 2nd hero, negative after |
| 9 | data | "Knee Deep in the Dead" starts red without a town: lost on day 7 in all 11 games, 9% of games | removed from `maps.GAME_MAPS` |
| 10 | hardening | Players knocked out together would have shared 1/n each | losers get 0 (`empire.shares`) |
| 11 | bug (time) | An endless siege in an eval game: Nullkiller2's hero (BattleAI, never retreats) against our garrison with no hero; VCMI has no round limit and our stalemate rule only covers us attacking. The game ran to its 900 s limit and held the eval, and with it all training, for 15 minutes | eval games capped at 240 s, training games at 360 s (normal: 10–30 s). It came back in 2 of 24 eval games, a hero-vs-hero battle too: AlphaBattleAI now retreats as the defender as well after 35 rounds without damage (the attacker's rule: 30), when it can flee. A defender without a hero still cannot: a round limit (engine or AI) stays your call |
| 13 | design (curriculum), open | Self-play never attacks: of 1,223 players in 400 games (version ~190), 2 were knocked out and 1,158 ended with just their starting town. Against Nullkiller2 we are knocked out in ~75% of games. On the same 12 eval maps the day-28 score rose 0.29 → 0.40–0.47 by version 80 (development learned) and stayed there; day 56 and the knock-outs (7–9 of 12) did not improve | your call: Nullkiller2 seats in a share of training games, as opponents, not teachers (only our players' decisions recorded); or wait for self-play to find aggression (a captured town is a large share gain at d+2..d+7) |
| 14 | learning, open | Campaign heroes retreat in ~60% of their own battles (13 of 21 over 6 games with the run's models at update 3009 / version ~230), 7 of them at their first decision: the campaign model walks into fights the battle model calls lost. This is why the battle updates' attack side retreats in ~45% of battles (v15-window: 12%) and the main way heroes are lost | the danger-ratio feature does not yet steer moves; candidates: Nullkiller2's expected-loss formula (army × ratio²) as an in-graph feature, or the battle model's own value estimate of a fight |
| 15 | resource | Memory is the binding resource: after 2.5 h the trainer held 18–19 GB (10 GB GPU, ~8 GB heap: the battle buffer kept full at its 16,000-decision cap by the surplus campaign battles), swap 4.1 GB, 40% system CPU. Flat over 10 minutes: no fast leak | next joint run: buffer_decisions ~6,000 (an update takes 3,200), or fewer campaign games |
| 16 | regression (battle model), open | From ~update 2600 the battle model drifts toward moving and waiting: training shooter-idle 0.05 → 0.30, the attack side's MOVE share 0.22 → 0.36, entropy 1.59 → 1.93; eval shooter-idle 0.05–0.09 → 0.39 (BattleAI) / 0.51 (MMAI) at 3139, castle vs BattleAI 0.10 → 0.02. Campaign battles (40% of each update, both sides our model) are 41% MOVE, 33% WAIT, 4% SHOOT (60 battle sides; one battle 373 decisions), and the drifting model plays the next ones: a loop. Not the stalemate rule (1 in ~30 battles); the same records path ran NK2's games in v15 without drift | kept checkpoints: 2629 (+0.082), 2840 (+0.076, every set positive). Options: a lower real_share for campaign battles (e.g. 0.1–0.15), weighting campaign battles by their size (data.example_weight), or mixing NK2 real games back in |
| 12 | design | Campaign games make ~5x the battle data the battle learner takes (40% of 3,200 decisions an update): most campaign battles age out unused | left as is (the generated share guards against forgetting); fewer campaign games would lose no battle data |

Added on your word: the game's own small maps (`campaign_model/maps.py`: 10 maps, 36×36, 2–4 players, no teams, standard victory) for a third of the games, picked with a new header tool (`build/bin/alpha_map_info`, VCMI's loader).

Checked and right: recorded log-probabilities and values match the torch model to 1e-6 on every action type seen; monster counts are quantity ids; our battle model in campaign games: 0 fallbacks, errors and rejections over 1,394 decisions; game maps play to the cap.

Known, not fixed: when a game ends in our AI's turn, VCMI can abort as it tears the AI down (exit 134; the records are complete); casts of view-only spells (View Air/Earth) flag as "changed nothing".

## 10. Open points

1. **The guesses** (the hero penalty 100·n², α 0.25, the decision price 1e-4): to be judged by the failure modes once training runs (your call). Watch heroes, wasted actions and decisions per day.
2. **The windows see 16 tiles around our heroes only.** Far mines, dwellings and artifacts are invisible. That is fine on 36×36 maps and blind on 72×72 and larger. The next input would be a coarse whole-map grid.
3. **Labels end at d+7.** Secondary skills and a main hero's growth pay over weeks; only the value head can carry that.
4. **`TYPE_PRIOR`** is the one hand-set starting bias (§4). You rejected a P(win) prior on moves; say if this one should go too.
5. **VCMI also shows a human a threat level** when they right-click a monster: 12 levels of its exact strength over the hero's (`CGCreature::getPopupText`, UI enhancements on by default). It is a human-visible view of the exact count; I left it out because you said "the range". It would be one more window input.
6. **Memory is the tight resource on this Mac** (above). On the 32 GB PC with the 5090 it would not be.
7. **Next decisions to hand to the model** (§3.3, in its order).
