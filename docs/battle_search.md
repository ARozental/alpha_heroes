# Battle search (MCTS): how to try moves and take them back

Status: stages 1-3 done and tested (see "Results" and "Stage 3"); stage 4 (training from search) to do. Written 2026-10-23.

## What search needs

MCTS (or any lookahead) has to:
1. take the current battle position,
2. apply an action and see the result, including dice (damage rolls, luck, morale),
3. go back and try another, thousands of times per real decision.

VCMI has no "undo". The vcmi-gym author gave up on MCTS for that reason.

## Where the battle state lives in VCMI

- **`BattleInfo`** (`lib/battle/BattleInfo.h`) is the state. It holds:
  - the sides (heroes, armies, how many spells each side cast);
  - every unit (`CStack`: position, count, health, shots, retaliations, flags, bonuses such as spell effects);
  - obstacles and walls;
  - the round number and the active unit.
  The server and every client keep their own copy inside their `CGameState`.
- **Every change goes through the `IBattleState` interface** (`moveUnit`, `setUnitState`, `addUnitBonus`, `nextTurn`, `nextRound`, `addObstacle`, ...). The server never edits a battle directly. Its rules compute a result and send network packs (`StacksInjured`, `BattleStackMoved`, `SetStackEffect`, ...). Each pack applies itself to *any* `IBattleState` with `applyBattle(IBattleState *)`, on the server and on each client.
- **The rules are the server's battle processors** (`server/battles/`, about 3,850 lines):
  - `BattleActionProcessor`: move, attack (with retaliation and every creature ability), shoot, spells, defend, wait.
  - `BattleFlowProcessor`: whose turn it is, morale, round start effects.
  - `BattleResultProcessor`: the end of the battle.
  They read the battle through a `const CBattleInfoCallback &`. They change it only by `gameHandler->sendAndApply(pack)`, roll dice with `gameHandler->randomizer` and the game RNG, and cast through `gameHandler->spellcastEnvironment()`. Their only other links to the game: the turn timer, surrender gold, charged artifacts, one game setting.
- **A copy-on-write battle already exists.** `HypotheticBattle` (`AI/BattleAI/StackWithBonuses.h`) is an `IBattleState` over a real battle. It copies a unit only when that unit changes, and has its own `ServerCallback` that applies packs to itself. BattleAI uses it to try spells with the real spell rules.
- **State outside `BattleInfo`** that a battle changes: the heroes' mana (pack `SetMana`, applied to the hero object), artifact charges, and the random generators.

## Options

| | How | Cost of one "try a move, take it back" | Works in a real game | Engine changes |
|---|---|---|---|---|
| A. Replay from the start | Fixed seed + `ALPHA_BATTLE_PREFIX` (done) | a new VCMI process and the whole prefix: ~1-2 s | no (needs the seed and the start) | none |
| B. Restart in process | `BattleProcessor::restartBattle`, then replay the prefix | the prefix: ~ms per action | no | small |
| C. Snapshot the live game | serialize `BattleInfo` + mana + RNG on the server, swap it back on server and clients | ~0.1-1 ms + resync | only with a hacked server | large and fragile (clients hold pointers into the battle) |
| D. fork() the process | the OS copies the whole game | ~ms | - | none, but **does not work**: VCMI is multithreaded and fork keeps one thread |
| **E. Local battle copy (recommended)** | the server's own rules run on a copy-on-write overlay of the battle, inside the AI | ~10-100 µs per action, a snapshot is a new overlay layer | **yes**: it only needs the AI's own view of the battle | medium, mechanical: the rules talk to an interface instead of `CGameHandler` |

E is the only option that is fast enough for MCTS *and* usable when our AI plays a real game. Nothing in it is a hack: it is VCMI's rules, VCMI's packs and VCMI's own overlay type, run without network and clients.

## Design E (as built: the copy is a copy of the game state, not a HypotheticBattle)

The first idea was to run the rules on a `HypotheticBattle` overlay. That does not work: the server's rules
read units as `CStack` (`battleGetStackByID`), and a `HypotheticBattle` only overlays `battle::Unit` queries,
so the rules would read the real, unchanged stacks. What works, and is what is built:

- **The copy:** `CGameState::cloneWithBattle(battleID)`, a new lib method.
  - The game state is copied in memory the way a savegame is written and loaded.
  - The battle is copied the way `BattleStart` sends it to a client.
  - Each unit's current state (position, health, shots, flags) is then copied as JSON, the way `BattleUnitsChanged` sends it. A unit serializes only how it *started* the battle.
- **The rules:** the server's `BattleActionProcessor` and `BattleFlowProcessor`, unchanged, behind `IBattleServer`/`IBattleOwner` (stage 1).
- **The packs:** applied to the copy with `CGameState::apply` (`GameStatePackVisitor`), exactly as on the server and the clients.
- **Dice:** the copy's own `CRandomGenerator` and `GameRandomizer` (luck, morale and ability rolls).
- **Encoding:** `CPlayerBattleCallback(copy's battle, player)` is a player's view of the copy, so the existing encoder and masks work on it unchanged.
- **Code:** `agents/AlphaBattleAI/BattleSim.{h,cpp}` (`BattleSimulation`: `copyOf`, `clone`, `apply`, `applyActionOnly`, `finished`, `winner`).

The original sketch follows.

### Original sketch (overlay)

```
          real game                                       search (inside AlphaBattleAI)
 BattleProcessor ── IBattleServer ── CGameHandler     BattleProcessor ── IBattleServer ── SimulatedServer
 (unchanged rules)   (interface)     (as today)       (the same code)                    │
                                                                                          ├─ packs -> applyBattle(overlay)
                                                                                          ├─ its own seeded RNG
                                                                                          └─ mana / charges in the overlay
 overlay = HypotheticBattle over the live battle (CPlayerBattleCallback): unchanged units are read from
           the real battle, changed ones are copies; a child node is an overlay over its parent's overlay
```

1. **`IBattleServer` interface** (`server/battles/IBattleServer.h`). It carries what the processors use: `sendAndApply`, `complain`, the randomizer and RNG, the spell-cast environment, the obstacle-trigger environment, settings, and hooks for the turn timer, surrender gold and artifact charges. `CGameHandler` implements it by forwarding. The processors hold an `IBattleServer *` instead of a `CGameHandler *`, so most call sites stay textually the same. This is a refactor without any change in behaviour, so it could go upstream.
2. **`SimulatedServer`** (our code) implements `IBattleServer` for one overlay:
   - packs are visited (`ICPackVisitor`) and applied with `applyBattle` to the overlay;
   - `SetMana` and charges go to per-side counters in the overlay;
   - its own random generator: seeded, or a chosen roll for chance nodes;
   - no clients, no timer, no queries;
   - the end of the battle is a terminal flag.
3. **The overlay**: `HypotheticBattle` (compiled from BattleAI's sources, or moved to lib) plus a mana proxy for the hero caster.
   - **Snapshot** = a new overlay over the current one: O(1), memory only for units that change.
   - **Restore** = drop to the parent overlay.
4. **Search API**, used from C++ (fast) and from Python through a small IPC server (the science stays in Python):
   - `root(battle)`, `clone(node)`;
   - `encode(node)`, the same tensors the model already reads (the encoder takes a battle callback);
   - `legal(node)` (the same masks), `apply(node, action, rng)`, `terminal(node)`, `reward(node)`.
5. **MCTS** (AlphaZero-style):
   - the network gives the priors (policy heads) and the leaf values (value heads);
   - chance (dice) is handled by sampling the RNG per simulation;
   - both sides are searched, the opponent with the same network;
   - training targets come from the search's visit counts.

### How to know the copy is right

- **Regression for step 1:** battles are deterministic given VCMI's seed and our actions. The same seeds and sampling seeds must produce byte-identical records before and after the refactor.
- **For step 2:** replay a recorded battle's actions in the copy and compare every unit after every action with the recorded state. Positions, turn order and spell effects must match exactly. Damage must be in range, or exact when the copy's RNG is fed the server's rolls.
- Server rejections must stay at zero: the copy's legality is the engine's.

### Risks

- Code in the processors that edits state without a pack, or that reads the global game state: to be found by the step 2 replay test. The audit so far found none in the action/flow processors besides the turn timer, surrender gold and artifact charges.
- Hidden information. In a real game the copy starts from what our client sees, so the enemy's spellbook and hidden skills are unknown. The copy then needs a guess, or a fixed "no spells" assumption.
- Speed of `HypotheticBattle` with deep overlays: every read walks the layers. Flatten a node when its depth exceeds a limit.

## Results (2026-10-23)

**Stage 1**, the `IBattleServer`/`IBattleOwner` refactor. 24 battles with fixed seeds produced byte-identical records before and after it: 882 decisions, covering normal play, self-play, Hypnotize, Resurrection, Animate Dead and Sacrifice.

**Stage 2**, the simulator. With `ALPHA_BATTLE_SIM_CHECK=1` the AI first plays each of its actions in a copy, then compares the copy with the real game once the real game has applied the action:

| Scenario | Actions compared | Refused by the copy | Mismatches not caused by dice | Count/health differences |
|---|---|---|---|---|
| castle_vs_inferno (incl. self-play) | 392 | 0 | 0 | 92, all after attacks |
| Hypnotize on both sides | 372 | 0 | 0 | 112, all after attacks |
| Resurrection, Animate Dead, Sacrifice, Lightning Bolt | 480 | 0 | 0 | 118, all after attacks |

- **Compared exactly:** positions, defending, waited, moved, shots, retaliations, spell effects and hero mana, for every unit.
- **Count and health** differ only after melee and shooting, because the copy rolls its own dice. They are exact after spells, moves, waits and defends.
- **Speed:** copy 3.4 ms (whole game state of a battle-only map plus the battle); one action 0.05 ms.

**Not covered yet:**
- **Turn flow is unchecked.** `apply` runs the next-unit activation (morale, new rounds) with the server's own code, but the check only compares right after the action.
- **Sieges** are not supported (`updateGateState` is a no-op).
- **The copy still includes the game state.** The battle needs nothing from the adventure map. Its world-map inputs will be one short "battle context" vector, made once at battle start by a function over what the player knows (later possibly an output of the campaign network). So the copy should come from the battle plus that context, never the map (TODO in `BattleSim.h`).

## Stages

1. **`IBattleServer` refactor.** No behaviour change; the regression test is identical records. DONE.
2. **`BattleSimulation`** (a copy of the game state, the server's rules, the copy's own dice), checked against the real game. DONE.
3. **Search in C++, next to the model**, with a readable Python copy checked against it. DONE (below).
4. **AlphaZero-style training** from search targets: the record must then carry the search's improved policy.

## Stage 3: the search (2026-10-23)

- **Algorithm:** Gumbel AlphaZero (Danihelka et al., ICLR 2022; mctx's formulas), `agents/AlphaBattleAI/GumbelSearch.h`.
  Gumbel-Top-k + Sequential Halving at the root, completed-Q improved policy below it. Chosen because it improves
  the policy even with few simulations (8-32), where visit-count MCTS needs hundreds; the alternatives found
  (Sampled MuZero, EfficientZero V2, Stochastic MuZero) are this plus more machinery.
- **Readable copy:** `battle_model/search.py`, only for review. `battle_model/tests/test_search.py` builds
  `alpha_search_selftest` and checks that both make the same decisions on a toy stochastic general-sum game
  (64 cases, every branch reached); changing one copy alone makes it fail (checked by mutating each).
- **General-sum:** each node's mover maximises its own Q. Terminal rewards are exact for both sides (`Reward.h`,
  checked against `data.summarize` on synthetic ends and on real battles: the end event's `cpp_rewards`). The model
  predicts only its own side's value, so a leaf's value for the other side is estimated from it
  (`reward.other_side_value`: the two rewards of a battle to the death add up to about 2).
- **Dice:** new dice in every simulation (open loop: each simulation replays the tree's actions on a fresh copy of
  the root). A child is split by the acting unit; at each node only the actions legal in the current state count.
- **Opponent in the tree:** our network, also against BattleAI (an approximation, only in eval).
- **In play:** `ALPHA_BATTLE_SEARCH_SIMULATIONS=N` (off by default), dev library only until runs end.
- **Cost (with two training runs loading the CPU):** 16 simulations ~130 ms a decision: copies ~70 ms (4 ms each),
  network ~40 ms (17 calls), encoding ~7 ms, actions ~2 ms. The copy is the first thing to make cheaper
  (battle-only copies, TODO in `BattleSim.h`).
