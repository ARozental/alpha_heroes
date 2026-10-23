# Battle search (MCTS): how to try moves and take them back

Status: design, with the first stage implemented (see the end). Written 2026-10-23.

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

## Design E

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

## Stages

1. **`IBattleServer` refactor.** No behaviour change; the regression test is identical records.
2. **`SimulatedServer` + overlay**, with the replay test against recorded battles.
3. **Search API and IPC to Python**, with a simple MCTS in Python.
4. **AlphaZero-style training** from search targets.
