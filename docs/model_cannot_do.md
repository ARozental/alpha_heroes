# What the current model cannot do

Things the campaign model can **never** learn, with any amount of data, because an input, an action, a gradient or a
contract size is missing. Not weak spots that more training fixes (those are the watch list). The order is the user's:
the most common situations first, then by weight. The list grows as we find more; an item leaves it when its fix is
deployed (move it to "Fixed", with the commit).

Audited 2026-11-03 against schema 10 (campaign_model/schema.py, constants.py), agents/AlphaMapAI (Encoder.cpp,
Actions.cpp, AlphaMapAI.cpp) and joint7's training (train.py). Nullkiller2's request counts are from the 2026-10-31
audit of two NK2-vs-NK2 games, 397 player-turns (`MoveHero` 12,981 for scale).

| # | Cannot | Why (where) | Fix sketch |
|---|---|---|---|
| 1 | **Disband a stack of its choosing**, or take an artifact off into the backpack, move one between worn slots, disassemble a combination. NK2's advice in joint7: `DisbandCreature` 3,555 and `ExchangeArtifacts` 450 unmatched over 941 player-games (2026-11-07). | DISMISS is reserved for heroes; DROP disbands only a hero's slowest or foreign stacks where no garrison takes them. ARTIFACT's targets are GIVE, EQUIP, ASSEMBLE. | ARMY to a "disband" holder (prior strongly negative: the user's rule, we do not remove our creatures); ARTIFACT targets UNEQUIP and a worn slot of its choosing. |
| 2b | **Tell a worn artifact from a carried one** in the hero's own vector: a sword in the backpack and the same sword worn give the same hero token, so only ARTIFACT's prior (and `h_primary`, which counts worn items' bonuses) knows what a hero actually wears. | `hero_fields` concatenates `h_artifacts` and `h_backpack` before the summed embedding (encoders.py). | A worn/backpack row (or the position's slot row) added to each artifact's embedding before the sum, zero-init. |
| 1b | **Learn to take a town's troops into its visiting hero** (and to upgrade there, and a main hero's spellbook): our heroes led 2% of our army at day 60; P(ARMY town->hero) 0.0002 where legal; BUY chosen 0 of 4,219 legal in town. | Nullkiller2 does all three inside its visit (AIGateway::objectVisited), none of them a request, so the advice (its turn's first request) never says so and the teacher taught against them. Replacing its label with them at every in-town decision (2026-11-05, reverted in 26bc2a1) collapsed the policy: eval 0.378 -> 0.248. | Label the visit's moves on its first decision only (Nullkiller2 does them once, on arriving), or as accepted entries beside its own label, or at a small weight of their own; measure the type drift (type_drift.py) after the first update. |
| 3b | **Remember what a decision needs many decisions later**: the memory's writer learns since 2026-11-05 (memory_steps 2), but its gradient reaches back one write, so keeping a fact across a day's ~30 decisions is learned only through each step's gate (each next decision's losses pulling on what it kept). | train.sequence_order / model_v2.forward_steps unroll pairs: a step graph replay 117 ms against 108 at 1 (the PC, 128 rows); longer sequences cost more and shrink each step's batch. | memory_steps 4-8 once the learner has the time; or writes once a day (the model's own gate), so one step spans a day. |
| 4 | **Buy the basic creature when the upgraded one is on sale** (from the week the upgraded dwelling stands). At an external dwelling with several creatures (Golem Factory, Elemental Conflux), choose which creature. | RECRUIT's mask carries `ids.back()`, the best creature per level (Encoder.cpp ~784, `t_available_creature`); `showRecruitmentDialog` picks "the dearest creature on offer" in C++ and the model picks only the amount (also against "Python owns the science"). | A basic/upgraded bit on RECRUIT; the dwelling's creatures as the options of RECRUIT_DWELLING. |
| 5 | **Weigh a quest**: what a Seer's Hut or Quest Guard pays and asks beyond the first number of its text. | A refusable yes/no dialog with the terms as components (`CQuest.cpp` sets `canRefuse`). `blockingQuestion`'s yes/no branch keeps `d_object`, `d_subtype`, `d_number` (first number) and `d_sound`; it encodes components only as a guarded object's guards (schema 11). | Encode the quest's components onto the YES option (`o_type`, `o_subtype`, `o_amount` exist). |
| 6 | **Know a town's spells**: plan "go learn Town Portal in that town", or value a hero's visit for its spells. | No input for a mage guild's spells; `t_built` gives the guild level, `h_spells` only what our heroes already learned. | `t_spells` (OWN_TOWNS × N_SPELLS bits). |
| 8b | **Buy at a Black Market, a Trading Post or an Altar of Sacrifice**: buy an artifact; trade at a map market; sacrifice artifacts or creatures for experience. | AlphaMapAI::showMarketWindow answers at once, unasked. | Each as a question of the University's kind. |
| 9 | **Beyond the contract's sizes**: maps above 36 × 36 (`MAP_TILES`); more than 8 rival heroes or 8 rival towns in view (the farthest vanish); more than 8 options (Town Portal with 9+ towns); the backpack past 8; map taverns (closed unasked, `showTavernWindow`); dismissing a hero (reserved); thieves' guild information (nothing). | Constants and stubs. Fine for the current maps. | Raise when the maps grow; map taverns as HIRE with a tavern holder. |

## Hard rather than never

- **Position, threat and exploration by walking** have no term in Φ (docs/empire_value.md §4), so defensive placement
  and scouting are learnable only through the end-of-game share: a weak signal in 28–112-day games that mostly end at
  the cap.
- **A MOVE reaches the hero's own window and one more** (schema 17, built 2026-11-10 on branch `move-windows`, not
  deployed yet). Before it a MOVE's targets were the 33 x 33 tiles around the hero on its own level: nothing past a gate
  or a monolith, and TRAVEL's areas are a day or more away, so a tile of the other level in reach today took a MOVE
  onto the gate first, a tile worth nothing by what stands on it. In a Nullkiller2 game of 40 days (36 x 36, two
  levels) 43% of the hero-turns had tiles in reach today outside the hero's window, a median of 31 against 77 inside.
  Now each hero gets one further window, in a window slot no hero holds: a square of the level with the most such
  tiles. What is still two decisions: a tile neither window shows (a second gate the same day; a far road on a map
  wider than two windows), and everything past an entrance whose exit we have not seen (VCMI's pathfinder does not go
  through one): that entrance is priced as a sight of new land instead (TILE_UNSEEN_EXIT).
- **A tile's features come from its own level**: the map network (U-Net) reads each level apart and no input says
  where a gate leads. A hero's token reads both levels, and its further window now shows it the tiles past a gate, so
  the pair (hero, tile) is informed; the tile's own vector is not. Linking the two ends in the network (the exit's
  cell added to the entrance's) is not built.

## Fixed

- **Arrange one army; swap stacks; put a hero in the garrison or take it out** (schema 16, 2026-11-07; the user: "make
  sure we can learn from all my actions"). Their first recorded game had 52 troop moves and 4 garrison swaps no action
  of ours could make; each now maps to one.
  - ARMY's target slot is the model's, a second draw like CAST's tile (army_slot_logp; the record's cell). A whole stack
    moves into an empty slot, merges onto its own creature or swaps with another. A part splits into an empty slot or
    onto its own creature. ONE's second half moves all but one, the one staying: the user's way to peel a stack into
    singles, 20 of their 54 moves. Priors: VCMI's old slot +4, a swap's slot -2, all but one -5.
  - ARMY within one army: the turn's own armies, and each army of an exchange (learned part -5, so the ARMY type keeps
    its score where only those are legal).
  - MULTI's SPLIT_SINGLES (a creature of the weakest stack into each free slot, NK2's addSingleCreatureStacks in VCMI's
    bulk-split order), MERGE_ALL (each creature's stacks into its first) and GARRISON_SWAP (VCMI's garrisonSwap rules).
    Each starts at -4.
  - unifyStacks keeps singles (7192bbd, the user's call, 2026-11-08): before each turn decision an army's duplicates
    merge unless one of them is a single creature. Without any merge (schema 16's first day) 6% of turn decisions had a
    hero holding a creature in two slots and the value heads' fresh loss rose ~40%; a split into halves still merges
    back (a deliberate 10/10 is not kept).
  - The matcher maps every ArrangeStacks (a swap from an empty slot as the move it is), GarrisonHeroSwap, BulkMergeStacks
    (MERGE_ALL), BulkSplitStack (SPLIT_SINGLES; approximate unless one stack fills every free slot), BulkSplitAndRebalance
    (half into the first free slot, approximate), BulkMoveArmy (GATHER; GIVE_ALL approximate), ExchangeArtifacts (GIVE,
    EQUIP) and AssembleArtifacts. A recorded request between two armies a turn does not pair is recorded as the exchange's
    decision.
  - Still not ours: taking an artifact off into the backpack, moving one between worn slots, disassembling; a stack
    split into an arbitrary count (the nearest of ALL, HALF, ONE, all but one: approximate).
  - Nullkiller2's advice newly matched (ARTIFACT, the three MULTI kinds, ARMY within an army or all but one) is not
    taught in joint7 until its label mix is measured (config.nk2_new_advice).

- **See a neutral town: its garrison a fight, taking it a capture** (296b06a; deployed 2026-11-05 22:59; the next eval, v1288, 0.476 ± 0.062 against 0.391, eliminated 15 against 18).
  - The bug: the encoder took the other towns from VCMI's getTownsInfo(false), which walks the players' own towns only.
  - So a neutral town was never in t_*: its garrison was no fight and its tile was worth ~0 to MOVE. All 17 of the hopeless MOVE attacks the features saw no fight in were neutral towns (40 joint7 games).
  - Now every visible town not ours is listed, by nearness (8 slots shared with the rivals').
  - On 3 games, P(step onto a hopeless neutral town) went 0.27 -> 0.002, and onto a winnable one 0.07-0.15 -> 0.75. Turn KL mean 0.037.
  - With it (635912f, deployed 2026-11-06 00:33): a guard that stands in fog. VCMI flags the visible tile next to it as guarded; with no monster we saw next to it, that fight was worth 0. Now it is the worst fight the window's monsters show.

- **Train the memory** (8458ede; deployed 2026-11-05 00:38, memory_steps 2 from then): each pass runs pairs of a player's
  consecutive decisions, the second reading the memory the first wrote, so every loss reaches the writer (it had none:
  PPO ran each decision on its recorded memory). Checked on the PC: the chain exact (400 of 400), the KL anchor's bias
  1.5e-5, the step graph against eager 1.5e-4; the first update's forward 173 s against 120-140.

- **Move artifacts: hand one to the hero met, wear one from the backpack, assemble a combination** (schema 13: 87e79b5,
  9fe35c9, d1950b5; deployed 2026-11-04 23:37). ARTIFACT (hero, position: worn slots then the backpack, target GIVE /
  EQUIP / ASSEMBLE); its prior is what the move does to Φ (an artifact's attack and defense move what each hero makes of
  its army; an assembly the combination's price over its parts'), less a move's cost (5 at the start, learned). The
  vocab exports each artifact's primary bonuses and a combination's parts. First 29 games: 42 GIVEs a game, 0 errors.

- **Remember a guarded object's guards; build a boat; learn a skill at a University** (schema 12: 542fdbe, deployed
  2026-11-04 22:46). An object whose guards a question showed is in a_* with its army as it is each time; a town with a
  shipyard builds a boat as BUILD's SHIP, a map shipyard asks a yes/no; a University's learnable skills are each a
  yes/no whose YES is the skill. Also new there: a purchase short of resources trades for itself.

- **Weigh a guarded object's guards before "attack?"; see a neutral garrison's army; take a monster's offer to join
  knowing its creatures and price** (schema 11: 778b92f, 6f54540, c01038f; deployed 2026-11-04 17:09).
  - The question's guards: a bank's exact list from its dialog; a mine's or a dwelling's as quantity ids.
  - o_battle on the option that fights, o_join and its price on a join. A monster's question takes the guard of the
    hero's tile (the engine's rule) when the destination holds no monster.
  - Garrisons' armies (a_*, 32) as fight tiles in MOVE's features.
  - The answer starts from the fight's value (as a MOVE onto a fight is valued) or the joining creatures' worth minus
    the price, plus zero-init learned corrections.
