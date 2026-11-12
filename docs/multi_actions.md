# Multi-actions: what Nullkiller2 does as one goal, and what MULTI makes of it

Schema 14 (e3f5503, 2218ada, 0eab9f0) added MULTI: one decision, several engine requests, the C++ making them
(RECRUIT_ALL, RECRUIT_ALL_HERO, GATHER, GIVE_ALL). This lists every place Nullkiller2 issues several requests from one
goal (AI/Nullkiller2: Goals/*.cpp accept(), AIGateway.cpp), what ours makes of each, and what is still missing, the
most common first. Counts: the live joint7 run's last 80 games (versions 1155–1159, 2026-11-05; the advice's
unmatched kinds, 216k advice asked) and the 2026-10-31 NK2-vs-NK2 request audit (397 player-turns).

Schema 16 (2026-11-07) added SPLIT_SINGLES, MERGE_ALL and GARRISON_SWAP, ARMY within one army, swaps and the target
slot (docs/model_cannot_do.md, Fixed). Rows 4 and 7 are built; row 1's one-army arrangement is ARMY's now.

Why it matters for learning: a sequence Nullkiller2 makes as one goal reaches our model as one label (its first
request) plus an accept set; the rest of the sequence is never taught, and the model must find it by exploration a
decision at a time. One entry per sequence gives one label and one gradient.

## What Nullkiller2 makes as one goal

| # | Nullkiller2's goal | Its requests, in order | Ours | Missing |
|---|---|---|---|---|
| 1 | **Pick the best army** (`AIGateway::pickBestCreatures`): on every hero exchange, every town arrival (`moveCreaturesToHero`), every garrison dialog, `buildArmyIn`. ArrangeStacks 1,532 per 397 turns, its most common non-move request. | `getBestArmy` picks the best 7 stacks of both armies for the receiver; moves each wanted creature into its slot (merging duplicates), moves unwanted stacks back to the source, dismisses a leftover stack worth < 1% of the army, splits one creature off when the source would be left with none. | GIVE_ALL (exchange) and GATHER (town) move everything that fits, VCMI's bulk move; ARMY one stack a decision, 8 a dialog. | A **TAKE_BEST** kind: the receiver ends with the best 7 stacks of both armies (NK2's `getBestArmy`, or by Φ's creature worth), the rest back, as one request batch. The one-army arrangement it needs (merge duplicates, fill slot 0) is the cannot-do list's row 1. The advice's "exchange (no move of ours toward its army)" is 2,793 of 98k exchange advice; "ArrangeStacks (within one army)" 287. |
| 2 | **Buy army in a town** (`BuyArmy::accept`). RecruitCreatures 711 per 397 turns. | `makePossibleUpgrades(town)` (every stack it can afford, best upgrade); recruit by level from the top, each level as many as it can pay for, into the upper army; when the garrison is full, dismiss its least valuable stack of another faction first; then, if a hero visits and no hero is garrisoned, `moveHeroToTile(town)`, which upgrades the hero's stacks, picks the best army into it and buys a spellbook for a MAIN hero. | RECRUIT_ALL, RECRUIT_ALL_HERO (levels from the top, pay as you go: the same order); UPGRADE one stack a decision; BUY spellbook its own decision. | **UPGRADE_ALL** (holder): NK2's `makePossibleUpgrades`, every stack whose best upgrade is affordable (UpgradeCreature was only 6 per 397 turns in NK2's own games, so by its count a low priority; a human upgrades at every visit). A **BUY_ARMY** kind doing the whole visit (upgrade all, recruit all, take best, spellbook) is one decision where ours takes four or five. |
| 3 | **Town arrival** (`performObjectInteraction` TOWN / HILL_FORT): on every `moveHeroToTile` that ends in a town. | Upgrade all the hero's stacks it can afford; pick the best army from the garrison into the hero (unless the garrisoned hero is locked for defence); buy the spellbook for a MAIN hero with a Mage Guild. Hill Fort: upgrade all. | GATHER (bulk, not best); UPGRADE per stack; BUY spellbook; at a Hill Fort UPGRADE per stack. | The arrival combination as one kind (**VISIT_TOWN**: upgrade all, take best, spellbook), legal for the hero visiting. Our MOVE into a town does none of it, so a visit costs 3–9 decisions after the move. |
| 4 | **Garrison hero swap** (`ExchangeSwapTownHeroes::accept`): the defence primitive. GarrisonHeroSwap 228 per 397 turns; the advice's unmatched 69 in 80 games (its first request only; most of the sequence is hidden behind a MoveHero). | Put a hero inside: swap the visitor out if another is in, upgrade the town's stacks, walk the hero in (= arrival's upgrades + best army), dismiss garrison stacks that cannot merge, swap, lock for defence. Take a hero out: swap, then `buildArmyIn` (upgrade hero and town, recruit all into the town, best army into the hero). | GARRISON_SWAP (schema 16): VCMI's garrisonSwap, one request (the swap alone, no best army, no lock). | A **GARRISON** action (town): swap the visiting and garrisoned heroes, as a plain action and as the two kinds **GARRISON_IN** (swap in + take best + lock nothing) and **GARRISON_OUT** (swap out + buildArmyIn). Without it GATHER has no counterpart: NK2 empties garrisons on arrival *because* its defence puts heroes back (0eab9f0's gather costs patch that hole from one side). |
| 5 | **Hire a hero** (`RecruitHero::accept`; `StartupBehavior`; `DefenceBehavior`). HireHero 86 per 397 turns. | Swap the visiting hero into the garrison when the visitor slot is taken, then hire; the Startup behavior follows with `ExchangeSwapTownHeroes(town, nullptr)`: swap the new hero in, `buildArmyIn` the other. Defence: swap, dismiss a weak hero, hire, as one sequence. | HIRE only when no hero visits (Encoder: `!town->getVisitingHero()`); then RECRUIT_ALL_HERO / GATHER next decision. | **HIRE with the visitor swapped in** (needs row 4's swap): today a town whose main hero stands inside cannot hire at all, which NK2 does routinely at the start. "Hire all" (every tavern hero) is not a Nullkiller2 pattern: it hires one at a time, each a goal, capped by `needToRecruitHero`. |
| 6 | **Hero exchange artifacts** (`pickBestArtifacts`, on every hero exchange). ExchangeArtifacts 67 per 397 turns; advice unmatched 191 in 80 games. | For the receiver: every artifact of either hero into an empty fitting slot, else swapped in when its score beats the worn one (the worse one to the other hero's slot or backpack), repeated until no change; then the same for the other hero with what is left. | GIVE_ALL moves all of the giver's artifacts (bulk: into free slots, else the backpack); ARTIFACT one move a decision (GIVE, EQUIP, ASSEMBLE). | **EQUIP_BEST** (hero, or the pair): NK2's `pickBestArtifacts` or by Φ's artifact worth. A hero carrying a better sword in its backpack never wears it unless the model finds the single EQUIP; one kind fixes it everywhere. |
| 7 | **Army formation before a fight** (`ExecuteHeroChain::accept` → `rearrangeArmyForSiege` when attacking an enemy town today; `WhirlpoolAction` → `addSingleCreatureStacks`). Part of the 1,532 ArrangeStacks. | Split one creature off the weakest stack into every free slot (7 stacks on the field; a whirlpool takes from a 1-stack); for a Citadel or better sort the stacks: fliers first, then by value (the slot order is the battlefield's starting order). | SPLIT_SINGLES and MERGE_ALL (schema 16); ARMY within one army moves, swaps and splits a stack into the slot of the model's choosing (the siege sort is a few of those). | **SPLIT_SINGLES** (hero): fill free slots with 1-creature stacks of the weakest; **MERGE_SINGLES** back. Cheap (no schema beyond a kind), and it hands the battle model blockers it never sees today. |
| 8 | **Recruit at a dwelling** (`showRecruitmentDialog` → `recruitCreatures`): every creature the dwelling offers, each level as many as affordable, merging two of the hero's duplicate stacks to make room. | One request per level. | RECRUIT_DWELLING: our C++ picks the dearest creature, the model only the amount (cannot-do row 4). | A **recruit all** option in that question (every creature, all affordable), beside the amounts of the dearest one: NK2's label, exactly. |
| 9 | **Trade for a purchase** (`tryRealize(Trade)`): sells every other resource until the needed one is covered. TradeOnMarketplace 686 per 397 turns. | One trade per resource sold. | Done: BUILD / RECRUIT short of resources trade for themselves (payByTrades, schema 12); TRADE in fractions. | Nothing. |
| 10 | **Town Portal** (`AdventureSpellCast::accept`): swap the destination's visitor into its empty garrison first, cast, then visit the town (arrival's upgrades, best army). | Swap, cast, move. | CAST with a destination; no swap, so a town with a visitor is not open to us. | Row 4's swap. |
| 11 | **Dismiss** (`DismissHero`, hero cap reached or a weak hero in the way of a hire; `BuyArmy` and `pickBestCreatures` dismiss small stacks). DisbandCreature 28, DismissHero in the advice 60 in 80 games. | One request. | DISMISS reserved; no disband. | Cannot-do row 1 (disband) and row 9 (dismiss a hero). |

Not multi, but in the same unmatched list: BuildBoat 756 (cannot-do row 8: the SHIP build entry exists since
2026-11-04 but NK2's BuildBoat is at map shipyards too); MoveHero into a teleport 2,470 (ours walks through
teleports as part of a path: no label lost); "RecruitCreatures (town: not legal for us)" 927 (unexplained: our RECRUIT
needs one creature affordable or tradeable and a slot; NK2's BuyArmy the same; to measure which).

## Review of schema 14's code (2026-11-05)

Read: Actions.cpp executeMulti, Encoder.cpp encodeMulti, Advice.cpp acceptMulti and adviseExchange, AlphaMapAI.cpp
exchange, model.py multi_gain / multi_prior / stronger_receiver / gather_loss, model_v2.py's MULTI head, VCMI's
bulkMoveArmy and bulkMoveArtifacts. Mac smoke games and the live run: 0 errors, 34 failed actions in 80 games,
27,754 multi-actions (87 a player-game).

- **Correct as far as read**: the recruit order and payment match NK2's (top level first, resources as they stand after
  each request; waitTillRealize is on); bulkMoveArmy keeps one creature of the weakest stack when all would go, else
  moves what fits; GIVE_ALL's rows and the prior's giver/receiver factors agree; the fingerprint now holds artifacts,
  so an artifact-only GIVE_ALL is not read as futile; older records read multi_mask at 0.
- **The accept set turns the teacher off where it accepts** (train.py: the advice loss is -log of the summed
  probability of the label and every accepted entry). GATHER is accepted at every town where it is legal
  (acceptMulti), so on those decisions the teacher no longer pulls toward NK2's own action: the model satisfies it
  with GATHER alone. That is the mechanism by which the first schema-14 games emptied every garrison; the gather
  costs (0eab9f0) lean against the prior, not against this. Accept GATHER only when NK2's advice itself is a move
  into that town or an action there (it gathers on arrival, never standing still), or weight accepted entries below
  the label.
- **GATHER's prior values the wrong army when a hero is garrisoned**: encodeMulti's GATHER moves the upper army,
  which is the garrisoned hero's when there is one, but multi_gain prices it at `f.town_gold` (the town's own stacks).
  With a hero inside the gain is near 0 for an army that may be large. Rare today (no action puts a hero inside), a
  bug once row 4 exists.
- **RECRUIT_ALL's acceptance needs NK2's first request to be the recruit** (acceptMulti): BuyArmy upgrades first, so
  whenever an upgrade is affordable the whole-recruit acceptance is lost. Rare by NK2's upgrade count.
- **GIVE_ALL ends the exchange** (exchange() breaks after a MULTI), so nothing can be taken back after it: intended
  (567 of 870 troop moves sent armies back before it), fine.
- **Tests** (test_model_v2.py) cover the prior's signs and the stronger receiver, not executeMulti; the C++ was
  checked by smoke games only. A unit of the recruit loop against a town with a full garrison (level skipped, not
  recruited elsewhere) would pin the one branch the games may not reach.

## Order to build (recommended)

1. TAKE_BEST (row 1): done (schema 15).
2. GARRISON swap (row 4): GARRISON_SWAP done (schema 16); GARRISON_IN / GARRISON_OUT and HIRE with the swap (row 5)
   still to build.
3. UPGRADE_ALL and the VISIT_TOWN combination (rows 2, 3).
4. EQUIP_BEST (row 6), recruit all at a dwelling (row 8). SPLIT_SINGLES (row 7): done (schema 16).
