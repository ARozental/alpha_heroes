# Campaign model v2: draft design

**Status: draft, not decided** (2026-10-30). The user: "they look good for now but I'm not sure about them".
`features.py` (domain knowledge), the priors, and the credit and labels stay as they are. The model around them is
what changes.

**Built, 2026-10-30.** The user asked for both models improved and trained.

| Piece | Where |
|---|---|
| The model, behind `campaign.model_architecture=v2` | `campaign_model/model_v2.py` |
| The input encoders | `campaign_model/encoders.py` |
| The foresight labels and loss | `campaign_model/foresight.py` |
| The tile grid and the battle records | schema 9 |
| Tests | `test_model_v2.py` |

Choices made while building it:

- **The tile width is 16, not 32.** At 32 the U-Net made inference 3x v1's: 27 ms against 9 ms a decision on one CPU
  thread. At 16 it costs 18.5 ms.
- **A type is as good as its best option.** Its score is its bias (from the priors), what its table's domain prior
  offers (the best legal prior entry: MOVE's the best tile's value as a logit, so value in reach holds END back), and
  the best legal entry's learned part. The first version used the log-mean-exp of the entries: a few good tiles
  among 300 counted for nothing, so MOVE could never beat END on their account. A plain logsumexp would have let
  the number of reachable tiles alone decide. Within each table, the probabilities are a plain softmax.
- **Every head's last layer starts at zero**, so the first policy is exactly the domain priors', the known-sane start.
- **Training is self-play only.** Nullkiller2 is the evaluation's opponent, never a training one (the user,
  2026-10-30: a model that learned to exploit its weaknesses is not one to play against).
- **The reach is an attention bias.** Each hero's reach to each tile enters the tile reader as a bias per head (4
  numbers). A 16-wide vector per (hero, tile) pair would have cost ~1.3 GB at a batch of 256.
- **MOVE stays in each hero's window.** It reads that window's tiles from the shared map, so the AI's contract is
  unchanged.
- **The verification run is battle only.** The battle model is checked first on its own (battle3). The campaign code
  can then change while that run trains.

## Why

The current model (`campaign_model/model.py`) has four problems:

- **Its learned parts get little signal.** The map encoder and each hero's window convolutions learn only from the
  policy gradient on the one chosen tile or area, and from 2-7-day share labels.
- **Inputs are processed ad hoc.** Every call site picks its own transform and constants. Some inputs never reach a
  learned layer (`h_luck` and `h_morale` feed only Φ; the learned part of TRAVEL never sees `travel_days`).
- **Tiles know too little about their surroundings.** Window tiles see 5x5 around them (two 3x3 convolutions), so a
  tile's features say nothing about the route from the hero to it. Heroes and towns are not on the tiles as vectors.
- **Choosing which hero acts is implicit.** Heroes learn about their windows only through a mean and a max over the
  window's tiles.

## Principles agreed in the discussion

- **`encode` only encodes raw inputs.** The higher-level facts (`known`) join after encoding, and join again late
  (`think_again`), so a single meaningful number (P(win), a tile's value) is not swallowed by lower layers.
- **Shared encoders.** One function turns any hero into a vector (ours, a rival's, one for hire). One function turns
  any army into a vector (a hero's, a garrison, a monster, a bank's guards).
- **Ownership is special.** It must strongly transform each vector, and there can be several rivals, so a sign flip is
  not enough.
- **"Large" is relative.** An army is large or small relative to the week and to the whole board, not only on a log
  scale.
- **A logit is a match, not a forward pass.** The actor's query is dotted with the target's key, plus a small network
  over the legal pairs only, plus the domain prior.
- **One softmax over all actions**, if compute allows.
- **Foresight heads are auxiliary.** Their predictions are training-only, never used by the AI. Their small hidden
  states, at least 8 wide, feed the later layers, and at least one non-linear layer comes before anything that changes
  the policy.
- **Any map size**, with part of the map hidden.

## The forward

```python
def forward(self, x):
    known = self.domain_knowledge(x)                    # facts from the raw inputs: Φ, fights, values, danger, reference sizes, priors
    state = self.encode(x)                              # raw inputs only -> entity vectors and the tile grid
    state = self.take_sides(state)                      # every vector transformed by who owns it: ours, nobody's, which rival's
    state = self.join(state, known)                     # known facts concatenated onto what they describe
    state = self.see(state)                             # entities onto their tiles, the map CNN, heroes and towns read the tiles
    state = self.think(state)                           # transformer over entity and player tokens
    foresight = self.foresee(state)                     # foresight hidden states, and predictions for the losses
    state = self.think_again(state, foresight, known)   # foresight and known joined again, more layers
    return self.decide(state, known), foresight         # every logit a match, plus its prior; one softmax
```

Each block calls many small functions. The forward names only the blocks. `state` is one small object (global, players,
heroes, towns, options, tiles, areas): every block takes it and returns it.

## 1. `domain_knowledge`: facts and references

Today's functions in `features.py`:

- Φ's parts
- army strength and worth
- hero worth and level gains
- town strength, value and danger
- object worth and guard strength
- win chance, expected loss and fight return
- exploration
- the travel prior
- Nullkiller2's habits
- hire gains

The tile functions move off the windows. Facts that do not depend on the hero go per map tile. Facts that do (P(win)
for this hero, its fight return, its danger) go per (hero, tile) pair.

New: **reference sizes**, which say what "large" means right now:

- our Φ
- our strongest hero's army
- our total army
- the strongest rival army seen
- our daily income
- each hero's daily movement
- the week number

## 2. `encode`: raw inputs only

| Function | Gives |
|---|---|
| `encode_army` | per-stack vectors and one army vector. Used for heroes, garrisons, monsters, bank guards, tavern heroes and the other army of an exchange |
| `encode_hero` | one vector per hero: ours, rivals' (unknown fields empty), the tavern's |
| `encode_town` | a town vector, and level vectors for RECRUIT |
| `encode_object` | an object vector |
| `encode_tiles` | the tile grid at the map's real size: terrain, road, flags, revealed or not, the object on the tile |
| `encode_global`, `encode_options` | tokens |

**Typed inputs.** Every schema input declares its kind and unit next to its doc, and one encoder per kind turns it into
vectors.

| Kind | Encoding |
|---|---|
| category (ids) | an embedding; -1 gives a zero vector |
| amount | log1p(x/s), x/(x+s/4), x/(x+s), x/(x+4s), x>0, then a linear layer; s is the unit's scale |
| signed amount | symlog(x/s), tanh(x/s), then a linear layer |
| level (skill levels, fort, luck, morale) | one-hot, then a linear layer |
| flag, bits | 0/1, then a linear layer |
| position | relative to the hero or town reading it: direction and distance in tiles |
| set (stacks, artifacts, options) | each element encoded from its fields, then pooled (sum and max) |
| day of the week | an embedding |
| week number | log-scaled and saturating |

- **Units.** There are about ten unit scales, set by hand and documented: gold 1,000, creatures 20, movement 1,500,
  mana 50, experience 5,000, days 3, tiles 10, and so on.
- **Tokens.** A token is the sum of its fields' encodings, then a LayerNorm.
- **Two tests.** Every schema input is consumed by some encoder. Every encoded value stays within a few units over the
  input's whole range.

## 3. `take_sides`: ownership transforms everything

- **Relation.** Every entity and tile has a relation to us: ours, nobody's, or a rival's. Each relation has its own
  learned affine transform per channel. It starts at +1 for ours and nobody's and at -1 for rivals: the battle model's
  signed half, generalized.
- **Rival identity.** Each rival adds its own identity embedding. Two rivals are both against us, but they are not the
  same rival.
- **Player tokens.** One per player (ours, and each rival's we know of), pooled from their entities. They make it
  explicit which rival is the threat. The value heads read our token against theirs.
- **Attention bias.** The trunk's attention gets a learned bias for each kind of pair:
  - same owner
  - us and a rival
  - a rival and the same rival
  - a rival and another rival
  - anything and nobody

## 4. Scale relative to the situation

Every amount enters twice: absolutely from `encode`, and relatively from `join`, as a clipped log ratio against its
unit's reference.

| Amount | Relative to |
|---|---|
| armies | our main army, our total army, our Φ, the strongest rival army seen |
| resources | days of our income, the next building's cost |
| movement | the hero's daily movement |
| distances | days of walking |
| mana | casts of the hero's dearest spell |
| counts on the map | the map's size |

References have floors (our Φ at least 1,000). Because the references grow with the weeks, "large" grows with them.

## 5. `see`: the map at tile resolution, any size

**One grid per level at the map's real size,** shared by every hero.

- Hidden tiles carry only an "unrevealed" flag.
- Off-map cells are masked after every layer.
- ONNX takes the height and width as dynamic axes.
- Records carry the map's size, and batches group decisions by size or pad them with the mask.

**Stamping.** Each hero's and town's vector, after ownership, is added onto its tile. The grid knows that *this* strong
rival hero stands here.

**U-Net.** Strides of 2 and back, so each tile sees the route to it and what lies a day or two around it. Its coarser
levels, pooled by area index into today's fixed 18x18 areas per level, are TRAVEL's targets: TRAVEL's contract stays.

**Reading.** Each of our heroes attends over the tiles, with its own reach added to the keys. Towns read the tiles around
them the same way.

**Per-hero reach stays a fixed 33x33 window** (today's `w_turns`, `w_move_left`), scattered onto the grid by position.
This keeps records small on big maps and keeps MOVE's window-relative indexing.

**Cost.**

- A 36x36x2 map is 2,592 tiles. That is less work than today's windows, which cover nearly a whole level once per hero.
- On 144x144 maps, the AI would compute the static part (terrain, objects) once per turn as a separate graph. That can
  wait until big maps are played.

## 6. `think`

A transformer over global, player, hero, town, option, tavern-hero and other-army tokens. Attention uses the relation
biases, plus today's bias by tile offset between tokens that have a place on the map.

## 7. `foresee` and `think_again`

Each foresight head keeps a hidden state of 8-16 values, joined back where it was computed.

| Head | Where | Targets |
|---|---|---|
| move | every legal (hero, tile) pair | battle or not; its score by the battle reward's formula; the change of each Φ part, over the consequence window |
| answer | every option of a question | the same |
| hero | each of our heroes | its share of today's remaining Φ change; alive and ours tomorrow |
| turn | global | Φ change to the end of our turn (%), and our share's change; each resource tomorrow against today |

**The consequence window** runs from a turn decision to our next turn decision, including every question and battle in
between. The chest's gold or experience, the University's skill, and the guarded dwelling we agree to attack all land
in that MOVE's window. A battle after a "yes" counts as the MOVE's battle, and as the answer's. END has no window.

**Labels.** All are free from the records (Φ in the graph is the ledger's, to the gold), except the battle score. That
needs `AlphaMapAI::battleStart`/`battleEnd` to record both sides' assets at the start, and the outcome and casualties
at the end.

**Losses.**

- Value changes: symlog targets, regressed as two-hot over bins (DreamerV3's recipe).
- Yes/no targets: BCE.
- Each head is weighted 0.1-0.3 and logged separately, so a head that hurts can be found and removed.
- Never exported to the AI.

**`think_again`** concatenates the foresight hidden states and the known facts again, onto entities and onto
(hero, tile) pairs, then runs more layers.

## 8. `decide`: every logit a match, one softmax

`logit = act_now(actor) + query(actor) · key(target | actor) + pairnet(actor, target, its foresight, its known facts) + prior`

| Type | Actor x target |
|---|---|
| MOVE | hero x its window's tiles (the key: the tile's map vector + the hero's reach) |
| TRAVEL | hero x area |
| BUILD | town x building |
| RECRUIT | town x level x amount x destination |
| HIRE | town x tavern hero |
| ARMY | stack x holder x amount |
| UPGRADE | holder x stack |
| TRADE | resource x resource x amount |
| CAST, BUY | hero x spell or item |
| questions | (global + the hero asked) x option |
| END | the global token alone |

- **Compute.** The dot products are batched matrix multiplies. The pair network runs only on legal entries, a few
  thousand per decision at most.
- **One softmax over every legal action.** A type's log-probability is the logsumexp of its entries, so END competes
  directly with the best MOVE. The hand-made couplings and the separate type head go away. Each type's act-now bias
  absorbs the difference in table sizes.
- **Which hero moves:** the joint softmax over hero x tile picks it. Each hero's vector has read its own tiles, and
  through the trunk the other heroes'.
- The AI still samples a type, then an entry.

## 9. Values

V2, V5, V7, growth and V_end read the global token and the player tokens: our share against each rival's.

## Contract changes

- **Added:** the tile grid at the map's size.
- **Removed:** the per-window terrain and object fields, and the 8,192-object list (objects live on their tiles).
- **Kept:** the windows, for reach and MOVE only. TRAVEL and the other tables do not change.
- **Training:** from scratch. Nothing of today's weights fits.

## Open decisions

1. **Reach and MOVE.** Keep per-hero reach and MOVE in fixed 33x33 windows on the shared variable-size map
   (recommended).
2. **Rival identity.** Order rivals by turn order relative to us (recommended), not by strength, which changes during a
   game.
3. **Unit scales.** Set by hand (recommended) or fitted from recorded games.

## Ideas discussed and set aside

| Idea | Why set aside |
|---|---|
| Φ per 9x9 block as an auxiliary target | unbounded; does not scale with map size |
| Φ per block relative to other blocks | a block holds 0 or 1 army |
| Hidden-content targets per area: the user's "collectable value / (that + 1k)", "visible value / total value", "has an unowned town" | bounded and local, but a target over unseen content is hidden information, and one map's answer repeats over hundreds of its decisions. If used: weight the loss per game, not per decision, and watch the loss on held-out games |
| "Where will our hero be tomorrow" | learns our own policy back, and moves as it changes |
| Long horizons (+7 days, the game's end) for local units | too much luck for a small hidden state |
| Area or hero targets with no clear use downstream | map-centric targets are harder to use than hero-centric ones: the model must still know which area is meant |
