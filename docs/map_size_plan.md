# One campaign model for every map size: the block grid

Written 2026-11-08 (e5) for the implementer. The user's constraints: training stays almost only on 36x36 maps with the
day caps as they are; small-map training must be left alone, in cost and in outputs; the model must be able to play
72x72, 108x108 and 144x144 maps (and HotA's larger ones) when evaluated later, having seen few or none of them; and it
must be possible to give it one large map to train on, once, to see what happens.

## Implemented 2026-11-08 (as below, with these differences)

- Step 5: the side embeddings use `padding_idx=1`. Row 1 (every small map) never learns; a zero-initialised row 1
  would have trained into a bias on 36 x 36 games that bigger maps then lack.
- Step 0: `features.block_side` already existed (TRAVEL's areas, the object counts): reused, no copy in model_v2.
- Step 1: water and blocked are judged over a block's revealed tiles, not its on-map ones (no reading through fog).
  Monsters are ranked by seen count x AI value, a quantity id's count being features.QUANTITY_COUNTS'.
- Step 6: `play.time_scale(size)` is the factor (side^2), applied in the supply's games, the evaluation, teacher.py,
  imitate.py and play.py's own games.
- Checked: small maps bit-identical (tests/test_map_size.py against outputs saved before the change), the 36 x 36
  forward's time unchanged (alternating old and new, 312-318 ms both), and a 144 x 144 self-play game (side 4, the
  grid all on the map, 0 errors, TRAVEL every day, recorded logp within 3e-5 of the trainer's).

## The design in one paragraph

The whole-map grid stays 36 x 36 cells a level (`MAP_TILES`), but a cell becomes a *block* of `side x side` tiles,
`side = ceil(max(width, height) / 36)`: 1 tile on every map up to 36 (today's model, bit for bit), 2 on 72, 3 on 108,
4 on 144, 7 on HotA's 252. The per-block tile counts (`g_tiles`, `MAP_GRID`) and TRAVEL's areas (`2 x side` tiles) are
already built this way in the encoder; only the eight `tile_*` arrays are not. Everything tile-exact stays tile-exact
at any size: the hero windows (`w_*`, 33 x 33 tiles), MOVE and its mask, every position on the wire (`h_pos`, `t_pos`,
`m_pos`, `a_pos`, in tiles), the object list, the armed objects. The model maps a tile position to its block with one
integer division, measures tile-reader distances in tiles (block distance x side), averages the reach it scatters onto a
block over the window tiles that land there, and receives the side as a zero-initialised input. The wire layout, the
records, the inference server, the trainer's batching and the CUDA graphs do not change, and the cost of a decision is
the same at every map size.

## Why this and not a true-size grid

Measured 2026-11-08: the contract carries 259k scalars a decision, of which the `tile_*` arrays are 20.7k (8%) and the
only ones that grow with the map (windows 87k, object list 82k, `g_tiles`, TRAVEL and the entities are fixed). On the
Mac's CPU the U-Net is 19% of the forward, the tile reader 16% and the tile encoder 2%: about 37% of the forward is
tile-bound. A tile-exact 144 x 144 grid has 16x the tiles, so that part alone would be ~6x today's whole forward and the
tile reader's per-decision tensors would go from ~3 MB to ~45 MB; carrying it also needs a contract per size class,
minibatches bucketed by class with a tile budget, graphs per class on the trainer and the server. Raising `MAP_TILES` to
144 for everyone is worse still: every small-map decision would carry 16x the tiles as padding and, since the graphs fix
shapes, every batch would be XL-shaped. The block grid costs nothing on small maps and keeps the whole map at the same
relative resolution the model learned, which is also the better transfer story: a block is "1/36 of the map" on every
map, and the model reads days and movement from the pathfinder's numbers (`w_turns`, `travel_days`), not from block
distance. Its price is a coarser far view on big maps, which `g_tiles`, TRAVEL's areas and the exact entity tokens
soften. If an XL eval shows the policy ignoring far objects it could see, the next step is a 72-cell grid class, not a
tile-exact one.

## Step 0: the side, on both sides of the wire

- C++: `side = std::max(1, (std::max(size.x, size.y) + MAP_GRID - 1) / MAP_GRID)` already exists in Encoder.cpp
  (encodeMap's `g_tiles` loop; TRAVEL's `area = 2 * side`). Reuse it.
- `campaign_model/constants.py`: `MAP_SIDE_MAX = 8` (generated into Schema.generated.h like the other constants). The
  encoder logs an error and counts it when `side > MAP_SIDE_MAX`.
- `campaign_model/model_v2.py`: one helper, `block_side(x)`:
  `torch.div(x["g_map"][:, :2].amax(dim=1) + S - 1, S, rounding_mode="floor").clamp(min=1)` -> `[B]` int64. Computed
  once per forward (in `domain_knowledge`, stored as `known.side`) and passed where needed. No host sync anywhere: it
  is a tensor, never `.item()`.

## Step 1: Encoder.cpp, `encodeTiles` per block

- `side == 1`: keep today's loop verbatim, so every training game's wire stays bit-identical. Branch once at the top.
- `side > 1`: for each level z < min(size.z, MAP_LEVELS) and each block (row, column) in [0, 36)^2, the tiles with
  `x in [column * side, min((column + 1) * side, size.x))` and `y` likewise; a block with no tile on the map is left at
  its fill values (as today past the map's edge). Per block:
  - `tile_flags`: `TILE_ON_MAP` if any tile is on the map; `TILE_REVEALED` if any tile is visible; `TILE_WATER` if more
    than half of the block's on-map tiles are water; `TILE_BLOCKED` if every revealed tile is blocked; `TILE_VISITABLE`
    if any revealed tile is visitable; `TILE_GUARDED` if any revealed tile is in a monster's zone;
    `TILE_VISITED_BY_PLAYER` from the chosen object (below); `TILE_EXACT_COUNT` from the chosen monster (below). An
    unrevealed block gets `TILE_ON_MAP` only, as an unrevealed tile does today.
  - `tile_terrain`: the most common terrain id among the revealed tiles (ties: the smallest id). `tile_road`: the most
    common nonzero road id among them, 0 if no tile has a road.
  - `tile_object`, `tile_subtype`, `tile_owner`: the block's chosen object, over the revealed tiles' `topNonHero`
    objects: rank 0 a town, rank 1 any object that is not a wandering monster and that we have not visited
    (`!obj->wasVisited(playerID)`), rank 2 a wandering monster, rank 3 the rest; the lowest rank wins; ties among
    monsters go to the largest seen count x `getAIValue()`, other ties to the first in scan order (y, then x).
    `TILE_VISITED_BY_PLAYER` is its `wasVisited(playerID)`.
  - `tile_creature`, `tile_count`: the block's strongest wandering monster (seen count x the creature's `getAIValue()`),
    whatever the chosen object, so a guard is never hidden by a mine in the same block; the count and `TILE_EXACT_COUNT`
    as today (`seenCount` over our heroes).
- Remove the `size.x > MAP_TILES` error. Nothing else in the encoder changes: positions stay in tiles, the windows,
  `move_mask`, TRAVEL and `g_tiles` are untouched. The label on the change: "block summary rule, 2026-11-08".

## Step 2: model_v2.py, tile positions to blocks

- `map_index(positions, b, side)`: `bx = torch.div(px, side[:, None], rounding_mode="floor")`, `by` likewise;
  `inside = (bx >= 0) & (bx < S) & (by >= 0) & (by < S) & (pz >= 0) & (pz < L)`; the flat index as today with `bx`,
  `by`. Caller: `see()`'s stamping of heroes and towns.
- `window_tile_index(x, windows, side)`: `tx`, `ty` in tiles as today; `side_k = side.index_select(0, decision)` with
  `decision = windows // HO`; `bx = floor(tx / side_k)`, `by = floor(ty / side_k)`; `inside` and the index as above.
  When `side > 1` several window tiles share one block; `index_select` with repeated indices is what the callers
  (`pairs`, the reach scatter) already do, so nothing else changes.
- `tile_positions()` stays as it is: block coordinates.
- Floor division everywhere (`torch.div(..., rounding_mode="floor")`): a negative tile lands at -1 and is outside. At
  `side == 1` every formula is exactly today's.

## Step 3: TileReader, distances in tiles

- `distance_buckets(delta, side=None)`: `distance = max(|dx|, |dy|) * side` with `side` broadcast `[B, 1, 1]` over
  `[B, N, M]`; the rest unchanged (the buckets already reach "96+ tiles").
- `TileReader.forward(..., side)`: `delta = tile_positions[None, None] - floor(positions / side)[:, :, None]`;
  `bucket = distance_buckets(delta, side)`. At `side == 1` identical to today. The token-to-token bias
  (`attention_bias`, `PairBias`) works on tile positions and does not change.

## Step 4: the reach scatter, averaged per block

In `see()`, after `reach = reach.index_add(0, torch.where(inside, index, 0), values)`, add
`count = torch.zeros(b * HO * L * S * S, 1).index_add(0, torch.where(inside, index, 0), inside.float().unsqueeze(-1))`
and `reach = reach / count.clamp(min=1)`. At `side == 1` at most one window tile of a hero lands in a cell, so the
count is 0 or 1 and the result is bit-identical. Graph-safe: fixed shapes, no data-dependent sizes.

## Step 5: the side as an input, zero-initialised

- `self.side_tile = zero_output(nn.Embedding(MAP_SIDE_MAX + 1, cell_dim))`, added to `state.tiles` right after
  `encode_tiles` (before `take_sides`); `self.side_global = zero_output(nn.Embedding(MAP_SIDE_MAX + 1, dim))`, added to
  `state.glob` in `encode`. Index `side.clamp(max=MAP_SIDE_MAX)`.
- Add `"side_tile"` and `"side_global"` to `model.ADDED_HEADS`, so an older checkpoint loads with them at 0 and gives
  the same outputs (the existing test pattern, `test_an_old_checkpoint_loads_...`).
- The global encoder already feeds the map's width and height as amounts (encoders.py, GlobalEncoder:
  `amount(size[:, 0], "tiles")`), so the model knows the map size today; these two embeddings are what a model trained
  on big maps can use to read a block differently from a tile. They stay at 0 until such training happens.

## Step 6: the knobs for playing a big map

- `config.eval_size: int = 36`; evaluate.py uses it instead of `EVAL_SIZE` (the eval map's name already carries the
  size, so the cached 36 maps stay valid). An eval at another size writes to `eval_<size>.csv` in the run directory,
  not the run's `eval.csv`: its columns are fixed and the live file must not change.
- `config.big_game_every: int = 0` and `config.big_game_size: int = 144`: every k-th training game drawn is a map of
  that size (0: never). This is the "one large game" switch; `config.sizes` stays `(36,)`.
- The per-game wall limit scales with the area: `game_seconds * side * side` wherever a game's `max_seconds` is set
  from the config (the learner's game runs, teacher.py, play.py). The day caps (`days_min`, `days_tail`, `days_max`,
  `eval_days`) stay as they are: that is the user's turn limit.
- `make_map(seed, players, size, ...)`, `play.py --size` and `teacher.py` already take a size. The windows budget
  (`windows_for`) is per hero and unaffected.

## Step 7: acceptance

1. Unit tests: `map_index`, `window_tile_index` and `distance_buckets` with `side == 1` equal today's formulas (keep
   the old formulas in the test) on random positions; with `side == 4` and positions in [0, 144): every index is inside
   `[0, L * S * S)`, one hero's 33 x 33 window maps to at most 9 x 9 distinct blocks, and each bucketed distance is
   within `side` tiles of the exact Chebyshev distance.
2. Bit-identity at 36: `tensors(rich_inputs(3, 2))` through the model before and after the change, `torch.equal` on
   `move_logp`, `type_logp`, `travel_logp`, `value`, `fs_move` (save the "before" outputs from the current commit as a
   golden file under campaign_model/tests/data). Then, outside the tests, a recorded minibatch from the live run's spool
   through the live checkpoint before and after: identical logits.
3. An XL synthetic forward: `rich_inputs` with `g_map = (144, 144, 2)` and `h_pos`, `t_pos` scaled x4 (the `tile_*`
   arrays keep their shape): the forward runs, every output is finite, and every output shape equals the 36 case (the
   graphs capture the same shapes).
4. An old checkpoint loads strictly after `with_added`, with the two embeddings at 0, and gives the same outputs.
5. Engine: one headless 144 x 144 game, 2 players, against Nullkiller2 (`play.py --size 144`, the wall limit x16): no
   `MAP_TILES` error, no "entry has no meaning", TRAVEL entries present every day, the records decode
   (`records.decode_wire`) and run through the model. The 36 path is the old code and needs only a 36 game to pass.
6. `runtime/check_graphs.py` on the PC before any deploy (docs: deploy safety): the new ops must stay capturable, so
   no `.item()`, no `nonzero`, no shape that depends on the data.
7. Cost: the update's seconds and `runtime/bench_inference.py` at 36 unchanged within noise.

Order of work: Step 1 first, so XL records exist to test the model against; then Steps 0, 2 to 5 together with the
tests of Step 7; Step 6 last.

## Not in this plan

- Raising `MAP_TILES`, a tile-exact grid, size classes, per-class graphs or a tile budget (see "Why this").
- The MOVE window radius (16). MOVE is a same-day move to a window tile (`move_mask`: `turns == 0`); a tile farther
  than a day is TRAVEL's, an 18 x 18 grid of areas a level (`2 x side` tiles each) with the days to each area
  (`travel_days`), the hero walking the path as far as today's movement goes. A same-day target beyond 16 tiles, a fast
  hero on a road, takes two MOVE decisions in the day; a far target takes TRAVEL, then MOVE once in range. The whole
  map is visible at block resolution, and TRAVEL is already 11% of the live policy's decisions and is labelled by the
  teacher, so nothing is unlearnable. On 144 maps a trip may take five TRAVEL days where small maps need one or two;
  the TRAVEL prior's distance discount covers the untrained part.
- The windows (`w_*`), the object list (`m_*`), `g_tiles`, TRAVEL, the server, the records, the spool.
