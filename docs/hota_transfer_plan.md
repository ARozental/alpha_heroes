# One model for SoD and HotA: the plan (2026-11-05)

The user's requirements (2026-11-05): the models must transfer to Horn of the Abyss through VCMI's port
(`vcmi-mods/horn-of-the-abyss` 1.8.005, VCMI ≥ 1.7.3); one model runs on both; ids must be matched and the match
**verified when an engine starts**; every object's class and worth must be right **before any training with HotA**;
identity embeddings for new creatures and artifacts may start at random, **their values must be true**.

The port's submods: `cove`, `factory`, `bulwark` (three towns), `cannon` (a fifth war machine), `interference` (a 29th
secondary skill), `highlandsTerrain`, `wastelandTerrain`, `artifacts`, `heroes`, `neutralCreatures`, `mapObjects`,
`templates` (HotA's random map templates), `spellResearch`, `airShip`, `gameBalance`, `confluxVaultofAshes`, a Bulwark
commander and stack experience (1.8.005). No new spells; creature abilities reuse VCMI's bonus types.

Inventory of what is bound to SoD today: campaign_model/constants.py and battle_model/constants.py (vocab sizes, fixed
arrays), agents/AlphaMapAI/Encoder.cpp and agents/AlphaBattleAI/Encoder.cpp (raw `getNum()` ids, `< N_*` guards that
drop the rest), agents/AlphaBattleAI/tools/ExportVocab.cpp (keys by `getJsonKey()`, no object subtype keys, no bank or
reward configs), campaign_model/priors (object_values.csv, object_subtypes.csv, bank_guards.csv from core config
only), constants.OBJECT_CLASSES (Obj ids), empire.py (building values by BuildingID), game_values (war machines and
the BUY list by id, DAMAGE_SPELLS, movement and morale constants copied from gameConfig), generator.py (`"core:"`
prefix, `short()` keys, hand-listed terrains and battlefields), MakeRandomMap.cpp (any template that fits), records
(contract hash only, no content hash).

## Part 1: content identity and the engine-start match

**Vocabulary.** vocab.json (ExportVocab) becomes the model's vocabulary: per table an ordered list of keys, growth
append-only, never reordered or removed (a retired key keeps a dead row). Tables: creatures, artifacts, spells,
secondary skills, heroes, hero classes, factions, buildings (per faction, rows in the faction's own list; core factions
keep BuildingID order 0..43 so the live weights map 1:1), object types, object subtypes (per type), terrains, roads,
battlefields, obstacles, bonus types, boat types. The model artifact (torch checkpoint and ONNX metadata) carries the
vocabulary and two hashes: the contract hash (shapes, as now) and the **vocabulary hash** (keys only, so numeric ids
and mod load order never enter it). Engine facts the model assumes (4 primary skills, 7 resources, 7 army slots, 19
worn slots, 187 hexes, the movement table, the morale and luck dice, the experience table) are in the vocabulary too
and checked, not copied into Python constants.

**The match: one function, one library, every process.** `alpha_content` (a small library both agents, the exporter
and the map tools link):

    Match ContentMatch::match(const Vocabulary & model)   // run at initGameInterface and in every tool's main

- For each table, for each loaded entity (`LIBRARY->creh->objects`, `arth`, `spellh`, `skillh`, `heroh`,
  `heroclassesh`, `townh` with each faction's buildings, `objtypeh->knownObjects()` with `knownSubObjects`, terrains,
  roads, battlefields, obstacles, `BonusType` names): its key → the model's row. Output per table: `rows[runtimeId]`,
  `unknown` (the engine has it, the model does not), `absent` (the model has it, the engine does not: fine), `collisions`
  (two entities one key: must be empty), and the engine's **content hash** (all loaded keys).
- Engine facts: compared value by value; any difference aborts.
- Two modes. **Strict** (`ALPHA_CONTENT_STRICT=1`, every training game): any `unknown` in any table aborts the game
  with the list of keys; the record header gets strict = 1. **Tolerant** (evaluation and play): unknown → the table's
  unknown row, each unknown key logged once, their counts in the game summary (`content_unknown` per table). Action
  tables (buildings, spells, skills, war machines, artifacts for BUY) have no safe unknown row, since two unknown
  buildings would share one mask bit, so in tolerant mode an unknown action target is encoded as not legal and
  counted (`content_unactionable`).
- The agents write **rows, not ids**, into records. The record header carries the model vocabulary hash, the vocabulary
  version (each table's length), the engine content hash and the strict flag. games.jsonl carries the same per game.
- The exporter runs the match against the vocabulary it is about to write and prints the diff against the previous
  vocab.json (what was added), so a re-export after installing a mod is a readable list.

**The Python mirror.** `content.match(engine_vocab, model_vocab)` with the same outputs, used by the trainer at start
(the games' user dir's exported vocabulary must match the model's), by supply (a record is accepted when strict = 1
and its vocabulary version is a prefix of the trainer's; otherwise dropped and counted, `dropped_content`), by the
checkpoint migration, and by `python -m content --check <user dir>` which prints the human diff ("unknown: 55
creatures, 46 artifacts, 59 heroes, 6 classes, 3 factions, 1 skill, 2 terrains, 41 object types, 3 battlefields").

**Verifying the match itself** (tests, Phase A):
- SoD vocabulary against the SoD engine: zero unknown, zero absent, zero collisions, and `rows` the identity on every
  core table (the bit-identical guarantee for the live checkpoints).
- A tiny test mod (one creature, one artifact, one building, one object with a subtype, one terrain) under a test user
  dir: the match reports exactly those as unknown; after a re-export, zero unknown and the new rows appended at the end.
- Load-order independence: the test mod loaded before and after another mod changes the numeric ids and changes
  nothing in `rows` by key or in the vocabulary hash.
- Record round trip: a record written by an agent in strict mode decodes in the trainer to the same rows the agent
  wrote; a record with a foreign vocabulary hash is dropped and counted.

## Part 2: arrays sized by the vocabulary, indexed by row

Everything indexed by an engine id today becomes indexed by row, sized by the vocabulary, with codegen writing the
sizes and the vocabulary hash into Schema.generated.h (so the C++ build, the model and the engine agree at start):

- Campaign: `h_skills`, `g_tavern_*` skills; `h_spells`, `h_spell_cost`, `g_tavern_spells`, `cast_mask` (the adventure
  spells are the vocabulary's spells with the adventure flag, not "ids 0..9"; TARGETED_SPELLS and INFO_SPELLS by key);
  `t_built` and `build_mask` as [town, building row of the town's faction]; `t_available_creature` and `recruit_mask`
  with **N_DWELLINGS = 8** (the Dungeon's Portal of Summoning at index 7 is skipped today, Factory's second tier 7
  would be); BUY by the artifact table's war-machine field (the cannon arrives by itself); `h_class`, `h_type`,
  `t_faction`, `tile_terrain`, `tile_road`, `tile_object`/`tile_subtype` (the dense pair rows generated from the
  vocabulary's object types and subtypes; object_subtypes.csv goes) all rows.
- Battle: `h_skills` [sides, skills]; the spell arrays and heads stay 70 wide but become vocabulary-sized by rule;
  SPELL_TELEPORT, SPELL_SACRIFICE, SECOND_TARGET_KIND, the timed-spell ids in rules.py by key; `u_creature`, `h_type`,
  `g_battlefield`, `x_obstacle_id`, `u_effect_id` rows; the ability multi-hot over the vocabulary's bonus list (the
  67-name ABILITIES tuple goes).
- Python model code: `vocab_index(ids, VOCAB)` keeps its unknown row; every `small_embedding(VOCAB + 1, d)` takes its
  size from the vocabulary module; the type-level vocab constants in constants.py are replaced by `vocab.sizes()`.
- The C++ guards `if(id < N_*)` go; the match guarantees coverage in strict mode.

## Part 3: true values for every piece of content (the gate before HotA training)

Identity rows may start at random (the user). What must be true are the tables Φ, the reward and the priors read:

- **Creatures and artifacts**: cost, fight value, aiValue, stats, speed, growth, upgrade, faction, abilities, artifact
  price, slot, class, combination parts, war-machine flag: all from the exporter, so true for whatever mod set it ran on.
  The exporter **fails** on a creature without cost or aiValue, an artifact without a price or slot, a faction without
  creatures per level, a building without a kind: nothing defaults silently.
- **Object classes and worth, derived from the engine's config.** The exporter emits per object type and subtype its
  key, handler name and valuing config: rewardable objects' reward lists (resources, experience, primary and secondary
  skills, spells, artifacts by class, creatures, mana, movement, luck, morale, bonuses; visit mode and limiters), banks'
  levels (guards, rewards, chances), dwellings' creatures, mines' produce, resources' kind, markets' modes, shrines'
  spell level. Python computes: the class from the handler (bank, dwelling, mine, resource, market, shrine, monster,
  town, hero, garrison, boat and shipyard, teleport, configurable → by its rewards: pickup, treasure, skill, boost,
  info) and Φ's immediate gain from the rewards at Φ's own rates (RESOURCE_PRICES, creature cost, artifact class mean
  price, experience as a share of a level at LEVEL_VALUE, skills at hero-value-design's values, spells at spell_value,
  luck and morale and movement at Φ v3's boost rates), with the limiters (once per hero, per player, weekly).
  object_values.csv stays as the override table and becomes the **regression test**: the derived SoD values are
  diffed against it; every difference is either a hand override with a reason in docs/empire_value.md or a fix to
  the derivation. No HotA number is trusted until the SoD diff is clean.
- **Buildings by kind** (empire.py): dwelling (level → growth × unit worth), mage guild (spells by level), produce
  (daily income at prices), fort/citadel/castle (growth and defence), marketplace, blacksmith, tavern, shipyard, horde,
  grail and special buildings from their produce, bonuses and configuration; anything the rules do not cover is 0
  **with a listed reason**, never a silent 0. Same regression diff against today's SoD values.
- **Skills**: Φ's per-skill values and the level-up priors keyed by skill key; a new skill (Interference) needs a value
  before strict training, enforced by the coverage test below.
- **Engine constants** (movement table, morale and luck dice, experience, the damage-spell table): read from the
  exported gameConfig, and the hand copies in game_values and features deleted or asserted equal.
- **The coverage gate**: a test and a trainer start-up check that, for the vocabulary in use, every visitable object
  type and subtype has a class other than OTHER or is on the explicit "other" list (signs, decorations, placeholders)
  with a reason, and has a worth rule; every building of every faction has a value rule or a listed 0; every skill has
  a value; every war machine a price. The trainer refuses to start a strict run while the gate lists anything. This is
  the user's "fixed before we even think of training with it".

## Part 4: generator, maps, scenarios, evals

- generator.py: full keys instead of `short()`, `guild_chance[faction key]` instead of `"core:" + faction`; terrains
  and special battlefields drawn from the vocabulary with weights in generator.toml keyed by key (an unknown key in
  the toml is an error); factions, classes and heroes already come from the vocabulary, so the new towns appear by
  themselves; the hard-case samplers pick by ability and hold. Scenario files name VCMI identifiers, which the battle
  map tool resolves with the mod enabled; the tool runs the match first and refuses a scenario naming unknown content.
- Campaign maps: MakeRandomMap gets a template allowlist from the config (`campaign.templates`: SoD templates, HotA
  templates, or any) and writes the template name and the content hash into games.jsonl (the template is already
  there). The RMG places the new factions' towns on its own. Training stays at 36 × 36 until the map-size axis
  (MAP_TILES, TRAVEL_GRID: cannot-do row 9) is raised; HotA's own templates are L/XL and wait for that.
- Mod settings: scripts/headless.sh and battle_model/play.py copy one modSettings.json into every user dir; the match
  at start is what catches a dir that differs. The trainer asserts the games' exported vocabulary equals the model's.
- Evals: keep every SoD eval (battle sets, the 48-game campaign eval) to measure regression; add HotA sets: battle
  scenario sets with the three towns (fixed seeds, as castle and hard are) and a campaign eval set generated with the
  mod (cached maps per machine; a new baseline). MMAI was trained on SoD, so its HotA numbers mean less; BattleAI reads
  the config and stays a fair baseline.

## Part 5: records, migration, training

- Records: rows, header with model vocabulary hash, vocabulary version, content hash, strict flag; supply accepts a
  record whose vocabulary version is a prefix of the trainer's (append-only growth makes every older record valid).
- Migration: `migrate_vocab(checkpoint, new_vocab)` appends rows to every table (identity rows at the module's random
  init, value tables from the vocabulary), for the campaign and battle checkpoints and the battle league's snapshots
  (checkpoint-compatibility rule); the BUILD head's per-faction rows keep core factions in BuildingID order.
- Training: one model, a mixed game pool (a share of games with the mod enabled, the rest SoD), strict mode on every
  training game. Expect the value heads to re-fit (new objects, new values): plan it like a Φ change and freeze other
  Φ edits around it (the 2026-11-05 value-loss jump cost ~100 updates).

## Part 6: what the port adds that needs a decision later (measured in Phase B, not guessed)

Spell research (a town action per guild level: a BUILD-like entry), the airship (a boat: cannot-do row 8), the cannon
(arrives through the war-machine field and the creature table), Interference (a row plus its Φ value), stack
experience (the battle encoder reads current stats; nothing to add), commanders (units in the 14 slots in battle;
ignored by the campaign at first), new bonus types the port may use (the bonus list is vocabulary-sized; the match
reports any the model lacks).

## Phases, with acceptance gates

**Phase A, SoD only, no behaviour change.** The content library and the match in both agents and all tools; rows
instead of ids; vocabulary-sized arrays and N_DWELLINGS 8; the exporter's additions (subtype keys, reward and bank
configs, bonus list, gameConfig, completeness failures); derived object and building classes and values with the SoD
regression diffs; the coverage gate; record headers and supply checks; the Python mirror and `--check`. Gates: the
four match tests pass; the live checkpoints give bit-identical outputs on SoD games through the translation; the SoD
derived-value diffs are clean or explained; smoke games 0 errors; deployed through drain_restart with a schema bump
on each contract. Order inside A: the library and the match first (it is the safety net for everything after), then
the exporter, then the derived tables and the gate, then the arrays.

**Phase B, the port on the Mac, nothing trains.** Install the port (it needs the player's HotA files) into a
separate user dir, enable its submods, re-export, read the `--check` diff. Nullkiller2 teacher games (campaign_model
.teacher) list the requests we have no action for. Smoke games in tolerant mode with the SoD checkpoints find crashes
and the unknown inventory. Fill the gate: every new object, building and skill valued by rule or by a listed override,
until `--check` and the coverage gate report nothing. Write the HotA eval sets. Gate: the coverage gate is empty for
the merged vocabulary and strict smoke games run 0 errors.

**Phase C, migrate and train one model.** Migrate both checkpoints and the league snapshots to the merged vocabulary;
start the mixed pool in strict mode; both eval suites run. Gates: the SoD evals hold within their standard errors over
the first day; the HotA evals rise; no `dropped_content`, no `content_unknown`.

**Phase D, later.** The map-size axis and HotA templates at native sizes; spell research as an action; boats and the
airship; commanders in the campaign.

## Not doing
Feature-projected initialisation of identity rows (the user: random is fine; values must be true, which Part 3
gives); an AI plugin for HotA's own engine; WoG.
