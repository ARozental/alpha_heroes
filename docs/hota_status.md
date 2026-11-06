# HotA branch: status, open decisions, TODO (2026-11-06)

The plan is docs/hota_transfer_plan.md. What the port loads, measured, is docs/hota_inventory.md. This file is where
the branch stands and what comes next.

## Where it stands

**Done** (commits on `hota`):
- 920284a: the plan.
- 6038bee: the inventory. Our VCMI 1.7.5 loads all 61 submods of the port. SoD ids stay put. HotA rebalances SoD content.
- 92a6ba9: Part 1's content match:
  - `agents/AlphaContent`: Content.h/.cpp, every table's keys by runtime id, matched by key against a vocabulary.
    It reports rows, unknown, absent, collisions and facts, plus FNV-1a content and vocabulary hashes.
  - `alpha_content_check`: the match from the command line.
  - The exporter's `vocab["content"]` section.
  - The Python mirror `content/`: match, append-only merge, is_prefix, `python -m content export|check|merge`.
  - 6 tests in content/tests (the real-export one needs ALPHA_CONTENT_EXPORTS).
- This commit: scripts/setup_hota.sh and this file.

**Verified:**
- SoD vocabulary against the SoD engine is strict-clean and the identity on every row.
- Against the HotA engine:
  - 775 unknown: HotA's content, under real keys.
  - 60 absent: SoD's two-way monoliths 41-100, whose subtype ids HotA reuses.
  - 0 collisions: HotA's hero camp, a second subtype also named "prison", is renamed core:prison#2.
- SoD grown by HotA (append-only) matches both engines strictly, with every SoD row at its runtime id.

**Training with HotA does not work yet.** If HotA games entered training today:
- The battle model has CREATURE_VOCAB 150. HotA's 53 creatures would share its unknown row.
- The campaign model's vocabularies are SoD-sized:
  - HotA factions, heroes and artifacts fall on untrained or unknown rows;
  - buildings past BuildingID 43 are not encoded (N_BUILDINGS 44).
- Φ, the reward and the priors have no value for HotA's towns, buildings or its 87 new object types: they count 0.
  The user: that must be fixed before any training with HotA.
- The agents do not run the match yet, so nothing stops such games from reaching the trainer.
- The battle generator is unchanged. It reads SoD's vocab.json and builds faction keys as "core:" + name
  (generator.py's guild_chance and short()), so it never makes a Cove, Factory or Bulwark battle.
- play.py and headless.sh copy SoD's mod preset into every user dir.

## Decisions waiting for the user

1. **Value tables per ruleset.** HotA's gameBalance submod changes SoD content itself:
   - 31 creatures: abilities, AI and fight value;
   - 8 artifacts;
   - 3 heroes' specialties;
   - Conflux's native terrain.

   One model for both games needs the values Φ, the reward and the priors read to be per ruleset, selected by the game's
   content hash, with identity rows shared by key. Recommended. The alternatives:
   - a key per ruleset for the changed entities (they lose their shared identity);
   - the port without gameBalance (not HotA as played).
2. **Battles first, or the plan's order?** Recommended: battles first. The battle model reads mostly stats from the
   engine, so it needs:
   - Part 1's rest;
   - Part 2 for the battle side (creatures, spells, abilities, obstacles, battlefields);
   - the creature values the exporter already gives;
   - the generator.

   That brings HotA towns into generated battles and battle training well before the campaign, which waits for Part 3's
   object and building values.

## TODO, by the plan's parts

**Part 1: the match in every process**
- [ ] Both agents (initGameInterface) and every tool (MakeBattleMap, MakeRandomMap, MapInfo) run
  `alpha_content::match` against the model's vocabulary. Where the vocabulary comes from (a file beside the model, or
  the inference server's handshake) is still to choose.
- [ ] Strict mode (`ALPHA_CONTENT_STRICT=1`, every training game): any unknown aborts the game with the list.
- [ ] Tolerant mode (evals): unknown goes to the table's unknown row, logged once, counted (`content_unknown`). Action
  targets with no safe unknown row (buildings, spells, skills, war machines, BUY) become not legal, counted
  (`content_unactionable`).
- [ ] Record headers and games.jsonl carry: the vocabulary hash, the vocabulary version (each table's length), the
  content hash, the strict flag.
- [ ] Supply accepts a record only with strict = 1 and a vocabulary version that is a prefix of the trainer's;
  otherwise it drops it and counts `dropped_content`. The trainer asserts at start that the games' exported vocabulary
  equals the model's.
- [ ] A tiny test mod (one creature, artifact, building, object with a subtype, terrain) under a test user dir: it shows
  exactly those as unknown; after a merge, 0 unknown with the new rows at the end.
- [ ] Load-order independence: the test mod before and after another mod gives the same rows by key and the same
  vocabulary hash.
- [ ] Record round trip: an agent's strict record decodes in the trainer to the same rows; a foreign vocabulary hash is
  dropped.
- [ ] The exporter's other tables (creatures, buildings, object_types...) by key too, so vocab.json has one keying.
  Today's tables stay as they are until their readers move (no behaviour change).

**Part 2: rows, not ids; arrays sized by the vocabulary**
- [ ] The agents write rows (the match's `rows[runtime id]`) into every content field:
  - battle: u_creature, h_type, g_battlefield, x_obstacle_id, u_effect_id, the spell and skill arrays;
  - campaign: h_type, h_class, t_faction, tile_terrain, tile_road, tile_object/tile_subtype (dense pair rows from the
    vocabulary), skills, spells, buildings per faction, artifacts.
- [ ] Codegen writes the vocabulary's sizes and hash into Schema.generated.h. The C++ `< N_*` guards that drop
  content go (the match covers it in strict mode).
- [ ] N_DWELLINGS 8: Dungeon's Portal of Summoning (index 7) is skipped today; Factory's 7th tier would be too.
- [ ] Python: `vocab.sizes()` replaces the vocab constants (constants.py CREATURE_VOCAB, ARTIFACT_VOCAB,
  FACTION_VOCAB, N_BUILDINGS...).
- [ ] The battle ability multi-hot over the vocabulary's bonus list (ABILITIES goes). SPELL_TELEPORT,
  SPELL_SACRIFICE, SECOND_TARGET_KIND and the timed-spell ids by key.
- [ ] Checkpoint migration `migrate_vocab(checkpoint, vocabulary)`: append rows to every table (identity rows at the
  module's init, value tables from the vocabulary), for campaign, battle and the league's snapshots. The live SoD
  checkpoints must give bit-identical outputs through it (the gate).

**Part 3: true values (the gate before any HotA training)**
- [ ] The exporter fails on a creature without cost or aiValue, an artifact without a price or slot, a faction without
  creatures per level, a building without a kind.
- [ ] The exporter emits object configs: reward lists, bank levels, dwellings' creatures, mines' produce, markets'
  modes, shrines' spell level, limiters.
- [ ] Python derives each object's class and Φ gain from them. object_values.csv becomes the override table and the
  regression test (the SoD diff clean or explained in docs/empire_value.md).
- [ ] Buildings valued by kind (dwelling, guild, produce, fort, market, blacksmith, tavern, shipyard, horde, grail,
  specials). Anything uncovered is 0 with a listed reason, never silently.
- [ ] Skills: Interference and Bulwark's Runes need Φ values and level-up priors.
- [ ] Engine constants (movement, morale and luck dice, experience, the damage-spell table) from the exported
  gameConfig. The hand copies in game_values and features are deleted or asserted equal.
- [ ] The coverage gate: a test plus a trainer start-up check. The trainer refuses a strict run while any object,
  building, skill or war machine lacks a rule.
- [ ] Per-ruleset value tables (decision 1).
- [ ] The 30 extra spell entries: projectile visuals (changedShootingVisuals), the cannon's siege shot. Check what each
  is and whether any is castable.

**Part 4: generator, maps, evals**
- [ ] generator.py: full keys instead of short(), `guild_chance[faction key]` instead of "core:" + faction, terrains
  and battlefields from the vocabulary with weights by key in generator.toml (an unknown key there is an error).
  Factions, classes and heroes come from the vocabulary, so the new towns appear by themselves. The hard-case
  samplers pick by ability.
- [ ] MakeBattleMap runs the match first and refuses a scenario naming unknown content.
- [ ] A mod preset per pool (SoD or HotA) in play.py and headless.sh. The match catches a user dir that differs.
- [ ] MakeRandomMap: a template allowlist (`campaign.templates`), and the template and content hash in games.jsonl.
  HotA's own templates are L/XL and wait for the map-size axis (cannot-do row 9).
- [ ] Eval sets:
  - HotA battle sets with the three towns, at fixed seeds;
  - a HotA campaign eval (cached maps per machine, a new baseline).
  - Keep every SoD eval to measure regression. MMAI was trained on SoD; BattleAI reads the config and stays fair.

**Part 5: training**
- [ ] Migrate both checkpoints and the league snapshots to the merged vocabulary.
- [ ] A mixed pool (a share of HotA games), strict on every training game. Freeze Φ edits around it: the value heads
  re-fit as after a Φ change.
- [ ] Gates: SoD evals hold within their standard errors over the first day; HotA evals rise; no `dropped_content` or
  `content_unknown`.

**Phase B items (measure, don't guess):** Nullkiller2 teacher games on HotA maps list the requests we have no action
for. Also open: spell research (a town action), the airship, the cannon (through the war-machine field), commanders,
stack experience.

## Building and running the branch

```sh
# the worktree's vcmi submodule is empty: build against the main checkout's engine source (same commit; the branch
# does not change the engine; if it ever must, run `git submodule update --init` in the worktree first)
H=$PWD; cmake -S ../alpha_heroes/vcmi -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DENABLE_MMAI=ON \
  -DENABLE_LAUNCHER=OFF -DENABLE_EDITOR=OFF -DENABLE_TEMPLATE_EDITOR=OFF -DENABLE_DISCORD=OFF -DENABLE_LOBBY=OFF \
  -DENABLE_TEST=OFF -DENABLE_GITVERSION=OFF -DEXTRA_AI_DIRS="$H/agents/UpLeftAI;$H/agents/AlphaBattleAI;$H/agents/AlphaMapAI"
nice ninja -C build -j 6     # all of it: VCMI runs from a build dir only with vcmiclient/vcmiserver beside config and Mods
scripts/setup_hota.sh        # the port and runtime/vcmi-user-hota

# the exports and the match
VCMI_USER_DIR=../alpha_heroes/runtime/vcmi-user build/bin/alpha_export_vocab /tmp/x/vocab_sod.json
VCMI_USER_DIR=$PWD/runtime/vcmi-user-hota build/bin/alpha_export_vocab /tmp/x/vocab_hota.json
VCMI_USER_DIR=$PWD/runtime/vcmi-user-hota build/bin/alpha_content_check /tmp/x/vocab_sod.json   # the HotA diff
python -m content merge /tmp/x/vocab_sod.json /tmp/x/vocab_hota.json /tmp/x/merged.json
ALPHA_CONTENT_EXPORTS=/tmp/x python -m pytest content/tests
```

AlphaContent is added by AlphaBattleAI's CMakeLists (`add_subdirectory`), so no machine's EXTRA_AI_DIRS changes. The
PC and the Mac's worker build pick it up at their next reconfigure.
