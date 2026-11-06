# HotA through VCMI: what our engine loads, measured (2026-11-06)

**The port.** `vcmi-mods/horn-of-the-abyss`, the release zip `horn-of-the-abyss-vcmi-1.7.zip`.
- Size: 442 MB, sha256 d10feca4…f7741.
- Contents: `content.zip` (the assets), `mod.json`, 30 submods.
- The repository entry in runtime/vcmi-user/downloads/hota.json says 1.8.005, VCMI ≥ 1.7.3. Our engine is 1.7.5.
- It ships its own assets, so no HotA game files are needed.
- Installed for the branch only, at runtime/vcmi-user-hota:
  - `Mods/hota` points at the extracted zip.
  - `Data` points at runtime/vcmi-user/Data (SoD's).
  - The preset is `vcmi, core, hota`.

**It loads.** The exporter (`alpha_export_vocab`, unchanged) on that user dir:
- All 61 submods "Loading mod: OK", and "All game content loaded".
- Errors are only cosmetic: SoD music files not found, and an unknown language entry "tchinese".

**What it adds** (the exporter's tables, SoD → HotA):

| table | SoD | HotA |
|---|---|---|
| creatures | 150 | 203 |
| artifacts | 144 | 167 |
| heroes | 163 | 245 |
| hero classes | 18 | 24 |
| factions | 10 | 13 |
| buildings | 342 | 458 |
| object types | 170 | 257 |
| spells | 104 | 134 |
| secondary skills | 28 | 30 |
| terrains | 10 | 12 |
| battlefields | 24 | 33 |
| obstacles | 125 | 172 |
| factions with creatures | 9 | 12 |

The 30 extra spell entries need a look: the plan expected none (creature abilities? spell research?).

**SoD ids stay put.** HotA appends after SoD in every keyed table: creatures 150/150 at the same index, artifacts 144/144, skills 28/28, spells 104/104, heroes 163/163, factions 10/10.

**HotA changes SoD content** (the `gameBalance` submod). The same key has different true values in the two games:
- 31 SoD creatures: abilities, ability values, AI and fight value (e.g. Monk, Zealot, Titan, Efreet Sultan, Naga).
- 8 artifacts: Ballista's price, Badge of Courage's bonuses, the class of four.
- 3 heroes' specialties: Thorgrim, Galthran, Haart (undead).
- Conflux's native terrain.
- Every spell's guild chances (new factions in the table).

**Decision needed (the plan assumed one value per key).** One model on both games needs value tables per ruleset:
- the stats, costs, AI values and abilities Φ, the reward and the priors read;
- selected by the game's content hash;
- identity rows shared by key.

The alternatives:
- a key per ruleset for the changed entities (31 creatures etc. lose their shared identity);
- the port without gameBalance (not HotA as played).

Recommended: per-ruleset value tables.

**Exporter gaps the match needs fixed** (Part 1):
- Buildings are not unique by key: core factions share `core:mageGuild1` etc. The key must carry the faction.
- Object types are keyed by display name: SoD's "Quest Guard" is missing under HotA (renamed). They need VCMI's identifiers, with subtype keys.

**The match by key (Part 1's library, agents/AlphaContent; Python mirror content/).**
- SoD vocabulary against the SoD engine, strict: 193 tables, 2,219 keys, all matched, the identity on every row.
- Against the HotA engine:
  - 775 unknown: HotA's content, now under its real keys (hota.cove:nix, hota.factory:mageGuild5, hota.bulwark:runes, hota.mapobjects:warlocksLab...).
  - 60 absent: SoD's two-way monolith subtypes 41-100, whose subtype ids HotA reuses for its decoration monoliths. An id-based encoder would have mapped those silently.
  - No collisions once HotA's second prison subtype gets its own key. Its hero camp is also named "prison", so it becomes `core:prison#2`, listed as a rename.
- The SoD vocabulary grown append-only by HotA (`python -m content merge`):
  - matches the HotA engine strictly with 0 unknown;
  - matches the SoD engine strictly too;
  - keeps every SoD row at its runtime id, so today's checkpoints map 1:1.
