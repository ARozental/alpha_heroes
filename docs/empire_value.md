# What an empire is worth (Φ), version 3

Φ is the campaign model's measure of an empire, in gold (`campaign_model/empire.py` from the daily ledger,
`features.empire_value` the same in the graph, equal to the gold). The campaign labels are a player's share of all
players' Φ 2, 5 and 7 days after a decision and at the game's end. So anything Φ leaves out is learned only from what
it leads to, over those horizons, with their noise. Anything Φ prices pays the day it happens.

This page lists everything Φ prices and the rule for each, then every map object and town building with what it gives
and what Φ adds for it, then what Φ leaves out and why. The numbers are `EmpireConfig`'s defaults.

The rule behind every price (the user, 2026-10-27): buying anything at its list price leaves Φ unchanged. Value appears
when what was bought produces: through pickups, fights and captures.

## 1. The rules

| Asset | What Φ counts | Numbers |
|---|---|---|
| Resources | the price list (`game_values.RESOURCE_PRICES`) | gold 1, wood and ore 250, mercury, sulfur, crystal and gems 500 |
| Town buildings | their cost | an income building (halls, resource silo, grail): the more of its cost and 5 days of the income it adds; the village hall: 5 days of its income; a dwelling: its cost less the first batch (u x growth) it comes with |
| Income without a price | 5 days of its daily income | mines and every other owned object with an income: a gold mine 5,000, a wood or ore mine 2,500, a rare one 2,500 |
| Creatures | cost + u each, wherever they are (heroes, town garrisons, mine garrisons) | u = 0.25 x the cost of the creature's unupgraded form; recruiting and upgrading at list price are neutral |
| Creatures for hire | u each, in our towns and our flagged dwellings | a week's growth adds Φ the day it arrives (Monday), never before |
| A hero | 1,500 + its skills; the strongest hero's skills in full, every other hero's at half; minus 100 x heroes² for all heroes | a hire (2,500) with a week's army is a gain, one with a lone creature a loss |
| Primary skills (without items) | the k-th point beyond a fresh hero's five: 400 x k | points 6, 7, 8: 400, 800, 1,200 |
| Secondary skills | each level beyond a fresh hero's two | 2,000 (a University's price) |
| Levels | each level 200 (its specialty), and the way to the next level at what that level-up brings | a level-up brings a primary point, a skill level and 200: ~2,600 at level 1, ~4,200 at 5, ~6,200 at 10. Experience short of a level counts pro rata |
| Spells (v3) | each spell in the hero's spellbook, among its skills | 500 x the spell's level: 500-2,500 |
| Luck and morale until the next battle (v3) | each 24th of battle chance they add, over the hero's other luck and morale, x the army they lift | 2% of the army's worth (cost + u) per 24th; luck lifts every creature, morale only the living (not undead, non-living or war machines); +1..+3 is 1..3 24ths, bad morale -1..-3 is -2..-6 24ths (lost turns), bad luck does nothing (VCMI's `gameConfig.json`); capped at +-3; gone when the next battle uses it |
| Artifacts | their price in the game's table | the spellbook 500; a ballista 1,250, an ammo cart 500, a first aid tent 375 (half price); the catapult and the Grail 0 |
| Mana | 2 gold x spell power x mana, for the mana 3 casts of the hero's dearest combat spell need | without a combat spell, nothing |
| Map knowledge | each tile the player ever revealed | 1 |
| A player out of the game | 0 | |

## 2. Map objects

The map objects in VCMI's `config/objects/*.json`, as the random maps place them. What each gives comes from that
config. Decorations (trees, rocks, lakes) give nothing and are left out.

### Picked up once

| Object | Gives | Φ adds |
|---|---|---|
| Resource pile | 5-10 wood or ore, 3-5 of a rare resource, or 500-1,000 gold | their price |
| Treasure chest | 1,000 / 1,500 / 2,000 gold, or 500 / 1,000 / 1,500 experience (the player picks); sometimes an artifact | the gold, or the experience by the level rule (a fresh main hero's 1,000 experience: a level, ~2,600) |
| Campfire | 4-6 of a resource and 400-600 gold | their price |
| Flotsam | nothing, 5 wood, 5 wood and 200 gold, or 10 wood and 500 gold | their price |
| Sea chest | gold, or an artifact | its price |
| Shipwreck survivor | an artifact | its price |
| Lean-to | 1-5 of a resource | its price |
| Wagon | a resource stack, an artifact, or nothing | its price |
| Corpse | sometimes an artifact | its price |
| Warrior's tomb | an artifact, and -3 morale until the next battle | the artifact's price, less the morale (v3: 6/24 of the living army's turns lost, -12% of its worth) |
| Artifact | the artifact | its price |
| Spell scroll | the scroll (an artifact) | its price |
| Pandora's box, event | anything below | each part by its rule |
| Ocean bottle, sign | a message | nothing |

### Visited each week

| Object | Gives | Φ adds |
|---|---|---|
| Windmill | 3-6 of a rare resource | their price |
| Water wheel | 500 gold (1,000 after the first week) | 500-1,000 |
| Mystical garden | 5 gems or 500 gold | 2,500 or 500 |
| Magic spring | mana to twice the maximum | the mana rule (within 3 casts of the dearest combat spell) |
| Magic well | mana to the maximum | the mana rule |

### Owned (flagged)

| Object | Gives | Φ adds |
|---|---|---|
| Mine, abandoned mine | its daily income | 5 days of it |
| Creature dwelling (one level or several) | creatures for hire each week | u for each creature there, now and as each week's growth arrives |
| Town | buildings, income, a garrison, creatures for hire | its buildings by the building rule, the garrison, the creatures for hire |
| Garrison | a guarded pass | the troops in it, as an army |
| Lighthouse | more movement at sea | nothing (movement) |
| Shipyard | boats | nothing |

### A hero's skills (once per hero)

The skills count in full for our strongest hero and at half for the others (section 1).

| Object | Gives | Φ adds |
|---|---|---|
| Arena | +2 attack or +2 defense | two primary points: 1,200 for a fresh hero, more the more points it has |
| Mercenary camp, Marletto tower, Star axis, Garden of revelation | +1 attack, defense, spell power or knowledge | one primary point |
| Library of enlightenment | +2 to each primary skill (from level 10, earlier with Diplomacy) | eight primary points |
| School of war, school of magic | +1 attack or defense / spell power or knowledge, for 1,000 gold | the point less 1,000: a loss until the hero's third point beyond five |
| Tree of knowledge | a level, free, or for 2,000 gold or 10 gems | the level-up less the price |
| Learning stone | 1,000 experience | the level rule |
| Witch hut | a secondary skill | 2,000 |
| University | secondary skills at 2,000 each | neutral for the strongest hero, a loss for the others |
| Scholar | a spell, a secondary skill or a primary point | by its rule |
| Shrine of magic incantation, gesture, thought | a level 1, 2 or 3 spell (with a spellbook; level 3 needs Wisdom) | v3: 500, 1,000 or 1,500 |
| Pyramid | a level 5 spell, after a fight | v3: 2,500, and the fight |

### Luck and morale until the next battle (v3)

The amounts below are for a hero leading a 20,000 army, starting with no luck or morale.

| Object | Gives | Φ adds |
|---|---|---|
| Buoy | +1 morale | +400 |
| Temple | +1 morale, +2 on the seventh day | +400 / +800 |
| Fountain of youth | +1 morale, 400 movement points | +400 |
| Watering hole | +1 morale, 400 movement points | +400 |
| Oasis | +1 morale, 800 movement points | +400 |
| Mermaids, faerie ring | +1 luck | +400 |
| Swan pond | +2 luck, and the day's movement ends | +800 |
| Fountain of fortune | -1..+3 luck at random | the luck it gives (bad luck: nothing) |
| Idol of fortune | +1 luck (days 1, 3, 5), +1 morale (2, 4, 6), both (7) | +400 / +800 |
| Rally flag | +1 luck, +1 morale, 400 movement points | +800 |
| Stables | +400 land movement for the week; cavaliers become champions | the champions' worth over the cavaliers'; nothing for the movement |

### Fights

| Object | Gives | Φ adds |
|---|---|---|
| Wandering monsters | experience, the way open; some join | the experience by the level rule; the troops lost and the troops joined at their worth |
| Creature banks (cyclops stockpile, dwarven treasury, griffin conservatory, imp cache, medusa stores, naga bank, dragon fly hive, crypt, derelict ship, shipwreck, dragon utopia) | gold, resources, creatures, artifacts | each by its rule; the troops lost |
| An enemy hero, an enemy town | its army destroyed, a town taken | the rival's Φ falls (the labels are shares); the experience; the town's worth |
| Prison | a hero | 1,500 and its skills |
| Sirens | experience for 30% of each stack | the experience less the troops |

### Trades (at or below the list price)

| Object | Gives | Φ adds |
|---|---|---|
| Trading post, black market, freelancers' guild, altar of sacrifice | resources, artifacts, creatures or experience, for others | each side at its price: an artifact at its price is neutral, a resource trade's spread a loss |
| Hill fort | upgrades at 0% (level 1), 25%, 50%, 75%, 100% (level 5 up) of the town's upgrade cost | the upgrade's worth less what it costs: a gain below level 5 |
| War machine factory | war machines at full price | half their price: a loss |
| Refugee camp | creatures for hire | u for each one bought |
| Tavern | heroes for hire | the hero's worth less 2,500 |
| Den of thieves | information | nothing |

### Map knowledge

| Object | Gives | Φ adds |
|---|---|---|
| Redwood observatory, pillar of fire | the land around | 1 a tile revealed |
| Cartographer | a whole map level (water, land or underground), for 1,000 gold | 1 a tile revealed, less 1,000 |
| Eye and hut of the magi | what the eyes see | 1 a tile revealed |
| Cover of darkness | hides our land from the others | nothing |

### Travel and access (nothing: means to reach things, credited when those are taken)

Monoliths, subterranean gates, whirlpools (they cost troops: their worth), boats, keymaster's tents, border guards and
gates, quest guards, seer's huts (their reward counts when paid), sanctuaries, the market of time (it forgets skills: a
loss), obelisks and the Grail (we have no DIG action), the battlefield terrains (cursed ground, magic plains, holy
ground and the rest).

## 3. Town buildings a hero visits

| Building | Gives | Φ adds |
|---|---|---|
| Mage guild 1-5 (and the Tower's library) | spells of each level built (levels 3-5 need Wisdom) | v3: 500 x level for each spell learned |
| Hall of Valhalla (Stronghold), Cage of warlords (Fortress), Order of fire (Inferno), Wall of knowledge (Tower) | +1 attack / defense / spell power / knowledge, once per hero | one primary point |
| Academy of battle scholars (Dungeon) | 1,000 experience, once per hero | the level rule |
| Mana vortex (Dungeon) | mana to twice the maximum, once a week | the mana rule |
| Stables (Castle) | +400 land movement for the week | nothing (movement) |
| Tavern | heroes for hire | the hero's worth less 2,500 |
| Blacksmith | a war machine | half its price |
| Marketplace, artifact merchants | trades | each side at its price |

Every other building (defense bonuses, growth, incomes) counts as its cost, an income building as the income rule.

## 4. What Φ leaves out, and why

- **Movement and its bonuses.** Φ is taken at the start of each day, after every hero's movement is reset. What extra
  movement reaches counts when it is picked up.
- **Information beyond the map.** The thieves' guild, the den of thieves and signs give nothing Φ can hold.
- **Access.** Keymasters, border gates, monoliths and boats are worth what they lead to, which counts when taken.
  Quests count when they pay.
- **The puzzle map and the Grail.** We have no DIG action, so an obelisk is worth nothing to us.
- **An owned dwelling's future weeks.** Φ counts the creatures there to hire, and each week's growth when it arrives.
- **Threats and positions.** Who can reach what is for the value heads to learn.

The MOVE prior's worth of each object (`campaign_model/priors/object_values.csv`) is separate: it is the policy's
starting estimate of what visiting an object is worth, before any learning. It uses Φ's gain where Φ has one, else
Nullkiller2's score.
