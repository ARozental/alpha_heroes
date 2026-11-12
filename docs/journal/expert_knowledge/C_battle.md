# C. Battle model: game-specific expert knowledge (snapshot 01de54d)

## 0. What is live

- **Live class: `battle_model/model.py` `BattleModel` ("v3").** `train.py:251` defaults `model_architecture="v3"`. `model_kwargs` (TR:1039-1040) stores `{"architecture": "v3"}`. The inference server builds every battle model through `model.build_model` (inference.py:64-74, 208), and `build_model` dispatches on the checkpoint's `architecture` (MO:790-802). export.py:18,101 exports `BattleModel`. The journal's battle line since 10-30 (J05: the function-preserving foresight and actor-query upgrades, saved as `battle_battle3_05829.pt`) is v3; the snapshot holds no run config, so this is inferred from the default and the journal, not read from the checkpoint. `model_v2.py` and `model_v1.py` remain only so old checkpoints and league snapshots still load (rows B-M04/M03).
- **Search is not in the deployed play path.** `search_simulations=0` by default (TR:248-250). When it is 0, `ALPHA_BATTLE_SEARCH_SIMULATIONS` is never set (TR:1339), so `searchSimulations=0` (AB:121-122) and `searchAction` is never called (AB:640-642). Evals are greedy without search (TR:248-249). No deploy note in the snapshot sets the option. The journal records that search hurt eval play (+0.62 to +0.28 at 32 sims). `aux_value` (the PPG value phase) is also off (TR:202).
- **No scripted-AI teacher in battle training.** League shares are 0.85 current / 0.15 past / 0.0 fixed AI (TR:131-132), and `real_games` vs the game's AI has been unused since 10-30 (TR:103). Scripted-AI judgement enters only through priors (tagged TEACHER).
- The C++ AI (`AlphaBattleAI`) sends the contract tensors (`encodeBattle` + `encodeAttackValues`, AB:608-609). It samples with `chooseAction` (PO:40-89) and plays the result.

Path abbreviations (all relative to SNAP). SC=battle_model/schema.py, CO=constants.py, FE=features.py, MO=model.py, LA=layers.py, RU=rules.py, PR=priors.py, ATc=priors/action_types.csv, SPc=priors/spells.csv, RW=reward.py, LO=losses.py, DA=data.py, TR=train.py, AUX=aux_value.py, DI=distributions.py, SE=search.py, M1/M2=model_v1/v2.py, SAI=stupid_ai.py, GEN=generator.py, GV=game_values/__init__.py. EN=agents/AlphaBattleAI/Encoder.cpp, AV=AttackValues.cpp, HI=History.cpp, BSIM=BattleSim.cpp/.h, BSE=BattleSearch.cpp (BSEh=.h), GS=GumbelSearch.h, RH=Reward.h, SG=Schema.generated.h, PO=Policy.cpp, AB=AlphaBattleAI.cpp.
Source codes: E = engine raw value; **C = computed in C++ (bends the raw-inputs rule; marked "C!")**; G = computed in the graph; T = table/CSV/constant; L = learned embedding only. Notation: "sc(c1,c2,c3)" means x/(x+c) at each c (FE:48-51); "ssc" means x/(abs(x)+c) (FE:54-57); "frac" means part/whole clamped to 0..1; "rel" means a/(a+b).

## A. Catalogue

### A1. Contract inputs (what C++ sends; how the graph reads each)

| ID | name | where | game meaning | tags | how counted | src | enters at | init/train | notes |
|---|---|---|---|---|---|---|---|---|---|
| B-I01 | g_our_side | SC:57; EN:640 | our battle side (0 attacker, 1 defender) | OWNERSHIP POSITION | int. Sign +1 if attacker on a_/s_/c_ values (FE:444,837); flips our_frame (FE:867-874) | E | all G ops | fixed | |
| B-I02 | g_round | SC:58; EN:641; FE:665 | battle round (0 = tactics phase) | TIME | sc(2,8,32); base for ages (B-I33/I46/I47) | E | battle tok | fixed | absolute count |
| B-I03 | g_battlefield | SC:59; EN:642; MO:148,154 | terrain/special field (magic plains, cursed ground...) | RULES MAGIC | embedding, 65 rows (64 + unknown) | E→L | battle tok | N(0,0.1) learned | the field's spell effect also arrives via B-I42 |
| B-I04 | g_tactic_distance | SC:60; EN:643; FE:667 | Tactics-skill columns | HERO TIME | frac of 7 (fixed: "the most Tactics gives") | E | battle tok | fixed | |
| B-I05 | g_active_unit | SC:61; EN:644; FE:109 | acting unit slot | STRUCTURE | index: `b.acting`, the "acting" flag, the acting token for queries (MO:568,777) | E | queries, features | — | |
| B-I06 | g_hero_can_cast | SC:62; EN:639,645; FE:668 | our hero may cast now (engine `battleCanCastSpell`) | MAGIC HERO | flag | E | battle tok; gates m_type HERO_SPELL | fixed | |
| B-I07 | g_gate_state | SC:63; EN:646; MO:149 | siege gate state | TOWN RULES | embedding, 6 rows | E→L | battle tok | learned | the gate is ignored by the FE reach maps (C-10) |
| B-I08 | g_creature_spell | SC:64; EN:509-537; MO:566,625,664 | spell the acting unit may cast | MAGIC | id → creature-spell query embedding (129 rows) and a learned no-health value row (B-P10) | C! (choice made in C++, B-C14) | logit: creature_spell targets | query zero-init | -1 (Master Genie) shares the "unknown" row |
| B-I09 | g_surrender_cost | SC:65; EN:647; FE:718-719 | gold to surrender (engine) | ECONOMY | gold; only inside surrender_reward | E | logit: escapes (via B-F62) | fixed | -1 makes the feature 0 |
| B-I10 | g_start_army | SC:66; AB:237; BSE:83-89 | each side's army value at battle start | VALUE ECONOMY | gold = Σ slots creatureValue (B-C20) × count | C! | outcome start assets (FE:706); hero-share denominator (MO:209) | fixed | war machines excluded (not army slots) |
| B-I11 | g_start_mana | SC:67; AB:238; FE:706 | each hero's start mana | MAGIC VALUE | 2 × power × mana gold (B-P22) | E | escape features | fixed | |
| B-I12 | u_present, u_alive | SC:69-70; EN:73-74 | slot used; stack alive (dead kept for resurrection) | STRUCTURE | flags; living masks; attention validity | E | unit tok | fixed | |
| B-I13 | u_creature | SC:71; EN:75; MO:253,268; RU:317-328 | creature type | IDENTITY RULES | embedding, 151 rows (core ids 0..149, other ids unknown) + lookups in creature tables (B-P16) and abilities (B-P14) | E→L/T | unit tok | N(0,0.1) | HotA ids (≥150) collapse to "unknown" |
| B-I14 | u_side vs u_owner_side | SC:72-73; EN:76-77; FE:111-112,312; MO:136-140 | battle side vs current controller (hypnosis) | OWNERSHIP | ours/enemy by controller; "on_our_side" flag by battle side | E | sign of token half, relation bias, every feature | fixed | events are signed by battle side (C-23) |
| B-I15 | u_hex, u_rear_hex | SC:74-75; EN:78-79; MO:256,266,269 | front hex; second hex of a two-hex unit | POSITION DISTANCE | hex-id embedding (188 rows incl. "no 2nd hex"), summed over both hexes; our_frame of the front hex (B-F66) | E→L,G | unit tok; geometry | learned | one of 3 independent hex embeddings (B §16) |
| B-I16 | u_count | SC:76; EN:80 | creatures in the stack | STRENGTH | never raw: enters through total HP, value, kills/count, wipe n=min(10,count), lost share | E | features | — | raw counts deliberately not inputs |
| B-I17 | u_first_hp, u_max_hp | SC:77-78; EN:81-82; FE:126-129; RU:412-413 | top creature HP; max HP | STRENGTH | health = first/max; total HP = (count-1)·max + first | E | features | — | |
| B-I18 | u_attack, u_defense | SC:79-80; EN:83-84; FE:207-216 | current melee attack/defence (hero skills, spells included) | STRENGTH HERO | sc(5,20,80) each | E | unit tok | fixed | ranged-only bonuses not visible (RU:30-31) |
| B-I19 | u_dmg_min, u_dmg_max | SC:81-82; EN:85-86; FE:182-193 | damage per creature | STRENGTH | rel(avg dmg, max_hp); spread (max-min)/(max+min) | E | unit tok | fixed | |
| B-I20 | u_speed | SC:83; EN:87; FE:243-247,404,575 | movement this round | DISTANCE | frac of 15 hexes (fixed scale); reach threshold steps ≤ speed; progress denominator | E | unit tok; maps | fixed | |
| B-I21 | u_shots (+can_shoot) | SC:84; EN:88; FE:107,250-258 | ammunition | STRENGTH | sc(2,8,32); can_shoot = alive ∧ SHOOTER ∧ shots>0 | E+T | unit tok; picks shot vs melee estimate | fixed | can_shoot ignores adjacency blocking |
| B-I22 | u_retaliations | SC:85; EN:89; FE:261-263,506 | retaliations left | RULES DANGER | min(avail,3)/3; unlimited sent as 3 | C! (cap) | unit tok; gates retaliation feature | fixed | unlimited = 3 (C-21) |
| B-I23 | u_casts | SC:86; EN:90; FE:266-269 | creature spell casts left | MAGIC | sc(1,4) | E | unit tok | fixed | |
| B-I24 | u_morale | SC:87; EN:91-93; FE:272-279 | effective morale (engine-clamped ±3, 0 if immune) | RULES TIME | log(1+P_good)+log(1-P_bad), P from 24-die tables (B-P17) = log expected actions per round | E→G | unit tok | fixed | immune and neutral both read 0 |
| B-I25 | u_luck | SC:88; EN:94; FE:282-288 | effective luck | RULES STRENGTH | log(1+P_good)+log(1-0.5·P_bad) = log expected damage multiplier (double / halve) | E→G | unit tok | fixed | bad-luck table is all 0 |
| B-I26 | u_magic_resistance | SC:89; EN:95 | resist chance % | MAGIC | — | E | **none** | — | UNUSED in Python; the doc promises "unit features will read it"; it only acts inside C++ castValue (B-C01) |
| B-I27 | u_magic_mirror | SC:90; EN:96; FE:844-851 | Magic Mirror reflect % | MAGIC | factor 1 - 2·mirror/100 on our hero's cast gain at enemy units ("a reflection costs ≈ what it gains") | E→G | spell_unit features | fixed | developer rule of thumb |
| B-I28 | u_moved, u_waited, u_defending | SC:91-93; EN:97-99; FE:313-315 | turn state | TIME RULES | flags | E | unit tok | fixed | |
| B-I29 | u_is_clone, u_is_summoned | SC:94-95; EN:100-101; FE:316-317; RU:489 | vanishes after the battle | VALUE | flags; value 0 in unit_value | E | unit tok; value | fixed | their HP still counts in Lanchester and health strengths (C-20) |
| B-I30 | u_kept | SC:96; EN:653-655,866-869 | creatures kept if the battle ended now | VALUE | whole creatures × table price (outcome army) | C! (`remainingAfterBattle`) | escape features | fixed | excludes war machines, summons, clones, temporarily raised creatures |
| B-I31 | u_queue_now, u_queue_next | SC:97-98; EN:118-127; FE:291-305,903 | turn-order position this and next round | TIME | 1 - pos/len (1 = first, 0 = absent); queue_now≥0 defines the "acts_this_round" subset | E | unit tok; strength | fixed | |
| B-I32 | u_effect_id/turns/value | SC:99-101; EN:103-116; MO:257-265 | ≤8 active spell effects: id, turns left, first bonus value | MAGIC TIME | effect embedding (129 rows) + turns sc(1,3,9) + value ssc(2,10,50) → SwiGLU(dim+6→16→dim) × present, summed | E→L | unit tok | learned | "first bonus value" means different things per spell |
| B-I33 | u_start_count, u_dealt, u_taken, u_kills, u_last_round | SC:103-108; HI:40-108; EN:579-596; MO:236-249 | what the battle showed of each unit | INFORMATION STRENGTH TIME | lost share (start-count)/start; dealt and taken /(start×max_hp) at sc(.25,1,4); kills sc(2,8,32); never-acted flag; rounds since acting sc(2,8,32) | C! (event accumulation) | unit tok | fixed | kills are an absolute count |
| B-I34 | u_last_target | SC:107; EN:594 | slot it last attacked | INFORMATION | — | C | **none** | — | UNUSED by v1, v2 and v3 |
| B-I35 | h_present | SC:110; EN:137,168 | side has a hero | HERO | flag; zeroes hero stats | E | hero tok | fixed | |
| B-I36 | h_type | SC:111; EN:138,169 | hero type (enemy: InfoAboutHero portrait id) | IDENTITY HERO | — | E | **none in v3** | — | deliberate (MO:167-169: overfitting); v1/v2 embedded it |
| B-I37 | h_attack, h_defense | SC:112-113; EN:139-140,173-174; MO:202 | hero primary attack/defence (enemy visible INBATTLE) | HERO STRENGTH | damage_factor(our atk - their def), damage_factor(their atk - our def) (B-F01) | E→G | hero tok | fixed | already inside u_attack/u_defense and p_* (B §2) |
| B-I38 | h_power, h_knowledge | SC:114-115; MO:203-204; FE:703 | spell power, knowledge | HERO MAGIC | sc(2,8,32); power also enters mana value | E | hero tok; escapes | fixed | |
| B-I39 | h_mana, h_mana_max | SC:116-117; MO:206-208; FE:638 | mana left, pool | MAGIC HERO | mana/pool; mana and pool sc(5,25,125); enemy mana/(mana+20) (B-F58) | E | hero, battle tok; escapes | fixed | |
| B-I40 | h_skills | SC:118; EN:145-146; MO:210 | 28 secondary skill levels | HERO RULES | level/3 each | E | hero tok | fixed | enemy all 0 (hidden) |
| B-I41 | h_spells + h_seen_spells | SC:119,126; EN:150,600; HI:110-123; MO:212 | castable (ours); seen cast (either side) | MAGIC INFORMATION | 70 flags: in book OR seen | E/C | hero tok | fixed | |
| B-I42 | h_spell_mastery | SC:120; EN:151; MO:614-620 | engine school level per spell (skills, tomes, specialty, battlefield) | MAGIC HERO | index into the learned [70×4] bias (B-P03) | E | logit: spell | prior, learnable | our hero only |
| B-I43 | h_level, h_level_progress | SC:121-122; EN:156-158,832-842; FE:724-735 | hero level + fraction to next | HERO VALUE | L + prog/1000 → sc(2,8,32); level value (B-F64) | C! (progress from the exp table) | hero tok; escapes | fixed | enemy 0, but the reward counts the true level (C-5) |
| B-I44 | h_worn_artifacts | SC:123; EN:153-155; MO:181-191,213 | our worn artifacts | HERO RULES | bag over "rule artifacts" (B-P15), 48 slots → Linear(→4) | E→G | hero tok | zero-init | enemy hidden (-1); stat artifacts show in stats instead |
| B-I45 | h_artifacts | SC:124; EN:159,844-859 | price of what a winner takes | ECONOMY VALUE HERO | gold, part of hero_asset | C! | hero tok share; escapes | fixed | enemy 0 |
| B-I46 | h_casts, h_last_cast_round | SC:127-128; HI:110-123; MO:215-217 | hero casts this battle; recency | MAGIC TIME INFORMATION | casts sc(2,8,32); never-cast flag; rounds since sc(2,8,32) | C! | hero tok | fixed | |
| B-I47 | e_* event log (16) | SC:130-141; HI:14-133; EN:604-622; MO:303-350 | last 16 actions of both sides | INFORMATION TIME | emb: type (14), spell (71), hex (188); actor and target = projected unit tokens (hero token if actor -1); age sc(1,3,9); dealt/taken HP sc(30,300,3000); kills/lost sc(2,8,32); signed by actor side | C! | event tok | learned | absolute HP scales; spell ids ≥70 collapse to "none" |
| B-I48 | x_access | SC:143; EN:185-187; MO:284,296; FE:348 | EAccessibility | POSITION RULES | emb (8 rows); ==ACCESSIBLE gives standability | E | hex tok; reach | learned | gate treated as blocked (C-10) |
| B-I49 | x_obstacle_type/side/turns | SC:144,146-147; EN:189-213; MO:285,292,297 | obstacle kind; spell obstacle owner and turns | POSITION DANGER MAGIC | type emb (5); flags ours / theirs / any; turns sc(1,4) | E | hex tok | learned/fixed | stopping tiles are not used by FE reach |
| B-I50 | x_obstacle_id | SC:145; EN:203 | obstacle/spell id | IDENTITY | — | E | **none in v3** | — | UNUSED live (v1/v2 emb; RU:448 quicksand is StupidAI only) |
| B-I51 | x_wall_state | SC:148; EN:215-222; MO:286,298 | siege wall part state, at its hex | TOWN | emb (5 rows) | E | hex tok | learned | |
| B-I52 | m_* legality masks | SC:150-161; EN:225-268,357-450,672-747 | engine legality per choice | LEGALITY RULES | bool; illegal logits -1e9 then -inf (LA:21-27). Reused as features: m_move_hex = "reachable" (FE:576), m_melee/shoot_target = unit flags (FE:321-322), m_shoot_target in full_shot (FE:764) | E (+C rules B-C10..16) | logits; features | fixed | |

### A2. Values computed in C++ before the network (raw-input rule bent)

| ID | name | where | game meaning | tags | how counted | src | enters at | init/train | notes |
|---|---|---|---|---|---|---|---|---|---|
| B-C01 | castValue → s_unit/s_hex/c_unit/c_hex_value | SC:162-169; EN:296-344 | predicted army-value change of each legal cast | VALUE MAGIC | gold, attacker − defender: Σ over affected units of engine `getHealthChange` hpΔ × price/max_hp (new creature type: count × price) × (1 − MR%) for negative magical spells; 0 for non-kept units and one-battle heals | C! (engine rules + reward prices) | B-F46..49 | fixed | war machines priced 0 here vs half price in Python (C-4) |
| B-C02 | castHits → s_/c_*_enemies/allies | SC:170-179; EN:346-355 | units a cast affects, by controller | OPPORTUNITY DANGER | counts → B-F32 | C! | spell features | fixed | |
| B-C03 | encodeAttackValues → a_melee_value, a_shoot_value | SC:180-184; AV:28-83,121-184 | engine-played gain of each legal attack (retaliation, area, Fire Shield, drain, death stare, jousting, rebirth, after-attack spells, skills, obstacles) | VALUE STRENGTH DANGER RULES | gold, attacker − defender: Δ keptValue (HP-fractional, kept units only) on a game copy; damage rolls at the mean (B-X03); luck/morale/ability rolls branched with game chances, ≤8 branches (most likely first), mass-renormalized | C! (BattleSimulation) | B-F30, B-F43, value prior B-P09 | fixed | mean dice ≠ mean value (C-15); war machines 0 (C-4) |
| B-C04 | a_*_enemies/allies | SC:185-188; AV:86-105,141-168 | units the attack hits from that hex (getAttackedCreatures) | OPPORTUNITY DANGER | counts (target included; attacker excluded) → B-F32 | C! | attack features | fixed | |
| B-C05 | encodePairDamage → p_melee/shot_min/max | SC:189-195; EN:539-568 | engine damage estimate i→j where they stand (every bonus, no luck, no charge) | STRENGTH DANGER RULES | HP min/max; only for opposed owners, target alive and on the field; shots only if canShoot | C! (battleEstimateDamage) | B-F22 | fixed | shot estimate includes the current distance and wall penalties |
| B-C06 | pickUnits | EN:37-60; CO:20-23 | which ≤14 units per side the model sees | STRUCTURE VALUE | acting unit first, then living units by creatureValue×count, then dead; slots in unit-id order | C! | slots | fixed | ranks with C++ prices (war machines at full price) |
| B-C07 | heroLevel | EN:832-842 | level + experience fraction | HERO | L + clamp((exp−req L)/(req L+1 − req L), 0, .999) | C! | B-I43 | fixed | |
| B-C08 | heroArtifacts | EN:844-859 | artifacts a winner takes | ECONOMY HERO | Σ market price of removable worn + backpack; excludes the Grail and war machines | C! | B-I45; reward | fixed | |
| B-C09 | keptAfterBattle / remainingAfterBattle | EN:861-869 | what a side keeps | VALUE RULES | not a clone/summon; valid slot or permanent summon; baseAmount − killed | C! | u_kept, B-C01/03, reward | fixed | |
| B-C10 | stopping-hex filter | EN:242-250 | an attack that moves onto moat/quicksand aborts | RULES LEGALITY | removes m_melee_from entries (front or tail hex) | C! hand rule | mask | fixed | hand-written legality |
| B-C11 | long-weapon reach | EN:258-260 | two-hex reach in a line | RULES | m_melee_from at the unit's own hex (engine isLongWeaponAttack) | C | mask | — | |
| B-C12 | shot alias check | EN:263-266 | the shot hits this unit | RULES | m_shoot_target only if the engine's unit at that hex is this one | C | mask | — | |
| B-C13 | castAffects spell masks | EN:270-294,357-450 | legal spell target = the cast affects that unit | RULES MAGIC | canBeCastAt + affected stacks contain the unit; spell2 rows are owned by one spell per first unit | C | masks | — | |
| B-C14 | creature spell choice | EN:503-537 | which spell the unit is offered | MAGIC RULES | first fixed SPELLCASTER spell with a target; else a weighted draw seeded by round·4096+unitId·2+waited (Faerie Dragon); Master Genie: our living units some beneficial spell suits | C! hand rule (mirrors the client) | B-I08 + masks | fixed | |
| B-C15 | surrender affordability | EN:702-705 | surrender only if gold ≥ cost + 2500 (hero rehire) | ECONOMY HERO LEGALITY | mask | C! hand rule, stricter than the engine | m_type | fixed | heuristic mask (C-16) |
| B-C16 | special modes | EN:674-697,727-747 | tactics, catapult, First Aid Tent | RULES STRUCTURE | TACTICS: zone on m_move_hex, TACTICS_END always; CATAPULT: attackable wall parts on m_move_hex; HEAL: own healable units on m_shoot_target + DEFEND; WAIT illegal once waited (EN:698) | C | masks/heads | fixed | |
| B-C17 | tactics placement order | AB:432-492 | which unit is placed next | STRUCTURE VALUE | most valuable first (creatureValue×count), each once, war machines skipped | C! behaviour | decision sequence | fixed | C-16 |
| B-C18 | forced retreats | AB:50-59,537-563 | battle-ending rules | RULES TIME VALUE | 30 rounds without damage → side not ahead retreats (defender +5); round ≥50 → not-ahead side retreats, ahead side at 55; ahead = aliveArmyValue (whole creatures, army only) | C! behaviour | overrides the model | fixed | C-16 |
| B-C19 | fallback play | AB:60,504-519 | default action | RULES | defend; catapult aims at the first wall part; after 8 refused choices | C | — | fixed | |
| B-C20 | creatureValue | EN:871-877; SG:377 | creature worth | VALUE ECONOMY | recruit cost × (wood/ore 250, rare 500, gold 1) | C!/T | B-I10, B-C01/03/06/17/18, RH | fixed | no war-machine half price (≠ B-P22) |
| B-C21 | attack-value sharing key | AV:146-180 | attacks from different hexes share a value when hits, retaliation set and charge distance (jousting) match and no triggering obstacle exists | RULES | cache key | C | a_melee_value | fixed | game-knowledge shortcut |
| B-C22 | History bookkeeping | HI:14-133 | action types, actor/target, damage per side | INFORMATION | victim side by unitSide (not controller); events newest first, ≤16 | C | B-I33/41/46/47 | fixed | C-23 |

### A3. Computations in the graph (features.py / rules.py; no parameters)

| ID | name | where | game meaning | tags | how counted | src | enters at | init/train | notes |
|---|---|---|---|---|---|---|---|---|---|
| B-F01 | damage_factor | FE:60-64; MO:202 | damage formula's attack−defence factor | RULES STRENGTH HERO | log(1+min(.05·a⁺,4)) + log(1−min(.025·a⁻,.7)); range -1.2..1.6 | G/T | hero tok | fixed | cap +400%; GV says +300% (C-6) |
| B-F02 | scales, signed_scales | FE:48-57 | multi-scale bounded magnitude | STRUCTURE | sc / ssc at constants B-P19 | G | many | fixed | |
| B-F03 | fraction, relative | FE:38-45 | shares | STRUCTURE | frac clamped 0..1; rel 0.5 = even | G | many | — | |
| B-F04 | read_battle | FE:105-113 | ours/enemy by controller; values; HP; flying | OWNERSHIP | flying = FLYING ability OR effect SpellID 6 | G | all features | fixed | SpellID 6 is the adventure Fly spell → dead (C-3) |
| B-F05 | health | FE:126-129 | top creature HP left | STRENGTH | first_hp/max_hp | G | unit tok | fixed | |
| B-F06 | value_in_battle | FE:132-137 | stack value vs the average living stack | VALUE STRENGTH | rel(v, mean living v) | G | unit tok | fixed | |
| B-F07 | value_in_army | FE:140-144 | share of its own army's value | VALUE | frac(v, side total) | G | unit tok | fixed | |
| B-F08 | allies_next_to (unit) | FE:147-173 | grouped allies (exposed to area attacks) | DANGER POSITION | value of adjacent same-side units / side value | G | unit tok | fixed | |
| B-F09 | enemies_next_to (unit) | FE:176-179 | adjacent enemies | DANGER OPPORTUNITY | adjacent other-side value / that side's value | G | unit tok | fixed | |
| B-F10 | damage_per_hit_point | FE:182-186 | hits hard vs takes | STRENGTH | rel(avg dmg, max_hp) | G | unit tok | fixed | |
| B-F11 | damage_spread | FE:189-193 | damage variance | RULES | (max−min)/(max+min) | G | unit tok | fixed | |
| B-F12 | attack_c, defense_c | FE:207-216 | stats | STRENGTH | sc(5,20,80) | G | unit tok | fixed | |
| B-F13 | attack_buff, defense_buff | FE:219-228 | bonus over the creature's base (hero, spells) | STRENGTH HERO MAGIC | (current − base table) ssc(5,20,80) | G+T | unit tok | fixed | |
| B-F14 | attack_style, creature_style | FE:231-240 | offensive vs defensive (now / base) | STRENGTH | rel(atk, def) | G+T | unit tok | fixed | |
| B-F15 | unit flags | FE:308-324 | alive, acting, ours, on_our_side, moved, waited, defending, clone, summoned, two_hex, flying, war_machine, melee_target, shot_target | OWNERSHIP RULES | 0/1 | G (+T, masks) | unit tok | fixed | |
| B-F16 | standable | FE:344-350 | where a unit can stand | POSITION RULES | x_access==ACCESSIBLE or its own footprint; two-hex units need the rear (hex−1 attacker / +1 defender) free | G | reach | fixed | open gate and stoppers ignored (C-10) |
| B-F17 | attack_positions, covered | FE:353-369 | hexes to melee from (CStack::meleeAttackHexes); a two-hex unit is attackable through either hex | RULES POSITION | 6 neighbours + shifted copies for two-hex units | G | reach, maps | fixed | |
| B-F18 | distance_maps.reach | FE:372-394; RU:522-555 | steps to every position | DISTANCE | walk BFS ≤24 steps (B-P20) or fly = hex distance; INFINITE beyond | G | B-F20 | fixed | |
| B-F19 | distance_maps.to_enemy | FE:372-394 | acting unit's steps from each position to an attack position on each unit | DISTANCE OPPORTUNITY | same search on its own footprint | G | B-F39/40 | fixed | |
| B-F20 | melee_reach / can_attack | FE:397-410,544-545 | can a unit attack a unit on hex h in its coming turn | DANGER OPPORTUNITY DISTANCE | min steps to an attack position ≤ speed; a shooter with ammo reaches every hex | G | maps, summaries | fixed | ignores blocking and others' moves (C-11) |
| B-F21 | hex_distance | RU:165-172 | BattleHex::getDistance | DISTANCE | exact hex metric | G | fly distance, B-F44 | fixed | |
| B-F22 | pair_damage, best_damage | FE:418-431 | avg/low/high damage of every pair | STRENGTH | p_* × (1 + extra_attacks); best = shot if can_shoot | C→G+T | B-F23..29, 36, 51-56, 65 | fixed | ×(1+ADDITIONAL_ATTACK) applied to melee and shot alike (C-13) |
| B-F23 | damage (ATTACK/SHOT) | FE:415,474,492 | share of the target's health taken | STRENGTH OPPORTUNITY | frac(avg, target HP) | G | logit: melee_target, shoot | ActionScore 0-init | |
| B-F24 | kills | FE:475,493; RU:477-482 | share of its creatures killed | STRENGTH | getCasualties at integer avg damage / count | G | same; threat_after | 0-init | kills at mean damage |
| B-F25 | wipe_chance | FE:456-467,494 | P(the whole stack dies) | RULES OPPORTUNITY | n = min(10, attacker count); n=1: uniform tail; else normal approximation of the mean of n uniform rolls on [low,high], continuity-corrected; no spread: step | G | same | 0-init | |
| B-F26 | value_destroyed | FE:477,495 | enemy army share removed | VALUE | v × damage share / enemy value | G | same | 0-init | uses Python value (war machines half price) |
| B-F27 | threat_removed | FE:434-437,478-479,496 | BattleAI's "damage reduce" | DANGER TEACHER | threat_share × (0.5·kills + 0.5·damage); threat_share = enemy's damage on the acting unit (capped at its HP) / Σ | G | same | 0-init | |
| B-F28 | retaliation | FE:480,499-507 | retaliation after the hit, melee only | DANGER RULES | target's avg melee estimate on the acting unit × surviving fraction × [count>0, retals>0, attacker lacks BLOCKS_RETALIATION] / my HP | G+T | logit: melee_target | 0-init | |
| B-F29 | value_lost | FE:481,509 | our army share lost to retaliation | VALUE DANGER | my value × retaliation / our value | G | same | 0-init | |
| B-F30 | net_value (attack_values) | FE:440-447,482-483,512-517 | engine-played gain (B-C03) | VALUE | sign·a_value / (both armies' value), clamp ±1; melee: best legal hex | C→G | logits; value prior | 0-init | mixes C++ and Python valuations (C-4) |
| B-F31 | versus_best | FE:450-453 | relative to the best legal option | VALUE | value / max abs(value) over legal options | G | logits | 0-init | |
| B-F32 | hits | FE:70-77 | units an attack or cast hits | OPPORTUNITY DANGER | enemies and allies at sc(1,4) | C→G | attack/from/shot/spell logits | 0-init | |
| B-F33 | reachable | FE:531,576 | acting unit can move there | DISTANCE | = m_move_hex | E | hex tok | fixed | in CATAPULT/TACTICS modes it means wall parts / zone |
| B-F34 | threat_value | FE:532,552 | enemy value able to attack the acting unit there | DANGER | Σ value of enemies whose can_attack covers its footprint / enemy value | G | hex tok; move/from Δ | fixed | |
| B-F35 | melee_threat_value | FE:533,553 | the same, melee only | DANGER | as B-F34 with melee_reach | G | hex tok; Δ | fixed | |
| B-F36 | danger | FE:534,555-556 | damage those enemies could deal there | DANGER STRENGTH | Σ avg dmg (shot if possible) / my HP, clamp 1 | G | hex tok; Δ; battle | fixed | not capped per enemy |
| B-F37 | blocked_shooters | FE:535,558-563 | enemy shooters it would block | OPPORTUNITY RULES | adjacent enemy shooters' value / enemy value | G | hex tok; Δ | fixed | |
| B-F38 | next_to_enemy | FE:536,564 | adjacent enemy (a shooter there cannot shoot) | RULES DANGER | flag | G | hex tok; shooter move feature | fixed | |
| B-F39 | new_targets | FE:537,569-572 | enemy value attackable next turn from there but not now | OPPORTUNITY DISTANCE | value with to_enemy ≤ speed, not attackable now / enemy value; 0 for shooters | G | hex tok; move/from | fixed | |
| B-F40 | progress | FE:538,573-575 | steps gained toward the nearest enemy | DISTANCE | (nearest_here − nearest_there)/speed, ±1 | G | hex tok; move | fixed | |
| B-F41 | allies/enemies_next_to (hex) | FE:539-541,565-567 | grouping if it stands there | DANGER POSITION | adjacent value shares | G | hex tok; Δ | fixed | |
| B-F42 | MOVE_FEATURES | FE:745-748,784-796 | a move vs standing still | DANGER DISTANCE OPPORTUNITY | 6 Δ (hex minus own hex: threat, melee threat, danger, blocked, allies, enemies) + new_targets + progress split melee/shooter + shooter_next_to_enemy | G | logit: move (HEX_TYPES) | 0-init | used for CATAPULT/TACTICS too (C-18) |
| B-F43 | melee_from features | FE:749,773-777,786-803 | position after the attack | DANGER VALUE | threat_after = threat − (target value × kills, if it reaches me)/enemy value, minus threat_here; other Δs; hex_value = attack value from this hex − best (≤0); hits | G | logit: melee_from | 0-init | danger Δ not reduced by the kills (C-25) |
| B-F44 | shot_types: full_shot, half_shot_only | FE:751-766 | can shoot at full damage (target ≤10 hexes or NO_DISTANCE_PENALTY) | RULES TEACHER | flags | G+T | logit: type (type_rules) | prior B-P08 | same rule as TR:745-758 |
| B-F45 | BATTLE_TYPE_FEATURES | FE:754,809 | acting unit's danger and threat where it stands; value share | DANGER STRENGTH | from B-F36/34/59 | G | logit: type | 0-init | |
| B-F46 | spell net_value | FE:815,823,847-853 | our gain of both armies' value | VALUE MAGIC | sign·value (×B-I27 factor for hero casts at units) / both, ±1 | C→G | logit: spell_unit/hex, creature targets (linear only); value prior | 0-init | one weight per feature shared by all 70 spells |
| B-F47 | spell versus_best | FE:824,852,857 | gain vs this spell's best target | VALUE | ours / max abs(ours) | G | same | 0-init | replaced by B-P10 when the spell changes no HP |
| B-F48 | changes_health | FE:825-826,854 | spell changes HP at any target | MAGIC | flag; switches versus_best ↔ learned value | G | same | 0-init | |
| B-F49 | target_value | FE:827-828,839-842,855 | value of the targeted unit (or hex) | VALUE | unit value / both; hex: Σ on hex, clamp 1 | G | same | 0-init | summons count 0 |
| B-F50 | hero_lands | FE:829-831,844-845 | Magic Mirror reflection | MAGIC | 1 − 2·mirror on enemy targets | G | B-F46 | fixed | see B-I27 |
| B-F51 | best_hit | FE:582-600 | most of an enemy's health it can take in reach | OPPORTUNITY | max over reachable opposed targets of avg/HP | G | unit tok | fixed | |
| B-F52 | worst_hit | FE:589,600 | most of its health one enemy in reach takes | DANGER | max over attackers | G | unit tok | fixed | |
| B-F53 | hit_by_me | FE:590,600 | the acting unit's best attack on it | OPPORTUNITY | avg/HP | G | unit tok | fixed | |
| B-F54 | hits_me | FE:591,600 | its attack on the acting unit | DANGER | avg / acting HP | G | unit tok | fixed | |
| B-F55 | loss_next_turn | FE:611,618-621,628-640 | our value lost before we act again | DANGER VALUE | Σ over enemies of max over our units of min(high est, HP)×value/HP, / our value, clamp 1 | G | battle tok; escapes | fixed | no reach, no spells: upper bound (C-12) |
| B-F56 | kill_next_turn | FE:622,641 | the mirror for our units | OPPORTUNITY VALUE | same | G | same | fixed | same |
| B-F57 | best_stack_share | FE:623,642 | our top stack's share ("one Blind + shooting") | DANGER VALUE | max v / our value | G | same | fixed | |
| B-F58 | enemy_spell_threat | FE:612,624-625,643 | enemy hero can hurt us with spells | MAGIC DANGER INFORMATION | enemy mana/(mana+20) | G/T | same | fixed | its book is hidden |
| B-F59 | battle strength | FE:649,664 | our value share | STRENGTH VALUE | rel(our value, enemy value) | G | battle tok; type rule | fixed | not Lanchester |
| B-F60 | danger_here, threat_here, my_share | FE:669-671 | acting unit's danger and threat at its own hex; its share of our value | DANGER VALUE | gathered from maps | G | battle tok | fixed | |
| B-F61 | retreat_reward | FE:679-720 | exact zero-sum reward if we retreat now | VALUE HERO ECONOMY | (Δenemy − Δours)/(starts), clamp ±1; ours = max(0, hero+mana−2500); enemy keeps army+hero+mana | G+T | logit: RETREAT (escape_rules) | 0-init | enemy hero counted 0 (C-5) |
| B-F62 | surrender_reward | FE:686-687,719 | the same for surrender | VALUE ECONOMY | ours = army(u_kept)+hero+mana−cost−2500; 0 if cost<0 | G+T | logit: SURRENDER | 0-init | |
| B-F63 | exposure → escapes | FE:679,721; MO:556 | B-F55..58 for both escapes | DANGER | one weight each, shared by RETREAT and SURRENDER | G | logit: escapes | 0-init | |
| B-F64 | hero_levels, level_value, hero_assets | FE:724-740 | hero worth | HERO VALUE | 2000·(Σ_{l<L} min(l,3)) + fraction·min(L,3)·2000 + artifacts | G+T | hero tok share (MO:209); B-F61/62 | fixed | enemy 0 |
| B-F65 | side_strengths (Lanchester) | FE:877-909; MO:18-23 | square law: winner = larger health × damage per round | STRENGTH | per variant (all, acts_this_round, melee, shooting): [log(H_us+1) − log(H_them+1), log(D_us+f) − log(D_them+f)], D = Σ capped avg dmg on each opponent × that opponent's HP share; f = 1% of the sum; clamp ±10 | G | Strength (B-P11) → context, type, V; escapes (fixed exponents) | fixed | health term identical across variants (C-20) |
| B-F66 | our_frame | FE:867-874 | position seen from our side | POSITION | (column/16, flipped for the defender; row/10) | G | unit, hex tok | fixed | |
| B-F67 | Context strengths | MO:473-503 | each side's strength by value and by HP, reweighted per unit | STRENGTH VALUE | 8 value + 8 HP sums × exp(clamp(w·token, ±5)); our share rel(a, b) | G+L | context | w zero-init = plain value and HP shares | unbounded factor up to ×148 |

### A4. Priors, tables, constants

| ID | name | where | game meaning | tags | how counted | src | enters at | init/train | notes |
|---|---|---|---|---|---|---|---|---|---|
| B-P01 | action_types.csv | ATc; PR:36-37; LA:13; MO:584 | starting odds of action types | HEURISTIC STRUCTURE | logits: MOVE .5, MELEE 1, SHOOT 1, HERO_SPELL .5, RETREAT −6, SURRENDER −6, all others 0 | T | logit: type bias | prior, learnable, unbounded | |
| B-P02 | spells.csv | SPc; PR:86-87; MO:591 | some spells are usually better | MAGIC HEURISTIC | per-spell bias −1 (Earthquake, Armageddon, Remove Obstacle) to +.75 (Implosion, Prayer, Slow); −.5 Quicksand/Land Mine/Force Field/Destroy Undead; adventure spells 0 | T | logit: spell bias | prior, learnable | |
| B-P03 | MASTERY_BIAS | PR:21,90-92; MO:573,614-620 | better mastery, better cast | MAGIC HERO | (0, .25, .5, .75) for none..expert, for all 70 spells | T+L | logit: spell | prior, learnable [70×4], unbounded | |
| B-P04 | escape_relief + escape_bias | PR:40-52; MO:558,682-684 | flee only when far weaker | STRENGTH HEURISTIC | relief = σ((−3.2 − z)/0.5), z = log(H·D ours / H·D theirs) (B-F65 "all", exponents fixed at 1); adds relief × escape_bias | T | logit: RETREAT, SURRENDER | fixed | **escape_bias = +30, not +6 (C-1)** |
| B-P05 | ESCAPE_AHEAD | PR:55-63,80-83; MO:560,686-687 | don't flee while V says we're ahead | VALUE HEURISTIC | w1·tanh(max(0,V)/0.1) + w2·max(0,V); V = final head, detached | T+L | logit: both escapes | init (−3,−3), unbounded | |
| B-P06 | ESCAPE_STRONG | PR:65-77; MO:562,685 | don't flee with stronger creatures | STRENGTH HEURISTIC | w·min(exp(3z), 8) | T+L | logit: both escapes | init −1, unbounded | cap to avoid a de-facto mask |
| B-P07 | escape_rules | MO:551-557,684 | weight of each outcome feature | VALUE | 6 weights × B-F61..63 via escape_types | L | logit: escapes | zero-init | |
| B-P08 | IDLE_SHOT_BIAS | PR:95-106; MO:585,604 | "a unit that can shoot at full damage shoots" (BattleAI) | TEACHER RULES | type_rules[DEFEND, full_shot] = −5; every other type-rule weight 0 | T | logit: type | prior, learnable | only DEFEND (C-14) |
| B-P09 | value prior | MO:36-41,518-526,581,607-612,646-666 | "take what gains" (BattleAI) | VALUE TEACHER | 2·g·tanh(v/0.02), v = engine value share; g = 5^tanh(10·r) per kind (shot, melee, spell, creature_spell) | T+L | logits: melee_from, shoot_target, spell_unit/hex, creature targets; reaches types via B-S07 | r = 0 (g = 1); bounded g ∈ [1/5, 5]; step ×10 | WAIT/DEFEND/MOVE get none |
| B-P10 | NO_HEALTH_VALUE | MO:544,574-575,622-632 | worth of spells that change no HP (Slow, Bless) | MAGIC VALUE | learned scalar per spell (70) and per creature spell (129) in place of versus_best | L | spell/creature target logits | init .5, unbounded | |
| B-P11 | Strength exponents | MO:506-515 | Lanchester exponents | STRENGTH | z_k = a_k·logH + b_k·logD_k (8 params) | L | context σ(z); B-P12/13 | init 1 (the law), unbounded | |
| B-P12 | type_strength | MO:550,586,688 | strength → type odds | STRENGTH | Linear(4 z → 13) | L | logit: type | zero-init | z unbounded (C-7) |
| B-P13 | value_strength | MO:721-726,778; LA:141-145 | strength → value | STRENGTH VALUE | final head + 0.5·z_all, so V = 2σ(z)−1; round heads 0 | T+L | value logits | prior .5, learnable | |
| B-P14 | rules.ABILITIES (68) | RU:95-113,159-161; MO:254-255,267 | creature abilities shared across creatures | RULES IDENTITY | one learned vector per ability (Linear 68→dim, no bias), summed per creature | T (engine vocab) + L | unit tok | N(0,0.1) | |
| B-P15 | RULE_ARTIFACT_BONUSES (19) | RU:116-137; MO:171-191 | artifacts that change rules (no fleeing, magic blocks, free shooting, necromancy...) | HERO RULES | bag of ≤48 → Linear(→4) | T + L | hero tok | zero-init | our hero only |
| B-P16 | CREATURE_TABLES (+hate) | RU:73-92,140-162 | per-creature facts | RULES VALUE | live: value, shooter, flying, no_distance_penalty, blocks_retaliation, base atk/def, extra_attacks, war_machine; StupidAI/v1 only: reductions, caps, immunities, hate, no_melee_penalty | T (engine vocab) | features | fixed | |
| B-P17 | morale/luck chances | RU:61-64 | per value on a 24-sided die | RULES | good morale 1,2,3/24; bad 2,4,6/24; good luck 1,2,3/24; bad luck 0 | T | B-I24/25 | fixed | |
| B-P18 | damage constants | RU:44-48 | attack +5%/pt (cap 80 pts), defence −2.5%/pt (cap 28), shooting penalty beyond 10 hexes | RULES | — | T | B-F01, B-F44 | fixed | ≠ GV:41 (C-6) |
| B-P19 | scale constants | FE:67-70,247,612,667; MO:230-232,279,311-312 | typical magnitudes | STRUCTURE | STAT (5,20,80); COUNT (2,8,32); MANA (5,25,125); HIT (1,4); casts (1,4); EFFECT_TURNS (1,3,9); EFFECT_STRENGTH (2,10,50); HISTORY (.25,1,4); OBSTACLE (1,4); AGE (1,3,9); HP (30,300,3000); speed /15; tactics /7; spell-threat mana 20; retaliations /3 | T | many | fixed | |
| B-P20 | hex geometry | RU:259-302; FE:35 | BattleHex grid | POSITION RULES | neighbour table; two-hex rear = hex∓1 by side; REACH_STEPS 24 | T | reach, adjacency | fixed | |
| B-P21 | vocab sizes | CO:46-58; LA:16-18 | id vocabularies | IDENTITY | creature 150, effect 128, battlefield 64, access 7, obstacle type 4, wall 4, gate 5 (+1 "unknown" each) | T | embeddings | — | |
| B-P22 | game_values prices | GV:21-34,163-170; RW:44-47,58-66 | what things are worth | ECONOMY VALUE | resources 250/500/1; hero hire 2500; level-up from L worth 2000·min(L,3); mana 2 gold per mana per power; war machines half price (ballista 1250, ammo cart 500, tent 375), catapult and towers 0 | T | reward; B-F26/30/61; unit values | fixed | |

### A5. Structure: action space, attachments, tokens

| ID | name | where | game meaning | tags | how counted | src | enters at | init/train | notes |
|---|---|---|---|---|---|---|---|---|---|
| B-S01 | 13 action types | CO:61-77 | WAIT DEFEND MOVE MELEE SHOOT HERO_SPELL RETREAT SURRENDER CREATURE_SPELL CATAPULT HEAL TACTICS_MOVE TACTICS_END | STRUCTURE RULES | type softmax | T | logp_type | — | |
| B-S02 | conditional decomposition | SC:198-216; DI:1-12; PO:40-89 | type → hex / target (→ melee from-hex) / spell → first target (unit, hex, none) → second target | STRUCTURE | chained masked log-softmaxes | T | outputs | — | |
| B-S03 | shared argument heads | CO:79-82; MO:670-671 | MOVE, TACTICS_MOVE and CATAPULT pick in the move head; SHOOT and HEAL in the shoot head | STRUCTURE | reuse | T | — | — | C-18 |
| B-S04 | SECOND_TARGET_KIND | CO:91-99; MO:576-579,653-659 | Teleport (63) → hex; Sacrifice (40) → unit | MAGIC STRUCTURE | static table; best second target added to the first unit's logit | T | spell2 heads | — | |
| B-S05 | pointer scoring | MO:567-569,587-589,637-666 | choice = query(context + acting unit) · key(unit/hex token) | STRUCTURE | 7 queries, 12 keys; per-spell query emb; creature-spell query emb | L | all argument logits | queries and actor queries zero (start = priors only) | |
| B-S06 | feature→logit attachment (ActionScore) | MO:534-543,572; LA:95-125 | each choice's own facts | STRUCTURE | ATTACK→melee_target, FROM→melee_from, SHOT→shoot_target, MOVE→move, SPELL→spell/creature targets; linear + tiny MLP + wide MLP over context and mean/max of legal choices (spell heads linear only) | L | logits | all zero-init | |
| B-S07 | best follow-up | MO:644-673; LA:84-86 | a choice is worth its best continuation | STRUCTURE VALUE | parent += max over legal children (0 if none) | G | type, target, spell logits | fixed op | WAIT, DEFEND, escapes, TACTICS_END get 0 |
| B-S08 | type logit sum | MO:688 | — | STRUCTURE | head(context) + type_rules(TYPE_FEATURES) + type_strength(z) + follow + escape terms | — | logp_type | — | |
| B-S09 | signed token halves | MO:14-17,129-133 | "ours is good, theirs bad" | OWNERSHIP | first half × (+1 ours / −1 enemy) for hero, unit, event tokens; RMSNorm (no centring) | G | tokens | fixed op | |
| B-S10 | relation bias | MO:353-390 | attention by same/opposite side | OWNERSHIP | 3 learned values per head (opposite, none, same) | L | trunk | zero-init | |
| B-S11 | token kinds | MO:393-413 | battle, hero, unit, hex, event | STRUCTURE | kind embedding (5) | L | trunk | learned | |
| B-S12 | Context | MO:473-503 | decision summary | STRUCTURE STRENGTH | SwiGLU([battle, acting unit, our hero, mean living units, 16 strength shares, 4 σ(z)] → 2·dim → dim) + RMSNorm | G+L | queries, value heads | learned | |
| B-S13 | masked softmax | LA:21-27 | legality only | LEGALITY | −1e9 then −inf | — | outputs | — | |
| B-S14 | Foresight heads | MO:442-470; DA:381-411; LO:146-150,318-335 | the enemy's reply before our next decision | DANGER TIME | per unit: dies, share of HP lost, HP dealt / other side's HP; per hex: an enemy stands there; BCE, weight 0.4 | label (data) + L | aux loss; hidden states joined into tokens | joins zero-init | |
| B-S15 | 14 unit slots per side | CO:20-24 | 7 stacks + 4 war machines + 3 towers fit | STRUCTURE | — | T | — | — | with more units, some are unseen (B-C06) |
| B-S16 | no hero identity | MO:164-169 | deliberate | IDENTITY | — | — | — | — | |

### A6. Reward, value, credit

| ID | name | where | game meaning | tags | how counted | src | enters at | init/train | notes |
|---|---|---|---|---|---|---|---|---|---|
| B-R01 | terminal reward | RW:1-24,99-109; DA:110-159 | zero-sum change in assets | VALUE ECONOMY HERO | ((enemy assets lost) − (ours lost)) / (both starts), clamp ±1; assets = army kept (start − casualties + permanent summons, whole creatures × price) + hero (level value + artifacts + surviving war machines at half price) + mana (2·power·mana) | T | V label, advantages | fixed | uses true enemy hero level/artifacts (C-5) |
| B-R02 | outcome rules | RW:84-96; RH:61-70 | what each ending leaves | RULES ECONOMY HERO | WIN: all + captured artifacts; LOSS/DRAW: 0; RETREAT: max(0, hero+mana−2500); SURRENDER: army+hero+mana−cost−2500 | T | reward | fixed | enemy surrender cost unknown = 0 (DA:117,156) |
| B-R03 | round_score labels | RW:112-115; DA:161-165; LO:153-168 | score if the battle ended at round r, r+1, r+2 | VALUE TIME | same formula on armies alive (whole creatures), heroes present, war machines at start value | T | V0..V2 labels | fixed | C-22 |
| B-R04 | value heads | LA:128-151; SC:215; CO:39-41; LO:211-221 | horizons: this round, +1, +2, battle end | VALUE TIME | tanh; V1, V2 add the previous logit (detached); V final stands alone; BCE on (y+1)/2 with logit 2z | L | value | zero + B-P13 | |
| B-R05 | window credit | LO:48-57,89-110,171-192 | judge a decision by V two rounds later | VALUE TIME CREDIT | A = 0.5·(V_later − V) + 0.5·(R − V); V_later = V at our first decision ≥2 rounds later, else R; the window is in rounds (game time) | — | advantages | fixed | outcome share fixed at .5 since 11-05 |
| B-R06 | policy_heads | LO:122-131,235-250 | the policy maximizes the final outcome only | VALUE CREDIT | weights (0,0,0,1) | — | — | fixed | |
| B-R07 | per-side normalization | LO:253-281 | attacking and defending are different problems | CREDIT OWNERSHIP | mean/std per side group, ≥2 battles, ≥8 decisions | — | — | fixed | |
| B-R08 | lopsided_scale | LO:117-121; DA:435-445 | lopsided battles move the share little | CREDIT STRENGTH | A × 1/(4p(1−p)), p = weaker side's asset share ≥ .02 (≤ ×12.8) | — | advantages | fixed | |
| B-R09 | example_weight | DA:370,414-423 | foregone battles teach little | CREDIT STRENGTH | weight = clip(p/0.1, .05, 1) | — | loss weights | fixed | |
| B-R10 | gamma 1, no per-step reward | LO:21-27,91 | battles are short | TIME | — | — | — | fixed | |
| B-R11 | known non-zero-sum gaps | MO:674-681; RW:25-39 | surrender payment destroyed, army lost on retreat, catapult damage, mana spent, experience, necromancy, Eagle Eye, >2 players | VALUE | not modelled (TODO) | — | — | — | |
| B-R12 | aux value phase | AUX:1-34,100-120; TR:196-211 | PPG value phase, 5 decisions per battle spread over its rounds | VALUE | — | — | — | off | |
| B-R13 | length_weight | DA:426-432; TR:122 | long battles are one outcome | CREDIT TIME | 1/(1+(n−1)/scale) | — | — | off (scale 0) | |
| B-R14 | training opponents | TR:103-104,131-132 | who the model learns against | TEACHER DATA | self .85 / past .15 / BattleAI 0; campaign battles real_share .5 | — | — | — | no scripted teacher |
| B-R15 | battle generator | GEN:1-60 (scenarios/generator.toml) | factions, levels, war machines, special fields, guild spells by level, Wisdom-limited spell levels | DATA | — | T | training data | — | outside the model |

### A7. Search (Gumbel AlphaZero; off in deployed play)

| ID | name | where | game meaning | tags | how counted | src | enters at | init/train | notes |
|---|---|---|---|---|---|---|---|---|---|
| B-X01 | deployment switch | TR:248-250,1339; AB:121-124,640-642 | — | — | 0 simulations by default | — | — | off | |
| B-X02 | BattleSimulation | BSIM.h:21-33; BSIM:89-257 | VCMI server battle rules on a copy of the game state | RULES | exact rules; new random dice per simulation | C | search; B-C03 | fixed | |
| B-X03 | MeanRandom | BSIM:20-45 | expected dice | RULES | uniform at the middle, rounding alternated; binomial at its mean | C | B-C03 | fixed | |
| B-X04 | ScriptedRolls | BSIM:47-86 | luck/morale/ability rolls | RULES | chance = settings vector/die size; ability % / 100; scripted outcome | C | B-C03 | fixed | |
| B-X05 | terminal reward (Reward.h) | RH:1-111; BSE:145-160 | C++ copy of B-R01/02 | VALUE | same formula, war machines left out | C/T | leaves | fixed | C-4 |
| B-X06 | leaf value | BSE:198-217 | network V final; the other side gets −V (zero-sum) | VALUE | — | — | — | — | both sides played by our network |
| B-X07 | round cap | AB:49; BSE:209 | round ≥30 is a leaf | TIME | — | T | — | — | |
| B-X08 | non-decisions | BSE:245-256; BSEh:31-32 | war machines defend in the tree | RULES | — | C | — | — | real play lets the model aim the catapult/tent (C-17) |
| B-X09 | encoding without history | BSE:183-196; EN:658-663 | simulated states | INFORMATION | u_start_count = count; no events, no seen spells | C | — | — | C-17 |
| B-X10 | open loop, node key | GS:22-27; BSE:178-181 | dice and morale as chance | STRUCTURE | split by acting unit id·2 + mover | — | — | — | |
| B-X11 | Gumbel config | GS:53-61; AB:123-124,726-729 | — | — | maxConsidered 16; valueScale .1; maxvisitInit 50; gumbel 1 (0 when greedy) | T | — | — | mctx defaults; generic |
| B-X12 | search_loss | LO:350-383 | AlphaZero target | CREDIT | cross-entropy to the improved policy | — | — | — | only when search is on |

### A8. Not in the live path (kept for old checkpoints, checks, diagnostics)

| ID | name | where | game meaning | tags | how counted | src | enters at | init/train | notes |
|---|---|---|---|---|---|---|---|---|---|
| B-M01 | rules.damage_range | RU:343-409 | Python DamageCalculator: hate, reductions, Shield/Air Shield, range, Forgetfulness, petrify, elementals, damage cap | RULES | float64, matches the engine | G | SAI:100-109 only | — | not used by v3 |
| B-M02 | rules.reachability | RU:424-474 | engine BFS incl. the gate for defenders, moat/quicksand stops | DISTANCE | — | G | StupidAI only | — | v3 uses the simpler B-F18 |
| B-M03 | least_damage, cross_stack_features, kill rule | RU:185-256; M1:70-71,111-113,308-316 | sure-kill rule, weight 1000 | RULES | — | G | v1 only | — | |
| B-M04 | v2 differences | M2:80-112 | hero-type embedding; skills x/(x+10); mastery from school skills (RU:493-509) | — | — | — | v2 only | — | |
| B-M05 | failure_counts | TR:745-800 | idle shooter, escape while ahead, kiting | TEACHER | — | — | eval diagnostics | — | not in the model |

## B. One game quantity, several representations

1. **Army/unit value (gold): at least 8 versions.** Python `unit_value` is HP-fractional, counts war machines at half price and summons at 0 (B-F04, used by F06/F07/F26/F49/F57/F59/F67). C++ `creatureValue` counts war machines at full price (B-C20 → B-I10, C06, C17). C++ `keptValue` is HP-fractional over kept units only, so war machines are 0 (B-C03). `castValue` prices value per HP × (1−MR) (B-C01). `aliveArmyValue` counts whole creatures (B-C18 "ahead", RH). `u_kept`×price plus war machines separately (B-I30 → B-F61/62). The reward uses start − casualties + summons, with war machines inside the hero (B-R01/R03). `data.army_values` uses u_count×price (credit check, DA:80-84). Each is then normalized differently: share of own side (F07), of both (F30/46/49), vs the mean stack (F06), as rel() (F59), or reweighted by learned exp factors (F67).
2. **Attack/damage power.** Raw stats at scales (B-I18/F12), buff over base (F13), style ratios (F14), avg dmg/HP (F10/F11), hero atk/def as a log formula factor (B-I37/F01), engine estimates ×(1+extra) (B-C05/F22), engine-played gold value (B-C03/F30), fractions killed/damaged/wiped (F23-25), Lanchester D as a log ratio (F65), danger share (F36), per-unit summaries (F51-54), the high estimate in exposure (F55/56). A hero's attack is counted 4 times: inside u_attack, attack_buff, the hero damage_factor, and p_*.
3. **Health.** Top-creature fraction (F05), total HP as a Lanchester log ratio (F65), HP-weighted context strengths (F67), share lost since start and dealt/taken per start HP (B-I33), foresight "lost" share (B-S14), share of the target taken (F23), retaliation and danger as shares of my HP (F28/F36).
4. **Balance of power: about 9 mechanisms.** Value share rel() (F59, battle tok + type rule). Lanchester log ratios × learned exponents (F65+P11) → σ(z) into context, z into type logits (P12) and into V (P13). 16 learned-reweighted value/HP shares (F67). Fixed-exponent Lanchester z for escape relief and escape_strong (P04/P06). The value head V for escape_ahead (P05). Exposure upper bounds (F55/56). The C++ "ahead" army-value test that forces retreats (C18). Start-asset share for credit weights (R08/R09). Transforms differ: share, log ratio clamped ±10, sigmoid, exp(3z) capped at 8, tanh(V/0.1).
5. **Distance/reach.** Hex-id embeddings (B-I15) and our_frame coordinates (F66), speed/15 (B-I20), BFS steps (F18/F19), the reach test (F20), hex distance ≤10 for full shot (F21/F44), progress in steps/speed (F40), new_targets (F39), the move mask as "reachable" (F33), the distance penalty inside p_shot (B-C05), path/jousting inside a_* (B-C03, C21).
6. **Time/turn order.** Queue positions (B-I31), moved/waited flags (B-I28), round sc(2,8,32) (B-I02), event age sc(1,3,9) (B-I47), rounds since acting/casting sc(2,8,32) (B-I33/I46), effect turns sc(1,3,9) (B-I32), obstacle turns sc(1,4) (B-I49), tactics/7 (B-I04), morale as expected actions (B-I24), the "acts_this_round" Lanchester subset (F65), label horizons r..r+2 and window 2 rounds (R03-R05), round caps 30/50/55 (C18, X07).
7. **Spell worth.** C++ gold prediction → net share (B-C01/F46), versus_best (F47) or a learned per-spell value when no HP changes (P10), the CSV bias (P02), the mastery bias (P03/B-I42), target value (F49), the tanh value prior (P09), castable flags in the hero token (B-I41).
8. **Hero worth.** Level+progress and artifact price (B-I43/I45 → F64) appear as a share in the hero token (MO:209), inside the outcome features (F61/62) and in the reward (R01). The enemy's is 0 in the first two but true in the third.
9. **Mana.** mana/pool and sc(5,25,125) (B-I39), enemy mana/(mana+20) (F58), 2·power·mana gold (F61/62, R01), start mana (B-I11).
10. **Units an attack hits.** C++ counts (B-C02/C04 → F32 at sc(1,4)) vs adjacency value shares (F08/F09/F41, "area attacks hit them together").
11. **Retaliation.** Count/3 (B-I22), estimate-based share and value lost (F28/F29), engine-played inside a_* (B-C03), the BLOCKS_RETALIATION table (P16).
12. **Shooting.** can_shoot and ammo (B-I21), full/half shot flags (F44) with the −5 DEFEND prior (P08), enemy shooters blocked (F37), shooter_next_to_enemy and shooter_progress (F42), melee_threat (F35), the shooting Lanchester variant (F65), p_shot (B-C05), shot value prior (P09).
13. **"Should we flee": 8 mechanisms.** CSV −6 (P01), relief (P04), ahead (P05), strong (P06), outcome-reward and exposure weights (P07×F61-63), the surrender affordability mask (C15), C++ forced retreats (C18), outcome rules in the reward (R02).
14. **Ownership.** Battle side vs controller (B-I14); signed halves (S09) and relation bias (S10) by controller for units, by battle side for events (B-I47/C22); flags ours vs on_our_side (F15).
15. **Kills/casualties.** Kill share at mean damage (F24), wipe probability (F25), u_kills sc(2,8,32) (B-I33), e_kills/e_lost sc(2,8,32) (B-I47), foresight "dies" (S14).
16. **Board identity: 3 independent hex-id embeddings** (unit position MO:256, hex token MO:283, event hex MO:319) plus our_frame (F66). **4 independent spell-id tables**: effect emb (MO:257, 129 rows), event spell emb (MO:318, 71), spell query emb (MO:565, 70), creature-spell query (MO:566, 129); plus 3 bias tables (P02, P03, P10). Creature identity = embedding (B-I13) + ability embeddings (P14) + fixed tables (P16).

## C. Surprises, inconsistencies, rule contradictions

1. **Suspected bug: escape_bias is 5× too large.** MO:554-558 builds `escape_bias = −prior × escapes.sum(dim=0)`. Since the 4 exposure rows were added (MO:556), each escape column sums to 5, so escape_bias = +30 instead of +6. The relief share is applied to 30. Intent per PR:40-45: 3× army −5.3, 5× army −3, 10× army −0.3. Actual: −2.4, **+9**, **+22**. `escape_any` (MO:563) correctly uses `>0`. The only test (tests/test_design.py:201-207) checks the relief share, not the bias.
2. **Unused contract inputs.** u_magic_resistance (B-I26; the schema promises a reader) and u_last_target (B-I34; read by no model version) are sent on every decision for nothing. h_type (B-I36) is unused by design. x_obstacle_id (B-I50) is unused in v3.
3. **Dead code.** `flying |= effect(FLY)` uses SpellID 6, the adventure-map Fly, which never appears on a battle unit (FE:108, RU:52).
4. **War machines are valued 4 ways.** Python half price (B-P22: F26/F49/F59/F65/F67 and the `both` denominator of F30/F46). C++ attack/cast values give 0 (not "kept"; B-C01/C03). C++ creatureValue uses full price (B-C06/C17). The reward uses half price as part of the hero (R01), and the search reward leaves them out (RH:12, X05). Shooting a ballista therefore gets value_destroyed > 0 but net_value = 0 and no value prior, while the reward pays +1250.
5. **Hidden enemy hero vs true reward.** The enemy's level, artifacts, skills, book and worn items are 0 or −1 in the inputs, but R01 prices its true level and artifacts. So B-F61/62/64 misstate escape rewards against heroes (acknowledged in FE:689-691).
6. **The same engine constant disagrees across models.** The battle uses an attack bonus cap of 80 points = +400% (RU:44-46, FE:61-64). game_values, used by the campaign's army_multiplier, says 60 points = +300% (GV:39-42).
7. **Unbounded inputs and weights despite "every feature bounded" (FE:5-8).** Lanchester log ratios run ±10 × unbounded exponents (P11) and enter type logits (P12) and V (P13) linearly. Context factors exp(±5) (F67). The escape weights P05/P06 are unbounded and can flip sign. P01-P03 and P10 are unbounded too. Only the value-prior gains (P09) are bounded.
8. **De-facto mask on escapes.** At init, when ahead and ≥2× stronger: −6 (P01) −8 (P06 cap) −6 (P05 at V≈1) ≈ −20 logits, which defeats the "never mask" intent behind the P06 cap.
9. **Not scale-free despite FE:5-8.** Event HP sc(30,300,3000) and kill/loss counts (B-I47), u_kills (B-I33), rounds, spell power.
10. **The geometry in features ignores stoppers and the gate.** FE reach treats the open gate as blocked for defenders and ignores moat/quicksand stops (F16/F18). rules.reachability models both but only StupidAI uses it (B-M02).
11. **Reach/threat simplifications.** Every shooter with ammo threatens every hex even when blocked (F20). Reach uses current positions and speeds. Danger is not capped per enemy (F36).
12. **Exposure ignores reach entirely.** Any enemy on any of our units, at the high estimate (F55/56): an upper bound named "loss next turn".
13. **Suspected over-count: extra attacks.** pair_damage multiplies melee and shot alike by 1+ADDITIONAL_ATTACK, and the table sums the bonus regardless of limiters (FE:425, RU:90). If the double shot (Marksmen, Grand Elves) is ranged-only, their melee estimate doubles. Unverified: vocab.json is not in the snapshot.
14. **Idle-shooter prior only on DEFEND (P08),** although WAIT and MOVE idling are also counted as failures (TR:793).
15. **Attack values are a certainty equivalent.** Damage is at its mean and kills are step functions of damage, so a_* is not E[value]. Branches beyond the 8 most likely are dropped and the rest renormalized (B-C03, AV:25,55-82).
16. **Behaviour and science decided in C++, contradicting the stated rules.** Forced retreats with an army-only "ahead" test (C18). Tactics placement order (C17). Fallbacks (C19). The surrender mask needs cost + 2500, stricter than the engine (C15). Faerie Dragon spell draw and Master Genie targets (C14). Unit truncation by value (C06). C++ applies reward prices and the magic-resistance expectation (C01/C03), bending "Python owns the science".
17. **Search mismatches (search is off).** Simulated states are encoded without history (X09). Terminal rewards omit war machines (X05). War machines defend in the tree, but the model aims the catapult and tent in real play (X08). Our network plays both sides.
18. **Shared heads reuse features built for something else.** CATAPULT picks a wall part with move features computed for walking the catapult. TACTICS_MOVE uses walking-distance features although placement is free within the zone. HEAL targets get all-zero attack features and no value prior, only the pointer (S03, FE:520).
19. **Docs drift.** FE:22 names "model.RuleScore" (now layers.ActionScore). BSEh:31 says war machines are left to StupidAI in real play, but CATAPULT/HEAL are model actions (B-C16). SC:89 promises a reader for magic resistance.
20. **Lanchester health term is the same in every variant (FE:901).** acts_this_round, melee and shooting differ only in D. Health includes summons and clones, while value excludes them (B-I29).
21. **Lossy codes.** Unlimited retaliations = 3 (B-I22). Morale-immune = neutral 0 (B-I24). The "first bonus value" of an effect has spell-specific meaning (B-I32). Event spell ids ≥70 collapse into "none" (B-I47). HotA creatures collapse into "unknown" (B-I13).
22. **Labels disagree on war machines.** V0..V2 labels count them at start value (DA:164, "machines lost on the way: not known"). The V label counts the survivors (DA:153).
23. **Events vs units on hypnosis.** Events are signed by the actor's battle side, and History books damage by unitSide (B-C22). Unit tokens are signed by controller, so a hypnotized unit's events and its token disagree in sign.
24. **melee_from adjusts only part of the position.** The threat after the attack drops the target's killed value, but danger Δ is not reduced (B-F43).
25. **pickUnits ranks with C++ prices** (full price for war machines). With more than 14 units on a side some are invisible: wrong but never illegal decisions (CO:20-22).

New tags used (beyond the given list): MAGIC (spells/mana), POSITION (board geometry), STRUCTURE (action/architecture decomposition), IDENTITY (id embeddings), LEGALITY (masks), CREDIT (training-signal design), HEURISTIC (developer hand-set priors), DATA (training distribution/opponents). Given tags not used: GROWTH.
