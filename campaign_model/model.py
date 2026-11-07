"""The campaign model: one network for every adventure-map decision (constants.py, schema.py).

The forward pass, in order (CampaignModel.forward):
1. make_custom_features (CustomFeatures, features.py): every hand-made feature, from the raw inputs and the game's
   tables: our empire's value Φ, army strengths and worth, each window tile's value (the object we may take there + the
   expected return of the fight stepping there starts + the worth of the land it reveals, in gold) with its danger,
   our towns' danger, each TRAVEL area's value discounted by the days to it. Each fixed prior has a few learnable
   parameters that start at Nullkiller2's value, stay within a sane range and are pulled back to it by weight decay.
2. Tokens: the global state (with Φ and the question asked, if any), the heroes, the towns, the other army of a
   garrison dialog and the options of a question. Each of our heroes' windows (the map around it, 33 x 33 tiles) is
   read by convolutions and joins its hero's token; the whole map (MapEncoder: blocks with every object we see,
   convolutions down to 9 x 9 blocks) is read by each hero by direction and distance (MapReader) and summarised into
   the global token.
3. A few self-attention blocks over the tokens, with a bias by the heroes' and towns' distances in tiles (PairBias).
4. The heads: each action table scored from the tokens it is about; MOVE's tiles and TRAVEL's areas also get their
   priors and features; the action types see what MOVE and TRAVEL offer; the value heads see Φ.

Counts are what a human sees (the user, 2026-10-27: Nullkiller2's exact counts are cheating): wandering monsters,
other players' armies and the monsters of a question are VCMI's quantity ids (few, several, pack, ...) unless VCMI
shows them exactly (Visions, a detailed view); the graph turns a quantity id into a typical count. Quantities enter
scale-free: x / (x + c) at a few scales c, fractions, log-ratios; every distance is in tiles, so one model serves every
map size.
"""
import functools
import math

import torch
from torch import nn
from torch.nn import functional as F

from battle_model.device import constant
from battle_model.features import scales
from battle_model.layers import ActionScore, Block, masked_log_softmax, small_embedding, vocab_index, zero_output

from . import constants as c
from . import features as ft
from . import priors
from .schema import HEADS, INPUTS, OUTPUTS

HO, TO, A, W, HD = c.OWN_HEROES, c.OWN_TOWNS, c.ARMY_SLOTS, c.WINDOW, c.N_HOLDERS
MO, G, L, TG = c.MAP_OBJECTS, c.MAP_GRID, c.MAP_LEVELS, c.TRAVEL_GRID
COARSE = G // 4  # the map encoder's coarsest blocks per side (9)
# Each prior parameter's effect per unit of its weight (features.bounded). Adam moves a weight about lr a step whatever
# its gradient's size: joint1's campaign took 19,233 steps in 9.5 hours, so at 1e-4 a plain weight could move ~1.9 in
# the whole run if every step agreed, a few tenths under noise: a gain that must reach x5 could not. At 10 a log-gain
# moves ~1e-3 a step: x5 in ~1,600 consistent steps, ~50 minutes of joint1's updates. (In effect a 10x rate on these
# few scalars; the KL control still bounds every update.)
CORRECTION_STEP = 10.0
MAX_LOG_GAIN = math.log(5.0)  # the gains stay within 1/5..5 times their prior
EXPLORE_GOLD = 600.0  # Nullkiller2's score for an exploring step (600 / its explore priority, 1 at best): a sight of new land
HERO_HIRE_COST = 2500.0  # game_values.HERO_HIRE_COST: a lost hero's replacement
# A town's defence (Nullkiller2, FuzzyHelper::evaluateDanger): a garrisoned town's strength plus this much for its fort
# level (none, fort, citadel, castle): its walls, towers and moat
FORT_STRENGTH = (0.0, 0.0, 4000.0, 10000.0)
# What taking a town we see adds to Φ at least (empire.py): its village hall and fort buildings at cost (the others are
# unseen), and the village hall's income for income_days
VILLAGE_HALL, FORT_BUILDINGS, VILLAGE_HALL_INCOME = 10, (7, 8, 9), 500.0
DWELLINGS, UPGRADED_DWELLINGS = 30, 37  # VCMI's building ids of the level 1 dwelling, and of its upgrade
# A guard we cannot see (it stands in fog), in a window that shows no wandering monster: its log strength ratio (ours over
# theirs), an e-fold stronger than us (features.hopeless's onset). Where the window shows some, it is the worst of theirs
UNSEEN_GUARD_RATIO = -1.0
PRIOR_SUBTYPES = 16  # subtypes 0..15 of an object may have a prior of their own (object_values.csv, bank_guards.csv)

GLOBAL_FEATURES = 7 + 4 + 3 + 3 * c.N_RESOURCES + 3 * c.N_RESOURCES + 3 + 1 + 3 + 1 + 3 + 2 + 2 * c.N_TAVERN + 3 + 3 + 1
PHI_FEATURES = len(ft.PHI_SCALES) + len(ft.PHI_PARTS)
HERO_FEATURES = 3 + 1 + 3 * c.N_PRIMARY + 2 + 1 + 3 + 1 + 1 + 1 + 3 + 1 + 6 + 1 + 3
TOWN_FEATURES = 3 + 4 + 1 + 3 + 2 + 1
OPTION_FEATURES = 3 + 3 + 1
RECRUIT_FIT = 5  # (CustomFeatures.recruit_fit: what buying a creature does to the hero's army)
QUESTION_FEATURES = 6 + 3 + 2 + RECRUIT_FIT  # (v2's option_knowledge: a question's fight on the option that fights, its join on the one that takes, the two flags; a purchase's fit)
MOVE_FEATURES = ("value", "object", "fight", "loss", "win", "explore", "danger", "defend", "move_left", "enemy", "behind", "home", "under_threat", "to_main")
CELL_FEATURES = c.N_TILE_FLAGS + 3 + 2 + 4 + 3 + 1 + len(MOVE_FEATURES)
OBJECT_FEATURES = 7 + 3 + 1
BLOCK_FEATURES = c.N_GRID_COUNTS + 2 * c.N_OBJECT_CLASSES + 2  # the tile counts, the object classes (2 scales), the block's size (2 scales)
TYPE_SUMMARY = 4  # what MOVE and TRAVEL offer, for the type head's learned part
HIRE_FEATURES = 4  # what a hire brings (features.tavern_worth at 3 scales, features.hire_gain), for the hire table's learned part


def game_tables():
	"""The game's tables the features need, as tensors (zeros or ones where the vocab is missing): per creature its fight
	value, gold cost, u (what one for hire is worth, empire.py), hit points, weekly growth and best upgrade; the resource
	prices; each faction's buildings' worth (empire.py), cost, daily income over the building it replaces, and a
	dwelling's creature (and, upgraded, the creature it upgrades); the artifacts' prices; the experience each level needs;
	each spell's worth in a spellbook, and whether each creature's luck and morale count in battle (empire.py)."""
	from game_values import RESOURCE_PRICES

	tables = {
		"fight_value": torch.ones(c.CREATURE_VOCAB + 1),
		"creature_cost": torch.zeros(c.CREATURE_VOCAB + 1),
		"creature_unit": torch.zeros(c.CREATURE_VOCAB + 1),
		"creature_health": torch.zeros(c.CREATURE_VOCAB + 1),
		"creature_attack": torch.zeros(c.CREATURE_VOCAB + 1),
		"creature_defense": torch.zeros(c.CREATURE_VOCAB + 1),
		"creature_damage": torch.zeros(c.CREATURE_VOCAB + 1),  # (its average damage a hit: min and max's mean)
		"creature_resources": torch.zeros(c.CREATURE_VOCAB + 1, len(RESOURCE_PRICES)),  # (its recruit cost, each resource)
		"resource_price": torch.tensor([float(v) for v in RESOURCE_PRICES.values()]),
		"building_value": torch.zeros(c.FACTION_VOCAB + 1, c.N_BUILDINGS),
		"artifact_price": torch.zeros(c.ARTIFACT_VOCAB + 1),
		"level_experience": torch.tensor([0.0, 1000.0]),
		"creature_growth": torch.zeros(c.CREATURE_VOCAB + 1),
		"creature_upgrade": torch.full((c.CREATURE_VOCAB + 1,), -1, dtype=torch.long),
		"building_cost": torch.zeros(c.FACTION_VOCAB + 1, c.N_BUILDINGS),
		"building_income": torch.zeros(c.FACTION_VOCAB + 1, c.N_BUILDINGS),
		"dwelling_creature": torch.full((c.FACTION_VOCAB + 1, c.N_BUILDINGS), -1, dtype=torch.long),
		"dwelling_base": torch.full((c.FACTION_VOCAB + 1, c.N_BUILDINGS), -1, dtype=torch.long),
		"building_resources": torch.zeros(c.FACTION_VOCAB + 1, c.N_BUILDINGS, len(RESOURCE_PRICES)),
		"spell_value": torch.zeros(c.N_SPELLS),
		"creature_luck": torch.zeros(c.CREATURE_VOCAB + 1),
		"creature_morale": torch.zeros(c.CREATURE_VOCAB + 1),
		"creature_faction": torch.full((c.CREATURE_VOCAB + 1,), -1, dtype=torch.long),  # (its faction's id; neutral's its own)
		"artifact_primary": torch.zeros(c.ARTIFACT_VOCAB + 1, c.N_PRIMARY),  # (what it adds to its hero's primary skills)
		"artifact_slot": torch.full((c.ARTIFACT_VOCAB + 1,), -1, dtype=torch.long),  # (the first worn slot it fits)
		"artifact_combination": torch.full((c.ARTIFACT_VOCAB + 1,), -1, dtype=torch.long),  # (the combination it is a part of)
		"creature_undead": torch.zeros(c.CREATURE_VOCAB + 1),
		"creature_speed": torch.zeros(c.CREATURE_VOCAB + 1),  # (its speed: the army's slowest sets its hero's movement)
	}
	try:
		from .empire import EmpireConfig, _unit, building_value
		from game_values import resources_value

		from .vocab import NO_LUCK, NO_MORALE, artifact_prices, buildings, creature_cost, creatures, experience_table, feels, raw, spell_levels, town_creatures

		faction_ids = {row["key"]: row["id"] for row in raw()["factions"]}

		for creature_id, row in creatures().items():
			if 0 <= creature_id < c.CREATURE_VOCAB:
				tables["fight_value"][creature_id] = float(max(row.get("fight_value", 1), 1))
				tables["creature_cost"][creature_id] = float(creature_cost(creature_id))
				tables["creature_unit"][creature_id] = float(_unit(creature_id, EmpireConfig()))
				tables["creature_health"][creature_id] = float(row.get("hit_points", 0))
				tables["creature_attack"][creature_id] = float(row.get("attack", 0))
				tables["creature_defense"][creature_id] = float(row.get("defense", 0))
				tables["creature_damage"][creature_id] = 0.5 * float(row.get("damage_min", 0) + row.get("damage_max", 0))
				for k, name in enumerate(RESOURCE_PRICES):
					tables["creature_resources"][creature_id, k] = float(row.get("cost", {}).get(name, 0))
				tables["creature_growth"][creature_id] = float(row.get("growth", 0))
				tables["creature_luck"][creature_id] = float(feels(creature_id, NO_LUCK))
				tables["creature_morale"][creature_id] = float(feels(creature_id, NO_MORALE))
				tables["creature_faction"][creature_id] = faction_ids.get(row.get("faction"), -1)
				tables["creature_undead"][creature_id] = float("UNDEAD" in row.get("abilities", []))
				tables["creature_speed"][creature_id] = float(row.get("speed", 0))
		for spell, level in spell_levels().items():
			if 0 <= spell < c.N_SPELLS:
				tables["spell_value"][spell] = EmpireConfig().spell_level * level
		by_key = {row["key"]: creature_id for creature_id, row in creatures().items()}
		for creature_id, row in creatures().items():  # the upgrade worth the most (Nullkiller2's makePossibleUpgrades)
			ups = [by_key[k] for k in row.get("upgrades", []) if k in by_key and by_key[k] != creature_id and by_key[k] < c.CREATURE_VOCAB]
			if ups and 0 <= creature_id < c.CREATURE_VOCAB:
				tables["creature_upgrade"][creature_id] = max(ups, key=lambda u: creatures()[u].get("fight_value", 0))
		rows = buildings()
		for faction, building in buildings():
			if 0 <= faction < c.FACTION_VOCAB and 0 <= building < c.N_BUILDINGS:
				tables["building_value"][faction, building] = building_value(faction, building)
				row = rows[(faction, building)]
				replaced = rows.get((faction, row.get("upgrade", -1)), {})
				tables["building_cost"][faction, building] = float(resources_value(row.get("cost") or {}))
				for r, name in enumerate(RESOURCE_PRICES):
					tables["building_resources"][faction, building, r] = float((row.get("cost") or {}).get(name, 0))
				tables["building_income"][faction, building] = float(resources_value(row.get("produce") or {}) - resources_value(replaced.get("produce") or {}))
		for faction, levels in town_creatures().items():  # dwellings: level 1..7 at DWELLINGS, upgraded at UPGRADED_DWELLINGS
			for level, forms in enumerate(levels[: c.N_LEVELS]):
				if 0 <= faction < c.FACTION_VOCAB and forms:
					tables["dwelling_creature"][faction, DWELLINGS + level] = forms[0]
					if len(forms) > 1:
						tables["dwelling_creature"][faction, UPGRADED_DWELLINGS + level] = forms[1]
						tables["dwelling_base"][faction, UPGRADED_DWELLINGS + level] = forms[0]
		for artifact, price in artifact_prices().items():
			if 0 <= artifact < c.ARTIFACT_VOCAB:
				tables["artifact_price"][artifact] = float(price)
		for row in raw()["artifacts"]:
			artifact = row.get("id", -1)
			if 0 <= artifact < c.ARTIFACT_VOCAB:
				tables["artifact_primary"][artifact] = torch.tensor([float(v) for v in (row.get("primary") or [0] * c.N_PRIMARY)[: c.N_PRIMARY]])
				slots = [s for s in row.get("slots", []) if 0 <= s < c.N_WORN_SLOTS]
				if slots:
					tables["artifact_slot"][artifact] = slots[0]
				for part in row.get("parts", []):
					if 0 <= part < c.ARTIFACT_VOCAB:
						tables["artifact_combination"][part] = artifact
		tables["level_experience"] = torch.tensor([float(v) for v in experience_table()])
	except (ImportError, FileNotFoundError):
		pass
	return tables


def flat_log_softmax(logits, mask):
	"""masked_log_softmax over all but the batch axis, in the table's shape."""
	shape = logits.shape
	return masked_log_softmax(logits.reshape(shape[0], -1), mask.reshape(shape[0], -1)).reshape(shape)


def dense(values, windows, n, b):
	"""[K, ...] per window -> [B, HO, ...]: each window at its hero's slot, zeros where no hero."""
	shape = tuple(values.shape[1:])
	return values.new_zeros((b * HO,) + shape).index_copy(0, windows[:n], values[:n]).reshape((b, HO) + shape)


class Features(dict):
	"""make_custom_features' results by name (f.name)."""

	__getattr__ = dict.__getitem__
	__setattr__ = dict.__setitem__


class CustomFeatures(nn.Module):
	"""Every hand-made feature of a decision (make_custom_features: features.py's functions in order), with the game's
	tables and the learnable parameters of the fixed priors. Each prior parameter starts at Nullkiller2's value (or the
	game's), stays within its range (features.bounded) and is a correction() the optimizer's weight decay pulls back:
	  object worth    gold per object (priors/object_values.csv; a town we see: what taking it adds to Φ at least, by its
	                  fort level) x a gain (global x per type, 1/5..5), + a delta (+-5,000)
	  fights          how sharply P(win) follows the strength ratio (4, 0.5..8: at 2, a 2:1 fight was lost 1 time in 5 and
	                  a lone monster at 10:1 still cost; Nullkiller2 calls 1.1:1 safe, with exact counts, while ours are
	                  often a quantity's range, ~x2.5 wide); a town's fort (Nullkiller2's +4,000 citadel, +10,000 castle
	                  of strength while it is defended, x 1/5..5); what destroying an enemy's army is worth
	                  to us (its gold x 1, 1/5..5); what losing one of our heroes costs beyond its army, levels and
	                  artifacts, which Φ prices at nothing though a dead hero collects nothing more (a hire: 2,500 gold,
	                  x 1/5..5); a creature bank's guards are what a player knows (priors/bank_guards.csv)
	  danger          what standing within an enemy hero's reach costs: its reach x P(it beats us) x what a lost battle takes
	                  (army, hero, a hire; x 1, 1/5..5), as the change from where the hero stands now
	  exploration     what a step revealing a sight of new land is worth (600 gold, x 1/20..5)
	  travel          each object class's worth in an area (x 1/5..5), the distance discount's exponent (1.3, 0.7..2) on
	                  the pathfinder's days there (travel_days)
	  buying an army  Nullkiller2 buys all it can of what strengthens its main hero, best creatures first, and a hero in
	                  its town takes the best army (BuyArmyBehavior, pickBestCreatures): RECRUIT's prior favours the
	                  largest amount (2, 0..4), the hero visiting the town (2, 0..4) and higher levels (0.4 a level, 0..1)
	  handing over    a whole stack from a town's garrison into a hero there (1.5, 0..4), and between heroes toward the
	                  stronger one (1, 0..4: Nullkiller2's main hero gathers the army), in ARMY's prior
	  plans across    weak pulls (0.5 logit, 0..2: pull_weights): a main hero toward a town of ours that can reinforce it,
	  days            any hero toward a town of ours under threat, a scout with troops toward a main hero (MOVE's tiles,
	                  TRAVEL's areas); hiring as Nullkiller2 would (HIRE); trading toward a building short of resources
	                  (TRADE); a wandering monster credited with half the best thing it blocks, as often as we win (0..1)
	  Nullkiller2's   its skill scores on a level-up's options (1 a point, 0..2: features.skill_scores); upgrading every
	  habits          stack the upgrade makes stronger (1.5, 0..4, as UPGRADE's offer too); a spellbook for a main hero
	                  without one (2, 0..4, as BUY's offer too); no army bought while short of gold before a City Hall
	                  (2, 0..4, off RECRUIT's type logit: features.recruit_pause)
	  type coupling   how much what MOVE and TRAVEL offer keeps a turn from ending (1.5, 0..4), and what MOVE offers keeps
	                  TRAVEL from walking past it (2, 0..4): Nullkiller2 always takes its best task; at 1 each, 4 of 42
	                  turns still ended and 3 of 8 TRAVELs left with a safe pile in reach (failures.py)
	  safety          Nullkiller2 never starts a fight it cannot win: a hopeless fight's tile costs MOVE 6 logits a unit of
                  hopeless (0..12; safety_weight)
  trades          what a trade loses (its gold of worth at value_logit's price, the price of every change of Φ) times 1
                  (0..2; trade_cost_weight): TRADE's prior, and a purchase trading for itself in its entry and its type's
                  offer (features.trade_options, purchase_trades)
  garrisons       what leaving our troops in a garrison not a town's (a mine's) costs in logits (4, 0..8): Φ prices
	                  them as if in the hero, so the labels never say it is dumb, which it almost always is; NK2 never
	                  does it"""

	def __init__(self):
		super().__init__()
		for name, table in game_tables().items():
			self.register_buffer(name, table, persistent=False)  # the engine's tables (vocab.json), not learned
		self.register_buffer("quantity_counts", torch.tensor(ft.QUANTITY_COUNTS, dtype=torch.float32), persistent=False)
		from .empire import EmpireConfig

		empire = EmpireConfig()
		self.income_days, self.hero_penalty, self.explored_tile = empire.income_days, empire.hero_penalty, empire.explored_tile
		from game_values import unled_share

		self.spell_step, self.unled_share = empire.spell_step, unled_share(empire.spell_step)  # (a hero's spellbook; armies no hero leads: game_values)
		self.speed_power, self.army_morale = empire.speed_power, empire.army_morale  # (a hero's army at its movement and its make-up's morale)
		for name in ("hero_base", "skill_level", "fresh_skill_levels", "primary_point", "primary_base", "level_bonus", "secondary_share", "boost", "armyless_share", "armed_army"):  # (empire.py's heroes)
			setattr(self, name, getattr(empire, name))
		scores, subtypes = priors.object_values()
		# Tables of a fixed shape, [object, subtype] (the last column: any other subtype), looked up in one gather: a
		# checkpoint keeps the priors it was trained from, and an edit of the CSVs still loads it
		subtype_score = torch.zeros(c.OBJECT_VOCAB + 1, PRIOR_SUBTYPES + 1)
		for obj, subtype, value in subtypes:
			if not 0 <= subtype < PRIOR_SUBTYPES:
				raise ValueError(f"object {obj} subtype {subtype}: past PRIOR_SUBTYPES ({PRIOR_SUBTYPES})")
			subtype_score[obj, subtype] = value
		self.register_buffer("object_score", scores)  # persistent: a checkpoint keeps the priors it was trained from
		self.register_buffer("subtype_score", subtype_score)
		self.register_buffer("class_value", priors.class_values())
		guards = torch.zeros(c.OBJECT_VOCAB + 1, PRIOR_SUBTYPES + 1, 2)  # a creature bank's guards: strength, hit points
		for (obj, subtype), guard in priors.bank_guards().items():
			if 0 <= obj < c.OBJECT_VOCAB and 0 <= subtype < PRIOR_SUBTYPES:
				guards[obj, subtype] = torch.tensor(guard, dtype=torch.float32)
		self.register_buffer("bank_guard", guards, persistent=False)
		zeros = lambda *shape: nn.Parameter(torch.zeros(shape))  # noqa: E731 (every correction starts at its prior)
		self.object_gain, self.fight_sharpness, self.enemy_value, self.explore_gain, self.hero_value = zeros(), zeros(), zeros(), zeros(), zeros()
		self.danger_gain, self.garrison_penalty, self.fort_gain = zeros(), zeros(), zeros()
		self.meeting_gain = zeros()  # (2026-11-05: what meeting a hero of ours or entering a town of ours gains, tile_features)
		self.multi_weight = zeros()  # (2026-11-05: how far a multi-action's gain to Φ biases it: multi_prior)
		self.give_all_cost = zeros()  # (2026-11-05: what a GIVE_ALL must gain to be worth making; give_all_cost_weight)
		self.gather_risk = zeros()  # (2026-11-05: how far a GATHER's risk to its town counts; gather_risk_weight)
		self.gather_cost = zeros()  # (2026-11-05: what a GATHER into a scout loses; gather_cost_weight)
		self.disband_cost = zeros()  # (2026-11-05: what a DROP that disbands loses; disband_cost_weight)
		self.drop_scale, self.drop_sign = zeros(), zeros()  # (2026-11-05: a DROP's gain as a share of its hero's army; drop_weights)
		self.trade_waste = zeros()  # (2026-11-05: a TRADE that unlocks no purchase; trade_waste_weight)
		self.skill_weight, self.upgrade_bias, self.book_bias, self.pause_weight = zeros(), zeros(), zeros(), zeros()
		self.pull_home, self.pull_defend, self.pull_ferry, self.block_weight, self.hire_weight, self.trade_weight = (zeros() for _ in range(6))
		self.travel_exponent, self.end_coupling, self.travel_coupling = zeros(), zeros(), zeros()
		self.move_discount = zeros()  # (not a prior correction: its 0 is the old score, not a prior; move_discount_power)
		self.safety = zeros()  # (2026-11-04: Nullkiller2 never starts a fight it cannot win; safety_weight)
		self.trade_unlock, self.trade_cost = zeros(), zeros()  # (2026-11-04: TRADE's prior; trade_cost_weight. trade_unlock unused since 2026-11-05, kept: checkpoints hold it)
		self.recruit_yes = zeros()  # (2026-11-04: a dwelling's "recruit?" answered YES, as Nullkiller2 does; recruit_yes_weight)
		self.visit_yes = zeros()  # (2026-11-04: a guard's "pass?", a quest's "complete it?" answered YES, as Nullkiller2 does)
		self.select_pick = zeros()  # (2026-11-04: Nullkiller2's pick among a selection's components; select_weight)
		self.artifact_cost = zeros()  # (2026-11-04: what an artifact's move must gain to be worth making; artifact_cost_weight)
		self.recruit_all, self.recruit_hero, self.recruit_level, self.army_take, self.army_gather = zeros(), zeros(), zeros(), zeros(), zeros()
		self.class_gain = zeros(c.N_OBJECT_CLASSES)
		self.object_type_gain, self.object_type_delta = nn.Embedding(c.OBJECT_VOCAB + 1, 1), nn.Embedding(c.OBJECT_VOCAB + 1, 1)
		self.object_subtype_delta = nn.Embedding(c.PAIR_ROWS, 1)  # (an (object, subtype) pair's: constants.OBJECT_PAIRS)
		self.register_buffer("pairs", ft.pair_table(), persistent=False)
		for table in (self.object_type_gain, self.object_type_delta, self.object_subtype_delta):
			nn.init.zeros_(table.weight)

	def move_discount_power(self):
		"""The power of Nullkiller2's movement discount on a MOVE tile's gain (tile_features): exactly 0 at the start (the
		score as before), within -1.5..1.5, moving about as fast as a prior correction (CORRECTION_STEP)."""
		return 1.5 * torch.tanh(CORRECTION_STEP * self.move_discount)

	def corrections(self):
		return [
			self.object_gain, self.object_type_gain.weight, self.object_type_delta.weight, self.object_subtype_delta.weight,
			self.fight_sharpness, self.enemy_value, self.hero_value, self.danger_gain, self.meeting_gain, self.explore_gain, self.class_gain, self.travel_exponent,
			self.end_coupling, self.travel_coupling, self.garrison_penalty, self.fort_gain, self.safety, self.trade_unlock, self.trade_cost, self.recruit_yes, self.visit_yes, self.select_pick, self.artifact_cost, self.multi_weight, self.give_all_cost, self.gather_risk, self.gather_cost, self.disband_cost, self.drop_scale, self.drop_sign, self.trade_waste,
			self.recruit_all, self.recruit_hero, self.recruit_level, self.army_take, self.army_gather,
			self.skill_weight, self.upgrade_bias, self.book_bias, self.pause_weight,
			self.pull_home, self.pull_defend, self.pull_ferry, self.block_weight, self.hire_weight, self.trade_weight,
		]

	# ---- the bounded prior parameters ----

	@staticmethod
	def gain(raw, low=-MAX_LOG_GAIN, high=MAX_LOG_GAIN):
		"""A multiplicative correction: exp of a log-gain within [low, high], 1 at the start."""
		return torch.exp(ft.bounded(raw, low, high, 0.0, CORRECTION_STEP))

	def explore_gold(self):
		"""What a step revealing a sight of new land is worth (gold)."""
		return EXPLORE_GOLD * self.gain(self.explore_gain, math.log(1 / 20), MAX_LOG_GAIN)

	def garrison_cost(self):
		"""The ARMY prior's cost, in logits, of a troop move into the other army of a garrison dialog (a mine's)."""
		return ft.bounded(self.garrison_penalty, 0.0, 8.0, 4.0, CORRECTION_STEP)

	def recruit_prior(self, danger):
		"""RECRUIT's prior logits [B, TO, N_LEVELS, N_RECRUIT_AMOUNTS, N_RECRUIT_TARGETS] (danger: our towns' [B, TO]): all
		that can be bought, into the hero visiting the town (less so as the town is threatened: defenders stay home,
		DefenceBehavior), higher levels first (Nullkiller2's BuyArmy and pickBestCreatures)."""
		level = torch.arange(c.N_LEVELS, device=self.recruit_all.device).float()
		all_of_it = (torch.arange(c.N_RECRUIT_AMOUNTS, device=level.device) == c.N_RECRUIT_AMOUNTS - 1).float()
		hero = (torch.arange(c.N_RECRUIT_TARGETS, device=level.device) == 1).float()
		safe = (1 - danger.clamp(0, 1))[:, :, None, None, None]
		return (
			ft.bounded(self.recruit_level, 0.0, 1.0, 0.4, CORRECTION_STEP) * level[:, None, None]
			+ ft.bounded(self.recruit_all, 0.0, 4.0, 2.0, CORRECTION_STEP) * all_of_it[None, :, None]
			+ ft.bounded(self.recruit_hero, 0.0, 4.0, 2.0, CORRECTION_STEP) * hero[None, None, :] * safe
		)

	def army_prior(self, hero_strength):
		"""ARMY's prior logits [B, HD, A, HD, N_ARMY_AMOUNTS] (hero_strength: our heroes' log strength [B, HO]): leaving
		our troops in a garrison not a town's costs (garrison_cost); a whole stack from a town's garrison into a hero
		there gains, and between heroes toward the stronger one (Nullkiller2's pickBestCreatures: the main hero gathers)."""
		device = hero_strength.device
		holders, everything = torch.arange(HD, device=device), (torch.arange(c.N_ARMY_AMOUNTS, device=device) == c.ArmyAmount.ALL)
		hero, town = holders < HO, (holders >= HO) & (holders < HO + TO)
		into_other = ((holders < c.OTHER_ARMY)[:, None] & (holders == c.OTHER_ARMY)[None, :]).float()[:, None, :, None]
		town_to_hero = (town[:, None] & hero[None, :]).float()[:, None, :, None] * everything.float()
		toward = 2 * torch.sigmoid(2 * (hero_strength[:, None, :] - hero_strength[:, :, None])) - 1  # [B, from, to]: + toward the stronger
		toward = F.pad(toward * (1 - torch.eye(HO, device=device)), (0, HD - HO, 0, HD - HO))[:, :, None, :, None] * everything.float()
		return (
			-self.garrison_cost() * into_other + ft.bounded(self.army_take, 0.0, 4.0, 1.5, CORRECTION_STEP) * town_to_hero
			+ ft.bounded(self.army_gather, 0.0, 4.0, 1.0, CORRECTION_STEP) * toward
		)

	def nk2_weights(self):
		"""The learnable weights of Nullkiller2's habits: (a skill score point, an upgrade, a wanted spellbook, the pause)."""
		bound = lambda raw, high, start: ft.bounded(raw, 0.0, high, start, CORRECTION_STEP)  # noqa: E731
		return bound(self.skill_weight, 2.0, 1.0), bound(self.upgrade_bias, 4.0, 1.5), bound(self.book_bias, 4.0, 2.0), bound(self.pause_weight, 4.0, 2.0)

	def pull_weights(self):
		"""Nullkiller2's plans across days, weak and easy to drop (the user, 2026-10-28): each a weight from 0.5 (logits a
		tanh of the gold at stake, PULL_SCALE) that can fall to 0 or rise to 2: (home: a main hero to a town of ours that can
		reinforce it, defend: to a town of ours under threat, ferry: a scout with troops to a main hero, hire, trade)."""
		weak = lambda raw: ft.bounded(raw, 0.0, 2.0, 0.5, CORRECTION_STEP)  # noqa: E731
		return weak(self.pull_home), weak(self.pull_defend), weak(self.pull_ferry), weak(self.hire_weight), weak(self.trade_weight)

	def safety_weight(self):
		"""Nullkiller2's rule as MOVE's prior: it never starts a fight it cannot win (its isSafeToVisit), so a tile whose
		fight is hopeless by every estimate (tile_features' hopeless: 1 at a guard 54x the hero's strength) costs this many
		logits a unit: 6 at the start, 0..12 (2026-11-04: with the mixer's -3 alone, joint7's heroes with less than 1,000
		gold of army still attacked 193 times in 30 games and lost 98%, the median such tile only -1.3 logits down)."""
		return ft.bounded(self.safety, 0.0, 12.0, 6.0, CORRECTION_STEP)

	def trade_cost_weight(self):
		"""The weight of what a trade loses, its gold of worth as value_logit prices every change of Φ (1 at the start,
		0..2): TRADE's prior (features.trade_options; the user, 2026-11-04: every trade is dumb unless it buys what could
		not be bought without it) and the trades a purchase short of resources makes (features.purchase_trades). joint7
		traded 135 times a game, 77% with no purchase that day, ~900 of Φ a trade at the median. 2026-11-05: at 0.5 x
		log(1 + lost / 1000), a 4,000 loss 0.8 logits against value_logit's 3.3 for a 4,000 gain, a fifth of the
		purchases traded for themselves, ~5,200 of worth each, ~150,000 a game (joint7 v1024's games). TRADE's bonus for
		a trade that unlocks a purchase (2 logits) went then too: with the purchase trading for itself at the full
		charge, TRADE and then BUILD came 2 logits cheaper than BUILD alone, and TRADE rose from 4 to 23 a game, 61% of
		them with nothing bought that day."""
		return ft.bounded(self.trade_cost, 0.0, 2.0, 1.0, CORRECTION_STEP)

	def trade_waste_weight(self):
		"""What a TRADE that unlocks no purchase (features.trade_options: a building, a creature for hire or a hero for hire
		we cannot pay for now and can after it) costs beside what it loses, in logits (5 at the start, 0..10; the user,
		2026-11-05: any trade that is not part of a buy or a hire is suspect)."""
		return ft.bounded(self.trade_waste, 0.0, 10.0, 5.0, CORRECTION_STEP)

	def recruit_yes_weight(self):
		"""What YES to a dwelling's "Would you like to recruit?" gains, in logits (4 at the start, 0..8): Nullkiller2 answers
		every unguarded yes/no YES (AIGateway::showBlockingDialog), and YES costs nothing here (the window it opens offers
		0 too: AlphaMapAI::showRecruitmentDialog). joint7 had answered NO to 467 of 481 of them with creatures for hire and
		the gold for one (2026-11-04: one player refused the same 14 Troglodytes on six days, with 1,100 to 4,100 gold)."""
		return ft.bounded(self.recruit_yes, 0.0, 8.0, 4.0, CORRECTION_STEP)

	def visit_yes_weight(self):
		"""What YES gains, in logits (4 at the start, 0..8), at a border guard ("pass?": YES removes it, we hold its key), a
		Seer's Hut's or a Quest Guard's "complete the quest?" (asked only when we can: CGSeerHut::onHeroVisit; a Quest
		Guard then lets us through) and a Tree of Knowledge's level for its price: Nullkiller2 answers them YES
		(AIGateway::showBlockingDialog: border and quest guards always). joint7 answered NO to all of the first 150
		(2026-11-04: one hero asked 53 times in 15 days at the same Quest Guard)."""
		return ft.bounded(self.visit_yes, 0.0, 8.0, 4.0, CORRECTION_STEP)

	def artifact_cost_weight(self):
		"""What an ARTIFACT move loses before what it gains, in logits (5 at the start, 0..12): a move that changes nothing
		of Φ is worse than none. The first schema-13 games handed artifacts back and forth, 460-1,417 moves a game: END's
		learned logit in an exchange sits near -3 (the model ends exchanges rarely, the 8 steps filled with what is legal),
		so a move must score under it; at 5 a move that changes nothing loses to END ~9 in 10 and a +2 sword for a main
		hero leading 20k (~900 gold of Φ) still wins ~6 in 10."""
		return ft.bounded(self.artifact_cost, 0.0, 12.0, 5.0, CORRECTION_STEP)

	def select_weight(self):
		"""What Nullkiller2's pick among a selection's components gains, in logits (3 at the start, 0..8; nk2_select)."""
		return ft.bounded(self.select_pick, 0.0, 8.0, 3.0, CORRECTION_STEP)

	def block_share(self):
		"""How much of what a wandering monster blocks counts for its fight (x P(win)): 0.5 at the start, 0..1."""
		return ft.bounded(self.block_weight, 0.0, 1.0, 0.5, CORRECTION_STEP)

	def couplings(self):
		"""(how much what the tables offer keeps a turn from ending, how much what MOVE offers holds TRAVEL back)."""
		return ft.bounded(self.end_coupling, 0.0, 4.0, 1.5, CORRECTION_STEP), ft.bounded(self.travel_coupling, 0.0, 4.0, 2.0, CORRECTION_STEP)

	@staticmethod
	def subtype_column(subtype):
		return torch.where((subtype >= 0) & (subtype < PRIOR_SUBTYPES), subtype, torch.full_like(subtype, PRIOR_SUBTYPES))

	def object_gold(self, obj, subtype, town_value=None):
		"""What visiting each object is worth in gold: Nullkiller2's score (with its subtype's where the table has one;
		a town's: town_value where given) x its gain, + its delta. Whether it may be taken is the caller's gate."""
		index = vocab_index(obj, c.OBJECT_VOCAB)
		score = self.object_score[index] + self.subtype_score[index, self.subtype_column(subtype)]
		if town_value is not None:
			score = torch.where(obj == ft.OBJ_TOWN, town_value, score)
		gain = self.gain(self.object_gain + self.object_type_gain(index).squeeze(-1))
		raw_delta = self.object_type_delta(index).squeeze(-1) + self.object_subtype_delta(ft.object_pair(self.pairs, obj, subtype)).squeeze(-1)
		return score * gain + ft.bounded(raw_delta, -5000.0, 5000.0, 0.0, CORRECTION_STEP)

	def bank_guards(self, obj, subtype):
		"""A creature bank's guards as a player knows them (priors/bank_guards.csv): (expected strength, expected hit
		points), 0 for anything else."""
		guard = self.bank_guard[vocab_index(obj, c.OBJECT_VOCAB), self.subtype_column(subtype)]
		return guard[..., 0], guard[..., 1]

	# ---- make_custom_features ----

	def forward(self, x, windows):
		return self.make_custom_features(x, windows)

	def make_custom_features(self, x, windows):
		"""Every hand-made feature of a batch of decisions, in order; windows: our heroes' windows to compute (flat
		decision x OWN_HEROES + hero indices, CampaignModel.windows)."""
		f = Features()
		f.phi_parts, f.phi_total = ft.empire_value(x, self)
		f.resource_worth = self.resource_worth(x)
		f.phi = ft.phi_features(f.phi_parts, f.phi_total)
		f.hero_counts, f.hero_power, f.hero_gold, f.hero_health = ft.army(x["h_army_creature"], x["h_army_count"], x["h_detailed"], self)
		f.hero_strength = ft.hero_strength(x["h_primary"], f.hero_power)
		# What each hero of ours makes of the army it leads (Φ's: its attack, defense and spells; features.army_multiplier)
		f.hero_multiplier = ft.hero_multipliers(x["h_primary"][:, :HO], x["h_artifacts"][:, :HO], self)
		f.hero_worth = ft.hero_worth(x, self)
		f.level_gain = ft.level_gains(x, self)
		f.town_counts, f.town_power, f.town_gold, f.town_health = ft.army(x["t_garrison_creature"], x["t_garrison_count"], x["t_detailed"], self)
		f.town_strength = self.town_strength(x, f.town_power)
		# The armies' fighting numbers (features.army_battle; a hero's attack and defense add its primary skills'), and our
		# heroes' spell damage a round (features.spell_damage): tile_features' damage-race estimates of a fight
		f.hero_damage, f.hero_attack, f.hero_defense = ft.army_battle(x["h_army_creature"], x["h_army_count"], x["h_detailed"], self)
		primary = x["h_primary"].clamp(min=0).float()
		f.hero_attack, f.hero_defense = f.hero_attack + primary[..., 0], f.hero_defense + primary[..., 1]
		f.hero_spells = ft.spell_damage(x, self)
		f.town_damage, f.town_attack, f.town_defense = ft.army_battle(x["t_garrison_creature"], x["t_garrison_count"], x["t_detailed"], self)
		# The armed objects that fight a hero stepping on them (schema 11's a_*: garrisons not ours), as a human sees them
		# (a record from before: none, zeros)
		armed = torch.where(x["a_present"].unsqueeze(-1) & (x["a_count"] > 0), x["a_creature"], torch.full_like(x["a_creature"], -1))
		_, f.armed_power, f.armed_gold, f.armed_health = ft.army(armed, x["a_count"], x["a_detailed"], self)
		f.armed_damage, f.armed_attack, f.armed_defense = ft.army_battle(armed, x["a_count"], x["a_detailed"], self)
		f.town_value = self.town_value(x)
		f.town_danger = ft.town_danger(x, f.hero_strength, f.town_strength)
		f.emptied_danger = ft.emptied_danger(x, f.hero_strength)
		f.map_worth, f.map_gate, f.map_monster_power = self.map_objects(x)
		f.object_counts = ft.object_counts(x)
		f.last_hero, f.last_town = ft.last_actors(x)
		# Nullkiller2's brain (features.py): its roles, skill scores, gold pressure and pause, buildings, upgrades, spellbook;
		# its plans across days as weak pulls: reinforcements in our towns, their stakes, hiring, trading toward a building
		f.main = ft.hero_roles(x, f.hero_strength)
		f.skill_scores = ft.skill_scores(x, f.main, self)
		f.gold_pressure = ft.gold_pressure(x, self)
		f.recruit_pause = ft.recruit_pause(x, f.gold_pressure, f.town_danger)
		f.build_rewards, f.upgrade_gains, f.spellbook_wanted = ft.build_rewards(x, self), ft.upgrade_gains(x, self), ft.spellbook_wanted(x, f.main)
		f.reinforcements, f.town_stakes = ft.reinforcements(x, self, f.town_gold), ft.town_stakes(x, self, f.town_gold)
		f.hire_want = ft.hire_want(x, f.gold_pressure, f.town_danger, ft.object_classes(x))
		f.tavern_worth = ft.tavern_worth(x, self)
		f.hire_gain = ft.hire_gain(x, self, f.tavern_worth)
		f.trade_needs = ft.trade_needs(x, self)
		f.trade_options = ft.trade_options(x, self)
		f.build_trade, f.recruit_trade, f.recruit_short = ft.purchase_trades(x, self)
		f.area_centres = ft.centres(x, TG)
		f.travel_prior = self.travel_prior(x, f)
		if windows.shape[0] > 0:
			f.update(self.tile_features(x, windows, f))
		f.option_prior, f.option_features = self.option_knowledge(x, f)
		f.artifact_prior = self.artifact_prior(x, f)
		f.multi_prior = self.multi_prior(x, f)
		return f

	def question_guards(self, x, f):
		"""A question's guards (schema 11: d_guard_*; the user, 2026-11-04: a battle decided as a MOVE's fight is, an offer to
		join easy to take): the hero asked (d_hero) against them by tile_features' estimates (army, total, race, spells),
		their hopeless view and the fight's value (its experience and a guarded object's worth, the bank's reward, as often
		as we win; its losses as fight_return counts them); and what taking them is worth (at cost + u) less the join
		option's price. (the fight's features [B, 6], its prior [B]: value_logit of its value less safety x hopeless, as
		MOVE's; the join's features [B, 3], its prior [B])."""
		hero = x["d_hero"].clamp(min=0, max=HO - 1)  # (our hero asked; with none, its options carry no battle or join)

		def asked(values):  # [B] of a [B, H or HO] per-hero value
			return values.gather(1, hero.unsqueeze(1)).squeeze(1)

		guards = torch.where(x["d_guard_count"] > 0, x["d_guard_creature"], torch.full_like(x["d_guard_creature"], -1))
		_, power, gold, health = ft.army(guards, x["d_guard_count"], x["d_guard_detailed"], self)
		damage, attack, defense = ft.army_battle(guards, x["d_guard_count"], x["d_guard_detailed"], self)
		theirs = torch.log1p(power)
		ours = (asked(f.hero_health), asked(f.hero_damage), asked(f.hero_attack), asked(f.hero_defense))
		ratios = torch.stack(
			[
				torch.log1p(asked(f.hero_power)) - theirs, asked(f.hero_strength) - theirs,
				ft.race_strength(*ours, health, damage, attack, defense), ft.race_strength(*ours, health, damage, attack, defense, asked(f.hero_spells)),
			],
			dim=-1,
		)  # [B, 4]
		hopeless = ft.hopeless(ratios.amax(dim=-1))
		win = ft.win_chance(ratios[:, 1], ft.bounded(self.fight_sharpness, 0.5, 8.0, 4.0, CORRECTION_STEP))
		loss = ft.expected_loss(ratios[:, 1], torch.ones_like(win))
		obj = x["d_object"]
		# A bank's reward (a mine's, a dwelling's): never a monster's question's (a join; d_object the monster, or our hero
		# itself on a step into its zone, or a town: object_gold of a hero was -760, a fake loss on 41 of 69 questions)
		holds = (obj >= 0) & (obj != ft.OBJ_MONSTER) & (obj != ft.OBJ_HERO) & (obj != ft.OBJ_TOWN) & ~x["o_join"].any(dim=-1)
		reward = self.object_gold(obj, x["d_subtype"]) * holds
		gain = ft.experience_value(asked(x["h_exp"].clamp(min=0).float()), health, asked(f.level_gain), self)
		at_stake = asked(f.hero_worth) + (HERO_HIRE_COST - self.hero_base) * self.gain(self.hero_value)
		fight = ft.fight_return(win, loss, asked(f.hero_gold), at_stake, gain) + win * reward
		price = (x["o_amount"].clamp(min=0).float() * x["o_join"]).sum(dim=-1)
		joined = gold - price
		return (
			torch.cat([ft.strength_view(ratios), hopeless.unsqueeze(-1), torch.tanh(ft.value_logit(fight) / 4).unsqueeze(-1)], dim=-1),
			ft.value_logit(fight) - self.safety_weight() * hopeless,
			torch.stack([torch.log1p(gold / 1000.0), torch.log1p(price / 1000.0), torch.tanh(ft.value_logit(joined) / 4)], dim=-1),
			ft.value_logit(joined),
		)

	def nk2_yes(self, x):
		"""What YES gains [B] where Nullkiller2 answers YES and joint7 had answered NO (2026-11-04): a dwelling's "recruit?"
		(recruit_yes_weight), a guard's "pass?", a quest's "complete it?", a Tree of Knowledge's level (visit_yes_weight);
		0 elsewhere, and wherever an option fights or takes the question's creatures."""
		obj = x["d_object"]
		plain = ~(x["o_battle"] | x["o_join"]).any(dim=-1)
		of = lambda objects: (functools.reduce(torch.logical_or, [obj == o for o in objects]) & plain).float()  # noqa: E731
		return of(ft.DWELLING_OBJECTS) * self.recruit_yes_weight() + of(ft.VISIT_OBJECTS) * self.visit_yes_weight()

	def nk2_select(self, x, f):
		"""Nullkiller2's pick among a selection's components [B, O], x select_weight (AIGateway::showBlockingDialog): the
		last one, but a Treasure Chest's gold (two components, the first a resource) for a hero not its main one or under
		gold pressure (as smooth as the pressure); 0 where no option is a component."""
		component = x["option_mask"] & (x["o_kind"] == int(c.OptionKind.COMPONENT))
		index = torch.arange(component.shape[1], device=component.device)
		count = component.sum(dim=-1, keepdim=True)
		last, first = (component & (index == count - 1)).float(), (component & (index == 0)).float()  # (AlphaMapAI lists the components first, then a NO)
		hero = x["d_hero"].clamp(0, HO - 1)
		main = (f.main.gather(1, hero.unsqueeze(1)).squeeze(1) & (x["d_hero"] >= 0)).float()
		chest = ((count.squeeze(-1) == 2) & (x["o_type"][:, 0] == int(c.ComponentType.RESOURCE))).float()
		gold = (chest * torch.maximum(1 - main, torch.sigmoid(10 * (f.gold_pressure - ft.MAX_GOLD_PRESSURE)))).unsqueeze(-1)
		return self.select_weight() * (gold * first + (1 - gold) * last)

	def nk2_amount(self, x):
		"""Nullkiller2 buys all it can at a dwelling (AIGateway::recruitCreatures): its largest amount option [B, O], x the
		town's recruit_all weight (the same habit there: recruit_prior)."""
		amounts = x["option_mask"] & (x["o_kind"] == int(c.OptionKind.AMOUNT))
		value = torch.where(amounts, x["o_amount"], torch.full_like(x["o_amount"], -1))
		most = amounts & (value == value.amax(dim=-1, keepdim=True)) & (value > 0)
		return most.float() * ft.bounded(self.recruit_all, 0.0, 4.0, 2.0, CORRECTION_STEP)

	def multi_gain(self, x, f):
		"""What each MULTI entry does to Φ [B, TO, N_MULTI] (gold; c.MultiKind; the user, 2026-11-05: multi-actions biased
		by what they gain): RECRUIT_ALL every creature the town has for hire we can pay for, the highest level first (each
		town's from all of our resources), at its worth (cost + u) x unled_share for its cost and the u it was worth for
		hire: its worth x (unled_share - 1), a little below 0. The others as Φ prices a hero's army (features.led_values:
		its worth x the hero's multiplier x its movement, and its make-up's morale), the armies joined: RECRUIT_ALL_HERO the
		visiting hero's army with the creatures bought, less their worth; GATHER its army with what of the garrison VCMI's
		bulk move takes (features.bulk_moves: onto its creature or into a free slot), less that at unled_share and what
		it adds to the town's expected loss (gather_loss); GIVE_ALL (rows 0 and 1: the exchange's receiving hero,
		x_heroes' first and second) the receiver's army with what of the other's the bulk move takes, and the other's
		left, less both apart. The multipliers are Φ's (f.hero_multiplier: features.hero_multipliers)."""
		b = x["g_day"].shape[0]
		have = x["g_resources"][:, None, :].clamp(min=0).float().expand(b, TO, -1)  # [B, TO, R]
		creatures = x["t_available_creature"][:, :TO]  # [B, TO, LV]
		index = vocab_index(creatures, c.CREATURE_VOCAB)
		counts = x["t_available"][:, :TO].clamp(min=0).float() * (creatures >= 0)
		def recruited(into):  # (executeMulti's: the highest level first, a level the army has no slot for left out)
			cash, free = have, (into < 0).sum(dim=-1)  # [B, TO]
			worth, bought = torch.zeros(b, TO, device=have.device), []
			for level in reversed(range(c.N_LEVELS)):
				creature = creatures[:, :, level]
				cost = self.creature_resources[index[:, :, level]]  # [B, TO, R]
				afford = torch.where(cost > 0, torch.floor(cash / cost.clamp(min=1e-6)), torch.full_like(cost, 1e9)).amin(dim=-1)
				there = ((into == creature.unsqueeze(-1)) & (into >= 0)).any(dim=-1)
				fits = there | (free > 0)
				amount = torch.minimum(counts[:, :, level], afford) * fits
				free = free - ((amount > 0) & ~there).long()
				cash = cash - amount.unsqueeze(-1) * cost
				worth = worth + amount * (self.creature_cost[index[:, :, level]] + self.creature_unit[index[:, :, level]])
				bought.append(amount)
			return worth, torch.stack(bought[::-1], dim=-1)  # [B, TO], [B, TO, LV]: each level's creatures bought
		# Our heroes' armies as Φ prices them, and the hero visiting each town (at most one)
		army_c, army_n = x["h_army_creature"][:, :HO], x["h_army_count"][:, :HO].float()
		rest = x["h_morale"][:, :HO].float() - ft.make_up_morales(army_c, army_n, self)  # (each hero's morale without its army's make-up's)
		led = ft.led_values(army_c, army_n, f.hero_multiplier, rest, self)  # [B, HO]
		ours = x["h_present"][:, :HO] & (x["h_owner"][:, :HO] == c.OWNER_SELF) & ~x["h_garrisoned"][:, :HO]
		at = (x["h_town"][:, :HO].unsqueeze(-1) == torch.arange(TO, device=have.device)) & ours.unsqueeze(-1)  # [B, HO, TO]
		has, who = at.any(dim=1).float(), at.float().argmax(dim=1)  # [B, TO]
		slots = lambda v: v.gather(1, who.unsqueeze(-1).expand(-1, -1, A))  # noqa: E731 ([B, HO, A] -> the visitor's [B, TO, A])
		visitor_c, visitor_n = slots(army_c), slots(army_n)
		visitor_m, visitor_r, visitor_led = f.hero_multiplier.gather(1, who), rest.gather(1, who), led.gather(1, who)
		# The town's upper army, which RECRUIT_ALL and GATHER act on (VCMI's getUpperArmy: Encoder.cpp): its garrison hero's
		# when one is there (the town's own stacks are then empty), else the town's (until 2026-11-05 always the town's)
		garrisoned = x["t_garrison_hero"][:, :TO]  # [B, TO]: hero token, -1 none
		upper = garrisoned.clamp(min=0).unsqueeze(-1).expand(-1, -1, A)
		in_hero = (garrisoned >= 0).unsqueeze(-1)
		garrison_c = torch.where(in_hero, x["h_army_creature"].gather(1, upper), x["t_garrison_creature"][:, :TO])
		garrison_n = torch.where(in_hero, x["h_army_count"].gather(1, upper), x["t_garrison_count"][:, :TO]).float()
		worth, _ = recruited(garrison_c)  # (RECRUIT_ALL's: into the upper army)
		hero_worth, bought = recruited(visitor_c)  # (RECRUIT_ALL_HERO's: into the visitor)
		with_recruits = ft.led_values(torch.cat([visitor_c, creatures], dim=-1), torch.cat([visitor_n, bought], dim=-1), visitor_m, visitor_r, self)
		gathered = ft.bulk_moves(visitor_c, garrison_c)  # [B, TO, A]: what of the garrison fits into the visitor (a garrison hero keeps one creature: not counted)
		with_garrison = ft.led_values(torch.cat([visitor_c, torch.where(gathered, garrison_c, torch.full_like(garrison_c, -1))], dim=-1), torch.cat([visitor_n, garrison_n * gathered], dim=-1), visitor_m, visitor_r, self)
		index = vocab_index(garrison_c, c.CREATURE_VOCAB)
		gathered_worth = (garrison_n * gathered * (self.creature_cost[index] + self.creature_unit[index]) * (garrison_c >= 0)).sum(dim=-1)  # [B, TO]
		gains = torch.zeros(b, TO, c.N_MULTI, device=have.device)
		gains[..., c.MultiKind.RECRUIT_ALL] = worth * (self.unled_share - 1.0)
		gains[..., c.MultiKind.RECRUIT_ALL_HERO] = (with_recruits - visitor_led - hero_worth) * has
		gains[..., c.MultiKind.GATHER] = (with_garrison - visitor_led - self.unled_share * gathered_worth) * has - self.gather_loss(f)
		pair = x["x_heroes"] - 1  # [B, 2]: the exchange's heroes (tokens + 1; -1 none)
		p = pair.clamp(min=0)
		pair_c, pair_n = army_c.gather(1, p.unsqueeze(-1).expand(-1, -1, A)), army_n.gather(1, p.unsqueeze(-1).expand(-1, -1, A))  # [B, 2, A]
		giver_c, giver_n = pair_c.flip(1), pair_n.flip(1)  # [B, 2, A]: row r's giver's army
		given = ft.bulk_moves(pair_c, giver_c)  # (GIVE_ALL's bulk move: what fits into row r's receiver)
		none = torch.full_like(giver_c, -1)
		joined = ft.led_values(torch.cat([pair_c, torch.where(given, giver_c, none)], dim=-1), torch.cat([pair_n, giver_n * given], dim=-1), f.hero_multiplier.gather(1, p), rest.gather(1, p), self)  # [B, 2]: row r's receiver leading its own and what came
		left = ft.led_values(torch.where(given, none, giver_c), giver_n * ~given, f.hero_multiplier.gather(1, p.flip(1)), rest.gather(1, p.flip(1)), self)  # (the giver's that did not fit)
		both = (pair >= 0).all(dim=1).float().unsqueeze(-1)
		apart = led.gather(1, p).sum(dim=1, keepdim=True)  # [B, 1]: the two armies before
		gains[:, :2, c.MultiKind.GIVE_ALL] = (joined + left - apart) * both
		# TAKE_BEST, TAKE_BEST_FAST: the armies after as the C++ makes them (x_take_*: receiving row, variant, side), the
		# receiver's led by it, the other's by the other
		take_c, take_n = x["x_take_creature"], x["x_take_count"].float()  # [B, 2, 2, 2, A]
		hero_of = torch.stack([p, p.flip(1)], dim=-1).unsqueeze(2).expand(-1, -1, 2, -1)  # [B, 2 rows, 2 variants, 2 sides]: the hero leading each
		after = ft.led_values(take_c, take_n, f.hero_multiplier.gather(1, hero_of.reshape(b, -1)).reshape(hero_of.shape), rest.gather(1, hero_of.reshape(b, -1)).reshape(hero_of.shape), self).sum(dim=-1)  # [B, 2, 2]
		for variant, kind in enumerate((c.MultiKind.TAKE_BEST, c.MultiKind.TAKE_BEST_FAST)):
			gains[:, :2, kind] = (after[:, :, variant] - apart) * both
		# DROP_SLOW, DROP_FOREIGN (a hero of ours a row): the army without the stacks dropped (h_drop), and those at
		# unled_share in the garrison that takes them (h_drop_kept), else disbanded
		drop = x["h_drop"].float()  # [B, HO, 2, A]
		keep_c = army_c.unsqueeze(2).expand(-1, -1, 2, -1)
		kept = ft.led_values(torch.where(drop > 0, torch.full_like(keep_c, -1), keep_c), army_n.unsqueeze(2) * (1 - drop), f.hero_multiplier.unsqueeze(-1), rest.unsqueeze(-1), self)  # [B, HO, 2]
		index = vocab_index(keep_c, c.CREATURE_VOCAB)
		dropped = (army_n.unsqueeze(2) * drop * (self.creature_cost[index] + self.creature_unit[index]) * (keep_c >= 0)).sum(dim=-1)  # [B, HO, 2]: their worth
		for variant, kind in enumerate((c.MultiKind.DROP_SLOW, c.MultiKind.DROP_FOREIGN)):
			gains[:, :HO, kind] = kept[:, :, variant] - led + self.unled_share * dropped[:, :, variant] * x["h_drop_kept"][:, :, variant].float()
		return gains

	def gather_risk_weight(self):
		"""How much of what a GATHER adds to its town's expected loss counts against it (gather_loss): 1 at the start, 0..2."""
		return ft.bounded(self.gather_risk, 0.0, 2.0, 1.0, CORRECTION_STEP)

	def gather_loss(self, f):
		"""What a GATHER adds to its town's expected loss [B, TO] (gold), at gather_risk_weight: the town without its army
		(f.emptied_danger x its buildings) less as it stands (f.town_danger x its buildings and army; an enemy that takes
		it either way takes the army too, unless its visitor has it). Nullkiller2 takes every garrison into its visitor
		(AIGateway::moveCreaturesToHero) and defends threatened towns apart (DefenceBehavior); ours took them with no
		defence: the first schema-14 games' garrisons held 21% of our armies (80% before), 31% of our towns stood empty,
		and we lost 0.13 towns a player-game (none before), 40% of them empty the day before; the eval had 28 of 48
		knocked out (19 before)."""
		stakes = f.town_stakes[:, :TO]
		risk = f.emptied_danger * (stakes - f.town_gold[:, :TO]) - f.town_danger[:, :TO] * stakes
		return self.gather_risk_weight() * risk

	def multi_weight_value(self):
		"""How far a multi-action's gain to Φ (multi_gain, as value_logit prices it) biases MULTI's entries: 1 at the start, 0..2."""
		return ft.bounded(self.multi_weight, 0.0, 2.0, 1.0, CORRECTION_STEP)

	def give_all_cost_weight(self):
		"""What a GIVE_ALL to the weaker of the exchange's two heroes (stronger_receiver) loses before what it gains, in
		logits (3 at the start, 0..8). The first schema-14 games, Φ's gain alone deciding the receiver, gave armies back
		and forth: 300 give-alls a game, their median gain 3 gold (two heroes' factors 0.022 apart at the median), 40%
		toward the hero making less of the army, a third sending it back the same day, and Nullkiller2's labels at the next
		exchange moving it back (its exchange labels ARMY 85%, from 52%). A cost on both ways instead (a give-all must gain
		to be made) left the army to troop moves: in a test game 82 of 265 exchanges made 8, the most allowed."""
		return ft.bounded(self.give_all_cost, 0.0, 8.0, 3.0, CORRECTION_STEP)

	def stronger_receiver(self, x, f):
		"""Which of GIVE_ALL's rows [B, 2] (its receiver: x_heroes' first or second) is the stronger hero, 1 or 0: the
		higher level, else the one making more of an army (Φ's factor, f.hero_multiplier), else the bigger army. The user's
		give-all is "to the strongest hero"; the level first as Nullkiller2's exchange (adviseExchange's destination: the
		higher level, else the stronger army), which names the same hero at 93% of the schema-14 games' exchanges and Φ's
		factor first at 71%."""
		pair = (x["x_heroes"] - 1).clamp(min=0)  # [B, 2]
		level = x["h_level"][:, :HO].gather(1, pair).float()
		factor, army = f.hero_multiplier.gather(1, pair), f.hero_gold[:, :HO].gather(1, pair)
		first = (level[:, 0] > level[:, 1]) | ((level[:, 0] == level[:, 1]) & ((factor[:, 0] > factor[:, 1]) | ((factor[:, 0] == factor[:, 1]) & (army[:, 0] >= army[:, 1]))))
		return torch.stack([first, ~first], dim=1).float()

	def gather_cost_weight(self):
		"""What a GATHER into a hero that is not one of our main heroes (main_visitor) loses before what it gains, in logits
		(3 at the start, 0..8): the user's armies go to the main hero. In two schema-14 test games (Arrogance, 28 days)
		scouts took half of the garrisons gathered (169k of 344k gold), and our towns stood empty (37-40% of them from day
		14)."""
		return ft.bounded(self.gather_cost, 0.0, 8.0, 3.0, CORRECTION_STEP)

	def disband_cost_weight(self):
		"""What a DROP_SLOW or DROP_FOREIGN that disbands its stacks (no garrison takes them: h_drop_kept) loses before what
		it gains, in logits (6 at the start, 0..10; the user, 2026-11-05: "we do not want to remove our creatures", a big
		minus, 5 or more): creatures destroyed. multi_gain prices the loss, but value_logit compresses it (a third of a
		hero's army about -3), and sampled at that the joint7 games of v1233-v1236 disbanded 98 times in 14 games, a median
		19% of the hero's army each time (12k gold a game), twice v1188-v1192's; at 3 (444ebfc) 5 in 12 games."""
		return ft.bounded(self.disband_cost, 0.0, 10.0, 6.0, CORRECTION_STEP)

	def drop_weights(self):
		"""DROP's prior over its gain as a share of its hero's army (drop_prior): (the share's weight, 20 at the start, 0..60:
		a 5% gain +1 logit, a 20% loss -4; the gain's sign's weight for a disband, 6 at the start, 0..12: as the disband's
		cost, so a disband that gains starts about neutral and one that loses at -12 or less). The user, 2026-11-05: a sign
		and the share, each with a trainable weight, easy to learn, to kill dumb dismisses; a disband with a positive
		value neutral, learned from real data."""
		return ft.bounded(self.drop_scale, 0.0, 60.0, 20.0, CORRECTION_STEP), ft.bounded(self.drop_sign, 0.0, 12.0, 6.0, CORRECTION_STEP)

	def drop_prior(self, x, f, gains):
		"""DROP_SLOW's and DROP_FOREIGN's prior logits [B, HO, 2] from their gains (multi_gain, gold): the gain as a share of
		the hero's army as Φ prices it led (features.led_values: what the drop loses, less what the rest wins in movement
		and morale), in -1..1, at the share's weight; and a disband (no garrison takes the stacks: h_drop_kept) pays
		disband_cost_weight and is pushed by its gain's sign at the sign's weight. At the start a disband that loses
		anything costs 12 logits or more (a fifth of the army -16), one that wins 5% of the army +1, about neutral (another
		hero would usually take the stacks better: learned from real data); a drop into a garrison its share's alone."""
		army_c, army_n = x["h_army_creature"][:, :HO], x["h_army_count"][:, :HO].float()
		rest = x["h_morale"][:, :HO].float() - ft.make_up_morales(army_c, army_n, self)
		led = ft.led_values(army_c, army_n, f.hero_multiplier, rest, self)  # [B, HO]
		share = (gains / led.clamp(min=1.0).unsqueeze(-1)).clamp(-1.0, 1.0)  # [B, HO, 2]
		disbands = 1.0 - x["h_drop_kept"][:, :HO].float()
		scale, sign = self.drop_weights()
		return scale * share + disbands * (sign * torch.sign(share) - self.disband_cost_weight())

	def main_visitor(self, x, f):
		"""Whether the hero visiting each town of ours is one of our main heroes [B, TO] (f.main: Nullkiller2's roles, by
		fighting strength), 1 or 0 (0 where none visits)."""
		ours = x["h_present"][:, :HO] & (x["h_owner"][:, :HO] == c.OWNER_SELF) & ~x["h_garrisoned"][:, :HO]
		at = (x["h_town"][:, :HO].unsqueeze(-1) == torch.arange(TO, device=ours.device)) & (ours & f.main).unsqueeze(-1)  # [B, HO, TO]
		return at.any(dim=1).float()

	def multi_prior(self, x, f):
		"""MULTI's prior logits [B, TO, N_MULTI]: what each multi-action does to Φ (multi_gain) as value_logit prices it, at
		multi_weight_value; the army sent to another than the strongest hero pays: a GIVE_ALL, TAKE_BEST or TAKE_BEST_FAST
		to the weaker hero give_all_cost_weight, a GATHER into a scout gather_cost_weight. DROP's its own (drop_prior)."""
		gains = self.multi_gain(x, f)
		prior = self.multi_weight_value() * ft.value_logit(gains)
		# (DROP rows are our heroes; one kind at a time: a list index is a CPU tensor, its copy breaks CUDA graph capture,
		# and every campaign step of daac154's went eager, 3-4x slower, 2026-11-05)
		slow, foreign = c.MultiKind.DROP_SLOW, c.MultiKind.DROP_FOREIGN
		drops = self.drop_prior(x, f, torch.stack([gains[:, :HO, slow], gains[:, :HO, foreign]], dim=-1))
		prior[:, :HO, slow], prior[:, :HO, foreign] = drops[..., 0], drops[..., 1]
		cost = torch.zeros_like(prior)
		weaker = (1.0 - self.stronger_receiver(x, f)) * self.give_all_cost_weight()
		for kind in (c.MultiKind.GIVE_ALL, c.MultiKind.TAKE_BEST, c.MultiKind.TAKE_BEST_FAST):
			cost[:, :2, kind] = weaker
		cost[..., c.MultiKind.GATHER] = (1.0 - self.main_visitor(x, f)) * self.gather_cost_weight()
		return prior - cost

	def artifact_prior(self, x, f):
		"""What each ARTIFACT entry does to Φ, as a prior logit [B, HO, N_ARTIFACT_POSITIONS, N_ARTIFACT_TARGETS]
		(value_logit; the user, 2026-11-04: move them to the strongest hero): an artifact's primary skills change what
		each hero makes of the army it leads (features.hero_multipliers: attack, defense, and with a spellbook spell power
		and knowledge); GIVE takes a worn one's off its hero and puts it on the
		other of the exchange (x_heroes); EQUIP wears one from the backpack in place of its slot's; ASSEMBLE turns the
		worn parts into their combination (its price over theirs, its bonuses in place of theirs). An artifact's price
		stays ours wherever it is: Φ's artifacts part moves only with an assembly."""
		ids = torch.cat([x["h_artifacts"][:, :HO], x["h_backpack"][:, :HO]], dim=-1)  # [B, HO, P]
		index = vocab_index(ids, c.ARTIFACT_VOCAB)
		present = ids >= 0
		worn = (torch.arange(ids.shape[-1], device=ids.device) < c.N_WORN_SLOTS) & present  # (a backpack artifact gives its hero nothing)
		bonus = self.artifact_primary[index] * present.unsqueeze(-1).float()  # [B, HO, P, N_PRIMARY]
		primary = x["h_primary"][:, :HO].clamp(min=0).float()  # [B, HO, N_PRIMARY]
		spellbook = (x["h_artifacts"][:, :HO] == ft.SPELLBOOK).any(dim=-1)  # [B, HO]
		army = f.hero_gold[:, :HO]

		def lead_of(skills, book):  # what an army counts at in Φ with a hero of these primary skills [.., 4] (features.hero_multipliers')
			return self.unled_share * ft.army_multiplier(skills[..., 0], skills[..., 1], skills[..., 2], skills[..., 3], book, self.spell_step)

		lead = lead_of(primary, spellbook)  # [B, HO]

		def change(gained):  # [B, HO, P]: what the hero's army gains in Φ with primary skills gained [B, HO, P, N_PRIMARY]
			return army.unsqueeze(-1) * (lead_of(primary.unsqueeze(2) + gained, spellbook.unsqueeze(-1)) - lead.unsqueeze(-1))

		on = lambda values, condition: values * condition.float()  # noqa: E731
		first, second = x["x_heroes"][:, 0:1] - 1, x["x_heroes"][:, 1:2] - 1  # (tokens + 1; 0, none: -1)
		heroes = torch.arange(HO, device=ids.device).unsqueeze(0)
		partner = torch.where(heroes == first, second, torch.where(heroes == second, first, torch.full_like(heroes.expand_as(first.expand(-1, HO)), -1)))
		other = partner.clamp(min=0)
		slot = self.artifact_slot[index]  # (where it is worn)
		# (the other wears it only if its slot there is free, else it lands in that hero's backpack, adding nothing: a
		# backpack sword counted as worn either way made giving it back a gain too, 1,049 moves in a game)
		free = x["h_artifacts"][:, :HO].gather(1, other.unsqueeze(-1).expand(-1, -1, c.N_WORN_SLOTS)).gather(2, slot.clamp(min=0)) < 0
		wears = (free & (slot >= 0)).float()
		their_skills = primary.gather(1, other.unsqueeze(-1).expand(-1, -1, primary.shape[-1]))  # [B, HO, N_PRIMARY]: the other hero's
		gained = army.gather(1, other).unsqueeze(-1) * (
			lead_of(their_skills.unsqueeze(2) + bonus * wears.unsqueeze(-1), spellbook.gather(1, other).unsqueeze(-1)) - lead.gather(1, other).unsqueeze(-1)
		)
		give = (change(-bonus * worn.unsqueeze(-1).float()) + gained) * (partner >= 0).unsqueeze(-1).float()
		there = x["h_artifacts"][:, :HO].gather(2, slot.clamp(min=0)) * (slot >= 0) + -1 * (slot < 0)
		replaced = self.artifact_primary[vocab_index(there, c.ARTIFACT_VOCAB)] * (there >= 0).unsqueeze(-1).float()
		equip = on(change(bonus - replaced), present & ~worn)
		combination = self.artifact_combination[index]  # [B, HO, P]
		parts = (combination.unsqueeze(-1) == combination.unsqueeze(-2)) & (combination >= 0).unsqueeze(-1) & worn.unsqueeze(-2)  # [B, HO, P, P (the parts worn)]
		parts_price = (parts.float() * self.artifact_price[index].unsqueeze(-2)).sum(dim=-1)
		parts_bonus = (parts.float().unsqueeze(-1) * bonus.unsqueeze(-3)).sum(dim=-2)  # [B, HO, P, N_PRIMARY]
		whole = self.artifact_primary[vocab_index(combination, c.ARTIFACT_VOCAB)]
		assemble = on(self.artifact_price[vocab_index(combination, c.ARTIFACT_VOCAB)] - parts_price + change(whole - parts_bonus), combination >= 0)
		return ft.value_logit(torch.stack([give, equip, assemble], dim=-1)) - self.artifact_cost_weight()  # (a move must gain to be made)

	def resource_worth(self, x):
		"""What one unit of each resource is worth to us [B, R] (schema 12's resource_worth: a purchase short of resources
		trades away what loses the least of it, AlphaMapAI's plan): VCMI's prices, the ones its market's offers come from
		and Φ counts resources at."""
		return self.resource_price.unsqueeze(0).expand(x["g_day"].shape[0], -1)

	def recruit_fit(self, x):
		"""What buying each AMOUNT option's creatures does to the asked hero's army [B, O, RECRUIT_FIT] (the user, 2026-11-04:
		do these creatures exist in the hero, do they belong to a faction it has; 39% of the dwelling purchases had lowered
		the army's morale): the creature already in it, its faction already in it, the morale change (H3's: one faction
		+1, two 0, each more -1; undead among the living -1) / 3, a new slot taken, the slots free after / 7; 0 for an
		option that buys nothing."""
		hero = x["d_hero"].clamp(0, HO - 1).reshape(-1, 1, 1).expand(-1, 1, A)
		army = x["h_army_creature"].gather(1, hero).squeeze(1)  # [B, A]
		present = (army >= 0) & (x["h_army_count"].gather(1, hero).squeeze(1) > 0)
		creature = x["o_type"]  # [B, O]
		buys = (x["o_kind"] == int(c.OptionKind.AMOUNT)) & (x["o_amount"] > 0) & x["option_mask"]
		faction = lambda ids: self.creature_faction[vocab_index(ids, c.CREATURE_VOCAB)]  # noqa: E731
		undead = lambda ids: self.creature_undead[vocab_index(ids, c.CREATURE_VOCAB)] > 0  # noqa: E731
		has = ((army.unsqueeze(1) == creature.unsqueeze(-1)) & present.unsqueeze(1)).any(dim=-1)
		kinds = faction(army).masked_fill(~present, -2)
		has_faction = ((kinds.unsqueeze(1) == faction(creature).unsqueeze(-1)) & present.unsqueeze(1)).any(dim=-1)
		ids = torch.arange(-1, c.FACTION_VOCAB + 1, device=army.device)  # (-1: a creature of no faction)
		factions = (kinds.unsqueeze(-1) == ids).any(dim=1).sum(dim=-1, keepdim=True).float()  # [B, 1]
		after = factions + (~has_faction).float()

		def morale(n, dead, living):
			return torch.where(n == 1, 1.0, torch.where(n <= 2, 0.0, 2.0 - n)) - (dead & living).float()

		dead, living = (undead(army) & present).any(dim=-1, keepdim=True), (~undead(army) & present).any(dim=-1, keepdim=True)
		change = morale(after, dead | undead(creature), living | ~undead(creature)) - morale(factions, dead, living)
		slot = (~has).float()
		free = (A - present.sum(dim=-1, keepdim=True).float() - slot) / A
		return torch.stack([has.float(), has_faction.float(), change / 3, slot, free], dim=-1) * buys.unsqueeze(-1).float()

	def option_knowledge(self, x, f):
		"""What is known of each option of a question (the user, 2026-11-04: every expert rule its own function, gathered
		here): (its prior [B, O]: the known AI's values with their corrections, the guards' fight on the option that fights
		them and their join on the one that takes them (question_guards), Nullkiller2's YES (nk2_yes), its pick among a
		selection's components (nk2_select), its dwelling purchase (nk2_amount) and its skill scores x their weight; its
		features [B, O, QUESTION_FEATURES] for the model's zero-init weights: the fight's 6 and the join's 3 on those
		options, those two flags, and a purchase's fit with the hero's army: recruit_fit)."""
		fight, battle_prior, join, join_prior = self.question_guards(x, f)
		battle, takes = x["o_battle"].float(), x["o_join"].float()
		yes = (x["o_kind"] == int(c.OptionKind.YES)).float()
		prior = (
			battle * battle_prior.unsqueeze(1) + takes * join_prior.unsqueeze(1) + yes * self.nk2_yes(x).unsqueeze(1)
			+ self.nk2_select(x, f) + self.nk2_amount(x) + self.nk2_weights()[0] * f.skill_scores
		)
		features = torch.cat([battle.unsqueeze(-1) * fight.unsqueeze(1), takes.unsqueeze(-1) * join.unsqueeze(1), battle.unsqueeze(-1), takes.unsqueeze(-1), self.recruit_fit(x)], dim=-1)
		return prior, features

	def town_strength(self, x, power):
		"""Each town's log strength [B, T]: its garrison's, plus its fort's while it is defended (FORT_STRENGTH x a gain)."""
		fort = constant(FORT_STRENGTH, power.device)[x["t_fort"].clamp(0, len(FORT_STRENGTH) - 1)]
		return torch.log1p(power + fort * self.gain(self.fort_gain) * (power > 0))

	def town_value(self, x):
		"""What taking each town adds to Φ at least [B, T]: its village hall and its fort level's buildings at cost (by
		its faction; the rest is unseen), and the village hall's income for income_days."""
		faction = vocab_index(x["t_faction"], c.FACTION_VOCAB)
		value = self.building_value[faction, VILLAGE_HALL] + self.income_days * VILLAGE_HALL_INCOME
		for level, building in enumerate(FORT_BUILDINGS, start=1):
			value = value + self.building_value[faction, building] * (x["t_fort"] >= level)
		return value

	def map_objects(self, x):
		"""The listed map objects for the map encoder: (their worth in gold, whether we may take them, wandering monsters'
		strength), each [B, MO]."""
		obj, owner, flags = x["m_object"], x["m_owner"], x["m_flags"]
		gate = (obj >= 0) & (obj != ft.OBJ_HERO) & (obj != ft.OBJ_MONSTER) & (owner != c.OWNER_SELF) & ~ft.bit(flags, c.OBJECT_VISITED_BY_PLAYER)
		creature = x["m_creature"]
		count = ft.typical_counts(x["m_count"], ft.bit(flags, c.OBJECT_EXACT_COUNT), self) * (creature >= 0)
		return self.object_gold(obj, x["m_subtype"]), gate, count * self.fight_value[vocab_index(creature, c.CREATURE_VOCAB)]

	def travel_prior(self, x, f):
		"""TRAVEL's prior logits [B, HO, L * TG^2]: each area's value in gold (its objects by class at each class's
		worth, the land it would reveal), discounted by the days each hero needs to get there (the pathfinder's,
		travel_days; Nullkiller2's evaluateMovement), as a logit (features.value_logit)."""
		contents, hidden = ft.area_contents(x, f.object_counts)
		value = (contents * self.class_value * self.gain(self.class_gain)).sum(dim=-1) + hidden * self.explore_gold()
		days = (x["travel_days"][:, :HO].reshape(value.shape[0], HO, -1).float() / 100).clamp(min=0)  # (-1: not legal, masked)
		discount = ft.movement_discount(days, ft.bounded(self.travel_exponent, 0.7, 2.0, 1.3, CORRECTION_STEP))
		home, defend, ferry, _, _ = self.pull_weights()
		pulls = self.travel_pulls(x, f, value.shape[-1])  # [B, HO, A, 3]
		weights = torch.stack([home, defend, ferry])
		return ft.value_logit(value.unsqueeze(1) * discount) + (weights * torch.tanh(pulls * discount.unsqueeze(-1) / ft.PULL_SCALE)).sum(dim=-1)

	def travel_pulls(self, x, f, areas):
		"""Nullkiller2's plans across days in each hero's TRAVEL areas [B, HO, A, 3] (gold): a main hero's pull to the area of
		a town of ours that can reinforce it, any hero's to one under threat, a scout's with troops to a main hero's."""
		b = x["g_day"].shape[0]
		town_area = ft.area_index(x, x["t_pos"][:, :TO])  # [B, TO]
		def scatter(values, index):  # [B, N] gold at areas [B, N] (-1 none) -> [B, A]: the most at each area
			index = torch.where(index >= 0, index, torch.full_like(index, areas))
			out = values.new_zeros(b, areas + 1).scatter_reduce(1, index, values, reduce="amax", include_self=True)
			return out[:, :areas]
		reinforce = scatter(f.reinforcements, town_area)
		threatened = scatter(f.town_danger * f.town_stakes, town_area)
		main_area = torch.where(f.main, ft.area_index(x, x["h_pos"][:, :HO]), torch.full_like(f.main, -1, dtype=torch.long))
		mains = scatter(f.main.float() * 1e9, main_area).clamp(max=1)  # [B, A]: a main hero there
		scout_army = f.hero_gold[:, :HO] * (~f.main) * (x["h_present"][:, :HO] & (x["h_owner"][:, :HO] == c.OWNER_SELF))
		return torch.stack(
			[
				f.main.float().unsqueeze(-1) * reinforce.unsqueeze(1),
				threatened.unsqueeze(1).expand(b, HO, areas),
				scout_army.unsqueeze(-1) * mains.unsqueeze(1),
			],
			dim=-1,
		)

	def meeting_value(self, x, f, decision, windows, on_tile):
		"""What stepping on a hero of ours (an exchange) or into a town of ours (its garrison) gains [K, W, W] (gold): the
		best the exchange or the gather there would make to Φ (the user, 2026-11-05: a meeting's expected gain, as a
		pickup's and a fight's), at least 0, priced as the actions are (features.led_values: each army at its hero's
		multiplier, movement and make-up's morale):
		  - a hero: the other's army brought to the stronger (the higher level, else the multiplier, else the army) as far
		    as VCMI's bulk move takes it (GIVE_ALL's way, features.bulk_moves), or the two joined less their slowest
		    creatures, which the other keeps (TAKE_BEST_FAST's way), over the two apart. Before (Φ v4's factors alone, 2026-11-05 morning) a hero that had handed its slow creatures over was
		    still a meeting worth making: schema 15's first games met 308 times a game, a pair up to 6 times a day;
		  - a town, for a main hero (f.main; a scout's GATHER pays gather_cost_weight): its army joined with the garrison,
		    less the garrison at unled_share and what taking it adds to the town's expected loss (gather_loss).
		Its weight learned (meeting_gain, 1 at the start)."""
		mine = x["h_present"][:, :HO] & (x["h_owner"][:, :HO] == c.OWNER_SELF)
		hero = torch.remainder(windows, HO)  # [K]: the window's hero
		others = on_tile(x["h_pos"][:, :HO], mine) & (torch.arange(HO, device=hero.device) != hero[:, None, None, None])  # [K, W, W, HO]
		army_c, army_n = x["h_army_creature"][:, :HO], x["h_army_count"][:, :HO].float()  # [B, HO, A]
		rest = x["h_morale"][:, :HO].float() - ft.make_up_morales(army_c, army_n, self)  # (each hero's morale without its army's make-up's)
		led = ft.led_values(army_c, army_n, f.hero_multiplier, rest, self)  # [B, HO]
		k = windows.shape[0]
		flat = lambda v: v.reshape((-1,) + tuple(v.shape[2:])).index_select(0, windows)  # noqa: E731 ([B, HO, ...] -> the window's hero's [K, ...])
		mover_c, mover_n, mover_m, mover_r, mover_v = flat(army_c), flat(army_n), flat(f.hero_multiplier), flat(rest), flat(led)  # [K, A] / [K]
		other_c, other_n = army_c.index_select(0, decision), army_n.index_select(0, decision)  # [K, HO, A]
		other_m, other_r, other_v = f.hero_multiplier.index_select(0, decision), rest.index_select(0, decision), led.index_select(0, decision)  # [K, HO]
		joined_c = torch.cat([mover_c.unsqueeze(1).expand(-1, HO, -1), other_c], dim=-1)  # [K, HO, 2A]
		joined_n = torch.cat([mover_n.unsqueeze(1).expand(-1, HO, -1), other_n], dim=-1)
		mine_c, mine_n = mover_c.unsqueeze(1).expand(-1, HO, -1), mover_n.unsqueeze(1).expand(-1, HO, -1)  # [K, HO, A]
		level = x["h_level"][:, :HO].float()
		mover_l, other_l = flat(level), level.index_select(0, decision)
		mover_a, other_a = flat(f.hero_gold[:, :HO]), f.hero_gold[:, :HO].index_select(0, decision)
		mover_first = (mover_l[:, None] > other_l) | ((mover_l[:, None] == other_l) & ((mover_m[:, None] > other_m) | ((mover_m[:, None] == other_m) & (mover_a[:, None] >= other_a))))  # [K, HO]
		receiver_m = torch.where(mover_first, mover_m[:, None].expand(-1, HO), other_m)
		receiver_r = torch.where(mover_first, mover_r[:, None].expand(-1, HO), other_r)
		giver_m = torch.where(mover_first, other_m, mover_m[:, None].expand(-1, HO))
		giver_r = torch.where(mover_first, other_r, mover_r[:, None].expand(-1, HO))
		mover_first_a = mover_first.unsqueeze(-1)
		receiver_c, receiver_n = torch.where(mover_first_a, mine_c, other_c), torch.where(mover_first_a, mine_n, other_n)
		giver_c, giver_n = torch.where(mover_first_a, other_c, mine_c), torch.where(mover_first_a, other_n, mine_n)
		given = ft.bulk_moves(receiver_c, giver_c)  # (GIVE_ALL's way: what fits)
		none = torch.full_like(giver_c, -1)
		all_in = ft.led_values(torch.cat([receiver_c, torch.where(given, giver_c, none)], dim=-1), torch.cat([receiver_n, giver_n * given], dim=-1), receiver_m, receiver_r, self) + ft.led_values(
			torch.where(given, none, giver_c), giver_n * ~given, giver_m, giver_r, self
		)
		speed = self.creature_speed[vocab_index(joined_c, c.CREATURE_VOCAB)]
		present = (joined_c >= 0) & (joined_n > 0)
		slowest = torch.where(present, speed, torch.full_like(speed, 1e9)).amin(dim=-1, keepdim=True)
		slow = present & (speed == slowest) & (speed < torch.where(present, speed, torch.full_like(speed, -1.0)).amax(dim=-1, keepdim=True))  # (none when all are as slow)
		fast = ft.led_values(torch.where(slow, torch.full_like(joined_c, -1), joined_c), joined_n * ~slow, receiver_m, receiver_r, self) + ft.led_values(
			torch.where(slow, joined_c, torch.full_like(joined_c, -1)), joined_n * slow, giver_m, giver_r, self
		)
		gain = (torch.maximum(all_in, fast) - mover_v[:, None] - other_v).clamp(min=0.0)  # [K, HO]
		met = (others.float() * gain[:, None, None, :]).amax(dim=-1)
		own = x["t_present"][:, :TO] & (x["t_owner"][:, :TO] == c.OWNER_SELF)
		garrison_c, garrison_n = x["t_garrison_creature"][:, :TO].index_select(0, decision), x["t_garrison_count"][:, :TO].float().index_select(0, decision)  # [K, TO, A]
		with_garrison = ft.led_values(
			torch.cat([mover_c.unsqueeze(1).expand(-1, TO, -1), garrison_c], dim=-1), torch.cat([mover_n.unsqueeze(1).expand(-1, TO, -1), garrison_n], dim=-1), mover_m[:, None], mover_r[:, None], self
		)  # [K, TO]
		gathered = with_garrison - mover_v[:, None] - self.unled_share * f.town_gold[:, :TO].index_select(0, decision) - self.gather_loss(f).index_select(0, decision)
		garrison = gathered.clamp(min=0.0) * flat(f.main.float())[:, None]
		entered = (on_tile(x["t_pos"][:, :TO], own).float() * garrison[:, None, None, :]).amax(dim=-1)
		return (met + entered) * self.gain(self.meeting_gain)

	def tile_features(self, x, windows, f):
		"""Each window tile's features [K, W, W, ...] (K windows): its value in gold (the object we may take there + the
		expected return of the fight stepping there starts + the land it reveals) and that value as the MOVE prior's
		logit; the features for ActionScore (MOVE_FEATURES) and for the window's convolutions (CELL_FEATURES)."""
		k = windows.shape[0]
		decision = torch.div(windows, HO, rounding_mode="floor")  # [K]

		def pick(name):  # [K, W, W] of a [B, HO, W, W] input
			return x[name].reshape((-1,) + tuple(x[name].shape[2:])).index_select(0, windows)

		def ours(values):  # [K] of a [B, H] per-hero value: the window's hero
			return values[:, :HO].reshape(-1).index_select(0, windows)

		flags = pick("w_flags")
		offsets = torch.arange(W, device=flags.device) - c.WINDOW_RADIUS
		dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
		hero_pos = x["h_pos"][:, :HO].reshape(-1, 3).index_select(0, windows)  # [K, 3]
		tile_x, tile_y = hero_pos[:, 0, None, None] + dx, hero_pos[:, 1, None, None] + dy  # [K, W, W] (rows: y, columns: x)
		tile_z = hero_pos[:, 2, None, None].expand_as(tile_x)

		def on_tile(pos, present):  # [K, W, W, N]: entity n of the window's decision stands on the tile
			pos, present = pos.index_select(0, decision), present.index_select(0, decision)
			return (
				(pos[:, None, None, :, 0] == tile_x.unsqueeze(-1)) & (pos[:, None, None, :, 1] == tile_y.unsqueeze(-1))
				& (pos[:, None, None, :, 2] == tile_z.unsqueeze(-1)) & present[:, None, None, :]
			)

		def of(here, values):  # [K, W, W]: the value of the entity on the tile (0 if none)
			return (here.float() * values.index_select(0, decision)[:, None, None, :]).amax(dim=-1)

		# What taking the tile's object is worth: only an object's entrance we may take (not a hero or a wandering
		# monster, not ours, not visited by us or by this hero: VCMI's own flags, what we know)
		obj, subtype, owner = pick("w_object"), pick("w_subtype"), pick("w_owner")
		gate = (
			ft.bit(flags, c.TILE_VISITABLE) & (obj >= 0) & (obj != ft.OBJ_HERO) & (obj != ft.OBJ_MONSTER) & (owner != c.OWNER_SELF)
			& ~ft.bit(flags, c.TILE_VISITED_BY_PLAYER) & ~ft.bit(flags, c.TILE_VISITED_BY_HERO)
		)
		# Who we would fight there: wandering monsters, an enemy hero, a town not ours, a creature bank's guards
		heroes_here = on_tile(x["h_pos"], x["h_present"] & (x["h_owner"] != c.OWNER_SELF))
		towns_here = on_tile(x["t_pos"], x["t_present"] & (x["t_owner"] != c.OWNER_SELF))
		armed_here = on_tile(x["a_pos"], x["a_present"])  # (a garrison not ours: its battle starts there)
		town_value = of(towns_here, f.town_value)
		object_gold = self.object_gold(obj, subtype, town_value) * gate
		creature = pick("w_creature")
		monster = creature >= 0
		index = vocab_index(creature, c.CREATURE_VOCAB)
		count = ft.typical_counts(pick("w_count"), ft.bit(flags, c.TILE_EXACT_COUNT), self) * monster
		bank_power, bank_health = self.bank_guards(obj, subtype)
		bank = gate & (bank_power > 0)
		on_hero, on_town = heroes_here.any(dim=-1), towns_here.any(dim=-1)
		has_enemy = monster | on_hero | on_town | bank | armed_here.any(dim=-1)
		their_strength = torch.maximum(
			torch.maximum(torch.log1p(count * self.fight_value[index]) * monster, torch.log1p(bank_power) * bank),
			torch.maximum(torch.maximum(of(heroes_here, f.hero_strength), of(towns_here, f.town_strength)), of(armed_here, torch.log1p(f.armed_power))),
		)
		our_strength = ours(f.hero_strength)
		ratio = (our_strength[:, None, None] - their_strength) * has_enemy  # log of ours / theirs
		# What the fight takes from a rival: its hero's or garrison's army, and a town of its own (wandering monsters, banks
		# and neutral towns are nobody's)
		rival_towns = on_tile(x["t_pos"], x["t_present"] & (x["t_owner"] != c.OWNER_SELF) & (x["t_owner"] != c.OWNER_NONE))
		their_gold = of(heroes_here, f.hero_gold) + of(towns_here, f.town_gold) + of(rival_towns, f.town_value)
		their_health = (
			count * self.creature_health[index] + bank_health * bank + of(heroes_here, f.hero_health) + of(towns_here, f.town_health)
			+ of(armed_here, f.armed_health) + ft.HERO_XP * on_hero + ft.TOWN_XP * on_town
		)
		# The fight's expected return (gold): win it and gain its experience and the enemy's loss, minus what it costs
		sharpness = ft.bounded(self.fight_sharpness, 0.5, 8.0, 4.0, CORRECTION_STEP)
		loss, win = ft.expected_loss(ratio, has_enemy), ft.win_chance(ratio, sharpness)
		gain = their_gold * self.gain(self.enemy_value) + ft.experience_value(ours(x["h_exp"].clamp(min=0).float())[:, None, None], their_health, ours(f.level_gain)[:, None, None], self)
		# What a lost battle takes besides the army: the hero's worth (a fresh one's HERO_BASE and what it has gained since)
		# and the rest of a hire's price (a fresh hero's loss: a hire's 2,500, as before skills were priced)
		hero = ours(f.hero_worth) + (HERO_HIRE_COST - self.hero_base) * self.gain(self.hero_value)
		fight = ft.fight_return(win, loss, ours(f.hero_gold)[:, None, None], hero[:, None, None], gain) * has_enemy
		# Kill a guard, take what it blocks (Nullkiller2's ClusterBehavior): a wandering monster's fight counts part of the
		# best object in a region we reach only through its zone (features.blocked_by_guards), as often as we win it
		zone = ft.bit(flags, c.TILE_GUARDED)
		unreachable = pick("w_turns") < 0
		region = ft.bit(flags, c.TILE_ON_MAP) & ft.bit(flags, c.TILE_REVEALED) & ~ft.bit(flags, c.TILE_BLOCKED) & unreachable & ~zone
		behind = ft.blocked_by_guards(object_gold * unreachable, region, zone.float(), monster.float())
		fight = fight + self.block_share() * win * behind
		# Stepping into a wandering monster's zone of control starts its fight: the worst guard's
		guarded = ft.bit(flags, c.TILE_GUARDED) & ~has_enemy
		# The fight's log strength ratio (ours over theirs) where one starts, the tile's own enemy's or the worst guard's
		# next to a tile of a monster's zone (model_v2's pair features; joint7, 2026-11-03: heroes with less than 1,000
		# gold of army stepped into fights 54x their strength, 97% lost)
		fight_ratio = torch.where(guarded, ft.worst_guard(ratio, monster, guarded), ratio)
		fight = fight + ft.worst_guard(fight, monster, guarded)
		loss = torch.maximum(loss, -ft.worst_guard(-loss, monster, guarded))
		win = torch.where(guarded, ft.worst_guard(win, monster, guarded), win * has_enemy)
		# A guard we cannot see: VCMI flags a visible tile of the zone of a monster that stands in fog (its
		# guardingCreaturePosition checks only the tile), and worst_guard finds no monster next to it, so the fight was
		# worth 0 and the tile's exploration came on top (2026-11-05: 5 of joint7's 46 hopeless MOVE attacks on fight
		# tiles, heroes of 50-1,300 gold against 2,420-7,350, each with fog next to the tile). Its fight is the worst this
		# window's monsters show (an RMG zone's are of one strength), UNSEEN_GUARD_RATIO's where it shows none; no gain
		unseen = guarded & ~ft.next_to_monster(monster)
		unseen_ratio = ft.window_worst(ratio, monster, UNSEEN_GUARD_RATIO)
		unseen_loss, unseen_win = ft.expected_loss(unseen_ratio, unseen), ft.win_chance(unseen_ratio, sharpness)
		fight_ratio = torch.where(unseen, unseen_ratio, fight_ratio)
		fight = torch.where(unseen, ft.fight_return(unseen_win, unseen_loss, ours(f.hero_gold)[:, None, None], hero[:, None, None], 0.0), fight)
		loss, win = torch.where(unseen, unseen_loss, loss), torch.where(unseen, unseen_win, win)
		fights = has_enemy | guarded
		# The fight in four estimates of strength (the user, 2026-11-04: several that agree are easier to act on), each a
		# log ratio ours over theirs, for this hero and for our main army (our strongest hero), where a fight starts (the
		# tile's own enemy's, or the worst guard's next to a tile of a monster's zone):
		#   army    count x fight value (Nullkiller2's getArmyStrength)
		#   total   that and the hero's attack and defense (Nullkiller2's getTotalStrength): `ratio`'s
		#   race    a damage race (features.race_strength): each side's hit points x its damage a round at its attack
		#           against the other's defense
		#   spells  the race with our best damage spell a round (features.spell_damage)
		# A creature bank's guards are known only by strength and hit points: their race is the army estimate
		monster_damage = count * self.creature_damage[index]
		monster_health = count * self.creature_health[index]
		their_army = torch.maximum(
			torch.maximum(torch.log1p(count * self.fight_value[index]) * monster, torch.log1p(bank_power) * bank),
			torch.maximum(torch.maximum(of(heroes_here, torch.log1p(f.hero_power)), of(towns_here, torch.log1p(f.town_power))), of(armed_here, torch.log1p(f.armed_power))),
		)
		their_hp = monster_health + of(heroes_here, f.hero_health) + of(towns_here, f.town_health) + of(armed_here, f.armed_health)
		their_damage = monster_damage + of(heroes_here, f.hero_damage) + of(towns_here, f.town_damage) + of(armed_here, f.armed_damage)
		their_attack = (
			monster_damage * self.creature_attack[index] + of(heroes_here, f.hero_damage * f.hero_attack) + of(towns_here, f.town_damage * f.town_attack) + of(armed_here, f.armed_damage * f.armed_attack)
		) / their_damage.clamp(min=1e-6)
		their_defense = (
			monster_health * self.creature_defense[index] + of(heroes_here, f.hero_health * f.hero_defense) + of(towns_here, f.town_health * f.town_defense) + of(armed_here, f.armed_health * f.armed_defense)
		) / their_hp.clamp(min=1e-6)
		own_slot = x["h_present"][:, :HO] & (x["h_owner"][:, :HO] == c.OWNER_SELF)
		strongest = torch.where(own_slot, f.hero_strength[:, :HO], torch.full_like(f.hero_strength[:, :HO], -1e9)).argmax(dim=1)  # [B]

		def of_main(values):  # [K] of a [B, H] per-hero value: our strongest hero's, for each window's decision
			return values[:, :HO].gather(1, strongest.unsqueeze(1)).squeeze(1).index_select(0, decision)

		def estimates(pick_hero):  # pick_hero: ours or of_main -> [K, W, W, 4] (army, total, race, spells)
			side = lambda values: pick_hero(values)[:, None, None]  # noqa: E731
			army = side(torch.log1p(f.hero_power)) - their_army
			race = ft.race_strength(side(f.hero_health), side(f.hero_damage), side(f.hero_attack), side(f.hero_defense), their_hp, their_damage, their_attack, their_defense)
			spells = ft.race_strength(side(f.hero_health), side(f.hero_damage), side(f.hero_attack), side(f.hero_defense), their_hp, their_damage, their_attack, their_defense, side(f.hero_spells))
			out = []
			for r in (army, side(f.hero_strength) - their_strength, torch.where(bank, army, race), torch.where(bank, army, spells)):
				r = r * has_enemy
				guard = torch.where(guarded, ft.worst_guard(r, monster, guarded), r)
				out.append(torch.where(unseen, ft.window_worst(r, monster, UNSEEN_GUARD_RATIO), guard))
			return torch.stack(out, dim=-1)

		this_hero, main_army = estimates(ours), estimates(of_main)
		# How hopeless the fight is by the kindest of this hero's estimates (features.hopeless: smooth; a guard 54x our
		# strength ~0.92)
		hopeless = ft.hopeless(this_hero.amax(dim=-1)) * fights
		explore = ft.unexplored_around(flags)
		# The object is ours only if we win the fight in front of it (a bank's guards, a monster's zone, a town's garrison)
		value = object_gold * torch.where(fights, win, torch.ones_like(win)) + fight + explore * self.explore_gold() + self.meeting_value(x, f, decision, windows, on_tile)
		# That gain discounted by the time to reach the tile as Nullkiller2 discounts its targets (PriorityEvaluator::
		# evaluateMovement, as travel_prior does TRAVEL's areas): by the part of a day's movement it costs, raised to
		# move_discount_power, 0 at the start (the score as before) and 1 its full discount. Nullkiller2 scores every
		# target, near or far, as value by cost, so a pickup on the way outranks a far goal (the user, 2026-11-02: value
		# per movement cost as an op on what the model has, the old score while it is unused). The danger below stays
		# where the move ends, undiscounted
		left_now = pick("w_move_left")
		days = torch.where(
			left_now >= 0,
			(ours(x["h_move"]).float()[:, None, None] - left_now.float()).clamp(min=0) / ours(x["h_move_max"]).float().clamp(min=1)[:, None, None],
			pick("w_turns").clamp(min=1).float(),
		)
		discount = ft.movement_discount(days, ft.bounded(self.travel_exponent, 0.7, 2.0, 1.3, CORRECTION_STEP))
		value = value * discount ** self.move_discount_power()
		# Danger: the enemy heroes we see, their reach to the tile x P(they beat this hero); what ending the move there
		# instead of where the hero stands changes in the expected loss (army, hero, its replacement: hero); our towns'
		# danger on them
		enemies = tuple(t.index_select(0, decision) for t in ft.enemies(x, f.hero_strength))
		near, theirs = ft.reach(enemies, tile_x, tile_y, tile_z)  # [K, W, W, E]
		danger = (near * torch.sigmoid(2 * (theirs - our_strength[:, None, None, None]))).amax(dim=-1)
		here = danger[:, c.WINDOW_RADIUS, c.WINDOW_RADIUS]
		at_stake = (ours(f.hero_gold) + hero) * self.gain(self.danger_gain)
		value = value + (here[:, None, None] - danger) * at_stake[:, None, None]
		own_towns = x["t_present"][:, :TO] & (x["t_owner"][:, :TO] == c.OWNER_SELF)
		defend = (on_tile(x["t_pos"][:, :TO], own_towns).float() * f.town_danger.index_select(0, decision)[:, None, None, :]).amax(dim=-1)
		move_left = pick("w_move_left")
		left = (move_left.clamp(min=0).float() / ours(x["h_move_max"]).clamp(min=1).float()[:, None, None]).clamp(max=1) * (move_left >= 0)
		# Nullkiller2's plans across days, as weak pulls in logits (pull_weights): a main hero toward a town of ours that can
		# reinforce it, any hero toward one under threat, a scout with troops toward a main hero
		own = x["t_present"][:, :TO] & (x["t_owner"][:, :TO] == c.OWNER_SELF)
		ours_here, mains_here = on_tile(x["t_pos"][:, :TO], own), on_tile(x["h_pos"][:, :HO], f.main)
		main = ours(f.main.float())  # the window's hero: a main hero?
		pulls = torch.tanh(
			torch.stack(
				[of(ours_here, f.reinforcements) * main[:, None, None], of(ours_here, f.town_danger * f.town_stakes), mains_here.any(dim=-1).float() * (ours(f.hero_gold) * (1 - main))[:, None, None]],
				dim=-1,
			) / ft.PULL_SCALE
		)  # [K, W, W, 3]
		home_weight, defend_weight, ferry_weight, _, _ = self.pull_weights()
		prior = ft.value_logit(value) + (torch.stack([home_weight, defend_weight, ferry_weight]) * pulls).sum(dim=-1) - self.safety_weight() * hopeless
		# model_v2's MOVE features beside MOVE_FEATURES, relative so they keep their meaning on any map and day (the user,
		# 2026-11-04): hopeless (above); the object as a share of our Φ (log(1 + 100 x share) / 3: 1% 0.23, 10% 0.8); where
		# a fight starts, what losing it takes (the army, the hero, its replacement) and what winning it takes (the enemy's
		# loss and the experience, the object there and what a guard blocks), each a share of Φ; our town here, its danger x
		# its stakes as a share of Φ; and what this hero can take into its army here (a town's reinforcements over its own
		# army; a scout's army over the main hero's it would join)
		phi = f.phi_total.clamp(min=1000.0).index_select(0, decision)[:, None, None]
		share = lambda gold: torch.log1p(100.0 * gold.clamp(min=0) / phi) / 3  # noqa: E731
		stake = (ours(f.hero_gold) + hero)[:, None, None] * fights
		taken = (torch.where(guarded, ft.worst_guard(gain, monster, guarded), gain * has_enemy) + object_gold + behind) * fights
		reinforce = torch.log1p(of(ours_here, f.reinforcements) / (ours(f.hero_gold)[:, None, None] + 500.0)) / 3
		ferry = torch.log1p(ours(f.hero_gold)[:, None, None] * (1 - main)[:, None, None] / (of(mains_here, f.hero_gold[:, :HO]) + 500.0)) / 3 * mains_here.any(dim=-1)
		relative = torch.stack(
			[hopeless, share(object_gold), share(stake), share(taken), share(of(ours_here, f.town_danger * f.town_stakes)), torch.maximum(reinforce, ferry)], dim=-1
		)  # [K, W, W, MOVE_RELATIVE]
		move = torch.cat(
			[
				torch.stack(
					[
						torch.tanh(prior / 4), torch.tanh(ft.value_logit(object_gold) / 4), torch.tanh(ft.value_logit(fight) / 4),
						loss, win, explore, danger, defend, left, fights.float(), torch.tanh(ft.value_logit(behind) / 4),
					],
					dim=-1,
				),
				pulls,
			],
			dim=-1,
		)  # MOVE_FEATURES
		turns = pick("w_turns")
		bits = torch.stack([torch.remainder(torch.div(flags, 1 << i, rounding_mode="floor"), 2) for i in range(c.N_TILE_FLAGS)], dim=-1).float()
		cells = torch.cat(
			[
				bits,
				scales(count, (5, 20, 80)),
				(move_left >= 0).float().unsqueeze(-1),
				left.unsqueeze(-1),
				torch.stack([turns == 0, turns == 1, turns >= 2, turns < 0], dim=-1).float(),
				(torch.stack([dx, dy, torch.maximum(dx.abs(), dy.abs())], dim=-1).float() / c.WINDOW_RADIUS).expand(k, W, W, 3),
				(ratio / 5).unsqueeze(-1),
				move,
			],
			dim=-1,
		)  # CELL_FEATURES
		return {
			"tile_value": value, "tile_prior": prior, "move_features": move, "cell_features": cells, "tile_object_gold": object_gold, "tile_fight_gold": fight, "tile_fight_ratio": fight_ratio, "tile_fights": fights.float(),
			"tile_stacks": (monster | guarded).float(),  # (a wandering monster or its zone: model_v2's battle head)
			"move_strength": ft.strength_view(torch.cat([this_hero, main_army], dim=-1)), "move_relative": relative,  # (model_v2: MOVE_STRENGTH, MOVE_RELATIVE)
		}


class MapEncoder(nn.Module):
	"""The whole map as a minimap: MAP_GRID x MAP_GRID blocks a level (a block features.block_side tiles a side, its size
	an input), each with its tile counts (g_tiles), its objects' counts by class (features.object_counts) and the objects
	themselves (m_*: every one we see) scattered onto it: each block's mean object and its most important one (a
	learned importance). Only the objects there are embedded: a small map's few, an XL map's thousands. A convolution pyramid reads it down to 18 x 18 (TRAVEL's areas) and 9 x 9
	blocks. Returns the coarse blocks' features [B, L * COARSE^2, dm], which are on the map, and the areas' features
	[B, L * TG^2, area_dim] (the 18 x 18 level's with its 9 x 9 block's)."""

	def __init__(self, dim, object_dim=32):
		super().__init__()
		emb = small_embedding
		self.object = emb(c.OBJECT_VOCAB + 1, object_dim)
		self.subtype = emb(c.SUBTYPE_ROWS, object_dim)
		self.owner = emb(c.OWNER_VOCAB + 1, object_dim)
		self.creature = emb(c.CREATURE_VOCAB + 1, object_dim)
		self.object_in = nn.Linear(OBJECT_FEATURES, object_dim)
		self.importance = nn.Linear(object_dim, 1)
		self.level = nn.Parameter(torch.zeros(L, 16))
		self.block_in = nn.Conv2d(BLOCK_FEATURES + 2 * object_dim + 1, 16, 1)
		self.down1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  # 36 -> 18
		self.conv1 = nn.Conv2d(32, 32, 3, padding=1)
		self.down2 = nn.Conv2d(32, dim, 3, stride=2, padding=1)  # 18 -> 9
		self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
		self.area_dim = 32 + dim

	def objects(self, x, f, rows):
		"""The vectors [P, object_dim] of the objects there (rows: their places in the flattened [B * MO] list)."""
		pick = lambda t: t.reshape((-1,) + tuple(t.shape[2:]))[rows]  # noqa: E731
		obj, owner, flags = pick(x["m_object"]), pick(x["m_owner"]), pick(x["m_flags"])
		numbers = torch.cat(
			[
				torch.stack(
					[
						ft.bit(flags, c.OBJECT_VISITED_BY_PLAYER).float(), ft.bit(flags, c.OBJECT_GUARDED).float(), ft.bit(flags, c.OBJECT_EXACT_COUNT).float(),
						(pick(x["m_visitors"]) != 0).float(), (owner == c.OWNER_SELF).float(), (owner > c.OWNER_SELF).float(), pick(f.map_gate).float(),
					],
					dim=-1,
				),
				scales(pick(f.map_monster_power), (1e3, 1e4, 1e5)),
				torch.tanh(ft.value_logit(pick(f.map_worth) * pick(f.map_gate)) / 4).unsqueeze(-1),
			],
			dim=-1,
		)
		return (
			self.object(vocab_index(obj, c.OBJECT_VOCAB)) + self.subtype(ft.bucket(obj, obj, pick(x["m_subtype"])))
			+ self.owner(vocab_index(owner, c.OWNER_VOCAB)) + self.creature(vocab_index(pick(x["m_creature"]), c.CREATURE_VOCAB)) + self.object_in(numbers)
		)

	def forward(self, x, f):
		b = x["g_day"].shape[0]
		present = x["m_object"] >= 0
		rows = torch.nonzero(present.reshape(-1)).squeeze(-1)  # the objects there: only they are embedded
		flat = self.objects(x, f, rows)
		# Each object's block: (decision, level, row, column)
		side = ft.block_side(x)
		pos = x["m_pos"].clamp(min=0)
		row = torch.div(pos[..., 1], side[:, None], rounding_mode="floor").clamp(max=G - 1)
		column = torch.div(pos[..., 0], side[:, None], rounding_mode="floor").clamp(max=G - 1)
		batch = torch.arange(b, device=pos.device)[:, None]
		cells = b * L * G * G
		index = (((batch * L + pos[..., 2].clamp(max=L - 1)) * G + row) * G + column).reshape(-1)[rows]
		weight = torch.exp(self.importance(flat).clamp(-10, 10))  # the block's most important objects count most
		zeros = flat.new_zeros(cells + 1, flat.shape[-1])
		total = zeros.index_add(0, index, flat)[:cells]
		count = flat.new_zeros(cells + 1, 1).index_add(0, index, torch.ones_like(weight))[:cells]
		weighted = zeros.index_add(0, index, flat * weight)[:cells]
		weights = flat.new_zeros(cells + 1, 1).index_add(0, index, weight)[:cells]
		tiles = x["g_tiles"].float().reshape(cells, c.N_GRID_COUNTS)
		on_map = tiles[:, c.GridCount.ON_MAP : c.GridCount.ON_MAP + 1]
		revealed = tiles[:, c.GridCount.REVEALED : c.GridCount.REVEALED + 1]
		block = side.float().repeat_interleave(L * G * G)[:, None]  # tiles a side of each block
		numbers = torch.cat(
			[
				on_map / block**2,
				revealed / on_map.clamp(min=1),
				tiles[:, c.GridCount.REVEALED + 1 :] / revealed.clamp(min=1),  # water, blocked, road: of the revealed
				scales(f.object_counts.reshape(cells, c.N_OBJECT_CLASSES), (1, 4)).flatten(-2),  # one? several?
				scales(block.squeeze(-1), (2, 8)),  # the map's scale: 1 tile a block on 36 x 36 maps, 4 on 144 x 144
			],
			dim=-1,
		)
		grid = torch.cat([numbers, total / count.clamp(min=1), weighted / weights.clamp(min=1e-6), torch.log1p(count) / 3], dim=-1)
		grid = grid.reshape(b * L, G, G, -1).permute(0, 3, 1, 2)
		hidden = F.gelu(self.block_in(grid) + self.level.repeat(b, 1)[:, :, None, None])
		hidden = F.gelu(self.down1(hidden))
		middle = hidden + self.conv1(F.gelu(hidden))  # [B * L, 32, TG, TG]
		hidden = F.gelu(self.down2(middle))
		hidden = hidden + self.conv2(F.gelu(hidden))  # [B * L, dm, COARSE, COARSE]
		valid = F.max_pool2d((on_map > 0).float().reshape(b * L, 1, G, G), G // COARSE).reshape(b, L * COARSE * COARSE) > 0
		blocks = hidden.reshape(b, L, -1, COARSE * COARSE).permute(0, 1, 3, 2).reshape(b, L * COARSE * COARSE, -1)
		areas = torch.cat([middle, F.interpolate(hidden, scale_factor=2, mode="nearest")], dim=1)  # what each area is part of
		areas = areas.reshape(b, L, self.area_dim, -1).permute(0, 1, 3, 2).reshape(b, L * TG * TG, self.area_dim)
		return blocks, valid, areas


class MapReader(nn.Module):
	"""Each of our heroes reads the map's coarse blocks by what they hold and where they lie from it: attention with a
	bias and a value term by direction and distance (features.relative_position)."""

	def __init__(self, dim, map_dim, heads=4):
		super().__init__()
		self.heads = heads
		self.q = nn.Linear(dim, dim)
		self.k = nn.Linear(map_dim, dim)
		self.v = nn.Linear(map_dim, dim)
		self.position_bias = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, heads))
		self.position_value = nn.Linear(8, dim)
		self.out = zero_output(nn.Linear(dim, dim))

	def forward(self, heroes, positions, blocks, valid, centres):
		"""heroes [B, N, d], positions [B, N, 3] -> [B, N, d] (0 at the start: out begins at 0)."""
		b, n, d = heroes.shape
		heads, dk = self.heads, d // self.heads
		relative = ft.relative_position(centres[:, None, :, :] - positions[:, :, None, :].float())  # [B, N, C, 8]
		q = self.q(heroes).reshape(b, n, heads, dk)
		k = self.k(blocks).reshape(b, -1, heads, dk)
		logits = torch.einsum("bnhd,bchd->bnhc", q, k) / math.sqrt(dk) + self.position_bias(relative).permute(0, 1, 3, 2)
		logits = logits.masked_fill(~valid[:, None, None, :], -1e4).softmax(dim=-1)  # [B, N, H, C]
		values = self.v(blocks)[:, None, :, :] + self.position_value(relative)  # [B, N, C, d]
		read = torch.einsum("bnhc,bnchd->bnhd", logits, values.reshape(b, n, -1, heads, dk)).reshape(b, n, d)
		return self.out(read)


class TravelScore(nn.Module):
	"""TRAVEL's learned logits [B, N, A] (its prior is CustomFeatures.travel_prior): for each hero and area, how its
	token matches the area's map features (a pointer), where the area lies from it (direction and distance in tiles)
	and what the area holds by itself."""

	def __init__(self, dim, area_dim):
		super().__init__()
		self.query = nn.Linear(dim, dim)
		self.key = nn.Linear(area_dim, dim)
		self.position = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 1))
		self.worth = nn.Linear(area_dim, 1)

	def forward(self, heroes, positions, areas, centres):
		relative = ft.relative_position(centres[:, None, :, :] - positions[:, :, None, :].float())  # [B, N, A, 8]
		pointer = torch.einsum("bnd,bad->bna", self.query(heroes), self.key(areas)) / math.sqrt(heroes.shape[-1])
		return pointer + self.position(relative).squeeze(-1) + self.worth(areas).squeeze(-1).unsqueeze(1)


class PairBias(nn.Module):
	"""An attention bias [B, heads, N, N] between tokens with a place on the map (heroes, towns, an option's object), by
	their offset in tiles (features.relative_position): which hero is near which town, whatever the map's size. It
	starts at 0."""

	def __init__(self, heads):
		super().__init__()
		self.net = nn.Sequential(nn.Linear(9, 16), nn.GELU(), zero_output(nn.Linear(16, heads)))

	def forward(self, positions, placed):
		both = (placed[:, :, None] & placed[:, None, :]).float().unsqueeze(-1)  # [B, N, N, 1]: query i, key j
		features = torch.cat([ft.relative_position(positions[:, None, :, :] - positions[:, :, None, :]), torch.ones_like(both)], dim=-1) * both
		return self.net(features).permute(0, 3, 1, 2)


# Parameters added since checkpoints were saved, each at 0 the model as before it existed: a checkpoint without them
# loads with them at 0 (with_added; 2026-11-02: features.move_discount, CustomFeatures.move_discount_power)
ADDED = ("features.move_discount", "features.safety", "features.trade_unlock", "features.trade_cost", "features.recruit_yes", "features.visit_yes", "features.select_pick", "features.artifact_cost", "features.meeting_gain", "features.multi_weight", "features.give_all_cost", "features.gather_risk", "features.gather_cost", "features.disband_cost", "features.drop_scale", "features.drop_sign", "features.trade_waste")  # (2026-11-04: at 0, their starts: safety 6; TRADE's unlock 2, cost 0.5; YES at a dwelling, a guard, a quest 4; a selection's pick 3)
# Parts added since, by prefix: a checkpoint without them loads them as a new model starts them, heads nothing the AI
# reads depends on (their losses only: 2026-11-02, foresight.plan*, model_v2.plan_of) or layers whose output starts at 0
# (2026-11-03: model_v2's pair_fight and move_fight, the fight features; 2026-11-04: its strength estimates' pair_strength,
# the mixer (move_mix*: its direct path as the fresh model has it, Nullkiller2's rule) and the foresight's battle heads)
ADDED_HEADS = (  # (+ trade_option: TRADE's options' cost and unlock; option_flags: a question's battle and join, at 0; the question's
	# decision state, its keys' context at 0, the options' known features' weights at 0 and the components' own rows:
	# model_v2.migrate_option_head)
	"foresight.plan", "pair_fight", "move_fight", "pair_strength", "move_mix", "foresight.tiles", "foresight.outcome", "trade_option", "option_flags",
	"decision", "option_context", "option_weights", "encode_options.component", "encode_options.primary", "encode_options.resource",
	"encode_options.numbers", "build_identity", "artifact_q", "artifact_target", "encode_hero.carried", "multi_q", "multi_kind",
	"privileged",  # (2026-11-06: the privileged critic, its corrections 0 at the start)
)


def with_added(state_dict, own):
	"""state_dict (a checkpoint's) with the ADDED parameters it lacks at 0 and the ADDED_HEADS' it lacks as the loading
	model has them; own: the loading model's state_dict."""
	missing = {key: torch.zeros_like(own[key]) for key in ADDED if key in own and key not in state_dict}
	missing.update({key: value.clone() for key, value in own.items() if key.startswith(ADDED_HEADS) and key not in state_dict})
	return {**state_dict, **missing} if missing else state_dict


class CampaignModel(nn.Module):
	def __init__(self, dim=64, cell_dim=32, heads=4, layers=2):
		super().__init__()
		d, dc = dim, cell_dim
		self.dim, self.cell_dim = d, dc
		self.features = CustomFeatures()
		emb = small_embedding
		self.creature = emb(c.CREATURE_VOCAB + 1, d)
		self.hero_type = emb(c.HERO_TYPE_VOCAB + 1, d)
		self.hero_class = emb(c.HERO_CLASS_VOCAB + 1, d)
		self.artifact = emb(c.ARTIFACT_VOCAB + 1, d)
		self.faction = emb(c.FACTION_VOCAB + 1, d)
		self.owner = emb(c.OWNER_VOCAB + 1, d)
		self.kind = emb(c.N_KINDS + 1, d)
		self.object = emb(c.OBJECT_VOCAB + 1, d)
		self.subtype = emb(c.SUBTYPE_ROWS, d)
		self.sound = emb(c.SOUND_VOCAB + 1, d)
		self.option_kind = emb(len(c.OptionKind) + 1, d)
		self.global_in = nn.Linear(GLOBAL_FEATURES, d)
		self.phi_in = nn.Linear(PHI_FEATURES, d)
		self.hero_in = nn.Linear(HERO_FEATURES, d)
		self.stack_in = nn.Linear(d + 6, d)
		self.skills_in = nn.Linear(c.N_SECONDARY_SKILLS, d)
		self.spells_in = nn.Linear(c.N_SPELLS, d)
		self.town_in = nn.Linear(TOWN_FEATURES, d)
		self.built_in = nn.Linear(c.N_BUILDINGS, d)
		self.level_in = nn.Linear(d + 7, d)
		self.option_in = nn.Linear(OPTION_FEATURES, d)
		# Our previous decision today (g_last_*), from 0: its type and whether it changed anything, the hero and town it
		# was about (where a chain of actions stands, the model's to read: 2026-10-28)
		self.last_in = zero_output(nn.Linear(c.N_ACTION_TYPES + 1 + 3, d))
		self.last_hero_in, self.last_town_in = zero_output(nn.Linear(1, d)), zero_output(nn.Linear(1, d))
		# Nullkiller2's brain as features (features.py), from 0: gold pressure and its pause; a hero's role; a building's
		# rewards and cost; an upgrade's gain and cost; a level-up option's skill score
		self.nk2_global, self.nk2_hero, self.nk2_option = zero_output(nn.Linear(3, d)), zero_output(nn.Linear(1, d)), zero_output(nn.Linear(1, d))
		self.nk2_build, self.nk2_upgrade, self.nk2_trade = zero_output(nn.Linear(4, 1)), zero_output(nn.Linear(2, 1)), zero_output(nn.Linear(2, 1))
		self.other_token = nn.Parameter(torch.zeros(d))
		# Tiles
		self.cell_terrain = emb(c.TERRAIN_VOCAB + 1, dc)
		self.cell_road = emb(c.ROAD_VOCAB + 1, dc)
		self.cell_object = emb(c.OBJECT_VOCAB + 1, dc)
		self.cell_subtype = emb(c.SUBTYPE_ROWS, dc)
		self.cell_owner = emb(c.OWNER_VOCAB + 1, dc)
		self.cell_creature = emb(c.CREATURE_VOCAB + 1, dc)
		self.cell_in = nn.Linear(CELL_FEATURES, dc)
		self.conv1 = nn.Conv2d(dc, dc, 3, padding=1)
		self.conv2 = nn.Conv2d(dc, dc, 3, padding=1)
		self.cell_pool = nn.Linear(2 * dc, d)
		# The whole map
		self.map = MapEncoder(d)
		self.map_reader = MapReader(d, d, heads)
		self.map_pool = nn.Linear(2 * d, d)
		self.pair_bias = PairBias(heads)
		self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
		self.norm = nn.LayerNorm(d)
		# Heads
		self.type_head = zero_output(nn.Linear(d, c.N_ACTION_TYPES))
		with torch.no_grad():  # its bias starts at the priors (priors/action_types.csv), learned from there
			self.type_head.bias.copy_(priors.action_types())
		self.type_summary = zero_output(nn.Linear(TYPE_SUMMARY, c.N_ACTION_TYPES))  # what MOVE and TRAVEL offer, learned from 0
		self.film = nn.Linear(d, 2 * dc)
		self.move_out = nn.Linear(dc, 1)
		self.move_score = ActionScore(MOVE_FEATURES, d)
		self.travel = TravelScore(d, self.map.area_dim)
		self.cast_cell_out = nn.Linear(dc, 1)
		self.building = emb(c.N_BUILDINGS, d)
		self.build_q = nn.Linear(d, d)
		self.build_bias = nn.Parameter(torch.zeros(c.N_BUILDINGS))
		self.recruit_head = nn.Sequential(nn.Linear(2 * d + 9, d), nn.GELU(), nn.Linear(d, c.N_RECRUIT_AMOUNTS * c.N_RECRUIT_TARGETS))
		self.hire_head = nn.Sequential(nn.Linear(2 * d + 2, d), nn.GELU(), nn.Linear(d, 1))
		self.hire_worth = zero_output(nn.Linear(HIRE_FEATURES, 1))  # what a hire brings, learned from 0 (schema 7)
		self.army_src = nn.Linear(2 * d, d)
		self.army_dst = nn.Linear(d, d)
		self.army_out = nn.Sequential(nn.GELU(), nn.Linear(d, c.N_ARMY_AMOUNTS))
		self.upgrade_out = nn.Sequential(nn.GELU(), nn.Linear(d, 1))
		self.trade_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, c.N_RESOURCES * c.N_RESOURCES * c.N_TRADE_AMOUNTS))
		self.spell = emb(c.N_ADVENTURE_SPELLS, d)
		self.cast_q = nn.Linear(d, d)
		self.buy_item = emb(c.N_BUY_ITEMS, d)
		self.buy_q = nn.Linear(d, d)
		self.dismiss_out = nn.Linear(d, 1)
		self.option_q = nn.Linear(2 * d, d)
		self.value_hidden = nn.Sequential(nn.Linear(2 * d + PHI_FEATURES, d), nn.GELU())
		self.value_heads = zero_output(nn.Linear(d, c.N_VALUE_OFFSETS))  # V2, V5, V7: a readout each, none chained
		self.growth_heads = zero_output(nn.Linear(d, c.N_VALUE_OFFSETS))  # our own growth to the same days (helpers)
		self.end_head = zero_output(nn.Linear(d, 1))  # V_end: the share at the game's end (labels.end_labels)

	def load_state_dict(self, state_dict, strict=True, assign=False):
		"""A checkpoint from before schema 7 (2026-10-29, no hire_worth) is migrated: the hire table's new part starts
		at 0 (its prior: what a hire does to Φ), and the type head's learned part goes back to its priors. It was learned
		in games whose towns had neither a tavern nor a dwelling (vcmi 47c9d57eb: random maps lost their towns' default
		buildings), where building had to come first, and had drifted to BUILD +4.3 and HIRE -2.3 logits over 290 updates:
		left so, it would hold every hire back however much the hire brought. A parameter added since (ADDED) starts at 0."""
		own = self.state_dict()
		state_dict = with_added(state_dict, own)
		missing = [key for key in own if key.startswith("hire_worth.") and key not in state_dict]
		if missing:
			state_dict = dict(state_dict)
			for key in missing:
				state_dict[key] = own[key]
			state_dict["type_head.weight"] = torch.zeros_like(own["type_head.weight"])
			state_dict["type_head.bias"] = priors.action_types().to(own["type_head.bias"])
		return super().load_state_dict(state_dict, strict=strict, assign=assign)

	def prior_corrections(self):
		"""The parameters that correct a fixed prior: the optimizer's weight decay pulls them back to 0 (to the prior)."""
		return self.features.corrections()

	# ---- the whole model ----

	def forward(self, x):
		b = x["g_day"].shape[0]
		windows, n = self.windows(x)  # our heroes' windows to compute
		f = self.features(x, windows)  # make_custom_features: every hand-made feature
		hero_tokens, hero_stacks = self.hero_tokens(x, f)
		town_tokens, town_stacks, town_levels = self.town_tokens(x, f)
		cells = None
		if n > 0:  # each hero's window: convolutions over its tiles, summarised into its token
			cells = self.window_cells(x, f, windows)
			hero_tokens = torch.cat([hero_tokens[:, :HO] + dense(self.window_summary(x, cells, windows), windows, n, b), hero_tokens[:, HO:]], dim=1)
		blocks, block_valid, areas = self.map(x, f)  # the whole map: each hero reads it, the global token gets its summary
		read = self.map_reader(hero_tokens[:, :HO], x["h_pos"][:, :HO], blocks, block_valid, ft.centres(x, COARSE)) * x["h_present"][:, :HO].unsqueeze(-1)
		hero_tokens = torch.cat([hero_tokens[:, :HO] + read, hero_tokens[:, HO:]], dim=1)
		tokens, valid, other_stacks = self.all_tokens(x, f, hero_tokens, town_tokens, blocks, block_valid)
		bias = self.pair_bias(*self.token_places(x))
		for block in self.blocks:
			tokens = block(tokens, valid, bias)
		tokens = self.norm(tokens)
		return self.outputs(x, f, tokens, cells, areas, windows, n, hero_stacks, town_stacks, town_levels, other_stacks)

	# ---- tokens ----

	def stacks(self, creatures, counts):
		"""Stack embeddings [..., A, d] from creatures and their counts as seen (features.army's)."""
		present = creatures >= 0
		index = vocab_index(creatures, c.CREATURE_VOCAB)
		power = counts * self.features.fight_value[index]
		return self.stack_in(torch.cat([self.creature(index), scales(counts, (5, 20, 80)), scales(power, (1e3, 1e4, 1e5))], dim=-1)) * present.unsqueeze(-1)

	def global_token(self, x, f):
		day = x["g_day"].clamp(min=1)
		price = self.features.resource_price
		size = x["g_map"].clamp(min=1).float()
		explored = x["g_explored"].clamp(min=0).float()
		numbers = torch.cat(
			[
				ft.one_hot(torch.remainder(day - 1, 7), 7),
				ft.one_hot(torch.remainder(torch.div(day - 1, 7, rounding_mode="floor"), 4), 4),
				scales(day, (7, 28, 112)),
				scales(x["g_resources"].clamp(min=0) * price / 1000, (1, 5, 20)).flatten(-2),  # each resource's worth, in thousands
				scales(x["g_income"].clamp(min=0) * price / 1000, (0.5, 2, 8)).flatten(-2),
				size / 144,
				x["g_players"].float().unsqueeze(-1) / c.MAX_PLAYERS,
				scales(x["g_decisions_today"], (4, 16, 64)),
				(explored / (size[:, 0] * size[:, 1] * size[:, 2])).unsqueeze(-1),
				scales(explored, (500, 2000, 8000)),
				scales(x["g_markets"], (1, 3)),
				scales(x["g_tavern_level"], (2, 8)).flatten(-2),
				scales(ft.typical_counts(x["d_count"], x["d_detailed"], self.features) * (x["d_creature"] >= 0), (5, 20, 80)),
				scales(x["d_number"], (500, 2000, 8000)),
				(x["d_hero"] >= 0).float().unsqueeze(-1),
			],
			dim=-1,
		)
		tavern = self.hero_type(vocab_index(x["g_tavern"], c.HERO_TYPE_VOCAB)) * (x["g_tavern"] >= 0).unsqueeze(-1)
		d_creature = self.creature(vocab_index(x["d_creature"], c.CREATURE_VOCAB)) * (x["d_creature"] >= 0).unsqueeze(-1)
		return (
			self.global_in(numbers) + self.phi_in(f.phi) + self.kind(vocab_index(x["g_kind"], c.N_KINDS)) + self.sound(vocab_index(x["d_sound"], c.SOUND_VOCAB))
			+ self.object(vocab_index(x["d_object"], c.OBJECT_VOCAB)) + self.subtype(ft.bucket(x["d_object"], x["d_object"], x["d_subtype"]))
			+ d_creature + tavern.sum(dim=1)
			+ self.last_in(torch.cat([ft.one_hot(x["g_last_type"].clamp(-1, c.N_ACTION_TYPES - 1) + 1, c.N_ACTION_TYPES + 1), ft.one_hot(x["g_last_changed"].clamp(-1, 1) + 1, 3)], dim=-1))
			+ self.nk2_global(torch.stack([f.gold_pressure / (1 + f.gold_pressure), f.recruit_pause, f.hire_want], dim=-1))
		)

	def hero_tokens(self, x, f):
		"""Hero tokens [B, H, d] and their stacks [B, H, A, d]."""
		stacks = self.stacks(x["h_army_creature"], f.hero_counts)
		size = x["g_map"].float().clamp(min=1)
		primary = x["h_primary"].clamp(min=0).float()
		level = ft.fractional_level(x["h_exp"].clamp(min=0).float(), self.features.level_experience)
		mana, mana_max, spell_cost = x["h_mana"].clamp(min=0).float(), x["h_mana_max"].clamp(min=0).float(), x["h_spell_cost"].clamp(min=0).float()
		move, move_max = x["h_move"].clamp(min=0).float(), x["h_move_max"].clamp(min=0).float()
		numbers = torch.cat(
			[
				scales(level, (2, 8, 32)),
				(level - level.floor()).unsqueeze(-1),  # the way to the next level
				scales(primary, (2, 8, 32)).flatten(-2),
				torch.log1p(0.05 * primary[..., :2]),  # attack and defense: the fighting-strength factors
				(mana / mana_max.clamp(min=1)).unsqueeze(-1),
				scales(mana, (5, 25, 125)),
				(spell_cost > 0).float().unsqueeze(-1),
				(mana / (3 * spell_cost).clamp(min=1)).clamp(max=1).unsqueeze(-1) * (spell_cost > 0).unsqueeze(-1),  # of a 3-round battle's casts
				(move / move_max.clamp(min=1)).clamp(max=1).unsqueeze(-1),  # the share of the day's walk left
				x["h_pos"].float() / size[:, None, :],
				move_max.unsqueeze(-1) / 1500,
				x["h_boat"].float().unsqueeze(-1),
				x["h_garrisoned"].float().unsqueeze(-1),
				x["h_detailed"].float().unsqueeze(-1),
				(x["h_town"] >= 0).float().unsqueeze(-1),
				(x["h_owner"] == c.OWNER_SELF).float().unsqueeze(-1),
				x["h_present"].float().unsqueeze(-1),
				f.hero_strength.unsqueeze(-1) / 10,
				scales(f.hero_power, (1e3, 1e4, 1e5)),
			],
			dim=-1,
		)
		artifacts = torch.cat([x["h_artifacts"], x["h_backpack"]], dim=-1)
		token = (
			self.hero_type(vocab_index(x["h_type"], c.HERO_TYPE_VOCAB)) + self.hero_class(vocab_index(x["h_class"], c.HERO_CLASS_VOCAB))
			+ self.owner(vocab_index(x["h_owner"], c.OWNER_VOCAB)) + self.hero_in(numbers) + stacks.sum(dim=-2)
			+ (self.artifact(vocab_index(artifacts, c.ARTIFACT_VOCAB)) * (artifacts >= 0).unsqueeze(-1)).sum(dim=-2)
			+ self.skills_in(x["h_skills"].float() / 3) + self.spells_in(x["h_spells"].float())
			+ self.last_hero_in(F.pad(f.last_hero, (0, c.N_HEROES - HO)).unsqueeze(-1))
			+ self.nk2_hero(F.pad(f.main.float(), (0, c.N_HEROES - HO)).unsqueeze(-1))
		)
		return token, stacks

	def town_tokens(self, x, f):
		"""Town tokens [B, T, d], garrison stacks [B, T, A, d] and level tokens [B, T, N_LEVELS, d]."""
		stacks = self.stacks(x["t_garrison_creature"], f.town_counts)
		size = x["g_map"].float().clamp(min=1)
		available = x["t_available"].clamp(min=0).float()
		level = torch.arange(c.N_LEVELS, device=available.device).float().expand_as(available) / c.N_LEVELS
		creature = x["t_available_creature"]
		levels = self.level_in(
			torch.cat([self.creature(vocab_index(creature, c.CREATURE_VOCAB)), scales(available, (5, 20, 80)), scales(x["t_growth"], (5, 20, 80)), level.unsqueeze(-1)], dim=-1)
		) * (creature >= 0).unsqueeze(-1)
		danger = torch.cat([f.town_danger, torch.zeros_like(f.town_danger)], dim=1)  # (none for the others' towns)
		numbers = torch.cat(
			[
				x["t_pos"].float() / size[:, None, :],
				ft.one_hot(x["t_fort"].clamp(0, 3), 4),
				(x["t_built_today"] > 0).float().unsqueeze(-1),
				scales(f.town_power, (1e3, 1e4, 1e5)),
				(x["t_visiting"] >= 0).float().unsqueeze(-1),
				(x["t_garrison_hero"] >= 0).float().unsqueeze(-1),
				danger.unsqueeze(-1),
			],
			dim=-1,
		)
		token = (
			self.faction(vocab_index(x["t_faction"], c.FACTION_VOCAB)) + self.owner(vocab_index(x["t_owner"], c.OWNER_VOCAB)) + self.town_in(numbers)
			+ self.built_in(x["t_built"].float()) + stacks.sum(dim=-2) + levels.sum(dim=-2)
			+ self.last_town_in(F.pad(f.last_town, (0, c.N_TOWNS - TO)).unsqueeze(-1))
		)
		return token, stacks, levels

	def options(self, x, f):
		size = x["g_map"].float().clamp(min=1)
		numbers = torch.cat([scales(x["o_amount"], (500, 2000, 8000)), x["o_pos"].float() / size[:, None, :], (x["o_type"] >= 0).float().unsqueeze(-1)], dim=-1)
		return (
			self.option_kind(vocab_index(x["o_kind"], len(c.OptionKind))) + self.subtype(ft.bucket(x["o_kind"], x["o_type"], x["o_subtype"])) + self.option_in(numbers)
			+ self.nk2_option(f.skill_scores.unsqueeze(-1) / 4)
		)

	def all_tokens(self, x, f, hero_tokens, town_tokens, blocks, block_valid):
		"""The transformer's tokens [B, N, d] (global, heroes, towns, the other army, options), which are there, and the
		other army's stacks."""
		valid = block_valid.unsqueeze(-1)
		map_mean = (blocks * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
		map_max = blocks.masked_fill(~valid, -1e4).amax(dim=1) * valid.any(dim=1)
		global_token = self.global_token(x, f) + self.map_pool(torch.cat([map_mean, map_max], dim=-1))
		other_counts, _, _, _ = ft.army(x["x_creature"], x["x_count"], torch.ones_like(x["x_creature"][:, 0], dtype=torch.bool), self.features)
		other_stacks = self.stacks(x["x_creature"], other_counts)
		other_present = (x["x_creature"] >= 0).any(dim=-1)
		other_token = (self.other_token + other_stacks.sum(dim=-2)).unsqueeze(1)
		tokens = torch.cat([global_token.unsqueeze(1), hero_tokens, town_tokens, other_token, self.options(x, f)], dim=1)
		valid = torch.cat([torch.ones_like(other_present).unsqueeze(1), x["h_present"], x["t_present"], other_present.unsqueeze(1), x["option_mask"]], dim=1)
		return tokens, valid, other_stacks

	def token_places(self, x):
		"""Each token's tile [B, N, 3] and whether it has one (heroes, towns, an option's object; not the rest)."""
		none = torch.zeros_like(x["h_pos"][:, :1])
		positions = torch.cat([none, x["h_pos"], x["t_pos"], none, x["o_pos"]], dim=1).float()
		nowhere = torch.zeros_like(x["h_present"][:, :1])
		placed = torch.cat([nowhere, x["h_present"], x["t_present"], nowhere, x["option_mask"] & (x["o_kind"] == c.OptionKind.OBJECT)], dim=1)
		return positions, placed

	# ---- windows ----

	def windows(self, x):
		"""Flat indices (decision x OWN_HEROES + hero) of the windows to compute, our heroes' that are there (a player
		has 1-3 of its 8 slots; the windows are most of the model's compute), and how many are real: on Apple's GPU
		padded with copies of the first to a power of two (it compiles and keeps a graph, ~100 MB, for every new
		shape); the copies are left out of the results."""
		windows = torch.nonzero(x["h_present"][:, :HO].reshape(-1)).squeeze(-1)
		n = windows.shape[0]
		if n > 0 and windows.device.type == "mps" and not torch.onnx.is_in_onnx_export():
			windows = torch.cat([windows, windows[:1].expand((1 << (n - 1).bit_length()) - n)])
		return windows, n

	def window_cells(self, x, f, windows):
		"""The window tiles after the convolutions [K, dc, W, W]: embeddings of what is there + its numbers (CELL_FEATURES)."""
		def pick(name):
			return x[name].reshape((-1,) + tuple(x[name].shape[2:])).index_select(0, windows)

		obj = pick("w_object")
		cell = (
			self.cell_terrain(vocab_index(pick("w_terrain"), c.TERRAIN_VOCAB)) + self.cell_road(vocab_index(pick("w_road"), c.ROAD_VOCAB))
			+ self.cell_object(vocab_index(obj, c.OBJECT_VOCAB)) + self.cell_subtype(ft.bucket(obj, obj, pick("w_subtype")))
			+ self.cell_owner(vocab_index(pick("w_owner"), c.OWNER_VOCAB)) + self.cell_creature(vocab_index(pick("w_creature"), c.CREATURE_VOCAB))
			+ self.cell_in(f.cell_features)
		)  # [K, W, W, dc]
		grid = cell.permute(0, 3, 1, 2)
		return grid + self.conv2(F.gelu(self.conv1(F.gelu(grid))))

	def window_summary(self, x, cells, windows):
		"""Each window's mean and max over its tiles on the map, as a token addition [K, d]."""
		on_map = (torch.remainder(x["w_flags"].reshape(-1, W, W).index_select(0, windows), 2) == 1).unsqueeze(1)  # TILE_ON_MAP
		mean = (cells * on_map).sum(dim=(-1, -2)) / on_map.sum(dim=(-1, -2)).clamp(min=1)
		top = cells.masked_fill(~on_map, -1e4).amax(dim=(-1, -2)) * on_map.any(dim=-1).any(dim=-1)
		return self.cell_pool(torch.cat([mean, top], dim=-1))

	# ---- the heads ----

	def outputs(self, x, f, tokens, cells, areas, windows, n, hero_stacks, town_stacks, town_levels, other_stacks):
		g = tokens[:, 0]
		heroes = tokens[:, 1 : 1 + c.N_HEROES]
		towns = tokens[:, 1 + c.N_HEROES : 1 + c.N_HEROES + c.N_TOWNS]
		other = tokens[:, 1 + c.N_HEROES + c.N_TOWNS]
		options = tokens[:, 2 + c.N_HEROES + c.N_TOWNS :]
		own_heroes, own_towns = heroes[:, :HO], towns[:, :TO]
		out = {}
		offers = self.move_and_travel(x, f, out, own_heroes, cells, areas, windows, n)
		hire_prior = self.town_actions(x, f, out, own_towns, town_levels)
		# HIRE's type gets its best legal hire's gain, signed: a hire that loses (a hero with a lone creature) holds HIRE
		# back as much as a good one pushes it (an offer, clipped at 0, let a learned taste for hiring take bad hires:
		# 10 hires, 25,000 gold, in a lost game against Nullkiller2, 2026-10-30)
		legal = x["hire_mask"].reshape(hire_prior.shape[0], -1)
		best = hire_prior.reshape(legal.shape).masked_fill(~legal, -1e4).amax(dim=1)
		offers["hire"] = torch.where(legal.any(dim=1), best, torch.zeros_like(best))
		offers["army"], offers["upgrade"] = self.army_actions(x, f, out, own_heroes, own_towns, other, hero_stacks, town_stacks, other_stacks)
		offers["buy"], offers["trade"] = self.hero_actions(x, f, out, g, own_heroes)
		self.question(x, f, out, g, heroes, options)
		self.action_types(x, f, out, g, offers)
		self.value(x, f, out, g, own_heroes, own_towns)
		out["memory"] = torch.zeros_like(x["mem"], dtype=torch.float32)  # (v1 has no memory: the contract's output, kept at 0)
		out["resource_worth"] = f.resource_worth
		out["artifact_logp"] = flat_log_softmax(f.artifact_prior, x["artifact_mask"])  # (v1: the prior only, schema 13's ARTIFACT)
		out["multi_logp"] = flat_log_softmax(f.multi_prior, x["multi_mask"])  # (v1: the prior only, schema 14's MULTI)
		return out

	def move_and_travel(self, x, f, out, own_heroes, cells, areas, windows, n):
		"""MOVE: each hero's token modulates its window; a tile's logit adds its prior (its value in gold as a logit)
		and its features' learned score. TRAVEL: each area's prior (its value discounted by the days to it) and its
		learned score. Returns what each offers [B] (features.offer), for the type head."""
		b = x["g_day"].shape[0]
		move = cast_cell = prior = own_heroes.new_zeros(b, HO, W, W)
		if n > 0:
			hero = own_heroes.reshape(b * HO, -1).index_select(0, windows)  # [K, d]
			gamma, beta = self.film(hero).chunk(2, dim=-1)  # [K, dc]
			tile = F.gelu(cells * (1 + gamma[..., None, None]) + beta[..., None, None]).permute(0, 2, 3, 1)  # [K, W, W, dc]
			legal = x["move_mask"].reshape(-1, W, W).index_select(0, windows)
			score = self.move_out(tile).squeeze(-1) + f.tile_prior + self.move_score(f.move_features, legal, hero)
			move, cast_cell, prior = (dense(v, windows, n, b) for v in (score, self.cast_cell_out(tile).squeeze(-1), f.tile_prior))
		out["move_logp"] = flat_log_softmax(move, x["move_mask"])
		out["cast_cell_logp"] = masked_log_softmax(cast_cell.reshape(b, HO, W * W), x["cast_cell_mask"].reshape(b, HO, W * W)).reshape(b, HO, W, W)
		travel_prior = f.travel_prior.reshape(b, HO, L, TG, TG)
		travel = self.travel(own_heroes, x["h_pos"][:, :HO], areas, f.area_centres).reshape(b, HO, L, TG, TG) + travel_prior
		out["travel_logp"] = flat_log_softmax(travel, x["travel_mask"])
		both = torch.cat([prior.reshape(b, -1), travel_prior.reshape(b, -1)], dim=1), torch.cat([x["move_mask"].reshape(b, -1), x["travel_mask"].reshape(b, -1)], dim=1)
		return {"move": ft.offer(prior, x["move_mask"]), "travel": ft.offer(travel_prior, x["travel_mask"]), "any": ft.offer(*both)}

	def town_actions(self, x, f, out, own_towns, town_levels):
		b, d = own_towns.shape[0], self.dim
		build = self.build_q(own_towns) @ self.building.weight.t() / math.sqrt(d) + self.build_bias + self.nk2_build(f.build_rewards).squeeze(-1)
		out["build_logp"] = flat_log_softmax(build, x["build_mask"])
		available = x["t_available"][:, :TO].clamp(min=0).float()
		gold = x["g_resources"][:, 6].clamp(min=0).float()
		recruit_in = torch.cat(
			[
				own_towns.unsqueeze(2).expand(b, TO, c.N_LEVELS, d),
				town_levels[:, :TO],
				scales(available, (5, 20, 80)),
				scales(gold, (1000, 5000, 20000)).reshape(b, 1, 1, 3).expand(b, TO, c.N_LEVELS, 3),
				scales(self.features.creature_cost[vocab_index(x["t_available_creature"][:, :TO], c.CREATURE_VOCAB)] * available, (1000, 5000, 20000)),
			],
			dim=-1,
		)
		recruit = self.recruit_head(recruit_in).reshape(b, TO, c.N_LEVELS, c.N_RECRUIT_AMOUNTS, c.N_RECRUIT_TARGETS)
		out["recruit_logp"] = flat_log_softmax(recruit + self.features.recruit_prior(f.town_danger), x["recruit_mask"])
		tavern = self.hero_type(vocab_index(x["g_tavern"], c.HERO_TYPE_VOCAB))  # [B, 2, d]
		hire_in = torch.cat(
			[
				own_towns.unsqueeze(2).expand(b, TO, c.N_TAVERN, d),
				tavern.unsqueeze(1).expand(b, TO, c.N_TAVERN, d),
				scales(x["g_tavern_level"], (2, 8)).reshape(b, 1, c.N_TAVERN, 2).expand(b, TO, c.N_TAVERN, 2),
			],
			dim=-1,
		)
		# What a hire does to Φ as its prior (features.hire_gain: + for a hero with the army the week's heroes bring, - for
		# one with a lone creature), the same at every town; and the learned part of what it brings, from 0
		gain = f.hire_gain.unsqueeze(1).expand(b, TO, c.N_TAVERN)
		worth = torch.cat([scales(f.tavern_worth.clamp(min=0), (1000, 3000, 10000)), torch.tanh(f.hire_gain / 2000).unsqueeze(-1)], dim=-1)
		hire_prior = ft.value_logit(gain)
		out["hire_logp"] = flat_log_softmax(self.hire_head(hire_in).squeeze(-1) + hire_prior + self.hire_worth(worth).squeeze(-1).unsqueeze(1), x["hire_mask"])
		return hire_prior

	def army_actions(self, x, f, out, own_heroes, own_towns, other, hero_stacks, town_stacks, other_stacks):
		b, d = own_heroes.shape[0], self.dim
		holders = torch.cat([own_heroes, own_towns, other.unsqueeze(1)], dim=1)  # [B, HD, d]
		holder_stacks = torch.cat([hero_stacks[:, :HO], town_stacks[:, :TO], other_stacks.unsqueeze(1)], dim=1)  # [B, HD, A, d]
		src = self.army_src(torch.cat([holders.unsqueeze(2).expand(b, HD, A, d), holder_stacks], dim=-1))  # [B, HD, A, d]
		dst = self.army_dst(holders)
		prior = self.features.army_prior(f.hero_strength[:, :HO])  # (CustomFeatures.army_prior: Nullkiller2's habits)
		out["army_logp"] = flat_log_softmax(self.army_out(src[:, :, :, None, :] + dst[:, None, None, :, :]) + prior, x["army_mask"])
		_, upgrade_weight, _, _ = self.features.nk2_weights()
		upgrade_prior = upgrade_weight * (f.upgrade_gains[..., 0] > 0).float()  # Nullkiller2 upgrades every stack it can
		out["upgrade_logp"] = flat_log_softmax(self.upgrade_out(src).squeeze(-1) + self.nk2_upgrade(f.upgrade_gains).squeeze(-1) + upgrade_prior, x["upgrade_mask"])
		# What ARMY offers, for its type: its best legal entry's prior (a whole stack from a town's garrison into a hero
		# there, or toward a stronger hero; below 0 if every legal move leaves troops behind)
		legal = x["army_mask"].reshape(b, -1)
		best = prior.expand(b, HD, A, HD, c.N_ARMY_AMOUNTS).reshape(b, -1).masked_fill(~legal, -1e4).amax(dim=1)
		upgrade_legal = x["upgrade_mask"].reshape(b, -1)
		upgrade_best = upgrade_prior.reshape(b, -1).masked_fill(~upgrade_legal, -1e4).amax(dim=1)
		return torch.where(legal.any(dim=1), best, torch.zeros_like(best)), torch.where(upgrade_legal.any(dim=1), upgrade_best, torch.zeros_like(upgrade_best))

	def hero_actions(self, x, f, out, g, own_heroes):
		"""TRADE, CAST, BUY, DISMISS; returns what BUY and TRADE offer [B] (their best legal entry's prior: a wanted
		spellbook; a trade that buys what a building short of resources lacks with what it does not need)."""
		b, d = own_heroes.shape[0], self.dim
		R = c.N_RESOURCES
		needs = torch.stack([f.trade_needs[:, None, :, 0].expand(b, R, R), f.trade_needs[:, :, None, 1].expand(b, R, R)], dim=-1)  # [B, sold, bought, 2]
		_, _, _, _, trade_weight = self.features.pull_weights()
		trade_prior = (trade_weight * ((needs[..., 0] > 0) & (needs[..., 1] > 0)).float()).unsqueeze(-1)  # [B, R, R, 1]
		trade = self.trade_head(g).reshape(b, R, R, c.N_TRADE_AMOUNTS) + self.nk2_trade(needs) + trade_prior
		out["trade_logp"] = flat_log_softmax(trade, x["trade_mask"])
		legal = x["trade_mask"].reshape(b, -1)
		best = trade_prior.expand(b, R, R, c.N_TRADE_AMOUNTS).reshape(b, -1).masked_fill(~legal, -1e4).amax(dim=1)
		trade_offer = torch.where(legal.any(dim=1), best, torch.zeros_like(best))
		out["cast_logp"] = flat_log_softmax(self.cast_q(own_heroes) @ self.spell.weight.t() / math.sqrt(d), x["cast_mask"])
		_, _, book_weight, _ = self.features.nk2_weights()
		book = (torch.arange(c.N_BUY_ITEMS, device=g.device) == c.BUY_ARTIFACTS.index(ft.SPELLBOOK)).float()
		buy_prior = book_weight * f.spellbook_wanted.float().unsqueeze(-1) * book  # [B, HO, N_BUY_ITEMS]
		out["buy_logp"] = flat_log_softmax(self.buy_q(own_heroes) @ self.buy_item.weight.t() / math.sqrt(d) + buy_prior, x["buy_mask"])
		out["dismiss_logp"] = flat_log_softmax(self.dismiss_out(own_heroes).squeeze(-1), x["dismiss_mask"])
		legal = x["buy_mask"].reshape(b, -1)
		best = buy_prior.reshape(b, -1).masked_fill(~legal, -1e4).amax(dim=1)
		return torch.where(legal.any(dim=1), best, torch.zeros_like(best)), trade_offer

	def question(self, x, f, out, g, heroes, options):
		"""The question's options, read with the hero it is about; a level-up's skills with Nullkiller2's scores as a prior."""
		b, d = g.shape[0], self.dim
		about = torch.gather(heroes, 1, x["d_hero"].clamp(min=0).reshape(b, 1, 1).expand(b, 1, d)).squeeze(1) * (x["d_hero"] >= 0).float().unsqueeze(-1)
		query = self.option_q(torch.cat([g, about], dim=-1))
		skill_weight, _, _, _ = self.features.nk2_weights()
		out["option_logp"] = masked_log_softmax((options * query.unsqueeze(1)).sum(dim=-1) / math.sqrt(d) + skill_weight * f.skill_scores, x["option_mask"])

	def action_types(self, x, f, out, g, offers):
		"""The action type: legal when its table has a legal entry (END in our turn and in an exchange). Its logits: the
		learned head (from the priors of priors/action_types.csv), a learned summary of what MOVE and TRAVEL offer (from
		0), and two couplings with the tables' priors (CustomFeatures.couplings, 1.5 and 2 at the start): what MOVE and TRAVEL
		offer holds END back (ending a turn with value left on the table is a loss), and what MOVE offers holds TRAVEL
		back (taking what is in reach first, then travelling, is never worse); and ARMY gets its table's best prior (an
		exchange whose only moves leave troops in a mine closes). None changes the odds among the other types, and no
		type is ever ruled out."""
		b = g.shape[0]
		kind = x["g_kind"]
		legal = [(kind == c.DecisionKind.TURN) | (kind == c.DecisionKind.EXCHANGE)]
		for action in list(c.ActionType)[1:]:
			legal.append(x[HEADS[action][1]].reshape(b, -1).any(dim=-1))
		summary = torch.stack([torch.tanh(offers["move"] / 4), torch.tanh(offers["travel"] / 4), (offers["move"] > 0).float(), (offers["travel"] > 0).float()], dim=-1)
		end_coupling, travel_coupling = self.features.couplings()
		types = torch.arange(c.N_ACTION_TYPES, device=g.device)
		couple = -(end_coupling * offers["any"]).unsqueeze(-1) * (types == c.ActionType.END) - (travel_coupling * offers["move"]).unsqueeze(-1) * (types == c.ActionType.TRAVEL)
		couple = couple + offers["army"].unsqueeze(-1) * (types == c.ActionType.ARMY)  # (an exchange that can only leave troops behind closes)
		# Nullkiller2's habits: an upgrade it can make, a spellbook it wants; no army while saving for a City Hall
		_, _, _, pause_weight = self.features.nk2_weights()
		couple = couple + offers["upgrade"].unsqueeze(-1) * (types == c.ActionType.UPGRADE) + offers["buy"].unsqueeze(-1) * (types == c.ActionType.BUY)
		couple = couple - (pause_weight * f.recruit_pause).unsqueeze(-1) * (types == c.ActionType.RECRUIT)
		_, _, _, hire_weight, _ = self.features.pull_weights()  # (weak pulls: hiring as Nullkiller2 would; a trade toward a building)
		couple = couple + (hire_weight * f.hire_want + offers["hire"]).unsqueeze(-1) * (types == c.ActionType.HIRE) + offers["trade"].unsqueeze(-1) * (types == c.ActionType.TRADE)
		out["type_logp"] = masked_log_softmax(self.type_head(g) + self.type_summary(summary) + couple, torch.stack(legal, dim=-1))

	def value(self, x, f, out, g, own_heroes, own_towns):
		"""The value heads, from the global token, our heroes' and towns' mean, and Φ straight in (see a pickup the
		moment it happens): V2, V5, V7, our share of all empires' value at the end of days d + 2, 5, 7, and the growth
		heads, our own value's log growth to the same days (labels.py: GROWTH_CAP), for training only, and V_end, our share
		at the game's end (training only: the update reads it, the AI never does). Each head its own
		readout, none chained (2026-10-28: a chained V7 starts as V2's answer to a shorter question, and so does V5
		two days on, which grades it; labels.ladder_targets passes what the short heads know up without that). The
		shares start at 1 / the players still in the game (the readout at 0, its prior logit(1 / players)), the growth
		at none."""
		mine = torch.cat([x["h_present"][:, :HO], x["t_present"][:, :TO]], dim=1).float().unsqueeze(-1)
		own = torch.cat([own_heroes, own_towns], dim=1)
		own_mean = (own * mine).sum(dim=1) / mine.sum(dim=1).clamp(min=1)
		hidden = self.value_hidden(torch.cat([g, own_mean, f.phi], dim=-1))
		players = x["g_players"].clamp(min=2).float()
		share_prior = -torch.log(players - 1)  # a share of 1 / players
		value_logit = self.value_heads(hidden) + share_prior.unsqueeze(-1)
		out["value_logit"] = value_logit
		out["value"] = torch.sigmoid(value_logit)
		out["growth_logit"] = self.growth_heads(hidden)
		out["end_logit"] = self.end_head(hidden).squeeze(-1) + share_prior
		out["phi"] = f.phi_total


def build_model(model_kwargs=None):
	"""The campaign model of a checkpoint's model_kwargs: v1 (CampaignModel here), or model_v2.CampaignModelV2 when its
	"architecture" says "v2" (docs/campaign_model_v2_draft.md)."""
	kwargs = dict(model_kwargs or {})
	if kwargs.pop("architecture", "v1") == "v2":
		from .model_v2 import CampaignModelV2

		return CampaignModelV2(**kwargs)
	return CampaignModel(**kwargs)


INPUT_NAMES = tuple(s.name for s in INPUTS)
OUTPUT_NAMES = tuple(s.name for s in OUTPUTS)
