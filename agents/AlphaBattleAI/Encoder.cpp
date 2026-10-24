#include "StdInc.h"
#include "Encoder.h"

#include "Policy.h"

#include "lib/CCreatureHandler.h"
#include "lib/CRandomGenerator.h"
#include "lib/CStack.h"
#include "lib/battle/BattleAction.h"
#include "lib/battle/BattleAttackInfo.h"
#include "lib/battle/CObstacleInstance.h"
#include "lib/battle/CPlayerBattleCallback.h"
#include "lib/GameLibrary.h"
#include "lib/entities/artifact/ArtifactUtils.h"
#include "lib/entities/artifact/CArtifact.h"
#include "lib/entities/hero/CHeroHandler.h"
#include "lib/bonuses/Bonus.h"
#include "lib/bonuses/BonusSelector.h"
#include "lib/callback/IGameInfoCallback.h"
#include "lib/constants/NumericConstants.h"
#include "lib/gameState/InfoAboutArmy.h"
#include "lib/spells/Problem.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/spells/CSpellHandler.h"
#include "lib/spells/ISpellMechanics.h"
#include "lib/spells/effects/UnitEffect.h"

#include <vcmi/Environment.h>
#include <vcmi/spells/Magic.h>

using namespace AlphaSchema;

namespace
{
/// The units the model sees: up to UNITS_PER_SIDE of each battle side: the acting unit, then living stacks by value,
/// then dead ones (Resurrection can target them); in the engine's order (unit ids), as the slots always were
std::vector<const CStack *> pickUnits(const CPlayerBattleCallback & battle, const CStack * active)
{
	auto stacks = battle.battleGetAllStacks(true);
	std::sort(stacks.begin(), stacks.end(), [](const CStack * a, const CStack * b) { return a->unitId() < b->unitId(); });
	std::array<std::vector<const CStack *>, 2> sides;
	for(const auto * s : stacks)
		if(s->unitSide() == BattleSide::ATTACKER || s->unitSide() == BattleSide::DEFENDER)
			sides[static_cast<int>(s->unitSide())].push_back(s);
	std::vector<const CStack *> picked;
	for(auto & side : sides)
	{
		auto value = [](const CStack * s) { return s->alive() ? creatureValue(s->unitType()) * s->getCount() : 0.0; };
		std::stable_sort(side.begin(), side.end(), [&](const CStack * a, const CStack * b)
		{
			return std::make_tuple(a != active, !a->alive(), -value(a)) < std::make_tuple(b != active, !b->alive(), -value(b));
		});
		side.resize(std::min(side.size(), static_cast<size_t>(UNITS_PER_SIDE)));
		picked.insert(picked.end(), side.begin(), side.end());
	}
	std::sort(picked.begin(), picked.end(), [](const CStack * a, const CStack * b) { return a->unitId() < b->unitId(); });
	return picked;
}

void encodeUnits(EncodedBattle & e, const CPlayerBattleCallback & battle, const std::vector<const CStack *> & stacks)
{
	auto & t = e.tensors;
	std::vector<battle::Units> queue;
	battle.battleGetTurnOrder(queue, MAX_UNITS, 2);

	for(size_t slot = 0; slot < stacks.size(); ++slot)
	{
		const int u = static_cast<int>(slot);
		const CStack * s = stacks[slot];
		e.units[slot] = s;
		t.set(In::u_present, u, 1);
		t.set(In::u_alive, u, s->alive());
		t.set(In::u_creature, u, s->creatureId().getNum());
		t.set(In::u_side, u, static_cast<int>(s->unitSide()));
		t.set(In::u_owner_side, u, static_cast<int>(battle.playerToSide(battle.battleGetOwner(s))));
		t.set(In::u_hex, u, s->getPosition().toInt());
		t.set(In::u_rear_hex, u, s->doubleWide() ? s->occupiedHex().toInt() : -1);
		t.set(In::u_count, u, s->getCount());
		t.set(In::u_first_hp, u, s->getFirstHPleft());
		t.set(In::u_max_hp, u, s->getMaxHealth());
		t.set(In::u_attack, u, s->getAttack(false));
		t.set(In::u_defense, u, s->getDefense(false));
		t.set(In::u_dmg_min, u, s->getMinDamage(false));
		t.set(In::u_dmg_max, u, s->getMaxDamage(false));
		t.set(In::u_speed, u, s->getMovementRange());
		t.set(In::u_shots, u, s->shots.available());
		t.set(In::u_retaliations, u, s->counterAttacks.isLimited() ? std::min(s->counterAttacks.available(), MAX_RETALIATIONS) : MAX_RETALIATIONS);
		t.set(In::u_casts, u, s->casts.available());
		// The engine's effective values (clamped to +-3; 0 for units morale or luck does not affect, e.g. undead), not the
		// raw sum of bonuses (bug fix 2026-10-23)
		t.set(In::u_morale, u, s->moraleVal());
		t.set(In::u_luck, u, s->luckVal());
		t.set(In::u_moved, u, s->moved());
		t.set(In::u_waited, u, s->waitedThisTurn);
		t.set(In::u_defending, u, s->defending);
		t.set(In::u_is_clone, u, s->isClone());
		t.set(In::u_is_summoned, u, s->summoned);

		int k = 0;
		for(const auto & spellId : s->activeSpells())
		{
			if(k >= MAX_EFFECTS)
				break;
			auto bonuses = s->getBonuses(Selector::source(BonusSource::SPELL_EFFECT, BonusSourceID(spellId)));
			t.set(In::u_effect_id, u, k, spellId.getNum());
			if(!bonuses->empty())
			{
				t.set(In::u_effect_turns, u, k, bonuses->front()->turnsRemain);
				t.set(In::u_effect_value, u, k, bonuses->front()->val);
			}
			++k;
		}

		for(int round = 0; round < static_cast<int>(queue.size()) && round < 2; ++round)
		{
			const auto & order = queue[round];
			for(size_t p = 0; p < order.size(); ++p)
				if(order[p]->unitId() == s->unitId())
				{
					t.set(round == 0 ? In::u_queue_now : In::u_queue_next, u, static_cast<int64_t>(p));
					break;
				}
		}
	}
}

void encodeHeroes(InputTensors & t, const CPlayerBattleCallback & battle, BattleSide mySide, const CGHeroInstance * enemyHero)
{
	// Our hero: everything
	const int my = static_cast<int>(mySide);
	if(const auto * hero = battle.battleGetMyHero())
	{
		t.set(In::h_present, my, 1);
		t.set(In::h_type, my, hero->getHeroTypeID().getNum());
		t.set(In::h_attack, my, hero->getPrimSkillLevel(PrimarySkill::ATTACK));
		t.set(In::h_defense, my, hero->getPrimSkillLevel(PrimarySkill::DEFENSE));
		t.set(In::h_power, my, hero->getPrimSkillLevel(PrimarySkill::SPELL_POWER));
		t.set(In::h_knowledge, my, hero->getPrimSkillLevel(PrimarySkill::KNOWLEDGE));
		t.set(In::h_mana, my, hero->mana);
		t.set(In::h_mana_max, my, hero->manaLimit());
		for(int skill = 0; skill < N_SECONDARY_SKILLS; ++skill)
			t.set(In::h_skills, my, skill, hero->getSecSkillLevel(SecondarySkill(skill)));
		if(hero->hasSpellbook())
			for(int spell = 0; spell < N_SPELLS; ++spell)
				t.set(In::h_spells, my, spell, hero->spellbookContainsSpell(SpellID(spell)));
		const double level = heroLevel(hero);
		t.set(In::h_level, my, static_cast<int64_t>(hero->level));
		t.set(In::h_level_progress, my, std::llround(1000 * (level - hero->level)));
		t.set(In::h_artifacts, my, std::llround(heroArtifacts(hero)));
	}
	// The enemy hero: only what a player sees in battle (no secondary skills, no spellbook). The player
	// callback's battleGetEnemyHero() hides even that (BASIC level), but the client shows a human both heroes'
	// in-battle stats (BattleWindow::createStickyHeroInfoWindows, INBATTLE level): use the same.
	const int enemy = 1 - my;
	if(battle.battleHasHero(static_cast<BattleSide>(enemy)))
	{
		const auto info = enemyHero ? InfoAboutHero(enemyHero, InfoAboutHero::EInfoLevel::INBATTLE) : battle.battleGetEnemyHero();
		t.set(In::h_present, enemy, 1);
		t.set(In::h_type, enemy, info.portraitSource.getNum());
		if(info.details)
		{
			const auto & primary = info.details->primskills;
			t.set(In::h_attack, enemy, primary.size() > 0 ? primary[0] : 0);
			t.set(In::h_defense, enemy, primary.size() > 1 ? primary[1] : 0);
			t.set(In::h_power, enemy, primary.size() > 2 ? primary[2] : 0);
			t.set(In::h_knowledge, enemy, primary.size() > 3 ? primary[3] : 0);
			t.set(In::h_mana, enemy, info.details->mana);
			t.set(In::h_mana_max, enemy, info.details->manaLimit);
		}
	}
}

void encodeHexes(InputTensors & t, const CPlayerBattleCallback & battle, BattleSide mySide, std::array<bool, N_HEXES> & stopping)
{
	const auto access = battle.getAccessibility();
	for(int h = 0; h < N_HEXES; ++h)
		t.set(In::x_access, h, static_cast<int>(access[h]));

	for(const auto & obstacle : battle.battleGetAllObstacles(mySide))
	{
		const auto * spellObstacle = dynamic_cast<const SpellCreatedObstacle *>(obstacle.get());
		auto tiles = obstacle->getAffectedTiles();
		tiles.insert(obstacle->getBlockedTiles());
		for(const auto & hex : tiles)
		{
			const int h = hex.toInt();
			if(h < 0 || h >= N_HEXES)
				continue;
			// Several obstacles on one hex: keep a spell-created one over terrain
			if(t.get(In::x_obstacle_type, h) >= 0 && !spellObstacle)
				continue;
			t.set(In::x_obstacle_type, h, static_cast<int>(obstacle->obstacleType));
			t.set(In::x_obstacle_id, h, obstacle->ID);
			if(spellObstacle)
			{
				t.set(In::x_obstacle_side, h, static_cast<int>(spellObstacle->casterSide));
				t.set(In::x_obstacle_turns, h, spellObstacle->turnsRemaining);
			}
		}
		for(const auto & hex : obstacle->getStoppingTile())
			if(hex.toInt() >= 0 && hex.toInt() < N_HEXES)
				stopping[hex.toInt()] = true;
	}

	for(int part = static_cast<int>(EWallPart::KEEP); part <= static_cast<int>(EWallPart::GATE); ++part)
	{
		const auto wallPart = static_cast<EWallPart>(part);
		const auto state = battle.battleGetWallState(wallPart);
		const int h = battle.wallPartToBattleHex(wallPart).toInt();
		if(state != EWallState::NONE && h >= 0 && h < N_HEXES)
			t.set(In::x_wall_state, h, static_cast<int>(state));
	}
}

void encodeUnitMasks(EncodedBattle & e, const CPlayerBattleCallback & battle, const CStack * active, const std::array<bool, N_HEXES> & stopping)
{
	auto & t = e.tensors;
	const auto activePos = active->getPosition();
	// Whom the unit may attack at all: the engine's rule (enemies by current owner, so hypnosis counts; not
	// invincible ones; only if the unit can melee), as the client applies it. Where from is checked per hex.
	std::array<bool, MAX_UNITS> canAttack{};
	for(int u = 0; u < MAX_UNITS; ++u)
		canAttack[u] = e.units[u] && battle.battleCanAttackUnit(active, e.units[u]);
	for(const auto & hex : battle.battleGetAvailableHexes(active, false))
	{
		const int h = hex.toInt();
		if(h < 0 || h >= N_HEXES)
			continue;
		if(hex != activePos)
			t.set(In::m_move_hex, h, 1);

		// Moving onto a stopping hex (moat, quicksand) aborts an attack; also when a wide unit's tail lands on one
		if(hex != activePos)
		{
			if(stopping[h])
				continue;
			const int rear = active->doubleWide() ? battle::Unit::occupiedHex(hex, true, active->unitSide()).toInt() : -1;
			if(rear >= 0 && rear < N_HEXES && stopping[rear])
				continue;
		}
		for(int u = 0; u < MAX_UNITS; ++u)
			if(canAttack[u] && CStack::isMeleeAttackPossible(active, e.units[u], hex))
				t.set(In::m_melee_from, u, h, 1);
	}
	for(int u = 0; u < MAX_UNITS; ++u)
	{
		const CStack * target = e.units[u];
		// A long weapon also reaches a unit two hexes away in a line, from where the unit stands
		if(canAttack[u] && battle.isLongWeaponAttack(active, target))
			t.set(In::m_melee_from, u, activePos.toInt(), 1);
		if(t.anyInRow(In::m_melee_from, u))
			t.set(In::m_melee_target, u, 1);
		// The engine's rule for shooting at the unit's hex (who counts as an enemy, berserk, range), and the
		// unit the engine would hit there must be this one
		if(target && battle.battleCanShoot(active, target->getPosition()) && battle.battleGetUnitByPos(target->getPosition(), true) == static_cast<const battle::Unit *>(target))
			t.set(In::m_shoot_target, u, 1);
	}
}

// The engine aims a creature-targeted spell at the unit's hex and picks the unit there by the spell's own rules
// (BattleSpellMechanics::transformSpellTarget, UnitEffect::transformTargetByRange): a slot is a legal target only if
// the cast really affects that unit. A corpse under a living unit is then no alias for the living one (Lightning Bolt
// would hit the living one), Resurrection and Animate Dead cannot revive a corpse any unit stands on
// (Heal::isValidTarget), and Sacrifice can still revive one an enemy stands on but not one under our own living unit
// (that one is picked instead).
bool castAffects(const spells::Mechanics & mechanics, const battle::Target & target, const CStack * unit)
{
	if(!mechanics.canBeCastAt(target))
		return false;
	const auto affected = mechanics.getAffectedStacks(target);
	return std::find(affected.begin(), affected.end(), unit) != affected.end();
}

/// True if what the spell heals at this level only lasts the battle (basic Resurrection: its "healPower" is "oneBattle")
bool healsForOneBattle(const CSpell & spell, int level)
{
	for(const auto & [name, effect] : spell.getLevelInfo(level).battleEffects.Struct())
		if(effect["healPower"].String() == "oneBattle")
			return true;
	return false;
}

/// What a cast aimed at `aim` would do, as the engine predicts it: each affected unit's health change by every unit
/// effect of the spell (UnitEffect::getHealthChange, the rules the cast itself uses), priced as the reward prices
/// armies: hit points at the creature's value per hit point, raised or summoned creatures at their value; nothing for
/// units the side does not keep after the battle, nor for health that only lasts the battle (healsForOneBattle).
/// The value for the attacker minus for the defender, in gold.
int64_t castValue(const CPlayerBattleCallback & battle, const CSpell & spell, const spells::Mechanics & mechanics, const battle::Target & aim)
{
	const bool temporary = healsForOneBattle(spell, mechanics.getEffectLevel()); // the caster's mastery (getSpellLevel is the spell's circle)
	const auto canonical = mechanics.canonicalizeTarget(aim);
	double value = 0;
	mechanics.forEachEffect([&](const spells::effects::Effect & effect)
	{
		const auto * unitEffect = dynamic_cast<const spells::effects::UnitEffect *>(&effect);
		if(!unitEffect)
			return false;
		// Unit by unit, each with the targets before it: a chain's damage depends on the target's place in it
		spells::EffectTarget before;
		spells::effects::SpellEffectValue total;
		for(const auto & destination : unitEffect->transformTarget(&mechanics, aim, canonical))
		{
			before.push_back(destination);
			const auto now = unitEffect->getHealthChange(&mechanics, before);
			const int64_t hp = now.hpDelta - total.hpDelta, units = now.unitsDelta - total.unitsDelta;
			total = now;
			const auto * stack = destination.unitValue ? battle.battleGetStackByID(destination.unitValue->unitId(), false) : nullptr;
			if(!stack)
				continue;
			const double sign = stack->unitSide() == BattleSide::ATTACKER ? 1.0 : -1.0;
			if(now.unitType != CreatureID::NONE && now.unitType != stack->creatureId()) // a new creature (Pit Lords' demons)
				value += sign * units * creatureValue(now.unitType.toCreature());
			else if(keptAfterBattle(stack) && !(temporary && hp > 0))
				value += sign * hp * creatureValue(stack->unitType()) / std::max(1, static_cast<int>(stack->getMaxHealth()));
		}
		return false;
	});
	return std::llround(value);
}

/// The living units a cast aimed at `aim` affects (the spell's own rule, as the cast uses it): (enemies of player,
/// player's own), by who controls them now
std::pair<int, int> castHits(const CPlayerBattleCallback & battle, const spells::Mechanics & mechanics, const battle::Target & aim, PlayerColor player)
{
	int enemies = 0, allies = 0;
	for(const auto * unit : mechanics.getAffectedStacks(aim))
		if(unit && unit->alive())
			++(battle.battleGetOwner(unit) == player ? allies : enemies);
	return {enemies, allies};
}

void encodeSpellMasks(EncodedBattle & e, const CPlayerBattleCallback & battle, const CGHeroInstance * hero, PlayerColor player)
{
	auto & t = e.tensors;
	std::array<int, MAX_UNITS> spell2Owner;
	spell2Owner.fill(-1);

	for(int s = 0; s < N_SPELLS; ++s)
	{
		const auto * spell = SpellID(s).toSpell();
		if(!spell || spell->isAdventure() || !spell->canBeCast(&battle, spells::Mode::HERO, hero))
			continue;
		spells::BattleCast cast(&battle, hero, spells::Mode::HERO, spell);
		auto mechanics = spell->battleMechanics(&cast);
		const auto aims = mechanics->getTargetTypes();
		const auto second = secondTargetKind(s);
		if(aims.empty() || aims.size() > 2 || (aims.size() == 2) != (second != TargetKind::NONE))
			continue; // shape the contract does not describe

		e.spellTargetKind[s] = aims[0] == spells::AimType::LOCATION ? TargetKind::HEX : aims[0] == spells::AimType::CREATURE ? TargetKind::UNIT : TargetKind::NONE;
		if(aims[0] == spells::AimType::NO_TARGET)
		{
			// The server's own check for spells without a destination
			spells::detail::ProblemImpl problem;
			if(mechanics->canBeCast(problem))
				t.set(In::m_spell, s, 1);
			continue;
		}
		if(aims[0] == spells::AimType::LOCATION)
		{
			for(int h = 0; h < N_HEXES; ++h)
				if(mechanics->canBeCastAt({battle::Destination(BattleHex(h))}))
				{
					t.set(In::m_spell_hex, s, h, 1);
					t.set(In::s_hex_value, s, h, castValue(battle, *spell, *mechanics, {battle::Destination(BattleHex(h))}));
					const auto [enemies, allies] = castHits(battle, *mechanics, {battle::Destination(BattleHex(h))}, player);
					t.set(In::s_hex_enemies, s, h, enemies);
					t.set(In::s_hex_allies, s, h, allies);
				}
		}
		else if(aims[0] == spells::AimType::CREATURE)
		{
			auto affects = [&](const battle::Target & target, const CStack * unit) { return castAffects(*mechanics, target, unit); };
			for(int u = 0; u < MAX_UNITS; ++u)
			{
				const CStack * unit = e.units[u];
				if(!unit)
					continue;
				const battle::Destination first(unit);
				if(second == TargetKind::NONE)
				{
					if(affects({first}, unit))
					{
						t.set(In::m_spell_unit, s, u, 1);
						t.set(In::s_unit_value, s, u, castValue(battle, *spell, *mechanics, {first}));
						const auto [enemies, allies] = castHits(battle, *mechanics, {first}, player);
						t.set(In::s_unit_enemies, s, u, enemies);
						t.set(In::s_unit_allies, s, u, allies);
					}
					continue;
				}
				if(spell2Owner[u] >= 0)
					continue; // second-target rows are per first unit; already used by another spell
				bool anySecond = false;
				if(second == TargetKind::HEX && aims[1] == spells::AimType::LOCATION)
				{
					for(int h = 0; h < N_HEXES; ++h)
						if(affects({first, battle::Destination(BattleHex(h))}, unit))
						{
							t.set(In::m_spell2_hex, u, h, 1);
							anySecond = true;
						}
				}
				else if(second == TargetKind::UNIT && aims[1] == spells::AimType::CREATURE)
				{
					for(int v = 0; v < MAX_UNITS; ++v)
						if(e.units[v] && affects({first, battle::Destination(e.units[v])}, unit))
						{
							t.set(In::m_spell2_unit, u, v, 1);
							anySecond = true;
						}
				}
				if(anySecond)
				{
					t.set(In::m_spell_unit, s, u, 1);
					spell2Owner[u] = s;
				}
			}
		}
		if(t.anyInRow(In::m_spell_unit, s) || t.anyInRow(In::m_spell_hex, s))
			t.set(In::m_spell, s, 1);
	}
}
}

namespace
{
/// The acting unit's legal targets for a spell it casts as its action (CREATURE_SPELL), at the level the server casts
/// it (doUnitSpellAction: its SPELLCASTER value). Only spells with one unit or hex target: creature spells without a
/// target or with two (none in the core game) are not offered. True if it has any.
bool encodeCreatureSpellTargets(EncodedBattle & e, const CPlayerBattleCallback & battle, const CStack * active, SpellID id, PlayerColor player)
{
	auto & t = e.tensors;
	const auto * spell = id.toSpell();
	if(!spell || !spell->canBeCast(&battle, spells::Mode::CREATURE_ACTIVE, active))
		return false;
	spells::BattleCast cast(&battle, active, spells::Mode::CREATURE_ACTIVE, spell);
	const auto caster = active->getBonus(Selector::typeSubtype(BonusType::SPELLCASTER, BonusSubtypeID(id)));
	cast.setSpellLevel(caster ? caster->val : 0);
	auto mechanics = spell->battleMechanics(&cast);
	const auto aims = mechanics->getTargetTypes();
	if(aims.size() != 1)
		return false;
	bool any = false;
	if(aims[0] == spells::AimType::LOCATION)
	{
		for(int h = 0; h < N_HEXES; ++h)
			if(mechanics->canBeCastAt({battle::Destination(BattleHex(h))}))
			{
				t.set(In::m_creature_spell_hex, h, 1);
				t.set(In::c_hex_value, h, castValue(battle, *spell, *mechanics, {battle::Destination(BattleHex(h))}));
				const auto [enemies, allies] = castHits(battle, *mechanics, {battle::Destination(BattleHex(h))}, player);
				t.set(In::c_hex_enemies, h, enemies);
				t.set(In::c_hex_allies, h, allies);
				any = true;
			}
	}
	else if(aims[0] == spells::AimType::CREATURE)
	{
		for(int u = 0; u < MAX_UNITS; ++u)
			if(e.units[u] && castAffects(*mechanics, {battle::Destination(e.units[u])}, e.units[u]))
			{
				t.set(In::m_creature_spell_unit, u, 1);
				t.set(In::c_unit_value, u, castValue(battle, *spell, *mechanics, {battle::Destination(e.units[u])}));
				const auto [enemies, allies] = castHits(battle, *mechanics, {battle::Destination(e.units[u])}, player);
				t.set(In::c_unit_enemies, u, enemies);
				t.set(In::c_unit_allies, u, allies);
				any = true;
			}
	}
	return any;
}

/// What the acting unit can cast as its action, as the client offers it (BattleActionsController,
/// tryActivateStackSpellcasting and actionIsLegal): its first fixed spell (SPELLCASTER without a weight) that has a
/// legal target; else the one of its weighted spells drawn for this turn (Faerie Dragons: the client and BattleAI draw
/// it, the server would accept any; drawn here from the round, the unit and whether it waited, so every encoding of
/// the same turn draws the same); else, for a random caster (Master Genie), any other living unit of ours that some
/// beneficial spell suits (the server draws the spell when it casts: g_creature_spell -1).
void encodeCreatureSpell(EncodedBattle & e, const CPlayerBattleCallback & battle, const CStack * active, PlayerColor player)
{
	auto & t = e.tensors;
	if(!active->canCast())
		return;
	std::vector<SpellID> spells;
	for(const auto & bonus : *active->getBonusesOfType(BonusType::SPELLCASTER))
		if(!bonus->parameters && bonus->subtype.as<SpellID>().hasValue())
			spells.push_back(bonus->subtype.as<SpellID>());
	CRandomGenerator draw(battle.battleGetRound() * 4096 + static_cast<int>(active->unitId()) * 2 + active->waitedThisTurn);
	const auto drawn = battle.getRandomCastedSpell(draw, active);
	if(drawn.hasValue())
		spells.push_back(drawn);
	for(const auto id : spells)
		if(encodeCreatureSpellTargets(e, battle, active, id, player))
		{
			t.set(In::g_creature_spell, id.getNum());
			return;
		}
	if(!active->hasBonusOfType(BonusType::RANDOM_SPELLCASTER))
		return;
	CRandomGenerator unused(0);
	for(int u = 0; u < MAX_UNITS; ++u)
	{
		const CStack * unit = e.units[u];
		if(unit && unit != active && unit->alive() && unit->unitOwner() == player && battle.getRandomBeneficialSpell(unused, active, unit) != SpellID::NONE)
			t.set(In::m_creature_spell_unit, u, 1);
	}
}

/// The engine's damage estimate of one attack of each unit on each other (schema.py p_*), as the game shows a player
void encodePairDamage(EncodedBattle & e, const CPlayerBattleCallback & battle)
{
	auto & t = e.tensors;
	for(int i = 0; i < MAX_UNITS; ++i)
	{
		const CStack * a = e.units[i];
		if(!a || !a->alive())
			continue;
		for(int j = 0; j < MAX_UNITS; ++j)
		{
			const CStack * b = e.units[j];
			if(!b || !b->alive() || battle.battleGetOwner(a) == battle.battleGetOwner(b))
				continue;
			for(const bool shooting : {false, true})
			{
				if(shooting && !a->canShoot())
					continue;
				BattleAttackInfo info(a, b, 0, shooting);
				info.attackerPos = a->getPosition();
				info.defenderPos = b->getPosition();
				const auto range = battle.battleEstimateDamage(info).damage;
				t.set(shooting ? In::p_shot_min : In::p_melee_min, i, j, range.min);
				t.set(shooting ? In::p_shot_max : In::p_melee_max, i, j, range.max);
			}
		}
	}
}
}

EncodedBattle encodeBattle(const CPlayerBattleCallback & battle, const CStack * active, const Environment & env, PlayerColor player, const CGHeroInstance * enemyHero, const AlphaReward::BattleEnd & start)
{
	EncodedBattle e;
	auto & t = e.tensors;
	const auto mySide = battle.battleGetMySide();
	const auto stacks = pickUnits(battle, active);
	encodeUnits(e, battle, stacks);
	for(int u = 0; u < MAX_UNITS; ++u)
		if(e.units[u] == active)
			e.activeSlot = u;
	if(e.activeSlot < 0)
		throw std::runtime_error("acting unit is not among the encoded units");

	const auto * hero = battle.battleGetMyHero();
	const bool heroCanCast = hero && battle.battleCanCastSpell(hero, spells::Mode::HERO) == ESpellCastProblem::OK;
	t.set(In::g_our_side, static_cast<int>(mySide));
	t.set(In::g_round, battle.battleGetRound());
	t.set(In::g_battlefield, battle.battleGetBattlefieldType().getNum());
	t.set(In::g_tactic_distance, battle.battleGetTacticDist());
	t.set(In::g_active_unit, e.activeSlot);
	t.set(In::g_hero_can_cast, heroCanCast);
	t.set(In::g_gate_state, static_cast<int>(battle.battleGetGateState()));
	t.set(In::g_surrender_cost, battle.battleGetSurrenderCost());
	for(int s = 0; s < N_SIDES; ++s)
	{
		t.set(In::g_start_army, s, std::llround(start.armyStart[s]));
		t.set(In::g_start_mana, s, start.manaStart[s]);
	}
	for(int u = 0; u < MAX_UNITS; ++u)
		if(e.units[u])
			t.set(In::u_kept, u, remainingAfterBattle(e.units[u]));
	encodePairDamage(e, battle);
	encodeHeroes(t, battle, mySide, enemyHero);

	std::array<bool, N_HEXES> stopping{};
	encodeHexes(t, battle, mySide, stopping);
	encodeUnitMasks(e, battle, active, stopping);
	if(heroCanCast)
		encodeSpellMasks(e, battle, hero, player);
	encodeCreatureSpell(e, battle, active, player);

	auto type = [&](ActionType a, bool legal) { t.set(In::m_type, static_cast<int>(a), legal); };
	type(ActionType::WAIT, !active->waitedThisTurn);
	type(ActionType::DEFEND, true);
	type(ActionType::HERO_SPELL, heroCanCast);
	type(ActionType::RETREAT, battle.battleCanFlee());
	// Surrender only when the hero can be bought back afterwards
	const int surrenderCost = battle.battleGetSurrenderCost();
	const bool canPay = surrenderCost >= 0 && player.isValidPlayer() && env.game()->getResource(player, EGameResID::GOLD) >= surrenderCost + GameConstants::HERO_GOLD_COST;
	type(ActionType::SURRENDER, canPay);
	refreshDerivedMasks(e);
	return e;
}

void refreshDerivedMasks(EncodedBattle & e)
{
	auto & t = e.tensors;
	for(int s = 0; s < N_SPELLS; ++s)
	{
		const auto second = secondTargetKind(s);
		for(int u = 0; u < MAX_UNITS && second != TargetKind::NONE; ++u)
			if(t.get(In::m_spell_unit, s, u) && !t.anyInRow(second == TargetKind::HEX ? In::m_spell2_hex : In::m_spell2_unit, u))
				t.set(In::m_spell_unit, s, u, 0);
		const bool hasTargets = t.anyInRow(In::m_spell_unit, s) || t.anyInRow(In::m_spell_hex, s);
		if(e.spellTargetKind[s] != TargetKind::NONE && !hasTargets)
			t.set(In::m_spell, s, 0);
	}
	for(int u = 0; u < MAX_UNITS; ++u)
		t.set(In::m_melee_target, u, t.anyInRow(In::m_melee_from, u));

	auto type = [&](ActionType a, bool legal) { t.set(In::m_type, static_cast<int>(a), legal); };
	type(ActionType::MOVE, t.any(In::m_move_hex));
	type(ActionType::MELEE, t.any(In::m_melee_target));
	type(ActionType::SHOOT, t.any(In::m_shoot_target));
	type(ActionType::HERO_SPELL, t.get(In::m_type, static_cast<int>(ActionType::HERO_SPELL)) && t.any(In::m_spell));
	type(ActionType::CREATURE_SPELL, t.any(In::m_creature_spell_unit) || t.any(In::m_creature_spell_hex));
}

void excludeAction(EncodedBattle & e, const ChosenAction & a)
{
	auto & t = e.tensors;
	auto clear = [&](In in, int i, int j = 0)
	{
		if(i >= 0 && j >= 0)
			t.set(in, i, j, 0);
	};
	switch(static_cast<ActionType>(a.type))
	{
		case ActionType::MOVE: clear(In::m_move_hex, a.hex); break;
		case ActionType::MELEE: clear(In::m_melee_from, a.unit, a.hex); break;
		case ActionType::SHOOT: clear(In::m_shoot_target, a.unit); break;
		case ActionType::HERO_SPELL:
			if(a.target2Hex >= 0)
				clear(In::m_spell2_hex, a.targetUnit, a.target2Hex);
			else if(a.target2Unit >= 0)
				clear(In::m_spell2_unit, a.targetUnit, a.target2Unit);
			else if(a.targetUnit >= 0)
				clear(In::m_spell_unit, a.spell, a.targetUnit);
			else if(a.targetHex >= 0)
				clear(In::m_spell_hex, a.spell, a.targetHex);
			else
				clear(In::m_spell, a.spell);
			break;
		case ActionType::CREATURE_SPELL:
			clear(In::m_creature_spell_unit, a.targetUnit);
			clear(In::m_creature_spell_hex, a.targetHex);
			break;
		default:
			if(a.type >= 0 && a.type < N_ACTION_TYPES)
				t.set(In::m_type, a.type, 0);
			break;
	}
	refreshDerivedMasks(e);
}

bool isLegal(const EncodedBattle & e, const ChosenAction & a)
{
	const auto & t = e.tensors;
	auto in = [](int v, int n) { return v >= 0 && v < n; };
	if(!in(a.type, N_ACTION_TYPES) || !t.get(In::m_type, a.type))
		return false;
	switch(static_cast<ActionType>(a.type))
	{
		case ActionType::MOVE: return in(a.hex, N_HEXES) && t.get(In::m_move_hex, a.hex);
		case ActionType::MELEE: return in(a.unit, MAX_UNITS) && in(a.hex, N_HEXES) && t.get(In::m_melee_target, a.unit) && t.get(In::m_melee_from, a.unit, a.hex);
		case ActionType::SHOOT: return in(a.unit, MAX_UNITS) && t.get(In::m_shoot_target, a.unit);
		case ActionType::HERO_SPELL:
		{
			if(!in(a.spell, N_SPELLS) || !t.get(In::m_spell, a.spell))
				return false;
			const auto kind = e.spellTargetKind[a.spell];
			if(kind == TargetKind::UNIT && !(in(a.targetUnit, MAX_UNITS) && t.get(In::m_spell_unit, a.spell, a.targetUnit)))
				return false;
			if(kind == TargetKind::HEX && !(in(a.targetHex, N_HEXES) && t.get(In::m_spell_hex, a.spell, a.targetHex)))
				return false;
			if(kind == TargetKind::NONE && (a.targetUnit >= 0 || a.targetHex >= 0))
				return false;
			const auto second = secondTargetKind(a.spell);
			if(second == TargetKind::HEX)
				return in(a.target2Hex, N_HEXES) && t.get(In::m_spell2_hex, a.targetUnit, a.target2Hex);
			if(second == TargetKind::UNIT)
				return in(a.target2Unit, MAX_UNITS) && t.get(In::m_spell2_unit, a.targetUnit, a.target2Unit);
			return a.target2Hex < 0 && a.target2Unit < 0;
		}
		case ActionType::CREATURE_SPELL:
			if(a.spell != t.get(In::g_creature_spell) || a.target2Hex >= 0 || a.target2Unit >= 0)
				return false;
			if(a.targetUnit >= 0)
				return a.targetHex < 0 && in(a.targetUnit, MAX_UNITS) && t.get(In::m_creature_spell_unit, a.targetUnit);
			return in(a.targetHex, N_HEXES) && t.get(In::m_creature_spell_hex, a.targetHex);
		default:
			return true;
	}
}

double heroLevel(const CGHeroInstance * hero)
{
	if(!hero)
		return 0.0;
	const auto & heroes = *LIBRARY->heroh;
	const auto level = hero->level;
	if(level >= heroes.maxSupportedLevel())
		return level;
	const double from = static_cast<double>(heroes.reqExp(level)), to = static_cast<double>(heroes.reqExp(level + 1));
	return level + (to > from ? std::clamp((static_cast<double>(hero->exp) - from) / (to - from), 0.0, 0.999) : 0.0);
}

double heroArtifacts(const CGHeroInstance * hero)
{
	if(!hero)
		return 0.0;
	double price = 0;
	for(const auto & slot : ArtifactUtils::commonWornSlots())
		if(const auto worn = hero->artifactsWorn.find(slot); worn != hero->artifactsWorn.end() && ArtifactUtils::isArtRemovable(*worn))
			price += worn->second.getArt()->getType()->getPrice();
	for(const auto & slot : hero->artifactsInBackpack)
		if(const auto * art = slot.getArt(); art && art->getTypeId() != ArtifactID::GRAIL)
			price += art->getType()->getPrice();
	return price;
}

bool keptAfterBattle(const CStack * unit)
{
	return !unit->isClone() && !unit->summoned && (unit->unitSlot().validSlot() || unit->unitSlot() == SlotID::SUMMONED_SLOT_PLACEHOLDER);
}

int remainingAfterBattle(const CStack * unit)
{
	return keptAfterBattle(unit) ? std::max(0, static_cast<int>(unit->unitBaseAmount()) - unit->getKilled()) : 0;
}

double creatureValue(const CCreature * creature)
{
	double value = 0;
	for(int r = 0; r < static_cast<int>(RESOURCE_PRICES.size()); ++r)
		value += RESOURCE_PRICES[r] * creature->getRecruitCost(GameResID(r));
	return value;
}

std::vector<ChosenAction> legalActions(const EncodedBattle & e)
{
	const auto & t = e.tensors;
	std::vector<ChosenAction> actions;
	for(int type = 0; type < N_ACTION_TYPES; ++type)
	{
		if(!t.get(In::m_type, type))
			continue;
		const ChosenAction base{.type = type};
		switch(static_cast<ActionType>(type))
		{
			case ActionType::MOVE:
				for(int h = 0; h < N_HEXES; ++h)
					if(t.get(In::m_move_hex, h))
						actions.push_back(ChosenAction{.type = type, .hex = h});
				break;
			case ActionType::MELEE:
				for(int u = 0; u < MAX_UNITS; ++u)
					if(t.get(In::m_melee_target, u))
						for(int h = 0; h < N_HEXES; ++h)
							if(t.get(In::m_melee_from, u, h))
								actions.push_back(ChosenAction{.type = type, .hex = h, .unit = u});
				break;
			case ActionType::SHOOT:
				for(int u = 0; u < MAX_UNITS; ++u)
					if(t.get(In::m_shoot_target, u))
						actions.push_back(ChosenAction{.type = type, .unit = u});
				break;
			case ActionType::HERO_SPELL:
				for(int s = 0; s < N_SPELLS; ++s)
				{
					if(!t.get(In::m_spell, s))
						continue;
					const ChosenAction spell{.type = type, .spell = s};
					if(t.anyInRow(In::m_spell_unit, s))
					{
						const auto second = secondTargetKind(s);
						for(int u = 0; u < MAX_UNITS; ++u)
						{
							if(!t.get(In::m_spell_unit, s, u))
								continue;
							auto first = spell;
							first.targetUnit = u;
							if(second == TargetKind::HEX)
							{
								for(int h = 0; h < N_HEXES; ++h)
									if(t.get(In::m_spell2_hex, u, h))
									{
										auto both = first;
										both.target2Hex = h;
										actions.push_back(both);
									}
							}
							else if(second == TargetKind::UNIT)
							{
								for(int v = 0; v < MAX_UNITS; ++v)
									if(t.get(In::m_spell2_unit, u, v))
									{
										auto both = first;
										both.target2Unit = v;
										actions.push_back(both);
									}
							}
							else
								actions.push_back(first);
						}
					}
					else if(t.anyInRow(In::m_spell_hex, s))
					{
						for(int h = 0; h < N_HEXES; ++h)
							if(t.get(In::m_spell_hex, s, h))
							{
								auto target = spell;
								target.targetHex = h;
								actions.push_back(target);
							}
					}
					else
						actions.push_back(spell);
				}
				break;
			case ActionType::CREATURE_SPELL:
			{
				const int spell = static_cast<int>(t.get(In::g_creature_spell));
				for(int u = 0; u < MAX_UNITS; ++u)
					if(t.get(In::m_creature_spell_unit, u))
						actions.push_back(ChosenAction{.type = type, .spell = spell, .targetUnit = u});
				for(int h = 0; h < N_HEXES; ++h)
					if(t.get(In::m_creature_spell_hex, h))
						actions.push_back(ChosenAction{.type = type, .spell = spell, .targetHex = h});
				break;
			}
			default:
				actions.push_back(base);
				break;
		}
	}
	return actions;
}

std::optional<ChosenAction> chosenAction(const EncodedBattle & e, const BattleAction & action)
{
	auto slotOf = [&](const BattleAction::DestinationInfo & d) -> int
	{
		for(int u = 0; u < MAX_UNITS; ++u)
		{
			const CStack * s = e.units[u];
			if(s && ((d.unitValue >= 0 && static_cast<int>(s->unitId()) == d.unitValue) || (d.unitValue < 0 && s->alive() && s->coversPos(d.hexValue))))
				return u;
		}
		return -1;
	};
	const auto & targets = action.target;
	ChosenAction a;
	switch(action.actionType)
	{
		case EActionType::WAIT: a.type = static_cast<int>(ActionType::WAIT); return a;
		case EActionType::DEFEND: a.type = static_cast<int>(ActionType::DEFEND); return a;
		case EActionType::WALK:
			if(targets.empty())
				return std::nullopt;
			a.type = static_cast<int>(ActionType::MOVE);
			a.hex = targets[0].hexValue.toInt();
			return a;
		case EActionType::WALK_AND_ATTACK: // makeMeleeAttack: the hex to attack from, then the target
			if(targets.size() < 2)
				return std::nullopt;
			a.type = static_cast<int>(ActionType::MELEE);
			a.hex = targets[0].hexValue.toInt();
			a.unit = slotOf(targets[1]);
			return a.unit >= 0 ? std::optional(a) : std::nullopt;
		case EActionType::SHOOT:
			if(targets.empty())
				return std::nullopt;
			a.type = static_cast<int>(ActionType::SHOOT);
			a.unit = slotOf(targets[0]);
			return a.unit >= 0 ? std::optional(a) : std::nullopt;
		case EActionType::MONSTER_SPELL: // another spell than the one encoded (another Faerie Dragon draw) has no model action
			if(targets.empty() || action.spell.getNum() != e.tensors.get(In::g_creature_spell))
				return std::nullopt;
			a.type = static_cast<int>(ActionType::CREATURE_SPELL);
			a.spell = action.spell.getNum();
			if(e.tensors.any(In::m_creature_spell_unit))
			{
				a.targetUnit = slotOf(targets[0]);
				// Aimed at a hex only: the legal target there, a corpse included (Resurrection)
				for(int u = 0; u < MAX_UNITS && a.targetUnit < 0; ++u)
					if(e.tensors.get(In::m_creature_spell_unit, u) && e.units[u]->coversPos(targets[0].hexValue))
						a.targetUnit = u;
			}
			else
				a.targetHex = targets[0].hexValue.toInt();
			return isLegal(e, a) ? std::optional(a) : std::nullopt;
		default:
			return std::nullopt;
	}
}

BattleAction engineAction(BattleSide side, const CStack * active, const ChosenAction & a, const std::function<const battle::Unit *(int)> & unit)
{
	auto target = [&](int i)
	{
		const auto * u = unit(i);
		if(!u)
			throw std::runtime_error("action refers to an unknown unit " + std::to_string(i));
		return u;
	};

	switch(static_cast<ActionType>(a.type))
	{
		case ActionType::WAIT:
			return BattleAction::makeWait(active);
		case ActionType::DEFEND:
			return BattleAction::makeDefend(active);
		case ActionType::MOVE:
			return BattleAction::makeMove(active, BattleHex(a.hex));
		case ActionType::MELEE:
			return BattleAction::makeMeleeAttack(active, target(a.unit), BattleHex(a.hex));
		case ActionType::SHOOT:
			return BattleAction::makeShotAttack(active, target(a.unit));
		case ActionType::RETREAT:
			return BattleAction::makeRetreat(side);
		case ActionType::SURRENDER:
			return BattleAction::makeSurrender(side);
		case ActionType::HERO_SPELL:
		{
			battle::Target destinations;
			if(a.targetUnit >= 0)
				destinations.emplace_back(target(a.targetUnit));
			if(a.targetHex >= 0)
				destinations.emplace_back(BattleHex(a.targetHex));
			if(a.target2Hex >= 0)
				destinations.emplace_back(BattleHex(a.target2Hex));
			if(a.target2Unit >= 0)
				destinations.emplace_back(target(a.target2Unit));
			BattleAction cast;
			cast.actionType = EActionType::HERO_SPELL;
			cast.spell = SpellID(a.spell);
			cast.side = side;
			cast.stackNumber = -1;
			cast.setTarget(destinations);
			return cast;
		}
		case ActionType::CREATURE_SPELL: // spell -1: a random caster's, which the server draws
		{
			battle::Target destinations;
			if(a.targetUnit >= 0)
				destinations.emplace_back(target(a.targetUnit));
			else
				destinations.emplace_back(BattleHex(a.targetHex));
			return BattleAction::makeCreatureSpellcast(active, destinations, SpellID(a.spell));
		}
		default:
			throw std::runtime_error("model chose action type " + std::to_string(a.type) + ", which is not supported");
	}
}
