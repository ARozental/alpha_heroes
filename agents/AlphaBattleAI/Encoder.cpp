#include "StdInc.h"
#include "Encoder.h"

#include "lib/CStack.h"
#include "lib/battle/CObstacleInstance.h"
#include "lib/battle/CPlayerBattleCallback.h"
#include "lib/bonuses/Bonus.h"
#include "lib/bonuses/BonusSelector.h"
#include "lib/callback/IGameInfoCallback.h"
#include "lib/constants/NumericConstants.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/spells/CSpellHandler.h"
#include "lib/spells/ISpellMechanics.h"

#include <vcmi/Environment.h>
#include <vcmi/spells/Magic.h>

using namespace AlphaSchema;

namespace
{
std::vector<const CStack *> pickUnits(const CPlayerBattleCallback & battle, const CStack * active)
{
	auto stacks = battle.battleGetAllStacks(true);
	std::sort(stacks.begin(), stacks.end(), [](const CStack * a, const CStack * b) { return a->unitId() < b->unitId(); });
	// Over capacity: drop dead stacks first, never the acting one
	while(stacks.size() > static_cast<size_t>(MAX_UNITS))
	{
		auto dead = std::find_if(stacks.rbegin(), stacks.rend(), [&](const CStack * s) { return !s->alive() && s != active; });
		if(dead != stacks.rend())
			stacks.erase(std::next(dead).base());
		else if(stacks.back() != active)
			stacks.pop_back();
		else
			stacks.erase(stacks.end() - 2);
	}
	return stacks;
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
		t.set(In::u_morale, u, s->valOfBonuses(BonusType::MORALE));
		t.set(In::u_luck, u, s->valOfBonuses(BonusType::LUCK));
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

void encodeHeroes(InputTensors & t, const CPlayerBattleCallback & battle, BattleSide mySide)
{
	for(int side = 0; side < N_SIDES; ++side)
	{
		const auto * hero = battle.battleGetFightingHero(static_cast<BattleSide>(side));
		if(!hero)
			continue;
		t.set(In::h_present, side, 1);
		t.set(In::h_type, side, hero->getHeroTypeID().getNum());
		t.set(In::h_attack, side, hero->getPrimSkillLevel(PrimarySkill::ATTACK));
		t.set(In::h_defense, side, hero->getPrimSkillLevel(PrimarySkill::DEFENSE));
		t.set(In::h_power, side, hero->getPrimSkillLevel(PrimarySkill::SPELL_POWER));
		t.set(In::h_knowledge, side, hero->getPrimSkillLevel(PrimarySkill::KNOWLEDGE));
		t.set(In::h_mana, side, hero->mana);
		t.set(In::h_mana_max, side, hero->manaLimit());
		for(int skill = 0; skill < N_SECONDARY_SKILLS; ++skill)
			t.set(In::h_skills, side, skill, hero->getSecSkillLevel(SecondarySkill(skill)));
		// A player cannot see the enemy's spellbook
		if(side == static_cast<int>(mySide) && hero->hasSpellbook())
			for(int spell = 0; spell < N_SPELLS; ++spell)
				t.set(In::h_spells, side, spell, hero->spellbookContainsSpell(SpellID(spell)));
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
		{
			const CStack * target = e.units[u];
			if(target && target->alive() && target->unitSide() != active->unitSide() && CStack::isMeleeAttackPossible(active, target, hex))
				t.set(In::m_melee_from, u, h, 1);
		}
	}
	for(int u = 0; u < MAX_UNITS; ++u)
	{
		const CStack * target = e.units[u];
		if(t.anyInRow(In::m_melee_from, u))
			t.set(In::m_melee_target, u, 1);
		if(target && target->alive() && target->unitSide() != active->unitSide() && battle.battleCanShoot(active, target->getPosition()))
			t.set(In::m_shoot_target, u, 1);
	}
}

void encodeSpellMasks(EncodedBattle & e, const CPlayerBattleCallback & battle, const CGHeroInstance * hero)
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

		if(aims[0] == spells::AimType::NO_TARGET)
		{
			if(mechanics->canBeCastAt(spells::Target()))
				t.set(In::m_spell, s, 1);
			continue;
		}
		if(aims[0] == spells::AimType::LOCATION)
		{
			for(int h = 0; h < N_HEXES; ++h)
				if(mechanics->canBeCastAt({battle::Destination(BattleHex(h))}))
					t.set(In::m_spell_hex, s, h, 1);
		}
		else if(aims[0] == spells::AimType::CREATURE)
		{
			for(int u = 0; u < MAX_UNITS; ++u)
			{
				const CStack * unit = e.units[u];
				if(!unit)
					continue;
				const battle::Destination first(unit);
				if(second == TargetKind::NONE)
				{
					if(mechanics->canBeCastAt({first}))
						t.set(In::m_spell_unit, s, u, 1);
					continue;
				}
				if(spell2Owner[u] >= 0)
					continue; // second-target rows are per first unit; already used by another spell
				bool anySecond = false;
				if(second == TargetKind::HEX && aims[1] == spells::AimType::LOCATION)
				{
					for(int h = 0; h < N_HEXES; ++h)
						if(mechanics->canBeCastAt({first, battle::Destination(BattleHex(h))}))
						{
							t.set(In::m_spell2_hex, u, h, 1);
							anySecond = true;
						}
				}
				else if(second == TargetKind::UNIT && aims[1] == spells::AimType::CREATURE)
				{
					for(int v = 0; v < MAX_UNITS; ++v)
						if(e.units[v] && mechanics->canBeCastAt({first, battle::Destination(e.units[v])}))
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

EncodedBattle encodeBattle(const CPlayerBattleCallback & battle, const CStack * active, const Environment & env, PlayerColor player)
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
	encodeHeroes(t, battle, mySide);

	std::array<bool, N_HEXES> stopping{};
	encodeHexes(t, battle, mySide, stopping);
	encodeUnitMasks(e, battle, active, stopping);
	if(heroCanCast)
		encodeSpellMasks(e, battle, hero);

	auto type = [&](ActionType a, bool legal) { t.set(In::m_type, static_cast<int>(a), legal); };
	type(ActionType::WAIT, !active->waitedThisTurn);
	type(ActionType::DEFEND, true);
	type(ActionType::MOVE, t.any(In::m_move_hex));
	type(ActionType::MELEE, t.any(In::m_melee_target));
	type(ActionType::SHOOT, t.any(In::m_shoot_target));
	type(ActionType::HERO_SPELL, heroCanCast && t.any(In::m_spell));
	type(ActionType::RETREAT, battle.battleCanFlee());
	// Surrender only when the hero can be bought back afterwards
	const int surrenderCost = battle.battleGetSurrenderCost();
	const bool canPay = surrenderCost >= 0 && player.isValidPlayer() && env.game()->getResource(player, EGameResID::GOLD) >= surrenderCost + GameConstants::HERO_GOLD_COST;
	type(ActionType::SURRENDER, canPay);
	return e;
}
