#include "StdInc.h"
#include "AlphaMapAI.h"
#include "Ledger.h"

#include "lib/GameConstants.h"
#include "lib/IGameSettings.h"
#include "lib/ResourceSet.h"
#include "lib/bonuses/BonusCustomTypes.h"
#include "lib/callback/CCallback.h"
#include "lib/entities/artifact/CArtifact.h"
#include "lib/entities/artifact/CArtifactInstance.h"
#include "lib/entities/artifact/ArtifactUtils.h"
#include "lib/entities/building/CBuilding.h"
#include "lib/entities/faction/CTown.h"
#include "lib/entities/hero/CHeroClass.h"
#include "lib/gameState/InfoAboutArmy.h"
#include "lib/gameState/UpgradeInfo.h"
#include "lib/mapObjects/CGCreature.h"
#include "lib/mapObjects/CGDwelling.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/mapObjects/CGTownInstance.h"
#include "lib/mapObjects/IMarket.h"
#include "lib/mapObjects/IOwnableObject.h"
#include "lib/mapping/TerrainTile.h"
#include "lib/pathfinder/CGPathNode.h"
#include "lib/pathfinder/PathfinderCache.h"
#include "lib/pathfinder/PathfinderOptions.h"
#include "lib/spells/CSpellHandler.h"
#include "lib/spells/ISpellMechanics.h"
#include "lib/spells/Problem.h"
#include "lib/GameLibrary.h"
#include "lib/networkPacks/Component.h"
#include "lib/texts/CGeneralTextHandler.h"

using namespace MapSchema;

namespace
{
int64_t clampInt(int64_t value)
{
	return std::clamp<int64_t>(value, -2000000000, 2000000000);
}

const CArmedInstance * armyOf(const CCallback & cb, ObjectInstanceID id)
{
	return id == ObjectInstanceID() ? nullptr : dynamic_cast<const CArmedInstance *>(cb.getObj(id, false));
}

/// The object a tile shows: its top visitable object other than `except` (the hero looking), else what blocks it
/// Wandering monsters' count as the hero sees it (CGCreature::getHoverText): VCMI's quantity id (few, several, pack,
/// ...), or the exact count when the hero's Visions reach them; and whether it is exact. Never more: the exact count
/// is what Nullkiller2 reads, and that is cheating (the user, 2026-10-27)
std::pair<int, bool> seenCount(const CGObjectInstance * monster, const CGHeroInstance * hero)
{
	const auto * army = dynamic_cast<const CArmedInstance *>(monster);
	const auto * stack = army ? army->getStackPtr(SlotID(0)) : nullptr;
	if(!stack)
		return {0, false};
	if(hero && hero->hasVisions(monster, BonusCustomSubtype::visionsMonsters))
		return {stack->getCount(), true};
	return {static_cast<int>(stack->getQuantityID()), false};
}

/// The object a map tile shows for no hero in particular (the tile grid's): its top visitable object other than a hero
/// (heroes and towns come in as their tokens), else what blocks it
const CGObjectInstance * topNonHero(const CCallback & cb, const int3 & pos)
{
	auto visitable = cb.getVisitableObjs(pos, false);
	for(auto it = visitable.rbegin(); it != visitable.rend(); ++it)
		if((*it)->ID != Obj::HERO)
			return *it;
	for(const auto * obj : cb.getBlockingObjs(pos))
		if(obj->ID != Obj::HERO)
			return obj;
	return nullptr;
}

/// Whether a question's text is VCMI's template's (a CGCreature dialog: it fills the template's %s and %d): the
/// template's longest fixed piece is in it
bool fromTemplate(const std::string & text, const std::string & templ)
{
	std::string longest, piece;
	for(size_t i = 0; i <= templ.size(); ++i)
	{
		if(i == templ.size() || (templ[i] == '%' && i + 1 < templ.size() && (templ[i + 1] == 's' || templ[i + 1] == 'd')))
		{
			if(piece.size() > longest.size())
				longest = piece;
			piece.clear();
			++i;
			continue;
		}
		piece += templ[i];
	}
	return longest.size() >= 8 && text.find(longest) != std::string::npos;
}

const CGObjectInstance * topObject(const CCallback & cb, const int3 & pos, const CGObjectInstance * except)
{
	auto visitable = cb.getVisitableObjs(pos, false);
	for(auto it = visitable.rbegin(); it != visitable.rend(); ++it)
		if(*it != except)
			return *it;
	auto blocking = cb.getBlockingObjs(pos);
	return blocking.empty() ? nullptr : blocking.front();
}

int distanceSq(const int3 & a, const int3 & b)
{
	return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z == b.z ? 0 : 10000);
}
}

std::optional<std::vector<Trade>> tradesFor(const IMarket * market, const TResources & cost, const TResources & have, const std::array<float, N_RESOURCES> & worth)
{
	if(!market)
		return std::nullopt;
	std::array<int, N_RESOURCES> need{};
	std::array<int, N_RESOURCES> spare{};
	for(int r = 0; r < N_RESOURCES; ++r)
	{
		need[r] = std::max(0, cost[GameResID(r)] - have[GameResID(r)]);
		spare[r] = std::max(0, have[GameResID(r)] - cost[GameResID(r)]);
	}
	std::vector<Trade> trades;
	for(int buy = 0; buy < N_RESOURCES; ++buy)
		while(need[buy] > 0)
		{
			int best = -1;
			int bestGive = 0;
			int bestGet = 0;
			float bestLoss = 0;
			for(int sell = 0; sell < N_RESOURCES; ++sell)
			{
				int give = 0;
				int get = 0;
				if(sell == buy || !market->getOffer(sell, buy, give, get, EMarketMode::RESOURCE_RESOURCE) || give <= 0 || get <= 0 || spare[sell] < give)
					continue;
				const float loss = worth[sell] * give / get; // (what a unit bought gives away)
				if(best < 0 || loss < bestLoss)
					std::tie(best, bestGive, bestGet, bestLoss) = std::tuple(sell, give, get, loss);
			}
			if(best < 0)
				return std::nullopt;
			const int lots = std::min((need[buy] + bestGet - 1) / bestGet, spare[best] / bestGive);
			trades.push_back(Trade{best, buy, lots * bestGive});
			spare[best] -= lots * bestGive;
			need[buy] -= lots * bestGet;
		}
	return trades;
}

std::array<float, N_RESOURCES> marketWorth(const IMarket * market)
{
	std::array<float, N_RESOURCES> worth{};
	for(int r = 0; r < N_RESOURCES; ++r)
	{
		int give = 0;
		int get = 0;
		if(r == EGameResID::GOLD)
			worth[r] = 1.0f;
		else if(market && market->getOffer(r, EGameResID::GOLD, give, get, EMarketMode::RESOURCE_RESOURCE) && give > 0)
			worth[r] = static_cast<float>(get) / give;
	}
	return worth;
}

const IMarket * AlphaMapAI::ourMarket() const
{
	for(const auto * town : cb->getTownsInfo(true))
		if(town->hasBuiltResourceMarketplace())
			return cb->getMarket(town->id);
	return nullptr;
}

int3 Decision::windowTile(int hero, int row, int column) const
{
	const auto & center = centers[hero];
	return int3(center.x + column - WINDOW_RADIUS, center.y + row - WINDOW_RADIUS, center.z);
}

int Decision::holderOf(ObjectInstanceID id) const
{
	if(id == ObjectInstanceID())
		return -1;
	for(int i = 0; i < N_HOLDERS; ++i)
		if(holders[i] == id)
			return i;
	return -1;
}

int AlphaMapAI::owner(PlayerColor color) const
{
	if(color == playerID)
		return OWNER_SELF;
	if(!color.isValidPlayer())
		return OWNER_NONE;
	return OWNER_SELF + 1 + (color.getNum() - playerID.getNum() - 1 + PlayerColor::PLAYER_LIMIT_I) % PlayerColor::PLAYER_LIMIT_I;
}

Decision AlphaMapAI::encode(DecisionKind kind, ObjectInstanceID exchangeFirst, ObjectInstanceID exchangeSecond)
{
	Decision decision;
	decision.kind = kind;
	decision.inputs.at(In::g_kind) = static_cast<int>(kind);

	// Token slots: ours by id, the others we see by distance to our nearest hero or town
	auto ownHeroes = cb->getHeroesInfo();
	std::sort(ownHeroes.begin(), ownHeroes.end(), [](auto * a, auto * b) { return a->id < b->id; });
	auto ownTowns = cb->getTownsInfo(true);
	std::sort(ownTowns.begin(), ownTowns.end(), [](auto * a, auto * b) { return a->id < b->id; });
	std::vector<int3> ours;
	for(const auto * h : ownHeroes)
		ours.push_back(h->visitablePos());
	for(const auto * t : ownTowns)
		ours.push_back(t->visitablePos());
	auto nearness = [&ours](const CGObjectInstance * obj)
	{
		int best = std::numeric_limits<int>::max();
		for(const auto & pos : ours)
			best = std::min(best, distanceSq(pos, obj->visitablePos()));
		return best;
	};
	std::vector<const CGObjectInstance *> otherHeroes;
	for(const auto * obj : cb->getAllVisitableObjs())
		if(obj->ID == Obj::HERO && obj->tempOwner != playerID)
			otherHeroes.push_back(obj);
	std::sort(otherHeroes.begin(), otherHeroes.end(), [&](auto * a, auto * b) { return nearness(a) < nearness(b); });
	std::vector<const CGTownInstance *> otherTowns;
	for(const auto * town : cb->getTownsInfo(false))
		if(town->tempOwner != playerID)
			otherTowns.push_back(town);
	std::sort(otherTowns.begin(), otherTowns.end(), [&](auto * a, auto * b) { return nearness(a) < nearness(b); });

	for(int i = 0; i < std::min<int>(ownHeroes.size(), OWN_HEROES); ++i)
		decision.heroes[i] = ownHeroes[i]->id;
	for(int i = 0; i < std::min<int>(otherHeroes.size(), OTHER_HEROES); ++i)
		decision.heroes[OWN_HEROES + i] = otherHeroes[i]->id;
	for(int i = 0; i < std::min<int>(ownTowns.size(), OWN_TOWNS); ++i)
		decision.towns[i] = ownTowns[i]->id;
	for(int i = 0; i < std::min<int>(otherTowns.size(), OTHER_TOWNS); ++i)
		decision.towns[OWN_TOWNS + i] = otherTowns[i]->id;
	for(int i = 0; i < OWN_HEROES; ++i)
		decision.holders[i] = decision.heroes[i];
	for(int i = 0; i < OWN_TOWNS; ++i)
		decision.holders[OWN_HEROES + i] = decision.towns[i];

	PathfinderCache paths(cb.get(), PathfinderOptions(*cb));
	encodeGlobal(decision);
	encodeTowns(decision);
	encodeHeroes(decision, paths);
	encodeWindows(decision, paths);
	encodeMap(decision, ours);
	encodeTiles(decision);
	if(kind == DecisionKind::TURN)
	{
		encodeTurnActions(decision, paths);
		encodeArtifacts(decision, -1, -1);
	}
	if(kind == DecisionKind::EXCHANGE)
	{
		// The two armies that may trade: our heroes or towns, or the dialog's other army (OTHER_ARMY)
		auto holder = [&](ObjectInstanceID id)
		{
			const int found = decision.holderOf(id);
			if(found >= 0)
				return found;
			decision.holders[OTHER_ARMY] = id;
			if(const auto * army = armyOf(*cb, id))
				for(int s = 0; s < ARMY_SLOTS; ++s)
					if(const auto * stack = army->getStackPtr(SlotID(s)))
					{
						decision.inputs.at(In::x_creature, s) = stack->getId().getNum();
						decision.inputs.at(In::x_count, s) = stack->getCount();
					}
			return static_cast<int>(OTHER_ARMY);
		};
		const int a = holder(exchangeFirst);
		const int b = holder(exchangeSecond);
		encodeArmyMoves(decision, {{a, b}, {b, a}});
		if(a < OWN_HEROES && b < OWN_HEROES) // (two heroes of ours: their artifacts may change hands)
		{
			decision.inputs.at(In::x_heroes, 0) = a + 1;
			decision.inputs.at(In::x_heroes, 1) = b + 1;
			encodeArtifacts(decision, a, b);
		}
	}
	dropFutile(decision);
	return decision;
}

void AlphaMapAI::dropFutile(Decision & decision)
{
	for(const auto & [type, index, cell] : futile)
	{
		const auto mask = head(static_cast<ActionType>(type)).second;
		if(mask == In::COUNT)
			continue;
		auto & values = decision.inputs.values[static_cast<size_t>(mask)];
		if(type == static_cast<int>(ActionType::CAST) && cell >= 0)
		{
			// A targeted spell that changed nothing there: that tile goes, the spell stays while another tile is left
			auto & cells = decision.inputs.values[static_cast<size_t>(In::cast_cell_mask)];
			const size_t first = static_cast<size_t>(index / N_ADVENTURE_SPELLS) * WINDOW * WINDOW;
			if(first + cell >= cells.size())
				continue;
			cells[first + cell] = 0;
			if(std::any_of(cells.begin() + first, cells.begin() + first + WINDOW * WINDOW, [](auto v) { return v != 0; }))
				continue;
		}
		if(index >= 0 && index < static_cast<int>(values.size()))
			values[index] = 0;
		decision.entries[type].erase(index);
	}
}

void AlphaMapAI::encodeGlobal(Decision & decision)
{
	auto & in = decision.inputs;
	in.at(In::g_day) = cb->getDate(Date::DAY);
	in.at(In::g_last_type) = lastChoice.type;
	in.at(In::g_last_index) = lastChoice.index;
	in.at(In::g_last_changed) = lastChanged;
	const auto resources = cb->getResourceAmount();
	for(int r = 0; r < N_RESOURCES; ++r)
		in.at(In::g_resources, r) = resources[GameResID(r)];
	ResourceSet income, objectIncome; // all of it, and the mines' and other objects' (not towns or heroes: the ledger's)
	for(const auto * obj : cb->getMyObjects())
		if(const auto * ownable = obj->asOwnable())
		{
			income += ownable->dailyIncome();
			if(obj->ID != Obj::HERO && obj->ID != Obj::TOWN)
				objectIncome += ownable->dailyIncome();
		}
	for(int r = 0; r < N_RESOURCES; ++r)
	{
		in.at(In::g_income, r) = income[GameResID(r)];
		in.at(In::g_object_income, r) = objectIncome[GameResID(r)];
	}
	// Our other armies and creatures for hire, as the ledger counts them (Ledger.cpp): the garrisons of our mines and
	// other owned objects, the creatures in our dwellings outside towns
	int owned = 0, hire = 0;
	for(const auto * obj : cb->getMyObjects())
	{
		if(obj->ID == Obj::HERO || obj->ID == Obj::TOWN)
			continue;
		if(const auto * dwelling = dynamic_cast<const CGDwelling *>(obj))
			for(const auto & [count, ids] : dwelling->creatures)
				if(count > 0 && !ids.empty() && hire < OWNED_STACKS)
				{
					in.at(In::g_dwelling_creature, hire) = ids.back().getNum();
					in.at(In::g_dwelling_count, hire++) = count;
				}
		if(const auto * army = dynamic_cast<const CArmedInstance *>(obj))
			for(int s = 0; s < ARMY_SLOTS; ++s)
				if(const auto * stack = army->getStackPtr(SlotID(s)); stack && owned < OWNED_STACKS)
				{
					in.at(In::g_owned_creature, owned) = stack->getId().getNum();
					in.at(In::g_owned_count, owned++) = stack->getCount();
				}
	}
	const auto size = cb->getMapSize();
	in.at(In::g_map, 0) = size.x;
	in.at(In::g_map, 1) = size.y;
	in.at(In::g_map, 2) = size.z;
	int players = 0;
	for(int p = 0; p < PlayerColor::PLAYER_LIMIT_I; ++p)
		if(cb->getPlayerStatus(PlayerColor(p), false) == EPlayerStatus::INGAME)
			++players;
	in.at(In::g_players) = players;
	in.at(In::g_decisions_today) = decisionsToday;
	int64_t revealed = 0;
	for(int z = 0; z < size.z; ++z)
		for(int y = 0; y < size.y; ++y)
			for(int x = 0; x < size.x; ++x)
				revealed += cb->isVisible(int3(x, y, z)) ? 1 : 0;
	in.at(In::g_explored) = revealed;
	int markets = 0;
	const CGTownInstance * tavern = nullptr;
	for(const auto * town : cb->getTownsInfo(true))
	{
		markets += town->hasBuiltResourceMarketplace() ? 1 : 0;
		if(!tavern && town->hasBuilt(BuildingID::TAVERN))
			tavern = town;
	}
	in.at(In::g_markets) = markets;
	if(tavern)
	{
		const auto heroes = cb->getAvailableHeroes(tavern);
		for(int i = 0; i < std::min<int>(heroes.size(), N_TAVERN); ++i)
		{
			const auto * h = heroes[i];
			in.at(In::g_tavern, i) = h->getHeroTypeID().getNum();
			in.at(In::g_tavern_level, i) = h->level;
			in.at(In::g_tavern_exp, i) = h->exp;
			for(int k = 0; k < N_PRIMARY; ++k)
				in.at(In::g_tavern_primary, i, k) = h->getBasePrimarySkillValue(PrimarySkill(k));
			for(const auto & [skill, level] : h->secSkills)
				if(skill.getNum() >= 0 && skill.getNum() < N_SECONDARY_SKILLS)
					in.at(In::g_tavern_skills, i, skill.getNum()) = level;
			// What the hire brings (as the tavern window shows it): its army and worn artifacts
			for(int s = 0; s < ARMY_SLOTS; ++s)
				if(const auto * stack = h->getStackPtr(SlotID(s)))
				{
					in.at(In::g_tavern_army_creature, i, s) = stack->getId().getNum();
					in.at(In::g_tavern_army_count, i, s) = stack->getCount();
				}
			for(int slot = 0; slot < N_WORN_SLOTS; ++slot)
				if(const auto * art = h->getArt(ArtifactPosition(slot)))
					in.at(In::g_tavern_artifacts, i, slot) = art->getTypeId().getNum();
			if(h->hasSpellbook())
				for(const auto & spell : h->getSpellsInSpellbook())
					if(spell.getNum() >= 0 && spell.getNum() < N_SPELLS)
						in.at(In::g_tavern_spells, i, spell.getNum()) = 1;
		}
	}
}

void AlphaMapAI::encodeTowns(Decision & decision)
{
	auto & in = decision.inputs;
	auto heroToken = [&decision](const CGHeroInstance * hero)
	{
		if(!hero)
			return -1;
		for(int i = 0; i < N_HEROES; ++i)
			if(decision.heroes[i] == hero->id)
				return i;
		return -1;
	};
	for(int i = 0; i < N_TOWNS; ++i)
	{
		const auto * town = cb->getTown(decision.towns[i]);
		if(decision.towns[i] == ObjectInstanceID() || !town)
			continue;
		in.at(In::t_present, i) = 1;
		in.at(In::t_owner, i) = owner(town->tempOwner);
		in.at(In::t_faction, i) = town->getFactionID().getNum();
		const auto pos = town->visitablePos();
		in.at(In::t_pos, i, 0) = pos.x;
		in.at(In::t_pos, i, 1) = pos.y;
		in.at(In::t_pos, i, 2) = pos.z;
		if(town->tempOwner == playerID)
		{
			for(const auto & building : town->getBuildings())
				if(building.getNum() >= 0 && building.getNum() < N_BUILDINGS)
					in.at(In::t_built, i, building.getNum()) = 1;
			in.at(In::t_fort, i) = static_cast<int>(town->fortLevel());
			in.at(In::t_built_today, i) = town->built;
			in.at(In::t_detailed, i) = 1;
			for(int s = 0; s < ARMY_SLOTS; ++s)
				if(const auto * stack = town->getStackPtr(SlotID(s)))
				{
					in.at(In::t_garrison_creature, i, s) = stack->getId().getNum();
					in.at(In::t_garrison_count, i, s) = stack->getCount();
				}
			for(int level = 0; level < std::min<int>(town->creatures.size(), N_LEVELS); ++level)
			{
				const auto & [count, ids] = town->creatures[level];
				in.at(In::t_available, i, level) = count;
				in.at(In::t_available_creature, i, level) = ids.empty() ? -1 : ids.back().getNum();
				in.at(In::t_growth, i, level) = town->creatureGrowth(level);
			}
			in.at(In::t_visiting, i) = heroToken(town->getVisitingHero());
			in.at(In::t_garrison_hero, i) = heroToken(town->getGarrisonHero());
		}
		else
		{
			InfoAboutTown info;
			if(cb->getTownInfo(town, info))
			{
				in.at(In::t_fort, i) = info.fortLevel;
				in.at(In::t_detailed, i) = info.army.isDetailed ? 1 : 0; // else counts are quantity ids
				for(const auto & [slot, stack] : info.army)
					if(slot.getNum() >= 0 && slot.getNum() < ARMY_SLOTS && stack.getId() != CreatureID::NONE)
					{
						in.at(In::t_garrison_creature, i, slot.getNum()) = stack.getId().getNum();
						in.at(In::t_garrison_count, i, slot.getNum()) = stack.getCount();
					}
			}
		}
	}
}

void AlphaMapAI::encodeHeroes(Decision & decision, PathfinderCache & paths)
{
	auto & in = decision.inputs;
	auto townToken = [&decision](const CGTownInstance * town)
	{
		if(!town)
			return -1;
		for(int i = 0; i < N_TOWNS; ++i)
			if(decision.towns[i] == town->id)
				return i;
		return -1;
	};
	for(int i = 0; i < N_HEROES; ++i)
	{
		if(decision.heroes[i] == ObjectInstanceID())
			continue;
		const auto * obj = cb->getObj(decision.heroes[i], false);
		if(!obj)
			continue;
		const auto pos = obj->visitablePos();
		in.at(In::h_present, i) = 1;
		in.at(In::h_owner, i) = owner(obj->tempOwner);
		in.at(In::h_pos, i, 0) = pos.x;
		in.at(In::h_pos, i, 1) = pos.y;
		in.at(In::h_pos, i, 2) = pos.z;
		const auto * h = i < OWN_HEROES ? cb->getHero(decision.heroes[i]) : nullptr;
		if(h)
		{
			in.at(In::h_type, i) = h->getHeroTypeID().getNum();
			in.at(In::h_class, i) = h->getHeroClassID().getNum();
			in.at(In::h_level, i) = h->level;
			in.at(In::h_exp, i) = clampInt(h->exp);
			for(int k = 0; k < N_PRIMARY; ++k)
			{
				in.at(In::h_primary, i, k) = h->getPrimSkillLevel(PrimarySkill(k));
				in.at(In::h_primary_base, i, k) = h->getBasePrimarySkillValue(PrimarySkill(k));
			}
			in.at(In::h_mana, i) = h->mana;
			in.at(In::h_mana_max, i) = h->manaLimit();
			int dearest = 0; // the dearest combat spell it can cast (as the ledger's spell_cost)
			if(h->hasSpellbook())
				for(const auto & spellID : h->getSpellsInSpellbook())
				{
					const auto * spell = spellID.toSpell();
					if(spell && spell->isCombat() && h->canCastThisSpell(spell))
						dearest = std::max(dearest, h->getSpellCost(spell));
				}
			in.at(In::h_spell_cost, i) = dearest;
			in.at(In::h_move, i) = h->movementPointsRemaining();
			in.at(In::h_move_max, i) = h->movementPointsLimit(!h->inBoat());
			for(int s = 0; s < ARMY_SLOTS; ++s)
				if(const auto * stack = h->getStackPtr(SlotID(s)))
				{
					in.at(In::h_army_creature, i, s) = stack->getId().getNum();
					in.at(In::h_army_count, i, s) = stack->getCount();
				}
			for(int slot = 0; slot < N_WORN_SLOTS; ++slot)
				if(const auto * art = h->getArt(ArtifactPosition(slot)))
					in.at(In::h_artifacts, i, slot) = art->getTypeId().getNum();
			for(int b = 0; b < std::min<int>(h->artifactsInBackpack.size(), N_BACKPACK); ++b)
				if(const auto * art = h->artifactsInBackpack[b].getArt())
					in.at(In::h_backpack, i, b) = art->getTypeId().getNum();
			for(const auto & [skill, level] : h->secSkills)
				if(skill.getNum() >= 0 && skill.getNum() < N_SECONDARY_SKILLS)
					in.at(In::h_skills, i, skill.getNum()) = level;
			if(h->hasSpellbook())
				for(const auto & spell : h->getSpellsInSpellbook())
					if(spell.getNum() >= 0 && spell.getNum() < N_SPELLS)
						in.at(In::h_spells, i, spell.getNum()) = 1;
			const auto boosts = luckAndMorale(*h); // (Ledger.cpp: as the ledger has them)
			in.at(In::h_luck, i) = boosts.luck;
			in.at(In::h_morale, i) = boosts.morale;
			in.at(In::h_battle_luck, i) = boosts.battleLuck;
			in.at(In::h_battle_morale, i) = boosts.battleMorale;
			in.at(In::h_boat, i) = h->inBoat() ? 1 : 0;
			in.at(In::h_town, i) = townToken(h->getVisitedTown());
			in.at(In::h_garrisoned, i) = h->isGarrisoned() ? 1 : 0;
			in.at(In::h_detailed, i) = 1;
		}
		else
		{
			InfoAboutHero info;
			if(cb->getHeroInfo(obj, info))
			{
				in.at(In::h_type, i) = info.portraitSource.getNum();
				in.at(In::h_class, i) = info.hclass ? info.hclass->getIndex() : -1;
				if(info.details)
				{
					for(int k = 0; k < std::min<int>(info.details->primskills.size(), N_PRIMARY); ++k)
						in.at(In::h_primary, i, k) = info.details->primskills[k];
					in.at(In::h_mana, i) = info.details->mana;
					in.at(In::h_mana_max, i) = info.details->manaLimit;
					in.at(In::h_luck, i) = info.details->luck;
					in.at(In::h_morale, i) = info.details->morale;
				}
				for(const auto & [slot, stack] : info.army)
					if(slot.getNum() >= 0 && slot.getNum() < ARMY_SLOTS && stack.getId() != CreatureID::NONE)
					{
						in.at(In::h_army_creature, i, slot.getNum()) = stack.getId().getNum();
						in.at(In::h_army_count, i, slot.getNum()) = stack.getCount();
					}
				in.at(In::h_detailed, i) = info.army.isDetailed ? 1 : 0;
			}
		}
	}
}

void AlphaMapAI::encodeWindows(Decision & decision, PathfinderCache & paths)
{
	auto & in = decision.inputs;
	const bool turn = decision.kind == DecisionKind::TURN;
	for(int i = 0; i < OWN_HEROES; ++i)
	{
		const auto * h = cb->getHero(decision.heroes[i]);
		if(decision.heroes[i] == ObjectInstanceID() || !h)
			continue;
		const auto center = h->visitablePos();
		decision.centers[i] = center;
		const auto info = paths.getPathsInfo(h);
		const bool canMove = turn && !h->isGarrisoned();
		for(int row = 0; row < WINDOW; ++row)
			for(int column = 0; column < WINDOW; ++column)
			{
				const auto pos = decision.windowTile(i, row, column);
				if(!cb->isInTheMap(pos))
					continue;
				int64_t flags = TILE_ON_MAP;
				if(!cb->isVisible(pos))
				{
					in.at(In::w_flags, i, row, column) = flags;
					continue;
				}
				flags |= TILE_REVEALED;
				const auto * tile = cb->getTile(pos, false);
				in.at(In::w_terrain, i, row, column) = tile->getTerrainID().getNum();
				in.at(In::w_road, i, row, column) = tile->getRoadID().getNum();
				flags |= tile->isWater() ? TILE_WATER : 0;
				flags |= tile->blocked() ? TILE_BLOCKED : 0;
				flags |= tile->visitable() ? TILE_VISITABLE : 0;
				flags |= cb->guardingCreaturePosition(pos) != int3(-1, -1, -1) ? TILE_GUARDED : 0;
				if(const auto * obj = topObject(*cb, pos, h))
				{
					in.at(In::w_object, i, row, column) = obj->ID.getNum();
					in.at(In::w_subtype, i, row, column) = obj->subID.getNum();
					in.at(In::w_owner, i, row, column) = owner(obj->tempOwner);
					flags |= obj->wasVisited(playerID) ? TILE_VISITED_BY_PLAYER : 0;
					flags |= obj->wasVisited(h) ? TILE_VISITED_BY_HERO : 0;
					// (a resource pile's amount is not shown to a human either: none)
					if(obj->ID == Obj::MONSTER)
						if(const auto * stack = dynamic_cast<const CArmedInstance *>(obj)->getStackPtr(SlotID(0)))
						{
							const auto [count, exact] = seenCount(obj, h);
							in.at(In::w_creature, i, row, column) = stack->getId().getNum();
							in.at(In::w_count, i, row, column) = count;
							flags |= exact ? TILE_EXACT_COUNT : 0;
						}
				}
				in.at(In::w_flags, i, row, column) = flags;
				const auto * node = info->getPathInfo(pos);
				if(node && node->reachable())
				{
					in.at(In::w_turns, i, row, column) = node->turns;
					if(node->turns == 0)
						in.at(In::w_move_left, i, row, column) = node->moveRemains;
					if(canMove && node->turns == 0 && pos != center && node->action != EPathNodeAction::UNKNOWN)
						in.at(In::move_mask, i, row, column) = 1;
				}
			}
	}
}

void AlphaMapAI::encodeMap(Decision & decision, const std::vector<int3> & ours)
{
	auto & in = decision.inputs;
	const auto size = cb->getMapSize();
	// Per block of the map (MAP_GRID x MAP_GRID per level, each block `side` tiles on a side): how many of its tiles
	// are on the map, revealed, and of those water, blocked, with a road (GridCount)
	const int side = std::max(1, (std::max(size.x, size.y) + MAP_GRID - 1) / MAP_GRID);
	for(int z = 0; z < std::min(size.z, MAP_LEVELS); ++z)
		for(int y = 0; y < size.y; ++y)
			for(int x = 0; x < size.x; ++x)
			{
				const int row = std::min(y / side, MAP_GRID - 1);
				const int column = std::min(x / side, MAP_GRID - 1);
				in.at(In::g_tiles, z, row, column, static_cast<int>(GridCount::ON_MAP)) += 1;
				const int3 pos(x, y, z);
				if(!cb->isVisible(pos))
					continue;
				const auto * tile = cb->getTile(pos, false);
				in.at(In::g_tiles, z, row, column, static_cast<int>(GridCount::REVEALED)) += 1;
				in.at(In::g_tiles, z, row, column, static_cast<int>(GridCount::WATER)) += tile->isWater() ? 1 : 0;
				in.at(In::g_tiles, z, row, column, static_cast<int>(GridCount::BLOCKED)) += tile->blocked() ? 1 : 0;
				in.at(In::g_tiles, z, row, column, static_cast<int>(GridCount::ROAD)) += tile->getRoadID().getNum() != 0 ? 1 : 0;
			}

	// Every object we see (visitable, on revealed tiles; VCMI hides events), nearest to our heroes and towns first. The
	// model reads them all (what matters is its to find, 2026-10-28): MAP_OBJECTS holds a whole 144 x 144 two-level map's;
	// should a map ever have more, the farthest are left out, and that is an error, never silent
	auto objects = cb->getAllVisitableObjs();
	auto nearness = [&ours](const CGObjectInstance * obj)
	{
		int best = std::numeric_limits<int>::max();
		for(const auto & pos : ours)
			best = std::min(best, distanceSq(pos, obj->visitablePos()));
		return best;
	};
	std::stable_sort(objects.begin(), objects.end(), [&](auto * a, auto * b) { return nearness(a) < nearness(b); });
	if(objects.size() > static_cast<size_t>(MAP_OBJECTS))
	{
		logAi->error("AlphaMapAI %s: %d objects seen, MAP_OBJECTS holds %d: the farthest are left out", playerID.toString(), static_cast<int>(objects.size()), MAP_OBJECTS);
		++errors;
	}
	std::vector<const CGHeroInstance *> heroes; // our hero tokens, for their visits and Visions
	for(int h = 0; h < OWN_HEROES; ++h)
		heroes.push_back(decision.heroes[h] == ObjectInstanceID() ? nullptr : cb->getHero(decision.heroes[h]));
	int armed = 0; // garrisons not ours with an army (a_*), the nearest first
	for(int i = 0; i < std::min<int>(objects.size(), MAP_OBJECTS); ++i)
	{
		const auto * obj = objects[i];
		const auto pos = obj->visitablePos();
		// A garrison fights a hero that steps on it (CGGarrison::onHeroVisit), its army as a human's right-click shows it
		// (CInfoBoxPopup: VCMI's quantity ids): the model's fight on that tile (2026-11-04: 9 of 49 lost fights of joint7's
		// near-empty heroes were garrisons it took for empty tiles)
		// An object whose guards a question showed us (revealedGuards: a bank, a guarded artifact) likewise, its army as it is
		// now (fewer after a battle there): a hero that declined once walked back 40 times to a guarded artifact it took
		// for a free pickup, 2026-11-04
		const bool garrison = obj->ID == Obj::GARRISON || obj->ID == Obj::GARRISON2;
		if((garrison || revealedGuards.count(obj->id)) && armed < ARMED_OBJECTS && owner(obj->tempOwner) != OWNER_SELF)
			if(const auto * army = dynamic_cast<const CArmedInstance *>(obj); army && army->stacksCount() > 0)
			{
				in.at(In::a_present, armed) = 1;
				in.at(In::a_pos, armed, 0) = pos.x;
				in.at(In::a_pos, armed, 1) = pos.y;
				in.at(In::a_pos, armed, 2) = pos.z;
				int s = 0;
				for(const auto & [slot, stack] : army->Slots())
					if(stack && s < ARMY_SLOTS)
					{
						in.at(In::a_creature, armed, s) = stack->getId().getNum();
						in.at(In::a_count, armed, s++) = static_cast<int>(stack->getQuantityID());
					}
				++armed;
			}
		in.at(In::m_object, i) = obj->ID.getNum();
		in.at(In::m_subtype, i) = obj->subID.getNum();
		in.at(In::m_owner, i) = owner(obj->tempOwner);
		in.at(In::m_pos, i, 0) = pos.x;
		in.at(In::m_pos, i, 1) = pos.y;
		in.at(In::m_pos, i, 2) = pos.z;
		int64_t flags = obj->wasVisited(playerID) ? OBJECT_VISITED_BY_PLAYER : 0;
		// (only on a tile we see: VCMI logs an error for each fogged one asked, ~2,300 lines in a game's first minute, 11%
		// of a campaign client's CPU, 2026-10-30)
		flags |= cb->isVisible(pos) && cb->guardingCreaturePosition(pos) != int3(-1, -1, -1) ? OBJECT_GUARDED : 0;
		int64_t visitors = 0;
		for(int h = 0; h < OWN_HEROES; ++h)
			if(heroes[h] && obj->wasVisited(heroes[h]))
				visitors |= int64_t(1) << h;
		if(obj->ID == Obj::MONSTER)
			if(const auto * stack = dynamic_cast<const CArmedInstance *>(obj)->getStackPtr(SlotID(0)))
			{
				auto [count, exact] = seenCount(obj, nullptr);
				for(const auto * hero : heroes) // exact when one of our heroes' Visions reaches it
					if(hero && !exact)
						std::tie(count, exact) = seenCount(obj, hero);
				in.at(In::m_creature, i) = stack->getId().getNum();
				in.at(In::m_count, i) = count;
				flags |= exact ? OBJECT_EXACT_COUNT : 0;
			}
		in.at(In::m_flags, i) = flags;
		in.at(In::m_visitors, i) = visitors;
	}
}

void AlphaMapAI::encodeTiles(Decision & decision)
{
	// The whole map tile by tile (MAP_TILES a side, schema 9): as a window's tiles but for no hero in particular, every
	// hero reading the same; a monster's count exact when any of our heroes' Visions reach it
	auto & in = decision.inputs;
	const auto size = cb->getMapSize();
	if(size.x > MAP_TILES || size.y > MAP_TILES)
	{
		logAi->error("AlphaMapAI %s: a %dx%d map, MAP_TILES is %d: the tiles past it are left out", playerID.toString(), size.x, size.y, MAP_TILES);
		++errors;
	}
	std::vector<const CGHeroInstance *> heroes;
	for(int h = 0; h < OWN_HEROES; ++h)
		if(decision.heroes[h] != ObjectInstanceID())
			if(const auto * hero = cb->getHero(decision.heroes[h]))
				heroes.push_back(hero);
	for(int z = 0; z < std::min(size.z, MAP_LEVELS); ++z)
		for(int y = 0; y < std::min(size.y, MAP_TILES); ++y)
			for(int x = 0; x < std::min(size.x, MAP_TILES); ++x)
			{
				const int3 pos(x, y, z);
				int64_t flags = TILE_ON_MAP;
				if(!cb->isVisible(pos))
				{
					in.at(In::tile_flags, z, y, x) = flags;
					continue;
				}
				flags |= TILE_REVEALED;
				const auto * tile = cb->getTile(pos, false);
				in.at(In::tile_terrain, z, y, x) = tile->getTerrainID().getNum();
				in.at(In::tile_road, z, y, x) = tile->getRoadID().getNum();
				flags |= tile->isWater() ? TILE_WATER : 0;
				flags |= tile->blocked() ? TILE_BLOCKED : 0;
				flags |= tile->visitable() ? TILE_VISITABLE : 0;
				flags |= cb->guardingCreaturePosition(pos) != int3(-1, -1, -1) ? TILE_GUARDED : 0;
				if(const auto * obj = topNonHero(*cb, pos))
				{
					in.at(In::tile_object, z, y, x) = obj->ID.getNum();
					in.at(In::tile_subtype, z, y, x) = obj->subID.getNum();
					in.at(In::tile_owner, z, y, x) = owner(obj->tempOwner);
					flags |= obj->wasVisited(playerID) ? TILE_VISITED_BY_PLAYER : 0;
					if(obj->ID == Obj::MONSTER)
						if(const auto * stack = dynamic_cast<const CArmedInstance *>(obj)->getStackPtr(SlotID(0)))
						{
							auto [count, exact] = seenCount(obj, nullptr);
							for(const auto * hero : heroes)
								if(!exact)
									std::tie(count, exact) = seenCount(obj, hero);
							in.at(In::tile_creature, z, y, x) = stack->getId().getNum();
							in.at(In::tile_count, z, y, x) = count;
							flags |= exact ? TILE_EXACT_COUNT : 0;
						}
				}
				in.at(In::tile_flags, z, y, x) = flags;
			}
}

void AlphaMapAI::encodeArtifacts(Decision & decision, int first, int second)
{
	// ARTIFACT (schema 13; the user, 2026-11-04: move them to the strongest hero, wear one from the backpack, assemble):
	// in an exchange of two heroes of ours (first, second), each artifact one can hand to the other (GIVE: into its
	// own slot there if free, else that hero's backpack: getArtAnyPosition); in a turn, a backpack artifact worn in its
	// slot (EQUIP: the one worn there into the backpack) and a worn one completing a combination with the hero's worn
	// parts (ASSEMBLE). The spellbook, the catapult, the grail and the parts locked in a combination stay
	auto & in = decision.inputs;
	const auto add = [&](int hero, int position, ArtifactTarget target, const Entry & entry)
	{
		in.at(In::artifact_mask, hero, position, static_cast<int>(target)) = 1;
		decision.entries[static_cast<int>(ActionType::ARTIFACT)][(hero * N_ARTIFACT_POSITIONS + position) * N_ARTIFACT_TARGETS + static_cast<int>(target)] = entry;
	};
	for(int h = 0; h < OWN_HEROES; ++h)
	{
		const auto * hero = decision.heroes[h] == ObjectInstanceID() ? nullptr : cb->getHero(decision.heroes[h]);
		if(!hero || (first >= 0 && h != first && h != second))
			continue;
		const auto * other = first >= 0 ? cb->getHero(decision.heroes[h == first ? second : first]) : nullptr;
		for(int p = 0; p < N_ARTIFACT_POSITIONS; ++p)
		{
			const ArtifactPosition position(p); // (worn slots, then the backpack: VCMI's BACKPACK_START is N_WORN_SLOTS)
			const auto * slot = hero->getSlot(position);
			const auto * art = hero->getArt(position);
			if(!art || !slot || slot->locked)
				continue;
			const auto id = art->getTypeId();
			if(id == ArtifactID::SPELLBOOK || id == ArtifactID::CATAPULT || id == ArtifactID::GRAIL || !ArtifactUtils::isArtRemovable(std::make_pair(position, *slot)))
				continue;
			if(first >= 0)
			{
				const auto to = other ? ArtifactUtils::getArtAnyPosition(other, id) : ArtifactPosition(ArtifactPosition::PRE_FIRST);
				if(to != ArtifactPosition::PRE_FIRST)
					add(h, p, ArtifactTarget::GIVE, Entry{hero->id, other->id, p, id.getNum(), to.num});
			}
			else if(ArtifactUtils::isSlotBackpack(position))
			{
				const auto & slots = art->getType()->getPossibleSlots();
				const auto found = slots.find(ArtBearer::HERO);
				if(found != slots.end())
					for(const auto & wear : found->second)
						if(ArtifactUtils::isSlotEquipment(wear) && art->canBePutAt(hero, wear, true))
						{
							add(h, p, ArtifactTarget::EQUIP, Entry{hero->id, {}, p, id.getNum(), wear.num});
							break;
						}
			}
			else if(const auto combinations = ArtifactUtils::assemblyPossibilities(hero, id, true); !combinations.empty())
				add(h, p, ArtifactTarget::ASSEMBLE, Entry{hero->id, {}, p, id.getNum(), combinations.front()->getId().getNum()});
		}
	}
}

void AlphaMapAI::encodeTurnActions(Decision & decision, PathfinderCache & paths)
{
	auto & in = decision.inputs;
	// TRAVEL: for each hero that can move today, each area (TRAVEL_GRID blocks a level) holding a tile it reaches in a
	// day or more; the entry is the area's nearest such tile (fewest days, then the most movement left), and the hero
	// walks the path there as far as it can today (moveHero). The paths are the windows' (the whole map, turnLimit 255)
	const auto size = cb->getMapSize();
	const int area = 2 * std::max(1, (std::max(size.x, size.y) + MAP_GRID - 1) / MAP_GRID); // tiles a side
	for(int i = 0; i < OWN_HEROES; ++i)
	{
		const auto * h = cb->getHero(decision.heroes[i]);
		if(decision.heroes[i] == ObjectInstanceID() || !h || h->isGarrisoned())
			continue;
		bool canMove = false;
		for(int cell = 0; cell < WINDOW * WINDOW && !canMove; ++cell)
			canMove = in.get(In::move_mask, i, cell / WINDOW, cell % WINDOW) != 0;
		if(!canMove)
			continue;
		const auto info = paths.getPathsInfo(h);
		std::map<int, std::pair<const CGPathNode *, int3>> best; // area index -> its nearest tile reached in a day or more
		for(int z = 0; z < std::min(size.z, MAP_LEVELS); ++z)
			for(int y = 0; y < size.y; ++y)
				for(int x = 0; x < size.x; ++x)
				{
					const int3 pos(x, y, z);
					const auto * node = info->getPathInfo(pos);
					if(!node || !node->reachable() || node->turns < 1)
						continue;
					const int index = ((i * MAP_LEVELS + z) * TRAVEL_GRID + std::min(y / area, TRAVEL_GRID - 1)) * TRAVEL_GRID + std::min(x / area, TRAVEL_GRID - 1);
					auto found = best.find(index);
					if(found == best.end() || node->turns < found->second.first->turns
						|| (node->turns == found->second.first->turns && node->moveRemains > found->second.first->moveRemains))
						best[index] = {node, pos};
				}
		for(const auto & [index, target] : best)
		{
			CGPath path; // only where today's first step is affordable: moveHero cannot start otherwise (a no-op)
			if(!info->getPath(path, target.second) || path.nodes.size() < 2 || path.nodes[path.nodes.size() - 2].turns != 0)
				continue;
			in.values[static_cast<size_t>(In::travel_mask)][index] = 1;
			// The days there: whole days and the part of the last day's movement (the prior's distance discount)
			const auto * node = target.first;
			const double day = std::max(1, h->movementPointsLimit(true));
			const double days = node->turns + std::clamp(1.0 - node->moveRemains / day, 0.0, 1.0);
			in.values[static_cast<size_t>(In::travel_days)][index] = static_cast<int64_t>(std::lround(100.0 * days));
			Entry entry;
			entry.first = h->id;
			entry.tile = target.second;
			decision.entries[static_cast<int>(ActionType::TRAVEL)][index] = entry;
		}
	}

	auto resources = cb->getResourceAmount(); // not const: ResourceSet's operator/ is not
	const int gold = resources[EGameResID::GOLD];
	const IMarket * market = ourMarket(); // (purchases short of resources trade there: tradesFor at its own prices)
	const auto prices = marketWorth(market);

	for(int t = 0; t < OWN_TOWNS; ++t)
	{
		const auto * town = cb->getTown(decision.towns[t]);
		if(decision.towns[t] == ObjectInstanceID() || !town)
			continue;
		// BUILD: the server's rules (CGameHandler::buildStructure): canBuildStructure for normal buildings; automatic and
		// special ones are never built by hand; the grail's needs a visiting hero with the grail (canBuildStructure
		// does not check that). One short of resources only is built too when our market's trades of what it leaves
		// over cover the rest (tradesFor; the entry's d: payByTrades first; the user, 2026-11-04: build+trade, no amount
		// for the model to set), still build_short: the model's prior prices the trades
		for(int b = 0; b < N_BUILDINGS; ++b)
		{
			const auto found = town->getTown()->buildings.find(BuildingID(b));
			if(found == town->getTown()->buildings.end())
				continue;
			const auto mode = found->second->mode;
			const bool grail = mode == CBuilding::BUILD_GRAIL && town->getVisitingHero() && town->getVisitingHero()->hasArt(ArtifactID::GRAIL);
			const auto state = cb->canBuildStructure(town, BuildingID(b));
			if((mode == CBuilding::BUILD_NORMAL || grail) && state == EBuildingState::ALLOWED)
			{
				in.at(In::build_mask, t, b) = 1;
				decision.entries[static_cast<int>(ActionType::BUILD)][t * N_BUILDINGS + b] = Entry{town->id, {}, b};
			}
			else if(mode == CBuilding::BUILD_NORMAL && state == EBuildingState::NO_RESOURCES)
			{
				in.at(In::build_short, t, b) = 1; // the engine's word: buildable but for resources
				if(tradesFor(market, found->second->resources, resources, prices))
				{
					in.at(In::build_mask, t, b) = 1;
					decision.entries[static_cast<int>(ActionType::BUILD)][t * N_BUILDINGS + b] = Entry{town->id, {}, b, 0, 0, 1};
				}
			}
		}
		// A boat at the town's shipyard (the user, 2026-11-04: another thing to build where the town can), BUILD's SHIP
		// entry: the shipyard built, its water free (IShipyard::shipyardStatus), its price ours (getBoatCost)
		if(const auto * yard = dynamic_cast<const IShipyard *>(town); yard && town->hasBuilt(BuildingID::SHIPYARD) && BuildingID::SHIP < N_BUILDINGS)
		{
			TResources price;
			yard->getBoatCost(price);
			if(yard->shipyardStatus() == IBoatGenerator::GOOD && resources.canAfford(price))
			{
				in.at(In::build_mask, t, BuildingID::SHIP) = 1;
				decision.entries[static_cast<int>(ActionType::BUILD)][t * N_BUILDINGS + BuildingID::SHIP] = Entry{town->id, {}, BuildingID::SHIP};
			}
		}
		// RECRUIT: into the town's garrison (its upper army: a hero garrisoned there takes them) or into the hero visiting
		// it, the model's choice (2026-10-28, the user: "buy and give it to the hero" in one decision, not two)
		const CGHeroInstance * visitor = town->getVisitingHero();
		const std::array<const CArmedInstance *, N_RECRUIT_TARGETS> into = {town->getUpperArmy(), visitor && visitor->tempOwner == playerID ? visitor : nullptr};
		for(int level = 0; level < std::min<int>(town->creatures.size(), N_LEVELS); ++level)
		{
			const auto & [count, ids] = town->creatures[level];
			if(ids.empty() || count <= 0)
				continue;
			const auto creature = ids.back();
			const auto & price = creature.toCreature()->getFullRecruitCost();
			int most = std::min<int>(count, resources / price);
			// Not even one affordable: as many as our market's trades of the rest make affordable (the user, 2026-11-04:
			// trade-hire when we cannot afford a single creature; the entry's d: payByTrades first)
			const bool trading = most <= 0;
			if(trading)
			{
				int low = 0;
				int high = count;
				while(low < high)
				{
					const int middle = (low + high + 1) / 2;
					if(tradesFor(market, price * middle, resources, prices))
						low = middle;
					else
						high = middle - 1;
				}
				most = low;
			}
			if(most <= 0)
				continue;
			for(int d = 0; d < N_RECRUIT_TARGETS; ++d)
			{
				const auto * destination = into[d];
				if(!destination || !destination->getSlotFor(creature).validSlot())
					continue;
				int previous = 0;
				for(int k = 0; k < N_RECRUIT_AMOUNTS; ++k)
				{
					const int amount = std::max(1, static_cast<int>(std::ceil(most * RECRUIT_FRACTIONS[k])));
					if(amount == previous)
						continue;
					previous = amount;
					in.at(In::recruit_mask, t, level, k, d) = 1;
					decision.entries[static_cast<int>(ActionType::RECRUIT)][((t * N_LEVELS + level) * N_RECRUIT_AMOUNTS + k) * N_RECRUIT_TARGETS + d] = Entry{town->id, destination->id, creature.getNum(), amount, level, trading ? 1 : 0};
				}
			}
		}
		// HIRE
		const auto & gameSettings = cb->getSettings();
		if(town->hasBuilt(BuildingID::TAVERN) && !town->getVisitingHero() && gold >= GameConstants::HERO_GOLD_COST
			&& cb->howManyHeroes(false) < gameSettings.getInteger(EGameSettings::HEROES_PER_PLAYER_ON_MAP_CAP)
			&& cb->howManyHeroes(true) < gameSettings.getInteger(EGameSettings::HEROES_PER_PLAYER_TOTAL_CAP))
		{
			const auto heroes = cb->getAvailableHeroes(town);
			for(int s = 0; s < std::min<int>(heroes.size(), N_TAVERN); ++s)
			{
				in.at(In::hire_mask, t, s) = 1;
				decision.entries[static_cast<int>(ActionType::HIRE)][t * N_TAVERN + s] = Entry{town->id, heroes[s]->id, s};
			}
		}
	}

	// ARMY: a town's garrison (or garrison hero) and the hero visiting it
	std::vector<std::pair<int, int>> pairs;
	for(int t = 0; t < OWN_TOWNS; ++t)
	{
		const auto * town = cb->getTown(decision.towns[t]);
		if(decision.towns[t] == ObjectInstanceID() || !town || !town->getVisitingHero())
			continue;
		const int visiting = decision.holderOf(town->getVisitingHero()->id);
		const int garrison = town->getGarrisonHero() ? decision.holderOf(town->getGarrisonHero()->id) : OWN_HEROES + t;
		if(visiting >= 0 && garrison >= 0)
		{
			pairs.emplace_back(garrison, visiting);
			pairs.emplace_back(visiting, garrison);
		}
	}
	encodeArmyMoves(decision, pairs);

	// UPGRADE: armies in our towns
	for(int holder = 0; holder < OWN_HEROES + OWN_TOWNS; ++holder)
	{
		const auto * army = armyOf(*cb, decision.holders[holder]);
		if(!army)
			continue;
		const auto * hero = dynamic_cast<const CGHeroInstance *>(army);
		const auto * town = hero ? hero->getVisitedTown() : dynamic_cast<const CGTownInstance *>(army);
		if(!town || town->tempOwner != playerID)
			continue;
		for(int s = 0; s < ARMY_SLOTS; ++s)
		{
			const auto * stack = army->getStackPtr(SlotID(s));
			if(!stack)
				continue;
			UpgradeInfo info(stack->getId());
			cb->fillUpgradeInfo(army, SlotID(s), info);
			if(!info.hasUpgrades())
				continue;
			CreatureID best = CreatureID::NONE;
			for(const auto & upgrade : info.getAvailableUpgrades())
				if(resources.canAfford(info.getUpgradeCostsFor(upgrade) * stack->getCount())
					&& (best == CreatureID::NONE || upgrade.toCreature()->getRecruitCost(EGameResID::GOLD) > best.toCreature()->getRecruitCost(EGameResID::GOLD)))
					best = upgrade;
			if(best == CreatureID::NONE)
				continue;
			in.at(In::upgrade_mask, holder, s) = 1;
			decision.entries[static_cast<int>(ActionType::UPGRADE)][holder * ARMY_SLOTS + s] = Entry{army->id, {}, s, best.getNum()};
		}
	}

	// TRADE: at a marketplace of ours (the rates improve with each one we own)
	if(market)
		for(int sell = 0; sell < N_RESOURCES; ++sell)
			for(int buy = 0; buy < N_RESOURCES; ++buy)
			{
				int give = 0;
				int get = 0;
				if(sell == buy || !market->getOffer(sell, buy, give, get, EMarketMode::RESOURCE_RESOURCE) || give <= 0 || get <= 0)
					continue;
				const int units = resources[GameResID(sell)] / give; // trades of `give` for `get`
				int previous = 0;
				for(int k = 0; units > 0 && k < N_TRADE_AMOUNTS; ++k)
				{
					const int trades = std::max(1, static_cast<int>(std::ceil(units * TRADE_FRACTIONS[k])));
					if(trades == previous)
						continue;
					previous = trades;
					in.at(In::trade_mask, sell, buy, k) = 1;
					decision.entries[static_cast<int>(ActionType::TRADE)][(sell * N_RESOURCES + buy) * N_TRADE_AMOUNTS + k] = Entry{market->getObjInstanceID(), {}, sell, buy, trades * give};
				}
			}

	for(int i = 0; i < OWN_HEROES; ++i)
	{
		const auto * h = cb->getHero(decision.heroes[i]);
		if(decision.heroes[i] == ObjectInstanceID() || !h)
			continue;
		// CAST: adventure spells; Dimension Door also needs a tile
		if(h->hasSpellbook() && !h->isGarrisoned())
			for(int s = 0; s < N_ADVENTURE_SPELLS; ++s)
			{
				const auto * spell = SpellID(s).toSpell();
				if(!spell || !h->canCastThisSpell(spell) || h->mana < h->getSpellCost(spell))
					continue;
				spells::detail::ProblemImpl problem;
				const auto & mechanics = spell->getAdventureMechanics();
				if(!mechanics.canBeCast(problem, cb.get(), h))
					continue;
				const bool targeted = std::find(TARGETED_SPELLS.begin(), TARGETED_SPELLS.end(), s) != TARGETED_SPELLS.end();
				if(s == SpellID::SCUTTLE_BOAT)
					continue; // targeted, not in the contract yet
				bool any = !targeted;
				if(targeted)
					for(int row = 0; row < WINDOW; ++row)
						for(int column = 0; column < WINDOW; ++column)
						{
							const auto pos = decision.windowTile(i, row, column);
							if(cb->isInTheMap(pos) && mechanics.canBeCastAt(problem, cb.get(), h, pos))
							{
								in.at(In::cast_cell_mask, i, row, column) = 1;
								any = true;
							}
						}
				if(!any)
					continue;
				in.at(In::cast_mask, i, s) = 1;
				decision.entries[static_cast<int>(ActionType::CAST)][i * N_ADVENTURE_SPELLS + s] = Entry{h->id, {}, s};
			}
		// BUY: a spellbook or a war machine in our town
		const auto * town = h->getVisitedTown();
		if(!town || town->tempOwner != playerID)
			continue;
		for(int item = 0; item < N_BUY_ITEMS; ++item)
		{
			const ArtifactID artifact(BUY_ARTIFACTS[item]);
			const bool spellbook = artifact == ArtifactID::SPELLBOOK;
			const bool legal = spellbook ? !h->hasSpellbook() && town->hasBuilt(BuildingID::MAGES_GUILD_1) && gold >= GameConstants::SPELLBOOK_GOLD_COST
										 : town->isWarMachineAvailable(artifact) && !h->hasArt(artifact) && gold >= static_cast<int>(artifact.toArtifact()->getPrice());
			if(!legal)
				continue;
			in.at(In::buy_mask, i, item) = 1;
			decision.entries[static_cast<int>(ActionType::BUY)][i * N_BUY_ITEMS + item] = Entry{h->id, {}, artifact.getNum()};
		}
	}
}

void AlphaMapAI::encodeArmyMoves(Decision & decision, const std::vector<std::pair<int, int>> & pairs)
{
	for(const auto & [from, to] : pairs)
	{
		const auto * source = armyOf(*cb, decision.holders[from]);
		const auto * target = armyOf(*cb, decision.holders[to]);
		if(!source || !target || from == to)
			continue;
		const bool sourceIsHero = source->ID == Obj::HERO;
		for(int s = 0; s < ARMY_SLOTS; ++s)
		{
			const auto * stack = source->getStackPtr(SlotID(s));
			if(!stack)
				continue;
			const auto slot = target->getSlotFor(stack->getId());
			if(!slot.validSlot())
				continue;
			const int count = stack->getCount();
			for(int amount = 0; amount < N_ARMY_AMOUNTS; ++amount)
			{
				const int moved = amount == static_cast<int>(ArmyAmount::ALL) ? count : amount == static_cast<int>(ArmyAmount::HALF) ? (count + 1) / 2 : 1;
				if(amount != static_cast<int>(ArmyAmount::ALL) && moved >= count)
					continue; // the same as moving the whole stack
				if(sourceIsHero && moved == count && source->stacksCount() == 1)
					continue; // a hero keeps at least one creature
				decision.inputs.at(In::army_mask, from, s, to, amount) = 1;
				const int index = ((from * ARMY_SLOTS + s) * N_HOLDERS + to) * N_ARMY_AMOUNTS + amount;
				decision.entries[static_cast<int>(ActionType::ARMY)][index] = Entry{source->id, target->id, s, slot.getNum(), moved};
			}
		}
	}
}

void AlphaMapAI::encodeOptionContext(Decision & decision, ObjectInstanceID heroID, int3 target)
{
	auto & in = decision.inputs;
	for(int i = 0; i < OWN_HEROES; ++i)
		if(heroID != ObjectInstanceID() && decision.heroes[i] == heroID)
			in.at(In::d_hero) = i;
	if(!target.isValid() || !cb->isInTheMap(target))
		return;
	const auto * hero = cb->getHero(heroID);
	if(const auto * obj = topObject(*cb, target, hero))
	{
		in.at(In::d_object) = obj->ID.getNum();
		in.at(In::d_subtype) = obj->subID.getNum();
		if(obj->ID == Obj::MONSTER)
			if(const auto * stack = dynamic_cast<const CArmedInstance *>(obj)->getStackPtr(SlotID(0)))
			{
				const auto [count, exact] = seenCount(obj, hero);
				in.at(In::d_creature) = stack->getId().getNum();
				in.at(In::d_count) = count;
				in.at(In::d_detailed) = exact ? 1 : 0;
			}
	}
}

void AlphaMapAI::encodeQuestionGuards(Decision & decision, const std::string & text, const std::vector<Component> & components)
{
	// What a yes/no question's options do with the creatures it is about (the model's fight and join features; the
	// user, 2026-11-04): a wandering monster's offer to join (YES takes them, for the gold component's price if any; NO
	// refuses, a battle as a rule), its flight (YES pursues: the battle; NO lets it go), a guarded object's "fight the
	// guards?" (a bank's, a mine's, a dwelling's: YES starts the battle). Its creatures as the dialog or the map shows them
	auto & in = decision.inputs;
	const auto * hero = cb->getHero(activeHero);
	const auto * obj = activeTarget.isValid() && cb->isInTheMap(activeTarget) ? topObject(*cb, activeTarget, hero) : nullptr;
	// A monster's questions by their texts (CGCreature: 86 joins for free, 90 for its gold component's price, 91 flees)
	const auto & texts = LIBRARY->generaltexth->advobtxt;
	int price = -1; // a join's price (0 a join for free's), -1 no join
	if(fromTemplate(text, texts[90]))
		for(const auto & component : components)
			if(component.type == ComponentType::RESOURCE && component.subType == GameResID(GameResID::GOLD))
				price = component.value.value_or(0);
	if(price < 0 && fromTemplate(text, texts[86]))
		price = 0;
	const bool flees = price < 0 && fromTemplate(text, texts[91]);
	// The monster asking: the move's destination (an attack on it), else the guard of the tile the hero stepped onto
	// (its zone, an object there too: the engine's guard of that tile, CGameHandler::moveHero), the case of most of them
	if((price >= 0 || flees) && hero && (!obj || obj->ID != Obj::MONSTER))
	{
		const int3 guard = cb->guardingCreaturePosition(hero->visitablePos());
		obj = guard.isValid() ? topObject(*cb, guard, hero) : nullptr;
	}
	const auto * army = dynamic_cast<const CArmedInstance *>(obj);
	if(!army || army->stacksCount() == 0 || dynamic_cast<const CGHeroInstance *>(obj) || dynamic_cast<const CGTownInstance *>(obj))
		return;
	constexpr int YES = 0;
	constexpr int NO = 1;
	if(obj->ID == Obj::MONSTER)
	{
		const auto [count, exact] = seenCount(obj, hero);
		if(const auto * stack = army->getStackPtr(SlotID(0)))
		{
			in.at(In::d_guard_creature, 0) = stack->getId().getNum();
			in.at(In::d_guard_count, 0) = count;
			in.at(In::d_guard_detailed) = exact ? 1 : 0;
		}
		if(price >= 0)
		{
			in.at(In::o_join, YES) = 1;
			in.at(In::o_amount, YES) = price;
			in.at(In::o_battle, NO) = 1;
		}
		else if(flees)
			in.at(In::o_battle, YES) = 1;
		return;
	}
	if(price >= 0 || flees)
		return; // (a monster's question whose monster is not found: nothing known of what its options do)
	in.at(In::o_battle, YES) = 1;
	revealedGuards.insert(obj->id); // (its army in a_* from now on)
	int i = 0;
	for(const auto & component : components) // (a bank's guards as its dialog shows them: exact)
		if(component.type == ComponentType::CREATURE && i < ARMY_SLOTS)
		{
			in.at(In::d_guard_creature, i) = component.subType.getNum();
			in.at(In::d_guard_count, i++) = component.value.value_or(0);
		}
	if(i > 0)
	{
		in.at(In::d_guard_detailed) = 1;
		return;
	}
	for(const auto & [slot, stack] : army->Slots()) // (a mine's or a dwelling's: as a human sees a monster stack)
		if(stack && i < ARMY_SLOTS)
		{
			in.at(In::d_guard_creature, i) = stack->getId().getNum();
			in.at(In::d_guard_count, i++) = static_cast<int>(stack->getQuantityID());
		}
}
