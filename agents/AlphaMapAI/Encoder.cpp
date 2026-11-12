#include "StdInc.h"
#include "AlphaMapAI.h"
#include "Armies.h"
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
#include "lib/mapObjects/MiscObjects.h"
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

/// Schema 17's two switches, the game's environment (every player's and every kind of game's alike: a run's unit
/// sets them once). ALPHA_MAP_FURTHER_WINDOWS: the further windows a hero may have (1 unless set; 0: none, the windows
/// as before schema 17). ALPHA_MAP_UNSEEN_EXITS: "0" leaves TILE_UNSEEN_EXIT and OBJECT_UNSEEN_EXIT unset (their
/// price and inputs as before)
int furtherWindowsAllowed()
{
	static const int allowed = []
	{
		const char * value = std::getenv("ALPHA_MAP_FURTHER_WINDOWS");
		return value && *value ? std::max(0, std::atoi(value)) : 1;
	}();
	return allowed;
}

bool unseenExitsShown()
{
	static const bool shown = []
	{
		const char * value = std::getenv("ALPHA_MAP_UNSEEN_EXITS");
		return !value || std::string(value) != "0";
	}();
	return shown;
}

int distanceSq(const int3 & a, const int3 & b)
{
	return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z == b.z ? 0 : 10000);
}

/// A holder's creature in a slot as the inputs have it (-1 empty): our heroes', our towns' garrisons, the other army's
int64_t holderCreature(const Decision & decision, int holder, int slot)
{
	if(holder < OWN_HEROES)
		return decision.inputs.get(In::h_army_creature, holder, slot);
	if(holder < OWN_HEROES + OWN_TOWNS)
		return decision.inputs.get(In::t_garrison_creature, holder - OWN_HEROES, slot);
	return decision.inputs.get(In::x_creature, slot);
}
}

bool armySlotAllowed(const Decision & decision, int index, int t)
{
	const int amount = index % N_ARMY_AMOUNTS;
	const int to = index / N_ARMY_AMOUNTS % N_HOLDERS;
	const int s = index / (N_ARMY_AMOUNTS * N_HOLDERS) % ARMY_SLOTS;
	const int from = index / (N_ARMY_AMOUNTS * N_HOLDERS * ARMY_SLOTS);
	if(index < 0 || from >= N_HOLDERS || t < 0 || t >= ARMY_SLOT_CHOICES)
		return false;
	if(t >= ARMY_SLOTS) // (the one stays)
	{
		if(amount != static_cast<int>(ArmyAmount::ONE))
			return false;
		t -= ARMY_SLOTS;
	}
	else if(amount == static_cast<int>(ArmyAmount::ALL))
		return decision.inputs.get(In::army_slot_mask, from, s, to, t) != 0;
	if(from == to && t == s)
		return false;
	const auto source = holderCreature(decision, from, s);
	const auto target = holderCreature(decision, to, t);
	return target < 0 || (source >= 0 && target == source);
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

int3 Decision::windowTile(int window, int row, int column) const
{
	const auto & center = centers[window];
	return int3(center.x + column - WINDOW_RADIUS, center.y + row - WINDOW_RADIUS, center.z);
}

int Decision::moveEntry(int hero, const int3 & tile) const
{
	for(int w = 0; w < OWN_HEROES; ++w)
	{
		if(windowHero[w] != hero)
			continue;
		const auto & center = centers[w];
		const int row = tile.y - center.y + WINDOW_RADIUS;
		const int column = tile.x - center.x + WINDOW_RADIUS;
		if(tile.z == center.z && row >= 0 && row < WINDOW && column >= 0 && column < WINDOW && inputs.get(In::move_mask, w, row, column) != 0)
			return (w * WINDOW + row) * WINDOW + column;
	}
	return -1;
}

int Decision::echoIndex(int type, int index) const
{
	const int cells = WINDOW * WINDOW;
	if(type != static_cast<int>(ActionType::MOVE) || index < 0 || index / cells >= OWN_HEROES || windowHero[index / cells] < 0)
		return index;
	return windowHero[index / cells] * cells + index % cells;
}

bool AlphaMapAI::revisitUseful(const CGHeroInstance * hero) const
{
	// (bundle 60, AlphaMapAI.h revisitEntry) what the hero stands on: an entrance with no exit seen, a map shipyard with a
	// boat to build now; the hero with movement left (a teleport takes some, a shipyard's question none: kept alike)
	if(!revisitEntry || hero->movementPointsRemaining() == 0)
		return false;
	for(const auto * obj : cb->getVisitableObjs(hero->visitablePos(), false))
	{
		if(obj == hero)
			continue;
		if(unseenExit(obj))
			return true;
		if(const auto * yard = dynamic_cast<const IShipyard *>(obj); yard && !dynamic_cast<const CGTownInstance *>(obj) && cb->getPlayerRelations(playerID, obj->getOwner()) != PlayerRelations::ENEMIES)
		{
			TResources price;
			yard->getBoatCost(price);
			if(yard->shipyardStatus() == IBoatGenerator::GOOD && cb->getResourceAmount().canAfford(price))
				return true;
		}
	}
	return false;
}

bool AlphaMapAI::unseenExit(const CGObjectInstance * obj) const
{
	// VCMI's pathfinder goes through a teleporter only to an exit the player sees (CPathfinderHelper::
	// getAllowedTeleportChannelExits, CGameInfoCallback::getVisibleTeleportObjects), so nothing past an entrance whose
	// exits we have not seen is anyone's target: the entrance itself is, and what it is worth is the land it shows
	// (Nullkiller2's ExplorationBehavior sends a hero into each such entrance too). A channel found to lead nowhere
	// (deadChannels) is none
	if(!unseenExitsShown())
		return false;
	const auto * teleport = dynamic_cast<const CGTeleport *>(obj);
	if(!teleport || !teleport->isEntrance() || deadChannels.count(teleport->channel))
		return false;
	for(const auto & exit : cb->getTeleportChannelExits(teleport->channel, playerID))
		if(exit != obj->id)
			return false;
	return true;
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
	// Every town we see that is not ours, neutral ones too: getTownsInfo(false) walks the players' own towns only, so a
	// neutral town was never listed and its garrison was no fight to the model (2026-11-05, 40 games of joint7: all 17
	// hopeless MOVE attacks our features saw no fight in were on neutral towns, 25-2,120 gold against 720-12,020)
	std::vector<const CGTownInstance *> otherTowns;
	for(const auto * obj : cb->getAllVisitableObjs())
		if(obj->ID == Obj::TOWN && obj->tempOwner != playerID)
			if(const auto * town = dynamic_cast<const CGTownInstance *>(obj))
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
		encodeMulti(decision, -1, -1);
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
		encodeArmyMoves(decision, {{a, b}, {b, a}, {a, a}, {b, b}}); // (schema 16: and each within itself)
		if(a < OWN_HEROES && b < OWN_HEROES) // (two heroes of ours: their artifacts may change hands)
		{
			decision.inputs.at(In::x_heroes, 0) = a + 1;
			decision.inputs.at(In::x_heroes, 1) = b + 1;
			encodeArtifacts(decision, a, b);
			encodeMulti(decision, a, b);
		}
	}
	dropFutile(decision);
	return decision;
}

void AlphaMapAI::dropFutile(Decision & decision)
{
	std::set<std::tuple<int, int, int>> dropped = futile;
	dropped.insert(loopsToday.begin(), loopsToday.end()); // (bundle 60's loop guard: AlphaMapAI.h)
	for(const auto & [type, index, cell] : dropped)
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
	const auto inside = [](const int3 & pos, const int3 & center)
	{
		return pos.z == center.z && std::abs(pos.x - center.x) <= WINDOW_RADIUS && std::abs(pos.y - center.y) <= WINDOW_RADIUS;
	};
	// One window: the tiles around `center` as hero h sees and reaches them, in window slot `slot`. MOVE's legal tiles
	// (canMove) are those it reaches today, each in the first of the hero's windows that shows it (earlier: the middles
	// of its windows before this one)
	const auto fill = [&](int slot, const CGHeroInstance * h, const int3 & center, const CPathsInfo & info, bool canMove, const std::vector<int3> & earlier)
	{
		const auto standing = h->visitablePos();
		decision.centers[slot] = center;
		for(int row = 0; row < WINDOW; ++row)
			for(int column = 0; column < WINDOW; ++column)
			{
				const auto pos = decision.windowTile(slot, row, column);
				if(!cb->isInTheMap(pos))
					continue;
				int64_t flags = TILE_ON_MAP;
				if(!cb->isVisible(pos))
				{
					in.at(In::w_flags, slot, row, column) = flags;
					continue;
				}
				flags |= TILE_REVEALED;
				const auto * tile = cb->getTile(pos, false);
				in.at(In::w_terrain, slot, row, column) = tile->getTerrainID().getNum();
				in.at(In::w_road, slot, row, column) = tile->getRoadID().getNum();
				flags |= tile->isWater() ? TILE_WATER : 0;
				flags |= tile->blocked() ? TILE_BLOCKED : 0;
				flags |= tile->visitable() ? TILE_VISITABLE : 0;
				flags |= cb->guardingCreaturePosition(pos) != int3(-1, -1, -1) ? TILE_GUARDED : 0;
				if(const auto * obj = topObject(*cb, pos, h))
				{
					in.at(In::w_object, slot, row, column) = obj->ID.getNum();
					in.at(In::w_subtype, slot, row, column) = obj->subID.getNum();
					in.at(In::w_owner, slot, row, column) = owner(obj->tempOwner);
					flags |= obj->wasVisited(playerID) ? TILE_VISITED_BY_PLAYER : 0;
					flags |= obj->wasVisited(h) ? TILE_VISITED_BY_HERO : 0;
					flags |= unseenExit(obj) ? TILE_UNSEEN_EXIT : 0;
					// (a resource pile's amount is not shown to a human either: none)
					if(obj->ID == Obj::MONSTER)
						if(const auto * stack = dynamic_cast<const CArmedInstance *>(obj)->getStackPtr(SlotID(0)))
						{
							const auto [count, exact] = seenCount(obj, h);
							in.at(In::w_creature, slot, row, column) = stack->getId().getNum();
							in.at(In::w_count, slot, row, column) = count;
							flags |= exact ? TILE_EXACT_COUNT : 0;
						}
				}
				in.at(In::w_flags, slot, row, column) = flags;
				const auto * node = info.getPathInfo(pos);
				if(node && node->reachable())
				{
					in.at(In::w_turns, slot, row, column) = node->turns;
					if(node->turns == 0)
						in.at(In::w_move_left, slot, row, column) = node->moveRemains;
					in.at(In::w_arrive_left, slot, row, column) = static_cast<int64_t>(node->moveRemains) + 1; // (schema 19: any day's arrival, 0 none)
					if(canMove && pos == standing && revisitUseful(h)) // (bundle 60: a re-visit, AlphaMapAI.h revisitEntry)
						in.at(In::move_mask, slot, row, column) = 1;
					if(canMove && node->turns == 0 && pos != standing && node->action != EPathNodeAction::UNKNOWN
						&& std::none_of(earlier.begin(), earlier.end(), [&](const int3 & c) { return inside(pos, c); }))
						in.at(In::move_mask, slot, row, column) = 1;
				}
			}
	};
	// Each hero's own window, around it. Then (schema 17, a turn's decisions) the tiles a hero reaches today that its
	// windows so far do not show: through a gate or a monolith whose exit we have seen (the pathfinder's paths go
	// through them: the other level, a far part of its own), or more than WINDOW_RADIUS tiles down a road. Before, none
	// of them was a MOVE's target, and TRAVEL's areas are a day or more away: the only way past a gate was a MOVE onto
	// the gate itself, a tile worth nothing by what stands on it
	struct Further
	{
		int hero;
		const CGHeroInstance * h;
		std::shared_ptr<const CPathsInfo> info;
		std::vector<int3> middles;                 // of its windows so far
		std::vector<std::pair<int, int3>> left;    // (movement left there, tile) of the tiles it reaches today, not shown yet
	};
	std::vector<Further> further;
	std::vector<int> spare; // the window slots no hero of ours holds
	const auto size = cb->getMapSize();
	for(int i = 0; i < OWN_HEROES; ++i)
	{
		if(decision.heroes[i] == ObjectInstanceID())
		{
			spare.push_back(i);
			continue;
		}
		const auto * h = cb->getHero(decision.heroes[i]);
		if(!h)
			continue;
		const auto center = h->visitablePos();
		const auto info = paths.getPathsInfo(h);
		const bool canMove = turn && !h->isGarrisoned();
		decision.windowHero[i] = i;
		fill(i, h, center, *info, canMove, {});
		if(!canMove || furtherWindowsAllowed() <= 0)
			continue;
		Further f{i, h, info, {center}, {}};
		for(int z = 0; z < std::min(size.z, MAP_LEVELS); ++z)
			for(int y = 0; y < size.y; ++y)
				for(int x = 0; x < size.x; ++x)
				{
					const int3 pos(x, y, z);
					if(inside(pos, center) || !cb->isVisible(pos))
						continue;
					const auto * node = info->getPathInfo(pos);
					if(node && node->reachable() && node->turns == 0 && node->action != EPathNodeAction::UNKNOWN)
						f.left.emplace_back(node->moveRemains, pos);
				}
		if(!f.left.empty())
			further.push_back(std::move(f));
	}
	// A further window: a square of one level, the level holding the most of the hero's tiles not shown yet, around the
	// soonest reached of them there (the most movement left; a tie: the first, row by row), kept on the map where the
	// map is wider than a window (nothing off the map shown for a tile near its edge: on a 36 x 36 map the square is
	// nearly the whole level). One a hero unless ALPHA_MAP_FURTHER_WINDOWS says otherwise, the heroes in turn while a
	// slot is free and a tile is left. One, because the first holds most of what there is: in a Nullkiller2 game of 40
	// days on a 36 x 36 map with its underground (2026-11-10) 43% of the hero-turns had tiles in reach today outside the
	// hero's own window, a median of 31 against the 77 inside it; a second window held a median of 8 more, a third 3,
	// and each window is as much work as a hero's own
	const auto middle = [](int at, int extent) { return extent > WINDOW ? std::clamp(at, WINDOW_RADIUS, extent - 1 - WINDOW_RADIUS) : extent / 2; };
	size_t next = 0;
	for(int round = 0; round < furtherWindowsAllowed() && next < spare.size(); ++round)
		for(auto & f : further)
		{
			if(f.left.empty() || next >= spare.size())
				continue;
			std::array<int, MAP_LEVELS> onLevel{};
			for(const auto & tile : f.left)
				++onLevel[tile.second.z];
			const int level = static_cast<int>(std::max_element(onLevel.begin(), onLevel.end()) - onLevel.begin());
			const std::pair<int, int3> * soonest = nullptr;
			for(const auto & tile : f.left)
				if(tile.second.z == level && (!soonest || tile.first > soonest->first))
					soonest = &tile;
			const int3 center(middle(soonest->second.x, size.x), middle(soonest->second.y, size.y), level);
			const int slot = spare[next++];
			decision.windowHero[slot] = f.hero;
			in.at(In::w_extra, slot) = f.hero + 1;
			in.at(In::w_extra_pos, slot, 0) = center.x;
			in.at(In::w_extra_pos, slot, 1) = center.y;
			in.at(In::w_extra_pos, slot, 2) = center.z;
			fill(slot, f.h, center, *f.info, true, f.middles);
			f.middles.push_back(center);
			f.left.erase(std::remove_if(f.left.begin(), f.left.end(), [&](const auto & tile) { return inside(tile.second, center); }), f.left.end());
		}
	// Our other heroes' time to each window tile (schema 20, F1): 1 + the fewest days (0 today) in which a hero of ours
	// other than the window's reaches it by its own paths (the pathfinder's turns), 0 where none does. A hero in a
	// garrison is left out (it walks nowhere until it leaves). Raw days only: the graph turns them into discounts
	if(!turn)
		return;
	std::vector<std::pair<int, std::shared_ptr<const CPathsInfo>>> walkers;
	for(int hero = 0; hero < OWN_HEROES; ++hero)  // (our heroes only: decision.heroes holds the others after them)
	{
		const auto * h = cb->getHero(decision.heroes[hero]);
		if(h && !h->isGarrisoned())
			walkers.emplace_back(hero, paths.getPathsInfo(h));
	}
	for(int slot = 0; slot < OWN_HEROES; ++slot)
	{
		const int hero = decision.windowHero[slot];
		if(hero < 0)
			continue;
		for(int row = 0; row < WINDOW; ++row)
			for(int column = 0; column < WINDOW; ++column)
			{
				const auto pos = decision.windowTile(slot, row, column);
				if(!cb->isInTheMap(pos) || !cb->isVisible(pos))
					continue;
				int fewest = -1;
				for(const auto & [other, info] : walkers)
				{
					if(other == hero)
						continue;
					const auto * node = info->getPathInfo(pos);
					if(node && node->reachable() && (fewest < 0 || node->turns < fewest))
						fewest = node->turns;
				}
				if(fewest >= 0)
					in.at(In::w_other_turns, slot, row, column) = fewest + 1;
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
		flags |= unseenExit(obj) ? OBJECT_UNSEEN_EXIT : 0;
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
	// hero reading the same; a monster's count exact when any of our heroes' Visions reach it. A map bigger than
	// MAP_TILES a side is read a block of side x side tiles a cell (encodeTileBlocks), as g_tiles' blocks and TRAVEL's
	// areas are: every map up to MAP_TILES a side (side 1) reads tile by tile, as before
	auto & in = decision.inputs;
	const auto size = cb->getMapSize();
	const int side = std::max(1, (std::max(size.x, size.y) + MAP_GRID - 1) / MAP_GRID);
	if(side > MAP_SIDE_MAX)
	{
		logAi->error("AlphaMapAI %s: a %dx%d map, blocks of %d tiles a side: the model reads sides up to %d", playerID.toString(), size.x, size.y, side, MAP_SIDE_MAX);
		++errors;
	}
	std::vector<const CGHeroInstance *> heroes;
	for(int h = 0; h < OWN_HEROES; ++h)
		if(decision.heroes[h] != ObjectInstanceID())
			if(const auto * hero = cb->getHero(decision.heroes[h]))
				heroes.push_back(hero);
	if(side > 1)
	{
		encodeTileBlocks(decision, heroes, side);
		return;
	}
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
					flags |= unseenExit(obj) ? TILE_UNSEEN_EXIT : 0;
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

void AlphaMapAI::encodeTileBlocks(Decision & decision, const std::vector<const CGHeroInstance *> & heroes, int side)
{
	// A map bigger than MAP_TILES a side, a block of side x side tiles a cell (docs/map_size_plan.md, "block summary rule,
	// 2026-11-08"), from its revealed tiles only (fog stays fog): on the map if any tile is, revealed if any is; water if
	// more than half the revealed tiles are, blocked if all are, visitable or guarded if any is; the most common terrain
	// and nonzero road (ties: the smallest id); the object to note (a town, else one we have not visited, else a
	// wandering monster, the strongest, else any; ties: the first, row by row), whose visit by us it carries; the
	// strongest wandering monster (its seen count x AI value) as the block's creature, whatever the object, so a guard is
	// never hidden by a mine beside it
	static constexpr std::array<int, 10> TYPICAL = {15, 3, 7, 15, 35, 75, 175, 375, 750, 1500}; // (a quantity id's count: features.QUANTITY_COUNTS)
	auto & in = decision.inputs;
	const auto size = cb->getMapSize();
	const auto seen = [&](const CGObjectInstance * obj)
	{
		auto [count, exact] = seenCount(obj, nullptr);
		for(const auto * hero : heroes)
			if(!exact)
				std::tie(count, exact) = seenCount(obj, hero);
		return std::pair(count, exact);
	};
	for(int z = 0; z < std::min(size.z, MAP_LEVELS); ++z)
		for(int row = 0; row < MAP_TILES; ++row)
			for(int column = 0; column < MAP_TILES; ++column)
			{
				const int x0 = column * side, y0 = row * side;
				if(x0 >= size.x || y0 >= size.y)
					continue; // (off the map: the fill values)
				int revealed = 0, water = 0, blocked = 0;
				bool visitable = false, guarded = false, unseen = false; // (unseen: an entrance with no exit seen, TILE_UNSEEN_EXIT, if any is)
				std::map<int, int> terrains, roads;
				const CGObjectInstance * chosen = nullptr;
				int chosenRank = 4;
				uint64_t chosenPower = 0;
				const CGObjectInstance * strongest = nullptr;
				uint64_t strongestPower = 0;
				std::pair<int, bool> strongestSeen{0, false};
				for(int y = y0; y < std::min(y0 + side, size.y); ++y)
					for(int x = x0; x < std::min(x0 + side, size.x); ++x)
					{
						const int3 pos(x, y, z);
						if(!cb->isVisible(pos))
							continue;
						++revealed;
						const auto * tile = cb->getTile(pos, false);
						++terrains[tile->getTerrainID().getNum()];
						if(tile->getRoadID().getNum() != 0)
							++roads[tile->getRoadID().getNum()];
						water += tile->isWater() ? 1 : 0;
						blocked += tile->blocked() ? 1 : 0;
						visitable = visitable || tile->visitable();
						guarded = guarded || cb->guardingCreaturePosition(pos) != int3(-1, -1, -1);
						const auto * obj = topNonHero(*cb, pos);
						if(!obj)
							continue;
						unseen = unseen || unseenExit(obj);
						uint64_t power = 0;
						if(obj->ID == Obj::MONSTER)
							if(const auto * stack = dynamic_cast<const CArmedInstance *>(obj)->getStackPtr(SlotID(0)))
							{
								const auto counted = seen(obj);
								const int count = counted.second ? counted.first : TYPICAL[std::clamp(counted.first, 0, static_cast<int>(TYPICAL.size()) - 1)];
								power = static_cast<uint64_t>(std::max(count, 1)) * static_cast<uint64_t>(std::max(stack->getCreature()->getAIValue(), 1));
								if(!strongest || power > strongestPower)
									std::tie(strongest, strongestPower, strongestSeen) = std::tuple(obj, power, counted);
							}
						const int rank = obj->ID == Obj::TOWN ? 0 : obj->ID == Obj::MONSTER ? 2 : !obj->wasVisited(playerID) ? 1 : 3;
						if(rank < chosenRank || (rank == 2 && chosenRank == 2 && power > chosenPower))
							std::tie(chosen, chosenRank, chosenPower) = std::tuple(obj, rank, power);
					}
				int64_t flags = TILE_ON_MAP;
				if(revealed == 0)
				{
					in.at(In::tile_flags, z, row, column) = flags;
					continue;
				}
				flags |= TILE_REVEALED;
				const auto mode = [](const std::map<int, int> & counts) // (the most common, the smallest id of a tie; 0 none)
				{
					int best = 0, most = 0;
					for(const auto & [id, n] : counts)
						if(n > most)
							std::tie(best, most) = std::tuple(id, n);
					return best;
				};
				in.at(In::tile_terrain, z, row, column) = mode(terrains);
				in.at(In::tile_road, z, row, column) = mode(roads);
				flags |= 2 * water > revealed ? TILE_WATER : 0;
				flags |= blocked == revealed ? TILE_BLOCKED : 0;
				flags |= visitable ? TILE_VISITABLE : 0;
				flags |= guarded ? TILE_GUARDED : 0;
				flags |= unseen ? TILE_UNSEEN_EXIT : 0;
				if(chosen)
				{
					in.at(In::tile_object, z, row, column) = chosen->ID.getNum();
					in.at(In::tile_subtype, z, row, column) = chosen->subID.getNum();
					in.at(In::tile_owner, z, row, column) = owner(chosen->tempOwner);
					flags |= chosen->wasVisited(playerID) ? TILE_VISITED_BY_PLAYER : 0;
				}
				if(strongest)
				{
					in.at(In::tile_creature, z, row, column) = dynamic_cast<const CArmedInstance *>(strongest)->getStackPtr(SlotID(0))->getId().getNum();
					in.at(In::tile_count, z, row, column) = strongestSeen.first;
					flags |= strongestSeen.second ? TILE_EXACT_COUNT : 0;
				}
				in.at(In::tile_flags, z, row, column) = flags;
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
				// Where it is worn (bundle 60, the ARTIFACT loop): a free slot it fits first, else the first it fits (the one
				// there into the backpack), as the prior reads it (model.artifact_prior). Before, the first it fits whatever
				// was there: a spell scroll swapped with the scroll worn there, a free misc slot beside it, 53 times in a game.
				// None that swaps it with its like (the same artifact, a scroll of the same spell): that changes nothing
				const auto & slots = art->getType()->getPossibleSlots();
				const auto found = slots.find(ArtBearer::HERO);
				std::optional<ArtifactPosition> chosen;
				if(found != slots.end())
				{
					for(const auto & wear : found->second)
						if(ArtifactUtils::isSlotEquipment(wear) && !hero->getArt(wear) && art->canBePutAt(hero, wear, true))
						{
							chosen = wear;
							break;
						}
					if(!chosen)
						for(const auto & wear : found->second)
							if(ArtifactUtils::isSlotEquipment(wear) && art->canBePutAt(hero, wear, true))
							{
								chosen = wear;
								break;
							}
				}
				const auto * there = chosen ? hero->getArt(*chosen) : nullptr;
				const bool alike = there && there->getTypeId() == id && (id != ArtifactID::SPELL_SCROLL || there->getScrollSpellID() == art->getScrollSpellID());
				if(chosen && !alike)
					add(h, p, ArtifactTarget::EQUIP, Entry{hero->id, {}, p, id.getNum(), chosen->num});
			}
			else if(const auto combinations = ArtifactUtils::assemblyPossibilities(hero, id, true); !combinations.empty())
				add(h, p, ArtifactTarget::ASSEMBLE, Entry{hero->id, {}, p, id.getNum(), combinations.front()->getId().getNum()});
		}
	}
}

namespace
{
// Whether Town Portal would take the hero anywhere, as the server casts it (TownPortalEffect, TownRelatedAdventureSpell-
// Effect): its movement left at least its school level's movementPointsTaken and movementPointsRequired (the spell's
// config, the base level's where its own level leaves one out), and a town of ours to land in: the nearest one (no
// town selection) or any (selection) without a visiting hero, its tile entrable from the hero's. Its canBeCast checks
// none of this: joint7's Town Portals that changed nothing (51 in the kept logs) were all the server's refusals, all on
// what the player sees
bool townPortalWorks(const CCallback & cb, const CGHeroInstance & hero, const CSpell & spell)
{
	const auto & effect = spell.getLevelInfo(hero.getSpellSchoolLevel(&spell)).adventureEffect;
	const auto & base = spell.getLevelInfo(0).adventureEffect;
	const auto field = [&](const std::string & name) -> const JsonNode & { return effect[name].isNull() ? base[name] : effect[name]; };
	const auto movement = static_cast<int64_t>(hero.movementPointsRemaining());
	if(movement < field("movementPointsTaken").Integer() || movement < field("movementPointsRequired").Integer())
		return false;
	const auto * from = cb.getTile(hero.visitablePos(), false);
	const auto lands = [&](const CGTownInstance * town)
	{
		const auto * tile = cb.getTile(town->visitablePos(), false);
		return town->getVisitingHero() == nullptr && from && tile && tile->entrableTerrain(from);
	};
	const auto towns = cb.getTownsInfo(true);
	if(field("allowTownSelection").Bool())
		return std::any_of(towns.begin(), towns.end(), lands);
	const CGTownInstance * nearest = nullptr;
	for(const auto * town : towns)
		if(!nearest || town->visitablePos().dist2dSQ(hero.visitablePos()) < nearest->visitablePos().dist2dSQ(hero.visitablePos()))
			nearest = town;
	return nearest && lands(nearest);
}

// Whether one creature of a town's creatures for hire at some level can be paid for and go into the army
bool recruitsAny(const CGTownInstance * town, const CArmedInstance * into, TResources resources) // (by value: ResourceSet's division is not const)
{
	for(int level = 0; level < std::min<int>(town->creatures.size(), N_LEVELS); ++level)
	{
		const auto & [count, ids] = town->creatures[level];
		if(!ids.empty() && count > 0 && resources / ids.back().toCreature()->getFullRecruitCost() > 0 && into->getSlotFor(ids.back()).validSlot())
			return true;
	}
	return false;
}

// Whether some of the army's creatures can go into the other: VCMI's bulk move keeps one of the source's when all of
// them would go (a hero must keep one), so a source of a single creature moves nothing (2026-11-05: GATHER of a lone
// creature changed nothing)
bool movesAny(const CArmedInstance * from, const CArmedInstance * to)
{
	int creatures = 0;
	bool fits = false;
	for(const auto & [slot, stack] : from->Slots())
	{
		creatures += stack->getCount();
		fits = fits || to->getSlotFor(stack->getCreature()).validSlot();
	}
	return fits && creatures > 1;
}
}

void AlphaMapAI::encodeMulti(Decision & decision, int first, int second)
{
	// MULTI (schema 14; the user, 2026-11-05: "multi-actions", several of the engine's requests in one decision, the C++
	// making them): in a turn, a town of ours: all we can pay for of what it has for hire, into its garrison
	// (RECRUIT_ALL) or into the hero of ours visiting it (RECRUIT_ALL_HERO), its garrison into that hero (GATHER); a hero
	// of ours (schema 15): its slowest creatures (DROP_SLOW) or those of its lesser factions (DROP_FOREIGN) into the
	// garrison of the town it visits, else disbanded (Armies.h; h_drop*); (schema 16, in the field or a garrison) a
	// creature of its weakest stack into each free slot (SPLIT_SINGLES), each creature's stacks merged (MERGE_ALL). A
	// town of ours (schema 16): its heroes in or out of the garrison (GARRISON_SWAP). In an exchange of two heroes of
	// ours, all of one's army (one creature kept) and artifacts to the other (GIVE_ALL), Nullkiller2's best army of the
	// two to it (TAKE_BEST, less its slowest creatures TAKE_BEST_FAST; x_take_*: the armies after), each way: row 0 the
	// first hero receiving. Entry: the source, the destination (none: disbanded), a the MultiKind
	auto & in = decision.inputs;
	const auto add = [&](int row, MultiKind kind, ObjectInstanceID from, ObjectInstanceID to)
	{
		in.at(In::multi_mask, row, static_cast<int>(kind)) = 1;
		decision.entries[static_cast<int>(ActionType::MULTI)][row * N_MULTI + static_cast<int>(kind)] = Entry{from, to, static_cast<int>(kind)};
	};
	if(first >= 0)
	{
		const std::array<const CGHeroInstance *, 2> pair = {cb->getHero(decision.heroes[first]), cb->getHero(decision.heroes[second])};
		for(int r = 0; r < 2; ++r)
		{
			const auto * receiver = pair[r];
			const auto * giver = pair[1 - r];
			if(!receiver || !giver)
				continue;
			bool artifacts = false; // (encodeArtifacts' GIVE entries from the giver to the receiver)
			for(const auto & [index, e] : decision.entries[static_cast<int>(ActionType::ARTIFACT)])
				artifacts = artifacts || (e.first == giver->id && e.second == receiver->id);
			if(movesAny(giver, receiver) || artifacts)
				add(r, MultiKind::GIVE_ALL, giver->id, receiver->id);
			for(int fast = 0; fast < 2; ++fast)
				if(const auto take = armies::takeBest(*receiver, *giver, cb->getSettings(), fast == 1))
				{
					for(int side = 0; side < 2; ++side)
					{
						const auto & army = side == 0 ? take->receiver : take->giver;
						for(int slot = 0; slot < static_cast<int>(army.size()) && slot < ARMY_SLOTS; ++slot)
						{
							in.at(In::x_take_creature, r, fast, side, slot) = army[slot].first.getNum();
							in.at(In::x_take_count, r, fast, side, slot) = army[slot].second;
						}
					}
					add(r, fast == 1 ? MultiKind::TAKE_BEST_FAST : MultiKind::TAKE_BEST, giver->id, receiver->id);
				}
		}
		return;
	}
	for(int h = 0; h < OWN_HEROES; ++h)
	{
		const auto * hero = decision.heroes[h] == ObjectInstanceID() ? nullptr : cb->getHero(decision.heroes[h]);
		if(!hero || hero->tempOwner != playerID)
			continue;
		if(armies::singleSource(*hero).validSlot())
			add(h, MultiKind::SPLIT_SINGLES, hero->id, hero->id);
		if(armies::duplicate(*hero).first.validSlot())
			add(h, MultiKind::MERGE_ALL, hero->id, hero->id);
		if(hero->isGarrisoned())
			continue;
		const auto * town = hero->getVisitedTown();
		const CArmedInstance * garrison = town && town->tempOwner == playerID ? town->getUpperArmy() : nullptr;
		for(int k = 0; k < 2; ++k)
		{
			const auto stacks = k == 0 ? armies::slowestStacks(*hero) : armies::foreignStacks(*hero);
			if(stacks.empty() || static_cast<int>(stacks.size()) >= hero->stacksCount()) // (never its whole army)
				continue;
			const bool kept = garrison && garrison != hero && armies::takes(*garrison, *hero, stacks);
			for(const auto & slot : stacks)
				in.at(In::h_drop, h, k, slot.getNum()) = 1;
			in.at(In::h_drop_kept, h, k) = kept ? 1 : 0;
			add(h, k == 0 ? MultiKind::DROP_SLOW : MultiKind::DROP_FOREIGN, hero->id, kept ? garrison->id : ObjectInstanceID());
		}
	}
	const auto resources = cb->getResourceAmount();
	for(int t = 0; t < OWN_TOWNS; ++t)
	{
		const auto * town = decision.towns[t] == ObjectInstanceID() ? nullptr : cb->getTown(decision.towns[t]);
		if(!town || town->tempOwner != playerID)
			continue;
		const auto * garrison = town->getUpperArmy();
		const auto * visitor = town->getVisitingHero();
		const bool visited = visitor && visitor->tempOwner == playerID;
		if(recruitsAny(town, garrison, resources))
			add(t, MultiKind::RECRUIT_ALL, town->id, garrison->id);
		if(visited && recruitsAny(town, visitor, resources))
			add(t, MultiKind::RECRUIT_ALL_HERO, town->id, visitor->id);
		if(visited && movesAny(garrison, visitor))
			add(t, MultiKind::GATHER, garrison->id, visitor->id);
		// GARRISON_SWAP, as VCMI's garrisonSwap allows: the visiting hero into the garrison (its army taking the town's),
		// the garrison hero out (below the cap of heroes on the map), or the two swapped
		const auto * inside = town->getGarrisonHero();
		const int cap = cb->getSettings().getInteger(EGameSettings::HEROES_PER_PLAYER_ON_MAP_CAP);
		if(inside && inside->tempOwner == playerID && (visitor ? visited : cb->howManyHeroes(false) < cap))
			add(t, MultiKind::GARRISON_SWAP, town->id, ObjectInstanceID());
		else if(!inside && visited && visitor->canBeMergedWith(*town))
			add(t, MultiKind::GARRISON_SWAP, town->id, ObjectInstanceID());
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
		for(int w = 0; w < OWN_HEROES && !canMove; ++w)
			if(decision.windowHero[w] == i)
				for(int cell = 0; cell < WINDOW * WINDOW && !canMove; ++cell)
					canMove = in.get(In::move_mask, w, cell / WINDOW, cell % WINDOW) != 0;
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
			// Where today's walk stops (schema 18): the path's last tile reached today (its nodes run from the target back
			// to the hero; moveHero walks on while it can pay the next step). The hero spends the night there: the prior
			// prices that night's danger (features.night_danger)
			int3 stop = path.nodes.back().coord;
			for(const auto & step : path.nodes)
				if(step.turns == 0)
				{
					stop = step.coord;
					break;
				}
			in.values[static_cast<size_t>(In::travel_stop)][index] = (static_cast<int64_t>(stop.z) * 256 + stop.y) * 256 + stop.x + 1;
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
			// (BUILD's SHIP is a boat, below, by the boat's own rules: VCMI's building 20, "ship" (buildingsLibrary.json),
			// is a hidden building of no cost that canBuildStructure allows whenever the shipyard stands, and its entry
			// here made a boat legal we could not pay for or put anywhere: 569 BUILDs changed nothing in joint7, the 12
			// looked at all boats, with 0-6 of the 10 wood a boat costs)
			if(b == BuildingID::SHIP)
				continue;
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
	for(int holder = 0; holder < OWN_HEROES + OWN_TOWNS; ++holder) // (schema 16: each army of ours arranged within itself)
		if(decision.holders[holder] != ObjectInstanceID())
			pairs.emplace_back(holder, holder);
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
				if(!mechanics.canBeCast(problem, cb.get(), h) || (s == SpellID::TOWN_PORTAL && !townPortalWorks(*cb, *h, *spell)))
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
	// Each stack of `from` into a slot of `to` (schema 16: `to` may be `from` itself, and the slot is the model's): a whole
	// stack into an empty slot (a move), onto its own creature (a merge) or onto another creature (a swap), its legal
	// slots army_slot_mask's; a part of it (HALF, ONE) into a fit slot, an empty one or its own creature's (never its
	// own slot: the model reads them from the armies' creatures, campaign_model.model.army_slots). A hero keeps at least
	// one creature: its last stack goes to another army only by a swap. The entry's b: the slot VCMI's getSlotFor gives
	// (where ARMY went before schema 16; -1 none, as within an army), its c the count moved
	for(const auto & [from, to] : pairs)
	{
		const auto * source = armyOf(*cb, decision.holders[from]);
		const auto * target = armyOf(*cb, decision.holders[to]);
		if(!source || !target)
			continue;
		const bool within = from == to;
		const bool lastOfHero = source->ID == Obj::HERO && source->stacksCount() == 1;
		for(int s = 0; s < ARMY_SLOTS; ++s)
		{
			const auto * stack = source->getStackPtr(SlotID(s));
			if(!stack)
				continue;
			const int count = stack->getCount();
			bool whole = false, part = false;
			for(int t = 0; t < ARMY_SLOTS; ++t)
			{
				if(within && t == s)
					continue;
				const auto * there = target->getStackPtr(SlotID(t));
				const bool fits = !there || there->getId() == stack->getId(); // (empty, or its own creature)
				part = part || fits;
				if(fits && !within && lastOfHero)
					continue; // (its last stack out whole: only by a swap, which leaves it the other)
				decision.inputs.at(In::army_slot_mask, from, s, to, t) = 1;
				whole = true;
			}
			const auto slot = within ? SlotID() : target->getSlotFor(stack->getId());
			for(int amount = 0; amount < N_ARMY_AMOUNTS; ++amount)
			{
				const bool all = amount == static_cast<int>(ArmyAmount::ALL);
				const int moved = all ? count : amount == static_cast<int>(ArmyAmount::HALF) ? (count + 1) / 2 : 1;
				if(!all && moved >= count)
					continue; // the same as moving the whole stack
				if(all ? !whole : !part)
					continue;
				decision.inputs.at(In::army_mask, from, s, to, amount) = 1;
				const int index = ((from * ARMY_SLOTS + s) * N_HOLDERS + to) * N_ARMY_AMOUNTS + amount;
				decision.entries[static_cast<int>(ActionType::ARMY)][index] = Entry{source->id, target->id, s, slot.validSlot() ? slot.getNum() : -1, moved};
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
