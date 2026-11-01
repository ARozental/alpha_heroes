#include "StdInc.h"
#include "AlphaMapAI.h"

#include "lib/callback/CCallback.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/mapObjects/CGTownInstance.h"
#include "lib/networkPacks/PacksForServer.h"

#include <cmath>

using namespace MapSchema;

// A request of Nullkiller2's (its recorder's, Teacher.cpp; its advice, Advice.cpp) as our turn decision's legal entry
// doing the same

namespace
{
const CArmedInstance * armedObject(const CCallback & cb, ObjectInstanceID id)
{
	return dynamic_cast<const CArmedInstance *>(cb.getObj(id, false));
}

// The entry of `entries` for which `key` is true whose amount `amount(e)` is nearest `wanted` (on a log scale); -1 none
template<typename Key, typename Amount>
int nearest(const std::unordered_map<int, Entry> & entries, Key key, Amount amount, double wanted, bool & exact)
{
	int best = -1;
	double gap = 1e18;
	for(const auto & [index, e] : entries)
	{
		if(!key(e))
			continue;
		const double g = std::abs(std::log(std::max(1.0, amount(e))) - std::log(std::max(1.0, wanted)));
		if(g < gap || (g == gap && index < best))
		{
			gap = g;
			best = index;
		}
	}
	exact = gap < 1e-9;
	return best;
}
}

bool AlphaMapAI::matchRequest(const Decision & decision, const CPackForServer * pack, Choice & choice, bool & exact, std::string & unmatched, int3 intended) const
{
	exact = true;
	const auto & entries = [&decision](ActionType type) -> const std::unordered_map<int, Entry> & { return decision.entries[static_cast<int>(type)]; };
	const auto pick = [&choice](ActionType type, int index)
	{
		if(index >= 0)
		{
			choice.type = static_cast<int>(type);
			choice.index = index;
		}
	};
	if(dynamic_cast<const EndTurn *>(pack))
		pick(ActionType::END, 0);
	else if(const auto * p = dynamic_cast<const MoveHero *>(pack))
	{
		const auto * h = cb->getHero(p->hid);
		// Where it is going (intended: its plan's tile for this hero, the caller's), else where this request goes: its
		// moves go a tile a request (AIGateway::moveHeroToTile), so the request's own tile is the path's first step
		const bool planned = h && intended.isValid() && intended != h->visitablePos();
		const int3 target = planned ? intended : h && !p->path.empty() ? h->convertToVisitablePos(p->path.back()) : int3(-1, -1, -1);
		if(!h || (p->path.empty() && !planned))
			unmatched = "MoveHero (no hero or path)";
		else if(target == h->visitablePos())
			unmatched = "MoveHero (into a teleport)"; // (our moves go through teleports as part of a path)
		else
		{
			for(int i = 0; i < OWN_HEROES; ++i)
			{
				if(decision.heroes[i] != p->hid)
					continue;
				const auto & center = decision.centers[i];
				const int row = target.y - center.y + WINDOW_RADIUS;
				const int column = target.x - center.x + WINDOW_RADIUS;
				if(target.z == center.z && row >= 0 && row < WINDOW && column >= 0 && column < WINDOW && decision.inputs.get(In::move_mask, i, row, column) != 0)
					pick(ActionType::MOVE, (i * WINDOW + row) * WINDOW + column);
				break;
			}
			if(choice.type < 0) // not a tile of its window in reach today: the area it heads for, the nearest
			{
				const auto distance = [&target](const Entry & e) { return 1.0 + e.tile.dist2dSQ(target) + (e.tile.z != target.z ? 1e6 : 0.0); };
				pick(ActionType::TRAVEL, nearest(entries(ActionType::TRAVEL), [p](const Entry & e) { return e.first == p->hid; }, distance, 1.0, exact));
				exact = false;
			}
			if(choice.type < 0)
				unmatched = "MoveHero (no move or travel of ours there)";
		}
	}
	else if(const auto * p = dynamic_cast<const BuildStructure *>(pack))
	{
		for(const auto & [index, e] : entries(ActionType::BUILD))
			if(e.first == p->tid && e.a == p->bid.getNum())
				pick(ActionType::BUILD, index);
		if(choice.type < 0)
			unmatched = "BuildStructure (not legal for us)";
	}
	else if(const auto * p = dynamic_cast<const RecruitCreatures *>(pack))
	{
		const auto key = [p](const Entry & e) { return e.first == p->tid && e.second == p->dst && e.a == p->crid.getNum(); };
		pick(ActionType::RECRUIT, nearest(entries(ActionType::RECRUIT), key, [](const Entry & e) { return double(e.b); }, double(p->amount), exact));
		if(choice.type < 0)
			unmatched = cb->getTown(p->tid) ? "RecruitCreatures (town: not legal for us)" : "RecruitCreatures (a dwelling's: a question of ours)";
	}
	else if(const auto * p = dynamic_cast<const HireHero *>(pack))
	{
		const auto * town = cb->getTown(p->tid);
		const auto heroes = town ? cb->getAvailableHeroes(town) : std::vector<const CGHeroInstance *>();
		for(const auto & [index, e] : entries(ActionType::HIRE))
			if(e.first == p->tid && e.a >= 0 && e.a < static_cast<int>(heroes.size()) && heroes[e.a]->getHeroTypeID() == p->hid)
				pick(ActionType::HIRE, index);
		if(choice.type < 0)
			unmatched = town ? "HireHero (not legal for us)" : "HireHero (a tavern outside our towns)";
	}
	else if(const auto * p = dynamic_cast<const UpgradeCreature *>(pack))
	{
		for(const auto & [index, e] : entries(ActionType::UPGRADE))
			if(e.first == p->id && e.a == p->pos.getNum())
			{
				pick(ActionType::UPGRADE, index);
				exact = e.b == p->cid.getNum(); // (ours upgrades to the best upgrade)
			}
		if(choice.type < 0)
			unmatched = "UpgradeCreature (not legal for us)";
	}
	else if(const auto * p = dynamic_cast<const TradeOnMarketplace *>(pack))
	{
		if(p->mode != EMarketMode::RESOURCE_RESOURCE || p->r1.size() != 1 || p->r2.size() != 1 || p->val.size() != 1)
			unmatched = "TradeOnMarketplace (not resource for resource)";
		else
		{
			const int sell = p->r1[0].getNum();
			const int buy = p->r2[0].getNum();
			const auto key = [p, sell, buy](const Entry & e) { return e.first == p->marketId && e.a == sell && e.b == buy; };
			pick(ActionType::TRADE, nearest(entries(ActionType::TRADE), key, [](const Entry & e) { return double(e.c); }, double(p->val[0]), exact));
			if(choice.type < 0)
				unmatched = "TradeOnMarketplace (not legal for us)";
		}
	}
	else if(const auto * p = dynamic_cast<const BuyArtifact *>(pack))
	{
		for(const auto & [index, e] : entries(ActionType::BUY))
			if(e.first == p->hid && e.a == p->aid.getNum())
				pick(ActionType::BUY, index);
		if(choice.type < 0)
			unmatched = "BuyArtifact (not a spellbook or war machine of ours)";
	}
	else if(const auto * p = dynamic_cast<const CastAdvSpell *>(pack))
	{
		for(const auto & [index, e] : entries(ActionType::CAST))
			if(e.first == p->hid && e.a == p->sid.getNum())
			{
				pick(ActionType::CAST, index);
				const int i = index / N_ADVENTURE_SPELLS;
				const auto & center = decision.centers[i];
				const int row = p->pos.y - center.y + WINDOW_RADIUS;
				const int column = p->pos.x - center.x + WINDOW_RADIUS;
				if(p->pos.isValid() && p->pos.z == center.z && row >= 0 && row < WINDOW && column >= 0 && column < WINDOW && decision.inputs.get(In::cast_cell_mask, i, row, column) != 0)
					choice.cell = row * WINDOW + column;
			}
		if(choice.type < 0)
			unmatched = "CastAdvSpell (not legal for us)";
	}
	else if(const auto * p = dynamic_cast<const ArrangeStacks *>(pack))
	{
		const auto * source = armedObject(*cb, p->id1);
		const auto * target = armedObject(*cb, p->id2);
		if(p->id1 == p->id2)
			unmatched = "ArrangeStacks (within one army)";
		else if(!source || !target)
			unmatched = "ArrangeStacks (no army)";
		else if(p->what == 1 && !target->slotEmpty(p->p2))
			unmatched = "ArrangeStacks (a swap of two stacks)";
		else
		{
			const double moved = p->what == 3 ? double(p->val - target->getStackCount(p->p2)) : double(source->getStackCount(p->p1));
			const auto key = [p](const Entry & e) { return e.first == p->id1 && e.second == p->id2 && e.a == p->p1.getNum() && e.b == p->p2.getNum(); };
			pick(ActionType::ARMY, nearest(entries(ActionType::ARMY), key, [](const Entry & e) { return double(e.c); }, moved, exact));
			if(choice.type < 0)
				unmatched = "ArrangeStacks (not legal for us)";
		}
	}
	else
		unmatched = typeid(*pack).name();
	return choice.type >= 0;
}
