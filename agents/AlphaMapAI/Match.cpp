#include "StdInc.h"
#include "AlphaMapAI.h"
#include "Armies.h"

#include "lib/callback/CCallback.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/mapObjects/CGTownInstance.h"
#include "lib/mapObjects/MiscObjects.h"
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

// alpha_heroes diagnostic (2026-11-11, logging only; e5's teleport question): Nullkiller2's request to move a hero onto
// the tile it stands on (in VCMI: visit again what is under it, a human's Space), named by what is there, for the
// advice summary's unmatched kinds: our town (a GATHER legal or not), another town, a teleporter (its kind, whether an
// exit other than itself is known to us), a boat, another object (its id), nothing; whether its plan's tile (intended)
// was set, and the request's path length (1: the tile itself)
std::string AlphaMapAI::ownTileRequest(const Decision & decision, const CGHeroInstance * hero, const int3 & intended, size_t path) const
{
	std::string what = "nothing";
	if(hero->inBoat())
		what = "in a boat";
	for(const auto * obj : cb->getVisitableObjs(hero->visitablePos(), false))
	{
		if(obj == hero)
			continue;
		if(const auto * town = dynamic_cast<const CGTownInstance *>(obj))
		{
			bool gather = false;
			for(const auto & [index, e] : decision.entries[static_cast<int>(ActionType::MULTI)])
				gather = gather || (static_cast<MultiKind>(e.a) == MultiKind::GATHER && e.second == hero->id);
			const bool garrisoned = town->getGarrisonHero() == hero; // (in the garrison: the move takes it out to the visiting slot, ExchangeSwapTownHeroes)
			what = town->tempOwner != playerID ? "another's town" : garrisoned ? (town->getVisitingHero() ? "our town, garrisoned, a hero visiting" : "our town, garrisoned, none visiting") : gather ? "our town, GATHER legal" : "our town, no GATHER";
		}
		else if(const auto * teleport = dynamic_cast<const CGTeleport *>(obj))
		{
			bool exit = false;
			for(const auto & id : cb->getTeleportChannelExits(teleport->channel, playerID))
				exit = exit || id != obj->id;
			what = "teleporter " + std::to_string(obj->ID.getNum()) + (exit ? ", an exit known" : ", no exit known");
		}
		else
			what = "object " + std::to_string(obj->ID.getNum());
		break;
	}
	return "MoveHero (own tile: " + what + (intended.isValid() ? "; its plan's tile set" : "; no plan tile") + (path <= 1 ? "; path 1)" : "; path longer)");
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
			unmatched = ownTileRequest(decision, h, intended, p->path.size()); // (alpha_heroes diagnostic, 2026-11-11: named by what is there)
		else
		{
			for(int i = 0; i < OWN_HEROES; ++i)
			{
				if(decision.heroes[i] != p->hid)
					continue;
				if(const int entry = decision.moveEntry(i, target); entry >= 0) // (any of its windows: schema 17)
					pick(ActionType::MOVE, entry);
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
		// Our ARMY entry of the same stack and destination (schema 16: within one army too), its target slot the
		// request's (the record's cell): a whole stack moved, merged or swapped (what 1, 2: ALL; VCMI swaps two stacks
		// of one creature where ours merge them: not exact); a part split off (what 3: val the count the target slot
		// ends with), the amount whose count is nearest (on a log scale), all but one ONE's second half (the one stays)
		const auto * source = armedObject(*cb, p->id1);
		const auto * target = armedObject(*cb, p->id2);
		if(source && target && p->what == 1 && source->slotEmpty(p->p1) && !target->slotEmpty(p->p2))
		{
			// A swap from an empty slot: VCMI's move of the other stack into it (Nullkiller2's formation for a siege)
			const ArrangeStacks back(1, p->p2, p->p1, p->id2, p->id1, 0);
			return matchRequest(decision, &back, choice, exact, unmatched, intended);
		}
		if(!source || !target)
			unmatched = "ArrangeStacks (no army)";
		else
		{
			const int count = source->getStackCount(p->p1);
			const double moved = p->what == 3 ? double(p->val - target->getStackCount(p->p2)) : double(count);
			int best = -1, bestCell = -1;
			double gap = 1e18;
			for(const auto & [index, e] : entries(ActionType::ARMY))
			{
				if(e.first != p->id1 || e.second != p->id2 || e.a != p->p1.getNum())
					continue;
				for(int half = 0; half < 2; ++half)
				{
					const int cell = p->p2.getNum() + half * ARMY_SLOTS;
					if(!armySlotAllowed(decision, index, cell))
						continue;
					const double g = std::abs(std::log(std::max(1, half ? count - 1 : e.c)) - std::log(std::max(1.0, moved)));
					if(g < gap || (g == gap && (index < best || (index == best && cell < bestCell))))
						std::tie(gap, best, bestCell) = std::tuple(g, index, cell);
				}
			}
			if(best >= 0)
			{
				pick(ActionType::ARMY, best);
				choice.cell = bestCell;
				const bool sameSwap = p->what == 1 && !target->slotEmpty(p->p2) && target->getCreature(p->p2) == source->getCreature(p->p1);
				exact = gap < 1e-9 && !sameSwap;
			}
			else
				unmatched = p->id1 == p->id2 ? "ArrangeStacks (within one army, not legal for us)" : "ArrangeStacks (not legal for us)";
		}
	}
	else if(const auto * p = dynamic_cast<const ExchangeArtifacts *>(pack))
	{
		// ARTIFACT: to the other hero of an exchange (GIVE; ours to its own slot there if free, else the backpack: exact
		// where the request's slot is that one), from the backpack into a worn slot (EQUIP, the same way); taking one off
		// into the backpack, or from one worn slot to another, is not among our actions
		const bool give = p->src.artHolder != p->dst.artHolder;
		const auto target = give ? ArtifactTarget::GIVE : ArtifactTarget::EQUIP;
		for(const auto & [index, e] : entries(ActionType::ARTIFACT))
			if(index % N_ARTIFACT_TARGETS == static_cast<int>(target) && e.first == p->src.artHolder && (!give || e.second == p->dst.artHolder) && e.a == p->src.slot.num)
			{
				pick(ActionType::ARTIFACT, index);
				exact = e.c == p->dst.slot.num;
			}
		if(choice.type < 0)
			unmatched = give ? "ExchangeArtifacts (not legal for us)" : "ExchangeArtifacts (within a hero: not among our actions)";
	}
	else if(const auto * p = dynamic_cast<const AssembleArtifacts *>(pack))
	{
		for(const auto & [index, e] : entries(ActionType::ARTIFACT))
			if(p->assemble && index % N_ARTIFACT_TARGETS == static_cast<int>(ArtifactTarget::ASSEMBLE) && e.first == p->heroID && e.a == p->artifactSlot.num && e.c == p->assembleTo.getNum())
				pick(ActionType::ARTIFACT, index);
		if(choice.type < 0)
			unmatched = p->assemble ? "AssembleArtifacts (not legal for us)" : "AssembleArtifacts (a disassembly: not among our actions)";
	}
	else if(const auto * p = dynamic_cast<const GarrisonHeroSwap *>(pack))
	{
		for(const auto & [index, e] : entries(ActionType::MULTI))
			if(e.first == p->tid && e.a == static_cast<int>(MultiKind::GARRISON_SWAP))
				pick(ActionType::MULTI, index);
		if(choice.type < 0)
			unmatched = "GarrisonHeroSwap (not legal for us)";
	}
	else if(const auto * p = dynamic_cast<const BulkMergeStacks *>(pack))
	{
		// MERGE_ALL: every creature's stacks into its first; VCMI's this creature's into this slot (exact when that is the
		// only creature in two slots and this its first)
		const auto * army = armedObject(*cb, p->srcOwner);
		for(const auto & [index, e] : entries(ActionType::MULTI))
			if(e.first == p->srcOwner && e.a == static_cast<int>(MultiKind::MERGE_ALL))
				pick(ActionType::MULTI, index);
		if(choice.type < 0 || !army)
			unmatched = "BulkMergeStacks (not legal for us)";
		else
		{
			const auto [later, earlier] = armies::duplicate(*army);
			const auto creature = army->getCreature(p->src);
			bool others = false;
			for(const auto & [slot, stack] : army->Slots())
				for(const auto & [other, more] : army->Slots())
					others = others || (other.getNum() > slot.getNum() && more->getCreature() == stack->getCreature() && stack->getCreature() != creature);
			exact = !others && earlier == p->src;
		}
	}
	else if(const auto * p = dynamic_cast<const BulkSplitStack *>(pack))
	{
		// A creature into each free slot (amount 1): SPLIT_SINGLES, exact when it splits this stack alone (its weakest of
		// more than one, enough creatures for every free slot); a larger amount: a part of it into the first free slot
		const auto * army = armedObject(*cb, p->srcOwner);
		const SlotID free = army ? army->getFreeSlot() : SlotID();
		if(p->amount == 1)
			for(const auto & [index, e] : entries(ActionType::MULTI))
				if(e.first == p->srcOwner && e.a == static_cast<int>(MultiKind::SPLIT_SINGLES))
					pick(ActionType::MULTI, index);
		if(choice.type >= 0)
			exact = armies::singleSource(*army) == p->src && army->getStackCount(p->src) - 1 >= static_cast<int>(army->getFreeSlots().size());
		else if(army && free.validSlot())
		{
			const ArrangeStacks split(3, p->src, free, p->srcOwner, p->srcOwner, p->amount);
			matchRequest(decision, &split, choice, exact, unmatched, intended);
			exact = false;
		}
		if(choice.type < 0)
			unmatched = "BulkSplitStack (not legal for us)";
	}
	else if(const auto * p = dynamic_cast<const BulkSplitAndRebalanceStack *>(pack))
	{
		// Its stack spread evenly over the free slots: the first step of it, half the stack into the first free slot
		const auto * army = armedObject(*cb, p->srcOwner);
		const SlotID free = army ? army->getFreeSlot() : SlotID();
		if(army && free.validSlot() && army->getStackCount(p->src) > 1)
		{
			const ArrangeStacks split(3, p->src, free, p->srcOwner, p->srcOwner, (army->getStackCount(p->src) + 1) / 2);
			matchRequest(decision, &split, choice, exact, unmatched, intended);
		}
		exact = false;
		if(choice.type < 0)
			unmatched = "BulkSplitAndRebalanceStack (not legal for us)";
	}
	else if(const auto * p = dynamic_cast<const BulkMoveArmy *>(pack))
	{
		// Every stack but a creature of one (VCMI's bulk move; ours keeps one of the weakest stack): a town's garrison into
		// its visitor (GATHER), an exchange's hero's into the other's (GIVE_ALL, its artifacts too: not exact)
		for(const auto & [index, e] : entries(ActionType::MULTI))
			if(e.first == p->srcArmy && e.second == p->destArmy && (e.a == static_cast<int>(MultiKind::GATHER) || e.a == static_cast<int>(MultiKind::GIVE_ALL)))
			{
				pick(ActionType::MULTI, index);
				exact = e.a == static_cast<int>(MultiKind::GATHER);
			}
		if(choice.type < 0)
			unmatched = "BulkMoveArmy (not legal for us)";
	}
	else
		unmatched = typeid(*pack).name();
	return choice.type >= 0;
}
