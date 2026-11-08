#include "StdInc.h"
#include "AlphaMapAI.h"
#include "Armies.h"

#include "lib/CCreatureHandler.h"
#include "lib/callback/CCallback.h"
#include "lib/entities/building/CBuilding.h"
#include "lib/entities/faction/CTown.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/mapObjects/CGTownInstance.h"
#include "lib/networkPacks/ArtifactLocation.h"
#include "lib/networkPacks/TradeItem.h"
#include "lib/pathfinder/CGPathNode.h"
#include "lib/pathfinder/PathfinderCache.h"
#include "lib/pathfinder/PathfinderOptions.h"

using namespace MapSchema;

namespace
{
/// A cheap fingerprint of what our actions change: resources, heroes (where, movement, mana, armies) and towns (what is
/// built, garrisons, creatures for hire). An action that leaves it unchanged did nothing.
std::vector<int64_t> fingerprint(const CCallback & cb)
{
	std::vector<int64_t> values;
	const auto resources = cb.getResourceAmount();
	for(int r = 0; r < 7; ++r)
		values.push_back(resources[GameResID(r)]);
	auto army = [&values](const CArmedInstance * a)
	{
		for(const auto & [slot, stack] : a->Slots())
			values.insert(values.end(), {slot.getNum(), stack->getId().getNum(), stack->getCount()});
	};
	for(const auto * h : cb.getHeroesInfo())
	{
		const auto pos = h->visitablePos();
		values.insert(values.end(), {h->id.getNum(), pos.x, pos.y, pos.z, h->movementPointsRemaining(), h->mana});
		army(h);
		// Its artifacts, which and where (their count alone: an EQUIP replacing the worn one, a GIVE from backpack to
		// backpack read as no effect, then futile, 2026-11-05)
		for(const auto & [position, slot] : h->artifactsWorn)
			values.insert(values.end(), {position.num, slot.getArt() ? slot.getArt()->getTypeId().getNum() : -1});
		values.push_back(static_cast<int64_t>(h->artifactsInBackpack.size()));
		for(const auto & slot : h->artifactsInBackpack)
			values.push_back(slot.getArt() ? slot.getArt()->getTypeId().getNum() : -1);
	}
	for(const auto * t : cb.getTownsInfo(true))
	{
		values.push_back(static_cast<int64_t>(t->getBuildings().size()));
		army(t);
		for(const auto & [count, ids] : t->creatures)
			values.push_back(count);
	}
	return values;
}

const CArmedInstance * armyOf(const CCallback & cb, ObjectInstanceID id)
{
	return dynamic_cast<const CArmedInstance *>(cb.getObj(id, false));
}

ObjectInstanceID topVisitable(const CCallback & cb, const int3 & pos)
{
	const auto objects = cb.getVisitableObjs(pos, false);
	return objects.empty() ? ObjectInstanceID() : objects.back()->id;
}
}

bool AlphaMapAI::execute(const Decision & decision, const Choice & choice)
{
	const auto type = static_cast<ActionType>(choice.type);
	const auto before = fingerprint(*cb);
	if(type == ActionType::MOVE)
	{
		const int cells = WINDOW * WINDOW;
		const int hero = choice.index / cells;
		const int cell = choice.index % cells;
		moveHero(decision.heroes[hero], decision.windowTile(hero, cell / WINDOW, cell % WINDOW));
		return fingerprint(*cb) != before;
	}
	const auto & entries = decision.entries[choice.type];
	const auto found = entries.find(choice.index);
	if(found == entries.end())
	{
		logAi->error("AlphaMapAI %s: action %d entry %d has no meaning (the masks and the entries disagree)", playerID.toString(), choice.type, choice.index);
		return false;
	}
	const auto & e = found->second;
	switch(type)
	{
		case ActionType::BUILD:
			if(const auto * town = cb->getTown(e.first))
			{
				if(e.a == BuildingID::SHIP)
				{
					if(const auto * yard = dynamic_cast<const IShipyard *>(town))
					{
						cb->buildBoat(yard);
						++boatsBuilt;
					}
				}
				else
				{
					if(e.d) // (short of resources: our market's trades first)
						payByTrades(town->getTown()->buildings.at(BuildingID(e.a))->resources, decision);
					cb->buildBuilding(town, BuildingID(e.a));
				}
			}
			break;
		case ActionType::RECRUIT:
		{
			const auto * town = cb->getTown(e.first);
			const auto * destination = armyOf(*cb, e.second);
			if(town && destination)
			{
				if(e.d) // (not one affordable: our market's trades first)
					payByTrades(CreatureID(e.a).toCreature()->getFullRecruitCost() * e.b, decision);
				cb->recruitCreatures(town, destination, CreatureID(e.a), e.b, e.c);
			}
			break;
		}
		case ActionType::HIRE:
		{
			const auto * town = cb->getTown(e.first);
			const auto heroes = town ? cb->getAvailableHeroes(town) : std::vector<const CGHeroInstance *>();
			if(e.a < static_cast<int>(heroes.size()))
				cb->recruitHero(town, heroes[e.a]);
			break;
		}
		case ActionType::ARMY:
		{
			const auto * source = armyOf(*cb, e.first);
			const auto * target = armyOf(*cb, e.second);
			if(!source || !target)
				break;
			// The model's target slot (schema 16; else VCMI's, as before): a whole stack moves into an empty slot, merges
			// onto its own creature, swaps with another; a part splits off into it (the count the slot ends with); ONE's
			// second half all but one into it, the one staying
			const SlotID from(e.a);
			const bool stays = choice.cell >= ARMY_SLOTS;
			const SlotID to(choice.cell >= 0 ? choice.cell % ARMY_SLOTS : e.b);
			const int count = source->getStackCount(from);
			if(!to.validSlot() || count <= 0)
				break;
			if(stays)
			{
				if(count > 1)
					cb->splitStack(source, target, from, to, target->getStackCount(to) + count - 1);
			}
			else if(e.c >= count)
			{
				if(target->slotEmpty(to) || target->getCreature(to) != source->getCreature(from))
					cb->swapCreatures(source, target, from, to);
				else
					cb->mergeStacks(source, target, from, to);
			}
			else
				cb->splitStack(source, target, from, to, target->getStackCount(to) + e.c);
			break;
		}
		case ActionType::UPGRADE:
			if(const auto * army = armyOf(*cb, e.first))
				cb->upgradeCreature(army, SlotID(e.a), CreatureID(e.b));
			break;
		case ActionType::TRADE:
			cb->trade(e.first, EMarketMode::RESOURCE_RESOURCE, GameResID(e.a), GameResID(e.b), e.c, nullptr);
			break;
		case ActionType::CAST:
		{
			const auto * hero = cb->getHero(e.first);
			if(!hero)
				break;
			int3 target(-1, -1, -1);
			if(std::find(TARGETED_SPELLS.begin(), TARGETED_SPELLS.end(), e.a) != TARGETED_SPELLS.end())
			{
				if(choice.cell < 0)
					break;
				const int slot = choice.index / N_ADVENTURE_SPELLS;
				target = decision.windowTile(slot, choice.cell / WINDOW, choice.cell % WINDOW);
			}
			activeHero = hero->id;
			activeTarget = target.isValid() ? target : hero->visitablePos();
			cb->castSpell(hero, SpellID(e.a), target);
			if(decision.kind == DecisionKind::TURN)
				waitTillFree(); // Town Portal asks where to; Dimension Door may land next to a fight
			break;
		}
		case ActionType::BUY:
			if(const auto * hero = cb->getHero(e.first))
				cb->buyArtifact(hero, ArtifactID(e.a));
			break;
		case ActionType::ARTIFACT: // (Encoder.cpp encodeArtifacts: the artifact's position a, its destination c)
			if(static_cast<ArtifactTarget>(choice.index % N_ARTIFACT_TARGETS) == ArtifactTarget::ASSEMBLE)
				cb->assembleArtifacts(e.first, ArtifactPosition(e.a), true, ArtifactID(e.c));
			else
				cb->swapArtifacts(ArtifactLocation(e.first, ArtifactPosition(e.a)), ArtifactLocation(e.second == ObjectInstanceID() ? e.first : e.second, ArtifactPosition(e.c)));
			break;
		case ActionType::TRAVEL: // as far along the path to the area's tile as today's movement goes (moveHero stops there)
			moveHero(e.first, e.tile);
			break;
		case ActionType::MULTI: // (Encoder.cpp encodeMulti: the source, the destination, a the MultiKind)
			executeMulti(e);
			break;
		default:
			logAi->error("AlphaMapAI %s: action type %d is not carried out", playerID.toString(), choice.type);
			return false;
	}
	return fingerprint(*cb) != before;
}

void AlphaMapAI::executeMulti(const Entry & e)
{
	const auto kind = static_cast<MultiKind>(e.a);
	if(kind == MultiKind::RECRUIT_ALL || kind == MultiKind::RECRUIT_ALL_HERO)
	{
		// Each level's creature for hire (the best of it, as RECRUIT's) from the highest level down, as many as we can pay
		// for (whole creatures, our resources as they are after the ones before) and the army can take
		const auto * town = cb->getTown(e.first);
		const auto * into = armyOf(*cb, e.second);
		if(!town || !into)
			return;
		for(int level = std::min<int>(town->creatures.size(), N_LEVELS) - 1; level >= 0; --level)
		{
			const auto & [count, ids] = town->creatures[level];
			if(ids.empty() || count <= 0 || !into->getSlotFor(ids.back()).validSlot())
				continue;
			const int amount = std::min<int>(count, cb->getResourceAmount() / ids.back().toCreature()->getFullRecruitCost());
			if(amount > 0)
				cb->recruitCreatures(town, into, ids.back(), amount, level);
		}
		++multiActions;
		return;
	}
	if(kind == MultiKind::TAKE_BEST || kind == MultiKind::TAKE_BEST_FAST)
	{
		// The armies the encoder priced (the same state: Armies.h), made by our own moves
		const auto * giver = armyOf(*cb, e.first);
		const auto * receiver = armyOf(*cb, e.second);
		if(!giver || !receiver)
			return;
		if(const auto take = armies::takeBest(*receiver, *giver, cb->getSettings(), kind == MultiKind::TAKE_BEST_FAST))
		{
			arrange(giver, receiver, *take);
			std::map<CreatureID, int> made, priced;
			for(const auto & [slot, stack] : receiver->Slots())
				made[stack->getCreatureID()] += stack->getCount();
			for(const auto & [creature, count] : take->receiver)
				priced[creature] += count;
			if(made != priced)
			{
				++takeMismatches;
				std::string text;
				for(const auto & [creature, count] : priced)
					text += " " + std::to_string(creature.getNum()) + "x" + std::to_string(count) + "/" + std::to_string(made[creature]);
				logAi->warn("AlphaMapAI %s: TAKE_BEST made another army than priced (creature x priced / made):%s", playerID.toString(), text);
			}
		}
		++multiActions;
		return;
	}
	if(kind == MultiKind::SPLIT_SINGLES)
	{
		// A creature into each free slot in order, from the hero's weakest stack of more than one (Armies.h)
		const auto * hero = cb->getHero(e.first);
		for(int guard = 0; hero && guard < ARMY_SLOTS; ++guard)
		{
			const auto from = armies::singleSource(*hero);
			const auto to = hero->getFreeSlot();
			if(!from.validSlot() || !to.validSlot())
				break;
			cb->splitStack(hero, hero, from, to, 1);
		}
		++multiActions;
		return;
	}
	if(kind == MultiKind::MERGE_ALL)
	{
		if(const auto * hero = cb->getHero(e.first))
			mergeDuplicates(hero);
		++multiActions;
		return;
	}
	if(kind == MultiKind::GARRISON_SWAP)
	{
		if(const auto * town = cb->getTown(e.first))
			cb->swapGarrisonHero(town);
		++multiActions;
		return;
	}
	if(kind == MultiKind::DROP_SLOW || kind == MultiKind::DROP_FOREIGN)
	{
		// The hero's stacks (Armies.h's rule, as encoded) into the garrison, else disbanded
		const auto * hero = cb->getHero(e.first);
		if(!hero)
			return;
		const auto * garrison = e.second == ObjectInstanceID() ? nullptr : armyOf(*cb, e.second);
		for(const auto & slot : kind == MultiKind::DROP_SLOW ? armies::slowestStacks(*hero) : armies::foreignStacks(*hero))
		{
			const auto * stack = hero->getStackPtr(slot);
			if(!stack)
				continue;
			if(!garrison)
			{
				cb->dismissCreature(hero, slot);
				continue;
			}
			const SlotID to = garrison->getSlotFor(stack->getCreatureID());
			if(!to.validSlot())
				continue;
			if(garrison->slotEmpty(to))
				cb->swapCreatures(hero, garrison, slot, to);
			else
				cb->mergeStacks(hero, garrison, slot, to);
		}
		++multiActions;
		return;
	}
	// GATHER, GIVE_ALL: the source's army into the destination as far as it fits (VCMI's bulk move, merging the same
	// creatures; one creature of the source's weakest stack stays if all would go), and GIVE_ALL's artifacts too (VCMI's
	// bulk move: each into its slot if free, else the backpack)
	const auto * from = armyOf(*cb, e.first);
	const auto * to = armyOf(*cb, e.second);
	if(!from || !to)
		return;
	SlotID weakest;
	uint64_t least = std::numeric_limits<uint64_t>::max();
	for(const auto & [slot, stack] : from->Slots())
		if(const uint64_t power = stack->getPower(); power < least)
			std::tie(weakest, least) = std::tuple(slot, power);
	if(weakest.validSlot())
		cb->bulkMoveArmy(from->id, to->id, weakest);
	if(kind == MultiKind::GIVE_ALL)
		cb->bulkMoveArtifacts(from->id, to->id, false, true, true);
	++multiActions;
}

void AlphaMapAI::arrange(const CArmedInstance * giver, const CArmedInstance * receiver, const armies::Take & take)
{
	// TAKE_BEST's requests (as Nullkiller2's pickBestCreatures, by our moves): each creature the receiver keeps made up
	// to its count there from the giver's stacks, onto its stack there, else into a free slot, else swapped whole for a
	// stack the receiver does not keep, the giver left its own count of it (the creature a hero giving keeps:
	// armies::takeBest's choice, whatever the order of the moves); then the receiver's stacks it does not keep to the
	// giver, onto their creature there or into a free slot
	const auto & target = take.receiver;
	const auto in = [](const armies::Army & army, CreatureID creature)
	{
		int n = 0;
		for(const auto & [kept, count] : army)
			n += kept == creature ? count : 0;
		return n;
	};
	const auto wanted = [&](CreatureID creature) { return in(target, creature); };
	const auto countOf = [](const CArmedInstance * army, CreatureID creature)
	{
		int n = 0;
		for(const auto & [slot, stack] : army->Slots())
			n += stack->getCreatureID() == creature ? stack->getCount() : 0;
		return n;
	};
	const auto total = [](const CArmedInstance * army)
	{
		int n = 0;
		for(const auto & [slot, stack] : army->Slots())
			n += stack->getCount();
		return n;
	};
	const bool keepsOne = dynamic_cast<const CGHeroInstance *>(giver) != nullptr;
	// A creature in two slots of one army takes a slot the receiver may need (an exchange's own troop moves and swaps
	// make them, and since schema 16 our own splits; before it, a merge of every army before each turn decision did not
	// reach an exchange's): 2 of the first 5,100 schema-15 TAKE_BESTs left a creature behind for want of a slot
	mergeDuplicates(receiver);
	mergeDuplicates(giver);
	for(const auto & [creature, count] : target)
		for(int guard = 0; guard < 2 * ARMY_SLOTS; ++guard) // (a request a stack at most)
		{
			const int have = countOf(receiver, creature);
			SlotID from;
			for(const auto & [slot, stack] : giver->Slots())
				if(stack->getCreatureID() == creature)
				{
					from = slot;
					break;
				}
			if(have >= count || !from.validSlot())
				break;
			const int there = giver->getStackCount(from);
			int move = std::min({count - have, there, countOf(giver, creature) - in(take.giver, creature)});
			SlotID out; // (a stack the receiver does not keep: swapped for the whole stack where it has no room, or where
			for(const auto & [slot, stack] : receiver->Slots()) // the move would leave a hero giving with nothing)
				if(wanted(stack->getCreatureID()) == 0)
				{
					out = slot;
					break;
				}
			if(keepsOne && move == total(giver))
			{
				if(out.validSlot() && move == there)
				{
					cb->swapCreatures(giver, receiver, from, out);
					continue;
				}
				--move; // (a hero keeps a creature)
			}
			if(move <= 0)
				break;
			const SlotID to = receiver->getSlotFor(creature);
			if(!to.validSlot())
			{
				if(mergeDuplicates(receiver)) // (a swap of ours may have made one)
					continue;
				if(!out.validSlot() || move < there)
					break;
				cb->swapCreatures(giver, receiver, from, out);
				continue;
			}
			if(move < there)
				cb->splitStack(giver, receiver, from, to, receiver->getStackCount(to) + move);
			else if(receiver->slotEmpty(to))
				cb->swapCreatures(giver, receiver, from, to);
			else
				cb->mergeStacks(giver, receiver, from, to);
		}
	std::vector<SlotID> leaving;
	for(const auto & [slot, stack] : receiver->Slots())
		if(wanted(stack->getCreatureID()) == 0)
			leaving.push_back(slot);
	for(const auto & slot : leaving)
	{
		const auto * stack = receiver->getStackPtr(slot);
		if(!stack)
			continue;
		const SlotID to = giver->getSlotFor(stack->getCreatureID());
		if(!to.validSlot())
			continue;
		if(giver->slotEmpty(to))
			cb->swapCreatures(receiver, giver, slot, to);
		else
			cb->mergeStacks(receiver, giver, slot, to);
	}
}

void AlphaMapAI::payByTrades(const TResources & cost, const Decision & decision)
{
	const auto * market = ourMarket();
	auto trades = tradesFor(market, cost, cb->getResourceAmount(), decision.worth);
	if(!trades) // (the model's worth found none: the market's own prices did, when the entry was made)
		trades = tradesFor(market, cost, cb->getResourceAmount(), marketWorth(market));
	if(!trades)
		return;
	for(const auto & trade : *trades)
		cb->trade(market->getObjInstanceID(), EMarketMode::RESOURCE_RESOURCE, GameResID(trade.sell), GameResID(trade.buy), trade.give, nullptr);
	++tradedPurchases;
}

bool AlphaMapAI::mergeDuplicates(const CArmedInstance * army)
{
	bool merged = false;
	for(int first = 0; first < ARMY_SLOTS; ++first)
		for(int second = first + 1; second < ARMY_SLOTS; ++second)
		{
			const auto * one = army->getStackPtr(SlotID(first));
			const auto * other = army->getStackPtr(SlotID(second));
			if(one && other && one->getId() == other->getId())
			{
				cb->mergeStacks(army, army, SlotID(second), SlotID(first)); // (the later slot into the earlier)
				++stacksMerged;
				merged = true;
			}
		}
	return merged;
}

bool AlphaMapAI::moveHero(ObjectInstanceID heroID, int3 destination)
{
	// The human client's movement (HeroMovementController): batches of steps that end at a teleport, a monster's zone
	// of control, a visitable object, or where the hero changes between land and water or air; repeated until the
	// hero is there, stops moving, or what was at the destination (a pile, a monster) is gone.
	activeHero = heroID;
	activeTarget = destination;
	const auto targetObject = topVisitable(*cb, destination);
	bool moved = false;
	for(int step = 0; step < 64 && !stopping; ++step)
	{
		const auto * h = cb->getHero(heroID);
		if(!h)
			return true;
		const auto start = h->visitablePos();
		if(start == destination)
			break;
		PathfinderCache cache(cb.get(), PathfinderOptions(*cb));
		const auto paths = cache.getPathsInfo(h);
		CGPath path;
		if(!paths->getPath(path, destination) || path.nodes.size() < 2)
			break;
		const auto & next = path.nodes[path.nodes.size() - 2];
		if(next.turns != 0)
			break;
		if(next.isTeleportAction())
		{
			intendedExit = next.coord;
			cb->moveHero(h, h->pos, false); // into the teleport the hero stands on
		}
		else
		{
			const bool transit = next.layer == EPathfindingLayer::AIR || next.layer == EPathfindingLayer::WATER;
			std::vector<int3> batch;
			for(auto node = path.nodes.rbegin(); node != path.nodes.rend(); ++node)
			{
				if(node->coord == start)
					continue;
				if(node->isTeleportAction() || node->turns != 0)
					break;
				if((node->layer == EPathfindingLayer::AIR || node->layer == EPathfindingLayer::WATER) != transit)
					break;
				batch.push_back(h->convertFromVisitablePos(node->coord));
				if(cb->guardingCreaturePosition(node->coord) != int3(-1, -1, -1) || !cb->getVisitableObjs(node->coord, false).empty())
					break;
			}
			if(batch.empty())
				break;
			cb->moveHero(h, batch, transit);
		}
		waitTillFree();
		intendedExit = int3(-1, -1, -1);
		h = cb->getHero(heroID);
		if(!h)
			return true;
		if(h->visitablePos() == start)
			break;
		moved = true;
		if(targetObject != ObjectInstanceID() && topVisitable(*cb, destination) != targetObject)
			break; // picked up, beaten or flagged: the move was for it
	}
	return moved;
}
