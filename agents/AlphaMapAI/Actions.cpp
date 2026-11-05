#include "StdInc.h"
#include "AlphaMapAI.h"

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
		values.insert(values.end(), {h->id.getNum(), pos.x, pos.y, pos.z, h->movementPointsRemaining(), h->mana, static_cast<int64_t>(h->artifactsWorn.size())});
		army(h);
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
			const SlotID from(e.a);
			const SlotID to(e.b);
			const int count = source->getStackCount(from);
			if(e.c >= count)
			{
				if(target->slotEmpty(to))
					cb->swapCreatures(source, target, from, to);
				else
					cb->mergeStacks(source, target, from, to);
			}
			else
				cb->splitStack(source, target, from, to, target->getStackCount(to) + e.c); // the count the target slot ends with
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
		default:
			logAi->error("AlphaMapAI %s: action type %d is not carried out", playerID.toString(), choice.type);
			return false;
	}
	return fingerprint(*cb) != before;
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

void AlphaMapAI::unifyStacks()
{
	std::vector<const CArmedInstance *> armies;
	for(const auto * hero : cb->getHeroesInfo()) // (ours)
		armies.push_back(hero);
	for(const auto * town : cb->getTownsInfo(true))
		armies.push_back(town);
	for(const auto * army : armies)
		for(int first = 0; first < ARMY_SLOTS; ++first)
			for(int second = first + 1; second < ARMY_SLOTS; ++second)
			{
				const auto * one = army->getStackPtr(SlotID(first));
				const auto * other = army->getStackPtr(SlotID(second));
				if(one && other && one->getId() == other->getId())
				{
					cb->mergeStacks(army, army, SlotID(second), SlotID(first)); // (the later slot into the earlier)
					++stacksMerged;
				}
			}
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
