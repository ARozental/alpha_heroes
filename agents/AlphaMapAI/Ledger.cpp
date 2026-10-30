#include "StdInc.h"
#include "Ledger.h"

#include "lib/CPlayerState.h"
#include "lib/entities/artifact/CArtifactInstance.h"
#include "lib/gameState/CGameState.h"
#include "lib/json/JsonNode.h"
#include "lib/mapObjects/CGDwelling.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/mapObjects/CGTownInstance.h"
#include "lib/mapObjects/IOwnableObject.h"
#include "lib/mapping/CMap.h"
#include "lib/spells/CSpellHandler.h"

namespace
{
// Tiles each player ever revealed: Cover of Darkness hides tiles again, so revealing them twice must not count twice
std::map<PlayerColor, std::vector<uint8_t>> explored;

JsonNode number(int64_t value)
{
	JsonNode node;
	node.Integer() = value;
	return node;
}

JsonNode stacks(const CArmedInstance & army)
{
	JsonNode list;
	list.Vector(); // an empty army is an empty list
	for(const auto & [slot, stack] : army.Slots())
	{
		JsonNode pair;
		pair.Vector().push_back(number(stack->getId().getNum()));
		pair.Vector().push_back(number(stack->getCount()));
		list.Vector().push_back(pair);
	}
	return list;
}

JsonNode forHire(const CGDwelling & dwelling)
{
	JsonNode list; // per level: [the best creature, how many]
	list.Vector();
	for(const auto & [count, ids] : dwelling.creatures)
	{
		JsonNode pair;
		pair.Vector().push_back(number(ids.empty() ? -1 : ids.back().getNum()));
		pair.Vector().push_back(number(count));
		list.Vector().push_back(pair);
	}
	return list;
}

int64_t revealed(const CGameState & gs, PlayerColor color)
{
	const auto size = gs.getMapSize();
	auto & seen = explored[color];
	seen.resize(static_cast<size_t>(size.x) * size.y * size.z, 0);
	int64_t count = 0;
	for(int z = 0; z < size.z; ++z)
		for(int y = 0; y < size.y; ++y)
			for(int x = 0; x < size.x; ++x)
			{
				auto & tile = seen[(static_cast<size_t>(z) * size.y + y) * size.x + x];
				if(!tile && gs.isVisibleFor(int3(x, y, z), color))
					tile = 1;
				count += tile;
			}
	return count;
}
}

std::string ledgerJson(const CGameState & gs, int day, bool final)
{
	JsonNode root;
	root["day"].Integer() = day;
	root["final"].Bool() = final;
	for(const auto & [color, state] : gs.players)
	{
		if(!color.isValidPlayer())
			continue;
		JsonNode player;
		player["color"].Integer() = color.getNum();
		player["status"].Integer() = static_cast<int>(state.status);
		for(int r = 0; r < 7; ++r)
			player["resources"].Vector().push_back(number(state.resources[GameResID(r)]));
		player["towns"].Vector();
		for(const auto * town : state.getTowns())
		{
			JsonNode t;
			t["faction"].Integer() = town->getFactionID().getNum();
			t["built"].Vector();
			for(const auto & building : town->getBuildings())
				t["built"].Vector().push_back(number(building.getNum()));
			t["garrison"] = stacks(*town);
			t["available"] = forHire(*town);
			player["towns"].Vector().push_back(t);
		}
		player["heroes"].Vector();
		for(const auto * hero : state.getHeroes())
		{
			JsonNode h;
			h["type"].Integer() = hero->getHeroTypeID().getNum();
			h["level"].Integer() = hero->level;
			h["exp"].Integer() = hero->exp;
			h["mana"].Integer() = hero->mana;
			h["power"].Integer() = hero->getPrimSkillLevel(PrimarySkill::SPELL_POWER);
			int dearest = 0; // the dearest combat spell it can cast
			if(hero->hasSpellbook())
				for(const auto & spellID : hero->getSpellsInSpellbook())
				{
					const auto * spell = spellID.toSpell();
					if(spell && spell->isCombat() && hero->canCastThisSpell(spell))
						dearest = std::max(dearest, hero->getSpellCost(spell));
				}
			h["spell_cost"].Integer() = dearest;
			// The hero's own skills (campaign_model/empire.py prices them): its primary skills without items or temporary
			// bonuses (its artifacts are priced apart), and its secondary skills with their levels
			h["primary"].Vector();
			for(int k = 0; k < 4; ++k)
				h["primary"].Vector().push_back(number(hero->getBasePrimarySkillValue(PrimarySkill(k))));
			h["skills"].Vector();
			for(const auto & [skill, level] : hero->secSkills)
			{
				JsonNode pair;
				pair.Vector() = {number(skill.getNum()), number(level)};
				h["skills"].Vector().push_back(pair);
			}
			h["army"] = stacks(*hero);
			h["artifacts"].Vector();
			for(const auto & [position, slot] : hero->artifactsWorn)
				if(slot.getArt())
					h["artifacts"].Vector().push_back(number(slot.getArt()->getTypeId().getNum()));
			for(const auto & slot : hero->artifactsInBackpack)
				if(slot.getArt())
					h["artifacts"].Vector().push_back(number(slot.getArt()->getTypeId().getNum()));
			player["heroes"].Vector().push_back(h);
		}
		player["dwellings"].Vector();
		player["armies"].Vector();
		ResourceSet income;
		for(const auto * obj : state.getOwnedObjects())
		{
			if(obj->ID == Obj::HERO || obj->ID == Obj::TOWN)
				continue;
			if(const auto * ownable = obj->asOwnable())
				income += ownable->dailyIncome();
			if(const auto * dwelling = dynamic_cast<const CGDwelling *>(obj))
			{
				JsonNode d;
				d["available"] = forHire(*dwelling);
				player["dwellings"].Vector().push_back(d);
			}
			if(const auto * army = dynamic_cast<const CArmedInstance *>(obj); army && army->stacksCount())
				for(const auto & pair : stacks(*army).Vector())
					player["armies"].Vector().push_back(pair);
		}
		for(int r = 0; r < 7; ++r)
			player["income"].Vector().push_back(number(income[GameResID(r)]));
		player["explored"].Integer() = revealed(gs, color);
		root["players"].Vector().push_back(player);
	}
	return root.toString();
}
