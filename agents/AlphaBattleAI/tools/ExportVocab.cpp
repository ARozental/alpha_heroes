// Writes the id -> name/stats tables of everything the battle model sees as an id
// (creatures, spells and effects, heroes, secondary skills, artifacts, battlefields, obstacles).
// The ids are the engine's own: exactly the values the C++ side puts into the model inputs.
//
// Usage (scripts/export_vocab.sh sets this up): VCMI_USER_DIR=... build/bin/alpha_export_vocab <out.json>
#include "StdInc.h"

#include "lib/BattleFieldHandler.h"
#include "lib/CBonusTypeHandler.h"
#include "lib/CConsoleHandler.h"
#include "lib/CCreatureHandler.h"
#include "lib/CSkillHandler.h"
#include "lib/GameLibrary.h"
#include "lib/ObstacleHandler.h"
#include "lib/VCMIDirs.h"
#include "lib/entities/artifact/CArtHandler.h"
#include "lib/entities/faction/CTownHandler.h"
#include "lib/entities/hero/CHeroClass.h"
#include "lib/entities/hero/CHeroHandler.h"
#include "lib/json/JsonNode.h"
#include "lib/logging/CBasicLogConfigurator.h"
#include "lib/spells/CSpellHandler.h"
#include "lib/bonuses/BonusEnum.h"

#include <fstream>

namespace
{
const std::array<const char *, 7> RESOURCES = {"wood", "mercury", "ore", "sulfur", "crystal", "gems", "gold"};

// Names of the engine's built-in bonus types, in BonusType order
#define BONUS_NAME(x) #x,
const std::vector<std::string> BONUS_NAMES = {BONUS_LIST};
#undef BONUS_NAME

std::string bonusName(BonusType type)
{
	const auto index = static_cast<size_t>(type);
	return index < BONUS_NAMES.size() ? BONUS_NAMES[index] : LIBRARY->bth->bonusToString(type);
}

template<typename Entity>
JsonNode entry(const Entity & e)
{
	JsonNode node;
	node["id"].Integer() = e.getIndex();
	node["key"].String() = e.getJsonKey();
	if constexpr(!std::is_same_v<Entity, CCreature>)
		node["name"].String() = e.getNameTranslated();
	return node;
}

JsonNode creatures()
{
	JsonNode list;
	for(const auto & c : LIBRARY->creh->objects)
	{
		if(!c)
			continue;
		auto node = entry(*c);
		node["name"].String() = c->getNameSingularTranslated();
		node["level"].Integer() = c->getLevel();
		node["faction"].String() = c->getFactionID().hasValue() ? c->getFactionID().toEntity(LIBRARY)->getJsonKey() : "";
		node["attack"].Integer() = c->getBaseAttack();
		node["defense"].Integer() = c->getBaseDefense();
		node["damage_min"].Integer() = c->getBaseDamageMin();
		node["damage_max"].Integer() = c->getBaseDamageMax();
		node["hit_points"].Integer() = c->getBaseHitPoints();
		node["speed"].Integer() = c->getBaseSpeed();
		node["shots"].Integer() = c->getBaseShots();
		node["growth"].Integer() = c->getGrowth();
		node["ai_value"].Integer() = c->getAIValue();
		node["fight_value"].Integer() = c->getFightValue();
		node["double_wide"].Bool() = c->isDoubleWide();
		for(int r = 0; r < static_cast<int>(RESOURCES.size()); ++r)
			if(int amount = c->getRecruitCost(GameResID(r)))
				node["cost"][RESOURCES[r]].Integer() = amount;
		std::set<std::string> abilities; // creature-type bonuses, base stats included (PRIMARY_SKILL, STACK_HEALTH, ...)
		for(const auto & bonus : c->getExportedBonusList())
		{
			abilities.insert(bonusName(bonus->type));
			// Each bonus's subtype (e.g. the creature a HATE is against) and value (e.g. its percentage), for the rules
			// battle_model/rules.py computes: {"HATE": [[subtype, value], ...], ...}
			JsonNode pair;
			pair.Vector().emplace_back(static_cast<si64>(bonus->subtype.getNum()));
			pair.Vector().emplace_back(static_cast<si64>(bonus->val));
			node["ability_values"][bonusName(bonus->type)].Vector().push_back(pair);
		}
		for(const auto & name : abilities)
		{
			JsonNode ability;
			ability.String() = name;
			node["abilities"].Vector().push_back(ability);
		}
		list.Vector().push_back(node);
	}
	return list;
}

JsonNode spellTable()
{
	JsonNode list;
	for(const auto & s : LIBRARY->spellh->objects)
	{
		if(!s)
			continue;
		auto node = entry(*s);
		node["level"].Integer() = s->getLevel();
		node["combat"].Bool() = s->isCombat();
		node["adventure"].Bool() = s->isAdventure();
		node["creature_ability"].Bool() = s->isCreatureAbility();
		node["positive"].Bool() = s->isPositive();
		node["negative"].Bool() = s->isNegative();
		for(int level = 0; level < 4; ++level)
		{
			JsonNode cost;
			cost.Integer() = s->getCost(level);
			node["mana_cost"].Vector().push_back(cost);
		}
		s->forEachSchool([&](const SpellSchool & school, bool & stop)
		{
			JsonNode name;
			name.String() = SpellSchool::encode(school.getNum());
			node["schools"].Vector().push_back(name);
		});
		list.Vector().push_back(node);
	}
	return list;
}

JsonNode heroes()
{
	JsonNode list;
	for(const auto & h : LIBRARY->heroh->objects)
	{
		if(!h)
			continue;
		auto node = entry(*h);
		if(h->heroClass)
		{
			node["class"].String() = h->heroClass->getJsonKey();
			node["faction"].String() = h->heroClass->faction.hasValue() ? h->heroClass->faction.toEntity(LIBRARY)->getJsonKey() : "";
		}
		list.Vector().push_back(node);
	}
	return list;
}

JsonNode artifacts()
{
	JsonNode list;
	for(const auto & a : LIBRARY->arth->objects)
	{
		if(!a)
			continue;
		auto node = entry(*a);
		node["price"].Integer() = a->getPrice();
		list.Vector().push_back(node);
	}
	return list;
}

template<typename Objects>
JsonNode simple(const Objects & objects)
{
	JsonNode list;
	for(const auto & o : objects)
		if(o)
			list.Vector().push_back(entry(*o));
	return list;
}
}

int main(int argc, const char * argv[])
{
	if(argc != 2)
	{
		std::cerr << "usage: alpha_export_vocab <out.json>\n";
		return 2;
	}
	const std::string out = boost::filesystem::absolute(argv[1]).string();
	// Data paths are relative to the executable's folder, as for vcmiserver
	boost::filesystem::current_path(boost::filesystem::system_complete(argv[0]).parent_path());

	CConsoleHandler console;
	CBasicLogConfigurator logConfigurator(VCMIDirs::get().userLogsPath() / "VCMI_ExportVocab_log.txt", &console);
	logConfigurator.configureDefault();
	LIBRARY = new GameLibrary;
	LIBRARY->initializeFilesystem(false);
	logConfigurator.configure();
	LIBRARY->initializeLibrary();

	JsonNode vocab;
	vocab["creatures"] = creatures();
	vocab["spells"] = spellTable();
	vocab["heroes"] = heroes();
	vocab["secondary_skills"] = simple(LIBRARY->skillh->objects);
	vocab["artifacts"] = artifacts();
	vocab["battlefields"] = simple(LIBRARY->battlefieldsHandler->objects);
	vocab["obstacles"] = simple(LIBRARY->obstacleHandler->objects);
	std::ofstream(out) << vocab.toString();
	std::cout << "wrote " << out << "\n";

	logConfigurator.deconfigure();
	delete LIBRARY;
	return 0;
}
