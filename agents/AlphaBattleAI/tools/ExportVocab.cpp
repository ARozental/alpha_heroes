// Writes the id -> name/stats tables of everything the battle model sees as an id
// (creatures, spells and effects, heroes, secondary skills, artifacts, battlefields, obstacles).
// The ids are the engine's own: exactly the values the C++ side puts into the model inputs.
// Also what the battle generator (battle_model/generator.py) samples from: upgrades, artifact classes and slots,
// hero classes and their level-up chances, factions, the mage guild's spell chances, the experience per level.
// For the campaign model (campaign_model/): town buildings (cost, daily produce, what each upgrades), each faction's
// creatures by level, horde levels, the map object types, terrains and roads.
//
// Usage (scripts/export_vocab.sh sets this up): VCMI_USER_DIR=... build/bin/alpha_export_vocab <out.json>
#include "StdInc.h"
#include "Content.h"

#include "lib/BattleFieldHandler.h"
#include "lib/CBonusTypeHandler.h"
#include "lib/CConsoleHandler.h"
#include "lib/CCreatureHandler.h"
#include "lib/CSkillHandler.h"
#include "lib/GameLibrary.h"
#include "lib/ObstacleHandler.h"
#include "lib/TerrainHandler.h"
#include "lib/VCMIDirs.h"
#include "lib/entities/artifact/CArtHandler.h"
#include "lib/entities/artifact/CArtifact.h"
#include "lib/entities/faction/CFaction.h"
#include "lib/entities/faction/CTown.h"
#include "lib/entities/building/CBuilding.h"
#include "lib/mapObjectConstructors/CObjectClassesHandler.h"
#include "lib/RoadHandler.h"
#include "lib/entities/hero/CHero.h"
#include "lib/entities/hero/CHeroClassHandler.h"
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
		node["special"].Bool() = c->special; // not recruitable: war machines, commanders, unused creatures
		for(const auto & upgrade : c->upgrades)
			node["upgrades"].Vector().emplace_back(upgrade.toEntity(LIBRARY)->getJsonKey());
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
		for(const auto & faction : LIBRARY->townh->objects) // the weight of being in a town's mage guild, by faction
			if(faction && faction->town)
				node["guild_chance"][faction->getJsonKey()].Integer() = s->getProbability(faction->getId());
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
		node["special"].Bool() = h->special; // campaign heroes, never in a normal game
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
		const std::map<EArtifactClass, std::string> classes = {{EArtifactClass::ART_SPECIAL, "special"}, {EArtifactClass::ART_TREASURE, "treasure"}, {EArtifactClass::ART_MINOR, "minor"}, {EArtifactClass::ART_MAJOR, "major"}, {EArtifactClass::ART_RELIC, "relic"}};
		node["class"].String() = classes.count(a->aClass) ? classes.at(a->aClass) : "special";
		node["combined"].Bool() = a->isCombined();
		std::set<std::string> bonuses; // what it does (battle_model/rules.py picks the ones that change rules)
		std::array<si64, 4> primary{}; // what it adds to its hero's attack, defense, spell power and knowledge
		for(const auto & bonus : a->getExportedBonusList())
		{
			bonuses.insert(bonusName(bonus->type));
			if(bonus->type == BonusType::PRIMARY_SKILL)
			{
				const int skill = bonus->subtype.as<PrimarySkill>().getNum();
				if(skill >= 0 && skill < 4)
					primary[skill] += bonus->val;
			}
		}
		for(const auto & name : bonuses)
			node["bonuses"].Vector().emplace_back(name);
		for(const auto value : primary)
			node["primary"].Vector().emplace_back(value);
		for(const auto * part : a->getConstituents()) // (a combination's parts: worn together they assemble into it)
			node["parts"].Vector().emplace_back(static_cast<si64>(part->getId().getNum()));
		if(a->getPossibleSlots().count(ArtBearer::HERO))
			for(const auto & slot : a->getPossibleSlots().at(ArtBearer::HERO))
				node["slots"].Vector().emplace_back(static_cast<si64>(slot.num)); // ArtifactPosition (encode names them wrongly)
		list.Vector().push_back(node);
	}
	return list;
}

JsonNode heroClasses()
{
	JsonNode list;
	for(const auto & c : LIBRARY->heroclassesh->objects)
	{
		if(!c)
			continue;
		auto node = entry(*c);
		node["faction"].String() = c->faction.hasValue() ? c->faction.toEntity(LIBRARY)->getJsonKey() : "";
		node["magic"].Bool() = c->affinity == CHeroClass::MAGIC;
		auto ints = [](const std::vector<int> & values)
		{
			JsonNode vector;
			for(const auto value : values)
				vector.Vector().emplace_back(static_cast<si64>(value));
			return vector;
		};
		node["primary_initial"] = ints(c->primarySkillInitial);
		node["primary_low"] = ints(c->primarySkillLowLevel); // % chance of each primary skill at a level up, levels 2-9
		node["primary_high"] = ints(c->primarySkillHighLevel); // from level 10
		for(const auto & [skill, weight] : c->secSkillProbability)
			node["secondary_weights"][skill.toEntity(LIBRARY)->getJsonKey()].Integer() = weight;
		list.Vector().push_back(node);
	}
	return list;
}

JsonNode factions()
{
	JsonNode list;
	for(const auto & f : LIBRARY->townh->objects)
	{
		if(!f)
			continue;
		auto node = entry(*f);
		node["town"].Bool() = f->town != nullptr;
		node["special"].Bool() = f->special;
		node["native_terrain"].String() = f->nativeTerrain.hasValue() ? f->nativeTerrain.toEntity(LIBRARY)->getJsonKey() : "";
		if(f->town)
			node["mage_guild"].Integer() = f->town->mageLevel;
		list.Vector().push_back(node);
	}
	return list;
}

JsonNode resourcesJson(const TResources & resources)
{
	JsonNode node;
	for(int r = 0; r < static_cast<int>(RESOURCES.size()); ++r)
		if(resources[GameResID(r)] != 0)
			node[RESOURCES[r]].Integer() = resources[GameResID(r)];
	return node;
}

JsonNode buildings()
{
	JsonNode list; // every faction's town buildings: (faction, id) is the key, ids are BuildingID numbers
	for(const auto & f : LIBRARY->townh->objects)
	{
		if(!f || !f->town)
			continue;
		for(const auto & [id, b] : f->town->buildings)
		{
			JsonNode node;
			node["faction"].Integer() = f->getIndex();
			node["id"].Integer() = id.getNum();
			node["key"].String() = b->getJsonKey();
			node["name"].String() = b->getNameTranslated();
			node["cost"] = resourcesJson(b->resources);
			node["produce"] = resourcesJson(b->produce); // a day; only the top building of an upgrade chain produces
			node["upgrade"].Integer() = b->upgrade.getNum(); // the building this one upgrades, -1 if none
			node["auto"].Bool() = b->mode == CBuilding::BUILD_AUTO;
			list.Vector().push_back(node);
		}
	}
	return list;
}

JsonNode townCreatures()
{
	JsonNode list; // for each faction with a town: creatures[level] = [base, upgrade, ...], horde_levels[0..1]
	for(const auto & f : LIBRARY->townh->objects)
	{
		if(!f || !f->town)
			continue;
		JsonNode node;
		node["faction"].Integer() = f->getIndex();
		for(const auto & level : f->town->creatures)
		{
			JsonNode ids;
			for(const auto & creature : level)
				ids.Vector().emplace_back(static_cast<si64>(creature.getNum()));
			node["creatures"].Vector().push_back(ids);
		}
		for(int horde = 0; horde < 2; ++horde)
			node["horde_levels"].Vector().emplace_back(static_cast<si64>(f->town->hordeLvl.count(horde) ? f->town->hordeLvl.at(horde) : -1));
		list.Vector().push_back(node);
	}
	return list;
}

JsonNode objectTypes()
{
	JsonNode list; // map object types (Obj ids), named by their first subtype, with their handler (e.g. "mine")
	for(const auto & type : LIBRARY->objtypeh->knownObjects())
	{
		JsonNode node;
		node["id"].Integer() = type.getNum();
		node["handler"].String() = LIBRARY->objtypeh->getObjectHandlerName(type);
		const auto subtypes = LIBRARY->objtypeh->knownSubObjects(type);
		node["name"].String() = subtypes.empty() ? "" : LIBRARY->objtypeh->getObjectName(type, *subtypes.begin());
		node["subtypes"].Integer() = static_cast<si64>(subtypes.size());
		list.Vector().push_back(node);
	}
	return list;
}

JsonNode experience()
{
	JsonNode list; // the experience a hero needs for level 1, 2, ...
	for(ui32 level = 1; level <= LIBRARY->heroh->maxSupportedLevel(); ++level)
		list.Vector().emplace_back(static_cast<si64>(LIBRARY->heroh->reqExp(level)));
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
	vocab["hero_classes"] = heroClasses();
	vocab["factions"] = factions();
	vocab["experience"] = experience();
	vocab["buildings"] = buildings();
	vocab["town_creatures"] = townCreatures();
	vocab["object_types"] = objectTypes();
	vocab["terrains"] = simple(LIBRARY->terrainTypeHandler->objects);
	vocab["roads"] = simple(LIBRARY->roadTypeHandler->objects);
	// Every table's keys by runtime id, the engine facts and hashes (alpha_content): what the match runs against
	vocab["content"] = alpha_content::readEngine().toJson();
	std::ofstream(out) << vocab.toString();
	std::cout << "wrote " << out << "\n";

	logConfigurator.deconfigure();
	delete LIBRARY;
	return 0;
}
