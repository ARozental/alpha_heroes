#include "StdInc.h"
#include "Content.h"

#include "lib/BattleFieldHandler.h"
#include "lib/CBonusTypeHandler.h"
#include "lib/CCreatureHandler.h"
#include "lib/CSkillHandler.h"
#include "lib/GameLibrary.h"
#include "lib/ObstacleHandler.h"
#include "lib/RiverHandler.h"
#include "lib/RoadHandler.h"
#include "lib/TerrainHandler.h"
#include "lib/battle/BattleHex.h"
#include "lib/bonuses/BonusEnum.h"
#include "lib/entities/artifact/CArtHandler.h"
#include "lib/entities/artifact/CArtifact.h"
#include "lib/entities/building/CBuilding.h"
#include "lib/entities/faction/CFaction.h"
#include "lib/entities/faction/CTown.h"
#include "lib/entities/faction/CTownHandler.h"
#include "lib/entities/hero/CHero.h"
#include "lib/entities/hero/CHeroClass.h"
#include "lib/entities/hero/CHeroClassHandler.h"
#include "lib/entities/hero/CHeroHandler.h"
#include "lib/json/JsonNode.h"
#include "lib/mapObjectConstructors/AObjectTypeHandler.h"
#include "lib/mapObjectConstructors/CObjectClassesHandler.h"
#include "lib/spells/CSpellHandler.h"

namespace alpha_content
{
namespace
{
// The engine's built-in bonus types by BonusType (its bonus type handler keeps them private)
#define BONUS_NAME(x) #x,
const std::vector<std::string> BONUS_NAMES = {BONUS_LIST};
#undef BONUS_NAME

void place(std::vector<std::string> & keys, int index, const std::string & key)
{
	if(index < 0)
		return;
	if(keys.size() <= static_cast<size_t>(index))
		keys.resize(index + 1);
	keys[index] = key;
}

template<typename Objects>
std::vector<std::string> byIndex(const Objects & objects)
{
	std::vector<std::string> keys;
	for(const auto & o : objects)
		if(o)
			place(keys, o->getIndex(), o->getJsonKey());
	return keys;
}

// FNV-1a, 64 bits: the Python mirror (content.py) computes the same over the same text
std::string fnv(const std::string & text)
{
	uint64_t hash = 14695981039346656037ULL;
	for(const unsigned char c : text)
	{
		hash ^= c;
		hash *= 1099511628211ULL;
	}
	char out[17];
	std::snprintf(out, sizeof out, "%016llx", static_cast<unsigned long long>(hash));
	return out;
}

std::string joined(const std::vector<std::string> & keys, size_t shown)
{
	std::string out;
	for(size_t i = 0; i < keys.size() && i < shown; ++i)
		out += (i ? ", " : "") + keys[i];
	if(keys.size() > shown)
		out += ", ... (" + std::to_string(keys.size() - shown) + " more)";
	return out;
}
}

std::string Content::contentHash() const
{
	std::string text;
	for(const auto & [name, keys] : tables)
	{
		std::vector<std::string> sorted;
		for(const auto & key : keys)
			if(!key.empty())
				sorted.push_back(key);
		std::sort(sorted.begin(), sorted.end());
		text += name + "\n";
		for(const auto & key : sorted)
			text += key + "\n";
		text += "\n";
	}
	return fnv(text);
}

std::string Content::vocabularyHash() const
{
	std::string text;
	for(const auto & [name, keys] : tables)
	{
		text += name + "\n";
		for(const auto & key : keys)
			text += key + "\n";
		text += "\n";
	}
	return fnv(text);
}

JsonNode Content::toJson() const
{
	JsonNode node;
	for(const auto & [name, keys] : tables)
	{
		auto & list = node["tables"][name];
		list.Vector();
		for(const auto & key : keys)
			list.Vector().emplace_back(key);
	}
	for(const auto & [name, value] : facts)
		node["facts"][name].Integer() = value;
	for(const auto & rename : renamed)
		node["renamed"].Vector().emplace_back(rename);
	node["content_hash"].String() = contentHash();
	node["vocabulary_hash"].String() = vocabularyHash();
	return node;
}

Content Content::fromJson(const JsonNode & node)
{
	Content content;
	for(const auto & [name, list] : node["tables"].Struct())
	{
		auto & keys = content.tables[name];
		for(const auto & key : list.Vector())
			keys.push_back(key.String());
	}
	for(const auto & [name, value] : node["facts"].Struct())
		content.facts[name] = value.Integer();
	for(const auto & rename : node["renamed"].Vector())
		content.renamed.push_back(rename.String());
	return content;
}

Content readEngine()
{
	Content content;
	auto & t = content.tables;
	t["creatures"] = byIndex(LIBRARY->creh->objects);
	t["artifacts"] = byIndex(LIBRARY->arth->objects);
	t["spells"] = byIndex(LIBRARY->spellh->objects);
	t["secondary_skills"] = byIndex(LIBRARY->skillh->objects);
	t["heroes"] = byIndex(LIBRARY->heroh->objects);
	t["hero_classes"] = byIndex(LIBRARY->heroclassesh->objects);
	t["factions"] = byIndex(LIBRARY->townh->objects);
	t["terrains"] = byIndex(LIBRARY->terrainTypeHandler->objects);
	t["roads"] = byIndex(LIBRARY->roadTypeHandler->objects);
	t["rivers"] = byIndex(LIBRARY->riverTypeHandler->objects);
	t["battlefields"] = byIndex(LIBRARY->battlefieldsHandler->objects);
	t["obstacles"] = byIndex(LIBRARY->obstacleHandler->objects);
	t["bonus_types"] = BONUS_NAMES;
	// A town's buildings, by BuildingID within its faction
	for(const auto & faction : LIBRARY->townh->objects)
		if(faction && faction->town)
		{
			auto & keys = t["buildings/" + faction->getJsonKey()];
			keys.clear();
			for(const auto & [id, building] : faction->town->buildings)
				if(building)
					place(keys, id.getNum(), building->getJsonKey());
		}
	// Map object types, each type's subtypes by subtype id
	auto & types = t["object_types"];
	for(const auto & type : LIBRARY->objtypeh->knownObjects())
	{
		const auto key = LIBRARY->objtypeh->getJsonKey(type);
		place(types, type.getNum(), key);
		auto & subtypes = t["subtypes/" + key];
		for(const auto & subtype : LIBRARY->objtypeh->knownSubObjects(type))
			if(const auto handler = LIBRARY->objtypeh->getHandlerFor(type, subtype))
				place(subtypes, subtype.getNum(), handler->getJsonKey());
	}
	// A key twice in a table (HotA's hero camp is a second prison subtype also named "prison"): the later ones are
	// key#2, key#3... in runtime id order, each listed, so every entity has its own row
	for(auto & [name, keys] : t)
	{
		std::map<std::string, int> seen;
		for(auto & key : keys)
			if(!key.empty() && ++seen[key] > 1)
			{
				const auto renamed = key + "#" + std::to_string(seen[key]);
				content.renamed.push_back(name + ": " + key + " -> " + renamed);
				key = renamed;
			}
	}
	auto & f = content.facts;
	f["primary_skills"] = GameConstants::PRIMARY_SKILLS;
	f["resources"] = GameConstants::RESOURCE_QUANTITY;
	f["army_slots"] = GameConstants::ARMY_SIZE;
	f["battle_hexes"] = GameConstants::BFIELD_SIZE;
	f["worn_slots"] = static_cast<int64_t>(ArtifactPosition::BACKPACK_START);
	f["max_hero_level"] = static_cast<int64_t>(LIBRARY->heroh->maxSupportedLevel());
	return content;
}

bool Match::clean(bool strict) const
{
	for(const auto & [name, table] : tables)
		if(!table.collisions.empty())
			return false;
	return factDiffs.empty() && (!strict || unknownCount() == 0);
}

size_t Match::unknownCount() const
{
	size_t n = 0;
	for(const auto & [name, table] : tables)
		n += table.unknown.size();
	return n;
}

std::string Match::report(size_t keysShown) const
{
	std::string out = "content hash " + contentHash + ", vocabulary hash " + vocabularyHash + "\n";
	size_t matched = 0, unknown = 0, absent = 0, collisions = 0, quiet = 0;
	for(const auto & [name, table] : tables)
	{
		for(const auto row : table.rows)
			matched += row >= 0;
		unknown += table.unknown.size();
		absent += table.absent.size();
		collisions += table.collisions.size();
		if(table.unknown.empty() && table.absent.empty() && table.collisions.empty())
		{
			++quiet;
			continue;
		}
		out += name + ":";
		if(!table.unknown.empty())
			out += " unknown " + std::to_string(table.unknown.size()) + " [" + joined(table.unknown, keysShown) + "]";
		if(!table.absent.empty())
			out += " absent " + std::to_string(table.absent.size()) + " [" + joined(table.absent, keysShown) + "]";
		if(!table.collisions.empty())
			out += " COLLISIONS " + std::to_string(table.collisions.size()) + " [" + joined(table.collisions, keysShown) + "]";
		out += "\n";
	}
	for(const auto & diff : factDiffs)
		out += "FACT DIFFERS: " + diff + "\n";
	out += std::to_string(tables.size()) + " tables (" + std::to_string(quiet) + " matched in full): " + std::to_string(matched) + " matched, " + std::to_string(unknown) + " unknown, " + std::to_string(absent) + " absent, " + std::to_string(collisions) + " collisions, " + std::to_string(factDiffs.size()) + " fact differences\n";
	return out;
}

Match match(const Content & engine, const Content & model)
{
	Match out;
	out.contentHash = engine.contentHash();
	out.vocabularyHash = model.vocabularyHash();
	std::set<std::string> names;
	for(const auto & [name, keys] : engine.tables)
		names.insert(name);
	for(const auto & [name, keys] : model.tables)
		names.insert(name);
	const std::vector<std::string> none;
	for(const auto & name : names)
	{
		const auto & engineKeys = engine.tables.count(name) ? engine.tables.at(name) : none;
		const auto & modelKeys = model.tables.count(name) ? model.tables.at(name) : none;
		auto & table = out.tables[name];
		std::map<std::string, int> rowOf;
		for(size_t row = 0; row < modelKeys.size(); ++row)
			if(!modelKeys[row].empty() && !rowOf.emplace(modelKeys[row], static_cast<int>(row)).second)
				table.collisions.push_back(modelKeys[row] + " (twice in the vocabulary)");
		std::map<std::string, size_t> idOf;
		table.rows.assign(engineKeys.size(), -1);
		for(size_t id = 0; id < engineKeys.size(); ++id)
		{
			const auto & key = engineKeys[id];
			if(key.empty())
				continue;
			const auto [first, fresh] = idOf.emplace(key, id);
			if(!fresh)
				table.collisions.push_back(key + " (engine ids " + std::to_string(first->second) + " and " + std::to_string(id) + ")");
			const auto row = rowOf.find(key);
			if(row != rowOf.end())
				table.rows[id] = row->second;
			else if(fresh)
				table.unknown.push_back(key);
		}
		for(const auto & [key, row] : rowOf)
			if(!idOf.count(key))
				table.absent.push_back(key);
	}
	for(const auto & [name, value] : model.facts)
	{
		const auto engineValue = engine.facts.find(name);
		if(engineValue == engine.facts.end())
			out.factDiffs.push_back(name + ": the vocabulary's " + std::to_string(value) + ", the engine has none");
		else if(engineValue->second != value)
			out.factDiffs.push_back(name + ": the vocabulary's " + std::to_string(value) + ", the engine's " + std::to_string(engineValue->second));
	}
	return out;
}
}
