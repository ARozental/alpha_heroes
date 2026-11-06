#pragma once

// The content match (docs/hota_transfer_plan.md, Part 1). The model's vocabulary (each table's keys, in row order)
// is matched by key against what the running engine loaded (each table's keys, by runtime id). Numeric ids and mod
// load order never enter a record or a model. Run at start by every process that encodes, builds or exports content:
// both agents, the exporter, the map tools.
//
// Tables:
// - Plain: "creatures", "artifacts", "spells", "secondary_skills", "heroes", "hero_classes", "factions",
//   "object_types", "terrains", "roads", "rivers", "battlefields", "obstacles", "bonus_types".
// - Per owner:
//   - "buildings/<faction key>" by BuildingID: a building's own key (core:mageGuild1) is shared by every town.
//   - "subtypes/<object type key>" by subtype id.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "Global.h"

VCMI_LIB_NAMESPACE_BEGIN
class JsonNode;
VCMI_LIB_NAMESPACE_END

namespace alpha_content
{
using Tables = std::map<std::string, std::vector<std::string>>;

struct Content
{
	// Engine content: keys by runtime id ("" where no entity has that id). A vocabulary: keys in row order.
	Tables tables;
	// Engine constants the model assumes (primary skills, resources, army slots, ...): compared, never copied
	std::map<std::string, int64_t> facts;
	// readEngine's renames of a key twice in a table ("table: key -> key#2"), shown by the report
	std::vector<std::string> renamed;

	// The engine's content hash: every table's keys, sorted. Load order and numeric ids never enter it
	std::string contentHash() const;
	// A vocabulary's hash: every table's keys in row order. A reorder is another vocabulary
	std::string vocabularyHash() const;
	JsonNode toJson() const;
	static Content fromJson(const JsonNode & node);
};

// What the engine loaded (LIBRARY initialised)
Content readEngine();

struct TableMatch
{
	std::vector<int> rows; // runtime id -> the model's row; -1: no entity there, or unknown to the model
	std::vector<std::string> unknown; // the engine has it, the model does not
	std::vector<std::string> absent; // the model has it, the engine does not (fine)
	std::vector<std::string> collisions; // one key at two runtime ids, or twice in the vocabulary: must be empty
};

struct Match
{
	std::map<std::string, TableMatch> tables;
	std::vector<std::string> factDiffs; // any difference aborts
	std::string contentHash, vocabularyHash;

	// No collision, no fact difference, and in strict mode no unknown either
	bool clean(bool strict) const;
	size_t unknownCount() const;
	// The readable diff: per table its counts and keys
	std::string report(size_t keysShown = 12) const;
};

// engine: readEngine's; model: the vocabulary (keys in row order)
Match match(const Content & engine, const Content & model);
}
