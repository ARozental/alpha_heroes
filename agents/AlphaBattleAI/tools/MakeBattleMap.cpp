// Builds a "battle only" map from a battle_model scenario file: two heroes next to each other, so
// the game starts directly with their battle and ends right after it (VCMI's Battle Only mode).
// Hero 0 attacks hero 1. Built with the same engine code as the lobby's Battle Only mode.
//
// Usage: VCMI_USER_DIR=... build/bin/alpha_make_battle_map <scenario.json> <out.vmap>
#include "StdInc.h"

#include "lib/CConsoleHandler.h"
#include "lib/GameLibrary.h"
#include "lib/VCMIDirs.h"
#include "lib/callback/EditorCallback.h"
#include "lib/entities/artifact/CArtifact.h"
#include "lib/entities/hero/CHero.h"
#include "lib/entities/hero/CHeroClass.h"
#include "lib/json/JsonNode.h"
#include "lib/logging/CBasicLogConfigurator.h"
#include "lib/mapObjectConstructors/AObjectTypeHandler.h"
#include "lib/mapObjectConstructors/CObjectClassesHandler.h"
#include "lib/mapObjects/CGHeroInstance.h"
#include "lib/mapping/CMap.h"
#include "lib/mapping/CMapEditManager.h"
#include "lib/mapping/CMapService.h"
#include "lib/modding/IdentifierStorage.h"
#include "lib/CRandomGenerator.h"
#include "lib/CSkillHandler.h"
#include "lib/mapping/MapFormat.h"

#include <fstream>
#include <sstream>

namespace
{
si32 lookup(const std::string & type, const std::string & name)
{
	auto id = LIBRARY->identifiers()->getIdentifier("core", type, name, true);
	if(!id)
		throw std::runtime_error("unknown " + type + " '" + name + "'");
	return *id;
}

void addHero(CMap & map, EditorCallback & cb, const JsonNode & side, PlayerColor color, const int3 & position)
{
	const HeroTypeID heroType(lookup("hero", side["hero"].String()));
	auto factory = LIBRARY->objtypeh->getHandlerFor(Obj::HERO, heroType.toHeroType()->heroClass->getId());
	auto hero = std::dynamic_pointer_cast<CGHeroInstance>(factory->create(&cb, factory->getTemplates().front()));
	hero->setHeroType(heroType);
	hero->setOwner(color);
	hero->pos = position;

	const auto & primary = side["primary_skills"].Vector();
	for(size_t i = 0; i < GameConstants::PRIMARY_SKILLS && i < primary.size(); i++)
		hero->pushPrimSkill(PrimarySkill(i), primary[i].Integer());

	hero->clearSlots();
	int slot = 0;
	for(const auto & stack : side["army"].Vector())
		hero->setCreature(SlotID(slot++), CreatureID(lookup("creature", stack.Vector()[0].String())), stack.Vector()[1].Integer());

	if(side["spellbook"].Bool() && !hero->getArt(ArtifactPosition::SPELLBOOK))
		hero->putArtifact(ArtifactPosition::SPELLBOOK, map.createArtifact(ArtifactID::SPELLBOOK));
	else if(!side["spellbook"].Bool() && hero->getArt(ArtifactPosition::SPELLBOOK))
		hero->removeArtifact(ArtifactPosition::SPELLBOOK);
	for(const auto & spell : side["spells"].Vector())
		hero->addSpellToSpellbook(SpellID(lookup("spell", spell.String())));

	if(side["war_machines"].Bool())
	{
		hero->putArtifact(ArtifactPosition::MACH1, map.createArtifact(ArtifactID::BALLISTA));
		hero->putArtifact(ArtifactPosition::MACH2, map.createArtifact(ArtifactID::AMMO_CART));
		hero->putArtifact(ArtifactPosition::MACH3, map.createArtifact(ArtifactID::FIRST_AID_TENT));
	}

	for(const auto & skill : LIBRARY->skillh->objects) // no secondary skills unless listed
		hero->setSecSkillLevel(SecondarySkill(skill->getId()), MasteryLevel::NONE, ChangeValueMode::ABSOLUTE);
	for(const auto & skill : side["secondary_skills"].Vector())
		hero->setSecSkillLevel(SecondarySkill(lookup("skill", skill.Vector()[0].String())), static_cast<MasteryLevel::Type>(skill.Vector()[1].Integer()), ChangeValueMode::ABSOLUTE);

	map.getEditManager()->insertObject(hero);
}
}

int main(int argc, const char * argv[])
{
	if(argc != 3)
	{
		std::cerr << "usage: alpha_make_battle_map <scenario.json> <out.vmap>\n";
		return 2;
	}
	const std::string scenarioPath = boost::filesystem::absolute(argv[1]).string();
	const std::string out = boost::filesystem::absolute(argv[2]).string();
	boost::filesystem::current_path(boost::filesystem::system_complete(argv[0]).parent_path());

	CConsoleHandler console;
	CBasicLogConfigurator logConfigurator(VCMIDirs::get().userLogsPath() / "VCMI_MakeBattleMap_log.txt", &console);
	logConfigurator.configureDefault();
	LIBRARY = new GameLibrary;
	LIBRARY->initializeFilesystem(false);
	logConfigurator.configure();
	LIBRARY->initializeLibrary();

	std::stringstream text;
	text << std::ifstream(scenarioPath).rdbuf();
	const std::string data = text.str();
	const JsonNode scenario(reinterpret_cast<const std::byte *>(data.data()), data.size(), scenarioPath);

	auto map = std::make_unique<CMap>(nullptr);
	map->version = EMapFormat::VCMI;
	map->creationDateTime = std::time(nullptr);
	map->width = 10;
	map->height = 10;
	map->mapLayers = {MapLayerId::SURFACE};
	map->battleOnly = true;
	map->name = MetaString::createFromRawString(scenario["name"].String());
	EditorCallback cb(map.get());
	map->cb = &cb;

	auto * rng = &CRandomGenerator::getDefault();
	map->initTerrain();
	map->getEditManager()->clearTerrain(rng);
	map->getEditManager()->getTerrainSelection().selectAll();
	map->getEditManager()->drawTerrain(TerrainId(lookup("terrain", scenario["terrain"].String())), 0, rng);

	map->players[0].canComputerPlay = true;
	map->players[0].canHumanPlay = true;
	map->players[1] = map->players[0];

	const auto & sides = scenario["sides"].Vector();
	if(sides.size() != 2)
		throw std::runtime_error("a scenario needs exactly 2 sides");
	addHero(*map, cb, sides[0], PlayerColor(0), int3(5, 6, 0)); // attacker
	addHero(*map, cb, sides[1], PlayerColor(1), int3(5, 5, 0)); // defender

	CMapService mapService;
	mapService.saveMap(map, out);
	std::cout << "wrote " << out << "\n";

	map.reset(); // the map refers to the library's texts: free it first
	logConfigurator.deconfigure();
	delete LIBRARY;
	return 0;
}
