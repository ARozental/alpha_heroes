// Writes a random map from VCMI's generator (lib/rmg), as the game's random map screen would make it, for self-play
// games of the campaign model (campaign_model/play.py). Water and the underground are left on: they are part of the
// game the model must learn.
//
// Usage: VCMI_USER_DIR=... build/bin/alpha_make_random_map <out.vmap> <seed> [size=36] [players=2] [levels=2]
//        [water=random|none|normal|islands] [template name, default: any that fits]
#include "StdInc.h"

#include "lib/CConsoleHandler.h"
#include "lib/GameLibrary.h"
#include "lib/VCMIDirs.h"
#include "lib/callback/EditorCallback.h"
#include "lib/logging/CBasicLogConfigurator.h"
#include "lib/mapping/CMap.h"
#include "lib/mapping/CMapService.h"
#include "lib/rmg/CMapGenOptions.h"
#include "lib/rmg/CMapGenerator.h"
#include "lib/rmg/CRmgTemplate.h"

int main(int argc, const char * argv[])
{
	if(argc < 3)
	{
		std::cerr << "usage: alpha_make_random_map <out.vmap> <seed> [size=36] [players=2] [levels=2] [water=random] [template]\n";
		return 2;
	}
	const std::string out = boost::filesystem::absolute(argv[1]).string();
	const int seed = std::stoi(argv[2]);
	const int size = argc > 3 ? std::stoi(argv[3]) : 36;
	const int players = argc > 4 ? std::stoi(argv[4]) : 2;
	const int levels = argc > 5 ? std::stoi(argv[5]) : 2;
	const std::string water = argc > 6 ? argv[6] : "random";
	const std::string templateName = argc > 7 ? argv[7] : "";
	boost::filesystem::current_path(boost::filesystem::system_complete(argv[0]).parent_path());

	CConsoleHandler console;
	CBasicLogConfigurator logConfigurator(VCMIDirs::get().userLogsPath() / "VCMI_MakeRandomMap_log.txt", &console);
	logConfigurator.configureDefault();
	LIBRARY = new GameLibrary;
	LIBRARY->initializeFilesystem(false);
	logConfigurator.configure();
	LIBRARY->initializeLibrary();

	const std::map<std::string, EWaterContent::EWaterContent> waterContents = {
		{"random", EWaterContent::RANDOM}, {"none", EWaterContent::NONE}, {"normal", EWaterContent::NORMAL}, {"islands", EWaterContent::ISLANDS}};
	CMapGenOptions options;
	options.setWidth(size);
	options.setHeight(size);
	options.setLevels(levels);
	options.setHumanOrCpuPlayerCount(players);
	options.setCompOnlyPlayerCount(0);
	options.setWaterContent(waterContents.at(water));
	options.setMonsterStrength(EMonsterStrength::RANDOM);
	if(!templateName.empty())
		options.setMapTemplate(templateName);

	int status = 0;
	try
	{
		EditorCallback cb(nullptr);
		CMapGenerator generator(options, &cb, seed);
		auto map = generator.generate();
		cb.setMap(map.get()); // saving reads the map's artifacts through it
		CMapService mapService;
		mapService.saveMap(map, out);
		std::cout << "wrote " << out << " (" << map->width << "x" << map->height << ", " << map->mapLayers.size() << " levels, template "
				  << (options.getMapTemplate() ? options.getMapTemplate()->getName() : std::string("?")) << ")\n";
	}
	catch(const std::exception & e)
	{
		std::cerr << "generating the map failed: " << e.what() << "\n";
		status = 1;
	}
	logConfigurator.deconfigure();
	return status;
}
