// Prints the headers of the game's maps as JSON lines, read by VCMI's own loader, so the campaign games can draw from
// the game's maps as well as generated ones (campaign_model/maps.py): size, levels, the players who can play, teams,
// the victory and loss conditions' identifiers (standardVictory, standardDefeat or the map's own).
//
// Usage: VCMI_USER_DIR=... build/bin/alpha_map_info "Maps/Elbow Room" ["Maps/..." ...]
#include "StdInc.h"

#include "lib/CConsoleHandler.h"
#include "lib/GameLibrary.h"
#include "lib/VCMIDirs.h"
#include "lib/filesystem/ResourcePath.h"
#include "lib/logging/CBasicLogConfigurator.h"
#include "lib/mapping/CMapHeader.h"
#include "lib/mapping/CMapService.h"

int main(int argc, const char * argv[])
{
	if(argc < 2)
	{
		std::cerr << "usage: alpha_map_info <map> [<map> ...]\n";
		return 2;
	}
	boost::filesystem::current_path(boost::filesystem::system_complete(argv[0]).parent_path());
	CConsoleHandler console;
	CBasicLogConfigurator logConfigurator(VCMIDirs::get().userLogsPath() / "VCMI_MapInfo_log.txt", &console);
	logConfigurator.configureDefault();
	LIBRARY = new GameLibrary;
	LIBRARY->initializeFilesystem(false);
	logConfigurator.configure();
	LIBRARY->initializeLibrary();

	CMapService mapService;
	for(int i = 1; i < argc; ++i)
	{
		JsonNode out;
		out["map"].String() = argv[i];
		try
		{
			const auto header = mapService.loadMapHeader(ResourcePath(argv[i], EResType::MAP));
			out["width"].Integer() = header->width;
			out["height"].Integer() = header->height;
			out["levels"].Integer() = header->levels();
			out["teams"].Integer() = header->howManyTeams;
			for(size_t p = 0; p < header->players.size(); ++p)
			{
				const auto & player = header->players[p];
				if(!player.canAnyonePlay())
					continue;
				JsonNode node;
				node["color"].Integer() = static_cast<int>(p);
				node["human"].Bool() = player.canHumanPlay;
				node["computer"].Bool() = player.canComputerPlay;
				node["team"].Integer() = player.team.getNum();
				node["main_town"].Bool() = player.hasMainTown;
				out["players"].Vector().push_back(node);
			}
			for(const auto & event : header->triggeredEvents)
			{
				JsonNode node;
				node.String() = event.identifier;
				out["events"].Vector().push_back(node);
			}
		}
		catch(const std::exception & e)
		{
			out["error"].String() = e.what();
		}
		std::cout << out.toCompactString() << "\n";
	}
	logConfigurator.deconfigure();
	return 0;
}
