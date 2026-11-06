// The content match from the command line (docs/hota_transfer_plan.md, Part 1). The engine as VCMI_USER_DIR loads
// it (its mods) against a vocabulary: per table, what is unknown to the model, absent from the engine, or colliding.
//
// Usage: VCMI_USER_DIR=... build/bin/alpha_content_check <vocabulary.json> [--strict] [--engine <out.json>]
// - The vocabulary: a vocab.json with the exporter's "content" section, or a content file itself.
// - --strict: any unknown fails too, as a training game would.
// - --engine: also writes what the engine loaded (keys by runtime id).
// Exit status: 0 clean, 1 not, 2 usage.
#include "StdInc.h"
#include "Content.h"

#include "lib/CConsoleHandler.h"
#include "lib/GameLibrary.h"
#include "lib/VCMIDirs.h"
#include "lib/json/JsonNode.h"
#include "lib/logging/CBasicLogConfigurator.h"

#include <fstream>
#include <sstream>

int main(int argc, const char * argv[])
{
	std::string vocabulary, engineOut;
	bool strict = false;
	for(int i = 1; i < argc; ++i)
	{
		const std::string arg = argv[i];
		if(arg == "--strict")
			strict = true;
		else if(arg == "--engine" && i + 1 < argc)
			engineOut = boost::filesystem::absolute(argv[++i]).string();
		else if(vocabulary.empty())
			vocabulary = boost::filesystem::absolute(arg).string();
		else
			vocabulary.clear(), i = argc;
	}
	if(vocabulary.empty())
	{
		std::cerr << "usage: alpha_content_check <vocabulary.json> [--strict] [--engine <out.json>]\n";
		return 2;
	}
	std::ifstream in(vocabulary);
	if(!in)
	{
		std::cerr << "cannot read " << vocabulary << "\n";
		return 2;
	}
	std::stringstream text;
	text << in.rdbuf();
	const std::string json = text.str();
	// Data paths are relative to the executable's folder, as for vcmiserver
	boost::filesystem::current_path(boost::filesystem::system_complete(argv[0]).parent_path());

	CConsoleHandler console;
	CBasicLogConfigurator logConfigurator(VCMIDirs::get().userLogsPath() / "VCMI_ContentCheck_log.txt", &console);
	logConfigurator.configureDefault();
	LIBRARY = new GameLibrary;
	LIBRARY->initializeFilesystem(false);
	logConfigurator.configure();
	LIBRARY->initializeLibrary();

	const JsonNode node(json.data(), json.size(), vocabulary);
	const auto model = alpha_content::Content::fromJson(node["content"].isNull() ? node : node["content"]);
	const auto engine = alpha_content::readEngine();
	const auto result = alpha_content::match(engine, model);
	for(const auto & rename : engine.renamed)
		std::cout << "renamed (a key twice in the engine): " << rename << "\n";
	std::cout << result.report();
	if(!engineOut.empty())
		std::ofstream(engineOut) << engine.toJson().toString();
	const bool clean = result.clean(strict);
	std::cout << (clean ? "clean" : "NOT CLEAN") << (strict ? " (strict)" : "") << "\n";

	logConfigurator.deconfigure();
	delete LIBRARY;
	return clean ? 0 : 1;
}
