// Drives MoraleWeight.h (alpha_heroes patch, the fresh run) for campaign_model/tests/test_morale_weight.py. Commands on
// stdin, one a line, one printed line each, with VCMI's tables (config/gameConfig.json: good 1 2 3, bad 2 4 6, dice 24):
//   mult MORALE WEIGHT                     the multiplier, %.17g
//   value WEIGHT POWER MORALE [POWER MORALE ...]   an army's weighed power (moraleWeighted's sum)
#include "../MoraleWeight.h"

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

int main()
{
	const std::vector<int> bad{2, 4, 6}, good{1, 2, 3};
	const int dice = 24;
	std::string line;
	while(std::getline(std::cin, line))
	{
		std::istringstream in(line);
		std::string command;
		in >> command;
		if(command == "mult")
		{
			int morale;
			double weight;
			in >> morale >> weight;
			std::printf("%.17g\n", armies::moraleMultiplier(morale, bad, good, dice, weight));
		}
		else if(command == "value")
		{
			double weight;
			in >> weight;
			uint64_t power, total = 0;
			int morale;
			while(in >> power >> morale)
				total += armies::weighedPower(power, morale, bad, good, dice, weight);
			std::printf("%llu\n", static_cast<unsigned long long>(total));
		}
		std::fflush(stdout);
	}
	return 0;
}
