#include "StdInc.h"
#include "Policy.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

using namespace AlphaSchema;

namespace
{
int choose(const float * logp, int n, std::mt19937 & rng, float temperature, const char * what)
{
	int best = -1;
	for(int i = 0; i < n; ++i)
		if(std::isfinite(logp[i]) && (best < 0 || logp[i] > logp[best]))
			best = i;
	if(best < 0)
		throw std::runtime_error(std::string("no legal choice for ") + what);
	if(temperature <= 0)
		return best;

	std::vector<double> weights(n, 0.0);
	for(int i = 0; i < n; ++i)
		if(std::isfinite(logp[i]))
			weights[i] = std::exp((static_cast<double>(logp[i]) - logp[best]) / temperature);
	std::discrete_distribution<int> distribution(weights.begin(), weights.end());
	return distribution(rng);
}

bool anyLegal(const float * logp, int n)
{
	for(int i = 0; i < n; ++i)
		if(std::isfinite(logp[i]))
			return true;
	return false;
}
}

ChosenAction chooseAction(const OutputTensors & out, std::mt19937 & rng, float temperature)
{
	ChosenAction a;
	auto pick = [&](Out o, int row, const char * what) { return choose(out.row(o, row), out.rowSize(o), rng, temperature, what); };

	a.type = pick(Out::logp_type, 0, "action type");
	switch(static_cast<ActionType>(a.type))
	{
		case ActionType::MOVE:
			a.hex = pick(Out::logp_move_hex, 0, "move hex");
			break;
		case ActionType::MELEE:
			a.unit = pick(Out::logp_melee_target, 0, "melee target");
			a.hex = pick(Out::logp_melee_from, a.unit, "melee hex");
			break;
		case ActionType::SHOOT:
			a.unit = pick(Out::logp_shoot_target, 0, "shoot target");
			break;
		case ActionType::HERO_SPELL:
		{
			a.spell = pick(Out::logp_spell, 0, "spell");
			if(anyLegal(out.row(Out::logp_spell_unit, a.spell), out.rowSize(Out::logp_spell_unit)))
			{
				a.targetUnit = pick(Out::logp_spell_unit, a.spell, "spell unit target");
				const auto second = secondTargetKind(a.spell);
				if(second == TargetKind::HEX)
					a.target2Hex = pick(Out::logp_spell2_hex, a.targetUnit, "spell second hex");
				else if(second == TargetKind::UNIT)
					a.target2Unit = pick(Out::logp_spell2_unit, a.targetUnit, "spell second unit");
			}
			else if(anyLegal(out.row(Out::logp_spell_hex, a.spell), out.rowSize(Out::logp_spell_hex)))
			{
				a.targetHex = pick(Out::logp_spell_hex, a.spell, "spell hex target");
			}
			break;
		}
		default:
			break;
	}
	return a;
}

std::string ChosenAction::toJson() const
{
	std::ostringstream s;
	s << "{\"type\": " << type << ", \"hex\": " << hex << ", \"unit\": " << unit << ", \"spell\": " << spell << ", \"target_unit\": " << targetUnit << ", \"target_hex\": " << targetHex << ", \"target2_hex\": " << target2Hex << ", \"target2_unit\": " << target2Unit << (forced ? ", \"forced\": true" : "") << "}";
	return s.str();
}
