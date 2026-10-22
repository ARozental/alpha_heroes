#pragma once

#include "Tensors.h"

#include <random>
#include <string>

/// One full action chosen from the model outputs; -1 for fields that do not apply.
/// Same fields as battle_model/distributions.py ACTION_FIELDS.
struct ChosenAction
{
	int type = -1;
	int hex = -1; // MOVE destination / MELEE hex to attack from
	int unit = -1; // MELEE / SHOOT target slot
	int spell = -1;
	int targetUnit = -1;
	int targetHex = -1;
	int target2Hex = -1;
	int target2Unit = -1;

	std::string toJson() const;
};

/// Sample (temperature > 0) or take the most likely (temperature == 0) action, mirroring
/// battle_model/distributions.py sample(). Throws if a needed choice has no legal entry.
ChosenAction chooseAction(const AlphaSchema::OutputTensors & out, std::mt19937 & rng, float temperature);
