#pragma once

#include "Tensors.h"

#include <string>

struct ChosenAction;

/// Writes one decision (all inputs and outputs as .npy, the chosen action as JSON) into
/// <dir>/<name>/ so battle_model/check_dumps.py can replay it through the model in Python.
void dumpDecision(const std::string & dir, const std::string & name, const AlphaSchema::InputTensors & inputs, const AlphaSchema::OutputTensors & outputs, const ChosenAction & action);
