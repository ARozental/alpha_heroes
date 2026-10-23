#pragma once

#include "Tensors.h"

#include <string>

struct ChosenAction;

/// Writes one decision (all inputs and outputs as .npy, the chosen action as action.json) into the
/// compressed archive <dir>/<name>.npz, read by battle_model/data.py and check_dumps.py.
void dumpDecision(const std::string & dir, const std::string & name, const AlphaSchema::InputTensors & inputs, const AlphaSchema::OutputTensors & outputs, const ChosenAction & action);
