#pragma once

#include "Schema.generated.h"

#include <array>
#include <string>
#include <vector>

/// The model's inputs for one decision, in contract order, at their fill values until set
struct Inputs
{
	std::array<std::vector<int64_t>, static_cast<size_t>(MapSchema::In::COUNT)> values;

	Inputs();
	int64_t & at(MapSchema::In name, int64_t i = 0, int64_t j = 0, int64_t k = 0, int64_t l = 0);
	int64_t get(MapSchema::In name, int64_t i = 0, int64_t j = 0, int64_t k = 0, int64_t l = 0) const;
	bool any(MapSchema::In name) const;
	/// The inference server's encoding (battle_model/inference.py): a bool as one byte, an int64 as an int32
	std::string wire() const;
};

/// The model's outputs for one decision (float32 tables in contract order)
struct Outputs
{
	std::array<std::vector<float>, static_cast<size_t>(MapSchema::Out::COUNT)> values;

	static Outputs fromWire(const std::string & bytes);
	const std::vector<float> & operator[](MapSchema::Out name) const { return values[static_cast<size_t>(name)]; }
};

/// Flat index of (i, j, k, l) in a tensor of this spec
int64_t flatIndex(const MapSchema::TensorSpec & spec, int64_t i, int64_t j, int64_t k, int64_t l);
