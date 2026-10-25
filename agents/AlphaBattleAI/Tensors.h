#pragma once

#include <algorithm>

#include "Schema.generated.h"

#include <array>
#include <cstdint>
#include <vector>

namespace AlphaSchema
{
inline const TensorSpec & spec(In in)
{
	return INPUTS[static_cast<int>(in)];
}

inline const TensorSpec & spec(Out out)
{
	return OUTPUTS[static_cast<int>(out)];
}

int64_t numel(const TensorSpec & spec);

/// Model inputs, all stored as int64 (bool masks too; converted when handed to onnxruntime)
class InputTensors
{
	std::array<std::vector<int64_t>, static_cast<int>(In::COUNT)> values;

	int64_t & at(In in, int i, int j);

public:
	InputTensors(); // every tensor at its fill value

	void set(In in, int64_t value) { at(in, 0, 0) = value; }
	void set(In in, int i, int64_t value) { at(in, i, 0) = value; }
	void set(In in, int i, int j, int64_t value) { at(in, i, j) = value; }
	int64_t get(In in, int i = 0, int j = 0) const;
	bool anyInRow(In in, int i) const; // rank-2 tensors: any non-zero entry in row i
	bool any(In in) const;
	void clear(In in) { std::fill(values[static_cast<int>(in)].begin(), values[static_cast<int>(in)].end(), 0); } // all 0 (masks)

	const std::vector<int64_t> & data(In in) const { return values[static_cast<int>(in)]; }
};

/// Model outputs: masked log-probabilities (-inf = illegal) and the value estimate
class OutputTensors
{
	std::array<std::vector<float>, static_cast<int>(Out::COUNT)> values;

public:
	std::vector<float> & data(Out out) { return values[static_cast<int>(out)]; }
	const std::vector<float> & data(Out out) const { return values[static_cast<int>(out)]; }
	/// Row i of a rank-2 output, or the whole rank-1 output when i is 0
	const float * row(Out out, int i = 0) const;
	int rowSize(Out out) const;
};
}
