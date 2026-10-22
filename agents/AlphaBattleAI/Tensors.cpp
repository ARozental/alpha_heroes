#include "StdInc.h"
#include "Tensors.h"

#include <stdexcept>
#include <string>

namespace AlphaSchema
{
int64_t numel(const TensorSpec & spec)
{
	int64_t n = 1;
	for(int d = 0; d < spec.rank; ++d)
		n *= spec.dims[d];
	return n;
}

InputTensors::InputTensors()
{
	for(int i = 0; i < static_cast<int>(In::COUNT); ++i)
		values[i].assign(numel(INPUTS[i]), INPUTS[i].fill);
}

int64_t & InputTensors::at(In in, int i, int j)
{
	const auto & s = spec(in);
	const int64_t index = s.rank == 2 ? i * s.dims[1] + j : i;
	auto & v = values[static_cast<int>(in)];
	if(index < 0 || index >= static_cast<int64_t>(v.size()))
		throw std::out_of_range(std::string("index out of range for input ") + s.name);
	return v[index];
}

int64_t InputTensors::get(In in, int i, int j) const
{
	return const_cast<InputTensors *>(this)->at(in, i, j);
}

bool InputTensors::anyInRow(In in, int i) const
{
	const auto & s = spec(in);
	const auto & v = data(in);
	for(int j = 0; j < s.dims[1]; ++j)
		if(v[i * s.dims[1] + j])
			return true;
	return false;
}

bool InputTensors::any(In in) const
{
	for(auto x : data(in))
		if(x)
			return true;
	return false;
}

const float * OutputTensors::row(Out out, int i) const
{
	return data(out).data() + static_cast<size_t>(i) * rowSize(out);
}

int OutputTensors::rowSize(Out out) const
{
	const auto & s = spec(out);
	return s.rank == 0 ? 1 : static_cast<int>(s.dims[s.rank - 1]);
}
}
