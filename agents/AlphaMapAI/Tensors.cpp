#include "StdInc.h"
#include "Tensors.h"

#include <cstring>
#include <stdexcept>

using namespace MapSchema;

int64_t flatIndex(const TensorSpec & spec, int64_t i, int64_t j, int64_t k, int64_t l)
{
	if(i < 0 || j < 0 || k < 0 || l < 0 || i >= spec.dims[0] || j >= spec.dims[1] || k >= spec.dims[2] || l >= spec.dims[3])
		throw std::out_of_range(std::string("index out of range in ") + spec.name);
	return ((i * spec.dims[1] + j) * spec.dims[2] + k) * spec.dims[3] + l;
}

Inputs::Inputs()
{
	for(size_t n = 0; n < INPUTS.size(); ++n)
		values[n].assign(INPUTS[n].size(), INPUTS[n].fill);
}

int64_t & Inputs::at(In name, int64_t i, int64_t j, int64_t k, int64_t l)
{
	const auto n = static_cast<size_t>(name);
	return values[n][flatIndex(INPUTS[n], i, j, k, l)];
}

int64_t Inputs::get(In name, int64_t i, int64_t j, int64_t k, int64_t l) const
{
	const auto n = static_cast<size_t>(name);
	return values[n][flatIndex(INPUTS[n], i, j, k, l)];
}

bool Inputs::any(In name) const
{
	for(auto value : values[static_cast<size_t>(name)])
		if(value)
			return true;
	return false;
}

std::string Inputs::wire() const
{
	std::string out;
	for(size_t n = 0; n < INPUTS.size(); ++n)
	{
		if(INPUTS[n].dtype == DType::BOOL)
		{
			for(auto value : values[n])
				out.push_back(value ? 1 : 0);
		}
		else
		{
			const auto start = out.size();
			out.resize(start + values[n].size() * 4);
			for(size_t i = 0; i < values[n].size(); ++i)
			{
				const auto value = static_cast<int32_t>(values[n][i]);
				std::memcpy(out.data() + start + i * 4, &value, 4);
			}
		}
	}
	return out;
}

Outputs Outputs::fromWire(const std::string & bytes)
{
	Outputs outputs;
	size_t offset = 0;
	for(size_t n = 0; n < OUTPUTS.size(); ++n)
	{
		const auto size = static_cast<size_t>(OUTPUTS[n].size());
		if(offset + size * 4 > bytes.size())
			throw std::runtime_error("inference reply too short");
		outputs.values[n].resize(size);
		std::memcpy(outputs.values[n].data(), bytes.data() + offset, size * 4);
		offset += size * 4;
	}
	if(offset != bytes.size())
		throw std::runtime_error("inference reply too long");
	return outputs;
}
