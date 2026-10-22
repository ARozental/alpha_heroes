#include "StdInc.h"
#include "Dump.h"

#include "Policy.h"

#include <filesystem>
#include <fstream>

using namespace AlphaSchema;

namespace
{
/// NumPy .npy format version 1.0 with a leading batch dimension of 1
void writeNpy(const std::filesystem::path & path, const char * descr, const TensorSpec & spec, const void * data, size_t bytes)
{
	std::string shape = "(1,";
	for(int d = 0; d < spec.rank; ++d)
		shape += " " + std::to_string(spec.dims[d]) + ",";
	shape += ")";
	std::string header = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
	const size_t total = 10 + header.size() + 1;
	header.append((64 - total % 64) % 64, ' ');
	header += '\n';

	std::ofstream file(path, std::ios::binary);
	file.write("\x93NUMPY\x01\x00", 8);
	const uint16_t length = static_cast<uint16_t>(header.size());
	file.put(static_cast<char>(length & 0xff));
	file.put(static_cast<char>(length >> 8));
	file.write(header.data(), header.size());
	file.write(static_cast<const char *>(data), bytes);
}
}

void dumpDecision(const std::string & dir, const std::string & name, const InputTensors & inputs, const OutputTensors & outputs, const ChosenAction & action)
{
	const auto path = std::filesystem::path(dir) / name;
	std::filesystem::create_directories(path);

	for(int i = 0; i < static_cast<int>(In::COUNT); ++i)
	{
		const auto & s = INPUTS[i];
		const auto & data = inputs.data(static_cast<In>(i));
		if(s.dtype == DType::BOOL)
		{
			std::vector<uint8_t> bytes(data.begin(), data.end());
			writeNpy(path / (std::string(s.name) + ".npy"), "|b1", s, bytes.data(), bytes.size());
		}
		else
		{
			writeNpy(path / (std::string(s.name) + ".npy"), "<i8", s, data.data(), data.size() * sizeof(int64_t));
		}
	}
	for(int o = 0; o < static_cast<int>(Out::COUNT); ++o)
	{
		const auto & data = outputs.data(static_cast<Out>(o));
		writeNpy(path / (std::string(OUTPUTS[o].name) + ".npy"), "<f4", OUTPUTS[o], data.data(), data.size() * sizeof(float));
	}
	std::ofstream(path / "action.json") << action.toJson() << "\n";
}
