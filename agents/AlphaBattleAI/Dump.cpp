#include "StdInc.h"
#include "Dump.h"

#include "Policy.h"

#include "lib/filesystem/CZipSaver.h"
#include "lib/filesystem/MinizipExtensions.h"

#include <filesystem>

using namespace AlphaSchema;

namespace
{
/// NumPy .npy format version 1.0 with a leading batch dimension of 1
std::string npy(const char * descr, const TensorSpec & spec, const void * data, size_t bytes)
{
	std::string shape = "(1,";
	for(int d = 0; d < spec.rank; ++d)
		shape += " " + std::to_string(spec.dims[d]) + ",";
	shape += ")";
	std::string header = std::string("{'descr': '") + descr + "', 'fortran_order': False, 'shape': " + shape + ", }";
	const size_t total = 10 + header.size() + 1;
	header.append((64 - total % 64) % 64, ' ');
	header += '\n';

	std::string file("\x93NUMPY\x01\x00", 8);
	const uint16_t length = static_cast<uint16_t>(header.size());
	file += static_cast<char>(length & 0xff);
	file += static_cast<char>(length >> 8);
	file += header;
	file.append(static_cast<const char *>(data), bytes);
	return file;
}

void add(CZipSaver & zip, const std::string & name, const std::string & bytes)
{
	auto stream = zip.addFile(name);
	stream->write(reinterpret_cast<const ui8 *>(bytes.data()), static_cast<si64>(bytes.size()));
}
}

void dumpDecision(const std::string & dir, const std::string & name, const InputTensors & inputs, const OutputTensors & outputs, const ChosenAction & action)
{
	// One compressed .npz per decision (numpy.load reads it): a .npy per tensor plus action.json.
	// The tensors are mostly zeros and -inf, and one file instead of ~75 keeps the file system quiet.
	std::filesystem::create_directories(dir);
	CZipSaver zip(std::make_shared<CDefaultIOApi>(), boost::filesystem::path(dir) / (name + ".npz"));

	for(int i = 0; i < static_cast<int>(In::COUNT); ++i)
	{
		const auto & s = INPUTS[i];
		const auto & data = inputs.data(static_cast<In>(i));
		if(s.dtype == DType::BOOL)
		{
			std::vector<uint8_t> bytes(data.begin(), data.end());
			add(zip, std::string(s.name) + ".npy", npy("|b1", s, bytes.data(), bytes.size()));
		}
		else
		{
			add(zip, std::string(s.name) + ".npy", npy("<i8", s, data.data(), data.size() * sizeof(int64_t)));
		}
	}
	for(int o = 0; o < static_cast<int>(Out::COUNT); ++o)
	{
		const auto & data = outputs.data(static_cast<Out>(o));
		add(zip, std::string(OUTPUTS[o].name) + ".npy", npy("<f4", OUTPUTS[o], data.data(), data.size() * sizeof(float)));
	}
	add(zip, "action.json", action.toJson() + "\n");
}
