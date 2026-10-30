#include "StdInc.h"
#include "Dump.h"

#include "Policy.h"

#include "lib/filesystem/CZipSaver.h"
#include "lib/filesystem/MinizipExtensions.h"

#include <climits>
#include <filesystem>
#include <stdexcept>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace AlphaSchema;

namespace
{
/// NumPy .npy format version 1.0 with a leading dimension (the batch: 1, or the decisions of a battle)
std::string npy(const char * descr, const TensorSpec & spec, const void * data, size_t bytes, int leading = 1)
{
	std::string shape = "(" + std::to_string(leading) + ",";
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

template<typename T>
void appendRaw(std::string & out, const T & value)
{
	out.append(reinterpret_cast<const char *>(&value), sizeof(T)); // little endian on every platform we build for
}

void appendEntry(std::string & out, const std::string & name, const std::string & bytes)
{
	appendRaw(out, static_cast<uint32_t>(name.size()));
	out += name;
	appendRaw(out, static_cast<uint64_t>(bytes.size()));
	out += bytes;
}
}

BattleRecord::BattleRecord()
	: inputs(static_cast<int>(In::COUNT))
	, outputs(static_cast<int>(Out::COUNT))
{
}

void BattleRecord::addEvent(const std::string & json)
{
	events += json + "\n";
}

void BattleRecord::addDecision(const InputTensors & in, const OutputTensors & out, const ChosenAction & action)
{
	for(int i = 0; i < static_cast<int>(In::COUNT); ++i)
	{
		const auto & data = in.data(static_cast<In>(i));
		auto & column = inputs[i];
		const size_t at = column.size();
		if(INPUTS[i].dtype == DType::BOOL)
		{
			column.resize(at + data.size());
			for(size_t k = 0; k < data.size(); ++k)
				column[at + k] = static_cast<char>(data[k] != 0);
		}
		else
		{
			// As int32, as the inference wire sends them (Model.cpp): half the bytes of int64, and the trainer keeps them so
			// (data.py compact read every int64 array to narrow it, ~1 ms of its CPU a decision)
			column.resize(at + data.size() * sizeof(int32_t));
			auto * out = reinterpret_cast<int32_t *>(column.data() + at);
			for(size_t k = 0; k < data.size(); ++k)
			{
				if(data[k] < INT32_MIN || data[k] > INT32_MAX)
					throw std::runtime_error(std::string("input '") + INPUTS[i].name + "' is out of the int32 range of the record");
				out[k] = static_cast<int32_t>(data[k]);
			}
		}
	}
	for(int o = 0; o < static_cast<int>(Out::COUNT); ++o)
	{
		const auto & data = out.data(static_cast<Out>(o));
		outputs[o].append(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
	}
	actions += action.toJson() + "\n";
	++decisions;
}

std::string BattleRecord::serialize() const
{
	std::string out = "ALPHAREC1\n";
	appendEntry(out, "events.jsonl", events);
	appendEntry(out, "actions.jsonl", actions);
	for(int i = 0; i < static_cast<int>(In::COUNT); ++i)
		appendEntry(out, std::string(INPUTS[i].name) + ".npy", npy(INPUTS[i].dtype == DType::BOOL ? "|b1" : "<i4", INPUTS[i], inputs[i].data(), inputs[i].size(), decisions));
	for(int o = 0; o < static_cast<int>(Out::COUNT); ++o)
		appendEntry(out, std::string(OUTPUTS[o].name) + ".npy", npy("<f4", OUTPUTS[o], outputs[o].data(), outputs[o].size(), decisions));
	return out;
}

void BattleRecord::send(const std::string & socketPath, const std::string & key) const
{
	sockaddr_un address{};
	address.sun_family = AF_UNIX;
	if(socketPath.size() >= sizeof(address.sun_path))
		throw std::runtime_error("record socket path too long: " + socketPath);
	std::copy(socketPath.begin(), socketPath.end(), address.sun_path);

	const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if(fd < 0)
		throw std::runtime_error("cannot create a socket for the record");
	if(connect(fd, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) != 0)
	{
		close(fd);
		throw std::runtime_error("cannot connect to the record socket " + socketPath);
	}
	std::string message;
	appendRaw(message, static_cast<uint32_t>(key.size()));
	message += key;
	message += serialize();
	for(size_t sent = 0; sent < message.size();)
	{
		const auto n = ::send(fd, message.data() + sent, message.size() - sent, 0);
		if(n <= 0)
		{
			close(fd);
			throw std::runtime_error("sending the record failed");
		}
		sent += static_cast<size_t>(n);
	}
	close(fd);
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
