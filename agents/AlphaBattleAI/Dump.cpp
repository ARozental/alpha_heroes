#include "StdInc.h"
#include "Dump.h"

#include "Policy.h"

#include "lib/filesystem/CZipSaver.h"
#include "lib/filesystem/MinizipExtensions.h"

#include <climits>
#include <cstring>
#include <filesystem>
#include <stdexcept>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

using namespace AlphaSchema;

namespace
{
/// The start of a NumPy .npy file (format version 1.0) with a leading dimension (the batch: 1, or the decisions of a
/// battle): its data follows
std::string npyHeader(const char * descr, const TensorSpec & spec, int leading)
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
	return file;
}

std::string npy(const char * descr, const TensorSpec & spec, const void * data, size_t bytes, int leading = 1)
{
	return npyHeader(descr, spec, leading).append(static_cast<const char *>(data), bytes);
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

BattleRecord::BattleRecord() = default;

void BattleRecord::addEvent(const std::string & json)
{
	events += json + "\n";
}

void BattleRecord::addDecision(const InputTensors & in, const OutputTensors & out, const ChosenAction & action)
{
	constexpr int nInputs = static_cast<int>(In::COUNT);
	constexpr int nOutputs = static_cast<int>(Out::COUNT);
	auto inputBytes = [&in](int i) { return in.data(static_cast<In>(i)).size() * (INPUTS[i].dtype == DType::BOOL ? 1 : sizeof(int32_t)); };
	auto outputBytes = [&out](int o) { return out.data(static_cast<Out>(o)).size() * sizeof(float); };
	if(tensorBytes.empty())
	{
		for(int i = 0; i < nInputs; ++i)
			tensorBytes.push_back(inputBytes(i));
		for(int o = 0; o < nOutputs; ++o)
			tensorBytes.push_back(outputBytes(o));
	}
	size_t total = 0;
	for(int k = 0; k < nInputs + nOutputs; ++k)
	{
		if((k < nInputs ? inputBytes(k) : outputBytes(k - nInputs)) != tensorBytes[k])
			throw std::runtime_error(std::string("tensor '") + (k < nInputs ? INPUTS[k].name : OUTPUTS[k - nInputs].name) + "' changed its size within a battle record");
		total += tensorBytes[k];
	}
	auto & bytes = decisionBytes.emplace_back(total, '\0');
	char * at = bytes.data();
	for(int i = 0; i < nInputs; ++i)
	{
		const auto & data = in.data(static_cast<In>(i));
		if(INPUTS[i].dtype == DType::BOOL)
		{
			for(size_t k = 0; k < data.size(); ++k)
				at[k] = static_cast<char>(data[k] != 0);
		}
		else
		{
			// As int32, as the inference wire sends them (Model.cpp): half the bytes of int64, and the trainer keeps them so
			// (data.py compact read every int64 array to narrow it, ~1 ms of its CPU a decision)
			for(size_t k = 0; k < data.size(); ++k)
			{
				if(data[k] < INT32_MIN || data[k] > INT32_MAX)
					throw std::runtime_error(std::string("input '") + INPUTS[i].name + "' is out of the int32 range of the record");
				const auto value = static_cast<int32_t>(data[k]);
				std::memcpy(at + k * sizeof(int32_t), &value, sizeof(int32_t)); // (a tensor after a bool one may be unaligned)
			}
		}
		at += tensorBytes[i];
	}
	for(int o = 0; o < nOutputs; ++o)
	{
		std::memcpy(at, out.data(static_cast<Out>(o)).data(), tensorBytes[nInputs + o]);
		at += tensorBytes[nInputs + o];
	}
	actions += action.toJson() + "\n";
	++decisions;
}

std::string BattleRecord::serialize() const
{
	std::string out;
	serializeInto(out);
	return out;
}

void BattleRecord::serializeInto(std::string & out) const
{
	// Each tensor's entry: its .npy header, then its bytes of every decision in turn; the message is sized first, so
	// that every byte is copied once (it was copied ~5 times: ~6% of a battle client's CPU)
	std::vector<std::string> names;
	std::vector<std::string> headers;
	for(int i = 0; i < static_cast<int>(In::COUNT); ++i)
	{
		names.push_back(std::string(INPUTS[i].name) + ".npy");
		headers.push_back(npyHeader(INPUTS[i].dtype == DType::BOOL ? "|b1" : "<i4", INPUTS[i], decisions));
	}
	for(int o = 0; o < static_cast<int>(Out::COUNT); ++o)
	{
		names.push_back(std::string(OUTPUTS[o].name) + ".npy");
		headers.push_back(npyHeader("<f4", OUTPUTS[o], decisions));
	}
	auto bytesOf = [this](size_t k) { return tensorBytes.empty() ? size_t(0) : tensorBytes[k]; }; // (no decisions: none)
	auto entrySize = [](const std::string & name, size_t bytes) { return sizeof(uint32_t) + name.size() + sizeof(uint64_t) + bytes; };
	const std::string magic = "ALPHAREC1\n";
	size_t total = magic.size() + entrySize("events.jsonl", events.size()) + entrySize("actions.jsonl", actions.size());
	for(size_t k = 0; k < names.size(); ++k)
		total += entrySize(names[k], headers[k].size() + bytesOf(k) * decisions);
	out.reserve(out.size() + total);

	out += magic;
	appendEntry(out, "events.jsonl", events);
	appendEntry(out, "actions.jsonl", actions);
	size_t offset = 0;
	for(size_t k = 0; k < names.size(); ++k)
	{
		appendRaw(out, static_cast<uint32_t>(names[k].size()));
		out += names[k];
		appendRaw(out, static_cast<uint64_t>(headers[k].size() + bytesOf(k) * decisions));
		out += headers[k];
		for(const auto & bytes : decisionBytes)
			out.append(bytes.data() + offset, bytesOf(k));
		offset += bytesOf(k);
	}
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
	serializeInto(message);
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
