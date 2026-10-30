#include "StdInc.h"
#include "InferenceClient.h"

#include <cstring>
#include <stdexcept>

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <zlib.h>

namespace
{
constexpr uint32_t ERROR_VERSION = 0xFFFFFFFF;

void sendAll(int fd, const char * data, size_t size)
{
	for(size_t sent = 0; sent < size;)
	{
		const auto n = ::send(fd, data + sent, size - sent, 0);
		if(n <= 0)
			throw std::runtime_error("inference server: sending failed");
		sent += static_cast<size_t>(n);
	}
}

void receiveAll(int fd, char * data, size_t size)
{
	for(size_t received = 0; received < size;)
	{
		const auto n = ::recv(fd, data + received, size - received, 0);
		if(n <= 0)
			throw std::runtime_error("inference server: the connection closed");
		received += static_cast<size_t>(n);
	}
}

void sendMessage(int fd, const std::string & payload)
{
	const auto length = static_cast<uint32_t>(payload.size());
	sendAll(fd, reinterpret_cast<const char *>(&length), 4);
	sendAll(fd, payload.data(), payload.size());
}

std::string receiveMessage(int fd)
{
	uint32_t length = 0;
	receiveAll(fd, reinterpret_cast<char *>(&length), 4);
	std::string payload(length, '\0');
	receiveAll(fd, payload.data(), length);
	return payload;
}

/// A message as a compressed connection sends it (inference.py pack): u32 size, then zlib at level 1
std::string pack(const std::string & message)
{
	uLongf size = compressBound(message.size());
	std::string out(4 + size, '\0');
	const auto length = static_cast<uint32_t>(message.size());
	std::memcpy(out.data(), &length, 4);
	if(compress2(reinterpret_cast<Bytef *>(out.data() + 4), &size, reinterpret_cast<const Bytef *>(message.data()), message.size(), 1) != Z_OK)
		throw std::runtime_error("inference server: compressing a request failed");
	out.resize(4 + size);
	return out;
}

std::string unpack(const std::string & packed)
{
	uint32_t length = 0;
	if(packed.size() < 4)
		throw std::runtime_error("inference server: a short compressed reply");
	std::memcpy(&length, packed.data(), 4);
	std::string out(length, '\0');
	uLongf size = length;
	if(uncompress(reinterpret_cast<Bytef *>(out.data()), &size, reinterpret_cast<const Bytef *>(packed.data() + 4), packed.size() - 4) != Z_OK || size != length)
		throw std::runtime_error("inference server: a damaged compressed reply");
	return out;
}

/// A TCP connection to "host:port" (a server on another machine: battle_model/worker.py), Nagle off (one small request
/// after another)
int connectTcp(const std::string & address)
{
	const auto colon = address.rfind(':');
	if(colon == std::string::npos)
		throw std::runtime_error("inference server address needs tcp:<host>:<port>: " + address);
	const auto host = address.substr(0, colon), port = address.substr(colon + 1);
	addrinfo hints{};
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	addrinfo * found = nullptr;
	if(getaddrinfo(host.c_str(), port.c_str(), &hints, &found) != 0 || !found)
		throw std::runtime_error("cannot resolve the inference server " + address);
	int fd = -1;
	for(auto * a = found; a && fd < 0; a = a->ai_next)
	{
		fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
		if(fd >= 0 && connect(fd, a->ai_addr, a->ai_addrlen) != 0)
		{
			close(fd);
			fd = -1;
		}
	}
	freeaddrinfo(found);
	if(fd < 0)
		throw std::runtime_error("cannot connect to the inference server at " + address);
	const int one = 1;
	setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
	return fd;
}
}

InferenceClient::InferenceClient(const std::string & socketPath, const std::string & family, const std::string & contractHash)
{
	if(socketPath.rfind("tcp:", 0) == 0) // another machine's server: compressed (a battle decision ~400 KB -> ~4 KB)
	{
		fd = connectTcp(socketPath.substr(4));
		compressed = true;
	}
	else
	{
		sockaddr_un address{};
		address.sun_family = AF_UNIX;
		if(socketPath.size() >= sizeof(address.sun_path))
			throw std::runtime_error("inference socket path too long: " + socketPath);
		std::copy(socketPath.begin(), socketPath.end(), address.sun_path);
		fd = socket(AF_UNIX, SOCK_STREAM, 0);
		if(fd < 0)
			throw std::runtime_error("cannot create a socket for the inference server");
		if(connect(fd, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) != 0)
		{
			close(fd);
			fd = -1;
			throw std::runtime_error("cannot connect to the inference server at " + socketPath);
		}
	}
	sendMessage(fd, "{\"family\": \"" + family + "\", \"contract_hash\": \"" + contractHash + "\"" + (compressed ? ", \"compress\": \"zlib\"" : "") + "}");
	const auto reply = receiveMessage(fd);
	if(reply.find("\"ok\": true") == std::string::npos)
	{
		close(fd);
		fd = -1;
		throw std::runtime_error("the inference server refused the connection: " + reply);
	}
}

InferenceClient::~InferenceClient()
{
	if(fd >= 0)
		close(fd);
}

uint32_t InferenceClient::request(const std::string & name, uint32_t decisions, const std::string & inputs, std::string & outputs)
{
	if(name.size() > 255)
		throw std::runtime_error("inference model name too long: " + name);
	std::string message;
	message.reserve(5 + name.size() + inputs.size());
	message.push_back(static_cast<char>(name.size()));
	message += name;
	message.append(reinterpret_cast<const char *>(&decisions), 4);
	message += inputs;

	std::scoped_lock lock(mutex);
	sendMessage(fd, compressed ? pack(message) : message);
	auto reply = compressed ? unpack(receiveMessage(fd)) : receiveMessage(fd);
	if(reply.size() < 4)
		throw std::runtime_error("inference server: a short reply");
	uint32_t version = 0;
	std::memcpy(&version, reply.data(), 4);
	if(version == ERROR_VERSION)
		throw std::runtime_error("inference server: " + reply.substr(4));
	outputs.assign(reply, 4, std::string::npos);
	return version;
}
