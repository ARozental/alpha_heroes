#include "StdInc.h"
#include "InferenceClient.h"

#include <cstring>
#include <stdexcept>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

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
}

InferenceClient::InferenceClient(const std::string & socketPath, const std::string & family, const std::string & contractHash)
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
	sendMessage(fd, "{\"family\": \"" + family + "\", \"contract_hash\": \"" + contractHash + "\"}");
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
	sendMessage(fd, message);
	auto reply = receiveMessage(fd);
	if(reply.size() < 4)
		throw std::runtime_error("inference server: a short reply");
	uint32_t version = 0;
	std::memcpy(&version, reply.data(), 4);
	if(version == ERROR_VERSION)
		throw std::runtime_error("inference server: " + reply.substr(4));
	outputs.assign(reply, 4, std::string::npos);
	return version;
}
