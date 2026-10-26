#pragma once

#include <cstdint>
#include <mutex>
#include <string>

/// A connection to the central inference server (battle_model/inference.py): it runs the trainer's live model for
/// every game, batched. Generic over model families: the caller encodes its contract's inputs and decodes the outputs.
/// One request at a time; thread safe. Throws std::runtime_error when the server refuses or the connection fails.
class InferenceClient
{
	int fd = -1;
	std::mutex mutex;

public:
	/// Connects and says which family and contract it speaks; the server refuses a different contract
	InferenceClient(const std::string & socketPath, const std::string & family, const std::string & contractHash);
	~InferenceClient();
	InferenceClient(const InferenceClient &) = delete;
	InferenceClient & operator=(const InferenceClient &) = delete;

	/// Runs the named model ("current", or e.g. "v350") on `decisions` decisions' inputs (wire encoded, in contract
	/// order); returns the weights' version and puts the outputs (float32, in contract order) into `outputs`.
	/// With 0 decisions it only asks for the version.
	uint32_t request(const std::string & name, uint32_t decisions, const std::string & inputs, std::string & outputs);
};
