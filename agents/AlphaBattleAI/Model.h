#pragma once

#include "Tensors.h"

#include <memory>
#include <string>
#include <vector>

namespace Ort
{
struct Session;
}

/// An ONNX battle model that implements the contract in Schema.generated.h.
/// load() rejects files whose schema version, contract hash, inputs or outputs differ.
class BattleModel
{
	std::unique_ptr<Ort::Session> session;
	std::vector<std::string> inputNames; // inputs the model declares, all from the contract
	std::vector<AlphaSchema::In> inputIds;

	explicit BattleModel(const std::string & path);

public:
	~BattleModel();

	/// Loaded once per path and shared by all battles; throws std::runtime_error on mismatch
	static std::shared_ptr<BattleModel> load(const std::string & path);

	AlphaSchema::OutputTensors run(const AlphaSchema::InputTensors & inputs) const;
};
