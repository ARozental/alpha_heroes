#pragma once

#include "Tensors.h"

#include <memory>
#include <string>
#include <vector>

namespace Ort
{
struct Session;
}

class InferenceClient;

/// A battle model of the contract in Schema.generated.h: an ONNX file (BattleModel) or the training's inference server
/// (RemoteBattleModel)
class IBattleModel
{
public:
	virtual ~IBattleModel() = default;
	virtual AlphaSchema::OutputTensors run(const AlphaSchema::InputTensors & inputs) const = 0;
};

/// An ONNX battle model.
/// load() rejects files whose schema version, contract hash, inputs or outputs differ.
class BattleModel : public IBattleModel
{
	std::unique_ptr<Ort::Session> session;
	std::vector<std::string> inputNames; // inputs the model declares, all from the contract
	std::vector<AlphaSchema::In> inputIds;

	explicit BattleModel(const std::string & path);

public:
	~BattleModel();

	/// Loaded once per file and shared by all battles; a link (training's current.onnx) is resolved first, so the model
	/// it points at now is the one returned. Throws std::runtime_error on mismatch
	static std::shared_ptr<BattleModel> load(const std::string & path);

	AlphaSchema::OutputTensors run(const AlphaSchema::InputTensors & inputs) const override;
};

/// The named model of the inference server (battle_model/inference.py, family "battle"): the trainer's live weights
/// ("current") or a version it keeps (e.g. "v350"), run for every game in batches. No ONNX session in the game.
class RemoteBattleModel : public IBattleModel
{
	std::unique_ptr<InferenceClient> client;
	std::string name;

public:
	RemoteBattleModel(const std::string & socketPath, const std::string & name);
	~RemoteBattleModel();

	AlphaSchema::OutputTensors run(const AlphaSchema::InputTensors & inputs) const override;
	/// The version of the weights the server runs under this name now
	uint32_t version() const;
};
