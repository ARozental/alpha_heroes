#include "StdInc.h"
#include "Model.h"

#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>

using namespace AlphaSchema;

namespace
{
Ort::Env & ortEnv()
{
	static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "AlphaBattleAI");
	return env;
}

ONNXTensorElementDataType ortType(DType dtype)
{
	switch(dtype)
	{
		case DType::INT64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
		case DType::BOOL: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
		case DType::FLOAT32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
	}
	throw std::logic_error("unknown dtype");
}

std::vector<int64_t> batchShape(const TensorSpec & spec)
{
	std::vector<int64_t> shape{1};
	for(int d = 0; d < spec.rank; ++d)
		shape.push_back(spec.dims[d]);
	return shape;
}

void checkShape(const TensorSpec & spec, const std::vector<int64_t> & shape)
{
	// Leading dimension is the batch (dynamic, -1); the rest must match the contract exactly
	if(shape.size() != static_cast<size_t>(spec.rank + 1))
		throw std::runtime_error(std::string("model tensor '") + spec.name + "' has rank " + std::to_string(shape.size() - 1) + ", contract says " + std::to_string(spec.rank));
	for(int d = 0; d < spec.rank; ++d)
		if(shape[d + 1] != spec.dims[d])
			throw std::runtime_error(std::string("model tensor '") + spec.name + "' dimension " + std::to_string(d) + " is " + std::to_string(shape[d + 1]) + ", contract says " + std::to_string(spec.dims[d]));
}

std::string metadata(Ort::Session & session, const char * key)
{
	Ort::AllocatorWithDefaultOptions allocator;
	auto value = session.GetModelMetadata().LookupCustomMetadataMapAllocated(key, allocator);
	return value ? std::string(value.get()) : std::string();
}
}

BattleModel::BattleModel(const std::string & path)
{
	Ort::SessionOptions options;
	options.SetIntraOpNumThreads(1);
	options.SetInterOpNumThreads(1);
	session = std::make_unique<Ort::Session>(ortEnv(), path.c_str(), options);

	const auto version = metadata(*session, "schema_version");
	if(version != std::to_string(SCHEMA_VERSION))
		throw std::runtime_error("model schema_version is '" + version + "', expected " + std::to_string(SCHEMA_VERSION));
	const auto hash = metadata(*session, "contract_hash");
	if(hash != CONTRACT_HASH)
		throw std::runtime_error("model contract_hash is '" + hash + "', expected " + CONTRACT_HASH);

	Ort::AllocatorWithDefaultOptions allocator;
	std::map<std::string, In> known;
	for(int i = 0; i < static_cast<int>(In::COUNT); ++i)
		known[INPUTS[i].name] = static_cast<In>(i);

	for(size_t i = 0; i < session->GetInputCount(); ++i)
	{
		std::string name = session->GetInputNameAllocated(i, allocator).get();
		auto it = known.find(name);
		if(it == known.end())
			throw std::runtime_error("model input '" + name + "' is not in the contract");
		const auto & s = spec(it->second);
		// The shape info is a view into the type info: keep the type info alive
		auto typeInfo = session->GetInputTypeInfo(i);
		auto info = typeInfo.GetTensorTypeAndShapeInfo();
		if(info.GetElementType() != ortType(s.dtype))
			throw std::runtime_error("model input '" + name + "' has the wrong element type");
		checkShape(s, info.GetShape());
		inputNames.push_back(name);
		inputIds.push_back(it->second);
	}

	for(int o = 0; o < static_cast<int>(Out::COUNT); ++o)
	{
		const auto & s = OUTPUTS[o];
		bool found = false;
		for(size_t i = 0; i < session->GetOutputCount() && !found; ++i)
		{
			if(std::string(session->GetOutputNameAllocated(i, allocator).get()) != s.name)
				continue;
			auto typeInfo = session->GetOutputTypeInfo(i);
			auto info = typeInfo.GetTensorTypeAndShapeInfo();
			if(info.GetElementType() != ortType(s.dtype))
				throw std::runtime_error(std::string("model output '") + s.name + "' has the wrong element type");
			checkShape(s, info.GetShape());
			found = true;
		}
		if(!found)
			throw std::runtime_error(std::string("model has no output '") + s.name + "'");
	}
}

BattleModel::~BattleModel() = default;

std::shared_ptr<BattleModel> BattleModel::load(const std::string & path)
{
	static std::mutex mutex;
	static std::map<std::string, std::shared_ptr<BattleModel>> cache;
	static std::map<std::string, std::string> failures; // a refused file is not retried every battle
	std::scoped_lock lock(mutex);
	if(auto failed = failures.find(path); failed != failures.end())
		throw std::runtime_error(failed->second);
	auto & model = cache[path];
	if(!model)
	{
		try
		{
			model.reset(new BattleModel(path));
		}
		catch(const std::exception & e)
		{
			cache.erase(path);
			failures[path] = e.what();
			throw;
		}
	}
	return model;
}

OutputTensors BattleModel::run(const InputTensors & inputs) const
{
	auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	std::vector<std::vector<uint8_t>> boolBuffers; // onnxruntime bool tensors are one byte per element
	boolBuffers.reserve(inputIds.size());
	std::vector<std::vector<int64_t>> shapes;
	shapes.reserve(inputIds.size());
	std::vector<Ort::Value> values;
	std::vector<const char *> names;

	for(size_t i = 0; i < inputIds.size(); ++i)
	{
		const auto & s = spec(inputIds[i]);
		const auto & data = inputs.data(inputIds[i]);
		shapes.push_back(batchShape(s));
		const auto & shape = shapes.back();
		if(s.dtype == DType::BOOL)
		{
			boolBuffers.emplace_back(data.begin(), data.end());
			auto & buffer = boolBuffers.back();
			values.push_back(Ort::Value::CreateTensor(memory, buffer.data(), buffer.size(), shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL));
		}
		else
		{
			auto * ptr = const_cast<int64_t *>(data.data());
			values.push_back(Ort::Value::CreateTensor<int64_t>(memory, ptr, data.size(), shape.data(), shape.size()));
		}
		names.push_back(inputNames[i].c_str());
	}

	std::vector<const char *> outputNames;
	for(const auto & s : OUTPUTS)
		outputNames.push_back(s.name);

	auto results = session->Run(Ort::RunOptions{nullptr}, names.data(), values.data(), values.size(), outputNames.data(), outputNames.size());

	OutputTensors out;
	for(int o = 0; o < static_cast<int>(Out::COUNT); ++o)
	{
		const auto n = numel(OUTPUTS[o]);
		const float * ptr = results[o].GetTensorData<float>();
		if(results[o].GetTensorTypeAndShapeInfo().GetElementCount() != static_cast<size_t>(n))
			throw std::runtime_error(std::string("model output '") + OUTPUTS[o].name + "' has the wrong size");
		auto & dst = out.data(static_cast<Out>(o));
		dst.assign(ptr, ptr + n);
		for(float v : dst)
			if(std::isnan(v))
				throw std::runtime_error(std::string("model output '") + OUTPUTS[o].name + "' contains NaN");
	}
	return out;
}
