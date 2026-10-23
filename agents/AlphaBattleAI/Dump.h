#pragma once

#include "Tensors.h"

#include <string>
#include <vector>

struct ChosenAction;

/// Writes one decision (all inputs and outputs as .npy, the chosen action as action.json) into the
/// compressed archive <dir>/<name>.npz, read by battle_model/data.py and check_dumps.py. For debugging:
/// training receives its records from BattleRecord, without files.
void dumpDecision(const std::string & dir, const std::string & name, const AlphaSchema::InputTensors & inputs, const AlphaSchema::OutputTensors & outputs, const ChosenAction & action);

/// One side's record of one battle, kept in memory: its events (JSON lines) and every decision (all input
/// and output tensors, the chosen action). At the end of the battle it is sent in one piece to the trainer
/// over a Unix socket (battle_model/play.py RecordServer), so training data never touches the disk.
///
/// Wire format, read by battle_model/data.py parse_record: "ALPHAREC1\n", then entries of u32 name length,
/// name, u64 size, bytes (little endian): "events.jsonl", "actions.jsonl", and "<tensor>.npy" per input and
/// output tensor with the decisions stacked on the first axis.
class BattleRecord
{
	std::string events;
	std::string actions;
	std::vector<std::string> inputs; // per input tensor: the raw values of every decision, concatenated
	std::vector<std::string> outputs;
	int decisions = 0;

public:
	BattleRecord();
	void addEvent(const std::string & json);
	void addDecision(const AlphaSchema::InputTensors & in, const AlphaSchema::OutputTensors & out, const ChosenAction & action);
	std::string serialize() const;
	/// Connects to socketPath and sends the key (u32 length + bytes) and then serialize(); throws on failure
	void send(const std::string & socketPath, const std::string & key) const;
};
