#pragma once

// MULTI's army arrangements (schema 15; the user, 2026-11-05: predetermined chains of the engine's own moves, the model
// choosing among them by what they make): the armies each makes, shared by the encoder (Encoder.cpp: the inputs, the
// legal entries) and the actions (Actions.cpp: the requests). Rules, not judgements of ours: Nullkiller2's best army,
// the slowest creatures by their speed, the creatures of the army's lesser factions.

#include "lib/constants/EntityIdentifiers.h"

#include <optional>
#include <utility>
#include <vector>

class CArmedInstance;
class CCreatureSet;
class IGameSettings;

namespace armies
{
using Army = std::vector<std::pair<CreatureID, int>>; // (creature, count) a stack, one a creature, at most ARMY_SLOTS

// TAKE_BEST (fast: TAKE_BEST_FAST): the receiver's army after and the giver's, or none when nothing would change or the
// giver could not hold what it gets back
struct Take
{
	Army receiver;
	Army giver;
};
std::optional<Take> takeBest(const CArmedInstance & receiver, const CArmedInstance & giver, const IGameSettings & settings, bool fast);

// DROP_SLOW: the army's stacks of its slowest speed (CCreature::getBaseSpeed, as vocab.json's "speed"), when it has more
// than one speed; none otherwise
std::vector<SlotID> slowestStacks(const CCreatureSet & army);
// DROP_FOREIGN: the stacks not of the army's main faction (the most power: count x AI value, VCMI's getPower), when it
// has more than one faction; none otherwise
std::vector<SlotID> foreignStacks(const CCreatureSet & army);
// Whether `into` can take those stacks of `from`: each onto its own creature there or into a free slot
bool takes(const CCreatureSet & into, const CCreatureSet & from, const std::vector<SlotID> & stacks);
}
