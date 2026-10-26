#pragma once

#include "Schema.generated.h"

#include <array>
#include <cstdint>
#include <deque>
#include <map>

VCMI_LIB_NAMESPACE_BEGIN
class BattleAction;
class CBattleInfoCallback;
struct BattleSpellCast;
struct BattleStackAttacked;
VCMI_LIB_NAMESPACE_END

/// What a player has seen of a battle so far, beyond its present state (the encoder's h_seen_*, u_start_count ..
/// u_last_round and the e_* event log): the spells each hero was seen casting, each unit's damage dealt and taken and
/// kills, whom it last went for, and the last actions of both sides. Only what the battle showed: an enemy hero's
/// spells stay unknown until it casts them. Kept by the AI from the engine's events, one per battle.
class BattleHistory
{
public:
	static constexpr uint32_t HERO = 0xFFFFFFFF; // an event's actor when a hero acted

	struct Unit
	{
		int64_t startCount = -1; // creatures when first seen (the battle's start, or when summoned)
		int64_t dealt = 0, taken = 0; // hit points
		int64_t kills = 0; // creatures killed
		int64_t lastTarget = -1; // unit id it last attacked or cast at
		int lastRound = -1; // round it last acted
	};

	struct Event
	{
		int round = 0;
		int side = 0; // the actor's battle side
		uint32_t actor = HERO; // unit id, or HERO
		int type = -1; // AlphaSchema::ActionType
		int64_t target = -1; // unit id, -1 if none
		int hex = -1; // target hex, -1 if none
		int spell = -1;
		int64_t dealt = 0, taken = 0; // hit points the actor's side dealt and took in it (retaliation, fire shield)
		int64_t kills = 0, lost = 0; // creatures killed of the other side, and of the actor's
	};

	std::map<uint32_t, Unit> units;
	std::array<std::array<int, AlphaSchema::N_SPELLS>, 2> seenSpells{};
	std::array<int, 2> casts{};
	std::array<int, 2> lastCastRound{-1, -1};
	std::deque<Event> events; // newest first, at most AlphaSchema::N_EVENTS

	void reset();
	/// The units' start counts: every stack it does not know yet
	void seeUnits(const CBattleInfoCallback & battle);
	void actionStarted(const CBattleInfoCallback & battle, const BattleAction & action);
	void stacksAttacked(const CBattleInfoCallback & battle, const std::vector<BattleStackAttacked> & attacked);
	void spellCast(const CBattleInfoCallback & battle, const BattleSpellCast & cast);
	void actionFinished();

private:
	bool inAction = false;
	Event current;
};
