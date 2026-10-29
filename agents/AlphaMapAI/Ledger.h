#pragma once

#include <string>

VCMI_LIB_NAMESPACE_BEGIN
class CGameState;
VCMI_LIB_NAMESPACE_END

/// Every player's assets as raw engine values (JSON), for the empire values the trainer computes
/// (campaign_model/empire.py): resources, towns (buildings, garrisons, creatures for hire), heroes (experience, mana,
/// spell power, their dearest combat spell, armies, artifacts), other armies, map dwellings, the income of mines and
/// other non-town objects, and the tiles each player ever revealed. It reads the whole game state: it is the reward's
/// view, never the model's. day: the day it was taken at the start of; final: at the end of the game.
std::string ledgerJson(const CGameState & gs, int day, bool final);
