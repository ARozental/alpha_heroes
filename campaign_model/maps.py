"""The game's own maps the campaign games draw from, next to generated ones (the user, 2026-10-27: "do not rely only on
generated ones"). Chosen with build/bin/alpha_map_info (VCMI's loader, agents/AlphaMapAI/tools/MapInfo.cpp) from the
159 maps in runtime/vcmi-user/Maps: 36 x 36 (small, as the generated ones), 2 to 4 players, no teams (the reward is
our share against every other player: an ally would be a rival it cannot attack) and the standard victory and loss
conditions (the model does not see a map's special ones, e.g. "gather 100k gold to win"). Some start a player with
heroes and no town (Freedom, The Newcomers): it has 7 days to take one, as in the game.

Left out after joint1 (2026-10-27): Knee Deep in the Dead, where red starts without a town and the untrained model lost
it on day 7 in all 11 games (the map decided them; a 7-day game also finishes fast, so it was 9% of the games).
"""

GAME_MAPS = {  # map name (as headless.sh takes it) -> players
	"Maps/Dungeon Keeper": 2,
	"Maps/Key to Victory": 2,
	"Maps/Chasing a Dream": 3,
	"Maps/Dead and Buried": 3,
	"Maps/Deluge": 3,
	"Maps/Freedom": 3,
	"Maps/Arrogance": 4,
	"Maps/Irrational Hostility": 4,
	"Maps/Sands of Blood": 4,
	"Maps/The Newcomers": 4,
}
