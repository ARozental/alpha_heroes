# alpha_heroes

Training a strong Heroes of Might and Magic III AI on the [VCMI](https://github.com/vcmi/vcmi) engine.

## Layout

- `vcmi/` — submodule: our VCMI branch `alpha_heroes`, based on the 1.7.5 release
  - upstream headless/testmap fixes cherry-picked from `develop`
  - `VCMI_USER_DIR` env var to keep game data, config, saves and logs out of `~/Library`
- `agents/` — our AI libraries, built into `build/bin/AI/` via VCMI's `EXTRA_AI_DIRS`
  - `UpLeftAI` — test battle AI: units walk to the top-left hex and hit adjacent enemies.
    Select it in the launcher under Settings → AI (Enemy / Neutral / Friendly combat AI).
- `scripts/vcmi.sh` — runs the local build with user data in `runtime/vcmi-user`
- `build/` — CMake build output (ignored)
- `runtime/` — game data, VCMI config, saves, logs (ignored; the game data is copyrighted)

## Setup (macOS, Apple Silicon)

```sh
git clone --recurse-submodules https://github.com/ARozental/alpha_heroes.git
cd alpha_heroes

brew install cmake ninja ccache innoextract boost minizip sdl2 sdl2_image sdl2_mixer \
  sdl2_ttf tbb libsquish onnxruntime fmt glaze ffmpeg qt

cmake -S vcmi -B build -G Ninja -D CMAKE_BUILD_TYPE=RelWithDebInfo \
  -D CMAKE_C_COMPILER_LAUNCHER=ccache -D CMAKE_CXX_COMPILER_LAUNCHER=ccache -D ENABLE_TEST=OFF \
  -D EXTRA_AI_DIRS="$(pwd)/agents/UpLeftAI"
cmake --build build
```

Game data comes from the GOG offline installer of *Heroes 3 Complete*:

```sh
mkdir -p runtime/vcmi-user
innoextract -s -g -d runtime/gog-extract <path>/setup_heroes_of_might_and_magic_3_complete_*.exe
mv runtime/gog-extract/{Data,Maps,Mp3} runtime/vcmi-user/ && rm -rf runtime/gog-extract
```

## Running

Run `scripts/make_app.sh` once to create `Heroes 3.app` in the repo root; double-click it to open the VCMI launcher (Play starts the game).

```sh
scripts/vcmi.sh client              # game main menu
scripts/vcmi.sh                     # launcher (play normally)
```

AI-only games without a window: always use `scripts/headless.sh`, never `vcmi.sh client --headless` directly.
Each run gets its own directory in `runtime/runs/` (own settings and logs, info-level logging) and is
killed on a time limit, a disk limit, a stalled log (a stack sample is saved first) or when the script exits:

```sh
scripts/headless.sh "Maps/Elbow Room" 120 100      # map, max seconds, max MB
SETTINGS='{"ai":{"combatEnemyAI":"UpLeftAI"}}' scripts/headless.sh "Maps/Elbow Room"
```
