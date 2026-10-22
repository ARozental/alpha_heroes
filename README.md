# alpha_heroes

Training a strong Heroes of Might and Magic III AI on the [VCMI](https://github.com/vcmi/vcmi) engine.

## Layout

- `vcmi/` — submodule: our VCMI branch `alpha_heroes`, based on the 1.7.5 release
  - upstream headless/testmap fixes cherry-picked from `develop`
  - `VCMI_USER_DIR` env var to keep game data, config, saves and logs out of `~/Library`
- `agents/` — our AI libraries, built into `build/bin/AI/` via VCMI's `EXTRA_AI_DIRS`
  - `UpLeftAI` — test battle AI: units walk to the top-left hex and hit adjacent enemies.
  - `AlphaBattleAI` — battle AI that plays with an ONNX model from `battle_model/` (see below).
  - Select either in the launcher under Settings → AI (Enemy / Neutral / Friendly combat AI).
- `battle_model/` — PyTorch battle model, its input/output contract, losses, reward and ONNX export
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
  -D EXTRA_AI_DIRS="$(pwd)/agents/UpLeftAI;$(pwd)/agents/AlphaBattleAI"
cmake --build build

python3 -m venv .venv && .venv/bin/pip install -r battle_model/requirements.txt
.venv/bin/python -m battle_model.export     # writes battle_model/artifacts/battle_model.onnx
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

## Battle model (`battle_model/`, `agents/AlphaBattleAI/`)

The model's interface is a contract shared by Python and C++:

- `battle_model/constants.py` and `schema.py` define every input tensor (raw engine values plus legality
  masks) and output tensor (masked conditional log-probabilities and a value). `codegen.py` generates
  `agents/AlphaBattleAI/Schema.generated.h` from them; a test fails if it is out of date.
- `export.py` stores `schema_version` and `contract_hash` in the ONNX metadata. AlphaBattleAI refuses
  any model whose metadata, input or output names, dtypes or shapes differ, and StupidAI plays instead.
- Any architecture works as long as it keeps the contract: export it and point `ALPHA_BATTLE_MODEL` at it.

```sh
.venv/bin/python -m pytest battle_model/tests       # model, distributions, losses, reward, export, codegen
.venv/bin/python -m battle_model.codegen            # after changing constants.py / schema.py
ALPHA_BATTLE_DUMP_DIR=$PWD/runtime/dumps SETTINGS='{"ai":{"combatNeutralAI":"AlphaBattleAI"}}' \
  scripts/headless.sh "Maps/Elbow Room" 120 100
.venv/bin/python -m battle_model.check_dumps runtime/dumps   # replays C++ decisions through the model in Python
```

AlphaBattleAI settings (environment variables): `ALPHA_BATTLE_MODEL` (default
`battle_model/artifacts/battle_model.onnx`, set by the scripts), `ALPHA_BATTLE_TEMPERATURE` (1 = sample,
0 = most likely action), `ALPHA_BATTLE_SEED`, `ALPHA_BATTLE_DUMP_DIR`, `ALPHA_BATTLE_DUMP_LIMIT` (default 50;
one decision is about 0.2 MB). War machines and the tactics phase are scripted with StupidAI, and so is
everything from round 30 on, so no battle can run forever.

Illegal actions never cost a turn: the masks use the engine's own legality checks, every chosen action is
checked against them before it is sent (an illegal one is excluded and the model chooses again), and if the
server still refuses a hero spell, retreat or surrender it asks the same unit again, which then takes its
next best legal action. `ALPHA_BATTLE_TEST_FAULT=surrender,spell,decoder` injects such failures for testing.
