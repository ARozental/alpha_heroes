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
  -D EXTRA_AI_DIRS="$(pwd)/agents/UpLeftAI;$(pwd)/agents/AlphaBattleAI;$(pwd)/agents/AlphaMapAI"
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

## Training (battle and campaign together)

```sh
.venv/bin/python -m campaign_model.joint --run NAME --hours 12        # a new run (battle model from runtime/saved/)
.venv/bin/python -m campaign_model.joint --run NAME --hours 12 --resume   # go on after a stop (kill -TERM <pid>)
.venv/bin/tensorboard --logdir runtime/train                          # tab eval: battle, campaign
```

`eval/battle` is the battle model against BattleAI and MMAI (mean reward, -1..1; greedy, on fixed battles that play
the same every time), `eval/campaign` the campaign model against Nullkiller2 (our share of all empires' value at day
56 x players: 1 = an average player); the eval tab holds nothing else (the per-set and per-opponent details are under
eval_details). Training itself is self-play, the wandering monsters BattleAI as in a real game.

### On several machines (one run)

The machine with the GPU (the master) trains; others (workers) play games for it. A worker runs only VCMI: every
decision is asked of the master's inference servers over the network (~4 KB a battle decision, compressed), and its
battle records and finished campaign games go to the master (`battle_model/remote.py`, `battle_model/worker.py`).

1. Every machine: the same code (`git pull`, the same `vcmi` submodule commit, rebuilt: `cmake --build build`) and the
   game data (`runtime/vcmi-user`, as above; the master refuses a worker whose model contracts differ from its own).
2. The master: the run with `remote_port` set (it listens there, its battle inference on `remote_port + 1 + i`, the
   campaign's on `remote_port + 51 + i`):
   ```sh
   .venv/bin/python -m campaign_model.joint --run NAME --hours 12 --machine configs/machines/rtx5090/joint-gpu-free.toml --set remote_port=7700
   ```
   The first such run writes the shared secret to `runtime/remote-token`; copy it to each worker's `runtime/`.
3. Each worker, its own sizes (`configs/machines/<machine>/worker.toml` or `--set generated_actors=N campaign.games=M`):
   ```sh
   .venv/bin/python -m battle_model.worker --master <master's address> --machine configs/machines/mac-m4/worker.toml
   ```
   It waits for the master and reconnects after a stop and resume of the run. The master logs `remote: worker ... joined`
   and, when one leaves, what it played.
4. The master in WSL2 (the RTX 5090 PC): other machines reach WSL only with mirrored networking (`networkingMode=mirrored`
   under `[wsl2]` in `C:\Users\<you>\.wslconfig`, then `wsl --shutdown`: it stops every WSL process) or a
   `netsh interface portproxy` per port. Inbound TCP 7700-7760 must pass Windows' firewall and, in mirrored mode, WSL's
   Hyper-V firewall (an administrator PowerShell: `New-NetFirewallRule -DisplayName alpha_heroes -Direction Inbound
   -Protocol TCP -LocalPort 7700-7760 -Action Allow` and `New-NetFirewallHyperVRule -Name alpha_heroes -DisplayName
   alpha_heroes -Direction Inbound -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts
   7700-7760`). Check from a worker with the run started: `nc -vz <master's address> 7700`. Home network only: the token
   is the only lock.

### On a second machine (an independent run beside this one)

Two machines do not share one run: the second runs its own (its own seed, or other settings) and the evals, which
are the same battles and maps everywhere, compare them.

1. Setup as above (macOS; on Linux the same packages from the distribution, and torch picks CUDA by itself). The
   repos are private: the machine needs an SSH key of the ARozental account for the clone and its submodule.
2. The game data (GOG, as above) into `runtime/vcmi-user/`, then `scripts/export_vocab.sh` (the engine's tables:
   `battle_model/artifacts/vocab.json`).
3. The MMAI mod (an eval opponent): copy `runtime/vcmi-user/Mods/mmai/` from the first machine (or the VCMI mod
   from its repository, its models from `Mods/mmai/models/sources.json`), list it in
   `runtime/vcmi-user/config/modSettings.json` (`"mods": ["vcmi", "core", "mmai"]`), and keep `"seed": 1` in
   `Mods/mmai/config/mmai-settings.json` (evals that play the same every time).
4. The starting models, not in git (`runtime/` is ignored): copy `runtime/saved/` (the battle checkpoints and the
   league's `models/`), and to start the campaign from where the first machine is, its run's
   `campaign/checkpoint_latest.pt`:
   ```sh
   rsync -a FIRST:alpha_heroes/runtime/saved/ runtime/saved/
   rsync -a FIRST:alpha_heroes/runtime/train/<run>/campaign/checkpoint_latest.pt runtime/saved/campaign_start.pt
   ```
5. Run with its own name and seeds; size the games and actors to the machine (24 GB: 6 games and 8 actors, the
   recipe's; 8 and 10 swapped):
   ```sh
   .venv/bin/python -m campaign_model.joint --run NAME2 --hours 12 --battle-from runtime/saved/<battle checkpoint> \
     --set seed=2 campaign.seed=2 campaign.init_from=runtime/saved/campaign_start.pt
   ```
   (campaign.init_from takes the campaign model's weights; its trust in the value heads is measured again.)

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

Everything about *what* the model learns lives in Python; the C++ side only turns the game into the
model's input tensors and the model's choice back into a game action.

The science (Python, the part to read and change):

| File | What it is |
|---|---|
| `model.py` | The network: raw inputs -> embeddings and in-graph transforms (log scaling, relative sides) -> attention over unit, hero, hex and global tokens -> conditional output heads. Also the starting action-type preferences (`ACTION_TYPE_PRIOR`) and the boosted value heads (V0..V2 = reward formula at the end of this and the next rounds, V = end of battle). Replace the architecture freely; only its inputs and outputs are fixed. |
| `distributions.py` | The maths of the output: log p(full action) as a sum of conditional choices, its exact entropy, and the reference sampler the C++ side mirrors. |
| `losses.py` | How the reward becomes a training signal: value-head labels, returns and advantages (GAE), and the PPO loss with its weights, explained step by step. |
| `reward.py` | What counts as good: the terminal reward, the same formula at the end of a round (`round_score`), its bounds, and the fixed value functions (units, heroes, retreat, surrender). |
| `vocab.py` | What the ids mean: creatures, spells, heroes, skills, artifacts, battlefields and obstacles by engine id, with names and base stats. |
| `scenarios/*.json` | Fixed battles to train on, e.g. `castle_vs_inferno.json`: both heroes, armies, skills, spells, battlefield. |
| `play.py` | Plays recorded battles in VCMI: a scenario becomes a battle-only map; our model on one side, another battle AI (BattleAI) on the other. |
| `data.py` | Battle records -> rewards, value labels and advantages -> per-decision training examples -> DataLoader. |
| `train.py` | The training loop: each epoch plays `games_per_side` battles per side, trains with PPO and logs win rates, rewards (-1..1) and the warning signs of forgetting (Adam signal ratio, weight change, KL, clip fraction) to `runtime/train/<run>/progress.csv` and TensorBoard. Nothing in it may depend on fixed seeds: the same training must work on real, ever-different battles. |
| `evaluate.py` | Win rates of a checkpoint, greedy (temperature 0: one deterministic battle per side) or sampled. |

The contract (shared with C++; change deliberately, then run codegen and bump `SCHEMA_VERSION`):

| File | What it is |
|---|---|
| `constants.py` | Sizes, action types, target kinds, embedding vocabulary sizes. |
| `schema.py` | Every input tensor (raw engine values and legality masks) and output tensor, with dtype, shape, fill value and meaning; random consistent inputs for tests. |

Plumbing (tools around the model):

| File | What it is |
|---|---|
| `export.py` | Writes the ONNX file with the contract hash in its metadata and checks it against torch. |
| `codegen.py` | Generates `agents/AlphaBattleAI/Schema.generated.h` from the contract. |
| `check_dumps.py` | Replays decisions dumped by the C++ side through the model in Python and checks the masks and actions. |
| `tests/` | Contract, maths, reward and vocabulary tests. |

`agents/AlphaBattleAI/` (C++, no modelling in it): `Encoder` reads the battle into the input tensors and
builds the legality masks from the engine's own rules; `Model` loads and validates the ONNX file and runs
it; `Policy` samples an action from the outputs; `AlphaBattleAI` executes it and falls back to StupidAI for
war machines, the tactics phase and round 30 onwards; `tools/ExportVocab` writes the id tables.

```sh
.venv/bin/python -m pytest battle_model/tests       # model, distributions, losses, reward, export, codegen, vocab
scripts/export_vocab.sh                             # id -> name/stats tables into battle_model/artifacts/vocab.json
.venv/bin/python -m battle_model.export             # default model into battle_model/artifacts/battle_model.onnx
.venv/bin/python -m battle_model.codegen            # after changing constants.py / schema.py
ALPHA_BATTLE_DUMP_DIR=$PWD/runtime/dumps SETTINGS='{"ai":{"combatNeutralAI":"AlphaBattleAI"}}' \
  scripts/headless.sh "Maps/Elbow Room" 120 100
.venv/bin/python -m battle_model.check_dumps runtime/dumps   # C++ and Python agree on every dumped decision
```

Training on a scenario. Battle records never touch the disk: AlphaBattleAI keeps each battle's record in memory
and sends it to the trainer over a Unix socket when the battle ends (`play.RecordServer`); `--set keep_records=true`
also writes them to `runtime/train/<run>/records` for debugging. Torch trains on the GPU (`device`, Apple MPS here)
while the CPU plays the next epoch's battles in the background (`prefetch`). Each run plays its own copy of the
scenario's map. A run's folder is `runtime/train/<start time>-<name>` (e.g. `20261023-1412-selfplay`), so name order is
time order; `--run NAME --resume` continues the newest run with that name. Any config field can be set with `--set`, including `ppo.*`, `advantage.*`, `reward.*`:

```sh
.venv/bin/python -m battle_model.train --run NAME            # 8 battles per side per epoch, fixed seeds
.venv/bin/python -m battle_model.train --run NAME --set random_seeds=true   # a fresh VCMI seed for every battle
.venv/bin/python -m battle_model.train --run NAME --set opponent=self       # self-play: our model on both sides, learning from both
# every eval_every (10) epochs an eval round against BattleAI (not trained on): eval.csv and eval/* in TensorBoard
.venv/bin/python -m battle_model.evaluate runtime/train/NAME/checkpoint_latest.pt --temperature 0   # greedy
.venv/bin/python -m battle_model.evaluate runtime/train/NAME/checkpoint_latest.pt --games 16 --map Maps/alpha/castle_vs_inferno-NAME
```

Battles are deterministic given VCMI's seed and our actions, so replaying a battle's first actions
(`play.Game.prefix`, `ALPHA_BATTLE_PREFIX`) restores any of its states: a primitive for search, not used by
training. To try C++ changes while runs use `AlphaBattleAI`, build `--target AlphaBattleAIDev` and pass
`--set ai=AlphaBattleAIDev`.

VCMI changes this relies on: `VCMI_BATTLE_AI_<COLOR>` chooses a player's battle AI, battle-only games end
only after their battle's result is decided, the `--testmap` start-up no longer races the network thread
(it crashed ~1 in 1000 headless games), and `server.seed` in the settings fixes the game's random seed.

To use a better model: export any architecture with `export.py` and point `ALPHA_BATTLE_MODEL` at the file.
AlphaBattleAI refuses a file whose contract differs (schema version, hash, input/output names, dtypes or
shapes); StupidAI then plays.

AlphaBattleAI settings (environment variables): `ALPHA_BATTLE_MODEL` (default
`battle_model/artifacts/battle_model.onnx`, set by the scripts), `ALPHA_BATTLE_TEMPERATURE` (1 = sample,
0 = most likely action), `ALPHA_BATTLE_SEED`, `ALPHA_BATTLE_DUMP_DIR`, `ALPHA_BATTLE_DUMP_LIMIT` (default 50;
one decision is a compressed .npz of about 16 KB), `ALPHA_BATTLE_PREFIX` (actions to replay first, JSON lines).

Illegal actions never cost a turn, and our own checks must never find one: every chosen action is checked
against the masks before it is sent, and a server refusal of a hero spell, retreat or surrender makes the
same unit choose its next best action. Either one logs `BUG:` and means our masks need fixing.
`ALPHA_BATTLE_TEST_FAULT=surrender,spell,decoder` injects such failures on purpose to test the recovery.
