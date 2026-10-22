#!/bin/sh
# Export the engine's id -> name/stats tables to battle_model/artifacts/vocab.json (read by battle_model/vocab.py).
# Generated, not in git: names and stats come from the Heroes 3 game data.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export VCMI_USER_DIR="$ROOT/runtime/vcmi-user"
mkdir -p "$ROOT/battle_model/artifacts"
exec "$ROOT/build/bin/alpha_export_vocab" "$ROOT/battle_model/artifacts/vocab.json"
