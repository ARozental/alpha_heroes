#!/bin/sh
# Run a binary from the local VCMI build with all user data kept inside the repo.
# Usage: scripts/vcmi.sh [launcher|client|server|mapeditor] [args...]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export VCMI_USER_DIR="$ROOT/runtime/vcmi-user"
export ALPHA_BATTLE_MODEL="${ALPHA_BATTLE_MODEL:-$ROOT/battle_model/artifacts/battle_model.onnx}"
TOOL="${1:-launcher}"
[ $# -gt 0 ] && shift
cd "$ROOT/build/bin"
exec "./vcmi$TOOL" "$@"
