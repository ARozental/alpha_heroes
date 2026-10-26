#!/bin/bash
# Run one AI-only headless game in its own run directory, with hard limits.
#
# Usage: scripts/headless.sh <map> [max_seconds=300] [max_mb=200] [stall_seconds=30]
#   map          e.g. "Maps/Elbow Room"
#   VCMI_LLDB    optional env var: run the client under lldb, which prints every thread's backtrace on a crash
#   VCMI_CLIENT  optional env var: the client binary in build/bin (default vcmiclient)
#   SETTINGS     optional env var: JSON merged into the run's settings.json,
#                e.g. SETTINGS='{"ai":{"combatEnemyAI":"UpLeftAI"}}'
#
# The game is killed when it exceeds max_seconds, when its run directory exceeds max_mb,
# when its log stops growing for stall_seconds while it does not compute either (a stack sample is saved first),
# or when this script exits for any reason. Prints run size and data rate at the end.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAP="${1:?usage: scripts/headless.sh <map> [max_seconds] [max_mb]}"
MAX_SECONDS="${2:-300}"
MAX_MB="${3:-200}"
STALL_SECONDS="${4:-30}"
SHARED="$ROOT/runtime/vcmi-user"
export ALPHA_BATTLE_MODEL="${ALPHA_BATTLE_MODEL:-$ROOT/battle_model/artifacts/battle_model.onnx}"

RUN_DIR="$ROOT/runtime/runs/$(date +%Y%m%d-%H%M%S)-$$-$(basename "$MAP" | tr -c 'A-Za-z0-9\n' '_')"
mkdir -p "$RUN_DIR/config"
for dir in Data Maps Mp3 Mods; do
	ln -s "$SHARED/$dir" "$RUN_DIR/$dir"
done
[ -f "$SHARED/config/modSettings.json" ] && cp "$SHARED/config/modSettings.json" "$RUN_DIR/config/"

python3 -I - "$RUN_DIR/config/settings.json" "${SETTINGS:-"{}"}" <<'EOF'
import json, sys

def merge(base, extra):
	for key, value in extra.items():
		if isinstance(value, dict) and isinstance(base.get(key), dict):
			merge(base[key], value)
		else:
			base[key] = value
	return base

settings = {"logging": {"loggers": [{"domain": "global", "level": "info"}, {"domain": "rng", "level": "info"}]}}
json.dump(merge(settings, json.loads(sys.argv[2])), open(sys.argv[1], "w"), indent=1)
EOF

cd "$ROOT/build/bin"
if [ -n "${VCMI_LLDB:-}" ]; then  # debugging a crash: every thread's backtrace lands in stdout.txt
	VCMI_USER_DIR="$RUN_DIR" lldb --batch -o "process handle SIGPIPE -n true -p true -s false" -o run -k "thread backtrace all" -k quit -- ./vcmiclient --headless --testmap "$MAP" > "$RUN_DIR/stdout.txt" 2>&1 &
else
	VCMI_USER_DIR="$RUN_DIR" "./${VCMI_CLIENT:-vcmiclient}" --headless --testmap "$MAP" > "$RUN_DIR/stdout.txt" 2>&1 &
fi
PID=$!

cleanup()
{
	if kill -0 "$PID" 2>/dev/null; then
		pkill -TERM -P "$PID" 2>/dev/null
		kill -TERM "$PID" 2>/dev/null
		sleep 1
		pkill -KILL -P "$PID" 2>/dev/null
		kill -KILL "$PID" 2>/dev/null
	fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

STATUS="finished"
START=$SECONDS
LAST_LOG_SIZE=-1
LAST_LOG_CHANGE=$SECONDS
cpu_seconds() { ps -o time= -p "$PID" 2>/dev/null | awk -F: '{ s = 0; for (i = 1; i <= NF; i++) s = s * 60 + $i; printf "%d", s }'; }
CPU_AT_CHANGE=$(cpu_seconds)
while kill -0 "$PID" 2>/dev/null; do
	sleep 1
	LOG_SIZE=$(stat -f %z "$RUN_DIR/logs/VCMI_Client_log.txt" 2>/dev/null || echo 0)
	if [ "$LOG_SIZE" != "$LAST_LOG_SIZE" ]; then
		LAST_LOG_SIZE=$LOG_SIZE
		LAST_LOG_CHANGE=$SECONDS
		CPU_AT_CHANGE=$(cpu_seconds)
	elif (( SECONDS - LAST_LOG_CHANGE >= STALL_SECONDS )) && (( $(cpu_seconds) - CPU_AT_CHANGE >= STALL_SECONDS / 4 )); then
		# silent but computing (e.g. BattleAI thinking for minutes in a big battle): not a stall; the time limit still holds
		LAST_LOG_CHANGE=$SECONDS
		CPU_AT_CHANGE=$(cpu_seconds)
	elif (( SECONDS - LAST_LOG_CHANGE >= STALL_SECONDS )); then
		# sample itself can hang on a stuck client (it did, for hours, blocking an eval and two real-game actors)
		sample "$PID" 3 -file "$RUN_DIR/stall_sample.txt" > /dev/null 2>&1 &
		SAMPLER=$!
		for _ in {1..20}; do kill -0 "$SAMPLER" 2>/dev/null || break; sleep 1; done
		kill -KILL "$SAMPLER" 2>/dev/null
		STATUS="killed: no log output for ${STALL_SECONDS}s, stack sample in stall_sample.txt"
		break
	fi
	if (( SECONDS - START >= MAX_SECONDS )); then
		STATUS="killed: time limit ${MAX_SECONDS}s"
		break
	fi
	if (( $(du -sm "$RUN_DIR" | cut -f1) >= MAX_MB )); then
		STATUS="killed: disk limit ${MAX_MB} MB"
		break
	fi
done
cleanup
wait "$PID" 2>/dev/null
EXIT_CODE=$?

ELAPSED=$(( SECONDS - START ))
SIZE_KB=$(du -sk "$RUN_DIR" | cut -f1)
echo "run:      $RUN_DIR"
echo "status:   $STATUS (exit code $EXIT_CODE)"
echo "elapsed:  ${ELAPSED}s"
awk -v kb="$SIZE_KB" -v s="$ELAPSED" 'BEGIN { printf "data:     %.1f MB (%.2f MB/min)\n", kb / 1024, (s > 0 ? kb / 1024 / s * 60 : 0) }'
