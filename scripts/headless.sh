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
# The game's TBB thread pool (vcmi/lib/AsyncRunner.h): one worker a hardware thread by default, 23 on the PC, of which
# ~13 spun between Nullkiller2's parallel loops at every AI request, ~60% of a game's CPU (2026-11-02). Four is the pool's
# minimum (AsyncRunner's requiredWorkersCount): a turn's blocking AI task and the one that unblocks it both get a thread
export VCMI_MAX_WORKERS="${VCMI_MAX_WORKERS:-4}"

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
LAST_TICK=$SECONDS
while kill -0 "$PID" 2>/dev/null; do
	sleep 1
	# The machine was asleep (a closed lid): the gap is no silence and no game time, so every timer moves past it
	if (( SECONDS - LAST_TICK > 10 )); then
		GAP=$(( SECONDS - LAST_TICK - 1 ))
		START=$(( START + GAP )); LAST_LOG_CHANGE=$(( LAST_LOG_CHANGE + GAP ))
	fi
	LAST_TICK=$SECONDS
	LOG_SIZE=$(stat -c %s "$RUN_DIR/logs/VCMI_Client_log.txt" 2>/dev/null || stat -f %z "$RUN_DIR/logs/VCMI_Client_log.txt" 2>/dev/null || echo 0)  # GNU, then BSD stat
	if [ "$LOG_SIZE" != "$LAST_LOG_SIZE" ]; then
		LAST_LOG_SIZE=$LOG_SIZE
		LAST_LOG_CHANGE=$SECONDS
		CPU_AT_CHANGE=$(cpu_seconds)
	elif (( SECONDS - LAST_LOG_CHANGE >= STALL_SECONDS )) && (( $(cpu_seconds) - CPU_AT_CHANGE >= STALL_SECONDS / 4 )); then
		# silent but computing (e.g. BattleAI thinking for minutes in a big battle): not a stall; the time limit still holds
		LAST_LOG_CHANGE=$SECONDS
		CPU_AT_CHANGE=$(cpu_seconds)
	elif (( SECONDS - LAST_LOG_CHANGE >= STALL_SECONDS )); then
		if command -v sample > /dev/null; then
			# sample itself can hang on a stuck client (it did, for hours, blocking an eval and two real-game actors)
			sample "$PID" 3 -file "$RUN_DIR/stall_sample.txt" > /dev/null 2>&1 &
			SAMPLER=$!
			for _ in {1..20}; do kill -0 "$SAMPLER" 2>/dev/null || break; sleep 1; done
			kill -KILL "$SAMPLER" 2>/dev/null
		else
			# Linux (the PC's WSL: no sample, no debugger, ptrace_scope 1): each thread's state and the kernel function it
			# waits in, and the client's descriptors. Every thread in futex_wait is a deadlock; one in a socket read waits
			# on the inference or the server; R is a busy loop. Readable without ptrace
			for P in "$PID" $(pgrep -P "$PID"); do  # (and a server process of its own, if any)
				echo "pid $P: $(tr '\0' ' ' < "/proc/$P/cmdline" 2>/dev/null)"
				for TASK in /proc/"$P"/task/*; do
					echo "$(basename "$TASK") $(cat "$TASK/comm" 2>/dev/null) state $(sed 's/.*) //' "$TASK/stat" 2>/dev/null | cut -d' ' -f1) wchan $(cat "$TASK/wchan" 2>/dev/null)"
				done
				ls -l /proc/"$P"/fd 2>/dev/null
				# Each socket's two ends (ss: the bytes unread on either side, the process that holds the other end), then
				# that process's threads counted by state and kernel wait. A battle AI waited 60 s for a reply while the
				# inference server showed no request waiting (2026-11-10 06:51, w4_g4_14; 16 such samples kept, most at a
				# battle's start): bytes unread on the server's end mean it never read the request, bytes unread on the
				# client's that the reply came and was not read, none on either that the request never came
				if command -v ss > /dev/null; then
					SOCKETS=$(ss -x -a -p -n 2>/dev/null)
					PEERS=""
					for INODE in $(ls -l /proc/"$P"/fd 2>/dev/null | sed -n 's/.*socket:\[\([0-9]*\)\].*/\1/p'); do
						ENDS=$(echo "$SOCKETS" | grep -E "(^| )$INODE( |\$)" | tr -s ' ')
						echo "$ENDS"
						PEERS="$PEERS $(echo "$ENDS" | grep -o 'pid=[0-9]*' | cut -d= -f2 | tr '\n' ' ')"
					done
					for PEER in $(echo $PEERS | tr ' ' '\n' | sort -un); do
						[ "$PEER" = "$P" ] && continue
						echo "peer pid $PEER: $(tr '\0' ' ' < "/proc/$PEER/cmdline" 2>/dev/null | cut -c1-200)"
						for TASK in /proc/"$PEER"/task/*; do
							echo "$(sed 's/[0-9][0-9]*/N/' "$TASK/comm" 2>/dev/null) state $(sed 's/.*) //' "$TASK/stat" 2>/dev/null | cut -d' ' -f1) wchan $(cat "$TASK/wchan" 2>/dev/null)"
						done | sort | uniq -c | sort -rn  # (a server's threads: its worker, the trainer's commands, a reader a connection)
					done
				fi
				echo
			done > "$RUN_DIR/stall_sample.txt" 2>&1
		fi
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
