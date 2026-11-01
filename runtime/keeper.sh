#!/bin/bash
# Starts a run again after the memory guard (runtime/memwatch.sh) stopped it, a minute later (its games' memory freed);
# a stop by hand is left alone (memwatch's CSV then does not end with its "stopping" line), and after 3 restarts in
# 6 hours it gives up (a run that keeps outgrowing the machine needs a change, not more restarts). joint7 stood stopped
# from 12:21 to 13:15 CST on 2026-11-01: the guard stops a run, nothing started it again. The run resumes by itself.
# Usage: runtime/keeper.sh <unit> <script>, as a unit of its own:
#   systemd-run --user --unit=alpha-joint7-keeper ~/projects/alpha_heroes/runtime/keeper.sh alpha-joint7 runtime/advice_run.sh
cd "$(dirname "$0")/.."
unit="$1"
script="$2"
log="runtime/keeper-$unit.log"
restarts=()
while true; do
	sleep 60
	case "$(systemctl --user show "$unit" -p ActiveState --value)" in
		active | activating | deactivating | reloading) continue ;;
	esac
	tail -1 "runtime/memwatch-$unit.csv" 2> /dev/null | grep -q ",stopping:" || continue
	now=$(date +%s)
	recent=0
	for t in "${restarts[@]}"; do
		[ $((now - t)) -lt 21600 ] && recent=$((recent + 1))
	done
	if [ "$recent" -ge 3 ]; then
		echo "$(date '+%F %T') $unit: stopped by the memory guard a 4th time in 6 h: left stopped" >> "$log"
		exit 1
	fi
	echo "$(date '+%F %T') $unit: stopped by the memory guard ($(tail -1 "runtime/memwatch-$unit.csv")): starting it again" >> "$log"
	systemctl --user reset-failed "$unit" 2> /dev/null
	systemd-run --user --unit="$unit" -p KillSignal=SIGINT -p TimeoutStopSec=900 -p KillMode=mixed -p LimitNOFILE=65536 "$PWD/$script" >> "$log" 2>&1
	restarts+=("$now")
done
