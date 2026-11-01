#!/bin/bash
# Samples a unit's memory every 10 s into runtime/memwatch-<unit>.csv (time, unit GB, WSL available GB, swap used GB,
# GB by process name, and the trainer's (the unit's main process) resident memory and peak), and keeps the kernel from
# ever choosing what to kill:
# - under SHED_AVAILABLE_GB (default 3) it ends the youngest game (its VCMI client: the least play lost; the run counts
#   it as failed, never trains on it), one each 20 s while memory stays under: a game lost, not the run;
# - under MIN_AVAILABLE_GB (default 1.5) it stops the unit (the last resort: shedding did not keep up).
# Each shed or stop writes what held the memory, process by process, to runtime/memwatch-<unit>-events.log.
# (The run's admission keeps the trainer's peak free and games under a floor: battle_model/supply.py, memory_allows;
# joint7 was stopped at 1.06 GB on 2026-11-01 12:21 CST, games let in at the trainer's low point between updates.)
# Usage: runtime/memwatch.sh <unit>    (ends with the unit)
cd "$(dirname "$0")/.."
unit="$1"
min="${MIN_AVAILABLE_GB:-1.5}"
shed="${SHED_AVAILABLE_GB:-3}"
out="runtime/memwatch-$unit.csv"
events="runtime/memwatch-$unit-events.log"
cg=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/$unit.service
echo "time,unit_gb,available_gb,swap_used_gb,by_process" > "$out"
under() { awk -v a="$1" -v m="$2" 'BEGIN {exit !(a < m)}'; }
snapshot() {
	echo "$(date '+%F %T') $1"
	for p in $(cat "$cg/cgroup.procs" 2>/dev/null); do
		echo "$(awk '/^VmRSS:/ {printf "%.2f", $2/2^20}' /proc/$p/status 2>/dev/null) GB  age $(ps -o etimes= -p $p 2>/dev/null | tr -d ' ')s  $p $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-120)"
	done | sort -rn | head -40
	echo
}
last_shed=0
while [ "$(systemctl --user show "$unit" -p ActiveState --value)" = active ]; do
	unit_gb=$(awk '{printf "%.2f", $1/2^30}' "$cg/memory.current" 2>/dev/null)
	available=$(awk '/MemAvailable/ {printf "%.2f", $2/2^20}' /proc/meminfo)
	swap=$(awk '/SwapTotal/ {t=$2} /SwapFree/ {f=$2} END {printf "%.2f", (t-f)/2^20}' /proc/meminfo)
	procs=$(for p in $(cat "$cg/cgroup.procs" 2>/dev/null); do ps -o comm=,rss= -p "$p" 2>/dev/null; done | awk '{n[$1]++; r[$1]+=$2} END {for (k in n) printf "%s:%d:%.2f ", k, n[k], r[k]/2^20}')
	main=$(systemctl --user show "$unit" -p MainPID --value)
	trainer=$(awk '/^VmRSS:/ {r=$2} /^VmHWM:/ {h=$2} END {if (r) printf "trainer_rss:%.2f trainer_peak:%.2f", r/2^20, h/2^20}' /proc/$main/status 2>/dev/null)
	echo "$(date +%H:%M:%S),$unit_gb,$available,$swap,$procs$trainer" >> "$out"
	if under "$available" "$min"; then
		echo "$(date +%H:%M:%S),stopping: available $available GB < $min GB" >> "$out"
		snapshot "stopping $unit: available $available GB < $min GB" >> "$events"
		systemctl --user stop "$unit"
		break
	fi
	if under "$available" "$shed" && [ $(($(date +%s) - last_shed)) -ge 20 ]; then
		victim=$(for p in $(cat "$cg/cgroup.procs" 2>/dev/null); do [ "$(cat /proc/$p/comm 2>/dev/null)" = vcmiclient ] && echo "$(ps -o etimes= -p $p | tr -d ' ') $p"; done | sort -n | head -1)
		if [ -n "$victim" ]; then
			snapshot "shedding: available $available GB < $shed GB: ending the youngest game (age ${victim% *}s, pid ${victim#* })" >> "$events"
			echo "$(date +%H:%M:%S),shedding: available $available GB < $shed GB: ended pid ${victim#* } (age ${victim% *}s)" >> "$out"
			kill -TERM "${victim#* }" 2>/dev/null
			last_shed=$(date +%s)
		fi
	fi
	sleep 10
done
