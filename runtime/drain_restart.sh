#!/bin/bash
# Restarts the PC's run (a deploy) losing no game: no new game starts (the run directory's "drain" file, supply.py's
# place), the games playing finish and reach the trainer (on disk from then on: learner.py's spool), then the unit is
# stopped (the battle buffer saved), and started again (it resumes: the spooled games and the buffer come back).
# Run on the Mac (it sees the Mac's slots too). Usage: runtime/drain_restart.sh [max minutes to wait, default 25]
pc=Owner@192.168.1.178
max=$((${1:-25} * 60))
on_pc() { echo "$1" | ssh -o BatchMode=yes $pc 'wsl --cd ~ bash -s' | tr -d '\0'; }
run=$(on_pc 'ls -d ~/projects/alpha_heroes/runtime/train/*-joint7 | tail -1')
on_pc "touch $run/drain"
echo "$(date +%T) draining $run"
cg=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/alpha-joint7.service
started=$(date +%s)
while true; do
	here=$(on_pc "for p in \$(cat $cg/cgroup.procs); do cat /proc/\$p/comm 2>/dev/null; done | grep -c vcmiclient")
	mac=$(pgrep -f vcmiclient | wc -l | tr -d ' ')
	[ "${here:-1}" -eq 0 ] && [ "$mac" -eq 0 ] && break
	[ $(($(date +%s) - started)) -ge $max ] && { echo "$(date +%T) still playing after ${1:-25} min (PC $here, Mac $mac): stopping anyway"; break; }
	sleep 15
done
sleep 20  # (the last games' records into the trainer, and onto its disk)
echo "$(date +%T) stopping"
on_pc "systemctl --user stop alpha-joint7; rm -f $run/drain; systemctl --user reset-failed 2>/dev/null; systemd-run --user --unit=alpha-joint7 -p KillSignal=SIGINT -p TimeoutStopSec=900 -p KillMode=mixed -p LimitNOFILE=65536 ~/projects/alpha_heroes/runtime/advice_run.sh"
echo "$(date +%T) started again"
