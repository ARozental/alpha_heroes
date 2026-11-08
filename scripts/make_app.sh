#!/bin/sh
# Build the double-click apps in the repo root:
#   "Heroes 3.app"             opens the VCMI launcher
#   "Heroes 3 (recorded).app"  plays a recorded game (campaign_model/record_human.py: the client opens, every move of
#                              yours is recorded into runtime/human/<start time>; a notification when it is saved)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

make_app() {  # name, shell command run in the repo (in the background)
	APP="$ROOT/$1.app"
	rm -rf "$APP"
	osacompile -o "$APP" -e "
set appDir to POSIX path of (path to me)
set repoDir to do shell script \"dirname \" & quoted form of appDir
do shell script \"cd \" & quoted form of repoDir & \" && ($2) > /dev/null 2>&1 &\""
	cp "$ROOT/vcmi/osx/vcmi.icns" "$APP/Contents/Resources/applet.icns" 2>/dev/null || true
	echo "Created $APP"
}

make_app "Heroes 3" "scripts/vcmi.sh launcher"
# (the session's output to runtime/human/sessions.log; the folder it saved in, in the notification)
make_app "Heroes 3 (recorded)" "mkdir -p runtime/human; .venv/bin/python -m campaign_model.record_human >> runtime/human/sessions.log 2>&1; osascript -e 'display notification \\\"Saved in runtime/human (see sessions.log)\\\" with title \\\"Heroes 3: game recorded\\\"'"
