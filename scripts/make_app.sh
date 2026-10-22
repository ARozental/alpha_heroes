#!/bin/sh
# Build "Heroes 3.app" in the repo root: double-click to open the VCMI main menu.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/Heroes 3.app"
rm -rf "$APP"
osacompile -o "$APP" -e '
set appDir to POSIX path of (path to me)
set repoDir to do shell script "dirname " & quoted form of appDir
do shell script quoted form of (repoDir & "/scripts/vcmi.sh") & " client > /dev/null 2>&1 &"'
cp "$ROOT/vcmi/osx/vcmi.icns" "$APP/Contents/Resources/applet.icns" 2>/dev/null || true
echo "Created $APP"
