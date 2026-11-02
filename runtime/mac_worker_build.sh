#!/bin/bash
# The Mac worker's own copy of the game binaries (alpha_heroes-e5/build -> build-own: a build of the main tree must not
# change what a running worker plays): a clone of the main build's bin, its rpaths to the copy, re-signed, swapped in
set -e
E5=/Users/alon/projects/alpha_heroes-e5
OLD=/Users/alon/projects/alpha_heroes/build/bin
NEW=$E5/build-own/bin
rm -rf $E5/build-own.new && mkdir -p $E5/build-own.new
cp -cR $OLD $E5/build-own.new/bin
cd $E5/build-own.new/bin
n=0
for f in $(find . -maxdepth 2 -type f \( -name "*.dylib" -o -perm -u+x \) | grep -v "\.a$"); do
	file "$f" | grep -q "Mach-O" || continue
	changed=0
	for rp in $(otool -l "$f" | grep -A2 LC_RPATH | awk '/path/ {print $2}'); do
		case "$rp" in $OLD*) install_name_tool -rpath "$rp" "$NEW${rp#$OLD}" "$f" 2>/dev/null && changed=1;; esac
	done
	[ $changed = 1 ] && { codesign --force --sign - "$f" 2>/dev/null; n=$((n+1)); }
done
cd $E5
rm -rf build-own.old; [ -d build-own ] && mv build-own build-own.old; mv build-own.new build-own; rm -rf build-own.old
ln -sfn build-own build
echo "$(date +%T) worker build refreshed: $n binaries re-pointed; AlphaMapAI $(stat -f %Sm -t %H:%M build/bin/AI/libAlphaMapAI.dylib)"
