#!/bin/sh
# HotA through VCMI's port for the hota branch (docs/hota_status.md).
# - Downloads the port's release once into $HOTA_MOD_DIR (default ~/projects/hota-mod): 442 MB, its own assets, no
#   HotA game files needed. It is checked by sha256 and extracted there, kept apart from any repo (downloaded data).
# - Makes runtime/vcmi-user-hota:
#   - Data: SoD's, from the main checkout's runtime/vcmi-user;
#   - Mods/hota: the extracted port;
#   - a mod preset of vcmi, core, hota.
# Then, for example:
#   VCMI_USER_DIR=$PWD/runtime/vcmi-user-hota build/bin/alpha_export_vocab /tmp/vocab_hota.json
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MAIN="${ALPHA_MAIN:-$HOME/projects/alpha_heroes}" # the checkout whose runtime/vcmi-user holds SoD's Data
MOD="${HOTA_MOD_DIR:-$HOME/projects/hota-mod}"
NAME=horn-of-the-abyss-vcmi-1.7
URL=https://github.com/vcmi-mods/horn-of-the-abyss/releases/download/1.7/$NAME.zip
SHA=d10feca4b21c750eed4d93974ed15c4397cd7bcdbe37e3ac7411b56ea1bf7741

mkdir -p "$MOD/zip" "$MOD/extracted"
[ -f "$MOD/zip/$NAME.zip" ] || curl -sSL --max-time 1800 -o "$MOD/zip/$NAME.zip" "$URL"
echo "$SHA  $MOD/zip/$NAME.zip" | shasum -a 256 -c -
[ -d "$MOD/extracted/$NAME" ] || (cd "$MOD/extracted" && unzip -q "../zip/$NAME.zip")

U="$ROOT/runtime/vcmi-user-hota"
mkdir -p "$U/Mods" "$U/config"
ln -sfn "$MAIN/runtime/vcmi-user/Data" "$U/Data"
ln -sfn "$MOD/extracted/$NAME" "$U/Mods/hota"
cat > "$U/config/modSettings.json" <<'EOF'
{
	"activeMods" : null,
	"activePreset" : "default",
	"presets" : {
		"default" : {
			"mods" : [ "vcmi", "core", "hota" ],
			"settings" : {  }
		}
	}
}
EOF
echo "ready: $U"
