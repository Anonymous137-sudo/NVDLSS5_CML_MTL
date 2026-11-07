#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
app="$project_dir/OBS DLSS NR.app"
source_app="$project_dir/../obs-studio/build_macos/frontend/Release/OBS.app"
if pgrep -f "^$app/Contents/MacOS/OBS" >/dev/null; then echo "Close the staged OBS app before repackaging";exit 1;fi
if [[ -d "$app/Contents/PlugIns/obs-dlss-nr.plugin" ]]; then rm -rf "$app";fi
[[ ! -e "$app" ]] || { echo "Unexpected app at staging destination";exit 1; }
cp -cR "$source_app" "$app"
plugin="$app/Contents/PlugIns/obs-dlss-nr.plugin"
rm -rf "$plugin"
bash "$project_dir/nr/package-plugin.sh" "$project_dir/build/nr-plugin/obs-dlss-nr.plugin" "$plugin"
/usr/libexec/PlistBuddy -c 'Set :CFBundleDisplayName OBS DLSS NR' "$app/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Set :CFBundleIdentifier local.obs-neural-post.OBS' "$app/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Set :NSCameraUsageDescription Display the PS5 HDMI capture feed in OBS DLSS NR.' "$app/Contents/Info.plist"
codesign --force --deep --sign - "$app"
codesign --verify --deep --strict "$app"
echo "$app"
