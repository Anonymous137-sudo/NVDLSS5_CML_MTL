#!/bin/bash
# Development build only. End users launch OBS DLSS NR.app directly.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
obs_dir="$project_dir/../obs-studio"
if pgrep -f "^$project_dir/OBS DLSS NR.app/Contents/MacOS/OBS" >/dev/null; then
 echo "Close OBS DLSS NR before rebuilding its bundle.";exit 1
fi
bash "$project_dir/nr/build-runtime.sh"
(cd "$obs_dir" && git submodule update --init --depth 1 plugins/obs-browser plugins/obs-websocket)
cmake --preset macos -S "$obs_dir" -B "$obs_dir/build_macos" \
 -DENABLE_BROWSER=OFF -DENABLE_SCRIPTING=OFF -DENABLE_AJA=OFF -DENABLE_DECKLINK=OFF \
 -DENABLE_VIRTUALCAM=OFF -DENABLE_VST=OFF -DENABLE_WEBSOCKET=OFF \
 -DCMAKE_OSX_ARCHITECTURES=arm64 -DSPARKLE_APPCAST_URL= -DSPARKLE_PUBLIC_KEY= \
 -DENABLE_NEURAL_POST=OFF -DENABLE_DLSS_NR=ON
xcodebuild -project "$obs_dir/build_macos/obs-studio.xcodeproj" -configuration Release \
 -target ALL_BUILD -jobs 6 CODE_SIGNING_ALLOWED=NO DEBUG_INFORMATION_FORMAT=dwarf
cmake -S "$project_dir/nr" -B "$project_dir/build/nr-plugin" -G Ninja
cmake --build "$project_dir/build/nr-plugin"
bash "$project_dir/nr/package.sh"
