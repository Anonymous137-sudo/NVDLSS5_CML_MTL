#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
obs_app="${OBS_APP:-/Applications/OBS.app}"
obs_version="$(/usr/libexec/PlistBuddy -c 'Print CFBundleShortVersionString' "$obs_app/Contents/Info.plist")"
obs_sdk="${OBS_SOURCE_DIR:-$project_dir/.deps/obs-studio-$obs_version}"
if [[ ! -f "$obs_sdk/libobs/obs.h" ]]; then
    echo "Provide matching OBS headers with OBS_SOURCE_DIR. Installed OBS version: $obs_version" >&2
    exit 1
fi
sdk_major="$(awk '/^#define LIBOBS_API_MAJOR_VER / {print $3}' "$obs_sdk/libobs/obs-config.h")"
if [[ "$sdk_major" != "${obs_version%%.*}" ]]; then
    echo "OBS header/app ABI mismatch: SDK $sdk_major, application $obs_version" >&2
    exit 1
fi
swift build -c release
cmake -S . -B build/plugin -G Ninja -DOBS_SOURCE_DIR="$obs_sdk" -DOBS_APP="$obs_app"
cmake --build build/plugin
plugin="$project_dir/build/plugin/obs-neural-post.plugin"
codesign --force --sign - "$plugin/Contents/Frameworks/libNeuralPost.dylib"
codesign --force --sign - "$plugin"
build/plugin/obs-neural-post-load-test "$plugin/Contents/MacOS/obs-neural-post" "$project_dir/models/rt4ksr-x3.safetensors"
echo "Built and load-tested: $plugin"
