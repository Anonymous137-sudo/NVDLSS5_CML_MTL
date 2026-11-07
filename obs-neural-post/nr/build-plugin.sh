#!/bin/bash
# Build the Apple Silicon plugin against an installed OBS release.
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="$project_dir/build/nr-plugin-stock"
bash "$project_dir/nr/build-runtime.sh"
cmake -S "$project_dir/nr" -B "$build_dir" -G Ninja -DCMAKE_OSX_ARCHITECTURES=arm64 "$@"
cmake --build "$build_dir"
bash "$project_dir/nr/package-plugin.sh" "$build_dir/obs-dlss-nr.plugin" "${DLSS_PLUGIN_OUTPUT:-$project_dir/build/stock-obs/obs-dlss-nr.plugin}"
