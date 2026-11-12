#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
backend="${MLX_DLSS_DIR:-$root/../MLX-DLSS}"
out="${NR5_BUILD_DIR:-$root/build/macos}"
runtime="$out/runtime"
MLX_DLSS_DIR="$backend" NR5_RUNTIME_OUTPUT="$runtime" bash "$root/src/macos/build-runtime.sh"
cmake -S "$root/macos" -B "$out/plugin" -G Ninja -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DOBS_HEADERS="${OBS_HEADERS:?Set OBS_HEADERS to matching OBS source}" -DNR5_RUNTIME_DIR="$runtime"
cmake --build "$out/plugin"
plugin="$out/obs-dlss5-nr.plugin"
if [[ -e "$plugin" ]]; then
  echo "Output exists. Choose a fresh NR5_BUILD_DIR: $out" >&2
  exit 1
fi
ditto "$out/plugin/obs-dlss5-nr.plugin" "$plugin"
mkdir -p "$plugin/Contents/Frameworks" "$plugin/Contents/Resources"
cp "$runtime/libobs_dlss5_nr.dylib" "$runtime/libnvgnx_dlss_mlx.dylib" "$plugin/Contents/Frameworks/"
cp "$backend/.build/release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" "$plugin/Contents/Frameworks/mlx.metallib"
cp "$plugin/Contents/Frameworks/mlx.metallib" "$plugin/Contents/Resources/mlx.metallib"
ditto "$root/data/locale" "$plugin/Contents/Resources/locale"
cp "$root/LICENSE" "$root/THIRD_PARTY_NOTICES.md" "$plugin/Contents/Resources/"
cp "$backend/LICENSE" "$plugin/Contents/Resources/MLX-DLSS-LICENSE"
cp "$backend/NOTICE" "$plugin/Contents/Resources/MLX-DLSS-NOTICE"
if [[ -n "${NR5_MODEL71:-}" ]]; then ditto "$NR5_MODEL71" "$plugin/Contents/Resources/NeuralRendering.dlssmodel"; fi
if [[ -n "${NR5_MODEL31:-}" ]]; then ditto "$NR5_MODEL31" "$plugin/Contents/Resources/NeuralRendering-Small.dlssmodel"; fi
for library in "$plugin"/Contents/Frameworks/*.dylib; do codesign --force --sign - "$library"; done
codesign --force --sign - "$plugin/Contents/Frameworks/mlx.metallib"
codesign --force --sign - "$plugin/Contents/Resources/mlx.metallib"
codesign --force --sign - "$plugin"
codesign --verify --deep --strict "$plugin"
echo "$plugin"
