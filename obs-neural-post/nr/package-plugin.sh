#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
module="${1:-$project_dir/build/nr-plugin-stock/obs-dlss-nr.plugin}"
plugin="${2:-$project_dir/build/stock-obs/obs-dlss-nr.plugin}"
if [[ -e "$plugin" ]]; then
  echo "Package destination already exists: $plugin" >&2
  exit 1
fi
mkdir -p "$(dirname "$plugin")"
cp -cR "$module" "$plugin"
mkdir -p "$plugin/Contents/Frameworks" "$plugin/Contents/Resources"
for library in libnvgnx_dlss_mlx libnvgnx_dlssnr librtss libnvidia_nr; do
  cp -c "$project_dir/build/nr/$library.dylib" "$plugin/Contents/Frameworks/"
  codesign --force --sign - "$plugin/Contents/Frameworks/$library.dylib"
done
cp -c "$project_dir/../MLX-DLSS/.build/release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" "$plugin/Contents/Frameworks/mlx.metallib"
cp -c "$plugin/Contents/Frameworks/mlx.metallib" "$plugin/Contents/Resources/mlx.metallib"
if [[ "${DLSS_BUNDLE_LOCAL_MODELS:-0}" == 1 ]]; then
  cp -cR "$project_dir/build/teacher/NeuralRendering.dlssmodel" "$plugin/Contents/Resources/"
  cp -cR "$project_dir/build/three-mode/NeuralRendering-Small.dlssmodel" "$plugin/Contents/Resources/"
  cp -c "$project_dir/models/rt4ksr-x3.safetensors" "$plugin/Contents/Resources/"
  cp -c "$project_dir/build/quality-first/framegen.safetensors" "$plugin/Contents/Resources/"
fi
cp "$project_dir/../MLX-DLSS/LICENSE" "$plugin/Contents/Resources/MLX-DLSS-LICENSE"
cp "$project_dir/../MLX-DLSS/NOTICE" "$plugin/Contents/Resources/MLX-DLSS-NOTICE"
cp "$project_dir/licenses/RT4KSR-LICENSE" "$plugin/Contents/Resources/"
codesign --force --sign - "$plugin/Contents/Frameworks/mlx.metallib"
codesign --force --sign - "$plugin/Contents/Resources/mlx.metallib"
codesign --force --sign - "$plugin"
codesign --verify --deep --strict "$plugin"
echo "$plugin"
