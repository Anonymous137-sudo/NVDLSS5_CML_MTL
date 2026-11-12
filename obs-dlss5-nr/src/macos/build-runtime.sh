#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
backend="${MLX_DLSS_DIR:-$root/../MLX-DLSS}"
if [[ ! -f "$backend/Sources/DLSSMLX/MLXVideoFrameIntensity.swift" ]]; then
  echo "Apply the local final-intensity extension before building:" >&2
  echo "git -C '$backend' apply '$root/patches/mlx-final-intensity.patch'" >&2
  exit 1
fi
if [[ ! -f "$backend/Sources/DLSSMLX/MLXNeuralStageTrace.swift" ]]; then
  echo "Apply the bounded live-stage diagnostic extension before building:" >&2
  echo "git -C '$backend' apply '$root/patches/mlx-stage-trace.patch'" >&2
  exit 1
fi
products="$backend/.build/release"
out="${NR5_RUNTIME_OUTPUT:-$root/build/macos-runtime}"
(cd "$backend" && swift build -c release --product mlxdlss)
mkdir -p "$out"
if [[ ! -f "$out/libnvgnx_dlss_mlx.dylib" ]]; then
  objects=()
  for name in DLSSCore DLSSMLX MLX MLXNN Cmlx ComplexModule RealModule _NumericsShims Numerics; do
    objects+=("$products/$name.o")
  done
  xcrun swiftc -O -target arm64-apple-macos14.0 -emit-library -module-name DLSSMLXBackend \
    "${objects[@]}" -framework Metal -framework Accelerate -framework Foundation -lc++ \
    -Xlinker -install_name -Xlinker @rpath/libnvgnx_dlss_mlx.dylib \
    -o "$out/libnvgnx_dlss_mlx.dylib"
fi
xcrun swiftc -O -target arm64-apple-macos14.0 -swift-version 5 -parse-as-library -emit-library \
  -module-name ObsDLSS5Native -emit-module -emit-module-path "$out/ObsDLSS5Native.swiftmodule" \
  -I "$products" -I "$backend/.build/checkouts/mlx-swift/Source/Cmlx/include" \
  -I "$backend/.build/checkouts/swift-numerics/Sources/_NumericsShims/include" \
  "$root/src/macos/NativeNR.swift" "$products/DLSSMedia.o" -L "$out" -lnvgnx_dlss_mlx \
  -framework AVFoundation -framework VideoToolbox -framework CoreMedia -framework CoreImage -framework ImageIO -framework UniformTypeIdentifiers -framework Vision -framework CoreVideo -framework IOSurface -framework Foundation -framework Metal \
  -Xlinker -rpath -Xlinker @loader_path \
  -Xlinker -install_name -Xlinker @rpath/libobs_dlss5_nr.dylib \
  -o "$out/libobs_dlss5_nr.dylib"
