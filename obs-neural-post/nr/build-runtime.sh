#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
backend_dir="$project_dir/../MLX-DLSS"
products="$backend_dir/.build/release"
out="$project_dir/build/nr"
(cd "$backend_dir" && swift build -c release --product mlxdlss)
mkdir -p "$out"

compiler=(xcrun swiftc -O -target arm64-apple-macos14.0 -swift-version 5
 -package-name DLSSNRPackage -parse-as-library -emit-library
 -Xlinker -rpath -Xlinker @loader_path)
imports=(-I "$products" -I "$out"
 -I "$backend_dir/.build/checkouts/mlx-swift/Source/Cmlx/include"
 -I "$backend_dir/.build/checkouts/swift-numerics/Sources/_NumericsShims/include")
objects=()
for name in DLSSCore DLSSMLX MLX MLXNN Cmlx ComplexModule RealModule _NumericsShims Numerics; do objects+=("$products/$name.o"); done

"${compiler[@]}" -module-name DLSSMLXBackend \
 -Xlinker -install_name -Xlinker @rpath/libnvgnx_dlss_mlx.dylib \
 "${objects[@]}" -framework Metal -framework Accelerate -framework Foundation -lc++ \
 -o "$out/libnvgnx_dlss_mlx.dylib"

"${compiler[@]}" -module-name DLSSRTSS -emit-module -emit-module-path "$out/DLSSRTSS.swiftmodule" \
 -Xlinker -install_name -Xlinker @rpath/librtss.dylib \
 "$project_dir/nr/NativeStatsOverlay.swift" -framework AppKit -framework IOKit -o "$out/librtss.dylib"

"${compiler[@]}" "${imports[@]}" -module-name DLSSNR -emit-module -emit-module-path "$out/DLSSNR.swiftmodule" \
 -Xlinker -install_name -Xlinker @rpath/libnvgnx_dlssnr.dylib \
 "$project_dir/nr/FrameDiagnostics.swift" "$project_dir/nr/CaptureGuides.swift" "$project_dir/nr/AppleFrameInterpolation.swift" "$project_dir/nr/NRConfiguration.swift" "$project_dir/nr/Runtime.swift" "$project_dir/nr/FrameInterpolation.swift" "$project_dir/nr/FrameSchedule.swift" \
 "$project_dir/tools/live_quality/ThreeModeEngine.swift" "$project_dir/nr/Presenter.swift" \
 "$project_dir/Sources/NeuralPost/Model.swift" -L "$out" -lnvgnx_dlss_mlx -lrtss \
 -framework Metal -framework CoreVideo -framework IOSurface -framework AppKit \
 -framework ImageIO -framework VideoToolbox -framework CoreMedia -framework MetalPerformanceShaders -framework QuartzCore \
 -o "$out/libnvgnx_dlssnr.dylib"

"${compiler[@]}" "${imports[@]}" -module-name NvidiaNRController \
 -Xlinker -install_name -Xlinker @rpath/libnvidia_nr.dylib \
 "$project_dir/nr/NRController.swift" -L "$out" -lnvgnx_dlssnr \
 -framework AppKit -framework CoreVideo -o "$out/libnvidia_nr.dylib"
