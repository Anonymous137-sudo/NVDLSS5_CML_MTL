#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
backend_dir="$project_dir/../MLX-DLSS"
products="$backend_dir/.build/release"
rife_dir="${DLSS_RIFE_METAL_DIR:-$project_dir/../../../rife-metal}"
rife_products="$rife_dir/.build/release"
out="$project_dir/build/nr"
(cd "$backend_dir" && swift build -c release --product mlxdlss)
(cd "$rife_dir" && swift build -c release --target RifeMetal)
mkdir -p "$out"
ditto "$rife_products/RifeMetal_RifeMetalCore.bundle" "$out/RifeMetal_RifeMetalCore.bundle"
cp "$rife_dir/Sources/RifeMetal/Resources/rife-v4.25.rmw" "$rife_dir/LICENSE" "$rife_dir/THIRD-PARTY.md" "$out/"

compiler=(xcrun swiftc -O -target arm64-apple-macos14.0 -swift-version 5
 -package-name DLSSNRPackage -parse-as-library -emit-library
 -Xlinker -rpath -Xlinker @loader_path)
imports=(-I "$products" -I "$out"
 -I "$backend_dir/.build/checkouts/mlx-swift/Source/Cmlx/include"
 -I "$backend_dir/.build/checkouts/swift-numerics/Sources/_NumericsShims/include")
rife_core=("$rife_dir"/Sources/RifeMetalCore/*.swift)
xcrun swiftc -O -whole-module-optimization -parse-as-library -target arm64-apple-macos14.0 \
 -package-name RifeMetal -module-name RifeMetalCore -emit-object "${rife_core[@]}" "$project_dir/nr/RifeMetalResources.swift" \
 -o "$out/RifeMetalCore.o"
imports+=(-I "$rife_products")
objects=()
for name in DLSSCore DLSSMLX MLX MLXNN Cmlx ComplexModule RealModule _NumericsShims Numerics; do objects+=("$products/$name.o"); done

"${compiler[@]}" -module-name DLSSMLXBackend \
 -Xlinker -install_name -Xlinker @rpath/libnvgnx_dlss_mlx.dylib \
 "${objects[@]}" -framework Metal -framework Accelerate -framework Foundation -lc++ \
 -o "$out/libnvgnx_dlss_mlx.dylib"

upstream_dir="${DLSS_OBS_NR_DIR:-$project_dir/../obs-dlss5-nr}"
MLX_DLSS_DIR="$backend_dir" NR5_RUNTIME_OUTPUT="$out" bash "$upstream_dir/src/macos/build-runtime.sh"

"${compiler[@]}" -module-name DLSSRTSS -emit-module -emit-module-path "$out/DLSSRTSS.swiftmodule" \
 -Xlinker -install_name -Xlinker @rpath/librtss.dylib \
 "$project_dir/nr/NativeStatsOverlay.swift" -framework AppKit -framework IOKit -o "$out/librtss.dylib"

"${compiler[@]}" "${imports[@]}" -module-name DLSSNR -emit-module -emit-module-path "$out/DLSSNR.swiftmodule" \
 -Xlinker -install_name -Xlinker @rpath/libnvgnx_dlssnr.dylib \
 "$out/RifeMetalCore.o" "$rife_products/RifeMetal.o" "$project_dir/nr/RifeFrameInterpolation.swift" "$project_dir/nr/FrameDiagnostics.swift" "$project_dir/nr/CaptureGuides.swift" "$project_dir/nr/AppleFrameInterpolation.swift" "$project_dir/nr/NRConfiguration.swift" "$project_dir/nr/Runtime.swift" "$project_dir/nr/FrameInterpolation.swift" "$project_dir/nr/FrameSchedule.swift" \
 "$project_dir/nr/UpstreamNREngine.swift" "$project_dir/nr/Presenter.swift" \
 "$project_dir/Sources/NeuralPost/Model.swift" -L "$out" -lnvgnx_dlss_mlx -lrtss -lobs_dlss5_nr \
 -framework Metal -framework MetalPerformanceShadersGraph -framework CoreVideo -framework IOSurface -framework AppKit \
 -framework ImageIO -framework VideoToolbox -framework CoreMedia -framework MetalPerformanceShaders -framework QuartzCore \
 -o "$out/libnvgnx_dlssnr.dylib"

"${compiler[@]}" "${imports[@]}" -module-name NvidiaNRController \
 -Xlinker -install_name -Xlinker @rpath/libnvidia_nr.dylib \
 "$project_dir/nr/NRController.swift" -L "$out" -lnvgnx_dlssnr \
 -framework AppKit -framework CoreVideo -o "$out/libnvidia_nr.dylib"
