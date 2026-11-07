#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
backend_dir="$project_dir/../MLX-DLSS"
products="$backend_dir/.build/release"
(cd "$backend_dir" && swift build -c release --product mlxdlss)
objects=()
for name in DLSSCore DLSSMLX MLX MLXNN Cmlx ComplexModule RealModule _NumericsShims Numerics; do objects+=("$products/$name.o"); done
xcrun swiftc -O -swift-version 5 -D QUALITY_LIVE -parse-as-library -I "$products" \
 -I "$backend_dir/.build/checkouts/mlx-swift/Source/Cmlx/include" \
 -I "$backend_dir/.build/checkouts/swift-numerics/Sources/_NumericsShims/include" \
 "$project_dir/nr/NRConfiguration.swift" "$project_dir/tools/live_quality/QualityWorker.swift" "$project_dir/tools/live_quality/StageIsland.swift" "$project_dir/tools/live_quality/PairedVideo.swift" "$project_dir/tools/live_quality/ThreeModeEngine.swift" "${LIVE_CAPTURE_SOURCE:-$project_dir/tools/live_quality/LiveCapture.swift}" \
 "${LIVE_PRESENTER_SOURCE:-$project_dir/tools/live_quality/Presenter.swift}" "$project_dir/Sources/NeuralPost/Model.swift" "${objects[@]}" \
 -framework Metal -framework Accelerate -framework CoreVideo -framework IOSurface -framework AVFoundation \
 -framework MetalPerformanceShaders -framework CoreML -framework AppKit -lc++ -o "${LIVE_OUTPUT:-$project_dir/build/quality-first/LiveQualityCapture}"
