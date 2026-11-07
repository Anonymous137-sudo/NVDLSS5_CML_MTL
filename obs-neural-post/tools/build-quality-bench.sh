#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
backend_dir="$project_dir/../MLX-DLSS"
(cd "$backend_dir" && swift build -c release --product mlxdlss)
products="$backend_dir/.build/release"
mkdir -p "$project_dir/build/quality-first"
cp "$products/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" "$project_dir/build/quality-first/mlx.metallib"
objects=()
for object_name in DLSSCore DLSSMLX MLX MLXNN Cmlx ComplexModule RealModule _NumericsShims Numerics; do
    objects+=("$products/$object_name.o")
done
for source_name in NativeTeacherBudget FrameGenBudget; do
    output_name=native-teacher-budget
    if [[ "$source_name" == FrameGenBudget ]]; then output_name=framegen-budget; fi
    xcrun swiftc -O -parse-as-library -I "$products" \
        -I "$backend_dir/.build/checkouts/mlx-swift/Source/Cmlx/include" \
        -I "$backend_dir/.build/checkouts/swift-numerics/Sources/_NumericsShims/include" \
        "$project_dir/tools/$source_name.swift" "${objects[@]}" \
        -framework Metal -framework Accelerate -framework CoreVideo -framework IOSurface -lc++ \
        -o "$project_dir/build/quality-first/$output_name"
done

xcrun swiftc -O -parse-as-library -I "$products" \
    -I "$backend_dir/.build/checkouts/mlx-swift/Source/Cmlx/include" \
    -I "$backend_dir/.build/checkouts/swift-numerics/Sources/_NumericsShims/include" \
    "$project_dir/tools/NativeQualityPipeline.swift" "$project_dir/Sources/NeuralPost/Model.swift" "${objects[@]}" \
    -framework Metal -framework Accelerate -framework CoreVideo -framework IOSurface -lc++ \
    -o "$project_dir/build/quality-first/native-quality-pipeline"
