#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$project_dir"
out=build/ane-full720
mkdir -p "$out"
# Keep the older converter's NumPy compatibility requirement isolated from the
# existing training environment. No model layers or processing pixels are cut.
if [[ ! -d "$out/python-deps/numpy" ]]; then
    .venv/bin/pip install --target "$out/python-deps" 'numpy==1.26.4'
fi
if [[ ! -d "$out/NeuralRendering720Spatial.mlpackage" ]]; then
    PYTHONPATH="$out/python-deps" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
        .venv/bin/python -u tools/ane_full_model.py --no-chunks \
        --export "$out/NeuralRendering720Spatial.mlpackage"
fi
xcrun swiftc -O -parse-as-library tools/FullModelANEAudit.swift -o "$out/full-model-ane-audit"
.venv/bin/python tools/run_ane_audit.py "$out/full-model-ane-audit" \
    "$out/NeuralRendering720Spatial.mlpackage" build/quality-first/features.f32 \
    "$out/spatial-audit.json" --seconds 300
