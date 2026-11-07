#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
app="$project_dir/build/StageSplitCaptureV3.app"
if [[ ! -x "$app/Contents/MacOS/CaptureAudit" ]]; then
    echo "The prebuilt stage-split viewer is missing: $app"
    exit 1
fi
# Reopening the running viewer must not create a second HDMI capture session.
if pgrep -f "^$app/Contents/MacOS/CaptureAudit " >/dev/null; then
    open -a "$app"
    exit 0
fi
run_dir="$project_dir/build/stage-split-live-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$run_dir"
open -n "$app" --args "$run_dir" \
    "--full-model=$project_dir/build/teacher/NeuralRendering.dlssmodel" \
    "--split-model=$project_dir/build/stage-split/Global720.mlpackage" --split-global \
    "--neural-resources=$project_dir/build/plugin/obs-neural-post.plugin/Contents/Resources" --live
