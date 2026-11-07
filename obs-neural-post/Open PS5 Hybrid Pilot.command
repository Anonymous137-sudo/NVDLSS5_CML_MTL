#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")" && pwd)"
model="$project_dir/build/teacher-distill/AppearancePilotV2.mlpackage"
if [[ ! -d "$model" ]]; then
    echo 'The experimental student model has not been built in this checkout.' >&2
    exit 1
fi
exec "$project_dir/tools/open-ps5-screen.sh" --neural "--ane-model=$model" "--output-dir=$project_dir/build/ps5-hybrid-pilot"
