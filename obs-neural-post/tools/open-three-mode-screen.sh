#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
mode="${1:-choose}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$mode" == choose ]]; then
    choice=$(osascript <<'APPLESCRIPT'
set selection to choose from list {"71-block HT: single pass", "31-block Small: single pass", "31-block Small: double pass"} with title "PS5 Neural - Three Modes" with prompt "720p appearance -> learned 4K reconstruction. All modes use MLX/Metal FP16; frame generation is off." default items {"31-block Small: double pass"} OK button name "Launch" cancel button name "Cancel"
if selection is false then return "cancel"
return item 1 of selection
APPLESCRIPT
)
    case "$choice" in
        "71-block HT: single pass") mode=71 ;;
        "31-block Small: single pass") mode=31 ;;
        "31-block Small: double pass") mode=31x2 ;;
        *) exit 0 ;;
    esac
fi
case "$mode" in 71|31|31x2|31then71-480|71x2-480) ;; *) echo "Choose 71, 31, 31x2, 31then71-480 or 71x2-480";exit 2;; esac
app="$project_dir/build/ThreeModeCapture.app"
if [[ "$mode" == *-480 ]]; then app="$project_dir/build/Hybrid480Capture.app";fi
ht="$project_dir/build/teacher/NeuralRendering.dlssmodel"
small="$project_dir/build/three-mode/NeuralRendering-Small.dlssmodel"
[[ -x "$app/Contents/MacOS/CaptureAudit" && -d "$ht" ]] || { echo "Prebuilt experiment app or71-block model is missing";exit 1; }
if [[ "$mode" != 71 && "$mode" != 71x2-480 && ! -d "$small" ]]; then echo "The genuine31-block checkpoint has not been imported";exit 1;fi
run_dir="$project_dir/build/three-mode-live-$mode-$(date +%Y%m%d-%H%M%S)"
duration=--live
extras=()
for arg in "$@"; do
    case "$arg" in
        --output=*) run_dir="${arg#--output=}" ;;
        --seconds=*) duration="$arg" ;;
        --compare-every=*|--record-pairs=*) extras+=("$arg") ;;
        *) echo "Unknown argument: $arg";exit 2 ;;
    esac
done
# These paths identify only this project's experimental viewers. A mode switch
# replaces its capture owner instead of opening a second HDMI session.
while IFS= read -r pid; do [[ -z "$pid" ]] || kill -TERM "$pid" 2>/dev/null || true; done < <(pgrep -f "^$project_dir/build/(ThreeModeCapture|Hybrid480Capture|StageSplitCapture[^/]*)[.]app/Contents/MacOS/CaptureAudit " || true)
mkdir -p "$run_dir"
open -n "$app" --args "$run_dir" "--experiment-mode=$mode" \
    "--model-71=$ht" "--model-31=$small" "--full-model=$ht" \
    "--neural-resources=$project_dir/build/plugin/obs-neural-post.plugin/Contents/Resources" \
    "$duration" "${extras[@]}"
echo "$run_dir"
