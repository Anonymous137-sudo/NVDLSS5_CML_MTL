#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
app="$project_dir/build/CaptureAudit.app"
mkdir -p "$app/Contents/MacOS"
cat > "$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>CaptureAudit</string>
<key>CFBundleIdentifier</key><string>local.obs-neural-post.capture-audit</string>
<key>CFBundleName</key><string>PS5 Screen</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>NSCameraUsageDescription</key><string>Display and measure the connected PS5 HDMI capture feed.</string>
</dict></plist>
PLIST
if [[ "${1:-}" == "--neural" ]]; then
    shift
    (cd "$project_dir" && swift build -c release --product PS5NeuralAudit)
    cp "$project_dir/.build/release/PS5NeuralAudit" "$app/Contents/MacOS/CaptureAudit"
    codesign --force --sign - "$app"
    destination="$project_dir/build/ps5-neural-live"
    continuous=true
    for option in "$@"; do
        if [[ "$option" == --seconds=* ]]; then continuous=false; fi
        if [[ "$option" == --output-dir=* ]]; then destination="${option#--output-dir=}"; fi
    done
    if [[ "$continuous" == true ]]; then
        open "$app" --args "$destination" "--neural-resources=$project_dir/build/plugin/obs-neural-post.plugin/Contents/Resources" --live "$@"
    else
        open "$app" --args "$destination" "--neural-resources=$project_dir/build/plugin/obs-neural-post.plugin/Contents/Resources" "$@"
    fi
else
    if [[ ! -x "$app/Contents/MacOS/CaptureAudit" || "$project_dir/tools/CaptureAudit.swift" -nt "$app/Contents/MacOS/CaptureAudit" ]]; then
        xcrun swiftc -O -parse-as-library -target arm64-apple-macosx14.0 "$project_dir/tools/CaptureAudit.swift" -o "$app/Contents/MacOS/CaptureAudit"
    fi
    codesign --force --sign - "$app"
    if [[ "${1:-}" == "--audit" ]]; then
        open "$app" --args "$project_dir/build/ps5-preview-audit"
    else
        open "$app" --args "$project_dir/build/ps5-live" --live
    fi
fi
