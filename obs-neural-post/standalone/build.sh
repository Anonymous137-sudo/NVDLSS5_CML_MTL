#!/bin/bash
set -euo pipefail
project_dir="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "${NVDLSS_SKIP_RUNTIME_BUILD:-0}" != 1 ]]; then
 bash "$project_dir/nr/build-runtime.sh"
fi
app="$project_dir/build/standalone/NVDLSS5.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Frameworks" "$app/Contents/Resources/Models"
for lib in libnvidia_nr libnvgnx_dlssnr libnvgnx_dlss_mlx librtss; do cp "$project_dir/build/nr/$lib.dylib" "$app/Contents/Frameworks/"; done
cp "$project_dir/build/nr/libobs_dlss5_nr.dylib" "$app/Contents/Frameworks/libdlss5_native.dylib"
install_name_tool -id @rpath/libdlss5_native.dylib "$app/Contents/Frameworks/libdlss5_native.dylib"
install_name_tool -change @rpath/libobs_dlss5_nr.dylib @rpath/libdlss5_native.dylib "$app/Contents/Frameworks/libnvgnx_dlssnr.dylib"
cp "$project_dir/../MLX-DLSS/.build/release/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib" "$app/Contents/Frameworks/mlx.metallib"
ditto "$project_dir/build/nr/RifeMetal_RifeMetalCore.bundle" "$app/Contents/Resources/RifeMetal_RifeMetalCore.bundle"
cp "$project_dir/build/nr/rife-v4.25.rmw" "$app/Contents/Resources/"
xcrun clang -O2 -fobjc-arc -fmodules -Wall -Wextra -target arm64-apple-macos14.0 \
 "$project_dir/standalone/main.m" "$project_dir/standalone/NativeInput.m" "$project_dir/nr/MacDisplayCapture.m" \
 -L "$app/Contents/Frameworks" -lnvidia_nr -Wl,-rpath,@executable_path/../Frameworks \
 -framework AppKit -framework ScreenCaptureKit -framework AVFoundation -framework VideoToolbox \
 -framework CoreVideo -framework IOSurface -framework CoreMedia -framework CoreGraphics \
 -o "$app/Contents/MacOS/NVDLSS5"
cat > "$app/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleIdentifier</key><string>dev.nvdlss5.native</string>
<key>CFBundleName</key><string>NVDLSS5</string>
<key>CFBundleExecutable</key><string>NVDLSS5</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>0.2.0</string>
<key>CFBundleVersion</key><string>2</string>
<key>LSMinimumSystemVersion</key><string>14.0</string>
<key>LSUIElement</key><true/>
<key>NSHighResolutionCapable</key><true/>
<key>NSCameraUsageDescription</key><string>Capture your external HDMI video device for neural rendering.</string>
<key>NSScreenCaptureUsageDescription</key><string>Capture the selected display beneath the click-through neural screen filter.</string>
</dict></plist>
PLIST
for file in "$app/Contents/Frameworks/"*.dylib; do codesign --force --sign - "$file"; done
mkdir -p "$app/Contents/Resources/Licenses"
cp "$project_dir/../LICENSE" "$app/Contents/Resources/Licenses/GPL-3.0.txt"
cp "$project_dir/../dependencies/LICENSE" "$app/Contents/Resources/Licenses/MLX-DLSS.txt"
cp "$project_dir/../dependencies/NOTICE" "$app/Contents/Resources/Licenses/MLX-DLSS-NOTICE.txt"
cp "$project_dir/../dependencies/RIFE-LICENSE" "$app/Contents/Resources/Licenses/RIFE.txt"
cp "$project_dir/../dependencies/RIFE-THIRD-PARTY.md" "$app/Contents/Resources/Licenses/RIFE-THIRD-PARTY.md"
cp "$project_dir/../obs-dlss5-nr/LICENSE" "$app/Contents/Resources/Licenses/obs-dlss5-nr.txt"
cp "$project_dir/../obs-dlss5-nr/THIRD_PARTY_NOTICES.md" "$app/Contents/Resources/Licenses/obs-dlss5-nr-NOTICES.md"
codesign --force --sign - "$app/Contents/Frameworks/mlx.metallib"
codesign --force --sign - "$app"
codesign --verify --deep --strict "$app"
echo "$app"
