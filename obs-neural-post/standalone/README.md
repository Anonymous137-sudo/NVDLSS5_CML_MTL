# Native NVDLSS5 host

This is the primary application host. It does not load OBS, libobs, the OBS
frontend, scenes, encoders or the OBS compositor. The existing model, MFG,
frame-pacing and ProMotion libraries remain native dylibs.

Two input/output modes:

- **Mac screen filter** captures the selected display beneath a borderless,
  floating, click-through Metal output. ScreenCaptureKit excludes this app's
  windows so its output does not feed back into inference. Capture dimensions
  match the selected display aspect ratio within 1920x1080; the system cursor
  is not captured, avoiding a delayed second cursor.
- **External HDMI / ProMotion window** captures a selected 1080p NV12 video
  device through AVFoundation and presents through the existing native window.

The five NR quality modes, independent final/per-layer intensity branches,
frame-generation factors and advanced model parameters are retained. Controls
are native AppKit widgets; the JSON editor exposes the complete runtime settings.
The menu-bar **DLSS 5 → Stop filter / output** command remains available while
clicks pass through the screen output. A performance-overlay checkbox controls
metrics without requiring the filter to receive keyboard focus.

## Run

Open `NVDLSS5.app`. Select a mode and source, then **Start / apply**. Use the
**Models** button to select a directory containing compatible
`NeuralRendering.dlssmodel` and `NeuralRendering-Small.dlssmodel` packages.
Individual paths can also be edited in the advanced settings. RIFE Metal's
weights are bundled; extracted NVIDIA NR checkpoints are not distributed.
Screen capture needs macOS Screen Recording permission for the new app;
HDMI needs camera permission. Settings persist under
`~/Library/Application Support/NVDLSS5/settings.json`.

The full model remains expensive. Full-resolution Fidelity is not currently a
playable desktop filter; this host split does not imply a neural-speed fix.
Use the NR toggle for a source/geometry check, while keeping that check distinct
from neural quality and frame-generation acceptance.

## Build without OBS

From the repository root:

```sh
python3 dependencies/bootstrap.py
bash obs-neural-post/standalone/build.sh
```

Requires Apple Silicon and Xcode/Swift; OBS headers, CMake and an installed OBS
application are not required for this target. The output is
`obs-neural-post/build/standalone/NVDLSS5.app`. An existing RIFE checkout may be
supplied with `DLSS_RIFE_METAL_DIR`. `NVDLSS_SKIP_RUNTIME_BUILD=1` rebuilds just
the host/package when the native libraries are already current.

The native NR port library is packaged as `libdlss5_native.dylib`. This is the
same native model code, with its install name and dependent references changed;
it has no libobs dependency. All packaged Mach-O dependencies were checked.

## Validation boundary

The standalone host was launched with OBS closed and captured the built-in Mac
display at 1670x1080 full-range NV12 on this machine. It processed neural frames
with no reported capture drops or inference errors, and opened the full-display
output. A non-neural geometry check was also exercised. Capture colour tags
were stable BT.709. Model timing remained roughly 2-3 base FPS in Fidelity.

Click-through is implemented using `ignoresMouseEvents`; its physical-click test
was inconclusive because the host/test window closed before their presence was
confirmed at the click. The user subsequently requested both stay closed, so
final UI/cursor/metrics refinements were built but not relaunched. No HDMI run
was attempted in the standalone host while the console was off. These remain
explicit runtime checks; compilation and dependency validation do not pass them.

The development fixture under `tests/` is separate from the app, and is not
launched by the host. Local evidence is in `build/standalone-test/` (not shipped).
