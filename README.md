# NVDLSS5_CML_MTL

Experimental Apple Silicon screen-space neural rendering with Metal, MLX and native capture. The current source includes the obs-dlss5-nr macOS model port, five NR modes, RIFE/MLX/Apple interpolation, bounded asynchronous NR/MFG workers, and HDMI or Mac display capture. A native ProMotion presenter reports fresh base frames, generated frames and display refresh separately.

The current capture path accepts 1080p NV12 SDR. Processing resolution is configurable, including a 2160p maximum; that maximum does not enlarge a 1080p source. Full-resolution Fidelity remains approximately 2–3 base FPS on the tested M4 Pro. Playable full-resolution throughput and temporal-quality acceptance are unresolved.

Direct ANEForge/IOSurface and mixed-precision experiments are in `obs-neural-post/tools/aneforge`. They are explicitly experimental: neither the rejected quantized candidates nor the slower hybrid partitions are enabled in live NR. See that directory's README for measured numerical gates and costs.

The primary host is now **NVDLSS5.app**, with two modes: **Mac screen filter** and **external HDMI / ProMotion window**. It uses native AppKit, ScreenCaptureKit, AVFoundation and Metal without loading OBS or its compositor. See the [native host guide](obs-neural-post/standalone/README.md). The previous OBS plugin remains as an optional legacy integration.

| Mode | Appearance passes |
| --- | --- |
| Fidelity | 71-block → 71-block |
| Balanced | 31-block → 71-block |
| Performance | Single 31-block pass |
| 71 Single | Single 71-block pass |
| 31 Double | 31-block → 31-block |

The DLSS 5 controls include temporal NR, a configurable HUD region, and mutually exclusive Final and Per-Layer intensity modes. Optional frame generation uses RIFE Metal, MLX or Apple VideoToolbox, with a cap of 6×. The native preview supports ProMotion and an RTSS-style overlay that separates fresh frames, generated frames, and display refresh.

## Native application

Open the native application from the source build or a matching release. Choose **Mac screen filter** or **External HDMI · ProMotion window**, select a source and use **Start / apply**. Use **Models** to configure local checkpoints. The menu-bar Stop command removes the filter and stops capture.

The Mac filter is a capture-and-present overlay, not a WindowServer injection. Its output and controls are excluded from capture; pointer events pass through. Full-resolution neural latency remains a major limitation. Mac capture was exercised, but click-through verification and standalone HDMI acceptance are still open. No new live run was made after the user requested that the apps remain closed.

```sh
python3 dependencies/bootstrap.py
bash obs-neural-post/standalone/build.sh
```

This target needs Xcode/Swift and Apple Silicon, **not an OBS installation or OBS headers**. Output: `obs-neural-post/build/standalone/NVDLSS5.app`.

## Optional legacy OBS plugin

1. Download the legacy OBS v0.1.0 plugin ZIP from [Releases](https://github.com/Anonymous137-sudo/NVDLSS5_CML_MTL/releases/tag/v0.1.0), extract it, and quit OBS.
2. Copy `obs-dlss-nr.plugin` into `~/Library/Application Support/obs-studio/plugins/` as a single bundle. The preview was tested on stock OBS 32.1.0 on Apple Silicon. Apple optical flow requires macOS 15.4 or later and supported hardware.
3. Open OBS, create or select a scene, then choose **Tools → DLSS 5**. In **Live tools**, select the HDMI capture card and grant camera access when macOS prompts. Use a 1920×1080 OBS canvas for this pipeline.
4. Open **Live tools → Original OBS source properties**. Point **Model files** and the frame-generation weights field at your local compatible assets. The table below lists the expected formats.
5. Open the native preview from **Live tools**. Press **F8** to toggle the performance overlay.

The release ZIP ships the executable components and Metal kernels. Model checkpoints are supplied locally.

| Asset | Needed for |
| --- | --- |
| `NeuralRendering.dlssmodel` | 71-block Fidelity and Balanced passes |
| `NeuralRendering-Small.dlssmodel` | 31-block Balanced and Performance passes |
| `rt4ksr-x3.safetensors` | The runtime's RT4KSR reconstruction model |
| `framegen.safetensors` | MLX frame generation |

See [MLX-DLSS model preparation](https://github.com/iamwavecut/MLX-DLSS#weights) for compatible NR and FG formats. The [RT4KSR project](https://github.com/eduardzamfir/RT4KSR) provides the x3 checkpoint. Convert it with `python3 obs-neural-post/tools/convert_rt4ksr.py rt4ksr_x3.pth rt4ksr-x3.safetensors` in an environment with PyTorch and SafeTensors.

## Components

| Binary | Responsibility |
| --- | --- |
| `NVDLSS5.app` | Native capture, screen-filter/HDMI window controls; no OBS core |
| `libdlss5_native.dylib` | Native recovered NR model port |
| `obs-dlss-nr.plugin` | OBS source, HDMI capture, settings, and delivery of finished frames |
| `libnvidia_nr.dylib` | Versioned C controller for settings, frame lifetime, telemetry, and preview |
| `libnvgnx_dlssnr.dylib` | Appearance pipeline, MFG orchestration, frame pacing, and native preview |
| `libnvgnx_dlss_mlx.dylib` | MLX inference and model execution |
| `librtss.dylib` | RTSS-style metrics overlay and telemetry |

The binary names are local compatibility names. This is an independent project with no NVIDIA, Apple, OBS Project, or RTSS affiliation.

## Optional OBS plugin build

Use Xcode with Swift 6.2 or later, CMake 3.28 or later, Ninja, and installed Apple Silicon OBS. From the repository root, prepare the pinned backend:

```sh
git clone https://github.com/iamwavecut/MLX-DLSS.git MLX-DLSS
git -C MLX-DLSS checkout --detach "$(cat dependencies/MLX-DLSS-REVISION)"
git -C MLX-DLSS apply ../dependencies/mlx-dlss.patch
```

Extract the [OBS 32.1.0 release source](https://github.com/obsproject/obs-studio/releases/tag/32.1.0) under `obs-neural-post/.deps/obs-studio-32.1.0` and [SIMDe 0.8.2](https://github.com/simd-everywhere/simde/releases/tag/v0.8.2) under `obs-neural-post/.deps/simde-0.8.2`. The OBS source tree supplies headers. The build links against the installed app's frameworks.

```sh
bash obs-neural-post/nr/build-plugin.sh
```

The signed bundle is staged under `obs-neural-post/build/stock-obs/`. Use `DLSS_PLUGIN_OUTPUT` for a different staging location. For a local-only build, `DLSS_BUNDLE_LOCAL_MODELS=1` also packages the model paths listed in `nr/package-plugin.sh`.

## Validation and limitations

The native host has been exercised on Mac screen capture; this does not establish a playable full-resolution rate or a completed click-through test. The original full-model bundle was exercised on a live PS5 feed in stock OBS 32.1.0. A historical lower-resolution Fidelity test at MLX 6× showed roughly 8 fresh and 38.5 generated frames per second in the 120 Hz preview. The 30–45 fresh FPS target remains unmet. Frame generation can show appearance discontinuities during camera motion. HDMI input supplies screen-space pixels, with motion estimated from those pixels.

## License

Project source is GPL-3.0. MLX-DLSS and its local patches retain Apache-2.0 terms. See [LICENSE](LICENSE), [MLX-DLSS notices](dependencies/NOTICE), and the component notices in the release archive. NVIDIA binaries and extracted checkpoints are outside the GPL project license and the release archive.

## Native port and dependencies

`obs-dlss5-nr/` contains the macOS model-runtime port and upstream notices. `dependencies/mlx-dlss.patch` records the current MLX-DLSS source changes against `dependencies/MLX-DLSS-REVISION`. Model weights and machine-local symlinks are not included. RIFE Metal and ANEForge are external dependencies; the latter is only needed for the experimental probes.
