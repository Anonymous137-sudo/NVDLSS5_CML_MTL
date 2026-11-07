# NVDLSS5_CML_MTL

An experimental Apple Silicon OBS plugin for screen-space neural post-processing of a console HDMI feed. It captures 720p video through AVFoundation, runs neural appearance processing at 480p, and resolves the output to 1080p. The standalone plugin runs in stock OBS. The project also includes earlier Core ML/ANE, Metal and MLX research tools. The current release runs appearance inference on MLX/Metal and uses MPS for scaling.

| Mode | Appearance passes |
| --- | --- |
| Fidelity | 71-block → 71-block |
| Balanced | 31-block → 71-block |
| Performance | Single 31-block pass |

The DLSS 5 controls include temporal NR, a configurable HUD region, and mutually exclusive Final and Per-Layer intensity modes. Optional frame generation uses MLX or Apple VideoToolbox, with a cap of 6×. The native preview supports ProMotion and an RTSS-style overlay that separates fresh frames, generated frames, and display refresh.

## Install and configure

1. Download the Apple Silicon ZIP from [Releases](../../releases), extract it, and quit OBS.
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
| `obs-dlss-nr.plugin` | OBS source, HDMI capture, settings, and delivery of finished frames |
| `libnvidia_nr.dylib` | Versioned C controller for settings, frame lifetime, telemetry, and preview |
| `libnvgnx_dlssnr.dylib` | Appearance pipeline, MFG orchestration, frame pacing, and native preview |
| `libnvgnx_dlss_mlx.dylib` | MLX inference and model execution |
| `librtss.dylib` | RTSS-style metrics overlay and telemetry |

The binary names are local compatibility names. This is an independent project with no NVIDIA, Apple, OBS Project, or RTSS affiliation.

## Build from source

Use Xcode with Swift 6.2 or later, CMake 3.28 or later, Ninja, and installed Apple Silicon OBS. From the repository root, prepare the pinned backend:

```sh
git clone https://github.com/iamwavecut/MLX-DLSS.git MLX-DLSS
git -C MLX-DLSS checkout --detach 7870371e8bf6c03967213f1a5d40db6c5a3bee38
git -C MLX-DLSS apply ../dependencies/mlx-dlss.patch
```

Extract the [OBS 32.1.0 release source](https://github.com/obsproject/obs-studio/releases/tag/32.1.0) under `obs-neural-post/.deps/obs-studio-32.1.0` and [SIMDe 0.8.2](https://github.com/simd-everywhere/simde/releases/tag/v0.8.2) under `obs-neural-post/.deps/simde-0.8.2`. The OBS source tree supplies headers. The build links against the installed app's frameworks.

```sh
bash obs-neural-post/nr/build-plugin.sh
```

The signed bundle is staged under `obs-neural-post/build/stock-obs/`. Use `DLSS_PLUGIN_OUTPUT` for a different staging location. For a local-only build, `DLSS_BUNDLE_LOCAL_MODELS=1` also packages the model paths listed in `nr/package-plugin.sh`.

## Validation and limitations

The original full-model bundle was exercised on a live PS5 feed in stock OBS 32.1.0. Fidelity at MLX 6× showed roughly 8 fresh and 38.5 generated frames per second in the 120 Hz preview. The 30–45 fresh FPS target remains unmet. Frame generation can show appearance discontinuities during camera motion. HDMI input supplies screen-space pixels, with motion estimated from those pixels.

## License

Project source is GPL-3.0. MLX-DLSS and its local patches retain Apache-2.0 terms. See [LICENSE](LICENSE), [MLX-DLSS notices](dependencies/NOTICE), and the component notices in the release archive. NVIDIA binaries and extracted checkpoints are outside the GPL project license and the release archive.
