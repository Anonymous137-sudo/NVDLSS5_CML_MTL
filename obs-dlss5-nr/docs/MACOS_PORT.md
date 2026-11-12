# macOS port and merge

The macOS implementation retains the obs-dlss5-nr effect filter and its source-size policy, worker queue, completed-frame cache, limiter, error reporting, history reset and retry controls. Its bridge runs the recovered NR checkpoint through MLX in FP16. This is a native reimplementation of the host contract. It does not execute NVIDIA's NGX DLL or NVIDIA driver code.

The merged Apple DLSS plugin registers this effect filter and also routes its existing HDMI source through the same native model implementation. The source's earlier ThreeModeEngine correction extraction, low-frequency history filter, perceptual correction transfer and display compositor are outside the active NR build. RIFE, MLX and Apple frame-generation implementations remain separate consumers of completed NR frames. Capture-guided interpolation remains disabled.

## Component mapping

| Upstream component | macOS implementation |
| --- | --- |
| OBS effect filter and processing dimensions | Shared src/dlss5-nr-filter.cpp and src/frame-policy.h |
| Smooth worker and newest pending frame | Shared upstream worker with per-filter bridge state |
| Low-latency inline mode | Shared upstream inline path, with synchronous native inference |
| D3D12 NGX feature 18 evaluation | src/macos/NativeNR.swift, recovered MLX NR graph, FP16 head |
| NVOFA current-to-previous motion | Existing MLX-DLSS NativeOpticalFlow, VideoToolbox or Vision, bidirectional confidence |
| Temporal model inputs and history | Existing MLX-DLSS device temporal feature processor and postprocessor |
| DXGI keyed-mutex shared textures | OBS CPU staging into owned IOSurfaces, then GPU inference |
| Windows caller shim and driver discovery | Native dylib ABI with packaged model and Metal-library discovery |
| Feature-creation retries | Five-second native model-load cooldown, immediate reset/retry on configuration change |
| Channel order | Explicit BGRA staging; an RGBA override swizzles once before native import |
| Style, tone, structure, skin, automask | Recovered model conditioning inputs |
| NVIDIA render presets 1–3 | No verified MLX mapping. Only recovered default graph is selectable |
| UI correction | Upstream keeps it off without UI-detection inputs |
| MFG and depth estimation | Not provided by this upstream filter. Existing source MFG is retained; depth remains later work |

The backend does not label model guidance or automatic skin conditioning as a game-engine depth map or a complete HUD mask.

## Model passes and resolution

The port exposes 71 single, 31 single, 31→71, 71→71 and 31→31. Every pass has its own history. Each pass consumes the preceding pass's RGB at the selected processing extent. The default processing budget is 1080p, preserving aspect ratio and avoiding enlargement of small sources.

The merged HDMI source retains the user's mutually exclusive intensity modes. Per-Layer intensity enters every model pass. Final intensity is applied once against the original source after the model output and history are resolved. The effect filter keeps the upstream single intensity control.

The older source's extra OKLab/HDR display controls, arbitrary motion multipliers, hand-entered noise index, rectangular HUD mask and custom SR path belonged to the retired compositor. Those controls are visibly disabled in the merged source. Saved values remain in OBS settings for migration, while the new model path does not consume them.

## Standalone build

The standalone filter works with an installed OBS application and matching OBS source headers. The local MLX-DLSS checkout includes the small final-intensity API extension. For a fresh checkout, apply patches/mlx-final-intensity.patch to MLX-DLSS before building. It does not require rebuilding OBS. Build the local MLX-DLSS backend and provide the model packages explicitly.

~~~sh
MLX_DLSS_DIR="/path/to/MLX-DLSS" \
OBS_HEADERS="/path/to/obs-studio-32.1.0" \
NR5_BUILD_DIR="/path/to/fresh-build-directory" \
NR5_MODEL71="/path/to/NeuralRendering.dlssmodel" \
NR5_MODEL31="/path/to/NeuralRendering-Small.dlssmodel" \
bash macos/build.sh
~~~

The result is obs-dlss5-nr.plugin containing the filter, libobs_dlss5_nr.dylib, the shared MLX runtime, Metal library, model packages and license notices. Building does not install the plugin or launch OBS. A fresh output directory prevents accidentally shipping an old package.

The Apple DLSS bundle builds the same filter source and native dylib through obs-neural-post/nr/build-plugin.sh. Install either the standalone filter or the merged bundle, since both register the same effect-filter ID.

## Validation boundary

Compilation, model inference, color/row-pitch checks, motion-direction tests and an isolated libobs renderer test can run with the OBS application stopped. The isolated test creates only a synthetic source and never opens HDMI or the user's scene collection.

These tests establish the implementation and API behavior. PS5 moving-scene quality, parity with NVIDIA's proprietary runtime, 1080p heavy-cascade performance, and the 30–45 fresh-FPS target require later validation. A successful native dispatch or enhanced-frame counter does not establish those outcomes.


### Native source transfer and timing

The HDMI source uses `nr5_process_surface` to pass its owned BGRA IOSurface to
MLX and retain the writer's output CVPixelBuffer. It does not copy full frames
through the row-pitch C ABI. The shared upstream effect filter retains the
row-pitch entry point for OBS's CPU staging buffers. Surface outputs have +1
ownership; callers must release them and must never overwrite a retained input.

`nr5_profile` reports import, optical flow, first pass, second pass, export,
history-reset flag, and C input/output copy wall times. Pass timings include
completed GPU work because the video backend evaluates before returning. These
are wall times, not hardware GPU counters. Runtime `stages` exposes them with
`nativeSurfaceBridge` and actual `nativeNRWidth`/`nativeNRHeight`.

`nativeProcessingHeight` remains a maximum extent: selecting 2160 does not
synthesize a 2160-line model input from a 1080-line capture. NR and MFG currently
share one serial worker; increasing MFG factor adds work to the next base-frame
interval. The 250 ms pair-discontinuity cutoff remains enforced. More phases do
not establish lower input latency or successful interpolation across rejected
pairs.
