# Direct ANE / Metal NR stage experiments

This directory adds an experimental, native ANEForge dispatch bridge and
checkpoint-stage exporters/probes. It does not install or enable an ANE backend
in OBS. The NR/MFG worker split is a separate runtime change, validated below.

## Implemented

- ANEForge's existing e5rt compiler/dispatcher is included at build time from
  the user-provided checkout; its source is not modified.
- `ANEForgeBridge.h` exports native compile/execute/destroy and IOSurface binding.
  Offline graph lowering uses Python; compiled-program execution does not.
- `SurfaceBridge.mm` wraps caller-created FP16 IOSurfaces using the current
  Espresso `e5rt_buffer_object_create_from_iosurface` API. It verifies the
  returned IOSurface ID and data address before binding a port.
- Metal textures view these same surfaces as R16Float `[S,C]`, matching the
  ANEForge graph's `[1,C,1,S]` port. S must be a multiple of 32 here, making
  each FP16 row a multiple of 64 bytes. This is one tested tensor contract,
  not a universal claim about all ANE graphs or all image dimensions.
- The test ring has three independent program/surface slots. GPU completion
  handlers submit ANE work to one serial owner queue; ANE completion precedes
  the dependent GPU read. Slots are not overwritten until GPU validation ends.
  This does not assume multiple simultaneous ANE evaluations or a Metal-event
  object directly consumable by e5rt.
- Tensor inputs and outputs are not copied by the bridge. The test inspects a
  GPU-generated validation buffer on CPU. Projection/FFN oracle readbacks are
  explicitly outside their measured warm execution loops. This does not prove
  that the private framework performs no internal transfers.
- Original block31 projection and FFN weights are loaded from the user's
  logical.safetensors, never bundled. Selected projections use an input x16 /
  output /16 scale to reduce measured cross-device numerical error. Logical
  weights and the mathematical projection are preserved.
- Recovered E4M3 publication uses explicit ties-to-even arithmetic. Native ANE
  `round` did not match the recovered publication at all tie cases on this Mac.
  Both the slower piecewise version and the compact log/power version remain
  available for reproducible comparison.

## Measured 2026-11-11

Apple M4 Pro, macOS 27.0.1 (26A434), ANEForge revision
`eba339e590d64c01e222dc72ea2e0c9e8988a4b8`.

The native-only three-slot affine test completed 120 distinct frame patterns,
with zero numerical error and verified surface identity. It used ANE mask 4,
no Core ML, and no ANEForge Python import in the native API validation driver.
This is an interoperability test, not moving-gameplay NR inference.

At `[1,1024,1,640]`, 50 warm predictions per case:

| Original checkpoint operation | ANE median ms | MLX FP16 median ms | Relative L2 vs MLX |
|---|---:|---:|---:|
| Block31 attention output projection, scaled | 0.169 | 0.372 | 0.001076 |
| Block31 QKV projection, scaled | 0.324 | 0.796 | 0.000571 |
| Block31 expansion projection, scaled | 0.430 | 0.942 | 0.000570 |
| Block31 FFN + gate + E4M3 + residual, compact publication | 4.621 | 3.502 | 0.003191 |

Projection surface outputs and the copied-input ANE outputs matched exactly in
these checks. The unscaled output projection failed the initial 1% relative-L2
gate (1.919% error). No tolerance was relaxed to pass it. The scaled version
passed; that result still needs validation on captured internal activations.
The FFN gate is 1% relative L2, not exact whole-stage equality.

The compact publication probe covered all 63,488 finite FP16 inputs and found
zero numerical mismatches against the recovered saturation/RNE reference.
There were 5,120 signed-zero differences, so this is not bitwise parity.
NaN/Inf behavior and different ANE generations are outside that result.

Compilation, input generation, oracle readbacks and model loading are excluded
from the warm operator timings. These are **not end-to-end 1080p frame times**.
The 640-token case is an internal feature workload, not a 640-pixel game input.

The complete FFN remains slower than its MLX reference. It is not promoted as a
performance win. Shared surfaces remove external copies; they do not remove the
cost of the required gate, E4M3 publication or hardware scheduling.

## Reproduce

Use the existing project Python environment that has numpy and mlx. Set paths
for your checkout, then:

```sh
python tools/aneforge/build.py --aneforge /path/to/ANEForge \
  --output build/aneforge-probe/libnvgnx_dlss_ane.dylib
python tools/aneforge/probe.py --aneforge /path/to/ANEForge \
  --library build/aneforge-probe/libnvgnx_dlss_ane.dylib \
  --weights /path/to/logical.safetensors --projection-scale 16 \
  --output build/aneforge-probe/scaled16
python tools/aneforge/publication_probe.py --aneforge /path/to/ANEForge \
  --library build/aneforge-probe/libnvgnx_dlss_ane.dylib \
  --output build/aneforge-probe
python tools/aneforge/ffn_probe.py --aneforge /path/to/ANEForge \
  --library build/aneforge-probe/libnvgnx_dlss_ane.dylib \
  --weights /path/to/logical.safetensors --output build/aneforge-probe/compact
```

Results are JSON files under the selected build directories. The native library
uses current private e5rt signatures and fails binding if the required symbols
or identity checks fail. It does not switch silently to CPU/GPU inference.
Metal compilation for this experimental bridge requires macOS 15+ SDK APIs.

## Remaining integration gate

A few isolated projections running faster does not establish an efficient
coarse model split. The complete NR backend must retain frame-specific skips,
publication points, model controls and temporal state; pack actual features
on GPU; and carry its exact capture timestamp through each ring slot. It must
then pass paired captured-feature parity, moving HDMI quality, resource-lifetime
and end-to-end timing checks under simultaneous FG and OBS presentation load.
No replacement softmax, removed quantization point, 480p substitution, or higher
MFG gap threshold is introduced by these experiments.

ANEForge is MIT licensed. See ../../licenses/ANEForge-LICENSE. The bridge and
scripts here are experimental additions to the surrounding project.

## Live captured-feature validation, 2026-11-11

The normal OBS plugin was run with the original Fidelity 71->71 model, the
user's 2160p processing limit, and the actual 1920x1080 HDMI feed. Capture-guided
appearance remained disabled. The network padded to 1920x1088; the observed
bottleneck input was NHWC `[1,20,32,1024]`, confirming 640 internal tokens.

The new optional MLX stage tracer captures a maximum of eight heads per process
(default four, after six warm-up heads). Every dump carries a capture timestamp
and pass-number label. It explicitly evaluates stage boundaries and exports only
the bottleneck feature tensor, not a full-resolution activation history. Those
synchronized stage walls must not be presented as uninstrumented FPS or GPU
hardware counters. With the environment unset the normal lazy graph is retained.

Six heads from three distinct live captures, including both Fidelity passes,
produced these median wall times per pass:

| Stage | Median ms |
|---|---:|
| Feature materialization | 2.06 |
| Encoder | 60.00 |
| Split encoder | 12.38 |
| Eight-block global attention stage | 21.30 |
| Decoder bridge | 0.35 |
| Decoder | 59.17 |
| Post block | 10.84 |

The encoder and decoder account for most of the model cost. Even eliminating
all 21 ms of global work in these samples would leave about 290 ms of model
work for two passes, before MFG. Bottleneck-only offload therefore cannot meet
the current 250 ms frame-pair window by itself. This is an estimate from
synchronized diagnostic samples, not a hardware-independent lower bound.

All six real bottleneck tensors were replayed through the ANE-only block31 FFN
and the MLX reference with the recovered gate/publication. All outputs were
finite; relative L2 error ranged from 0.001053 to 0.001136 (0.1053-0.1136%).
They passed the same 1% feature-parity gate used for the initial branch probe.
This is captured-feature parity, not whole-frame perceptual or temporal parity.
Replay uses one explicit initialization copy from the recorded tensor; it is
not a production GPU->ANE transfer measurement. OBS stayed active while replay
ran and the controls changed, so these replay timings do not establish an
idle-GPU speed comparison.

A separate functional regression compared traced and untraced native inference
on identical 320x320 inputs: single71 and double71 both had zero changed output
bytes. This checks instrumentation behavior, not live performance acceptance.
The gameplay recording contains an NR enable/disable change and is not a
controlled temporal-quality comparison.

Artifacts: `build/aneforge-probe/latest-live-stage.txt` identifies the saved
capture directory with `report.json`, six labelled feature dumps, stage reports,
replay results, settings and a short native-preview recording. The trace parity
result is `build/aneforge-probe/trace-parity/trace-parity.json`.

The extension is shipped in `obs-dlss5-nr/patches/mlx-stage-trace.patch` relative
to the macOS project root. The native bridge's build checks for this extension.
Opt-in environment variables:

```sh
DLSS_NR_STAGE_TRACE=/path/to/bounded-output
DLSS_NR_STAGE_TRACE_FRAMES=6
DLSS_NR_STAGE_TRACE_SKIP=6
```

`ffn_probe.py --feature /path/to/global-input.f16` loads the actual NHWC tensor
and shape sidecar. `--program-dir` reuses one compiled stage across captures.
`trace_parity.py --plugin /path/to/obs-dlss-nr.plugin --output /path/to/report`
reproduces the native output-byte regression.

No complete ANE stage is selected in the production NR renderer yet. The next
performance candidate needs to include suitable encoder/decoder work; the full
1080p throughput, temporal-quality and frame-generation gates remain open.

## Complete 32-channel window-block port

`nr_ops.window32` now expresses a complete recovered block1 in one ANEForge
program: original expansion/projection weights, quadratic gate, E4M3 publication,
residuals, QKV projection, fragment-tree cosine normalization, recovered bias
swizzle, bit-affine softmax, attention, output projection and final publication.
It does not use standard softmax or native SDPA in place of the recovered math.
Independent prepartitioned windows can occupy H or N. Image padding/partition
remains a separate boundary, with actual window origins recorded by the tracer.

`window_probe.py` validates the graph against the independent recovered PyTorch
reference and, when supplied, the actual captured fused-Metal output. A failed
1% relative-L2 output gate exits with status 2 and is never selected in OBS.
`window_diagnostics.py` exports numerical intermediates in one multi-output
ANE-only diagnostic program to locate disagreement.

The live tracer now captures eight interior block1 input/output windows for each
admitted head, plus the full block's synchronized GPU wall time. It does not dump
or retain the whole 32-channel image on CPU. The input/expected tensors use NWTC
`[1,8,64,32]`. On the current 1080p feed, the actual block image is
`[1,544,960,32]` and block1 has zero window origin. The two captured GPU calls
took 2.741 and 2.837 ms.

The direct ANE candidate was rejected on both gates:

- Captured-output relative L2 was 0.014059 and 0.013610 (1.406% and 1.361%),
  above the unchanged 1% gate. Outputs were finite. These are paired actual
  fused-GPU outputs from both Fidelity passes of one captured source frame.
- Eight seeded windows took about 0.255 ms; 256 windows took 7.629 ms. An
  alternative N-batch layout gave 7.024 ms for 256, with matching sampled error.
- A shifted/padded frame-sized 8,349-window volume took 396.469 ms, excluding
  initialization and host copies. It checked 24 distributed output windows and
  scanned all outputs for finiteness. It is not the identical full image as the
  unshifted live block1, which has 8,160 windows; the scale result still fails
  the usable latency gate by a large margin.

The small seeded probe passed 1% error, but the real captured tensors did not.
That distinction is preserved in the reports. No threshold was relaxed and the
renderer was not switched to the failed candidate.

Numerical intermediate checks on the first captured tile group found relative
L2 errors of about 0.011% after expansion, 0.069% at the FFN residual, 0.104%
after QKV, 0.892%/0.779% at published Q/K, 1.298% at probabilities and 1.411%
at attended values. These comparisons are against the recovered independent
reference, while the final paired gate uses actual fused-GPU output.

Separately compiled 256-window prefixes measured 0.184 ms through expansion,
2.781 ms through gate/publication, 4.192 ms through attention scores and
5.818 ms through probability publication; the complete prefix was 7.583 ms.
These help target the emulated nonlinear/publication graph. Prefixes have
different compiler fusion plans; they must not be subtracted or summed as exact
per-operation hardware time. They ran with OBS active, not in the idle benchmark
conditions of the frame-sized feasibility test.

The trace-only code passed a fresh output regression: ordinary versus traced
single71 and double71 inference both had zero differing output bytes on the
functional fixture. Trace environments still admit only a bounded set of heads.

`build/aneforge-probe/latest-window-stage.txt` identifies the saved report,
captured windows, paired outputs and numerical stage attribution. The rejected
candidate remains in tools, not in the production NR path. At the time of the window tests the async NR/MFG worker change was separate
and not installed. Its subsequent live validation is recorded below.
The playable full-resolution throughput and temporal-quality targets remain open.


## INT8 activation and weight experiment

`precision_probe.py` compares FP16, weight-only INT8 (W8A16), and explicit
activation-QDQ plus INT8 weights through the direct ANE-only backend. The local
MIL lowering is in `quantized_graphs.py`; ANEForge itself is unchanged. No Core
ML model execution is invoked. ANEForge's existing `int8=True` compresses constant
weights; it must not be described as W8A8 activation computation by itself.

Apple documents an M4 ANE int8-int8 path for W8A8:
https://apple.github.io/coremltools/docs-guides/source/opt-overview.html
That hardware capability does not prove which packed arithmetic mode our private
compiler selects. Reports explicitly retain `nativePackedINT8Verified: false`.
QDQ in MIL and a speed difference establish a graph experiment, not that proof.

The probe uses the original block31 expansion matrix and captured global inputs
from the live 1080p Fidelity run. Four heads from the first two source frames
calibrate; both heads from the third frame are held out. This is a very small
indoor calibration set, not broad gameplay coverage. It tests the projection
only; attention normalization, E4M3 publication, residuals and colour processing
are not replaced. Initial host input loading and output reads are outside the
warm execute timing, so these are not complete interop or frame-latency numbers.

Latest local warm medians (50 executions after five warmups):

| Graph | Median ms | Held-out relative L2 | 1% operator gate |
| --- | ---: | ---: | --- |
| FP16 | 0.414 | 0.00168% | Pass |
| W8A16 | 0.371 | 0.499-0.504% | Pass |
| INT8 weights + activation QDQ, max calibration | 0.292 | 1.267-1.284% | Fail |
| INT8 weights + activation QDQ, MSE calibration | 0.275 | 1.100-1.161% | Fail |

The MSE scale is selected from 21 clipping candidates using calibration inputs
only, never the held-out tensors. It clips about 0.009-0.012% of held-out
activations. The report and scale search are under
`build/aneforge-probe/precision-calibration/`. The script records rejected modes
alongside accepted modes and exits successfully when the comparison completed;
`allPassed` in each row is the numerical gate. Passing this operator gate does
not establish image-quality or temporal parity after repeated layers.

No quantized candidate was installed in OBS. The roughly 1.51x speed difference
for this projection is not a full-model speedup, nor an explanation for all of
the previously measured window-block cost. The window implementation's nonlinear
publication graph remains a separate bottleneck. FP8 Metal execution has not
been benchmarked by this experiment.


## Continuous asynchronous scheduling

The surface ring now refills each slot after its dependent GPU verification
completes. It no longer waits for all three slots at a batch barrier. Metal
completion handlers dispatch to a serial ANE owner; the same slot is never
written or reused until GPU consumption ends. This permits independent jobs
from different slots to overlap without assuming concurrent ANE evaluations.
There are at most three active slot chains. The only group wait is in the
standalone validation caller, after launch; callbacks do not wait on GPU work.

`async_ring_probe.py` executes the native API without importing ANEForge Python.
The previously exported affine graph remains a functional interop fixture, not
a substitute NR workload. On this M4 Pro, 600 distinct patterns per case gave:

| Slots | Total wall ms | Maximum numerical error | External tensor copies |
| --- | ---: | ---: | ---: |
| 1 | 221.861 | 0 | 0 |
| 3 | 95.587 | 0 | 0 |

GPU hardware timestamps overlapped ANE host-dispatch intervals for 3.556 ms in
the three-slot run, versus zero in the one-slot run. Host-dispatch intervals
include runtime overhead; these are not ANE hardware counters. The fixture's
2.32x throughput difference is not a full-model acceleration claim. Reports:
`build/aneforge-probe/async-ring/`.

Separately, the native OBS runtime now uses serial NR and MFG workers, with a
single newest pending item at each boundary, instead of executing MFG inside the
NR loop. MFG uses a separate MLX GPU stream. Output surfaces stay retained by
their owning packets. Epoch checks reject obsolete results after controls
change. Stop drains both owners before releasing models. The stage history adds
NR start, NR ready, MFG start and publication timestamps; the new
`live_async_summary.py` reports host overlap and queue delay explicitly.

The build was installed and tested on live PS5 GTA at 1920x1080, retaining the
2160 processing maximum and RIFE 6x. A single31 sample reported 8.168 processed
FPS, zero NR/MFG errors and zero MFG superseded jobs, with 0.168 ms median and
0.272 ms maximum MFG handoff delay across the saved published rows. At that
snapshot 1,275 generated frames had completed (session counter). The trace
contains overlapping NR/MFG host intervals; it does not establish simultaneous
execution of their GPU kernels. Its saved summary is `async/current-summary.json`.

The user changed style, intensity branch/value and mode during the live session.
The later `performance.json` actually has double31 settings; it must not be
reported as a fixed single31 benchmark. The serial/async Fidelity samples also
have different styles and are not a controlled speed comparison. Fidelity
remained about three base FPS, with no generated frames when source gaps exceed
the existing RIFE cutoff. A later mixed-mode presentation sample showed 4.5 base
plus 23 generated shown FPS, but also long presentation gaps. Neither smoothness
nor a full-resolution throughput target passed. No quantized ANE NR stage was
installed. Capture-guided appearance stays disabled.

The installed runtime is backed up under
`build/aneforge-probe/async-live/serial-frameworks/`. Live reports are under the
same `async-live` directory. The latest user-selected double31 mode was restored
after the scheduler check; other live control changes were preserved.


## Hybrid complete FFN feasibility

`HybridFFN.mm` and `hybrid_ffn_probe.py` implement two original ANE projections,
with the model's quadratic gate and E4M3 publication on Metal between them.
The second ANE program includes the original cosine residual and reads the same
retained original-input IOSurface. Four surfaces are allocated and bound once;
no CPU tensor transfer occurs in the warm loop. Input initialization and final
oracle reads in the captured replay are explicitly outside the warm timing.

The measurement includes both ANE calls and the dependent Metal handoff, not
just isolated matrix times. This synchronous feasibility entry point must run
on an inference owner worker; its Metal wait is not suitable for the capture
callback or presentation thread. It is not installed in the running OBS model.

The reference now loads the actual quadratic-gate fused shader and original
E4M3 Metal header from MLX-DLSS source. The earlier 3.53 ms portable Python
reference used an unfused gate chain and is not the correct production-speed
baseline. The hybrid probe compares against the fused-gate reference instead.

With OBS continuing to run, 50 interleaved warm stage calls measured:

| Graph | Complete wall ms | GPU fused-gate reference ms | Captured relative L2 |
| --- | ---: | ---: | ---: |
| ANE FP16 projections + Metal gate | 1.870 | 1.702 | 0.0271-0.0301% |
| ANE INT8 weights / FP16 activations + Metal gate | 1.598 | 1.706 | 0.712-0.727% |

All six saved captured feature tensors passed the unchanged 1% operator gate.
They cover both Fidelity passes from three live source frames, not varied-scene
or full-model temporal acceptance. The INT8 variant is weight-only compression,
not a claim of packed W8A8 arithmetic. Its GPU gate took about 0.266 ms; the
complete gate handoff cost about 0.396 ms. Per-component medians need not sum
exactly to the median complete time.

These stage results exclude NHWC-to-port packing/unpacking and integration into
the full model scheduler. The roughly 0.108 ms advantage over the fused GPU
reference is therefore not yet a net pipeline speedup. Freeing GPU execution
for other work is a potential benefit, not a measured whole-frame result.
Reports: `build/aneforge-probe/hybrid-ffn-fp16/hybrid-result.json` and
`build/aneforge-probe/hybrid-ffn-w8a16/hybrid-result.json`.

The compact all-ANE W8A16 FFN retained the existing gate/publication and passed
one captured case at 0.728% relative L2, but took 4.360 ms. It remains rejected on
cost. The larger attention-block candidate's existing rejection is unchanged.

A shorter half-ULP E4M3 rounding candidate (`e4m3_magic`) was also tested, using
the identity already present in MLX-DLSS Metal. On this ANE graph it differs from
the required reference at 126 of the 63,488 finite FP16 inputs, including exact
halfway cases, with a maximum error of 32. It is a negative experimental case,
never the default publication. `publication_probe.py --publication magic`
exits with a parity failure; the compact default is unchanged. Evidence is in
`build/aneforge-probe/magic-publication/e4m3-magic-parity.json`.

The next integration gate must account for actual feature-layout transfers and
paired complete-model output before any candidate is promoted to live NR.
No production runtime libraries or controls were changed by these experiments.


## Actual MLX layout handoff gate

`HybridLayoutProbe.swift` now exercises an actual contiguous FP16 MLX array,
using MLX's existing `asMTLBuffer(device:noCopy:)`. It asserts contiguous strides
before requesting the wrapper and retains the source array through every GPU
read. Metal packs NHWC into the bound NCHW input surface. After hybrid FFN,
Metal can unpack into an NHWC Metal buffer consumed by a managed MLX array.

There is also a retained output-IOSurface route: MLX views `[1024,640]` and
transposes it as a strided view, removing the unpack dispatch. The surface is
retained by the managed array's finalizer. Every consumer must complete before
the context is reused and overwrites that surface. This is a synchronous
worker-only experiment, not an asynchronous production backend. The native
multi-input compiler wrapper supports the original residual input without
Python in native execution.

The optional layout probe builds with `build.py --mlx-project /path/to/MLX-DLSS`.
It needs the existing native runtime libraries under `build/nr/` and their
matching `mlx.metallib` colocated there. The experimental library now declares
a macOS 15 deployment target to match its public Metal API use; private runtime
compatibility is only tested on this macOS 27 M4 Pro host.

On 2026-11-11, with OBS verified **not running** at both ends of these tests,
50 interleaved stage calls and 30 interleaved calls with the following QKV
projection measured:

| W8A16 route | FFN Swift caller ms | GPU FFN reference ms | FFN + following QKV ms | GPU FFN + QKV ms |
| --- | ---: | ---: | ---: | ---: |
| GPU pack and unpack | 1.868 | 1.713 | 2.618 | 2.299 |
| GPU pack, retained strided output | 1.760 | 1.693 | 2.569 | 2.288 |

The full pack/unpack route spent about 0.165 ms and 0.138 ms in the respective
handoffs, although GPU kernels themselves took about 0.011 and 0.013 ms. The
strided route removed the unpack, but including the actual next MLX matmul
still cost more than staying on GPU. Per-case validation covers all six saved
captured tensors and the following QKV output; both routes passed the existing
1% relative-L2 gate. GPU unpack and the retained surface view were bitwise
identical across all tested half words. Invalid null input, aliasing output and
undersized output buffers are rejected before dispatch.

These are captured-feature replays, not new moving-gameplay acceptance. Initial
capture loading and the upstream feature producer are excluded. The reports
record OBS presence explicitly rather than assuming a concurrent live workload.
The approximately 0.108 ms pre-layout advantage from the prior operator test did
not survive this more complete handoff gate. Neither hybrid route was installed
in OBS. The full-resolution playable target remains open.

Native probe invocation (repeat with a different output filename and append
`--output-view` to test the strided route):

```sh
build/aneforge-probe/hybrid-layout-probe \
  build/aneforge-probe/hybrid-ffn-w8a16/manifest.json \
  build/aneforge-probe/live-stage-r3txx4ha/stages \
  /path/to/logical.safetensors \
  /path/to/MLX-DLSS/Sources/DLSSMLX/NeuralRenderingTransformerOperations.swift \
  build/aneforge-probe/hybrid-ffn-w8a16/layout-result.json
```

The manifest is emitted by `hybrid_ffn_probe.py` when lowering the two graphs.
Results are `layout-result.json` and `layout-view-result.json` in the same
`hybrid-ffn-w8a16` directory.


## Coarse FFN + QKV and complete-block gate

`prepare_coarse_ffn.py` lowers the residual projection and following QKV into
one ANE program, retaining **both** outputs. The original attention and final
residual remain on Metal. The two-output native compiler/binding API and retained
QKV surface avoid a separate GPU QKV projection without losing the mid residual.
`HybridLayoutProbe.swift` now measures through the complete original global
block, with compiled production attention and residual kernels accessed through
`MLXGlobalBlockProbe.swift`. There are no production renderer call-site changes.
The optional diagnostic API is distributed as
`../obs-dlss5-nr/patches/mlx-global-block-probe.patch`; reverse apply-check passed.

The previous FFN-only error gate was insufficient for promoting a candidate.
On all six captured feature tensors, full-block comparisons found:

| Coarse graph | Complete global block ms | Compiled GPU reference ms | Full-block relative L2 |
| --- | ---: | ---: | ---: |
| FP16 ANE projections | 3.037 | 2.789 | 1.499-1.560% |
| Weight INT8 / FP16 activation | 2.828 | 2.799 | 2.806-2.939% |

The unchanged 1% gate rejects both. The INT8 graph is under 1% at the FFN and
QKV outputs, but exceeds it after original attention and E4M3 publication. The
GPU split reference and original full-block reference matched exactly on all
six cases, helping separate a harness mismatch from cross-device numerical
error. This does not establish the cause of any visible live artifact, nor a
new moving-scene quality result. No tolerance was relaxed.

The latter replays overlapped an OBS Mac-display capture session (presence is
recorded per report); they are stage measurements with concurrent background
work, not controlled full-frame throughput. Reports are
`coarse-ffn-qkv-fp16/block-result.json` and
`coarse-ffn-qkv-w8a16/block-live-load-result.json` under
`build/aneforge-probe/`. Neither coarse candidate was installed in live NR.
