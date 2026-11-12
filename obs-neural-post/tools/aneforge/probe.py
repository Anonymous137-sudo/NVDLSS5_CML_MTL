"""ANE-only shared-surface and checkpoint projection hardware checks.

These are isolated interoperability/operator probes. They do not measure OBS
throughput or validate the full neural renderer on moving gameplay.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import mlx.core as mx

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--aneforge', type=Path, required=True)
parser.add_argument('--library', type=Path, required=True)
parser.add_argument('--weights', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--tokens', type=int, default=640)
parser.add_argument('--projection-scale', type=float, default=1)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
os.environ['ANEFORGE_DYLIB'] = str(args.library.resolve())
sys.path.insert(0, str(args.aneforge.resolve()))
import aneforge as af

lib = ctypes.CDLL(str(args.library.resolve()))
lib.dlss_af_probe.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
lib.dlss_af_probe.restype = ctypes.c_int
lib.dlss_af_make_surface.argtypes = [ctypes.c_int, ctypes.c_int]
lib.dlss_af_make_surface.restype = ctypes.c_void_p
lib.dlss_af_release_surface.argtypes = [ctypes.c_void_p]
lib.dlss_af_bind_surface.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
lib.dlss_af_fill_surface.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

from nr_ops import load_weight

# Distinct frame patterns plus three retained surfaces catch stale buffers and
# accidental ring-slot reuse. GPU fills and checks the tensors.
x = af.input((1, 32, 1, args.tokens))
y = (x * .5 + .25).relu()
models = [af.compile(y, build_dir=str(args.output/'affine')) for _ in range(3)]
try:
    handles = (ctypes.c_void_p * 3)(*[m._prog._handle for m in models])
    rc = lib.dlss_af_probe(handles, 3, models[0]._inputs[0][0].encode(),
        models[0]._out_name.encode(), 32, args.tokens, 120,
        str((args.output/'surface-affine.json').resolve()).encode())
    if rc:
        raise RuntimeError(f'shared-surface round trip failed: {rc}')
finally:
    for model in models:
        model.release()

results = []
for name in ['block31.layer4.projection_weight', 'block31.layer2.qkv_weight', 'block31.layer0.weight']:
    w = load_weight(args.weights, name)
    cin, cout = w.shape
    x = af.input((1, cin, 1, args.tokens))
    graph = af.conv(x * args.projection_scale, w.T.reshape(cout, cin, 1, 1)) / args.projection_scale
    started = time.perf_counter()
    model = af.compile(graph, build_dir=str(args.output/name))
    compile_seconds = time.perf_counter() - started
    surfaces = []
    try:
        for channels, port, is_input in [(cin, model._inputs[0][0], 1), (cout, model._out_name, 0)]:
            surface = lib.dlss_af_make_surface(channels, args.tokens)
            if not surface:
                raise RuntimeError('IOSurface allocation failed')
            surfaces.append(surface)
            rc = lib.dlss_af_bind_surface(model._prog._handle, port.encode(), is_input, surface)
            if rc:
                raise RuntimeError(f'{name}: surface bind failed {rc}')
        if lib.dlss_af_fill_surface(surfaces[0], 17):
            raise RuntimeError('Metal input fill failed')
        model.execute()
        # CPU reads are validation only, outside the warm execution timing.
        source = model.input_view().reshape(cin, args.tokens).astype(np.float32)
        actual = model.output_view().reshape(cout, args.tokens).astype(np.float32)
        reference = w.astype(np.float32).T @ source
        gpu_input = mx.array(source.T.copy(), dtype=mx.float16)
        gpu_weight = mx.array(w, dtype=mx.float16)
        gpu_output = gpu_input @ gpu_weight
        mx.eval(gpu_output)
        gpu_reference = np.array(gpu_output).T.astype(np.float32)
        error = actual - gpu_reference
        relative_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(gpu_reference), 1e-12))
        cosine = float(np.sum(actual*gpu_reference, dtype=np.float64) /
            max(float(np.linalg.norm(actual)*np.linalg.norm(gpu_reference)), 1e-12))
        parity_passed = bool(np.isfinite(actual).all() and relative_l2 <= .01 and cosine >= .999)
        # Compare both I/O paths separately from the reference arithmetic gate.
        copied = af.compile(graph, build_dir=str(args.output/name))
        try:
            host_result = copied(source.reshape(1, cin, 1, args.tokens)).reshape(cout, args.tokens)
            io_error = float(np.max(np.abs(actual-host_result)))
            if io_error != 0:
                raise RuntimeError(f'{name}: shared-surface and copied ANE outputs disagree: {io_error}')
        finally:
            copied.release()
        for _ in range(5):
            model.execute()
        times = []
        for _ in range(50):
            start = time.perf_counter()
            model.execute()
            times.append((time.perf_counter()-start)*1000)
        gpu_times = []
        for _ in range(50):
            start = time.perf_counter()
            output = gpu_input @ gpu_weight
            mx.eval(output)
            gpu_times.append((time.perf_counter()-start)*1000)
        row = dict(weight=name, projectionScale=args.projection_scale, parityPassed=parity_passed, sharedVsCopiedMaxAbs=io_error,
            gpuFP16MedianMS=statistics.median(gpu_times),
            numpyFP32RelativeL2=float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference),1e-12)),
            referenceRMS=float(np.sqrt(np.mean(gpu_reference**2))), outputMax=float(np.max(np.abs(actual))), inputShape=[1, cin, 1, args.tokens],
            outputShape=[1, cout, 1, args.tokens], deviceMask=4,
            compileSeconds=compile_seconds, warmMedianMS=statistics.median(times),
            warmP95MS=float(np.percentile(times, 95)), relativeL2=relative_l2,
            cosine=cosine, maxAbsError=float(np.max(np.abs(error))), externalTensorCopies=0,
            scope='Original-checkpoint 1x1 projection, not the complete NR block or live pipeline')
        results.append(row)
        print(json.dumps(row), flush=True)
        (args.output/'checkpoint-projections.json').write_text(json.dumps(results, indent=2))
    finally:
        model.release()
        for surface in surfaces:
            lib.dlss_af_release_surface(surface)
print('Shared-surface I/O PASS; model projection parity:', all(r['parityPassed'] for r in results), flush=True)
