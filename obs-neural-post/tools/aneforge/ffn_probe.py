"""Compile and verify one original 71-block model FFN on ANE-only surfaces.

A stage feasibility check, not an installed OBS backend. The GPU reference
retains the recovered quadratic gate and E4M3 intermediate publication.
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

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--aneforge', type=Path, required=True)
p.add_argument('--library', type=Path, required=True)
p.add_argument('--weights', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--tokens', type=int, default=640)
p.add_argument('--program-dir', type=Path, help='Reuse one compiled stage across captured samples')
p.add_argument('--weight-int8', action='store_true', help='Compress projection weights; activations and publication remain FP16')
p.add_argument('--feature', type=Path, help='Captured NHWC global-input.f16 with shape sidecar')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
captured = None
if a.feature:
    metadata = json.loads(a.feature.with_suffix('.json').read_text())
    shape = metadata['shape']
    if len(shape) != 4 or shape[0] != 1 or shape[3] != 1024 or metadata['dtype'] != 'float16':
        raise ValueError('Expected one captured 1024-channel NHWC FP16 feature grid')
    captured = np.fromfile(a.feature, np.float16).reshape(shape)
    a.tokens = shape[1]*shape[2]
    if a.tokens % 32:
        raise ValueError('Captured token count needs a separate padded port contract')
os.environ['ANEFORGE_DYLIB'] = str(a.library.resolve())
sys.path.insert(0, str(a.aneforge.resolve()))
import aneforge as af
from nr_ops import load_weight, feed_forward

lib = ctypes.CDLL(str(a.library.resolve()))
lib.dlss_af_make_surface.argtypes = [ctypes.c_int, ctypes.c_int]
lib.dlss_af_make_surface.restype = ctypes.c_void_p
lib.dlss_af_release_surface.argtypes = [ctypes.c_void_p]
lib.dlss_af_bind_surface.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
lib.dlss_af_fill_surface.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
w0 = load_weight(a.weights, 'block31.layer0.weight')
w1 = load_weight(a.weights, 'block31.layer1.weight')
cos = load_weight(a.weights, 'block31.layer1.ffn_cos_skip')
x = af.input((1, 1024, 1, a.tokens))
start = time.perf_counter()
model = af.compile(feed_forward(x, w0, w1, cos), int8=a.weight_int8, opt=0, build_dir=str(a.program_dir or a.output/'ffn31'))
compile_time = time.perf_counter()-start
surfaces = []
try:
    for port, is_input in [(model._inputs[0][0], 1), (model._out_name, 0)]:
        surface = lib.dlss_af_make_surface(1024, a.tokens)
        if not surface or lib.dlss_af_bind_surface(model._prog._handle, port.encode(), is_input, surface):
            raise RuntimeError('FFN IOSurface binding failed')
        surfaces.append(surface)
    if captured is None:
        if lib.dlss_af_fill_surface(surfaces[0], 43):
            raise RuntimeError('Metal fill failed')
    else:
        # Offline captured-feature replay only. This initialization copy is
        # excluded from warm timings and is not the production zero-copy route.
        np.copyto(model.input_view(), captured.reshape(a.tokens,1024).T.reshape(1,1024,1,a.tokens))
    model.execute()
    values = model.input_view().reshape(1024, a.tokens).T.copy()
    actual = model.output_view().reshape(1024, a.tokens).T.astype(np.float32)
    source = mx.array(values, dtype=mx.float16)
    expansion = mx.array(w0, dtype=mx.float16)
    projection = mx.array(w1, dtype=mx.float16)
    cosine = mx.array(cos, dtype=mx.float16)

    def gpu():
        expanded = source @ expansion
        clipped = mx.clip(expanded, -4, 4).astype(mx.float32)
        linear = (mx.abs(clipped)*-0.055908203125 + 0.447265625).astype(mx.float16)
        gate = (clipped*linear.astype(mx.float32) + 0.89453125).astype(mx.float16)
        activated = (expanded*gate).astype(mx.float16)
        magnitude = mx.minimum(mx.abs(activated.astype(mx.float32)), 448)
        exponent = mx.floor(mx.log2(mx.maximum(magnitude, 2**-6)))
        step = mx.where(magnitude < 2**-6, 2**-9, mx.power(2., exponent-3))
        published = (mx.sign(activated)*mx.round(magnitude/step)*step).astype(mx.float16)
        return ((published @ projection) + (source*cosine).astype(mx.float16)).astype(mx.float16)

    result = gpu()
    mx.eval(result)
    reference = np.array(result).astype(np.float32)
    relative = float(np.linalg.norm(actual-reference)/max(np.linalg.norm(reference), 1e-12))
    times, gpu_times = [], []
    for _ in range(5):
        model.execute()
    for _ in range(50):
        start = time.perf_counter(); model.execute()
        times.append((time.perf_counter()-start)*1000)
    for _ in range(50):
        start = time.perf_counter(); result=gpu(); mx.eval(result)
        gpu_times.append((time.perf_counter()-start)*1000)
    report = dict(scope='Block31 FFN including quadratic gate, E4M3 and residual; not complete attention block',
        weightINT8=a.weight_int8,activationPrecision='FP16',publication='compact-RNE',
        shape=[1,1024,1,a.tokens], deviceMask=4, hotLoopExternalTensorCopies=0,
        featureSource=str(a.feature) if a.feature else 'Metal-generated probe',
        initializationCopies=1 if captured is not None else 0,
        compileSeconds=compile_time, relativeL2=relative,
        finite=bool(np.isfinite(actual).all()), maxAbsError=float(np.max(abs(actual-reference))),
        warmMedianMS=statistics.median(times), warmP95MS=float(np.percentile(times,95)),
        gpuFP16MedianMS=statistics.median(gpu_times),
        parityPassed=bool(np.isfinite(actual).all() and relative <= .01))
    (a.output/'ffn-result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
finally:
    model.release()
    for surface in surfaces:
        lib.dlss_af_release_surface(surface)
