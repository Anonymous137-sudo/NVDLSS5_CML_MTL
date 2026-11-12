"""Check that opt-in stage materialization preserves native NR output bytes.

A focused functional regression, not a performance or gameplay-quality test.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--plugin', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--worker', action='store_true')
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
if a.worker:
    lib = ctypes.CDLL(str(a.plugin/'Contents/Frameworks/libobs_dlss5_nr.dylib'))
    lib.nr5_create.argtypes = [ctypes.c_char_p]
    lib.nr5_create.restype = ctypes.c_void_p
    lib.nr5_destroy.argtypes = [ctypes.c_void_p]
    lib.nr5_process.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint64]
    lib.nr5_process.restype = ctypes.c_bool
    lib.nr5_error.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    handle = lib.nr5_create(str(a.plugin/'Contents/Resources').encode())
    if not handle:
        raise RuntimeError('Native session creation failed')
    image = np.zeros((320,320,4), np.uint8)
    yy, xx = np.mgrid[:320,:320]
    image[:,:,0] = (xx*3+yy)%256
    image[:,:,1] = (yy*2)%256
    image[:,:,2] = xx%256
    image[:,:,3] = 255
    try:
        for mode in [0,3]:
            options = dict(mode=mode,style=0,preset=0,intensity=1,finalIntensity=1,
                tone=1,structure=1,skin=-1,automask=False,temporal=False,
                reset=True,channelOrder=0)
            output = np.zeros_like(image)
            ok = lib.nr5_process(handle,image.ctypes.data,image.strides[0],
                output.ctypes.data,output.strides[0],320,320,
                json.dumps(options).encode(),1_000_000_000)
            if not ok:
                error = ctypes.create_string_buffer(1024)
                lib.nr5_error(handle,error,1024)
                raise RuntimeError(error.value.decode())
            np.save(a.output/f'mode-{mode}.npy',output)
    finally:
        lib.nr5_destroy(handle)
else:
    for name in ['ordinary','traced']:
        env = dict(os.environ)
        env.pop('DLSS_NR_STAGE_TRACE',None)
        if name == 'traced':
            env.update(DLSS_NR_STAGE_TRACE=str(a.output/'stage-traces'),
                DLSS_NR_STAGE_TRACE_SKIP='0',DLSS_NR_STAGE_TRACE_FRAMES='3')
        subprocess.run([sys.executable,__file__,'--worker','--plugin',str(a.plugin),
                        '--output',str(a.output/name)],env=env,check=True)
    results = []
    for mode in [0,3]:
        ordinary = np.load(a.output/'ordinary'/f'mode-{mode}.npy').astype(np.int16)
        traced = np.load(a.output/'traced'/f'mode-{mode}.npy').astype(np.int16)
        delta = np.abs(ordinary-traced)
        results.append(dict(mode=mode,maxByteDelta=int(delta.max()),
                            changedBytes=int(np.count_nonzero(delta))))
    (a.output/'trace-parity.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2),flush=True)
    if any(r['maxByteDelta'] for r in results):
        raise RuntimeError('Tracing changes output bytes')
