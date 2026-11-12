"""ANE-only complete recovered window-block gate, including captured GPU output.

Prepartitioned windows are a stage contract, not a smaller game-rendering input.
No complete renderer is replaced by this probe.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--aneforge',type=Path,required=True)
p.add_argument('--library',type=Path,required=True)
p.add_argument('--weights',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True,help='MLX-DLSS checkout')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--windows',type=int,default=8)
p.add_argument('--iterations',type=int,default=50)
p.add_argument('--batch-layout',action='store_true',help='Put independent windows in N rather than H')
p.add_argument('--capture',type=Path,help='Directory containing window1-input/output.f16')
a=p.parse_args()
if a.windows<1 or a.iterations<1:p.error("windows and iterations must be positive")
a.output.mkdir(parents=True,exist_ok=True)
os.environ['ANEFORGE_DYLIB']=str(a.library.resolve())
sys.path[:0]=[str(a.aneforge.resolve()),str(a.reference.resolve()/'python')]
import aneforge as af
import torch
from mlxdlss import model as ref
from nr_ops import window32,load_weight,recover_single_head_bias

names=['weight1','weight2','ffn_cos_skip','qkv_weight','attn_scale','attn_bias','projection_weight','attn_cos_skip']
w={name:load_weight(a.weights,'block1.layer0.'+name,allow_float32=True) for name in names}
if a.capture:
    source=np.fromfile(a.capture/'window1-input.f16',np.float16).reshape(1,8,64,32)
    expected=np.fromfile(a.capture/'window1-output.f16',np.float16).reshape(1,8,64,32).astype(np.float32)
else:
    source=(np.random.default_rng(21).standard_normal((1,a.windows,64,32))*.3).astype(np.float16)
    expected=None
windows=source.shape[1]
indices=np.unique(np.linspace(0,windows-1,min(windows,24),dtype=int))
x=af.input((windows,32,1,64) if a.batch_layout else (1,32,windows,64));start=time.perf_counter()
model=af.compile(window32(x,w),opt=0,build_dir=str(a.output/'window1'))
compile_time=time.perf_counter()-start
try:
    tensor=(source[0].transpose(0,2,1)[:,:,None,:] if a.batch_layout else source.transpose(0,3,1,2)).copy()
    raw=model(tensor)
    actual=raw[:,:,0,:].transpose(0,2,1)[None,...] if a.batch_layout else raw.transpose(0,2,3,1)
    # Independent recovered PyTorch math is an extra oracle; captured fused-GPU
    # output is the authoritative paired-output gate when it is supplied.
    with torch.inference_mode():
        value=torch.from_numpy(source[0,indices].copy())
        tw={name:torch.from_numpy(array.copy()) for name,array in w.items()}
        hidden=ref.e4m3_round_trip(ref.quadratic_gate_activation(value@tw['weight1']))
        mid=(hidden@tw['weight2'])+value*tw['ffn_cos_skip']
        branch=ref.cosine_attention(mid,qkv_weight=tw['qkv_weight'],attention_scale=tw['attn_scale'],
            attention_bias=torch.from_numpy(recover_single_head_bias(w['attn_bias']).reshape(1,64,64).copy()),
            projection_weight=tw['projection_weight'],head_count=1)
        portable=ref.e4m3_round_trip(branch+mid*tw['attn_cos_skip']).numpy().reshape(1,len(indices),64,32).astype(np.float32)
    if expected is None:expected=portable
    selected=actual[:,indices]
    if a.capture: expected=expected[:,indices]
    error=selected-expected
    relative=float(np.linalg.norm(error)/max(np.linalg.norm(expected),1e-12))
    # Warm execution excludes input/output host copies and includes only the
    # ANE program. Full interop/OBS cost must be measured after a stage passes.
    for _ in range(5):model.execute()
    times=[]
    for _ in range(a.iterations):
        start=time.perf_counter();model.execute();times.append((time.perf_counter()-start)*1000)
    result=dict(scope='Complete block1 on prepartitioned windows; sampled output parity, not live renderer throughput',
        totalWindows=windows,windowBatchLayout=a.batch_layout,validatedWindowIndices=indices.tolist(),
        source=str(a.capture) if a.capture else 'seeded feature probe',compileSeconds=compile_time,
        finite=bool(np.isfinite(actual).all()),relativeL2=relative,
        maxAbsError=float(np.max(abs(error))),meanAbsError=float(np.mean(abs(error))),
        changedFraction=float(np.mean(selected!=expected)),warmMedianMS=statistics.median(times),
        warmP95MS=float(np.percentile(times,95)),
        portableReferenceRelativeL2=float(np.linalg.norm(selected-portable)/max(np.linalg.norm(portable),1e-12)),
        passed=bool(np.isfinite(actual).all() and relative<=.01),deviceMask=4,
        initializationCopies=1,hotLoopExternalTensorCopies=0)
    np.save(a.output/'sampled-actual.npy',selected);np.save(a.output/'expected.npy',expected)
    (a.output/'window-result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
finally:model.release()
if not result["passed"]:
    raise SystemExit(2)
