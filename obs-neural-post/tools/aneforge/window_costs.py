"""Diagnostic prefix timings for the recovered ANE window graph.

Prefixes are separately compiled, so fusion/planning may differ. Their times
must not be subtracted or summed as exact per-operator hardware costs.
"""
import argparse,json,os,statistics,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
for name in ['aneforge','library','weights','output']:p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
os.environ['ANEFORGE_DYLIB']=str(a.library.resolve());sys.path.insert(0,str(a.aneforge.resolve()))
import aneforge as af
from nr_ops import window32,load_weight
names=['weight1','weight2','ffn_cos_skip','qkv_weight','attn_scale','attn_bias','projection_weight','attn_cos_skip']
w={n:load_weight(a.weights,'block1.layer0.'+n,allow_float32=True) for n in names}
x=af.input((1,32,256,64));stages=window32(x,w,diagnostics=True)
source=(np.random.default_rng(21).standard_normal(x.shape)*.3).astype(np.float16)
rows=[]
for name in ['expanded','ff_published','mid','qkv','query','scores','probabilities','output']:
    net=af.compile(stages[name],opt=0,build_dir=str(a.output/name))
    try:
        np.copyto(net.input_view(),source)
        for _ in range(3):net.execute()
        times=[]
        for _ in range(10):
            start=time.perf_counter();net.execute();times.append((time.perf_counter()-start)*1000)
        row=dict(prefix=name,medianMS=statistics.median(times),outputShape=list(stages[name].shape))
        rows.append(row);print(json.dumps(row),flush=True)
    finally:net.release()
(a.output/'costs.json').write_text(json.dumps(rows,indent=2))
