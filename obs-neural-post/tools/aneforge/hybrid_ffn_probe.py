"""Captured FFN: ANE projections + exact Metal gate, including handoff cost.

The FP16 GPU baseline loads the gate shader from the actual MLX-DLSS source.
This is a stage experiment, not a live throughput acceptance test.
"""
import argparse,ctypes,json,os,re,statistics,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
p=argparse.ArgumentParser(description=__doc__)
for key in ['aneforge','library','weights','captures','mlx-source','output']:
    p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--weight-int8',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
os.environ['ANEFORGE_DYLIB']=str(a.library.resolve());sys.path.insert(0,str(a.aneforge.resolve()))
import aneforge as af
from nr_ops import load_weight
w0=load_weight(a.weights,'block31.layer0.weight');w1=load_weight(a.weights,'block31.layer1.weight')
cos=load_weight(a.weights,'block31.layer1.ffn_cos_skip')
files=sorted(a.captures.glob('*/global-input.f16'))
if not files:raise ValueError('No captured features')
source_text=a.mlx_source.read_text()
header=re.search(r'private static let e4m3MetalHeader = #"""(.*?)"""#',source_text,re.S)[1]
kernel=re.search(r'private static let quadraticGatePublishKernel.*?source: #"""(.*?)"""#',source_text,re.S)[1]
gate=mx.fast.metal_kernel(name='dlss_hybrid_ffn_reference_gate',input_names=['input','params'],
    output_names=['output'],source=kernel,header=header)
lib=ctypes.CDLL(str(a.library.resolve()))
lib.dlss_af_hybrid_create.argtypes=[ctypes.c_void_p]*2+[ctypes.c_char_p]*5+[ctypes.c_int]
lib.dlss_af_hybrid_create.restype=ctypes.c_void_p
lib.dlss_af_hybrid_execute.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double)]
lib.dlss_af_hybrid_destroy.argtypes=[ctypes.c_void_p]
models=[];context=None
try:
    x=af.input((1,1024,1,640))
    expand=af.compile(af.conv(x*16,w0.T.reshape(4096,1024,1,1))/16,int8=a.weight_int8,
                      opt=0,build_dir=str(a.output/'expand'));models.append(expand)
    published=af.input((1,4096,1,640));skip=af.input((1,1024,1,640))
    branch=af.compile(af.conv(published*16,w1.T.reshape(1024,4096,1,1))/16+skip*cos.reshape(1,1024,1,1),
                      int8=a.weight_int8,opt=0,build_dir=str(a.output/'branch'));models.append(branch)
    branch_input=next(n for n,s in branch._inputs if s[1]==4096)
    branch_skip=next(n for n,s in branch._inputs if s[1]==1024)
    manifest=dict(tokens=640,expand=dict(directory=str((a.output/'expand').resolve()),
        inputs=[dict(name=n,bytes=int(np.prod(s))*2) for n,s in expand._inputs],
        output=expand._out_name,outputBytes=4096*640*2),
        branch=dict(directory=str((a.output/'branch').resolve()),
        inputs=[dict(name=n,bytes=int(np.prod(s))*2) for n,s in branch._inputs],
        output=branch._out_name,outputBytes=1024*640*2),
        branchInput=branch_input,branchSkip=branch_skip,weightINT8=a.weight_int8)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    context=lib.dlss_af_hybrid_create(expand._prog._handle,branch._prog._handle,
        expand._inputs[0][0].encode(),expand._out_name.encode(),branch_input.encode(),branch_skip.encode(),
        branch._out_name.encode(),640)
    if not context:raise RuntimeError('Hybrid IOSurface initialization failed')
    expansion=mx.array(w0);projection=mx.array(w1);cosine=mx.array(cos);params=mx.array([640*4096//4],mx.uint32)
    mx.eval(expansion,projection,cosine,params)
    def gpu(source):
        expanded=source@expansion
        activated=gate(inputs=[expanded,params],grid=(640*4096//4,1,1),threadgroup=(256,1,1),
            output_shapes=[expanded.shape],output_dtypes=[mx.float16])[0]
        return ((activated@projection)+(source*cosine).astype(mx.float16)).astype(mx.float16)
    timings=(ctypes.c_double*5)();cases=[]
    for feature in files:
        shape=json.loads(feature.with_suffix('.json').read_text())['shape']
        if shape!=[1,20,32,1024]:raise ValueError(f'Unexpected shape: {shape}')
        values=np.fromfile(feature,np.float16).reshape(640,1024)
        np.copyto(expand.input_view(),values.T.reshape(1,1024,1,640))
        if lib.dlss_af_hybrid_execute(context,timings):raise RuntimeError('Hybrid execute failed')
        actual=branch.output_view().reshape(1024,640).T.astype(np.float32)
        source=mx.array(values);reference=gpu(source);mx.eval(reference)
        expected=np.array(reference).astype(np.float32)
        relative=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-12))
        cases.append(dict(feature=str(feature),relativeL2=relative,finite=bool(np.isfinite(actual).all()),
            maxAbsError=float(np.max(abs(actual-expected))),passed=bool(np.isfinite(actual).all() and relative<=.01)))
    # Interleave controls to reduce bias from the running OBS GPU workload.
    for _ in range(5):
        lib.dlss_af_hybrid_execute(context,timings)
        result=gpu(source);mx.eval(result)
    rows=[];gpu_times=[]
    for _ in range(50):
        if lib.dlss_af_hybrid_execute(context,timings):raise RuntimeError('Warm hybrid execute failed')
        rows.append(list(timings))
        start=time.perf_counter();result=gpu(source);mx.eval(result);gpu_times.append((time.perf_counter()-start)*1000)
    names=['expandANEWallMS','gateHandoffWallMS','branchANEWallMS','completeWallMS','gateGPUKernelMS']
    report=dict(scope='Captured complete block31 FFN; includes both ANE calls and GPU handoff',
        weightINT8=a.weight_int8,activationPrecision='FP16',deviceMask=4,externalHotLoopTensorCopies=0,
        initializationCopiesPerCase=1,cases=cases,allPassed=all(r['passed'] for r in cases),
        medians={name:statistics.median(r[i] for r in rows) for i,name in enumerate(names)},
        fusedGateGPUReferenceMedianMS=statistics.median(gpu_times),
        productionInstalled=False,nativePackedINT8Verified=False,
        timingScope='Interleaved warm stage calls with OBS running; NHWC port packing and full-model scheduling excluded')
    (a.output/'hybrid-result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
finally:
    if context:lib.dlss_af_hybrid_destroy(context)
    for model in models:model.release()
