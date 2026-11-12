"""Compare FP16, weight INT8, and activation+weight INT8 on captured expansion inputs.

No Core ML invocation. Quantization is tested on the exact block31 expansion
input; normalization, residuals, publications and display controls are unchanged.
This is a held-out operator experiment, not a live throughput acceptance test.
"""
import argparse,json,os,statistics,sys,time
from pathlib import Path
import numpy as np
import mlx.core as mx
p=argparse.ArgumentParser(description=__doc__)
for key in ['aneforge','library','weights','captures','output']:
    p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
os.environ['ANEFORGE_DYLIB']=str(a.library.resolve());sys.path.insert(0,str(a.aneforge.resolve()))
import aneforge as af
from nr_ops import load_weight
from quantized_graphs import activation_qdq
files=sorted(a.captures.glob('*/global-input.f16'))
if len(files)<4:raise ValueError('Need at least two capture pairs for separate calibration and held-out cases')
frames=[]
for f in files:
    metadata=json.loads(f.with_suffix('.json').read_text());shape=metadata['shape']
    if shape != [1,20,32,1024]:raise ValueError(f'Unexpected expansion input: {shape}')
    frames.append(np.fromfile(f,np.float16).reshape(640,1024).T.reshape(1,1024,1,640).copy())
calibration=frames[:-2];held_out=frames[-2:]
maximum=max(float(np.max(abs(v))) for v in calibration)
scale=float(np.float16(max(maximum/127,np.finfo(np.float16).tiny)))
w=load_weight(a.weights,'block31.layer0.weight');gpu_weight=mx.array(w,dtype=mx.float16)
# Select clipping from calibration activations only. Held-out frames never
# participate in choosing the scale. Report the search so clipping is explicit.
calibration_samples=np.concatenate([v.reshape(-1) for v in calibration]).astype(np.float32)
scale_search=[]
for factor in np.linspace(.5,1.,21):
    candidate=float(np.float16(scale*factor))
    reconstructed=np.clip(np.rint(calibration_samples/candidate),-128,127)*candidate
    scale_search.append(dict(scale=candidate,mse=float(np.mean((reconstructed-calibration_samples)**2))))
calibrated_scale=min(scale_search,key=lambda row:row['mse'])['scale']
(a.output/'calibration-search.json').write_text(json.dumps(scale_search,indent=2))
results=[]
for mode in ['fp16','w8a16','w8a8_qdq_candidate','w8a8_calibrated_candidate']:
    mode_scale=calibrated_scale if mode=='w8a8_calibrated_candidate' else scale
    x=af.input((1,1024,1,640));inp=x*16
    if mode.startswith('w8a8'):inp=activation_qdq(inp,np.float16(mode_scale*16))
    graph=af.conv(inp,w.T.reshape(4096,1024,1,1))/16
    started=time.perf_counter()
    # The new QDQ node is explicitly lowered here. The native ANE-only compiler
    # remains the hardware-support gate; no CPU/GPU fallback is requested.
    net=af.compile(graph,int8=mode!='fp16',opt=0,_check_precision=False,
                   build_dir=str(a.output/mode))
    compile_time=time.perf_counter()-started
    cases=[]
    try:
        for index,source in enumerate(held_out):
            actual=net(source).reshape(4096,640).T
            gpu_input=mx.array(source.reshape(1024,640).T.copy(),dtype=mx.float16)
            reference=gpu_input@gpu_weight;mx.eval(reference)
            reference=np.array(reference).astype(np.float32)
            error=actual-reference
            case=dict(capture=str(files[-2+index]),relativeL2=float(np.linalg.norm(error)/max(np.linalg.norm(reference),1e-12)),
                finite=bool(np.isfinite(actual).all()),maxAbsError=float(np.max(abs(error))),
                activationClipFraction=float(np.mean((source.astype(np.float32)<-mode_scale*128) |
                    (source.astype(np.float32)>mode_scale*127))))
            case['passed']=bool(case['finite'] and case['relativeL2']<=.01)
            cases.append(case)
        for _ in range(5):net.execute()
        times=[]
        for _ in range(50):
            start=time.perf_counter();net.execute();times.append((time.perf_counter()-start)*1000)
        mil=(a.output/mode/'model.mil').read_text()
        row=dict(mode=mode,compileSeconds=compile_time,warmMedianMS=statistics.median(times),
            warmP95MS=float(np.percentile(times,95)),cases=cases,allPassed=all(r['passed'] for r in cases),
            activationScale=mode_scale,calibrationCaptures=[str(f) for f in files[:-2]],
            quantizedWeightOp='constexpr_affine_dequantize' in mil,
            activationQDQ=(' = quantize(' in mil and ' = dequantize(' in mil),
            nativePackedINT8Verified=False,
            scope='Direct ANE-only projection graph; graph precision does not prove hardware arithmetic mode')
        results.append(row);print(json.dumps(row),flush=True)
        (a.output/'precision-results.json').write_text(json.dumps(results,indent=2))
    finally:net.release()
