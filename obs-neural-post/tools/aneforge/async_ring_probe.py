"""Native-only GPU->ANE->GPU continuous-ring validation; no live NR claim.

Uses a previously exported affine MIL graph. No Python ANEForge import or CPU
input tensor touches occur. CPU reads only the GPU-generated validation errors.
"""
import argparse
import ctypes
import json
import re
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--library',type=Path,required=True)
p.add_argument('--graph',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--iterations',type=int,default=600)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
text=(a.graph/'model.mil').read_text()
match=re.search(r'tensor<fp16, \[1, (\d+), 1, (\d+)\]> (\w+)\)',text)
output=re.search(r'-> \((\w+)\);',text)
if not match or not output:raise ValueError('Expected single-input affine NCHW graph')
c,s=int(match[1]),int(match[2]);input_name=match[3].encode();output_name=output[1].encode()
lib=ctypes.CDLL(str(a.library.resolve()))
lib.dlss_ane_compile_directory.argtypes=[ctypes.c_char_p]*3+[ctypes.c_size_t]*2
lib.dlss_ane_compile_directory.restype=ctypes.c_void_p
lib.dlss_ane_destroy.argtypes=[ctypes.c_void_p]
lib.dlss_af_probe.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_int,
    ctypes.c_char_p,ctypes.c_char_p]+[ctypes.c_int]*3+[ctypes.c_char_p]
results=[]
for slots in [1,3]:
    handles=[]
    try:
        for _ in range(slots):
            handle=lib.dlss_ane_compile_directory(str(a.graph.resolve()).encode(),input_name,output_name,c*s*2,c*s*2)
            if not handle:raise RuntimeError('ANE-only compile failed')
            handles.append(handle)
        report=a.output/f'ring-{slots}.json'
        rc=lib.dlss_af_probe((ctypes.c_void_p*slots)(*handles),slots,input_name,output_name,c,s,
                            a.iterations,str(report.resolve()).encode())
        if rc:raise RuntimeError(f'Ring failed: {rc}')
        data=json.loads(report.read_text());rows=data['frames']
        if len(rows)!=a.iterations or sorted(r['frame'] for r in rows)!=list(range(a.iterations)):
            raise RuntimeError('Missing or duplicate frame patterns')
        # Hardware GPU timestamps and host ANE execute intervals share host time.
        # This measures overlap with the host dispatch interval, not ANE counters.
        intervals=sorted((r[k+'Start'],r[k+'End']) for r in rows for k in ['gpuWrite','gpuRead'])
        merged=[]
        for start,end in intervals:
            if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
            else:merged.append([start,end])
        overlap=sum(max(0,min(r['aneEnd'],end)-max(r['aneStart'],start))
                    for r in rows for start,end in merged)*1000
        row=dict(slots=slots,frames=len(rows),wallMS=data['wallMS'],
                 framesPerSecond=len(rows)*1000/data['wallMS'],
                 gpuOverlapANEHostIntervalMS=overlap,
                 maxAbsError=max(r['maxAbsError'] for r in rows),
                 externalTensorCopies=data['externalTensorCopies'])
        results.append(row);print(json.dumps(row),flush=True)
    finally:
        for handle in handles:lib.dlss_ane_destroy(handle)
(a.output/'comparison.json').write_text(json.dumps(results,indent=2))
