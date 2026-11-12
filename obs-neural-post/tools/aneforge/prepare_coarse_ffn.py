"""Lower original FFN branch, cosine residual and QKV into one ANE program.

Both mid and QKV remain outputs: the original attention still needs the mid
residual. No attention nonlinearities or softmax are approximated here.
"""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
for key in ['aneforge','library','weights','output']:p.add_argument('--'+key,type=Path,required=True)
p.add_argument('--weight-int8',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
os.environ['ANEFORGE_DYLIB']=str(a.library.resolve());sys.path.insert(0,str(a.aneforge.resolve()))
import aneforge as af
from aneforge._compile import compile_multi
from nr_ops import load_weight
w0=load_weight(a.weights,'block31.layer0.weight');w1=load_weight(a.weights,'block31.layer1.weight')
cos=load_weight(a.weights,'block31.layer1.ffn_cos_skip');wqkv=load_weight(a.weights,'block31.layer2.qkv_weight')
x=af.input((1,1024,1,640));models=[]
try:
    expand=af.compile(af.conv(x*16,w0.T.reshape(4096,1024,1,1))/16,int8=a.weight_int8,
                      opt=0,build_dir=str(a.output/'expand'));models.append(expand)
    published=af.input((1,4096,1,640));skip=af.input((1,1024,1,640))
    mid=af.conv(published*16,w1.T.reshape(1024,4096,1,1))/16+skip*cos.reshape(1,1024,1,1)
    qkv=af.conv(mid*16,wqkv.T.reshape(3072,1024,1,1))/16
    branch=compile_multi([mid,qkv],int8=a.weight_int8,build_dir=str(a.output/'branch'));models.append(branch)
    inputs=[dict(name=n,bytes=int(np.prod(t.shape))*2) for t,n in branch.input_ports]
    outputs=[dict(name=n,bytes=int(np.prod(t.shape))*2) for t,n in branch.output_ports]
    manifest=dict(tokens=640,coarseQKV=True,weightINT8=a.weight_int8,
        expand=dict(directory=str((a.output/'expand').resolve()),
        inputs=[dict(name=n,bytes=int(np.prod(s))*2) for n,s in expand._inputs],
        output=expand._out_name,outputBytes=4096*640*2),
        branch=dict(directory=str((a.output/'branch').resolve()),inputs=inputs,outputs=outputs,
                    output=outputs[0]['name'],outputBytes=outputs[0]['bytes']),
        branchInput=next(n for t,n in branch.input_ports if t.shape[1]==4096),
        branchSkip=next(n for t,n in branch.input_ports if t.shape[1]==1024),qkvOutput=outputs[1]['name'])
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
finally:
    for model in models:model.release()
