"""Attribute complete-window ANE error on captured activations, no renderer changes."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
for key in ['aneforge','library','weights','reference','capture','output']:
    p.add_argument('--'+key,type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
os.environ['ANEFORGE_DYLIB']=str(a.library.resolve())
sys.path[:0]=[str(a.aneforge.resolve()),str(a.reference.resolve()/'python')]
import aneforge as af
from aneforge._compile import compile_multi
from nr_ops import window32,load_weight,recover_single_head_bias
import torch
from mlxdlss import model as ref
names=['weight1','weight2','ffn_cos_skip','qkv_weight','attn_scale','attn_bias','projection_weight','attn_cos_skip']
w={n:load_weight(a.weights,'block1.layer0.'+n,allow_float32=True) for n in names}
source=np.fromfile(a.capture/'window1-input.f16',np.float16).reshape(1,8,64,32)
x=af.input((1,32,8,64));stages=window32(x,w,diagnostics=True)
net=compile_multi(list(stages.values()),build_dir=str(a.output/'stages'))
try:
    outputs=net(source.transpose(0,3,1,2).copy())
    actual={key:outputs[port] for key,(_,port) in zip(stages,net.output_ports)}
finally:net.release()
with torch.inference_mode():
    value=torch.from_numpy(source.copy());tw={k:torch.from_numpy(v.copy()) for k,v in w.items()}
    expected={}
    expected['expanded']=value@tw['weight1']
    expected['activated']=ref.quadratic_gate_activation(expected['expanded'])
    expected['ff_published']=ref.e4m3_round_trip(expected['activated'])
    expected['branch']=expected['ff_published']@tw['weight2']
    expected['mid']=expected['branch']+value*tw['ffn_cos_skip']
    expected['qkv']=expected['mid']@tw['qkv_weight']
    q,k,v=expected['qkv'].chunk(3,dim=-1)
    q=ref.vendor_cosine_publish(q,tw['attn_scale'])
    k=ref.vendor_cosine_publish(k);v=ref.e4m3_round_trip(v)
    expected.update(query=q,key=k,value=v)
    bias=torch.from_numpy(recover_single_head_bias(w['attn_bias']).copy())
    expected['scores']=(q@k.transpose(-1,-2))+bias
    expected['probabilities']=ref.vendor_approximate_softmax(expected['scores'])
    expected['attended']=ref.e4m3_round_trip(expected['probabilities']@v)
    expected['projection']=expected['attended']@tw['projection_weight']
    expected['output']=ref.e4m3_round_trip(expected['projection']+expected['mid']*tw['attn_cos_skip'])
rows=[]
channels_first={'expanded','activated','ff_published','branch','mid','qkv','projection','output'}
for key,computed in actual.items():
    if key in channels_first:computed=computed.transpose(0,2,3,1)
    target=expected[key].numpy().astype(np.float32)
    rows.append(dict(stage=key,relativeL2=float(np.linalg.norm(computed-target)/max(np.linalg.norm(target),1e-12)),
        maxAbsError=float(np.max(abs(computed-target))),changedFraction=float(np.mean(computed!=target))))
(a.output/'stages.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2),flush=True)
