"""Offline paired distillation. Runtime remains native Core ML + MLX + Metal.

Manifest is JSON with color_space='sRGB', records [{clip,frame,split,source,
target,teacher,geometry_verified,protect?,confidence?,flow_to_previous?,valid?}].
Paths are relative to manifest. Splits are clip-disjoint. A single unpaired
reference is deliberately insufficient. Flow, if present, is current->previous
sampling displacement in pixels at 384x216; valid excludes disocclusions.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from model import AppearanceStudent,compose,WIDTH,HEIGHT,CONTRACT

def load_manifest(path,allow_pilot=False):
    d=json.loads(path.read_text())
    if d.get('color_space')!='sRGB': raise ValueError('Explicit sRGB training contract required')
    records=d.get('records',[]); seen=set();clips={};content_splits={}
    for r in records:
        key=(r['clip'],r['frame'])
        if key in seen: raise ValueError('Duplicate clip/frame key')
        seen.add(key)
        if not d.get('diagnostic_only') and (not r.get('protect') or not r.get('confidence')):
            raise ValueError('Production data requires explicit UI protection and trusted-target confidence masks')
        with Image.open(path.parent/r['source']) as source:
            digest=hashlib.sha256(source.convert('RGB').tobytes()).hexdigest()
        content_splits.setdefault(digest,set()).add(r['split'])
        if r['split'] not in ('train','validation','test'): raise ValueError('Unknown split')
        clips.setdefault(r['clip'],set()).add(r['split'])
        if not r.get('teacher'): raise ValueError('Teacher provenance required')
        if not r.get('geometry_verified') and not (allow_pilot and d.get('pilot_only')): raise ValueError('Registered geometry required; pilot labels need explicit --pilot')
        for key in ('source','target'):
            with Image.open(path.parent/r[key]) as im:
                if im.width*9 != im.height*16: raise ValueError('Expected full 16:9 coverage, not a center crop')
        with Image.open(path.parent/r['source']) as a,Image.open(path.parent/r['target']) as b:
            if a.size!=b.size: raise ValueError('Input/target dimensions must match; register explicitly offline')
        if 'flow_to_previous' in r:
            if not r.get('valid'): raise ValueError('Flow requires explicit nonocclusion validity mask')
            previous=(r['clip'],r['frame']-1)
            if not any((q['clip'],q['frame'])==previous and q['split']==r['split'] for q in records):
                raise ValueError('Temporal previous frame must exist in the same split')
    if any(len(s)!=1 for s in clips.values()): raise ValueError('Clip leakage across train/validation')
    if any(len(s)!=1 for s in content_splits.values()): raise ValueError('Identical frame content leaks across splits')
    if not all(any(r['split']==s for r in records) for s in ('train','validation')): raise ValueError('Need separate training and held-out clips')
    return d

def image(path,device,mask=False,full=False):
    with Image.open(path) as im:
        im=im.convert('L' if mask else 'RGB').resize((1280,720) if full else (WIDTH,HEIGHT),Image.Resampling.NEAREST if mask else Image.Resampling.BOX)
        arr=np.asarray(im,dtype=np.float32).copy()/255
    if mask: arr=arr[...,None]
    return torch.from_numpy(arr).permute(2,0,1)[None].to(device)

def warp(previous,flow):
    y,x=torch.meshgrid(torch.arange(HEIGHT,device=previous.device),torch.arange(WIDTH,device=previous.device),indexing='ij')
    coords=torch.stack((x,y),dim=-1).float()[None]+flow
    grid=torch.stack((2*coords[...,0]/(WIDTH-1)-1,2*coords[...,1]/(HEIGHT-1)-1),dim=-1)
    return F.grid_sample(previous,grid,mode='bilinear',padding_mode='border',align_corners=True)

def objective(model,x,y,protect,confidence=None):
    fields=model(F.interpolate(x,size=(HEIGHT,WIDTH),mode="bilinear",align_corners=False)); out=compose(x,fields,protect)
    data=sum(F.l1_loss(F.avg_pool2d(out,s),F.avg_pool2d(y,s)) for s in (1,2,4))/3
    edge=F.l1_loss(out[:,:,:,1:]-out[:,:,:,:-1],y[:,:,:,1:]-y[:,:,:,:-1])+F.l1_loss(out[:,:,1:]-out[:,:,:-1],y[:,:,1:]-y[:,:,:-1])
    smooth=(fields[:,:,1:]-fields[:,:,:-1]).abs().mean()+(fields[:,:,:,1:]-fields[:,:,:,:-1]).abs().mean()
    trust=fields[:,15:16].square().mean()*0.001
    if confidence is not None: trust=F.binary_cross_entropy_with_logits(fields[:,15:16],F.interpolate(confidence,size=fields.shape[-2:],mode='bilinear',align_corners=False))
    return data+0.2*edge+0.001*smooth+0.01*trust,out

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--full-resolution',action='store_true');p.add_argument('--head-only',action='store_true');p.add_argument('--pilot',action='store_true');p.add_argument('--width',type=int,default=24,choices=(24,48,64,96));p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--steps',type=int,default=2000);p.add_argument('--device',default='mps');p.add_argument('--seed',type=int,default=73)
    a=p.parse_args();torch.manual_seed(a.seed)
    d=load_manifest(a.manifest,allow_pilot=a.pilot); records=d['records']; device=torch.device(a.device)
    if (d.get('diagnostic_only') or d.get('pilot_only')) and 'build' not in a.output.resolve().parts: p.error('Diagnostic weights must remain in build/')
    if a.full_resolution and any('flow_to_previous' in r for r in records): p.error('Full-resolution temporal training requires full-resolution flow support; current flow contract is 384x216')
    cached={}
    for r in records:
        x=image(a.manifest.parent/r['source'],device,full=a.full_resolution);y=image(a.manifest.parent/r['target'],device,full=a.full_resolution)
        protect=image(a.manifest.parent/r['protect'],device,True,full=a.full_resolution) if r.get('protect') else torch.zeros_like(x[:,:1])
        # A target must preserve protected UI; do not train away incorrect labels.
        if ((x-y).abs()*protect).max().item()>0.03: raise ValueError('Teacher modified protected pixels')
        confidence=image(a.manifest.parent/r['confidence'],device,True,full=a.full_resolution) if r.get('confidence') else None
        cached[(r['clip'],r['frame'])]=(x,y,protect,confidence)
    model=AppearanceStudent(width=a.width).to(device)
    if a.head_only:
        for name,param in model.named_parameters(): param.requires_grad_(name.startswith("fields."))
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=a.learning_rate)
    train=[r for r in records if r['split']=='train'];valid=[r for r in records if r['split']=='validation']
    def evaluate():
        with torch.no_grad():
            return sum((compose(cached[(r['clip'],r['frame'])][0],model(F.interpolate(cached[(r['clip'],r['frame'])][0],size=(HEIGHT,WIDTH),mode='bilinear',align_corners=False)),cached[(r['clip'],r['frame'])][2])-cached[(r['clip'],r['frame'])][1]).abs().mean().item() for r in valid)/len(valid)
    before=evaluate(); history=[];best=before;best_state=None
    for step in range(a.steps):
        r=train[step%len(train)];x,y,protect,confidence=cached[(r['clip'],r['frame'])]
        opt.zero_grad();loss,out=objective(model,x,y,protect,confidence)
        if r.get('flow_to_previous'):
            px,py,pm,pc=cached[(r['clip'],r['frame']-1)]
            prev=compose(px,model(px),pm)
            f=np.load(a.manifest.parent/r['flow_to_previous'],allow_pickle=False)
            if f.shape!=(HEIGHT,WIDTH,2) or not np.isfinite(f).all(): raise ValueError('Invalid current-to-previous flow')
            flow=torch.from_numpy(f.copy()).float()[None].to(device)
            mask=image(a.manifest.parent/r['valid'],device,True)*(1-protect)*(1-warp(pm,flow))
            delta=(out-warp(prev,flow))-(y-warp(py,flow))
            loss=loss+0.1*(delta.abs()*mask).sum()/(mask.sum()*3+1e-6)
        if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        if step%25==0 or step==a.steps-1:
            score=evaluate();history.append({'step':step,'training_loss':loss.item(),'held_out_loss':score})
            if score<best: best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            print(json.dumps(history[-1]),flush=True)
    if best_state is None: raise ValueError('No improvement on held-out clips; checkpoint rejected')
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(best_state,a.output)
    report={'contract':CONTRACT,'diagnostic_only':d.get('diagnostic_only',False),'pilot_only':d.get('pilot_only',False),'unreviewed_geometry_count':sum(not r.get('geometry_verified',False) for r in records),'full_resolution':a.full_resolution,'head_only':a.head_only,'width':a.width,'steps':a.steps,'seed':a.seed,'learning_rate':a.learning_rate,
            'manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest(),'validation_before':before,'validation_best':best,
            'history':history,'temporal_supervised_pairs':sum('flow_to_previous' in r for r in train),
            'deployment_ready':False,'quality_scope':'384x216 paired appearance supervision only. Final 4K composed video and runtime latency remain unvalidated.'}
    a.output.with_suffix('.training.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
