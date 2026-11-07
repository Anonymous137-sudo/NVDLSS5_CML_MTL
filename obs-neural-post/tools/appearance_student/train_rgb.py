"""Pilot native-resolution residual student. Train patches; evaluate full held-out frames."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
CONTRACT='neural-post.rgb-residual.v1'
class RGBStudent(nn.Module):
 def __init__(self):
  super().__init__();self.layers=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,3,3,padding=1))
  nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
 def forward(self,x):return 0.35*torch.tanh(self.layers(x))
def main():
 p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--steps',type=int,default=1500);a=p.parse_args()
 d=json.loads(a.manifest.read_text());assert d.get('pilot_only') and 'build' in a.output.resolve().parts
 torch.manual_seed(73);rng=np.random.default_rng(73);m=RGBStudent().to('mps');opt=torch.optim.Adam(m.parameters(),lr=0.0003)
 cached=[]
 for r in d['records']:
  def load(key,mask=False):
   z=np.asarray(Image.open(a.manifest.parent/r[key]).convert('L' if mask else 'RGB'),dtype=np.float32).copy()/255
   if mask:z=z[...,None]
   return torch.from_numpy(z).permute(2,0,1)[None].to('mps')
  cached.append((r,load('source'),load('target'),load('protect',True)))
 def evaluate(split):
  rows=[]
  with torch.no_grad():
   for r,x,y,mask in cached:
    if r['split']!=split:continue
    out=(x+m(x)*(1-mask)).clamp(0,1)
    rows.append({'clip':r['clip'],'frame':r['frame'],'baseline_mae':(x-y).abs().mean().item(),'student_mae':(out-y).abs().mean().item()})
  return rows
 before=evaluate('validation')
 if a.checkpoint:m.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
 best=np.mean([r['student_mae'] for r in before]);state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()} if a.checkpoint else None;history=[]
 train=[z for z in cached if z[0]['split']=='train']
 for step in range(a.steps):
  _,x,y,mask=train[int(rng.integers(len(train)))];top=int(rng.integers(720-128));left=int(rng.integers(1280-128))
  x=x[:,:,top:top+128,left:left+128];y=y[:,:,top:top+128,left:left+128];mask=mask[:,:,top:top+128,left:left+128]
  out=(x+m(x)*(1-mask)).clamp(0,1)
  # Exclude patch-padding edges from supervision; native full-frame borders are measured separately.
  out=out[:,:,4:-4,4:-4];y=y[:,:,4:-4,4:-4]
  loss=F.l1_loss(out,y)+.15*(F.l1_loss(out[:,:,:,1:]-out[:,:,:,:-1],y[:,:,:,1:]-y[:,:,:,:-1])+F.l1_loss(out[:,:,1:]-out[:,:,:-1],y[:,:,1:]-y[:,:,:-1]))
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  if step%100==0 or step==a.steps-1:
   rows=evaluate('validation');score=float(np.mean([r['student_mae'] for r in rows]));history.append({'step':step,'loss':loss.item(),'validation_mae':score});print(json.dumps(history[-1]),flush=True)
   if score<best:best=score;state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
 if state is None:raise ValueError('No held-out improvement')
 m.load_state_dict(state);torch.save(state,a.output)
 result={'contract':CONTRACT,'pilot_only':True,'deployment_ready':False,'steps':a.steps,'validation_before':float(np.mean([r['baseline_mae'] for r in before])),'validation_best':best,'history':history,'splits':{s:evaluate(s) for s in ['train','validation','test']}}
 for split,rows in result['splits'].items():
  result[split+'_relative_improvement']=1-np.mean([r['student_mae'] for r in rows])/np.mean([r['baseline_mae'] for r in rows])
 result['pilot_quality_gate']=bool(result['validation_relative_improvement']>=.15 and result['test_relative_improvement']>=0)
 a.output.with_suffix('.training.json').write_text(json.dumps(result,indent=2))
 with torch.no_grad():
  for r,x,y,mask in cached:
   if r['split']=='validation':
    rgb=(x+m(x)*(1-mask)).clamp(0,1)[0].permute(1,2,0).cpu().numpy();Image.fromarray(np.round(rgb*255).astype(np.uint8)).save(a.output.parent/f"rgb-{r['clip']}-{r['frame']}.png")
 print(json.dumps({k:v for k,v in result.items() if 'improvement' in k or k=='pilot_quality_gate'}))
if __name__=='__main__':main()
