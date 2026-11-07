"""Evaluate an explicit pilot on held-out source/teacher frames, never on art refs."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from model import AppearanceStudent,compose,CONTRACT,WIDTH,HEIGHT

def main():
 p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('checkpoint',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
 report=json.loads(a.checkpoint.with_suffix('.training.json').read_text())
 if report['contract']!=CONTRACT: p.error('Decoder mismatch')
 m=AppearanceStudent(report['width']).to('mps');m.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True));m.eval()
 records=json.loads(a.manifest.read_text())['records'];results=[];a.output.mkdir(parents=True,exist_ok=True)
 for r in records:
  def tensor(key,mask=False):
   im=Image.open(a.manifest.parent/r[key]).convert('L' if mask else 'RGB');arr=np.array(im,dtype=np.float32)/255
   if mask:arr=arr[...,None]
   return torch.from_numpy(arr).permute(2,0,1)[None].to('mps')
  x=tensor('source');y=tensor('target');mask=tensor('protect',True)
  with torch.no_grad():
   fields=m(F.interpolate(x,size=(HEIGHT,WIDTH),mode='bilinear',align_corners=False));out=compose(x,fields,mask)
   before=(x-y).abs().mean().item();after=(out-y).abs().mean().item();protected=((out-x).abs()*mask).max().item()
   rgb=out[0].permute(1,2,0).cpu().numpy()
  Image.fromarray(np.round(rgb*255).astype(np.uint8)).save(a.output/f"{r['clip']}-{r['frame']}-student.png")
  results.append({'clip':r['clip'],'frame':r['frame'],'split':r['split'],'source_teacher_mae':before,'student_teacher_mae':after,'protected_max_error':protected})
 summary={}
 for split in ('train','validation','test'):
  items=[r for r in results if r['split']==split]
  if not items:continue
  before=float(np.mean([r['source_teacher_mae'] for r in items]));after=float(np.mean([r['student_teacher_mae'] for r in items]))
  summary[split]={'frames':len(items),'baseline_mae':before,'student_mae':after,'relative_improvement':1-after/before}
 result={'contract':CONTRACT,'splits':summary,'frames':results,'deployment_ready':False,
  'pilot_quality_gate':summary['validation']['relative_improvement']>=.15 and summary.get('test',{}).get('relative_improvement',-1)>=0 and all(r['protected_max_error']==0 for r in results),
  'scope':'Sparse session-held-out pilot stills, not temporal or unseen-game generalization. Gate requires 15% validation MAE reduction and no test regression; it does not establish approved teacher appearance equivalence.'}
 (a.output/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2));print('pilot_quality_gate',result['pilot_quality_gate'])
if __name__=='__main__':main()
