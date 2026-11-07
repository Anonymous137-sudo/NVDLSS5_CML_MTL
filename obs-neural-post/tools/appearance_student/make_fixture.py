"""Synthetic contract/training smoke test. Never a photorealistic teacher dataset."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from model import WIDTH,HEIGHT
root=Path('build/appearance-student/fixture');root.mkdir(parents=True,exist_ok=True)
sources=[Path('build/gta5-neural-motion-180')/f'pair-{i}-input.png' for i in (65,623,1826)]
sources.append(Path('build/gta5-neural-live-60/pair-1834-input.png'))
records=[]
for i,path in enumerate(sources):
 x=np.asarray(Image.open(path).convert('RGB').resize((WIDTH,HEIGHT),Image.Resampling.BOX)).copy()
 protect=np.zeros((HEIGHT,WIDTH),np.uint8);protect[-64:,:96]=255
 valid=np.full((HEIGHT,WIDTH),255,np.uint8);valid[:,-1]=0
 flow=np.zeros((HEIGHT,WIDTH,2),np.float32);flow[...,0]=1
 np.save(root/'flow.npy',flow);Image.fromarray(valid).save(root/'valid.png')
 Image.fromarray(protect).save(root/'protect.png')
 for frame in range(2):
  if frame: x=np.concatenate((x[:,1:],x[:,-1:]),axis=1)
  target=np.clip(x.astype(np.float32)+np.array([8,3,-2]),0,255).astype(np.uint8)
  target[protect>0]=x[protect>0]
  prefix=f'{i}-{frame}'
  Image.fromarray(x).save(root/f'{prefix}-input.png');Image.fromarray(target).save(root/f'{prefix}-target.png')
  r={'clip':f'diagnostic-{i}','frame':frame,'split':'train' if i<2 else 'validation',
     'source':f'{prefix}-input.png','target':f'{prefix}-target.png','protect':'protect.png',
     'teacher':'synthetic fixed color offset; not DLSSMac or photorealistic supervision','geometry_verified':True}
  if frame:r.update(flow_to_previous='flow.npy',valid='valid.png')
  records.append(r)
(root/'manifest.json').write_text(json.dumps({'color_space':'sRGB','diagnostic_only':True,'records':records},indent=2))
print(root/'manifest.json')
