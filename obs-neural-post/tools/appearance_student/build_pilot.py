"""Generate real teacher supervision from captured sessions. Explicitly pilot-only."""
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
root=Path('build/teacher-distill');root.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(Path('../MLX-DLSS/python').resolve()))
from mlxdlss.pipeline import NeuralRenderingPipeline
pipeline=NeuralRenderingPipeline.from_safetensors('build/teacher/logical.safetensors',device='mps',precision='fast')
specs=[('motion180','train',Path('build/gta5-neural-motion-180'),[65,623,1826]),
       ('restart60','train',Path('build/gta5-neural-no-post-start-60'),[65,623,1860]),
       ('raw30','validation',Path('build/gta5-raw-metal-30'),[30,300,600]),
       ('night60','test',Path('build/gta5-neural-live-60'),[1834])]
records=[]
for clip,split,directory,sequences in specs:
 for sequence in sequences:
  source=directory/(f'frame-{sequence}.png' if clip=='raw30' else f'pair-{sequence}-input.png')
  folder=root/clip/str(sequence);folder.mkdir(parents=True,exist_ok=True)
  if not (folder/'teacher.png').exists():
   image=Image.open(source).convert('RGB');x=np.asarray(image,dtype=np.float32)/255
   start=time.perf_counter();result=pipeline.enhance(x,frame_index=0)
   y=np.asarray(result.image)
   if not np.isfinite(y).all() or y.shape!=x.shape:raise ValueError('Invalid teacher')
   # Conservative user-visible GTA minimap reservation, not automatic HUD segmentation.
   mask=np.zeros(x.shape[:2],np.uint8);mask[int(.70*x.shape[0]):,:int(.26*x.shape[1])]=255
   guard=np.zeros_like(mask);guard[int(.68*x.shape[0]):,:int(.28*x.shape[1])]=255
   y=np.where(guard[...,None]>0,x,np.clip(y,0,1))
   image.save(folder/'input.png');Image.fromarray(np.round(y*255).astype(np.uint8)).save(folder/'teacher.png')
   Image.fromarray(mask).save(folder/'protect.png')
   # Training gate: trusted away from UI only provisionally; no claim of calibrated uncertainty.
   Image.fromarray(255-mask).save(folder/'confidence.png')
   (folder/'provenance.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'teacher_weights_sha256':'76c063a509a56cb0d7a82791e0879a3400f45435ff298e044b89635253a41d2f',
      'seconds':time.perf_counter()-start,'geometry_verified':False,'review':'Same captured frame; dense geometry/identity review pending'},indent=2))
  relative=folder.relative_to(root)
  records.append({'clip':clip,'frame':sequence,'split':split,'source':str(relative/'input.png'),
     'target':str(relative/'teacher.png'),'protect':str(relative/'protect.png'),'confidence':str(relative/'confidence.png'),
     'teacher':'user DLL HT, MLX-DLSS MPS float16 reference','geometry_verified':False})
  print(clip,sequence,split,flush=True)
(root/'manifest.json').write_text(json.dumps({'color_space':'sRGB','pilot_only':True,'records':records},indent=2))
