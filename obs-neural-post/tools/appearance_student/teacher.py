"""Offline teacher probe using the user-supplied MLX-DLSS backend and weights.

Never imported by OBS or the native viewer. Outputs are paired by source frame,
but require geometry/appearance review and protection/confidence masks before
being admitted as production training data.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image

def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
 p.add_argument('--backend',type=Path,default=Path('../MLX-DLSS'))
 p.add_argument('--weights',type=Path,default=Path('build/teacher/logical.safetensors'))
 p.add_argument('--width',type=int,default=384);p.add_argument('--frame-index',type=int,default=0)
 a=p.parse_args()
 if a.width%16 or a.width<320: p.error('Use width >=320 divisible by16; 384 gives full 16:9 384x216 context')
 sys.path.insert(0,str((a.backend/'python').resolve()))
 from mlxdlss.pipeline import NeuralRenderingPipeline
 import torch
 if not torch.backends.mps.is_available():p.error('This probe requires MPS; do not silently fall back to CPU')
 with Image.open(a.source) as im:
  if im.width*9!=im.height*16:p.error('Source must cover full16:9 frame')
  image=im.convert('RGB').resize((a.width,a.width*9//16),Image.Resampling.BOX)
 x=np.asarray(image,dtype=np.float32)/255
 started=time.perf_counter();pipeline=NeuralRenderingPipeline.from_safetensors(a.weights,device='mps',precision='fast')
 print('Loaded teacher',time.perf_counter()-started,flush=True)
 start=time.perf_counter();result=pipeline.enhance(x,frame_index=a.frame_index)
 torch.mps.synchronize();elapsed=time.perf_counter()-start
 y=np.asarray(result.image,dtype=np.float32)
 if y.shape!=x.shape or not np.isfinite(y).all(): raise ValueError('Invalid teacher output')
 a.output.mkdir(parents=True,exist_ok=True)
 image.save(a.output/'input.png');Image.fromarray(np.round(np.clip(y,0,1)*255).astype(np.uint8)).save(a.output/'teacher.png')
 report={'backend':str(a.backend.resolve()),'weights_sha256':digest(a.weights),'original_source':str(a.source.resolve()),
  'original_sha256':digest(a.source),'input_width':image.width,'input_height':image.height,
  'network_extent':list(result.network_extent),'precision':'MPS float16 fast reference port',
  'frame_index':a.frame_index,'inference_and_composition_seconds':elapsed,'backend_timings':result.timings,
  'mean_abs_rgb_change':float(np.abs(y-x).mean()),'max_abs_rgb_change':float(np.abs(y-x).max()),
  'min':float(y.min()),'max':float(y.max()),'finite':True,'production_training_ready':False,
  'review':'Review geometry/identity, exposure, materials and temporal behavior; add explicit HUD protection and trusted-target confidence before training. This probe is not native NVIDIA DLSS validation.'}
 (a.output/'teacher-probe.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
