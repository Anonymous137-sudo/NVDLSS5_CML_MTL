"""Full-model resolution experiments; resampling comparisons are explicitly non-neural."""
import sys,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path('../MLX-DLSS/python').resolve()))
from mlxdlss.pipeline import NeuralRenderingPipeline
root=Path('build/quality-first/resolutions');root.mkdir(parents=True,exist_ok=True)
source=Image.open('build/teacher/gta-720/input.png').convert('RGB');x=np.asarray(source,dtype=np.float32)/255
reference=np.asarray(Image.open('build/teacher/gta-720/teacher.png').convert('RGB'),dtype=np.float32)/255
p=NeuralRenderingPipeline.from_safetensors('build/teacher/logical.safetensors',device='cpu')
rows=[]
for label,w,h in [('480p',854,480),('360p',640,360),('144p',256,144)]:
 d=root/label;d.mkdir(exist_ok=True);im=source.resize((w,h),Image.Resampling.LANCZOS);im.save(d/'input.png')
 low=np.asarray(im,dtype=np.float32)/255;features=p.prepare(low).features;features.astype('<f4').tofile(d/'features.f32')
 if not (d/'transformed.png').exists():
  run=subprocess.run(['../MLX-DLSS/.build/release/mlxdlss','render-image',str(d/'input.png'),'build/teacher/NeuralRendering.dlssmodel','--output',str(d/'transformed.png'),'--execution','metal-fused','--precision','float16','--profile','standard'],text=True,capture_output=True,check=True)
  (d/'cold-render.json').write_text(run.stdout)
 bench=subprocess.run(['build/quality-first/native-teacher-budget','build/teacher/NeuralRendering.dlssmodel',str(d/'features.f32'),'build/quality-first/mlx.metallib','metal-fused',str(features.shape[1]),str(features.shape[0])],capture_output=True,text=True,check=True)
 (d/'warm-bench.json').write_text(bench.stdout);time=json.loads(bench.stdout)
 transformed=Image.open(d/'transformed.png').convert('RGB');naive=np.asarray(transformed.resize((1280,720),Image.Resampling.LANCZOS),np.float32)/255
 delta=np.asarray(transformed,dtype=np.float32)/255-low
 # Float-plane interpolation of the low-resolution edit; preserve original 720p detail.
 up=np.stack([np.asarray(Image.fromarray(delta[...,c]).resize((1280,720),Image.Resampling.BILINEAR)) for c in range(3)],axis=-1)
 guided=np.clip(x+up,0,1)
 Image.fromarray(np.round(naive*255).astype(np.uint8)).save(d/'naive-720.png')
 Image.fromarray(np.round(guided*255).astype(np.uint8)).save(d/'guided-720.png')
 row={'resolution':label,'input_width':w,'input_height':h,'network_width':features.shape[1],'network_height':features.shape[0],
  'forward_median_ms':time['forwardMedianMS'],'render_readback_median_ms':time['renderMedianMS'],
  'naive_to_approved_mae':float(np.abs(naive-reference).mean()),'guided_to_approved_mae':float(np.abs(guided-reference).mean()),
  'note':'Single captured image; both 720p reconstruction comparisons use explicit spatial resampling, not an unimplemented learned low-resolution-to-720p model.'}
 rows.append(row);print(json.dumps(row),flush=True)
 (root/'summary.json').write_text(json.dumps(rows,indent=2))
