"""Test the backend's supported match-output geometry; do not weaken guards."""
import sys,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path('../MLX-DLSS/python').resolve()))
from mlxdlss.features import make_features
root=Path('build/quality-first/resolutions/144p-tight');root.mkdir(exist_ok=True)
source=Image.open('build/teacher/gta-720/input.png').convert('RGB');low=np.asarray(source.resize((256,144),Image.Resampling.LANCZOS))
padded=np.pad(low,((0,48),(0,0),(0,0)),mode='reflect');Image.fromarray(padded).save(root/'padded.png')
make_features(padded.astype(np.float32)/255).astype('<f4').tofile(root/'features.f32')
r=subprocess.run(['../MLX-DLSS/.build/release/mlxdlss','render-image',str(root/'padded.png'),'build/teacher/NeuralRendering.dlssmodel','--output',str(root/'transformed-padded.png'),'--execution','metal-fused','--precision','float16','--profile','standard','--network-geometry','match-output'],capture_output=True,text=True,check=True)
(root/'cold-render.json').write_text(r.stdout)
r=subprocess.run(['build/quality-first/native-teacher-budget','build/teacher/NeuralRendering.dlssmodel',str(root/'features.f32'),'build/quality-first/mlx.metallib','metal-fused','256','192'],capture_output=True,text=True,check=True);(root/'warm-bench.json').write_text(r.stdout)
y=np.asarray(Image.open(root/'transformed-padded.png').convert('RGB'))[:144];Image.fromarray(y).save(root/'transformed.png')
naive=Image.fromarray(y).resize((1280,720),Image.Resampling.LANCZOS);naive.save(root/'naive-720.png')
delta=(y.astype(np.float32)-low.astype(np.float32))/255
up=np.stack([np.asarray(Image.fromarray(delta[...,c]).resize((1280,720),Image.Resampling.BILINEAR)) for c in range(3)],axis=-1)
guided=np.clip(np.asarray(source,np.float32)/255+up,0,1);Image.fromarray(np.round(guided*255).astype(np.uint8)).save(root/'guided-720.png')
ref=np.asarray(Image.open('build/teacher/gta-720/teacher.png').convert('RGB'),np.float32)/255
report={'logical':[256,144],'network':[256,192],'geometry':'supported match-output with reflected bottom padding; unchanged model guard, manual crop to144 rows',
        'forward_median_ms':json.loads(r.stdout)['forwardMedianMS'],'guided_to_approved_mae':float(np.abs(guided-ref).mean()),'naive_to_approved_mae':float(np.abs(np.asarray(naive,np.float32)/255-ref).mean()),
        'note':'Nonvendor geometry is an experiment; reduced padding can change appearance. No temporal proof or end-to-end FPS claim.'}
(root/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
