"""Native full-transformer repeated single-pass timing; not the student pipeline."""
import sys,time,json
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path('../MLX-DLSS/python').resolve()))
from mlxdlss.mlxdlss_stream import MLXDLSSStreamSession
model=Path('build/teacher/NeuralRendering.dlssmodel').resolve();binary=str(Path('../MLX-DLSS/.build/release/mlxdlss').resolve())
x=np.asarray(Image.open('build/teacher/gta-720/input.png').convert('RGB'),dtype=np.float32)/255
times=[]
s=MLXDLSSStreamSession(model,1280,720,temporal=False,motion='zero',mlxdlss=binary,execution='metal-fused',precision='float16',protocol_version=3)
try:
 for i in range(6):
  start=time.perf_counter();y=s.process_frame(x);times.append((time.perf_counter()-start)*1000);print(i,times[-1],flush=True)
  if not np.isfinite(y).all():raise ValueError('Nonfinite native output')
finally:summary=s.close()
warm=times[1:];report={'source':'captured 720p static frame, native MLX metal-fused full HT transformer single pass per input','passes':1,'times_ms':times,'warm_median_ms':float(np.median(warm)),'warm_max_ms':float(max(warm)),'warm_completion_fps':1000/float(np.mean(warm)),'native_summary':summary,'note':'Includes Python/native pipe transport and output readback; excludes neural SR, FG and OBS. Not a live gameplay test. No temporal mode or generated frames.'}
Path('build/quality-first/native-stream.json').write_text(json.dumps(report,indent=2))
Image.fromarray(np.round(np.clip(y,0,1)*255).astype(np.uint8)).save('build/quality-first/native-stream-last.png')
