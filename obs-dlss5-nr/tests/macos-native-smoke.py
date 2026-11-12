import ctypes,json,sys,time
from pathlib import Path
import numpy as np
stage=Path(sys.argv[1])
lib=ctypes.CDLL(str(stage/'Contents/Frameworks/libobs_dlss5_nr.dylib'))
lib.nr5_create.argtypes=[ctypes.c_char_p];lib.nr5_create.restype=ctypes.c_void_p
lib.nr5_destroy.argtypes=[ctypes.c_void_p]
lib.nr5_error.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_size_t]
lib.nr5_process.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint64]
lib.nr5_process.restype=ctypes.c_bool
lib.nr5_profile.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t]
lib.nr5_profile.restype=ctypes.c_size_t
h=lib.nr5_create(str(stage/'Contents/Resources').encode())
assert h
w,hgt=320,320
pitch=w*4+64
a=np.zeros((hgt,pitch),dtype=np.uint8)
p=a[:,:w*4].reshape(hgt,w,4)
yy,xx=np.mgrid[:hgt,:w]
p[:,:,0]=(xx*3+yy)%256;p[:,:,1]=(yy*2)%256;p[:,:,2]=xx%256;p[:,:,3]=255
options=dict(style=0,preset=0,intensity=0,tone=1,structure=1,skin=-1,automask=False,temporal=False,reset=False,mode=1,channelOrder=0)
b=np.full_like(a,193)
def process(label,frame,options,t):
 start=time.monotonic()
 ok=lib.nr5_process(h,frame.ctypes.data,pitch,b.ctypes.data,pitch,w,hgt,json.dumps(options).encode(),t)
 err=ctypes.create_string_buffer(1024);lib.nr5_error(h,err,1024)
 print(label,ok,round((time.monotonic()-start)*1000,1),err.value.decode(),flush=True)
 return ok
assert process('FP16 Small intensity0',a,options,1_000_000_000)
assert np.max(np.abs(b[:,:w*4].astype(int)-a[:,:w*4].astype(int)))<=1
assert np.all(b[:,w*4:]==193)
profile=(ctypes.c_double*8)()
assert lib.nr5_profile(h,profile,8)==8
assert all(np.isfinite(x) and x>=0 for x in profile)
assert profile[2]>0 and profile[3]==0
assert sum(profile[i] for i in (0,1,2,3,4,6,7))>0
print('completed stage profile',list(profile),flush=True)
options['intensity']=1
assert process('FP16 Small model',a,options,1_020_000_000)
assert np.all(b[:,:w*4].reshape(hgt,w,4)[:,:,3]==255)
print('mean model byte change',np.abs(b[:,:w*4].astype(float)-a[:,:w*4]).mean(),flush=True)
options['temporal']=True
assert process('temporal first',a,options,1_040_000_000)
moved=a.copy()
moved[:,:w*4]=np.roll(p,4,axis=1).reshape(hgt,w*4)
assert process('temporal translated',moved,options,1_060_000_000)
options['temporal']=False
options['reset']=True
options['finalIntensity']=0
assert process('final intensity after history',a,options,1_080_000_000)
assert np.max(np.abs(b[:,:w*4].astype(int)-a[:,:w*4].astype(int)))<=1
options.pop('finalIntensity')
options['reset']=False
options['model31']='/nonexistent/NeuralRendering-Small.dlssmodel'
assert not process('missing replacement model',a,options,1_100_000_000)
options.pop('model31')
assert process('recover original model',a,options,1_120_000_000)
assert np.abs(b[:,:w*4].astype(float)-a[:,:w*4]).mean()>0.1
options['preset']=3
assert not process('unsupported preset',a,options,1_080_000_000)
lib.nr5_destroy(h)
print('native bridge smoke PASS',flush=True)
