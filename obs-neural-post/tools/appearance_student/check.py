"""Contract regressions; deliberately tests coordinates/protection and leakage."""
import json,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from model import AppearanceStudent,compose,WIDTH,HEIGHT
from train import load_manifest,warp

def main():
 torch.set_num_threads(2);torch.manual_seed(73)
 x=torch.rand(1,3,HEIGHT,WIDTH);model=AppearanceStudent()
 with torch.no_grad():
  fields=model(x);assert fields.shape==(1,16,108,192)
  assert torch.equal(compose(x,fields),x),'Zero-edit identity must be exact'
  edits=torch.randn_like(fields)*4;protect=torch.zeros_like(x[:,:1]);protect[:,:,:40,:80]=1
  out=compose(x,edits,protect)
  assert torch.equal(out[:,:,:40,:80],x[:,:,:40,:80]),'Protected HUD changed'
  assert out.min()>=0 and out.max()<=1 and torch.isfinite(out).all()
  # Flow convention: current pixel samples one pixel right in previous frame.
  ramp=torch.arange(WIDTH).float()[None,None,None].expand(1,3,HEIGHT,WIDTH)/(WIDTH-1)
  flow=torch.zeros(1,HEIGHT,WIDTH,2);flow[...,0]=1
  assert torch.allclose(warp(ramp,flow)[:,:,:,:-1],ramp[:,:,:,1:],atol=1e-6)
 with tempfile.TemporaryDirectory() as name:
  root=Path(name);Image.fromarray(np.zeros((HEIGHT,WIDTH,3),np.uint8)).save(root/'frame.png')
  r={'clip':'same','frame':0,'split':'train','source':'frame.png','target':'frame.png','teacher':'diagnostic','geometry_verified':True}
  v=dict(r,frame=1,split='validation');path=root/'manifest.json'
  path.write_text(json.dumps({'color_space':'sRGB','diagnostic_only':True,'records':[r,v]}))
  try:load_manifest(path)
  except ValueError as e:assert 'leakage' in str(e)
  else:raise AssertionError('Clip leakage accepted')
  v['clip']='other';path.write_text(json.dumps({'color_space':'sRGB','diagnostic_only':True,'records':[r,v]}))
  try:load_manifest(path)
  except ValueError as e:assert 'Identical frame' in str(e)
  else:raise AssertionError('Identical frames across distinct clip labels accepted')
  Image.fromarray(np.ones((HEIGHT,WIDTH,3),np.uint8)).save(root/'other.png');v['source']='other.png'
  path.write_text(json.dumps({'color_space':'sRGB','diagnostic_only':True,'records':[r,v]}));load_manifest(path)
 print('PASS: identity, spatial field shape, exact HUD protection, gamut/finite output, flow coordinates, clip and identical-frame split rejection')
if __name__=='__main__':main()
