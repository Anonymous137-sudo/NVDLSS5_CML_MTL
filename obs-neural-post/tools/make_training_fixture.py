"""Procedural pairs ONLY for verifying the appearance training/inference path.
These images are not game footage or evidence of photorealistic quality.
"""
from pathlib import Path
import numpy as np
from PIL import Image
root=Path(__file__).resolve().parents[1]/"build"/"appearance-fixture"
for side in ["input","target"]: (root/side).mkdir(parents=True,exist_ok=True)
y,x=np.mgrid[0:180,0:320]
for i in range(2):
    rgb=np.stack([(x+i*17)%256,(y+i*32)%256,(x+y)%256],axis=-1).astype(np.float32)/255
    target=np.clip(rgb*np.array([1.03,0.98,1.01])+np.array([0.01,0.0,-0.01]),0,1)
    Image.fromarray((rgb*255).round().astype(np.uint8)).save(root/"input"/f"{i}.png")
    Image.fromarray((target*255).round().astype(np.uint8)).save(root/"target"/f"{i}.png")
print(root)
