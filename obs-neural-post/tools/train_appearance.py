"""Train the small MLX appearance predictor from matched image pairs.

INPUT_DIR/name.png is the SDR capture; TARGET_DIR/name.png is the desired
appearance (e.g. an offline DLSSMac output of that exact frame). No teacher runs
inside OBS. Deliberately requires real pairs; it does not invent a pretrained look.
Dependencies for this development tool: mlx, pillow, numpy. Runtime is native Swift.
"""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim


class Appearance(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0=nn.Conv2d(3,16,3,stride=2,padding=1)
        self.conv1=nn.Conv2d(16,16,3,stride=2,padding=1)
        self.head=nn.Conv2d(16,12,1)
        self.head.weight=mx.zeros_like(self.head.weight)
        self.head.bias=mx.zeros_like(self.head.bias)
    def __call__(self,x):
        return self.head(nn.relu(self.conv1(nn.relu(self.conv0(x)))))


def image(path):
    with Image.open(path) as im:
        return mx.array(np.asarray(im.convert("RGB").resize((320,180),Image.Resampling.BOX),dtype=np.float32)/255)[None]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("inputs",type=Path); p.add_argument("targets",type=Path); p.add_argument("output",type=Path)
    p.add_argument("--steps",type=int,default=2000); p.add_argument("--seed",type=int,default=123)
    a=p.parse_args()
    pairs=[(f,a.targets/f.name) for f in sorted(a.inputs.glob("*.png")) if (a.targets/f.name).is_file()]
    if len(pairs)<2: p.error("Need at least two correctly aligned SDR PNG pairs; reserve separate validation clips")
    mx.random.seed(a.seed)
    model=Appearance(); optimizer=optim.Adam(learning_rate=0.001)
    def loss(model,x,y):
        c=model(x).reshape(1,45,80,3,4)
        rgb=mx.mean(x.reshape(1,45,4,80,4,3),axis=(2,4))
        target=mx.mean(y.reshape(1,45,4,80,4,3),axis=(2,4))
        features=mx.concatenate([rgb,mx.ones((1,45,80,1))],axis=-1)
        edit=mx.sum(c*features[...,None,:],axis=-1)
        # Constrain edits and coefficient variation. This is an appearance
        # approximation objective, not a physically based relighting loss.
        data=mx.mean(mx.abs(rgb+mx.clip(edit,-0.25,0.25)-target))
        smooth=mx.mean(mx.abs(c[:,1:]-c[:,:-1]))+mx.mean(mx.abs(c[:,:,1:]-c[:,:,:-1]))
        return data+0.001*smooth+0.0001*mx.mean(c*c)
    gradient=nn.value_and_grad(model,loss)
    for step in range(a.steps):
        left,right=pairs[step%len(pairs)]
        value,grads=gradient(model,image(left),image(right))
        optimizer.update(model,grads); mx.eval(model.parameters(),optimizer.state,value)
        if step%100==0: print(step,float(value.item()),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    model.save_weights(str(a.output))
    print(f"Saved {a.output}; validate held-out video and temporal stability before enabling live")

if __name__=="__main__": main()
