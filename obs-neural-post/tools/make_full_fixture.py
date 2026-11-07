"""Generate sparse full-720p/4K PyTorch reference samples for the Metal pack test."""
import json
from pathlib import Path
import torch
from convert_rt4ksr import reference

root=Path(__file__).resolve().parents[1]
s={k.removeprefix("module."):v for k,v in torch.load(root/"models/rt4ksr_x3.pth",map_location="cpu",weights_only=True)["state_dict"].items()}
x=torch.arange(1280).view(1,1280).expand(720,-1)
y=torch.arange(720).view(720,1).expand(-1,1280)
rgb=torch.stack([(x+y)%256,y%256,x%256],dim=0).float()[None]/255
with torch.inference_mode(): out=reference(rgb,s).clamp(0,1)
coords=[(0,0),(3839,0),(0,2159),(3839,2159),(1,1),(5,5),(6,6),(100,100),(1920,1080)]
coords += [((i*131+17)%3840,(i*79+31)%2160) for i in range(64)]
samples=[{"x":x,"y":y,"rgb":out[0,:,y,x].tolist()} for x,y in coords]
(root/"models/full-parity.json").write_text(json.dumps(samples,indent=2)+"\n")
print(f"Wrote {len(samples)} full-frame reference samples")
