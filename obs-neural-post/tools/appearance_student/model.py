"""Screen-space ANE candidate. Teacher-distilled pilot weights are generated separately, never random defaults."""
import torch
from torch import nn
from torch.nn import functional as F

WIDTH, HEIGHT = 384, 216
CONTRACT = 'neural-post.appearance-fields.v2'

class Block(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.spatial = nn.Conv2d(width,width,3,padding=dilation,dilation=dilation,groups=width)
        self.mix = nn.Conv2d(width,width,1)
    def forward(self,x):
        return x + 0.1*self.mix(F.relu(self.spatial(x)))

class AppearanceStudent(nn.Module):
    """Predict 12 affine coefficients, 3 content-guided RGB detail gains and confidence.

    Input: RGB [0,1], [1,3,216,384]. Output [1,16,108,192]. Fixed aspect;
    no center crop. Global context plus dilated local context; no geometry input.
    Raw fields are decoded by compose(), not interpreted as an RGB image.
    """
    def __init__(self,width=24):
        super().__init__()
        self.encoder = nn.Conv2d(3,width,3,stride=2,padding=1)
        self.blocks = nn.Sequential(*(Block(width,d) for d in (1,2,4,8)))
        self.context = nn.Conv2d(width,width,1)
        self.fields = nn.Conv2d(width,16,1)
        # Zero edits are a safe initialization, not a trained appearance model.
        nn.init.zeros_(self.fields.weight); nn.init.zeros_(self.fields.bias)
    def forward(self,rgb):
        local = self.blocks(F.relu(self.encoder(rgb)))
        context = self.context(local.mean(dim=(2,3),keepdim=True))
        return self.fields(F.relu(local+context))

def compose(rgb,fields,protect=None):
    """Differentiable GPU-compositor reference, including soft gamut protection.

    protection=1 preserves HUD/identity pixels; confidence is learned only when
    explicit trusted-target supervision exists. No fake semantic labeling.
    """
    f=F.interpolate(fields,size=rgb.shape[-2:],mode='bilinear',align_corners=False)
    coefficients=0.125*torch.tanh(f[:,:12]).reshape(-1,3,4,*rgb.shape[-2:])
    features=torch.cat((rgb,torch.ones_like(rgb[:,:1])),dim=1)
    affine=(coefficients*features[:,None]).sum(dim=2)
    highpass=rgb-F.avg_pool2d(F.pad(rgb,(1,1,1,1),mode="replicate"),3,stride=1)
    residual=2*torch.tanh(f[:,12:15])*highpass
    confidence=torch.sigmoid(f[:,15:16])
    edit=confidence*(affine+residual)
    if protect is not None: edit=edit*(1-protect.clamp(0,1))
    # Preserve RGB ratios while reducing changes near gamut boundaries.
    headroom=torch.where(edit>=0,1-rgb,rgb)
    scale=(headroom/(edit.abs()+1e-6)).amin(dim=1,keepdim=True).clamp(0,1)
    return (rgb+edit*scale).clamp(0,1)
