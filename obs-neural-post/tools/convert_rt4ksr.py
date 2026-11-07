"""Fuse the authors' RT4KSR x3 checkpoint for native MLX; no Core ML.

Run with the development venv. torch.load uses weights_only=True. The resulting
runtime archive is tensor data, not executable Python. Original architecture:
https://github.com/eduardzamfir/RT4KSR (Apache-2.0).
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import save_file


def branch(x, state, prefix):
    def t(s): return state[prefix + s]
    expanded = F.conv2d(x, t(".expand_conv.weight"), t(".expand_conv.bias"))
    # Upstream pads with expansion bias, rather than zero, before middle conv.
    padded = F.pad(expanded - t(".expand_conv.bias")[None, :, None, None], (1,1,1,1))
    padded += t(".expand_conv.bias")[None, :, None, None]
    middle = F.conv2d(padded, t(".fea_conv.weight"), t(".fea_conv.bias")) + expanded
    return F.conv2d(middle, t(".reduce_conv.weight"), t(".reduce_conv.bias")) + x


def fuse(state, prefix):
    a = state[prefix + ".expand_conv.weight"][:, :, 0, 0].double()
    b = state[prefix + ".fea_conv.weight"].double().clone()
    c = state[prefix + ".reduce_conv.weight"][:, :, 0, 0].double()
    for i in range(b.shape[0]): b[i,i,1,1] += 1
    weight = torch.einsum("om,mnhw,ni->oihw", c, b, a)
    for i in range(weight.shape[0]): weight[i,i,1,1] += 1
    bias = c @ (b.sum((2,3)) @ state[prefix + ".expand_conv.bias"].double()
                + state[prefix + ".fea_conv.bias"].double()) + state[prefix + ".reduce_conv.bias"].double()
    return weight.float(), bias.float()


def reference(x, s):
    def norm(x, p):
        centered = x-x.mean(1,keepdim=True)
        return centered/torch.sqrt((centered*centered).mean(1,keepdim=True)+1e-6) * s[p+".weight"][None,:,None,None]+s[p+".bias"][None,:,None,None]
    x = F.pixel_unshuffle(x,2)
    x = F.conv2d(x,s["head.0.weight"],s["head.0.bias"],padding=1)
    for i in range(4): x = F.gelu(branch(norm(x,f"body.{i}.norm"),s,f"body.{i}.conv1"))
    x = branch(norm(x,"tail.0"),s,"tail.1")
    return F.pixel_shuffle(F.conv2d(x,s["upsample.0.weight"],s["upsample.0.bias"],padding=1),6)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint",type=Path); p.add_argument("output",type=Path)
    args = p.parse_args()
    s = {k.removeprefix("module."):v for k,v in torch.load(args.checkpoint,map_location="cpu",weights_only=True)["state_dict"].items()}
    torch.manual_seed(123)
    converted = {k:v for k,v in s.items() if k.startswith(("head.","upsample.","tail.0.")) or ".norm." in k}
    max_error = 0
    for prefix in [f"body.{i}.conv1" for i in range(4)]+["tail.1"]:
        w,b = fuse(s,prefix)
        test = torch.rand(1,24,16,20)
        error = (branch(test,s,prefix)-F.conv2d(test,w,b,padding=1)).abs().max().item()
        max_error = max(max_error,error)
        assert error < 1e-4, (prefix,error)
        converted[prefix+".rep_conv.weight"] = w
        converted[prefix+".rep_conv.bias"] = b
    # MLX convolution weights are OHWI rather than OIHW.
    converted = {k:(v.permute(0,2,3,1).contiguous() if v.ndim == 4 else v.contiguous()) for k,v in converted.items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    save_file(converted,str(args.output),metadata={"architecture":"rt4ksr-x3-24-4", "source":"eduardzamfir/RT4KSR", "scale":"3"})
    x = torch.rand(1,3,48,64)
    expected = reference(x,s)
    save_file({"input":x.permute(0,2,3,1).contiguous(),"expected":expected.permute(0,2,3,1).contiguous()},str(args.output.with_suffix(".parity.safetensors")))
    record = {"checkpointSHA256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "archiveSHA256":hashlib.sha256(args.output.read_bytes()).hexdigest(),"fusionMaxAbsError":max_error,
              "source":"https://github.com/eduardzamfir/RT4KSR", "scale":3,"features":24,"blocks":4}
    args.output.with_suffix(".json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))

if __name__ == "__main__": main()
