"""Export blocks23...47 at their native internal grid for live HDMI testing.
No benchmark or synthetic quality evaluation: zero input is tracing only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import coremltools as ct
import ane_full_model as adapter

p = argparse.ArgumentParser()
p.add_argument('destination', type=Path)
p.add_argument('--global-only', action='store_true')
a = p.parse_args()
if a.destination.exists():
    p.error('Destination already exists')
adapter.reference.CHUNK_TOKENS = 0
adapter.install_adapters(False)
r = adapter.reference

class Center(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = r.load_model('build/teacher/logical.safetensors').half().eval()
    def forward(self, value):
        m = self.model
        x = value.permute(0, 2, 3, 1)
        if a.global_only:
            for index in m.topology.global_stage:
                x = m._global(x, index)
            return x.permute(0, 3, 1, 2)
        for index in m.topology.split_encoder:
            x = m._split_window(x, index)
        skip = x
        x = r.pad_spatial_end(x, 8)
        x = r.downsample(x, weight=m.weight('block30.layer4.weight'))
        x = r.e4m3_round_trip(x)
        for index in m.topology.global_stage:
            x = m._global(x, index)
        x = r.decoder_input_merge(x @ m.weight('block39.layer0.conv_weight'),
                                 skip=skip, skip_sine=m.weight('block39.layer0.inp_upsample_sin'))
        x = r.e4m3_round_trip(x)
        for index in m.topology.split_decoder:
            x = m._split_window(x, index)
        return x.permute(0, 3, 1, 2)

shape = (1,1024,12,20) if a.global_only else (1,512,24,40)
with torch.inference_mode():
    traced = torch.jit.trace(Center().eval(), torch.zeros(shape, dtype=torch.float16))
converted = ct.convert(traced, convert_to='mlprogram', minimum_deployment_target=ct.target.macOS14,
    compute_precision=ct.precision.FLOAT16, skip_model_load=True,
    inputs=[ct.TensorType(name='features', shape=shape, dtype=np.float16)],
    outputs=[ct.TensorType(name='restored', dtype=np.float16)])
converted.short_description = 'Full HT blocks23...47 at original720p internal resolution; GPU owns0...22 and48...70'
converted.user_defined_metadata['stage_contract'] = 'ht-720-global-31-38-v1' if a.global_only else 'ht-720-center-23-47-v1'
converted.save(str(a.destination))
print(json.dumps({'package':str(a.destination),'inputNCHW':shape,
                  'outputNCHW':shape,'coremlBlocks':[31,38] if a.global_only else [23,47],
                  'gpuBlocks':[[0,30],[39,70]] if a.global_only else [[0,22],[48,70]],'inputImage':[1280,720],
                  'scope':'Export only. Performance and parity must be measured on live frames.'}),flush=True)
