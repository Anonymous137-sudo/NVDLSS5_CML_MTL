"""Experimental activation-QDQ MIL lowering for direct ANE-only tests.

ANEForge's existing int8 option compresses weights. This local node additionally
quantizes activations. It does not by itself prove packed hardware execution;
that requires compiler/runtime evidence and a matched performance comparison.
The ANEForge checkout is not modified.
"""
import math
import aneforge as af
from aneforge._compile import op


def activation_qdq(value, scale):
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Quantization scale must be finite and positive')
    return af.Tensor(value.shape, 'dlss_activation_qdq', [value], {'scale':scale})


@op('dlss_activation_qdq')
def emit_activation_qdq(emitter, tensor, name, inputs):
    scale = float(tensor.attrs['scale']).hex()
    emitter.line(f'fp16 {name}_scale = const()[name = string("{name}_scale"), val = fp16({scale})];')
    emitter.line(f'string {name}_dtype = const()[name = string("{name}_dtype"), val = string("int8")];')
    emitter.line(f'{emitter.ty_dt(tensor.shape,"int8")} {name}_q = quantize(input = {inputs[0]}, '
                 f'output_dtype = {name}_dtype, scale = {name}_scale)[name = string("{name}_q")];')
    emitter.line(f'{emitter.ty(tensor.shape)} {name} = dequantize(input = {name}_q, '
                 f'scale = {name}_scale)[name = string("{name}")];')
