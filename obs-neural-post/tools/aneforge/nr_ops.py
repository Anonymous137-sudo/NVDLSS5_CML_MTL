"""Recovered NR publication operation expressed with ANEForge primitives.

This retains E4M3 saturation and ties-to-even rounding. Hardware parity must
pass before using it inside a checkpoint stage; no standard-softmax substitute.
"""
import numpy as np
import aneforge as af
from aneforge.graph import _const


def e4m3_piecewise(value):
    magnitude = value.abs().clip(0, 448)
    step = _const(np.array([[[[2**-9]]]], dtype=np.float16))
    for exponent in range(-5, 9):
        threshold = _const(np.array([[[[2**exponent]]]], dtype=np.float16))
        increment = _const(np.array([[[[2**(exponent-3)]]]], dtype=np.float16))
        step = af.select(magnitude.greater_equal(threshold), increment, step)
    scaled = magnitude / step
    lower = scaled.floor()
    fraction = scaled - lower
    half = _const(np.array([[[[.5]]]], dtype=np.float16))
    # The ANE round op differs at E4M3 ties on this chip. Express RNE explicitly.
    odd = lower - (lower * .5).floor() * 2
    rounded = af.select(fraction.greater(half), lower + 1,
                        af.select(fraction.equal(half), lower + odd, lower))
    return value.sign() * rounded * step


def e4m3(value):
    magnitude = value.abs().clip(0, 448)
    exponent = (magnitude.threshold(2**-6).log() * 1.4426950408889634).floor()
    two = _const(np.array([[[[2]]]], dtype=np.float16))
    step = two.pow(exponent - 3)
    scaled = magnitude / step
    lower = scaled.floor()
    fraction = scaled - lower
    half = _const(np.array([[[[.5]]]], dtype=np.float16))
    # The ANE round op differs at E4M3 ties on this chip. Express RNE explicitly.
    odd = lower - (lower * .5).floor() * 2
    rounded = af.select(fraction.greater(half), lower + 1,
                        af.select(fraction.equal(half), lower + odd, lower))
    return value.sign() * rounded * step


def e4m3_magic(value):
    # Same finite-input E4M3 grid as e4m3, using the existing MLX Metal
    # implementation's half-ULP rounding identity. The ANE compiler must
    # preserve the intermediate FP16 rounding; exhaustive parity is mandatory.
    magnitude = value.abs().clip(0, 448)
    exponent = (magnitude.threshold(2**-6).log() * 1.4426950408889634).floor()
    two = _const(np.array([[[[2]]]], dtype=np.float16))
    magic = two.pow(exponent + 7)
    rounded = (magnitude + magic) - magic
    return value.sign() * rounded


def load_weight(path, name, allow_float32=False):
    import json
    import struct
    with path.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        item = json.loads(stream.read(size))[name]
        if item['dtype'] != 'F16' and not (allow_float32 and item['dtype'] == 'F32'):
            raise ValueError(f'{name}: expected original F16 checkpoint')
        start, end = item['data_offsets']
        stream.seek(8 + size + start)
        dtype = np.float32 if item['dtype'] == 'F32' else np.float16
        return np.frombuffer(stream.read(end-start), dtype).reshape(item['shape']).astype(np.float16)


def feed_forward(value, expansion, projection, cosine, scale=16, publication=e4m3):
    expanded = af.conv(value * scale, expansion.T.reshape(4096, 1024, 1, 1)) / scale
    clipped = expanded.clip(-4, 4)
    linear = clipped.abs().linear_activation(alpha=-0.055908203125, beta=0.447265625)
    gate = clipped * linear + 0.89453125
    published = publication(expanded * gate)
    branch = af.conv(published * scale, projection.T.reshape(1024, 4096, 1, 1)) / scale
    return branch + value * cosine.reshape(1, 1024, 1, 1)


def _last_slice(value, begin, count):
    start = [0] * len(value.shape)
    start[-1] = begin
    size = list(value.shape)
    size[-1] = count
    return value.slice_by_size(start, size)


def cosine_normalize32(value):
    chunks = [_last_slice(value, i*8, 8) for i in range(4)]
    first = chunks[0]*chunks[0] + chunks[1]*chunks[1]
    second = chunks[2]*chunks[2] + chunks[3]*chunks[3]
    partial = first + second
    even = _last_slice(partial,0,2) + _last_slice(partial,4,2)
    odd = _last_slice(partial,2,2) + _last_slice(partial,6,2)
    pair = even + odd
    norm = _last_slice(pair,0,1) + _last_slice(pair,1,1)
    return value * norm.threshold(0.00006198883056640625).rsqrt()


def recovered_softmax(value):
    # Arithmetic form of the recovered half-bit affine transform; this is not
    # standard exp-softmax or the hardware SDPA operator.
    affine = value.linear_activation(alpha=0.044921875, beta=1.30078125).clip(1.03125,1.5693359375)
    bits = (affine - 1) * 32768
    exponent = (bits / 1024).floor()
    mantissa = bits - exponent * 1024
    two = _const(np.array([[[[2]]]], dtype=np.float16))
    weights = (mantissa/1024 + 1) * two.pow(exponent-15)
    total = weights @ np.ones((value.shape[-1],1),dtype=np.float16)
    one = _const(np.array([[[[1]]]],dtype=np.float16))
    return e4m3(weights * (one/total))


def recover_single_head_bias(stored):
    indices = []
    for query in range(64):
        for key in range(64):
            qy,qx,ky,kx = query//8,query%8,key//8,key%8
            bit = lambda value,index: (value >> index) & 1
            indices.append((bit(qy,2)<<11)|(bit(qx,2)<<10)|(bit(ky,2)<<9)|(bit(kx,2)<<8)|
                (bit(qy,0)<<7)|(bit(qx,1)<<6)|(bit(qx,0)<<5)|(bit(ky,0)<<4)|
                (bit(kx,1)<<3)|(bit(ky,1)<<2)|(bit(qy,1)<<1)|bit(kx,0))
    return stored.reshape(1,4096)[:,indices].reshape(1,1,64,64)


def window32(value, weights, scale=16, diagnostics=False):
    """One complete prepartitioned 32-channel/one-head recovered window block.

    Port layout [1,32,windows,64] or [windows,32,1,64]; caller owns
    image padding/window partition.
    Normalization uses the recovered fragment-tree grouping. ANE arithmetic must
    still pass captured-output parity; FP16 FMA behavior is not assumed exact.
    """
    def projection(x, w):
        return af.conv(x*scale,w.T.reshape(w.shape[1],w.shape[0],1,1))/scale
    expanded = projection(value,weights['weight1'])
    clipped = expanded.clip(-4,4)
    linear = clipped.abs().linear_activation(alpha=-0.055908203125,beta=0.447265625)
    activated = expanded * (clipped*linear + 0.89453125)
    ff_published = e4m3(activated)
    branch = projection(ff_published,weights['weight2'])
    mid = branch + value*weights['ffn_cos_skip'].reshape(1,32,1,1)
    qkv = projection(mid,weights['qkv_weight'])
    q,k,v = af.split(qkv,3,axis=1)
    q,k,v = [item.transpose([0,2,3,1]) for item in [q,k,v]]
    q = e4m3(cosine_normalize32(q)*weights['attn_scale'].reshape(1,1,1,1))
    k,v = e4m3(cosine_normalize32(k)),e4m3(v)
    scores = (q @ k.transpose([0,1,3,2])) + recover_single_head_bias(weights['attn_bias'])
    probabilities = recovered_softmax(scores)
    attended = e4m3(probabilities @ v)
    output = projection(attended.transpose([0,3,1,2]),weights['projection_weight'])
    final = e4m3(output + mid*weights['attn_cos_skip'].reshape(1,32,1,1))
    if diagnostics:
        return dict(expanded=expanded,activated=activated,ff_published=ff_published,
            branch=branch,mid=mid,qkv=qkv,query=q,key=k,value=v,scores=scores,
            probabilities=probabilities,attended=attended,projection=output,output=final)
    return final
